//! Context compiler — hard-budget context assembly as a query primitive.
//!
//! Composes exact BPE counting, temporal decay, weighted RRF fusion, and MMR
//! diversity into a single entry point returning a packed block ≤ budget B.

use crate::exact_token_counter::count_tokens_exact;
use crate::temporal_decay::{TemporalDecayConfig, TemporalScorer};
use crate::unified_fusion::fuse_rrf_weighted;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type DocId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSpec {
    pub budget: usize,
    pub bm25_weight: f32,
    pub trigram_weight: f32,
    pub vector_weight: f32,
    pub mmr_lambda: f32,
    pub decay_half_life_secs: f64,
    pub template: ContextTemplate,
}

impl Default for ContextSpec {
    fn default() -> Self {
        Self {
            budget: 4096,
            bm25_weight: 0.4,
            trigram_weight: 0.2,
            vector_weight: 0.4,
            mmr_lambda: 0.7,
            decay_half_life_secs: 86_400.0 * 7.0,
            template: ContextTemplate::Markdown,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContextTemplate {
    Markdown,
    Toon,
    Plain,
}

#[derive(Debug, Clone)]
pub struct ContextCandidate {
    pub doc_id: DocId,
    pub text: String,
    pub relevance: f32,
    pub timestamp_secs: f64,
    pub episode_id: Option<u64>,
    pub t_valid_from: Option<u64>,
    pub t_valid_to: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledFact {
    pub text: String,
    pub tokens: usize,
    pub episode_id: Option<u64>,
    pub t_valid_from: Option<u64>,
    pub t_valid_to: Option<u64>,
    pub trust_hint: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledContext {
    pub body: String,
    pub exact_tokens: usize,
    pub budget: usize,
    pub facts: Vec<CompiledFact>,
    pub truncated: bool,
}

/// Greedy MMR selection with exact BPE running sum; stops when budget exhausted.
pub struct ContextCompiler {
    decay: TemporalScorer,
}

impl ContextCompiler {
    pub fn new(spec: &ContextSpec) -> Self {
        let decay_cfg = TemporalDecayConfig {
            half_life_secs: spec.decay_half_life_secs,
            ..TemporalDecayConfig::default()
        };
        Self {
            decay: TemporalScorer::new(decay_cfg),
        }
    }

    pub fn compile(
        &self,
        spec: &ContextSpec,
        bm25: &[(DocId, f32)],
        trigram: &[(DocId, f32)],
        vector: &[(DocId, f32)],
        texts: &HashMap<DocId, ContextCandidate>,
    ) -> CompiledContext {
        let fused = self.fuse_lanes(spec, bm25, trigram, vector);
        let mut decayed: Vec<(DocId, f32)> = fused
            .into_iter()
            .filter_map(|(id, score)| {
                texts.get(&id).map(|c| {
                    let final_score = self.decay.final_score(score, c.timestamp_secs);
                    (id, final_score)
                })
            })
            .collect();
        decayed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected = self.mmr_select(&decayed, texts, spec.budget, spec.mmr_lambda);
        self.render(spec, &selected, texts)
    }

    fn fuse_lanes(
        &self,
        spec: &ContextSpec,
        bm25: &[(DocId, f32)],
        trigram: &[(DocId, f32)],
        vector: &[(DocId, f32)],
    ) -> HashMap<DocId, f32> {
        use crate::filtered_vector_search::ScoredResult;
        use crate::unified_fusion::RankedList;

        let to_scored = |hits: &[(DocId, f32)]| {
            hits.iter()
                .map(|(id, score)| ScoredResult::new(*id, *score))
                .collect::<Vec<_>>()
        };

        let bm25_scored = to_scored(bm25);
        let trigram_scored = to_scored(trigram);
        let vector_scored = to_scored(vector);

        let mut lists = Vec::new();
        if !bm25_scored.is_empty() {
            lists.push(RankedList {
                results: &bm25_scored,
                weight: spec.bm25_weight,
            });
        }
        if !trigram_scored.is_empty() {
            lists.push(RankedList {
                results: &trigram_scored,
                weight: spec.trigram_weight,
            });
        }
        if !vector_scored.is_empty() {
            lists.push(RankedList {
                results: &vector_scored,
                weight: spec.vector_weight,
            });
        }
        fuse_rrf_weighted(&lists, 60.0)
            .into_iter()
            .map(|(id, score)| (id.0, score))
            .collect()
    }

    fn mmr_select(
        &self,
        ranked: &[(DocId, f32)],
        texts: &HashMap<DocId, ContextCandidate>,
        budget: usize,
        lambda: f32,
    ) -> Vec<DocId> {
        let mut selected: Vec<DocId> = Vec::new();
        let mut used_tokens = 0usize;
        let mut remaining: Vec<(DocId, f32)> = ranked.to_vec();

        while !remaining.is_empty() && used_tokens < budget {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;
            for (i, (id, rel)) in remaining.iter().enumerate() {
                let Some(cand) = texts.get(id) else { continue };
                let tok = count_tokens_exact(&cand.text);
                if used_tokens + tok > budget {
                    continue;
                }
                let max_sim = selected
                    .iter()
                    .filter_map(|sid| texts.get(sid))
                    .map(|s| jaccard(&cand.text, &s.text))
                    .fold(0.0f32, f32::max);
                let mmr = lambda * rel - (1.0 - lambda) * max_sim;
                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }
            let (id, _) = remaining.remove(best_idx);
            let Some(cand) = texts.get(&id) else { break };
            used_tokens += count_tokens_exact(&cand.text);
            selected.push(id);
        }
        selected
    }

    fn render(
        &self,
        spec: &ContextSpec,
        selected: &[DocId],
        texts: &HashMap<DocId, ContextCandidate>,
    ) -> CompiledContext {
        let mut facts = Vec::new();
        let mut body_parts = Vec::new();
        let mut total_tokens = 0usize;

        for id in selected {
            let Some(c) = texts.get(id) else { continue };
            let tok = count_tokens_exact(&c.text);
            if total_tokens + tok > spec.budget {
                break;
            }
            total_tokens += tok;
            let rendered = match spec.template {
                ContextTemplate::Markdown => {
                    format!("### Memory {}\n{}\n", id, c.text)
                }
                ContextTemplate::Toon => format!("mem{}|{}\n", id, c.text.replace('\n', " ")),
                ContextTemplate::Plain => c.text.clone(),
            };
            body_parts.push(rendered);
            facts.push(CompiledFact {
                text: c.text.clone(),
                tokens: tok,
                episode_id: c.episode_id,
                t_valid_from: c.t_valid_from,
                t_valid_to: c.t_valid_to,
                trust_hint: c.relevance,
            });
        }

        let truncated = facts.len() < selected.len();
        CompiledContext {
            body: body_parts.join("\n"),
            exact_tokens: total_tokens,
            budget: spec.budget,
            facts,
            truncated,
        }
    }
}

fn jaccard(a: &str, b: &str) -> f32 {
    let sa: std::collections::HashSet<_> = a.split_whitespace().collect();
    let sb: std::collections::HashSet<_> = b.split_whitespace().collect();
    if sa.is_empty() && sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn compile_respects_exact_budget() {
        let spec = ContextSpec {
            budget: 50,
            ..ContextSpec::default()
        };
        let compiler = ContextCompiler::new(&spec);
        let bm25 = vec![(1u64, 1.0), (2, 0.8)];
        let mut texts = HashMap::new();
        texts.insert(
            1,
            ContextCandidate {
                doc_id: 1,
                text: "short memory".into(),
                relevance: 1.0,
                timestamp_secs: 0.0,
                episode_id: Some(1),
                t_valid_from: None,
                t_valid_to: None,
            },
        );
        texts.insert(
            2,
            ContextCandidate {
                doc_id: 2,
                text: "a much longer memory entry that would exceed the token budget if included"
                    .into(),
                relevance: 0.8,
                timestamp_secs: 0.0,
                episode_id: Some(2),
                t_valid_from: None,
                t_valid_to: None,
            },
        );
        let out = compiler.compile(&spec, &bm25, &[], &[], &texts);
        assert!(out.exact_tokens <= spec.budget);
        assert!(!out.facts.is_empty());
    }
}