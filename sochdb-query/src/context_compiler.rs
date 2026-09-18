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
    /// What this entry costs *as rendered into the body*, not the token count
    /// of `text`. The two differ by the template's per-entry framing, and the
    /// budget is spent on the rendered form.
    pub tokens: usize,
    pub episode_id: Option<u64>,
    pub t_valid_from: Option<u64>,
    pub t_valid_to: Option<u64>,
    pub trust_hint: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledContext {
    pub body: String,
    /// Exact token count of `body`, measured after assembly. Guaranteed
    /// `<= budget`.
    pub exact_tokens: usize,
    pub budget: usize,
    pub facts: Vec<CompiledFact>,
    /// Whether eligible evidence was left out because the budget ran out.
    pub truncated: bool,
}

/// Greedy MMR selection with exact BPE running sum; stops when budget exhausted.
pub struct ContextCompiler {
    decay: TemporalScorer,
}

/// Separator placed between rendered entries in the body.
const BODY_SEPARATOR: &str = "\n";

/// Render one selected candidate exactly as it appears in the body.
///
/// Budget accounting and body construction must go through this together.
/// Counting `c.text` and then emitting `format!("### Memory {id}\n{text}\n")`
/// makes `exact_tokens` a count of something the caller never receives, and the
/// difference is not a rounding error: a heading and a fence per entry is real
/// context the model has to pay for. The compiler's one promise is a block that
/// fits in `budget`, so what it counts has to be what it emits.
fn render_entry(template: ContextTemplate, id: DocId, text: &str) -> String {
    match template {
        ContextTemplate::Markdown => format!("### Memory {}\n{}\n", id, text),
        ContextTemplate::Toon => format!("mem{}|{}\n", id, text.replace('\n', " ")),
        ContextTemplate::Plain => text.to_string(),
    }
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

        let eligible = decayed.len();
        let selected = self.mmr_select(&decayed, texts, spec, spec.mmr_lambda);
        self.render(spec, &selected, texts, eligible)
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
        spec: &ContextSpec,
        lambda: f32,
    ) -> Vec<DocId> {
        let budget = spec.budget;
        let separator = count_tokens_exact(BODY_SEPARATOR);

        // Everything that does not depend on what has already been selected is
        // computed once, here. Rendering a candidate, tokenizing that render,
        // and splitting its text into words all produce the same answer on
        // every pass, but doing them inside the selection loop repeated each
        // one `selected × remaining` times -- which is what made compiling a
        // few thousand candidates take seconds rather than milliseconds.
        let mut remaining: Vec<MmrCandidate<'_>> = ranked
            .iter()
            .filter_map(|(id, relevance)| {
                let cand = texts.get(id)?;
                Some(MmrCandidate {
                    id: *id,
                    relevance: *relevance,
                    base_cost: count_tokens_exact(&render_entry(spec.template, *id, &cand.text)),
                    words: cand.text.split_whitespace().collect(),
                    similarity: 0.0,
                })
            })
            .collect();

        let mut selected: Vec<DocId> = Vec::new();
        let mut used_tokens = 0usize;

        while !remaining.is_empty() && used_tokens < budget {
            // Cost is what the entry costs *in the body*: its rendered form
            // plus the separator that precedes it. Selecting against the bare
            // source text picks entries that cannot actually fit.
            let separator_cost = if selected.is_empty() { 0 } else { separator };
            let mut best_idx = None;
            let mut best_mmr = f32::NEG_INFINITY;
            let mut best_cost = 0usize;

            for (i, cand) in remaining.iter().enumerate() {
                let cost = cand.base_cost + separator_cost;
                if used_tokens + cost > budget {
                    continue;
                }
                let mmr = lambda * cand.relevance - (1.0 - lambda) * cand.similarity;
                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = Some(i);
                    best_cost = cost;
                }
            }

            // Nothing left fits. Previously the loop fell back to index 0 and
            // selected it anyway, pushing a candidate it had just rejected past
            // the budget.
            let Some(best_idx) = best_idx else { break };
            let chosen = remaining.remove(best_idx);
            used_tokens += best_cost;
            selected.push(chosen.id);

            // Similarity to the selected set is a running maximum, so only the
            // document just added can raise it. Recomputing against the whole
            // selected set each round redid comparisons whose result could not
            // have changed.
            for cand in &mut remaining {
                cand.similarity = cand.similarity.max(jaccard(&cand.words, &chosen.words));
            }
        }
        selected
    }

    /// The pre-optimisation selection loop, kept only so a test can prove the
    /// optimised one picks exactly the same documents. Not used in production.
    #[cfg(test)]
    fn mmr_select_legacy(
        &self,
        ranked: &[(DocId, f32)],
        texts: &HashMap<DocId, ContextCandidate>,
        spec: &ContextSpec,
        lambda: f32,
    ) -> Vec<DocId> {
        let budget = spec.budget;
        let separator = count_tokens_exact(BODY_SEPARATOR);
        let mut selected: Vec<DocId> = Vec::new();
        let mut used_tokens = 0usize;
        let mut remaining: Vec<(DocId, f32)> = ranked.to_vec();

        while !remaining.is_empty() && used_tokens < budget {
            let mut best_idx = None;
            let mut best_mmr = f32::NEG_INFINITY;
            let mut best_cost = 0usize;
            for (i, (id, rel)) in remaining.iter().enumerate() {
                let Some(cand) = texts.get(id) else { continue };
                // Cost is what the entry costs *in the body*: its rendered form
                // plus the separator that precedes it. Selecting against the
                // bare source text picks entries that cannot actually fit.
                let rendered = render_entry(spec.template, *id, &cand.text);
                let cost =
                    count_tokens_exact(&rendered) + if selected.is_empty() { 0 } else { separator };
                if used_tokens + cost > budget {
                    continue;
                }
                let max_sim = selected
                    .iter()
                    .filter_map(|sid| texts.get(sid))
                    .map(|s| {
                        jaccard(
                            &cand.text.split_whitespace().collect(),
                            &s.text.split_whitespace().collect(),
                        )
                    })
                    .fold(0.0f32, f32::max);
                let mmr = lambda * rel - (1.0 - lambda) * max_sim;
                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = Some(i);
                    best_cost = cost;
                }
            }
            // Nothing left fits. Previously the loop fell back to index 0 and
            // selected it anyway, pushing a candidate it had just rejected past
            // the budget.
            let Some(best_idx) = best_idx else { break };
            let (id, _) = remaining.remove(best_idx);
            used_tokens += best_cost;
            selected.push(id);
        }
        selected
    }

    fn render(
        &self,
        spec: &ContextSpec,
        selected: &[DocId],
        texts: &HashMap<DocId, ContextCandidate>,
        eligible: usize,
    ) -> CompiledContext {
        let separator = count_tokens_exact(BODY_SEPARATOR);
        let mut facts = Vec::new();
        let mut parts: Vec<String> = Vec::new();
        let mut estimated = 0usize;

        for id in selected {
            let Some(c) = texts.get(id) else { continue };
            let rendered = render_entry(spec.template, *id, &c.text);
            let tok = count_tokens_exact(&rendered);
            let cost = tok + if parts.is_empty() { 0 } else { separator };
            if estimated + cost > spec.budget {
                break;
            }
            estimated += cost;
            parts.push(rendered);
            facts.push(CompiledFact {
                text: c.text.clone(),
                tokens: tok,
                episode_id: c.episode_id,
                t_valid_from: c.t_valid_from,
                t_valid_to: c.t_valid_to,
                trust_hint: c.relevance,
            });
        }

        // Final feasibility check against the bytes actually returned. The loop
        // above sums per-entry counts, and BPE is not obliged to be additive
        // across a join, so the assembled body is measured rather than
        // inferred. Normally this costs one tokenization and drops nothing.
        let mut body = parts.join(BODY_SEPARATOR);
        let mut exact_tokens = count_tokens_exact(&body);
        while exact_tokens > spec.budget && !parts.is_empty() {
            parts.pop();
            facts.pop();
            body = parts.join(BODY_SEPARATOR);
            exact_tokens = count_tokens_exact(&body);
        }

        // Measured against the *eligible* candidates, not against the selection.
        // Selection is now budget-aware and stops before it overruns, so
        // comparing the two would report `false` for every query — including
        // one that dropped most of its evidence — and a caller that widens its
        // budget on `truncated` would never know to.
        let truncated = facts.len() < eligible;
        CompiledContext {
            body,
            exact_tokens,
            budget: spec.budget,
            facts,
            truncated,
        }
    }
}

/// A candidate with everything that is invariant across the MMR loop already
/// computed, plus its running similarity to the selected set.
struct MmrCandidate<'a> {
    id: DocId,
    relevance: f32,
    /// Token cost of this entry's rendered form, excluding any separator.
    base_cost: usize,
    words: std::collections::HashSet<&'a str>,
    /// Highest Jaccard similarity against any already-selected document.
    similarity: f32,
}

/// Jaccard similarity over pre-split word sets.
///
/// Takes sets rather than strings because the caller reuses each document's set
/// across the whole selection; splitting on every comparison was the dominant
/// cost of context compilation.
fn jaccard(a: &std::collections::HashSet<&str>, b: &std::collections::HashSet<&str>) -> f32 {
    let intersection = a.intersection(b).count() as f32;
    // |A ∪ B| = |A| + |B| - |A ∩ B|, so the second full pass a union() walk
    // would cost is unnecessary.
    let union = a.len() as f32 + b.len() as f32 - intersection;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn candidate(doc_id: DocId, text: &str, relevance: f32) -> ContextCandidate {
        ContextCandidate {
            doc_id,
            text: text.to_string(),
            relevance,
            timestamp_secs: 0.0,
            episode_id: Some(doc_id),
            t_valid_from: None,
            t_valid_to: None,
        }
    }

    /// The optimised selector hoists rendering, tokenizing and word-splitting
    /// out of the loop and tracks similarity incrementally. None of that may
    /// change which documents come back, so it is checked against the original
    /// implementation over pseudo-random corpora with overlapping vocabulary --
    /// overlap matters because it is what drives the MMR diversity penalty.
    #[test]
    fn optimised_mmr_selects_exactly_what_the_original_did() {
        const WORDS: [&str; 12] = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu",
        ];
        let mut seed = 0x5eed_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for trial in 0..40 {
            let n = 5 + (trial * 7) % 60;
            let mut texts = HashMap::new();
            let mut ranked = Vec::new();
            for i in 0..n {
                let len = 3 + (next() % 8) as usize;
                let text: Vec<&str> = (0..len)
                    .map(|_| WORDS[(next() % WORDS.len() as u64) as usize])
                    .collect();
                let id = i as DocId;
                let rel = (next() % 1000) as f32 / 1000.0;
                texts.insert(id, candidate(id, &text.join(" "), rel));
                ranked.push((id, rel));
            }

            for template in [
                ContextTemplate::Toon,
                ContextTemplate::Markdown,
                ContextTemplate::Plain,
            ] {
                for budget in [0, 1, 17, 64, 256, 4096] {
                    for lambda in [0.0, 0.5, 1.0] {
                        let spec = ContextSpec {
                            budget,
                            template,
                            ..ContextSpec::default()
                        };
                        let compiler = ContextCompiler::new(&spec);
                        assert_eq!(
                            compiler.mmr_select(&ranked, &texts, &spec, lambda),
                            compiler.mmr_select_legacy(&ranked, &texts, &spec, lambda),
                            "trial={trial} template={template:?} budget={budget} lambda={lambda}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compile_respects_exact_budget() {
        let spec = ContextSpec {
            budget: 50,
            ..ContextSpec::default()
        };
        let compiler = ContextCompiler::new(&spec);
        let bm25 = vec![(1u64, 1.0), (2, 0.8)];
        let mut texts = HashMap::new();
        texts.insert(1, candidate(1, "short memory", 1.0));
        texts.insert(
            2,
            candidate(
                2,
                "a much longer memory entry that would exceed the token budget if included",
                0.8,
            ),
        );
        let out = compiler.compile(&spec, &bm25, &[], &[], &texts);
        assert!(out.exact_tokens <= spec.budget);
        assert!(!out.facts.is_empty());
    }

    /// `exact_tokens` must describe the body the caller receives.
    ///
    /// Counting the source text and rendering with a heading and a fence per
    /// entry let a "50-token" context be materially larger than 50 tokens —
    /// the one number the compiler exists to guarantee.
    #[test]
    fn exact_tokens_counts_the_rendered_body_not_the_source_text() {
        for template in [
            ContextTemplate::Markdown,
            ContextTemplate::Toon,
            ContextTemplate::Plain,
        ] {
            let spec = ContextSpec {
                budget: 4096,
                template,
                ..ContextSpec::default()
            };
            let compiler = ContextCompiler::new(&spec);
            let mut texts = HashMap::new();
            texts.insert(1, candidate(1, "alpha beta gamma", 1.0));
            texts.insert(2, candidate(2, "delta epsilon zeta", 0.9));

            let out = compiler.compile(&spec, &[(1, 1.0), (2, 0.9)], &[], &[], &texts);
            assert_eq!(out.facts.len(), 2, "both fit in 4096 tokens");
            assert_eq!(
                out.exact_tokens,
                count_tokens_exact(&out.body),
                "{template:?}: reported tokens must equal the body's tokens"
            );
        }
    }

    /// The per-entry framing has to be charged to the budget, or a tight budget
    /// silently overspends by the framing of every entry it admits.
    #[test]
    fn markdown_framing_is_charged_against_a_tight_budget() {
        let text = "alpha beta gamma delta";
        let source_tokens = count_tokens_exact(text);
        let rendered_tokens = count_tokens_exact(&render_entry(ContextTemplate::Markdown, 1, text));
        assert!(
            rendered_tokens > source_tokens,
            "the markdown template must cost something, else this proves nothing"
        );

        // A budget that fits the source text but not its rendered form.
        let spec = ContextSpec {
            budget: rendered_tokens - 1,
            template: ContextTemplate::Markdown,
            ..ContextSpec::default()
        };
        let compiler = ContextCompiler::new(&spec);
        let mut texts = HashMap::new();
        texts.insert(1, candidate(1, text, 1.0));

        let out = compiler.compile(&spec, &[(1, 1.0)], &[], &[], &texts);
        assert!(
            out.facts.is_empty(),
            "the entry does not fit and is not emitted"
        );
        assert!(out.exact_tokens <= spec.budget);
    }

    /// A budget too small for anything must produce an empty context, not the
    /// first candidate emitted anyway.
    #[test]
    fn an_impossible_budget_yields_an_empty_context() {
        let spec = ContextSpec {
            budget: 1,
            template: ContextTemplate::Markdown,
            ..ContextSpec::default()
        };
        let compiler = ContextCompiler::new(&spec);
        let mut texts = HashMap::new();
        texts.insert(
            1,
            candidate(1, "a memory entry far larger than one token", 1.0),
        );

        let out = compiler.compile(&spec, &[(1, 1.0)], &[], &[], &texts);
        assert!(out.facts.is_empty());
        assert!(out.body.is_empty());
        assert_eq!(out.exact_tokens, 0);
    }

    /// The separator between entries is part of the body and must be paid for.
    #[test]
    fn budget_holds_across_many_entries_including_separators() {
        let spec = ContextSpec {
            budget: 64,
            template: ContextTemplate::Markdown,
            ..ContextSpec::default()
        };
        let compiler = ContextCompiler::new(&spec);
        let mut texts = HashMap::new();
        let mut hits = Vec::new();
        for id in 1u64..=12 {
            texts.insert(
                id,
                candidate(id, &format!("memory number {id} of twelve"), 1.0),
            );
            hits.push((id, 1.0 - id as f32 * 0.01));
        }

        let out = compiler.compile(&spec, &hits, &[], &[], &texts);
        assert_eq!(out.exact_tokens, count_tokens_exact(&out.body));
        assert!(
            out.exact_tokens <= spec.budget,
            "{} tokens exceeds the {} budget",
            out.exact_tokens,
            spec.budget
        );
        assert!(out.truncated, "not everything fits in 64 tokens");
    }
}
