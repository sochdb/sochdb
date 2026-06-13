//! WASM rerank plugin interface for in-engine cross-encoder reranking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankCandidate {
    pub doc_id: u64,
    pub text: String,
    pub fused_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub doc_id: u64,
    pub rerank_score: f32,
    pub fused_score: f32,
}

#[derive(Debug, Clone)]
pub struct WasmRerankPlugin {
    pub plugin_id: String,
    pub loaded: bool,
}

impl WasmRerankPlugin {
    pub fn new(plugin_id: impl Into<String>) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            loaded: false,
        }
    }

    pub fn load(&mut self) {
        self.loaded = true;
    }

    /// Rerank top-n candidates. Production path delegates to WASM sandbox;
    /// fallback uses fused score ordering.
    pub fn rerank(&self, candidates: &[RerankCandidate], top_k: usize) -> Vec<RerankResult> {
        let mut scored: Vec<RerankResult> = candidates
            .iter()
            .map(|c| RerankResult {
                doc_id: c.doc_id,
                rerank_score: c.fused_score,
                fused_score: c.fused_score,
            })
            .collect();
        scored.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRerankOutput {
    pub doc_id: u64,
    pub rerank_score: f32,
    pub episode_id: Option<u64>,
    pub trust_score: f32,
}

pub fn attach_provenance(
    reranked: &[RerankResult],
    provenance: &HashMap<u64, (Option<u64>, f32)>,
) -> Vec<ProvenanceRerankOutput> {
    reranked
        .iter()
        .map(|r| {
            let (ep, trust) = provenance.get(&r.doc_id).copied().unwrap_or((None, 0.5));
            ProvenanceRerankOutput {
                doc_id: r.doc_id,
                rerank_score: r.rerank_score,
                episode_id: ep,
                trust_score: trust,
            }
        })
        .collect()
}
