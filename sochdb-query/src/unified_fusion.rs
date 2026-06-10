// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Unified Hybrid Fusion with Mandatory Pre-Filtering (Task 7)
//!
//! This module implements hybrid retrieval (vector + BM25) that **never**
//! post-filters. The key insight is:
//!
//! > Both vector and BM25 executors receive the **same** AllowedSet,
//! > produce candidates **guaranteed** within it, then fusion merges by doc_id.
//!
//! ## Anti-Pattern (What We Avoid)
//!
//! ```text
//! BAD: vector_search() → candidates → filter → too few
//!      bm25_search() → candidates → filter → inconsistent
//!      fusion(unfiltered_v, unfiltered_b) → filter at end → broken!
//! ```
//!
//! ## Correct Pattern
//!
//! ```text
//! GOOD: compute AllowedSet from FilterIR
//!       vector_search(query, allowed_set) → filtered_v
//!       bm25_search(query, allowed_set) → filtered_b
//!       fusion(filtered_v, filtered_b) → already correct!
//! ```
//!
//! ## Fusion Cost
//!
//! With pre-filtered candidates:
//! - Fusion is O(k_v + k_b) with hash-join or two-pointer merge
//! - Total work is proportional to constrained candidate sizes
//! - No wasted scoring on disallowed documents

use std::collections::HashMap;
use std::sync::Arc;

use crate::candidate_gate::AllowedSet;
use crate::filter_ir::{AuthScope, FilterIR};
use crate::filtered_vector_search::ScoredResult;
use crate::grep_executor::GrepMode;
use crate::namespace::NamespaceScope;

// ============================================================================
// Fusion Configuration
// ============================================================================

/// Fusion method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion: score = Σ wᵢ / (k + rankᵢ), rank 1-indexed.
    Rrf {
        k: f32,
        vector_weight: f32,
        bm25_weight: f32,
    },

    /// Linear combination of normalized scores
    Linear {
        vector_weight: f32,
        bm25_weight: f32,
    },

    /// Take max score across modalities
    Max,

    /// Cascade: use one modality to filter, other to rank
    Cascade { primary: Modality },
}

/// Search modality
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Modality {
    Vector,
    Bm25,
    /// Trigram-accelerated regex (grep) lane.
    Grep,
}

impl Default for FusionMethod {
    fn default() -> Self {
        Self::Rrf {
            k: 60.0,
            vector_weight: 1.0,
            bm25_weight: 1.0,
        }
    }
}

/// Configuration for hybrid fusion
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Fusion method
    pub method: FusionMethod,

    /// Number of candidates to retrieve from each modality
    pub candidates_per_modality: usize,

    /// Final result limit
    pub final_k: usize,

    /// Minimum score threshold (after fusion)
    pub min_score: Option<f32>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::default(),
            candidates_per_modality: 100,
            final_k: 10,
            min_score: None,
        }
    }
}

// ============================================================================
// Unified Hybrid Query
// ============================================================================

/// A hybrid query that enforces pre-filtering
#[derive(Debug, Clone)]
pub struct UnifiedHybridQuery {
    /// Namespace scope (mandatory)
    pub namespace: NamespaceScope,

    /// Vector query (optional)
    pub vector_query: Option<VectorQuerySpec>,

    /// BM25 query (optional)
    pub bm25_query: Option<Bm25QuerySpec>,

    /// Grep (regex) query (optional)
    pub grep_query: Option<GrepQuerySpec>,

    /// User-provided filter
    pub filter: FilterIR,

    /// Fusion configuration
    pub fusion_config: FusionConfig,
}

/// Vector query specification
#[derive(Debug, Clone)]
pub struct VectorQuerySpec {
    /// Query embedding
    pub embedding: Vec<f32>,
    /// ef_search for HNSW
    pub ef_search: usize,
}

/// BM25 query specification
#[derive(Debug, Clone)]
pub struct Bm25QuerySpec {
    /// Query text (will be tokenized)
    pub text: String,
    /// Fields to search
    pub fields: Vec<String>,
}

/// Grep (regex) query specification for the third lane.
///
/// The [`GrepMode`] determines how the lane participates in fusion:
/// - [`GrepMode::Rank`] contributes a ranked list weighted by `weight`.
/// - [`GrepMode::Gate`] narrows the `AllowedSet` *before* the vector and BM25
///   lanes run (a cascade); `weight` is unused in that case.
#[derive(Debug, Clone)]
pub struct GrepQuerySpec {
    /// Regular expression pattern.
    pub pattern: String,
    /// How the lane is consumed by fusion.
    pub mode: GrepMode,
    /// Fusion weight (used only for [`GrepMode::Rank`]).
    pub weight: f32,
}

impl UnifiedHybridQuery {
    /// Create a new hybrid query (namespace is mandatory)
    pub fn new(namespace: NamespaceScope) -> Self {
        Self {
            namespace,
            vector_query: None,
            bm25_query: None,
            grep_query: None,
            filter: FilterIR::all(),
            fusion_config: FusionConfig::default(),
        }
    }

    /// Add vector search
    pub fn with_vector(mut self, embedding: Vec<f32>) -> Self {
        self.vector_query = Some(VectorQuerySpec {
            embedding,
            ef_search: 100,
        });
        self
    }

    /// Add BM25 search
    pub fn with_bm25(mut self, text: impl Into<String>) -> Self {
        self.bm25_query = Some(Bm25QuerySpec {
            text: text.into(),
            fields: vec!["content".to_string()],
        });
        self
    }

    /// Add a grep (regex) lane with the default weight of `1.0`.
    pub fn with_grep(mut self, pattern: impl Into<String>, mode: GrepMode) -> Self {
        self.grep_query = Some(GrepQuerySpec {
            pattern: pattern.into(),
            mode,
            weight: 1.0,
        });
        self
    }

    /// Add a grep (regex) lane with an explicit fusion weight.
    pub fn with_grep_weighted(
        mut self,
        pattern: impl Into<String>,
        mode: GrepMode,
        weight: f32,
    ) -> Self {
        self.grep_query = Some(GrepQuerySpec {
            pattern: pattern.into(),
            mode,
            weight,
        });
        self
    }

    /// Add filter
    pub fn with_filter(mut self, filter: FilterIR) -> Self {
        self.filter = filter;
        self
    }

    /// Set fusion config
    pub fn with_fusion(mut self, config: FusionConfig) -> Self {
        self.fusion_config = config;
        self
    }

    /// Compute the complete effective filter
    ///
    /// This combines namespace scope + user filter. Auth scope is added later.
    pub fn effective_filter(&self) -> FilterIR {
        self.namespace.to_filter_ir().and(self.filter.clone())
    }
}

// ============================================================================
// Filtered Candidates
// ============================================================================

/// Candidates from a single modality (already filtered)
#[derive(Debug)]
pub struct FilteredCandidates {
    /// Modality source
    pub modality: Modality,
    /// Scored results (doc_id, score)
    pub results: Vec<ScoredResult>,
    /// Whether the allowed set was applied
    pub filtered: bool,
}

impl FilteredCandidates {
    /// Create from vector search results
    pub fn from_vector(results: Vec<ScoredResult>) -> Self {
        Self {
            modality: Modality::Vector,
            results,
            filtered: true,
        }
    }

    /// Create from BM25 results
    pub fn from_bm25(results: Vec<ScoredResult>) -> Self {
        Self {
            modality: Modality::Bm25,
            results,
            filtered: true,
        }
    }

    /// Create from grep (regex) results
    pub fn from_grep(results: Vec<ScoredResult>) -> Self {
        Self {
            modality: Modality::Grep,
            results,
            filtered: true,
        }
    }
}

// ============================================================================
// Canonical Document Identity + RRF Kernel
// ============================================================================

/// Canonical document identity consumed by the fusion kernel.
///
/// A newtype (rather than a bare `u64`) so a retrieval-space document id can
/// never be silently confused with a raw vector offset, record id, or rank.
/// The fusion kernel keys exclusively on this type; executors convert their
/// native ids into a `DocId` at the boundary.
///
/// ## Identity-space contract (Task 1)
///
/// `DocId` is the **retrieval-universe** identity, a dense `u64` shared by
/// every lane that participates in pre-filtered fusion:
///
/// | Lane | Native key | Relationship to `DocId` |
/// |------|-----------|--------------------------|
/// | BM25 / inverted index | `u64` doc id | identical (`DocId(id)`) |
/// | Grep / trigram lane | `DocId = u64` alias | identical |
/// | `AllowedSet` membership | `u64` | identical (`AllowedSet::contains(d.get())`) |
/// | HNSW vector graph | `u128` storage id | **mapped** at the boundary |
///
/// The first three are the *same* key space, so an `AllowedSet` produced by one
/// lane gates every other lane with an O(1) membership test on `DocId.0`. The
/// HNSW graph keys on a wider `u128` *storage* id; the vector executor narrows
/// that to a retrieval `DocId` when it emits candidates (and accepts an
/// `allowed(u128)` predicate over the same mapping — see
/// `HnswIndex::search_allowed`). Threading the newtype down into the `u128`
/// storage layer is deliberately **out of scope** here: it touches the durable
/// id space and yields no behavioral change while this kernel keys on `DocId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DocId(pub u64);

impl DocId {
    /// The underlying retrieval-universe id.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for DocId {
    fn from(id: u64) -> Self {
        DocId(id)
    }
}

impl From<DocId> for u64 {
    fn from(d: DocId) -> Self {
        d.0
    }
}

/// A ranked candidate list paired with the weight it contributes to fusion.
///
/// Results must be ordered best-first; the element at index 0 is treated as
/// rank 1.
pub struct RankedList<'a> {
    /// Candidates ordered best-first.
    pub results: &'a [ScoredResult],
    /// Fusion weight for this list.
    pub weight: f32,
}

/// A modality's pre-filtered candidates paired with the weight it contributes
/// to N-ary fusion. This is the unit consumed by [`FusionEngine::fuse_multi`],
/// allowing an arbitrary number of lanes (vector, BM25, grep, …) to be fused by
/// a single call.
pub struct WeightedLane {
    /// Pre-filtered candidates for one modality.
    pub candidates: FilteredCandidates,
    /// Fusion weight for this lane.
    pub weight: f32,
}

/// The single, canonical Reciprocal Rank Fusion kernel: **weighted, 1-indexed,
/// N-ary**.
///
/// ```text
/// score(d) = Σᵢ  weightᵢ / (k + rankᵢ(d))
/// ```
///
/// where `rankᵢ(d)` is the **1-indexed** position of document `d` in list `i`
/// (the top result has rank 1) and `weightᵢ` is that list's weight. Every
/// higher-level fusion path funnels through this function so the weighting and
/// the rank offset can never diverge across the codebase again.
pub fn fuse_rrf_weighted(lists: &[RankedList<'_>], k: f32) -> HashMap<DocId, f32> {
    let mut scores: HashMap<DocId, f32> = HashMap::new();
    for list in lists {
        for (rank, result) in list.results.iter().enumerate() {
            let contribution = list.weight / (k + (rank as f32 + 1.0));
            *scores.entry(DocId(result.doc_id)).or_insert(0.0) += contribution;
        }
    }
    scores
}

// ============================================================================
// Fusion Engine
// ============================================================================

/// The fusion engine that combines candidates from multiple modalities
pub struct FusionEngine {
    config: FusionConfig,
}

impl FusionEngine {
    /// Create a new fusion engine
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Fuse candidates from vector and BM25 search
    ///
    /// INVARIANT: Both candidate sets are already filtered to AllowedSet.
    /// This function does NOT apply any additional filtering.
    ///
    /// This is the two-lane convenience over [`FusionEngine::fuse_multi`]: it
    /// builds weighted lanes from the configured method weights and delegates,
    /// so the two-lane and N-ary paths share exactly one scoring implementation.
    pub fn fuse(
        &self,
        vector_candidates: Option<FilteredCandidates>,
        bm25_candidates: Option<FilteredCandidates>,
    ) -> FusionResult {
        // Validate that candidates are pre-filtered
        if let Some(ref vc) = vector_candidates {
            debug_assert!(vc.filtered, "Vector candidates must be pre-filtered!");
        }
        if let Some(ref bc) = bm25_candidates {
            debug_assert!(bc.filtered, "BM25 candidates must be pre-filtered!");
        }

        // Cascade is intrinsically two-modality (primary filters, secondary
        // ranks) and keeps its dedicated path.
        if let FusionMethod::Cascade { primary } = self.config.method {
            return self.fuse_cascade(vector_candidates, bm25_candidates, primary);
        }

        let (vector_weight, bm25_weight) = self.method_weights();
        let mut lanes: Vec<WeightedLane> = Vec::with_capacity(2);
        if let Some(vc) = vector_candidates {
            lanes.push(WeightedLane {
                candidates: vc,
                weight: vector_weight,
            });
        }
        if let Some(bc) = bm25_candidates {
            lanes.push(WeightedLane {
                candidates: bc,
                weight: bm25_weight,
            });
        }
        self.fuse_multi(lanes)
    }

    /// The per-modality weights implied by the configured fusion method.
    ///
    /// RRF and Linear carry explicit vector/BM25 weights; Max and Cascade do
    /// not weight their inputs, so they report a neutral `1.0`.
    pub(crate) fn method_weights(&self) -> (f32, f32) {
        match self.config.method {
            FusionMethod::Rrf {
                vector_weight,
                bm25_weight,
                ..
            } => (vector_weight, bm25_weight),
            FusionMethod::Linear {
                vector_weight,
                bm25_weight,
            } => (vector_weight, bm25_weight),
            FusionMethod::Max | FusionMethod::Cascade { .. } => (1.0, 1.0),
        }
    }

    /// N-ary fusion across any number of pre-filtered modality lanes.
    ///
    /// This is the canonical multi-lane path: vector, BM25, and grep (or any
    /// future lane) are fused by a single call. RRF funnels through
    /// [`fuse_rrf_weighted`]; Linear and Max combine per-lane normalized scores
    /// weighted by each lane's weight.
    ///
    /// INVARIANT: every lane is already filtered to the AllowedSet. No
    /// additional filtering happens here.
    pub fn fuse_multi(&self, lanes: Vec<WeightedLane>) -> FusionResult {
        for lane in &lanes {
            debug_assert!(
                lane.candidates.filtered,
                "Fusion lanes must be pre-filtered!"
            );
        }

        match self.config.method {
            FusionMethod::Rrf { k, .. } => {
                let ranked: Vec<RankedList<'_>> = lanes
                    .iter()
                    .map(|lane| RankedList {
                        results: &lane.candidates.results,
                        weight: lane.weight,
                    })
                    .collect();
                let scores = fuse_rrf_weighted(&ranked, k)
                    .into_iter()
                    .map(|(doc, score)| (doc.0, score))
                    .collect();
                self.collect_top_k(scores)
            }
            FusionMethod::Linear { .. } => {
                let mut scores: HashMap<u64, f32> = HashMap::new();
                for lane in &lanes {
                    for (doc_id, score) in self.normalize_scores(&lane.candidates.results) {
                        *scores.entry(doc_id).or_insert(0.0) += score * lane.weight;
                    }
                }
                self.collect_top_k(scores)
            }
            FusionMethod::Max => {
                let mut scores: HashMap<u64, f32> = HashMap::new();
                for lane in &lanes {
                    for (doc_id, score) in self.normalize_scores(&lane.candidates.results) {
                        let entry = scores.entry(doc_id).or_insert(0.0);
                        *entry = entry.max(score);
                    }
                }
                self.collect_top_k(scores)
            }
            FusionMethod::Cascade { primary } => {
                // Cascade is two-modality: reconstruct the vector and BM25 lanes
                // by modality and apply the primary/secondary logic. A grep Rank
                // lane is not part of a cascade and is ignored here (grep's
                // cascade shape is Gate, applied before fusion in `execute`).
                let mut vector = None;
                let mut bm25 = None;
                for lane in lanes {
                    match lane.candidates.modality {
                        Modality::Vector => vector = Some(lane.candidates),
                        Modality::Bm25 => bm25 = Some(lane.candidates),
                        Modality::Grep => {}
                    }
                }
                self.fuse_cascade(vector, bm25, primary)
            }
        }
    }

    /// Cascade fusion: use primary modality to filter, secondary to rank
    fn fuse_cascade(
        &self,
        vector: Option<FilteredCandidates>,
        bm25: Option<FilteredCandidates>,
        primary: Modality,
    ) -> FusionResult {
        let (primary_candidates, secondary_candidates) = match primary {
            Modality::Vector => (vector, bm25),
            Modality::Bm25 => (bm25, vector),
            // Grep is not a cascade ranking modality (its cascade shape is the
            // Gate, applied to the AllowedSet before fusion). Fall back to a
            // vector-primary cascade so the method stays total.
            Modality::Grep => (vector, bm25),
        };

        // Get primary doc IDs
        let primary_ids: std::collections::HashSet<u64> = primary_candidates
            .as_ref()
            .map(|c| c.results.iter().map(|r| r.doc_id).collect())
            .unwrap_or_default();

        // Score by secondary, but only docs in primary
        let mut scores: HashMap<u64, f32> = HashMap::new();

        if let Some(sc) = secondary_candidates {
            for result in &sc.results {
                if primary_ids.contains(&result.doc_id) {
                    scores.insert(result.doc_id, result.score);
                }
            }
        }

        // If secondary doesn't score some docs, use primary order
        if let Some(pc) = primary_candidates {
            for (rank, result) in pc.results.iter().enumerate() {
                scores.entry(result.doc_id).or_insert(-(rank as f32));
            }
        }

        self.collect_top_k(scores)
    }

    /// Normalize scores to [0, 1] using min-max normalization
    fn normalize_scores(&self, results: &[ScoredResult]) -> Vec<(u64, f32)> {
        if results.is_empty() {
            return vec![];
        }

        let min = results
            .iter()
            .map(|r| r.score)
            .fold(f32::INFINITY, f32::min);
        let max = results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range == 0.0 {
            return results.iter().map(|r| (r.doc_id, 1.0)).collect();
        }

        results
            .iter()
            .map(|r| (r.doc_id, (r.score - min) / range))
            .collect()
    }

    /// Collect top-k results from score map
    fn collect_top_k(&self, scores: HashMap<u64, f32>) -> FusionResult {
        let mut results: Vec<ScoredResult> = scores
            .into_iter()
            .map(|(doc_id, score)| ScoredResult::new(doc_id, score))
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply min_score filter
        if let Some(min) = self.config.min_score {
            results.retain(|r| r.score >= min);
        }

        // Truncate to k
        results.truncate(self.config.final_k);

        FusionResult {
            results,
            method: self.config.method,
        }
    }
}

/// Result of fusion
#[derive(Debug)]
pub struct FusionResult {
    /// Final ranked results
    pub results: Vec<ScoredResult>,
    /// Method used
    pub method: FusionMethod,
}

// ============================================================================
// Unified Hybrid Executor
// ============================================================================

/// Trait for vector search executor
pub trait VectorExecutor {
    fn search(&self, query: &[f32], k: usize, allowed: &AllowedSet) -> Vec<ScoredResult>;
}

/// Trait for BM25 executor
pub trait Bm25Executor {
    fn search(&self, query: &str, k: usize, allowed: &AllowedSet) -> Vec<ScoredResult>;
}

/// Trait for the grep (trigram-accelerated regex) lane.
///
/// Implementations MUST honor `allowed` — every returned id must be a member
/// (the same `result ⊆ allowed` contract the other lanes enforce). For
/// [`GrepMode::Rank`] the scores rank documents by match density; for
/// [`GrepMode::Gate`] only the returned doc-ids matter (they form the cascade
/// gate) and the scores are not meaningful.
pub trait GrepLaneExecutor {
    fn grep(
        &self,
        pattern: &str,
        k: usize,
        allowed: &AllowedSet,
        mode: GrepMode,
    ) -> Vec<ScoredResult>;
}

/// The unified hybrid executor
///
/// This is the main entry point that enforces the "no post-filtering" contract.
pub struct UnifiedHybridExecutor<V: VectorExecutor, B: Bm25Executor> {
    vector_executor: Arc<V>,
    bm25_executor: Arc<B>,
    grep_executor: Option<Arc<dyn GrepLaneExecutor>>,
    fusion_engine: FusionEngine,
}

impl<V: VectorExecutor, B: Bm25Executor> UnifiedHybridExecutor<V, B> {
    /// Create a new executor
    pub fn new(
        vector_executor: Arc<V>,
        bm25_executor: Arc<B>,
        fusion_config: FusionConfig,
    ) -> Self {
        Self {
            vector_executor,
            bm25_executor,
            grep_executor: None,
            fusion_engine: FusionEngine::new(fusion_config),
        }
    }

    /// Attach a grep lane executor, enabling three-lane fusion.
    ///
    /// Without one, any `grep_query` on a [`UnifiedHybridQuery`] is ignored and
    /// the executor behaves exactly as the two-lane vector+BM25 path.
    pub fn with_grep_executor(mut self, grep_executor: Arc<dyn GrepLaneExecutor>) -> Self {
        self.grep_executor = Some(grep_executor);
        self
    }

    /// Execute a hybrid query with mandatory pre-filtering
    ///
    /// # Contract
    ///
    /// 1. Computes `effective_filter = auth_scope ∧ query_filter`
    /// 2. Converts to `AllowedSet` (via metadata index)
    /// 3. A grep `Gate` lane (if present) narrows that `AllowedSet` *first*
    /// 4. Passes the SAME `AllowedSet` to the vector, BM25 and grep-`Rank` lanes
    /// 5. Fuses all already-filtered lanes with one N-ary `fuse_multi` call
    ///
    /// NO POST-FILTERING occurs in this function.
    pub fn execute(
        &self,
        query: &UnifiedHybridQuery,
        _auth_scope: &AuthScope,
        allowed_set: &AllowedSet, // Pre-computed from FilterIR + AuthScope
    ) -> FusionResult {
        // Short-circuit if empty
        if allowed_set.is_empty() {
            return FusionResult {
                results: vec![],
                method: self.fusion_engine.config.method,
            };
        }

        let k = self.fusion_engine.config.candidates_per_modality;

        // ---- Lane 3 (grep) planning -------------------------------------
        // Gate: run grep first and intersect its matches into the AllowedSet so
        //       the other lanes only ever see grep-approved documents.
        // Rank: run grep as an additional ranked lane alongside vector/BM25.
        let mut grep_rank: Option<FilteredCandidates> = None;
        let mut grep_weight = 1.0_f32;
        let mut gated: Option<AllowedSet> = None;
        if let (Some(gq), Some(grep)) = (query.grep_query.as_ref(), self.grep_executor.as_ref()) {
            match gq.mode {
                GrepMode::Gate => {
                    // `k = 0` = unlimited: the gate must be the full match set.
                    let hits = grep.grep(&gq.pattern, 0, allowed_set, GrepMode::Gate);
                    gated = Some(AllowedSet::from_iter(hits.into_iter().map(|r| r.doc_id)));
                }
                GrepMode::Rank => {
                    let hits = grep.grep(&gq.pattern, k, allowed_set, GrepMode::Rank);
                    grep_rank = Some(FilteredCandidates::from_grep(hits));
                    grep_weight = gq.weight;
                }
            }
        }

        // The effective AllowedSet every other lane is gated by.
        let effective_allowed: &AllowedSet = gated.as_ref().unwrap_or(allowed_set);
        if effective_allowed.is_empty() {
            return FusionResult {
                results: vec![],
                method: self.fusion_engine.config.method,
            };
        }

        // Vector search (with the effective AllowedSet)
        let vector_candidates = query.vector_query.as_ref().map(|vq| {
            let results = self
                .vector_executor
                .search(&vq.embedding, k, effective_allowed);
            FilteredCandidates::from_vector(results)
        });

        // BM25 search (with the SAME effective AllowedSet)
        let bm25_candidates = query.bm25_query.as_ref().map(|bq| {
            let results = self.bm25_executor.search(&bq.text, k, effective_allowed);
            FilteredCandidates::from_bm25(results)
        });

        // ---- Fuse all lanes with a single N-ary call --------------------
        let (vector_weight, bm25_weight) = self.fusion_engine.method_weights();
        let mut lanes: Vec<WeightedLane> = Vec::with_capacity(3);
        if let Some(vc) = vector_candidates {
            lanes.push(WeightedLane {
                candidates: vc,
                weight: vector_weight,
            });
        }
        if let Some(bc) = bm25_candidates {
            lanes.push(WeightedLane {
                candidates: bc,
                weight: bm25_weight,
            });
        }
        if let Some(gc) = grep_rank {
            lanes.push(WeightedLane {
                candidates: gc,
                weight: grep_weight,
            });
        }

        self.fusion_engine.fuse_multi(lanes)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let config = FusionConfig {
            method: FusionMethod::Rrf {
                k: 60.0,
                vector_weight: 1.0,
                bm25_weight: 1.0,
            },
            candidates_per_modality: 10,
            final_k: 5,
            min_score: None,
        };

        let engine = FusionEngine::new(config);

        let vector = FilteredCandidates::from_vector(vec![
            ScoredResult::new(1, 0.9),
            ScoredResult::new(2, 0.8),
            ScoredResult::new(3, 0.7),
        ]);

        let bm25 = FilteredCandidates::from_bm25(vec![
            ScoredResult::new(2, 5.0), // doc 2 is in both
            ScoredResult::new(4, 4.0),
            ScoredResult::new(1, 3.0), // doc 1 is in both
        ]);

        let result = engine.fuse(Some(vector), Some(bm25));

        // Doc 2 should score highest (rank 2 in vector, rank 1 in BM25)
        // Doc 1 should also score well (rank 1 in vector, rank 3 in BM25)
        assert!(!result.results.is_empty());

        // Docs 1 and 2 should be near the top
        let top_ids: Vec<u64> = result.results.iter().map(|r| r.doc_id).collect();
        assert!(top_ids.contains(&1));
        assert!(top_ids.contains(&2));
    }

    #[test]
    fn test_fuse_rrf_weighted_is_1_indexed_and_weighted() {
        // Single list: the rank-1 document must score weight / (k + 1), proving
        // the kernel is 1-indexed (top result is rank 1, not 0) and honors the
        // per-list weight.
        let k = 60.0_f32;
        let docs = [ScoredResult::new(7, 0.9), ScoredResult::new(8, 0.5)];
        let scores = fuse_rrf_weighted(
            &[RankedList {
                results: &docs,
                weight: 2.0,
            }],
            k,
        );

        let s7 = scores[&DocId(7)];
        let s8 = scores[&DocId(8)];
        assert!(
            (s7 - 2.0 / (k + 1.0)).abs() < 1e-6,
            "rank-1 must use 1-indexed weighted score"
        );
        assert!(
            (s8 - 2.0 / (k + 2.0)).abs() < 1e-6,
            "rank-2 must use 1-indexed weighted score"
        );
        assert!(s7 > s8, "earlier rank must score higher");

        // A document present in two weighted lists accumulates both contributions.
        let list_a = [ScoredResult::new(1, 0.0)];
        let list_b = [ScoredResult::new(1, 0.0)];
        let merged = fuse_rrf_weighted(
            &[
                RankedList {
                    results: &list_a,
                    weight: 1.0,
                },
                RankedList {
                    results: &list_b,
                    weight: 3.0,
                },
            ],
            k,
        );
        let expected = 1.0 / (k + 1.0) + 3.0 / (k + 1.0);
        assert!(
            (merged[&DocId(1)] - expected).abs() < 1e-6,
            "weights must sum across lists"
        );
    }

    #[test]
    fn test_linear_fusion() {
        let config = FusionConfig {
            method: FusionMethod::Linear {
                vector_weight: 0.6,
                bm25_weight: 0.4,
            },
            candidates_per_modality: 10,
            final_k: 5,
            min_score: None,
        };

        let engine = FusionEngine::new(config);

        let vector = FilteredCandidates::from_vector(vec![
            ScoredResult::new(1, 1.0),
            ScoredResult::new(2, 0.5),
        ]);

        let bm25 = FilteredCandidates::from_bm25(vec![
            ScoredResult::new(2, 10.0), // Different scale
            ScoredResult::new(3, 5.0),
        ]);

        let result = engine.fuse(Some(vector), Some(bm25));

        // After normalization, doc 2 should benefit from both
        assert!(!result.results.is_empty());
    }

    #[test]
    fn test_empty_allowed_set() {
        let config = FusionConfig::default();
        let engine = FusionEngine::new(config);

        // No candidates = empty result
        let result = engine.fuse(None, None);
        assert!(result.results.is_empty());
    }

    #[test]
    fn test_score_normalization() {
        let config = FusionConfig::default();
        let engine = FusionEngine::new(config);

        let results = vec![
            ScoredResult::new(1, 100.0),
            ScoredResult::new(2, 50.0),
            ScoredResult::new(3, 0.0),
        ];

        let normalized = engine.normalize_scores(&results);

        // Should be normalized to [0, 1]
        assert_eq!(normalized.len(), 3);
        let scores: HashMap<u64, f32> = normalized.into_iter().collect();
        assert!((scores[&1] - 1.0).abs() < 0.001);
        assert!((scores[&2] - 0.5).abs() < 0.001);
        assert!((scores[&3] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_no_post_filter_invariant() {
        // This test verifies the core invariant:
        // result-set ⊆ allowed-set
        //
        // If this invariant is violated, it indicates a security issue.

        let allowed: std::collections::HashSet<u64> = [1, 2, 3, 5, 8].into_iter().collect();
        let allowed_set = AllowedSet::from_iter(allowed.iter().copied());

        // Simulate filtered candidates (these should already respect AllowedSet)
        let vector = FilteredCandidates::from_vector(vec![
            ScoredResult::new(1, 0.9), // in allowed set
            ScoredResult::new(2, 0.8), // in allowed set
            ScoredResult::new(5, 0.7), // in allowed set
        ]);

        let bm25 = FilteredCandidates::from_bm25(vec![
            ScoredResult::new(2, 5.0), // in allowed set
            ScoredResult::new(3, 4.0), // in allowed set
            ScoredResult::new(8, 3.0), // in allowed set
        ]);

        let config = FusionConfig::default();
        let engine = FusionEngine::new(config);
        let result = engine.fuse(Some(vector), Some(bm25));

        // INVARIANT: Every result doc_id must be in the allowed set
        for doc in &result.results {
            assert!(
                allowed_set.contains(doc.doc_id),
                "INVARIANT VIOLATION: doc_id {} not in allowed set",
                doc.doc_id
            );
        }
    }

    // ---- Three-lane fusion: grep (Task 5) wired into hybrid (Task 7) -------

    use crate::grep_executor::GrepMode;
    use crate::namespace::Namespace;
    use crate::trigram_index::TrigramIndex;

    struct MockVector(Vec<ScoredResult>);
    impl VectorExecutor for MockVector {
        fn search(&self, _q: &[f32], k: usize, allowed: &AllowedSet) -> Vec<ScoredResult> {
            self.0
                .iter()
                .filter(|r| allowed.contains(r.doc_id))
                .take(k)
                .cloned()
                .collect()
        }
    }

    struct MockBm25(Vec<ScoredResult>);
    impl Bm25Executor for MockBm25 {
        fn search(&self, _q: &str, k: usize, allowed: &AllowedSet) -> Vec<ScoredResult> {
            self.0
                .iter()
                .filter(|r| allowed.contains(r.doc_id))
                .take(k)
                .cloned()
                .collect()
        }
    }

    /// Grep lane backed by the real trigram index + grep executor — this proves
    /// the wiring drives the actual Task 5 machinery, not a stub.
    struct RealGrep {
        index: TrigramIndex,
    }
    impl GrepLaneExecutor for RealGrep {
        fn grep(
            &self,
            pattern: &str,
            k: usize,
            allowed: &AllowedSet,
            mode: GrepMode,
        ) -> Vec<ScoredResult> {
            let exec = crate::grep_executor::GrepExecutor::new(&self.index);
            match exec.search(pattern, allowed, k, mode) {
                Ok(results) => results
                    .hits
                    .into_iter()
                    .map(|h| ScoredResult::new(h.doc_id, h.score))
                    .collect(),
                Err(_) => Vec::new(),
            }
        }
    }

    fn test_query() -> UnifiedHybridQuery {
        UnifiedHybridQuery::new(NamespaceScope::single(Namespace::new("test").unwrap()))
    }

    fn grep_index() -> TrigramIndex {
        let mut idx = TrigramIndex::new();
        idx.insert(1, "fn alpha() { compute_idf() }");
        idx.insert(2, "fn beta() { unrelated helper }");
        idx.insert(3, "fn gamma() { compute_idf() twice compute_idf() }");
        idx.insert(4, "struct Config { compute_idf: bool }");
        idx
    }

    #[test]
    fn test_three_lane_rank_fusion_respects_allowed_set() {
        // Vector + BM25 favor docs {1,2,4}; grep(Rank) for "compute_idf" finds
        // {1,3,4}. Fusing all three must (a) stay within the allowed set and
        // (b) surface doc 3, which only the grep lane contributes.
        let vector = MockVector(vec![
            ScoredResult::new(2, 0.9),
            ScoredResult::new(1, 0.8),
            ScoredResult::new(4, 0.2),
        ]);
        let bm25 = MockBm25(vec![ScoredResult::new(2, 5.0), ScoredResult::new(1, 3.0)]);
        let grep = RealGrep {
            index: grep_index(),
        };

        let allowed = AllowedSet::from_iter([1, 2, 3, 4]);
        let executor =
            UnifiedHybridExecutor::new(Arc::new(vector), Arc::new(bm25), FusionConfig::default())
                .with_grep_executor(Arc::new(grep));

        let query = test_query()
            .with_vector(vec![0.0; 4])
            .with_bm25("anything")
            .with_grep("compute_idf", GrepMode::Rank);

        let result = executor.execute(&query, &AuthScope::for_namespace("test"), &allowed);

        assert!(!result.results.is_empty());
        for r in &result.results {
            assert!(
                allowed.contains(r.doc_id),
                "result {} escaped allowed set",
                r.doc_id
            );
        }
        let ids: Vec<u64> = result.results.iter().map(|r| r.doc_id).collect();
        assert!(
            ids.contains(&3),
            "grep-only doc 3 should appear via the third lane, got {ids:?}"
        );
    }

    #[test]
    fn test_grep_gate_narrows_before_other_lanes() {
        // grep(Gate) for "compute_idf" matches {1,3,4}. Even though the vector
        // lane ranks doc 2 first, the gate must exclude it entirely (cascade).
        let vector = MockVector(vec![
            ScoredResult::new(2, 0.9),
            ScoredResult::new(1, 0.8),
            ScoredResult::new(4, 0.7),
            ScoredResult::new(3, 0.6),
        ]);
        let bm25 = MockBm25(vec![ScoredResult::new(2, 5.0)]);
        let grep = RealGrep {
            index: grep_index(),
        };

        let allowed = AllowedSet::from_iter([1, 2, 3, 4]);
        let executor =
            UnifiedHybridExecutor::new(Arc::new(vector), Arc::new(bm25), FusionConfig::default())
                .with_grep_executor(Arc::new(grep));

        let query = test_query()
            .with_vector(vec![0.0; 4])
            .with_bm25("anything")
            .with_grep("compute_idf", GrepMode::Gate);

        let result = executor.execute(&query, &AuthScope::for_namespace("test"), &allowed);

        assert!(!result.results.is_empty());
        let gate: std::collections::HashSet<u64> = [1, 3, 4].into_iter().collect();
        for r in &result.results {
            assert!(
                gate.contains(&r.doc_id),
                "doc {} not in grep gate {{1,3,4}}",
                r.doc_id
            );
        }
        assert!(
            !result.results.iter().any(|r| r.doc_id == 2),
            "doc 2 (no compute_idf) must be gated out"
        );
    }

    #[test]
    fn test_grep_query_ignored_without_grep_executor() {
        // A grep_query with no configured grep executor degrades cleanly to the
        // two-lane vector+BM25 path (no panic, grep simply absent).
        let vector = MockVector(vec![ScoredResult::new(1, 0.9)]);
        let bm25 = MockBm25(vec![ScoredResult::new(2, 5.0)]);
        let allowed = AllowedSet::from_iter([1, 2, 3, 4]);
        let executor =
            UnifiedHybridExecutor::new(Arc::new(vector), Arc::new(bm25), FusionConfig::default());

        let query = test_query()
            .with_vector(vec![0.0; 4])
            .with_bm25("anything")
            .with_grep("compute_idf", GrepMode::Gate);

        let result = executor.execute(&query, &AuthScope::for_namespace("test"), &allowed);
        let ids: Vec<u64> = result.results.iter().map(|r| r.doc_id).collect();
        assert!(
            ids.contains(&1) && ids.contains(&2),
            "without a grep executor both lanes survive, got {ids:?}"
        );
    }
}

// ============================================================================
// Invariant Verification
// ============================================================================

/// Verify that a fusion result respects the no-post-filtering invariant
///
/// This function should be used in tests and optionally in debug builds
/// to verify that the security invariant holds.
///
/// # Invariant
///
/// `∀ doc ∈ result: doc.id ∈ allowed_set`
///
/// This is the "monotone property" from the architecture document.
pub fn verify_no_post_filter_invariant(
    result: &FusionResult,
    allowed_set: &AllowedSet,
) -> InvariantVerification {
    let mut violations = Vec::new();

    for doc in &result.results {
        if !allowed_set.contains(doc.doc_id) {
            violations.push(doc.doc_id);
        }
    }

    if violations.is_empty() {
        InvariantVerification::Valid
    } else {
        InvariantVerification::Violated {
            doc_ids: violations,
        }
    }
}

/// Result of invariant verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvariantVerification {
    /// Invariant holds
    Valid,
    /// Invariant violated - these doc IDs should not be in results
    Violated { doc_ids: Vec<u64> },
}

impl InvariantVerification {
    /// Check if the invariant holds
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Panic if the invariant is violated (for testing)
    pub fn assert_valid(&self) {
        match self {
            Self::Valid => {}
            Self::Violated { doc_ids } => {
                panic!(
                    "NO-POST-FILTER INVARIANT VIOLATED: {} docs not in allowed set: {:?}",
                    doc_ids.len(),
                    doc_ids
                );
            }
        }
    }
}
