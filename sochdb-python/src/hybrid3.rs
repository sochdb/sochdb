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

//! Three-lane hybrid retrieval binding (grep + BM25 + HNSW → RRF).
//!
//! This exposes the real `sochdb_query::UnifiedHybridExecutor` to Python so the
//! benchmark harness can exercise the full fusion path — not the legacy
//! two-lane Python glue. The three lanes are concrete bridges over the same
//! engines used elsewhere:
//!
//! - **vector**: `sochdb_index::hnsw::HnswIndex::search_allowed` (in-traversal
//!   AllowedSet filtering)
//! - **BM25**: `sochdb_query::DisjunctiveBm25Executor` over a concrete
//!   `InvertedIndex` built here (OR semantics, AllowedSet pushdown)
//! - **grep**: `sochdb_query::GrepExecutor` over a `TrigramIndex`
//!
//! All three receive the SAME `AllowedSet` and the fusion never post-filters.

use std::collections::HashMap;
use std::sync::Arc;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};
use sochdb_index::vector_quantized::Precision;

use sochdb_query::bm25_filtered::{
    Bm25Params, DisjunctiveBm25Executor, InvertedIndex, PostingList,
};
use sochdb_query::candidate_gate::AllowedSet;
use sochdb_query::filter_ir::AuthScope;
use sochdb_query::filtered_vector_search::ScoredResult;
use sochdb_query::grep_executor::{GrepExecutor, GrepMode};
use sochdb_query::namespace::{Namespace, NamespaceScope};
use sochdb_query::trigram_index::TrigramIndex;
use sochdb_query::unified_fusion::FusionMethod;
use sochdb_query::unified_fusion::{
    Bm25Executor, FusionConfig, GrepLaneExecutor, UnifiedHybridExecutor, UnifiedHybridQuery,
    VectorExecutor,
};

// ============================================================================
// Tokenization (shared by index + query side)
// ============================================================================

/// Normalize free text into BM25 tokens: lowercase, split on non-alphanumeric,
/// drop tokens shorter than 2 chars. The query side normalizes identically and
/// joins with single spaces so the executor's internal `split_whitespace`
/// tokenizer sees the same tokens that were indexed.
fn normalize_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            if cur.len() >= 2 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

// ============================================================================
// Concrete InvertedIndex for the BM25 lane
// ============================================================================

/// In-memory inverted index built from the corpus, implementing the
/// `sochdb_query::InvertedIndex` trait so it can drive the real filtered BM25
/// executors.
struct SimpleInvertedIndex {
    /// term -> sorted (doc_id, term_frequency)
    postings: HashMap<String, Vec<(u64, u32)>>,
    /// doc_id -> document length in tokens
    doc_len: HashMap<u64, u32>,
    params: Bm25Params,
}

impl SimpleInvertedIndex {
    fn build(doc_ids: &[u64], texts: &[String]) -> Self {
        let mut postings: HashMap<String, HashMap<u64, u32>> = HashMap::new();
        let mut doc_len: HashMap<u64, u32> = HashMap::new();
        let mut total_len: u64 = 0;

        for (&doc_id, text) in doc_ids.iter().zip(texts.iter()) {
            let tokens = normalize_tokens(text);
            doc_len.insert(doc_id, tokens.len() as u32);
            total_len += tokens.len() as u64;
            for tok in tokens {
                *postings.entry(tok).or_default().entry(doc_id).or_insert(0) += 1;
            }
        }

        // Flatten + sort each posting list by doc_id (the executors expect
        // ascending doc-id postings).
        let postings: HashMap<String, Vec<(u64, u32)>> = postings
            .into_iter()
            .map(|(term, by_doc)| {
                let mut entries: Vec<(u64, u32)> = by_doc.into_iter().collect();
                entries.sort_unstable_by_key(|(id, _)| *id);
                (term, entries)
            })
            .collect();

        let n = doc_ids.len().max(1) as f32;
        let avgdl = if doc_ids.is_empty() {
            1.0
        } else {
            (total_len as f32 / n).max(1.0)
        };

        let params = Bm25Params {
            k1: 1.2,
            b: 0.75,
            avgdl,
            total_docs: doc_ids.len() as u64,
        };

        Self {
            postings,
            doc_len,
            params,
        }
    }
}

impl InvertedIndex for SimpleInvertedIndex {
    fn get_posting_list(&self, term: &str) -> Option<PostingList> {
        self.postings
            .get(term)
            .map(|entries| PostingList::new(term, entries.clone()))
    }

    fn get_doc_length(&self, doc_id: u64) -> Option<u32> {
        self.doc_len.get(&doc_id).copied()
    }

    fn get_params(&self) -> &Bm25Params {
        &self.params
    }
}

// ============================================================================
// Lane executors
// ============================================================================

/// Vector lane: HNSW with in-traversal AllowedSet filtering.
struct VectorLane {
    hnsw: Arc<HnswIndex>,
    ef: usize,
}

impl VectorExecutor for VectorLane {
    fn search(&self, query: &[f32], k: usize, allowed: &AllowedSet) -> Vec<ScoredResult> {
        match self
            .hnsw
            .search_allowed(query, k, self.ef, |id| allowed.contains(id as u64))
        {
            Ok(hits) => hits
                .into_iter()
                // Convert distance (lower is better) into a similarity score
                // (higher is better); ordering is already nearest-first.
                .map(|(id, dist)| ScoredResult::new(id as u64, 1.0 / (1.0 + dist)))
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// BM25 lane: disjunctive (OR) filtered BM25 over the concrete inverted index.
struct Bm25Lane {
    exec: DisjunctiveBm25Executor<SimpleInvertedIndex>,
}

impl Bm25Executor for Bm25Lane {
    fn search(&self, query: &str, k: usize, allowed: &AllowedSet) -> Vec<ScoredResult> {
        self.exec.search(query, k, allowed)
    }
}

/// Grep lane: trigram-accelerated regex over the corpus.
struct GrepLane {
    index: Arc<TrigramIndex>,
}

impl GrepLaneExecutor for GrepLane {
    fn grep(
        &self,
        pattern: &str,
        k: usize,
        allowed: &AllowedSet,
        mode: GrepMode,
    ) -> Vec<ScoredResult> {
        let exec = GrepExecutor::new(&self.index);
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

// ============================================================================
// Python class
// ============================================================================

/// Three-lane hybrid retrieval index (HNSW + BM25 + grep → RRF fusion).
///
/// Unlike the pure-Python `HybridSearchIndex`, this drives the native
/// `UnifiedHybridExecutor`: all lanes share one `AllowedSet` and fusion never
/// post-filters. It exists primarily so benchmarks can measure the real
/// three-lane path.
#[pyclass(name = "ThreeLaneHybridIndex")]
pub struct PyThreeLaneHybridIndex {
    dimension: usize,
    ef_search: usize,
    hnsw: Arc<HnswIndex>,
    bm25: Option<Arc<SimpleInvertedIndex>>,
    trigram: Arc<TrigramIndex>,
    /// numeric doc-id (1-based) -> original string id
    id_map: Vec<String>,
}

#[pymethods]
impl PyThreeLaneHybridIndex {
    /// Create a new three-lane hybrid index.
    ///
    /// Args:
    ///     dimension: Embedding dimension.
    ///     m: HNSW max connections per node (default 32, matching the engine
    ///        HnswConfig::default; m=16 capped recall ~0.86 on 768D+ data).
    ///     ef_construction: HNSW construction beam width (default 256).
    ///     ef_search: HNSW query beam width (default 128).
    ///     metric: Distance metric ("cosine", "euclidean", "dot").
    #[new]
    #[pyo3(signature = (dimension, m=32, ef_construction=256, ef_search=128, metric="cosine"))]
    fn new(
        dimension: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
    ) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("dimension must be > 0"));
        }
        let distance_metric = match metric.to_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "euclidean" | "l2" => DistanceMetric::Euclidean,
            "dot" | "dot_product" | "inner_product" => DistanceMetric::DotProduct,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown metric: {other}. Use 'cosine', 'euclidean', or 'dot'"
                )));
            }
        };

        let config = HnswConfig {
            max_connections: m,
            max_connections_layer0: m * 2,
            level_multiplier: 1.0 / (m as f32).ln(),
            ef_construction,
            metric: distance_metric,
            quantization_precision: Some(Precision::F32),
            ..Default::default()
        };

        Ok(Self {
            dimension,
            ef_search,
            hnsw: Arc::new(HnswIndex::new(dimension, config)),
            bm25: None,
            trigram: Arc::new(TrigramIndex::new()),
            id_map: Vec::new(),
        })
    }

    /// Build all three lanes over the same corpus.
    ///
    /// Args:
    ///     doc_ids: List of string document ids (length N).
    ///     texts: List of document texts (length N), used for BM25 + grep.
    ///     embeddings: float32 array of shape (N, dimension) for HNSW.
    fn build<'py>(
        &mut self,
        py: Python<'py>,
        doc_ids: Vec<String>,
        texts: Vec<String>,
        embeddings: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<usize> {
        let n = doc_ids.len();
        if texts.len() != n {
            return Err(PyValueError::new_err(
                "doc_ids and texts must have the same length",
            ));
        }
        let shape = embeddings.shape();
        if shape[0] != n {
            return Err(PyValueError::new_err(format!(
                "embeddings rows {} != doc count {}",
                shape[0], n
            )));
        }
        if shape[1] != self.dimension {
            return Err(PyValueError::new_err(format!(
                "embedding dimension {} != index dimension {}",
                shape[1], self.dimension
            )));
        }
        if !embeddings.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "embeddings must be C-contiguous (use np.ascontiguousarray)",
            ));
        }

        // Numeric ids are 1..=N; id_map[numeric-1] = original string id.
        self.id_map = doc_ids;
        let numeric_ids: Vec<u64> = (1..=n as u64).collect();

        // ---- BM25 lane: concrete inverted index ----
        let bm25 = SimpleInvertedIndex::build(&numeric_ids, &texts);
        self.bm25 = Some(Arc::new(bm25));

        // ---- Grep lane: trigram index ----
        let mut trigram = TrigramIndex::new();
        for (&id, text) in numeric_ids.iter().zip(texts.iter()) {
            trigram.insert(id, text);
        }
        self.trigram = Arc::new(trigram);

        // ---- Vector lane: HNSW (release the GIL for the heavy build) ----
        let vec_slice = embeddings
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("embeddings not contiguous: {e}")))?;
        let d = self.dimension;
        let hnsw = Arc::clone(&self.hnsw);
        let ids = numeric_ids.clone();
        py.allow_threads(move || hnsw.insert_batch_contiguous_u64(&ids, vec_slice, d))
            .map_err(|e| PyValueError::new_err(format!("HNSW build failed: {e}")))?;

        Ok(n)
    }

    /// Run a three-lane hybrid search.
    ///
    /// Args:
    ///     query_embedding: float32 vector of length `dimension`.
    ///     query_text: Free text for the BM25 lane (normalized internally).
    ///     k: Number of results to return.
    ///     grep_pattern: Optional regex for the grep lane.
    ///     grep_mode: "rank" (third RRF lane) or "gate" (cascade pre-filter).
    ///     method: Fusion method ("rrf", "linear", "max").
    ///     vector_weight / bm25_weight / grep_weight: Per-lane weights.
    ///     rrf_k: RRF k constant (default 60).
    ///
    /// Returns:
    ///     List of (doc_id, score) tuples, best first.
    #[pyo3(signature = (
        query_embedding,
        query_text,
        k=10,
        grep_pattern=None,
        grep_mode="rank",
        method="rrf",
        vector_weight=1.0,
        bm25_weight=1.0,
        grep_weight=1.0,
        rrf_k=60.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &self,
        query_embedding: PyReadonlyArray1<'py, f32>,
        query_text: &str,
        k: usize,
        grep_pattern: Option<&str>,
        grep_mode: &str,
        method: &str,
        vector_weight: f32,
        bm25_weight: f32,
        grep_weight: f32,
        rrf_k: f32,
    ) -> PyResult<Vec<(String, f32)>> {
        let bm25 = self
            .bm25
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built yet; call build() first"))?;

        let embedding = query_embedding
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("query not contiguous: {e}")))?;
        if embedding.len() != self.dimension {
            return Err(PyValueError::new_err(format!(
                "query dimension {} != index dimension {}",
                embedding.len(),
                self.dimension
            )));
        }

        let fusion_method = match method.to_lowercase().as_str() {
            "rrf" => FusionMethod::Rrf {
                k: rrf_k,
                vector_weight,
                bm25_weight,
            },
            "linear" => FusionMethod::Linear {
                vector_weight,
                bm25_weight,
            },
            "max" => FusionMethod::Max,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown fusion method: {other}. Use 'rrf', 'linear', or 'max'"
                )));
            }
        };

        // Pull more candidates per lane than the final k so fusion has overlap.
        let per_lane = (k * 10).max(k + 50);
        let fusion_config = FusionConfig {
            method: fusion_method,
            candidates_per_modality: per_lane,
            final_k: k,
            min_score: None,
        };

        let vector_lane = Arc::new(VectorLane {
            hnsw: Arc::clone(&self.hnsw),
            ef: self.ef_search.max(per_lane),
        });
        let bm25_lane = Arc::new(Bm25Lane {
            exec: DisjunctiveBm25Executor::new(Arc::clone(bm25)),
        });
        let grep_lane = Arc::new(GrepLane {
            index: Arc::clone(&self.trigram),
        });

        let executor = UnifiedHybridExecutor::new(vector_lane, bm25_lane, fusion_config)
            .with_grep_executor(grep_lane);

        // Build the query. Namespace is required by the API but unconstrained
        // here (the benchmark has a single namespace).
        let namespace = Namespace::new("bench")
            .map_err(|e| PyValueError::new_err(format!("namespace error: {e:?}")))?;
        let mut query = UnifiedHybridQuery::new(NamespaceScope::single(namespace))
            .with_vector(embedding.to_vec())
            .with_bm25(normalize_tokens(query_text).join(" "));

        if let Some(pattern) = grep_pattern {
            let mode = match grep_mode.to_lowercase().as_str() {
                "rank" => GrepMode::Rank,
                "gate" => GrepMode::Gate,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown grep_mode: {other}. Use 'rank' or 'gate'"
                    )));
                }
            };
            query = query.with_grep_weighted(pattern, mode, grep_weight);
        }

        // No metadata filter in the benchmark → all documents are allowed. The
        // grep Gate lane (if any) narrows this inside execute().
        let result = executor.execute(&query, &AuthScope::for_namespace("bench"), &AllowedSet::All);

        let out: Vec<(String, f32)> = result
            .results
            .into_iter()
            .filter_map(|r| {
                let idx = (r.doc_id as usize).checked_sub(1)?;
                self.id_map.get(idx).map(|sid| (sid.clone(), r.score))
            })
            .collect();
        Ok(out)
    }
}
