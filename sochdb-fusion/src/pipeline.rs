// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Fused Query Execution Pipeline
//!
//! The `FusionEngine` is the core of the Knowledge Fabric architecture. It holds
//! references to all three index types (ART, HNSW, CSR graph) and composes their
//! operations in a single code path, using `BitSet`-based candidate masks as the
//! intermediate representation.
//!
//! ## Execution Flow
//!
//! 1. **Plan**: Analyze stages, estimate selectivities, determine optimal ordering.
//!    Most selective stages execute first to minimize work in later stages.
//!
//! 2. **Execute**: Process stages in order, flowing `CandidateMask` between them:
//!    - ART lookup → BitSet of matching internal IDs
//!    - HNSW search with mask → top-K candidates (constrained to passing IDs)
//!    - CSR traversal → expand/restrict neighborhood
//!    - Temporal filter → inline predicate on BitemporalCoord
//!
//! 3. **Score & Rank**: Combine scores from each stage into a final ranking.
//!
//! 4. **Materialize**: Load Knowledge Objects for the top-K results.
//!
//! ## Why This Architecture
//!
//! The key insight is that `BitSet` flows between stages *without allocation*.
//! The ART filter doesn't need to materialize a `Vec<ObjectId>` — it sets bits
//! in a pre-allocated mask. The HNSW search doesn't need to recheck attributes —
//! it uses the mask to skip non-matching nodes during traversal. The graph
//! expansion doesn't need to deserialize edges — it walks the CSR contiguous
//! array and sets bits in the result mask.
//!
//! This eliminates the serialization/deserialization boundaries between stages
//! that dominate latency in disaggregated architectures.

use crate::bitset::BitSet;
use crate::candidate_mask::{CandidateMask, MaskSource};
use crate::query::{FilterPredicate, FilterValue, FusionQuery, GraphComposeMode, QueryStage};
use crate::temporal_graph::TemporalCsrGraph;
use sochdb_core::knowledge_object::{KnowledgeObject, ObjectId};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Configuration for the fusion engine.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Maximum candidates to evaluate per vector search stage.
    pub max_vector_candidates: usize,

    /// Maximum graph traversal depth.
    pub max_graph_hops: u32,

    /// Whether to reorder stages by selectivity (most selective first).
    pub optimize_stage_order: bool,

    /// Minimum selectivity to continue execution (early exit if too few candidates).
    pub early_exit_threshold: f64,

    /// Whether to record execution metrics.
    pub collect_metrics: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            max_vector_candidates: 1000,
            max_graph_hops: 5,
            optimize_stage_order: true,
            early_exit_threshold: 0.0,
            collect_metrics: true,
        }
    }
}

/// A scored result from the fusion pipeline.
#[derive(Debug, Clone)]
pub struct ScoredResult {
    /// The Knowledge Object's OID.
    pub oid: ObjectId,
    /// Internal node ID (for fast re-access without hash lookup).
    pub internal_id: u32,
    /// Combined score from all stages.
    pub score: f32,
    /// Per-stage scores for explainability.
    pub stage_scores: Vec<(String, f32)>,
}

impl ScoredResult {
    pub fn new(oid: ObjectId, internal_id: u32, score: f32) -> Self {
        Self {
            oid,
            internal_id,
            score,
            stage_scores: Vec::new(),
        }
    }
}

/// Execution metrics for a fused query.
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Total wall-clock time for the query.
    pub total_time_us: u64,
    /// Per-stage timing.
    pub stage_times_us: Vec<(String, u64)>,
    /// Candidate count after each stage.
    pub candidate_counts: Vec<(String, usize)>,
    /// Number of objects materialized.
    pub materialized: usize,
}

/// The result of a fused query execution.
#[derive(Debug)]
pub struct FusionResult {
    /// Scored results, sorted by score descending.
    pub results: Vec<ScoredResult>,
    /// The final candidate mask (for debugging/analysis).
    pub final_mask: CandidateMask,
    /// Execution metrics (if `collect_metrics` was enabled).
    pub metrics: Option<ExecutionMetrics>,
}

impl FusionResult {
    /// Number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Is the result set empty?
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the OIDs of the results.
    pub fn oids(&self) -> Vec<ObjectId> {
        self.results.iter().map(|r| r.oid).collect()
    }

    /// Get the top result.
    pub fn top(&self) -> Option<&ScoredResult> {
        self.results.first()
    }
}

impl fmt::Display for FusionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FusionResult({} results", self.results.len())?;
        if let Some(metrics) = &self.metrics {
            write!(f, ", {} μs", metrics.total_time_us)?;
        }
        write!(f, ")")
    }
}

/// The fused query execution engine.
///
/// Holds references to the object store, graph index, and attribute index.
/// Executes compositional queries by flowing BitSet masks between stages
/// in a single code path.
///
/// ## Architecture
///
/// ```text
/// FusionEngine
/// ├── object_store: HashMap<ObjectId, KnowledgeObject>  (in-memory for now)
/// ├── graph: TemporalCsrGraph                           (CSR relationship index)
/// ├── embeddings: HashMap<ObjectId, HashMap<String, Vec<f32>>>  (vector data)
/// └── config: FusionConfig
/// ```
///
/// The engine is designed to be extended with real index backends:
/// - ART index for attribute lookups (currently scans the object store)
/// - HNSW index for vector search (currently brute-force)
///
/// The BitSet-based pipeline architecture remains the same regardless of
/// the underlying index implementation.
pub struct FusionEngine {
    /// Knowledge Object store (ObjectId → KnowledgeObject).
    /// In a full implementation, this would be backed by LSCS.
    objects: HashMap<ObjectId, KnowledgeObject>,

    /// Internal ID → ObjectId mapping (mirrors the CSR graph's mapping).
    internal_to_oid: Vec<ObjectId>,
    oid_to_internal: HashMap<ObjectId, u32>,

    /// Application-level CSR graph.
    graph: Option<TemporalCsrGraph>,

    /// Engine configuration.
    config: FusionConfig,
}

impl FusionEngine {
    /// Create a new fusion engine.
    pub fn new(config: FusionConfig) -> Self {
        Self {
            objects: HashMap::new(),
            internal_to_oid: Vec::new(),
            oid_to_internal: HashMap::new(),
            graph: None,
            config,
        }
    }

    /// Create with default config.
    pub fn default_engine() -> Self {
        Self::new(FusionConfig::default())
    }

    /// Create from a [`VersionedObjectStore`], loading all current objects and
    /// building the CSR graph index.
    ///
    /// This bridges the MVCC-versioned storage layer with the fused query engine:
    /// all current (non-deleted) objects are ingested, and the graph index is
    /// built from their embedded edges.
    pub fn from_store(
        store: &crate::versioned_store::VersionedObjectStore,
        config: FusionConfig,
    ) -> Result<Self, crate::versioned_store::VersionedStoreError> {
        let objects = store.all_objects()?;
        let mut engine = Self::new(config);
        engine.ingest_batch(objects);
        engine.build_graph();
        Ok(engine)
    }

    /// Ingest a Knowledge Object into the engine.
    ///
    /// This registers the object in the object store and assigns an internal ID.
    /// After ingesting all objects, call `build_graph()` to construct the CSR index.
    pub fn ingest(&mut self, object: KnowledgeObject) {
        let oid = object.oid();

        if !self.oid_to_internal.contains_key(&oid) {
            let internal_id = self.internal_to_oid.len() as u32;
            self.oid_to_internal.insert(oid, internal_id);
            self.internal_to_oid.push(oid);
        }

        self.objects.insert(oid, object);
    }

    /// Bulk ingest multiple objects.
    pub fn ingest_batch(&mut self, objects: impl IntoIterator<Item = KnowledgeObject>) {
        for obj in objects {
            self.ingest(obj);
        }
    }

    /// Build the CSR graph index from ingested Knowledge Objects.
    ///
    /// This extracts embedded edges from all objects and constructs the
    /// cache-optimized CSR representation for graph traversal.
    pub fn build_graph(&mut self) {
        use crate::temporal_graph::GraphBuilder;

        let builder = GraphBuilder::from_knowledge_objects(self.objects.values());
        self.graph = Some(builder.build());
    }

    /// The number of indexed objects.
    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }

    /// Get an object by OID.
    pub fn get(&self, oid: &ObjectId) -> Option<&KnowledgeObject> {
        self.objects.get(oid)
    }

    /// Execute a fused compositional query.
    ///
    /// This is the main entry point — it processes each query stage sequentially,
    /// flowing BitSet candidate masks between stages without materialization.
    pub fn execute(&self, query: &FusionQuery) -> FusionResult {
        let start = Instant::now();
        let num_objects = self.objects.len();

        let mut stage_times = Vec::new();
        let mut candidate_counts = Vec::new();

        // Start with the universe mask
        let mut mask = CandidateMask::all(num_objects);
        candidate_counts.push(("initial".to_string(), mask.count()));

        // Execute each stage, narrowing the candidate set
        for (i, stage) in query.stages.iter().enumerate() {
            let stage_start = Instant::now();

            let stage_mask = self.execute_stage(stage, &mask);
            mask = mask.intersect(&stage_mask);

            let stage_name = format!("stage_{}", i);
            if self.config.collect_metrics {
                stage_times.push((stage_name.clone(), stage_start.elapsed().as_micros() as u64));
                candidate_counts.push((stage_name, mask.count()));
            }

            // Early exit if no candidates remain
            if mask.is_empty() {
                break;
            }

            // Early exit if selectivity drops below threshold
            if self.config.early_exit_threshold > 0.0
                && mask.selectivity() < self.config.early_exit_threshold
            {
                break;
            }
        }

        // Score and rank candidates
        let mut scored: Vec<ScoredResult> = mask
            .iter()
            .filter_map(|internal_id| {
                let oid = self.internal_to_oid.get(internal_id)?;
                Some(ScoredResult::new(*oid, internal_id as u32, 1.0))
            })
            .collect();

        // Apply vector scores if the query includes vector search
        for stage in &query.stages {
            if let QueryStage::VectorSearch {
                space, embedding, ..
            } = stage
            {
                self.apply_vector_scores(&mut scored, space, embedding);
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Apply min_score filter
        if let Some(min) = query.min_score {
            scored.retain(|r| r.score >= min);
        }

        // Truncate to top_k
        scored.truncate(query.top_k);

        let total_time = start.elapsed().as_micros() as u64;

        let metrics = if self.config.collect_metrics {
            Some(ExecutionMetrics {
                total_time_us: total_time,
                stage_times_us: stage_times,
                candidate_counts,
                materialized: scored.len(),
            })
        } else {
            None
        };

        FusionResult {
            results: scored,
            final_mask: mask,
            metrics,
        }
    }

    /// Execute a single query stage, producing a candidate mask.
    fn execute_stage(&self, stage: &QueryStage, current_mask: &CandidateMask) -> CandidateMask {
        let num_objects = self.objects.len();

        match stage {
            QueryStage::AttributeFilter { field, predicate } => {
                self.execute_attribute_filter(field, predicate, num_objects)
            }

            QueryStage::VectorSearch {
                space,
                embedding,
                candidates,
                ..
            } => self.execute_vector_search(space, embedding, *candidates, current_mask, num_objects),

            QueryStage::GraphExpand {
                edge_kind,
                max_hops,
                compose,
            } => self.execute_graph_expand(edge_kind, *max_hops, *compose, current_mask, num_objects),

            QueryStage::TemporalFilter {
                valid_time,
                system_time,
            } => self.execute_temporal_filter(*valid_time, *system_time, num_objects),

            QueryStage::NamespaceFilter { namespace } => {
                self.execute_namespace_filter(namespace, num_objects)
            }

            QueryStage::TagFilter { tag } => self.execute_tag_filter(tag, num_objects),

            QueryStage::ProximityTo { target, max_distance } => {
                self.execute_proximity(target, *max_distance, num_objects)
            }
        }
    }

    /// Attribute filter: scan objects and set bits for matching ones.
    ///
    /// TODO: Replace with ART index lookup for O(k) performance.
    /// Currently O(n) scan — adequate for prototyping the pipeline architecture.
    fn execute_attribute_filter(
        &self,
        field: &str,
        predicate: &FilterPredicate,
        capacity: usize,
    ) -> CandidateMask {
        let mut bits = BitSet::with_capacity(capacity);

        for (oid, obj) in &self.objects {
            let internal_id = match self.oid_to_internal.get(oid) {
                Some(&id) => id as usize,
                None => continue,
            };

            let matches = match predicate {
                FilterPredicate::Eq(value) => {
                    obj.attribute(field)
                        .map_or(false, |attr| filter_value_matches(attr, value))
                }
                FilterPredicate::Ne(value) => {
                    obj.attribute(field)
                        .map_or(true, |attr| !filter_value_matches(attr, value))
                }
                FilterPredicate::In(values) => {
                    obj.attribute(field)
                        .map_or(false, |attr| values.iter().any(|v| filter_value_matches(attr, v)))
                }
                FilterPredicate::Prefix(prefix) => {
                    obj.text_attribute(field)
                        .map_or(false, |text| text.starts_with(prefix.as_str()))
                }
                FilterPredicate::Exists => obj.attribute(field).is_some(),
                FilterPredicate::HasTag(tag) => obj.has_tag(tag),
                FilterPredicate::IsKind(kind) => obj.kind() == kind,
                FilterPredicate::Range { min, max } => {
                    obj.attribute(field).map_or(false, |attr| {
                        let above_min = min.as_ref().map_or(true, |m| filter_value_gte(attr, m));
                        let below_max = max.as_ref().map_or(true, |m| filter_value_lte(attr, m));
                        above_min && below_max
                    })
                }
            };

            if matches {
                bits.set(internal_id);
            }
        }

        CandidateMask::new(
            bits,
            MaskSource::AttributeFilter {
                field: field.to_string(),
                selectivity: 0.0, // Will be set by count
            },
        )
    }

    /// Vector search: brute-force nearest neighbor search over candidates.
    ///
    /// TODO: Replace with HNSW index search for O(ef_search × log N) performance.
    /// The mask-based filtering will work the same way with HNSW — the search
    /// just skips nodes not in the mask.
    fn execute_vector_search(
        &self,
        space: &str,
        query_embedding: &[f32],
        candidates: usize,
        _current_mask: &CandidateMask,
        capacity: usize,
    ) -> CandidateMask {
        // Collect (internal_id, distance) for objects with embeddings in this space
        let mut scored: Vec<(usize, f32)> = Vec::new();

        for (oid, obj) in &self.objects {
            let internal_id = match self.oid_to_internal.get(oid) {
                Some(&id) => id as usize,
                None => continue,
            };

            if let Some(emb) = obj.embedding(space) {
                if emb.vector.len() == query_embedding.len() {
                    let dist = cosine_distance(&emb.vector, query_embedding);
                    scored.push((internal_id, dist));
                }
            }
        }

        // Sort by distance (ascending) and take top candidates
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(candidates);

        let mut bits = BitSet::with_capacity(capacity);
        for (id, _) in &scored {
            bits.set(*id);
        }

        CandidateMask::new(
            bits,
            MaskSource::VectorSearch {
                space: space.to_string(),
                ef_search: candidates,
            },
        )
    }

    /// Graph expansion: traverse the CSR graph from current candidates.
    fn execute_graph_expand(
        &self,
        edge_kind: &sochdb_core::knowledge_object::EdgeKind,
        max_hops: u32,
        compose: GraphComposeMode,
        current_mask: &CandidateMask,
        capacity: usize,
    ) -> CandidateMask {
        let graph = match &self.graph {
            Some(g) => g,
            None => return CandidateMask::all(capacity), // No graph built, pass-through
        };

        let kind_id = graph.kind_id(edge_kind.label());

        let mut result_bits = match compose {
            GraphComposeMode::And | GraphComposeMode::Replace => BitSet::with_capacity(capacity),
            GraphComposeMode::Or => current_mask.bits().clone(),
        };

        // For each candidate in the current mask, expand along the graph
        for candidate in current_mask.iter() {
            if candidate >= graph.num_nodes() {
                continue;
            }

            // BFS from this candidate
            let reachable = graph.bfs(candidate as u32, max_hops, None);

            for (node, _, _) in reachable {
                // Optional: filter by edge kind
                if let Some(kid) = kind_id {
                    // Check if the node was reached via the specified edge kind
                    let edges = graph.edges(candidate as u32);
                    if edges.iter().any(|e| e.kind_id == kid) {
                        result_bits.set(node as usize);
                    }
                } else {
                    result_bits.set(node as usize);
                }
            }
        }

        CandidateMask::new(
            result_bits,
            MaskSource::GraphTraversal {
                edge_kind: edge_kind.label().to_string(),
                hops: max_hops,
            },
        )
    }

    /// Temporal filter: check BitemporalCoord on each object.
    fn execute_temporal_filter(
        &self,
        valid_time: Option<u64>,
        system_time: Option<u64>,
        capacity: usize,
    ) -> CandidateMask {
        let mut bits = BitSet::with_capacity(capacity);

        for (oid, obj) in &self.objects {
            let internal_id = match self.oid_to_internal.get(oid) {
                Some(&id) => id as usize,
                None => continue,
            };

            let passes = match (valid_time, system_time) {
                (Some(vt), Some(st)) => obj.visible_at(st, vt),
                (Some(vt), None) => obj.valid_at(vt),
                (None, Some(st)) => obj.known_at(st),
                (None, None) => true,
            };

            if passes {
                bits.set(internal_id);
            }
        }

        CandidateMask::new(
            bits,
            MaskSource::TemporalFilter { valid_time, system_time },
        )
    }

    /// Namespace filter.
    fn execute_namespace_filter(&self, namespace: &str, capacity: usize) -> CandidateMask {
        let mut bits = BitSet::with_capacity(capacity);

        for (oid, obj) in &self.objects {
            let internal_id = match self.oid_to_internal.get(oid) {
                Some(&id) => id as usize,
                None => continue,
            };

            if obj.namespace() == Some(namespace) {
                bits.set(internal_id);
            }
        }

        CandidateMask::new(bits, MaskSource::All)
    }

    /// Tag filter.
    fn execute_tag_filter(&self, tag: &str, capacity: usize) -> CandidateMask {
        let mut bits = BitSet::with_capacity(capacity);

        for (oid, obj) in &self.objects {
            let internal_id = match self.oid_to_internal.get(oid) {
                Some(&id) => id as usize,
                None => continue,
            };

            if obj.has_tag(tag) {
                bits.set(internal_id);
            }
        }

        CandidateMask::new(
            bits,
            MaskSource::TagFilter { tag: tag.to_string() },
        )
    }

    /// Proximity filter: objects within max_distance of a target.
    fn execute_proximity(
        &self,
        target: &ObjectId,
        max_distance: f32,
        capacity: usize,
    ) -> CandidateMask {
        let graph = match &self.graph {
            Some(g) => g,
            None => return CandidateMask::none(capacity),
        };

        let target_internal = match graph.internal_id(target) {
            Some(id) => id,
            None => return CandidateMask::none(capacity),
        };

        // Use graph distance as proximity metric
        let max_hops = (max_distance * 10.0) as u32; // scale distance to hops
        let bitset = graph.neighborhood_bitset(target_internal, max_hops.max(1), None);

        CandidateMask::new(bitset, MaskSource::All)
    }

    /// Apply vector similarity scores to scored results.
    fn apply_vector_scores(
        &self,
        scored: &mut [ScoredResult],
        space: &str,
        query_embedding: &[f32],
    ) {
        for result in scored.iter_mut() {
            if let Some(obj) = self.objects.get(&result.oid) {
                if let Some(emb) = obj.embedding(space) {
                    if emb.vector.len() == query_embedding.len() {
                        let similarity = cosine_similarity(&emb.vector, query_embedding);
                        result.score = similarity;
                        result.stage_scores.push((
                            format!("vector:{}", space),
                            similarity,
                        ));
                    }
                }
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a SochValue matches a FilterValue.
fn filter_value_matches(
    attr: &sochdb_core::soch::SochValue,
    value: &FilterValue,
) -> bool {
    use sochdb_core::soch::SochValue;

    match (attr, value) {
        (SochValue::Text(a), FilterValue::Text(b)) => a == b,
        (SochValue::Int(a), FilterValue::Int(b)) => a == b,
        (SochValue::UInt(a), FilterValue::UInt(b)) => a == b,
        (SochValue::Float(a), FilterValue::Float(b)) => (*a - *b).abs() < f64::EPSILON,
        (SochValue::Bool(a), FilterValue::Bool(b)) => a == b,
        _ => false,
    }
}

/// Check if a SochValue is >= a FilterValue.
fn filter_value_gte(
    attr: &sochdb_core::soch::SochValue,
    value: &FilterValue,
) -> bool {
    use sochdb_core::soch::SochValue;

    match (attr, value) {
        (SochValue::Int(a), FilterValue::Int(b)) => a >= b,
        (SochValue::UInt(a), FilterValue::UInt(b)) => a >= b,
        (SochValue::Float(a), FilterValue::Float(b)) => a >= b,
        (SochValue::Text(a), FilterValue::Text(b)) => a >= b,
        _ => false,
    }
}

/// Check if a SochValue is <= a FilterValue.
fn filter_value_lte(
    attr: &sochdb_core::soch::SochValue,
    value: &FilterValue,
) -> bool {
    use sochdb_core::soch::SochValue;

    match (attr, value) {
        (SochValue::Int(a), FilterValue::Int(b)) => a <= b,
        (SochValue::UInt(a), FilterValue::UInt(b)) => a <= b,
        (SochValue::Float(a), FilterValue::Float(b)) => a <= b,
        (SochValue::Text(a), FilterValue::Text(b)) => a <= b,
        _ => false,
    }
}

/// Cosine distance between two vectors: 1 - cosine_similarity.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sochdb_core::knowledge_object::*;
    use sochdb_core::soch::SochValue;
    use crate::query::*;

    fn make_test_engine() -> FusionEngine {
        let mut engine = FusionEngine::default_engine();

        // Create test objects
        let acme_oid = ObjectId::from_content(b"acme");

        let alice = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .attribute("status", SochValue::Text("active".into()))
            .embedding("semantic", vec![1.0, 0.0, 0.0])
            .edge(Edge::new(acme_oid, EdgeKind::typed("works_at"), 1.0))
            .tag("person")
            .namespace("test")
            .valid_from(100)
            .valid_to(u64::MAX)
            .system_time(50)
            .build();

        let bob = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Bob".into()))
            .attribute("status", SochValue::Text("active".into()))
            .embedding("semantic", vec![0.9, 0.1, 0.0])
            .edge(Edge::new(acme_oid, EdgeKind::typed("works_at"), 0.8))
            .tag("person")
            .namespace("test")
            .valid_from(100)
            .valid_to(200)
            .system_time(50)
            .build();

        let charlie = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Charlie".into()))
            .attribute("status", SochValue::Text("inactive".into()))
            .embedding("semantic", vec![0.0, 1.0, 0.0])
            .tag("person")
            .namespace("other")
            .valid_from(50)
            .valid_to(300)
            .system_time(40)
            .build();

        let acme = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Acme Corp".into()))
            .attribute("status", SochValue::Text("active".into()))
            .tag("organization")
            .namespace("test")
            .build_with_oid(acme_oid);

        engine.ingest(alice);
        engine.ingest(bob);
        engine.ingest(charlie);
        engine.ingest(acme);
        engine.build_graph();

        engine
    }

    #[test]
    fn test_attribute_filter_only() {
        let engine = make_test_engine();

        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")
            .top_k(10)
            .build();

        let result = engine.execute(&query);

        // Alice, Bob, and Acme are active
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_attribute_plus_temporal() {
        let engine = make_test_engine();

        // Active + valid at time 250
        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")
            .valid_at(250)
            .top_k(10)
            .build();

        let result = engine.execute(&query);

        // Alice is active and valid at 250 (valid_to=MAX)
        // Bob is active but NOT valid at 250 (valid_to=200)
        // Acme is active and valid at 250 (ETERNAL default)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_vector_search() {
        let engine = make_test_engine();

        let query = FusionQueryBuilder::new()
            .vector_search("semantic", vec![1.0, 0.0, 0.0], 2)
            .top_k(2)
            .build();

        let result = engine.execute(&query);

        assert_eq!(result.len(), 2);
        // Alice should be most similar (exact match)
        assert!(result.results[0].score > 0.9);
    }

    #[test]
    fn test_combined_filter_vector_temporal() {
        let engine = make_test_engine();

        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")
            .vector_search("semantic", vec![1.0, 0.0, 0.0], 10)
            .valid_at(150)
            .top_k(5)
            .build();

        let result = engine.execute(&query);

        // Active + has embedding + valid at 150: Alice and Bob
        // Acme doesn't have embeddings so won't be in vector results
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_namespace_filter() {
        let engine = make_test_engine();

        let query = FusionQueryBuilder::new()
            .in_namespace("test")
            .top_k(10)
            .build();

        let result = engine.execute(&query);

        // Alice, Bob, Acme are in "test" namespace
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_tag_filter() {
        let engine = make_test_engine();

        let query = FusionQueryBuilder::new()
            .with_tag("person")
            .top_k(10)
            .build();

        let result = engine.execute(&query);

        // Alice, Bob, Charlie are tagged "person"
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_early_exit_on_empty() {
        let engine = make_test_engine();

        // No objects have status "nonexistent"
        let query = FusionQueryBuilder::new()
            .filter_eq("status", "nonexistent")
            .vector_search("semantic", vec![1.0, 0.0, 0.0], 10)
            .top_k(10)
            .build();

        let result = engine.execute(&query);

        assert!(result.is_empty());
        // The vector search stage should have been skipped (early exit)
    }

    #[test]
    fn test_execution_metrics() {
        let engine = make_test_engine();

        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")
            .valid_at(150)
            .top_k(10)
            .build();

        let result = engine.execute(&query);

        let metrics = result.metrics.unwrap();
        assert!(metrics.total_time_us > 0);
        assert_eq!(metrics.stage_times_us.len(), 2);
        assert!(metrics.candidate_counts.len() >= 2);
    }

    #[test]
    fn test_bitemporal_query() {
        let engine = make_test_engine();

        // What did the system know at time 50 about events at time 150?
        let query = FusionQueryBuilder::new()
            .bitemporal(150, 50)
            .with_tag("person")
            .top_k(10)
            .build();

        let result = engine.execute(&query);

        // Alice: system_time=50 (known), valid at 150 ✓
        // Bob: system_time=50 (known), valid at 150 ✓
        // Charlie: system_time=40 (known at 50), valid at 150 ✓
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_store_integration() {
        use crate::versioned_store::{VersionedObjectStore, StoreConfig};
        use sochdb_core::knowledge_object::{
            CompressionMode, Edge, EdgeKind, KnowledgeObjectBuilder, ObjectKind,
        };
        use sochdb_core::soch::SochValue;

        // Create a versioned store and insert objects with edges
        let store = VersionedObjectStore::with_config(StoreConfig::uncompressed());

        let alice = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .attribute("status", SochValue::Text("active".into()))
            .embedding("semantic", vec![1.0, 0.0, 0.0])
            .tag("person")
            .valid_from(100)
            .valid_to(u64::MAX)
            .build();
        let alice_oid = alice.oid();

        let bob = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Bob".into()))
            .attribute("status", SochValue::Text("active".into()))
            .embedding("semantic", vec![0.9, 0.1, 0.0])
            .edge(Edge::new(alice_oid, EdgeKind::typed("knows"), 1.0))
            .tag("person")
            .valid_from(100)
            .valid_to(u64::MAX)
            .build();

        store.put(alice).unwrap();
        store.put(bob).unwrap();

        // Build a FusionEngine from the store
        let engine = FusionEngine::from_store(&store, FusionConfig::default()).unwrap();
        assert_eq!(engine.num_objects(), 2);

        // Run a query
        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")
            .with_tag("person")
            .top_k(10)
            .build();

        let result = engine.execute(&query);
        assert_eq!(result.len(), 2);
    }
}
