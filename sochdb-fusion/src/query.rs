// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Compositional Query Types
//!
//! Defines the query language for fused Knowledge Fabric queries. Queries are
//! composed of **stages** that are fused into a single execution pipeline.
//!
//! ## Query Composition
//!
//! ```rust,ignore
//! let query = FusionQueryBuilder::new()
//!     .filter("status", FilterPredicate::Eq("active".into()))
//!     .vector_search("semantic", embedding, 100)
//!     .graph_expand(EdgeKind::typed("works_at"), 1)
//!     .valid_at(now)
//!     .top_k(10)
//!     .build();
//! ```
//!
//! Each stage narrows the candidate set via BitSet composition, operating
//! entirely in-process without serialization boundaries.

use sochdb_core::knowledge_object::{EdgeKind, ObjectId, ObjectKind};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A predicate for attribute filtering (evaluated against ART index).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterPredicate {
    /// Exact match: `field == value`
    Eq(FilterValue),
    /// Not equal: `field != value`
    Ne(FilterValue),
    /// Range: `min <= field <= max`
    Range {
        min: Option<FilterValue>,
        max: Option<FilterValue>,
    },
    /// Set membership: `field IN (v1, v2, ...)`
    In(Vec<FilterValue>),
    /// Prefix match: `field STARTS WITH prefix`
    Prefix(String),
    /// Existence: `field IS NOT NULL`
    Exists,
    /// Tag predicate: object has the given tag
    HasTag(String),
    /// Kind predicate: object is of the given kind
    IsKind(ObjectKind),
}

/// A typed value for filter predicates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterValue {
    Text(String),
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
}

impl From<&str> for FilterValue {
    fn from(s: &str) -> Self {
        FilterValue::Text(s.to_string())
    }
}

impl From<String> for FilterValue {
    fn from(s: String) -> Self {
        FilterValue::Text(s)
    }
}

impl From<i64> for FilterValue {
    fn from(v: i64) -> Self {
        FilterValue::Int(v)
    }
}

impl From<u64> for FilterValue {
    fn from(v: u64) -> Self {
        FilterValue::UInt(v)
    }
}

impl From<f64> for FilterValue {
    fn from(v: f64) -> Self {
        FilterValue::Float(v)
    }
}

impl From<bool> for FilterValue {
    fn from(v: bool) -> Self {
        FilterValue::Bool(v)
    }
}

/// A stage in the fused query execution pipeline.
///
/// Each stage produces a `CandidateMask` (BitSet) that is composed with the
/// masks from other stages. The execution engine determines the optimal
/// ordering based on estimated selectivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryStage {
    /// ART attribute filter: `field OP value`
    ///
    /// Selectivity: typically 0.01–0.1 (filters out 90–99% of candidates)
    /// Cost: O(k) ART lookup where k = key length
    AttributeFilter {
        field: String,
        predicate: FilterPredicate,
    },

    /// HNSW vector search in a named embedding space.
    ///
    /// Produces the top-K nearest neighbors, optionally restricted by a
    /// pre-computed candidate mask.
    ///
    /// Cost: O(ef_search × log(N)) with CSR graph traversal
    VectorSearch {
        /// Which embedding space to search in.
        space: String,
        /// The query embedding vector.
        embedding: Vec<f32>,
        /// ef_search parameter (accuracy/speed trade-off).
        ef_search: usize,
        /// Maximum candidates to return from vector search.
        candidates: usize,
    },

    /// CSR graph traversal: expand from current candidate set along edges.
    ///
    /// For each candidate, traverse edges of the given kind up to `max_hops`,
    /// adding reached nodes to the candidate set (OR) or restricting to reached
    /// nodes (AND).
    GraphExpand {
        /// Which edge kind to traverse.
        edge_kind: EdgeKind,
        /// Maximum traversal depth.
        max_hops: u32,
        /// How to compose with existing candidates:
        /// - `And`: candidates must be reachable from existing candidates
        /// - `Or`: add reachable nodes to existing candidates
        compose: GraphComposeMode,
    },

    /// Temporal filter: restrict to objects valid at a given time.
    ///
    /// Cost: O(1) per object (inline check on BitemporalCoord)
    TemporalFilter {
        /// Filter by valid time.
        valid_time: Option<u64>,
        /// Filter by system time (as-of query).
        system_time: Option<u64>,
    },

    /// Namespace filter: restrict to a specific namespace.
    NamespaceFilter {
        namespace: String,
    },

    /// Tag filter: restrict to objects with a specific tag.
    TagFilter {
        tag: String,
    },

    /// Proximity to a specific object (graph distance or embedding distance).
    ProximityTo {
        target: ObjectId,
        max_distance: f32,
    },
}

/// How graph expansion composes with existing candidates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GraphComposeMode {
    /// Intersection: candidates must be in the graph neighborhood.
    And,
    /// Union: add graph neighbors to candidates.
    Or,
    /// Replace: graph neighbors become the new candidate set.
    Replace,
}

/// A complete fused query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionQuery {
    /// Ordered list of query stages.
    pub stages: Vec<QueryStage>,

    /// Maximum number of results to return.
    pub top_k: usize,

    /// Minimum score threshold (0.0–1.0).
    pub min_score: Option<f32>,

    /// Whether to include the full Knowledge Object in results,
    /// or just OIDs and scores.
    pub include_payload: bool,

    /// Whether to include provenance information in results.
    pub include_provenance: bool,

    /// Optional: restrict to a specific namespace (applied as a pre-filter).
    pub namespace: Option<String>,
}

impl FusionQuery {
    /// Estimate the selectivity of this query (product of stage selectivities).
    pub fn estimated_selectivity(&self) -> f64 {
        let mut sel = 1.0;
        for stage in &self.stages {
            sel *= match stage {
                QueryStage::AttributeFilter { .. } => 0.05,   // ~5% pass typical attribute filter
                QueryStage::VectorSearch { candidates, .. } => {
                    // Vector search returns a fixed number of candidates
                    0.01_f64.max(*candidates as f64 / 1_000_000.0)
                }
                QueryStage::GraphExpand { max_hops, .. } => {
                    // Graph expansion: ~10 neighbors per hop
                    let expansion = 10.0_f64.powi(*max_hops as i32);
                    (expansion / 1_000_000.0).min(1.0)
                }
                QueryStage::TemporalFilter { .. } => 0.5,     // ~50% pass temporal filter
                QueryStage::NamespaceFilter { .. } => 0.1,    // ~10% per namespace
                QueryStage::TagFilter { .. } => 0.1,          // ~10% per tag
                QueryStage::ProximityTo { .. } => 0.01,       // ~1% proximity match
            };
        }
        sel
    }
}

impl fmt::Display for FusionQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FusionQuery(stages={}, top_k={})", self.stages.len(), self.top_k)
    }
}

/// Builder for constructing fused queries with a fluent API.
pub struct FusionQueryBuilder {
    stages: Vec<QueryStage>,
    top_k: usize,
    min_score: Option<f32>,
    include_payload: bool,
    include_provenance: bool,
    namespace: Option<String>,
}

impl FusionQueryBuilder {
    /// Create a new query builder.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            top_k: 10,
            min_score: None,
            include_payload: true,
            include_provenance: false,
            namespace: None,
        }
    }

    /// Add an attribute filter stage.
    pub fn filter(mut self, field: impl Into<String>, predicate: FilterPredicate) -> Self {
        self.stages.push(QueryStage::AttributeFilter {
            field: field.into(),
            predicate,
        });
        self
    }

    /// Add an exact-match attribute filter.
    pub fn filter_eq(self, field: impl Into<String>, value: impl Into<FilterValue>) -> Self {
        self.filter(field, FilterPredicate::Eq(value.into()))
    }

    /// Add a vector search stage.
    pub fn vector_search(
        mut self,
        space: impl Into<String>,
        embedding: Vec<f32>,
        candidates: usize,
    ) -> Self {
        self.stages.push(QueryStage::VectorSearch {
            space: space.into(),
            embedding,
            ef_search: candidates * 2, // Default: ef_search = 2× candidates
            candidates,
        });
        self
    }

    /// Add a vector search stage with custom ef_search.
    pub fn vector_search_with_ef(
        mut self,
        space: impl Into<String>,
        embedding: Vec<f32>,
        candidates: usize,
        ef_search: usize,
    ) -> Self {
        self.stages.push(QueryStage::VectorSearch {
            space: space.into(),
            embedding,
            ef_search,
            candidates,
        });
        self
    }

    /// Add a graph expansion stage (AND compose by default).
    pub fn graph_expand(mut self, edge_kind: EdgeKind, max_hops: u32) -> Self {
        self.stages.push(QueryStage::GraphExpand {
            edge_kind,
            max_hops,
            compose: GraphComposeMode::And,
        });
        self
    }

    /// Add a graph expansion stage with explicit compose mode.
    pub fn graph_expand_with_mode(
        mut self,
        edge_kind: EdgeKind,
        max_hops: u32,
        compose: GraphComposeMode,
    ) -> Self {
        self.stages.push(QueryStage::GraphExpand {
            edge_kind,
            max_hops,
            compose,
        });
        self
    }

    /// Add a temporal filter (valid_at).
    pub fn valid_at(mut self, valid_time: u64) -> Self {
        self.stages.push(QueryStage::TemporalFilter {
            valid_time: Some(valid_time),
            system_time: None,
        });
        self
    }

    /// Add a temporal filter (as_of system time).
    pub fn as_of(mut self, system_time: u64) -> Self {
        self.stages.push(QueryStage::TemporalFilter {
            valid_time: None,
            system_time: Some(system_time),
        });
        self
    }

    /// Add a combined bitemporal filter.
    pub fn bitemporal(mut self, valid_time: u64, system_time: u64) -> Self {
        self.stages.push(QueryStage::TemporalFilter {
            valid_time: Some(valid_time),
            system_time: Some(system_time),
        });
        self
    }

    /// Add a namespace filter.
    pub fn in_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Add a tag filter.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.stages.push(QueryStage::TagFilter {
            tag: tag.into(),
        });
        self
    }

    /// Add a kind filter.
    pub fn of_kind(self, kind: ObjectKind) -> Self {
        self.filter("_kind", FilterPredicate::IsKind(kind))
    }

    /// Set the maximum number of results.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set the minimum score threshold.
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Include full Knowledge Object payload in results.
    pub fn with_payload(mut self) -> Self {
        self.include_payload = true;
        self
    }

    /// Exclude payload from results (OIDs and scores only — faster).
    pub fn without_payload(mut self) -> Self {
        self.include_payload = false;
        self
    }

    /// Include provenance information in results.
    pub fn with_provenance(mut self) -> Self {
        self.include_provenance = true;
        self
    }

    /// Build the query.
    pub fn build(self) -> FusionQuery {
        let mut stages = self.stages;

        // Prepend namespace filter if set
        if let Some(ns) = &self.namespace {
            stages.insert(0, QueryStage::NamespaceFilter {
                namespace: ns.clone(),
            });
        }

        FusionQuery {
            stages,
            top_k: self.top_k,
            min_score: self.min_score,
            include_payload: self.include_payload,
            include_provenance: self.include_provenance,
            namespace: self.namespace,
        }
    }
}

impl Default for FusionQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder() {
        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")
            .vector_search("semantic", vec![0.1, 0.2, 0.3], 100)
            .graph_expand(EdgeKind::typed("works_at"), 1)
            .valid_at(1700000000)
            .top_k(10)
            .build();

        assert_eq!(query.stages.len(), 4);
        assert_eq!(query.top_k, 10);
    }

    #[test]
    fn test_namespace_prepended() {
        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")
            .in_namespace("tenant-1")
            .build();

        assert_eq!(query.stages.len(), 2);
        // Namespace should be first
        assert!(matches!(&query.stages[0], QueryStage::NamespaceFilter { .. }));
    }

    #[test]
    fn test_selectivity_estimation() {
        let query = FusionQueryBuilder::new()
            .filter_eq("status", "active")  // ~5%
            .vector_search("semantic", vec![0.1], 100) // ~0.01%
            .valid_at(1700000000)            // ~50%
            .build();

        let sel = query.estimated_selectivity();
        assert!(sel < 0.01); // Very selective
        assert!(sel > 0.0);  // But not zero
    }

    #[test]
    fn test_bitemporal_query() {
        let query = FusionQueryBuilder::new()
            .bitemporal(1700000000, 1699999000)
            .build();

        assert_eq!(query.stages.len(), 1);
        match &query.stages[0] {
            QueryStage::TemporalFilter { valid_time, system_time } => {
                assert_eq!(*valid_time, Some(1700000000));
                assert_eq!(*system_time, Some(1699999000));
            }
            _ => panic!("Expected TemporalFilter"),
        }
    }
}
