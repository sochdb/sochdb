// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Candidate Mask — Composable Filter Results
//!
//! `CandidateMask` wraps a [`BitSet`] with metadata about how it was produced
//! and operations for composing multiple filter results. This is the glue
//! that connects ART attribute filters, HNSW candidate sets, CSR graph
//! neighborhoods, and temporal predicates.
//!
//! ## Composition Rules
//!
//! The fused pipeline composes masks using boolean operations:
//!
//! - `AND`: intersection (attribute filter AND vector candidates)
//! - `OR`: union (match in semantic space OR code space)
//! - `AND NOT`: exclusion (candidates NOT in blocked set)
//!
//! Each composition operates on the underlying BitSet in O(n/64) time.

use crate::bitset::BitSet;
use std::fmt;

/// How a candidate mask was produced — for query planning and cost estimation.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskSource {
    /// Produced by an ART attribute lookup.
    AttributeFilter {
        field: String,
        selectivity: f64,
    },
    /// Produced by an HNSW vector search.
    VectorSearch {
        space: String,
        ef_search: usize,
    },
    /// Produced by CSR graph traversal.
    GraphTraversal {
        edge_kind: String,
        hops: u32,
    },
    /// Produced by a temporal predicate.
    TemporalFilter {
        valid_time: Option<u64>,
        system_time: Option<u64>,
    },
    /// Produced by a tag filter.
    TagFilter {
        tag: String,
    },
    /// Produced by composing other masks.
    Composed {
        operation: MaskOp,
        sources: Vec<MaskSource>,
    },
    /// Universe (all candidates).
    All,
    /// Empty set (no candidates).
    None,
}

/// Boolean operation for mask composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// Intersection: both masks must match.
    And,
    /// Union: either mask may match.
    Or,
    /// Difference: first mask matches, second does not.
    AndNot,
}

impl fmt::Display for MaskOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskOp::And => write!(f, "AND"),
            MaskOp::Or => write!(f, "OR"),
            MaskOp::AndNot => write!(f, "AND NOT"),
        }
    }
}

/// A candidate mask with provenance information.
///
/// This is the primary data structure that flows between stages of the
/// fused query pipeline. Each stage produces a `CandidateMask`, and stages
/// are composed via boolean operations to narrow the candidate set.
#[derive(Clone)]
pub struct CandidateMask {
    /// The underlying bit vector.
    bits: BitSet,
    /// How this mask was produced.
    source: MaskSource,
}

impl CandidateMask {
    /// Create a mask from a BitSet and source.
    pub fn new(bits: BitSet, source: MaskSource) -> Self {
        Self { bits, source }
    }

    /// Create a universe mask (all candidates allowed).
    pub fn all(capacity: usize) -> Self {
        Self {
            bits: BitSet::all(capacity),
            source: MaskSource::All,
        }
    }

    /// Create an empty mask (no candidates).
    pub fn none(capacity: usize) -> Self {
        Self {
            bits: BitSet::with_capacity(capacity),
            source: MaskSource::None,
        }
    }

    /// Create a mask from an iterator of candidate IDs.
    pub fn from_ids(capacity: usize, ids: impl IntoIterator<Item = usize>, source: MaskSource) -> Self {
        Self {
            bits: BitSet::from_iter(capacity, ids),
            source,
        }
    }

    /// The underlying bit set.
    #[inline]
    pub fn bits(&self) -> &BitSet {
        &self.bits
    }

    /// Mutable access to the underlying bit set.
    #[inline]
    pub fn bits_mut(&mut self) -> &mut BitSet {
        &mut self.bits
    }

    /// How this mask was produced.
    pub fn source(&self) -> &MaskSource {
        &self.source
    }

    /// Number of candidates that pass this mask.
    #[inline]
    pub fn count(&self) -> usize {
        self.bits.count()
    }

    /// Is this mask empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Selectivity (fraction of candidates that pass).
    #[inline]
    pub fn selectivity(&self) -> f64 {
        self.bits.selectivity()
    }

    /// Test if a candidate ID passes this mask.
    #[inline]
    pub fn contains(&self, id: usize) -> bool {
        self.bits.contains(id)
    }

    /// Iterate over candidate IDs that pass this mask.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter()
    }

    // =========================================================================
    // Composition Operations
    // =========================================================================

    /// Intersect with another mask (AND).
    ///
    /// Returns only candidates that pass *both* masks. This is the most common
    /// composition: "attribute filter AND vector similarity candidates".
    pub fn intersect(mut self, other: &CandidateMask) -> Self {
        let source = MaskSource::Composed {
            operation: MaskOp::And,
            sources: vec![self.source.clone(), other.source.clone()],
        };
        self.bits.intersect(&other.bits);
        self.source = source;
        self
    }

    /// Union with another mask (OR).
    ///
    /// Returns candidates that pass *either* mask. Useful for multi-space
    /// embedding queries: "similar in semantic space OR similar in code space".
    pub fn union(mut self, other: &CandidateMask) -> Self {
        let source = MaskSource::Composed {
            operation: MaskOp::Or,
            sources: vec![self.source.clone(), other.source.clone()],
        };
        self.bits.union(&other.bits);
        self.source = source;
        self
    }

    /// Difference with another mask (AND NOT).
    ///
    /// Returns candidates that pass this mask but NOT the other. Useful for
    /// exclusion: "similar entities NOT already seen".
    pub fn difference(mut self, other: &CandidateMask) -> Self {
        let source = MaskSource::Composed {
            operation: MaskOp::AndNot,
            sources: vec![self.source.clone(), other.source.clone()],
        };
        self.bits.difference(&other.bits);
        self.source = source;
        self
    }

    /// Apply an operation with another mask, returning a new mask.
    pub fn compose(self, other: &CandidateMask, op: MaskOp) -> Self {
        match op {
            MaskOp::And => self.intersect(other),
            MaskOp::Or => self.union(other),
            MaskOp::AndNot => self.difference(other),
        }
    }
}

impl fmt::Debug for CandidateMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CandidateMask(count={}, selectivity={:.4}, source={:?})",
            self.count(),
            self.selectivity(),
            self.source
        )
    }
}

impl fmt::Display for CandidateMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mask[{}/{} = {:.2}%]",
            self.count(),
            self.bits.capacity(),
            self.selectivity() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect_masks() {
        let attr_mask = CandidateMask::from_ids(
            100,
            vec![1, 5, 10, 20, 50],
            MaskSource::AttributeFilter {
                field: "status".into(),
                selectivity: 0.05,
            },
        );

        let vector_mask = CandidateMask::from_ids(
            100,
            vec![5, 10, 15, 25, 50, 75],
            MaskSource::VectorSearch {
                space: "semantic".into(),
                ef_search: 100,
            },
        );

        let result = attr_mask.intersect(&vector_mask);

        assert_eq!(result.count(), 3); // 5, 10, 50
        assert!(result.contains(5));
        assert!(result.contains(10));
        assert!(result.contains(50));
        assert!(!result.contains(1));
        assert!(!result.contains(75));

        // Source should record composition
        match result.source() {
            MaskSource::Composed { operation, .. } => {
                assert_eq!(*operation, MaskOp::And);
            }
            _ => panic!("Expected composed source"),
        }
    }

    #[test]
    fn test_union_masks() {
        let semantic = CandidateMask::from_ids(
            100,
            vec![1, 5, 10],
            MaskSource::VectorSearch { space: "semantic".into(), ef_search: 50 },
        );

        let code = CandidateMask::from_ids(
            100,
            vec![5, 10, 20],
            MaskSource::VectorSearch { space: "code".into(), ef_search: 50 },
        );

        let result = semantic.union(&code);

        assert_eq!(result.count(), 4); // 1, 5, 10, 20
    }

    #[test]
    fn test_difference_masks() {
        let candidates = CandidateMask::from_ids(100, vec![1, 5, 10, 20, 50], MaskSource::All);
        let blocked = CandidateMask::from_ids(100, vec![5, 10], MaskSource::None);

        let result = candidates.difference(&blocked);

        assert_eq!(result.count(), 3); // 1, 20, 50
        assert!(!result.contains(5));
        assert!(!result.contains(10));
    }

    #[test]
    fn test_all_and_none() {
        let all = CandidateMask::all(1000);
        assert_eq!(all.count(), 1000);
        assert!((all.selectivity() - 1.0).abs() < f64::EPSILON);

        let none = CandidateMask::none(1000);
        assert_eq!(none.count(), 0);
        assert!(none.is_empty());
    }
}
