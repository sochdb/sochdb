//! Common types used throughout the engine.

use bytemuck::{Pod, Zeroable};
use half::f16;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Vector ID - unique identifier for a vector within a segment
pub type VectorId = u32;

/// Dimension index
pub type DimIndex = u16;

/// Segment ID - unique across the collection
pub type SegmentId = u64;

/// Score type for ranking (higher = better for dot product)
pub type Score = f32;

/// Distance type for BPS (lower = better, L1 distance)
pub type Distance = u16;

/// Block index for BPS
pub type BlockIndex = u16;

/// Stripe ID for RDF posting lists
pub type StripeId = u32;

/// Configuration constants
pub const MAGIC: [u8; 8] = *b"SVSEGM\x00\x00";
pub const SEGMENT_VERSION: u32 = 1;

/// Default configuration values
pub const DEFAULT_DIM: u32 = 768;
pub const DEFAULT_BPS_BLOCK_SIZE: u16 = 16;
pub const DEFAULT_BPS_PROJECTIONS: u16 = 1;
pub const DEFAULT_RDF_TOP_T: u16 = 32;
pub const DEFAULT_STRIPE_SHIFT: u8 = 8; // 256 vids per stripe
pub const DEFAULT_NUM_OUTLIERS: u8 = 8;
pub const DEFAULT_STOP_DIM_THRESHOLD: u32 = 2048;

/// A scored candidate from search
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredCandidate {
    pub id: VectorId,
    pub score: Score,
}

impl Eq for ScoredCandidate {}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher score is better, so reverse ordering for min-heap usage
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Outlier entry for a vector (stored separately for precision)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OutlierEntry {
    pub dim_id: DimIndex,
    pub value: u16, // f16 stored as u16 bits
}

impl OutlierEntry {
    pub fn new(dim_id: DimIndex, value: f16) -> Self {
        Self {
            dim_id,
            value: value.to_bits(),
        }
    }

    pub fn get_value(&self) -> f16 {
        f16::from_bits(self.value)
    }
}

/// RDF posting entry (stored in striped chunks)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RdfPosting {
    pub vid_in_stripe: u8,      // Local ID within stripe (0-255 for shift=8)
    pub sign_and_mag: u8,       // High bit = sign, low 7 bits = magnitude
}

impl RdfPosting {
    pub fn new(vid_in_stripe: u8, sign: bool, mag: u8) -> Self {
        let sign_and_mag = if sign {
            0x80 | (mag & 0x7F)
        } else {
            mag & 0x7F
        };
        Self { vid_in_stripe, sign_and_mag }
    }

    #[inline]
    pub fn sign(&self) -> bool {
        (self.sign_and_mag & 0x80) != 0
    }

    #[inline]
    pub fn magnitude(&self) -> u8 {
        self.sign_and_mag & 0x7F
    }
}

/// Stripe chunk header for RDF posting lists
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct StripeChunkHeader {
    pub stripe_id: StripeId,
    pub count: u16,
    pub _pad: u16,
}

/// Query parameters for search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    /// Number of results to return
    pub k: usize,
    /// RDF candidate limit
    pub l_a: usize,
    /// BPS candidate limit  
    pub l_b: usize,
    /// Rerank candidate limit
    pub r: usize,
    /// Enable adaptive widening
    pub adaptive: bool,
    /// Filter bitset (if any)
    pub filter: Option<Vec<u64>>,
}

impl Default for QueryParams {
    fn default() -> Self {
        Self {
            k: 10,
            l_a: 5000,
            l_b: 20000,
            r: 500,
            adaptive: true,
            filter: None,
        }
    }
}

/// Query result with timing information
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub candidates: Vec<ScoredCandidate>,
    pub stats: QueryStats,
}

/// Statistics from query execution
#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    pub rdf_candidates: usize,
    pub bps_candidates: usize,
    pub union_size: usize,
    pub post_filter_size: usize,
    pub rerank_count: usize,
    pub widening_applied: bool,
    pub time_rotate_ns: u64,
    pub time_rdf_ns: u64,
    pub time_bps_ns: u64,
    pub time_filter_ns: u64,
    pub time_rerank_ns: u64,
    pub total_time_ns: u64,
}

impl fmt::Display for QueryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RDF:{} BPS:{} Union:{} Filtered:{} Rerank:{} Widen:{} Total:{:.2}ms",
            self.rdf_candidates,
            self.bps_candidates,
            self.union_size,
            self.post_filter_size,
            self.rerank_count,
            self.widening_applied,
            self.total_time_ns as f64 / 1_000_000.0
        )
    }
}

/// Similarity metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    DotProduct,
    Cosine,
}

impl Default for Metric {
    fn default() -> Self {
        Metric::DotProduct
    }
}

/// Segment state in LSM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentState {
    /// Currently accepting writes
    Mutable,
    /// Sealed, immutable, being written
    Sealing,
    /// Immutable, ready for queries
    Sealed,
    /// Marked for compaction
    Compacting,
    /// Deleted (tombstone)
    Deleted,
}
