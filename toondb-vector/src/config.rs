//! Engine configuration.

use crate::types::*;
use serde::{Deserialize, Serialize};

/// Main engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Vector dimension
    pub dim: u32,
    
    /// Similarity metric
    pub metric: Metric,

    /// BPS configuration
    pub bps: BpsConfig,

    /// RDF configuration
    pub rdf: RdfConfig,

    /// Rerank configuration
    pub rerank: RerankConfig,

    /// Router configuration
    pub router: RouterConfig,

    /// LSM/Segment configuration
    pub lsm: LsmConfig,

    /// Query defaults
    pub query: QueryConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dim: DEFAULT_DIM,
            metric: Metric::DotProduct,
            bps: BpsConfig::default(),
            rdf: RdfConfig::default(),
            rerank: RerankConfig::default(),
            router: RouterConfig::default(),
            lsm: LsmConfig::default(),
            query: QueryConfig::default(),
        }
    }
}

impl EngineConfig {
    /// Create config for a specific dimension
    pub fn with_dim(dim: u32) -> Self {
        let num_blocks = (dim + DEFAULT_BPS_BLOCK_SIZE as u32 - 1) / DEFAULT_BPS_BLOCK_SIZE as u32;
        Self {
            dim,
            bps: BpsConfig {
                block_size: DEFAULT_BPS_BLOCK_SIZE,
                num_blocks: num_blocks as u16,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Get padded dimension (power of 2 for Hadamard)
    pub fn padded_dim(&self) -> u32 {
        let mut p = 1u32;
        while p < self.dim {
            p *= 2;
        }
        p
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.dim == 0 {
            return Err(crate::Error::Config("Dimension must be > 0".into()));
        }
        if self.bps.block_size == 0 {
            return Err(crate::Error::Config("BPS block size must be > 0".into()));
        }
        if self.rdf.top_t == 0 {
            return Err(crate::Error::Config("RDF top_t must be > 0".into()));
        }
        Ok(())
    }
}

/// BPS (Block Projection Sketch) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpsConfig {
    /// Dimensions per block
    pub block_size: u16,
    /// Number of blocks (computed from dim)
    pub num_blocks: u16,
    /// Number of projections per block (1 or 2)
    pub num_projections: u16,
}

impl BpsConfig {
    /// Maximum safe value for num_blocks × num_projections to prevent u16 overflow.
    /// With max L1 diff of 255 per slot: 65535 / 255 = 257
    pub const MAX_SAFE_SLOTS: u32 = 257;
    
    /// Validate configuration to ensure BPS distance won't overflow u16.
    /// Returns error if num_blocks × num_projections × 255 > u16::MAX.
    pub fn validate(&self) -> Result<(), String> {
        let total_slots = self.num_blocks as u32 * self.num_projections as u32;
        if total_slots > Self::MAX_SAFE_SLOTS {
            return Err(format!(
                "BPS configuration would overflow u16: {} blocks × {} projections = {} slots (max {})",
                self.num_blocks, self.num_projections, total_slots, Self::MAX_SAFE_SLOTS
            ));
        }
        Ok(())
    }
    
    /// Theoretical maximum L1 distance for this configuration
    pub fn max_distance(&self) -> u32 {
        self.num_blocks as u32 * self.num_projections as u32 * 255
    }
}

impl Default for BpsConfig {
    fn default() -> Self {
        let num_blocks = (DEFAULT_DIM + DEFAULT_BPS_BLOCK_SIZE as u32 - 1) / DEFAULT_BPS_BLOCK_SIZE as u32;
        Self {
            block_size: DEFAULT_BPS_BLOCK_SIZE,
            num_blocks: num_blocks as u16,
            num_projections: DEFAULT_BPS_PROJECTIONS,
        }
    }
}

/// RDF (Rare-Dominant Fingerprint) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfConfig {
    /// Number of top dimensions to select per vector
    pub top_t: u16,
    /// Stripe shift (log2 of stripe size)
    pub stripe_shift: u8,
    /// Stop-dimension document frequency threshold
    pub stop_dim_threshold: u32,
    /// IDF weight (alpha)
    pub idf_weight: f32,
    /// Variance weight (beta)
    pub var_weight: f32,
}

impl Default for RdfConfig {
    fn default() -> Self {
        Self {
            top_t: DEFAULT_RDF_TOP_T,
            stripe_shift: DEFAULT_STRIPE_SHIFT,
            stop_dim_threshold: DEFAULT_STOP_DIM_THRESHOLD,
            idf_weight: 0.5,
            var_weight: 0.5,
        }
    }
}

/// Rerank configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Number of outliers to store per vector
    pub num_outliers: u8,
    /// Use percentile-based quantization
    pub percentile_quantization: bool,
    /// Percentile for scale computation (e.g., 0.99)
    pub scale_percentile: f32,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            num_outliers: DEFAULT_NUM_OUTLIERS,
            percentile_quantization: true,
            scale_percentile: 0.99,
        }
    }
}

/// Router configuration (for partitioned search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Number of partitions/lists
    pub n_lists: u32,
    /// Number of lists to probe
    pub n_probe: u32,
    /// Enable router (only for large datasets)
    pub enabled: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            n_lists: 128,
            n_probe: 8,
            enabled: false,
        }
    }
}

/// LSM/Segment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LsmConfig {
    /// Maximum vectors per mutable segment before sealing
    pub max_mutable_size: usize,
    /// Maximum number of segments before triggering compaction
    pub max_segments: usize,
    /// Compaction ratio (merge N segments into 1)
    pub compaction_ratio: usize,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            max_mutable_size: 100_000,
            max_segments: 8,
            compaction_ratio: 4,
        }
    }
}

/// Default query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Default k
    pub k: usize,
    /// Default L_A (RDF candidates)
    pub l_a: usize,
    /// Default L_B (BPS candidates)
    pub l_b: usize,
    /// Default R (rerank candidates)
    pub r: usize,
    /// Enable adaptive widening by default
    pub adaptive: bool,
    /// Widening factor for low confidence
    pub widening_factor: f32,
    /// Score gap threshold for confidence
    pub score_gap_threshold: f32,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            k: 10,
            l_a: 5000,
            l_b: 20000,
            r: 500,
            adaptive: true,
            widening_factor: 2.0,
            score_gap_threshold: 0.1,
        }
    }
}
