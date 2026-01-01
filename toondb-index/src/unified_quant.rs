// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Unified Quantized Vector Contract
//!
//! This module provides a single abstraction for all vector quantization formats:
//! - **F32**: Full precision (baseline)
//! - **F16/BF16**: Half precision (2× compression)
//! - **I8**: 8-bit integer quantization (4× compression)
//! - **PQ**: Product quantization (32× compression)
//! - **BPS**: Block Projection Sketch (coarse filtering)
//!
//! # Architecture
//!
//! ```text
//! Query → BPS Scan (coarse) → PQ Score (refine) → I8 Rerank (exact) → Results
//!            ↓                     ↓                    ↓
//!         1000 cands            100 cands            10 results
//! ```
//!
//! # Fallback Ladder
//!
//! The pipeline automatically falls back when:
//! - PQ codebooks not trained → Skip PQ, use I8 directly
//! - I8 not available → Use F32 rerank
//! - BPS not built → Start with PQ/I8 scan
//!
//! # Usage
//!
//! ```rust,ignore
//! let scorer = UnifiedScorer::new(&config);
//! let results = scorer.search(&query, k, recall_target)?;
//! ```

use std::fmt;

/// Quantization level (ordered by compression ratio).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum QuantLevel {
    /// Full precision f32 (1.0× compression)
    F32 = 0,
    /// Half precision f16 (2× compression)
    F16 = 1,
    /// Brain float bf16 (2× compression, better range)
    BF16 = 2,
    /// 8-bit integer (4× compression)
    I8 = 3,
    /// Product quantization (32× compression typical)
    PQ = 4,
    /// Block projection sketch (coarse filtering only)
    BPS = 5,
}

impl QuantLevel {
    /// Bytes per dimension for this level.
    pub const fn bytes_per_dim(self) -> f32 {
        match self {
            QuantLevel::F32 => 4.0,
            QuantLevel::F16 => 2.0,
            QuantLevel::BF16 => 2.0,
            QuantLevel::I8 => 1.0,
            QuantLevel::PQ => 0.125, // ~1 byte per 8 dims (typical)
            QuantLevel::BPS => 0.0625, // ~1 byte per 16 dims
        }
    }

    /// Expected recall at this level (rough estimates).
    pub const fn expected_recall(self) -> f32 {
        match self {
            QuantLevel::F32 => 1.0,
            QuantLevel::F16 => 0.999,
            QuantLevel::BF16 => 0.998,
            QuantLevel::I8 => 0.995,
            QuantLevel::PQ => 0.90,
            QuantLevel::BPS => 0.70,
        }
    }

    /// Compute cost per vector (relative to F32 = 1.0).
    pub const fn relative_cost(self) -> f32 {
        match self {
            QuantLevel::F32 => 1.0,
            QuantLevel::F16 => 0.6,
            QuantLevel::BF16 => 0.6,
            QuantLevel::I8 => 0.3,
            QuantLevel::PQ => 0.1,
            QuantLevel::BPS => 0.05,
        }
    }
}

impl fmt::Display for QuantLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantLevel::F32 => write!(f, "F32"),
            QuantLevel::F16 => write!(f, "F16"),
            QuantLevel::BF16 => write!(f, "BF16"),
            QuantLevel::I8 => write!(f, "I8"),
            QuantLevel::PQ => write!(f, "PQ"),
            QuantLevel::BPS => write!(f, "BPS"),
        }
    }
}

/// Unified storage format for quantized vectors.
///
/// This is the canonical representation that all vector data flows through.
/// Each format has a specific layout optimized for its access pattern.
#[derive(Clone)]
pub enum UnifiedQuantizedVector {
    /// Full precision f32 vector.
    F32(Vec<f32>),

    /// Half precision f16 (stored as u16 bitpattern).
    F16(Vec<u16>),

    /// Brain float bf16 (stored as u16 bitpattern).
    BF16(Vec<u16>),

    /// 8-bit integer with scale and zero-point.
    I8 {
        data: Vec<i8>,
        scale: f32,
        zero_point: i8,
    },

    /// Product quantization codes.
    PQ {
        /// PQ codes (one byte per subspace).
        codes: Vec<u8>,
        /// Number of subspaces.
        num_subspaces: usize,
        /// Precomputed scale for reconstruction.
        scale: f32,
    },

    /// Block projection sketch (for coarse filtering).
    BPS {
        /// Sketch bytes (one per block).
        sketch: Vec<u8>,
        /// Number of blocks.
        num_blocks: usize,
    },
}

impl UnifiedQuantizedVector {
    /// Get the quantization level.
    pub fn level(&self) -> QuantLevel {
        match self {
            UnifiedQuantizedVector::F32(_) => QuantLevel::F32,
            UnifiedQuantizedVector::F16(_) => QuantLevel::F16,
            UnifiedQuantizedVector::BF16(_) => QuantLevel::BF16,
            UnifiedQuantizedVector::I8 { .. } => QuantLevel::I8,
            UnifiedQuantizedVector::PQ { .. } => QuantLevel::PQ,
            UnifiedQuantizedVector::BPS { .. } => QuantLevel::BPS,
        }
    }

    /// Get dimension (or approximate for compressed formats).
    pub fn dimension(&self) -> usize {
        match self {
            UnifiedQuantizedVector::F32(v) => v.len(),
            UnifiedQuantizedVector::F16(v) => v.len(),
            UnifiedQuantizedVector::BF16(v) => v.len(),
            UnifiedQuantizedVector::I8 { data, .. } => data.len(),
            UnifiedQuantizedVector::PQ { num_subspaces, .. } => *num_subspaces * 8, // Approximate
            UnifiedQuantizedVector::BPS { num_blocks, .. } => *num_blocks * 16, // Approximate
        }
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            UnifiedQuantizedVector::F32(v) => v.len() * 4,
            UnifiedQuantizedVector::F16(v) => v.len() * 2,
            UnifiedQuantizedVector::BF16(v) => v.len() * 2,
            UnifiedQuantizedVector::I8 { data, .. } => data.len() + 5, // data + scale + zero
            UnifiedQuantizedVector::PQ { codes, .. } => codes.len() + 8, // codes + metadata
            UnifiedQuantizedVector::BPS { sketch, .. } => sketch.len() + 4,
        }
    }

    /// Convert to f32 vector (decode).
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            UnifiedQuantizedVector::F32(v) => v.clone(),
            UnifiedQuantizedVector::F16(v) => v.iter().map(|&x| f16_to_f32(x)).collect(),
            UnifiedQuantizedVector::BF16(v) => v.iter().map(|&x| bf16_to_f32(x)).collect(),
            UnifiedQuantizedVector::I8 {
                data,
                scale,
                zero_point,
            } => data
                .iter()
                .map(|&x| (x as f32 - *zero_point as f32) * scale)
                .collect(),
            UnifiedQuantizedVector::PQ { .. } => {
                // PQ decode requires codebooks - return zeros as placeholder
                vec![0.0; self.dimension()]
            }
            UnifiedQuantizedVector::BPS { .. } => {
                // BPS is not meant for reconstruction
                vec![0.0; self.dimension()]
            }
        }
    }

    /// Create from f32 vector with specified quantization level.
    pub fn from_f32(data: &[f32], level: QuantLevel) -> Self {
        match level {
            QuantLevel::F32 => UnifiedQuantizedVector::F32(data.to_vec()),
            QuantLevel::F16 => UnifiedQuantizedVector::F16(data.iter().map(|&x| f32_to_f16(x)).collect()),
            QuantLevel::BF16 => UnifiedQuantizedVector::BF16(data.iter().map(|&x| f32_to_bf16(x)).collect()),
            QuantLevel::I8 => {
                // Simple symmetric quantization
                let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                let quantized: Vec<i8> = data
                    .iter()
                    .map(|&x| (x / scale).clamp(-127.0, 127.0) as i8)
                    .collect();
                UnifiedQuantizedVector::I8 {
                    data: quantized,
                    scale,
                    zero_point: 0,
                }
            }
            QuantLevel::PQ => {
                // PQ requires training - return placeholder
                let num_subspaces = (data.len() + 7) / 8;
                UnifiedQuantizedVector::PQ {
                    codes: vec![0u8; num_subspaces],
                    num_subspaces,
                    scale: 1.0,
                }
            }
            QuantLevel::BPS => {
                // BPS projection
                let num_blocks = (data.len() + 15) / 16;
                let sketch: Vec<u8> = (0..num_blocks)
                    .map(|b| {
                        let start = b * 16;
                        let end = (start + 16).min(data.len());
                        let sum: f32 = data[start..end].iter().sum();
                        ((sum * 10.0).clamp(0.0, 255.0)) as u8
                    })
                    .collect();
                UnifiedQuantizedVector::BPS { sketch, num_blocks }
            }
        }
    }
}

/// Stage in the multi-stage retrieval pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Coarse filtering (BPS scan).
    Coarse,
    /// Refinement (PQ scoring).
    Refine,
    /// Final reranking (I8 or F32 exact).
    Rerank,
}

/// Configuration for the quantization pipeline.
#[derive(Debug, Clone)]
pub struct QuantPipelineConfig {
    /// Target recall (0.0 to 1.0).
    pub target_recall: f32,

    /// Maximum latency budget in microseconds.
    pub latency_budget_us: u64,

    /// Candidate count at each stage.
    pub stage_candidates: StageCandidates,

    /// Available quantization levels.
    pub available_levels: Vec<QuantLevel>,

    /// Whether to use adaptive stage selection.
    pub adaptive: bool,
}

/// Candidate counts at each pipeline stage.
#[derive(Debug, Clone)]
pub struct StageCandidates {
    /// Candidates after coarse stage.
    pub after_coarse: usize,
    /// Candidates after refinement stage.
    pub after_refine: usize,
    /// Final k results.
    pub final_k: usize,
}

impl Default for QuantPipelineConfig {
    fn default() -> Self {
        Self {
            target_recall: 0.95,
            latency_budget_us: 1000, // 1ms
            stage_candidates: StageCandidates {
                after_coarse: 1000,
                after_refine: 100,
                final_k: 10,
            },
            available_levels: vec![QuantLevel::F32, QuantLevel::I8],
            adaptive: true,
        }
    }
}

/// Unified scorer that handles all quantization formats.
pub struct UnifiedScorer {
    /// Pipeline configuration.
    config: QuantPipelineConfig,

    /// Best available level for reranking.
    rerank_level: QuantLevel,
}

impl UnifiedScorer {
    /// Create a new unified scorer.
    pub fn new(config: QuantPipelineConfig) -> Self {
        // Determine best rerank level from available
        let rerank_level = config
            .available_levels
            .iter()
            .filter(|l| matches!(l, QuantLevel::F32 | QuantLevel::I8))
            .min()
            .copied()
            .unwrap_or(QuantLevel::F32);

        Self {
            config,
            rerank_level,
        }
    }

    /// Get the rerank level.
    pub fn rerank_level(&self) -> QuantLevel {
        self.rerank_level
    }

    /// Compute I8 dot product between query and vector.
    #[inline]
    pub fn dot_i8(query: &[i8], vector: &[i8]) -> i32 {
        // Use SIMD if available
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::dot_i8_avx2(query, vector) };
            }
        }

        // Scalar fallback
        query
            .iter()
            .zip(vector.iter())
            .map(|(&a, &b)| (a as i32) * (b as i32))
            .sum()
    }

    /// Compute I8 dot product with dequantization.
    #[inline]
    pub fn dot_i8_dequant(
        query: &[i8],
        query_scale: f32,
        vector: &[i8],
        vector_scale: f32,
    ) -> f32 {
        let int_dot = Self::dot_i8(query, vector);
        int_dot as f32 * query_scale * vector_scale
    }

    /// AVX2 implementation of I8 dot product.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
        use std::arch::x86_64::*;

        let len = a.len().min(b.len());
        let chunks = len / 32;

        let mut acc = _mm256_setzero_si256();

        for i in 0..chunks {
            let idx = i * 32;
            let va = _mm256_loadu_si256(a.as_ptr().add(idx) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(idx) as *const __m256i);

            // Sign-extend and multiply-add
            let va_lo = _mm256_castsi256_si128(va);
            let va_hi = _mm256_extracti128_si256(va, 1);
            let vb_lo = _mm256_castsi256_si128(vb);
            let vb_hi = _mm256_extracti128_si256(vb, 1);

            let a_lo_16 = _mm256_cvtepi8_epi16(va_lo);
            let a_hi_16 = _mm256_cvtepi8_epi16(va_hi);
            let b_lo_16 = _mm256_cvtepi8_epi16(vb_lo);
            let b_hi_16 = _mm256_cvtepi8_epi16(vb_hi);

            let prod_lo = _mm256_madd_epi16(a_lo_16, b_lo_16);
            let prod_hi = _mm256_madd_epi16(a_hi_16, b_hi_16);

            acc = _mm256_add_epi32(acc, prod_lo);
            acc = _mm256_add_epi32(acc, prod_hi);
        }

        // Horizontal sum
        let acc_lo = _mm256_castsi256_si128(acc);
        let acc_hi = _mm256_extracti128_si256(acc, 1);
        let sum128 = _mm_add_epi32(acc_lo, acc_hi);
        let sum128 = _mm_hadd_epi32(sum128, sum128);
        let sum128 = _mm_hadd_epi32(sum128, sum128);
        let mut total = _mm_cvtsi128_si32(sum128);

        // Handle remainder
        for i in (chunks * 32)..len {
            total += (a[i] as i32) * (b[i] as i32);
        }

        total
    }

    /// Estimate recall at given candidate count for a level.
    pub fn estimate_recall(&self, level: QuantLevel, candidates: usize, total_vectors: usize) -> f32 {
        let base_recall = level.expected_recall();
        let coverage = (candidates as f32 / total_vectors as f32).min(1.0);
        base_recall * coverage.sqrt() // Rough model
    }

    /// Choose optimal pipeline stages given constraints.
    pub fn plan_pipeline(&self, total_vectors: usize) -> Vec<(PipelineStage, QuantLevel, usize)> {
        let mut stages = Vec::new();

        let has_bps = self.config.available_levels.contains(&QuantLevel::BPS);
        let has_pq = self.config.available_levels.contains(&QuantLevel::PQ);

        // Coarse stage (if BPS available and dataset is large enough)
        if has_bps && total_vectors > 10_000 {
            stages.push((
                PipelineStage::Coarse,
                QuantLevel::BPS,
                self.config.stage_candidates.after_coarse,
            ));
        }

        // Refine stage (if PQ available)
        if has_pq && total_vectors > 1_000 {
            stages.push((
                PipelineStage::Refine,
                QuantLevel::PQ,
                self.config.stage_candidates.after_refine,
            ));
        }

        // Rerank stage (always)
        stages.push((
            PipelineStage::Rerank,
            self.rerank_level,
            self.config.stage_candidates.final_k,
        ));

        stages
    }
}

// ============================================================================
// F16/BF16 conversion utilities
// ============================================================================

/// Convert f32 to f16 (IEEE 754 half-precision).
#[inline]
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN
        (sign << 15) | 0x7C00 | ((frac >> 13) as u16)
    } else if exp > 142 {
        // Overflow to infinity
        (sign << 15) | 0x7C00
    } else if exp < 113 {
        // Underflow to zero or subnormal
        if exp < 103 {
            sign << 15
        } else {
            let frac = frac | 0x800000;
            let shift = 126 - exp;
            (sign << 15) | ((frac >> shift) as u16)
        }
    } else {
        let new_exp = ((exp - 127 + 15) as u16) << 10;
        let new_frac = (frac >> 13) as u16;
        (sign << 15) | new_exp | new_frac
    }
}

/// Convert f16 to f32.
#[inline]
fn f16_to_f32(x: u16) -> f32 {
    let sign = ((x >> 15) as u32) << 31;
    let exp = ((x >> 10) & 0x1F) as u32;
    let frac = (x & 0x3FF) as u32;

    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal
            let mut frac = frac;
            let mut exp = 1u32;
            while (frac & 0x400) == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x3FF;
            sign | ((exp + 127 - 15) << 23) | (frac << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        sign | 0x7F800000 | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };

    f32::from_bits(bits)
}

/// Convert f32 to bf16 (Brain float).
#[inline]
fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Convert bf16 to f32.
#[inline]
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_level_ordering() {
        assert!(QuantLevel::F32 < QuantLevel::I8);
        assert!(QuantLevel::I8 < QuantLevel::PQ);
        assert!(QuantLevel::PQ < QuantLevel::BPS);
    }

    #[test]
    fn test_unified_vector_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vec = UnifiedQuantizedVector::from_f32(&data, QuantLevel::F32);
        
        assert_eq!(vec.level(), QuantLevel::F32);
        assert_eq!(vec.dimension(), 4);
        assert_eq!(vec.to_f32(), data);
    }

    #[test]
    fn test_unified_vector_i8() {
        let data = vec![0.5, -0.3, 0.8, -0.1];
        let vec = UnifiedQuantizedVector::from_f32(&data, QuantLevel::I8);
        
        assert_eq!(vec.level(), QuantLevel::I8);
        assert_eq!(vec.dimension(), 4);
        
        // Check reconstruction accuracy
        let reconstructed = vec.to_f32();
        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 0.1, "I8 reconstruction error too large");
        }
    }

    #[test]
    fn test_unified_vector_f16() {
        let data = vec![1.5, -2.25, 0.0, 100.0];
        let vec = UnifiedQuantizedVector::from_f32(&data, QuantLevel::F16);
        
        assert_eq!(vec.level(), QuantLevel::F16);
        
        let reconstructed = vec.to_f32();
        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 0.01 * orig.abs().max(1.0), 
                "F16 reconstruction error too large");
        }
    }

    #[test]
    fn test_unified_vector_bf16() {
        let data = vec![1.5, -2.25, 0.0, 100.0];
        let vec = UnifiedQuantizedVector::from_f32(&data, QuantLevel::BF16);
        
        assert_eq!(vec.level(), QuantLevel::BF16);
        
        let reconstructed = vec.to_f32();
        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            // BF16 has lower precision than F16
            assert!((orig - recon).abs() < 0.1 * orig.abs().max(1.0), 
                "BF16 reconstruction error too large");
        }
    }

    #[test]
    fn test_dot_i8() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b: Vec<i8> = vec![8, 7, 6, 5, 4, 3, 2, 1];
        
        let result = UnifiedScorer::dot_i8(&a, &b);
        let expected: i32 = a.iter().zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pipeline_planning() {
        let config = QuantPipelineConfig {
            available_levels: vec![QuantLevel::F32, QuantLevel::I8, QuantLevel::BPS, QuantLevel::PQ],
            ..Default::default()
        };
        
        let scorer = UnifiedScorer::new(config);
        
        // Small dataset: should skip coarse and refine
        let stages_small = scorer.plan_pipeline(500);
        assert_eq!(stages_small.len(), 1);
        assert_eq!(stages_small[0].0, PipelineStage::Rerank);
        
        // Large dataset: should use all stages
        let stages_large = scorer.plan_pipeline(100_000);
        assert!(stages_large.len() >= 2);
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 100.0, -0.001, 65504.0]; // Max F16 value
        
        for &v in &values {
            let f16_bits = f32_to_f16(v);
            let back = f16_to_f32(f16_bits);
            
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel_error = ((v - back) / v).abs();
                assert!(rel_error < 0.001 || (v - back).abs() < 0.001, 
                    "F16 roundtrip failed for {}: got {}", v, back);
            }
        }
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values = [0.0, 1.0, -1.0, 100.0, 1e10, -1e-10];
        
        for &v in &values {
            let bf16_bits = f32_to_bf16(v);
            let back = bf16_to_f32(bf16_bits);
            
            if v == 0.0 {
                assert!(back.abs() < 1e-10);
            } else {
                let rel_error = ((v - back) / v).abs();
                assert!(rel_error < 0.01, 
                    "BF16 roundtrip failed for {}: got {} (error {})", v, back, rel_error);
            }
        }
    }

    #[test]
    fn test_memory_usage() {
        let dim = 768;
        let data: Vec<f32> = (0..dim).map(|i| i as f32 / 1000.0).collect();
        
        let f32_vec = UnifiedQuantizedVector::from_f32(&data, QuantLevel::F32);
        let i8_vec = UnifiedQuantizedVector::from_f32(&data, QuantLevel::I8);
        let f16_vec = UnifiedQuantizedVector::from_f32(&data, QuantLevel::F16);
        
        assert_eq!(f32_vec.memory_bytes(), dim * 4);
        assert_eq!(i8_vec.memory_bytes(), dim + 5); // data + scale + zero
        assert_eq!(f16_vec.memory_bytes(), dim * 2);
        
        // Verify compression ratios
        assert!(i8_vec.memory_bytes() < f32_vec.memory_bytes() / 3);
        assert!(f16_vec.memory_bytes() < f32_vec.memory_bytes());
    }
}
