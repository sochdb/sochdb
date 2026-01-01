//! Int8 Dot Product Kernel
//!
//! This module provides SIMD-accelerated int8 dot product computation
//! for reranking candidates after the initial BPS scan.
//!
//! # Algorithm
//!
//! ```text
//! dot(Q, V) = Σ_{d=0}^{D-1} Q[d] × V[d]
//! ```
//!
//! # Overflow Analysis
//!
//! For D=768 dimensions with i8 values in [-127, 127]:
//! ```text
//! max_product = 127 × 127 = 16,129
//! max_sum = 768 × 16,129 = 12,387,072 < 2^31 - 1 (i32 max)
//! ```
//! Thus, i32 accumulation is sufficient.
//!
//! # Implementation Strategy
//!
//! ## x86_64 AVX2
//! Uses sign-extension to i16 followed by `_mm256_madd_epi16`:
//! 1. Load 32 i8 values
//! 2. Sign-extend to 2×16 i16 values
//! 3. Multiply-add pairs: (a0*b0 + a1*b1) -> i32
//! 4. Accumulate i32 results
//!
//! ## ARM NEON
//! Uses `vmull_s8` to multiply 8 i8 pairs to i16, then `vpadalq_s16` to
//! widen and accumulate to i32.
//!
//! ## Future: VNNI/SDOT
//! - AVX-512 VNNI: `_mm256_dpbssd_epi32` (single instruction, i8×i8→i32)
//! - ARM v8.2 SDOT: `vdotq_s32` (single instruction, i8×i8→i32)

use super::dispatch::cpu_features;

/// Compute the dot product of two i8 vectors.
///
/// # Arguments
/// * `a` - First vector (i8)
/// * `b` - Second vector (i8, same length as `a`)
///
/// # Returns
/// The dot product as i32
///
/// # Panics
/// Panics if `a.len() != b.len()`
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    
    let features = cpu_features();
    
    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx2 {
            // Safety: AVX2 feature is verified
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if features.has_neon {
            // Safety: NEON is mandatory on aarch64
            return unsafe { dot_i8_neon(a, b) };
        }
    }
    
    dot_i8_scalar(a, b)
}

/// Compute dot products for a batch of vectors with dequantization.
///
/// Computes: `result[i] = dot(query, vectors[i * dim..(i+1) * dim]) * scales[i]`
///
/// # Arguments
/// * `query` - Query vector (i8)
/// * `vectors` - Flattened database vectors (i8, n_vec × dim)
/// * `scales` - Per-vector dequantization scales
/// * `dim` - Dimension of each vector
/// * `results` - Output dequantized dot products
///
/// # Panics
/// Panics if buffer sizes are inconsistent
#[inline]
pub fn dot_i8_batch(
    query: &[i8],
    vectors: &[i8],
    scales: &[f32],
    dim: usize,
    results: &mut [f32],
) {
    let n_vec = scales.len();
    assert!(query.len() >= dim, "query too short");
    assert!(vectors.len() >= n_vec * dim, "vectors buffer too small");
    assert!(results.len() >= n_vec, "results buffer too small");
    
    let features = cpu_features();
    
    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx2 {
            unsafe { dot_i8_batch_avx2(query, vectors, scales, dim, results) };
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if features.has_neon {
            unsafe { dot_i8_batch_neon(query, vectors, scales, dim, results) };
            return;
        }
    }
    
    dot_i8_batch_scalar(query, vectors, scales, dim, results);
}

/// Compute dot products for indexed candidates.
///
/// # Arguments
/// * `query` - Query vector (i8)
/// * `vectors` - All vectors (i8, total_vecs × dim)
/// * `cand_ids` - Candidate indices to compute
/// * `dim` - Dimension of each vector
/// * `out_scores` - Output i32 dot products
#[inline]
pub fn dot_i8_indexed(
    query: &[i8],
    vectors: &[i8],
    cand_ids: &[u32],
    dim: usize,
    out_scores: &mut [i32],
) {
    assert!(query.len() >= dim);
    assert!(out_scores.len() >= cand_ids.len());
    
    for (i, &cand_id) in cand_ids.iter().enumerate() {
        let offset = cand_id as usize * dim;
        let vec = &vectors[offset..offset + dim];
        out_scores[i] = dot_i8(&query[..dim], vec);
    }
}

// ============================================================================
// x86_64 AVX2 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    
    unsafe {
        let len = a.len();
        let dim_aligned = (len / 32) * 32;
        
        let mut acc = _mm256_setzero_si256();
        
        // Main loop: process 32 dimensions per iteration
        for d in (0..dim_aligned).step_by(32) {
            // Load 32 bytes from each vector
            let q = _mm256_loadu_si256(a.as_ptr().add(d) as *const __m256i);
            let v = _mm256_loadu_si256(b.as_ptr().add(d) as *const __m256i);
            
            // For signed × signed, we use sign extension to i16 then madd
            // Extract low and high 128-bit lanes
            let q_lo = _mm256_castsi256_si128(q);
            let q_hi = _mm256_extracti128_si256(q, 1);
            let v_lo = _mm256_castsi256_si128(v);
            let v_hi = _mm256_extracti128_si256(v, 1);
            
            // Sign-extend i8 to i16
            let q_lo_16 = _mm256_cvtepi8_epi16(q_lo);
            let q_hi_16 = _mm256_cvtepi8_epi16(q_hi);
            let v_lo_16 = _mm256_cvtepi8_epi16(v_lo);
            let v_hi_16 = _mm256_cvtepi8_epi16(v_hi);
            
            // Multiply i16 × i16 → i32 with horizontal add (madd)
            // madd: (a0*b0 + a1*b1, a2*b2 + a3*b3, ...) -> 8 i32
            let prod_lo = _mm256_madd_epi16(q_lo_16, v_lo_16);
            let prod_hi = _mm256_madd_epi16(q_hi_16, v_hi_16);
            
            // Accumulate
            acc = _mm256_add_epi32(acc, prod_lo);
            acc = _mm256_add_epi32(acc, prod_hi);
        }
        
        // Horizontal sum of acc (8 × i32)
        let acc_lo = _mm256_castsi256_si128(acc);
        let acc_hi = _mm256_extracti128_si256(acc, 1);
        let sum128 = _mm_add_epi32(acc_lo, acc_hi);
        
        // Horizontal add within 128-bit register
        let sum128 = _mm_hadd_epi32(sum128, sum128);
        let sum128 = _mm_hadd_epi32(sum128, sum128);
        
        let mut result = _mm_cvtsi128_si32(sum128);
        
        // Handle remaining dimensions
        for d in dim_aligned..len {
            result += (a[d] as i32) * (b[d] as i32);
        }
        
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_batch_avx2(
    query: &[i8],
    vectors: &[i8],
    scales: &[f32],
    dim: usize,
    results: &mut [f32],
) {
    unsafe {
        let n_vec = scales.len();
        
        for v in 0..n_vec {
            let offset = v * dim;
            let vec = &vectors[offset..offset + dim];
            let int_dot = dot_i8_avx2(&query[..dim], vec);
            results[v] = int_dot as f32 * scales[v];
        }
    }
}

// ============================================================================
// aarch64 NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    
    unsafe {
        let len = a.len();
        let mut acc = vdupq_n_s32(0);
        
        let mut i = 0;
        
        // Process 16 elements at a time
        while i + 16 <= len {
            // Load 16 i8 values each
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            
            // Widen to i16 and multiply
            let lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            let hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            
            // Widen to i32 and accumulate
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
            
            i += 16;
        }
        
        // Horizontal sum
        let mut result = vaddvq_s32(acc);
        
        // Handle remainder
        while i < len {
            result += (a[i] as i32) * (b[i] as i32);
            i += 1;
        }
        
        result
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_batch_neon(
    query: &[i8],
    vectors: &[i8],
    scales: &[f32],
    dim: usize,
    results: &mut [f32],
) {
    unsafe {
        let n_vec = scales.len();
        
        for v in 0..n_vec {
            let offset = v * dim;
            let vec = &vectors[offset..offset + dim];
            let int_dot = dot_i8_neon(&query[..dim], vec);
            results[v] = int_dot as f32 * scales[v];
        }
    }
}

// ============================================================================
// Scalar Fallback
// ============================================================================

/// Scalar dot product
#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32) * (y as i32))
        .sum()
}

/// Scalar batch with dequantization
#[inline]
fn dot_i8_batch_scalar(
    query: &[i8],
    vectors: &[i8],
    scales: &[f32],
    dim: usize,
    results: &mut [f32],
) {
    for (i, &scale) in scales.iter().enumerate() {
        let offset = i * dim;
        let vec = &vectors[offset..offset + dim];
        let int_dot = dot_i8_scalar(&query[..dim], vec);
        results[i] = int_dot as f32 * scale;
    }
}

// ============================================================================
// L2 Distance (bonus)
// ============================================================================

/// Compute squared L2 distance between two i8 vectors.
///
/// dist = sum((a[i] - b[i])^2)
#[inline]
pub fn l2_distance_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "aarch64")]
    {
        let features = cpu_features();
        if features.has_neon {
            return unsafe { l2_distance_i8_neon(a, b) };
        }
    }
    
    // Scalar fallback
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = (x as i32) - (y as i32);
            diff * diff
        })
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_distance_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    
    unsafe {
        let len = a.len();
        let mut acc = vdupq_n_s32(0);
        let mut i = 0;
        
        while i + 16 <= len {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            
            // Compute difference (widen to avoid overflow)
            let diff_lo = vsubl_s8(vget_low_s8(va), vget_low_s8(vb));
            let diff_hi = vsubl_s8(vget_high_s8(va), vget_high_s8(vb));
            
            // Square and accumulate
            acc = vmlal_s16(acc, vget_low_s16(diff_lo), vget_low_s16(diff_lo));
            acc = vmlal_s16(acc, vget_high_s16(diff_lo), vget_high_s16(diff_lo));
            acc = vmlal_s16(acc, vget_low_s16(diff_hi), vget_low_s16(diff_hi));
            acc = vmlal_s16(acc, vget_high_s16(diff_hi), vget_high_s16(diff_hi));
            
            i += 16;
        }
        
        let mut result = vaddvq_s32(acc);
        
        while i < len {
            let diff = (a[i] as i32) - (b[i] as i32);
            result += diff * diff;
            i += 1;
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dot_i8_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b: Vec<i8> = vec![8, 7, 6, 5, 4, 3, 2, 1];
        
        let result = dot_i8(&a, &b);
        let expected: i32 = a.iter().zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum();
        
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_dot_i8_large() {
        // Test with typical embedding dimension
        let dim = 768;
        let a: Vec<i8> = (0..dim).map(|i| ((i % 256) as i8).wrapping_add(-128)).collect();
        let b: Vec<i8> = (0..dim).map(|i| ((i * 7 % 256) as i8).wrapping_add(-128)).collect();
        
        let result = dot_i8(&a, &b);
        let expected = dot_i8_scalar(&a, &b);
        
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_dot_i8_batch() {
        let dim = 128;
        let n_vec = 10;
        let query: Vec<i8> = (0..dim).map(|i| (i % 127) as i8).collect();
        let vectors: Vec<i8> = (0..n_vec * dim).map(|i| ((i * 3) % 127) as i8).collect();
        let scales: Vec<f32> = (0..n_vec).map(|i| 0.01 * (i + 1) as f32).collect();
        let mut results = vec![0.0f32; n_vec];
        
        dot_i8_batch(&query, &vectors, &scales, dim, &mut results);
        
        // Verify against scalar
        let mut expected = vec![0.0f32; n_vec];
        dot_i8_batch_scalar(&query, &vectors, &scales, dim, &mut expected);
        
        for (r, e) in results.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "result={}, expected={}", r, e);
        }
    }
    
    #[test]
    fn test_l2_distance() {
        let a: Vec<i8> = vec![10, 20, 30, 40];
        let b: Vec<i8> = vec![11, 22, 33, 44];
        
        let result = l2_distance_i8(&a, &b);
        // (10-11)^2 + (20-22)^2 + (30-33)^2 + (40-44)^2 = 1 + 4 + 9 + 16 = 30
        assert_eq!(result, 30);
    }
}
