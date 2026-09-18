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
pub fn dot_i8_batch(query: &[i8], vectors: &[i8], scales: &[f32], dim: usize, results: &mut [f32]) {
    let n_vec = scales.len();
    assert!(query.len() >= dim, "query too short");
    assert!(vectors.len() >= n_vec * dim, "vectors buffer too small");
    assert!(results.len() >= n_vec, "results buffer too small");

    let features = cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx_vnni && dim >= 32 && dim <= MAX_VNNI_DIM {
            unsafe { dot_i8_batch_avxvnni(query, vectors, scales, dim, results) };
            return;
        }
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
    dot_i8_indexed_prefix(query, vectors, cand_ids, dim, dim, out_scores);
}

/// Dot products over only the first `prefix` dimensions of each candidate.
///
/// `stride` stays the full stored dimension, so this reads a contiguous
/// `prefix`-byte head of each record and skips the rest. Because the segment
/// stores Hadamard-rotated vectors, energy is spread evenly across dimensions
/// and a prefix dot is an unbiased estimate of the full dot - which makes this
/// usable as a cheap ranking key for pruning before full scoring.
///
/// Reading `prefix` of `stride` bytes touches `ceil(prefix/64)` cache lines per
/// record instead of `ceil(stride/64)`, and the scattered gather is
/// bandwidth-bound, so the saving is close to the byte ratio.
#[inline]
pub fn dot_i8_indexed_prefix(
    query: &[i8],
    vectors: &[i8],
    cand_ids: &[u32],
    stride: usize,
    prefix: usize,
    out_scores: &mut [i32],
) {
    assert!(prefix <= stride, "prefix cannot exceed stride");
    assert!(query.len() >= prefix);
    assert!(out_scores.len() >= cand_ids.len());

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-VNNI folds the multiply and the widening accumulate into one
        // instruction, and the batch form hoists the query correction term out
        // of the per-candidate loop. Only worth entering when there is a whole
        // 32-lane block to work on.
        if cpu_features().has_avx_vnni && prefix >= 32 && prefix <= MAX_VNNI_DIM {
            // Safety: AVX-VNNI is verified present, and the bounds above keep
            // the biased accumulator inside i32.
            unsafe { dot_i8_indexed_avxvnni(query, vectors, cand_ids, stride, prefix, out_scores) };
            return;
        }
    }

    for (i, &cand_id) in cand_ids.iter().enumerate() {
        let offset = cand_id as usize * stride;
        let vec = &vectors[offset..offset + prefix];
        out_scores[i] = dot_i8(&query[..prefix], vec);
    }
}

/// How many candidates ahead to request records for.
///
/// A scattered record is a cold miss costing on the order of 80 ns while
/// scoring one costs a few nanoseconds, so requesting only the next candidate
/// hides barely a tenth of the stall and leaves the loop latency-bound: the
/// measured gather rate was several times below this machine's DRAM bandwidth,
/// which is the signature of too few outstanding misses rather than too little
/// bandwidth. The distance has to cover the miss with `distance x per-candidate
/// work`, which lands in the low tens.
///
/// Sixteen was chosen by sweeping 1, 4, 8, 16, 32 and 64 - the curve falls
/// steeply to 16 and is flat after. Scaling it down for records that span more
/// cache lines was also measured and was consistently worse, so both the narrow
/// prefix pass and full scoring use the same depth.
const PREFETCH_DISTANCE: usize = 16;

/// Largest dimension for which the biased AVX-VNNI accumulator cannot overflow.
///
/// `VPDPBUSD` accumulates products of an unsigned byte and a signed byte, so
/// after biasing the stored vector each term is bounded by `255 * 128 = 32_640`
/// rather than the `127 * 127` of the unbiased signed product. The accumulator
/// is i32, so the dimension must satisfy `dim * 32_640 <= i32::MAX`. Beyond this
/// the scalar and AVX2 paths still apply, since they never form the biased term.
const MAX_VNNI_DIM: usize = (i32::MAX as usize) / 32_640;

/// AVX-VNNI indexed batch dot product.
///
/// `VPDPBUSD` multiplies an *unsigned* byte by a *signed* byte, but both the
/// query and the stored vectors are signed. Biasing the stored vector into
/// unsigned form with `v ^ 0x80` — which is exactly `v + 128` for a signed byte
/// — gives
///
/// ```text
/// dpbusd(v + 128, q) = Σ (vᵢ + 128)·qᵢ = Σ vᵢqᵢ + 128·Σ qᵢ
/// ```
///
/// so the true dot product is the accumulator minus `128·Σ qᵢ`. The correction
/// depends only on the query, which is why this is written as a batch: `Σ qᵢ` is
/// computed once for every candidate instead of once per candidate. That leaves
/// roughly four uops per 32 lanes against the twelve the sign-extend-and-madd
/// path needs, and the results are bit-identical because every step is integer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn dot_i8_indexed_avxvnni(
    query: &[i8],
    vectors: &[i8],
    cand_ids: &[u32],
    stride: usize,
    prefix: usize,
    out_scores: &mut [i32],
) {
    use std::arch::x86_64::*;

    unsafe {
        let dim_aligned = (prefix / 32) * 32;
        // Hoisted query correction: the whole reason for the batch form. Only
        // the lanes that go through the biased path are corrected, so the sum
        // covers the aligned prefix and the scalar tail is left untouched.
        let sum_q: i32 = query[..dim_aligned].iter().map(|&x| x as i32).sum();
        let correction = 128 * sum_q;
        let bias = _mm256_set1_epi8(0x80u8 as i8);

        for (i, &cand_id) in cand_ids.iter().enumerate() {
            let offset = cand_id as usize * stride;
            let vptr = vectors.as_ptr().add(offset);
            let qptr = query.as_ptr();

            // Candidate ids arrive in arbitrary order, so the hardware stride
            // prefetcher cannot help; every record is a cold, scattered miss.
            // Issue the request far enough ahead that the line has arrived by
            // the time the loop reaches it - see `PREFETCH_DISTANCE`.
            if let Some(&future) = cand_ids.get(i + PREFETCH_DISTANCE) {
                let fptr = vectors.as_ptr().add(future as usize * stride);
                let mut byte = 0usize;
                while byte < prefix {
                    _mm_prefetch(fptr.add(byte), _MM_HINT_T0);
                    byte += 64;
                }
            }

            let biased = dot_i8_vnni_core(qptr, vptr, dim_aligned, bias);

            // Tail below a full 32-lane block never went through the bias, so it
            // contributes its plain signed product with no correction.
            let mut tail = 0i32;
            for k in dim_aligned..prefix {
                tail +=
                    (*query.get_unchecked(k) as i32) * (*vectors.get_unchecked(offset + k) as i32);
            }

            *out_scores.get_unchecked_mut(i) = biased - correction + tail;
        }
    }
}

/// One biased VNNI dot product over `dim_aligned` lanes (a multiple of 32).
///
/// Returns `Σ (vᵢ + 128)·qᵢ`; the caller subtracts the hoisted `128·Σ qᵢ`. Two
/// accumulators keep the dpbusd latency chain off the critical path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
#[inline]
unsafe fn dot_i8_vnni_core(
    qptr: *const i8,
    vptr: *const i8,
    dim_aligned: usize,
    bias: std::arch::x86_64::__m256i,
) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();

        let mut d = 0usize;
        while d + 64 <= dim_aligned {
            let v0 = _mm256_loadu_si256(vptr.add(d) as *const __m256i);
            let q0 = _mm256_loadu_si256(qptr.add(d) as *const __m256i);
            let v1 = _mm256_loadu_si256(vptr.add(d + 32) as *const __m256i);
            let q1 = _mm256_loadu_si256(qptr.add(d + 32) as *const __m256i);
            acc0 = _mm256_dpbusd_avx_epi32(acc0, _mm256_xor_si256(v0, bias), q0);
            acc1 = _mm256_dpbusd_avx_epi32(acc1, _mm256_xor_si256(v1, bias), q1);
            d += 64;
        }
        while d + 32 <= dim_aligned {
            let v0 = _mm256_loadu_si256(vptr.add(d) as *const __m256i);
            let q0 = _mm256_loadu_si256(qptr.add(d) as *const __m256i);
            acc0 = _mm256_dpbusd_avx_epi32(acc0, _mm256_xor_si256(v0, bias), q0);
            d += 32;
        }

        let acc = _mm256_add_epi32(acc0, acc1);
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256(acc, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_hadd_epi32(s, s);
        let s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(s)
    }
}

/// AVX-VNNI contiguous batch dot product.
///
/// Same bias identity as the indexed kernel, but the candidates are laid out
/// consecutively, so the hardware stride prefetcher covers the loads and no
/// software prefetch is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn dot_i8_batch_avxvnni(
    query: &[i8],
    vectors: &[i8],
    scales: &[f32],
    dim: usize,
    results: &mut [f32],
) {
    use std::arch::x86_64::*;

    unsafe {
        let dim_aligned = (dim / 32) * 32;
        let sum_q: i32 = query[..dim_aligned].iter().map(|&x| x as i32).sum();
        let correction = 128 * sum_q;
        let bias = _mm256_set1_epi8(0x80u8 as i8);
        let qptr = query.as_ptr();

        for (i, &scale) in scales.iter().enumerate() {
            let offset = i * dim;
            let vptr = vectors.as_ptr().add(offset);
            let biased = dot_i8_vnni_core(qptr, vptr, dim_aligned, bias);

            let mut tail = 0i32;
            for k in dim_aligned..dim {
                tail +=
                    (*query.get_unchecked(k) as i32) * (*vectors.get_unchecked(offset + k) as i32);
            }

            *results.get_unchecked_mut(i) = (biased - correction + tail) as f32 * scale;
        }
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
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_i8_large() {
        // Test with typical embedding dimension
        let dim = 768;
        let a: Vec<i8> = (0..dim)
            .map(|i| ((i % 256) as i8).wrapping_add(-128))
            .collect();
        let b: Vec<i8> = (0..dim)
            .map(|i| ((i * 7 % 256) as i8).wrapping_add(-128))
            .collect();

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

    /// Deterministic full-range i8 values, including both endpoints. The bias
    /// trick maps `-128` to unsigned `0` and `127` to `255`, so the endpoints
    /// are exactly where an off-by-one in the correction would show up.
    fn pseudo_i8(seed: u64, n: usize) -> Vec<i8> {
        let mut s = seed | 1;
        (0..n)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % 8 {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    2 => 0,
                    _ => (s >> 56) as i8,
                }
            })
            .collect()
    }

    #[test]
    fn indexed_batch_dot_is_bit_identical_to_the_scalar_reference() {
        // The batch kernel may use AVX-VNNI, which cannot multiply two signed
        // bytes and so computes a biased product and subtracts a per-query
        // correction. That correction is applied only to the lanes that went
        // through the vectorised path, so dimensions that are not a multiple of
        // 32 are the interesting case: a correction computed over the whole
        // query instead of the aligned prefix would be wrong only there.
        // Integer arithmetic throughout means "close enough" is not the bar —
        // the results must be equal.
        for &dim in &[1usize, 7, 31, 32, 33, 63, 64, 96, 127, 128, 384, 768, 1000] {
            let n_vec = 17usize;
            let vectors = pseudo_i8(0xA5A5 ^ dim as u64, n_vec * dim);
            let query = pseudo_i8(0x1234 ^ dim as u64, dim);
            let cand_ids: Vec<u32> = (0..n_vec as u32).rev().collect();

            let mut got = vec![0i32; n_vec];
            dot_i8_indexed(&query, &vectors, &cand_ids, dim, &mut got);

            for (slot, &cand) in cand_ids.iter().enumerate() {
                let off = cand as usize * dim;
                let want: i32 = (0..dim)
                    .map(|k| query[k] as i32 * vectors[off + k] as i32)
                    .sum();
                assert_eq!(
                    got[slot], want,
                    "dim={} candidate={} mismatch (batch {} vs reference {})",
                    dim, cand, got[slot], want
                );
            }
        }
    }

    #[test]
    fn indexed_batch_dot_matches_the_single_vector_kernel() {
        // Both kernels are public and callers mix them, so they must agree.
        for &dim in &[32usize, 100, 768] {
            let n_vec = 9usize;
            let vectors = pseudo_i8(0xBEEF ^ dim as u64, n_vec * dim);
            let query = pseudo_i8(0xF00D ^ dim as u64, dim);
            let cand_ids: Vec<u32> = (0..n_vec as u32).collect();

            let mut got = vec![0i32; n_vec];
            dot_i8_indexed(&query, &vectors, &cand_ids, dim, &mut got);

            for (slot, &cand) in cand_ids.iter().enumerate() {
                let off = cand as usize * dim;
                let single = dot_i8(&query, &vectors[off..off + dim]);
                assert_eq!(got[slot], single, "dim={} candidate={}", dim, cand);
            }
        }
    }

    #[test]
    fn contiguous_batch_dot_is_bit_identical_to_the_scalar_reference() {
        // The biased VNNI accumulator only covers whole 32-lane blocks, so the
        // hoisted correction must span the aligned prefix and not the whole
        // query. Dimensions that are not multiples of 32 are the only ones that
        // can catch a correction applied over the wrong range.
        for dim in [32usize, 33, 63, 64, 96, 127, 128, 255, 768, 769] {
            let query = pseudo_i8(0x1234_5678, dim);
            let n_vec = 9usize;
            let vectors = pseudo_i8(0x9ABC_DEF0, n_vec * dim);
            let scales: Vec<f32> = (0..n_vec).map(|i| 0.5 + i as f32 * 0.25).collect();

            let mut got = vec![0.0f32; n_vec];
            dot_i8_batch(&query, &vectors, &scales, dim, &mut got);

            for (i, &scale) in scales.iter().enumerate() {
                let want: i32 = (0..dim)
                    .map(|k| query[k] as i32 * vectors[i * dim + k] as i32)
                    .sum();
                assert_eq!(
                    got[i],
                    want as f32 * scale,
                    "dim={} vector={} batch dot diverged from the scalar reference",
                    dim,
                    i
                );
            }
        }
    }

    #[test]
    fn prefix_dot_reads_the_head_of_each_record_and_skips_the_rest() {
        // The whole point of the prefix kernel is that stride and scored length
        // differ. Mixing them up still produces plausible numbers, so the test
        // fills the unscored tail of every record with values that would
        // dominate the result if they were ever read.
        let stride = 768usize;
        let n_vec = 40usize;
        let query = pseudo_i8(0x5151, stride);

        for prefix in [32usize, 64, 96, 128, 129, 256, 768] {
            let mut vectors = vec![0i8; n_vec * stride];
            for v in 0..n_vec {
                let head = pseudo_i8(0x7000 + v as u64, prefix);
                vectors[v * stride..v * stride + prefix].copy_from_slice(&head);
                // Poison the tail: large values that must not reach the score.
                for k in prefix..stride {
                    vectors[v * stride + k] = if k % 2 == 0 { 127 } else { -128 };
                }
            }

            let cand_ids: Vec<u32> = (0..n_vec as u32).filter(|i| i % 3 != 0).collect();
            let mut got = vec![0i32; cand_ids.len()];
            dot_i8_indexed_prefix(&query, &vectors, &cand_ids, stride, prefix, &mut got);

            for (i, &cand) in cand_ids.iter().enumerate() {
                let base = cand as usize * stride;
                let want: i32 = (0..prefix)
                    .map(|k| query[k] as i32 * vectors[base + k] as i32)
                    .sum();
                assert_eq!(
                    got[i], want,
                    "prefix={} candidate={} scored the wrong byte range",
                    prefix, cand
                );
            }
        }
    }
}
