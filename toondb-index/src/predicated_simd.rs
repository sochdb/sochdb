// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Predicated SIMD Kernels
//!
//! This module provides branch-free SIMD operations using masked loads/stores
//! and predicated execution to eliminate branch mispredictions in hot paths.
//!
//! # Branch Misprediction Problem
//!
//! Traditional HNSW search has branches like:
//! ```text
//! if distance < worst_distance {
//!     update_candidate_list();  // Branch misprediction: 20+ cycle penalty
//! }
//! ```
//!
//! With random data, this branch is unpredictable (~50% taken rate),
//! causing ~10 mispredictions per search at 20 cycles each = 200 wasted cycles.
//!
//! # Predicated Solution
//!
//! Using masked operations:
//! ```text
//! mask = distance < worst_distance;  // No branch
//! result = blend(old_value, new_value, mask);  // Branchless select
//! ```
//!
//! # Supported Operations
//!
//! - **Masked min/max**: `min_mask(a, b, mask)` returns `mask ? min(a,b) : a`
//! - **Conditional store**: Only write if condition is true
//! - **Blend/select**: Choose between two values based on mask
//! - **Horizontal reduce with mask**: Only reduce active lanes
//!
//! # Architecture Support
//!
//! - **AVX-512**: Native mask registers (k0-k7), masked loads/stores
//! - **AVX2**: Simulate masks with `vblendvps`
//! - **NEON**: Use `vbslq` for blending
//! - **Scalar**: Ternary operator fallback

/// Mask type for predicated operations.
/// True lanes are processed, false lanes are preserved/ignored.
#[derive(Debug, Clone, Copy)]
pub struct SimdMask {
    /// Bitmask for 4-wide operations (bits 0-3 active)
    bits: u8,
}

impl SimdMask {
    /// All lanes active.
    pub const ALL: SimdMask = SimdMask { bits: 0x0F };
    
    /// No lanes active.
    pub const NONE: SimdMask = SimdMask { bits: 0x00 };

    /// Create mask from bitmask.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self { bits: bits & 0x0F }
    }

    /// Create mask from comparison results.
    #[inline]
    pub fn from_cmp_lt(a: [f32; 4], b: [f32; 4]) -> Self {
        let mut bits = 0u8;
        for i in 0..4 {
            if a[i] < b[i] {
                bits |= 1 << i;
            }
        }
        Self { bits }
    }

    /// Create mask from comparison with scalar.
    #[inline]
    pub fn from_cmp_lt_scalar(a: [f32; 4], b: f32) -> Self {
        let mut bits = 0u8;
        for i in 0..4 {
            if a[i] < b {
                bits |= 1 << i;
            }
        }
        Self { bits }
    }

    /// Get the underlying bits.
    #[inline]
    pub const fn bits(self) -> u8 {
        self.bits
    }

    /// Check if lane is active.
    #[inline]
    pub const fn is_active(self, lane: usize) -> bool {
        (self.bits >> lane) & 1 == 1
    }

    /// Count active lanes.
    #[inline]
    pub const fn count(self) -> u32 {
        self.bits.count_ones()
    }

    /// Check if any lane is active.
    #[inline]
    pub const fn any(self) -> bool {
        self.bits != 0
    }

    /// Check if all lanes are active.
    #[inline]
    pub const fn all(self) -> bool {
        self.bits == 0x0F
    }

    /// AND two masks.
    #[inline]
    pub const fn and(self, other: Self) -> Self {
        Self { bits: self.bits & other.bits }
    }

    /// OR two masks.
    #[inline]
    pub const fn or(self, other: Self) -> Self {
        Self { bits: self.bits | other.bits }
    }

    /// NOT mask.
    #[inline]
    pub const fn not(self) -> Self {
        Self { bits: (!self.bits) & 0x0F }
    }
}

/// Predicated SIMD operations (4-wide f32).
pub struct PredicatedSimd;

impl PredicatedSimd {
    /// Blend two vectors based on mask: mask ? b : a
    #[inline]
    #[allow(unreachable_code)]
    pub fn blend(a: [f32; 4], b: [f32; 4], mask: SimdMask) -> [f32; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::blend_avx2(a, b, mask) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::blend_neon(a, b, mask) };
        }

        // Scalar fallback
        Self::blend_scalar(a, b, mask)
    }

    /// Scalar blend implementation.
    #[inline]
    fn blend_scalar(a: [f32; 4], b: [f32; 4], mask: SimdMask) -> [f32; 4] {
        [
            if mask.is_active(0) { b[0] } else { a[0] },
            if mask.is_active(1) { b[1] } else { a[1] },
            if mask.is_active(2) { b[2] } else { a[2] },
            if mask.is_active(3) { b[3] } else { a[3] },
        ]
    }

    /// AVX2 blend using vblendvps.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blend_avx2(a: [f32; 4], b: [f32; 4], mask: SimdMask) -> [f32; 4] {
        use std::arch::x86_64::*;

        // Expand mask bits to full 32-bit masks
        let mask_expanded: [i32; 4] = [
            if mask.is_active(0) { -1i32 } else { 0 },
            if mask.is_active(1) { -1i32 } else { 0 },
            if mask.is_active(2) { -1i32 } else { 0 },
            if mask.is_active(3) { -1i32 } else { 0 },
        ];

        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let vm = _mm_loadu_si128(mask_expanded.as_ptr() as *const __m128i);
        
        let result = _mm_blendv_ps(va, vb, _mm_castsi128_ps(vm));
        
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), result);
        out
    }

    /// NEON blend using vbslq.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn blend_neon(a: [f32; 4], b: [f32; 4], mask: SimdMask) -> [f32; 4] {
        use std::arch::aarch64::*;

        // Expand mask bits to full 32-bit masks
        let mask_expanded: [u32; 4] = [
            if mask.is_active(0) { 0xFFFFFFFF } else { 0 },
            if mask.is_active(1) { 0xFFFFFFFF } else { 0 },
            if mask.is_active(2) { 0xFFFFFFFF } else { 0 },
            if mask.is_active(3) { 0xFFFFFFFF } else { 0 },
        ];

        // SAFETY: We're in an unsafe function, NEON intrinsics are safe with valid pointers
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vm = vld1q_u32(mask_expanded.as_ptr());

            // vbslq: bitwise select - picks bits from b where mask is 1, a where 0
            let result = vbslq_f32(vm, vb, va);

            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            out
        }
    }

    /// Predicated minimum: mask ? min(a, b) : a
    #[inline]
    pub fn min_pred(a: [f32; 4], b: [f32; 4], mask: SimdMask) -> [f32; 4] {
        let min_ab = Self::min(a, b);
        Self::blend(a, min_ab, mask)
    }

    /// Predicated maximum: mask ? max(a, b) : a
    #[inline]
    pub fn max_pred(a: [f32; 4], b: [f32; 4], mask: SimdMask) -> [f32; 4] {
        let max_ab = Self::max(a, b);
        Self::blend(a, max_ab, mask)
    }

    /// Element-wise minimum.
    #[inline]
    #[allow(unreachable_code)]
    pub fn min(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::min_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::min_neon(a, b) };
        }

        [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2]), a[3].min(b[3])]
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min_avx2(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        use std::arch::x86_64::*;
        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let result = _mm_min_ps(va, vb);
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), result);
        out
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn min_neon(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        use std::arch::aarch64::*;
        // SAFETY: We're in an unsafe function, NEON intrinsics are safe with valid pointers
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let result = vminq_f32(va, vb);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            out
        }
    }

    /// Element-wise maximum.
    #[inline]
    #[allow(unreachable_code)]
    pub fn max(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::max_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::max_neon(a, b) };
        }

        [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2]), a[3].max(b[3])]
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn max_avx2(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        use std::arch::x86_64::*;
        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let result = _mm_max_ps(va, vb);
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), result);
        out
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn max_neon(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        use std::arch::aarch64::*;
        // SAFETY: We're in an unsafe function, NEON intrinsics are safe with valid pointers
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let result = vmaxq_f32(va, vb);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            out
        }
    }

    /// Horizontal minimum (returns min of all lanes).
    #[inline]
    pub fn horizontal_min(v: [f32; 4]) -> f32 {
        v[0].min(v[1]).min(v[2]).min(v[3])
    }

    /// Horizontal minimum with mask (only consider active lanes).
    #[inline]
    pub fn horizontal_min_masked(v: [f32; 4], mask: SimdMask) -> f32 {
        let mut min_val = f32::MAX;
        for i in 0..4 {
            if mask.is_active(i) {
                min_val = min_val.min(v[i]);
            }
        }
        min_val
    }

    /// Find index of minimum value (masked).
    #[inline]
    pub fn argmin_masked(v: [f32; 4], mask: SimdMask) -> Option<usize> {
        let mut min_val = f32::MAX;
        let mut min_idx = None;
        for i in 0..4 {
            if mask.is_active(i) && v[i] < min_val {
                min_val = v[i];
                min_idx = Some(i);
            }
        }
        min_idx
    }

    /// Compare less-than and return mask.
    #[inline]
    pub fn cmp_lt(a: [f32; 4], b: [f32; 4]) -> SimdMask {
        SimdMask::from_cmp_lt(a, b)
    }

    /// Compare less-than with scalar and return mask.
    #[inline]
    pub fn cmp_lt_scalar(a: [f32; 4], threshold: f32) -> SimdMask {
        SimdMask::from_cmp_lt_scalar(a, threshold)
    }

    /// Predicated conditional update for candidate distances.
    /// Updates `distances` and `indices` where `new_distances < distances`.
    #[inline]
    pub fn update_best_candidates(
        distances: &mut [f32; 4],
        indices: &mut [u32; 4],
        new_distances: [f32; 4],
        new_indices: [u32; 4],
    ) -> SimdMask {
        let mask = Self::cmp_lt(new_distances, *distances);
        
        if mask.any() {
            *distances = Self::blend(*distances, new_distances, mask);
            
            // Update indices using mask
            for i in 0..4 {
                if mask.is_active(i) {
                    indices[i] = new_indices[i];
                }
            }
        }
        
        mask
    }

    /// Batch distance comparison with threshold.
    /// Returns mask of lanes where distance < threshold.
    #[inline]
    pub fn filter_by_distance(
        distances: [f32; 4],
        threshold: f32,
    ) -> SimdMask {
        Self::cmp_lt_scalar(distances, threshold)
    }

    /// Compact active lanes to front of array.
    /// Returns (compacted values, count).
    pub fn compact(values: [f32; 4], indices: [u32; 4], mask: SimdMask) -> (Vec<f32>, Vec<u32>) {
        let mut compact_vals = Vec::with_capacity(4);
        let mut compact_idxs = Vec::with_capacity(4);
        
        for i in 0..4 {
            if mask.is_active(i) {
                compact_vals.push(values[i]);
                compact_idxs.push(indices[i]);
            }
        }
        
        (compact_vals, compact_idxs)
    }
}

/// 8-wide predicated operations for AVX2/NEON.
pub struct PredicatedSimd8;

impl PredicatedSimd8 {
    /// Blend 8 values: mask ? b : a
    #[inline]
    pub fn blend(a: [f32; 8], b: [f32; 8], mask: u8) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::blend_avx2(a, b, mask) };
            }
        }

        // Scalar fallback
        let mut out = a;
        for i in 0..8 {
            if (mask >> i) & 1 == 1 {
                out[i] = b[i];
            }
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blend_avx2(a: [f32; 8], b: [f32; 8], mask: u8) -> [f32; 8] {
        use std::arch::x86_64::*;

        // Expand mask to full 256-bit
        let mask_expanded: [i32; 8] = [
            if (mask & 0x01) != 0 { -1i32 } else { 0 },
            if (mask & 0x02) != 0 { -1i32 } else { 0 },
            if (mask & 0x04) != 0 { -1i32 } else { 0 },
            if (mask & 0x08) != 0 { -1i32 } else { 0 },
            if (mask & 0x10) != 0 { -1i32 } else { 0 },
            if (mask & 0x20) != 0 { -1i32 } else { 0 },
            if (mask & 0x40) != 0 { -1i32 } else { 0 },
            if (mask & 0x80) != 0 { -1i32 } else { 0 },
        ];

        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let vm = _mm256_loadu_si256(mask_expanded.as_ptr() as *const __m256i);

        let result = _mm256_blendv_ps(va, vb, _mm256_castsi256_ps(vm));

        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), result);
        out
    }

    /// 8-wide minimum.
    #[inline]
    pub fn min(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::min_avx2(a, b) };
            }
        }

        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = a[i].min(b[i]);
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min_avx2(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        use std::arch::x86_64::*;
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let result = _mm256_min_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), result);
        out
    }

    /// 8-wide less-than comparison, returns mask.
    #[inline]
    pub fn cmp_lt(a: [f32; 8], b: [f32; 8]) -> u8 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::cmp_lt_avx2(a, b) };
            }
        }

        let mut mask = 0u8;
        for i in 0..8 {
            if a[i] < b[i] {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_lt_avx2(a: [f32; 8], b: [f32; 8]) -> u8 {
        use std::arch::x86_64::*;
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let cmp = _mm256_cmp_ps(va, vb, _CMP_LT_OQ);
        _mm256_movemask_ps(cmp) as u8
    }

    /// Horizontal minimum of 8 values.
    #[inline]
    pub fn horizontal_min(v: [f32; 8]) -> f32 {
        let a = v[0].min(v[1]).min(v[2]).min(v[3]);
        let b = v[4].min(v[5]).min(v[6]).min(v[7]);
        a.min(b)
    }

    /// Count set bits (popcount).
    #[inline]
    pub fn popcount(mask: u8) -> u32 {
        mask.count_ones()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_basic() {
        let mask = SimdMask::from_bits(0b1010);
        
        assert!(!mask.is_active(0));
        assert!(mask.is_active(1));
        assert!(!mask.is_active(2));
        assert!(mask.is_active(3));
        assert_eq!(mask.count(), 2);
    }

    #[test]
    fn test_mask_from_cmp() {
        let a = [1.0, 5.0, 3.0, 7.0];
        let b = [2.0, 4.0, 4.0, 6.0];
        
        let mask = SimdMask::from_cmp_lt(a, b);
        
        assert!(mask.is_active(0)); // 1 < 2
        assert!(!mask.is_active(1)); // 5 > 4
        assert!(mask.is_active(2)); // 3 < 4
        assert!(!mask.is_active(3)); // 7 > 6
    }

    #[test]
    fn test_blend_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let mask = SimdMask::from_bits(0b0101);
        
        let result = PredicatedSimd::blend(a, b, mask);
        
        assert_eq!(result[0], 10.0); // mask[0] = 1, use b
        assert_eq!(result[1], 2.0);  // mask[1] = 0, use a
        assert_eq!(result[2], 30.0); // mask[2] = 1, use b
        assert_eq!(result[3], 4.0);  // mask[3] = 0, use a
    }

    #[test]
    fn test_min_max() {
        let a = [1.0, 5.0, 3.0, 7.0];
        let b = [2.0, 4.0, 4.0, 6.0];
        
        let min_result = PredicatedSimd::min(a, b);
        let max_result = PredicatedSimd::max(a, b);
        
        assert_eq!(min_result, [1.0, 4.0, 3.0, 6.0]);
        assert_eq!(max_result, [2.0, 5.0, 4.0, 7.0]);
    }

    #[test]
    fn test_predicated_min() {
        let a = [5.0, 5.0, 5.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mask = SimdMask::from_bits(0b0110); // Only lanes 1 and 2
        
        let result = PredicatedSimd::min_pred(a, b, mask);
        
        assert_eq!(result[0], 5.0); // mask[0] = 0, keep a
        assert_eq!(result[1], 2.0); // mask[1] = 1, min(5, 2) = 2
        assert_eq!(result[2], 3.0); // mask[2] = 1, min(5, 3) = 3
        assert_eq!(result[3], 5.0); // mask[3] = 0, keep a
    }

    #[test]
    fn test_horizontal_min_masked() {
        let v = [10.0, 1.0, 100.0, 5.0];
        let mask = SimdMask::from_bits(0b1001); // Only lanes 0 and 3
        
        let min = PredicatedSimd::horizontal_min_masked(v, mask);
        
        assert_eq!(min, 5.0); // min(10.0, 5.0)
    }

    #[test]
    fn test_argmin_masked() {
        let v = [10.0, 1.0, 100.0, 5.0];
        let mask = SimdMask::from_bits(0b1101); // Lanes 0, 2, 3 (not 1)
        
        let idx = PredicatedSimd::argmin_masked(v, mask);
        
        assert_eq!(idx, Some(3)); // Lane 3 has 5.0, smallest among active
    }

    #[test]
    fn test_update_best_candidates() {
        let mut distances = [10.0, 20.0, 30.0, 40.0];
        let mut indices = [0u32, 1, 2, 3];
        let new_distances = [5.0, 25.0, 15.0, 35.0];
        let new_indices = [100u32, 101, 102, 103];
        
        let mask = PredicatedSimd::update_best_candidates(
            &mut distances,
            &mut indices,
            new_distances,
            new_indices,
        );
        
        // 5 < 10, 25 > 20, 15 < 30, 35 < 40
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert!(mask.is_active(2));
        assert!(mask.is_active(3));
        
        assert_eq!(distances, [5.0, 20.0, 15.0, 35.0]);
        assert_eq!(indices, [100, 1, 102, 103]);
    }

    #[test]
    fn test_compact() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let indices = [10u32, 20, 30, 40];
        let mask = SimdMask::from_bits(0b1010); // Lanes 1 and 3
        
        let (compact_vals, compact_idxs) = PredicatedSimd::compact(values, indices, mask);
        
        assert_eq!(compact_vals, vec![2.0, 4.0]);
        assert_eq!(compact_idxs, vec![20, 40]);
    }

    #[test]
    fn test_simd8_blend() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mask = 0b10101010u8; // Lanes 1, 3, 5, 7
        
        let result = PredicatedSimd8::blend(a, b, mask);
        
        assert_eq!(result, [1.0, 20.0, 3.0, 40.0, 5.0, 60.0, 7.0, 80.0]);
    }

    #[test]
    fn test_simd8_cmp_lt() {
        let a = [1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 9.0];
        let b = [2.0, 4.0, 4.0, 6.0, 3.0, 7.0, 5.0, 8.0];
        
        let mask = PredicatedSimd8::cmp_lt(a, b);
        
        // a[i] < b[i]: 1<2, 5>4, 3<4, 7>6, 2<3, 8>7, 4<5, 9>8
        assert_eq!(mask, 0b01010101);
    }

    #[test]
    fn test_filter_by_distance() {
        let distances = [0.5, 1.5, 0.3, 2.0];
        let threshold = 1.0;
        
        let mask = PredicatedSimd::filter_by_distance(distances, threshold);
        
        assert!(mask.is_active(0)); // 0.5 < 1.0
        assert!(!mask.is_active(1)); // 1.5 > 1.0
        assert!(mask.is_active(2)); // 0.3 < 1.0
        assert!(!mask.is_active(3)); // 2.0 > 1.0
    }
}
