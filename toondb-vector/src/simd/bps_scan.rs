//! BPS (Block Projection Sketch) L1 Distance Kernel
//!
//! This implements the vertical SIMD approach for computing L1 distances
//! between a query sketch and many vector sketches stored in SoA layout.
//!
//! # Algorithm
//!
//! For each query sketch Q[0..n_blocks]:
//!     For each vector V[i] in SoA layout:
//!         distance[i] = Σ |Q[slot] - V[slot * n_vec + i]|
//!
//! # Memory Layout
//!
//! The BPS data uses Structure-of-Arrays (SoA) layout:
//! - `bps[slot * n_vec + vec_id]` gives the sketch value for vector `vec_id` at `slot`
//!
//! # SIMD Strategy
//!
//! - **AVX2**: Process 32 vectors per iteration using 256-bit registers
//! - **NEON**: Process 16 vectors per iteration using 128-bit registers
//! - **Scalar**: Fallback for unsupported platforms
//!
//! # Math
//!
//! The L1 distance uses the identity:
//! ```text
//! |a - b| = max(a - b, 0) + max(b - a, 0) = (a ⊖ b) ∨ (b ⊖ a)
//! ```
//! where `⊖` is saturating subtraction and `∨` is bitwise OR.

use super::dispatch::cpu_features;

/// Compute BPS L1 distances between query and database vectors.
///
/// # Arguments
/// * `bps` - BPS data in SoA layout: `bps[slot * n_vec + vec_id]`
/// * `n_vec` - Number of vectors in the database
/// * `n_blocks` - Number of blocks in each sketch
/// * `query` - Query sketch values
/// * `out` - Output distances (u16)
///
/// # Panics
/// Panics if `query.len() < n_blocks` or `out.len() < n_vec`
#[inline]
pub fn bps_scan(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u16]) {
    assert!(query.len() >= n_blocks, "query too short");
    assert!(out.len() >= n_vec, "output buffer too small");
    
    let features = cpu_features();
    
    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx2 {
            // Safety: AVX2 feature is verified
            unsafe { bps_scan_avx2(bps, n_vec, n_blocks, query, out) };
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if features.has_neon {
            // Safety: NEON is mandatory on aarch64
            unsafe { bps_scan_neon(bps, n_vec, n_blocks, query, out) };
            return;
        }
    }
    
    // Scalar fallback
    bps_scan_scalar(bps, n_vec, n_blocks, query, out);
}

/// Compute BPS L1 distances with u32 output.
///
/// Same as `bps_scan` but outputs u32 distances for larger accumulations.
#[inline]
pub fn bps_scan_u32(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u32]) {
    assert!(query.len() >= n_blocks, "query too short");
    assert!(out.len() >= n_vec, "output buffer too small");
    
    let features = cpu_features();
    
    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx2 {
            unsafe { bps_scan_avx2_u32(bps, n_vec, n_blocks, query, out) };
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if features.has_neon {
            unsafe { bps_scan_neon_u32(bps, n_vec, n_blocks, query, out) };
            return;
        }
    }
    
    bps_scan_scalar_u32(bps, n_vec, n_blocks, query, out);
}

// ============================================================================
// x86_64 AVX2 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bps_scan_avx2(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u16]) {
    use std::arch::x86_64::*;
    
    // Process 32 vectors at a time (256 bits / 8 bits = 32)
    let vec_aligned = (n_vec / 32) * 32;
    
    // Zero output
    out.iter_mut().take(n_vec).for_each(|d| *d = 0);
    
    // Main loop: process 32 vectors at a time
    for chunk_start in (0..vec_aligned).step_by(32) {
        // Accumulators for 32 vectors (split into 2x16 u16)
        let mut acc_lo = _mm256_setzero_si256(); // Vectors 0-15
        let mut acc_hi = _mm256_setzero_si256(); // Vectors 16-31
        
        for slot in 0..n_blocks {
            let base = slot * n_vec + chunk_start;
            
            // Load 32 vector values
            let v = _mm256_loadu_si256(bps.as_ptr().add(base) as *const __m256i);
            
            // Broadcast query value
            let qv = _mm256_set1_epi8(query[slot] as i8);
            
            // Compute absolute difference: |a - b| = (a ⊖ b) ∨ (b ⊖ a)
            let d1 = _mm256_subs_epu8(v, qv);
            let d2 = _mm256_subs_epu8(qv, v);
            let diff = _mm256_or_si256(d1, d2);
            
            // Widen u8 → u16 and accumulate
            // Extract low and high 128-bit lanes
            let diff_lo128 = _mm256_castsi256_si128(diff);
            let diff_hi128 = _mm256_extracti128_si256(diff, 1);
            
            // Zero-extend u8 to u16
            let lo16 = _mm256_cvtepu8_epi16(diff_lo128);
            let hi16 = _mm256_cvtepu8_epi16(diff_hi128);
            
            // Accumulate
            acc_lo = _mm256_add_epi16(acc_lo, lo16);
            acc_hi = _mm256_add_epi16(acc_hi, hi16);
        }
        
        // Store results
        _mm256_storeu_si256(out.as_mut_ptr().add(chunk_start) as *mut __m256i, acc_lo);
        _mm256_storeu_si256(out.as_mut_ptr().add(chunk_start + 16) as *mut __m256i, acc_hi);
    }
    
    // Handle remaining vectors with scalar code
    for i in vec_aligned..n_vec {
        let mut sum: u16 = 0;
        for slot in 0..n_blocks {
            let v = bps[slot * n_vec + i];
            let qv = query[slot];
            let diff = if v > qv { v - qv } else { qv - v };
            sum = sum.saturating_add(diff as u16);
        }
        out[i] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bps_scan_avx2_u32(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u32]) {
    use std::arch::x86_64::*;
    
    // Process 32 vectors at a time
    let vec_aligned = (n_vec / 32) * 32;
    
    // Zero output
    out.iter_mut().take(n_vec).for_each(|d| *d = 0);
    
    // Main loop: process 32 vectors at a time
    for chunk_start in (0..vec_aligned).step_by(32) {
        // Accumulators - need 8 x 4 = 32 u32 values
        // We'll use intermediate u16 accumulators and widen at the end
        let mut acc_lo = _mm256_setzero_si256(); // Vectors 0-15 as u16
        let mut acc_hi = _mm256_setzero_si256(); // Vectors 16-31 as u16
        
        for slot in 0..n_blocks {
            let base = slot * n_vec + chunk_start;
            let v = _mm256_loadu_si256(bps.as_ptr().add(base) as *const __m256i);
            let qv = _mm256_set1_epi8(query[slot] as i8);
            
            let d1 = _mm256_subs_epu8(v, qv);
            let d2 = _mm256_subs_epu8(qv, v);
            let diff = _mm256_or_si256(d1, d2);
            
            let diff_lo128 = _mm256_castsi256_si128(diff);
            let diff_hi128 = _mm256_extracti128_si256(diff, 1);
            
            let lo16 = _mm256_cvtepu8_epi16(diff_lo128);
            let hi16 = _mm256_cvtepu8_epi16(diff_hi128);
            
            acc_lo = _mm256_add_epi16(acc_lo, lo16);
            acc_hi = _mm256_add_epi16(acc_hi, hi16);
        }
        
        // Widen u16 to u32 and store
        // acc_lo contains 16 u16 values for vectors 0-15
        // acc_hi contains 16 u16 values for vectors 16-31
        
        // Extract and widen acc_lo
        let acc_lo_128_0 = _mm256_castsi256_si128(acc_lo);
        let acc_lo_128_1 = _mm256_extracti128_si256(acc_lo, 1);
        let out_0 = _mm256_cvtepu16_epi32(acc_lo_128_0); // 8 u32
        let out_1 = _mm256_cvtepu16_epi32(acc_lo_128_1); // 8 u32
        
        _mm256_storeu_si256(out.as_mut_ptr().add(chunk_start) as *mut __m256i, out_0);
        _mm256_storeu_si256(out.as_mut_ptr().add(chunk_start + 8) as *mut __m256i, out_1);
        
        // Extract and widen acc_hi
        let acc_hi_128_0 = _mm256_castsi256_si128(acc_hi);
        let acc_hi_128_1 = _mm256_extracti128_si256(acc_hi, 1);
        let out_2 = _mm256_cvtepu16_epi32(acc_hi_128_0);
        let out_3 = _mm256_cvtepu16_epi32(acc_hi_128_1);
        
        _mm256_storeu_si256(out.as_mut_ptr().add(chunk_start + 16) as *mut __m256i, out_2);
        _mm256_storeu_si256(out.as_mut_ptr().add(chunk_start + 24) as *mut __m256i, out_3);
    }
    
    // Handle remaining vectors
    for i in vec_aligned..n_vec {
        let mut sum: u32 = 0;
        for slot in 0..n_blocks {
            let v = bps[slot * n_vec + i];
            let qv = query[slot];
            let diff = if v > qv { v - qv } else { qv - v };
            sum += diff as u32;
        }
        out[i] = sum;
    }
}

// ============================================================================
// aarch64 NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bps_scan_neon(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u16]) {
    use std::arch::aarch64::*;
    
    unsafe {
        // Process 16 vectors at a time (128 bits / 8 bits = 16)
        let vec_aligned = (n_vec / 16) * 16;
        
        // Zero output
        out.iter_mut().take(n_vec).for_each(|d| *d = 0);
        
        for chunk_start in (0..vec_aligned).step_by(16) {
            // Accumulators for 16 vectors as u16 (split into 2x8)
            let mut acc_lo = vdupq_n_u16(0);
            let mut acc_hi = vdupq_n_u16(0);
            
            for slot in 0..n_blocks {
                let base = slot * n_vec + chunk_start;
                
                // Broadcast query byte
                let q = vdupq_n_u8(query[slot]);
                
                // Load 16 database bytes
                let db = vld1q_u8(bps.as_ptr().add(base));
                
                // Compute |q - db| using vabdq_u8 (single instruction on NEON!)
                let diff = vabdq_u8(q, db);
                
                // Widen to u16 and accumulate
                acc_lo = vaddw_u8(acc_lo, vget_low_u8(diff));
                acc_hi = vaddw_u8(acc_hi, vget_high_u8(diff));
            }
            
            // Store 16 distances
            vst1q_u16(out.as_mut_ptr().add(chunk_start), acc_lo);
            vst1q_u16(out.as_mut_ptr().add(chunk_start + 8), acc_hi);
        }
        
        // Handle remainder
        for i in vec_aligned..n_vec {
            let mut sum: u16 = 0;
            for slot in 0..n_blocks {
                let v = bps[slot * n_vec + i];
                let qv = query[slot];
                let diff = if v > qv { v - qv } else { qv - v };
                sum = sum.saturating_add(diff as u16);
            }
            out[i] = sum;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bps_scan_neon_u32(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u32]) {
    use std::arch::aarch64::*;
    
    unsafe {
        let vec_aligned = (n_vec / 16) * 16;
        
        out.iter_mut().take(n_vec).for_each(|d| *d = 0);
        
        for chunk_start in (0..vec_aligned).step_by(16) {
            let mut acc_lo = vdupq_n_u16(0);
            let mut acc_hi = vdupq_n_u16(0);
            
            for slot in 0..n_blocks {
                let base = slot * n_vec + chunk_start;
                let q = vdupq_n_u8(query[slot]);
                let db = vld1q_u8(bps.as_ptr().add(base));
                let diff = vabdq_u8(q, db);
                
                acc_lo = vaddw_u8(acc_lo, vget_low_u8(diff));
                acc_hi = vaddw_u8(acc_hi, vget_high_u8(diff));
            }
            
            // Widen u16 to u32 and store
            let d0 = vmovl_u16(vget_low_u16(acc_lo));
            let d1 = vmovl_u16(vget_high_u16(acc_lo));
            let d2 = vmovl_u16(vget_low_u16(acc_hi));
            let d3 = vmovl_u16(vget_high_u16(acc_hi));
            
            vst1q_u32(out.as_mut_ptr().add(chunk_start), d0);
            vst1q_u32(out.as_mut_ptr().add(chunk_start + 4), d1);
            vst1q_u32(out.as_mut_ptr().add(chunk_start + 8), d2);
            vst1q_u32(out.as_mut_ptr().add(chunk_start + 12), d3);
        }
        
        for i in vec_aligned..n_vec {
            let mut sum: u32 = 0;
            for slot in 0..n_blocks {
                let v = bps[slot * n_vec + i];
                let qv = query[slot];
                let diff = if v > qv { v - qv } else { qv - v };
                sum += diff as u32;
            }
            out[i] = sum;
        }
    }
}

// ============================================================================
// Scalar Fallback
// ============================================================================

/// Scalar fallback for BPS scan (u16 output)
#[inline]
fn bps_scan_scalar(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u16]) {
    // Zero output
    out.iter_mut().take(n_vec).for_each(|d| *d = 0);
    
    for slot in 0..n_blocks {
        let q = query[slot];
        let base = slot * n_vec;
        
        for vec_id in 0..n_vec {
            let v = bps[base + vec_id];
            let diff = if v > q { v - q } else { q - v };
            out[vec_id] = out[vec_id].saturating_add(diff as u16);
        }
    }
}

/// Scalar fallback for BPS scan (u32 output)
#[inline]
fn bps_scan_scalar_u32(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u32]) {
    out.iter_mut().take(n_vec).for_each(|d| *d = 0);
    
    for slot in 0..n_blocks {
        let q = query[slot];
        let base = slot * n_vec;
        
        for vec_id in 0..n_vec {
            let v = bps[base + vec_id];
            let diff = if v > q { v - q } else { q - v };
            out[vec_id] += diff as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bps_scan_basic() {
        let n_vec = 100;
        let n_blocks = 8;
        let bps: Vec<u8> = (0..n_vec * n_blocks).map(|i| (i % 256) as u8).collect();
        let query: Vec<u8> = (0..n_blocks).map(|i| (i * 10) as u8).collect();
        let mut out = vec![0u16; n_vec];
        
        bps_scan(&bps, n_vec, n_blocks, &query, &mut out);
        
        // Verify against scalar
        let mut expected = vec![0u16; n_vec];
        bps_scan_scalar(&bps, n_vec, n_blocks, &query, &mut expected);
        
        assert_eq!(out, expected);
    }
    
    #[test]
    fn test_bps_scan_u32_basic() {
        let n_vec = 100;
        let n_blocks = 8;
        let bps: Vec<u8> = (0..n_vec * n_blocks).map(|i| (i % 256) as u8).collect();
        let query: Vec<u8> = (0..n_blocks).map(|i| (i * 10) as u8).collect();
        let mut out = vec![0u32; n_vec];
        
        bps_scan_u32(&bps, n_vec, n_blocks, &query, &mut out);
        
        let mut expected = vec![0u32; n_vec];
        bps_scan_scalar_u32(&bps, n_vec, n_blocks, &query, &mut expected);
        
        assert_eq!(out, expected);
    }
    
    #[test]
    fn test_bps_scan_alignment() {
        // Test with sizes that don't align to SIMD width
        for n_vec in [1, 15, 17, 31, 33, 63, 65, 127] {
            let n_blocks = 4;
            let bps: Vec<u8> = (0..n_vec * n_blocks).map(|i| (i % 256) as u8).collect();
            let query: Vec<u8> = vec![128; n_blocks];
            let mut out = vec![0u16; n_vec];
            
            bps_scan(&bps, n_vec, n_blocks, &query, &mut out);
            
            let mut expected = vec![0u16; n_vec];
            bps_scan_scalar(&bps, n_vec, n_blocks, &query, &mut expected);
            
            assert_eq!(out, expected, "Mismatch for n_vec={}", n_vec);
        }
    }
}
