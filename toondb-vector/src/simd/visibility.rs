//! MVCC Visibility Check Kernel
//!
//! This module provides SIMD-accelerated visibility checking for MVCC
//! (Multi-Version Concurrency Control) operations.
//!
//! # Algorithm
//!
//! A row is visible if:
//! ```text
//! visible[i] = (commit_ts[i] != 0) && (commit_ts[i] < snapshot_ts)
//! ```
//!
//! With transaction ID awareness:
//! ```text
//! visible[i] = ((commit_ts[i] != 0) && (commit_ts[i] < snapshot_ts)) || (txn_id[i] == current_txn)
//! ```
//!
//! # Boolean Logic
//!
//! ```text
//! visible = (commit ≠ 0) ∧ (commit < snapshot)
//!         = ¬(commit = 0) ∧ (commit < snapshot)
//! ```
//!
//! # SIMD Strategy
//!
//! - **AVX2**: Process 4 u64 timestamps per 256-bit register
//! - **NEON**: Process 2 u64 timestamps per 128-bit register

use super::dispatch::cpu_features;

/// Check visibility for a batch of rows based on commit timestamps.
///
/// # Arguments
/// * `commit_timestamps` - Array of commit timestamps (0 = uncommitted)
/// * `snapshot_ts` - The snapshot timestamp for visibility check
/// * `visible_mask` - Output: 1 if visible, 0 if not visible
///
/// # Panics
/// Panics if `visible_mask.len() < commit_timestamps.len()`
#[inline]
pub fn visibility_check(
    commit_timestamps: &[u64],
    snapshot_ts: u64,
    visible_mask: &mut [u8],
) {
    let n_rows = commit_timestamps.len();
    assert!(visible_mask.len() >= n_rows, "visible_mask buffer too small");
    
    if n_rows == 0 {
        return;
    }
    
    let features = cpu_features();
    
    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx2 {
            unsafe { visibility_check_avx2(commit_timestamps, snapshot_ts, visible_mask) };
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if features.has_neon {
            unsafe { visibility_check_neon(commit_timestamps, snapshot_ts, visible_mask) };
            return;
        }
    }
    
    visibility_check_scalar(commit_timestamps, snapshot_ts, visible_mask);
}

/// Check visibility with transaction ID awareness (for self-visibility).
///
/// A row is visible if:
/// - `(commit_ts != 0 && commit_ts < snapshot_ts)`, OR
/// - `txn_id == current_txn_id` (self-visibility)
///
/// # Arguments
/// * `commit_timestamps` - Array of commit timestamps (0 = uncommitted)
/// * `txn_ids` - Array of transaction IDs that wrote each row
/// * `snapshot_ts` - The snapshot timestamp for visibility check
/// * `current_txn_id` - The current transaction's ID
/// * `visible_mask` - Output: 1 if visible, 0 if not visible
#[inline]
pub fn visibility_check_with_txn(
    commit_timestamps: &[u64],
    txn_ids: &[u64],
    snapshot_ts: u64,
    current_txn_id: u64,
    visible_mask: &mut [u8],
) {
    let n_rows = commit_timestamps.len();
    assert_eq!(txn_ids.len(), n_rows, "txn_ids length mismatch");
    assert!(visible_mask.len() >= n_rows, "visible_mask buffer too small");
    
    if n_rows == 0 {
        return;
    }
    
    let features = cpu_features();
    
    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx2 {
            unsafe {
                visibility_check_with_txn_avx2(
                    commit_timestamps, txn_ids, snapshot_ts, current_txn_id, visible_mask
                )
            };
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if features.has_neon {
            unsafe {
                visibility_check_with_txn_neon(
                    commit_timestamps, txn_ids, snapshot_ts, current_txn_id, visible_mask
                )
            };
            return;
        }
    }
    
    visibility_check_with_txn_scalar(commit_timestamps, txn_ids, snapshot_ts, current_txn_id, visible_mask);
}

// ============================================================================
// x86_64 AVX2 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn visibility_check_avx2(
    commit_timestamps: &[u64],
    snapshot_ts: u64,
    visible_mask: &mut [u8],
) {
    use std::arch::x86_64::*;
    
    unsafe {
        let n_rows = commit_timestamps.len();
        let snapshot_vec = _mm256_set1_epi64x(snapshot_ts as i64);
        let zero_vec = _mm256_setzero_si256();
        
        let mut i = 0;
        
        // Process 4 rows at a time (256 bits / 64 bits = 4)
        while i + 4 <= n_rows {
            // Load 4 commit timestamps
            let commits = _mm256_loadu_si256(commit_timestamps.as_ptr().add(i) as *const __m256i);
            
            // Check: commit_ts != 0
            let eq_zero = _mm256_cmpeq_epi64(commits, zero_vec);
            // Invert: not_zero = ~eq_zero
            let not_zero = _mm256_xor_si256(eq_zero, _mm256_set1_epi64x(-1));
            
            // Check: commit_ts < snapshot_ts (using snapshot > commit)
            let less_than = _mm256_cmpgt_epi64(snapshot_vec, commits);
            
            // Combine: not_zero AND less_than
            let visible = _mm256_and_si256(not_zero, less_than);
            
            // Extract to mask bytes: take bit 63 of each 64-bit lane
            let mask_bits = _mm256_movemask_pd(_mm256_castsi256_pd(visible));
            
            visible_mask[i] = if mask_bits & 1 != 0 { 1 } else { 0 };
            visible_mask[i + 1] = if mask_bits & 2 != 0 { 1 } else { 0 };
            visible_mask[i + 2] = if mask_bits & 4 != 0 { 1 } else { 0 };
            visible_mask[i + 3] = if mask_bits & 8 != 0 { 1 } else { 0 };
            
            i += 4;
        }
        
        // Scalar tail
        while i < n_rows {
            let commit = commit_timestamps[i];
            visible_mask[i] = if commit != 0 && commit < snapshot_ts { 1 } else { 0 };
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn visibility_check_with_txn_avx2(
    commit_timestamps: &[u64],
    txn_ids: &[u64],
    snapshot_ts: u64,
    current_txn_id: u64,
    visible_mask: &mut [u8],
) {
    use std::arch::x86_64::*;
    
    unsafe {
        let n_rows = commit_timestamps.len();
        let snapshot_vec = _mm256_set1_epi64x(snapshot_ts as i64);
        let zero_vec = _mm256_setzero_si256();
        let current_txn_vec = _mm256_set1_epi64x(current_txn_id as i64);
        
        let mut i = 0;
        
        while i + 4 <= n_rows {
            // Load 4 commit timestamps and txn IDs
            let commits = _mm256_loadu_si256(commit_timestamps.as_ptr().add(i) as *const __m256i);
            let txns = _mm256_loadu_si256(txn_ids.as_ptr().add(i) as *const __m256i);
            
            // Check: txn_id == current_txn_id (own writes always visible)
            let own_write = _mm256_cmpeq_epi64(txns, current_txn_vec);
            
            // Check: commit_ts != 0
            let eq_zero = _mm256_cmpeq_epi64(commits, zero_vec);
            let not_zero = _mm256_xor_si256(eq_zero, _mm256_set1_epi64x(-1));
            
            // Check: commit_ts < snapshot_ts
            let less_than = _mm256_cmpgt_epi64(snapshot_vec, commits);
            
            // Combine: (not_zero AND less_than) OR own_write
            let committed_visible = _mm256_and_si256(not_zero, less_than);
            let visible = _mm256_or_si256(committed_visible, own_write);
            
            // Extract to mask bytes
            let mask_bits = _mm256_movemask_pd(_mm256_castsi256_pd(visible));
            
            visible_mask[i] = if mask_bits & 1 != 0 { 1 } else { 0 };
            visible_mask[i + 1] = if mask_bits & 2 != 0 { 1 } else { 0 };
            visible_mask[i + 2] = if mask_bits & 4 != 0 { 1 } else { 0 };
            visible_mask[i + 3] = if mask_bits & 8 != 0 { 1 } else { 0 };
            
            i += 4;
        }
        
        // Scalar tail
        while i < n_rows {
            let commit = commit_timestamps[i];
            let txn = txn_ids[i];
            let visible = (commit != 0 && commit < snapshot_ts) || txn == current_txn_id;
            visible_mask[i] = if visible { 1 } else { 0 };
            i += 1;
        }
    }
}

// ============================================================================
// aarch64 NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn visibility_check_neon(
    commit_timestamps: &[u64],
    snapshot_ts: u64,
    visible_mask: &mut [u8],
) {
    use std::arch::aarch64::*;
    
    unsafe {
        let n_rows = commit_timestamps.len();
        let snapshot_vec = vdupq_n_u64(snapshot_ts);
        let zero_vec = vdupq_n_u64(0);
        
        let mut i = 0;
        
        // Process 2 rows at a time (128 bits / 64 bits = 2)
        while i + 2 <= n_rows {
            // Load 2 commit timestamps
            let commits = vld1q_u64(commit_timestamps.as_ptr().add(i));
            
            // Check: commit_ts != 0
            let eq_zero = vceqq_u64(commits, zero_vec);
            // not_zero via bitwise NOT on the bytes
            let not_zero = vmvnq_u8(vreinterpretq_u8_u64(eq_zero));
            
            // Check: commit_ts < snapshot_ts
            // NEON doesn't have vcltq_u64, use subtraction trick
            // If commit < snapshot, then (commit - snapshot) will have high bit set (underflow)
            let diff = vsubq_u64(commits, snapshot_vec);
            let less_than = vshrq_n_u64(diff, 63); // Get sign bit (1 if underflowed)
            
            // Combine: not_zero AND (less_than == 1)
            let visible = vandq_u64(
                vreinterpretq_u64_u8(not_zero),
                vceqq_u64(less_than, vdupq_n_u64(1)),
            );
            
            // Extract to mask bytes
            visible_mask[i] = if vgetq_lane_u64(visible, 0) != 0 { 1 } else { 0 };
            visible_mask[i + 1] = if vgetq_lane_u64(visible, 1) != 0 { 1 } else { 0 };
            
            i += 2;
        }
        
        // Scalar tail
        while i < n_rows {
            let commit = commit_timestamps[i];
            visible_mask[i] = if commit != 0 && commit < snapshot_ts { 1 } else { 0 };
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn visibility_check_with_txn_neon(
    commit_timestamps: &[u64],
    txn_ids: &[u64],
    snapshot_ts: u64,
    current_txn_id: u64,
    visible_mask: &mut [u8],
) {
    use std::arch::aarch64::*;
    
    unsafe {
        let n_rows = commit_timestamps.len();
        let snapshot_vec = vdupq_n_u64(snapshot_ts);
        let zero_vec = vdupq_n_u64(0);
        let current_txn_vec = vdupq_n_u64(current_txn_id);
        
        let mut i = 0;
        
        while i + 2 <= n_rows {
            let commits = vld1q_u64(commit_timestamps.as_ptr().add(i));
            let txns = vld1q_u64(txn_ids.as_ptr().add(i));
            
            // Check: txn_id == current_txn_id
            let own_write = vceqq_u64(txns, current_txn_vec);
            
            // Check: commit_ts != 0
            let eq_zero = vceqq_u64(commits, zero_vec);
            let not_zero = vmvnq_u8(vreinterpretq_u8_u64(eq_zero));
            
            // Check: commit_ts < snapshot_ts
            let diff = vsubq_u64(commits, snapshot_vec);
            let less_than = vshrq_n_u64(diff, 63);
            
            // Combine
            let committed_visible = vandq_u64(
                vreinterpretq_u64_u8(not_zero),
                vceqq_u64(less_than, vdupq_n_u64(1)),
            );
            let visible = vorrq_u64(committed_visible, own_write);
            
            visible_mask[i] = if vgetq_lane_u64(visible, 0) != 0 { 1 } else { 0 };
            visible_mask[i + 1] = if vgetq_lane_u64(visible, 1) != 0 { 1 } else { 0 };
            
            i += 2;
        }
        
        // Scalar tail
        while i < n_rows {
            let commit = commit_timestamps[i];
            let txn = txn_ids[i];
            let visible = (commit != 0 && commit < snapshot_ts) || txn == current_txn_id;
            visible_mask[i] = if visible { 1 } else { 0 };
            i += 1;
        }
    }
}

// ============================================================================
// Scalar Fallback
// ============================================================================

/// Scalar fallback for visibility check
#[inline]
fn visibility_check_scalar(
    commit_timestamps: &[u64],
    snapshot_ts: u64,
    visible_mask: &mut [u8],
) {
    for (i, &commit) in commit_timestamps.iter().enumerate() {
        visible_mask[i] = if commit != 0 && commit < snapshot_ts { 1 } else { 0 };
    }
}

/// Scalar fallback for visibility check with txn
#[inline]
fn visibility_check_with_txn_scalar(
    commit_timestamps: &[u64],
    txn_ids: &[u64],
    snapshot_ts: u64,
    current_txn_id: u64,
    visible_mask: &mut [u8],
) {
    for i in 0..commit_timestamps.len() {
        let commit = commit_timestamps[i];
        let txn = txn_ids[i];
        let visible = (commit != 0 && commit < snapshot_ts) || txn == current_txn_id;
        visible_mask[i] = if visible { 1 } else { 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_visibility_basic() {
        let commits = vec![0, 100, 200, 300, 400];
        let snapshot = 250;
        let mut mask = vec![0u8; 5];
        
        visibility_check(&commits, snapshot, &mut mask);
        
        // Expected:
        // 0: commit=0 (uncommitted) -> not visible
        // 100: 100 < 250 -> visible
        // 200: 200 < 250 -> visible
        // 300: 300 >= 250 -> not visible
        // 400: 400 >= 250 -> not visible
        assert_eq!(mask, vec![0, 1, 1, 0, 0]);
    }
    
    #[test]
    fn test_visibility_with_txn() {
        let commits = vec![0, 100, 200, 300, 0];
        let txn_ids = vec![10, 20, 30, 40, 50];
        let snapshot = 250;
        let current_txn = 50;
        let mut mask = vec![0u8; 5];
        
        visibility_check_with_txn(&commits, &txn_ids, snapshot, current_txn, &mut mask);
        
        // Expected:
        // 0: commit=0, txn=10 != 50 -> not visible
        // 100: commit < snapshot -> visible
        // 200: commit < snapshot -> visible
        // 300: commit >= snapshot, txn=40 != 50 -> not visible
        // 0: commit=0, txn=50 == 50 -> visible (self-visibility)
        assert_eq!(mask, vec![0, 1, 1, 0, 1]);
    }
    
    #[test]
    fn test_visibility_alignment() {
        // Test with sizes that don't align to SIMD width
        for n_rows in [1, 2, 3, 4, 5, 7, 9, 15, 17] {
            let commits: Vec<u64> = (0..n_rows).map(|i| (i * 100) as u64).collect();
            let snapshot = 500;
            let mut mask = vec![0u8; n_rows];
            
            visibility_check(&commits, snapshot, &mut mask);
            
            // Verify against scalar
            let mut expected = vec![0u8; n_rows];
            visibility_check_scalar(&commits, snapshot, &mut expected);
            
            assert_eq!(mask, expected, "Mismatch for n_rows={}", n_rows);
        }
    }
    
    #[test]
    fn test_visibility_edge_cases() {
        // All zeros
        let commits = vec![0u64; 10];
        let mut mask = vec![1u8; 10];
        visibility_check(&commits, 100, &mut mask);
        assert!(mask.iter().all(|&m| m == 0));
        
        // All equal to snapshot
        let commits = vec![100u64; 10];
        let mut mask = vec![1u8; 10];
        visibility_check(&commits, 100, &mut mask);
        assert!(mask.iter().all(|&m| m == 0));
        
        // All less than snapshot
        let commits = vec![99u64; 10];
        let mut mask = vec![0u8; 10];
        visibility_check(&commits, 100, &mut mask);
        assert!(mask.iter().all(|&m| m == 1));
    }
}
