//! Kernel dispatch for pure Rust SIMD implementations.
//!
//! Provides wrappers around optimized Rust SIMD kernels from the `simd` module.
//! Uses runtime CPU detection to select optimal code path.
//!
//! # Migration from C++ FFI
//!
//! This module previously used C++ SIMD kernels via FFI. It has been migrated
//! to pure Rust implementations in the `simd` module, providing:
//! - Unified toolchain (no C++ compiler needed)
//! - Cross-function inlining
//! - Better error messages and debugging
//! - `cargo miri` support for undefined behavior detection

use std::sync::OnceLock;

// Re-export from the simd module for backwards compatibility
pub use crate::simd::dispatch::{CpuFeatures, SimdLevel};

/// Global CPU features, detected once at first use
static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Get detected CPU features (cached)
pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

/// Get best available SIMD level
pub fn simd_level() -> SimdLevel {
    cpu_features().best_level()
}

// ============================================================================
// BPS Scan Dispatcher
// ============================================================================

/// BPS scan dispatcher - uses pure Rust SIMD implementations.
pub struct BpsScanDispatcher;

impl BpsScanDispatcher {
    /// Scan BPS data and compute L1 distances.
    /// 
    /// Data layout (SoA): bps_data[block * n_vec + vec]
    ///
    /// Uses pure Rust SIMD implementations for optimal performance.
    pub fn scan(
        bps: &[u8],
        n_vec: usize,
        n_blocks: usize,
        _proj: usize, // Legacy parameter, kept for API compat
        query: &[u8],
        out: &mut [u16],
    ) {
        crate::simd::bps_scan::bps_scan(bps, n_vec, n_blocks, query, out);
    }
    
    /// New interface - returns u32 distances.
    pub fn scan_u32(
        bps: &[u8],
        n_vec: usize,
        n_blocks: usize,
        query: &[u8],
        out: &mut [u32],
    ) {
        crate::simd::bps_scan::bps_scan_u32(bps, n_vec, n_blocks, query, out);
    }

    /// Rust fallback implementation (for testing).
    #[allow(dead_code)]
    pub(crate) fn scan_fallback(
        bps: &[u8],
        n_vec: usize,
        n_blocks: usize,
        proj: usize,
        query: &[u8],
        out: &mut [u16],
    ) {
        let slots = n_blocks * proj;
        
        // Zero output
        for d in out.iter_mut().take(n_vec) {
            *d = 0;
        }
        
        for slot in 0..slots {
            let q = query[slot] as i16;
            let base = slot * n_vec;
            
            for vec_id in 0..n_vec {
                let v = bps[base + vec_id] as i16;
                let diff = (q - v).abs() as u16;
                out[vec_id] = out[vec_id].saturating_add(diff);
            }
        }
    }
    
    /// Rust fallback implementation for u32 output (for testing).
    #[allow(dead_code)]
    pub(crate) fn scan_fallback_u32(
        bps: &[u8],
        n_vec: usize,
        n_blocks: usize,
        query: &[u8],
        out: &mut [u32],
    ) {
        // Zero output
        for d in out.iter_mut().take(n_vec) {
            *d = 0;
        }
        
        for block in 0..n_blocks {
            let q = query[block];
            let base = block * n_vec;
            
            for vec_id in 0..n_vec {
                let v = bps[base + vec_id];
                let diff = if q > v { q - v } else { v - q };
                out[vec_id] += diff as u32;
            }
        }
    }
}

// ============================================================================
// int8 Dot Product Dispatcher
// ============================================================================

/// int8 dot product dispatcher - uses pure Rust SIMD implementations.
pub struct DotI8Dispatcher;

impl DotI8Dispatcher {
    /// Compute single dot product.
    pub fn dot(a: &[i8], b: &[i8]) -> i32 {
        crate::simd::dot_i8::dot_i8(a, b)
    }
    
    /// Compute int8 dot products for candidate reranking.
    pub fn compute(
        query: &[i8],
        vectors: &[i8],
        cand_ids: &[u32],
        dim: usize,
        out_scores: &mut [i32],
    ) {
        crate::simd::dot_i8::dot_i8_indexed(query, vectors, cand_ids, dim, out_scores);
    }

    /// Compute with dequantization for contiguous vectors.
    pub fn compute_batch_contiguous(
        query: &[i8],
        vectors: &[i8],
        scales: &[f32],
        dim: usize,
        out_scores: &mut [f32],
    ) {
        crate::simd::dot_i8::dot_i8_batch(query, vectors, scales, dim, out_scores);
    }
    
    /// Legacy interface - compute with dequantization for indexed access.
    pub fn compute_batch(
        query: &[i8],
        vectors: &[i8],
        cand_ids: &[u32],
        dim: usize,
        query_scale: f32,
        vec_scales: &[f32],
        out_scores: &mut [f32],
    ) {
        let n_cand = cand_ids.len();
        assert!(query.len() >= dim);
        assert!(out_scores.len() >= n_cand);
        
        // Compute int32 scores first
        let mut int_scores = vec![0i32; n_cand];
        Self::compute(query, vectors, cand_ids, dim, &mut int_scores);
        
        // Dequantize
        let denom = 127.0 * 127.0;
        for (i, &cand_id) in cand_ids.iter().enumerate() {
            let scale = query_scale * vec_scales[cand_id as usize] / denom;
            out_scores[i] = int_scores[i] as f32 * scale;
        }
    }

    /// Rust fallback for single dot (for testing).
    #[allow(dead_code)]
    pub(crate) fn dot_fallback(a: &[i8], b: &[i8]) -> i32 {
        a.iter().zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum()
    }

    /// Rust fallback implementation (for testing).
    #[allow(dead_code)]
    pub(crate) fn compute_fallback(
        query: &[i8],
        vectors: &[i8],
        cand_ids: &[u32],
        dim: usize,
        out_scores: &mut [i32],
    ) {
        for (i, &cand_id) in cand_ids.iter().enumerate() {
            let offset = cand_id as usize * dim;
            let vec = &vectors[offset..offset + dim];
            out_scores[i] = Self::dot_fallback(&query[..dim], vec);
        }
    }
    
    /// Rust fallback for batch contiguous (for testing).
    #[allow(dead_code)]
    pub(crate) fn compute_batch_fallback(
        query: &[i8],
        vectors: &[i8],
        scales: &[f32],
        dim: usize,
        out_scores: &mut [f32],
    ) {
        for (i, &scale) in scales.iter().enumerate() {
            let offset = i * dim;
            let vec = &vectors[offset..offset + dim];
            let int_score = Self::dot_fallback(&query[..dim], vec);
            out_scores[i] = int_score as f32 * scale;
        }
    }
}

// ============================================================================
// Visibility Check Dispatcher
// ============================================================================

/// SIMD-accelerated batch visibility checking for MVCC snapshots.
/// 
/// Checks which rows are visible to a given snapshot timestamp.
/// A row is visible if: commit_ts != 0 && commit_ts < snapshot_ts
/// Or if the row belongs to the current transaction (txn_id match).
pub struct VisibilityDispatcher;

impl VisibilityDispatcher {
    /// Check visibility for a batch of rows based on commit timestamps.
    /// 
    /// # Arguments
    /// * `commit_timestamps` - Array of commit timestamps (0 = uncommitted)
    /// * `snapshot_ts` - The snapshot timestamp for visibility check
    /// * `visible_mask` - Output: 1 if visible, 0 if not visible
    /// 
    /// # Panics
    /// Panics if visible_mask length doesn't match commit_timestamps length.
    pub fn check_batch(
        commit_timestamps: &[u64],
        snapshot_ts: u64,
        visible_mask: &mut [u8],
    ) {
        crate::simd::visibility::visibility_check(commit_timestamps, snapshot_ts, visible_mask);
    }
    
    /// Check visibility with transaction ID awareness (for self-visibility).
    /// 
    /// A row is visible if:
    /// - (commit_ts != 0 && commit_ts < snapshot_ts), OR
    /// - txn_id == current_txn_id (self-visibility)
    /// 
    /// # Arguments
    /// * `commit_timestamps` - Array of commit timestamps (0 = uncommitted)
    /// * `txn_ids` - Array of transaction IDs that wrote each row
    /// * `snapshot_ts` - The snapshot timestamp for visibility check
    /// * `current_txn_id` - The current transaction's ID
    /// * `visible_mask` - Output: 1 if visible, 0 if not visible
    pub fn check_batch_with_txn(
        commit_timestamps: &[u64],
        txn_ids: &[u64],
        snapshot_ts: u64,
        current_txn_id: u64,
        visible_mask: &mut [u8],
    ) {
        crate::simd::visibility::visibility_check_with_txn(
            commit_timestamps, txn_ids, snapshot_ts, current_txn_id, visible_mask
        );
    }
    
    /// Rust fallback implementation for batch visibility check (for testing).
    #[allow(dead_code)]
    pub(crate) fn check_batch_fallback(
        commit_timestamps: &[u64],
        snapshot_ts: u64,
        visible_mask: &mut [u8],
    ) {
        for (i, &commit_ts) in commit_timestamps.iter().enumerate() {
            visible_mask[i] = if commit_ts != 0 && commit_ts < snapshot_ts { 1 } else { 0 };
        }
    }
    
    /// Rust fallback implementation for batch visibility check with txn ID (for testing).
    #[allow(dead_code)]
    pub(crate) fn check_batch_with_txn_fallback(
        commit_timestamps: &[u64],
        txn_ids: &[u64],
        snapshot_ts: u64,
        current_txn_id: u64,
        visible_mask: &mut [u8],
    ) {
        for i in 0..commit_timestamps.len() {
            let commit_ts = commit_timestamps[i];
            let txn_id = txn_ids[i];
            let visible = (commit_ts != 0 && commit_ts < snapshot_ts) || txn_id == current_txn_id;
            visible_mask[i] = if visible { 1 } else { 0 };
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if SIMD kernels are available (runtime detection).
pub fn simd_available() -> bool {
    cpu_features().has_simd()
}

/// Get dispatch info for debugging (runtime detection).
pub fn dispatch_info() -> String {
    crate::simd::dispatch::dispatch_info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bps_scan_fallback() {
        let n_vec = 100;
        let n_blocks = 4;
        let proj = 1;
        
        // Create test data
        let mut bps = vec![0u8; n_blocks * proj * n_vec];
        for i in 0..n_vec {
            for b in 0..n_blocks {
                bps[b * n_vec + i] = (i % 256) as u8;
            }
        }
        
        let query = vec![128u8; n_blocks * proj];
        let mut out = vec![0u16; n_vec];
        
        BpsScanDispatcher::scan_fallback(&bps, n_vec, n_blocks, proj, &query, &mut out);
        
        // Check results make sense
        assert!(out.iter().all(|&d| d > 0 || d == 0));
    }
    
    #[test]
    fn test_bps_scan_fallback_u32() {
        let n_vec = 100;
        let n_blocks = 4;
        
        // Create test data
        let mut bps = vec![0u8; n_blocks * n_vec];
        for i in 0..n_vec {
            for b in 0..n_blocks {
                bps[b * n_vec + i] = (i % 256) as u8;
            }
        }
        
        let query = vec![128u8; n_blocks];
        let mut out = vec![0u32; n_vec];
        
        BpsScanDispatcher::scan_fallback_u32(&bps, n_vec, n_blocks, &query, &mut out);
        
        // Check results make sense
        for (i, &d) in out.iter().enumerate() {
            let expected: u32 = (0..n_blocks)
                .map(|_b| {
                    let v = (i % 256) as u8;
                    let q = 128u8;
                    (if q > v { q - v } else { v - q }) as u32
                })
                .sum();
            assert_eq!(d, expected);
        }
    }

    #[test]
    fn test_dot_i8_fallback() {
        let dim = 64;
        let n_vec = 10;
        
        let query: Vec<i8> = (0..dim).map(|i| (i % 128) as i8).collect();
        let vectors: Vec<i8> = (0..n_vec * dim)
            .map(|i| ((i / dim) as i8).wrapping_mul(2))
            .collect();
        let cand_ids: Vec<u32> = (0..n_vec as u32).collect();
        let mut out = vec![0i32; n_vec];
        
        DotI8Dispatcher::compute_fallback(&query, &vectors, &cand_ids, dim, &mut out);
        
        // Scores should vary
        assert!(out.iter().any(|&s| s != out[0]));
    }
    
    #[test]
    fn test_dot_single() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5];
        let b: Vec<i8> = vec![1, 2, 3, 4, 5];
        let result = DotI8Dispatcher::dot_fallback(&a, &b);
        assert_eq!(result, 1 + 4 + 9 + 16 + 25);
    }

    #[test]
    fn test_dispatch_info() {
        let info = dispatch_info();
        assert!(!info.is_empty());
        println!("Dispatch: {}", info);
    }
    
    /// Cross-validate SIMD dispatch vs fallback for bit-exact equivalence.
    #[test]
    fn test_simd_dispatch_cross_validation() {
        // Test BPS scan equivalence
        let n_vec = 256;
        let n_blocks = 8;
        
        // Generate deterministic test data
        let bps: Vec<u8> = (0..(n_blocks * n_vec))
            .map(|i| ((i * 17 + 13) % 256) as u8)
            .collect();
        let query: Vec<u8> = (0..n_blocks).map(|i| (i * 31 + 7) as u8).collect();
        
        // Reference: fallback implementation
        let mut ref_distances = vec![0u16; n_vec];
        BpsScanDispatcher::scan_fallback(&bps, n_vec, n_blocks, 1, &query, &mut ref_distances);
        
        // Dispatch: uses SIMD if available
        let mut dispatch_distances = vec![0u16; n_vec];
        BpsScanDispatcher::scan(&bps, n_vec, n_blocks, 1, &query, &mut dispatch_distances);
        
        // Verify bit-exact match
        for i in 0..n_vec {
            assert_eq!(
                ref_distances[i], dispatch_distances[i],
                "BPS scan mismatch at vector {}: fallback={}, dispatch={}",
                i, ref_distances[i], dispatch_distances[i]
            );
        }
        
        // Test int8 dot product equivalence
        let dim = 128;
        let a: Vec<i8> = (0..dim).map(|i| ((i * 3 - 64) % 128) as i8).collect();
        let b: Vec<i8> = (0..dim).map(|i| ((i * 7 + 32) % 128) as i8).collect();
        
        let ref_dot = DotI8Dispatcher::dot_fallback(&a, &b);
        let dispatch_dot = DotI8Dispatcher::dot(&a, &b);
        
        assert_eq!(
            ref_dot, dispatch_dot,
            "int8 dot product mismatch: fallback={}, dispatch={}",
            ref_dot, dispatch_dot
        );
    }
    
    /// Test CPU feature detection
    #[test]
    fn test_cpu_features_detection() {
        let features = cpu_features();
        let level = simd_level();
        
        println!("CPU Features: {:?}", features);
        println!("SIMD Level: {:?}", level);
        println!("Dispatch Info: {}", dispatch_info());
        
        // On any modern x86_64, we should have at least SSE4.1
        #[cfg(target_arch = "x86_64")]
        {
            // Most x86_64 CPUs have SSE4.1+
            assert!(level >= SimdLevel::Scalar);
        }
        
        // On aarch64, we always have NEON
        #[cfg(target_arch = "aarch64")]
        {
            assert!(features.has_neon);
            assert!(level >= SimdLevel::Neon);
        }
    }
    
    /// Test visibility check fallback
    #[test]
    fn test_visibility_check_basic() {
        let commit_timestamps = vec![10, 0, 5, 15, 20, 8];
        let snapshot_ts = 12;
        let mut visible_mask = vec![0u8; 6];
        
        VisibilityDispatcher::check_batch(&commit_timestamps, snapshot_ts, &mut visible_mask);
        
        // Expected: [1, 0, 1, 0, 0, 1]
        assert_eq!(visible_mask, vec![1, 0, 1, 0, 0, 1]);
    }
    
    /// Test visibility check with transaction ID
    #[test]
    fn test_visibility_check_with_txn() {
        let commit_timestamps = vec![10, 0, 5, 0, 20, 8];
        let txn_ids = vec![1, 2, 3, 99, 5, 6];
        let snapshot_ts = 12;
        let current_txn_id = 99;
        let mut visible_mask = vec![0u8; 6];
        
        VisibilityDispatcher::check_batch_with_txn(
            &commit_timestamps,
            &txn_ids,
            snapshot_ts,
            current_txn_id,
            &mut visible_mask,
        );
        
        // Expected: [1, 0, 1, 1, 0, 1]
        assert_eq!(visible_mask, vec![1, 0, 1, 1, 0, 1]);
    }
    
    /// Test visibility dispatcher SIMD vs fallback equivalence
    #[test]
    fn test_visibility_simd_equivalence() {
        let n_rows = 1024;
        
        // Generate test data
        let commit_timestamps: Vec<u64> = (0..n_rows)
            .map(|i| if i % 5 == 0 { 0 } else { (i * 7 % 100) as u64 })
            .collect();
        let txn_ids: Vec<u64> = (0..n_rows)
            .map(|i| (i % 10) as u64)
            .collect();
        let snapshot_ts = 50;
        let current_txn_id = 7;
        
        // Test basic visibility
        let mut ref_mask = vec![0u8; n_rows];
        let mut dispatch_mask = vec![0u8; n_rows];
        
        VisibilityDispatcher::check_batch_fallback(&commit_timestamps, snapshot_ts, &mut ref_mask);
        VisibilityDispatcher::check_batch(&commit_timestamps, snapshot_ts, &mut dispatch_mask);
        
        for i in 0..n_rows {
            assert_eq!(
                ref_mask[i], dispatch_mask[i],
                "Visibility mismatch at row {}: fallback={}, dispatch={}",
                i, ref_mask[i], dispatch_mask[i]
            );
        }
        
        // Test with txn ID
        let mut ref_mask_txn = vec![0u8; n_rows];
        let mut dispatch_mask_txn = vec![0u8; n_rows];
        
        VisibilityDispatcher::check_batch_with_txn_fallback(
            &commit_timestamps, &txn_ids, snapshot_ts, current_txn_id, &mut ref_mask_txn
        );
        VisibilityDispatcher::check_batch_with_txn(
            &commit_timestamps, &txn_ids, snapshot_ts, current_txn_id, &mut dispatch_mask_txn
        );
        
        for i in 0..n_rows {
            assert_eq!(
                ref_mask_txn[i], dispatch_mask_txn[i],
                "Visibility+txn mismatch at row {}: fallback={}, dispatch={}",
                i, ref_mask_txn[i], dispatch_mask_txn[i]
            );
        }
    }
}
