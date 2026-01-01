//! Property-based tests for SIMD kernel equivalence.
//!
//! These tests ensure that SIMD implementations produce identical results
//! to scalar implementations across all inputs.

use proptest::prelude::*;
use toondb_vector::simd::{bps_scan, dot_i8, visibility};

// ============================================================================
// BPS Scan Equivalence Tests
// ============================================================================

/// Scalar reference implementation for BPS scan
fn bps_scan_reference(bps: &[u8], n_vec: usize, n_blocks: usize, query: &[u8], out: &mut [u16]) {
    for d in out.iter_mut().take(n_vec) {
        *d = 0;
    }
    
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn bps_scan_equivalence(
        n_vec in 1..1000usize,
        n_blocks in 1..64usize,
    ) {
        // Generate random BPS data
        let bps: Vec<u8> = (0..n_vec * n_blocks)
            .map(|i| ((i * 17 + 13) % 256) as u8)
            .collect();
        let query: Vec<u8> = (0..n_blocks)
            .map(|i| ((i * 31 + 7) % 256) as u8)
            .collect();
        
        let mut out_ref = vec![0u16; n_vec];
        let mut out_simd = vec![0u16; n_vec];
        
        bps_scan_reference(&bps, n_vec, n_blocks, &query, &mut out_ref);
        bps_scan::bps_scan(&bps, n_vec, n_blocks, &query, &mut out_simd);
        
        prop_assert_eq!(out_ref, out_simd);
    }
    
    #[test]
    fn bps_scan_random_data(
        n_vec in 32..256usize,
        n_blocks in 4..32usize,
        seed in 0u64..1000,
    ) {
        // Generate random data using seed
        let bps: Vec<u8> = (0..n_vec * n_blocks)
            .map(|i| ((i as u64 * seed + 17) % 256) as u8)
            .collect();
        let query: Vec<u8> = (0..n_blocks)
            .map(|i| ((i as u64 * seed + 31) % 256) as u8)
            .collect();
        
        let mut out_ref = vec![0u16; n_vec];
        let mut out_simd = vec![0u16; n_vec];
        
        bps_scan_reference(&bps, n_vec, n_blocks, &query, &mut out_ref);
        bps_scan::bps_scan(&bps, n_vec, n_blocks, &query, &mut out_simd);
        
        prop_assert_eq!(out_ref, out_simd);
    }
}

// ============================================================================
// Int8 Dot Product Equivalence Tests
// ============================================================================

/// Scalar reference implementation for i8 dot product
fn dot_i8_reference(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32) * (y as i32))
        .sum()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn dot_i8_equivalence(
        dim in 1..1024usize,
        seed in 0u64..1000,
    ) {
        let a: Vec<i8> = (0..dim)
            .map(|i| ((i as u64 * seed + 17) % 256) as u8 as i8)
            .collect();
        let b: Vec<i8> = (0..dim)
            .map(|i| ((i as u64 * seed + 31) % 256) as u8 as i8)
            .collect();
        
        let ref_result = dot_i8_reference(&a, &b);
        let simd_result = dot_i8::dot_i8(&a, &b);
        
        prop_assert_eq!(ref_result, simd_result);
    }
    
    #[test]
    fn dot_i8_batch_equivalence(
        dim in 64..256usize,
        n_vec in 1..32usize,
        seed in 0u64..1000,
    ) {
        let query: Vec<i8> = (0..dim)
            .map(|i| (((i as u64 * seed) % 127) as i8))
            .collect();
        let vectors: Vec<i8> = (0..n_vec * dim)
            .map(|i| (((i as u64 * seed + 17) % 127) as i8))
            .collect();
        let scales: Vec<f32> = (0..n_vec)
            .map(|i| 0.01 + (i as f32 * 0.001))
            .collect();
        
        let mut out_simd = vec![0.0f32; n_vec];
        dot_i8::dot_i8_batch(&query, &vectors, &scales, dim, &mut out_simd);
        
        // Reference: compute each dot product individually
        for (i, &scale) in scales.iter().enumerate() {
            let offset = i * dim;
            let vec = &vectors[offset..offset + dim];
            let ref_dot = dot_i8_reference(&query, vec);
            let ref_result = ref_dot as f32 * scale;
            
            prop_assert!((out_simd[i] - ref_result).abs() < 1e-5,
                "Mismatch at vec {}: simd={}, ref={}", i, out_simd[i], ref_result);
        }
    }
}

// ============================================================================
// Visibility Check Equivalence Tests
// ============================================================================

/// Scalar reference for visibility check
fn visibility_reference(commits: &[u64], snapshot: u64, mask: &mut [u8]) {
    for (i, &commit) in commits.iter().enumerate() {
        mask[i] = if commit != 0 && commit < snapshot { 1 } else { 0 };
    }
}

/// Scalar reference for visibility check with txn
fn visibility_with_txn_reference(
    commits: &[u64],
    txns: &[u64],
    snapshot: u64,
    current_txn: u64,
    mask: &mut [u8],
) {
    for i in 0..commits.len() {
        let visible = (commits[i] != 0 && commits[i] < snapshot) || txns[i] == current_txn;
        mask[i] = if visible { 1 } else { 0 };
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn visibility_equivalence(
        n_rows in 1..512usize,
        snapshot in 1u64..1000,
        seed in 0u64..1000,
    ) {
        let commits: Vec<u64> = (0..n_rows)
            .map(|i| {
                let v = ((i as u64 * seed + 17) % 150) as u64;
                if v > 100 { 0 } else { v } // Some uncommitted
            })
            .collect();
        
        let mut ref_mask = vec![0u8; n_rows];
        let mut simd_mask = vec![0u8; n_rows];
        
        visibility_reference(&commits, snapshot, &mut ref_mask);
        visibility::visibility_check(&commits, snapshot, &mut simd_mask);
        
        prop_assert_eq!(ref_mask, simd_mask);
    }
    
    #[test]
    fn visibility_with_txn_equivalence(
        n_rows in 1..256usize,
        snapshot in 1u64..1000,
        current_txn in 0u64..10,
        seed in 0u64..1000,
    ) {
        let commits: Vec<u64> = (0..n_rows)
            .map(|i| {
                let v = ((i as u64 * seed + 17) % 150) as u64;
                if v > 100 { 0 } else { v }
            })
            .collect();
        let txns: Vec<u64> = (0..n_rows)
            .map(|i| ((i as u64 * seed + 3) % 10))
            .collect();
        
        let mut ref_mask = vec![0u8; n_rows];
        let mut simd_mask = vec![0u8; n_rows];
        
        visibility_with_txn_reference(&commits, &txns, snapshot, current_txn, &mut ref_mask);
        visibility::visibility_check_with_txn(&commits, &txns, snapshot, current_txn, &mut simd_mask);
        
        prop_assert_eq!(ref_mask, simd_mask);
    }
}

// ============================================================================
// Boundary Tests (non-proptest)
// ============================================================================

#[test]
fn test_bps_boundary_sizes() {
    // Test sizes at SIMD lane boundaries
    for n_vec in [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
        let n_blocks = 4;
        let bps: Vec<u8> = (0..n_vec * n_blocks).map(|i| (i % 256) as u8).collect();
        let query: Vec<u8> = vec![128; n_blocks];
        
        let mut out_ref = vec![0u16; n_vec];
        let mut out_simd = vec![0u16; n_vec];
        
        bps_scan_reference(&bps, n_vec, n_blocks, &query, &mut out_ref);
        bps_scan::bps_scan(&bps, n_vec, n_blocks, &query, &mut out_simd);
        
        assert_eq!(out_ref, out_simd, "Mismatch for n_vec={}", n_vec);
    }
}

#[test]
fn test_dot_i8_boundary_sizes() {
    // Test sizes at SIMD lane boundaries
    for dim in [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257] {
        let a: Vec<i8> = (0..dim).map(|i| ((i * 3) % 127) as i8).collect();
        let b: Vec<i8> = (0..dim).map(|i| ((i * 7) % 127) as i8).collect();
        
        let ref_result = dot_i8_reference(&a, &b);
        let simd_result = dot_i8::dot_i8(&a, &b);
        
        assert_eq!(ref_result, simd_result, "Mismatch for dim={}", dim);
    }
}

#[test]
fn test_visibility_boundary_sizes() {
    // Test sizes at SIMD lane boundaries
    for n_rows in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
        let commits: Vec<u64> = (0..n_rows).map(|i| (i * 10) as u64).collect();
        let snapshot = 50;
        
        let mut ref_mask = vec![0u8; n_rows];
        let mut simd_mask = vec![0u8; n_rows];
        
        visibility_reference(&commits, snapshot, &mut ref_mask);
        visibility::visibility_check(&commits, snapshot, &mut simd_mask);
        
        assert_eq!(ref_mask, simd_mask, "Mismatch for n_rows={}", n_rows);
    }
}

#[test]
fn test_extreme_values() {
    // Test with extreme i8 values
    let a: Vec<i8> = vec![127, -128, 0, 127, -128, 0, 1, -1];
    let b: Vec<i8> = vec![-128, 127, 0, -128, 127, 0, -1, 1];
    
    let ref_result = dot_i8_reference(&a, &b);
    let simd_result = dot_i8::dot_i8(&a, &b);
    
    assert_eq!(ref_result, simd_result);
    // 127*-128 + -128*127 + 0*0 + 127*-128 + -128*127 + 0*0 + 1*-1 + -1*1
    // = -16256 - 16256 + 0 - 16256 - 16256 + 0 - 1 - 1 = -65026
    assert_eq!(simd_result, -65026);
}

#[test]
fn test_all_zeros() {
    let n_vec = 100;
    let n_blocks = 10;
    let bps = vec![0u8; n_vec * n_blocks];
    let query = vec![0u8; n_blocks];
    let mut out = vec![0u16; n_vec];
    
    bps_scan::bps_scan(&bps, n_vec, n_blocks, &query, &mut out);
    
    assert!(out.iter().all(|&d| d == 0));
}

#[test]
fn test_all_max() {
    let n_vec = 100;
    let n_blocks = 10;
    let bps = vec![255u8; n_vec * n_blocks];
    let query = vec![0u8; n_blocks];
    let mut out = vec![0u16; n_vec];
    
    bps_scan::bps_scan(&bps, n_vec, n_blocks, &query, &mut out);
    
    // Each block contributes 255, so total should be 255 * n_blocks
    assert!(out.iter().all(|&d| d == 255 * n_blocks as u16));
}
