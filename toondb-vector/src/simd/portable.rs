//! Portable SIMD Layer using `core::simd`
//!
//! This module provides a portable SIMD abstraction that generates optimal
//! code for each target architecture. It uses Rust's `core::simd` (nightly)
//! for write-once, run-optimally-everywhere SIMD code.
//!
//! # Status
//!
//! This module requires the `portable-simd` feature flag and nightly Rust.
//! It is intended as a future migration path when `core::simd` stabilizes.
//!
//! # Usage
//!
//! Enable with `--features portable-simd` on nightly:
//!
//! ```toml
//! [features]
//! portable-simd = []
//! ```
//!
//! # Advantages
//!
//! - **Single Codebase**: Write once, runs optimally on x86, ARM, WASM, RISC-V
//! - **Safe Abstractions**: No `unsafe` required for core operations
//! - **Compiler Optimization**: LLVM auto-vectorizes and selects optimal instructions
//! - **WASM Support**: Generates WASM SIMD instructions for browser-based vector search

#![cfg(feature = "portable-simd")]
#![feature(portable_simd)]

use std::simd::{Simd, SimdPartialOrd, SimdUint, SimdInt};

/// Portable L1 distance for u8 vectors.
///
/// Computes the sum of absolute differences between two u8 slices.
pub fn l1_distance_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    
    const LANES: usize = 32;
    let mut sum = Simd::<u32, LANES>::splat(0);
    
    let chunks = a.len() / LANES;
    
    for i in 0..chunks {
        let offset = i * LANES;
        let a_vec = Simd::<u8, LANES>::from_slice(&a[offset..offset + LANES]);
        let b_vec = Simd::<u8, LANES>::from_slice(&b[offset..offset + LANES]);
        
        // Compute absolute difference
        let diff = a_vec.abs_diff(b_vec);
        
        // Widen to u32 and accumulate
        sum += diff.cast::<u32>();
    }
    
    // Handle remainder
    let mut scalar_sum = sum.reduce_sum();
    for i in (chunks * LANES)..a.len() {
        scalar_sum += a[i].abs_diff(b[i]) as u32;
    }
    
    scalar_sum
}

/// Portable dot product for i8 vectors.
///
/// Computes the sum of products of two i8 slices.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    
    const LANES: usize = 16;
    let mut sum = Simd::<i32, LANES>::splat(0);
    
    let chunks = a.len() / LANES;
    
    for i in 0..chunks {
        let offset = i * LANES;
        let a_vec = Simd::<i8, LANES>::from_slice(&a[offset..offset + LANES]);
        let b_vec = Simd::<i8, LANES>::from_slice(&b[offset..offset + LANES]);
        
        // Widen to i16, multiply, then widen to i32
        let a_i16 = a_vec.cast::<i16>();
        let b_i16 = b_vec.cast::<i16>();
        let prod = a_i16 * b_i16;
        
        sum += prod.cast::<i32>();
    }
    
    // Handle remainder
    let mut scalar_sum = sum.reduce_sum();
    for i in (chunks * LANES)..a.len() {
        scalar_sum += (a[i] as i32) * (b[i] as i32);
    }
    
    scalar_sum
}

/// Portable dot product for f32 vectors.
///
/// Computes the sum of products of two f32 slices.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    const LANES: usize = 8;
    let mut sum = Simd::<f32, LANES>::splat(0.0);
    
    let chunks = a.len() / LANES;
    
    for i in 0..chunks {
        let offset = i * LANES;
        let a_vec = Simd::<f32, LANES>::from_slice(&a[offset..offset + LANES]);
        let b_vec = Simd::<f32, LANES>::from_slice(&b[offset..offset + LANES]);
        
        sum += a_vec * b_vec;
    }
    
    // Handle remainder
    let mut scalar_sum = sum.reduce_sum();
    for i in (chunks * LANES)..a.len() {
        scalar_sum += a[i] * b[i];
    }
    
    scalar_sum
}

/// Portable BPS scan using portable SIMD.
///
/// This is a reference implementation that should match the
/// architecture-specific implementations in `bps_scan.rs`.
pub fn bps_scan_portable(
    bps: &[u8],
    n_vec: usize,
    n_blocks: usize,
    query: &[u8],
    out: &mut [u16],
) {
    assert!(query.len() >= n_blocks);
    assert!(out.len() >= n_vec);
    
    const LANES: usize = 16;
    let vec_aligned = (n_vec / LANES) * LANES;
    
    // Zero output
    out.iter_mut().take(n_vec).for_each(|d| *d = 0);
    
    for chunk_start in (0..vec_aligned).step_by(LANES) {
        let mut acc = Simd::<u16, LANES>::splat(0);
        
        for slot in 0..n_blocks {
            let base = slot * n_vec + chunk_start;
            let v = Simd::<u8, LANES>::from_slice(&bps[base..base + LANES]);
            let q = Simd::<u8, LANES>::splat(query[slot]);
            
            // Compute |v - q|
            let diff = v.abs_diff(q);
            
            // Widen to u16 and accumulate
            acc += diff.cast::<u16>();
        }
        
        // Store
        acc.copy_to_slice(&mut out[chunk_start..chunk_start + LANES]);
    }
    
    // Handle remainder
    for i in vec_aligned..n_vec {
        let mut sum: u16 = 0;
        for slot in 0..n_blocks {
            let v = bps[slot * n_vec + i];
            let q = query[slot];
            sum = sum.saturating_add(v.abs_diff(q) as u16);
        }
        out[i] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l1_distance() {
        let a: Vec<u8> = (0..64).collect();
        let b: Vec<u8> = (0..64).map(|x| x + 1).collect();
        
        let result = l1_distance_u8(&a, &b);
        assert_eq!(result, 64); // Each diff is 1
    }
    
    #[test]
    fn test_dot_i8() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b: Vec<i8> = vec![8, 7, 6, 5, 4, 3, 2, 1];
        
        let result = dot_i8(&a, &b);
        let expected: i32 = a.iter().zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum();
        
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_dot_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0];
        
        let result = dot_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_bps_scan_portable() {
        let n_vec = 32;
        let n_blocks = 4;
        let bps: Vec<u8> = (0..n_vec * n_blocks).map(|i| (i % 256) as u8).collect();
        let query: Vec<u8> = vec![128; n_blocks];
        let mut out = vec![0u16; n_vec];
        
        bps_scan_portable(&bps, n_vec, n_blocks, &query, &mut out);
        
        // Verify non-zero results
        assert!(out.iter().any(|&x| x > 0));
    }
}
