//! Pure Rust SIMD kernels for ToonDB vector operations.
//!
//! This module provides high-performance SIMD implementations for:
//! - **BPS Scan**: L1 distance computation for Block Projection Sketches
//! - **Int8 Dot Product**: Quantized dot products for reranking
//! - **Visibility Check**: MVCC visibility bitmap generation
//!
//! # Architecture
//!
//! The module uses a trait-based abstraction (`SimdBackend`) that allows
//! swapping between architecture-specific intrinsics (`core::arch`) and
//! portable SIMD (`core::simd`) when it stabilizes.
//!
//! # CPU Detection
//!
//! Runtime CPU feature detection is used to select the optimal code path:
//! - x86_64: AVX-512 > AVX2 > SSE4.1 > Scalar
//! - aarch64: NEON (mandatory) + optional SVE
//! - Other: Scalar fallback
//!
//! # Safety
//!
//! SIMD intrinsic functions are `unsafe` because they require specific CPU
//! features. The dispatch layer handles feature detection and ensures
//! that intrinsics are only called when the CPU supports them.

pub mod bps_scan;
pub mod dot_i8;
pub mod visibility;
pub mod dispatch;

#[cfg(feature = "portable-simd")]
pub mod portable;

// Re-export main types and functions
pub use dispatch::{CpuFeatures, SimdLevel, cpu_features, simd_level};
pub use bps_scan::{bps_scan, bps_scan_u32};
pub use dot_i8::{dot_i8, dot_i8_batch};
pub use visibility::{visibility_check, visibility_check_with_txn};

/// Trait for SIMD backend abstraction.
/// 
/// This allows the algorithm code to be generic over the backend,
/// enabling A/B testing and gradual migration to `core::simd`.
pub trait SimdBackend {
    /// 32 x u8 vector type
    type U8x32;
    /// 32 x i8 vector type
    type I8x32;
    /// 8 x f32 vector type
    type F32x8;
    /// 4 x u64 vector type
    type U64x4;
    
    /// Compute L1 distance between two u8 vectors
    fn l1_distance_u8(a: Self::U8x32, b: Self::U8x32) -> Self::U8x32;
    
    /// Compute dot product of two i8 vectors
    fn dot_i8(a: Self::I8x32, b: Self::I8x32) -> i32;
    
    /// Compute dot product of two f32 vectors
    fn dot_f32(a: Self::F32x8, b: Self::F32x8) -> f32;
}
