//! Build script for ToonDB Vector.
//!
//! This build script handles feature detection for the pure Rust SIMD
//! implementations. No C++ compilation is required - all SIMD kernels
//! are now implemented in pure Rust using `core::arch` intrinsics.
//!
//! # SIMD Support
//!
//! - x86_64: AVX2, AVX-512 (runtime detected)
//! - aarch64: NEON (mandatory), SVE (optional)
//! - Other: Scalar fallback

fn main() {
    // Detect target architecture for conditional compilation hints
    let target = std::env::var("TARGET").unwrap_or_default();
    
    // Emit cfg flags for architecture-specific code
    if target.contains("x86_64") {
        println!("cargo:rustc-cfg=has_x86_simd");
    } else if target.contains("aarch64") || target.contains("arm64") {
        println!("cargo:rustc-cfg=has_arm_simd");
    }
    
    // Rerun if simd modules change
    println!("cargo:rerun-if-changed=src/simd/");
    
    // Note: No C++ compilation needed!
    // All SIMD kernels are now pure Rust in src/simd/
    //
    // Benefits:
    // - Unified toolchain: `cargo build` is sufficient
    // - Cross-function inlining enabled
    // - Smaller binaries (no libstdc++ dependency)
    // - Better error messages and debugging
    // - `cargo miri` support for undefined behavior detection
}
