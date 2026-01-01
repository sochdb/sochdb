//! Unified SIMD Dispatch Architecture
//!
//! This module provides compile-time and runtime CPU feature detection
//! and dispatch for SIMD operations.
//!
//! # Architecture
//!
//! The dispatch system supports two modes:
//! 1. **Compile-Time Dispatch** (`#[cfg]` attributes): Used when target is known
//! 2. **Runtime Dispatch** (CPUID/feature detection): For portable binaries
//!
//! # Usage
//!
//! ```rust,ignore
//! use toondb_vector::simd::dispatch::{cpu_features, simd_level, SimdLevel};
//!
//! let features = cpu_features();
//! if features.has_avx2 {
//!     println!("AVX2 is available!");
//! }
//!
//! match simd_level() {
//!     SimdLevel::Avx512 => println!("Using AVX-512"),
//!     SimdLevel::Avx2 => println!("Using AVX2"),
//!     SimdLevel::Neon => println!("Using NEON"),
//!     _ => println!("Using scalar fallback"),
//! }
//! ```

use std::sync::OnceLock;

/// CPU feature flags detected at runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    /// SSE 4.1 support (x86)
    pub has_sse4_1: bool,
    /// AVX2 support (x86)
    pub has_avx2: bool,
    /// AVX-512F support (x86)
    pub has_avx512f: bool,
    /// AVX-512BW support (x86)
    pub has_avx512bw: bool,
    /// AVX-512 VNNI support (x86, for int8 acceleration)
    pub has_vnni: bool,
    /// NEON support (ARM, mandatory on aarch64)
    pub has_neon: bool,
    /// SVE support (ARM v8.2+)
    pub has_sve: bool,
    /// Dot product instruction support (ARM v8.2+)
    pub has_dotprod: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86()
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_arm()
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    fn detect_x86() -> Self {
        Self {
            has_sse4_1: is_x86_feature_detected!("sse4.1"),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512f: is_x86_feature_detected!("avx512f"),
            has_avx512bw: is_x86_feature_detected!("avx512bw"),
            has_vnni: is_x86_feature_detected!("avx512vnni"),
            has_neon: false,
            has_sve: false,
            has_dotprod: false,
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    fn detect_arm() -> Self {
        // NEON is mandatory on aarch64
        Self {
            has_sse4_1: false,
            has_avx2: false,
            has_avx512f: false,
            has_avx512bw: false,
            has_vnni: false,
            has_neon: true,
            // SVE and dotprod detection would require reading system registers
            // For now, we rely on compile-time detection
            has_sve: cfg!(target_feature = "sve"),
            has_dotprod: cfg!(target_feature = "dotprod"),
        }
    }
    
    /// Get the best SIMD level available.
    pub fn best_level(&self) -> SimdLevel {
        if self.has_avx512f && self.has_avx512bw {
            SimdLevel::Avx512
        } else if self.has_avx2 {
            SimdLevel::Avx2
        } else if self.has_neon {
            SimdLevel::Neon
        } else if self.has_sse4_1 {
            SimdLevel::Sse4
        } else {
            SimdLevel::Scalar
        }
    }
    
    /// Check if any SIMD acceleration is available.
    pub fn has_simd(&self) -> bool {
        self.has_avx2 || self.has_neon || self.has_sse4_1
    }
}

/// SIMD capability level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SimdLevel {
    /// No SIMD, scalar operations only
    Scalar = 0,
    /// SSE 4.1 (128-bit, x86)
    Sse4 = 1,
    /// NEON (128-bit, ARM)
    Neon = 2,
    /// AVX2 (256-bit, x86)
    Avx2 = 3,
    /// AVX-512 (512-bit, x86)
    Avx512 = 4,
}

impl SimdLevel {
    /// Elements per SIMD register for u8 operations.
    pub const fn u8_width(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse4 => 16,
            SimdLevel::Neon => 16,
            SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
        }
    }
    
    /// Elements per SIMD register for u64 operations.
    pub const fn u64_width(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse4 => 2,
            SimdLevel::Neon => 2,
            SimdLevel::Avx2 => 4,
            SimdLevel::Avx512 => 8,
        }
    }
    
    /// Elements per SIMD register for f32 operations.
    pub const fn f32_width(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse4 => 4,
            SimdLevel::Neon => 4,
            SimdLevel::Avx2 => 8,
            SimdLevel::Avx512 => 16,
        }
    }
    
    /// Register width in bits.
    pub const fn width_bits(self) -> usize {
        match self {
            SimdLevel::Scalar => 64,
            SimdLevel::Sse4 => 128,
            SimdLevel::Neon => 128,
            SimdLevel::Avx2 => 256,
            SimdLevel::Avx512 => 512,
        }
    }
    
    /// Theoretical speedup factor over scalar for byte operations.
    pub const fn speedup_factor(self) -> usize {
        self.u8_width()
    }
    
    /// Human-readable name.
    pub const fn name(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "Scalar",
            SimdLevel::Sse4 => "SSE4.1",
            SimdLevel::Neon => "NEON",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Avx512 => "AVX-512",
        }
    }
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Global CPU features, detected once at first use.
static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Get detected CPU features (cached).
#[inline]
pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

/// Get best available SIMD level.
#[inline]
pub fn simd_level() -> SimdLevel {
    cpu_features().best_level()
}

/// Check if SIMD acceleration is available.
#[inline]
pub fn simd_available() -> bool {
    cpu_features().has_simd()
}

/// Get a human-readable description of SIMD capabilities.
pub fn dispatch_info() -> String {
    let features = cpu_features();
    let level = features.best_level();
    
    let mut info = format!(
        "SIMD Level: {} ({}-bit)\n",
        level.name(),
        level.width_bits()
    );
    
    #[cfg(target_arch = "x86_64")]
    {
        info.push_str(&format!("  SSE4.1: {}\n", features.has_sse4_1));
        info.push_str(&format!("  AVX2: {}\n", features.has_avx2));
        info.push_str(&format!("  AVX-512F: {}\n", features.has_avx512f));
        info.push_str(&format!("  AVX-512BW: {}\n", features.has_avx512bw));
        info.push_str(&format!("  VNNI: {}\n", features.has_vnni));
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        info.push_str(&format!("  NEON: {}\n", features.has_neon));
        info.push_str(&format!("  SVE: {}\n", features.has_sve));
        info.push_str(&format!("  DOTPROD: {}\n", features.has_dotprod));
    }
    
    info
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_detection() {
        let features = cpu_features();
        let level = features.best_level();
        
        println!("Detected SIMD level: {:?}", level);
        println!("Features: {:?}", features);
        
        // At minimum, detection should work without panicking
        assert!(level >= SimdLevel::Scalar);
    }
    
    #[test]
    fn test_simd_widths() {
        assert_eq!(SimdLevel::Scalar.u8_width(), 1);
        assert_eq!(SimdLevel::Avx2.u8_width(), 32);
        assert_eq!(SimdLevel::Neon.u8_width(), 16);
        assert_eq!(SimdLevel::Avx512.u8_width(), 64);
    }
    
    #[test]
    fn test_dispatch_info() {
        let info = dispatch_info();
        println!("{}", info);
        assert!(!info.is_empty());
    }
}
