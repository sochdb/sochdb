// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Memory Residency Controls (Task 1)
//!
//! Eliminates page-fault storms during HNSW construction by ensuring
//! all vector data is RAM-resident before the random-access build phase.
//!
//! ## Problem
//!
//! HNSW construction performs O(N · ef_construction · avg_traversal_depth)
//! distance computations with adversarially non-sequential access patterns.
//! When vectors reside in mmap-backed storage without residency guarantees,
//! each random access can trigger a major page fault (10-100μs on NVMe).
//!
//! ## Solution
//!
//! Convert random-access page faults into a single sequential prefetch:
//! - `T_fault = p · #distance_evals · latency_per_fault = O(N · ef · L · p · t_fault)`
//! - `T_prefault = O(N · D · t_sequential_read)` (one-time)
//!
//! Sequential read throughput: ~5-6 GB/s (NVMe)
//! Random 4KB fault throughput: ~50-100 MB/s effective
//! Improvement: 50-100× on I/O-bound portion

use std::io;

/// Memory advice hints for mmap regions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemAdvice {
    /// Sequential access pattern (default)
    Sequential,
    /// Random access pattern
    Random,
    /// Will need this data soon (prefetch)
    WillNeed,
    /// Won't need this data (can be paged out)
    DontNeed,
}

/// Prefault an mmap region into RAM (Task 1: Option D fallback)
///
/// This function ensures all pages in the given memory region are
/// resident in RAM by reading one byte per page. This converts
/// O(N · random_faults) into O(sequential_read).
///
/// # Performance
///
/// For 1GB of vectors at 4KB pages:
/// - Pages to touch: 262,144
/// - Time at ~1μs per prefault: ~0.3s
/// - Sequential read equivalent: ~0.2s
///
/// This is acceptable overhead compared to the alternative of
/// ~10,000+ major faults at ~100μs each = 1s+ of stall time.
#[inline(never)]
pub fn prefault_region(ptr: *const u8, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    
    const PAGE_SIZE: usize = 4096;
    let mut pages_touched = 0usize;
    
    // Touch every page to force it into RAM
    // Using volatile read to prevent optimization away
    for offset in (0..len).step_by(PAGE_SIZE) {
        unsafe {
            let _byte: u8 = std::ptr::read_volatile(ptr.add(offset));
        }
        pages_touched += 1;
    }
    
    // Also touch the last byte to ensure final partial page is loaded
    if len > 0 && len % PAGE_SIZE != 0 {
        unsafe {
            let _byte: u8 = std::ptr::read_volatile(ptr.add(len - 1));
        }
    }
    
    pages_touched
}

/// Prefault with progress callback for large regions
pub fn prefault_region_with_progress<F>(
    ptr: *const u8,
    len: usize,
    mut progress: F,
) -> usize 
where
    F: FnMut(usize, usize),  // (bytes_done, total_bytes)
{
    if len == 0 {
        return 0;
    }
    
    const PAGE_SIZE: usize = 4096;
    const PROGRESS_INTERVAL: usize = 1024 * 1024; // Report every 1MB
    
    let mut pages_touched = 0usize;
    let mut last_report = 0usize;
    
    for offset in (0..len).step_by(PAGE_SIZE) {
        unsafe {
            let _byte: u8 = std::ptr::read_volatile(ptr.add(offset));
        }
        pages_touched += 1;
        
        if offset - last_report >= PROGRESS_INTERVAL {
            progress(offset, len);
            last_report = offset;
        }
    }
    
    if len > 0 && len % PAGE_SIZE != 0 {
        unsafe {
            let _byte: u8 = std::ptr::read_volatile(ptr.add(len - 1));
        }
    }
    
    progress(len, len);
    pages_touched
}

/// Apply madvise hint to a memory region (Unix only)
#[cfg(target_os = "linux")]
pub fn madvise(ptr: *mut u8, len: usize, advice: MemAdvice) -> io::Result<()> {
    use libc::{madvise as libc_madvise, MADV_WILLNEED, MADV_RANDOM, MADV_SEQUENTIAL, MADV_DONTNEED};
    
    let advice_int = match advice {
        MemAdvice::Sequential => MADV_SEQUENTIAL,
        MemAdvice::Random => MADV_RANDOM,
        MemAdvice::WillNeed => MADV_WILLNEED,
        MemAdvice::DontNeed => MADV_DONTNEED,
    };
    
    let result = unsafe {
        libc_madvise(ptr as *mut libc::c_void, len, advice_int)
    };
    
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Apply madvise hint to a memory region (macOS)
#[cfg(target_os = "macos")]
pub fn madvise(ptr: *mut u8, len: usize, advice: MemAdvice) -> io::Result<()> {
    use libc::{madvise as libc_madvise, MADV_WILLNEED, MADV_RANDOM, MADV_SEQUENTIAL, MADV_FREE};
    
    let advice_int = match advice {
        MemAdvice::Sequential => MADV_SEQUENTIAL,
        MemAdvice::Random => MADV_RANDOM,
        MemAdvice::WillNeed => MADV_WILLNEED,
        MemAdvice::DontNeed => MADV_FREE,
    };
    
    let result = unsafe {
        libc_madvise(ptr as *mut libc::c_void, len, advice_int)
    };
    
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Fallback for non-Unix platforms
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn madvise(_ptr: *mut u8, _len: usize, _advice: MemAdvice) -> io::Result<()> {
    // No-op on unsupported platforms
    Ok(())
}

/// Residency statistics for telemetry
#[derive(Debug, Clone, Default)]
pub struct ResidencyStats {
    /// Size of memory region in bytes
    pub region_bytes: usize,
    /// Number of pages in region
    pub total_pages: usize,
    /// Time spent in prefault (nanoseconds)
    pub prefault_ns: u64,
    /// Prefault throughput in MB/s
    pub prefault_mb_per_sec: f64,
}

impl ResidencyStats {
    /// Create stats for a prefault operation
    pub fn from_prefault(region_bytes: usize, elapsed_ns: u64) -> Self {
        let total_pages = (region_bytes + 4095) / 4096;
        let prefault_mb_per_sec = if elapsed_ns > 0 {
            (region_bytes as f64 / 1024.0 / 1024.0) / (elapsed_ns as f64 / 1e9)
        } else {
            0.0
        };
        
        Self {
            region_bytes,
            total_pages,
            prefault_ns: elapsed_ns,
            prefault_mb_per_sec,
        }
    }
    
    /// Print human-readable summary
    pub fn print_summary(&self, label: &str) {
        eprintln!("[residency] {}: {:.1} MB, {} pages, {:.2}s ({:.0} MB/s)",
            label,
            self.region_bytes as f64 / 1024.0 / 1024.0,
            self.total_pages,
            self.prefault_ns as f64 / 1e9,
            self.prefault_mb_per_sec,
        );
    }
}

/// Ensure memory region is resident and optimally configured for HNSW build
///
/// This is the main entry point for Task 1. Call this before HNSW construction
/// to eliminate page-fault storms during random graph traversal.
///
/// # Strategy
///
/// 1. `madvise(MADV_WILLNEED)` - Hint to kernel to prefetch asynchronously
/// 2. `madvise(MADV_RANDOM)` - Disable readahead (harmful for random access)
/// 3. Explicit prefault loop - Guarantee residency before build starts
pub fn ensure_resident_for_hnsw(
    ptr: *mut u8,
    len: usize,
    verbose: bool,
) -> ResidencyStats {
    use std::time::Instant;
    
    if len == 0 {
        return ResidencyStats::default();
    }
    
    let start = Instant::now();
    
    // Step 1: Hint for async prefetch
    if let Err(e) = madvise(ptr, len, MemAdvice::WillNeed) {
        if verbose {
            eprintln!("[residency] madvise(WILLNEED) failed: {} (falling back to explicit prefault)", e);
        }
    }
    
    // Step 2: Disable readahead (harmful for HNSW's random access pattern)
    if let Err(e) = madvise(ptr, len, MemAdvice::Random) {
        if verbose {
            eprintln!("[residency] madvise(RANDOM) failed: {}", e);
        }
    }
    
    // Step 3: Explicit prefault to guarantee residency
    let pages_touched = if verbose {
        prefault_region_with_progress(ptr as *const u8, len, |done, total| {
            if done == total {
                eprintln!("[residency] Prefaulted {:.1} MB", total as f64 / 1024.0 / 1024.0);
            } else {
                eprint!("\r[residency] Prefaulting... {:.0}%", 100.0 * done as f64 / total as f64);
            }
        })
    } else {
        prefault_region(ptr as *const u8, len)
    };
    
    let elapsed = start.elapsed();
    let stats = ResidencyStats::from_prefault(len, elapsed.as_nanos() as u64);
    
    if verbose {
        eprintln!("[residency] {} pages resident in {:.2}s ({:.0} MB/s)",
            pages_touched,
            elapsed.as_secs_f64(),
            stats.prefault_mb_per_sec,
        );
    }
    
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prefault() {
        // Allocate some memory and prefault it
        let size = 64 * 4096; // 64 pages
        let mut data = vec![0u8; size];
        
        let pages = prefault_region(data.as_ptr(), size);
        assert_eq!(pages, 64);
        
        // Verify data is still accessible
        data[0] = 42;
        assert_eq!(data[0], 42);
    }
}
