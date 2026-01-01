// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Page Fault Telemetry (Task 2)
//!
//! Provides instrumentation to validate the mmap/page-fault hypothesis.
//! Measures major/minor faults before and after operations to attribute
//! performance regressions to memory subsystem behavior.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let mut telemetry = FaultTelemetry::capture_start();
//! // ... perform operation ...
//! telemetry.capture_end();
//! telemetry.print_summary("HNSW Build");
//! ```

use std::time::{Duration, Instant};

/// Page fault statistics from getrusage()
#[derive(Debug, Clone, Default)]
pub struct FaultStats {
    /// Major page faults (required disk I/O)
    pub major_faults: u64,
    /// Minor page faults (no disk I/O, just page table update)
    pub minor_faults: u64,
    /// Maximum resident set size in KB
    pub max_rss_kb: u64,
    /// User CPU time in microseconds
    pub user_time_us: u64,
    /// System CPU time in microseconds
    pub sys_time_us: u64,
}

impl FaultStats {
    /// Capture current resource usage
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub fn capture() -> Self {
        use libc::{getrusage, rusage, RUSAGE_SELF};
        use std::mem::MaybeUninit;
        
        let mut usage = MaybeUninit::<rusage>::uninit();
        let result = unsafe {
            getrusage(RUSAGE_SELF, usage.as_mut_ptr())
        };
        
        if result != 0 {
            return Self::default();
        }
        
        let usage = unsafe { usage.assume_init() };
        
        Self {
            major_faults: usage.ru_majflt as u64,
            minor_faults: usage.ru_minflt as u64,
            #[cfg(target_os = "linux")]
            max_rss_kb: usage.ru_maxrss as u64, // Already in KB on Linux
            #[cfg(target_os = "macos")]
            max_rss_kb: (usage.ru_maxrss as u64) / 1024, // Bytes on macOS
            user_time_us: usage.ru_utime.tv_sec as u64 * 1_000_000 + usage.ru_utime.tv_usec as u64,
            sys_time_us: usage.ru_stime.tv_sec as u64 * 1_000_000 + usage.ru_stime.tv_usec as u64,
        }
    }
    
    /// Fallback for non-Unix platforms
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn capture() -> Self {
        Self::default()
    }
    
    /// Compute delta from another stats snapshot
    pub fn delta_from(&self, earlier: &Self) -> Self {
        Self {
            major_faults: self.major_faults.saturating_sub(earlier.major_faults),
            minor_faults: self.minor_faults.saturating_sub(earlier.minor_faults),
            max_rss_kb: self.max_rss_kb,  // Max RSS is cumulative, not delta
            user_time_us: self.user_time_us.saturating_sub(earlier.user_time_us),
            sys_time_us: self.sys_time_us.saturating_sub(earlier.sys_time_us),
        }
    }
}

/// Telemetry capture for before/after comparison
#[derive(Debug)]
pub struct FaultTelemetry {
    /// Stats at operation start
    pub start_stats: FaultStats,
    /// Stats at operation end
    pub end_stats: Option<FaultStats>,
    /// Wall-clock start time
    pub start_time: Instant,
    /// Wall-clock end time
    pub end_time: Option<Instant>,
    /// Operation label
    pub label: String,
}

impl FaultTelemetry {
    /// Start capturing telemetry for an operation
    pub fn capture_start() -> Self {
        Self {
            start_stats: FaultStats::capture(),
            end_stats: None,
            start_time: Instant::now(),
            end_time: None,
            label: String::new(),
        }
    }
    
    /// Start capturing with a label
    pub fn capture_start_labeled(label: &str) -> Self {
        let mut t = Self::capture_start();
        t.label = label.to_string();
        t
    }
    
    /// Capture end state
    pub fn capture_end(&mut self) {
        self.end_stats = Some(FaultStats::capture());
        self.end_time = Some(Instant::now());
    }
    
    /// Get the delta stats (requires capture_end to have been called)
    pub fn delta(&self) -> Option<FaultStats> {
        self.end_stats.as_ref().map(|end| end.delta_from(&self.start_stats))
    }
    
    /// Get wall-clock duration
    pub fn duration(&self) -> Duration {
        self.end_time.unwrap_or_else(Instant::now).duration_since(self.start_time)
    }
    
    /// Print a summary to stderr
    pub fn print_summary(&self, label: &str) {
        let delta = self.delta().unwrap_or_default();
        let elapsed = self.duration();
        
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  Page Fault Telemetry: {:39} ║", label);
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  Wall time:      {:>10.3}s                                 ║", elapsed.as_secs_f64());
        eprintln!("║  Major faults:   {:>10}                                   ║", delta.major_faults);
        eprintln!("║  Minor faults:   {:>10}                                   ║", delta.minor_faults);
        eprintln!("║  Max RSS:        {:>10.1} MB                               ║", delta.max_rss_kb as f64 / 1024.0);
        eprintln!("║  User CPU:       {:>10.3}s                                 ║", delta.user_time_us as f64 / 1e6);
        eprintln!("║  Sys CPU:        {:>10.3}s                                 ║", delta.sys_time_us as f64 / 1e6);
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
        
        // Diagnostic interpretation
        let fault_rate = if elapsed.as_secs_f64() > 0.0 {
            delta.major_faults as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        
        if delta.major_faults > 1000 {
            eprintln!();
            eprintln!("⚠️  HIGH MAJOR FAULT COUNT: {} faults ({:.0}/s)", 
                delta.major_faults, fault_rate);
            eprintln!("   This indicates memory is not resident before access.");
            eprintln!("   Estimated fault overhead: {:.1}s @ 100μs/fault",
                delta.major_faults as f64 * 100e-6);
            eprintln!("   Consider: prefaulting, mlock, or direct read() instead of mmap");
        }
        
        let sys_ratio = if delta.user_time_us > 0 {
            delta.sys_time_us as f64 / delta.user_time_us as f64
        } else {
            0.0
        };
        
        if sys_ratio > 0.5 {
            eprintln!();
            eprintln!("⚠️  HIGH SYSTEM TIME RATIO: {:.1}% sys vs user",
                100.0 * sys_ratio / (1.0 + sys_ratio));
            eprintln!("   This may indicate excessive kernel involvement (page faults, syscalls).");
        }
    }
    
    /// Check if major fault count indicates a problem
    pub fn has_fault_problem(&self) -> bool {
        self.delta().map(|d| d.major_faults > 1000).unwrap_or(false)
    }
    
    /// Get estimated time spent in faults (assuming 100μs per major fault)
    pub fn estimated_fault_overhead_secs(&self) -> f64 {
        self.delta()
            .map(|d| d.major_faults as f64 * 100e-6)
            .unwrap_or(0.0)
    }
}

/// Telemetry gate for CI/benchmark validation
///
/// Use this to assert that major faults stay below a threshold.
/// Fails the build if page fault overhead is excessive.
#[derive(Debug)]
pub struct FaultGate {
    max_major_faults: u64,
    max_fault_ratio: f64,  // major_faults / N vectors
}

impl FaultGate {
    /// Create a gate with absolute fault limit
    pub fn with_max_faults(max_major_faults: u64) -> Self {
        Self {
            max_major_faults,
            max_fault_ratio: f64::MAX,
        }
    }
    
    /// Create a gate with per-vector fault ratio limit
    pub fn with_max_ratio(max_fault_ratio: f64) -> Self {
        Self {
            max_major_faults: u64::MAX,
            max_fault_ratio,
        }
    }
    
    /// Check if telemetry passes the gate
    pub fn check(&self, telemetry: &FaultTelemetry, n_vectors: usize) -> Result<(), String> {
        let delta = telemetry.delta().unwrap_or_default();
        
        if delta.major_faults > self.max_major_faults {
            return Err(format!(
                "FAULT GATE FAILED: {} major faults > {} max",
                delta.major_faults, self.max_major_faults
            ));
        }
        
        let ratio = delta.major_faults as f64 / n_vectors.max(1) as f64;
        if ratio > self.max_fault_ratio {
            return Err(format!(
                "FAULT GATE FAILED: {:.4} faults/vector > {:.4} max",
                ratio, self.max_fault_ratio
            ));
        }
        
        Ok(())
    }
}

/// Quick telemetry for a closure
pub fn with_telemetry<F, R>(label: &str, f: F) -> (R, FaultTelemetry)
where
    F: FnOnce() -> R,
{
    let mut telemetry = FaultTelemetry::capture_start_labeled(label);
    let result = f();
    telemetry.capture_end();
    (result, telemetry)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fault_stats_capture() {
        let stats = FaultStats::capture();
        // Just verify it doesn't crash
        println!("Captured stats: {:?}", stats);
    }
    
    #[test]
    fn test_telemetry() {
        let mut t = FaultTelemetry::capture_start();
        
        // Do some work that might cause page faults
        let mut v = Vec::with_capacity(1024 * 1024);
        for i in 0..1024 * 1024 {
            v.push(i as u8);
        }
        
        t.capture_end();
        
        let delta = t.delta().unwrap();
        println!("Delta: {:?}", delta);
        
        // Verify we captured something
        assert!(t.duration().as_nanos() > 0);
    }
}
