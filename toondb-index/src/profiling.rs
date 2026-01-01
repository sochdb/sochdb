// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! End-to-End Profiling for HNSW Operations
//!
//! This module provides detailed profiling of HNSW operations that can be
//! enabled via the `TOONDB_PROFILING=1` environment variable.
//!
//! ## Usage
//!
//! ```bash
//! TOONDB_PROFILING=1 python my_script.py
//! ```
//!
//! ## Output
//!
//! When enabled, profiling data is written to:
//! - `TOONDB_PROFILE_FILE` environment variable (default: `/tmp/toondb_profile.json`)
//!
//! ## Profiled Operations
//!
//! - FFI boundary crossing
//! - Memory allocation
//! - ID conversion
//! - HNSW insert (single and batch)
//! - Graph operations (layer assignment, neighbor selection, connection)
//! - Distance computations
//! - Search operations

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};
use std::time::Instant;
use std::io::Write;

// ============================================================================
// Configuration
// ============================================================================

static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILING_INIT: Once = Once::new();

/// Check if profiling is enabled (via TOONDB_PROFILING env var)
#[inline]
pub fn is_profiling_enabled() -> bool {
    PROFILING_INIT.call_once(|| {
        let enabled = std::env::var("TOONDB_PROFILING")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        PROFILING_ENABLED.store(enabled, Ordering::Relaxed);
        
        if enabled {
            eprintln!("[TOONDB] Profiling enabled. Output: {}", get_profile_path());
        }
    });
    PROFILING_ENABLED.load(Ordering::Relaxed)
}

fn get_profile_path() -> String {
    std::env::var("TOONDB_PROFILE_FILE")
        .unwrap_or_else(|_| "/tmp/toondb_profile.json".to_string())
}

// ============================================================================
// Timing Structures
// ============================================================================

/// High-resolution timer
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    /// Start a new timer
    #[inline]
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }
    
    /// Stop timer and record elapsed time in nanoseconds
    #[inline]
    pub fn stop(self) -> u64 {
        let elapsed_ns = self.start.elapsed().as_nanos() as u64;
        if is_profiling_enabled() {
            PROFILE_COLLECTOR.record(&self.name, elapsed_ns);
        }
        elapsed_ns
    }
    
    /// Stop timer and record with count (for batch operations)
    #[inline]
    pub fn stop_with_count(self, count: usize) -> u64 {
        let elapsed_ns = self.start.elapsed().as_nanos() as u64;
        if is_profiling_enabled() {
            PROFILE_COLLECTOR.record_with_count(&self.name, elapsed_ns, count);
        }
        elapsed_ns
    }
}

/// Macro for timed sections (no-op when profiling disabled)
#[macro_export]
macro_rules! profile_section {
    ($name:expr, $code:block) => {{
        if $crate::profiling::is_profiling_enabled() {
            let timer = $crate::profiling::Timer::start($name);
            let result = $code;
            timer.stop();
            result
        } else {
            $code
        }
    }};
}

/// Macro for batch-timed sections
#[macro_export]
macro_rules! profile_batch {
    ($name:expr, $count:expr, $code:block) => {{
        if $crate::profiling::is_profiling_enabled() {
            let timer = $crate::profiling::Timer::start($name);
            let result = $code;
            timer.stop_with_count($count);
            result
        } else {
            $code
        }
    }};
}

// ============================================================================
// Profile Data Collection
// ============================================================================

/// Statistics for a single operation type
#[derive(Debug, Default, Clone)]
pub struct OpStats {
    pub count: u64,
    pub total_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub item_count: u64,  // For batch operations
}

impl OpStats {
    fn record(&mut self, elapsed_ns: u64, count: usize) {
        self.count += 1;
        self.total_ns += elapsed_ns;
        self.item_count += count as u64;
        
        if self.min_ns == 0 || elapsed_ns < self.min_ns {
            self.min_ns = elapsed_ns;
        }
        if elapsed_ns > self.max_ns {
            self.max_ns = elapsed_ns;
        }
    }
    
    pub fn mean_ns(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64
        }
    }
    
    pub fn per_item_ns(&self) -> f64 {
        if self.item_count == 0 {
            self.mean_ns()
        } else {
            self.total_ns as f64 / self.item_count as f64
        }
    }
}

/// Global profile collector
pub struct ProfileCollector {
    stats: Mutex<HashMap<String, OpStats>>,
    start_time: Instant,
}

impl ProfileCollector {
    fn new() -> Self {
        Self {
            stats: Mutex::new(HashMap::new()),
            start_time: Instant::now(),
        }
    }
    
    /// Record a timing
    pub fn record(&self, name: &str, elapsed_ns: u64) {
        self.record_with_count(name, elapsed_ns, 1);
    }
    
    /// Record a timing with item count
    pub fn record_with_count(&self, name: &str, elapsed_ns: u64, count: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.entry(name.to_string())
                .or_default()
                .record(elapsed_ns, count);
        }
    }
    
    /// Increment a counter (no timing)
    pub fn increment(&self, name: &str, count: u64) {
        if let Ok(mut stats) = self.stats.lock() {
            let entry = stats.entry(name.to_string()).or_default();
            entry.count += count;
            entry.item_count += count;
        }
    }
    
    /// Get all stats
    pub fn get_stats(&self) -> HashMap<String, OpStats> {
        self.stats.lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
    
    /// Write profile to JSON file
    pub fn write_profile(&self) -> std::io::Result<()> {
        let stats = self.get_stats();
        let elapsed_total = self.start_time.elapsed();
        
        let path = get_profile_path();
        let mut file = std::fs::File::create(&path)?;
        
        writeln!(file, "{{")?;
        writeln!(file, "  \"total_elapsed_ms\": {:.2},", elapsed_total.as_secs_f64() * 1000.0)?;
        writeln!(file, "  \"operations\": {{")?;
        
        let mut entries: Vec<_> = stats.iter().collect();
        entries.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns)); // Sort by total time
        
        for (i, (name, op)) in entries.iter().enumerate() {
            let comma = if i < entries.len() - 1 { "," } else { "" };
            writeln!(file, "    \"{}\": {{", name)?;
            writeln!(file, "      \"count\": {},", op.count)?;
            writeln!(file, "      \"total_ms\": {:.3},", op.total_ns as f64 / 1_000_000.0)?;
            writeln!(file, "      \"mean_us\": {:.2},", op.mean_ns() / 1_000.0)?;
            writeln!(file, "      \"min_us\": {:.2},", op.min_ns as f64 / 1_000.0)?;
            writeln!(file, "      \"max_us\": {:.2},", op.max_ns as f64 / 1_000.0)?;
            if op.item_count > 0 && op.item_count != op.count {
                writeln!(file, "      \"item_count\": {},", op.item_count)?;
                writeln!(file, "      \"per_item_us\": {:.2}", op.per_item_ns() / 1_000.0)?;
            } else {
                writeln!(file, "      \"item_count\": {}", op.item_count)?;
            }
            writeln!(file, "    }}{}", comma)?;
        }
        
        writeln!(file, "  }}")?;
        writeln!(file, "}}")?;
        
        eprintln!("[TOONDB] Profile written to: {}", path);
        Ok(())
    }
    
    /// Print summary to stderr
    pub fn print_summary(&self) {
        let stats = self.get_stats();
        let elapsed_total = self.start_time.elapsed();
        
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║                     ToonDB HNSW Profiling Summary                        ║");
        eprintln!("╠══════════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ Total elapsed: {:>8.2} ms                                              ║", 
                  elapsed_total.as_secs_f64() * 1000.0);
        eprintln!("╠════════════════════════════════╦═════════╦══════════╦═════════╦══════════╣");
        eprintln!("║ Operation                      ║  Count  ║ Total ms ║ Mean µs ║ Items    ║");
        eprintln!("╠════════════════════════════════╬═════════╬══════════╬═════════╬══════════╣");
        
        let mut entries: Vec<_> = stats.iter().collect();
        entries.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
        
        for (name, op) in entries.iter().take(15) {
            let name_truncated: String = if name.len() > 30 {
                format!("{}...", &name[..27])
            } else {
                name.to_string()
            };
            eprintln!("║ {:30} ║ {:>7} ║ {:>8.2} ║ {:>7.1} ║ {:>8} ║",
                     name_truncated,
                     op.count,
                     op.total_ns as f64 / 1_000_000.0,
                     op.mean_ns() / 1_000.0,
                     op.item_count);
        }
        
        eprintln!("╚════════════════════════════════╩═════════╩══════════╩═════════╩══════════╝\n");
    }
}

// Global singleton
lazy_static::lazy_static! {
    pub static ref PROFILE_COLLECTOR: ProfileCollector = ProfileCollector::new();
}

// ============================================================================
// FFI Functions for External Access
// ============================================================================

/// Enable profiling programmatically
#[unsafe(no_mangle)]
pub extern "C" fn toondb_profiling_enable() {
    PROFILING_ENABLED.store(true, Ordering::Relaxed);
}

/// Disable profiling
#[unsafe(no_mangle)]
pub extern "C" fn toondb_profiling_disable() {
    PROFILING_ENABLED.store(false, Ordering::Relaxed);
}

/// Write profile to file and print summary
#[unsafe(no_mangle)]
pub extern "C" fn toondb_profiling_dump() {
    if is_profiling_enabled() {
        PROFILE_COLLECTOR.print_summary();
        let _ = PROFILE_COLLECTOR.write_profile();
    }
}

/// Record a custom timing (for external profiling)
#[unsafe(no_mangle)]
pub extern "C" fn toondb_profiling_record(
    name: *const std::os::raw::c_char,
    elapsed_ns: u64,
    count: usize,
) {
    if !is_profiling_enabled() || name.is_null() {
        return;
    }
    
    unsafe {
        if let Ok(name_str) = std::ffi::CStr::from_ptr(name).to_str() {
            PROFILE_COLLECTOR.record_with_count(name_str, elapsed_ns, count);
        }
    }
}

// ============================================================================
// Convenience Functions for Internal Use
// ============================================================================

/// Profile an FFI call
#[inline]
pub fn profile_ffi<T>(name: &str, f: impl FnOnce() -> T) -> T {
    if is_profiling_enabled() {
        let timer = Timer::start(format!("ffi.{}", name));
        let result = f();
        timer.stop();
        result
    } else {
        f()
    }
}

/// Profile a batch FFI call
#[inline]
pub fn profile_ffi_batch<T>(name: &str, count: usize, f: impl FnOnce() -> T) -> T {
    if is_profiling_enabled() {
        let timer = Timer::start(format!("ffi.{}", name));
        let result = f();
        timer.stop_with_count(count);
        result
    } else {
        f()
    }
}

/// Profile HNSW operation
#[inline]
pub fn profile_hnsw<T>(name: &str, f: impl FnOnce() -> T) -> T {
    if is_profiling_enabled() {
        let timer = Timer::start(format!("hnsw.{}", name));
        let result = f();
        timer.stop();
        result
    } else {
        f()
    }
}

/// Profile batch HNSW operation
#[inline]
pub fn profile_hnsw_batch<T>(name: &str, count: usize, f: impl FnOnce() -> T) -> T {
    if is_profiling_enabled() {
        let timer = Timer::start(format!("hnsw.{}", name));
        let result = f();
        timer.stop_with_count(count);
        result
    } else {
        f()
    }
}

// ============================================================================
// Specific Operation Profilers
// ============================================================================

/// Profile ID conversion (u64[] -> u128[])
#[inline]
pub fn profile_id_conversion<T>(count: usize, f: impl FnOnce() -> T) -> T {
    profile_ffi_batch("id_conversion", count, f)
}

/// Profile memory allocation
#[inline]
pub fn profile_allocation<T>(name: &str, f: impl FnOnce() -> T) -> T {
    profile_hnsw(&format!("alloc.{}", name), f)
}

/// Profile distance computation
#[inline]
pub fn profile_distance<T>(f: impl FnOnce() -> T) -> T {
    if is_profiling_enabled() {
        let timer = Timer::start("hnsw.distance_compute");
        let result = f();
        timer.stop();
        PROFILE_COLLECTOR.increment("hnsw.distance_count", 1);
        result
    } else {
        f()
    }
}

/// Profile graph operations
pub mod graph {
    use super::*;
    
    #[inline]
    pub fn layer_assignment<T>(f: impl FnOnce() -> T) -> T {
        profile_hnsw("graph.layer_assignment", f)
    }
    
    #[inline]
    pub fn neighbor_selection<T>(layer: usize, f: impl FnOnce() -> T) -> T {
        if is_profiling_enabled() {
            let timer = Timer::start(format!("hnsw.graph.neighbor_select_L{}", layer));
            let result = f();
            timer.stop();
            result
        } else {
            f()
        }
    }
    
    #[inline]
    pub fn add_connection<T>(f: impl FnOnce() -> T) -> T {
        profile_hnsw("graph.add_connection", f)
    }
    
    #[inline]
    pub fn update_entry_point<T>(f: impl FnOnce() -> T) -> T {
        profile_hnsw("graph.update_entry_point", f)
    }
}

/// Profile search operations
pub mod search {
    use super::*;
    
    #[inline]
    pub fn layer_search<T>(layer: usize, f: impl FnOnce() -> T) -> T {
        if is_profiling_enabled() {
            let timer = Timer::start(format!("hnsw.search.layer_L{}", layer));
            let result = f();
            timer.stop();
            result
        } else {
            f()
        }
    }
    
    #[inline]
    pub fn candidate_selection<T>(f: impl FnOnce() -> T) -> T {
        profile_hnsw("search.candidate_selection", f)
    }
    
    #[inline]
    pub fn result_collection<T>(f: impl FnOnce() -> T) -> T {
        profile_hnsw("search.result_collection", f)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timer() {
        let timer = Timer::start("test_op");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        
        // Should be at least 10ms = 10_000_000 ns
        assert!(elapsed >= 9_000_000, "Expected >= 9ms, got {}ns", elapsed);
    }
    
    #[test]
    fn test_op_stats() {
        let mut stats = OpStats::default();
        stats.record(1000, 1);
        stats.record(2000, 1);
        stats.record(3000, 1);
        
        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_ns, 6000);
        assert_eq!(stats.min_ns, 1000);
        assert_eq!(stats.max_ns, 3000);
        assert!((stats.mean_ns() - 2000.0).abs() < 0.01);
    }
    
    #[test]
    fn test_batch_stats() {
        let mut stats = OpStats::default();
        stats.record(10_000, 100);  // 10µs for 100 items
        
        assert_eq!(stats.count, 1);
        assert_eq!(stats.item_count, 100);
        assert!((stats.per_item_ns() - 100.0).abs() < 0.01);  // 100ns per item
    }
}
