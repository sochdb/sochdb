//! Performance Guardrails
//!
//! This module provides runtime checks and warnings to prevent silent
//! performance regressions.
//!
//! ## Guardrails
//!
//! 1. **Safe Mode Warning**: Loud warning when TOONDB_BATCH_SAFE_MODE=1
//! 2. **Contiguity Check**: Verify vectors are in contiguous memory
//! 3. **Insert Path Logging**: Log which insertion primitive is selected
//! 4. **Throughput Tripwire**: Fail fast if throughput is unexpectedly low

use std::sync::Once;
use std::time::Duration;

// =============================================================================
// Safe Mode Detection
// =============================================================================

static SAFE_MODE_WARNING: Once = Once::new();
static INSERT_PATH_LOG: Once = Once::new();

/// Check if safe mode is enabled and emit warning.
pub fn check_safe_mode() -> bool {
    let enabled = std::env::var("TOONDB_BATCH_SAFE_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    
    if enabled {
        SAFE_MODE_WARNING.call_once(|| {
            eprintln!(
                "\n\
                ╔══════════════════════════════════════════════════════════════╗\n\
                ║  WARNING: TOONDB_BATCH_SAFE_MODE=1 is active                 ║\n\
                ║  Batch inserts are running 10-100× SLOWER than normal.       ║\n\
                ║  Unset this variable for production/benchmarking.            ║\n\
                ╚══════════════════════════════════════════════════════════════╝\n"
            );
        });
    }
    enabled
}

/// Check if performance debug logging is enabled.
pub fn is_debug_enabled() -> bool {
    std::env::var("TOONDB_DEBUG_PERF").is_ok()
        || std::env::var("TOONDB_DEBUG_INSERT").is_ok()
}

// =============================================================================
// Insert Path Logging
// =============================================================================

/// Log which insertion path is being used.
pub fn log_insert_path(path: &str, batch_size: usize, dimension: usize) {
    if is_debug_enabled() {
        INSERT_PATH_LOG.call_once(|| {
            eprintln!(
                "[toondb-perf] Insert path: {} | batch={} | dim={}",
                path, batch_size, dimension
            );
        });
    }
}

/// Log detailed batch info (only in debug mode).
pub fn log_batch_detail(batch_idx: usize, batch_size: usize, elapsed: Duration) {
    if std::env::var("TOONDB_DEBUG_BATCH").is_ok() {
        let rate = batch_size as f64 / elapsed.as_secs_f64();
        eprintln!(
            "[toondb-perf] Batch {}: {} vectors in {:.2}ms ({:.0} vec/s)",
            batch_idx, batch_size, elapsed.as_millis(), rate
        );
    }
}

// =============================================================================
// Throughput Tripwire
// =============================================================================

/// Minimum expected throughput thresholds by dimension.
///
/// If throughput falls below these values, something is wrong.
/// These are conservative - actual throughput should be 2-10x higher.
pub fn min_expected_throughput(dimension: usize) -> f64 {
    // Conservative minimums based on empirical data
    match dimension {
        d if d <= 128 => 5000.0,   // 5K vec/s minimum for small dims
        d if d <= 384 => 3000.0,   // 3K vec/s for embedding dims
        d if d <= 768 => 1500.0,   // 1.5K vec/s for large embeddings
        d if d <= 1536 => 750.0,   // 750 vec/s for very large embeddings
        _ => 300.0,               // 300 vec/s for huge dimensions
    }
}

/// Check if throughput is within expected range.
///
/// Returns an error message if throughput is suspiciously low.
pub fn check_throughput(
    vectors_inserted: usize,
    elapsed: Duration,
    dimension: usize,
) -> Option<String> {
    if vectors_inserted < 100 {
        return None; // Not enough data for meaningful check
    }
    
    let rate = vectors_inserted as f64 / elapsed.as_secs_f64();
    let min_rate = min_expected_throughput(dimension);
    
    if rate < min_rate {
        Some(format!(
            "PERFORMANCE WARNING: {:.0} vec/s is below minimum expected {:.0} vec/s for {}D vectors.\n\
             Possible causes:\n\
             - TOONDB_BATCH_SAFE_MODE=1 is set\n\
             - Vectors are not contiguous in memory\n\
             - Using wrong insertion primitive\n\
             - System under memory pressure\n\
             Set TOONDB_DEBUG_PERF=1 for more details.",
            rate, min_rate, dimension
        ))
    } else {
        None
    }
}

// =============================================================================
// Contiguity Check
// =============================================================================

/// Verify that a vector slice is likely contiguous.
///
/// This is a heuristic check - we verify that the slice length
/// matches expected size and pointers are aligned.
pub fn verify_contiguous(vectors: &[f32], num_vectors: usize, dimension: usize) -> bool {
    let expected_len = num_vectors * dimension;
    if vectors.len() != expected_len {
        return false;
    }
    
    // Check alignment (f32 should be 4-byte aligned)
    let ptr = vectors.as_ptr() as usize;
    if ptr % std::mem::align_of::<f32>() != 0 {
        return false;
    }
    
    true
}

// =============================================================================
// Performance Summary
// =============================================================================

/// Print a performance summary with warnings if needed.
pub fn print_perf_summary(
    vectors: usize,
    dimension: usize,
    elapsed: Duration,
    output_bytes: Option<u64>,
) {
    let rate = vectors as f64 / elapsed.as_secs_f64();
    
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                    Build Summary                             ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║  Vectors:    {:>12}                                    ║", vectors);
    eprintln!("║  Dimension:  {:>12}                                    ║", dimension);
    eprintln!("║  Time:       {:>12.2}s                                   ║", elapsed.as_secs_f64());
    eprintln!("║  Throughput: {:>12.0} vec/s                              ║", rate);
    if let Some(bytes) = output_bytes {
        eprintln!("║  Output:     {:>12.1} MB                                 ║", bytes as f64 / 1024.0 / 1024.0);
    }
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    
    // Check for performance issues
    if let Some(warning) = check_throughput(vectors, elapsed, dimension) {
        eprintln!();
        eprintln!("⚠️  {}", warning);
    }
    
    // Report if safe mode was active
    if check_safe_mode() {
        eprintln!();
        eprintln!("⚠️  Note: TOONDB_BATCH_SAFE_MODE was active during this build.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_throughput_check() {
        // Good throughput
        assert!(check_throughput(10000, Duration::from_secs(1), 768).is_none());
        
        // Bad throughput
        let warning = check_throughput(1000, Duration::from_secs(10), 768);
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("PERFORMANCE WARNING"));
    }
    
    #[test]
    fn test_contiguity() {
        let vectors = vec![0.0f32; 1000 * 128];
        assert!(verify_contiguous(&vectors, 1000, 128));
        assert!(!verify_contiguous(&vectors, 1000, 64)); // Wrong dimension
    }
}
