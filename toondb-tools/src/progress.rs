// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Licensed under the Apache License, Version 2.0

//! Progress reporting for bulk operations

use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// Progress reporter for bulk operations
pub struct ProgressReporter {
    bar: ProgressBar,
    start: Instant,
    last_update: Instant,
    vectors_processed: u64,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(total: u64, message: &str) -> Self {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {per_sec}")
                .unwrap()
                .progress_chars("█▓▒░"),
        );
        bar.set_message(message.to_string());
        
        let now = Instant::now();
        Self {
            bar,
            start: now,
            last_update: now,
            vectors_processed: 0,
        }
    }
    
    /// Create a spinner for indeterminate progress
    pub fn spinner(message: &str) -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::default_spinner()
                .template("{msg} {spinner:.green} [{elapsed_precise}] {pos} vectors ({per_sec})")
                .unwrap(),
        );
        bar.set_message(message.to_string());
        bar.enable_steady_tick(Duration::from_millis(100));
        
        let now = Instant::now();
        Self {
            bar,
            start: now,
            last_update: now,
            vectors_processed: 0,
        }
    }
    
    /// Update progress
    pub fn update(&mut self, count: u64) {
        self.vectors_processed += count;
        self.bar.set_position(self.vectors_processed);
        
        // Update rate every 100ms
        if self.last_update.elapsed() > Duration::from_millis(100) {
            let elapsed = self.start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                let rate = self.vectors_processed as f64 / elapsed;
                self.bar.set_message(format!("{:.0} vec/s", rate));
            }
            self.last_update = Instant::now();
        }
    }
    
    /// Set absolute progress
    pub fn set(&mut self, count: u64) {
        self.vectors_processed = count;
        self.bar.set_position(count);
    }
    
    /// Finish with a message
    pub fn finish(&self, message: &str) {
        self.bar.finish_with_message(message.to_string());
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    /// Get vectors per second
    pub fn rate(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.vectors_processed as f64 / elapsed
        } else {
            0.0
        }
    }
    
    /// Get total vectors processed
    pub fn total(&self) -> u64 {
        self.vectors_processed
    }
}

impl Drop for ProgressReporter {
    fn drop(&mut self) {
        if !self.bar.is_finished() {
            self.bar.abandon();
        }
    }
}

/// Summary statistics for a bulk operation
#[derive(Debug, Clone)]
pub struct BulkStats {
    /// Total vectors processed
    pub vectors: u64,
    /// Elapsed time
    pub elapsed: Duration,
    /// Vectors per second
    pub rate: f64,
    /// Peak memory (if available)
    pub peak_memory_mb: Option<f64>,
    /// Output size in bytes
    pub output_bytes: Option<u64>,
}

impl BulkStats {
    /// Create stats from a progress reporter
    pub fn from_progress(progress: &ProgressReporter) -> Self {
        Self {
            vectors: progress.total(),
            elapsed: progress.elapsed(),
            rate: progress.rate(),
            peak_memory_mb: None,
            output_bytes: None,
        }
    }
    
    /// Print formatted summary
    pub fn print_summary(&self) {
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║                    BULK OPERATION SUMMARY                    ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  Vectors:     {:>12}                                   ║", self.vectors);
        eprintln!("║  Time:        {:>12.2}s                                  ║", self.elapsed.as_secs_f64());
        eprintln!("║  Throughput:  {:>12.0} vec/s                             ║", self.rate);
        if let Some(mem) = self.peak_memory_mb {
            eprintln!("║  Peak Memory: {:>12.1} MB                                ║", mem);
        }
        if let Some(size) = self.output_bytes {
            let mb = size as f64 / 1024.0 / 1024.0;
            eprintln!("║  Output Size: {:>12.1} MB                                ║", mb);
        }
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }
}
