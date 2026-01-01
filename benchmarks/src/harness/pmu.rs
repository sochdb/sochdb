// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! PMU/Performance Counter integration for Linux systems.
//!
//! This module provides access to hardware performance counters via the
//! Linux perf_event subsystem. On non-Linux systems, it provides a stub
//! implementation that returns None for all metrics.

use serde::{Deserialize, Serialize};

/// PMU counter types to collect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PmuCounter {
    Cycles,
    Instructions,
    BranchInstructions,
    BranchMisses,
    L1DLoads,
    L1DLoadMisses,
    LLCLoads,
    LLCLoadMisses,
    DTLBLoadMisses,
    ContextSwitches,
    PageFaults,
}

/// PMU snapshot containing all collected counter values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PmuSnapshot {
    pub cycles: Option<u64>,
    pub instructions: Option<u64>,
    pub branch_instructions: Option<u64>,
    pub branch_misses: Option<u64>,
    pub l1d_loads: Option<u64>,
    pub l1d_load_misses: Option<u64>,
    pub llc_loads: Option<u64>,
    pub llc_load_misses: Option<u64>,
    pub dtlb_load_misses: Option<u64>,
    pub context_switches: Option<u64>,
    pub page_faults: Option<u64>,
}

impl PmuSnapshot {
    /// Calculate instructions per cycle (IPC).
    pub fn ipc(&self) -> Option<f64> {
        match (self.cycles, self.instructions) {
            (Some(c), Some(i)) if c > 0 => Some(i as f64 / c as f64),
            _ => None,
        }
    }

    /// Calculate branch miss rate.
    pub fn branch_miss_rate(&self) -> Option<f64> {
        match (self.branch_instructions, self.branch_misses) {
            (Some(i), Some(m)) if i > 0 => Some(m as f64 / i as f64),
            _ => None,
        }
    }

    /// Calculate L1D cache miss rate.
    pub fn l1d_miss_rate(&self) -> Option<f64> {
        match (self.l1d_loads, self.l1d_load_misses) {
            (Some(l), Some(m)) if l > 0 => Some(m as f64 / l as f64),
            _ => None,
        }
    }

    /// Calculate LLC cache miss rate.
    pub fn llc_miss_rate(&self) -> Option<f64> {
        match (self.llc_loads, self.llc_load_misses) {
            (Some(l), Some(m)) if l > 0 => Some(m as f64 / l as f64),
            _ => None,
        }
    }
}

// ============================================================================
// Linux Implementation
// ============================================================================

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use std::time::Duration;

    /// PMU collector using Linux perf stat.
    pub struct PmuCollector {
        counters: Vec<PmuCounter>,
        perf_process: Option<std::process::Child>,
        output_file: Option<String>,
    }

    impl PmuCollector {
        /// Create a new PMU collector with default counters.
        pub fn new() -> Self {
            Self {
                counters: vec![
                    PmuCounter::Cycles,
                    PmuCounter::Instructions,
                    PmuCounter::BranchInstructions,
                    PmuCounter::BranchMisses,
                    PmuCounter::L1DLoads,
                    PmuCounter::L1DLoadMisses,
                    PmuCounter::ContextSwitches,
                    PmuCounter::PageFaults,
                ],
                perf_process: None,
                output_file: None,
            }
        }

        /// Check if perf is available.
        pub fn is_available() -> bool {
            Command::new("perf")
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        }

        /// Start collecting counters for the current process.
        pub fn start(&mut self) -> Result<(), String> {
            if !Self::is_available() {
                return Err("perf is not available".to_string());
            }

            let pid = std::process::id();
            let output_file = format!("/tmp/perf_stat_{}.txt", pid);
            
            let events = self.counters.iter()
                .map(|c| counter_to_event(*c))
                .collect::<Vec<_>>()
                .join(",");

            let child = Command::new("perf")
                .args([
                    "stat",
                    "-e", &events,
                    "-p", &pid.to_string(),
                    "-o", &output_file,
                    "--", "sleep", "infinity"
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("Failed to start perf: {}", e))?;

            self.perf_process = Some(child);
            self.output_file = Some(output_file);

            // Give perf time to attach
            std::thread::sleep(Duration::from_millis(100));

            Ok(())
        }

        /// Stop collecting and return the snapshot.
        pub fn stop(&mut self) -> Result<PmuSnapshot, String> {
            // Kill perf process
            if let Some(mut proc) = self.perf_process.take() {
                let _ = proc.kill();
                let _ = proc.wait();
            }

            // Parse output file
            let output_file = self.output_file.take()
                .ok_or("No output file")?;

            self.parse_perf_output(&output_file)
        }

        fn parse_perf_output(&self, path: &str) -> Result<PmuSnapshot, String> {
            let file = File::open(path)
                .map_err(|e| format!("Failed to open perf output: {}", e))?;
            
            let reader = BufReader::new(file);
            let mut snapshot = PmuSnapshot::default();

            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
                let parts: Vec<&str> = line.split_whitespace().collect();
                
                if parts.len() < 2 {
                    continue;
                }

                // Parse "123,456 event-name" format
                if let Ok(value) = parts[0].replace(",", "").parse::<u64>() {
                    let event = parts[1];
                    match event {
                        "cycles" => snapshot.cycles = Some(value),
                        "instructions" => snapshot.instructions = Some(value),
                        "branch-instructions" => snapshot.branch_instructions = Some(value),
                        "branch-misses" => snapshot.branch_misses = Some(value),
                        "L1-dcache-loads" => snapshot.l1d_loads = Some(value),
                        "L1-dcache-load-misses" => snapshot.l1d_load_misses = Some(value),
                        "LLC-loads" => snapshot.llc_loads = Some(value),
                        "LLC-load-misses" => snapshot.llc_load_misses = Some(value),
                        "dTLB-load-misses" => snapshot.dtlb_load_misses = Some(value),
                        "context-switches" | "cs" => snapshot.context_switches = Some(value),
                        "page-faults" | "faults" => snapshot.page_faults = Some(value),
                        _ => {}
                    }
                }
            }

            // Clean up temp file
            let _ = std::fs::remove_file(path);

            Ok(snapshot)
        }
    }

    impl Default for PmuCollector {
        fn default() -> Self {
            Self::new()
        }
    }

    fn counter_to_event(counter: PmuCounter) -> &'static str {
        match counter {
            PmuCounter::Cycles => "cycles",
            PmuCounter::Instructions => "instructions",
            PmuCounter::BranchInstructions => "branch-instructions",
            PmuCounter::BranchMisses => "branch-misses",
            PmuCounter::L1DLoads => "L1-dcache-loads",
            PmuCounter::L1DLoadMisses => "L1-dcache-load-misses",
            PmuCounter::LLCLoads => "LLC-loads",
            PmuCounter::LLCLoadMisses => "LLC-load-misses",
            PmuCounter::DTLBLoadMisses => "dTLB-load-misses",
            PmuCounter::ContextSwitches => "context-switches",
            PmuCounter::PageFaults => "page-faults",
        }
    }

    /// Run a closure with PMU collection.
    pub fn with_pmu<F, R>(f: F) -> (R, Option<PmuSnapshot>)
    where
        F: FnOnce() -> R,
    {
        let mut collector = PmuCollector::new();

        // Try to start collection
        let started = collector.start().is_ok();

        // Run the closure
        let result = f();

        // Get snapshot if we started successfully
        let snapshot = if started {
            collector.stop().ok()
        } else {
            None
        };

        (result, snapshot)
    }
}

// ============================================================================
// Non-Linux Stub Implementation
// ============================================================================

#[cfg(not(target_os = "linux"))]
mod stub {
    use super::*;

    /// Stub PMU collector for non-Linux systems.
    pub struct PmuCollector;

    impl PmuCollector {
        pub fn new() -> Self {
            Self
        }

        pub fn is_available() -> bool {
            false
        }

        pub fn start(&mut self) -> Result<(), String> {
            Err("PMU counters not available on this platform".to_string())
        }

        pub fn stop(&mut self) -> Result<PmuSnapshot, String> {
            Err("PMU counters not available on this platform".to_string())
        }
    }

    impl Default for PmuCollector {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Stub: Run a closure without PMU collection.
    pub fn with_pmu<F, R>(f: F) -> (R, Option<PmuSnapshot>)
    where
        F: FnOnce() -> R,
    {
        (f(), None)
    }
}

// Re-export the appropriate implementation
#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(not(target_os = "linux"))]
pub use stub::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pmu_snapshot_calculations() {
        let snapshot = PmuSnapshot {
            cycles: Some(1_000_000),
            instructions: Some(2_000_000),
            branch_instructions: Some(100_000),
            branch_misses: Some(5_000),
            l1d_loads: Some(500_000),
            l1d_load_misses: Some(10_000),
            ..Default::default()
        };

        assert!((snapshot.ipc().unwrap() - 2.0).abs() < 0.001);
        assert!((snapshot.branch_miss_rate().unwrap() - 0.05).abs() < 0.001);
        assert!((snapshot.l1d_miss_rate().unwrap() - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_pmu_collector_create() {
        let _collector = PmuCollector::new();
        // Just verify it doesn't panic
    }
}
