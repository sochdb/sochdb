// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Environment information and reproducibility controls.

use serde::{Deserialize, Serialize};
use std::process::Command;

/// Comprehensive environment information for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Operating system
    pub os: String,
    /// OS version/kernel
    pub os_version: String,
    /// CPU model name
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// CPU microcode version (Linux only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_microcode: Option<String>,
    /// RAM size in GB
    pub ram_gb: u64,
    /// RAM speed (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_speed: Option<String>,
    /// Storage type (NVMe, SSD, HDD)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_type: Option<String>,
    /// CPU scaling governor
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_governor: Option<String>,
    /// Turbo boost enabled
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turbo_enabled: Option<bool>,
    /// Hugepages configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hugepages: Option<String>,
    /// Transparent Huge Pages setting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thp_setting: Option<String>,
    /// Rust compiler version
    pub rustc_version: String,
    /// Git commit SHA
    pub git_sha: String,
    /// Git dirty status
    pub git_dirty: bool,
    /// Build profile (debug/release)
    pub build_profile: String,
    /// Enabled features
    pub features: Vec<String>,
    /// Is this a reproducible environment?
    pub is_locked: bool,
    /// Warnings about non-reproducible settings
    pub warnings: Vec<String>,
}

impl EnvironmentInfo {
    /// Collect comprehensive environment information.
    pub fn collect() -> Self {
        let mut warnings = Vec::new();
        let mut is_locked = true;

        // OS info
        let os = std::env::consts::OS.to_string();
        let os_version = get_os_version();

        // CPU info
        let cpu_model = get_cpu_model();
        let cpu_cores = std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(1);
        let cpu_microcode = get_cpu_microcode();

        // RAM info
        let ram_gb = get_ram_gb();
        let ram_speed = None; // Platform-specific, complex to detect

        // Storage info
        let storage_type = None; // Platform-specific

        // CPU governor (Linux only)
        let cpu_governor = get_cpu_governor();
        if let Some(ref gov) = cpu_governor {
            if gov != "performance" {
                warnings.push(format!(
                    "CPU governor is '{}', should be 'performance' for reproducible results",
                    gov
                ));
                is_locked = false;
            }
        }

        // Turbo boost (Linux only)
        let turbo_enabled = get_turbo_enabled();
        if turbo_enabled == Some(true) {
            warnings.push("Turbo boost is enabled, may cause variance".to_string());
            // Note: turbo enabled is not necessarily bad, just variance
        }

        // Hugepages
        let hugepages = get_hugepages_info();
        let thp_setting = get_thp_setting();

        // Rust/build info
        let rustc_version = get_rustc_version();
        let (git_sha, git_dirty) = get_git_info();
        if git_dirty {
            warnings.push("Git working directory is dirty".to_string());
        }

        let build_profile = if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        };

        if build_profile == "debug" {
            warnings.push("Running in debug mode - results not meaningful".to_string());
            is_locked = false;
        }

        let mut features = Vec::new();
        if cfg!(feature = "jemalloc") {
            features.push("jemalloc".to_string());
        }

        Self {
            os,
            os_version,
            cpu_model,
            cpu_cores,
            cpu_microcode,
            ram_gb,
            ram_speed,
            storage_type,
            cpu_governor,
            turbo_enabled,
            hugepages,
            thp_setting,
            rustc_version,
            git_sha,
            git_dirty,
            build_profile,
            features,
            is_locked,
            warnings,
        }
    }

    /// Generate a fingerprint string for comparison.
    pub fn fingerprint(&self) -> String {
        format!(
            "{}-{}-{}-{}-{}",
            self.os,
            self.cpu_model.replace(' ', "_"),
            self.cpu_cores,
            self.ram_gb,
            self.git_sha,
        )
    }
}

fn get_os_version() -> String {
    if cfg!(target_os = "macos") {
        Command::new("sw_vers")
            .args(["-productVersion"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    } else if cfg!(target_os = "linux") {
        Command::new("uname")
            .args(["-r"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        "unknown".to_string()
    }
}

fn get_cpu_model() -> String {
    if cfg!(target_os = "macos") {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    } else if cfg!(target_os = "linux") {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        "unknown".to_string()
    }
}

fn get_cpu_microcode() -> Option<String> {
    if cfg!(target_os = "linux") {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("microcode"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
    } else {
        None
    }
}

fn get_ram_gb() -> u64 {
    if cfg!(target_os = "macos") {
        Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|b| b / (1024 * 1024 * 1024))
            .unwrap_or(0)
    } else if cfg!(target_os = "linux") {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|n| n.parse::<u64>().ok())
                    })
            })
            .map(|kb| kb / (1024 * 1024))
            .unwrap_or(0)
    } else {
        0
    }
}

fn get_cpu_governor() -> Option<String> {
    if cfg!(target_os = "linux") {
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

fn get_turbo_enabled() -> Option<bool> {
    if cfg!(target_os = "linux") {
        // Intel
        if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/intel_pstate/no_turbo") {
            return Some(content.trim() == "0");
        }
        // AMD (boost)
        if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/cpufreq/boost") {
            return Some(content.trim() == "1");
        }
    }
    None
}

fn get_hugepages_info() -> Option<String> {
    if cfg!(target_os = "linux") {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                let total = s.lines()
                    .find(|l| l.starts_with("HugePages_Total"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|n| n.parse::<u64>().ok())
                    .unwrap_or(0);
                let size = s.lines()
                    .find(|l| l.starts_with("Hugepagesize"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|n| n.parse::<u64>().ok())
                    .unwrap_or(0);
                if total > 0 {
                    Some(format!("{}x{}kB", total, size))
                } else {
                    Some("disabled".to_string())
                }
            })
    } else {
        None
    }
}

fn get_thp_setting() -> Option<String> {
    if cfg!(target_os = "linux") {
        std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
            .ok()
            .and_then(|s| {
                // Format: "always [madvise] never" - find the bracketed one
                s.split_whitespace()
                    .find(|w| w.starts_with('['))
                    .map(|w| w.trim_matches(|c| c == '[' || c == ']').to_string())
            })
    } else {
        None
    }
}

fn get_rustc_version() -> String {
    Command::new("rustc")
        .args(["--version"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn get_git_info() -> (String, bool) {
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    (sha, dirty)
}

/// Check if the environment is suitable for reproducible benchmarks.
pub fn check_reproducibility() -> Vec<String> {
    let env = EnvironmentInfo::collect();
    env.warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_collect() {
        let env = EnvironmentInfo::collect();
        assert!(!env.os.is_empty());
        assert!(env.cpu_cores > 0);
        assert!(env.ram_gb > 0);
    }

    #[test]
    fn test_fingerprint() {
        let env = EnvironmentInfo::collect();
        let fp = env.fingerprint();
        assert!(!fp.is_empty());
        assert!(!fp.contains(' '));
    }
}
