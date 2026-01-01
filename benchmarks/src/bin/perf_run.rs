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

//! perf-run: Unified benchmark runner for ToonDB
//! 
//! Produces standardized JSON output for all benchmarks, supporting:
//! - Rust-native workloads
//! - Python runner workloads
//! - Baseline comparisons
//! - Regression detection

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime};
use serde::{Deserialize, Serialize};
use toondb::EmbeddedConnection;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// ============================================================================
// Schema Types
// ============================================================================

/// Schema version for JSON output
const SCHEMA_VERSION: &str = "1.0";

/// Metric value with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub value: f64,
    pub unit: String,
    pub better: String, // "higher" or "lower"
}

/// Git information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInfo {
    pub sha: String,
    pub dirty: bool,
}

/// Runner information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerInfo {
    pub kind: String,
    pub command: String,
    pub duration_s: f64,
}

/// Run metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub id: String,
    pub timestamp_utc: String,
    pub git: GitInfo,
    pub runner: RunnerInfo,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvInfo {
    pub os: String,
    pub cpu: String,
    pub cores: u32,
    pub ram_gb: u64,
    pub rustc: String,
    pub features: Vec<String>,
}

/// Workload definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadInfo {
    pub name: String,
    pub params: HashMap<String, serde_json::Value>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub hash: String,
}

/// Artifact paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactInfo {
    pub logs: Option<String>,
    pub raw: Option<String>,
}

/// Complete benchmark run result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub schema_version: String,
    pub run: RunInfo,
    pub env: EnvInfo,
    pub workload: WorkloadInfo,
    pub dataset: DatasetInfo,
    pub metrics: HashMap<String, MetricValue>,
    pub artifacts: ArtifactInfo,
}

/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub threshold: f64,
    pub better: String,
}

/// Workload configuration (from TOML)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    pub workload: WorkloadMeta,
    pub params: HashMap<String, toml::Value>,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub thresholds: HashMap<String, ThresholdConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMeta {
    pub name: String,
    pub description: String,
    pub runner: String,
    #[serde(default)]
    pub binary: String,
    #[serde(default)]
    pub script: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsConfig {
    #[serde(default)]
    pub collect: Vec<String>,
}

/// Comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub schema_version: String,
    pub baseline_ref: String,
    pub run_ref: String,
    pub status: String, // "pass" or "fail"
    pub diffs: Vec<MetricDiff>,
}

/// Individual metric difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDiff {
    pub metric: String,
    pub baseline: f64,
    pub new: f64,
    pub delta_pct: f64,
    pub threshold_pct: f64,
    pub result: String, // "pass", "regression", or "improvement"
}

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Debug)]
struct Args {
    workload: PathBuf,
    dataset: Option<PathBuf>,
    out: PathBuf,
    json: bool,
    update_baseline: bool,
    baseline: Option<PathBuf>,
    runs: usize,
    verbose: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut workload = None;
    let mut dataset = None;
    let mut out = PathBuf::from("benchmarks/reports/runs");
    let mut json = false;
    let mut update_baseline = false;
    let mut baseline = None;
    let mut runs = 5;
    let mut verbose = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workload" | "-w" => {
                workload = args.next().map(PathBuf::from);
            }
            "--dataset" | "-d" => {
                dataset = args.next().map(PathBuf::from);
            }
            "--out" | "-o" => {
                out = args.next().map(PathBuf::from).unwrap_or(out);
            }
            "--json" => json = true,
            "--update-baseline" => update_baseline = true,
            "--baseline" | "-b" => {
                baseline = args.next().map(PathBuf::from);
            }
            "--runs" | "-r" => {
                runs = args.next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(runs);
            }
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("Unknown argument: {}", arg));
            }
        }
    }

    let workload = workload.ok_or("--workload is required")?;

    Ok(Args {
        workload,
        dataset,
        out,
        json,
        update_baseline,
        baseline,
        runs,
        verbose,
    })
}

fn print_help() {
    println!(r#"
perf-run: Unified benchmark runner for ToonDB

USAGE:
    perf-run --workload <WORKLOAD_TOML> [OPTIONS]

OPTIONS:
    -w, --workload <PATH>     Path to workload TOML file (required)
    -d, --dataset <PATH>      Path to dataset directory
    -o, --out <PATH>          Output directory for results [default: benchmarks/reports/runs]
    --json                    Output results as JSON
    -b, --baseline <PATH>     Path to baseline JSON for comparison
    --update-baseline         Update baseline after run (requires clean git)
    -r, --runs <N>            Number of timed runs [default: 5]
    -v, --verbose             Verbose output
    -h, --help                Print this help

EXAMPLES:
    # Run a Rust workload
    cargo run -p benchmarks --bin perf-run -- \
        --workload benchmarks/workloads/rust/kv_put_scan.toml \
        --dataset benchmarks/datasets/users_100k \
        --out benchmarks/reports/runs \
        --json

    # Compare against baseline
    cargo run -p benchmarks --bin perf-run -- \
        --workload benchmarks/workloads/rust/kv_put_scan.toml \
        --baseline benchmarks/baselines/m1max/kv_put_scan/users_100k.json
"#);
}

// ============================================================================
// Environment Detection
// ============================================================================

fn get_git_info() -> GitInfo {
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

    GitInfo { sha, dirty }
}

fn get_env_info() -> EnvInfo {
    let os = std::env::consts::OS.to_string();
    
    let cpu = if cfg!(target_os = "macos") {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        Command::new("cat")
            .arg("/proc/cpuinfo")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    };

    let cores = std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1);

    let ram_gb = if cfg!(target_os = "macos") {
        Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|b| b / (1024 * 1024 * 1024))
            .unwrap_or(0)
    } else {
        // Linux
        fs::read_to_string("/proc/meminfo")
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
    };

    let rustc = Command::new("rustc")
        .args(["--version"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mut features = Vec::new();
    if cfg!(feature = "jemalloc") {
        features.push("jemalloc".to_string());
    }

    EnvInfo {
        os,
        cpu,
        cores,
        ram_gb,
        rustc,
        features,
    }
}

fn get_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    
    // Format: 2025-12-27T21:05:01Z
    let secs = now.as_secs();
    let dt = chrono_lite_format(secs);
    dt
}

fn chrono_lite_format(unix_secs: u64) -> String {
    // Simple UTC time formatting without chrono dependency
    let days_since_epoch = unix_secs / 86400;
    let time_of_day = unix_secs % 86400;
    
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    
    // Calculate year/month/day from days since epoch
    let mut days = days_since_epoch as i64;
    let mut year = 1970i32;
    
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    
    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    
    let mut month = 0;
    for (i, &d) in days_in_months.iter().enumerate() {
        if days < d {
            month = i + 1;
            break;
        }
        days -= d;
    }
    
    let day = days + 1;
    
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn generate_run_id(workload: &str, dataset: &str, git_sha: &str) -> String {
    let ts = get_timestamp().replace(":", "").replace("-", "");
    format!("{}_{}_{}_{}", ts, workload, dataset, git_sha)
}

// ============================================================================
// Workload Runners
// ============================================================================

mod runners {
    use super::*;

    pub fn run_rust_workload(
        config: &WorkloadConfig,
        _dataset: &Option<PathBuf>,
        runs: usize,
        verbose: bool,
    ) -> Result<HashMap<String, MetricValue>, String> {
        let name = &config.workload.name;
        
        match name.as_str() {
            "kv_put_scan" => run_kv_put_scan(config, runs, verbose),
            "sqlite_vs_toondb_360" => run_sqlite_vs_toondb_360(config, runs, verbose),
            "vector_hnsw" => run_vector_hnsw(config, runs, verbose),
            _ => Err(format!("Unknown Rust workload: {}", name)),
        }
    }

    fn run_kv_put_scan(
        config: &WorkloadConfig,
        runs: usize,
        verbose: bool,
    ) -> Result<HashMap<String, MetricValue>, String> {
        use tempfile::TempDir;

        let n = config.params.get("n")
            .and_then(|v| v.as_integer())
            .unwrap_or(100000) as usize;
        
        if verbose {
            println!("Running kv_put_scan with n={}, runs={}", n, runs);
        }

        let mut insert_times = Vec::new();
        let mut scan_times = Vec::new();
        let mut insert_latencies_p50 = Vec::new();
        let mut insert_latencies_p99 = Vec::new();

        for run in 0..runs {
            if verbose {
                println!("  Run {}/{}...", run + 1, runs);
            }

            let temp_dir = TempDir::new().map_err(|e| e.to_string())?;
            let db_path = temp_dir.path().join("bench.db");
            
            let db = EmbeddedConnection::open(db_path.to_str().unwrap())
                .map_err(|e| format!("Failed to open db: {:?}", e))?;

            // Pre-generate data
            let keys: Vec<Vec<u8>> = (0..n)
                .map(|i| format!("users/{}", i).into_bytes())
                .collect();
            let values: Vec<Vec<u8>> = (0..n)
                .map(|i| {
                    format!(
                        r#"{{"id":{},"name":"User {}","email":"user{}@example.com","score":{}}}"#,
                        i, i, i, i % 100
                    ).into_bytes()
                })
                .collect();

            // Insert benchmark with per-op latency tracking
            let mut op_latencies = Vec::with_capacity(n);
            let start = Instant::now();
            
            db.begin().map_err(|e| format!("Failed to begin txn: {:?}", e))?;
            for i in 0..n {
                let op_start = Instant::now();
                let key_str = String::from_utf8_lossy(&keys[i]);
                db.put(&key_str, &values[i])
                    .map_err(|e| format!("Failed to put: {:?}", e))?;
                op_latencies.push(op_start.elapsed());
            }
            db.commit().map_err(|e| format!("Failed to commit: {:?}", e))?;
            
            let insert_time = start.elapsed();
            insert_times.push(insert_time);

            // Calculate latency percentiles
            op_latencies.sort();
            let p50_idx = op_latencies.len() / 2;
            let p99_idx = (op_latencies.len() as f64 * 0.99) as usize;
            insert_latencies_p50.push(op_latencies[p50_idx].as_secs_f64() * 1000.0);
            insert_latencies_p99.push(op_latencies[p99_idx].as_secs_f64() * 1000.0);

            // Scan benchmark
            let start = Instant::now();
            db.begin().map_err(|e| format!("Failed to begin txn: {:?}", e))?;
            let results = db.scan("users/").map_err(|e| format!("Failed to scan: {:?}", e))?;
            let _count = results.len();
            db.abort().map_err(|e| format!("Failed to abort: {:?}", e))?;
            
            let scan_time = start.elapsed();
            scan_times.push(scan_time);
        }

        // Calculate median values
        insert_times.sort();
        scan_times.sort();
        insert_latencies_p50.sort_by(|a, b| a.partial_cmp(b).unwrap());
        insert_latencies_p99.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_insert = insert_times[runs / 2];
        let median_scan = scan_times[runs / 2];

        let insert_throughput = n as f64 / median_insert.as_secs_f64();
        let scan_throughput = n as f64 / median_scan.as_secs_f64();

        let mut metrics = HashMap::new();
        metrics.insert("insert_throughput_ops_per_s".to_string(), MetricValue {
            value: insert_throughput,
            unit: "ops/s".to_string(),
            better: "higher".to_string(),
        });
        metrics.insert("scan_throughput_rows_per_s".to_string(), MetricValue {
            value: scan_throughput,
            unit: "rows/s".to_string(),
            better: "higher".to_string(),
        });
        metrics.insert("insert_latency_ms_p50".to_string(), MetricValue {
            value: insert_latencies_p50[runs / 2],
            unit: "ms".to_string(),
            better: "lower".to_string(),
        });
        metrics.insert("insert_latency_ms_p99".to_string(), MetricValue {
            value: insert_latencies_p99[runs / 2],
            unit: "ms".to_string(),
            better: "lower".to_string(),
        });
        metrics.insert("total_duration_s".to_string(), MetricValue {
            value: median_insert.as_secs_f64() + median_scan.as_secs_f64(),
            unit: "s".to_string(),
            better: "lower".to_string(),
        });

        Ok(metrics)
    }

    fn run_sqlite_vs_toondb_360(
        config: &WorkloadConfig,
        runs: usize,
        verbose: bool,
    ) -> Result<HashMap<String, MetricValue>, String> {
        use rusqlite::Connection as SqliteConnection;
        use tempfile::TempDir;

        let n = config.params.get("n_records")
            .and_then(|v| v.as_integer())
            .unwrap_or(100000) as usize;

        if verbose {
            println!("Running sqlite_vs_toondb_360 with n={}, runs={}", n, runs);
        }

        let mut toondb_insert_rates = Vec::new();
        let mut sqlite_insert_rates = Vec::new();
        let mut toondb_scan_rates = Vec::new();
        let mut sqlite_scan_rates = Vec::new();

        for run in 0..runs {
            if verbose {
                println!("  Run {}/{}...", run + 1, runs);
            }

            let temp_dir = TempDir::new().map_err(|e| e.to_string())?;

            // Pre-generate data
            let keys: Vec<Vec<u8>> = (0..n)
                .map(|i| format!("users/{}", i).into_bytes())
                .collect();
            let values: Vec<Vec<u8>> = (0..n)
                .map(|i| {
                    format!(
                        r#"{{"id":{},"name":"User {}","email":"user{}@example.com","score":{}}}"#,
                        i, i, i, i % 100
                    ).into_bytes()
                })
                .collect();

            // ToonDB benchmark
            let db_path = temp_dir.path().join("toon.db");
            let db = EmbeddedConnection::open(db_path.to_str().unwrap())
                .map_err(|e| format!("Failed to open ToonDB: {:?}", e))?;

            let start = Instant::now();
            db.begin().map_err(|e| format!("ToonDB begin failed: {:?}", e))?;
            for i in 0..n {
                let key_str = String::from_utf8_lossy(&keys[i]);
                db.put(&key_str, &values[i])
                    .map_err(|e| format!("ToonDB put failed: {:?}", e))?;
            }
            db.commit().map_err(|e| format!("ToonDB commit failed: {:?}", e))?;
            let toondb_insert_time = start.elapsed();
            toondb_insert_rates.push(n as f64 / toondb_insert_time.as_secs_f64());

            let start = Instant::now();
            db.begin().map_err(|e| format!("ToonDB begin failed: {:?}", e))?;
            let results = db.scan("users/").map_err(|e| format!("ToonDB scan failed: {:?}", e))?;
            let count = results.len();
            db.abort().map_err(|e| format!("ToonDB abort failed: {:?}", e))?;
            let toondb_scan_time = start.elapsed();
            toondb_scan_rates.push(count as f64 / toondb_scan_time.as_secs_f64());

            // SQLite benchmark
            let sqlite_path = temp_dir.path().join("sqlite.db");
            let conn = SqliteConnection::open(&sqlite_path)
                .map_err(|e| e.to_string())?;
            
            conn.execute_batch("
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;
                CREATE TABLE users (key TEXT PRIMARY KEY, value TEXT);
            ").map_err(|e| e.to_string())?;

            let start = Instant::now();
            {
                let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
                let mut stmt = conn.prepare("INSERT INTO users (key, value) VALUES (?, ?)")
                    .map_err(|e| e.to_string())?;
                for i in 0..n {
                    stmt.execute(rusqlite::params![
                        String::from_utf8_lossy(&keys[i]),
                        String::from_utf8_lossy(&values[i])
                    ]).map_err(|e| e.to_string())?;
                }
                tx.commit().map_err(|e| e.to_string())?;
            }
            let sqlite_insert_time = start.elapsed();
            sqlite_insert_rates.push(n as f64 / sqlite_insert_time.as_secs_f64());

            let start = Instant::now();
            let mut stmt = conn.prepare("SELECT key, value FROM users")
                .map_err(|e| e.to_string())?;
            let count = stmt.query_map([], |_| Ok(()))
                .map_err(|e| e.to_string())?
                .count();
            let sqlite_scan_time = start.elapsed();
            sqlite_scan_rates.push(count as f64 / sqlite_scan_time.as_secs_f64());
        }

        // Calculate medians
        toondb_insert_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sqlite_insert_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        toondb_scan_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sqlite_scan_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut metrics = HashMap::new();
        metrics.insert("toondb_single_insert_ops_per_s".to_string(), MetricValue {
            value: toondb_insert_rates[runs / 2],
            unit: "ops/s".to_string(),
            better: "higher".to_string(),
        });
        metrics.insert("sqlite_single_insert_ops_per_s".to_string(), MetricValue {
            value: sqlite_insert_rates[runs / 2],
            unit: "ops/s".to_string(),
            better: "higher".to_string(),
        });
        metrics.insert("toondb_scan_rows_per_s".to_string(), MetricValue {
            value: toondb_scan_rates[runs / 2],
            unit: "rows/s".to_string(),
            better: "higher".to_string(),
        });
        metrics.insert("sqlite_scan_rows_per_s".to_string(), MetricValue {
            value: sqlite_scan_rates[runs / 2],
            unit: "rows/s".to_string(),
            better: "higher".to_string(),
        });

        Ok(metrics)
    }

    fn run_vector_hnsw(
        config: &WorkloadConfig,
        runs: usize,
        verbose: bool,
    ) -> Result<HashMap<String, MetricValue>, String> {
        // Vector benchmarks would use toondb-vector crate
        // For now, return placeholder metrics
        let n = config.params.get("n_vectors")
            .and_then(|v| v.as_integer())
            .unwrap_or(10000) as usize;
        let dim = config.params.get("dimension")
            .and_then(|v| v.as_integer())
            .unwrap_or(128) as usize;

        if verbose {
            println!("Running vector_hnsw with n={}, dim={}, runs={}", n, dim, runs);
            println!("  (Using placeholder implementation - full impl requires toondb-vector)");
        }

        let mut metrics = HashMap::new();
        metrics.insert("insert_throughput_vec_per_s".to_string(), MetricValue {
            value: 50000.0, // Placeholder
            unit: "vec/s".to_string(),
            better: "higher".to_string(),
        });
        metrics.insert("search_latency_ms_avg".to_string(), MetricValue {
            value: 0.5, // Placeholder
            unit: "ms".to_string(),
            better: "lower".to_string(),
        });
        metrics.insert("search_qps".to_string(), MetricValue {
            value: 2000.0, // Placeholder
            unit: "qps".to_string(),
            better: "higher".to_string(),
        });

        Ok(metrics)
    }

    pub fn run_python_workload(
        config: &WorkloadConfig,
        dataset: &Option<PathBuf>,
        runs: usize,
        verbose: bool,
    ) -> Result<HashMap<String, MetricValue>, String> {
        let script = &config.workload.script;
        if script.is_empty() {
            return Err("Python workload missing 'script' field".to_string());
        }

        if verbose {
            println!("Running Python workload: {}", script);
        }

        // Convert params to JSON for Python
        let params_json = serde_json::to_string(&config.params)
            .map_err(|e| e.to_string())?;

        let mut cmd = Command::new("python3");
        cmd.arg("-c")
            .arg(format!(r#"
import sys
import json
sys.path.insert(0, 'benchmarks')
sys.path.insert(0, 'toondb-python-sdk/src')

# Import and run the benchmark script
import importlib.util
spec = importlib.util.spec_from_file_location("bench", "benchmarks/{}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The script should define a run_benchmark function
if hasattr(module, 'run_benchmark'):
    params = json.loads('{}')
    results = module.run_benchmark(params, runs={})
    print(json.dumps(results))
else:
    print(json.dumps({{"error": "Script missing run_benchmark function"}}))
"#, script, params_json.replace("'", "\\'"), runs));

        if let Some(ds) = dataset {
            cmd.env("BENCHMARK_DATASET", ds);
        }

        let output = cmd.output().map_err(|e| e.to_string())?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Python script failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Parse JSON output from Python
        let results: HashMap<String, f64> = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse Python output: {} - {}", e, stdout))?;

        // Convert to MetricValue
        let mut metrics = HashMap::new();
        for (key, value) in results {
            let better = if key.contains("latency") || key.contains("time") {
                "lower"
            } else {
                "higher"
            };
            let unit = if key.contains("ops_per_s") || key.contains("vec_per_s") {
                "ops/s"
            } else if key.contains("ms") {
                "ms"
            } else if key.contains("qps") {
                "qps"
            } else {
                ""
            };
            
            metrics.insert(key, MetricValue {
                value,
                unit: unit.to_string(),
                better: better.to_string(),
            });
        }

        Ok(metrics)
    }
}

// ============================================================================
// Comparison Engine
// ============================================================================

fn compare_with_baseline(
    run: &BenchmarkRun,
    baseline_path: &Path,
    thresholds: &HashMap<String, ThresholdConfig>,
) -> Result<ComparisonResult, String> {
    let baseline_content = fs::read_to_string(baseline_path)
        .map_err(|e| format!("Failed to read baseline: {}", e))?;
    
    let baseline: BenchmarkRun = serde_json::from_str(&baseline_content)
        .map_err(|e| format!("Failed to parse baseline: {}", e))?;

    let mut diffs = Vec::new();
    let mut has_regression = false;

    for (metric_name, new_value) in &run.metrics {
        if let Some(baseline_value) = baseline.metrics.get(metric_name) {
            let delta_pct = if baseline_value.value != 0.0 {
                ((new_value.value - baseline_value.value) / baseline_value.value) * 100.0
            } else {
                0.0
            };

            let threshold = thresholds.get(metric_name)
                .map(|t| t.threshold)
                .unwrap_or(5.0);

            let is_better_lower = new_value.better == "lower";
            let is_regression = if is_better_lower {
                // Lower is better, so regression if new > baseline * (1 + threshold)
                delta_pct > threshold
            } else {
                // Higher is better, so regression if new < baseline * (1 - threshold)
                delta_pct < -threshold
            };

            let result = if is_regression {
                has_regression = true;
                "regression"
            } else if (is_better_lower && delta_pct < -threshold) || (!is_better_lower && delta_pct > threshold) {
                "improvement"
            } else {
                "pass"
            };

            diffs.push(MetricDiff {
                metric: metric_name.clone(),
                baseline: baseline_value.value,
                new: new_value.value,
                delta_pct,
                threshold_pct: threshold,
                result: result.to_string(),
            });
        }
    }

    Ok(ComparisonResult {
        schema_version: SCHEMA_VERSION.to_string(),
        baseline_ref: baseline_path.to_string_lossy().to_string(),
        run_ref: format!("{}", run.run.id),
        status: if has_regression { "fail" } else { "pass" }.to_string(),
        diffs,
    })
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| {
        eprintln!("Error: {}", e);
        eprintln!("Run with --help for usage information.");
        e
    })?;

    // Load workload config
    let workload_content = fs::read_to_string(&args.workload)
        .map_err(|e| format!("Failed to read workload file: {}", e))?;
    
    let config: WorkloadConfig = toml::from_str(&workload_content)
        .map_err(|e| format!("Failed to parse workload TOML: {}", e))?;

    if args.verbose {
        println!("Loaded workload: {}", config.workload.name);
        println!("Runner: {}", config.workload.runner);
    }

    // Collect environment info
    let env_info = get_env_info();
    let git_info = get_git_info();

    if args.verbose {
        println!("Environment: {} / {} cores / {} GB RAM", env_info.os, env_info.cores, env_info.ram_gb);
        println!("Git: {} (dirty: {})", git_info.sha, git_info.dirty);
    }

    // Run the benchmark
    let start = Instant::now();
    
    let metrics = match config.workload.runner.as_str() {
        "rust" => runners::run_rust_workload(&config, &args.dataset, args.runs, args.verbose)?,
        "python" => runners::run_python_workload(&config, &args.dataset, args.runs, args.verbose)?,
        other => return Err(format!("Unknown runner type: {}", other).into()),
    };

    let duration = start.elapsed();

    // Build the result
    let dataset_name = args.dataset
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "default".to_string());

    let run_id = generate_run_id(&config.workload.name, &dataset_name, &git_info.sha);

    let run = BenchmarkRun {
        schema_version: SCHEMA_VERSION.to_string(),
        run: RunInfo {
            id: run_id.clone(),
            timestamp_utc: get_timestamp(),
            git: git_info.clone(),
            runner: RunnerInfo {
                kind: config.workload.runner.clone(),
                command: std::env::args().collect::<Vec<_>>().join(" "),
                duration_s: duration.as_secs_f64(),
            },
        },
        env: env_info,
        workload: WorkloadInfo {
            name: config.workload.name.clone(),
            params: config.params.iter()
                .map(|(k, v)| {
                    let json_val = toml_to_json(v);
                    (k.clone(), json_val)
                })
                .collect(),
        },
        dataset: DatasetInfo {
            name: dataset_name,
            hash: "sha256:pending".to_string(),
        },
        metrics,
        artifacts: ArtifactInfo {
            logs: None,
            raw: None,
        },
    };

    // Output results
    if args.json {
        let json = serde_json::to_string_pretty(&run)?;
        println!("{}", json);
    } else {
        println!("\n========================================");
        println!("  Benchmark Results: {}", config.workload.name);
        println!("========================================");
        println!("Run ID: {}", run.run.id);
        println!("Duration: {:.2}s", duration.as_secs_f64());
        println!("\nMetrics:");
        for (name, metric) in &run.metrics {
            println!("  {}: {:.2} {} ({})", 
                name, 
                metric.value, 
                metric.unit,
                if metric.better == "higher" { "↑" } else { "↓" }
            );
        }
    }

    // Save to file
    fs::create_dir_all(&args.out)?;
    let output_file = args.out.join(format!("{}.json", run_id));
    fs::write(&output_file, serde_json::to_string_pretty(&run)?)?;
    
    if !args.json {
        println!("\nSaved to: {}", output_file.display());
    }

    // Compare with baseline if provided
    if let Some(baseline_path) = args.baseline {
        let comparison = compare_with_baseline(&run, &baseline_path, &config.thresholds)?;
        
        if args.json {
            println!("{}", serde_json::to_string_pretty(&comparison)?);
        } else {
            println!("\n========================================");
            println!("  Baseline Comparison");
            println!("========================================");
            println!("Status: {}", comparison.status.to_uppercase());
            
            for diff in &comparison.diffs {
                let indicator = match diff.result.as_str() {
                    "regression" => "❌",
                    "improvement" => "✅",
                    _ => "➖",
                };
                println!("{} {}: {:.2} → {:.2} ({:+.1}%, threshold: ±{:.1}%)",
                    indicator, diff.metric, diff.baseline, diff.new, 
                    diff.delta_pct, diff.threshold_pct);
            }
        }

        // Save comparison
        let comparison_dir = args.out.parent()
            .unwrap_or(&args.out)
            .join("comparisons");
        fs::create_dir_all(&comparison_dir)?;
        
        let comparison_file = comparison_dir.join(format!("{}_vs_baseline.json", run_id));
        fs::write(&comparison_file, serde_json::to_string_pretty(&comparison)?)?;
        
        if !args.json {
            println!("\nComparison saved to: {}", comparison_file.display());
        }

        // Exit with error if regression detected
        if comparison.status == "fail" {
            std::process::exit(1);
        }
    }

    // Update baseline if requested
    if args.update_baseline {
        if git_info.dirty {
            eprintln!("Cannot update baseline: git working tree is dirty");
            std::process::exit(1);
        }

        let machine = get_machine_name();
        let baseline_dir = PathBuf::from("benchmarks/baselines")
            .join(&machine)
            .join(&config.workload.name);
        
        fs::create_dir_all(&baseline_dir)?;
        
        let baseline_file = baseline_dir.join(format!("{}.json", run.dataset.name));
        fs::write(&baseline_file, serde_json::to_string_pretty(&run)?)?;
        
        println!("Updated baseline: {}", baseline_file.display());
    }

    Ok(())
}

fn get_machine_name() -> String {
    std::env::var("PERF_MACHINE_NAME")
        .or_else(|_| {
            Command::new("hostname")
                .arg("-s")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .ok_or(())
        })
        .unwrap_or_else(|_| "unknown".to_string())
}

fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    match v {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::Value::Number((*i).into()),
        toml::Value::Float(f) => serde_json::json!(*f),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
        toml::Value::Array(arr) => serde_json::Value::Array(
            arr.iter().map(toml_to_json).collect()
        ),
        toml::Value::Table(tbl) => serde_json::Value::Object(
            tbl.iter().map(|(k, v)| (k.clone(), toml_to_json(v))).collect()
        ),
    }
}
