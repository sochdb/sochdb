//! Shared types, traits, data generators and latency recording for sochdb-bench.

pub mod adapters;
pub mod report;
pub mod workloads;

use hdrhistogram::Histogram;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;
use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ────────────────────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────────────────────

pub type BenchResult<T> = std::result::Result<T, BenchError>;

#[derive(Debug)]
pub enum BenchError {
    Io(std::io::Error),
    Database(String),
    Config(String),
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchError::Io(e) => write!(f, "IO error: {}", e),
            BenchError::Database(s) => write!(f, "Database error: {}", s),
            BenchError::Config(s) => write!(f, "Config error: {}", s),
        }
    }
}

impl std::error::Error for BenchError {}

impl From<std::io::Error> for BenchError {
    fn from(e: std::io::Error) -> Self {
        BenchError::Io(e)
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// BenchDb trait — every adapter implements this
// ────────────────────────────────────────────────────────────────────────────────

/// Row for analytics workloads.
#[derive(Debug, Clone)]
pub struct AnalyticsRow {
    pub id: u64,
    pub timestamp: i64,
    pub amount: f64,
    pub category: String,
    pub description: String,
}

/// Unified database adapter trait.
pub trait BenchDb: Send {
    fn name(&self) -> &str;

    // ── setup / teardown ──
    fn setup_kv_table(&mut self) -> BenchResult<()>;
    fn setup_analytics_table(&mut self) -> BenchResult<()>;
    fn setup_vector_table(&mut self, dim: usize) -> BenchResult<()>;
    fn teardown(&mut self) -> BenchResult<()>;

    // ── key-value ops ──
    fn put(&mut self, key: &[u8], value: &[u8]) -> BenchResult<()>;
    fn get(&mut self, key: &[u8]) -> BenchResult<Option<Vec<u8>>>;
    fn delete(&mut self, key: &[u8]) -> BenchResult<()>;
    fn batch_put(&mut self, pairs: &[(&[u8], &[u8])]) -> BenchResult<()>;

    // ── analytics ops ──
    fn insert_analytics_row(&mut self, row: &AnalyticsRow) -> BenchResult<()>;
    fn insert_analytics_batch(&mut self, rows: &[AnalyticsRow]) -> BenchResult<()>;
    fn scan_filter_amount_gt(&mut self, threshold: f64) -> BenchResult<usize>;
    fn aggregate_sum_amount(&mut self) -> BenchResult<f64>;
    fn group_by_category_count(&mut self) -> BenchResult<Vec<(String, u64)>>;
    fn range_scan_ts(&mut self, start: i64, end: i64) -> BenchResult<usize>;

    // ── vector ops ──
    fn insert_vector(&mut self, id: u64, vector: &[f32], metadata: Option<&str>) -> BenchResult<()>;
    fn insert_vector_batch(
        &mut self,
        vectors: &[(u64, Vec<f32>, Option<String>)],
    ) -> BenchResult<()>;
    fn vector_search(&mut self, query: &[f32], k: usize) -> BenchResult<Vec<(u64, f32)>>;

    // ── storage size ──
    fn db_size_bytes(&self) -> BenchResult<u64>;
}

// ────────────────────────────────────────────────────────────────────────────────
// Data generator (deterministic via ChaCha8Rng)
// ────────────────────────────────────────────────────────────────────────────────

pub struct DataGen {
    rng: ChaCha8Rng,
}

impl DataGen {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Generate a KV key: `kv:{id:08x}`.
    pub fn kv_key(&self, id: u64) -> Vec<u8> {
        format!("kv:{:08x}", id).into_bytes()
    }

    /// Generate a random value of `size` bytes.
    pub fn random_value(&mut self, size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; size];
        self.rng.fill_bytes(&mut buf);
        buf
    }

    /// Generate a random analytics row.
    pub fn analytics_row(&mut self, id: u64) -> AnalyticsRow {
        let categories = [
            "electronics",
            "clothing",
            "food",
            "books",
            "toys",
            "tools",
            "sports",
            "music",
        ];
        let ts_base = 1_700_000_000i64;
        AnalyticsRow {
            id,
            timestamp: ts_base + self.rng.gen_range(0..86_400 * 365),
            amount: self.rng.gen_range(1.0..10_000.0),
            category: categories[self.rng.gen_range(0..categories.len())].to_string(),
            description: format!("desc-{:06}", id),
        }
    }

    /// Generate a random f32 vector (normalised).
    pub fn random_vector(&mut self, dim: usize) -> Vec<f32> {
        let normal = Normal::new(0.0f32, 1.0).unwrap();
        let v: Vec<f32> = (0..dim).map(|_| self.rng.sample(normal)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        }
    }

    /// Generate a random u64.
    pub fn random_u64(&mut self) -> u64 {
        self.rng.gen()
    }

    /// Generate a range [0..n) in shuffled order.
    pub fn shuffled_indices(&mut self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut self.rng);
        indices
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Latency recorder (HDR histogram)
// ────────────────────────────────────────────────────────────────────────────────

pub struct LatencyRecorder {
    hist: Histogram<u64>,
    total: Duration,
    ops: u64,
}

impl LatencyRecorder {
    pub fn new() -> Self {
        Self {
            hist: Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap(),
            total: Duration::ZERO,
            ops: 0,
        }
    }

    /// Start a latency measurement.
    #[inline(always)]
    pub fn start(&self) -> Instant {
        Instant::now()
    }

    /// Record the elapsed time since `start`.
    #[inline(always)]
    pub fn record(&mut self, start: Instant) {
        let elapsed = start.elapsed();
        let nanos = elapsed.as_nanos() as u64;
        let _ = self.hist.record(nanos.max(1));
        self.total += elapsed;
        self.ops += 1;
    }

    /// Record `n` ops that collectively took `elapsed`.
    pub fn record_batch(&mut self, elapsed: Duration, n: u64) {
        let per_op = elapsed.as_nanos() as u64 / n.max(1);
        for _ in 0..n {
            let _ = self.hist.record(per_op.max(1));
        }
        self.total += elapsed;
        self.ops += n;
    }

    pub fn ops(&self) -> u64 {
        self.ops
    }

    pub fn total_secs(&self) -> f64 {
        self.total.as_secs_f64()
    }

    pub fn throughput(&self) -> f64 {
        if self.total.as_secs_f64() > 0.0 {
            self.ops as f64 / self.total.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Percentile in nanoseconds.
    pub fn percentile_ns(&self, p: f64) -> u64 {
        self.hist.value_at_percentile(p)
    }

    /// Percentile in microseconds.
    pub fn percentile_us(&self, p: f64) -> f64 {
        self.percentile_ns(p) as f64 / 1_000.0
    }

    /// Mean latency in microseconds.
    pub fn mean_us(&self) -> f64 {
        self.hist.mean() / 1_000.0
    }
}

impl Default for LatencyRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Benchmark output types
// ────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct WorkloadResult {
    pub db_name: String,
    pub workload: String,
    pub ops: u64,
    pub total_secs: f64,
    pub throughput: f64,           // ops/sec
    pub p50_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
    pub mean_us: f64,
    pub extra: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchSuite {
    pub system_info: SystemInfo,
    pub results: Vec<WorkloadResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpus: usize,
    pub timestamp: String,
}

impl SystemInfo {
    pub fn collect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpus: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            timestamp: chrono_now(),
        }
    }
}

fn chrono_now() -> String {
    // simple ISO-ish timestamp without pulling in chrono
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    format!("{}s-since-epoch", d.as_secs())
}

impl WorkloadResult {
    pub fn from_recorder(
        db_name: &str,
        workload: &str,
        rec: &LatencyRecorder,
    ) -> Self {
        Self {
            db_name: db_name.to_string(),
            workload: workload.to_string(),
            ops: rec.ops(),
            total_secs: rec.total_secs(),
            throughput: rec.throughput(),
            p50_us: rec.percentile_us(50.0),
            p99_us: rec.percentile_us(99.0),
            p999_us: rec.percentile_us(99.9),
            mean_us: rec.mean_us(),
            extra: HashMap::new(),
        }
    }

    pub fn with_extra(mut self, key: &str, val: &str) -> Self {
        self.extra.insert(key.to_string(), val.to_string());
        self
    }
}
