//! Shared types, traits, data generators and latency recording for sochdb-bench.

pub mod adapters;
pub mod memory_bench;
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
    fn insert_vector(&mut self, id: u64, vector: &[f32], metadata: Option<&str>)
        -> BenchResult<()>;
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
    pub throughput: f64, // ops/sec
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
    pub fn from_recorder(db_name: &str, workload: &str, rec: &LatencyRecorder) -> Self {
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

// ────────────────────────────────────────────────────────────────────────────────
// Durable scratch space
// ────────────────────────────────────────────────────────────────────────────────

/// Filesystems on which `fsync` returns without doing any I/O.
///
/// These are RAM-backed, so a durability benchmark pointed at one measures
/// memcpy and reports it as durable throughput.
const VOLATILE_FILESYSTEMS: &[&str] = &["tmpfs", "ramfs"];

/// The filesystem type backing `path`, by longest-prefix match against
/// `/proc/mounts`. Returns `None` when the table cannot be read, which is
/// treated as "unknown" rather than "safe" by the caller.
fn filesystem_type(path: &std::path::Path) -> Option<String> {
    let target = path.canonicalize().ok()?;
    let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
    filesystem_type_in(&mounts, &target)
}

/// Longest-prefix mount lookup, split from the `/proc/mounts` read so the
/// matching is testable.
///
/// Longest-prefix is required rather than merely tidy: `/` is a prefix of every
/// path, so a first-match scan reports the root filesystem for a `/tmp` that is
/// separately mounted as tmpfs -- precisely the configuration this guard exists
/// to catch.
fn filesystem_type_in(mounts: &str, target: &std::path::Path) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        // A malformed or blank line must skip, not abort the scan: giving up
        // early reports "unknown", which the caller treats as safe.
        let mut f = line.split_whitespace();
        let (mount_point, fstype) = match (f.next(), f.next(), f.next()) {
            (Some(_dev), Some(m), Some(t)) => (m, t),
            _ => continue,
        };
        // /proc/mounts octal-escapes spaces and tabs in mount points.
        let mount_point = mount_point.replace("\\040", " ").replace("\\011", "\t");
        if target.starts_with(&mount_point)
            && best
                .as_ref()
                .is_none_or(|(len, _)| mount_point.len() > *len)
        {
            best = Some((mount_point.len(), fstype.to_string()));
        }
    }
    best.map(|(_, fstype)| fstype)
}

/// Scratch directory for benchmarks whose result depends on `fsync` actually
/// reaching stable storage.
///
/// `tempfile::TempDir::new` follows `TMPDIR`, which on most Linux desktops is
/// `/tmp` and is frequently `tmpfs`. `fsync` on `tmpfs` is a no-op, so a
/// durability benchmark run there does not fail or warn -- it returns numbers
/// that are two to three orders of magnitude too high and, worse, can invert
/// the ranking of the configurations under test. Group commit exists to
/// amortize fsync across committers; with fsync free it looks like pure
/// overhead, so a tmpfs run recommends turning off the one setting that makes
/// durable writes scale.
///
/// This refuses to run on such a filesystem rather than annotating the output,
/// because the failure is silent and the numbers are quotable.
///
/// `SOCHDB_BENCH_TMP` overrides the location. `SOCHDB_BENCH_ALLOW_TMPFS=1`
/// suppresses the check for runs that are deliberately measuring the non-fsync
/// path.
pub fn durable_temp_dir() -> std::io::Result<tempfile::TempDir> {
    let base = std::env::var_os("SOCHDB_BENCH_TMP")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&base)?;

    if std::env::var("SOCHDB_BENCH_ALLOW_TMPFS").as_deref() != Ok("1") {
        if let Some(fstype) = filesystem_type(&base) {
            if VOLATILE_FILESYSTEMS.contains(&fstype.as_str()) {
                return Err(std::io::Error::other(format!(
                    "{} is {}, where fsync is a no-op, so durable throughput measured \
                     here is meaningless (observed: 2210x too high at 1 thread, and \
                     group commit appears 709x slower than no group commit when on \
                     real disk it is 18x faster at 64 threads).\n\
                     Set SOCHDB_BENCH_TMP to a directory on a real filesystem, or \
                     SOCHDB_BENCH_ALLOW_TMPFS=1 to measure the non-fsync path on purpose.",
                    base.display(),
                    fstype,
                )));
            }
        }
    }
    tempfile::TempDir::new_in(&base)
}

#[cfg(test)]
mod durable_scratch_tests {
    use super::*;
    use std::path::Path;

    const MOUNTS: &str = "\
/dev/nvme0n1p2 / ext4 rw,relatime 0 0
tmpfs /tmp tmpfs rw,nosuid,nodev 0 0
/dev/nvme0n1p1 /boot/efi vfat rw 0 0
tmpfs /run/user/1000 tmpfs rw,nosuid 0 0
";

    #[test]
    fn a_separately_mounted_tmp_is_reported_as_tmpfs_and_not_as_the_root_filesystem() {
        // `/` is a prefix of `/tmp`, so a first-match scan would answer ext4
        // here and the guard would pass on a filesystem where fsync is free.
        assert_eq!(
            filesystem_type_in(MOUNTS, Path::new("/tmp/dc_data")).as_deref(),
            Some("tmpfs")
        );
        assert_eq!(
            filesystem_type_in(MOUNTS, Path::new("/home/user/scratch")).as_deref(),
            Some("ext4")
        );
    }

    #[test]
    fn a_blank_or_truncated_mounts_line_skips_rather_than_ending_the_scan() {
        // Aborting early yields None, which the caller reads as "unknown" and
        // therefore lets the run proceed -- the same silent pass the guard is
        // meant to prevent.
        let ragged = format!("\n{}bad-line-no-fields\n\n", MOUNTS);
        assert_eq!(
            filesystem_type_in(&ragged, Path::new("/tmp/dc_data")).as_deref(),
            Some("tmpfs")
        );
    }

    #[test]
    fn mount_points_containing_escaped_whitespace_are_decoded_before_matching() {
        let mounts = "tmpfs /mnt/my\\040disk tmpfs rw 0 0\n";
        assert_eq!(
            filesystem_type_in(mounts, Path::new("/mnt/my disk/x")).as_deref(),
            Some("tmpfs")
        );
    }

    #[test]
    fn every_filesystem_named_volatile_is_one_whose_fsync_does_no_io() {
        assert!(VOLATILE_FILESYSTEMS.contains(&"tmpfs"));
        assert!(VOLATILE_FILESYSTEMS.contains(&"ramfs"));
        assert!(!VOLATILE_FILESYSTEMS.contains(&"ext4"));
        assert!(!VOLATILE_FILESYSTEMS.contains(&"xfs"));
    }
}
