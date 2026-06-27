//! Concurrent durable write micro-harness (statistically hardened).
//!
//! First-principles motivation: the main `sochdb-bench` is single-threaded
//! (`&mut dyn BenchDb`) and runs `SyncMode::Off`, so it cannot exercise the
//! path that decides whether SochDB is competitive with fsync-on, many-client
//! engines (e.g. SurrealDB's crud-bench). Durable throughput under concurrency
//! is governed by exactly two things:
//!   1. fsync latency — amortized only if many commits share one fsync.
//!   2. WAL append serialization — every committer contends on one writer lock.
//!
//! Measurement integrity: this machine (a thermally-throttled laptop) drifts up
//! to ~3.5x run-to-run, so absolute numbers across separate invocations are not
//! comparable. To get trustworthy A/B DELTAS we:
//!   - run multiple timed REPEATS per data point and discard the first (warmup);
//!   - INTERLEAVE the configs being compared within each repeat round, so both
//!     are measured microseconds apart and any thermal drift hits both equally;
//!   - report median + [min,max] per (config, threads), and the median ratio
//!     between the first two configs as the A/B speedup.
//!
//! Env knobs:
//!   DC_THREADS   comma list of thread counts to sweep   (default "1,8,32,64,128")
//!   DC_OPS       timed commits PER THREAD               (default 1000)
//!   DC_REPEATS   timed repeats per point (1st discarded) (default 4)
//!   DC_VALSIZE   value bytes                             (default 256)
//!   DC_CONFIGS   comma list of "sync:group" to A/B,      (default "full:1")
//!                interleaved; sync in {off,normal,full}, group in {0,1}.
//!                Back-compat: if unset, DC_SYNC + DC_GROUP build a single config.

use std::sync::{Arc, Barrier};
use std::time::Instant;

use sochdb_storage::{Database, DatabaseConfig, SyncMode};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[derive(Clone, Copy)]
struct Cfg {
    sync: SyncMode,
    group: bool,
}

impl Cfg {
    fn label(&self) -> String {
        let s = match self.sync {
            SyncMode::Off => "off",
            SyncMode::Normal => "normal",
            SyncMode::Full => "full",
        };
        format!("{}:g{}", s, self.group as u8)
    }
}

fn parse_sync(s: &str) -> SyncMode {
    match s {
        "off" => SyncMode::Off,
        "normal" => SyncMode::Normal,
        _ => SyncMode::Full,
    }
}

/// One timed run: open a fresh durable DB with `cfg`, fan `nthreads` committers
/// at it for `ops_per_thread` commits each, return aggregate ops/s.
fn run_point(cfg: Cfg, nthreads: usize, ops_per_thread: usize, valsize: usize) -> f64 {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let path = tmp.path().join("dc_data");
    std::fs::create_dir_all(&path).unwrap();

    let mut config = DatabaseConfig::throughput_optimized();
    config.sync_mode = cfg.sync;
    config.group_commit = cfg.group;
    let db = Arc::new(Database::open_with_config(&path, config).expect("open durable db"));

    let value = vec![0xABu8; valsize];
    let barrier = Arc::new(Barrier::new(nthreads + 1));

    let mut handles = Vec::with_capacity(nthreads);
    for t in 0..nthreads {
        let db = Arc::clone(&db);
        let value = value.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for i in 0..ops_per_thread {
                let key = ((t as u64) << 40 | i as u64).to_be_bytes();
                let txn = db.begin_write_only().expect("begin");
                db.put_raw(txn, &key, &value).expect("put");
                db.commit(txn).expect("commit");
            }
        }));
    }

    barrier.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let wall = start.elapsed();
    (nthreads * ops_per_thread) as f64 / wall.as_secs_f64()
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() {
    let threads: Vec<usize> = std::env::var("DC_THREADS")
        .unwrap_or_else(|_| "1,8,32,64,128".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let ops_per_thread = env_usize("DC_OPS", 1000);
    let valsize = env_usize("DC_VALSIZE", 256);
    let repeats = env_usize("DC_REPEATS", 4).max(2); // >=2 so we can discard warmup

    // Configs to A/B (interleaved).
    let configs: Vec<Cfg> = match std::env::var("DC_CONFIGS") {
        Ok(s) => s
            .split(',')
            .filter_map(|c| {
                let mut it = c.trim().split(':');
                let sync = parse_sync(it.next()?);
                let group = it.next().map(|g| g == "1").unwrap_or(true);
                Some(Cfg { sync, group })
            })
            .collect(),
        Err(_) => {
            let sync = parse_sync(&std::env::var("DC_SYNC").unwrap_or_else(|_| "full".into()));
            let group = std::env::var("DC_GROUP").map(|v| v == "1").unwrap_or(true);
            vec![Cfg { sync, group }]
        }
    };

    println!(
        "durable_concurrent (hardened): ops/thread={} repeats={} (1st=warmup) valsize={}B",
        ops_per_thread, repeats, valsize
    );
    println!("configs (interleaved A/B): {:?}", configs.iter().map(|c| c.label()).collect::<Vec<_>>());
    println!(
        "\n{:>8} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "threads", "config", "median o/s", "min", "max", "vs[0]"
    );

    for &nthreads in &threads {
        // samples[config_index] = Vec of ops/s across repeats (excluding warmup)
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
        for rep in 0..repeats {
            for (ci, cfg) in configs.iter().enumerate() {
                let ops_s = run_point(*cfg, nthreads, ops_per_thread, valsize);
                if rep > 0 {
                    samples[ci].push(ops_s); // discard rep 0 (warmup)
                }
            }
        }
        let med0 = median(&samples[0]);
        for (ci, cfg) in configs.iter().enumerate() {
            let s = &samples[ci];
            let med = median(s);
            let mn = s.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = s.iter().cloned().fold(0.0_f64, f64::max);
            let ratio = if med0 > 0.0 { med / med0 } else { 0.0 };
            println!(
                "{:>8} {:>10} {:>12.0} {:>12.0} {:>12.0} {:>9.2}x",
                if ci == 0 { nthreads.to_string() } else { String::new() },
                cfg.label(),
                med,
                mn,
                mx,
                ratio
            );
        }
    }
}
