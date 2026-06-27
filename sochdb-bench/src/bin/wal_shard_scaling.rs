//! WAL fsync-stream scaling validation.
//!
//! The audit (`map-durable-write-serialization`) found that on the live commit
//! path the ONLY lock held across fsync is the single `TxnWal.writer` mutex
//! (txn_wal.rs:626, sync_all at 1050), one per DB — everything else (MVCC,
//! commit_ts, memtable) is lock-free / per-key and released before durability.
//! So the hypothesis for iteration 3 is: split the WAL into N INDEPENDENT
//! TxnWal instances (N files, N writer mutexes, N fsync streams) and aggregate
//! durable throughput should scale ~N× — UNTIL some other serializer dominates.
//!
//! This is the cheap, decisive validation BEFORE touching the production
//! DurableStorage commit/recovery path: it exercises raw TxnWal directly, so a
//! "yes" green-lights the real (recovery-correct) refactor and a "no" kills it.
//!
//! NOTE: deliberately uses N independent `TxnWal`, NOT the dead-code
//! `ShardedWal` (txn_wal.rs:1518), which funnels every shard through one
//! central fsync lock and so would NOT show real parallelism.
//!
//! Each worker thread does `append_sync` (append + flush + fsync) — the
//! worst-case, batching-free, purely fsync-bound durable commit record — routed
//! to shard `(thread_id % N)`. Group-commit batching is orthogonal and would
//! multiply each stream's rate; this isolates the *stream* scaling question.
//!
//! Env:
//!   WS_SHARDS   comma list of shard counts to A/B (default "1,2,4")
//!   WS_THREADS  worker threads                     (default 64)
//!   WS_OPS      append_sync ops per thread         (default 400)
//!   WS_REPEATS  timed repeats (1st discarded)      (default 4)

use std::sync::{Arc, Barrier};
use std::time::Instant;

use sochdb_storage::txn_wal::{TxnWal, TxnWalEntry};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn run_point(shards: usize, threads: usize, ops_per_thread: usize) -> f64 {
    let tmp = tempfile::TempDir::new().expect("tmp");
    // N independent WAL files, each its own writer mutex + fsync stream.
    let wals: Vec<Arc<TxnWal>> = (0..shards)
        .map(|s| {
            let path = tmp.path().join(format!("wal_{s}.log"));
            Arc::new(TxnWal::new(&path).expect("open TxnWal"))
        })
        .collect();
    let wals = Arc::new(wals);
    let barrier = Arc::new(Barrier::new(threads + 1));

    let mut handles = Vec::with_capacity(threads);
    for t in 0..threads {
        let wals = Arc::clone(&wals);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            let wal = Arc::clone(&wals[t % wals.len()]);
            barrier.wait();
            for i in 0..ops_per_thread {
                let txn_id = ((t as u64) << 40) | i as u64;
                let entry = TxnWalEntry::txn_commit(txn_id);
                wal.append_sync(&entry).expect("append_sync");
            }
        }));
    }

    barrier.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let wall = start.elapsed();
    (threads * ops_per_thread) as f64 / wall.as_secs_f64()
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
    let shard_counts: Vec<usize> = std::env::var("WS_SHARDS")
        .unwrap_or_else(|_| "1,2,4".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let threads = env_usize("WS_THREADS", 64);
    let ops = env_usize("WS_OPS", 400);
    let repeats = env_usize("WS_REPEATS", 4).max(2);

    println!(
        "wal_shard_scaling: threads={} ops/thread={} repeats={} (1st=warmup)",
        threads, ops, repeats
    );
    println!(
        "\n{:>8} {:>14} {:>12} {:>12} {:>10}",
        "shards", "median o/s", "min", "max", "vs[0]"
    );

    // Interleave shard-counts within each repeat round so thermal drift hits all
    // equally; discard the first round as warmup.
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); shard_counts.len()];
    for rep in 0..repeats {
        for (i, &n) in shard_counts.iter().enumerate() {
            let v = run_point(n, threads, ops);
            if rep > 0 {
                samples[i].push(v);
            }
        }
    }

    let med0 = median(&samples[0]);
    for (i, &n) in shard_counts.iter().enumerate() {
        let s = &samples[i];
        let med = median(s);
        let mn = s.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = s.iter().cloned().fold(0.0_f64, f64::max);
        let ratio = if med0 > 0.0 { med / med0 } else { 0.0 };
        println!(
            "{:>8} {:>14.0} {:>12.0} {:>12.0} {:>9.2}x",
            n, med, mn, mx, ratio
        );
    }
}
