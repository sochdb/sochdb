//! End-to-end query profile against the real segment pipeline.
//!
//! The per-stage timers already in `QueryStats` are the attribution source of
//! truth. The number this probe adds is the *unattributed* remainder
//! (total - sum(stages)), because that is where the candidate union, the merge
//! and the final sort live — none of which have their own timer, and all of
//! which scale with the candidate count rather than with the result count.

use sochdb_vector::config::EngineConfig;
use sochdb_vector::query::engine::QueryEngine;
use sochdb_vector::segment::{Segment, writer::SegmentWriter};
use sochdb_vector::types::QueryParams;
use std::sync::Arc;

/// Deterministic pseudo-random vectors. A fixed generator keeps the profile
/// comparable across runs; real embeddings differ in distribution but the
/// pipeline's cost is driven by counts and byte volume, not by values.
fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = Vec::with_capacity(dim);
        let mut norm = 0.0f32;
        for _ in 0..dim {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            v.push(x);
            norm += x * x;
        }
        let inv = 1.0 / norm.sqrt().max(1e-9);
        for x in v.iter_mut() {
            *x *= inv;
        }
        out.push(v);
    }
    out
}

fn main() {
    let n_vec: usize = std::env::var("N_VEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(768);

    eprintln!("building segment: {} vectors x {} dim ...", n_vec, dim);
    let cfg = EngineConfig::with_dim(dim as u32);

    let path = std::env::temp_dir().join(format!("sochdb_prof_{}_{}.seg", n_vec, dim));
    if !path.exists() {
        let vectors = gen_vectors(n_vec, dim, 0x5eed);
        let mut w = SegmentWriter::new(cfg.clone()).expect("writer");
        for v in &vectors {
            w.add(v).expect("add");
        }
        w.build(&path).expect("build");
    }
    let seg = Segment::open(&path).expect("open");
    let mut engine = QueryEngine::new(cfg).expect("engine");
    engine.add_segment(Arc::new(seg)).expect("add segment");

    let queries = gen_vectors(64, dim, 0xC0FFEE);
    let mut params = QueryParams::default();
    if let Some(v) = std::env::var("L_A").ok().and_then(|v| v.parse().ok()) {
        params.l_a = v;
    }
    if let Some(v) = std::env::var("L_B").ok().and_then(|v| v.parse().ok()) {
        params.l_b = v;
    }

    // Warm the page cache and any lazily-built structures so the profile
    // measures steady-state query cost, not first-touch faults.
    for q in queries.iter().take(8) {
        let _ = engine.search(q, &params).expect("warmup");
    }

    let threads: usize = std::env::var("THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if threads > 0 {
        // search(&self) means throughput scales by running whole queries
        // concurrently rather than splitting one query across cores, which
        // keeps per-query cache locality intact and avoids fan-out sync.
        let engine = Arc::new(engine);
        let queries = Arc::new(queries);
        let params = params.clone();
        let per = 64usize;
        let t0 = std::time::Instant::now();
        let mut handles = Vec::new();
        for t in 0..threads {
            let e = Arc::clone(&engine);
            let q = Arc::clone(&queries);
            let p = params.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..per {
                    let _ = e.search(&q[(i + t) % q.len()], &p).expect("search");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let el = t0.elapsed().as_secs_f64();
        let total_q = threads * per;
        println!();
        println!("## Concurrent throughput ({} vectors x {} dim)", n_vec, dim);
        println!(
            "   threads={} queries={} elapsed={:.2}s",
            threads, total_q, el
        );
        println!("   aggregate QPS: {:.0}", total_q as f64 / el);
        println!(
            "   per-query latency: {:.0} us",
            el * 1e6 / (total_q as f64 / threads as f64)
        );
        return;
    }

    // Min-of-N, not mean: this box is shared and interference can only ever add
    // time, so a mean drifts with whatever else is running while the minimum
    // stays a stable lower bound on what the code itself costs.
    let mut acc = [u64::MAX; 6];
    let mut counts = [0usize; 5];
    let runs = 64;
    for i in 0..runs {
        let r = engine
            .search(&queries[i % queries.len()], &params)
            .expect("search");
        let s = &r.stats;
        acc[0] = acc[0].min(s.time_rotate_ns);
        acc[1] = acc[1].min(s.time_rdf_ns);
        acc[2] = acc[2].min(s.time_bps_ns);
        acc[3] = acc[3].min(s.time_filter_ns);
        acc[4] = acc[4].min(s.time_rerank_ns);
        acc[5] = acc[5].min(s.total_time_ns);
        counts[0] = s.rdf_candidates;
        counts[1] = s.bps_candidates;
        counts[2] = s.union_size;
        counts[3] = s.post_filter_size;
        counts[4] = s.rerank_count;
    }

    let total = acc[5] as f64;
    let attributed: u64 = acc[..5].iter().sum();
    let unattributed = total - attributed as f64;

    println!();
    println!(
        "## End-to-end query profile ({} vectors x {} dim)",
        n_vec, dim
    );
    println!(
        "   k={} l_a={} l_b={} r={}",
        params.k, params.l_a, params.l_b, params.r
    );
    println!();
    println!("{:>16} {:>12} {:>9}", "stage", "us/query", "% total");
    let names = ["rotate", "rdf", "bps", "filter", "rerank"];
    for (i, name) in names.iter().enumerate() {
        let us = acc[i] as f64 / 1000.0;
        println!(
            "{:>16} {:>12.1} {:>8.1}%",
            name,
            us,
            100.0 * us * 1000.0 / total
        );
    }
    println!(
        "{:>16} {:>12.1} {:>8.1}%   <-- union(HashSet) + merge + final sort",
        "UNATTRIBUTED",
        unattributed / 1000.0,
        100.0 * unattributed / total
    );
    println!("{:>16} {:>12.1}", "TOTAL", total / 1000.0);
    println!();
    println!("{:>16} {:>12}", "candidate flow", "per query");
    let cn = [
        "rdf_candidates",
        "bps_candidates",
        "union_size",
        "post_filter",
        "rerank_count",
    ];
    for (i, name) in cn.iter().enumerate() {
        println!("{:>16} {:>12}", name, counts[i]);
    }
    println!();
    println!("   QPS (single thread): {:.0}", 1e9 / total);
}
