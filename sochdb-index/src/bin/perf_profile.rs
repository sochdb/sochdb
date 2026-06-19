//! Configurable perf harness for HNSW insert/search profiling.
//! Env: PP_N (vectors), PP_DIM, PP_PHASE=insert|search|both, PP_NQ (queries), PP_K, PP_EFS
use rand::Rng;
use rand::SeedableRng;
use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let n = env_usize("PP_N", 20000);
    let dim = env_usize("PP_DIM", 768);
    let nq = env_usize("PP_NQ", 5000);
    let k = env_usize("PP_K", 10);
    let efs = env_usize("PP_EFS", 100);
    let phase = std::env::var("PP_PHASE").unwrap_or_else(|_| "both".into());
    let iters = env_usize("PP_ITERS", 1);

    eprintln!("perf_profile: N={n} dim={dim} nq={nq} k={k} efs={efs} phase={phase} iters={iters}");

    let config = HnswConfig {
        max_connections: 16,
        max_connections_layer0: 32,
        ef_construction: 200,
        ef_search: efs,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let ids: Vec<u128> = (0..n as u128).collect();
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    for _ in 0..n * dim { flat.push(rng.r#gen::<f32>()); }
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect();

    let do_insert = phase == "insert" || phase == "both";
    let do_search = phase == "search" || phase == "both";

    if phase == "search" {
        // build silently first (not profiled region for search-only run)
        eprintln!("building index (not profiled)...");
    }

    let index = HnswIndex::new(dim, config);

    if do_insert {
        let t = Instant::now();
        let _ = index.insert_batch_contiguous(&ids, &flat, dim);
        let el = t.elapsed();
        eprintln!("INSERT N={n} dim={dim} took {:?} ({:.1} vec/s)", el, n as f64 / el.as_secs_f64());
    } else {
        // need a populated index for search-only
        let _ = index.insert_batch_contiguous(&ids, &flat, dim);
        eprintln!("built index for search, len={}", index.len());
    }

    if do_search {
        let qrefs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let t = Instant::now();
        let mut sink = 0usize;
        for _ in 0..iters {
            for q in &qrefs {
                let r = index.search_with_ef(q, k, efs).unwrap();
                sink = sink.wrapping_add(r.len());
            }
        }
        let el = t.elapsed();
        let total = nq * iters;
        eprintln!("SEARCH nq={total} dim={dim} efs={efs} took {:?} ({:.1} q/s) sink={sink}", el, total as f64 / el.as_secs_f64());
    }
    eprintln!("len={}", index.len());
}
