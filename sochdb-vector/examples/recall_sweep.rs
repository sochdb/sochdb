//! Recall vs. candidate-budget sweep.
//!
//! The pipeline's defaults rerank ~24_000 candidates to return k=10. That ratio
//! is only justified if recall actually depends on it. This probe measures
//! recall@k against exact brute-force ground truth across a sweep of `l_a`
//! (RDF budget) and `l_b` (BPS budget) so the budget can be chosen from the
//! recall/latency curve instead of from a default constant.
//!
//! Two ceilings matter and are reported separately:
//!
//! * the *quantization ceiling* - quality when every vector is a candidate,
//!   which is what int8 + rotation cost on their own;
//! * the *budget quality* - what a given (l_a, l_b) achieves relative to it.
//!
//! Confusing the two makes a budget look lossy when the loss is really in the
//! quantizer.
//!
//! Quality is a score ratio rather than exact-id recall@k. On data whose top-k
//! are near-tied, exact-id recall counts a statistically identical vector as a
//! miss and so measures tie-breaking noise instead of answer quality.

use sochdb_vector::config::EngineConfig;
use sochdb_vector::query::engine::QueryEngine;
use sochdb_vector::segment::{Segment, writer::SegmentWriter};
use sochdb_vector::types::QueryParams;
use std::sync::Arc;

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

/// Exact top-k by brute-force dot product. Vectors are unit-normalised above,
/// so dot product ranks identically to cosine.
fn ground_truth(vectors: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
            (i as u32, d)
        })
        .collect();
    scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Clustered vectors: `n_clusters` random centroids, each point a centroid plus
/// small isotropic noise. Real embeddings have this structure and random unit
/// vectors do not -- in 768 dimensions independent unit vectors are nearly
/// orthogonal, so their "true" top-10 is statistically indistinguishable from
/// their top-10_000 and recall measured on them says nothing about the index.
fn gen_clustered(n: usize, dim: usize, n_clusters: usize, spread: f32, seed: u64) -> Vec<Vec<f32>> {
    let centroids = gen_vectors(n_clusters, dim, seed);
    let mut s = seed ^ 0xA5A5_A5A5;
    let mut out = Vec::with_capacity(n);
    // Per-dimension U(-a, a) noise has norm a*sqrt(dim/3). Centroids are unit
    // vectors, so `a` must be scaled by 1/sqrt(dim/3) for `spread` to mean
    // "noise norm as a fraction of the centroid norm". Without this the noise
    // swamps the signal at high dim and the data is random, not clustered.
    let a = spread * (3.0 / dim as f32).sqrt();
    for i in 0..n {
        let c = &centroids[i % n_clusters];
        let mut v = Vec::with_capacity(dim);
        let mut norm = 0.0f32;
        for &cd in c.iter().take(dim) {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = (((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * a;
            let x = cd + noise;
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
        .unwrap_or(200_000);
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(768);
    let n_query: usize = std::env::var("N_QUERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let k: usize = 10;

    let clustered = std::env::var("DATA")
        .map(|v| v == "cluster")
        .unwrap_or(false);
    let spread: f32 = std::env::var("SPREAD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.30);
    let cfg = EngineConfig::with_dim(dim as u32);
    let vectors = if clustered {
        gen_clustered(n_vec, dim, 200, spread, 0x5eed)
    } else {
        gen_vectors(n_vec, dim, 0x5eed)
    };
    eprintln!(
        "data: {}",
        if clustered {
            format!("clustered (200 clusters, noise/signal = {})", spread)
        } else {
            "uniform-random".to_string()
        }
    );

    // Cache the built segment: construction is minutes at 200k and the sweep
    // needs it unchanged across many runs.
    let path = std::env::temp_dir().join(format!(
        "sochdb_recall_{}_{}_{}.seg",
        n_vec,
        dim,
        if clustered {
            format!("cl{}", (spread * 100.0) as u32)
        } else {
            "uni".to_string()
        }
    ));
    if !path.exists() {
        eprintln!("building segment: {} x {} ...", n_vec, dim);
        let mut w = SegmentWriter::new(cfg.clone()).expect("writer");
        for v in &vectors {
            w.add(v).expect("add");
        }
        w.build(&path).expect("build");
    } else {
        eprintln!("reusing cached segment {}", path.display());
    }
    let seg = Segment::open(&path).expect("open");
    let mut engine = QueryEngine::new(cfg).expect("engine");
    engine.add_segment(Arc::new(seg)).expect("add segment");

    let queries = if clustered {
        gen_clustered(n_query, dim, 200, spread, 0x5eed ^ 0xBEEF)
    } else {
        gen_vectors(n_query, dim, 0xC0FFEE)
    };
    eprintln!("computing exact ground truth for {} queries ...", n_query);
    let truth_scores: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| {
            vectors
                .iter()
                .map(|v| v.iter().zip(q).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect();
    let truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| ground_truth(&vectors, q, k))
        .collect();

    if std::env::var("DEBUG").is_ok() {
        // Distinguish "returns wrong vectors" from "returns equally-good
        // vectors": if the pipeline's picks have true scores close to the true
        // top-10, recall@10 understates quality and the ceiling is a tie
        // artifact. If their true ranks are deep, scoring is genuinely wrong.
        let params = QueryParams {
            k,
            l_a: n_vec,
            l_b: n_vec,
            r: 500,
            adaptive: false,
            filter: None,
        };
        let exact = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        for (qi, q) in queries.iter().take(3).enumerate() {
            let r = engine.search(q, &params).expect("search");
            let mut all: Vec<f32> = vectors.iter().map(|v| exact(v, q)).collect();
            let mut sorted = all.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let rank_of = |sc: f32| sorted.iter().position(|&x| x <= sc).unwrap_or(all.len());
            println!("\n-- query {} --", qi);
            println!(
                "   true top-10 scores: {:?}",
                sorted
                    .iter()
                    .take(10)
                    .map(|x| (x * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
            println!("   returned (id, pipeline_score, true_score, true_rank):");
            for c in r.candidates.iter().take(10) {
                let ts = all[c.id as usize];
                println!(
                    "      {:>7}  {:>10.4}  {:>8.4}  rank {}",
                    c.id,
                    c.score,
                    ts,
                    rank_of(ts)
                );
            }
            all.clear();
        }
        return;
    }

    let budgets: Vec<(usize, usize)> = vec![
        (n_vec, n_vec),
        (5000, 20000),
        (4000, 12000),
        (3000, 8000),
        (2000, 5000),
        (1500, 3000),
        (1000, 2000),
        (750, 1500),
        (500, 1000),
        (300, 600),
        (200, 400),
        (100, 200),
    ];

    println!();
    println!(
        "## Score-quality@{} vs candidate budget ({} vectors x {} dim, {} queries)",
        k, n_vec, dim, n_query
    );
    println!();
    println!(
        "{:>7} {:>7} {:>10} {:>9} {:>9} {:>10} {:>8}",
        "l_a", "l_b", "cands", "quality", "vs ceil", "us/query", "QPS"
    );

    let mut ceiling = 0.0f64;
    for (bi, &(l_a, l_b)) in budgets.iter().enumerate() {
        let params = QueryParams {
            k,
            l_a,
            l_b,
            r: 500,
            adaptive: false,
            filter: None,
        };

        for q in queries.iter().take(3) {
            let _ = engine.search(q, &params).expect("warmup");
        }

        let mut hits = 0usize;
        let mut cands = 0usize;
        let mut best_ns = u64::MAX;
        let mut quality = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            let r = engine.search(q, &params).expect("search");
            let got: Vec<u32> = r.candidates.iter().map(|x| x.id).collect();
            hits += truth[qi].iter().filter(|t| got.contains(t)).count();
            // Exact-id recall counts a miss even when the returned vector is
            // statistically identical to the one it displaced, which is the
            // common case whenever the top-k are tied. Score ratio measures the
            // similarity actually delivered, so it degrades only when the
            // answer is genuinely worse.
            let got_sum: f64 = got
                .iter()
                .map(|&i| truth_scores[qi][i as usize] as f64)
                .sum();
            let best_sum: f64 = truth[qi]
                .iter()
                .map(|&i| truth_scores[qi][i as usize] as f64)
                .sum();
            quality += got_sum / best_sum.max(1e-9);
            cands += r.stats.post_filter_size;
            best_ns = best_ns.min(r.stats.total_time_ns);
        }

        let recall = quality / n_query as f64;
        let exact = hits as f64 / (n_query * k) as f64;
        let _ = exact;
        if bi == 0 {
            ceiling = recall;
        }
        let us = best_ns as f64 / 1000.0;
        println!(
            "{:>7} {:>7} {:>10} {:>8.1}% {:>8.1}% {:>10.1} {:>8.0}",
            l_a,
            l_b,
            cands / n_query,
            100.0 * recall,
            100.0 * recall / ceiling.max(1e-9),
            us,
            1e6 / us
        );
    }
    println!();
    println!("   row 1 is the quantization ceiling (every vector a candidate).");
}
