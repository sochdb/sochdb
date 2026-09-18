//! Recall and latency under attribute filters.
//!
//! Every other harness in this directory passes `filter: None`, so the entire
//! measured quality and speed story is the unfiltered one. Production queries
//! are rarely unfiltered: they carry a tenant id, a permission set, a category
//! or a date range. This probe measures what the pipeline actually returns when
//! only a fraction of the corpus is eligible.
//!
//! Two properties make this a different question from unfiltered recall rather
//! than a harder version of it:
//!
//! * **Ground truth must be computed inside the eligible set.** Scoring against
//!   the global top-k would charge the engine for correctly excluding
//!   ineligible vectors, which is the one thing a filter is for.
//! * **Filter geometry matters more than filter strength.** A filter drawn at
//!   random removes candidates uniformly, so a candidate list that was good
//!   before filtering is still representative after it. A filter aligned with
//!   the vector distribution -- whole clusters eligible, the rest not -- can
//!   place every eligible vector outside the candidate list no matter how good
//!   the ANN scoring was, because the pipeline filters *after* generating
//!   candidates. Real filters (tenant, language, category) are correlated, so
//!   the random case is the optimistic one and should never be quoted alone.
//!
//! Env: `N_VEC`, `DIM`, `N_QUERY`, `SPREAD`, `L_A`, `L_B`, `R`.
//!
//! # Measured result (200k x 768, clustered, spread 0.30, 30 queries)
//!
//! At the default budget (`l_a` 5_000 / `l_b` 20_000, ~24_400 candidates)
//! quality stays at 99% for random filters down to 0.1% selectivity, and
//! degrades only in the adversarial correlated case -- a single eligible
//! cluster -- to 91.7%, where it also starts failing to fill `k`. Zero filter
//! violations were observed across 300 queries, so the post-filter is sound.
//!
//! # The load-bearing finding: the candidate budget is not slack
//!
//! `recall_sweep` reports that quality is flat from ~24_000 candidates down to
//! ~300, which reads as an 82x budget reduction available for free. It is not.
//! That sweep runs unfiltered, and the budget it calls redundant is the only
//! reason filtered search works.
//!
//! At `l_a=100 l_b=200` (~300 candidates) this harness measures:
//!
//! | geometry | selectivity | quality | returned |
//! |---|---|---|---|
//! | (none)   | 1.0   | 98.7% | 10.0 |
//! | random   | 0.001 | 13.8% |  1.4 |
//! | cluster  | 0.01  |  8.9% |  0.9 |
//! | cluster  | 0.001 |  5.6% |  0.6 |
//!
//! The mechanism is that the filter is applied *after* candidate generation, so
//! a query retains roughly `budget * selectivity` usable candidates. At 300
//! candidates and 0.1% selectivity that expectation is well under one, and the
//! engine returns almost nothing -- while every other harness in this directory,
//! all of which pass `filter: None`, still reports 98.7% and no regression.
//!
//! So the budget must be cut against *this* harness, not against `recall_sweep`
//! alone. The alternative is to make the filter participate in candidate
//! generation rather than follow it, which would decouple the two.

use sochdb_vector::config::EngineConfig;
use sochdb_vector::query::engine::QueryEngine;
use sochdb_vector::segment::{Segment, writer::SegmentWriter};
use sochdb_vector::types::QueryParams;
use std::sync::Arc;
use std::time::Instant;

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Clustered vectors, matching `recall_sweep`: independent unit vectors in high
/// dimension are near-orthogonal, so their top-10 is indistinguishable from
/// their top-10_000 and recall measured on them says nothing about the index.
fn gen_clustered(
    n: usize,
    dim: usize,
    n_clusters: usize,
    spread: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut cs = seed;
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| lcg(&mut cs)).collect();
            let inv = 1.0 / v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in v.iter_mut() {
                *x *= inv;
            }
            v
        })
        .collect();
    let mut s = seed ^ 0xA5A5_A5A5;
    let a = spread * (3.0 / dim as f32).sqrt();
    let mut out = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let ci = i % n_clusters;
        let mut v = Vec::with_capacity(dim);
        for &cd in centroids[ci].iter() {
            v.push(cd + lcg(&mut s) * a);
        }
        let inv = 1.0 / v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in v.iter_mut() {
            *x *= inv;
        }
        out.push(v);
        labels.push(ci);
    }
    (out, labels)
}

/// Exact top-k restricted to `eligible`, with scores. This is the definition of
/// correctness under a filter; global top-k is the wrong reference.
fn eligible_truth(vectors: &[Vec<f32>], eligible: &[u32], q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = eligible
        .iter()
        .map(|&i| {
            let v = &vectors[i as usize];
            (i, v.iter().zip(q).map(|(a, b)| a * b).sum())
        })
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// Uniform in [0, 1). `lcg` returns a signed value whose range is only half the
/// unit interval, so sampling selectivity from it directly would silently halve
/// every requested fraction.
fn unit(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 33) as f32 / (1u64 << 31) as f32
}

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n_vec: usize = env("N_VEC", 200_000);
    let dim: usize = env("DIM", 768);
    let n_query: usize = env("N_QUERY", 50);
    let spread: f32 = env("SPREAD", 0.30);
    let l_a: usize = env("L_A", 5_000);
    let l_b: usize = env("L_B", 20_000);
    let r: usize = env("R", 500);
    let k: usize = 10;
    let n_clusters = 200;

    let (vectors, labels) = gen_clustered(n_vec, dim, n_clusters, spread, 0x5eed);

    // Share the cached segment with recall_sweep: identical generator, seed and
    // parameters, so rebuilding would produce the same bytes.
    let path = std::env::temp_dir().join(format!(
        "sochdb_recall_{}_{}_cl{}.seg",
        n_vec,
        dim,
        (spread * 100.0) as u32
    ));
    if !path.exists() {
        eprintln!("building segment: {} x {} ...", n_vec, dim);
        let mut w = SegmentWriter::new(EngineConfig::with_dim(dim as u32)).expect("writer");
        for v in &vectors {
            w.add(v).expect("add");
        }
        w.build(&path).expect("build");
    } else {
        eprintln!("reusing cached segment {}", path.display());
    }
    let seg = Segment::open(&path).expect("open");
    let mut engine = QueryEngine::new(EngineConfig::with_dim(dim as u32)).expect("engine");
    engine.add_segment(Arc::new(seg)).expect("add segment");

    let (queries, _) = gen_clustered(n_query, dim, n_clusters, spread, 0x5eed ^ 0xBEEF);

    // Two filter geometries at each selectivity. `random` samples ids
    // independently of position in vector space; `cluster` makes whole clusters
    // eligible, which is what a tenant or language filter looks like.
    let selectivities = [1.0f32, 0.5, 0.1, 0.01, 0.001];

    println!(
        "{:<9} {:<8} {:>9} {:>9} {:>9} {:>10} {:>11} {:>9}",
        "geometry", "select", "eligible", "quality", "returned", "cands", "violations", "us/query"
    );
    println!("{}", "-".repeat(82));

    for geometry in ["random", "cluster"] {
        for &sel in &selectivities {
            let eligible: Vec<u32> = if geometry == "random" {
                let mut s = 0xD00D_u64;
                (0..n_vec as u32).filter(|_| unit(&mut s) < sel).collect()
            } else {
                // Whole clusters eligible. At 0.001 this is a single cluster, so
                // the eligible set is a tight ball far from most queries.
                let keep = ((n_clusters as f32 * sel).round() as usize).max(1);
                (0..n_vec as u32)
                    .filter(|&i| labels[i as usize] < keep)
                    .collect()
            };
            if eligible.is_empty() {
                continue;
            }
            let bits = eligible.iter().map(|&i| i as u64).collect::<Vec<u64>>();
            let params = QueryParams {
                k,
                l_a,
                l_b,
                r,
                adaptive: true,
                filter: if sel >= 1.0 { None } else { Some(bits.clone()) },
            };

            let mut got_score = 0.0f64;
            let mut truth_score = 0.0f64;
            let mut returned = 0usize;
            let mut cands = 0usize;
            let mut violations = 0usize;
            let mut best_ns = u64::MAX;
            let elig_set: Vec<bool> = {
                let mut m = vec![false; n_vec];
                for &i in &eligible {
                    m[i as usize] = true;
                }
                m
            };
            for q in &queries {
                let truth = eligible_truth(&vectors, &eligible, q, k);
                let t0 = Instant::now();
                let res = engine.search(q, &params).expect("search");
                best_ns = best_ns.min(t0.elapsed().as_nanos() as u64);

                truth_score += truth.iter().map(|(_, s)| *s as f64).sum::<f64>();
                for c in res.candidates.iter().take(k) {
                    if !elig_set[c.id as usize] {
                        violations += 1;
                        continue;
                    }
                    let v = &vectors[c.id as usize];
                    got_score += v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() as f64;
                }
                returned += res.candidates.len().min(k);
                cands += res.stats.post_filter_size;
            }
            println!(
                "{:<9} {:<8} {:>9} {:>8.1}% {:>9.1} {:>10.0} {:>11} {:>9.0}",
                geometry,
                if sel >= 1.0 {
                    "none".to_string()
                } else {
                    format!("{}", sel)
                },
                eligible.len(),
                100.0 * got_score / truth_score.max(1e-9),
                returned as f32 / n_query as f32,
                cands as f32 / n_query as f32,
                violations,
                best_ns as f32 / 1000.0,
            );
        }
    }
}
