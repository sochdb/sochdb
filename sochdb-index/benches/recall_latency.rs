// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Recall@10 + search-latency gate for the HNSW hot path.
//!
//! This is the benchmark gate the HD-1 (contiguous vector arena) and HD-3
//! (single-ownership insert) refactors must pass: a memory-layout change is only
//! acceptable if it preserves recall@10 while improving (or not regressing) p95
//! query latency at the embedding dimensions those tasks target (768/1536/3072).
//!
//! Unlike the criterion benches, this uses a custom harness so it can report
//! recall@10 alongside p50/p95/p99 latency in one pass. Vectors are produced by
//! a seeded xorshift PRNG, so the numbers are reproducible run-to-run.
//!
//! Run (in-repo smoke baseline):
//!     cargo bench -p sochdb-index --bench recall_latency
//!
//! Run (the real gate — scale before drawing conclusions; the HD-1 scatter
//! penalty only surfaces at large N):
//!     RECALL_N=100000 RECALL_QUERIES=500 cargo bench -p sochdb-index --bench recall_latency
//!
//! Note: vectors are uniform-random (no cluster structure), which is a harder
//! recall regime than real embeddings; treat the absolute recall as a relative
//! baseline for before/after comparison, not as a production recall figure.

use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};
use std::time::Instant;

/// Deterministic xorshift64* PRNG — reproducible, no external dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A float in [-0.5, 0.5).
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed | 1);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.unit()).collect())
        .collect()
}

/// `count` points scattered around the given cluster centers (center + noise),
/// round-robin over centers. Approximates how real embeddings group by topic —
/// a higher-recall regime than uniform random.
fn around_centers(centers: &[Vec<f32>], count: usize, noise: f32, rng: &mut Rng) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let c = &centers[i % centers.len()];
            c.iter().map(|&x| x + noise * rng.unit()).collect()
        })
        .collect()
}

/// Cosine distance on raw vectors. Order-equivalent to the index's normalized
/// cosine, so it is a valid ground truth for recall regardless of normalization.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na < 1e-12 || nb < 1e-12 {
        1.0
    } else {
        1.0 - dot / (na.sqrt() * nb.sqrt())
    }
}

/// Exhaustive top-k ids by cosine distance (recall ground truth).
fn brute_force_topk(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine_distance(query, v), i))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = env_usize("RECALL_N", 3000);
    let q = env_usize("RECALL_QUERIES", 100);
    let k = 10usize;
    // RECALL_DIMS=768,1536 restricts the dimensions probed (faster iteration);
    // defaults to the three the HD-1/HD-3 refactors target.
    let dims: Vec<usize> = std::env::var("RECALL_DIMS")
        .ok()
        .map(|s| s.split(',').filter_map(|d| d.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![768, 1536, 3072]);

    // RECALL_DATA=clustered uses a Gaussian-mixture (realistic, higher-recall
    // regime); default "uniform" is the pessimistic random regime.
    let clustered = std::env::var("RECALL_DATA")
        .map(|s| s == "clustered")
        .unwrap_or(false);
    let clusters = env_usize("RECALL_CLUSTERS", 64).max(1);

    // RECALL_PRESET selects the HnswConfig preset under test (default = crate Default).
    let preset = std::env::var("RECALL_PRESET").unwrap_or_else(|_| "default".to_string());
    let base = match preset.as_str() {
        "high_recall" => HnswConfig::high_recall(),
        "balanced" => HnswConfig::balanced(),
        "fast" => HnswConfig::fast(),
        _ => HnswConfig::default(),
    };

    let data_label = if clustered {
        format!("clustered/{clusters}")
    } else {
        "uniform".to_string()
    };
    println!(
        "HNSW recall@{k} gate  (N={n}, q={q}, data={data_label}, preset={preset}, m0={}, ef_c={}, seeded)\n",
        base.max_connections_layer0, base.ef_construction
    );
    println!(
        "{:>6} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "dim", "recall@10", "build_ms", "p50_us", "p95_us", "p99_us", "mean_us"
    );

    for &dim in &dims {
        let (data, queries) = if clustered {
            let mut rng = Rng((0x00C0_FFEE ^ dim as u64) | 1);
            let centers: Vec<Vec<f32>> = (0..clusters)
                .map(|_| (0..dim).map(|_| rng.unit()).collect())
                .collect();
            let data = around_centers(&centers, n, 0.15, &mut rng);
            let queries = around_centers(&centers, q, 0.15, &mut rng);
            (data, queries)
        } else {
            (
                gen_vectors(n, dim, 0x00C0_FFEE ^ dim as u64),
                gen_vectors(q, dim, 0x0000_BEEF ^ dim as u64),
            )
        };

        let config = HnswConfig {
            metric: DistanceMetric::Cosine,
            ..base.clone()
        };
        let index = HnswIndex::new(dim, config);

        let batch: Vec<(u128, Vec<f32>)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u128, v.clone()))
            .collect();
        let t_build = Instant::now();
        index.insert_batch(&batch).expect("insert_batch failed");
        let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

        let mut latencies_us: Vec<f64> = Vec::with_capacity(q);
        let (mut hits, mut total) = (0usize, 0usize);
        for query in &queries {
            let truth: std::collections::HashSet<usize> =
                brute_force_topk(query, &data, k).into_iter().collect();

            let t = Instant::now();
            let res = index.search(query, k).expect("search failed");
            latencies_us.push(t.elapsed().as_secs_f64() * 1e6);

            hits += res
                .iter()
                .filter(|(id, _)| truth.contains(&(*id as usize)))
                .count();
            total += truth.len();
        }

        let recall = hits as f64 / total as f64;
        latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = latencies_us.iter().sum::<f64>() / latencies_us.len().max(1) as f64;

        println!(
            "{:>6} {:>10.4} {:>10.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
            dim,
            recall,
            build_ms,
            percentile(&latencies_us, 50.0),
            percentile(&latencies_us, 95.0),
            percentile(&latencies_us, 99.0),
            mean,
        );
    }
}
