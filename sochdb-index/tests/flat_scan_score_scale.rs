// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! The score a search returns must mean the same thing at every dataset size.
//!
//! `HnswIndex::search` has two implementations behind one signature: a
//! brute-force flat scan for datasets below a size threshold, and the graph
//! traversal above it. They are meant to be indistinguishable to a caller
//! except in speed.
//!
//! They were not. The flat scan carried its own copy of the distance
//! computation, and that copy had drifted from the canonical one:
//!
//! * `Euclidean` returned the **squared** distance, while the graph path
//!   returned the rooted distance;
//! * `DotProduct` returned `1 - dot`, while the graph path returned `-dot`.
//!
//! Both differences are monotonic in the true distance, so the *ordering* of
//! results was identical and every recall benchmark stayed green. Only the
//! reported score changed -- and it changed the moment a dataset grew past the
//! threshold. Any caller that had written a distance threshold, cached a score,
//! or compared a score against one computed elsewhere would silently start
//! getting different answers as its data grew, with nothing in the results to
//! indicate it.
//!
//! These tests pin the score itself rather than the ordering, because ordering
//! is exactly the property that stayed correct while the value was wrong.

use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};

/// Deterministic, unnormalised vectors. Unnormalised on purpose: normalising
/// would make `1 - dot` accidentally correct for cosine and hide one of the
/// three cases under test.
fn corpus(dimension: usize, count: usize) -> Vec<(u128, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let mut v = vec![0.0f32; dimension];
            v[i % dimension] = 1.0 + (i as f32) * 0.01;
            v[(i * 5 + 2) % dimension] += 0.25 + (i as f32) * 0.005;
            v[(i * 3 + 1) % dimension] += 0.75;
            (i as u128, v)
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// What each metric is defined to return, computed independently of the index.
fn expected(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f64 {
    match metric {
        DistanceMetric::Euclidean => a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(x - y).powi(2))
            .sum::<f64>()
            .sqrt(),
        DistanceMetric::Cosine => 1.0 - dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()),
        // The index negates so that lower is always nearer.
        DistanceMetric::DotProduct => -dot(a, b),
    }
}

fn build(metric: DistanceMetric, dimension: usize, rows: &[(u128, Vec<f32>)]) -> HnswIndex {
    let index = HnswIndex::new(
        dimension,
        HnswConfig {
            metric,
            // Small graph parameters keep the test fast; they affect which
            // neighbours are found, never what a distance means.
            max_connections: 8,
            max_connections_layer0: 16,
            ef_construction: 32,
            ef_search: 64,
            ..HnswConfig::default()
        },
    );
    let batch: Vec<(u128, Vec<f32>)> = rows.to_vec();
    index.insert_batch(&batch).expect("batch insert");
    index
}

/// The flat-scan path reports the distance the metric is defined to produce.
///
/// This is the regression: `Euclidean` here used to report the square of the
/// value asserted below.
#[test]
fn the_flat_scan_path_reports_the_defined_distance_for_every_metric() {
    let dimension = 12;
    let rows = corpus(dimension, 64);

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let index = build(metric, dimension, &rows);
        let query = &rows[9].1;
        let hits = index.search(query, 6).expect("search");
        assert!(!hits.is_empty(), "{metric:?} returned nothing");

        for (id, score) in hits {
            let stored = &rows
                .iter()
                .find(|(row_id, _)| *row_id == id)
                .expect("a hit names an inserted row")
                .1;
            let want = expected(metric, query, stored);
            assert!(
                (f64::from(score) - want).abs() < 1e-3,
                "{metric:?}: search reported {score} for id {id} but the metric is defined as \
                 {want}"
            );
        }
    }
}

/// Crossing the flat-scan threshold does not change what a score means.
///
/// The threshold is 10,000 for dimensions up to 128, so one index sits below it
/// and one above, and the same query is asked of both. Before the fix the two
/// answers differed by a square for `Euclidean` and by a constant for
/// `DotProduct`.
///
/// Marked `#[ignore]` because building more than ten thousand vectors is too
/// slow for the default run. It is the test that actually exercises both
/// implementations, so it is worth running deliberately:
/// `cargo test -p sochdb-index --test flat_scan_score_scale -- --ignored`.
#[test]
#[ignore = "builds >10k vectors to cross the flat-scan threshold; run deliberately"]
fn the_score_scale_does_not_change_when_a_dataset_crosses_the_flat_scan_threshold() {
    let dimension = 8;
    let below = corpus(dimension, 500);
    let above = corpus(dimension, 10_500);

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let query = below[3].1.clone();

        let flat = build(metric, dimension, &below);
        let graph = build(metric, dimension, &above);

        // Ask each for the distance to the same stored vector by finding it in
        // the results, rather than comparing top-k membership: the two indexes
        // hold different data, so only the scores of a shared row are
        // comparable.
        let target = below[17].0;
        let flat_score = flat
            .search(&query, 500)
            .expect("flat search")
            .into_iter()
            .find(|(id, _)| *id == target)
            .map(|(_, score)| f64::from(score));
        let graph_score = graph
            .search(&query, 500)
            .expect("graph search")
            .into_iter()
            .find(|(id, _)| *id == target)
            .map(|(_, score)| f64::from(score));

        if let (Some(flat_score), Some(graph_score)) = (flat_score, graph_score) {
            assert!(
                (flat_score - graph_score).abs() < 1e-3,
                "{metric:?}: the flat-scan path reported {flat_score} and the graph path reported \
                 {graph_score} for the same pair; a caller's distance threshold would move as its \
                 data grew"
            );
        }
    }
}
