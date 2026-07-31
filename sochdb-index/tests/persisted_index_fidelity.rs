// SPDX-License-Identifier: AGPL-3.0-or-later
//
// SochDB - A high-performance vector database
// Copyright (C) 2026 Sushanth Reddy Vanagala
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! What a persisted index must still be after it is loaded back.
//!
//! Persistence is easy to get almost right. A round trip that restores the
//! vectors but loses the configuration produces an index that works -- it
//! accepts queries, returns neighbours, and reports plausible scores -- while
//! ranking by a metric nobody asked for. Nothing about the result set says so.
//!
//! These tests therefore check the *settings* as well as the contents, and
//! check the answers rather than just the node count, because a restored index
//! that has the right number of nodes and the wrong graph is the failure that
//! silent persistence bugs actually produce.

use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};

/// Deterministic, well-spread vectors so a comparison is about the index.
fn corpus(dimension: usize, count: usize) -> Vec<(u128, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let vector = (0..dimension)
                .map(|d| ((i * 7 + d * 13) % 61) as f32 / 61.0)
                .collect();
            (i as u128, vector)
        })
        .collect()
}

fn build(dimension: usize, metric: DistanceMetric, rows: &[(u128, Vec<f32>)]) -> HnswIndex {
    let index = HnswIndex::new(
        dimension,
        HnswConfig {
            metric,
            ..HnswConfig::default()
        },
    );
    index.insert_batch(rows).unwrap();
    index
}

/// The accessors must report what the index was actually built with.
///
/// They exist so a caller can verify a restored index against the schema it
/// expected, which is only useful if they read the live configuration rather
/// than a default.
#[test]
fn the_accessors_report_the_configuration_the_index_was_built_with() {
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let index = build(11, metric, &corpus(11, 20));
        assert_eq!(index.dimension(), 11, "{metric:?}: wrong dimension");
        assert_eq!(index.metric(), metric, "{metric:?}: wrong metric");
    }
}

/// A round trip must preserve the metric, not just the vectors.
///
/// If the configuration is not part of the payload, a restored index falls back
/// to whatever the default is and silently ranks by the wrong notion of
/// nearness. The scores stay well-formed, so nothing downstream can detect it.
#[test]
fn a_saved_index_keeps_its_metric_and_dimension_across_a_round_trip() {
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let index = build(9, metric, &corpus(9, 50));
        let mut bytes = Vec::new();
        index.save_to_writer(&mut bytes).unwrap();
        let restored = HnswIndex::load_from_reader(&mut bytes.as_slice()).unwrap();

        assert_eq!(
            restored.metric(),
            metric,
            "a round trip changed the metric from {metric:?} to {:?}",
            restored.metric()
        );
        assert_eq!(restored.dimension(), 9, "a round trip changed the width");
        assert_eq!(restored.len(), 50, "a round trip lost vectors");
    }
}

/// A restored index must give the same answers, with the same scores.
///
/// Node count is not enough: an index whose graph was rebuilt from scratch, or
/// whose links were dropped, still holds every vector and still answers. The
/// only check that distinguishes a faithful restore from a plausible one is
/// putting the same query to both and comparing the results exactly.
#[test]
fn a_restored_index_answers_identically_to_the_one_that_was_saved() {
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let rows = corpus(12, 200);
        let index = build(12, metric, &rows);

        let mut bytes = Vec::new();
        index.save_to_writer(&mut bytes).unwrap();
        let restored = HnswIndex::load_from_reader(&mut bytes.as_slice()).unwrap();

        for probe in [0usize, 37, 199] {
            let before = index.search(&rows[probe].1, 10).unwrap();
            let after = restored.search(&rows[probe].1, 10).unwrap();
            assert_eq!(
                before.len(),
                after.len(),
                "{metric:?}: restored index returned a different number of neighbours"
            );
            for (i, ((id_a, score_a), (id_b, score_b))) in
                before.iter().zip(after.iter()).enumerate()
            {
                assert_eq!(
                    id_a, id_b,
                    "{metric:?}: neighbour {i} of query {probe} changed across a round trip"
                );
                assert!(
                    (score_a - score_b).abs() < 1e-6,
                    "{metric:?}: score for neighbour {i} of query {probe} changed from \
                     {score_a} to {score_b}"
                );
            }
        }
    }
}

/// Truncated or corrupt bytes must fail loudly.
///
/// A half-written snapshot is a realistic outcome of a crash during
/// checkpointing. Loading one and carrying on with whatever vectors happened to
/// be readable would give an index that is quietly missing rows, which reads as
/// "no near matches" rather than as an error.
#[test]
fn a_truncated_snapshot_is_rejected_rather_than_partially_loaded() {
    let index = build(8, DistanceMetric::Cosine, &corpus(8, 100));
    let mut bytes = Vec::new();
    index.save_to_writer(&mut bytes).unwrap();

    let truncated = &bytes[..bytes.len() / 2];
    assert!(
        HnswIndex::load_from_reader(&mut &truncated[..]).is_err(),
        "a half-written snapshot loaded successfully, so a crash during checkpoint \
         would leave a silently incomplete index"
    );

    assert!(
        HnswIndex::load_from_reader(&mut &b"not an index at all"[..]).is_err(),
        "arbitrary bytes were accepted as an index"
    );
}
