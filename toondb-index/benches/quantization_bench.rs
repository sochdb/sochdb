// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Quantization Performance Benchmarks
//!
//! Measures the impact of vector quantization (F16/BF16) on:
//! - Memory usage
//! - Search latency
//! - Insert throughput

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use toondb_index::hnsw::{HnswConfig, HnswIndex};
use toondb_index::vector_quantized::Precision;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn generate_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| generate_random_vector(dim)).collect()
}

/// Benchmark search latency with different quantization levels
fn bench_quantization_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_search");

    let vectors = generate_test_vectors(10_000, 128);

    for precision in [Precision::F32, Precision::F16, Precision::BF16] {
        group.bench_with_input(
            BenchmarkId::new("precision", format!("{:?}", precision)),
            &precision,
            |b, &precision| {
                let config = HnswConfig {
                    quantization_precision: Some(precision),
                    ..Default::default()
                };
                let index = HnswIndex::new(128, config);

                // Insert vectors
                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(i as u128, vec.clone()).unwrap();
                }

                let query = generate_random_vector(128);

                b.iter(|| {
                    let results = index.search(&query, 10).unwrap();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark insert throughput with different quantization levels
fn bench_quantization_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_insert");

    let vectors = generate_test_vectors(5_000, 128);

    for precision in [Precision::F32, Precision::F16, Precision::BF16] {
        group.bench_with_input(
            BenchmarkId::new("precision", format!("{:?}", precision)),
            &precision,
            |b, &precision| {
                b.iter(|| {
                    let config = HnswConfig {
                        quantization_precision: Some(precision),
                        ..Default::default()
                    };
                    let index = HnswIndex::new(128, config);

                    for (i, vec) in vectors.iter().enumerate() {
                        index.insert(i as u128, vec.clone()).unwrap();
                    }

                    black_box(index);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage (reports stats, not a timing benchmark)
fn bench_quantization_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_memory");

    let vectors = generate_test_vectors(10_000, 768); // Larger vectors to see memory impact

    for precision in [Precision::F32, Precision::F16, Precision::BF16] {
        group.bench_with_input(
            BenchmarkId::new("precision", format!("{:?}", precision)),
            &precision,
            |b, &precision| {
                let config = HnswConfig {
                    quantization_precision: Some(precision),
                    ..Default::default()
                };
                let index = HnswIndex::new(768, config);

                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(i as u128, vec.clone()).unwrap();
                }

                b.iter(|| {
                    let stats = index.memory_stats();
                    black_box(stats);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quantization_search,
    bench_quantization_insert,
    bench_quantization_memory
);
criterion_main!(benches);
