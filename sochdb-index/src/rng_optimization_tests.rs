//! Comprehensive tests for HNSW RNG optimizations
//!
//! This test suite validates the core optimizations implemented:
//! 1. Normalize-at-ingest with L2 distance on unit sphere for cosine similarity
//! 2. Triangle inequality gating to skip most candidate↔selected distance computations
//! 3. Threshold-aware early-abort distance calculations
//! 4. Batch-oriented RNG with incremental min distance tracking

#[cfg(test)]
mod rng_optimization_tests {
    use crate::hnsw::{DistanceMetric, HnswConfig, HnswIndex, RngOptimizationConfig};
    use crate::vector_quantized::{
        Precision, QuantizedVector, cosine_distance_normalized_quantized, dot_product_quantized,
        l2_squared_normalized_quantized,
    };
    use ndarray::Array1;
    use std::collections::HashSet;

    /// Lock the HnswConfig speed/recall presets so their values cannot drift
    /// silently (the recall_latency characterization is keyed to them).
    #[test]
    fn test_config_presets() {
        let d = HnswConfig::default();
        let hr = HnswConfig::high_recall();
        // Default IS the high-recall preset (the safe, deep-1M-tuned config).
        assert_eq!(hr.max_connections_layer0, 64);
        assert_eq!(hr.ef_construction, 256);
        assert_eq!(d.max_connections_layer0, hr.max_connections_layer0);
        assert_eq!(d.ef_construction, hr.ef_construction);

        let b = HnswConfig::balanced();
        assert_eq!(
            (
                b.max_connections,
                b.max_connections_layer0,
                b.ef_construction
            ),
            (16, 32, 200)
        );

        let f = HnswConfig::fast();
        assert_eq!(
            (
                f.max_connections,
                f.max_connections_layer0,
                f.ef_construction
            ),
            (12, 24, 128)
        );

        // Strictly decreasing graph degree: fast < balanced < high_recall.
        assert!(f.max_connections_layer0 < b.max_connections_layer0);
        assert!(b.max_connections_layer0 < hr.max_connections_layer0);
        // level_multiplier stays consistent with m (1/ln(m)).
        assert!((b.level_multiplier - 1.0 / (16.0_f32).ln()).abs() < 1e-6);
        assert!((f.level_multiplier - 1.0 / (12.0_f32).ln()).abs() < 1e-6);
    }

    /// Test vector normalization during ingestion
    #[test]
    fn test_normalize_at_ingest() {
        let config = HnswConfig {
            metric: DistanceMetric::Cosine,
            rng_optimization: RngOptimizationConfig {
                normalize_at_ingest: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let index = HnswIndex::new(3, config);

        // Insert some non-unit vectors
        let vectors = vec![
            vec![3.0, 4.0, 0.0], // length = 5
            vec![1.0, 0.0, 0.0], // length = 1 (already normalized)
            vec![2.0, 2.0, 1.0], // length = 3
        ];

        for (i, vector) in vectors.iter().enumerate() {
            index
                .insert(i as u128, vector.clone())
                .expect("Insert failed");
        }

        // Verify vectors are normalized in storage
        for i in 0..vectors.len() {
            if let Some(node) = index.nodes.get(&(i as u128)) {
                let stored_vec = node.vector.to_f32();

                // Calculate L2 norm
                let norm_squared: f32 = stored_vec.iter().map(|&x| x * x).sum();
                let norm = norm_squared.sqrt();

                assert!(
                    (norm - 1.0).abs() < 1e-6,
                    "Vector {} not normalized: norm = {} (expected 1.0)",
                    i,
                    norm
                );
            }
        }
    }

    /// Test that optimized distance functions give equivalent results
    #[test]
    fn test_optimized_distance_equivalence() {
        // Create test vectors
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        // Test normalized distance equivalence
        let a_norm = QuantizedVector::from_f32_normalized(a.clone(), Precision::F32);
        let b_norm = QuantizedVector::from_f32_normalized(b.clone(), Precision::F32);

        // For unit vectors: cosine_distance = 1 - dot_product
        let dot_product = crate::vector_quantized::dot_product_quantized(&a_norm, &b_norm);
        let cosine_dist = cosine_distance_normalized_quantized(&a_norm, &b_norm);

        assert!(
            (cosine_dist - (1.0 - dot_product)).abs() < 1e-6,
            "Cosine distance optimization failed: {} vs {}",
            cosine_dist,
            1.0 - dot_product
        );

        // For unit vectors: ||a-b||² = 2 - 2*dot_product
        let l2_squared = l2_squared_normalized_quantized(&a_norm, &b_norm);
        assert!(
            (l2_squared - (2.0 - 2.0 * dot_product)).abs() < 1e-6,
            "L2 squared distance optimization failed: {} vs {}",
            l2_squared,
            2.0 - 2.0 * dot_product
        );
    }

    /// Regression test for the AVX2/FMA runtime-dispatch guard (HD-7).
    ///
    /// The dimension-specialized inline kernels execute AVX2+FMA intrinsics with
    /// no per-call feature check; `calculate_distance` only enters them when
    /// `dim_specialized_kernels_available()` is true. This forces that guard
    /// false — the path an x86_64 CPU without AVX2/FMA takes — for every
    /// specialized dimension and metric, asserting the generic fallback (a) does
    /// not panic / SIGILL and (b) matches the native specialized path to within
    /// floating-point reordering error. Runs on every host, so the fallback is
    /// covered even on AVX2-capable CI runners.
    #[test]
    fn test_distance_fallback_matches_specialized_kernels() {
        use crate::hnsw::FORCE_GENERIC_DISTANCE;

        // Deterministic pseudo-random vector with components in [-0.5, 0.5).
        let make = |dim: usize, seed: u64| -> QuantizedVector {
            let mut s = seed;
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect();
            QuantizedVector::F32(Array1::from_vec(v))
        };

        // RAII guard so the thread-local resets even if an assertion panics.
        struct ForceGeneric;
        impl ForceGeneric {
            fn on() -> Self {
                FORCE_GENERIC_DISTANCE.with(|f| f.set(true));
                ForceGeneric
            }
        }
        impl Drop for ForceGeneric {
            fn drop(&mut self) {
                FORCE_GENERIC_DISTANCE.with(|f| f.set(false));
            }
        }

        let dims = [128usize, 256, 384, 512, 768, 1024, 1536, 3072];
        let metrics = [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
        ];

        for &dim in &dims {
            let a = make(dim, 0x1234_5678);
            let b = make(dim, 0x9abc_def0);
            for &metric in &metrics {
                let config = HnswConfig {
                    metric,
                    // Disable the normalized fast path so calculate_distance
                    // reaches the dimension-specialized kernel dispatch.
                    rng_optimization: RngOptimizationConfig {
                        normalize_at_ingest: false,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let index = HnswIndex::new(dim, config);

                // Native path: inline AVX2 on x86+AVX2, NEON on aarch64.
                let native = index.calculate_distance(&a, &b);

                // Forced generic fallback: the AVX2-absent x86_64 path.
                let fallback = {
                    let _g = ForceGeneric::on();
                    index.calculate_distance(&a, &b)
                };

                assert!(
                    native.is_finite() && fallback.is_finite(),
                    "non-finite distance: dim={dim} metric={metric:?} native={native} fallback={fallback}"
                );
                let tol = 1e-3 * (1.0 + native.abs().max(fallback.abs()));
                assert!(
                    (native - fallback).abs() <= tol,
                    "fallback diverges from specialized kernel: dim={dim} metric={metric:?} \
                     native={native} fallback={fallback} tol={tol}"
                );
            }
        }
    }

    /// Test that triangle inequality gating produces same results as original RNG
    #[test]
    fn test_triangle_inequality_equivalence() {
        let config_optimized = HnswConfig {
            metric: DistanceMetric::Cosine,
            ef_construction: 20,
            max_connections: 8,
            rng_optimization: RngOptimizationConfig {
                normalize_at_ingest: true,
                triangle_inequality_gating: true,
                early_abort_distance: true,
                batch_oriented_rng: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let config_original = HnswConfig {
            metric: DistanceMetric::Cosine,
            ef_construction: 20,
            max_connections: 8,
            rng_optimization: RngOptimizationConfig {
                normalize_at_ingest: true,
                triangle_inequality_gating: false,
                early_abort_distance: false,
                batch_oriented_rng: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let index_optimized = HnswIndex::new(4, config_optimized);
        let index_original = HnswIndex::new(4, config_original);

        // Insert same vectors into both indices
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];

        for (i, vector) in vectors.iter().enumerate() {
            index_optimized
                .insert(i as u128, vector.clone())
                .expect("Insert failed");
            index_original
                .insert(i as u128, vector.clone())
                .expect("Insert failed");
        }

        // Test search quality - both should find similar neighbors
        let query = vec![0.5, 0.5, 0.5, 0.5];
        let results_optimized = index_optimized.search(&query, 5).unwrap();
        let results_original = index_original.search(&query, 5).unwrap();

        // Check that we get reasonable recall between the two methods
        let optimized_ids: HashSet<u128> = results_optimized.iter().map(|r| r.0).collect();
        let original_ids: HashSet<u128> = results_original.iter().map(|r| r.0).collect();

        let intersection_size = optimized_ids.intersection(&original_ids).count();
        let recall = intersection_size as f32 / results_original.len().min(5) as f32;

        assert!(
            recall >= 0.6,
            "Low recall between optimized and original: {:.2} (intersection: {}, original: {})",
            recall,
            intersection_size,
            results_original.len()
        );
    }

    /// Test threshold-aware distance calculation
    #[test]
    fn test_threshold_aware_distance() {
        use crate::simd_distance::l2_squared_threshold;

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1];

        // Calculate true L2 squared distance
        let true_dist_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();

        // Test with threshold above true distance - should return exact result
        let high_threshold = true_dist_sq + 1.0;
        let result_high = l2_squared_threshold(&a, &b, high_threshold);
        assert!(
            (result_high - true_dist_sq).abs() < 1e-6,
            "High threshold should return exact distance: {} vs {}",
            result_high,
            true_dist_sq
        );

        // Test with threshold below true distance - should return value > threshold
        let low_threshold = true_dist_sq - 0.01;
        let result_low = l2_squared_threshold(&a, &b, low_threshold);
        assert!(
            result_low > low_threshold,
            "Low threshold should trigger early abort: {} should be > {}",
            result_low,
            low_threshold
        );
    }

    /// Performance comparison test (disabled by default to avoid slowing down tests)
    #[test]
    #[ignore]
    fn test_performance_improvement() {
        use std::time::Instant;

        let config_optimized = HnswConfig {
            metric: DistanceMetric::Cosine,
            ef_construction: 200,
            max_connections: 16,
            rng_optimization: RngOptimizationConfig {
                normalize_at_ingest: true,
                triangle_inequality_gating: true,
                early_abort_distance: true,
                batch_oriented_rng: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let config_original = HnswConfig {
            metric: DistanceMetric::Cosine,
            ef_construction: 200,
            max_connections: 16,
            rng_optimization: RngOptimizationConfig {
                normalize_at_ingest: false,
                triangle_inequality_gating: false,
                early_abort_distance: false,
                batch_oriented_rng: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let dimension = 768;
        let num_vectors = 1000;

        // Generate random vectors
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| {
                (0..dimension)
                    .map(|_| rand::random::<f32>() - 0.5)
                    .collect()
            })
            .collect();

        // Test optimized version
        let start = Instant::now();
        let index_optimized = HnswIndex::new(dimension, config_optimized);
        for (i, vector) in vectors.iter().enumerate() {
            index_optimized.insert(i as u128, vector.clone()).unwrap();
        }
        let optimized_time = start.elapsed();

        // Test original version
        let start = Instant::now();
        let index_original = HnswIndex::new(dimension, config_original);
        for (i, vector) in vectors.iter().enumerate() {
            index_original.insert(i as u128, vector.clone()).unwrap();
        }
        let original_time = start.elapsed();

        let speedup = original_time.as_secs_f64() / optimized_time.as_secs_f64();

        println!("Original time: {:?}", original_time);
        println!("Optimized time: {:?}", optimized_time);
        println!("Speedup: {:.2}x", speedup);

        // We expect at least some improvement
        assert!(speedup > 1.1, "Expected >1.1x speedup, got {:.2}x", speedup);
    }
}
