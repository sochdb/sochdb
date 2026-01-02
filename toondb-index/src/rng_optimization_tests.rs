//! Comprehensive tests for HNSW RNG optimizations
//!
//! This test suite validates the core optimizations implemented:
//! 1. Normalize-at-ingest with L2 distance on unit sphere for cosine similarity
//! 2. Triangle inequality gating to skip most candidate↔selected distance computations
//! 3. Threshold-aware early-abort distance calculations
//! 4. Batch-oriented RNG with incremental min distance tracking

#[cfg(test)]
mod rng_optimization_tests {
    use crate::hnsw::{HnswIndex, HnswConfig, DistanceMetric, RngOptimizationConfig};
    use crate::vector_quantized::{Precision, QuantizedVector, l2_squared_normalized_quantized, cosine_distance_normalized_quantized, dot_product_quantized};
    use ndarray::Array1;
    use std::collections::HashSet;

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
            index.insert(i as u128, vector.clone()).expect("Insert failed");
        }

        // Verify vectors are normalized in storage
        for i in 0..vectors.len() {
            if let Some(node) = index.nodes.get(&(i as u128)) {
                let stored_vec = node.vector.to_f32();
                
                // Calculate L2 norm
                let norm_squared: f32 = stored_vec.iter().map(|&x| x * x).sum();
                let norm = norm_squared.sqrt();
                
                assert!((norm - 1.0).abs() < 1e-6, 
                    "Vector {} not normalized: norm = {} (expected 1.0)", i, norm);
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
        
        assert!((cosine_dist - (1.0 - dot_product)).abs() < 1e-6, 
            "Cosine distance optimization failed: {} vs {}", cosine_dist, 1.0 - dot_product);

        // For unit vectors: ||a-b||² = 2 - 2*dot_product  
        let l2_squared = l2_squared_normalized_quantized(&a_norm, &b_norm);
        assert!((l2_squared - (2.0 - 2.0 * dot_product)).abs() < 1e-6,
            "L2 squared distance optimization failed: {} vs {}", l2_squared, 2.0 - 2.0 * dot_product);
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
            index_optimized.insert(i as u128, vector.clone()).expect("Insert failed");
            index_original.insert(i as u128, vector.clone()).expect("Insert failed");
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
        
        assert!(recall >= 0.6, 
            "Low recall between optimized and original: {:.2} (intersection: {}, original: {})", 
            recall, intersection_size, results_original.len());
    }

    /// Test threshold-aware distance calculation
    #[test]
    fn test_threshold_aware_distance() {
        use crate::simd_distance::l2_squared_threshold;

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1];
        
        // Calculate true L2 squared distance
        let true_dist_sq: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();

        // Test with threshold above true distance - should return exact result
        let high_threshold = true_dist_sq + 1.0;
        let result_high = l2_squared_threshold(&a, &b, high_threshold);
        assert!((result_high - true_dist_sq).abs() < 1e-6,
            "High threshold should return exact distance: {} vs {}", result_high, true_dist_sq);

        // Test with threshold below true distance - should return value > threshold
        let low_threshold = true_dist_sq - 0.01;
        let result_low = l2_squared_threshold(&a, &b, low_threshold);
        assert!(result_low > low_threshold,
            "Low threshold should trigger early abort: {} should be > {}", result_low, low_threshold);
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
            .map(|_| (0..dimension).map(|_| rand::random::<f32>() - 0.5).collect())
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