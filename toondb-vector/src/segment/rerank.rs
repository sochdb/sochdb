//! Rerank builder and int8 quantization with outlier handling.
//!
//! Uses percentile-based symmetric quantization with separate outlier storage
//! to preserve dot product accuracy.

use crate::config::RerankConfig;
use crate::dispatch::DotI8Dispatcher;
use crate::types::*;
use half::f16;

/// Builder for rerank data (int8 embeddings + outliers)
pub struct RerankBuilder<'a> {
    config: &'a RerankConfig,
    vectors: &'a [Vec<f32>],
}

impl<'a> RerankBuilder<'a> {
    /// Create a new rerank builder
    pub fn new(config: &'a RerankConfig, rotated_vectors: &'a [Vec<f32>]) -> Self {
        Self {
            config,
            vectors: rotated_vectors,
        }
    }

    /// Build int8 embeddings with per-vector scales
    /// Returns (i8_data, scales)
    pub fn build_i8(&self) -> (Vec<i8>, Vec<f32>) {
        let n_vec = self.vectors.len();
        if n_vec == 0 {
            return (Vec::new(), Vec::new());
        }
        
        let dim = self.vectors[0].len();
        let mut i8_data = Vec::with_capacity(n_vec * dim);
        let mut scales = Vec::with_capacity(n_vec);
        
        for vec in self.vectors {
            // Find outlier indices (we'll zero them in i8)
            let outlier_indices = self.find_outlier_indices(vec);
            
            // Compute scale using percentile (excluding outliers)
            let scale = self.compute_scale(vec, &outlier_indices);
            scales.push(scale);
            
            // Quantize
            let inv_scale = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
            for (i, &v) in vec.iter().enumerate() {
                if outlier_indices.contains(&(i as u16)) {
                    // Zero out outlier positions (will be added back during rerank)
                    i8_data.push(0);
                } else {
                    let quantized = (v * inv_scale * 127.0).clamp(-127.0, 127.0) as i8;
                    i8_data.push(quantized);
                }
            }
        }
        
        (i8_data, scales)
    }

    /// Build outlier entries
    pub fn build_outliers(&self) -> Vec<OutlierEntry> {
        let n_vec = self.vectors.len();
        let num_outliers = self.config.num_outliers as usize;
        let mut outliers = Vec::with_capacity(n_vec * num_outliers);
        
        for vec in self.vectors {
            let outlier_entries = self.extract_outliers(vec);
            for entry in outlier_entries {
                outliers.push(entry);
            }
        }
        
        outliers
    }

    /// Find indices of top-o outliers by absolute value
    fn find_outlier_indices(&self, vec: &[f32]) -> Vec<DimIndex> {
        let num_outliers = self.config.num_outliers as usize;
        if num_outliers == 0 {
            return Vec::new();
        }
        
        let mut indexed: Vec<(usize, f32)> = vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        
        if indexed.len() <= num_outliers {
            return indexed.iter().map(|&(i, _)| i as DimIndex).collect();
        }
        
        indexed.select_nth_unstable_by(num_outliers - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap()
        });
        
        indexed.iter()
            .take(num_outliers)
            .map(|&(i, _)| i as DimIndex)
            .collect()
    }

    /// Compute scale using percentile-based approach
    fn compute_scale(&self, vec: &[f32], outlier_indices: &[DimIndex]) -> f32 {
        // Collect non-outlier absolute values
        let mut values: Vec<f32> = vec
            .iter()
            .enumerate()
            .filter(|&(i, _)| !outlier_indices.contains(&(i as DimIndex)))
            .map(|(_, &v)| v.abs())
            .collect();
        
        if values.is_empty() {
            return 1.0;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Use percentile
        let idx = ((values.len() as f32) * self.config.scale_percentile) as usize;
        let idx = idx.min(values.len() - 1);
        
        values[idx].max(1e-10)
    }

    /// Extract outliers with their values
    fn extract_outliers(&self, vec: &[f32]) -> Vec<OutlierEntry> {
        let num_outliers = self.config.num_outliers as usize;
        let mut entries = Vec::with_capacity(num_outliers);
        
        let mut indexed: Vec<(usize, f32)> = vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        // Sort by absolute value descending
        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        
        for &(dim_id, value) in indexed.iter().take(num_outliers) {
            entries.push(OutlierEntry::new(dim_id as DimIndex, f16::from_f32(value)));
        }
        
        // Pad with zeros if needed
        while entries.len() < num_outliers {
            entries.push(OutlierEntry::new(0, f16::from_f32(0.0)));
        }
        
        entries
    }
}

/// Reranker for computing int8 dot products with outlier correction
pub struct Reranker<'a> {
    i8_data: &'a [i8],
    scales: &'a [f32],
    outliers: &'a [OutlierEntry],
    dim: usize,
    num_outliers: usize,
}

impl<'a> Reranker<'a> {
    /// Create a new reranker
    pub fn new(
        i8_data: &'a [i8],
        scales: &'a [f32],
        outliers: &'a [OutlierEntry],
        dim: usize,
        num_outliers: usize,
    ) -> Self {
        Self {
            i8_data,
            scales,
            outliers,
            dim,
            num_outliers,
        }
    }

    /// Compute dot product score for a single candidate
    ///
    /// Uses SIMD-accelerated C++ kernels via FFI when available:
    /// - AVX2: 32 int8 ops per cycle (8x speedup for dim=768)
    /// - AVX512: 64 int8 ops per cycle (16x speedup)
    /// - NEON: 16 int8 ops per cycle (4x speedup)
    pub fn score(&self, vid: VectorId, query_i8: &[i8], query_scale: f32) -> f32 {
        // Delegate to score_with_fp32 with None for outlier query values
        // This maintains backward compatibility while the approximation is used
        self.score_with_fp32(vid, query_i8, query_scale, None)
    }
    
    /// Compute dot product score with optional fp32 query for accurate outlier computation.
    ///
    /// When `query_fp32` is provided, outlier contributions use exact fp32 values
    /// instead of reconstructing from quantized int8, reducing error from O(1/127)
    /// to floating-point epsilon.
    ///
    /// # Arguments
    /// * `vid` - Vector ID to score
    /// * `query_i8` - Quantized query vector (for main dot product)
    /// * `query_scale` - Query quantization scale
    /// * `query_fp32` - Optional original fp32 query (for accurate outlier scoring)
    pub fn score_with_fp32(
        &self, 
        vid: VectorId, 
        query_i8: &[i8], 
        query_scale: f32,
        query_fp32: Option<&[f32]>,
    ) -> f32 {
        let vid = vid as usize;
        let offset = vid * self.dim;
        
        if offset + self.dim > self.i8_data.len() {
            return f32::NEG_INFINITY;
        }
        
        let vec_i8 = &self.i8_data[offset..offset + self.dim];
        let vec_scale = self.scales[vid];
        
        // SIMD-accelerated int8 dot product via C++ FFI
        let dot_i8: i32 = DotI8Dispatcher::dot(&query_i8[..self.dim], vec_i8);
        
        // Dequantize
        let mut score = (dot_i8 as f32) * query_scale * vec_scale / (127.0 * 127.0);
        
        // Add outlier contributions
        if self.num_outliers > 0 {
            let outlier_offset = vid * self.num_outliers;
            if outlier_offset + self.num_outliers <= self.outliers.len() {
                let vec_outliers = &self.outliers[outlier_offset..outlier_offset + self.num_outliers];
                
                for outlier in vec_outliers {
                    let dim_id = outlier.dim_id as usize;
                    if dim_id < self.dim {
                        let v_val = outlier.get_value().to_f32();
                        
                        // Use fp32 query if available (accurate), otherwise approximate from int8
                        let q_val = if let Some(fp32) = query_fp32 {
                            // Exact fp32 value - no quantization error
                            fp32[dim_id]
                        } else {
                            // Approximate: reconstruct from int8 (introduces ~0.78% error per dim)
                            (query_i8[dim_id] as f32) * query_scale / 127.0
                        };
                        
                        score += q_val * v_val;
                    }
                }
            }
        }
        
        score
    }

    /// Score multiple candidates in batch
    pub fn score_batch(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        query_scale: f32,
    ) -> Vec<ScoredCandidate> {
        candidates
            .iter()
            .map(|&vid| ScoredCandidate {
                id: vid,
                score: self.score(vid, query_i8, query_scale),
            })
            .collect()
    }
    
    /// Score multiple candidates with fp32 query for accurate outlier computation
    pub fn score_batch_with_fp32(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        query_scale: f32,
        query_fp32: &[f32],
    ) -> Vec<ScoredCandidate> {
        candidates
            .iter()
            .map(|&vid| ScoredCandidate {
                id: vid,
                score: self.score_with_fp32(vid, query_i8, query_scale, Some(query_fp32)),
            })
            .collect()
    }

    /// Rerank and return top R candidates
    pub fn rerank(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        query_scale: f32,
        r: usize,
    ) -> Vec<ScoredCandidate> {
        let mut scored = self.score_batch(candidates, query_i8, query_scale);
        
        if scored.len() <= r {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            return scored;
        }
        
        scored.select_nth_unstable_by(r - 1, |a, b| {
            b.score.partial_cmp(&a.score).unwrap()
        });
        scored.truncate(r);
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        scored
    }
    
    /// Rerank with fp32 query for accurate outlier computation
    pub fn rerank_with_fp32(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        query_scale: f32,
        query_fp32: &[f32],
        r: usize,
    ) -> Vec<ScoredCandidate> {
        let mut scored = self.score_batch_with_fp32(candidates, query_i8, query_scale, query_fp32);
        
        if scored.len() <= r {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            return scored;
        }
        
        scored.select_nth_unstable_by(r - 1, |a, b| {
            b.score.partial_cmp(&a.score).unwrap()
        });
        scored.truncate(r);
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        scored
    }
}

/// Quantize a query vector for reranking
pub fn quantize_query(query: &[f32], config: &RerankConfig) -> (Vec<i8>, f32) {
    // Compute scale using percentile
    let mut abs_values: Vec<f32> = query.iter().map(|&v| v.abs()).collect();
    abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let idx = ((abs_values.len() as f32) * config.scale_percentile) as usize;
    let idx = idx.min(abs_values.len() - 1);
    let scale = abs_values[idx].max(1e-10);
    
    // Quantize
    let inv_scale = 1.0 / scale;
    let i8_data: Vec<i8> = query
        .iter()
        .map(|&v| (v * inv_scale * 127.0).clamp(-127.0, 127.0) as i8)
        .collect();
    
    (i8_data, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_build() {
        let config = RerankConfig {
            num_outliers: 4,
            percentile_quantization: true,
            scale_percentile: 0.99,
        };

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..64).map(|j| {
                    if j < 4 { (i as f32 + j as f32) * 0.1 }
                    else { (j as f32 - 32.0) * 0.01 }
                }).collect()
            })
            .collect();

        let builder = RerankBuilder::new(&config, &vectors);
        let (i8_data, scales) = builder.build_i8();
        let outliers = builder.build_outliers();

        assert_eq!(i8_data.len(), 100 * 64);
        assert_eq!(scales.len(), 100);
        assert_eq!(outliers.len(), 100 * 4);
    }

    #[test]
    fn test_dot_product() {
        let config = RerankConfig {
            num_outliers: 2,
            percentile_quantization: true,
            scale_percentile: 0.99,
        };

        // Create orthogonal-ish vectors
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        let builder = RerankBuilder::new(&config, &vectors);
        let (i8_data, scales) = builder.build_i8();
        let outliers = builder.build_outliers();

        let reranker = Reranker::new(&i8_data, &scales, &outliers, 4, 2);

        // Query similar to first vector
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let (q_i8, q_scale) = quantize_query(&query, &config);

        let score0 = reranker.score(0, &q_i8, q_scale);
        let score1 = reranker.score(1, &q_i8, q_scale);
        let score2 = reranker.score(2, &q_i8, q_scale);

        // Vector 0 should have highest score (most similar to query)
        assert!(score0 > score1);
        assert!(score0 > score2);
    }
}
