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

        let mut indexed: Vec<(usize, f32)> =
            vec.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();

        if indexed.len() <= num_outliers {
            return indexed.iter().map(|&(i, _)| i as DimIndex).collect();
        }

        indexed.select_nth_unstable_by(num_outliers - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .iter()
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

        let mut indexed: Vec<(usize, f32)> = vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();

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
                let vec_outliers =
                    &self.outliers[outlier_offset..outlier_offset + self.num_outliers];

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
        self.score_batch_inner(candidates, query_i8, query_scale, None)
    }

    /// Score multiple candidates with fp32 query for accurate outlier computation
    pub fn score_batch_with_fp32(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        query_scale: f32,
        query_fp32: &[f32],
    ) -> Vec<ScoredCandidate> {
        self.score_batch_inner(candidates, query_i8, query_scale, Some(query_fp32))
    }

    /// Shared batch scoring.
    ///
    /// The int8 dot products are taken in one pass so the kernel can hoist the
    /// per-query work out of the candidate loop and prefetch across candidates;
    /// scoring one candidate at a time forfeits both. Dequantization and the
    /// outlier corrections stay per-candidate and unchanged, so the result is
    /// identical to calling `score_with_fp32` in a loop.
    fn score_batch_inner(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        query_scale: f32,
        query_fp32: Option<&[f32]>,
    ) -> Vec<ScoredCandidate> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // A candidate whose vector is not fully present cannot be scored by the
        // batch kernel, which indexes without bounds checks. Those are rare
        // enough (only truncated or corrupt segments produce them) that routing
        // them through the scalar path keeps the fast path branch-free.
        let max_vid = self.i8_data.len() / self.dim.max(1);
        if candidates.iter().any(|&v| (v as usize) >= max_vid) {
            return candidates
                .iter()
                .map(|&vid| ScoredCandidate {
                    id: vid,
                    score: self.score_with_fp32(vid, query_i8, query_scale, query_fp32),
                })
                .collect();
        }

        let mut dots = vec![0i32; candidates.len()];
        crate::simd::dot_i8::dot_i8_indexed(
            &query_i8[..self.dim],
            self.i8_data,
            candidates,
            self.dim,
            &mut dots,
        );

        let denom = 127.0 * 127.0;
        candidates
            .iter()
            .zip(dots.iter())
            .map(|(&vid, &dot)| {
                let v = vid as usize;
                let mut score = (dot as f32) * query_scale * self.scales[v] / denom;
                score += self.outlier_correction(v, query_i8, query_scale, query_fp32);
                ScoredCandidate { id: vid, score }
            })
            .collect()
    }

    /// Outlier contribution for one vector, factored out so the batch and
    /// single-candidate paths cannot drift apart.
    fn outlier_correction(
        &self,
        vid: usize,
        query_i8: &[i8],
        query_scale: f32,
        query_fp32: Option<&[f32]>,
    ) -> f32 {
        if self.num_outliers == 0 {
            return 0.0;
        }
        let outlier_offset = vid * self.num_outliers;
        if outlier_offset + self.num_outliers > self.outliers.len() {
            return 0.0;
        }

        let mut acc = 0.0f32;
        for outlier in &self.outliers[outlier_offset..outlier_offset + self.num_outliers] {
            let dim_id = outlier.dim_id as usize;
            if dim_id < self.dim {
                let v_val = outlier.get_value().to_f32();
                let q_val = match query_fp32 {
                    Some(fp32) => fp32[dim_id],
                    None => (query_i8[dim_id] as f32) * query_scale / 127.0,
                };
                acc += q_val * v_val;
            }
        }
        acc
    }

    /// Dimensions scored in the pruning pass.
    ///
    /// 128 bytes is exactly two cache lines, so a pruning pass reads two lines
    /// per record instead of the twelve a 768-dim record needs.
    const PREFILTER_DIMS: usize = 64;
    /// Candidates kept per requested result.
    const PREFILTER_OVERSAMPLE: usize = 4;
    /// Floor on survivors, so a small `r` still leaves a wide net.
    const PREFILTER_MIN_KEEP: usize = 2048;
    /// Only prune when the candidate set is this many times the survivor count;
    /// below that the extra pass costs more than the scoring it removes.
    const PREFILTER_MIN_RATIO: usize = 3;

    /// Cheaply narrow a large candidate set before full scoring.
    ///
    /// Segments store Hadamard-rotated vectors, so vector energy is spread
    /// evenly over the dimensions and the dot product of a dimension prefix is
    /// an unbiased estimate of the full dot product. That makes
    /// `prefix_dot * scale` a usable ranking key: `query_scale / 127^2` is a
    /// positive constant across candidates and does not affect the order, while
    /// `scale` varies per vector and must be included.
    ///
    /// The estimate is far too coarse to pick the final top-k - it only has to
    /// be good enough that the true winners survive a net
    /// `PREFILTER_OVERSAMPLE` times wider than the result set. Full scoring
    /// still decides the ranking, so this trades a small recall risk for
    /// reading roughly a sixth of the bytes.
    ///
    /// Returns `None` when pruning does not apply, in which case the caller
    /// scores the original set.
    fn prefilter(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        r: usize,
    ) -> Option<Vec<VectorId>> {
        // Below this there is no tail worth skipping.
        if self.dim < Self::PREFILTER_DIMS * 2 || query_i8.len() < Self::PREFILTER_DIMS {
            return None;
        }
        let keep = (r * Self::PREFILTER_OVERSAMPLE).max(Self::PREFILTER_MIN_KEEP);
        if candidates.len() < keep.saturating_mul(Self::PREFILTER_MIN_RATIO) {
            return None;
        }
        // The prefix kernel indexes without bounds checks.
        let max_vid = self.i8_data.len() / self.dim.max(1);
        if candidates.iter().any(|&v| (v as usize) >= max_vid) {
            return None;
        }

        let mut dots = vec![0i32; candidates.len()];
        crate::simd::dot_i8::dot_i8_indexed_prefix(
            &query_i8[..Self::PREFILTER_DIMS],
            self.i8_data,
            candidates,
            self.dim,
            Self::PREFILTER_DIMS,
            &mut dots,
        );

        let mut keyed: Vec<(VectorId, f32)> = candidates
            .iter()
            .zip(dots.iter())
            .map(|(&vid, &dot)| (vid, dot as f32 * self.scales[vid as usize]))
            .collect();
        keyed.select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
        keyed.truncate(keep);

        let mut kept: Vec<VectorId> = keyed.into_iter().map(|(vid, _)| vid).collect();
        // Selection leaves survivors in arbitrary order; full scoring gathers
        // 768-byte records, so restore ascending ids to keep that pass
        // sequential.
        kept.sort_unstable();
        Some(kept)
    }

    /// Rerank and return top R candidates
    pub fn rerank(
        &self,
        candidates: &[VectorId],
        query_i8: &[i8],
        query_scale: f32,
        r: usize,
    ) -> Vec<ScoredCandidate> {
        let pruned = self.prefilter(candidates, query_i8, r);
        let candidates = pruned.as_deref().unwrap_or(candidates);
        let mut scored = self.score_batch(candidates, query_i8, query_scale);

        if scored.len() <= r {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            return scored;
        }

        scored.select_nth_unstable_by(r - 1, |a, b| b.score.partial_cmp(&a.score).unwrap());
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
        let pruned = self.prefilter(candidates, query_i8, r);
        let candidates = pruned.as_deref().unwrap_or(candidates);
        let mut scored = self.score_batch_with_fp32(candidates, query_i8, query_scale, query_fp32);

        if scored.len() <= r {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            return scored;
        }

        scored.select_nth_unstable_by(r - 1, |a, b| b.score.partial_cmp(&a.score).unwrap());
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
                (0..64)
                    .map(|j| {
                        if j < 4 {
                            (i as f32 + j as f32) * 0.1
                        } else {
                            (j as f32 - 32.0) * 0.01
                        }
                    })
                    .collect()
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
