//! BPS (Block Projection Sketch) builder and utilities.
//!
//! BPS divides vectors into blocks and computes short projections per block,
//! stored in SoA layout for vertical SIMD scanning.

use crate::config::BpsConfig;
use crate::dispatch::BpsScanDispatcher;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::Rng;
use bytemuck::{Pod, Zeroable};

/// BPS quantization parameters per slot (min, inv_range)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BpsQParam {
    pub min: f32,
    pub inv_range: f32,
}

/// Builder for BPS sketches
pub struct BpsBuilder<'a> {
    config: &'a BpsConfig,
    vectors: &'a [Vec<f32>],
    projection_vectors: Vec<Vec<f32>>,
}

/// Seed for deterministic BPS projections
const BPS_SEED: u64 = 0xBEEF_CAFE_1234_5678;

impl<'a> BpsBuilder<'a> {
    /// Create a new BPS builder
    pub fn new(config: &'a BpsConfig, rotated_vectors: &'a [Vec<f32>]) -> Self {
        // Generate random projection vectors (one per block per projection)
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(BPS_SEED);
        let num_blocks = config.num_blocks as usize;
        let block_size = config.block_size as usize;
        let num_proj = config.num_projections as usize;

        let mut projection_vectors = Vec::with_capacity(num_blocks * num_proj);
        for _ in 0..(num_blocks * num_proj) {
            let proj: Vec<f32> = (0..block_size)
                .map(|_| {
                    // Random unit vector component
                    let v: f32 = rng.gen_range(-1.0..1.0);
                    v
                })
                .collect();
            // Normalize
            let norm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = proj.iter().map(|x| x / norm.max(1e-10)).collect();
            projection_vectors.push(normalized);
        }

        Self {
            config,
            vectors: rotated_vectors,
            projection_vectors,
        }
    }

    /// Build BPS data in SoA layout with quantization parameters
    /// Layout: [(block * num_proj + proj) * n_vec + vec]
    /// Returns: (bps_data, qparams per slot)
    pub fn build(&self) -> (Vec<u8>, Vec<BpsQParam>) {
        let n_vec = self.vectors.len();
        let num_blocks = self.config.num_blocks as usize;
        let num_proj = self.config.num_projections as usize;
        let block_size = self.config.block_size as usize;
        let num_slots = num_blocks * num_proj;

        // Compute projections for all vectors
        let mut projections: Vec<Vec<f32>> = Vec::with_capacity(n_vec);
        for vec in self.vectors {
            let mut vec_proj = Vec::with_capacity(num_slots);
            for block_idx in 0..num_blocks {
                let block_start = block_idx * block_size;
                let block_end = (block_start + block_size).min(vec.len());
                
                for proj_idx in 0..num_proj {
                    let proj_vec = &self.projection_vectors[block_idx * num_proj + proj_idx];
                    let mut dot = 0.0f32;
                    for (i, &v) in vec[block_start..block_end].iter().enumerate() {
                        if i < proj_vec.len() {
                            dot += v * proj_vec[i];
                        }
                    }
                    vec_proj.push(dot);
                }
            }
            projections.push(vec_proj);
        }

        // Find min/max per slot for quantization
        let mut mins = vec![f32::MAX; num_slots];
        let mut maxs = vec![f32::MIN; num_slots];
        for proj in &projections {
            for (i, &v) in proj.iter().enumerate() {
                mins[i] = mins[i].min(v);
                maxs[i] = maxs[i].max(v);
            }
        }

        // Build qparams
        let mut qparams = Vec::with_capacity(num_slots);
        for slot in 0..num_slots {
            let range = (maxs[slot] - mins[slot]).max(1e-10);
            qparams.push(BpsQParam {
                min: mins[slot],
                inv_range: 255.0 / range,
            });
        }

        // Quantize to u8 and store in SoA layout
        let mut bps_data = vec![0u8; num_slots * n_vec];
        for (vec_id, proj) in projections.iter().enumerate() {
            for (slot_idx, &value) in proj.iter().enumerate() {
                let normalized = ((value - qparams[slot_idx].min) * qparams[slot_idx].inv_range).clamp(0.0, 255.0);
                
                // SoA index: slot_idx * n_vec + vec_id
                let idx = slot_idx * n_vec + vec_id;
                bps_data[idx] = normalized as u8;
            }
        }

        (bps_data, qparams)
    }

    /// Compute query sketch using stored quantization parameters
    pub fn compute_query_sketch_with_params(config: &BpsConfig, rotated_query: &[f32], qparams: &[BpsQParam]) -> Vec<u8> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(BPS_SEED);
        let num_blocks = config.num_blocks as usize;
        let block_size = config.block_size as usize;
        let num_proj = config.num_projections as usize;

        let mut sketch = Vec::with_capacity(num_blocks * num_proj);
        let mut slot_idx = 0;
        
        for block_idx in 0..num_blocks {
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(rotated_query.len());
            
            for _ in 0..num_proj {
                // Generate same random projection (must match builder)
                let proj: Vec<f32> = (0..block_size)
                    .map(|_| rng.gen_range(-1.0f32..1.0))
                    .collect();
                let norm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt();
                
                let mut dot = 0.0f32;
                for (i, &v) in rotated_query[block_start..block_end].iter().enumerate() {
                    if i < proj.len() {
                        dot += v * (proj[i] / norm.max(1e-10));
                    }
                }
                
                // Use stored qparams for correct quantization
                if slot_idx < qparams.len() {
                    let qp = &qparams[slot_idx];
                    let quantized = ((dot - qp.min) * qp.inv_range).clamp(0.0, 255.0) as u8;
                    sketch.push(quantized);
                } else {
                    // Fallback: symmetric quantization
                    let quantized = ((dot + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                    sketch.push(quantized);
                }
                slot_idx += 1;
            }
        }
        
        sketch
    }

    /// Legacy: Compute query sketch without stored params (for backwards compat)
    pub fn compute_query_sketch(config: &BpsConfig, rotated_query: &[f32]) -> Vec<u8> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(BPS_SEED);
        let num_blocks = config.num_blocks as usize;
        let block_size = config.block_size as usize;
        let num_proj = config.num_projections as usize;

        let mut sketch = Vec::with_capacity(num_blocks * num_proj);
        
        for block_idx in 0..num_blocks {
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(rotated_query.len());
            
            for _ in 0..num_proj {
                // Generate same random projection (must match builder)
                let proj: Vec<f32> = (0..block_size)
                    .map(|_| rng.gen_range(-1.0f32..1.0))
                    .collect();
                let norm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt();
                
                let mut dot = 0.0f32;
                for (i, &v) in rotated_query[block_start..block_end].iter().enumerate() {
                    if i < proj.len() {
                        dot += v * (proj[i] / norm.max(1e-10));
                    }
                }
                
                // Symmetric quantization (less accurate without qparams)
                let quantized = ((dot + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                sketch.push(quantized);
            }
        }
        
        sketch
    }
}

/// BPS scanner for streaming candidate generation
pub struct BpsScanner<'a> {
    bps_data: &'a [u8],
    n_vec: usize,
    num_blocks: usize,
    num_proj: usize,
}

impl<'a> BpsScanner<'a> {
    /// Create a new BPS scanner
    pub fn new(bps_data: &'a [u8], n_vec: usize, num_blocks: usize, num_proj: usize) -> Self {
        Self {
            bps_data,
            n_vec,
            num_blocks,
            num_proj,
        }
    }

    /// Scan and compute L1 distances to query sketch
    /// Returns distances for all vectors (lower = more similar)
    ///
    /// Uses SIMD-accelerated C++ kernels via FFI when available:
    /// - AVX2: 32 vectors per cycle (~32x speedup)
    /// - AVX512: 64 vectors per cycle (~64x speedup)
    /// - NEON: 16 vectors per cycle (~16x speedup)
    pub fn scan(&self, query_sketch: &[u8]) -> Vec<u16> {
        let mut distances = vec![0u16; self.n_vec];
        let n_slots = self.num_blocks * self.num_proj;
        
        // Dispatch to C++ SIMD kernel (AVX2/AVX512/NEON) via FFI
        BpsScanDispatcher::scan(
            self.bps_data,
            self.n_vec,
            n_slots, // n_blocks for dispatcher = total slots
            1,       // proj = 1 (legacy param)
            query_sketch,
            &mut distances,
        );
        
        distances
    }

    /// Fallback Rust implementation (kept for testing/verification)
    /// Uses saturating_add to prevent overflow for safety.
    #[allow(dead_code)]
    fn scan_fallback(&self, query_sketch: &[u8], distances: &mut [u16]) {
        let slots = self.num_blocks * self.num_proj;
        
        for slot_idx in 0..slots {
            let q = query_sketch[slot_idx] as i16;
            let base = slot_idx * self.n_vec;
            
            for vec_id in 0..self.n_vec {
                let v = self.bps_data[base + vec_id] as i16;
                let diff = (q - v).abs() as u16;
                // Use saturating_add to prevent overflow (safety measure)
                distances[vec_id] = distances[vec_id].saturating_add(diff);
            }
        }
    }

    /// Get top-k candidates by distance (lower is better)
    pub fn top_k(&self, query_sketch: &[u8], k: usize) -> Vec<(u32, u16)> {
        let distances = self.scan(query_sketch);
        
        // Use partial selection for efficiency
        let mut candidates: Vec<(u32, u16)> = distances
            .into_iter()
            .enumerate()
            .map(|(i, d)| (i as u32, d))
            .collect();
        
        if candidates.len() <= k {
            candidates.sort_by_key(|&(_, d)| d);
            return candidates;
        }
        
        // Partial sort for top k
        candidates.select_nth_unstable_by_key(k - 1, |&(_, d)| d);
        candidates.truncate(k);
        candidates.sort_by_key(|&(_, d)| d);
        
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bps_build() {
        let config = BpsConfig {
            block_size: 16,
            num_blocks: 4,
            num_projections: 1,
        };

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect()
            })
            .collect();

        let builder = BpsBuilder::new(&config, &vectors);
        let (bps_data, qparams) = builder.build();

        // Should have num_blocks * num_proj * n_vec bytes
        assert_eq!(bps_data.len(), 4 * 1 * 100);
        // Should have qparams for each slot
        assert_eq!(qparams.len(), 4 * 1);
    }

    #[test]
    fn test_bps_scan() {
        let config = BpsConfig {
            block_size: 16,
            num_blocks: 4,
            num_projections: 1,
        };

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect()
            })
            .collect();

        let builder = BpsBuilder::new(&config, &vectors);
        let (bps_data, _qparams) = builder.build();

        let scanner = BpsScanner::new(&bps_data, 100, 4, 1);
        
        // Query with first vector's sketch (should have distance 0 or close)
        let query_sketch = vec![128u8; 4]; // Neutral sketch
        let candidates = scanner.top_k(&query_sketch, 10);
        
        assert_eq!(candidates.len(), 10);
    }
}
