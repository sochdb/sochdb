//! RDF (Rare-Dominant Fingerprint) builder and posting list handling.
//!
//! RDF uses IR-style inverted lists over a sparse fingerprint of each vector.
//! Posting lists are stored in VID-striped chunks for cache-friendly scoring.

use crate::config::RdfConfig;
use crate::types::*;
use crate::segment::format::PostingListEntry;
use std::collections::HashMap;

/// Builder for RDF posting lists
pub struct RdfBuilder<'a> {
    config: &'a RdfConfig,
    dim: u32,
    vectors: &'a [Vec<f32>],
    dim_weights: Vec<f32>,
    doc_freqs: Vec<u32>,
}

impl<'a> RdfBuilder<'a> {
    /// Create a new RDF builder
    pub fn new(config: &'a RdfConfig, dim: u32, rotated_vectors: &'a [Vec<f32>]) -> Self {
        let n_vec = rotated_vectors.len();
        let dim_usize = dim as usize;

        // Compute dimension statistics
        let mut sum = vec![0.0f64; dim_usize];
        let mut sum_sq = vec![0.0f64; dim_usize];
        let mut doc_freqs = vec![0u32; dim_usize];
        
        // Track which dims appear in each vector's top-t
        let top_t = config.top_t as usize;
        
        for vec in rotated_vectors {
            // Find top-t dims by absolute value
            let mut scored: Vec<(usize, f32)> = vec
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v.abs()))
                .collect();
            let nth_idx = top_t.min(scored.len()).saturating_sub(1);
            if nth_idx < scored.len() {
                scored.select_nth_unstable_by(nth_idx, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap()
                });
            }
            
            for &(dim_idx, _) in scored.iter().take(top_t) {
                doc_freqs[dim_idx] += 1;
            }
            
            for (i, &v) in vec.iter().enumerate() {
                sum[i] += v as f64;
                sum_sq[i] += (v * v) as f64;
            }
        }

        // Compute dimension weights: w[d] = α·idf[d] + β·sqrt(var[d])
        let n = n_vec as f64;
        let mut dim_weights = Vec::with_capacity(dim_usize);
        
        for d in 0..dim_usize {
            let mean = sum[d] / n;
            let var = (sum_sq[d] / n - mean * mean).max(0.0);
            let std_dev = var.sqrt();
            
            // IDF-like weight: log(N / df)
            let df = doc_freqs[d].max(1) as f64;
            let idf = (n / df).ln();
            
            // Combined weight
            let weight = config.idf_weight as f64 * idf 
                       + config.var_weight as f64 * std_dev;
            dim_weights.push(weight as f32);
        }

        Self {
            config,
            dim,
            vectors: rotated_vectors,
            dim_weights,
            doc_freqs,
        }
    }

    /// Get dimension weights
    pub fn dim_weights(&self) -> Vec<f32> {
        self.dim_weights.clone()
    }

    /// Build posting lists with striped storage
    /// Returns (directory, concatenated posting data)
    pub fn build(&self) -> (Vec<PostingListEntry>, Vec<u8>) {
        let dim_usize = self.dim as usize;
        let top_t = self.config.top_t as usize;
        let stripe_shift = self.config.stripe_shift;
        let _stripe_size = 1usize << stripe_shift;
        
        // Collect postings per dimension
        // Each posting: (vid, sign, magnitude)
        let mut dim_postings: Vec<Vec<(VectorId, bool, u8)>> = vec![Vec::new(); dim_usize];
        
        // Compute per-dimension magnitude scales for quantization
        let mut dim_max_mag = vec![0.0f32; dim_usize];
        
        for (_vid, vec) in self.vectors.iter().enumerate() {
            // Score each dim: |v[d]| * w[d]
            let mut scored: Vec<(usize, f32, f32)> = vec
                .iter()
                .enumerate()
                .map(|(d, &v)| (d, v.abs() * self.dim_weights[d], v))
                .collect();
            
            // Select top-t by score
            if scored.len() > top_t {
                scored.select_nth_unstable_by(top_t - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap()
                });
                scored.truncate(top_t);
            }
            
            for &(dim_idx, _, value) in &scored {
                let mag = value.abs();
                dim_max_mag[dim_idx] = dim_max_mag[dim_idx].max(mag);
            }
        }
        
        // Second pass: collect postings with quantized magnitudes
        for (vid, vec) in self.vectors.iter().enumerate() {
            let mut scored: Vec<(usize, f32, f32)> = vec
                .iter()
                .enumerate()
                .map(|(d, &v)| (d, v.abs() * self.dim_weights[d], v))
                .collect();
            
            if scored.len() > top_t {
                scored.select_nth_unstable_by(top_t - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap()
                });
                scored.truncate(top_t);
            }
            
            for &(dim_idx, _, value) in &scored {
                let sign = value >= 0.0;
                let mag = value.abs();
                let max_mag = dim_max_mag[dim_idx].max(1e-10);
                let mag8 = ((mag / max_mag) * 127.0).min(127.0) as u8;
                
                dim_postings[dim_idx].push((vid as VectorId, sign, mag8));
            }
        }
        
        // Build striped posting lists
        let mut directory = Vec::with_capacity(dim_usize);
        let mut data = Vec::new();
        
        for dim_idx in 0..dim_usize {
            let postings = &dim_postings[dim_idx];
            
            if postings.is_empty() {
                directory.push(PostingListEntry {
                    offset: data.len() as u64,
                    length: 0,
                    num_stripes: 0,
                    flags: 0,
                });
                continue;
            }
            
            let offset = data.len() as u64;
            
            // Check if this is a stop-dim
            let is_stopword = self.doc_freqs[dim_idx] > self.config.stop_dim_threshold;
            let flags = if is_stopword { PostingListEntry::FLAG_STOPWORD } else { 0 };
            
            // Group postings by stripe
            let mut stripes: HashMap<StripeId, Vec<(u8, bool, u8)>> = HashMap::new();
            for &(vid, sign, mag) in postings {
                let stripe_id = vid >> stripe_shift;
                let vid_in_stripe = (vid & ((1 << stripe_shift) - 1)) as u8;
                stripes.entry(stripe_id).or_default().push((vid_in_stripe, sign, mag));
            }
            
            // Sort stripes by stripe_id
            let mut stripe_ids: Vec<StripeId> = stripes.keys().copied().collect();
            stripe_ids.sort();
            
            // Write stripe chunks
            for stripe_id in &stripe_ids {
                let entries = stripes.get(stripe_id).unwrap();
                
                // Write stripe header
                let header = StripeChunkHeader {
                    stripe_id: *stripe_id,
                    count: entries.len() as u16,
                    _pad: 0,
                };
                data.extend_from_slice(bytemuck::bytes_of(&header));
                
                // Write entries sorted by vid_in_stripe
                let mut sorted_entries = entries.clone();
                sorted_entries.sort_by_key(|e| e.0);
                
                for (vid_in_stripe, sign, mag) in sorted_entries {
                    let posting = RdfPosting::new(vid_in_stripe, sign, mag);
                    data.extend_from_slice(bytemuck::bytes_of(&posting));
                }
            }
            
            directory.push(PostingListEntry {
                offset,
                length: postings.len() as u32,
                num_stripes: stripe_ids.len() as u16,
                flags,
            });
        }
        
        (directory, data)
    }
}

/// RDF scorer for query-time candidate generation
pub struct RdfScorer<'a> {
    directory: &'a [PostingListEntry],
    rdf_data: &'a [u8],
    dim_weights: &'a [f32],
    stripe_shift: u8,
    stripe_size: usize,
    n_vec: u32,
}

impl<'a> RdfScorer<'a> {
    /// Create a new RDF scorer
    pub fn new(
        directory: &'a [PostingListEntry],
        rdf_data: &'a [u8],
        dim_weights: &'a [f32],
        stripe_shift: u8,
        n_vec: u32,
    ) -> Self {
        Self {
            directory,
            rdf_data,
            dim_weights,
            stripe_shift,
            stripe_size: 1 << stripe_shift,
            n_vec,
        }
    }

    /// Score candidates using RDF
    /// Returns top L_A candidates by score (higher = better)
    pub fn score(&self, query: &[f32], top_t: usize, l_a: usize) -> Vec<ScoredCandidate> {
        if self.directory.is_empty() {
            return Vec::new();
        }

        let _dim = query.len();
        
        // Find top-t query dimensions by |q[d]| * w[d]
        let mut scored: Vec<(usize, f32, f32)> = query
            .iter()
            .enumerate()
            .map(|(d, &v)| {
                let w = if d < self.dim_weights.len() { self.dim_weights[d] } else { 1.0 };
                (d, v.abs() * w, v)
            })
            .collect();
        
        if scored.len() > top_t {
            scored.select_nth_unstable_by(top_t - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap()
            });
            scored.truncate(top_t);
        }
        
        // Get query dims (excluding stopwords)
        let query_dims: Vec<(usize, f32, f32)> = scored
            .into_iter()
            .filter(|&(d, _, _)| {
                d < self.directory.len() && !self.directory[d].is_stopword()
            })
            .collect();
        
        if query_dims.is_empty() {
            return Vec::new();
        }
        
        // Use stripe-based accumulation
        let num_stripes = (self.n_vec as usize + self.stripe_size - 1) / self.stripe_size;
        let mut global_candidates = Vec::new();
        
        // Process stripe by stripe for cache locality
        for stripe_id in 0..num_stripes as u32 {
            let mut stripe_acc = vec![0.0f32; self.stripe_size];
            
            for &(dim_idx, _, q_value) in &query_dims {
                let entry = &self.directory[dim_idx];
                if entry.length == 0 {
                    continue;
                }
                
                // Find and process the stripe chunk for this dimension
                self.accumulate_stripe(
                    entry,
                    stripe_id,
                    q_value,
                    self.dim_weights[dim_idx],
                    &mut stripe_acc,
                );
            }
            
            // Collect non-zero scores from this stripe
            let base_vid = stripe_id << self.stripe_shift;
            for (i, &score) in stripe_acc.iter().enumerate() {
                if score > 0.0 {
                    let vid = base_vid + i as u32;
                    if vid < self.n_vec {
                        global_candidates.push(ScoredCandidate { id: vid, score });
                    }
                }
            }
        }
        
        // Select top L_A
        if global_candidates.len() <= l_a {
            global_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            return global_candidates;
        }
        
        global_candidates.select_nth_unstable_by(l_a - 1, |a, b| {
            b.score.partial_cmp(&a.score).unwrap()
        });
        global_candidates.truncate(l_a);
        global_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        global_candidates
    }

    /// Accumulate scores for a specific stripe from one dimension's posting list
    fn accumulate_stripe(
        &self,
        entry: &PostingListEntry,
        target_stripe_id: StripeId,
        q_value: f32,
        dim_weight: f32,
        stripe_acc: &mut [f32],
    ) {
        let mut offset = entry.offset as usize;
        let header_size = std::mem::size_of::<StripeChunkHeader>();
        let posting_size = std::mem::size_of::<RdfPosting>();
        
        for _ in 0..entry.num_stripes {
            if offset + header_size > self.rdf_data.len() {
                break;
            }
            
            let header: StripeChunkHeader = unsafe {
                std::ptr::read_unaligned(self.rdf_data.as_ptr().add(offset) as *const _)
            };
            offset += header_size;
            
            let count = header.count as usize;
            
            if header.stripe_id == target_stripe_id {
                // Process this stripe
                for _ in 0..count {
                    if offset + posting_size > self.rdf_data.len() {
                        break;
                    }
                    
                    let posting: RdfPosting = unsafe {
                        std::ptr::read_unaligned(self.rdf_data.as_ptr().add(offset) as *const _)
                    };
                    offset += posting_size;
                    
                    let vid_in_stripe = posting.vid_in_stripe as usize;
                    let sign = if posting.sign() { 1.0 } else { -1.0 };
                    let mag = posting.magnitude() as f32 / 127.0;
                    
                    // Score contribution: q_value * sign * mag * weight
                    let contribution = q_value * sign * mag * dim_weight;
                    stripe_acc[vid_in_stripe] += contribution;
                }
                return;
            } else {
                // Skip this stripe
                offset += count * posting_size;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdf_build() {
        let config = RdfConfig {
            top_t: 8,
            stripe_shift: 4, // 16 vids per stripe
            stop_dim_threshold: 1000,
            idf_weight: 0.5,
            var_weight: 0.5,
        };

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..32).map(|j| {
                    if j == (i % 32) { 1.0 } else { 0.1 }
                }).collect()
            })
            .collect();

        let builder = RdfBuilder::new(&config, 32, &vectors);
        let (directory, data) = builder.build();

        assert_eq!(directory.len(), 32);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_rdf_scorer() {
        let config = RdfConfig {
            top_t: 4,
            stripe_shift: 4,
            stop_dim_threshold: 1000,
            idf_weight: 0.5,
            var_weight: 0.5,
        };

        // Create vectors with distinctive patterns
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..16).map(|j| {
                    if j == (i % 16) { 1.0 } else { 0.0 }
                }).collect()
            })
            .collect();

        let builder = RdfBuilder::new(&config, 16, &vectors);
        let dim_weights = builder.dim_weights();
        let (directory, data) = builder.build();

        let scorer = RdfScorer::new(&directory, &data, &dim_weights, 4, 50);

        // Query matching vector 0 pattern
        let query: Vec<f32> = (0..16).map(|j| if j == 0 { 1.0 } else { 0.0 }).collect();
        let candidates = scorer.score(&query, 4, 10);

        // Should find vector 0 (and others with same pattern) as top candidates
        assert!(!candidates.is_empty());
    }
}
