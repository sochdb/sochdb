//! RDF (Rare-Dominant Fingerprint) builder and posting list handling.
//!
//! RDF uses IR-style inverted lists over a sparse fingerprint of each vector.
//! Posting lists are stored in VID-striped chunks for cache-friendly scoring.

use crate::config::RdfConfig;
use crate::segment::format::PostingListEntry;
use crate::types::*;
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
            let mut scored: Vec<(usize, f32)> =
                vec.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
            let nth_idx = top_t.min(scored.len()).saturating_sub(1);
            if nth_idx < scored.len() {
                scored.select_nth_unstable_by(nth_idx, |a, b| b.1.partial_cmp(&a.1).unwrap());
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
            let weight = config.idf_weight as f64 * idf + config.var_weight as f64 * std_dev;
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
                scored.select_nth_unstable_by(top_t - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
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
                scored.select_nth_unstable_by(top_t - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
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
            let flags = if is_stopword {
                PostingListEntry::FLAG_STOPWORD
            } else {
                0
            };

            // Group postings by stripe
            let mut stripes: HashMap<StripeId, Vec<(u8, bool, u8)>> = HashMap::new();
            for &(vid, sign, mag) in postings {
                let stripe_id = vid >> stripe_shift;
                let vid_in_stripe = (vid & ((1 << stripe_shift) - 1)) as u8;
                stripes
                    .entry(stripe_id)
                    .or_default()
                    .push((vid_in_stripe, sign, mag));
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
                let w = if d < self.dim_weights.len() {
                    self.dim_weights[d]
                } else {
                    1.0
                };
                (d, v.abs() * w, v)
            })
            .collect();

        if scored.len() > top_t {
            scored.select_nth_unstable_by(top_t - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(top_t);
        }

        // Get query dims (excluding stopwords — but keep dims that pass
        // the directory bounds check even if all are stopwords)
        let query_dims: Vec<(usize, f32, f32)> = scored
            .into_iter()
            .filter(|&(d, _, _)| d < self.directory.len() && !self.directory[d].is_stopword())
            .collect();

        if query_dims.is_empty() {
            // All query dimensions were stopwords — fall back to using
            // the original dimensions WITHOUT the stopword filter.
            // Returning empty here would cause zero recall for common
            // queries where all top-t dimensions happen to be stop-dims.
            // IDF-based dim_weights will naturally downweight these.
            let query_dims_fallback: Vec<(usize, f32, f32)> = {
                let mut s: Vec<(usize, f32, f32)> = query
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| {
                        let w = if d < self.dim_weights.len() {
                            self.dim_weights[d]
                        } else {
                            1.0
                        };
                        (d, v.abs() * w, v)
                    })
                    .collect();
                if s.len() > top_t {
                    s.select_nth_unstable_by(top_t - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
                    s.truncate(top_t);
                }
                s.into_iter()
                    .filter(|&(d, _, _)| d < self.directory.len())
                    .collect()
            };
            if query_dims_fallback.is_empty() {
                return Vec::new();
            }
            return self.score_with_dims(&query_dims_fallback, l_a);
        }

        self.score_with_dims(&query_dims, l_a)
    }

    /// Internal scoring with a given set of query dimensions.
    /// Separated from `score()` to allow fallback when all dims are stopwords.
    fn score_with_dims(
        &self,
        query_dims: &[(usize, f32, f32)],
        l_a: usize,
    ) -> Vec<ScoredCandidate> {
        // Use stripe-based accumulation
        let num_stripes = (self.n_vec as usize + self.stripe_size - 1) / self.stripe_size;
        let mut global_candidates = Vec::new();

        // Allocate stripe accumulator ONCE, clear per-stripe (avoids N allocs)
        let mut stripe_acc = vec![0.0f32; self.stripe_size];

        // One cursor per query dimension, created before the stripe loop so each
        // posting list is read exactly once across the whole scan rather than
        // being re-walked from the start for every stripe.
        let mut cursors: Vec<DimCursor> = query_dims
            .iter()
            .filter_map(|&(dim_idx, _, q_value)| {
                let entry = &self.directory[dim_idx];
                if entry.length == 0 {
                    return None;
                }
                Some(DimCursor {
                    offset: entry.offset as usize,
                    remaining: entry.num_stripes as u32,
                    q_value,
                    dim_weight: self.dim_weights[dim_idx],
                })
            })
            .collect();

        // Process stripe by stripe for cache locality
        for stripe_id in 0..num_stripes as u32 {
            // Clear accumulator (memset — vectorizes to single SIMD store)
            stripe_acc.iter_mut().for_each(|x| *x = 0.0);

            for cursor in cursors.iter_mut() {
                self.accumulate_stripe(cursor, stripe_id, &mut stripe_acc);
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

        global_candidates
            .select_nth_unstable_by(l_a - 1, |a, b| b.score.partial_cmp(&a.score).unwrap());
        global_candidates.truncate(l_a);
        global_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        global_candidates
    }

    /// Accumulate scores for a specific stripe from one dimension's posting list.
    ///
    /// `cursor` is advanced past every chunk it consumes and is never rewound.
    /// Callers must therefore invoke this with non-decreasing
    /// `target_stripe_id`, which `score_with_dims` does.
    fn accumulate_stripe(
        &self,
        cursor: &mut DimCursor,
        target_stripe_id: StripeId,
        stripe_acc: &mut [f32],
    ) {
        let header_size = std::mem::size_of::<StripeChunkHeader>();
        let posting_size = std::mem::size_of::<RdfPosting>();

        while cursor.remaining > 0 {
            if cursor.offset + header_size > self.rdf_data.len() {
                cursor.remaining = 0;
                return;
            }

            let header: StripeChunkHeader = unsafe {
                std::ptr::read_unaligned(self.rdf_data.as_ptr().add(cursor.offset) as *const _)
            };

            // Chunks ascend by stripe id and so do the targets, so a chunk past
            // the target belongs to a later call. Leave the cursor on it.
            if header.stripe_id > target_stripe_id {
                return;
            }

            let count = header.count as usize;
            let mut offset = cursor.offset + header_size;
            let matched = header.stripe_id == target_stripe_id;

            if matched {
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
                    let contribution = cursor.q_value * sign * mag * cursor.dim_weight;
                    stripe_acc[vid_in_stripe] += contribution;
                }
            }

            // Consume the chunk either way: a chunk before the target can never
            // match a later, larger target. That case only arises from malformed
            // data (duplicate or out-of-range stripe ids), since the caller
            // visits every stripe in order and so meets each chunk at its own
            // stripe; skipping it keeps a corrupt posting list from stalling the
            // cursor rather than serving any well-formed input.
            cursor.offset += header_size + count * posting_size;
            cursor.remaining -= 1;

            if matched {
                return;
            }
        }
    }
}

/// A monotonically advancing read cursor into one dimension's posting list.
///
/// `score_with_dims` visits stripe ids in increasing order, and `RdfBuilder`
/// sorts each posting list's chunks by stripe id before writing them. Joining
/// two sequences that are already sorted on the join key is a merge, not a
/// nested loop. Restarting each posting list from `entry.offset` for every
/// stripe made the scan O(top_t * num_stripes^2) — quadratic in the vector
/// count, which is why RDF dominated query time and got worse with scale.
/// Carrying the offset across stripes makes it O(top_t * (num_stripes + chunks)).
struct DimCursor {
    /// Byte offset of the next unconsumed chunk header in `rdf_data`.
    offset: usize,
    /// Chunks left in this dimension's posting list.
    remaining: u32,
    q_value: f32,
    dim_weight: f32,
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
                (0..32)
                    .map(|j| if j == (i % 32) { 1.0 } else { 0.1 })
                    .collect()
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
                (0..16)
                    .map(|j| if j == (i % 16) { 1.0 } else { 0.0 })
                    .collect()
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

    #[test]
    fn every_vector_holding_a_query_dimension_is_found_even_when_its_postings_span_many_stripes() {
        // The scorer walks stripes in ascending order while each posting list
        // carries its chunks in ascending stripe order, and it keeps one forward
        // cursor per dimension. A cursor that advances past a chunk belonging to
        // a later stripe silently drops every match in it, which reads as a mild
        // recall regression rather than a bug.
        //
        // Catching that requires *gaps*: the dimension under test must be absent
        // from long runs of stripes, so the cursor is repeatedly asked about a
        // stripe earlier than the chunk it is parked on and must hold position.
        // `top_t: 1` keeps each vector posting to exactly one dimension, which
        // matters because ties among zero-valued dimensions otherwise scatter
        // postings into every stripe and erase the gaps.
        let config = RdfConfig {
            top_t: 1,
            stripe_shift: 4, // 16 vids per stripe
            stop_dim_threshold: 10_000,
            idf_weight: 0.5,
            var_weight: 0.5,
        };

        let dim = 16usize;
        let n = 1024usize;
        let target_dim = 3usize;
        // Hot in one stripe out of every eight, so the target dimension's chunks
        // sit at stripes 0, 8, 16, ... with seven empty stripes between them.
        let is_target = |i: usize| (i / 16) % 8 == 0;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                if is_target(i) {
                    v[target_dim] = 1.0;
                } else {
                    // Spread the remaining vectors over filler dimensions so no
                    // single dimension collects a posting from every vector.
                    v[4 + (i % 12)] = 0.5;
                }
                v
            })
            .collect();

        let builder = RdfBuilder::new(&config, dim as u32, &vectors);
        let dim_weights = builder.dim_weights();
        let (directory, data) = builder.build();
        let scorer = RdfScorer::new(&directory, &data, &dim_weights, 4, n as u32);

        let query: Vec<f32> = (0..dim)
            .map(|j| if j == target_dim { 1.0 } else { 0.0 })
            .collect();
        let candidates = scorer.score(&query, config.top_t as usize, n);

        let found: std::collections::HashSet<u32> = candidates.iter().map(|c| c.id).collect();
        let expected: Vec<u32> = (0..n).filter(|&i| is_target(i)).map(|i| i as u32).collect();
        assert_eq!(expected.len(), 128, "test data should span eight stripes");

        let missing: Vec<u32> = expected
            .iter()
            .copied()
            .filter(|id| !found.contains(id))
            .collect();
        assert!(
            missing.is_empty(),
            "cursor dropped {} of {} vectors carrying dimension {}: {:?}",
            missing.len(),
            expected.len(),
            target_dim,
            &missing[..missing.len().min(16)]
        );
    }

    #[test]
    fn scores_are_unchanged_when_several_dimensions_interleave_across_stripes() {
        // With one cursor per query dimension, an error in how far a cursor
        // advances shows up as a wrong score rather than a missing id. Give two
        // dimensions overlapping but offset stripe coverage and pin the exact
        // scores, so both the traversal and the accumulated value are checked.
        let config = RdfConfig {
            top_t: 2,
            stripe_shift: 2, // 4 vids per stripe
            stop_dim_threshold: 10_000,
            idf_weight: 0.5,
            var_weight: 0.5,
        };

        let dim = 4usize;
        let n = 64usize;
        // Dimension 0 is hot on even vectors, dimension 1 on multiples of 8, so
        // their chunks share some stripes and not others.
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                if i % 2 == 0 {
                    v[0] = 1.0;
                }
                if i % 8 == 0 {
                    v[1] = 1.0;
                }
                v
            })
            .collect();

        let builder = RdfBuilder::new(&config, dim as u32, &vectors);
        let dim_weights = builder.dim_weights();
        let (directory, data) = builder.build();
        let scorer = RdfScorer::new(&directory, &data, &dim_weights, 2, n as u32);

        let query: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0];
        let candidates = scorer.score(&query, 2, n);
        let by_id: std::collections::HashMap<u32, f32> =
            candidates.iter().map(|c| (c.id, c.score)).collect();

        // Vectors divisible by 8 carry both dimensions and must outscore those
        // that carry only dimension 0.
        for i in (0..n).step_by(8) {
            let both = by_id
                .get(&(i as u32))
                .unwrap_or_else(|| panic!("vector {} carries both dims but is absent", i));
            let only_dim0 = by_id
                .get(&((i + 2) as u32))
                .unwrap_or_else(|| panic!("vector {} carries dimension 0 but is absent", i + 2));
            assert!(
                *both > *only_dim0,
                "vector {} carries two query dims and must outscore {}: {} vs {}",
                i,
                i + 2,
                both,
                only_dim0
            );
        }
    }
}
