//! Query engine implementation.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::filter::BitsetFilter;
use crate::rotation::Rotator;
use crate::segment::bps::{BpsBuilder, BpsScanner};
use crate::segment::rdf::RdfScorer;
use crate::segment::rerank::{quantize_query, Reranker};
use crate::segment::Segment;
use crate::types::*;

/// Query engine for executing vector searches
pub struct QueryEngine {
    config: EngineConfig,
    segments: Vec<Arc<Segment>>,
    rotator: Rotator,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new(config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let rotator = Rotator::new(config.dim);
        
        Ok(Self {
            config,
            segments: Vec::new(),
            rotator,
        })
    }

    /// Add a segment to the engine
    pub fn add_segment(&mut self, segment: Arc<Segment>) -> Result<()> {
        if segment.dim() != self.config.dim {
            return Err(Error::DimensionMismatch {
                expected: self.config.dim,
                got: segment.dim(),
            });
        }
        self.segments.push(segment);
        Ok(())
    }

    /// Load a segment from file
    pub fn load_segment(&mut self, path: &str) -> Result<()> {
        let segment = Segment::open(path)?;
        self.add_segment(Arc::new(segment))
    }

    /// Execute a query
    pub fn search(&self, query: &[f32], params: &QueryParams) -> Result<QueryResult> {
        if query.len() != self.config.dim as usize {
            return Err(Error::DimensionMismatch {
                expected: self.config.dim,
                got: query.len() as u32,
            });
        }

        if self.segments.is_empty() {
            return Err(Error::EmptyIndex);
        }

        let total_start = Instant::now();
        let mut stats = QueryStats::default();

        // Step 0: Rotate query
        let rotate_start = Instant::now();
        let rotated_query = self.rotator.rotate(query);
        stats.time_rotate_ns = rotate_start.elapsed().as_nanos() as u64;

        // Prepare filter
        let filter = params.filter.as_ref().map(|bits| {
            BitsetFilter::from_ids(
                self.total_vectors(),
                &bits.iter().map(|&id| id as VectorId).collect::<Vec<_>>(),
            )
        });

        // Search each segment
        let mut all_candidates: Vec<ScoredCandidate> = Vec::new();

        for segment in &self.segments {
            let segment_result = self.search_segment(
                segment,
                &rotated_query,
                query,
                params,
                filter.as_ref(),
                &mut stats,
            )?;
            all_candidates.extend(segment_result);
        }

        // Merge and select top k
        all_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_candidates.truncate(params.k);

        stats.total_time_ns = total_start.elapsed().as_nanos() as u64;

        Ok(QueryResult {
            candidates: all_candidates,
            stats,
        })
    }

    /// Search within a single segment
    fn search_segment(
        &self,
        segment: &Segment,
        rotated_query: &[f32],
        _original_query: &[f32],
        params: &QueryParams,
        filter: Option<&BitsetFilter>,
        stats: &mut QueryStats,
    ) -> Result<Vec<ScoredCandidate>> {
        let header = segment.header();
        let _n_vec = header.n_vec as usize;

        // Compute selectivity for filter-aware widening
        let selectivity = filter.map(|f| {
            let mut f = f.clone();
            f.selectivity()
        }).unwrap_or(1.0);
        
        let widening_factor = if selectivity < 1.0 && params.adaptive {
            (1.0 / selectivity).min(4.0)
        } else {
            1.0
        };

        // Step 2: RDF candidate generation
        let rdf_start = Instant::now();
        let rdf_candidates = if header.flags.has(crate::segment::format::SegmentFlags::HAS_RDF) {
            let l_a_widened = ((params.l_a as f32) * widening_factor) as usize;
            self.rdf_search(segment, rotated_query, l_a_widened)
        } else {
            Vec::new()
        };
        stats.time_rdf_ns += rdf_start.elapsed().as_nanos() as u64;
        stats.rdf_candidates += rdf_candidates.len();

        // Step 3: BPS candidate generation
        let bps_start = Instant::now();
        let bps_candidates = if header.flags.has(crate::segment::format::SegmentFlags::HAS_BPS) {
            let l_b_widened = ((params.l_b as f32) * widening_factor) as usize;
            self.bps_search(segment, rotated_query, l_b_widened)
        } else {
            Vec::new()
        };
        stats.time_bps_ns += bps_start.elapsed().as_nanos() as u64;
        stats.bps_candidates += bps_candidates.len();

        // Step 4: Union candidates
        let mut candidate_set: HashSet<VectorId> = HashSet::new();
        for c in &rdf_candidates {
            candidate_set.insert(c.id);
        }
        for (vid, _) in &bps_candidates {
            candidate_set.insert(*vid);
        }
        stats.union_size += candidate_set.len();

        // Apply filter
        let filter_start = Instant::now();
        let filtered_candidates: Vec<VectorId> = if let Some(f) = filter {
            candidate_set.into_iter()
                .filter(|&id| f.contains(id) && !segment.is_tombstoned(id))
                .collect()
        } else {
            candidate_set.into_iter()
                .filter(|&id| !segment.is_tombstoned(id))
                .collect()
        };
        stats.time_filter_ns += filter_start.elapsed().as_nanos() as u64;
        stats.post_filter_size += filtered_candidates.len();

        // Step 5: Rerank
        let rerank_start = Instant::now();
        let reranked = self.rerank(segment, rotated_query, &filtered_candidates, params.r)?;
        stats.time_rerank_ns += rerank_start.elapsed().as_nanos() as u64;
        stats.rerank_count += reranked.len();

        // Adaptive widening check
        if params.adaptive && reranked.len() < params.k {
            stats.widening_applied = true;
            // Could widen and retry here
        }

        Ok(reranked)
    }

    /// RDF-based candidate generation
    fn rdf_search(&self, segment: &Segment, rotated_query: &[f32], l_a: usize) -> Vec<ScoredCandidate> {
        let directory = segment.rdf_directory();
        if directory.is_empty() {
            return Vec::new();
        }

        let rdf_data = unsafe {
            let ptr = segment.rdf_data_ptr();
            let len = segment.header().file_len as usize - segment.header().off_rdf_data as usize;
            std::slice::from_raw_parts(ptr, len.min(1024 * 1024 * 100)) // Cap at 100MB for safety
        };

        let dim_weights = segment.dim_weights();
        let scorer = RdfScorer::new(
            directory,
            rdf_data,
            dim_weights,
            segment.header().rdf_stripe_shift,
            segment.num_vectors(),
        );

        scorer.score(rotated_query, self.config.rdf.top_t as usize, l_a)
    }

    /// BPS-based candidate generation
    fn bps_search(&self, segment: &Segment, rotated_query: &[f32], l_b: usize) -> Vec<(VectorId, Distance)> {
        let header = segment.header();
        let bps_data = segment.bps_data();
        
        // Compute query sketch
        let query_sketch = BpsBuilder::compute_query_sketch(&self.config.bps, rotated_query);
        
        let scanner = BpsScanner::new(
            bps_data,
            header.n_vec as usize,
            header.num_bps_blocks() as usize,
            header.bps_proj as usize,
        );

        scanner.top_k(&query_sketch, l_b)
    }

    /// Rerank candidates using int8 dot product
    fn rerank(
        &self,
        segment: &Segment,
        rotated_query: &[f32],
        candidates: &[VectorId],
        r: usize,
    ) -> Result<Vec<ScoredCandidate>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let header = segment.header();
        let i8_data = segment.i8_data();
        let scales = segment.scales_data();
        
        // Quantize query
        let (query_i8, query_scale) = quantize_query(rotated_query, &self.config.rerank);

        // Get outliers if available
        let outliers = if header.flags.has(crate::segment::format::SegmentFlags::HAS_OUTLIERS) {
            unsafe {
                std::slice::from_raw_parts(
                    segment.outliers_ptr(),
                    header.n_vec as usize * header.num_outliers as usize,
                )
            }
        } else {
            &[]
        };

        let reranker = Reranker::new(
            i8_data,
            &scales[..header.n_vec as usize], // One scale per vector
            outliers,
            header.dim as usize,
            header.num_outliers as usize,
        );

        Ok(reranker.rerank(candidates, &query_i8, query_scale, r))
    }

    /// Optional verification with fp32
    pub fn verify(&self, segment: &Segment, candidates: &[ScoredCandidate], query: &[f32], k: usize) -> Vec<ScoredCandidate> {
        if let Some(fp32_data) = segment.fp32_data() {
            let dim = segment.dim() as usize;
            let mut verified: Vec<ScoredCandidate> = candidates.iter()
                .map(|c| {
                    let offset = c.id as usize * dim;
                    if offset + dim <= fp32_data.len() {
                        let vec = &fp32_data[offset..offset + dim];
                        let score = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                        ScoredCandidate { id: c.id, score }
                    } else {
                        *c
                    }
                })
                .collect();
            
            verified.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            verified.truncate(k);
            verified
        } else {
            candidates.to_vec()
        }
    }

    /// Get total vector count across all segments
    pub fn total_vectors(&self) -> u32 {
        self.segments.iter().map(|s| s.num_vectors()).sum()
    }

    /// Get config
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get segments
    pub fn segments(&self) -> &[Arc<Segment>] {
        &self.segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::SegmentWriter;
    use tempfile::NamedTempFile;

    fn create_test_index() -> (NamedTempFile, EngineConfig) {
        let config = EngineConfig::with_dim(64);
        let mut writer = SegmentWriter::new(config.clone()).unwrap();

        // Add vectors with known patterns
        for i in 0..1000 {
            let mut vec = vec![0.0f32; 64];
            // Create a distinctive pattern
            vec[i % 64] = 1.0;
            vec[(i + 1) % 64] = 0.5;
            writer.add(&vec).unwrap();
        }

        let file = NamedTempFile::new().unwrap();
        writer.build(file.path()).unwrap();
        
        (file, config)
    }

    #[test]
    fn test_query_engine_basic() {
        let (file, config) = create_test_index();
        
        let mut engine = QueryEngine::new(config).unwrap();
        engine.load_segment(file.path().to_str().unwrap()).unwrap();

        // Query similar to vector 0
        let mut query = vec![0.0f32; 64];
        query[0] = 1.0;
        query[1] = 0.5;

        let params = QueryParams {
            k: 10,
            l_a: 100,
            l_b: 200,
            r: 50,
            adaptive: false,
            filter: None,
        };

        let result = engine.search(&query, &params).unwrap();
        
        assert!(!result.candidates.is_empty());
        println!("Query stats: {}", result.stats);
    }

    #[test]
    fn test_query_with_filter() {
        let (file, config) = create_test_index();
        
        let mut engine = QueryEngine::new(config).unwrap();
        engine.load_segment(file.path().to_str().unwrap()).unwrap();

        let mut query = vec![0.0f32; 64];
        query[0] = 1.0;

        // Filter to only even IDs
        let filter: Vec<u64> = (0..500).map(|i| i * 2).collect();

        let params = QueryParams {
            k: 10,
            l_a: 100,
            l_b: 200,
            r: 50,
            adaptive: false,
            filter: Some(filter),
        };

        let result = engine.search(&query, &params).unwrap();
        
        // All results should have even IDs
        for c in &result.candidates {
            assert!(c.id % 2 == 0, "Expected even ID, got {}", c.id);
        }
    }
}
