//! Segment writer for building immutable segments.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::rotation::Rotator;
use crate::types::*;
use super::format::*;
use super::bps::BpsBuilder;
use super::rdf::RdfBuilder;
use super::rerank::RerankBuilder;

/// Builder for creating segment files
pub struct SegmentWriter {
    config: EngineConfig,
    rotator: Rotator,
    vectors: Vec<Vec<f32>>,
    rotated: Vec<Vec<f32>>,
}

impl SegmentWriter {
    /// Create a new segment writer
    pub fn new(config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let rotator = Rotator::new(config.dim);
        Ok(Self {
            config,
            rotator,
            vectors: Vec::new(),
            rotated: Vec::new(),
        })
    }

    /// Add a vector to the segment
    pub fn add(&mut self, vector: &[f32]) -> Result<VectorId> {
        if vector.len() != self.config.dim as usize {
            return Err(Error::DimensionMismatch {
                expected: self.config.dim,
                got: vector.len() as u32,
            });
        }

        let vid = self.vectors.len() as VectorId;
        let vec_owned = vector.to_vec();
        
        // Apply rotation
        let rotated = self.rotator.rotate(&vec_owned);
        
        self.vectors.push(vec_owned);
        self.rotated.push(rotated);
        
        Ok(vid)
    }

    /// Add multiple vectors
    pub fn add_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<VectorId>> {
        vectors.iter().map(|v| self.add(v)).collect()
    }

    /// Number of vectors added
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Build and write segment to file
    pub fn build<P: AsRef<Path>>(self, path: P) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(Error::EmptyIndex);
        }

        let n_vec = self.vectors.len() as u32;
        let dim = self.config.dim;
        let _num_blocks = self.config.bps.num_blocks as usize;

        // Create builders
        let bps_builder = BpsBuilder::new(&self.config.bps, &self.rotated);
        let rdf_builder = RdfBuilder::new(&self.config.rdf, dim, &self.rotated);
        let rerank_builder = RerankBuilder::new(&self.config.rerank, &self.rotated);

        // Open file for writing
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Reserve space for header
        let mut header = SegmentHeader::new(n_vec, dim);
        header.bps_block = self.config.bps.block_size;
        header.bps_proj = self.config.bps.num_projections;
        header.rdf_t = self.config.rdf.top_t;
        header.rdf_stripe_shift = self.config.rdf.stripe_shift;
        header.num_outliers = self.config.rerank.num_outliers;
        header.flags.set(SegmentFlags::HAS_BPS);
        header.flags.set(SegmentFlags::HAS_RDF);
        header.flags.set(SegmentFlags::HAS_OUTLIERS);
        header.flags.set(SegmentFlags::ROTATED);
        header.flags.set(SegmentFlags::HAS_FP32); // Store originals for verification

        // Write placeholder header (will rewrite at end)
        writer.write_all(&[0u8; SegmentHeader::SIZE])?;
        let mut offset = SegmentHeader::SIZE as u64;

        // Write BPS data (SoA layout)
        header.off_bps = offset;
        let (bps_data, bps_qparams) = bps_builder.build();
        writer.write_all(&bps_data)?;
        offset += bps_data.len() as u64;

        // Write BPS quantization parameters
        header.off_bps_qparams = offset;
        writer.write_all(bytemuck::cast_slice(&bps_qparams))?;
        offset += (bps_qparams.len() * std::mem::size_of::<super::bps::BpsQParam>()) as u64;

        // Write int8 embeddings
        header.off_i8 = offset;
        let (i8_data, scales) = rerank_builder.build_i8();
        writer.write_all(bytemuck::cast_slice(&i8_data))?;
        offset += i8_data.len() as u64;

        // Write scales
        header.off_scales = offset;
        writer.write_all(bytemuck::cast_slice(&scales))?;
        offset += (scales.len() * 4) as u64;

        // Write outliers
        header.off_outliers = offset;
        let outliers = rerank_builder.build_outliers();
        writer.write_all(bytemuck::cast_slice(&outliers))?;
        offset += (outliers.len() * std::mem::size_of::<OutlierEntry>()) as u64;

        // Write tombstone bitset (all zeros = no tombstones)
        header.off_tombstone = offset;
        let tombstone_words = (n_vec as usize + 63) / 64;
        let tombstone_data = vec![0u64; tombstone_words];
        writer.write_all(bytemuck::cast_slice(&tombstone_data))?;
        offset += (tombstone_words * 8) as u64;

        // Write RDF directory
        header.off_rdf_dir = offset;
        let (rdf_dir, rdf_data) = rdf_builder.build();
        writer.write_all(bytemuck::cast_slice(&rdf_dir))?;
        offset += (rdf_dir.len() * std::mem::size_of::<PostingListEntry>()) as u64;

        // Write RDF posting list data
        header.off_rdf_data = offset;
        writer.write_all(&rdf_data)?;
        offset += rdf_data.len() as u64;

        // Write dimension weights
        header.off_dim_weights = offset;
        let weights = rdf_builder.dim_weights();
        writer.write_all(bytemuck::cast_slice(&weights))?;
        offset += (weights.len() * 4) as u64;

        // Write original fp32 vectors
        header.off_fp32 = offset;
        for vec in &self.vectors {
            writer.write_all(bytemuck::cast_slice(vec))?;
        }
        offset += (n_vec as usize * dim as usize * 4) as u64;

        // Update header with final file length
        header.file_len = offset;

        // Seek back and write final header
        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(bytemuck::bytes_of(&header))?;
        writer.flush()?;

        Ok(())
    }

    /// Get config
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use crate::segment::Segment;

    #[test]
    fn test_segment_write_read() {
        let config = EngineConfig::with_dim(64);
        let mut writer = SegmentWriter::new(config).unwrap();

        // Add some random vectors
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let vec: Vec<f32> = (0..64).map(|_| rng.gen_range(-1.0..1.0)).collect();
            writer.add(&vec).unwrap();
        }

        // Write segment
        let file = NamedTempFile::new().unwrap();
        writer.build(file.path()).unwrap();

        // Read back
        let segment = Segment::open(file.path()).unwrap();
        assert_eq!(segment.num_vectors(), 100);
        assert_eq!(segment.dim(), 64);
    }
}
