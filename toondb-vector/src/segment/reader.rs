//! Segment reader with mmap support.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::types::*;
use super::format::*;

/// An immutable segment backed by mmap
pub struct Segment {
    /// Memory-mapped file
    mmap: Arc<Mmap>,
    /// Parsed header
    header: SegmentHeader,
    /// File path
    path: String,
}

impl Segment {
    /// Open a segment file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < SegmentHeader::SIZE {
            return Err(Error::Segment("File too small for header".into()));
        }

        // Parse header
        let header: SegmentHeader = unsafe {
            std::ptr::read_unaligned(mmap.as_ptr() as *const SegmentHeader)
        };
        header.validate()?;

        // Validate file length
        if mmap.len() < header.file_len as usize {
            return Err(Error::Segment(format!(
                "File size {} < declared length {}",
                mmap.len(),
                header.file_len
            )));
        }

        Ok(Self {
            mmap: Arc::new(mmap),
            header,
            path: path_str,
        })
    }

    /// Get segment header
    #[inline]
    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    /// Number of vectors
    #[inline]
    pub fn num_vectors(&self) -> u32 {
        self.header.n_vec
    }

    /// Vector dimension
    #[inline]
    pub fn dim(&self) -> u32 {
        self.header.dim
    }

    /// Get raw pointer to BPS data
    #[inline]
    pub fn bps_ptr(&self) -> *const u8 {
        unsafe { self.mmap.as_ptr().add(self.header.off_bps as usize) }
    }

    /// Get BPS data slice
    pub fn bps_data(&self) -> &[u8] {
        let size = self.header.bps_size();
        unsafe {
            std::slice::from_raw_parts(self.bps_ptr(), size)
        }
    }

    /// Get raw pointer to int8 embedding data
    #[inline]
    pub fn i8_ptr(&self) -> *const i8 {
        unsafe { self.mmap.as_ptr().add(self.header.off_i8 as usize) as *const i8 }
    }

    /// Get int8 embedding data slice
    pub fn i8_data(&self) -> &[i8] {
        let size = self.header.i8_size();
        unsafe {
            std::slice::from_raw_parts(self.i8_ptr(), size)
        }
    }

    /// Get int8 vector for a specific ID
    pub fn get_i8_vector(&self, vid: VectorId) -> Option<&[i8]> {
        if vid >= self.header.n_vec {
            return None;
        }
        let dim = self.header.dim as usize;
        let offset = vid as usize * dim;
        Some(&self.i8_data()[offset..offset + dim])
    }

    /// Get raw pointer to quantization scales
    #[inline]
    pub fn scales_ptr(&self) -> *const f32 {
        unsafe { self.mmap.as_ptr().add(self.header.off_scales as usize) as *const f32 }
    }

    /// Get quantization scales
    pub fn scales_data(&self) -> &[f32] {
        let num_blocks = self.header.num_bps_blocks() as usize;
        // One scale per block per vector
        let size = num_blocks * self.header.n_vec as usize;
        unsafe {
            std::slice::from_raw_parts(self.scales_ptr(), size)
        }
    }

    /// Get raw pointer to outlier data
    #[inline]
    pub fn outliers_ptr(&self) -> *const OutlierEntry {
        unsafe { self.mmap.as_ptr().add(self.header.off_outliers as usize) as *const OutlierEntry }
    }

    /// Get outliers for a specific vector
    pub fn get_outliers(&self, vid: VectorId) -> Option<&[OutlierEntry]> {
        if vid >= self.header.n_vec || !self.header.flags.has(SegmentFlags::HAS_OUTLIERS) {
            return None;
        }
        let num_outliers = self.header.num_outliers as usize;
        let offset = vid as usize * num_outliers;
        unsafe {
            Some(std::slice::from_raw_parts(
                self.outliers_ptr().add(offset),
                num_outliers,
            ))
        }
    }

    /// Get raw pointer to tombstone bitset
    #[inline]
    pub fn tombstone_ptr(&self) -> *const u64 {
        unsafe { self.mmap.as_ptr().add(self.header.off_tombstone as usize) as *const u64 }
    }

    /// Get tombstone bitset
    pub fn tombstone_data(&self) -> &[u64] {
        let num_words = (self.header.n_vec as usize + 63) / 64;
        unsafe {
            std::slice::from_raw_parts(self.tombstone_ptr(), num_words)
        }
    }

    /// Check if a vector is tombstoned
    pub fn is_tombstoned(&self, vid: VectorId) -> bool {
        if vid >= self.header.n_vec {
            return true;
        }
        let word_idx = vid as usize / 64;
        let bit_idx = vid as usize % 64;
        let tombstones = self.tombstone_data();
        if word_idx >= tombstones.len() {
            return false;
        }
        (tombstones[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Get RDF posting list directory
    pub fn rdf_directory(&self) -> &[PostingListEntry] {
        if !self.header.flags.has(SegmentFlags::HAS_RDF) {
            return &[];
        }
        let dim = self.header.dim as usize;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(self.header.off_rdf_dir as usize) as *const PostingListEntry,
                dim,
            )
        }
    }

    /// Get raw pointer to RDF posting list data
    #[inline]
    pub fn rdf_data_ptr(&self) -> *const u8 {
        unsafe { self.mmap.as_ptr().add(self.header.off_rdf_data as usize) }
    }

    /// Get dimension weights for RDF
    pub fn dim_weights(&self) -> &[f32] {
        if !self.header.flags.has(SegmentFlags::HAS_RDF) {
            return &[];
        }
        let dim = self.header.dim as usize;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(self.header.off_dim_weights as usize) as *const f32,
                dim,
            )
        }
    }

    /// Get optional fp32 vectors for verification
    pub fn fp32_data(&self) -> Option<&[f32]> {
        if !self.header.flags.has(SegmentFlags::HAS_FP32) {
            return None;
        }
        let size = self.header.n_vec as usize * self.header.dim as usize;
        unsafe {
            Some(std::slice::from_raw_parts(
                self.mmap.as_ptr().add(self.header.off_fp32 as usize) as *const f32,
                size,
            ))
        }
    }

    /// Get fp32 vector for a specific ID
    pub fn get_fp32_vector(&self, vid: VectorId) -> Option<&[f32]> {
        let fp32 = self.fp32_data()?;
        let dim = self.header.dim as usize;
        let offset = vid as usize * dim;
        Some(&fp32[offset..offset + dim])
    }

    /// Get file path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Clone the mmap handle (cheap, Arc-backed)
    pub fn clone_mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }
}

impl std::fmt::Debug for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Segment")
            .field("path", &self.path)
            .field("n_vec", &self.header.n_vec)
            .field("dim", &self.header.dim)
            .field("flags", &self.header.flags)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_segment() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        
        let n_vec = 100u32;
        let dim = 64u32;
        let num_blocks = (dim + 15) / 16;
        
        let mut header = SegmentHeader::new(n_vec, dim);
        header.flags.set(SegmentFlags::HAS_BPS);
        
        // Calculate offsets
        let mut offset = SegmentHeader::SIZE as u64;
        
        // BPS data
        header.off_bps = offset;
        let bps_size = (num_blocks as usize * n_vec as usize) as u64;
        offset += bps_size;
        
        // i8 data
        header.off_i8 = offset;
        let i8_size = (n_vec as usize * dim as usize) as u64;
        offset += i8_size;
        
        // Scales
        header.off_scales = offset;
        let scales_size = (num_blocks as usize * n_vec as usize * 4) as u64;
        offset += scales_size;
        
        // Tombstone
        header.off_tombstone = offset;
        let tombstone_size = ((n_vec as usize + 63) / 64 * 8) as u64;
        offset += tombstone_size;
        
        header.file_len = offset;
        
        // Write header
        file.write_all(bytemuck::bytes_of(&header)).unwrap();
        
        // Write BPS data (zeros)
        file.write_all(&vec![0u8; bps_size as usize]).unwrap();
        
        // Write i8 data (zeros)
        file.write_all(&vec![0u8; i8_size as usize]).unwrap();
        
        // Write scales (ones)
        for _ in 0..(num_blocks * n_vec) {
            file.write_all(&1.0f32.to_le_bytes()).unwrap();
        }
        
        // Write tombstone (zeros = no tombstones)
        file.write_all(&vec![0u8; tombstone_size as usize]).unwrap();
        
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_segment_open() {
        let file = create_test_segment();
        let segment = Segment::open(file.path()).unwrap();
        
        assert_eq!(segment.num_vectors(), 100);
        assert_eq!(segment.dim(), 64);
    }

    #[test]
    fn test_tombstone_check() {
        let file = create_test_segment();
        let segment = Segment::open(file.path()).unwrap();
        
        // No tombstones set
        assert!(!segment.is_tombstoned(0));
        assert!(!segment.is_tombstoned(50));
        assert!(!segment.is_tombstoned(99));
        
        // Out of range should return true
        assert!(segment.is_tombstoned(100));
    }
}
