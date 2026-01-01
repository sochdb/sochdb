//! Segment binary format definitions.
//!
//! The segment file format is designed for mmap and sequential SIMD scans:
//! - Little-endian, fixed header
//! - Offset table to SoA blocks
//! - BPS stored as block-major SoA
//! - RDF posting lists stored in VID-striped chunks

use bytemuck::{Pod, Zeroable};
use crate::types::*;

/// Segment header - fixed size at file start
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SegmentHeader {
    /// Magic bytes: b"SVSEGM\0\0"
    pub magic: [u8; 8],
    /// Format version
    pub version: u32,
    /// Feature flags
    pub flags: SegmentFlags,
    /// Number of vectors in segment
    pub n_vec: u32,
    /// Vector dimension
    pub dim: u32,
    /// BPS block size (e.g., 16)
    pub bps_block: u16,
    /// BPS projections per block (1 or 2)
    pub bps_proj: u16,
    /// RDF top-t dimensions per vector
    pub rdf_t: u16,
    /// RDF stripe shift (log2 of stripe size)
    pub rdf_stripe_shift: u8,
    /// Number of outliers per vector
    pub num_outliers: u8,

    // Offset table (bytes from file start)
    /// BPS SoA array
    pub off_bps: u64,
    /// int8 embeddings (blocked SoA)
    pub off_i8: u64,
    /// Quantization scales per block
    pub off_scales: u64,
    /// Outlier entries
    pub off_outliers: u64,
    /// Tombstone bitset
    pub off_tombstone: u64,
    /// RDF posting list directory
    pub off_rdf_dir: u64,
    /// RDF posting list data
    pub off_rdf_data: u64,
    /// Dimension weights for RDF
    pub off_dim_weights: u64,
    /// Original fp32 vectors (optional, for verification)
    pub off_fp32: u64,
    /// BPS quantization parameters (min, inv_range per slot)
    pub off_bps_qparams: u64,
    /// Total file length
    pub file_len: u64,

    /// Padding for alignment (to 256 bytes total)
    /// 8 + 4 + 4 + 4 + 4 + 2 + 2 + 2 + 1 + 1 + (10 * 8) + 8 = 120, so we need 136 reserved
    pub _reserved1: [u8; 128],
    pub _reserved2: [u8; 8],
}

impl SegmentHeader {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create a new header with magic and version
    pub fn new(n_vec: u32, dim: u32) -> Self {
        Self {
            magic: MAGIC,
            version: SEGMENT_VERSION,
            flags: SegmentFlags::empty(),
            n_vec,
            dim,
            bps_block: DEFAULT_BPS_BLOCK_SIZE,
            bps_proj: DEFAULT_BPS_PROJECTIONS,
            rdf_t: DEFAULT_RDF_TOP_T,
            rdf_stripe_shift: DEFAULT_STRIPE_SHIFT,
            num_outliers: DEFAULT_NUM_OUTLIERS,
            off_bps: 0,
            off_i8: 0,
            off_scales: 0,
            off_outliers: 0,
            off_tombstone: 0,
            off_rdf_dir: 0,
            off_rdf_data: 0,
            off_dim_weights: 0,
            off_fp32: 0,
            off_bps_qparams: 0,
            file_len: 0,
            _reserved1: [0; 128],
            _reserved2: [0; 8],
        }
    }

    /// Validate header
    pub fn validate(&self) -> crate::Result<()> {
        if self.magic != MAGIC {
            return Err(crate::Error::InvalidMagic);
        }
        if self.version != SEGMENT_VERSION {
            return Err(crate::Error::UnsupportedVersion(self.version));
        }
        Ok(())
    }

    /// Number of BPS blocks
    pub fn num_bps_blocks(&self) -> u32 {
        (self.dim + self.bps_block as u32 - 1) / self.bps_block as u32
    }

    /// Size of BPS data in bytes
    pub fn bps_size(&self) -> usize {
        self.num_bps_blocks() as usize 
            * self.n_vec as usize 
            * self.bps_proj as usize
    }

    /// Size of int8 embedding data in bytes
    pub fn i8_size(&self) -> usize {
        self.n_vec as usize * self.dim as usize
    }

    /// Stripe size (number of vids per stripe)
    pub fn stripe_size(&self) -> usize {
        1usize << self.rdf_stripe_shift
    }
}

/// Segment feature flags
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq)]
pub struct SegmentFlags(pub u32);

impl SegmentFlags {
    pub const NONE: u32 = 0;
    pub const HAS_FP32: u32 = 1 << 0;
    pub const HAS_OUTLIERS: u32 = 1 << 1;
    pub const HAS_RDF: u32 = 1 << 2;
    pub const HAS_BPS: u32 = 1 << 3;
    pub const NORMALIZED: u32 = 1 << 4;
    pub const ROTATED: u32 = 1 << 5;

    pub fn empty() -> Self {
        Self(Self::NONE)
    }

    pub fn has(&self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }

    pub fn set(&mut self, flag: u32) {
        self.0 |= flag;
    }
}

/// RDF posting list directory entry
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PostingListEntry {
    /// Offset to list data
    pub offset: u64,
    /// Number of postings (total across all stripes)
    pub length: u32,
    /// Number of stripe chunks
    pub num_stripes: u16,
    /// Flags (is_stopword, etc.)
    pub flags: u16,
}

impl PostingListEntry {
    pub const FLAG_STOPWORD: u16 = 1 << 0;

    pub fn is_stopword(&self) -> bool {
        (self.flags & Self::FLAG_STOPWORD) != 0
    }
}

/// Block quantization scale
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlockScale {
    /// Scale factor for the block
    pub scale: f32,
}

/// Align a value to the next multiple of alignment
#[inline]
pub const fn align_to(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Compute the offset of BPS data for a specific block and vector
#[inline]
pub const fn bps_offset(block: usize, vec_id: usize, n_vec: usize, proj: usize) -> usize {
    // SoA layout: bps[(block * proj + p) * n_vec + vec]
    // For proj=1: bps[block * n_vec + vec]
    if proj == 1 {
        block * n_vec + vec_id
    } else {
        (block * 2) * n_vec + vec_id * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(SegmentHeader::SIZE, 256);
    }

    #[test]
    fn test_header_validation() {
        let mut header = SegmentHeader::new(1000, 768);
        assert!(header.validate().is_ok());

        header.magic = [0; 8];
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_flags() {
        let mut flags = SegmentFlags::empty();
        assert!(!flags.has(SegmentFlags::HAS_BPS));
        
        flags.set(SegmentFlags::HAS_BPS);
        assert!(flags.has(SegmentFlags::HAS_BPS));
        assert!(!flags.has(SegmentFlags::HAS_RDF));
    }
}
