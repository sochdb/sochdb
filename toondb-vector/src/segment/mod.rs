//! Segment format and reading functionality.
//!
//! Segments are immutable mmap-able files with SoA layouts for streaming SIMD.

pub mod format;
pub mod reader;
pub mod writer;
pub mod bps;
pub mod rdf;
pub mod rerank;

pub use format::*;
pub use reader::Segment;
pub use writer::SegmentWriter;
