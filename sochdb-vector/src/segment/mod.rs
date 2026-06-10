//! Segment format and reading functionality.
//!
//! Segments are immutable mmap-able files with SoA layouts for streaming SIMD.

pub mod bps;
pub mod format;
pub mod rdf;
pub mod reader;
pub mod rerank;
pub mod writer;

pub use format::*;
pub use reader::Segment;
pub use writer::SegmentWriter;
