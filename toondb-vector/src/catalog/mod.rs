//! ToonDB-based catalog for segment metadata.
//!
//! Uses ToonDB storage for unified transaction semantics instead of SQLite.
//! Vector data is stored in mmap-able segment files.

mod toondb_catalog;

pub use toondb_catalog::{Catalog, CollectionInfo, SegmentInfo};
