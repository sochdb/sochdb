//! Database adapter modules.

pub mod duckdb_adapter;
pub mod sochdb_adapter;
pub mod sqlite_adapter;

#[cfg(feature = "lancedb-bench")]
pub mod lancedb_adapter;
