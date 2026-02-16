//! Database adapter modules.

pub mod sochdb_adapter;
pub mod sqlite_adapter;
pub mod duckdb_adapter;

#[cfg(feature = "lancedb-bench")]
pub mod lancedb_adapter;
