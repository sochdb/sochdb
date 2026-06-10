// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! SochDB Tools - Command-line utilities for high-performance bulk operations
//!
//! This crate provides the `sochdb-bulk` CLI for bypassing Python FFI overhead
//! during large-scale vector index construction.
//!
//! ## Why Use This?
//!
//! Python FFI adds O(N·d) memcpy overhead for each batch. The bulk CLI runs
//! entirely in Rust, achieving 100% of native throughput via:
//! - Zero-copy memory-mapped I/O
//! - Direct HNSW insertion (no FFI marshalling)
//! - Optimal batch sizing and threading
//!
//! ## Performance Comparison
//!
//! | Method | 768D Throughput | Overhead |
//! |--------|-----------------|----------|
//! | Python FFI | ~130 vec/s | 12× slower |
//! | Rust Bulk CLI | ~1,600 vec/s | 1.0× baseline |
//!
//! ## Usage
//!
//! ```bash
//! # Build index from raw f32 file
//! sochdb-bulk build-index \
//!     --input embeddings.bin \
//!     --output index.hnsw \
//!     --dimension 768
//!
//! # Build from NumPy .npy file
//! sochdb-bulk build-index \
//!     --input embeddings.npy \
//!     --output index.hnsw \
//!     --format npy
//!
//! # Custom HNSW parameters
//! sochdb-bulk build-index \
//!     --input data.bin \
//!     --output index.hnsw \
//!     --dimension 768 \
//!     --max-connections 32 \
//!     --ef-construction 200
//!
//! # With optional ID file
//! sochdb-bulk build-index \
//!     --input embeddings.bin \
//!     --ids ids.u64 \
//!     --output index.hnsw \
//!     --dimension 768
//! ```

pub mod builder;
pub mod error;
pub mod guardrails;
pub mod io;
pub mod ordering;
pub mod progress;

pub use builder::{BuildConfig, BuildError, BuildResult, build_hnsw_index, build_hnsw_index_u128};
pub use error::ToolsError;
pub use guardrails::{check_safe_mode, check_throughput, log_insert_path, print_perf_summary};
pub use io::raw::write_raw_f32;
pub use io::{
    FaultStats, FaultTelemetry, ensure_resident_for_hnsw, prefault_region, with_telemetry,
};
pub use io::{OwnedVectors, load_bulk, load_npy_bulk, load_vectors_bulk};
pub use io::{VectorFormat, VectorMeta, VectorReader, ids_from_mmap, open_ids, open_vectors};
pub use ordering::{OrderingStrategy, ReorderResult, compute_ordering};
pub use progress::{BulkStats, ProgressReporter};
