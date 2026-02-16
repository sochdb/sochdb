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

//! # SochDB Fusion — Compositional Query Execution for Knowledge Fabric
//!
//! This crate is the central thesis of the Knowledge Fabric architecture: **fusing**
//! ART attribute lookups, HNSW vector search, and CSR graph traversal into a single,
//! zero-copy query pipeline that operates over Knowledge Objects.
//!
//! ## The Cost of Non-Fusion
//!
//! In a traditional disaggregated architecture, a compositional query requires:
//!
//! ```text
//! Attribute filter → serialize → vector search → serialize → graph traverse → serialize
//!                  ↑           ↑                ↑            ↑
//!              ~1 ms each: allocation, syscalls, cache flushes
//! ```
//!
//! **Total: ~11 ms** for a 3-hop compositional query.
//!
//! ## Fused Execution
//!
//! ```text
//! ART lookup → BitSet mask → HNSW search(mask) → CSR traverse → results
//!            ↑              ↑                    ↑
//!        BitSet flow: no serialization, no allocation, in-cache
//! ```
//!
//! **Total: ~300 μs** — a 30–50× improvement.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                FusionEngine                      │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
//! │  │ ART      │  │ HNSW     │  │ CSR      │      │
//! │  │ Index    │──│ Index    │──│ Graph    │      │
//! │  └──────────┘  └──────────┘  └──────────┘      │
//! │       ↓              ↓             ↓            │
//! │  ┌──────────────────────────────────────┐       │
//! │  │         BitSet Candidate Mask         │       │
//! │  │    (flows between stages, no alloc)   │       │
//! │  └──────────────────────────────────────┘       │
//! │       ↓                                         │
//! │  ┌──────────────────────────────────────┐       │
//! │  │    Temporal + Provenance Filter       │       │
//! │  └──────────────────────────────────────┘       │
//! │       ↓                                         │
//! │  ┌──────────────────────────────────────┐       │
//! │  │         Scored Knowledge Objects      │       │
//! │  └──────────────────────────────────────┘       │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Modules
//!
//! - [`bitset`] — Dense bit-vector for candidate masks (no allocation between stages)
//! - [`candidate_mask`] — Candidate mask operations and composition
//! - [`temporal_graph`] — Application-level CSR graph with temporal edges
//! - [`pipeline`] — Fused query execution pipeline
//! - [`query`] — Compositional query types and builder

pub mod bitset;
pub mod candidate_mask;
pub mod temporal_graph;
pub mod pipeline;
pub mod query;
pub mod versioned_store;

// Re-export primary types
pub use bitset::BitSet;
pub use candidate_mask::{CandidateMask, MaskOp};
pub use temporal_graph::{TemporalCsrGraph, TemporalEdge, GraphBuilder};
pub use pipeline::{FusionEngine, FusionResult, FusionConfig};
pub use query::{FusionQuery, FusionQueryBuilder, QueryStage};
pub use versioned_store::{VersionedObjectStore, StoreConfig, VersionedStoreError};
