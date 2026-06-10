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

//! SochDB gRPC Services
//!
//! This crate provides a comprehensive gRPC interface for SochDB operations.
//! It implements a "Thick Server / Thin Client" architecture where all business
//! logic lives in the Rust server, enabling thin SDK wrappers in any language.
//!
//! ## Services
//!
//! - **VectorIndexService**: HNSW vector operations
//! - **GraphService**: Graph overlay for agent memory
//! - **PolicyService**: Policy evaluation and enforcement
//! - **ContextService**: LLM context assembly with token budgets
//! - **CollectionService**: Collection management
//! - **NamespaceService**: Multi-tenant namespace management
//! - **SemanticCacheService**: Semantic caching for LLM queries
//! - **TraceService**: Trace/span management
//! - **CheckpointService**: State checkpoint and restore
//! - **McpService**: MCP tool routing
//! - **KvService**: Basic key-value operations
//!
//! ## Usage
//!
//! ```bash
//! # Start the gRPC server
//! sochdb-grpc-server --port 50051
//!
//! # From Python client
//! import grpc
//! from sochdb.proto import sochdb_pb2, sochdb_pb2_grpc
//!
//! channel = grpc.insecure_channel('localhost:50051')
//! stub = sochdb_pb2_grpc.VectorIndexServiceStub(channel)
//! ```

pub mod proto {
    // Include generated protobuf code
    tonic::include_proto!("sochdb.v1");
}

pub mod error;
pub mod server;

// Production hardening modules (Tasks 1-3, 8-9)
pub mod auth_interceptor; // gRPC authentication interceptor (Task 7)
pub mod blocking_pool; // Async boundary hardening with bounded pools
pub mod health_service; // Probe semantics + degraded mode + watchdog
pub mod metrics_server; // Prometheus /metrics HTTP endpoint (Task 8)
pub mod observability; // Observability hardening: cardinality, slow query, SLOs
pub mod security; // Security baseline: mTLS, AuthZ, audit, rate limiting

// Service implementations
pub mod checkpoint_server;
pub mod collection_server;
pub mod context_server;
pub mod memory_backend;
pub mod graph_server;
pub mod kv_server;
pub mod mcp_server;
pub mod namespace_server;
pub mod pg_wire;
pub mod policy_server;
pub mod semantic_cache_server;
pub mod subscription_server;
pub mod trace_server;
pub mod ws_server; // WebSocket gateway (Task 4) // PostgreSQL wire protocol (Task 5)

pub use blocking_pool::{BlockingPool, BlockingPoolConfig, BlockingPoolManager, PoolType};
pub use error::GrpcError;
pub use health_service::{
    DegradedCondition, HealthCheckResult, HealthService, HealthServiceConfig,
};
pub use security::{AuthError, Capability, Principal, SecurityConfig, SecurityService};
pub use server::VectorIndexServer;
