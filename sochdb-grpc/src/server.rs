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

//! gRPC Server implementation for Vector Index Service

use crate::auth_interceptor::{extract_principal, require_capability};
use crate::error::GrpcError;
use crate::namespace_server::NamespaceServer;
use crate::proto::{
    self, CreateIndexRequest, CreateIndexResponse, DropIndexRequest, DropIndexResponse,
    GetStatsRequest, GetStatsResponse, HealthCheckRequest, HealthCheckResponse,
    HnswConfig as ProtoHnswConfig, IndexInfo, IndexStats, InsertBatchRequest, InsertBatchResponse,
    InsertStreamRequest, InsertStreamResponse, QueryResults, SearchBatchRequest,
    SearchBatchResponse, SearchRequest, SearchResponse, SearchResult,
    vector_index_service_server::{VectorIndexService, VectorIndexServiceServer},
};
use crate::security::Capability;
use dashmap::DashMap;
use sochdb_index::hnsw::{DistanceMetric, HnswConfig, HnswIndex};
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

/// Server version
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Metadata for an index
#[allow(dead_code)]
struct IndexEntry {
    index: Arc<HnswIndex>,
    name: String,
    dimension: usize,
    metric: proto::DistanceMetric,
    config: ProtoHnswConfig,
    created_at: u64,
}

/// Vector Index gRPC Server
pub struct VectorIndexServer {
    /// Map of index name -> index entry (keyed as "namespace:name")
    indexes: DashMap<String, IndexEntry>,
    /// Shared namespace server for quota enforcement
    ns_server: Option<NamespaceServer>,
}

impl VectorIndexServer {
    /// Create a new server instance
    pub fn new() -> Self {
        Self {
            indexes: DashMap::new(),
            ns_server: None,
        }
    }

    /// Create with a shared NamespaceServer for quota enforcement.
    pub fn with_namespace_server(ns: NamespaceServer) -> Self {
        Self {
            indexes: DashMap::new(),
            ns_server: Some(ns),
        }
    }

    /// Build a namespace-prefixed index key.
    fn index_key(namespace: &str, name: &str) -> String {
        if namespace.is_empty() {
            name.to_string()
        } else {
            format!("{}:{}", namespace, name)
        }
    }

    /// Create the gRPC service
    pub fn into_service(self) -> VectorIndexServiceServer<Self> {
        VectorIndexServiceServer::new(self)
    }

    /// Get an index by name and its dimension
    fn get_index_with_dim(&self, name: &str) -> Result<(Arc<HnswIndex>, usize), GrpcError> {
        self.indexes
            .get(name)
            .map(|entry| (entry.index.clone(), entry.dimension))
            .ok_or_else(|| GrpcError::IndexNotFound(name.to_string()))
    }

    /// Get an index by name together with its dimension and configured metric.
    ///
    /// Used by the search paths so the response can report which metric the
    /// `distance` values are expressed in (see `metric_label`).
    fn get_index_with_meta(
        &self,
        name: &str,
    ) -> Result<(Arc<HnswIndex>, usize, proto::DistanceMetric), GrpcError> {
        self.indexes
            .get(name)
            .map(|entry| (entry.index.clone(), entry.dimension, entry.metric))
            .ok_or_else(|| GrpcError::IndexNotFound(name.to_string()))
    }

    /// Get an index by name
    fn get_index(&self, name: &str) -> Result<Arc<HnswIndex>, GrpcError> {
        self.indexes
            .get(name)
            .map(|entry| entry.index.clone())
            .ok_or_else(|| GrpcError::IndexNotFound(name.to_string()))
    }

    /// Convert proto metric to internal metric
    fn convert_metric(metric: proto::DistanceMetric) -> DistanceMetric {
        match metric {
            proto::DistanceMetric::L2 => DistanceMetric::Euclidean,
            proto::DistanceMetric::Cosine => DistanceMetric::Cosine,
            proto::DistanceMetric::DotProduct => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine, // Default
        }
    }

    /// Stable, lower-case label for a metric, surfaced on `SearchResult.metric`
    /// so callers can interpret `distance` without out-of-band knowledge.
    fn metric_label(metric: proto::DistanceMetric) -> &'static str {
        match metric {
            proto::DistanceMetric::L2 => "euclidean",
            proto::DistanceMetric::Cosine => "cosine",
            proto::DistanceMetric::DotProduct => "dot_product",
            proto::DistanceMetric::Unspecified => "unspecified",
        }
    }
}

impl Default for VectorIndexServer {
    fn default() -> Self {
        Self::new()
    }
}
#[tonic::async_trait]
impl VectorIndexService for VectorIndexServer {
    async fn create_index(
        &self,
        request: Request<CreateIndexRequest>,
    ) -> Result<Response<CreateIndexResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::ManageCollections)?;
        let name = req.name.clone();
        // SECURITY: scope the index key to the caller's tenant so two tenants
        // cannot read/overwrite/destroy each other's indexes by reusing a name.
        // The key is derived from the AUTHENTICATED principal, never from input.
        let key = Self::index_key(&principal.tenant_id, &name);

        // Check if index already exists
        if self.indexes.contains_key(&key) {
            return Ok(Response::new(CreateIndexResponse {
                success: false,
                error: format!("Index '{}' already exists", name),
                info: None,
            }));
        }

        // Build config
        let proto_config = req.config.unwrap_or_default();
        // Fall back to HnswConfig::default() for any field the client leaves
        // unset, so the gRPC path inherits the same 95+-recall defaults
        // (m=32, m0=64, ef_construction=256, F32 precision) as every other
        // entry point. Previously these fallbacks were hardcoded to the old
        // cheap values (m=16/m0=32/efc=200), so a client that didn't specify
        // an HNSW config silently got a low-recall graph (~0.90 vs ~0.97 on
        // hard 768-d data) — the exact default-drift #27 removed elsewhere.
        let def = HnswConfig::default();
        let config = HnswConfig {
            max_connections: if proto_config.max_connections > 0 {
                proto_config.max_connections as usize
            } else {
                def.max_connections
            },
            max_connections_layer0: if proto_config.max_connections_layer0 > 0 {
                proto_config.max_connections_layer0 as usize
            } else {
                def.max_connections_layer0
            },
            ef_construction: if proto_config.ef_construction > 0 {
                proto_config.ef_construction as usize
            } else {
                def.ef_construction
            },
            ef_search: if proto_config.ef_search > 0 {
                proto_config.ef_search as usize
            } else {
                def.ef_search
            },
            metric: Self::convert_metric(req.metric()),
            ..def
        };

        let dimension = req.dimension as usize;
        let index = HnswIndex::new(dimension, config.clone());
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = IndexEntry {
            index: Arc::new(index),
            name: name.clone(),
            dimension,
            metric: req.metric(),
            config: proto_config.clone(),
            created_at,
        };

        self.indexes.insert(key, entry);

        tracing::info!("Created index '{}' with dimension {}", name, dimension);

        Ok(Response::new(CreateIndexResponse {
            success: true,
            error: String::new(),
            info: Some(IndexInfo {
                name,
                dimension: dimension as u32,
                metric: req.metric.into(),
                config: Some(proto_config),
                created_at,
            }),
        }))
    }

    async fn drop_index(
        &self,
        request: Request<DropIndexRequest>,
    ) -> Result<Response<DropIndexResponse>, Status> {
        let principal = extract_principal(&request);
        require_capability(&principal, &Capability::ManageCollections)?;
        let name = request.into_inner().name;
        let key = Self::index_key(&principal.tenant_id, &name);

        match self.indexes.remove(&key) {
            Some(_) => {
                tracing::info!("Dropped index '{}'", name);
                Ok(Response::new(DropIndexResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(DropIndexResponse {
                success: false,
                error: format!("Index '{}' not found", name),
            })),
        }
    }

    async fn insert_batch(
        &self,
        request: Request<InsertBatchRequest>,
    ) -> Result<Response<InsertBatchResponse>, Status> {
        let start = Instant::now();
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Write)?;

        let key = Self::index_key(&principal.tenant_id, &req.index_name);
        let (index, dimension) = self.get_index_with_dim(&key)?;

        // Validate input
        if req.vectors.len() != req.ids.len() * dimension {
            return Err(Status::invalid_argument(format!(
                "Vector data size mismatch: expected {} floats, got {}",
                req.ids.len() * dimension,
                req.vectors.len()
            )));
        }

        // Convert IDs to u128
        let ids: Vec<u128> = req.ids.iter().map(|&id| id as u128).collect();

        // Offload CPU-heavy HNSW insertion to blocking thread pool to avoid
        // starving the tokio runtime (which handles health checks, streams, etc.)
        let index_name = req.index_name.clone();
        let vectors = req.vectors;
        let result =
            tokio::task::spawn_blocking(move || index.insert_batch_flat(&ids, &vectors, dimension))
                .await
                .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match result {
            Ok(count) => {
                let duration_us = start.elapsed().as_micros() as u64;
                tracing::info!(
                    "Inserted {} vectors into '{}' in {}µs ({}ms)",
                    count,
                    index_name,
                    duration_us,
                    duration_us / 1000
                );
                Ok(Response::new(InsertBatchResponse {
                    inserted_count: count as u32,
                    error: String::new(),
                    duration_us,
                }))
            }
            Err(e) => Ok(Response::new(InsertBatchResponse {
                inserted_count: 0,
                error: e,
                duration_us: start.elapsed().as_micros() as u64,
            })),
        }
    }

    async fn insert_stream(
        &self,
        request: Request<Streaming<InsertStreamRequest>>,
    ) -> Result<Response<InsertStreamResponse>, Status> {
        let start = Instant::now();
        // Auth check: extract principal from request metadata
        let principal = extract_principal(&request);
        require_capability(&principal, &Capability::Write)?;
        let mut stream = request.into_inner();

        let mut index_name: Option<String> = None;
        let mut index: Option<(Arc<HnswIndex>, usize)> = None;
        let mut total_inserted = 0u32;
        let mut errors = Vec::new();

        // Micro-batch buffer: accumulate streamed vectors and flush in batches
        // to amortize lock acquisition overhead
        const MICRO_BATCH_SIZE: usize = 128;
        let mut batch_ids: Vec<u128> = Vec::with_capacity(MICRO_BATCH_SIZE);
        let mut batch_vectors: Vec<f32> = Vec::new();
        let mut dimension: usize = 0;

        while let Some(result) = stream.next().await {
            match result {
                Ok(req) => {
                    // Get index on first message
                    if index.is_none() {
                        if req.index_name.is_empty() {
                            errors.push("First message must include index_name".to_string());
                            continue;
                        }
                        index_name = Some(req.index_name.clone());
                        let key = Self::index_key(&principal.tenant_id, &req.index_name);
                        match self.get_index_with_dim(&key) {
                            Ok((idx, dim)) => {
                                dimension = dim;
                                batch_vectors.reserve(MICRO_BATCH_SIZE * dim);
                                index = Some((idx, dim));
                            }
                            Err(e) => {
                                errors.push(e.to_string());
                                break;
                            }
                        }
                    }

                    // Buffer the vector
                    if index.is_some() {
                        batch_ids.push(req.id as u128);
                        batch_vectors.extend_from_slice(&req.vector);

                        // Flush micro-batch when full
                        if batch_ids.len() >= MICRO_BATCH_SIZE {
                            if let Some((ref idx, _)) = index {
                                match idx.insert_batch_flat(&batch_ids, &batch_vectors, dimension) {
                                    Ok(count) => total_inserted += count as u32,
                                    Err(e) => errors.push(e),
                                }
                            }
                            batch_ids.clear();
                            batch_vectors.clear();
                        }
                    }
                }
                Err(e) => {
                    errors.push(format!("Stream error: {}", e));
                    break;
                }
            }
        }

        // Flush remaining buffered vectors
        if !batch_ids.is_empty() {
            if let Some((ref idx, _)) = index {
                match idx.insert_batch_flat(&batch_ids, &batch_vectors, dimension) {
                    Ok(count) => total_inserted += count as u32,
                    Err(e) => errors.push(e),
                }
            }
        }

        let duration_us = start.elapsed().as_micros() as u64;

        if let Some(name) = &index_name {
            tracing::debug!(
                "Stream inserted {} vectors into '{}' in {}µs",
                total_inserted,
                name,
                duration_us
            );
        }

        Ok(Response::new(InsertStreamResponse {
            total_inserted,
            errors,
            duration_us,
        }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let start = Instant::now();
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Read)?;

        let key = Self::index_key(&principal.tenant_id, &req.index_name);
        let (index, dimension, metric_enum) = self.get_index_with_meta(&key)?;
        let metric = Self::metric_label(metric_enum);

        // Validate dimension
        if req.query.len() != dimension {
            return Err(Status::invalid_argument(format!(
                "Query dimension mismatch: expected {}, got {}",
                dimension,
                req.query.len()
            )));
        }

        let k = req.k.max(1) as usize;

        // Honor per-query ef override for recall tuning; fall back to index default
        let ef = if req.ef > 0 { req.ef as usize } else { 0 };

        // Offload CPU-heavy HNSW search to blocking thread pool
        let query = req.query;
        let results = tokio::task::spawn_blocking(move || {
            if ef > 0 {
                index.search_with_ef(&query, k, ef.max(k))
            } else {
                index.search(&query, k)
            }
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match results {
            Ok(r) => {
                let duration_us = start.elapsed().as_micros() as u64;
                Ok(Response::new(SearchResponse {
                    results: r
                        .into_iter()
                        .map(|(id, distance)| SearchResult {
                            id: id as u64,
                            distance,
                            metric: metric.to_string(),
                        })
                        .collect(),
                    duration_us,
                    error: String::new(),
                }))
            }
            Err(e) => Ok(Response::new(SearchResponse {
                results: vec![],
                duration_us: start.elapsed().as_micros() as u64,
                error: e,
            })),
        }
    }

    async fn search_batch(
        &self,
        request: Request<SearchBatchRequest>,
    ) -> Result<Response<SearchBatchResponse>, Status> {
        let start = Instant::now();
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Read)?;

        let key = Self::index_key(&principal.tenant_id, &req.index_name);
        let (index, dimension, metric_enum) = self.get_index_with_meta(&key)?;
        let metric = Self::metric_label(metric_enum);
        let num_queries = req.num_queries as usize;
        let k = req.k.max(1) as usize;

        // Validate
        if req.queries.len() != num_queries * dimension {
            return Err(Status::invalid_argument(format!(
                "Query data size mismatch: expected {} floats, got {}",
                num_queries * dimension,
                req.queries.len()
            )));
        }

        // Parallel batch search via rayon on blocking thread pool
        let queries = req.queries;
        let all_results = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            (0..num_queries)
                .into_par_iter()
                .map(|i| {
                    let query = &queries[i * dimension..(i + 1) * dimension];
                    let results = match index.search(query, k) {
                        Ok(r) => r,
                        Err(_) => vec![],
                    };
                    QueryResults {
                        results: results
                            .into_iter()
                            .map(|(id, distance)| SearchResult {
                                id: id as u64,
                                distance,
                                metric: metric.to_string(),
                            })
                            .collect(),
                    }
                })
                .collect::<Vec<_>>()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        let duration_us = start.elapsed().as_micros() as u64;

        Ok(Response::new(SearchBatchResponse {
            results: all_results,
            duration_us,
        }))
    }

    async fn get_stats(
        &self,
        request: Request<GetStatsRequest>,
    ) -> Result<Response<GetStatsResponse>, Status> {
        let principal = extract_principal(&request);
        require_capability(&principal, &Capability::Read)?;
        let name = request.into_inner().index_name;
        let key = Self::index_key(&principal.tenant_id, &name);

        match self.indexes.get(&key) {
            Some(entry) => {
                let stats = entry.index.stats();
                Ok(Response::new(GetStatsResponse {
                    stats: Some(IndexStats {
                        num_vectors: stats.num_vectors as u64,
                        dimension: entry.dimension as u32,
                        max_layer: stats.max_layer as u32,
                        memory_bytes: 0, // Memory stats available via separate call
                        avg_connections: stats.avg_connections,
                    }),
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(GetStatsResponse {
                stats: None,
                error: format!("Index '{}' not found", name),
            })),
        }
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        // SECURITY (CWE-200): a health probe is typically unauthenticated and is
        // not tenant-scoped, so it must NOT enumerate index names — doing so
        // leaked every tenant's index inventory across the tenant boundary. Use
        // the authenticated, tenant-scoped GetStats RPC to inspect a specific
        // index instead. The probe reports only liveness + version.
        Ok(Response::new(HealthCheckResponse {
            status: proto::health_check_response::Status::Serving.into(),
            version: VERSION.to_string(),
            indexes: Vec::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create an index, insert vectors, search, and return the search results.
    async fn create_insert_search(metric: proto::DistanceMetric) -> Vec<SearchResult> {
        let server = VectorIndexServer::new();
        let index_name = "metric_test_index";

        server
            .create_index(Request::new(CreateIndexRequest {
                name: index_name.to_string(),
                dimension: 4,
                config: None,
                metric: metric as i32,
            }))
            .await
            .expect("create_index");

        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: index_name.to_string(),
                ids: vec![1, 2, 3],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
            }))
            .await
            .expect("insert_batch");

        let resp = server
            .search(Request::new(SearchRequest {
                index_name: index_name.to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 3,
                ef: 0,
            }))
            .await
            .expect("search")
            .into_inner();

        assert!(resp.error.is_empty(), "search error: {}", resp.error);
        assert!(!resp.results.is_empty(), "expected at least one result");
        resp.results
    }

    #[tokio::test]
    async fn search_result_reports_cosine_metric() {
        let results = create_insert_search(proto::DistanceMetric::Cosine).await;
        for r in &results {
            assert_eq!(r.metric, "cosine", "every result must carry its metric");
        }
    }

    #[tokio::test]
    async fn search_result_reports_euclidean_metric() {
        let results = create_insert_search(proto::DistanceMetric::L2).await;
        for r in &results {
            assert_eq!(r.metric, "euclidean");
        }
    }

    #[tokio::test]
    async fn search_result_reports_dot_product_metric() {
        let results = create_insert_search(proto::DistanceMetric::DotProduct).await;
        for r in &results {
            assert_eq!(r.metric, "dot_product");
        }
    }

    #[tokio::test]
    async fn search_batch_results_report_metric() {
        let server = VectorIndexServer::new();
        let index_name = "metric_batch_index";

        server
            .create_index(Request::new(CreateIndexRequest {
                name: index_name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .expect("create_index");

        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: index_name.to_string(),
                ids: vec![1, 2],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                ],
            }))
            .await
            .expect("insert_batch");

        let resp = server
            .search_batch(Request::new(SearchBatchRequest {
                index_name: index_name.to_string(),
                #[rustfmt::skip]
                queries: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                ],
                num_queries: 2,
                k: 2,
                ef: 0,
            }))
            .await
            .expect("search_batch")
            .into_inner();

        let mut saw_result = false;
        for query_results in &resp.results {
            for r in &query_results.results {
                saw_result = true;
                assert_eq!(r.metric, "cosine");
            }
        }
        assert!(saw_result, "expected at least one batch result");
    }

    /// SECURITY (CWE-639): vector indexes are scoped to the authenticated tenant.
    /// Two tenants using the SAME index name get isolated indexes; neither can
    /// read, overwrite, or drop the other's.
    #[tokio::test]
    async fn vector_indexes_are_tenant_isolated() {
        use crate::security::{AuthMethod, Principal};
        use std::collections::HashSet;

        fn authed<T>(msg: T, tenant: &str) -> Request<T> {
            let mut r = Request::new(msg);
            r.extensions_mut().insert(Principal {
                id: "u".to_string(),
                tenant_id: tenant.to_string(),
                capabilities: HashSet::from([
                    Capability::Read,
                    Capability::Write,
                    Capability::ManageCollections,
                ]),
                expires_at: None,
                auth_method: AuthMethod::Anonymous,
            });
            r
        }

        let server = VectorIndexServer::new();

        // tenant-a creates "shared" and inserts a vector.
        server
            .create_index(authed(
                CreateIndexRequest {
                    name: "shared".to_string(),
                    dimension: 4,
                    config: None,
                    metric: proto::DistanceMetric::Cosine as i32,
                },
                "tenant-a",
            ))
            .await
            .unwrap();
        server
            .insert_batch(authed(
                InsertBatchRequest {
                    index_name: "shared".to_string(),
                    ids: vec![1],
                    vectors: vec![1.0, 0.0, 0.0, 0.0],
                },
                "tenant-a",
            ))
            .await
            .unwrap();

        // tenant-b searching "shared" must NOT find tenant-a's index.
        let resp_b = server
            .search(authed(
                SearchRequest {
                    index_name: "shared".to_string(),
                    query: vec![1.0, 0.0, 0.0, 0.0],
                    k: 1,
                    ef: 0,
                },
                "tenant-b",
            ))
            .await;
        assert!(resp_b.is_err(), "tenant-b must not access tenant-a's index");

        // tenant-a sees its own index fine.
        let resp_a = server
            .search(authed(
                SearchRequest {
                    index_name: "shared".to_string(),
                    query: vec![1.0, 0.0, 0.0, 0.0],
                    k: 1,
                    ef: 0,
                },
                "tenant-a",
            ))
            .await
            .unwrap()
            .into_inner();
        assert!(resp_a.error.is_empty() && !resp_a.results.is_empty());
    }
}
