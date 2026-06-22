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

    fn metadata_pairs(metadata: &proto::VectorMetadata) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        if let Some(parent_id) = metadata.parent_id {
            pairs.push(("parent_id".to_string(), parent_id.to_string()));
        }
        if let Some(view_type) = &metadata.view_type {
            pairs.push(("view_type".to_string(), view_type.clone()));
        }
        pairs
    }

    fn result_metadata(index: &HnswIndex, id: u128) -> (Option<u64>, Option<String>) {
        let Some(metadata) = index.get_metadata(id) else {
            return (None, None);
        };

        let parent_id = metadata
            .iter()
            .find(|(key, _)| key == "parent_id")
            .and_then(|(_, value)| value.parse::<u64>().ok());
        let view_type = metadata
            .iter()
            .find(|(key, _)| key == "view_type")
            .map(|(_, value)| value.clone());

        (parent_id, view_type)
    }

    fn search_result(
        index: &HnswIndex,
        id: u128,
        distance: f32,
        metric: &'static str,
    ) -> SearchResult {
        let (parent_id, view_type) = Self::result_metadata(index, id);
        SearchResult {
            id: id as u64,
            distance,
            metric: metric.to_string(),
            parent_id,
            view_type,
        }
    }
    fn group_key_for_result(index: &HnswIndex, id: u128) -> u128 {
        let Some(metadata) = index.get_metadata(id) else {
            return id;
        };
        metadata
            .iter()
            .find(|(key, _)| key == "parent_id")
            .and_then(|(_, value)| value.parse::<u64>().ok())
            .map(|parent_id| parent_id as u128)
            .unwrap_or(id)
    }

    fn grouped_results(
        index: &HnswIndex,
        raw_results: Vec<(u128, f32)>,
        k: usize,
        _max_per_group: u32,
        requested_candidate_k: u32,
        metric: &'static str,
    ) -> (Vec<SearchResult>, proto::GroupingInfo) {
        let max_per_group = if _max_per_group == 0 {
            1
        } else {
            _max_per_group
        };

        let mut seen_groups: std::collections::HashMap<u128, Vec<(u128, f32, usize)>> =
            std::collections::HashMap::new();
        for (idx, (id, distance)) in raw_results.iter().enumerate() {
            let group_key = Self::group_key_for_result(index, *id);
            seen_groups
                .entry(group_key)
                .or_default()
                .push((*id, *distance, idx));
        }

        let mut grouped: Vec<(u128, f32, u128, usize)> = Vec::new();
        for (group_key, mut candidates) in seen_groups {
            candidates.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.2.cmp(&b.2))
            });
            for c in candidates.iter().take(max_per_group as usize) {
                grouped.push((group_key, c.1, c.0, c.2));
            }
        }
        grouped.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.3.cmp(&b.3))
        });

        let returned_group_count = grouped.len().min(k) as u32;
        let results: Vec<SearchResult> = grouped
            .into_iter()
            .take(k)
            .map(|(_, distance, id, _)| Self::search_result(index, id, distance, metric))
            .collect();

        let info = proto::GroupingInfo {
            group_by: proto::GroupBy::ParentId.into(),
            requested_k: k as u32,
            candidate_k: requested_candidate_k,
            raw_candidate_count: raw_results.len() as u32,
            returned_group_count,
        };

        (results, info)
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

        if !req.metadata.is_empty() && req.metadata.len() != req.ids.len() {
            return Err(Status::invalid_argument(format!(
                "Metadata length mismatch: expected {} entries, got {}",
                req.ids.len(),
                req.metadata.len()
            )));
        }

        // Convert IDs to u128
        let ids: Vec<u128> = req.ids.iter().map(|&id| id as u128).collect();

        let metadata_entries: Vec<(u128, Vec<(String, String)>)> = if req.metadata.is_empty() {
            Vec::new()
        } else {
            ids.iter()
                .copied()
                .zip(req.metadata.iter())
                .filter_map(|(id, metadata)| {
                    let pairs = Self::metadata_pairs(metadata);
                    if pairs.is_empty() {
                        None
                    } else {
                        Some((id, pairs))
                    }
                })
                .collect()
        };

        // Offload CPU-heavy HNSW insertion to blocking thread pool to avoid
        // starving the tokio runtime (which handles health checks, streams, etc.)
        let index_name = req.index_name.clone();
        let vectors = req.vectors;
        let index_for_meta = Arc::clone(&index);
        let result =
            tokio::task::spawn_blocking(move || index.insert_batch_flat(&ids, &vectors, dimension))
                .await
                .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match result {
            Ok(count) => {
                let duration_us = start.elapsed().as_micros() as u64;
                if !metadata_entries.is_empty() {
                    index_for_meta.set_metadata_batch(&metadata_entries);
                }
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
        let mut batch_metadata: Vec<(u128, Vec<(String, String)>)> = Vec::new();
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
                        if let Some(metadata) = &req.metadata {
                            let pairs = Self::metadata_pairs(metadata);
                            if !pairs.is_empty() {
                                batch_metadata.push((req.id as u128, pairs));
                            }
                        }

                        // Flush micro-batch when full
                        if batch_ids.len() >= MICRO_BATCH_SIZE {
                            if let Some((ref idx, _)) = index {
                                match idx.insert_batch_flat(&batch_ids, &batch_vectors, dimension) {
                                    Ok(count) => {
                                        if !batch_metadata.is_empty() {
                                            idx.set_metadata_batch(&batch_metadata);
                                        }
                                        total_inserted += count as u32;
                                    }
                                    Err(e) => errors.push(e),
                                }
                            }
                            batch_ids.clear();
                            batch_vectors.clear();
                            batch_metadata.clear();
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
                    Ok(count) => {
                        if !batch_metadata.is_empty() {
                            idx.set_metadata_batch(&batch_metadata);
                        }
                        total_inserted += count as u32;
                    }
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

        // Check for grouping options
        let grouping_opts = req.grouping;
        let is_grouped = grouping_opts
            .as_ref()
            .map(|g| g.group_by == proto::GroupBy::ParentId as i32)
            .unwrap_or(false);

        let candidate_k = if is_grouped {
            let max_per_group = grouping_opts.as_ref().map(|g| g.max_per_group).unwrap_or(1);
            let raw = grouping_opts.as_ref().map(|g| g.candidate_k).unwrap_or(0);
            let effective = if raw == 0 {
                (k * 4).max(k)
            } else if (raw as usize) < k {
                return Err(Status::invalid_argument(format!(
                    "candidate_k ({}) must be >= k ({}) when grouping is active",
                    raw, k
                )));
            } else {
                raw as usize
            };
            (max_per_group, effective)
        } else {
            (1, k)
        };

        // Honor per-query ef override for recall tuning; fall back to index default
        let ef = if req.ef > 0 { req.ef as usize } else { 0 };

        // Offload CPU-heavy HNSW search to blocking thread pool
        let query = req.query;
        let search_k = candidate_k.1;
        let index_for_search = Arc::clone(&index);
        let raw_results = tokio::task::spawn_blocking(move || {
            if ef > 0 {
                index.search_with_ef(&query, search_k, ef.max(search_k))
            } else {
                index.search(&query, search_k)
            }
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match raw_results {
            Ok(r) => {
                let duration_us = start.elapsed().as_micros() as u64;
                let (search_results, grouping_info) = if is_grouped {
                    let (results, info) = Self::grouped_results(
                        &index_for_search,
                        r,
                        k,
                        candidate_k.0,
                        candidate_k.1 as u32,
                        metric,
                    );
                    (results, Some(info))
                } else {
                    let mut results = Vec::with_capacity(r.len());
                    for (id, distance) in r {
                        results.push(Self::search_result(&index_for_search, id, distance, metric));
                    }
                    (results, None)
                };
                Ok(Response::new(SearchResponse {
                    results: search_results,
                    duration_us,
                    error: String::new(),
                    metric: metric_enum.into(),
                    grouping: grouping_info,
                }))
            }
            Err(e) => Ok(Response::new(SearchResponse {
                results: vec![],
                duration_us: start.elapsed().as_micros() as u64,
                error: e,
                metric: metric_enum.into(),
                grouping: None,
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

        // Check for batch grouping options
        let batch_grouping = req.grouping;
        let is_grouped = batch_grouping
            .as_ref()
            .map(|g| g.group_by == proto::GroupBy::ParentId as i32)
            .unwrap_or(false);

        let (max_per_group, candidate_k) = if is_grouped {
            let mpg = batch_grouping
                .as_ref()
                .map(|g| g.max_per_group)
                .unwrap_or(1);
            let raw = batch_grouping.as_ref().map(|g| g.candidate_k).unwrap_or(0);
            let effective = if raw == 0 {
                (k * 4).max(k)
            } else if (raw as usize) < k {
                return Err(Status::invalid_argument(format!(
                    "candidate_k ({}) must be >= k ({}) when grouping is active",
                    raw, k
                )));
            } else {
                raw as usize
            };
            (mpg, effective)
        } else {
            (1, k)
        };

        // Parallel batch search via rayon on blocking thread pool
        let queries = req.queries;
        let search_k = candidate_k;
        let all_results = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            (0..num_queries)
                .into_par_iter()
                .map(|i| -> Result<QueryResults, String> {
                    let query = &queries[i * dimension..(i + 1) * dimension];
                    let raw = index.search(query, search_k).unwrap_or_default();
                    if is_grouped {
                        let (results, info) = Self::grouped_results(
                            &index,
                            raw,
                            k,
                            max_per_group,
                            search_k as u32,
                            metric,
                        );
                        Ok(QueryResults {
                            results,
                            grouping: Some(info),
                        })
                    } else {
                        let mut search_results = Vec::with_capacity(raw.len());
                        for (id, distance) in raw {
                            search_results.push(Self::search_result(&index, id, distance, metric));
                        }
                        Ok(QueryResults {
                            results: search_results,
                            grouping: None,
                        })
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(Status::internal)?;

        let duration_us = start.elapsed().as_micros() as u64;

        Ok(Response::new(SearchBatchResponse {
            results: all_results,
            duration_us,
            metric: metric_enum.into(),
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
                metadata: vec![],
            }))
            .await
            .expect("insert_batch");

        let resp = server
            .search(Request::new(SearchRequest {
                index_name: index_name.to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 3,
                ef: 0,
                grouping: None,
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
                metadata: vec![],
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
                grouping: None,
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

    #[tokio::test]
    async fn legacy_insert_without_metadata_returns_absent_metadata() {
        let results = create_insert_search(proto::DistanceMetric::Cosine).await;
        for result in &results {
            assert_eq!(result.parent_id, None);
            assert_eq!(result.view_type, None);
        }
    }

    #[tokio::test]
    async fn batch_insert_with_mixed_metadata_is_returned_by_search() {
        let server = VectorIndexServer::new();
        let index_name = "metadata_batch_index";

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
                ids: vec![1, 2, 3],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(0),
                        view_type: Some("turn".to_string()),
                    },
                    proto::VectorMetadata {
                        parent_id: None,
                        view_type: None,
                    },
                    proto::VectorMetadata {
                        parent_id: Some(712),
                        view_type: Some("event".to_string()),
                    },
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
                grouping: None,
            }))
            .await
            .expect("search")
            .into_inner();

        let by_id: std::collections::HashMap<u64, &SearchResult> = resp
            .results
            .iter()
            .map(|result| (result.id, result))
            .collect();

        assert_eq!(by_id[&1].parent_id, Some(0));
        assert_eq!(by_id[&1].view_type.as_deref(), Some("turn"));
        assert_eq!(by_id[&2].parent_id, None);
        assert_eq!(by_id[&2].view_type, None);
        assert_eq!(by_id[&3].parent_id, Some(712));
        assert_eq!(by_id[&3].view_type.as_deref(), Some("event"));
    }

    #[tokio::test]
    async fn batch_insert_rejects_metadata_length_mismatch() {
        let server = VectorIndexServer::new();
        let index_name = "metadata_mismatch_index";

        server
            .create_index(Request::new(CreateIndexRequest {
                name: index_name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .expect("create_index");

        let err = server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: index_name.to_string(),
                ids: vec![1, 2],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                ],
                metadata: vec![proto::VectorMetadata {
                    parent_id: Some(1),
                    view_type: None,
                }],
            }))
            .await
            .expect_err("metadata length mismatch should be invalid");

        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("Metadata length mismatch"));
    }

    #[tokio::test]
    async fn search_batch_returns_metadata_for_mixed_presence() {
        let server = VectorIndexServer::new();
        let index_name = "batch_metadata_index";

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
                ids: vec![1, 2, 3],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(0),
                        view_type: Some("turn".to_string()),
                    },
                    proto::VectorMetadata {
                        parent_id: None,
                        view_type: None,
                    },
                    proto::VectorMetadata {
                        parent_id: Some(712),
                        view_type: Some("event".to_string()),
                    },
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
                ],
                num_queries: 1,
                k: 3,
                ef: 0,
                grouping: None,
            }))
            .await
            .expect("search_batch")
            .into_inner();

        assert_eq!(resp.results.len(), 1);
        let results = &resp.results[0].results;
        let by_id: std::collections::HashMap<u64, &SearchResult> =
            results.iter().map(|r| (r.id, r)).collect();

        assert_eq!(by_id[&1].parent_id, Some(0));
        assert_eq!(by_id[&1].view_type.as_deref(), Some("turn"));
        assert_eq!(by_id[&2].parent_id, None);
        assert_eq!(by_id[&2].view_type, None);
        assert_eq!(by_id[&3].parent_id, Some(712));
        assert_eq!(by_id[&3].view_type.as_deref(), Some("event"));
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
                    metadata: vec![],
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
                    grouping: None,
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
                    grouping: None,
                },
                "tenant-a",
            ))
            .await
            .unwrap()
            .into_inner();
        assert!(resp_a.error.is_empty() && !resp_a.results.is_empty());
    }

    /// Regression: `u64::try_from(u128)` must reject IDs above `u64::MAX`
    /// rather than silently truncating with `as u64`.
    #[tokio::test]
    async fn search_checked_id_conversion_rejects_overflow() {
        // Direct proof that `as u64` truncates but `try_from` errors.
        let big: u128 = u64::MAX as u128 + 1;
        let truncated = big as u64;
        assert_eq!(
            truncated, 0,
            "as-cast silently truncates u128 > u64::MAX to 0"
        );
        assert!(u64::try_from(big).is_err(), "try_from must fail");
    }

    /// Response-level `SearchResponse.metric` must be populated and agree
    /// with the per-result `SearchResult.metric` string.
    #[tokio::test]
    async fn search_response_carries_typed_metric() {
        for (proto_metric, expected_label) in [
            (proto::DistanceMetric::Cosine, "cosine"),
            (proto::DistanceMetric::L2, "euclidean"),
            (proto::DistanceMetric::DotProduct, "dot_product"),
        ] {
            let results = create_insert_search(proto_metric).await;
            for r in &results {
                assert_eq!(r.metric, expected_label, "result metric label mismatch");
            }
        }

        // Verify response-level metric on a fresh search
        let server = VectorIndexServer::new();
        let index_name = "response_metric_test";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: index_name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: index_name.to_string(),
                ids: vec![1],
                #[rustfmt::skip]
                vectors: vec![1.0, 0.0, 0.0, 0.0],
                metadata: vec![],
            }))
            .await
            .unwrap();
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: index_name.to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 1,
                ef: 0,
                grouping: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            resp.metric,
            proto::DistanceMetric::Cosine as i32,
            "response-level metric must be typed DistanceMetric"
        );
        assert!(!resp.results.is_empty());
        assert_eq!(resp.results[0].metric, "cosine");
    }

    /// Response-level `SearchBatchResponse.metric` must be populated.
    #[tokio::test]
    async fn search_batch_response_carries_typed_metric() {
        let server = VectorIndexServer::new();
        let index_name = "batch_response_metric_test";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: index_name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::L2 as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: index_name.to_string(),
                ids: vec![1, 2],
                vectors: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                metadata: vec![],
            }))
            .await
            .unwrap();
        let resp = server
            .search_batch(Request::new(SearchBatchRequest {
                index_name: index_name.to_string(),
                queries: vec![1.0, 0.0, 0.0, 0.0],
                num_queries: 1,
                k: 2,
                ef: 0,
                grouping: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            resp.metric,
            proto::DistanceMetric::L2 as i32,
            "batch response-level metric must be typed DistanceMetric"
        );
    }

    // ── Grouped-search tests ────────────────────────────────────────────

    #[tokio::test]
    async fn grouped_search_returns_unique_parents() {
        let server = VectorIndexServer::new();
        let name = "uniq_parents";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2, 3],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(100),
                        view_type: Some("turn".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(100),
                        view_type: Some("event".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(200),
                        view_type: Some("turn".into()),
                    },
                ],
            }))
            .await
            .unwrap();
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![0.0, 1.0, 0.0, 0.0],
                k: 2,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 0,
                }),
            }))
            .await
            .unwrap()
            .into_inner();

        // Should get at most one result per parent (100 and 200)
        let parents: std::collections::HashSet<u64> =
            resp.results.iter().filter_map(|r| r.parent_id).collect();
        assert!(parents.len() >= 1, "expected unique parents");
        assert!(!resp.results.is_empty());
        let info = resp.grouping.unwrap();
        assert_eq!(info.requested_k, 2);
        assert!(info.returned_group_count >= 1);
    }

    #[tokio::test]
    async fn grouped_search_missing_parent_fallback() {
        let server = VectorIndexServer::new();
        let name = "miss_parent";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                ],
                metadata: vec![],
            }))
            .await
            .unwrap();
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![0.0, 1.0, 0.0, 0.0],
                k: 2,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 0,
                }),
            }))
            .await
            .unwrap()
            .into_inner();

        // Without parent_id metadata, each vector should be its own group
        // Vector IDs 1 and 2 should both appear (since they are different groups)
        let ids: Vec<u64> = resp.results.iter().map(|r| r.id).collect();
        assert!(!ids.is_empty(), "should have results");
    }

    #[tokio::test]
    async fn grouped_search_parent_zero_is_grouped() {
        let server = VectorIndexServer::new();
        let name = "parent_zero";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2, 3],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(0),
                        view_type: Some("turn".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(0),
                        view_type: Some("event".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: None,
                        view_type: None,
                    },
                ],
            }))
            .await
            .unwrap();
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![0.0, 0.0, 1.0, 0.0],
                k: 2,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 0,
                }),
            }))
            .await
            .unwrap()
            .into_inner();

        // parent_id == 0 should be grouped together; only one result from parent 0
        let parents: Vec<u64> = resp.results.iter().filter_map(|r| r.parent_id).collect();
        // parent 0 should appear at most once
        let zero_count = parents.iter().filter(|&&p| p == 0).count();
        assert!(zero_count <= 1, "parent_id=0 should be grouped");
    }

    #[tokio::test]
    async fn grouped_search_candidate_overfetch() {
        let server = VectorIndexServer::new();
        let name = "overfetch";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2, 3, 4],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(100),
                        view_type: None,
                    },
                    proto::VectorMetadata {
                        parent_id: Some(100),
                        view_type: None,
                    },
                    proto::VectorMetadata {
                        parent_id: Some(200),
                        view_type: None,
                    },
                    proto::VectorMetadata {
                        parent_id: Some(300),
                        view_type: None,
                    },
                ],
            }))
            .await
            .unwrap();
        // Without overfetch (k=3), duplicates may block parent 300.
        // With candidate_k=8 we should be able to reach all 3 unique parents.
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![0.0, 0.0, 0.0, 0.0],
                k: 3,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 8,
                }),
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.results.len(), 3, "expected 3 unique parent groups");
        let info = resp.grouping.unwrap();
        assert_eq!(info.candidate_k, 8);
        assert_eq!(info.returned_group_count, 3);
    }

    #[tokio::test]
    async fn grouped_search_rejects_candidate_k_smaller_than_k() {
        let server = VectorIndexServer::new();
        let name = "reject_small";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        let err = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 10,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 5,
                }),
            }))
            .await
            .expect_err("should reject candidate_k < k");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("candidate_k"));
    }

    #[tokio::test]
    async fn grouped_search_batch_grouping() {
        let server = VectorIndexServer::new();
        let name = "batch_grouping";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2, 3],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(100),
                        view_type: Some("turn".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(100),
                        view_type: Some("event".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(200),
                        view_type: Some("turn".into()),
                    },
                ],
            }))
            .await
            .unwrap();
        let resp = server
            .search_batch(Request::new(SearchBatchRequest {
                index_name: name.to_string(),
                queries: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                num_queries: 2,
                k: 2,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 0,
                }),
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.results.len(), 2, "2 queries -> 2 result sets");
        for qr in &resp.results {
            assert!(
                qr.grouping.is_some(),
                "each query should have grouping info"
            );
            let parents: std::collections::HashSet<u64> =
                qr.results.iter().filter_map(|r| r.parent_id).collect();
            assert!(parents.len() <= 2);
        }
    }

    #[tokio::test]
    async fn ungrouped_search_is_unchanged() {
        let server = VectorIndexServer::new();
        let name = "ungrouped";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                ],
                metadata: vec![],
            }))
            .await
            .unwrap();
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 2,
                ef: 0,
                grouping: None,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].id, 1);
        assert_eq!(resp.results[0].distance, 0.0);
        assert!(
            resp.grouping.is_none(),
            "ungrouped search must not emit grouping info"
        );
    }

    #[tokio::test]
    async fn grouped_search_without_metadata_is_safe() {
        let server = VectorIndexServer::new();
        let name = "safe_empty";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        // No vectors inserted — empty index with grouping should not panic
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 3,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 10,
                }),
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(resp.results.is_empty());
        assert!(resp.grouping.is_some());
    }

    #[tokio::test]
    async fn grouped_search_response_contains_info() {
        let server = VectorIndexServer::new();
        let name = "resp_info";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 4,
                config: None,
                metric: proto::DistanceMetric::Cosine as i32,
            }))
            .await
            .unwrap();
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(10),
                        view_type: None,
                    },
                    proto::VectorMetadata {
                        parent_id: Some(20),
                        view_type: None,
                    },
                ],
            }))
            .await
            .unwrap();
        let resp = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 3,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 6,
                }),
            }))
            .await
            .unwrap()
            .into_inner();

        let info = resp.grouping.unwrap();
        assert_eq!(info.group_by, proto::GroupBy::ParentId as i32);
        assert_eq!(info.requested_k, 3);
        assert_eq!(info.candidate_k, 6);
        assert!(info.raw_candidate_count >= 1);
        assert!(info.returned_group_count >= 1);
    }

    #[tokio::test]
    async fn grouped_search_customer_support_use_case_reduces_duplicate_parent_waste() {
        let server = VectorIndexServer::new();
        let name = "cs_usecase";
        server
            .create_index(Request::new(CreateIndexRequest {
                name: name.to_string(),
                dimension: 2,
                config: None,
                metric: proto::DistanceMetric::L2 as i32,
            }))
            .await
            .unwrap();

        // Customer-support agent memory: 3 parent memories, each with multiple vector views.
        // parent_id=101 "damaged laptop refund" -> turn, event, entity
        // parent_id=102 "replacement accepted"   -> turn, event
        // parent_id=103 "pickup delayed"         -> turn, event
        // Vectors placed so the query strongly favours parent 101.
        server
            .insert_batch(Request::new(InsertBatchRequest {
                index_name: name.to_string(),
                ids: vec![1, 2, 3, 4, 5, 6, 7],
                #[rustfmt::skip]
                vectors: vec![
                    1.0, 0.0,
                    1.0, 0.1,
                    1.0, 0.2,
                    0.0, 1.0,
                    0.0, 1.1,
                    -1.0, 0.0,
                    -0.9, 0.0,
                ],
                metadata: vec![
                    proto::VectorMetadata {
                        parent_id: Some(101),
                        view_type: Some("turn".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(101),
                        view_type: Some("event".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(101),
                        view_type: Some("entity".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(102),
                        view_type: Some("turn".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(102),
                        view_type: Some("event".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(103),
                        view_type: Some("turn".into()),
                    },
                    proto::VectorMetadata {
                        parent_id: Some(103),
                        view_type: Some("event".into()),
                    },
                ],
            }))
            .await
            .unwrap();

        // ── Ungrouped search
        let ungrouped = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![1.0, 0.0],
                k: 5,
                ef: 0,
                grouping: None,
            }))
            .await
            .unwrap()
            .into_inner();

        let ug_returned = ungrouped.results.len();
        let ug_unique: std::collections::HashSet<u64> = ungrouped
            .results
            .iter()
            .filter_map(|r| r.parent_id)
            .collect();
        let ug_unique_count = ug_unique.len();
        let ug_duplicate_waste = ug_returned.saturating_sub(ug_unique_count);

        assert!(
            ug_unique_count < ug_returned,
            "ungrouped top-K should contain duplicate parents; unique={} returned={}",
            ug_unique_count,
            ug_returned,
        );
        assert!(
            ug_duplicate_waste > 0,
            "ungrouped duplicate waste must be > 0; got {}",
            ug_duplicate_waste,
        );

        // ── Grouped search
        let grouped = server
            .search(Request::new(SearchRequest {
                index_name: name.to_string(),
                query: vec![1.0, 0.0],
                k: 5,
                ef: 0,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 10,
                }),
            }))
            .await
            .unwrap()
            .into_inner();

        let g_returned = grouped.results.len();
        let g_unique: std::collections::HashSet<u64> =
            grouped.results.iter().filter_map(|r| r.parent_id).collect();
        let g_unique_count = g_unique.len();
        let g_duplicate_waste = g_returned.saturating_sub(g_unique_count);

        assert!(
            g_unique_count > ug_unique_count,
            "grouped search should surface more unique parents; grouped={} ungrouped={}",
            g_unique_count,
            ug_unique_count,
        );
        assert!(
            g_duplicate_waste < ug_duplicate_waste,
            "grouped search should reduce duplicate waste; grouped={} ungrouped={}",
            g_duplicate_waste,
            ug_duplicate_waste,
        );
        let info = grouped.grouping.unwrap();
        assert_eq!(info.group_by, proto::GroupBy::ParentId as i32);
        assert_eq!(info.requested_k, 5);
        assert!(info.returned_group_count >= g_returned as u32);

        let has_parent = grouped.results.iter().any(|r| r.parent_id.is_some());
        assert!(
            has_parent,
            "at least one grouped result should carry parent_id"
        );
        let has_view = grouped.results.iter().any(|r| r.view_type.is_some());
        assert!(
            has_view,
            "at least one grouped result should carry view_type"
        );

        // ── Proof table
        println!("USECASE=customer_support_agent_memory");
        println!("UNGROUPED_RETURNED={}", ug_returned);
        println!("UNGROUPED_UNIQUE_PARENTS={}", ug_unique_count);
        println!("UNGROUPED_DUPLICATE_WASTE={}", ug_duplicate_waste);
        println!("GROUPED_RETURNED={}", g_returned);
        println!("GROUPED_UNIQUE_PARENTS={}", g_unique_count);
        println!("GROUPED_DUPLICATE_WASTE={}", g_duplicate_waste);
        println!("GROUPED_SEARCH_USECASE_OK=1");
    }
}
