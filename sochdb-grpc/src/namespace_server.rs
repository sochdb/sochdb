// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// you may not use this file except in compliance with the License.

//! Namespace Service gRPC Implementation
//!
//! Provides namespace management for multi-tenant isolation via gRPC.

use crate::auth_interceptor::{extract_principal, require_capability, require_namespace_access};
use crate::proto::{
    namespace_service_server::{NamespaceService, NamespaceServiceServer},
    CreateNamespaceRequest, CreateNamespaceResponse, DeleteNamespaceRequest,
    DeleteNamespaceResponse, GetNamespaceRequest, GetNamespaceResponse, ListNamespacesRequest,
    ListNamespacesResponse, Namespace, NamespaceQuota, NamespaceStats, SetQuotaRequest,
    SetQuotaResponse,
};
use crate::security::Capability;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::{Request, Response, Status};

/// Namespace gRPC Server
///
/// Cheaply cloneable (inner DashMap is Arc-wrapped) so that other services
/// can hold a clone and call quota / stat-tracking helpers.
#[derive(Clone)]
pub struct NamespaceServer {
    namespaces: Arc<DashMap<String, Namespace>>,
}

impl NamespaceServer {
    pub fn new() -> Self {
        Self {
            namespaces: Arc::new(DashMap::new()),
        }
    }

    pub fn into_service(self) -> NamespaceServiceServer<Self> {
        NamespaceServiceServer::new(self)
    }

    /// Check whether a namespace has capacity for an additional collection.
    /// Returns Ok(()) if under quota, Err(Status) if exceeded.
    pub fn check_collection_quota(&self, namespace: &str) -> Result<(), Status> {
        if let Some(ns) = self.namespaces.get(namespace) {
            if let (Some(quota), Some(stats)) = (&ns.quota, &ns.stats) {
                if quota.max_collections > 0
                    && stats.collection_count >= quota.max_collections
                {
                    return Err(Status::resource_exhausted(format!(
                        "Namespace '{}' collection quota exceeded ({}/{})",
                        namespace, stats.collection_count, quota.max_collections
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check whether a namespace has capacity for additional vectors.
    /// Returns Ok(()) if under quota, Err(Status) if exceeded.
    pub fn check_vector_quota(&self, namespace: &str, additional: u64) -> Result<(), Status> {
        if let Some(ns) = self.namespaces.get(namespace) {
            if let (Some(quota), Some(stats)) = (&ns.quota, &ns.stats) {
                if quota.max_vectors > 0
                    && stats.vector_count + additional > quota.max_vectors
                {
                    return Err(Status::resource_exhausted(format!(
                        "Namespace '{}' vector quota exceeded ({} + {} > {})",
                        namespace, stats.vector_count, additional, quota.max_vectors
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check whether a namespace has capacity for additional storage bytes.
    /// Returns Ok(()) if under quota, Err(Status) if exceeded.
    pub fn check_storage_quota(&self, namespace: &str, additional_bytes: u64) -> Result<(), Status> {
        if let Some(ns) = self.namespaces.get(namespace) {
            if let (Some(quota), Some(stats)) = (&ns.quota, &ns.stats) {
                if quota.max_storage_bytes > 0
                    && stats.storage_bytes + additional_bytes > quota.max_storage_bytes
                {
                    return Err(Status::resource_exhausted(format!(
                        "Namespace '{}' storage quota exceeded ({} + {} > {} bytes)",
                        namespace, stats.storage_bytes, additional_bytes, quota.max_storage_bytes
                    )));
                }
            }
        }
        Ok(())
    }

    /// Increment the collection count for a namespace.
    pub fn increment_collection_count(&self, namespace: &str) {
        if let Some(mut ns) = self.namespaces.get_mut(namespace) {
            if let Some(ref mut stats) = ns.stats {
                stats.collection_count += 1;
            }
        }
    }

    /// Decrement the collection count for a namespace.
    pub fn decrement_collection_count(&self, namespace: &str) {
        if let Some(mut ns) = self.namespaces.get_mut(namespace) {
            if let Some(ref mut stats) = ns.stats {
                stats.collection_count = stats.collection_count.saturating_sub(1);
            }
        }
    }

    /// Increment the vector count for a namespace.
    pub fn increment_vector_count(&self, namespace: &str, count: u64) {
        if let Some(mut ns) = self.namespaces.get_mut(namespace) {
            if let Some(ref mut stats) = ns.stats {
                stats.vector_count += count;
            }
        }
    }

    /// Increment the storage bytes for a namespace.
    pub fn increment_storage_bytes(&self, namespace: &str, bytes: u64) {
        if let Some(mut ns) = self.namespaces.get_mut(namespace) {
            if let Some(ref mut stats) = ns.stats {
                stats.storage_bytes += bytes;
            }
        }
    }

    /// Get a reference to the internal namespaces map (for testing).
    pub fn namespaces(&self) -> &DashMap<String, Namespace> {
        &self.namespaces
    }
}

impl Default for NamespaceServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl NamespaceService for NamespaceServer {
    async fn create_namespace(
        &self,
        request: Request<CreateNamespaceRequest>,
    ) -> Result<Response<CreateNamespaceResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::ManageCollections)?;

        if self.namespaces.contains_key(&req.name) {
            return Ok(Response::new(CreateNamespaceResponse {
                success: false,
                namespace: None,
                error: format!("Namespace '{}' already exists", req.name),
            }));
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let namespace = Namespace {
            name: req.name.clone(),
            description: req.description,
            created_at: now,
            quota: req.quota,
            stats: Some(NamespaceStats {
                storage_bytes: 0,
                vector_count: 0,
                collection_count: 0,
            }),
            metadata: req.metadata,
        };

        self.namespaces.insert(req.name, namespace.clone());

        Ok(Response::new(CreateNamespaceResponse {
            success: true,
            namespace: Some(namespace),
            error: String::new(),
        }))
    }

    async fn get_namespace(
        &self,
        request: Request<GetNamespaceRequest>,
    ) -> Result<Response<GetNamespaceResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Read)?;
        require_namespace_access(&principal, &req.name)?;

        match self.namespaces.get(&req.name) {
            Some(ns) => Ok(Response::new(GetNamespaceResponse {
                namespace: Some(ns.clone()),
                error: String::new(),
            })),
            None => Ok(Response::new(GetNamespaceResponse {
                namespace: None,
                error: format!("Namespace '{}' not found", req.name),
            })),
        }
    }

    async fn list_namespaces(
        &self,
        _request: Request<ListNamespacesRequest>,
    ) -> Result<Response<ListNamespacesResponse>, Status> {
        let principal = extract_principal(&_request);
        require_capability(&principal, &Capability::Read)?;
        let namespaces: Vec<Namespace> = self
            .namespaces
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        Ok(Response::new(ListNamespacesResponse { namespaces }))
    }

    async fn delete_namespace(
        &self,
        request: Request<DeleteNamespaceRequest>,
    ) -> Result<Response<DeleteNamespaceResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Admin)?;
        require_namespace_access(&principal, &req.name)?;

        match self.namespaces.remove(&req.name) {
            Some(_) => Ok(Response::new(DeleteNamespaceResponse {
                success: true,
                error: String::new(),
            })),
            None => Ok(Response::new(DeleteNamespaceResponse {
                success: false,
                error: format!("Namespace '{}' not found", req.name),
            })),
        }
    }

    async fn set_quota(
        &self,
        request: Request<SetQuotaRequest>,
    ) -> Result<Response<SetQuotaResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Admin)?;
        require_namespace_access(&principal, &req.namespace)?;

        match self.namespaces.get_mut(&req.namespace) {
            Some(mut ns) => {
                ns.quota = req.quota;
                Ok(Response::new(SetQuotaResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(SetQuotaResponse {
                success: false,
                error: format!("Namespace '{}' not found", req.namespace),
            })),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server_with_ns(name: &str, max_collections: u64, max_vectors: u64, max_storage: u64) -> NamespaceServer {
        let server = NamespaceServer::new();
        let ns = Namespace {
            name: name.to_string(),
            description: String::new(),
            created_at: 0,
            quota: Some(NamespaceQuota {
                max_storage_bytes: max_storage,
                max_vectors,
                max_collections,
            }),
            stats: Some(NamespaceStats {
                storage_bytes: 0,
                vector_count: 0,
                collection_count: 0,
            }),
            metadata: Default::default(),
        };
        server.namespaces.insert(name.to_string(), ns);
        server
    }

    #[test]
    fn test_collection_quota_allows_under_limit() {
        let server = make_server_with_ns("prod", 5, 0, 0);
        assert!(server.check_collection_quota("prod").is_ok());
    }

    #[test]
    fn test_collection_quota_rejects_at_limit() {
        let server = make_server_with_ns("prod", 2, 0, 0);
        server.increment_collection_count("prod");
        server.increment_collection_count("prod");
        let result = server.check_collection_quota("prod");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn test_vector_quota_allows_under_limit() {
        let server = make_server_with_ns("prod", 0, 1000, 0);
        assert!(server.check_vector_quota("prod", 500).is_ok());
    }

    #[test]
    fn test_vector_quota_rejects_over_limit() {
        let server = make_server_with_ns("prod", 0, 1000, 0);
        server.increment_vector_count("prod", 800);
        assert!(server.check_vector_quota("prod", 300).is_err());
    }

    #[test]
    fn test_storage_quota_rejects_over_limit() {
        let server = make_server_with_ns("prod", 0, 0, 1_000_000);
        server.increment_storage_bytes("prod", 900_000);
        assert!(server.check_storage_quota("prod", 200_000).is_err());
        assert!(server.check_storage_quota("prod", 100_000).is_ok());
    }

    #[test]
    fn test_quota_zero_means_unlimited() {
        // max_collections = 0 means unlimited
        let server = make_server_with_ns("prod", 0, 0, 0);
        for _ in 0..100 {
            server.increment_collection_count("prod");
        }
        assert!(server.check_collection_quota("prod").is_ok());
    }

    #[test]
    fn test_unknown_namespace_passes_quota() {
        let server = NamespaceServer::new();
        // Unknown namespace → no quota to enforce → passes
        assert!(server.check_collection_quota("phantom").is_ok());
    }

    #[test]
    fn test_decrement_collection_count() {
        let server = make_server_with_ns("prod", 2, 0, 0);
        server.increment_collection_count("prod");
        server.increment_collection_count("prod");
        assert!(server.check_collection_quota("prod").is_err());
        server.decrement_collection_count("prod");
        assert!(server.check_collection_quota("prod").is_ok());
    }
}
