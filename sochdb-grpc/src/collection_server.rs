// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// you may not use this file except in compliance with the License.

//! Collection Service gRPC Implementation
//!
//! Provides collection management for vectors/documents via gRPC.

use crate::auth_interceptor::{extract_principal, require_capability, require_namespace_access};
use crate::proto::{
    collection_service_server::{CollectionService, CollectionServiceServer},
    AddDocumentsRequest, AddDocumentsResponse, Collection, CreateCollectionRequest,
    CreateCollectionResponse, DeleteCollectionRequest, DeleteCollectionResponse,
    DeleteDocumentRequest, DeleteDocumentResponse, Document, DocumentResult,
    GetCollectionRequest, GetCollectionResponse, GetDocumentRequest, GetDocumentResponse,
    ListCollectionsRequest, ListCollectionsResponse, SearchCollectionRequest,
    SearchCollectionResponse,
};
use crate::namespace_server::NamespaceServer;
use crate::security::Capability;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::{Request, Response, Status};

/// In-memory collection storage
struct CollectionData {
    info: Collection,
    documents: DashMap<String, Document>,
}

/// Collection gRPC Server
pub struct CollectionServer {
    collections: DashMap<String, Arc<CollectionData>>,
    /// Shared namespace server for quota enforcement
    ns_server: Option<NamespaceServer>,
}

impl CollectionServer {
    pub fn new() -> Self {
        Self {
            collections: DashMap::new(),
            ns_server: None,
        }
    }

    /// Create with a shared NamespaceServer for quota enforcement.
    pub fn with_namespace_server(ns: NamespaceServer) -> Self {
        Self {
            collections: DashMap::new(),
            ns_server: Some(ns),
        }
    }

    pub fn into_service(self) -> CollectionServiceServer<Self> {
        CollectionServiceServer::new(self)
    }

    fn collection_key(namespace: &str, name: &str) -> String {
        format!("{}:{}", namespace, name)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

impl Default for CollectionServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl CollectionService for CollectionServer {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::ManageCollections)?;
        require_namespace_access(&principal, &req.namespace)?;

        // Enforce collection quota before creating
        if let Some(ref ns) = self.ns_server {
            ns.check_collection_quota(&req.namespace)?;
        }

        let key = Self::collection_key(&req.namespace, &req.name);

        if self.collections.contains_key(&key) {
            return Ok(Response::new(CreateCollectionResponse {
                success: false,
                collection: None,
                error: format!("Collection '{}' already exists", req.name),
            }));
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let collection = Collection {
            name: req.name.clone(),
            namespace: req.namespace.clone(),
            dimension: req.dimension,
            metric: req.metric,
            document_count: 0,
            created_at: now,
            metadata: req.metadata,
        };

        let data = Arc::new(CollectionData {
            info: collection.clone(),
            documents: DashMap::new(),
        });

        self.collections.insert(key, data);

        // Track the new collection in namespace stats
        if let Some(ref ns) = self.ns_server {
            ns.increment_collection_count(&req.namespace);
        }

        Ok(Response::new(CreateCollectionResponse {
            success: true,
            collection: Some(collection),
            error: String::new(),
        }))
    }

    async fn get_collection(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<GetCollectionResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Read)?;
        require_namespace_access(&principal, &req.namespace)?;
        let key = Self::collection_key(&req.namespace, &req.name);

        match self.collections.get(&key) {
            Some(data) => {
                let mut info = data.info.clone();
                info.document_count = data.documents.len() as u64;
                Ok(Response::new(GetCollectionResponse {
                    collection: Some(info),
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(GetCollectionResponse {
                collection: None,
                error: format!("Collection '{}' not found", req.name),
            })),
        }
    }

    async fn list_collections(
        &self,
        request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Read)?;
        if !req.namespace.is_empty() {
            require_namespace_access(&principal, &req.namespace)?;
        }

        let collections: Vec<Collection> = self
            .collections
            .iter()
            .filter(|entry| {
                req.namespace.is_empty() || entry.value().info.namespace == req.namespace
            })
            .map(|entry| {
                let mut info = entry.value().info.clone();
                info.document_count = entry.value().documents.len() as u64;
                info
            })
            .collect();

        Ok(Response::new(ListCollectionsResponse { collections }))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteCollectionResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::ManageCollections)?;
        require_namespace_access(&principal, &req.namespace)?;
        let key = Self::collection_key(&req.namespace, &req.name);

        match self.collections.remove(&key) {
            Some(_) => {
                // Decrement collection count in namespace stats
                if let Some(ref ns) = self.ns_server {
                    ns.decrement_collection_count(&req.namespace);
                }
                Ok(Response::new(DeleteCollectionResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(DeleteCollectionResponse {
                success: false,
                error: format!("Collection '{}' not found", req.name),
            })),
        }
    }

    async fn add_documents(
        &self,
        request: Request<AddDocumentsRequest>,
    ) -> Result<Response<AddDocumentsResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Write)?;
        require_namespace_access(&principal, &req.namespace)?;

        // Enforce vector quota before adding documents
        if let Some(ref ns) = self.ns_server {
            ns.check_vector_quota(&req.namespace, req.documents.len() as u64)?;
        }

        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => {
                let added_count = req.documents.len() as u64;
                let mut ids = Vec::new();
                for doc in req.documents {
                    let id = if doc.id.is_empty() {
                        uuid::Uuid::new_v4().to_string()
                    } else {
                        doc.id.clone()
                    };
                    ids.push(id.clone());

                    let mut stored_doc = doc;
                    stored_doc.id = id.clone();
                    data.documents.insert(id, stored_doc);
                }

                // Track vectors added in namespace stats
                if let Some(ref ns) = self.ns_server {
                    ns.increment_vector_count(&req.namespace, added_count);
                }

                Ok(Response::new(AddDocumentsResponse {
                    added_count: ids.len() as u32,
                    ids,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(AddDocumentsResponse {
                added_count: 0,
                ids: vec![],
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }

    async fn search_collection(
        &self,
        request: Request<SearchCollectionRequest>,
    ) -> Result<Response<SearchCollectionResponse>, Status> {
        let start = std::time::Instant::now();
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Read)?;
        require_namespace_access(&principal, &req.namespace)?;
        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => {
                let tenant_id = principal.tenant_id.clone();
                let is_admin = principal.has_capability(&Capability::Admin);
                let mut scored: Vec<(Document, f32)> = data
                    .documents
                    .iter()
                    .filter(|entry| {
                        // Record-level ACL: non-admin users can only see docs
                        // owned by their tenant (if _tenant_id metadata is set)
                        if !is_admin {
                            if let Some(doc_tenant) = entry.value().metadata.get("_tenant_id") {
                                if doc_tenant != &tenant_id && tenant_id != "default" {
                                    return false;
                                }
                            }
                        }
                        // Apply user metadata filter
                        if req.filter.is_empty() {
                            return true;
                        }
                        for (filter_key, filter_val) in &req.filter {
                            if let Some(doc_val) = entry.value().metadata.get(filter_key) {
                                if doc_val != filter_val {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                        true
                    })
                    .map(|entry| {
                        let doc = entry.value().clone();
                        let score = Self::cosine_similarity(&req.query, &doc.embedding);
                        (doc, score)
                    })
                    .collect();

                // Sort by score descending
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top k
                let results: Vec<DocumentResult> = scored
                    .into_iter()
                    .take(req.k as usize)
                    .map(|(doc, score)| DocumentResult {
                        document: Some(doc),
                        score,
                    })
                    .collect();

                Ok(Response::new(SearchCollectionResponse {
                    results,
                    duration_us: start.elapsed().as_micros() as u64,
                    error: String::new(),
                }))
            }
            None => Ok(Response::new(SearchCollectionResponse {
                results: vec![],
                duration_us: 0,
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }

    async fn get_document(
        &self,
        request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Read)?;
        require_namespace_access(&principal, &req.namespace)?;
        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => match data.documents.get(&req.document_id) {
                Some(doc) => Ok(Response::new(GetDocumentResponse {
                    document: Some(doc.clone()),
                    error: String::new(),
                })),
                None => Ok(Response::new(GetDocumentResponse {
                    document: None,
                    error: format!("Document '{}' not found", req.document_id),
                })),
            },
            None => Ok(Response::new(GetDocumentResponse {
                document: None,
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }

    async fn delete_document(
        &self,
        request: Request<DeleteDocumentRequest>,
    ) -> Result<Response<DeleteDocumentResponse>, Status> {
        let principal = extract_principal(&request);
        let req = request.into_inner();
        require_capability(&principal, &Capability::Write)?;
        require_namespace_access(&principal, &req.namespace)?;
        let key = Self::collection_key(&req.namespace, &req.collection_name);

        match self.collections.get(&key) {
            Some(data) => match data.documents.remove(&req.document_id) {
                Some(_) => Ok(Response::new(DeleteDocumentResponse {
                    success: true,
                    error: String::new(),
                })),
                None => Ok(Response::new(DeleteDocumentResponse {
                    success: false,
                    error: format!("Document '{}' not found", req.document_id),
                })),
            },
            None => Ok(Response::new(DeleteDocumentResponse {
                success: false,
                error: format!("Collection '{}' not found", req.collection_name),
            })),
        }
    }
}
