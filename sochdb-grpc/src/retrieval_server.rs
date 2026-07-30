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

//! The v2 retrieval service.
//!
//! This runs beside `VectorIndexService`, over the same indexes. v1 is not
//! changed and not deprecated here; a client that is happy with it keeps
//! working. What v2 adds is everything a caller needs to know whether an
//! answer can be used:
//!
//! * indexes are addressed by a stable object id, not by a mutable name, so
//!   renaming an index cannot silently redirect a query to different data;
//! * ids are 128 bits end to end, which is the width the index actually uses;
//! * a caller can pin the generation it wants and be refused rather than
//!   answered from a different one;
//! * every request carries a deadline and a scoped, expiring capability;
//! * a retried write is recognised as the same write and not applied twice;
//! * every response carries a digest of what it committed to.
//!
//! Two limits are worth stating plainly rather than leaving to be discovered.
//!
//! The record of which operations have been applied lives in memory. A
//! restart forgets it, so a batch replayed across a restart is applied again.
//! Closing that needs the applied set to be published with the generation it
//! belongs to; it is not done here.
//!
//! `searchable_watermark` reports the published generation, not an ingestion
//! timestamp. It answers "how far back can this answer be reproduced", which
//! is the question a caller pinning a generation is actually asking.

use crate::auth_interceptor::{extract_principal, require_capability};
use crate::proto::{
    self, BindIndexRequest, BindIndexResponse, RetrievalIngestRequest, RetrievalIngestResponse,
    RetrievalRequest, RetrievalResponse,
    retrieval_service_server::{RetrievalService, RetrievalServiceServer},
};
use crate::retrieval_protocol::{
    CapabilityIssuer, ProtocolError, check_contract, check_deadline, check_generation,
    evidence_digest, from_wire, now_unix_ms, operation_id, to_wire,
};
use crate::security::Capability;
use crate::server::VectorIndexServer;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};

/// What a completed ingest recorded, so a replay can answer identically
/// without doing the work again.
#[derive(Clone)]
struct AppliedOperation {
    inserted_count: u32,
    generation: u64,
}

pub struct RetrievalServer {
    vectors: Arc<VectorIndexServer>,
    issuer: CapabilityIssuer,
    /// Stable object id -> the index key it names. A binding is per tenant
    /// because the key it resolves to is.
    bindings: DashMap<u128, String>,
    applied: DashMap<String, AppliedOperation>,
}

impl RetrievalServer {
    pub fn new(vectors: Arc<VectorIndexServer>, issuer: CapabilityIssuer) -> Self {
        Self {
            vectors,
            issuer,
            bindings: DashMap::new(),
            applied: DashMap::new(),
        }
    }

    /// Issue a capability. Exposed so that whoever authorises a request can
    /// mint the grant that carries that decision to the executing service.
    pub fn issuer(&self) -> &CapabilityIssuer {
        &self.issuer
    }

    pub fn into_service(self) -> RetrievalServiceServer<Self> {
        RetrievalServiceServer::new(self)
    }

    /// Resolve an object id to the index key it names, refusing to cross a
    /// tenant boundary even if the binding exists.
    fn resolve(&self, object_id: u128, tenant: &str) -> Result<String, ProtocolError> {
        let key = self
            .bindings
            .get(&object_id)
            .map(|k| k.clone())
            .ok_or(ProtocolError::UnboundIndex { object_id })?;
        // A binding is a name inside a tenant. Returning one tenant's key to
        // another would defeat the isolation the key prefix exists to provide,
        // even though the capability check would also catch it -- two
        // independent checks, because this one is cheap.
        if !key.starts_with(&format!("{}:", tenant)) {
            return Err(ProtocolError::UnboundIndex { object_id });
        }
        Ok(key)
    }
}

/// Map a protocol failure onto a status code that says what the caller should
/// do about it. Collapsing these all onto `invalid_argument` would tell a
/// caller to fix its request when the right action is often to retry, refresh
/// a token, or stop.
fn to_status(e: ProtocolError) -> Status {
    let message = e.to_string();
    match e {
        ProtocolError::MissingCapability
        | ProtocolError::InvalidSignature
        | ProtocolError::Expired { .. }
        | ProtocolError::LifetimeTooLong { .. } => Status::unauthenticated(message),
        ProtocolError::WrongIndex { .. }
        | ProtocolError::WrongTenant { .. }
        | ProtocolError::WrongOperation
        | ProtocolError::PolicyScopeMismatch { .. } => Status::permission_denied(message),
        ProtocolError::DeadlineExceeded { .. } => Status::deadline_exceeded(message),
        // Stale generation is `aborted`, not `failed_precondition`: the caller
        // can succeed by retrying at the generation it was told about, which is
        // exactly what `aborted` means in gRPC.
        ProtocolError::StaleGeneration { .. } => Status::aborted(message),
        ProtocolError::UnboundIndex { .. } => Status::not_found(message),
        ProtocolError::IncompatibleMajor { .. }
        | ProtocolError::ServerTooOld { .. }
        | ProtocolError::Malformed(_) => Status::invalid_argument(message),
    }
}

#[tonic::async_trait]
impl RetrievalService for RetrievalServer {
    async fn bind_index(
        &self,
        request: Request<BindIndexRequest>,
    ) -> Result<Response<BindIndexResponse>, Status> {
        let principal = extract_principal(&request);
        require_capability(&principal, &Capability::ManageCollections)?;
        let req = request.into_inner();

        let object_id = match req.index_object_id.as_ref() {
            Some(id) => from_wire(id),
            None => {
                return Ok(Response::new(BindIndexResponse {
                    success: false,
                    error: "index_object_id is required".to_string(),
                }));
            }
        };
        // Zero is the value a caller gets by forgetting to set the field, so
        // it must not be a usable identity.
        if object_id == 0 {
            return Ok(Response::new(BindIndexResponse {
                success: false,
                error: "index_object_id must not be zero".to_string(),
            }));
        }

        let key = VectorIndexServer::key_for(&principal.tenant_id, &req.index_name);
        if self.vectors.index_for(&key).is_none() {
            return Ok(Response::new(BindIndexResponse {
                success: false,
                error: format!("Index '{}' not found", req.index_name),
            }));
        }

        // Rebinding an id to a different index is exactly the redirection this
        // protocol exists to prevent, so it is refused rather than allowed as
        // an update.
        if let Some(existing) = self.bindings.get(&object_id) {
            if *existing != key {
                return Ok(Response::new(BindIndexResponse {
                    success: false,
                    error: format!(
                        "object id is already bound to a different index ('{}')",
                        *existing
                    ),
                }));
            }
        }

        self.bindings.insert(object_id, key);
        Ok(Response::new(BindIndexResponse {
            success: true,
            error: String::new(),
        }))
    }

    async fn retrieve(
        &self,
        request: Request<RetrievalRequest>,
    ) -> Result<Response<RetrievalResponse>, Status> {
        let started = Instant::now();
        let principal = extract_principal(&request);
        require_capability(&principal, &Capability::Read)?;
        let req = request.into_inner();
        let now = now_unix_ms();

        check_contract(req.contract_version.as_ref()).map_err(to_status)?;
        check_deadline(req.deadline_unix_ms, now).map_err(to_status)?;

        let object_id = req
            .index_object_id
            .as_ref()
            .map(from_wire)
            .ok_or_else(|| Status::invalid_argument("index_object_id is required"))?;
        let key = self
            .resolve(object_id, &principal.tenant_id)
            .map_err(to_status)?;

        self.issuer
            .verify(
                req.capability.as_ref(),
                &key,
                &principal.tenant_id,
                proto::RetrievalOperation::Search,
                &req.policy_scope_digest,
                now,
            )
            .map_err(to_status)?;

        let (index, dimension) = self
            .vectors
            .index_for(&key)
            .ok_or_else(|| Status::not_found(format!("Index '{}' is not served", key)))?;

        let generation = self.vectors.published_generation(&key);
        check_generation(req.required_index_generation, generation).map_err(to_status)?;

        if req.query_vectors.len() != dimension {
            return Err(Status::invalid_argument(format!(
                "expected a {}-dimensional query, got {} floats",
                dimension,
                req.query_vectors.len()
            )));
        }
        if req.top_k == 0 {
            return Err(Status::invalid_argument("top_k must be at least 1"));
        }

        let mut warnings = Vec::new();
        // The candidate budget bounds work, so it can only ever be raised to
        // top_k, never used to return fewer results than asked for.
        let budget = req.candidate_budget.max(req.top_k) as usize;
        if req.candidate_budget != 0 && req.candidate_budget < req.top_k {
            warnings.push(format!(
                "candidate_budget {} was below top_k {} and was raised",
                req.candidate_budget, req.top_k
            ));
        }
        if generation == 0 {
            warnings.push(
                "this index has no published generation, so this answer cannot be reproduced \
                 after a restart"
                    .to_string(),
            );
        }

        let search_started = Instant::now();
        let query = req.query_vectors.clone();
        let k = req.top_k as usize;
        let hits = tokio::task::spawn_blocking(move || index.search(&query, budget))
            .await
            .map_err(|e| Status::internal(format!("search task failed: {e}")))?
            .map_err(Status::internal)?;
        let search_us = search_started.elapsed().as_micros() as u64;
        let candidates_before_truncation = hits.len();

        let hits: Vec<(u128, f32)> = hits.into_iter().take(k).collect();

        let digest = evidence_digest(object_id, generation, &req.source_snapshot_id, &hits);

        Ok(Response::new(RetrievalResponse {
            index_generation: generation,
            source_snapshot_id: req.source_snapshot_id,
            searchable_watermark: generation,
            candidates: hits
                .iter()
                .map(|(id, distance)| proto::RetrievalCandidate {
                    id: Some(to_wire(*id)),
                    distance: *distance,
                })
                .collect(),
            // HNSW is approximate. Reporting otherwise would let a caller
            // present these results as exhaustive.
            approximate: true,
            recall_configuration: format!("hnsw;candidate_budget={budget}"),
            stages: vec![
                proto::RetrievalStage {
                    name: "hnsw_search".to_string(),
                    duration_us: search_us,
                    candidates_in: 0,
                    candidates_out: candidates_before_truncation as u32,
                },
                proto::RetrievalStage {
                    name: "top_k".to_string(),
                    duration_us: 0,
                    candidates_in: candidates_before_truncation as u32,
                    candidates_out: hits.len() as u32,
                },
            ],
            total_duration_us: started.elapsed().as_micros() as u64,
            warnings,
            evidence_digest: digest,
            query_id: req.query_id,
        }))
    }

    async fn ingest(
        &self,
        request: Request<RetrievalIngestRequest>,
    ) -> Result<Response<RetrievalIngestResponse>, Status> {
        let principal = extract_principal(&request);
        require_capability(&principal, &Capability::Write)?;
        let req = request.into_inner();
        let now = now_unix_ms();

        check_contract(req.contract_version.as_ref()).map_err(to_status)?;
        check_deadline(req.deadline_unix_ms, now).map_err(to_status)?;

        let object_id = req
            .index_object_id
            .as_ref()
            .map(from_wire)
            .ok_or_else(|| Status::invalid_argument("index_object_id is required"))?;
        let key = self
            .resolve(object_id, &principal.tenant_id)
            .map_err(to_status)?;

        self.issuer
            .verify(
                req.capability.as_ref(),
                &key,
                &principal.tenant_id,
                proto::RetrievalOperation::Ingest,
                &req.policy_scope_digest,
                now,
            )
            .map_err(to_status)?;

        let op = operation_id(
            object_id,
            req.index_generation,
            &req.source_snapshot_id,
            req.batch_sequence,
        );

        // The duplicate check comes before validation deliberately. A caller
        // retrying a batch it already sent must get the same answer, and a
        // batch that was accepted once cannot become invalid later.
        if let Some(previous) = self.applied.get(&op) {
            return Ok(Response::new(RetrievalIngestResponse {
                operation_id: op,
                applied: false,
                inserted_count: previous.inserted_count,
                index_generation: previous.generation,
            }));
        }

        let (index, dimension) = self
            .vectors
            .index_for(&key)
            .ok_or_else(|| Status::not_found(format!("Index '{}' is not served", key)))?;

        if req.vectors.len() != req.ids.len() * dimension {
            return Err(Status::invalid_argument(format!(
                "expected {} floats for {} vectors of dimension {}, got {}",
                req.ids.len() * dimension,
                req.ids.len(),
                dimension,
                req.vectors.len()
            )));
        }

        let ids: Vec<u128> = req.ids.iter().map(from_wire).collect();
        let vectors = req.vectors;
        let inserted = tokio::task::spawn_blocking(move || {
            let mut count = 0u32;
            for (i, id) in ids.iter().enumerate() {
                let start = i * dimension;
                index.insert(*id, vectors[start..start + dimension].to_vec())?;
                count += 1;
            }
            Ok::<u32, String>(count)
        })
        .await
        .map_err(|e| Status::internal(format!("ingest task failed: {e}")))?
        .map_err(Status::internal)?;

        self.vectors.note_inserts(&key, inserted as u64).await;
        let generation = self.vectors.published_generation(&key);

        // Recorded only after the vectors are in. Recording first would make a
        // failed ingest look applied, and the retry -- the one thing that would
        // have fixed it -- would be answered from the record.
        self.applied.insert(
            op.clone(),
            AppliedOperation {
                inserted_count: inserted,
                generation,
            },
        );

        Ok(Response::new(RetrievalIngestResponse {
            operation_id: op,
            applied: true,
            inserted_count: inserted,
            index_generation: generation,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::vector_index_service_server::VectorIndexService;
    use crate::proto::{CreateIndexRequest, RetrievalContractVersion};
    use crate::retrieval_protocol::CapabilityScope;
    use crate::security::{AuthMethod, Principal};
    use crate::server::VectorIndexServer;
    use std::collections::HashSet;
    use tonic::Request as TonicRequest;

    fn authed<T>(msg: T, tenant: &str) -> TonicRequest<T> {
        let mut r = TonicRequest::new(msg);
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

    fn version() -> Option<RetrievalContractVersion> {
        Some(RetrievalContractVersion { major: 2, minor: 0 })
    }

    /// Build a server with one 4-dimensional index named `docs`, owned by
    /// tenant `t` and bound to object id 42.
    async fn create_index(vectors: &VectorIndexServer, tenant: &str, name: &str) {
        let response = vectors
            .create_index(authed(
                CreateIndexRequest {
                    name: name.to_string(),
                    dimension: 4,
                    config: None,
                    metric: proto::DistanceMetric::Cosine as i32,
                },
                tenant,
            ))
            .await
            .expect("index creation failed")
            .into_inner();
        assert!(response.success, "{}", response.error);
    }

    async fn fixture() -> RetrievalServer {
        let vectors = Arc::new(VectorIndexServer::new());
        create_index(&vectors, "t", "docs").await;
        let server = RetrievalServer::new(vectors, CapabilityIssuer::new(b"test-key".to_vec()));
        let bound = server
            .bind_index(authed(
                BindIndexRequest {
                    index_object_id: Some(to_wire(42)),
                    index_name: "docs".to_string(),
                },
                "t",
            ))
            .await
            .unwrap()
            .into_inner();
        assert!(bound.success, "binding failed: {}", bound.error);
        server
    }

    fn token(
        server: &RetrievalServer,
        key: &str,
        tenant: &str,
        op: proto::RetrievalOperation,
    ) -> Option<proto::RetrievalCapabilityToken> {
        Some(server.issuer().issue(CapabilityScope {
            index_key: key.to_string(),
            tenant_id: tenant.to_string(),
            expires_at_unix_ms: now_unix_ms() + 60_000,
            policy_scope_digest: "policy-a".to_string(),
            operation: op as i32,
        }))
    }

    fn search_req(server: &RetrievalServer, object_id: u128) -> RetrievalRequest {
        RetrievalRequest {
            contract_version: version(),
            query_id: "q1".to_string(),
            trace_context: String::new(),
            table_object_id: None,
            index_object_id: Some(to_wire(object_id)),
            required_index_generation: 0,
            source_snapshot_id: "snap-1".to_string(),
            query_vectors: vec![1.0, 0.0, 0.0, 0.0],
            query_count: 1,
            top_k: 2,
            candidate_budget: 16,
            filter_ir: None,
            retrieval_profile: String::new(),
            deadline_unix_ms: 0,
            capability: token(server, "t:docs", "t", proto::RetrievalOperation::Search),
            policy_scope_digest: "policy-a".to_string(),
        }
    }

    fn ingest_req(server: &RetrievalServer, sequence: u64) -> RetrievalIngestRequest {
        RetrievalIngestRequest {
            contract_version: version(),
            index_object_id: Some(to_wire(42)),
            index_generation: 0,
            source_snapshot_id: "snap-1".to_string(),
            batch_sequence: sequence,
            // An id above u64::MAX, which v1 could not have carried back.
            ids: vec![to_wire(u64::MAX as u128 + 7), to_wire(3)],
            vectors: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            deadline_unix_ms: 0,
            capability: token(server, "t:docs", "t", proto::RetrievalOperation::Ingest),
            policy_scope_digest: "policy-a".to_string(),
        }
    }

    /// The reason 128-bit ids exist in v2: an id that does not fit in 64 bits
    /// must come back exactly as it went in. Under v1 this id would have been
    /// reported as 6.
    #[tokio::test]
    async fn an_id_too_large_for_v1_survives_a_round_trip() {
        let server = fixture().await;
        server
            .ingest(authed(ingest_req(&server, 0), "t"))
            .await
            .unwrap();

        let response = server
            .retrieve(authed(search_req(&server, 42), "t"))
            .await
            .unwrap()
            .into_inner();

        let ids: Vec<u128> = response
            .candidates
            .iter()
            .map(|c| from_wire(c.id.as_ref().unwrap()))
            .collect();
        assert_eq!(ids[0], u64::MAX as u128 + 7);
    }

    /// A caller that cannot tell whether its write arrived must be able to
    /// simply send it again. The second call reports `applied: false` and the
    /// index is not larger.
    #[tokio::test]
    async fn a_replayed_batch_is_not_applied_twice() {
        let server = fixture().await;

        let first = server
            .ingest(authed(ingest_req(&server, 0), "t"))
            .await
            .unwrap()
            .into_inner();
        assert!(first.applied);
        assert_eq!(first.inserted_count, 2);

        let (index, _) = server.vectors.index_for("t:docs").unwrap();
        let after_first = index.len();

        let second = server
            .ingest(authed(ingest_req(&server, 0), "t"))
            .await
            .unwrap()
            .into_inner();
        assert!(!second.applied, "the replay was applied a second time");
        assert_eq!(second.operation_id, first.operation_id);
        assert_eq!(
            second.inserted_count, first.inserted_count,
            "a replay must report what the original did"
        );
        assert_eq!(index.len(), after_first, "the replay changed the index");
    }

    /// A genuinely different batch must not be mistaken for a replay, or real
    /// data is silently discarded.
    #[tokio::test]
    async fn a_different_batch_is_still_applied() {
        let server = fixture().await;
        server
            .ingest(authed(ingest_req(&server, 0), "t"))
            .await
            .unwrap();
        let second = server
            .ingest(authed(ingest_req(&server, 1), "t"))
            .await
            .unwrap()
            .into_inner();
        assert!(second.applied);
    }

    /// Pinning a generation the server cannot produce must fail loudly. The
    /// failure mode this prevents is a caller joining these results to other
    /// data derived from generation 9 and getting a silently different answer.
    #[tokio::test]
    async fn a_request_pinned_to_an_unavailable_generation_is_refused() {
        let server = fixture().await;
        let mut req = search_req(&server, 42);
        req.required_index_generation = 9;
        let status = server
            .retrieve(authed(req, "t"))
            .await
            .expect_err("a pinned generation that does not exist was answered anyway");
        assert_eq!(status.code(), tonic::Code::Aborted);
    }

    /// An unbound object id is not reachable at all. This is what stops a v2
    /// caller from falling back to addressing an index by name.
    #[tokio::test]
    async fn an_unbound_object_id_cannot_be_queried() {
        let server = fixture().await;
        let status = server
            .retrieve(authed(search_req(&server, 999), "t"))
            .await
            .expect_err("an unbound object id was resolved");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    /// Rebinding is the redirection attack in one call: bind the id another
    /// tenant's plan already uses to an index you control.
    #[tokio::test]
    async fn an_object_id_cannot_be_rebound_to_another_index() {
        let server = fixture().await;
        create_index(&server.vectors, "t", "other").await;
        let response = server
            .bind_index(authed(
                BindIndexRequest {
                    index_object_id: Some(to_wire(42)),
                    index_name: "other".to_string(),
                },
                "t",
            ))
            .await
            .unwrap()
            .into_inner();
        assert!(!response.success, "an object id was redirected");
    }

    /// Binding the same id to the same index again is not a redirection and
    /// must stay idempotent, or a retried bind breaks a working client.
    #[tokio::test]
    async fn rebinding_the_same_pair_is_idempotent() {
        let server = fixture().await;
        let response = server
            .bind_index(authed(
                BindIndexRequest {
                    index_object_id: Some(to_wire(42)),
                    index_name: "docs".to_string(),
                },
                "t",
            ))
            .await
            .unwrap()
            .into_inner();
        assert!(response.success, "{}", response.error);
    }

    /// A binding made by one tenant must not be resolvable by another, even
    /// with an otherwise valid request.
    #[tokio::test]
    async fn a_binding_does_not_cross_a_tenant_boundary() {
        let server = fixture().await;
        let mut req = search_req(&server, 42);
        req.capability = token(
            &server,
            "other:docs",
            "other",
            proto::RetrievalOperation::Search,
        );
        let status = server
            .retrieve(authed(req, "other"))
            .await
            .expect_err("one tenant resolved another tenant's binding");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    /// The capability is not decoration. A request without one is refused even
    /// though the caller is authenticated and holds the Read capability.
    #[tokio::test]
    async fn a_request_without_a_capability_is_refused() {
        let server = fixture().await;
        let mut req = search_req(&server, 42);
        req.capability = None;
        let status = server
            .retrieve(authed(req, "t"))
            .await
            .expect_err("a request with no capability was served");
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    /// A search grant must not perform a write, and the refusal must be
    /// `permission_denied` so the caller does not retry it.
    #[tokio::test]
    async fn a_search_capability_cannot_ingest() {
        let server = fixture().await;
        let mut req = ingest_req(&server, 0);
        req.capability = token(&server, "t:docs", "t", proto::RetrievalOperation::Search);
        let status = server
            .ingest(authed(req, "t"))
            .await
            .expect_err("a search grant performed a write");
        assert_eq!(status.code(), tonic::Code::PermissionDenied);
    }

    /// An already-passed deadline means the caller has stopped waiting. Doing
    /// the work anyway burns capacity on an answer nobody will read.
    #[tokio::test]
    async fn a_passed_deadline_is_refused_before_any_work() {
        let server = fixture().await;
        let mut req = search_req(&server, 42);
        req.deadline_unix_ms = 1;
        let status = server
            .retrieve(authed(req, "t"))
            .await
            .expect_err("an expired deadline was served");
        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
    }

    #[tokio::test]
    async fn a_client_from_a_different_major_version_is_refused() {
        let server = fixture().await;
        let mut req = search_req(&server, 42);
        req.contract_version = Some(RetrievalContractVersion { major: 1, minor: 0 });
        let status = server
            .retrieve(authed(req, "t"))
            .await
            .expect_err("a v1 contract was accepted by the v2 service");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    /// The digest exists so two parties can compare answers without exchanging
    /// them: identical answers agree, different ones do not.
    #[tokio::test]
    async fn the_evidence_digest_distinguishes_different_answers() {
        let server = fixture().await;
        server
            .ingest(authed(ingest_req(&server, 0), "t"))
            .await
            .unwrap();

        let a = server
            .retrieve(authed(search_req(&server, 42), "t"))
            .await
            .unwrap()
            .into_inner();
        let b = server
            .retrieve(authed(search_req(&server, 42), "t"))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(a.evidence_digest, b.evidence_digest);
        assert!(!a.evidence_digest.is_empty());

        let mut different = search_req(&server, 42);
        different.query_vectors = vec![0.0, 1.0, 0.0, 0.0];
        let c = server
            .retrieve(authed(different, "t"))
            .await
            .unwrap()
            .into_inner();
        assert_ne!(a.evidence_digest, c.evidence_digest);
    }

    /// An answer that cannot be reproduced must say so rather than look like
    /// one that can.
    #[tokio::test]
    async fn an_unpublished_index_warns_that_its_answer_is_not_reproducible() {
        let server = fixture().await;
        server
            .ingest(authed(ingest_req(&server, 0), "t"))
            .await
            .unwrap();
        let response = server
            .retrieve(authed(search_req(&server, 42), "t"))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.index_generation, 0);
        assert!(
            response.warnings.iter().any(|w| w.contains("restart")),
            "no warning about reproducibility: {:?}",
            response.warnings
        );
        assert!(response.approximate, "an HNSW answer was reported as exact");
    }

    /// A budget below top_k would quietly return fewer results than asked for.
    /// It is raised instead, and the caller is told.
    #[tokio::test]
    async fn a_candidate_budget_below_top_k_is_raised_and_reported() {
        let server = fixture().await;
        server
            .ingest(authed(ingest_req(&server, 0), "t"))
            .await
            .unwrap();
        let mut req = search_req(&server, 42);
        req.top_k = 2;
        req.candidate_budget = 1;
        let response = server
            .retrieve(authed(req, "t"))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.candidates.len(), 2);
        assert!(
            response
                .warnings
                .iter()
                .any(|w| w.contains("candidate_budget")),
            "the budget was raised without telling the caller"
        );
    }
}
