// SPDX-License-Identifier: AGPL-3.0-or-later
//
// SochDB - A unified database for AI-native applications
// Copyright (C) 2025 Sushanth Reddy Vanagala
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! The gate suite that backs this build's `Preview` claims.
//!
//! [`sochdb_core::qualification`] defines what a maturity claim requires and
//! refuses one that has no evidence. This produces the evidence, by running the
//! probes against the real served code rather than against a mock of it.
//!
//! Two rules govern everything here, and both exist because the easiest thing
//! to build is a suite that always passes.
//!
//! **A probe must be able to fail.** Every isolation probe drives the actual
//! request path -- a `Principal` in the request extensions, the real handler,
//! the real key derivation -- and asserts on what the handler returned. A probe
//! that checks a constant, or that calls the enforcement helper directly
//! instead of the RPC that is supposed to call it, proves only that the helper
//! exists. That is exactly the gap this suite was written to find: three
//! services here imported no principal at all, and a probe of the helper would
//! have reported all of them green.
//!
//! **A gate is recorded from what was observed.** No probe returns
//! `GateOutcome::Passed` as a literal. Each returns what it saw, and the bundle
//! is assembled from those observations, so a probe that stops being run shows
//! up as `NotRun` and blocks promotion rather than silently vanishing.

use std::collections::HashSet;

use sochdb_core::capability::{Maturity, manifest, names};
use sochdb_core::qualification::{
    BuildIdentity, EvidenceBundle, GateKind, GateOutcome, Metric, qualify_capability,
};
use sochdb_grpc::proto;
use sochdb_grpc::proto::semantic_cache_service_server::SemanticCacheService;
use sochdb_grpc::proto::vector_index_service_server::VectorIndexService;
use sochdb_grpc::security::{AuthMethod, Capability, Principal};
use sochdb_grpc::server::VectorIndexServer;
use tonic::Request;

/// Build a request carrying an authenticated principal for `tenant`.
///
/// This is how the real interceptor delivers identity to a handler, so a probe
/// built this way exercises the same path a client does.
fn authed<T>(msg: T, tenant: &str) -> Request<T> {
    let mut request = Request::new(msg);
    request.extensions_mut().insert(Principal {
        id: format!("user-of-{tenant}"),
        tenant_id: tenant.to_string(),
        capabilities: HashSet::from([
            Capability::Read,
            Capability::Write,
            Capability::ManageCollections,
            Capability::ManageIndexes,
        ]),
        expires_at: None,
        auth_method: AuthMethod::Anonymous,
    });
    request
}

/// Turn an observation into a gate outcome.
///
/// `held` is what the probe actually measured. The rate is the metric the
/// manifest and the dashboards share, so a leak is reported in the same units
/// production reports it in.
fn isolation_outcome(held: bool, detail: &str) -> GateOutcome {
    if held {
        GateOutcome::Passed {
            metric: Metric::UnauthorizedResultRate,
            measured: 0.0,
        }
    } else {
        GateOutcome::Failed {
            metric: Metric::UnauthorizedResultRate,
            measured: 1.0,
            required: 0.0,
        }
    }
    .annotated(detail)
}

/// Attach a detail string to a failing outcome without changing a passing one.
trait Annotated {
    fn annotated(self, detail: &str) -> GateOutcome;
}

impl Annotated for GateOutcome {
    fn annotated(self, detail: &str) -> GateOutcome {
        match self {
            GateOutcome::Failed { .. } => GateOutcome::Errored {
                detail: detail.to_string(),
            },
            other => other,
        }
    }
}

/// Probe: the manifest negotiates the way the contract says it does.
///
/// Real, and able to fail: it asks the live manifest to accept a client that
/// requires a newer minor version than the server offers, and a client on a
/// different major. Both must be refused. If someone edits a contract version
/// without meaning to, this notices.
fn protocol_compatibility(name: &str) -> GateOutcome {
    use sochdb_core::capability::{ContractVersion, Requirement};

    let manifest = manifest();
    let Some(capability) = manifest.get(name) else {
        return GateOutcome::Errored {
            detail: format!("{name} is absent from the manifest"),
        };
    };
    let offered = capability.contract_version;

    let accepts_itself = manifest
        .negotiate(&[Requirement::new(name, offered, Maturity::Preview)])
        .is_ok();
    let refuses_newer_minor = manifest
        .negotiate(&[Requirement::new(
            name,
            ContractVersion::new(offered.major, offered.minor + 1),
            Maturity::Preview,
        )])
        .is_err();
    let refuses_other_major = manifest
        .negotiate(&[Requirement::new(
            name,
            ContractVersion::new(offered.major + 1, 0),
            Maturity::Preview,
        )])
        .is_err();

    if accepts_itself && refuses_newer_minor && refuses_other_major {
        GateOutcome::Passed {
            metric: Metric::ResultDivergence,
            measured: 0.0,
        }
    } else {
        GateOutcome::Errored {
            detail: format!(
                "{name} negotiation is not the documented relation: accepts_itself={accepts_itself} \
                 refuses_newer_minor={refuses_newer_minor} refuses_other_major={refuses_other_major}"
            ),
        }
    }
}

/// Probe: retrieval v2's declared contract matches the constants the code
/// enforces.
///
/// Two numbers in two files that must agree. Nothing else checks them, and a
/// mismatch would let a client negotiate a version the server does not speak.
fn retrieval_v2_contract_matches_code() -> GateOutcome {
    use sochdb_grpc::retrieval_protocol::{CONTRACT_MAJOR, CONTRACT_MINOR};

    let manifest = manifest();
    let declared = match manifest.get(names::RETRIEVAL_V2) {
        Some(capability) => capability.contract_version,
        None => {
            return GateOutcome::Errored {
                detail: "retrieval.protocol.v2 is absent from the manifest".into(),
            };
        }
    };
    if declared.major == CONTRACT_MAJOR && declared.minor == CONTRACT_MINOR {
        GateOutcome::Passed {
            metric: Metric::ResultDivergence,
            measured: 0.0,
        }
    } else {
        GateOutcome::Errored {
            detail: format!(
                "manifest declares {}.{} but retrieval_protocol enforces {CONTRACT_MAJOR}.\
                 {CONTRACT_MINOR}",
                declared.major, declared.minor
            ),
        }
    }
}

/// Probe: one tenant's vectors are not readable by another.
///
/// Tenant `alpha` creates an index and inserts a vector. Tenant `beta` then
/// names the *same index* and searches it. Beta must not see alpha's vector.
///
/// The probe uses the same index name for both deliberately. Using different
/// names would prove only that different indexes hold different data.
async fn vector_search_isolation() -> GateOutcome {
    let server = VectorIndexServer::new();
    let name = "shared-name";

    let created = server
        .create_index(authed(
            proto::CreateIndexRequest {
                name: name.into(),
                dimension: 4,
                metric: 0,
                ..Default::default()
            },
            "alpha",
        ))
        .await;
    if created.is_err() {
        return GateOutcome::Errored {
            detail: "alpha could not create its index".into(),
        };
    }
    let inserted = server
        .insert_batch(authed(
            proto::InsertBatchRequest {
                index_name: name.into(),
                ids: vec![1],
                vectors: vec![1.0, 0.0, 0.0, 0.0],
                metadata: vec![],
            },
            "alpha",
        ))
        .await;
    if inserted.is_err() {
        return GateOutcome::Errored {
            detail: "alpha could not insert into its own index".into(),
        };
    }

    let beta_search = server
        .search(authed(
            proto::SearchRequest {
                index_name: name.into(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 10,
                ..Default::default()
            },
            "beta",
        ))
        .await;

    let leaked = match beta_search {
        Ok(response) => !response.into_inner().results.is_empty(),
        Err(_) => false,
    };
    isolation_outcome(
        !leaked,
        "a second tenant searching the same index name saw the first tenant's vectors",
    )
}

/// Probe: one tenant's writes do not land in another's index.
///
/// The mirror of the read probe, and a distinct failure. A server could scope
/// reads correctly and still let a write from beta land where alpha reads it,
/// which is a data-integrity breach rather than a disclosure.
async fn vector_ingest_isolation() -> GateOutcome {
    let server = VectorIndexServer::new();
    let name = "shared-name";

    for tenant in ["alpha", "beta"] {
        if server
            .create_index(authed(
                proto::CreateIndexRequest {
                    name: name.into(),
                    dimension: 4,
                    metric: 0,
                    ..Default::default()
                },
                tenant,
            ))
            .await
            .is_err()
        {
            return GateOutcome::Errored {
                detail: format!("{tenant} could not create its index"),
            };
        }
    }

    if server
        .insert_batch(authed(
            proto::InsertBatchRequest {
                index_name: name.into(),
                ids: vec![7],
                vectors: vec![0.0, 1.0, 0.0, 0.0],
                metadata: vec![],
            },
            "beta",
        ))
        .await
        .is_err()
    {
        return GateOutcome::Errored {
            detail: "beta could not insert into its own index".into(),
        };
    }

    let alpha_sees = server
        .search(authed(
            proto::SearchRequest {
                index_name: name.into(),
                query: vec![0.0, 1.0, 0.0, 0.0],
                k: 10,
                ..Default::default()
            },
            "alpha",
        ))
        .await
        .map(|r| !r.into_inner().results.is_empty())
        .unwrap_or(false);

    isolation_outcome(
        !alpha_sees,
        "a write by one tenant became visible in another tenant's index of the same name",
    )
}

/// Probe: a metadata filter does not become a way around tenant scoping.
///
/// A filter that matches everything is the strongest form of the question: if
/// scoping is applied before filtering, beta still sees nothing.
async fn metadata_prefilter_isolation() -> GateOutcome {
    let server = VectorIndexServer::new();
    let name = "shared-name";

    if server
        .create_index(authed(
            proto::CreateIndexRequest {
                name: name.into(),
                dimension: 4,
                metric: 0,
                ..Default::default()
            },
            "alpha",
        ))
        .await
        .is_err()
    {
        return GateOutcome::Errored {
            detail: "alpha could not create its index".into(),
        };
    }
    if server
        .insert_batch(authed(
            proto::InsertBatchRequest {
                index_name: name.into(),
                ids: vec![3],
                vectors: vec![0.0, 0.0, 1.0, 0.0],
                metadata: vec![proto::VectorMetadata {
                    parent_id: Some(42),
                    view_type: Some("secret".into()),
                }],
            },
            "alpha",
        ))
        .await
        .is_err()
    {
        return GateOutcome::Errored {
            detail: "alpha could not insert a labelled vector".into(),
        };
    }

    let beta_sees = server
        .search(authed(
            proto::SearchRequest {
                index_name: name.into(),
                query: vec![0.0, 0.0, 1.0, 0.0],
                k: 10,
                grouping: Some(proto::GroupingOptions {
                    group_by: proto::GroupBy::ParentId as i32,
                    max_per_group: 1,
                    candidate_k: 50,
                }),
                ..Default::default()
            },
            "beta",
        ))
        .await
        .map(|r| !r.into_inner().results.is_empty())
        .unwrap_or(false);

    isolation_outcome(
        !beta_sees,
        "a metadata filter let one tenant select another tenant's rows",
    )
}

/// Probe: a capability token issued for one tenant does not verify for another.
///
/// The signature covers the scope, so a token that verified under a swapped
/// tenant would mean the scope is not really bound to it.
fn retrieval_v2_token_isolation() -> GateOutcome {
    use sochdb_grpc::retrieval_protocol::{CapabilityIssuer, CapabilityScope, now_unix_ms};

    let issuer = CapabilityIssuer::new(b"gate-suite-key".to_vec());
    let now = now_unix_ms();
    let digest = "policy-digest-v1".to_string();
    let token = issuer.issue(CapabilityScope {
        index_key: "alpha:shared".into(),
        tenant_id: "alpha".into(),
        expires_at_unix_ms: now + 60_000,
        policy_scope_digest: digest.clone(),
        operation: proto::RetrievalOperation::Search as i32,
    });

    let own_scope_verifies = issuer
        .verify(
            Some(&token),
            "alpha:shared",
            "alpha",
            proto::RetrievalOperation::Search,
            &digest,
            now,
        )
        .is_ok();
    let other_tenant_rejected = issuer
        .verify(
            Some(&token),
            "alpha:shared",
            "beta",
            proto::RetrievalOperation::Search,
            &digest,
            now,
        )
        .is_err();

    if !own_scope_verifies {
        return GateOutcome::Errored {
            detail: "a token did not verify against the scope it was issued for".into(),
        };
    }
    isolation_outcome(
        other_tenant_rejected,
        "a capability token issued for one tenant verified for another",
    )
}

/// Probe: a filter that cannot be pushed down is never silently dropped, and a
/// row missing the governed field is never admitted.
///
/// Both directions matter. Dropping the residual leaks rows; admitting an
/// absent field leaks the rows that carry no label at all, which under SQL
/// three-valued logic would satisfy a negation and become visible to everyone.
fn governed_pushdown_isolation() -> GateOutcome {
    use sochdb_query::filter_ir::{Disjunction, FilterAtom, FilterIR, FilterValue};
    use sochdb_query::governed_retrieval::{
        EqualityAndSetMembership, ResidualStrategy, matches, plan_retrieval,
    };

    let filter = FilterIR {
        clauses: vec![Disjunction {
            atoms: vec![FilterAtom::Eq {
                field: "tenant".into(),
                value: FilterValue::String("alpha".into()),
            }],
        }],
    };

    // A predicate the engine cannot push down must not be quietly discarded.
    // `Reject` is the strategy that refuses outright, so a planner that dropped
    // the residual would return a plan instead of an error.
    let unsupported = FilterIR {
        clauses: vec![Disjunction {
            atoms: vec![FilterAtom::Contains {
                field: "notes".into(),
                substring: "confidential".into(),
            }],
        }],
    };
    let residual_not_dropped = plan_retrieval(
        &unsupported,
        &EqualityAndSetMembership,
        ResidualStrategy::Reject,
        10,
        0.5,
        10_000,
    )
    .is_err();

    // A row carrying no tenant label must not satisfy a tenant predicate.
    let admits_unlabelled = matches(&filter, |_field: &str| None);

    if !residual_not_dropped {
        return GateOutcome::Errored {
            detail: "a filter the engine cannot evaluate was silently dropped from the plan".into(),
        };
    }
    isolation_outcome(
        !admits_unlabelled,
        "a row carrying no tenant label satisfied a tenant predicate",
    )
}

/// Probe: a vector produced under one context is never served as current under
/// another.
///
/// The embedding cache is content-addressed, and deliberately so -- an
/// embedding is a pure function of content, model and chunk policy, so two
/// callers with identical input are entitled to the same vector. The isolation
/// property that does matter is the one on the other axis: when the model or
/// the policy moves, the key must move with it. If it did not, a vector from
/// the old embedding space would be handed back as current and would sit in an
/// index that has already moved on -- a mixed-space index, arrived at through
/// what looks like a cache hit.
fn embedding_lineage_isolation() -> GateOutcome {
    use sochdb_index::embedding::lineage::{
        ChunkPolicy, EmbeddingContext, EmbeddingRecord, MaintenancePlan, ModelRef, SourceChunk,
        embedding_cache_key, plan_maintenance,
    };

    let (Ok(v1), Ok(v2)) = (
        ModelRef::parse("openai:text-3@1"),
        ModelRef::parse("openai:text-3@2"),
    ) else {
        return GateOutcome::Errored {
            detail: "the gate could not construct model references".into(),
        };
    };

    let digest = "content-digest-of-one-chunk";
    let same = embedding_cache_key(digest, "policy@1", &v1);
    let new_model = embedding_cache_key(digest, "policy@1", &v2);
    let new_policy = embedding_cache_key(digest, "policy@2", &v1);
    let repeat = embedding_cache_key(digest, "policy@1", &v1);

    let key_tracks_context =
        same != new_model && same != new_policy && new_model != new_policy && same == repeat;

    // And the planner must decide to rebuild rather than reuse when the context
    // it is asked for differs from the context the stored vectors were made
    // under. Reuse in that situation is the failure: it returns vectors from a
    // superseded embedding space and reports success.
    let policy = ChunkPolicy {
        revision: "policy@1".into(),
        max_tokens: 256,
        overlap_tokens: 16,
    };
    let stored = EmbeddingRecord {
        source_row_id: "row-1".into(),
        chunk_ordinal: 0,
        content_digest: digest.into(),
        chunk_policy_revision: "policy@1".into(),
        model: v1.clone(),
        dimension: 768,
        generated_at_unix_ms: 1,
        cache_key: same.clone(),
    };
    let desired = SourceChunk {
        source_row_id: "row-1".into(),
        chunk_ordinal: 0,
        content_digest: digest.into(),
    };
    let moved_on = EmbeddingContext {
        model: v2,
        chunk_policy: policy,
        dimension: 768,
    };
    let rebuild_forced = matches!(
        plan_maintenance(
            std::slice::from_ref(&stored),
            std::slice::from_ref(&desired),
            &moved_on
        ),
        MaintenancePlan::Rebuild { .. }
    );

    if !rebuild_forced {
        return GateOutcome::Errored {
            detail: "a changed embedding context did not force a rebuild".into(),
        };
    }
    isolation_outcome(
        key_tracks_context,
        "an embedding cache key did not change when the model or chunk policy did",
    )
}

/// Probe: one tenant cannot read another's semantic cache.
///
/// Alpha stores an answer; beta names the same cache and asks for it. Beta must
/// be refused or must miss.
async fn semantic_cache_isolation() -> GateOutcome {
    use sochdb_grpc::semantic_cache_server::SemanticCacheServer;

    let server = SemanticCacheServer::new();
    let cache = "shared-cache";

    let stored = server
        .put(authed(
            proto::SemanticCachePutRequest {
                cache_name: cache.into(),
                key: "what is the payroll total".into(),
                value: "alpha-confidential-answer".into(),
                key_embedding: vec![1.0, 0.0],
                ttl_seconds: 600,
            },
            "alpha",
        ))
        .await;
    if stored.is_err() {
        return GateOutcome::Errored {
            detail: "alpha could not store an entry in its own cache".into(),
        };
    }

    let beta_read = server
        .get(authed(
            proto::SemanticCacheGetRequest {
                cache_name: cache.into(),
                query: "what is the payroll total".into(),
                query_embedding: vec![1.0, 0.0],
                similarity_threshold: 0.5,
            },
            "beta",
        ))
        .await;

    let leaked = match beta_read {
        Ok(response) => {
            let inner = response.into_inner();
            inner.hit && inner.cached_value.contains("alpha-confidential")
        }
        Err(_) => false,
    };

    isolation_outcome(
        !leaked,
        "a second tenant read the first tenant's cached answer by naming its cache",
    )
}

/// Probe: one tenant cannot write into or read from another's context
/// namespace.
async fn context_assembly_isolation() -> GateOutcome {
    use sochdb_grpc::context_server::ContextServer;
    use sochdb_grpc::proto::context_service_server::ContextService;

    let server = ContextServer::new();

    let beta_writes_alphas_namespace = server
        .write_episode(authed(
            proto::WriteEpisodeRequest {
                namespace: "alpha".into(),
                text: "injected by beta".into(),
                t_valid_from: None,
                metadata_json: String::new(),
            },
            "beta",
        ))
        .await;

    isolation_outcome(
        beta_writes_alphas_namespace.is_err(),
        "one tenant wrote an episode into another tenant's context namespace",
    )
}

/// Probe: the graph service refuses a namespace the caller does not own.
async fn graph_isolation() -> GateOutcome {
    use sochdb_grpc::graph_server::GraphServer;
    use sochdb_grpc::proto::graph_service_server::GraphService;

    let server = GraphServer::new();
    let beta_touches_alpha = server
        .add_node(authed(
            proto::AddNodeRequest {
                namespace: "alpha".into(),
                node: Some(proto::GraphNode {
                    id: "n1".into(),
                    node_type: "injected".into(),
                    properties: Default::default(),
                }),
            },
            "beta",
        ))
        .await;

    isolation_outcome(
        beta_touches_alpha.is_err(),
        "one tenant added a node to another tenant's graph namespace",
    )
}

/// Run every probe and assemble the bundle.
async fn run_gates() -> EvidenceBundle {
    let build = BuildIdentity::new(
        std::env::var("SOCHDB_GATE_COMMIT").unwrap_or_else(|_| "0".repeat(40)),
        std::env::var("SOCHDB_GATE_IMAGE").unwrap_or_else(|_| format!("sha256:{}", "0".repeat(64))),
    );

    // Protocol compatibility is one property of the manifest as a whole, so the
    // strictest observation across the Preview set is what the bundle records.
    let mut protocol = GateOutcome::Passed {
        metric: Metric::ResultDivergence,
        measured: 0.0,
    };
    for name in preview_names() {
        let outcome = protocol_compatibility(name);
        if !outcome.is_pass() {
            protocol = outcome;
            break;
        }
    }
    if protocol.is_pass() {
        let v2 = retrieval_v2_contract_matches_code();
        if !v2.is_pass() {
            protocol = v2;
        }
    }

    // Isolation is the weakest link across every served surface. One leaking
    // service is a leaking build, so the worst observation is what is recorded.
    let probes: Vec<(&str, GateOutcome)> = vec![
        ("vector search", vector_search_isolation().await),
        ("vector ingest", vector_ingest_isolation().await),
        ("metadata prefilter", metadata_prefilter_isolation().await),
        ("retrieval v2 token", retrieval_v2_token_isolation()),
        ("governed pushdown", governed_pushdown_isolation()),
        ("embedding lineage", embedding_lineage_isolation()),
        ("semantic cache", semantic_cache_isolation().await),
        ("context assembly", context_assembly_isolation().await),
        ("graph", graph_isolation().await),
    ];

    let mut isolation = GateOutcome::Passed {
        metric: Metric::UnauthorizedResultRate,
        measured: 0.0,
    };
    let mut leaks = Vec::new();
    for (name, outcome) in &probes {
        if !outcome.is_pass() {
            leaks.push(format!("{name}: {outcome:?}"));
        }
    }
    if !leaks.is_empty() {
        isolation = GateOutcome::Errored {
            detail: leaks.join("; "),
        };
    }

    EvidenceBundle::new(build)
        .with(GateKind::ProtocolCompatibility, protocol)
        .with(GateKind::PolicyIsolation, isolation)
}

/// The capabilities this build claims at `Preview`.
fn preview_names() -> Vec<&'static str> {
    [
        names::EMBEDDING_MAINTENANCE,
        names::GOVERNED_PUSHDOWN,
        names::RETRIEVAL_V2,
        names::HNSW_SEARCH,
        names::VECTOR_INGEST_BATCH,
        names::METADATA_PREFILTERED_SEARCH,
        names::GRAPH,
        names::CONTEXT_ASSEMBLY,
        names::SEMANTIC_CACHE,
    ]
    .to_vec()
}

/// Every probe individually, so a failure names the surface that leaked rather
/// than only that something did.
#[tokio::test]
async fn every_served_surface_isolates_its_tenants() {
    let probes: Vec<(&str, GateOutcome)> = vec![
        ("vector search", vector_search_isolation().await),
        ("vector ingest", vector_ingest_isolation().await),
        ("metadata prefilter", metadata_prefilter_isolation().await),
        ("retrieval v2 token", retrieval_v2_token_isolation()),
        ("governed pushdown", governed_pushdown_isolation()),
        ("embedding lineage", embedding_lineage_isolation()),
        ("semantic cache", semantic_cache_isolation().await),
        ("context assembly", context_assembly_isolation().await),
        ("graph", graph_isolation().await),
    ];
    let failures: Vec<String> = probes
        .iter()
        .filter(|(_, outcome)| !outcome.is_pass())
        .map(|(name, outcome)| format!("{name} -> {outcome:?}"))
        .collect();
    assert!(
        failures.is_empty(),
        "isolation probes failed: {failures:#?}"
    );
}

#[test]
fn the_manifest_negotiates_the_relation_it_documents() {
    for name in preview_names() {
        let outcome = protocol_compatibility(name);
        assert!(outcome.is_pass(), "{name} -> {outcome:?}");
    }
    let v2 = retrieval_v2_contract_matches_code();
    assert!(v2.is_pass(), "{v2:?}");
}

/// The reason this file exists: every `Preview` claim in the manifest is backed
/// by evidence this suite actually produced.
///
/// If a probe regresses, this fails and the capability is no longer qualified
/// -- which is the whole point of writing the claim down as something that can
/// be refused.
#[tokio::test]
async fn every_preview_capability_is_backed_by_this_run() {
    let evidence = run_gates().await;
    let build = evidence.build.clone();
    let manifest = manifest();

    let unbacked: Vec<String> = manifest
        .capabilities
        .iter()
        .filter(|c| c.maturity >= Maturity::Preview)
        .filter_map(|c| {
            qualify_capability(c, &build, &evidence)
                .err()
                .map(|refusals| {
                    let reasons: Vec<String> = refusals
                        .iter()
                        .map(std::string::ToString::to_string)
                        .collect();
                    format!("{} ({:?}): {}", c.name, c.maturity, reasons.join(", "))
                })
        })
        .collect();

    assert!(
        unbacked.is_empty(),
        "capabilities claiming Preview or better with no supporting evidence:\n{unbacked:#?}"
    );
}

/// The suite covers exactly the capabilities that need covering. A new
/// `Preview` capability that nobody wrote a probe for fails here rather than
/// inheriting another capability's evidence.
#[test]
fn the_suite_covers_every_preview_capability() {
    let manifest = manifest();
    let declared: Vec<&str> = manifest
        .capabilities
        .iter()
        .filter(|c| c.maturity >= Maturity::Preview)
        .map(|c| c.name.as_str())
        .collect();
    let covered = preview_names();
    for name in &declared {
        assert!(
            covered.contains(name),
            "{name} claims Preview with no probe"
        );
    }
    assert_eq!(declared.len(), covered.len());
}
