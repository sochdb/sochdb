//! Integration test: ContextServer SEARCH section uses sochdb-memory + ContextCompiler.

use sochdb_grpc::context_server::ContextServer;
use sochdb_grpc::proto::{
    ContextQueryRequest, ContextSection, ContextSectionType, OutputFormat, WriteEpisodeRequest,
    context_service_server::ContextService,
};
use sochdb_grpc::security::{AuthMethod, Capability, Principal};
use std::collections::HashSet;
use tonic::Request;

/// A request carrying the principal the interceptor would have attached.
///
/// The context service authorises every namespace it is asked to touch, so a
/// bare `Request::new` is now correctly refused: it arrives as `anonymous`,
/// whose tenant is `default`, and `default` does not own `sess-42`. These tests
/// exercise the service's own behaviour rather than the refusal, so they
/// present the principal that legitimately owns the namespace under test.
fn authed<T>(msg: T, tenant: &str) -> Request<T> {
    let mut request = Request::new(msg);
    request.extensions_mut().insert(Principal {
        id: format!("user-of-{tenant}"),
        tenant_id: tenant.to_string(),
        capabilities: HashSet::from([Capability::Read, Capability::Write]),
        expires_at: None,
        auth_method: AuthMethod::Anonymous,
    });
    request
}

#[tokio::test]
async fn context_search_uses_memory_backend() {
    let server = ContextServer::new();
    let store = server.memory_store();

    store
        .write_episode(sochdb_memory::EpisodeWrite {
            namespace: "sess-42".into(),
            text: "Melanie ran a charity race the Sunday before 25 May 2023.".into(),
            t_valid_from: None,
            metadata: None,
        })
        .unwrap();

    let req = ContextQueryRequest {
        session_id: "sess-42".into(),
        token_limit: 1024,
        sections: vec![ContextSection {
            name: "memory".into(),
            priority: 0,
            section_type: ContextSectionType::ContextSectionSearch as i32,
            query: "charity race".into(),
            options: Default::default(),
        }],
        format: OutputFormat::Markdown as i32,
        include_schema: false,
    };

    let resp = server
        .query(authed(req, "sess-42"))
        .await
        .unwrap()
        .into_inner();

    assert!(resp.error.is_empty());
    assert!(resp.total_tokens > 0);
    assert!(
        resp.context.to_lowercase().contains("charity")
            || resp.context.to_lowercase().contains("race")
    );
}

#[tokio::test]
async fn write_episode_rpc_then_search() {
    let server = ContextServer::with_memory_store_and_lifecycle(
        std::sync::Arc::new(sochdb_memory::MemoryStore::with_defaults()),
        false,
    );

    let write_resp = server
        .write_episode(authed(
            WriteEpisodeRequest {
                namespace: "agent-99".into(),
                text: "Alice adopted a rescue dog named Biscuit in March 2024.".into(),
                t_valid_from: None,
                metadata_json: String::new(),
            },
            "agent-99",
        ))
        .await
        .unwrap()
        .into_inner();

    assert!(write_resp.error.is_empty());
    assert!(write_resp.lexical_indexed);
    assert!(write_resp.episode_id > 0);

    let search_resp = server
        .query(authed(
            ContextQueryRequest {
                session_id: "agent-99".into(),
                token_limit: 512,
                sections: vec![ContextSection {
                    name: "recall".into(),
                    priority: 0,
                    section_type: ContextSectionType::ContextSectionSearch as i32,
                    query: "rescue dog Biscuit".into(),
                    options: Default::default(),
                }],
                format: OutputFormat::Markdown as i32,
                include_schema: false,
            },
            "agent-99",
        ))
        .await
        .unwrap()
        .into_inner();

    assert!(search_resp.error.is_empty());
    assert!(search_resp.context.to_lowercase().contains("biscuit"));
}

/// Every output format the service offers, so a regression in one cannot hide
/// behind another passing.
const ALL_FORMATS: [(OutputFormat, &str); 4] = [
    (OutputFormat::Toon, "toon"),
    (OutputFormat::Json, "json"),
    (OutputFormat::Markdown, "markdown"),
    (OutputFormat::Text, "text"),
];

/// A principal allowed to reach any namespace, needed to put two sections of
/// one request into two different namespaces.
fn admin<T>(msg: T) -> Request<T> {
    let mut request = Request::new(msg);
    request.extensions_mut().insert(Principal {
        id: "admin".into(),
        tenant_id: "admin".into(),
        capabilities: HashSet::from([Capability::Read, Capability::Write, Capability::Admin]),
        expires_at: None,
        auth_method: AuthMethod::Anonymous,
    });
    request
}

/// Two searches over two namespaces, each recalling a word the other cannot.
fn two_section_request(token_limit: u32, format: OutputFormat) -> ContextQueryRequest {
    ContextQueryRequest {
        session_id: "multi-sess".into(),
        token_limit,
        sections: vec![
            ContextSection {
                name: "first".into(),
                priority: 0,
                section_type: ContextSectionType::ContextSectionSearch as i32,
                query: "sourdough".into(),
                options: Default::default(),
            },
            ContextSection {
                name: "second".into(),
                priority: 1,
                section_type: ContextSectionType::ContextSectionSearch as i32,
                query: "tunnelling".into(),
                options: std::collections::HashMap::from([(
                    "namespace".to_string(),
                    "other-ns".to_string(),
                )]),
            },
        ],
        format: format as i32,
        include_schema: false,
    }
}

fn server_with_two_namespaces() -> ContextServer {
    let server = ContextServer::new();
    let store = server.memory_store();
    store
        .write_episode(sochdb_memory::EpisodeWrite {
            namespace: "multi-sess".into(),
            text: "Ravi baked sourdough on Tuesday.".into(),
            t_valid_from: None,
            metadata: None,
        })
        .unwrap();
    store
        .write_episode(sochdb_memory::EpisodeWrite {
            namespace: "other-ns".into(),
            text: "Quantum tunnelling explains the junction current.".into(),
            t_valid_from: None,
            metadata: None,
        })
        .unwrap();
    server
}

/// The multi-section path used to re-render the joined sections through the
/// formatter as a `CompiledContext` with no facts. Formats that render from
/// facts -- including Toon, the default -- therefore returned an envelope
/// around nothing, discarding every section. Only single-section requests were
/// covered, so the suite stayed green while the default format returned no
/// content at all.
#[tokio::test]
async fn multi_section_content_survives_assembly_in_every_format() {
    for (format, label) in ALL_FORMATS {
        let server = server_with_two_namespaces();
        let resp = server
            .query(admin(two_section_request(2048, format)))
            .await
            .unwrap()
            .into_inner();

        assert!(resp.error.is_empty(), "{label}: {}", resp.error);
        let lower = resp.context.to_lowercase();
        assert!(
            lower.contains("sourdough"),
            "{label}: first section missing from {:?}",
            resp.context
        );
        assert!(
            lower.contains("tunnelling"),
            "{label}: second section missing from {:?}",
            resp.context
        );
    }
}

/// The per-section budget is handed out in body tokens but the response is
/// emitted in wrapped tokens, so charging one against the other let the
/// assembled context run past the caller's limit.
///
/// The assertion measures `resp.context` itself rather than trusting
/// `resp.total_tokens`: the reported figure used to be a sum of the parts, and
/// that is precisely how an overrunning context passed for a compliant one.
#[tokio::test]
async fn assembled_context_never_exceeds_the_requested_token_limit() {
    for (format, label) in ALL_FORMATS {
        for token_limit in [1u32, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let server = server_with_two_namespaces();
            let resp = server
                .query(admin(two_section_request(token_limit, format)))
                .await
                .unwrap()
                .into_inner();

            let measured =
                sochdb_grpc::memory_backend::MemoryBackend::estimate_tokens_exact(&resp.context);
            // A format with a mandatory envelope cannot go below it, so that
            // floor -- not zero -- is the smallest honest bound.
            let floor = match format {
                OutputFormat::Json => 2,
                _ => 0,
            };
            assert!(
                measured <= token_limit.max(floor),
                "{label} at limit {token_limit}: context is {measured} tokens: {:?}",
                resp.context
            );
            for section in &resp.section_results {
                assert!(
                    section.tokens_used <= token_limit,
                    "{label} at limit {token_limit}: section {} used {}",
                    section.name,
                    section.tokens_used
                );
            }
        }
    }
}

/// A client parses the response without knowing how much survived the budget,
/// so the shape must not depend on that. JSON used to come back as a bare
/// object for one section, an array for two, and an empty string for none.
#[tokio::test]
async fn json_output_is_always_an_array_whatever_the_budget_allows() {
    for token_limit in [1u32, 16, 128, 2048] {
        let server = server_with_two_namespaces();
        let resp = server
            .query(admin(two_section_request(token_limit, OutputFormat::Json)))
            .await
            .unwrap()
            .into_inner();

        assert!(
            resp.context.starts_with('[') && resp.context.ends_with(']'),
            "limit {token_limit}: not a JSON array: {:?}",
            resp.context
        );
    }
}

/// Charging a section more than it was allocated starves whatever follows it:
/// the loop sees no budget left and breaks, silently dropping later sections.
#[tokio::test]
async fn a_generous_budget_starves_no_section() {
    for (format, label) in ALL_FORMATS {
        let server = server_with_two_namespaces();
        let resp = server
            .query(admin(two_section_request(4096, format)))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(
            resp.section_results.len(),
            2,
            "{label}: a section was dropped"
        );
        for section in &resp.section_results {
            assert!(
                !section.content.is_empty(),
                "{label}: section {} came back empty",
                section.name
            );
        }
    }
}

/// `total_tokens` is what the caller uses to plan the rest of its prompt, so it
/// has to describe the string actually returned rather than the sum of the
/// parts that went into it.
#[tokio::test]
async fn reported_total_tokens_describes_the_returned_context() {
    for (format, label) in ALL_FORMATS {
        let server = server_with_two_namespaces();
        let resp = server
            .query(admin(two_section_request(2048, format)))
            .await
            .unwrap()
            .into_inner();

        let measured =
            sochdb_grpc::memory_backend::MemoryBackend::estimate_tokens_exact(&resp.context);
        assert_eq!(
            resp.total_tokens, measured,
            "{label}: reported {} for a context of {measured}",
            resp.total_tokens
        );
    }
}
