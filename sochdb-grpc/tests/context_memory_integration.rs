//! Integration test: ContextServer SEARCH section uses sochdb-memory + ContextCompiler.

use sochdb_grpc::context_server::ContextServer;
use sochdb_grpc::proto::{
    ContextQueryRequest, ContextSection, ContextSectionType, OutputFormat, WriteEpisodeRequest,
    context_service_server::ContextService,
};
use tonic::Request;

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

    let resp = server.query(Request::new(req)).await.unwrap().into_inner();

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
        .write_episode(Request::new(WriteEpisodeRequest {
            namespace: "agent-99".into(),
            text: "Alice adopted a rescue dog named Biscuit in March 2024.".into(),
            t_valid_from: None,
            metadata_json: String::new(),
        }))
        .await
        .unwrap()
        .into_inner();

    assert!(write_resp.error.is_empty());
    assert!(write_resp.lexical_indexed);
    assert!(write_resp.episode_id > 0);

    let search_resp = server
        .query(Request::new(ContextQueryRequest {
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
        }))
        .await
        .unwrap()
        .into_inner();

    assert!(search_resp.error.is_empty());
    assert!(search_resp.context.to_lowercase().contains("biscuit"));
}
