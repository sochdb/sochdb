// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// you may not use this file except in compliance with the License.

//! Context Service gRPC Implementation
//!
//! Wires LLM context assembly to `sochdb-memory` (write-time lexical recall)
//! and `sochdb-query::ContextCompiler` (exact-BPE budget packing).

use crate::memory_backend::{ContextOutputFormat, MemoryBackend};
use crate::proto::{
    ContextQueryRequest, ContextQueryResponse, ContextSectionType, EstimateTokensRequest,
    EstimateTokensResponse, FormatContextRequest, FormatContextResponse, OutputFormat,
    SectionResult, WriteEpisodeRequest, WriteEpisodeResponse,
    context_service_server::{ContextService, ContextServiceServer},
};
use sochdb_memory::{LifecycleConfig, MemoryLifecycleDaemon, MemoryStore};
use sochdb_query::ContextTemplate;
use std::sync::Arc;
use tonic::{Request, Response, Status};

fn proto_format(fmt: i32) -> ContextOutputFormat {
    match fmt {
        x if x == OutputFormat::Json as i32 => ContextOutputFormat::Json,
        x if x == OutputFormat::Markdown as i32 => ContextOutputFormat::Markdown,
        x if x == OutputFormat::Text as i32 => ContextOutputFormat::Text,
        _ => ContextOutputFormat::Toon,
    }
}

fn compiler_template(fmt: ContextOutputFormat) -> ContextTemplate {
    match fmt {
        ContextOutputFormat::Markdown => ContextTemplate::Markdown,
        ContextOutputFormat::Text => ContextTemplate::Plain,
        _ => ContextTemplate::Toon,
    }
}

use std::collections::HashMap;

fn section_namespace(session_id: &str, options: &HashMap<String, String>) -> String {
    options
        .get("namespace")
        .cloned()
        .unwrap_or_else(|| session_id.to_string())
}

/// Context gRPC Server backed by sochdb-memory + ContextCompiler.
pub struct ContextServer {
    backend: MemoryBackend,
    lifecycle: Option<MemoryLifecycleDaemon>,
}

impl ContextServer {
    pub fn new() -> Self {
        Self::with_memory_store(Arc::new(MemoryStore::with_defaults()))
    }

    pub fn with_memory_store(store: Arc<MemoryStore>) -> Self {
        Self::with_memory_store_and_lifecycle(store, true)
    }

    /// Share a `MemoryStore` across the process; optionally start the lifecycle daemon.
    pub fn with_memory_store_and_lifecycle(store: Arc<MemoryStore>, start_lifecycle: bool) -> Self {
        let lifecycle = if start_lifecycle {
            let config = LifecycleConfig::default();
            let daemon = MemoryLifecycleDaemon::new(Arc::clone(&store), config.clone());
            daemon.start(&config);
            Some(daemon)
        } else {
            None
        };
        Self {
            backend: MemoryBackend::new(store),
            lifecycle,
        }
    }

    pub fn memory_store(&self) -> Arc<MemoryStore> {
        Arc::clone(self.backend.store())
    }

    pub fn into_service(self) -> ContextServiceServer<Self> {
        ContextServiceServer::new(self)
    }
}

impl Drop for ContextServer {
    fn drop(&mut self) {
        if let Some(daemon) = self.lifecycle.take() {
            daemon.stop();
        }
    }
}

impl Default for ContextServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl ContextService for ContextServer {
    async fn query(
        &self,
        request: Request<ContextQueryRequest>,
    ) -> Result<Response<ContextQueryResponse>, Status> {
        let req = request.into_inner();
        let token_limit = req.token_limit.max(1) as usize;
        let output_fmt = proto_format(req.format);
        let template = compiler_template(output_fmt);

        let mut section_results = Vec::new();
        let mut total_tokens = 0u32;
        let mut context_parts = Vec::new();

        let mut sections = req.sections;
        sections.sort_by_key(|s| s.priority);

        for section in sections {
            let remaining_budget = token_limit.saturating_sub(total_tokens as usize);
            if remaining_budget == 0 {
                break;
            }

            let ns = section_namespace(&req.session_id, &section.options);
            let ns = ns.as_str();

            let (final_content, tokens_used, truncated) = match section.section_type {
                x if x == ContextSectionType::ContextSectionSearch as i32 => {
                    let lanes = MemoryBackend::parse_lanes(&section.options);
                    match self.backend.search_and_compile(
                        ns,
                        &section.query,
                        remaining_budget,
                        lanes,
                        template,
                    ) {
                        Ok(compiled) => {
                            let truncated =
                                compiled.truncated || compiled.exact_tokens > remaining_budget;
                            let content = MemoryBackend::format_compiled(&compiled, output_fmt);
                            let tokens = compiled.exact_tokens as u32;
                            (content, tokens, truncated)
                        }
                        Err(e) => (format!("# {} (error)\n{}\n", section.name, e), 0, true),
                    }
                }
                x if x == ContextSectionType::ContextSectionGet as i32 => {
                    if let Some(text) = section.options.get("episode_text") {
                        match self.backend.write_episode(ns, text, None, None) {
                            Ok(wr) => {
                                let content = format!(
                                    "# {} (ingested)\nepisode_id={} lag_us={} lexical={}\n",
                                    section.name,
                                    wr.episode_id.0,
                                    wr.ingestion_lag_us,
                                    wr.lexical_indexed
                                );
                                let tokens = MemoryBackend::estimate_tokens_exact(&content);
                                (content, tokens, false)
                            }
                            Err(e) => (
                                format!("# {} (write error)\n{}\n", section.name, e),
                                0,
                                true,
                            ),
                        }
                    } else if let Some(doc_id) = section
                        .options
                        .get("doc_id")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        let text = self
                            .backend
                            .get_episode_text(ns, doc_id)
                            .unwrap_or_else(|| format!("[episode {doc_id} not found]"));
                        let content = format!("# {}\n{}\n", section.name, text);
                        let tokens = MemoryBackend::estimate_tokens_exact(&content);
                        (content, tokens, false)
                    } else {
                        let content = format!("# {}\n[path: {}]\n", section.name, section.query);
                        let tokens = MemoryBackend::estimate_tokens_exact(&content);
                        (content, tokens, false)
                    }
                }
                x if x == ContextSectionType::ContextSectionLast as i32 => {
                    let count = self.backend.store().episode_count(ns);
                    let content = format!(
                        "# {} (Recent)\nnamespace={} episodes={}\n",
                        section.name, ns, count
                    );
                    let tokens = MemoryBackend::estimate_tokens_exact(&content);
                    (content, tokens, false)
                }
                x if x == ContextSectionType::ContextSectionSelect as i32 => {
                    let content = format!("# {} (Query)\n[SQL: {}]\n", section.name, section.query);
                    let tokens = MemoryBackend::estimate_tokens_exact(&content);
                    (content, tokens, false)
                }
                _ => {
                    let content = format!("# {}\n", section.name);
                    let tokens = MemoryBackend::estimate_tokens_exact(&content);
                    (content, tokens, false)
                }
            };

            total_tokens += tokens_used;
            context_parts.push(final_content.clone());
            section_results.push(SectionResult {
                name: section.name,
                tokens_used,
                truncated,
                content: final_content,
            });
        }

        let context = if context_parts.len() == 1 {
            context_parts.into_iter().next().unwrap_or_default()
        } else {
            MemoryBackend::format_compiled(
                &sochdb_query::CompiledContext {
                    body: context_parts.join("\n---\n"),
                    exact_tokens: total_tokens as usize,
                    budget: token_limit,
                    facts: vec![],
                    truncated: total_tokens as usize >= token_limit,
                },
                output_fmt,
            )
        };

        Ok(Response::new(ContextQueryResponse {
            context,
            total_tokens,
            section_results,
            error: String::new(),
        }))
    }

    async fn write_episode(
        &self,
        request: Request<WriteEpisodeRequest>,
    ) -> Result<Response<WriteEpisodeResponse>, Status> {
        let req = request.into_inner();
        if req.text.is_empty() {
            return Ok(Response::new(WriteEpisodeResponse {
                error: "text must not be empty".into(),
                ..Default::default()
            }));
        }

        let metadata = if req.metadata_json.is_empty() {
            None
        } else {
            match serde_json::from_str(&req.metadata_json) {
                Ok(v) => Some(v),
                Err(e) => {
                    return Ok(Response::new(WriteEpisodeResponse {
                        error: format!("invalid metadata_json: {e}"),
                        ..Default::default()
                    }));
                }
            }
        };

        match self
            .backend
            .write_episode(&req.namespace, &req.text, req.t_valid_from, metadata)
        {
            Ok(wr) => Ok(Response::new(WriteEpisodeResponse {
                episode_id: wr.episode_id.0,
                t_created: wr.t_created,
                lexical_indexed: wr.lexical_indexed,
                ingestion_lag_us: wr.ingestion_lag_us,
                enrichment_queued: wr.enrichment_queued,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(WriteEpisodeResponse {
                error: e,
                ..Default::default()
            })),
        }
    }

    async fn estimate_tokens(
        &self,
        request: Request<EstimateTokensRequest>,
    ) -> Result<Response<EstimateTokensResponse>, Status> {
        let req = request.into_inner();
        let token_count = MemoryBackend::estimate_tokens_exact(&req.content);

        Ok(Response::new(EstimateTokensResponse { token_count }))
    }

    async fn format_context(
        &self,
        request: Request<FormatContextRequest>,
    ) -> Result<Response<FormatContextResponse>, Status> {
        let req = request.into_inner();
        let fmt = proto_format(req.format);

        let formatted = match fmt {
            ContextOutputFormat::Json => serde_json::json!({ "content": req.content }).to_string(),
            ContextOutputFormat::Markdown => {
                format!("```\n{}\n```", req.content)
            }
            ContextOutputFormat::Text => req.content,
            ContextOutputFormat::Toon => format!("<toon>{}</toon>", req.content),
        };

        Ok(Response::new(FormatContextResponse { formatted }))
    }
}
