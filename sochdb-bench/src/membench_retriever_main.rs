//! MemoryAgentBench-compatible retriever using `sochdb-memory` lexical lanes.
//!
//! Reads clawdesk-style JSON from stdin or a file path argument, returns ranked
//! chunk texts per query on stdout.

use serde::{Deserialize, Serialize};
use sochdb_memory::{EpisodeWrite, MemoryQuery, MemoryStore, QueryLanes};
use std::io::{self, Read};
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Payload {
    top_k: usize,
    #[serde(default = "default_k1")]
    bm25_k1: f32,
    #[serde(default = "default_b")]
    bm25_b: f32,
    contexts: Vec<ContextPayload>,
}

fn default_k1() -> f32 {
    1.2
}
fn default_b() -> f32 {
    0.75
}

#[derive(Debug, Deserialize)]
struct ContextPayload {
    context_id: u32,
    chunks: Vec<String>,
    queries: Vec<QueryPayload>,
}

#[derive(Debug, Deserialize)]
struct QueryPayload {
    query_id: u32,
    query: String,
}

#[derive(Debug, Serialize)]
struct RetrieverOutput {
    retriever: &'static str,
    bm25_k1: f32,
    bm25_b: f32,
    results: Vec<QueryResult>,
}

#[derive(Debug, Serialize)]
struct QueryResult {
    context_id: u32,
    query_id: u32,
    retrieved_ids: Vec<u64>,
    retrieved_texts: Vec<String>,
    build_ms: f64,
    query_ms: f64,
}

fn main() -> Result<(), String> {
    let input = if let Some(path) = std::env::args().nth(1) {
        std::fs::read_to_string(path).map_err(|e| e.to_string())?
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).map_err(|e| e.to_string())?;
        buf
    };

    let payload: Payload = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    let top_k = payload.top_k.max(1);
    let mut results = Vec::new();

    for ctx in payload.contexts {
        let build_start = Instant::now();
        let store = MemoryStore::with_defaults();
        let ns = format!("ctx-{}", ctx.context_id);

        for (idx, chunk) in ctx.chunks.iter().enumerate() {
            store
                .write_episode(EpisodeWrite {
                    namespace: ns.clone(),
                    text: chunk.clone(),
                    t_valid_from: None,
                    metadata: None,
                })
                .map_err(|e| e.to_string())?;
            // doc_id is 1-based sequential per namespace
            let _ = idx;
        }

        let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

        for q in ctx.queries {
            let query_start = Instant::now();
            let mq = MemoryQuery {
                namespace: ns.clone(),
                query: q.query,
                as_of: None,
                lanes: QueryLanes::lexical_only(),
                k: top_k,
            };
            let hits = store.query(&mq);
            let retrieved_ids: Vec<u64> = hits.hits.iter().map(|h| h.doc_id).collect();
            let retrieved_texts: Vec<String> = retrieved_ids
                .iter()
                .filter_map(|id| store.episode_text(&ns, *id))
                .collect();
            let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;

            results.push(QueryResult {
                context_id: ctx.context_id,
                query_id: q.query_id,
                retrieved_ids,
                retrieved_texts,
                build_ms,
                query_ms,
            });
        }
    }

    let out = RetrieverOutput {
        retriever: "sochdb-memory-lexical",
        bm25_k1: payload.bm25_k1,
        bm25_b: payload.bm25_b,
        results,
    };

    println!("{}", serde_json::to_string(&out).map_err(|e| e.to_string())?);
    Ok(())
}