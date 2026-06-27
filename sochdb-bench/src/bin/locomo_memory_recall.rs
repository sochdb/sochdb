//! LoCoMo agent-memory retrieval recall on SochDB's NATIVE three_lane memory.
//!
//! This is the publishable "fast + accurate" number for SochDB's hybrid memory:
//! per-conversation, ingest every dialogue turn as an episode, then for each QA
//! query the three_lane path and score whether the gold-evidence turn(s) land in
//! the top-k. We run the IDENTICAL retrieval code the gRPC ContextService.query
//! uses (MemoryBackend -> MemoryStore::query), so this in-process recall is the
//! number the server would report — without the server build or the network.
//!
//! mock (lexical-only, vector lane gated off) vs fastembed bge-small (semantic
//! hybrid) — the gap is the value of the semantic lane on real agent memory.
//!
//! Build: cargo build --release -p sochdb-bench --bin locomo-memory-recall --features fastembed
//! Run:   cd sochdb-query && FASTEMBED_CACHE_DIR=$PWD/.fastembed_cache \
//!          LOCOMO_JSON=/path/to/locomo10.json ../target/release/locomo-memory-recall

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use sochdb_memory::{EpisodeWrite, Lane, MemoryQuery, MemoryStore, MemoryStoreConfig, QueryLanes};
use sochdb_query::EmbeddingProvider;

// Canonical LoCoMo category map (verified against the dataset counts).
fn cat_name(c: &str) -> &'static str {
    match c {
        "1" => "multi_hop",
        "2" => "temporal",
        "3" => "open_domain",
        "4" => "single_hop",
        "5" => "adversarial",
        _ => "other",
    }
}

/// Embedding provider backed by a precomputed text->vector cache (JSON map).
/// Lets us evaluate a strong HTTP embedder (e.g. OpenAI text-embedding-3-small)
/// without per-turn network calls: embed all LoCoMo texts once (batched, in
/// Python), then look them up by exact text here.
struct CachedEmbedder {
    map: std::collections::HashMap<String, Vec<f32>>,
    dim: usize,
}
impl sochdb_query::EmbeddingProvider for CachedEmbedder {
    fn model_name(&self) -> &str {
        "cached"
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    fn max_length(&self) -> usize {
        8192
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn embed(
        &self,
        text: &str,
    ) -> sochdb_query::embedding_provider::EmbeddingResult<Vec<f32>> {
        self.map.get(text).cloned().ok_or_else(|| {
            sochdb_query::embedding_provider::EmbeddingError::ProviderError(format!(
                "no cached embedding for: {:.50}",
                text
            ))
        })
    }
}

#[derive(Default)]
struct Agg {
    hit: f64,
    recall: f64,
    n: usize,
}
impl Agg {
    fn add(&mut self, hit: f64, recall: f64) {
        self.hit += hit;
        self.recall += recall;
        self.n += 1;
    }
}

fn run(label: &str, embedder: Arc<dyn EmbeddingProvider>, data: &[Value], k: usize) {
    let mut overall = Agg::default();
    let mut by_cat: HashMap<String, Agg> = HashMap::new();
    let mut latency_us = 0u128;
    let mut vector_any = false;

    for conv in data {
        let store = MemoryStore::with_embedder(
            None,
            MemoryStoreConfig {
                enrich_on_write: true,
                ..MemoryStoreConfig::default()
            },
            Arc::clone(&embedder),
        );
        let ns = "loco";
        let mut dia2doc: HashMap<String, u64> = HashMap::new();

        if let Some(obj) = conv["conversation"].as_object() {
            // WINDOW=N groups N consecutive turns per session into one episode
            // (0 = the whole session as one episode). Larger windows give each
            // episode more conversational context, so a question can match the
            // block that CONTAINS its evidence turn even when the turn itself is
            // short/context-stripped — the retrieval-structure lever.
            let window: usize = std::env::var("WINDOW")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            for (key, val) in obj {
                if !(key.starts_with("session_") && !key.contains("date")) {
                    continue;
                }
                let Some(turns) = val.as_array() else { continue };
                // (dia_id, "speaker: text") for each non-empty turn.
                let items: Vec<(String, String)> = turns
                    .iter()
                    .filter_map(|t| {
                        let dia = t["dia_id"].as_str()?;
                        let text = t["text"].as_str()?;
                        if text.trim().is_empty() {
                            return None;
                        }
                        let line = match t["speaker"].as_str() {
                            Some(sp) if !sp.is_empty() => format!("{sp}: {text}"),
                            _ => text.to_string(),
                        };
                        Some((dia.to_string(), line))
                    })
                    .collect();
                let chunk_size = if window == 0 { items.len().max(1) } else { window };
                for chunk in items.chunks(chunk_size) {
                    let text = chunk
                        .iter()
                        .map(|(_, l)| l.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    if text.trim().is_empty() {
                        continue;
                    }
                    let wr = store
                        .write_episode(EpisodeWrite {
                            namespace: ns.into(),
                            text,
                            t_valid_from: None,
                            metadata: None,
                        })
                        .expect("write_episode");
                    // Every turn in the chunk maps to this episode's doc_id.
                    for (dia, _) in chunk {
                        dia2doc.insert(dia.clone(), wr.episode_id.0);
                    }
                }
            }
        }

        let Some(qas) = conv["qa"].as_array() else { continue };
        for qa in qas {
            let Some(question) = qa["question"].as_str() else {
                continue;
            };
            let cat = qa["category"]
                .as_str()
                .map(|s| s.to_string())
                .or_else(|| qa["category"].as_i64().map(|n| n.to_string()))
                .unwrap_or_else(|| "?".into());
            // gold evidence dia_ids -> distinct episode (chunk) doc_ids
            let gold: HashSet<u64> = qa["evidence"]
                .as_array()
                .map(|ev| {
                    ev.iter()
                        .filter_map(|e| e.as_str())
                        .filter_map(|d| dia2doc.get(d).copied())
                        .collect()
                })
                .unwrap_or_default();
            if gold.is_empty() {
                continue; // unanswerable / no resolvable evidence
            }

            let t = Instant::now();
            let r = store.query(&MemoryQuery {
                namespace: ns.into(),
                query: question.into(),
                as_of: None,
                lanes: QueryLanes::three_lane(),
                k,
            });
            latency_us += t.elapsed().as_micros();
            vector_any |= r.lanes_used.contains(&Lane::Vector);

            let topk: HashSet<u64> = r.hits.iter().map(|h| h.doc_id).collect();
            let found = gold.iter().filter(|g| topk.contains(g)).count();
            let hit = if found > 0 { 1.0 } else { 0.0 };
            let recall = found as f64 / gold.len() as f64;
            overall.add(hit, recall);
            by_cat.entry(cat).or_default().add(hit, recall);
        }
    }

    let n = overall.n.max(1) as f64;
    println!(
        "\n{label}:\n  overall  n={:4}  hit@{k}={:.3}  recall@{k}={:.3}  vector_lane={}  {:.0} us/query",
        overall.n,
        overall.hit / n,
        overall.recall / n,
        if vector_any { "ON" } else { "off(gated)" },
        latency_us as f64 / n
    );
    let mut cats: Vec<_> = by_cat.iter().collect();
    cats.sort_by(|a, b| a.0.cmp(b.0));
    for (c, a) in cats {
        let an = a.n.max(1) as f64;
        println!(
            "    {:12} n={:4}  hit@{k}={:.3}  recall@{k}={:.3}",
            cat_name(c),
            a.n,
            a.hit / an,
            a.recall / an
        );
    }
}

fn main() {
    let path = std::env::var("LOCOMO_JSON")
        .unwrap_or_else(|_| "/Users/sushanth/git-clone/locomo/data/locomo10.json".into());
    let data: Vec<Value> = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&path).expect("open LOCOMO_JSON"),
    ))
    .expect("parse locomo json");
    let k = std::env::var("K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10usize);

    println!(
        "LoCoMo recall on SochDB native three_lane memory ({} conversations, per-conversation scope, k={})",
        data.len(),
        k
    );

    run(
        "mock (lexical-only)",
        Arc::new(sochdb_query::MockEmbeddingProvider::new(384)),
        &data,
        k,
    );

    // Strong HTTP embedder via precomputed cache (no per-turn network / ONNX).
    if let Ok(cache_path) = std::env::var("EMBED_CACHE") {
        let map: HashMap<String, Vec<f32>> = serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(&cache_path).expect("open EMBED_CACHE"),
        ))
        .expect("parse EMBED_CACHE");
        let dim = map.values().next().map(|v| v.len()).unwrap_or(0);
        let label = std::env::var("EMBED_LABEL").unwrap_or_else(|_| "cached embedder".into());
        run(
            &format!("{label} (hybrid, dim={dim})"),
            Arc::new(CachedEmbedder { map, dim }),
            &data,
            k,
        );
        return;
    }

    #[cfg(feature = "fastembed")]
    {
        // One embedder shared across all conversations (avoids reloading the model).
        // Model selectable via EMBED_MODEL (e.g. bge-small-en, bge-base-en, bge-large-en).
        let model = std::env::var("EMBED_MODEL").unwrap_or_else(|_| "bge-small-en".into());
        let emb =
            sochdb_query::embedding_provider::embedder_from_spec(&format!("fastembed:{model}"));
        run(&format!("fastembed {model} (hybrid)"), emb, &data, k);
    }
    #[cfg(not(feature = "fastembed"))]
    println!("\n(rebuild with --features fastembed to measure the semantic path)");
}
