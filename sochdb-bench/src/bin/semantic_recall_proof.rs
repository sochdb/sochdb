//! Proof of the "fast + accurate agent memory" selling point.
//!
//! Ingests a corpus into SochDB's NATIVE three_lane memory path and queries it
//! with PARAPHRASES — questions that share essentially no content words with
//! their relevant episode. Lexical retrieval (BM25/trigram) cannot bridge that
//! vocabulary gap; only a semantic vector lane can. We compare:
//!   - mock embedder  -> the is_semantic() gate skips the vector lane, so
//!     three_lane degrades to lexical. This is the old default's behaviour.
//!   - fastembed (bge-small-en) -> a real semantic vector lane, fused with the
//!     lexical lanes via RRF.
//! A large recall gap between the two IS the proof: SochDB's hybrid memory is
//! accurate (finds paraphrases) only with a real embedder — which the shipped
//! defaults now make safe + on-by-default.
//!
//! Build: cargo build --release -p sochdb-bench --bin semantic-recall-proof --features fastembed
//! Run from sochdb-query/ so the cached model (.fastembed_cache) is found.

use std::sync::Arc;
use std::time::Instant;

use sochdb_memory::{EpisodeWrite, MemoryQuery, MemoryStore, MemoryStoreConfig, QueryLanes};
use sochdb_query::EmbeddingProvider;

/// (episode_text, paraphrase_query): the query is a semantic restatement of the
/// episode with deliberately minimal keyword overlap.
const CASES: &[(&str, &str)] = &[
    (
        "The patient was prescribed metformin to manage type 2 diabetes.",
        "what medication did the doctor give for high blood sugar?",
    ),
    (
        "She adopted a rescue dog named Biscuit last spring.",
        "when did they welcome a new pet into the family?",
    ),
    (
        "The startup closed a Series B round led by Accel.",
        "who financed the company's latest fundraising stage?",
    ),
    (
        "He flew to Tokyo for the robotics conference in March.",
        "which city did he travel to for the engineering event?",
    ),
    (
        "The recipe calls for two cups of all-purpose flour.",
        "how much wheat powder does the dish need?",
    ),
    (
        "Maria defended her PhD thesis on coral reef ecology.",
        "what subject did she earn her doctorate studying?",
    ),
    (
        "The bridge was closed for repairs after the earthquake.",
        "why was the crossing shut down following the tremor?",
    ),
    (
        "Their flight was delayed three hours by a snowstorm.",
        "what caused the long wait before takeoff?",
    ),
    (
        "The museum acquired a Monet painting at auction.",
        "what artwork did the gallery purchase recently?",
    ),
    (
        "Quarterly sales grew forty percent after the ad campaign launched.",
        "how did revenue change once the marketing push began?",
    ),
];

fn run(label: &str, embedder: Arc<dyn EmbeddingProvider>, k: usize) {
    let store = MemoryStore::with_embedder(
        None,
        MemoryStoreConfig {
            enrich_on_write: true,
            ..MemoryStoreConfig::default()
        },
        embedder,
    );

    // Ingest all episodes; remember each one's doc_id for scoring.
    let mut ids = Vec::with_capacity(CASES.len());
    for (text, _) in CASES {
        let wr = store
            .write_episode(EpisodeWrite {
                namespace: "proof".into(),
                text: (*text).into(),
                t_valid_from: None,
                metadata: None,
            })
            .expect("write_episode");
        ids.push(wr.episode_id.0);
    }

    let mut hits = 0usize;
    let mut latency_us = 0u128;
    let mut vector_active = false;
    for (i, (_, query)) in CASES.iter().enumerate() {
        let t = Instant::now();
        let r = store.query(&MemoryQuery {
            namespace: "proof".into(),
            query: (*query).into(),
            as_of: None,
            lanes: QueryLanes::three_lane(),
            k,
        });
        latency_us += t.elapsed().as_micros();
        vector_active |= r.lanes_used.contains(&sochdb_memory::Lane::Vector);
        if r.hits.iter().any(|h| h.doc_id == ids[i]) {
            hits += 1;
        }
    }
    println!(
        "  {label:30} recall@{k} = {hits:2}/{} = {:.2}   vector_lane={}   {:.0} us/query",
        CASES.len(),
        hits as f64 / CASES.len() as f64,
        if vector_active { "ON" } else { "off(gated)" },
        latency_us as f64 / CASES.len() as f64
    );
}

fn main() {
    let k = 3;
    println!(
        "Paraphrase recall@{k} on SochDB native three_lane memory ({} cases, queries share ~no keywords with their episode):\n",
        CASES.len()
    );

    run(
        "mock (lexical-only)",
        Arc::new(sochdb_query::MockEmbeddingProvider::new(384)),
        k,
    );

    #[cfg(feature = "fastembed")]
    {
        let emb = sochdb_query::embedding_provider::embedder_from_spec("fastembed:bge-small-en");
        run("fastembed bge-small (hybrid)", emb, k);
    }
    #[cfg(not(feature = "fastembed"))]
    println!("  (rebuild with --features fastembed to measure the semantic path)");
}
