use crate::enrichment::{EnrichmentJob, EnrichmentQueue};
use crate::episode::{ConversationTurn, Episode, EpisodeId, EpisodeWrite};
use crate::fact::{FactEdge, FactId};
use parking_lot::RwLock;
use sochdb_query::{EmbeddingProvider, MockEmbeddingProvider, trigram_index::TrigramIndex};
use sochdb_storage::hlc::HybridLogicalClock;
use sochdb_vector::bm25::BM25Config;
use sochdb_vector::inverted_index::InvertedIndex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("namespace not found: {0}")]
    NamespaceNotFound(String),
    #[error("episode not found: {0}")]
    EpisodeNotFound(u64),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type MemoryResult<T> = Result<T, MemoryError>;

#[derive(Debug, Clone)]
pub struct MemoryStoreConfig {
    pub max_enrichment_queue: usize,
    /// Run embedding + HNSW insert synchronously on write (bench/tests).
    pub enrich_on_write: bool,
}

impl Default for MemoryStoreConfig {
    fn default() -> Self {
        Self {
            max_enrichment_queue: 10_000,
            enrich_on_write: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WriteResult {
    pub episode_id: EpisodeId,
    pub t_created: u64,
    pub lexical_indexed: bool,
    pub ingestion_lag_us: u64,
    pub enrichment_queued: bool,
}

pub(crate) struct NamespaceIndexes {
    pub(crate) bm25: InvertedIndex,
    pub(crate) trigram: TrigramIndex,
    pub(crate) vectors: HashMap<u64, Vec<f32>>,
    pub(crate) episodes: HashMap<u64, Episode>,
    facts: Vec<FactEdge>,
    next_episode_id: u64,
    next_fact_id: u64,
}

impl NamespaceIndexes {
    fn new() -> Self {
        Self {
            bm25: InvertedIndex::new(BM25Config::default()),
            trigram: TrigramIndex::new(),
            vectors: HashMap::new(),
            episodes: HashMap::new(),
            facts: Vec::new(),
            next_episode_id: 1,
            next_fact_id: 1,
        }
    }
}

/// Independently owned mutable state for one namespace.
pub(crate) type NamespaceHandle = Arc<RwLock<NamespaceIndexes>>;

/// Agent memory store: write-time lexical recall + async enrichment queue.
pub struct MemoryStore {
    hlc: HybridLogicalClock,
    /// Directory of namespaces — *lookup and creation only*.
    ///
    /// Each namespace owns its indexes behind its own lock, so this outer lock
    /// is held just long enough for a hash lookup and an `Arc` clone, never
    /// across index mutation. Holding one store-wide write guard while updating
    /// BM25, trigram and episode state made every agent's write serialise
    /// against every other agent's write, and block every reader, over state
    /// they do not share: namespaces have nothing in common but this map.
    pub(crate) namespaces: RwLock<HashMap<String, NamespaceHandle>>,
    pub(crate) enrichment: EnrichmentQueue,
    pub(crate) embedder: Arc<dyn EmbeddingProvider>,
    config: MemoryStoreConfig,
}

fn default_embedder() -> Arc<dyn EmbeddingProvider> {
    Arc::new(MockEmbeddingProvider::new(384))
}

impl MemoryStore {
    pub fn new(_data_dir: Option<&Path>, config: MemoryStoreConfig) -> Self {
        Self::with_embedder(_data_dir, config, default_embedder())
    }

    pub fn with_embedder(
        _data_dir: Option<&Path>,
        config: MemoryStoreConfig,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            hlc: HybridLogicalClock::new(),
            namespaces: RwLock::new(HashMap::new()),
            enrichment: EnrichmentQueue::new(config.max_enrichment_queue),
            embedder,
            config,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(None, MemoryStoreConfig::default())
    }

    /// Build with the embedder selected by the `SOCHDB_EMBEDDER` environment
    /// variable (e.g. `fastembed:bge-small-en`, or `mock`/unset for the default
    /// mock embedder). See [`sochdb_query::embedding_provider::embedder_from_env`].
    pub fn from_env() -> Self {
        Self::with_embedder(
            None,
            MemoryStoreConfig::default(),
            sochdb_query::embedding_provider::embedder_from_env(),
        )
    }

    pub fn enrichment_queue(&self) -> &EnrichmentQueue {
        &self.enrichment
    }

    /// Handle for an existing namespace, or `None`.
    ///
    /// The store-wide read guard is dropped before the caller can touch the
    /// namespace, so a long read of one namespace never blocks a write to
    /// another.
    pub(crate) fn namespace(&self, name: &str) -> Option<NamespaceHandle> {
        self.namespaces.read().get(name).cloned()
    }

    /// Handle for a namespace, creating it if this is its first write.
    ///
    /// Takes the cheap read path first: after the first write to a namespace,
    /// which is the overwhelming majority of calls, no writer ever touches the
    /// directory. `entry` on the write path keeps a concurrent creation of the
    /// same namespace from producing two sets of indexes.
    pub(crate) fn namespace_or_create(&self, name: &str) -> NamespaceHandle {
        if let Some(handle) = self.namespaces.read().get(name) {
            return Arc::clone(handle);
        }
        Arc::clone(
            self.namespaces
                .write()
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(RwLock::new(NamespaceIndexes::new()))),
        )
    }

    /// Write episode: lexical lanes indexed synchronously; enrichment queued async.
    pub fn write_episode(&self, write: EpisodeWrite) -> MemoryResult<WriteResult> {
        let start = Instant::now();
        let t_created = self.hlc.next();
        // Default validity-start to wall-clock unix milliseconds, matching the
        // `as_of=<unix_ms>` query contract. `t_created` is a raw HLC tick
        // (`physical_micros << 16 | logical`, ~1e21), so defaulting to it made
        // the bi-temporal filter `t_valid_from <= as_of` always false for any
        // realistic `as_of` — silently returning zero results for every
        // episode written without an explicit validity time (the common case).
        // Callers that pass `t_valid_from` keep their own time domain.
        let t_valid = write
            .t_valid_from
            .unwrap_or_else(|| HybridLogicalClock::physical_time(t_created) / 1000);

        let handle = self.namespace_or_create(&write.namespace);
        let episode_id;
        {
            let mut ns = handle.write();

            episode_id = EpisodeId(ns.next_episode_id);
            ns.next_episode_id += 1;

            let doc_id = episode_id.0;
            ns.bm25.add_document_with_id(doc_id, &write.text);
            ns.trigram.insert(doc_id, &write.text);

            let episode = Episode {
                id: episode_id,
                namespace: write.namespace.clone(),
                text: write.text.clone(),
                t_created,
                t_valid_from: t_valid,
                enriched: false,
                metadata: write.metadata.clone(),
            };
            ns.episodes.insert(doc_id, episode);
        }

        let job = EnrichmentJob {
            namespace: write.namespace.clone(),
            episode_id: episode_id.0,
            text: write.text.clone(),
        };

        let enrichment_queued = self.enrichment.try_enqueue(job.clone()).is_ok();
        let ingestion_lag_us = start.elapsed().as_micros() as u64;

        let result = WriteResult {
            episode_id,
            t_created,
            lexical_indexed: true,
            ingestion_lag_us,
            enrichment_queued,
        };

        if self.config.enrich_on_write {
            let _ = self.enrich_episode(&job);
        }

        Ok(result)
    }

    /// Ingest conversation turns as WINDOWED, speaker-prefixed episodes — the
    /// ingestion shape that maximizes retrieval recall.
    ///
    /// Writing one bare-text episode per turn strips the conversational context
    /// an episode needs to be retrievable: a short, context-dependent turn like
    /// "Yeah, May 7th" cannot be matched to "When did she go?" by any embedder.
    /// Grouping `window` consecutive turns into a single `"speaker: text"`-
    /// formatted episode restores that context, and a `stride` smaller than
    /// `window` produces OVERLAPPING episodes so no turn is stranded near a chunk
    /// boundary without context.
    ///
    /// Measured on LoCoMo (same 384-d embedder) — retrieval *structure*, not
    /// embedder strength, is the dominant recall lever:
    /// - one bare turn per episode:            hit@10 ~0.61
    /// - `window=5,  stride=5` (disjoint):     hit@10 ~0.89
    /// - `window=10, stride=4` (≈40% overlap): hit@10 ~0.91  (best)
    ///
    /// Good defaults for chat: `window≈10`, `stride≈window*0.4`. Too-large
    /// windows over-coarsen (the vector lane loses discrimination); too-small a
    /// stride (e.g. 1) crowds top-k with near-duplicate episodes and *hurts*.
    /// `window` is clamped to ≥1 and `stride` to `1..=window`.
    ///
    /// Returns one [`WriteResult`] per episode written (i.e. per window).
    pub fn write_turns(
        &self,
        namespace: &str,
        turns: &[ConversationTurn],
        window: usize,
        stride: usize,
        t_valid_from: Option<u64>,
    ) -> MemoryResult<Vec<WriteResult>> {
        let w = window.max(1);
        let s = stride.clamp(1, w);
        let mut out = Vec::new();
        let mut start = 0;
        while start < turns.len() {
            let end = (start + w).min(turns.len());
            let group = &turns[start..end];
            let text = group
                .iter()
                .map(|t| {
                    if t.speaker.is_empty() {
                        t.text.clone()
                    } else {
                        format!("{}: {}", t.speaker, t.text)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !text.trim().is_empty() {
                out.push(self.write_episode(EpisodeWrite {
                    namespace: namespace.to_string(),
                    text,
                    t_valid_from,
                    metadata: None,
                })?);
            }
            if end == turns.len() {
                break;
            }
            start += s;
        }
        Ok(out)
    }

    pub fn get_episode(&self, namespace: &str, id: EpisodeId) -> MemoryResult<Episode> {
        let handle = self
            .namespace(namespace)
            .ok_or_else(|| MemoryError::NamespaceNotFound(namespace.to_string()))?;
        let ns = handle.read();
        ns.episodes
            .get(&id.0)
            .cloned()
            .ok_or(MemoryError::EpisodeNotFound(id.0))
    }

    pub fn namespace_bm25(&self, namespace: &str) -> Option<Arc<InvertedIndex>> {
        // BM25 index is behind the namespace lock — expose search via store methods instead
        let _ = namespace;
        None
    }

    pub fn search_bm25(&self, namespace: &str, query: &str, k: usize) -> Vec<(u64, f32)> {
        let Some(handle) = self.namespace(namespace) else {
            return Vec::new();
        };
        let ns = handle.read();
        ns.bm25.search(query, k)
    }

    pub fn search_trigram_literal(
        &self,
        namespace: &str,
        literal: &str,
        k: usize,
    ) -> Vec<(u64, f32)> {
        let trigrams = sochdb_query::trigram_index::trigrams_of(literal);
        if trigrams.is_empty() {
            return Vec::new();
        }
        let Some(handle) = self.namespace(namespace) else {
            return Vec::new();
        };
        let ns = handle.read();
        ns.trigram
            .candidates(&trigrams)
            .into_iter()
            .take(k)
            .map(|doc_id| (doc_id, 1.0))
            .collect()
    }

    pub fn episode_text(&self, namespace: &str, doc_id: u64) -> Option<String> {
        let handle = self.namespace(namespace)?;
        let ns = handle.read();
        ns.episodes.get(&doc_id).map(|e| e.text.clone())
    }

    pub fn add_fact(&self, namespace: &str, mut fact: FactEdge) -> MemoryResult<FactId> {
        let handle = self.namespace_or_create(namespace);
        let mut ns = handle.write();
        let id = FactId(ns.next_fact_id);
        ns.next_fact_id += 1;
        fact.id = id;
        ns.facts.push(fact);
        Ok(id)
    }

    pub fn facts_valid_at(&self, namespace: &str, tau: u64) -> Vec<FactEdge> {
        let Some(handle) = self.namespace(namespace) else {
            return Vec::new();
        };
        let ns = handle.read();
        ns.facts
            .iter()
            .filter(|f| f.is_valid_at(tau))
            .cloned()
            .collect()
    }

    pub fn invalidate_fact(&self, namespace: &str, fact_id: FactId, t_invalid: u64) -> bool {
        let Some(handle) = self.namespace(namespace) else {
            return false;
        };
        let mut ns = handle.write();
        if let Some(fact) = ns.facts.iter_mut().find(|f| f.id == fact_id) {
            fact.invalidate(t_invalid);
            return true;
        }
        false
    }

    pub fn episode_count(&self, namespace: &str) -> usize {
        self.namespace(namespace)
            .map(|handle| handle.read().episodes.len())
            .unwrap_or(0)
    }
}
