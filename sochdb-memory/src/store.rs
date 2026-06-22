use crate::enrichment::{EnrichmentJob, EnrichmentQueue};
use crate::episode::{Episode, EpisodeId, EpisodeWrite};
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

/// Agent memory store: write-time lexical recall + async enrichment queue.
pub struct MemoryStore {
    hlc: HybridLogicalClock,
    pub(crate) namespaces: RwLock<HashMap<String, NamespaceIndexes>>,
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

        let mut namespaces = self.namespaces.write();
        let ns = namespaces
            .entry(write.namespace.clone())
            .or_insert_with(NamespaceIndexes::new);

        let episode_id = EpisodeId(ns.next_episode_id);
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

        let job = EnrichmentJob {
            namespace: write.namespace.clone(),
            episode_id: doc_id,
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

        // Release namespace lock before enrichment (embed + vector insert re-lock).
        drop(namespaces);

        if self.config.enrich_on_write {
            let _ = self.enrich_episode(&job);
        }

        Ok(result)
    }

    pub fn get_episode(&self, namespace: &str, id: EpisodeId) -> MemoryResult<Episode> {
        let namespaces = self.namespaces.read();
        let ns = namespaces
            .get(namespace)
            .ok_or_else(|| MemoryError::NamespaceNotFound(namespace.to_string()))?;
        ns.episodes
            .get(&id.0)
            .cloned()
            .ok_or_else(|| MemoryError::EpisodeNotFound(id.0))
    }

    pub fn namespace_bm25(&self, namespace: &str) -> Option<Arc<InvertedIndex>> {
        // BM25 index is behind RwLock in namespace — expose search via store methods instead
        let _ = namespace;
        None
    }

    pub fn search_bm25(&self, namespace: &str, query: &str, k: usize) -> Vec<(u64, f32)> {
        let namespaces = self.namespaces.read();
        namespaces
            .get(namespace)
            .map(|ns| ns.bm25.search(query, k))
            .unwrap_or_default()
    }

    pub fn search_trigram_literal(
        &self,
        namespace: &str,
        literal: &str,
        k: usize,
    ) -> Vec<(u64, f32)> {
        let namespaces = self.namespaces.read();
        let Some(ns) = namespaces.get(namespace) else {
            return Vec::new();
        };
        let trigrams = sochdb_query::trigram_index::trigrams_of(literal);
        if trigrams.is_empty() {
            return Vec::new();
        }
        let candidates = ns.trigram.candidates(&trigrams);
        candidates
            .into_iter()
            .take(k)
            .map(|doc_id| (doc_id, 1.0))
            .collect()
    }

    pub fn episode_text(&self, namespace: &str, doc_id: u64) -> Option<String> {
        let namespaces = self.namespaces.read();
        namespaces
            .get(namespace)?
            .episodes
            .get(&doc_id)
            .map(|e| e.text.clone())
    }

    pub fn add_fact(&self, namespace: &str, mut fact: FactEdge) -> MemoryResult<FactId> {
        let mut namespaces = self.namespaces.write();
        let ns = namespaces
            .entry(namespace.to_string())
            .or_insert_with(NamespaceIndexes::new);
        let id = FactId(ns.next_fact_id);
        ns.next_fact_id += 1;
        fact.id = id;
        ns.facts.push(fact);
        Ok(id)
    }

    pub fn facts_valid_at(&self, namespace: &str, tau: u64) -> Vec<FactEdge> {
        let namespaces = self.namespaces.read();
        namespaces
            .get(namespace)
            .map(|ns| {
                ns.facts
                    .iter()
                    .filter(|f| f.is_valid_at(tau))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn invalidate_fact(&self, namespace: &str, fact_id: FactId, t_invalid: u64) -> bool {
        let mut namespaces = self.namespaces.write();
        let Some(ns) = namespaces.get_mut(namespace) else {
            return false;
        };
        if let Some(fact) = ns.facts.iter_mut().find(|f| f.id == fact_id) {
            fact.invalidate(t_invalid);
            return true;
        }
        false
    }

    pub fn episode_count(&self, namespace: &str) -> usize {
        self.namespaces
            .read()
            .get(namespace)
            .map(|ns| ns.episodes.len())
            .unwrap_or(0)
    }
}
