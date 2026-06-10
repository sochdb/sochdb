//! Episode embedding + per-namespace vector store for the vector retrieval lane.

use crate::enrichment::EnrichmentJob;
use crate::store::MemoryStore;
use sochdb_query::EmbeddingProvider;
use std::sync::Arc;

/// Cosine similarity for L2-normalized embeddings.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    dot
}

impl MemoryStore {
    pub fn embedder(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.embedder
    }

    /// Embed an episode and store its vector for semantic retrieval.
    pub fn enrich_episode(&self, job: &EnrichmentJob) -> Result<(), String> {
        let mut embedding = self
            .embedder
            .embed(&job.text)
            .map_err(|e| e.to_string())?;
        self.embedder.normalize(&mut embedding);

        let mut namespaces = self.namespaces.write();
        let ns = namespaces
            .get_mut(&job.namespace)
            .ok_or_else(|| format!("namespace not found: {}", job.namespace))?;

        ns.vectors.insert(job.episode_id, embedding);

        if let Some(episode) = ns.episodes.get_mut(&job.episode_id) {
            episode.enriched = true;
        }

        Ok(())
    }

    /// Drain all pending enrichment jobs synchronously (tests / bench warmup).
    pub fn drain_enrichment_queue(&self) -> usize {
        let mut processed = 0usize;
        while let Some(job) = self.enrichment.pop() {
            if self.enrich_episode(&job).is_ok() {
                processed += 1;
            }
            self.enrichment.mark_processed();
        }
        processed
    }

    /// Vector lane search over enriched episodes (brute-force; tuned for agent-memory scale).
    pub fn search_vector(&self, namespace: &str, query: &str, k: usize) -> Vec<(u64, f32)> {
        let mut query_emb = match self.embedder.embed(query) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        self.embedder.normalize(&mut query_emb);

        let namespaces = self.namespaces.read();
        let Some(ns) = namespaces.get(namespace) else {
            return Vec::new();
        };
        if ns.vectors.is_empty() {
            return Vec::new();
        }

        let mut ranked: Vec<(u64, f32)> = ns
            .vectors
            .iter()
            .map(|(id, vec)| (*id, cosine_similarity(&query_emb, vec)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }

    pub fn enriched_episode_count(&self, namespace: &str) -> usize {
        self.namespaces
            .read()
            .get(namespace)
            .map(|ns| ns.vectors.len())
            .unwrap_or(0)
    }
}