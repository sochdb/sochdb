//! Episode embedding + per-namespace vector store for the vector retrieval lane.

use crate::enrichment::EnrichmentJob;
use crate::store::MemoryStore;
use crate::topk;
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
    ///
    /// The embedding is computed *before* the namespace lock is taken: it is
    /// the expensive part and it needs nothing from the index, so holding the
    /// lock across it would stall every reader and writer of that namespace for
    /// the duration of a model call.
    pub fn enrich_episode(&self, job: &EnrichmentJob) -> Result<(), String> {
        let mut embedding = self.embedder.embed(&job.text).map_err(|e| e.to_string())?;
        self.embedder.normalize(&mut embedding);

        let handle = self
            .namespace(&job.namespace)
            .ok_or_else(|| format!("namespace not found: {}", job.namespace))?;
        let mut ns = handle.write();

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

    /// Apply the pending enrichment for one namespace, and only that namespace.
    ///
    /// This is the freshness a reader of `namespace` can actually observe.
    /// Draining the whole queue instead makes a foreground query wait for
    /// embedding work belonging to unrelated tenants — unbounded latency
    /// imported from a queue the caller does not read — while leaving its *own*
    /// read-your-writes guarantee no stronger.
    ///
    /// Returns the number of episodes enriched.
    pub fn drain_enrichment_for(&self, namespace: &str) -> usize {
        let mut processed = 0usize;
        for job in self.enrichment.take_namespace(namespace) {
            if self.enrich_episode(&job).is_ok() {
                processed += 1;
            }
            self.enrichment.mark_processed();
        }
        processed
    }

    /// Vector lane search over enriched episodes (brute-force; tuned for agent-memory scale).
    ///
    /// Scoring is `O(Nd)` and unavoidable without an index. Selection is
    /// `O(N log k)` via a bounded heap rather than `O(N log N)` from sorting
    /// every score to discard all but `k` of them.
    pub fn search_vector(&self, namespace: &str, query: &str, k: usize) -> Vec<(u64, f32)> {
        if k == 0 {
            return Vec::new();
        }

        // Asymmetric: embed the QUERY with the model's query instruction (BGE),
        // not as a document — aligns it with the indexed doc embeddings.
        let mut query_emb = match self.embedder.embed_query(query) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        self.embedder.normalize(&mut query_emb);

        let Some(handle) = self.namespace(namespace) else {
            return Vec::new();
        };
        let ns = handle.read();
        if ns.vectors.is_empty() {
            return Vec::new();
        }

        topk::top_k(
            ns.vectors
                .iter()
                .map(|(id, vec)| (*id, cosine_similarity(&query_emb, vec))),
            k,
        )
    }

    pub fn enriched_episode_count(&self, namespace: &str) -> usize {
        self.namespace(namespace)
            .map(|handle| handle.read().vectors.len())
            .unwrap_or(0)
    }
}
