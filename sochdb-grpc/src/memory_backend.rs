//! Shared bridge from gRPC ContextService to sochdb-memory + ContextCompiler.

use sochdb_memory::{EpisodeWrite, MemoryQuery, MemoryStore, QueryLanes, WriteResult};
use sochdb_query::{
    CompiledContext, ContextCandidate, ContextCompiler, ContextSpec, ContextTemplate,
    count_tokens_exact,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Output format for compiled context (mirrors proto OutputFormat).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextOutputFormat {
    Toon,
    Json,
    Markdown,
    Text,
}

pub struct MemoryBackend {
    store: Arc<MemoryStore>,
}

/// Retrieval results for one section, reusable across compilations.
///
/// Packing a context to a budget can need more than one attempt when the
/// output format wraps the compiled body, and retrieval must not be repeated
/// for each attempt: the search is the expensive half and its results do not
/// depend on the budget.
pub struct SectionCandidates {
    /// Fused hits, best first.
    pub ranked: Vec<(u64, f32)>,
    /// Candidate texts by doc id.
    pub texts: HashMap<u64, ContextCandidate>,
}

impl MemoryBackend {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    pub fn write_episode(
        &self,
        namespace: &str,
        text: &str,
        t_valid_from: Option<u64>,
        metadata: Option<serde_json::Value>,
    ) -> Result<WriteResult, String> {
        self.store
            .write_episode(EpisodeWrite {
                namespace: namespace.to_string(),
                text: text.to_string(),
                t_valid_from,
                metadata,
            })
            .map_err(|e| e.to_string())
    }

    pub fn get_episode_text(&self, namespace: &str, doc_id: u64) -> Option<String> {
        self.store.episode_text(namespace, doc_id)
    }

    /// Three-lane retrieval, without packing.
    pub fn search_candidates(
        &self,
        namespace: &str,
        query: &str,
        lanes: QueryLanes,
    ) -> SectionCandidates {
        // Wait only for this namespace's enrichment. The store-wide queue also
        // holds work for namespaces this query will never read, and blocking on
        // that imports another tenant's backlog into this request's latency
        // without making this request's results any fresher.
        if lanes.vector && self.store.enrichment_queue().depth_for(namespace) > 0 {
            self.store.drain_enrichment_for(namespace);
        }

        let mq = MemoryQuery {
            namespace: namespace.to_string(),
            query: query.to_string(),
            as_of: None,
            lanes,
            k: 32,
        };
        let hits = self.store.query(&mq);

        let ranked: Vec<(u64, f32)> = hits.hits.iter().map(|h| (h.doc_id, h.score)).collect();

        let mut texts = HashMap::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        for hit in &hits.hits {
            if let Ok(ep) = self
                .store
                .get_episode(namespace, sochdb_memory::EpisodeId(hit.doc_id))
            {
                texts.insert(
                    hit.doc_id,
                    ContextCandidate {
                        doc_id: hit.doc_id,
                        text: ep.text,
                        relevance: hit.score,
                        timestamp_secs: now,
                        episode_id: Some(hit.doc_id),
                        t_valid_from: Some(ep.t_valid_from),
                        t_valid_to: None,
                    },
                );
            }
        }

        SectionCandidates { ranked, texts }
    }

    /// Pack already-retrieved candidates into a context of at most `budget`
    /// tokens.
    pub fn compile_candidates(
        candidates: &SectionCandidates,
        budget: usize,
        template: ContextTemplate,
    ) -> CompiledContext {
        let spec = ContextSpec {
            budget,
            template,
            ..ContextSpec::default()
        };
        let compiler = ContextCompiler::new(&spec);
        compiler.compile(&spec, &candidates.ranked, &[], &[], &candidates.texts)
    }

    /// Three-lane retrieval + context compiler under an exact token budget.
    pub fn search_and_compile(
        &self,
        namespace: &str,
        query: &str,
        budget: usize,
        lanes: QueryLanes,
        template: ContextTemplate,
    ) -> Result<CompiledContext, String> {
        let candidates = self.search_candidates(namespace, query, lanes);
        Ok(Self::compile_candidates(&candidates, budget, template))
    }

    /// Render a compiled context in the requested wire format.
    ///
    /// Every format emits `compiled.body`. The compiler packed that body to the
    /// budget using the matching [`ContextTemplate`], so re-rendering the facts
    /// here with different framing would emit something the budget never
    /// measured — and would silently drop the content of any `CompiledContext`
    /// carrying a body with no facts beside it.
    pub fn format_compiled(compiled: &CompiledContext, format: ContextOutputFormat) -> String {
        match format {
            ContextOutputFormat::Json => serde_json::json!({
                "body": compiled.body,
                "exact_tokens": compiled.exact_tokens,
                "budget": compiled.budget,
                "truncated": compiled.truncated,
                "facts": compiled.facts,
            })
            .to_string(),
            ContextOutputFormat::Markdown | ContextOutputFormat::Text => compiled.body.clone(),
            ContextOutputFormat::Toon => format!(
                "<context tokens=\"{}\" budget=\"{}\" truncated=\"{}\">\n{}\n</context>",
                compiled.exact_tokens, compiled.budget, compiled.truncated, compiled.body
            ),
        }
    }

    pub fn estimate_tokens_exact(content: &str) -> u32 {
        count_tokens_exact(content) as u32
    }

    /// Parse retrieval lanes from section options (`lanes=lexical|three_lane|hybrid`).
    pub fn parse_lanes(options: &HashMap<String, String>) -> QueryLanes {
        match options.get("lanes").map(|s| s.as_str()) {
            Some("three_lane") | Some("hybrid") | Some("all") => QueryLanes::three_lane(),
            Some("bm25") => QueryLanes {
                bm25: true,
                trigram: false,
                vector: false,
                bm25_weight: 1.0,
                trigram_weight: 0.0,
                vector_weight: 0.0,
            },
            Some("trigram") => QueryLanes {
                bm25: false,
                trigram: true,
                vector: false,
                bm25_weight: 0.0,
                trigram_weight: 1.0,
                vector_weight: 0.0,
            },
            // Default: full hybrid. The vector lane auto-degrades (skips) when
            // no semantic embedder is configured, so an unspecified caller gets
            // semantic+lexical fusion when it's available and clean lexical
            // results when it isn't — never mock-vector noise. Callers can still
            // opt down to lanes=bm25/trigram for latency.
            _ => QueryLanes::three_lane(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sochdb_memory::MemoryStore;

    #[test]
    fn ingest_search_compile_pipeline() {
        let store = Arc::new(MemoryStore::with_defaults());
        let backend = MemoryBackend::new(store);
        let wr = backend
            .write_episode(
                "agent-1",
                "Caroline attended the LGBTQ support group on 7 May 2023.",
                None,
                None,
            )
            .unwrap();
        assert!(wr.lexical_indexed);

        let compiled = backend
            .search_and_compile(
                "agent-1",
                "LGBTQ support group",
                512,
                QueryLanes::lexical_only(),
                ContextTemplate::Markdown,
            )
            .unwrap();

        assert!(!compiled.body.is_empty());
        assert!(compiled.exact_tokens <= 512);
        assert!(!compiled.facts.is_empty());
    }
}
