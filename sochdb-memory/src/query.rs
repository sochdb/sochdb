use crate::provenance::{ProvenanceBundle, TrustScore, TrustScoreConfig};
use crate::store::MemoryStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lane {
    Bm25,
    Trigram,
    Vector,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryLanes {
    pub bm25: bool,
    pub trigram: bool,
    pub vector: bool,
    pub bm25_weight: f32,
    pub trigram_weight: f32,
    pub vector_weight: f32,
}

impl QueryLanes {
    pub fn lexical_only() -> Self {
        Self {
            bm25: true,
            trigram: true,
            vector: false,
            bm25_weight: 0.6,
            trigram_weight: 0.4,
            vector_weight: 0.0,
        }
    }

    pub fn three_lane() -> Self {
        Self {
            bm25: true,
            trigram: true,
            vector: true,
            bm25_weight: 0.4,
            trigram_weight: 0.2,
            vector_weight: 0.4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub namespace: String,
    pub query: String,
    pub as_of: Option<u64>,
    pub lanes: QueryLanes,
    pub k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHit {
    pub doc_id: u64,
    pub score: f32,
    pub lane: Lane,
    pub snippet: String,
    pub provenance: ProvenanceBundle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQueryResult {
    pub hits: Vec<MemoryHit>,
    pub query_latency_us: u64,
    pub lanes_used: Vec<Lane>,
}

impl MemoryStore {
    /// Three-lane fusion: BM25 + trigram (+ vector when enriched).
    pub fn query(&self, q: &MemoryQuery) -> MemoryQueryResult {
        let start = Instant::now();
        let k = q.k.max(1);
        let mut scores: HashMap<u64, f32> = HashMap::new();
        let mut lanes_used = Vec::new();

        if q.lanes.bm25 {
            lanes_used.push(Lane::Bm25);
            for (doc_id, score) in self.search_bm25(&q.namespace, &q.query, k * 2) {
                *scores.entry(doc_id).or_default() += score * q.lanes.bm25_weight;
            }
        }

        if q.lanes.trigram {
            lanes_used.push(Lane::Trigram);
            for (doc_id, score) in self.search_trigram_literal(&q.namespace, &q.query, k * 2) {
                *scores.entry(doc_id).or_default() += score * q.lanes.trigram_weight;
            }
        }

        if q.lanes.vector {
            lanes_used.push(Lane::Vector);
            for (doc_id, score) in self.search_vector(&q.namespace, &q.query, k * 2) {
                *scores.entry(doc_id).or_default() += score * q.lanes.vector_weight;
            }
        }

        let tau = q.as_of.unwrap_or(u64::MAX);
        let trust_cfg = TrustScoreConfig::default();

        let mut ranked: Vec<(u64, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);

        let hits: Vec<MemoryHit> = ranked
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let text = self.episode_text(&q.namespace, doc_id)?;
                let episode = self
                    .get_episode(&q.namespace, crate::episode::EpisodeId(doc_id))
                    .ok()?;
                let snippet: String = text.chars().take(256).collect();
                let provenance = ProvenanceBundle {
                    episode_id: doc_id,
                    t_valid_from: episode.t_valid_from,
                    t_valid_to: if tau < u64::MAX { tau } else { u64::MAX },
                    trust: TrustScore::compute(&trust_cfg, 1, episode.t_created, 0),
                };
                Some(MemoryHit {
                    doc_id,
                    score,
                    lane: Lane::Bm25,
                    snippet,
                    provenance,
                })
            })
            .collect();

        MemoryQueryResult {
            hits,
            query_latency_us: start.elapsed().as_micros() as u64,
            lanes_used,
        }
    }
}
