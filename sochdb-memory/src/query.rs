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

/// Which retrieval lanes a query runs, and how their scores are weighted.
///
/// # Why `Default` is written out
///
/// Deriving it produced a value that was *syntactically* valid and
/// *semantically* inert: every lane `false`, every weight `0.0`. A query built
/// with `..Default::default()` therefore searched nothing and returned an empty
/// result set — not an error, not a warning, just zero hits, which is
/// indistinguishable from a namespace that genuinely holds no match. The failure
/// mode is silent and points at the data rather than at the query, which is the
/// expensive kind to debug.
///
/// The default mirrors [`Self::three_lane`] because that is already the
/// codebase's answer to "the caller expressed no preference": it is what the
/// gRPC backend selects for an unrecognised or absent `lanes` option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLanes {
    pub bm25: bool,
    pub trigram: bool,
    pub vector: bool,
    pub bm25_weight: f32,
    pub trigram_weight: f32,
    pub vector_weight: f32,
}

impl Default for QueryLanes {
    fn default() -> Self {
        Self::three_lane()
    }
}

impl QueryLanes {
    /// Every lane off. The only way to ask for a search that finds nothing.
    ///
    /// Exists so that the inert value remains reachable for callers that build
    /// lanes up field by field, now that it is no longer what `Default` hands
    /// out by accident.
    pub fn none() -> Self {
        Self {
            bm25: false,
            trigram: false,
            vector: false,
            bm25_weight: 0.0,
            trigram_weight: 0.0,
            vector_weight: 0.0,
        }
    }

    /// BM25 + trigram: everything indexed synchronously at write time.
    ///
    /// Unlike [`Self::three_lane`], returns an episode the instant it is
    /// written, because neither lane waits on the enrichment daemon.
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

        // Reciprocal Rank Fusion (RRF): fuse lanes by RANK, not raw score.
        // The raw per-lane scores are incomparable — BM25 is unbounded, cosine
        // is [-1,1], and the trigram lane emits a flat constant — so a weighted
        // SUM of raw scores let BM25 magnitude dominate and made the nominal lane
        // weights meaningless. RRF (weight / (RRF_K + rank)) is scale-invariant:
        // each lane contributes purely by where a doc ranks within that lane, so
        // the lane weights are honored and no lane can drown out another. Lanes
        // return results already sorted best-first, so the enumeration index is
        // the rank. RRF_K=60 is the standard damping constant.
        const RRF_K: f32 = 60.0;
        let rrf = |rank: usize| 1.0 / (RRF_K + rank as f32 + 1.0);

        if q.lanes.bm25 {
            lanes_used.push(Lane::Bm25);
            for (rank, (doc_id, _)) in self
                .search_bm25(&q.namespace, &q.query, k * 2)
                .into_iter()
                .enumerate()
            {
                *scores.entry(doc_id).or_default() += q.lanes.bm25_weight * rrf(rank);
            }
        }

        if q.lanes.trigram {
            lanes_used.push(Lane::Trigram);
            for (rank, (doc_id, _)) in self
                .search_trigram_literal(&q.namespace, &q.query, k * 2)
                .into_iter()
                .enumerate()
            {
                *scores.entry(doc_id).or_default() += q.lanes.trigram_weight * rrf(rank);
            }
        }

        // Vector lane contributes ONLY when the embedder is semantic. Fusing the
        // cosine of a hash/mock embedder would inject noise and silently degrade
        // recall, so a non-semantic embedder makes three_lane gracefully behave
        // as lexical instead — which is what makes defaulting to three_lane safe.
        if q.lanes.vector && self.embedder.is_semantic() {
            lanes_used.push(Lane::Vector);
            for (rank, (doc_id, _)) in self
                .search_vector(&q.namespace, &q.query, k * 2)
                .into_iter()
                .enumerate()
            {
                *scores.entry(doc_id).or_default() += q.lanes.vector_weight * rrf(rank);
            }
        }

        let tau = q.as_of.unwrap_or(u64::MAX);
        let trust_cfg = TrustScoreConfig::default();

        // Bounded selection with a stable tie-break, not sort-everything-then-
        // truncate. `scores` is a `HashMap`, so a comparator that calls equal
        // scores equal let the same query return different documents on
        // different runs depending on hash iteration order.
        let ranked = crate::topk::top_k(scores, k);

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
