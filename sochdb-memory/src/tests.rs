#[cfg(test)]
mod tests {
    use crate::{EpisodeWrite, MemoryQuery, MemoryStore, MemoryStoreConfig, QueryLanes};

    /// RRF fusion: the doc that matches across lanes (bm25 + trigram) must rank
    /// first, and fused scores are the small rank-based RRF values (not raw
    /// per-lane magnitudes). Locks the fix that replaced the un-normalized
    /// weighted-sum (where unbounded BM25 scores dominated).
    #[test]
    fn rrf_fusion_ranks_best_match_first() {
        let store = MemoryStore::with_defaults();
        for text in [
            "Caroline joined the LGBTQ support group in May",
            "the weather was sunny on the beach yesterday",
            "quarterly revenue report shows strong growth",
        ] {
            store
                .write_episode(EpisodeWrite {
                    namespace: "t".into(),
                    text: text.into(),
                    t_valid_from: None,
                    metadata: None,
                })
                .unwrap();
        }
        let r = store.query(&MemoryQuery {
            namespace: "t".into(),
            query: "LGBTQ support group".into(),
            as_of: None,
            lanes: QueryLanes::lexical_only(),
            k: 3,
        });
        assert!(!r.hits.is_empty(), "RRF fusion returned no hits");
        assert!(
            r.hits[0].snippet.contains("LGBTQ"),
            "best multi-lane match must rank first under RRF, got: {}",
            r.hits[0].snippet
        );
        // RRF contributions are weight / (60 + rank): strictly positive, well below 1.
        assert!(
            r.hits[0].score > 0.0 && r.hits[0].score < 1.0,
            "RRF score outside expected range: {}",
            r.hits[0].score
        );
    }

    /// write_turns groups N turns per episode and prefixes each with its speaker
    /// — the ingestion shape proven to maximize recall (vs one bare turn per
    /// episode). 6 turns @ window=3 => 2 episodes; each carries speaker-prefixed
    /// lines for its turns and is retrievable.
    #[test]
    fn write_turns_windows_and_prefixes() {
        let store = MemoryStore::with_defaults();
        let turns: Vec<crate::episode::ConversationTurn> = (0..6)
            .map(|i| crate::episode::ConversationTurn {
                speaker: if i % 2 == 0 { "Alice" } else { "Bob" }.into(),
                text: format!("message number {i}"),
            })
            .collect();

        let results = store.write_turns("conv", &turns, 3, None).unwrap();
        assert_eq!(results.len(), 2, "6 turns @ window=3 should be 2 episodes");

        let ep0 = store.episode_text("conv", results[0].episode_id.0).unwrap();
        assert!(ep0.contains("Alice: message number 0"), "speaker prefix + turn 0");
        assert!(ep0.contains("Bob: message number 1"), "turn 1 grouped in");
        assert!(ep0.contains("message number 2"), "turn 2 grouped in");

        let r = store.query(&MemoryQuery {
            namespace: "conv".into(),
            query: "message number 4".into(),
            as_of: None,
            lanes: QueryLanes::lexical_only(),
            k: 5,
        });
        assert!(!r.hits.is_empty(), "windowed episode must be retrievable");
    }

    #[test]
    fn write_time_lexical_recall() {
        let store = MemoryStore::with_defaults();
        let wr = store
            .write_episode(EpisodeWrite {
                namespace: "test".into(),
                text: "Caroline went to the LGBTQ support group on 7 May 2023".into(),
                t_valid_from: None,
                metadata: None,
            })
            .unwrap();
        assert!(wr.lexical_indexed);
        assert!(wr.ingestion_lag_us < 1_000_000);

        let result = store.query(&MemoryQuery {
            namespace: "test".into(),
            query: "LGBTQ support group".into(),
            as_of: None,
            lanes: QueryLanes::lexical_only(),
            k: 5,
        });
        assert!(!result.hits.is_empty());
    }

    /// Deterministic test embedder that reports `is_semantic() == true` so the
    /// vector lane runs — lets us exercise the semantic-gated hybrid path without
    /// the heavy fastembed ONNX model. (Embed quality is irrelevant here; the
    /// lexical lanes supply the hits, this just makes the vector lane active.)
    struct SemanticTestEmbedder(sochdb_query::MockEmbeddingProvider);
    impl sochdb_query::EmbeddingProvider for SemanticTestEmbedder {
        fn model_name(&self) -> &str {
            "test-semantic"
        }
        fn dimension(&self) -> usize {
            self.0.dimension()
        }
        fn max_length(&self) -> usize {
            self.0.max_length()
        }
        fn is_semantic(&self) -> bool {
            true
        }
        fn embed(
            &self,
            text: &str,
        ) -> sochdb_query::embedding_provider::EmbeddingResult<Vec<f32>> {
            self.0.embed(text)
        }
    }

    #[test]
    fn enrichment_enables_vector_lane() {
        let store = MemoryStore::with_embedder(
            None,
            MemoryStoreConfig {
                enrich_on_write: true,
                ..MemoryStoreConfig::default()
            },
            std::sync::Arc::new(SemanticTestEmbedder(
                sochdb_query::MockEmbeddingProvider::new(384),
            )),
        );

        store
            .write_episode(EpisodeWrite {
                namespace: "vec-ns".into(),
                text: "The patient underwent cardiac surgery in Boston on March 12".into(),
                t_valid_from: None,
                metadata: None,
            })
            .unwrap();

        assert_eq!(store.enriched_episode_count("vec-ns"), 1);

        let result = store.query(&MemoryQuery {
            namespace: "vec-ns".into(),
            query: "cardiac surgery Boston".into(),
            as_of: None,
            lanes: QueryLanes::three_lane(),
            k: 5,
        });

        // Semantic embedder + enrichment -> vector lane is active.
        assert!(result.lanes_used.contains(&crate::Lane::Vector));
        assert!(!result.hits.is_empty());
    }

    /// The gate: with a NON-semantic (mock) embedder, three_lane must auto-skip
    /// the vector lane so mock-cosine noise never enters fusion — this is what
    /// makes defaulting to three_lane safe.
    #[test]
    fn mock_embedder_skips_vector_lane() {
        let store = MemoryStore::with_embedder(
            None,
            MemoryStoreConfig {
                enrich_on_write: true,
                ..MemoryStoreConfig::default()
            },
            std::sync::Arc::new(sochdb_query::MockEmbeddingProvider::new(384)),
        );
        store
            .write_episode(EpisodeWrite {
                namespace: "m".into(),
                text: "cardiac surgery in Boston".into(),
                t_valid_from: None,
                metadata: None,
            })
            .unwrap();
        let result = store.query(&MemoryQuery {
            namespace: "m".into(),
            query: "cardiac surgery".into(),
            as_of: None,
            lanes: QueryLanes::three_lane(),
            k: 5,
        });
        // Vector lane suppressed; lexical lanes still return the hit.
        assert!(!result.lanes_used.contains(&crate::Lane::Vector));
        assert!(!result.hits.is_empty());
    }

    #[test]
    fn drain_enrichment_queue_indexes_vectors() {
        let store = MemoryStore::with_defaults();
        store
            .write_episode(EpisodeWrite {
                namespace: "async-ns".into(),
                text: "Melanie adopted a rescue dog named Biscuit".into(),
                t_valid_from: None,
                metadata: None,
            })
            .unwrap();

        assert_eq!(store.enriched_episode_count("async-ns"), 0);
        assert_eq!(store.drain_enrichment_queue(), 1);
        assert_eq!(store.enriched_episode_count("async-ns"), 1);

        let vector_hits = store.search_vector("async-ns", "rescue dog Biscuit", 5);
        assert!(!vector_hits.is_empty());
    }
}
