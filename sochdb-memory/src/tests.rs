#[cfg(test)]
mod tests {
    use crate::{EpisodeWrite, MemoryQuery, MemoryStore, MemoryStoreConfig, QueryLanes};

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

    #[test]
    fn enrichment_enables_vector_lane() {
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

        assert!(result.lanes_used.contains(&crate::Lane::Vector));
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