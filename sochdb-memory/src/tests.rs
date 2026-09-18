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

        // Disjoint windows: 6 turns @ window=3 stride=3 -> 2 episodes.
        let results = store.write_turns("conv", &turns, 3, 3, None).unwrap();
        assert_eq!(
            results.len(),
            2,
            "6 turns @ window=3 stride=3 should be 2 episodes"
        );

        let ep0 = store.episode_text("conv", results[0].episode_id.0).unwrap();
        assert!(
            ep0.contains("Alice: message number 0"),
            "speaker prefix + turn 0"
        );
        assert!(ep0.contains("Bob: message number 1"), "turn 1 grouped in");
        assert!(ep0.contains("message number 2"), "turn 2 grouped in");

        // Overlapping windows: window=3 stride=2 -> starts at 0,2,4 -> 3 episodes.
        let overlapped = store.write_turns("conv2", &turns, 3, 2, None).unwrap();
        assert_eq!(
            overlapped.len(),
            3,
            "window=3 stride=2 over 6 turns should be 3 episodes"
        );

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
        fn embed(&self, text: &str) -> sochdb_query::embedding_provider::EmbeddingResult<Vec<f32>> {
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
        )
        .expect("an in-memory store opens no files and cannot fail");

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
        )
        .expect("an in-memory store opens no files and cannot fail");
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

    /// A reader of one namespace must not be made to do another namespace's
    /// enrichment. Draining the whole queue instead let one busy tenant's
    /// backlog become every other tenant's query latency.
    #[test]
    fn enrichment_waits_are_scoped_to_the_queried_namespace() {
        let store = MemoryStore::with_defaults();
        for ns in ["busy", "busy", "busy", "quiet"] {
            store
                .write_episode(EpisodeWrite {
                    namespace: ns.into(),
                    text: format!("an episode belonging to {ns}"),
                    t_valid_from: None,
                    metadata: None,
                })
                .unwrap();
        }
        assert_eq!(store.enrichment_queue().depth(), 4);
        assert_eq!(store.enrichment_queue().depth_for("busy"), 3);

        assert_eq!(store.drain_enrichment_for("quiet"), 1);

        assert_eq!(
            store.enriched_episode_count("quiet"),
            1,
            "the queried namespace is fresh"
        );
        assert_eq!(
            store.enriched_episode_count("busy"),
            0,
            "the unrelated namespace's work was not done on this query's time"
        );
        assert_eq!(
            store.enrichment_queue().depth(),
            3,
            "and it is still queued for the worker"
        );
    }

    /// The store-wide lock must not be held across namespace mutation: two
    /// agents writing to different namespaces share nothing but the directory.
    #[test]
    fn concurrent_writes_to_different_namespaces_do_not_lose_episodes() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(MemoryStore::with_defaults());
        let mut handles = Vec::new();
        for agent in 0..8u32 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..50u32 {
                    store
                        .write_episode(EpisodeWrite {
                            namespace: format!("agent-{agent}"),
                            text: format!("agent {agent} step {i}"),
                            t_valid_from: None,
                            metadata: None,
                        })
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        for agent in 0..8u32 {
            assert_eq!(
                store.episode_count(&format!("agent-{agent}")),
                50,
                "agent-{agent} lost writes"
            );
        }
    }

    /// Concurrent first writes to the *same* namespace must not produce two
    /// sets of indexes: the read-then-write lookup has to settle on one.
    #[test]
    fn concurrent_creation_of_one_namespace_yields_one_index_set() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let store = Arc::new(MemoryStore::with_defaults());
        let go = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for i in 0..8u32 {
            let store = Arc::clone(&store);
            let go = Arc::clone(&go);
            handles.push(thread::spawn(move || {
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                store
                    .write_episode(EpisodeWrite {
                        namespace: "shared".into(),
                        text: format!("racing write {i}"),
                        t_valid_from: None,
                        metadata: None,
                    })
                    .unwrap()
            }));
        }
        go.store(true, Ordering::Release);

        let mut ids: Vec<u64> = handles
            .into_iter()
            .map(|h| h.join().unwrap().episode_id.0)
            .collect();
        ids.sort_unstable();

        assert_eq!(store.episode_count("shared"), 8);
        assert_eq!(
            ids,
            (1..=8).collect::<Vec<u64>>(),
            "episode ids must come from a single counter"
        );
    }
}

/// Recovery: what a restart must and must not lose.
///
/// Every test here restarts the store the way a crash does — drop the whole
/// `MemoryStore`, build a new one over the same directory — rather than calling
/// a recovery entry point directly. A recovery path that is only ever exercised
/// through a test-only door is a recovery path that has never been shown to run
/// on the code the operator actually restarts.
#[cfg(test)]
mod durability_tests {
    use crate::episode::EpisodeWrite;
    use crate::fact::{FactEdge, FactId, FactKind};
    use crate::store::{MemoryStore, MemoryStoreConfig};
    use crate::{Durability, MemoryQuery, QueryLanes};
    use sochdb_core::knowledge_object::BitemporalCoord;
    use std::path::Path;
    use tempfile::TempDir;

    fn durable_config() -> MemoryStoreConfig {
        MemoryStoreConfig {
            durability: Durability::Sync,
            ..MemoryStoreConfig::default()
        }
    }

    fn open(dir: &Path) -> MemoryStore {
        MemoryStore::new(Some(dir), durable_config()).expect("store opens")
    }

    fn write(store: &MemoryStore, ns: &str, text: &str) -> u64 {
        store
            .write_episode(EpisodeWrite {
                namespace: ns.to_string(),
                text: text.to_string(),
                t_valid_from: Some(1),
                metadata: None,
            })
            .expect("write succeeds")
            .episode_id
            .0
    }

    fn fact(episode_id: u64, subject: &str, object: &str) -> FactEdge {
        FactEdge {
            id: FactId(0),
            episode_id,
            subject: subject.to_string(),
            predicate: "likes".to_string(),
            object: object.to_string(),
            kind: FactKind::UserAsserted,
            temporal: BitemporalCoord::new(1, 1),
        }
    }

    #[test]
    fn episodes_written_before_a_restart_are_still_there_after_it() {
        let dir = TempDir::new().unwrap();

        let ids: Vec<u64> = {
            let store = open(dir.path());
            let ids = vec![
                write(&store, "agent-a", "the deployment failed at 03:00"),
                write(&store, "agent-a", "rollback completed successfully"),
                write(&store, "agent-b", "unrelated tenant episode"),
            ];
            assert_eq!(store.episode_count("agent-a"), 2);
            ids
        };

        let store = open(dir.path());
        assert_eq!(store.episode_count("agent-a"), 2);
        assert_eq!(store.episode_count("agent-b"), 1);
        assert_eq!(
            store.episode_text("agent-a", ids[0]).as_deref(),
            Some("the deployment failed at 03:00")
        );
        assert_eq!(
            store.episode_text("agent-b", ids[2]).as_deref(),
            Some("unrelated tenant episode")
        );
    }

    #[test]
    fn a_recovered_episode_keeps_the_id_its_client_was_given() {
        let dir = TempDir::new().unwrap();
        let handed_out = {
            let store = open(dir.path());
            for i in 0..5 {
                write(&store, "ns", &format!("episode number {i}"));
            }
            write(&store, "ns", "the one the client remembers")
        };

        let store = open(dir.path());
        assert_eq!(
            store.episode_text("ns", handed_out).as_deref(),
            Some("the one the client remembers"),
            "an id a client stored must still resolve after recovery"
        );
    }

    #[test]
    fn ids_issued_after_recovery_do_not_collide_with_recovered_ones() {
        let dir = TempDir::new().unwrap();
        let before: Vec<u64> = {
            let store = open(dir.path());
            (0..4)
                .map(|i| write(&store, "ns", &format!("old {i}")))
                .collect()
        };

        let store = open(dir.path());
        let after: Vec<u64> = (0..4)
            .map(|i| write(&store, "ns", &format!("new {i}")))
            .collect();

        for id in &after {
            assert!(
                !before.contains(id),
                "id {id} was reissued after recovery and now names two episodes"
            );
        }
        assert_eq!(store.episode_count("ns"), 8);
    }

    #[test]
    fn lexical_lanes_are_rebuilt_so_recovered_episodes_are_searchable() {
        let dir = TempDir::new().unwrap();
        {
            let store = open(dir.path());
            write(&store, "ns", "postgres connection pool exhausted");
            write(&store, "ns", "redis eviction policy changed");
        }

        let store = open(dir.path());

        let bm25 = store.search_bm25("ns", "postgres connection", 5);
        assert!(
            !bm25.is_empty(),
            "BM25 postings were not rebuilt: a recovered episode is unreachable"
        );

        let trigram = store.search_trigram_literal("ns", "eviction", 5);
        assert!(
            !trigram.is_empty(),
            "trigram postings were not rebuilt: literal search is broken after restart"
        );
    }

    #[test]
    fn a_full_query_works_against_a_store_that_was_only_ever_recovered() {
        let dir = TempDir::new().unwrap();
        {
            let store = open(dir.path());
            write(&store, "ns", "the incident was caused by a bad migration");
        }

        let store = open(dir.path());
        let result = store.query(&MemoryQuery {
            namespace: "ns".to_string(),
            query: "bad migration".to_string(),
            k: 5,
            as_of: Some(u64::MAX),
            lanes: QueryLanes::lexical_only(),
        });
        assert!(
            !result.hits.is_empty(),
            "end-to-end query returned nothing against a recovered store"
        );
    }

    #[test]
    fn facts_and_their_invalidations_both_survive_a_restart() {
        let dir = TempDir::new().unwrap();
        let (kept, retracted) = {
            let store = open(dir.path());
            let ep = write(&store, "ns", "source episode");
            let kept = store.add_fact("ns", fact(ep, "ana", "coffee")).unwrap();
            let retracted = store.add_fact("ns", fact(ep, "ana", "tea")).unwrap();
            assert!(store.invalidate_fact("ns", retracted, 10).unwrap());
            (kept, retracted)
        };

        let store = open(dir.path());
        let valid_later: Vec<FactId> = store
            .facts_valid_at("ns", 50)
            .into_iter()
            .map(|f| f.id)
            .collect();

        assert!(
            valid_later.contains(&kept),
            "a live fact was lost across restart"
        );
        assert!(
            !valid_later.contains(&retracted),
            "a retracted fact came back to life after restart"
        );
    }

    #[test]
    fn recovered_episodes_are_re_queued_for_the_embedding_they_never_logged() {
        let dir = TempDir::new().unwrap();
        {
            let store = open(dir.path());
            write(&store, "ns", "needs an embedding");
            write(&store, "ns", "also needs an embedding");
            store.drain_enrichment_queue();
            assert_eq!(store.enriched_episode_count("ns"), 2);
        }

        let store = open(dir.path());
        assert_eq!(
            store.enriched_episode_count("ns"),
            0,
            "vectors are not logged, so none should exist before re-enrichment"
        );
        assert_eq!(
            store.enrichment_queue().depth_for("ns"),
            2,
            "recovery must re-queue enrichment or the vector lane stays empty forever"
        );

        store.drain_enrichment_queue();
        assert_eq!(
            store.enriched_episode_count("ns"),
            2,
            "re-queued enrichment did not restore the vector lane"
        );
    }

    #[test]
    fn a_torn_trailing_record_costs_only_that_record() {
        use std::io::Write;

        let dir = TempDir::new().unwrap();
        {
            let store = open(dir.path());
            write(&store, "ns", "first survives");
            write(&store, "ns", "second survives");
        }

        // Simulate a crash midway through appending a third record.
        let wal = dir.path().join("memory.wal");
        let mut f = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
        f.write_all(&[0xFF; 9]).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let store = open(dir.path());
        assert_eq!(
            store.episode_count("ns"),
            2,
            "a partial tail must not cost records that were fully written"
        );
    }

    #[test]
    fn asking_for_durability_without_a_directory_is_refused_not_ignored() {
        for tier in [Durability::Buffered, Durability::Sync] {
            let outcome = MemoryStore::new(
                None,
                MemoryStoreConfig {
                    durability: tier,
                    ..MemoryStoreConfig::default()
                },
            );
            let Err(err) = outcome else {
                panic!("durability {tier:?} with nowhere to log must not be accepted");
            };
            assert!(
                err.to_string().contains("requires a data directory"),
                "error must say why: {err}"
            );
        }
    }

    #[test]
    fn a_store_without_a_directory_reports_that_it_is_not_durable() {
        let store = MemoryStore::with_defaults();
        assert!(!store.is_durable());
        assert!(store.sync().is_ok(), "sync is a no-op, not an error");

        let dir = TempDir::new().unwrap();
        assert!(open(dir.path()).is_durable());
    }

    #[test]
    fn buffered_durability_survives_the_process_dying_without_a_clean_shutdown() {
        let dir = TempDir::new().unwrap();
        let config = MemoryStoreConfig {
            durability: Durability::Buffered,
            ..MemoryStoreConfig::default()
        };

        {
            let store = MemoryStore::new(Some(dir.path()), config.clone()).unwrap();
            write(&store, "ns", "buffered write reaches the kernel");
            // Deliberately NOT calling sync(), and leaking the store so its
            // Drop never runs — which is what `kill -9` does.
            std::mem::forget(store);
        }

        let store = MemoryStore::new(Some(dir.path()), config).unwrap();
        assert_eq!(
            store.episode_count("ns"),
            1,
            "Buffered promises survival of process death; the record was still in userspace"
        );
    }

    #[test]
    fn nothing_is_written_to_disk_when_durability_is_off() {
        let dir = TempDir::new().unwrap();
        {
            let store = MemoryStore::new(
                Some(dir.path()),
                MemoryStoreConfig {
                    durability: Durability::None,
                    ..MemoryStoreConfig::default()
                },
            )
            .unwrap();
            write(&store, "ns", "this is not logged");
        }

        let store = open(dir.path());
        assert_eq!(
            store.episode_count("ns"),
            0,
            "Durability::None must not log, however tempting the open directory is"
        );
    }

    #[test]
    fn concurrent_writers_to_one_namespace_all_survive_a_restart() {
        use std::sync::Arc;

        const THREADS: usize = 8;
        const PER_THREAD: usize = 25;

        let dir = TempDir::new().unwrap();
        let live_ids: Vec<u64> = {
            let store = Arc::new(open(dir.path()));
            let mut handles = Vec::new();
            for t in 0..THREADS {
                let store = Arc::clone(&store);
                handles.push(std::thread::spawn(move || {
                    (0..PER_THREAD)
                        .map(|i| write(&store, "shared", &format!("thread {t} episode {i}")))
                        .collect::<Vec<u64>>()
                }));
            }
            let mut ids: Vec<u64> = handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect();
            ids.sort_unstable();

            let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
            assert_eq!(
                unique.len(),
                THREADS * PER_THREAD,
                "concurrent writers were handed the same episode id"
            );
            assert_eq!(store.episode_count("shared"), THREADS * PER_THREAD);
            ids
        };

        let store = open(dir.path());
        let mut recovered: Vec<u64> = store
            .namespace("shared")
            .unwrap()
            .read()
            .episodes
            .keys()
            .copied()
            .collect();
        recovered.sort_unstable();

        assert_eq!(
            recovered, live_ids,
            "a concurrently written store must recover exactly the ids it acknowledged"
        );

        // Ids issued after recovery must not collide with any of them.
        let next = write(&store, "shared", "written after recovery");
        assert!(
            !live_ids.contains(&next),
            "id {next} was reissued after recovering a concurrently written log"
        );
    }

    /// A real `ENOSPC`, not a mock: `/dev/full` fails every write with
    /// "no space left on device", which is the failure a long-lived deployment
    /// actually meets, since nothing truncates or checkpoints this log.
    #[test]
    fn a_write_that_could_not_be_logged_is_invisible_and_stops_the_store() {
        let dir = TempDir::new().unwrap();
        std::os::unix::fs::symlink("/dev/full", dir.path().join("memory.wal")).unwrap();

        let store = MemoryStore::new(Some(dir.path()), durable_config())
            .expect("a store whose log cannot be written to still opens");

        let failed = store.write_episode(EpisodeWrite {
            namespace: "ns".to_string(),
            text: "this cannot be logged".to_string(),
            t_valid_from: Some(1),
            metadata: None,
        });
        let err = failed.expect_err("an unloggable write must not report success");
        assert!(
            err.to_string().contains("No space left on device"),
            "the caller must be told why: {err}"
        );
        assert_eq!(
            store.episode_count("ns"),
            0,
            "a write that could not be logged must never become visible"
        );

        // Fail-stop. Once the log and memory can disagree, continuing to accept
        // writes only widens the gap, and a failed append is indeterminate:
        // BufWriter retains bytes it could not write, so the record the caller
        // was told had failed may still reach the disk behind a later flush.
        let after = store.write_episode(EpisodeWrite {
            namespace: "ns".to_string(),
            text: "after the failure".to_string(),
            t_valid_from: Some(1),
            metadata: None,
        });
        let after_err = after.expect_err("the store kept accepting writes after its log failed");
        assert!(
            after_err.to_string().contains("no longer accepting writes"),
            "later writes must fail loudly and name the original cause: {after_err}"
        );
        assert_eq!(store.episode_count("ns"), 0);

        std::mem::forget(store);
    }

    #[test]
    fn an_unreadable_durability_setting_is_refused_rather_than_downgraded() {
        // Guards the tier the operator actually asked for: reading `fsync` as
        // `Buffered` hands them page-cache durability under a name that claims
        // to survive power loss.
        let dir = TempDir::new().unwrap();
        temp_env(
            &[
                ("SOCHDB_MEMORY_DIR", Some(dir.path().to_str().unwrap())),
                ("SOCHDB_MEMORY_DURABILITY", Some("fsync")),
            ],
            || {
                let err = MemoryStore::from_env()
                    .err()
                    .expect("an unrecognised durability tier must not be silently defaulted");
                assert!(
                    err.to_string().contains("not one of none, buffered, sync"),
                    "error must name the valid tiers: {err}"
                );
            },
        );
    }

    #[test]
    fn a_named_memory_directory_is_logged_to_by_default() {
        let dir = TempDir::new().unwrap();
        temp_env(
            &[
                ("SOCHDB_MEMORY_DIR", Some(dir.path().to_str().unwrap())),
                ("SOCHDB_MEMORY_DURABILITY", None),
            ],
            || {
                let store = MemoryStore::from_env().expect("opens");
                assert!(
                    store.is_durable(),
                    "setting SOCHDB_MEMORY_DIR must be enough to get persistence"
                );
            },
        );
    }

    #[test]
    fn every_recovered_episode_is_re_queued_even_past_the_live_queue_bound() {
        let dir = TempDir::new().unwrap();
        // A bound far below the number of episodes, as a real deployment's
        // 10_000 is below a real corpus.
        let small_bound = MemoryStoreConfig {
            durability: Durability::Sync,
            max_enrichment_queue: 4,
            ..MemoryStoreConfig::default()
        };

        {
            let store = MemoryStore::new(Some(dir.path()), small_bound.clone()).unwrap();
            for i in 0..20 {
                store
                    .write_episode(EpisodeWrite {
                        namespace: "ns".to_string(),
                        text: format!("episode {i}"),
                        t_valid_from: Some(1),
                        metadata: None,
                    })
                    .expect("write succeeds even when enrichment admission is refused");
                store.drain_enrichment_queue();
            }
        }

        let store = MemoryStore::new(Some(dir.path()), small_bound).unwrap();
        assert_eq!(store.episode_count("ns"), 20);
        assert_eq!(
            store.recovery_report().enrichment_requeued,
            20,
            "the live admission bound must not silently strand recovered episodes              outside the vector lane"
        );

        store.drain_enrichment_queue();
        assert_eq!(
            store.enriched_episode_count("ns"),
            20,
            "every recovered episode must end up back in the vector lane"
        );
    }

    #[test]
    fn a_record_this_build_cannot_decode_fails_recovery_instead_of_being_skipped() {
        use sochdb_storage::{TxnWal, TxnWalEntry};

        let dir = TempDir::new().unwrap();
        {
            let store = open(dir.path());
            write(&store, "ns", "written by this build");
        }

        // A record that is intact on disk — correct framing, valid CRC — but
        // whose payload this build does not understand. That is what a
        // downgrade looks like, not corruption.
        {
            let wal = TxnWal::new(dir.path().join("memory.wal")).unwrap();
            let entry = TxnWalEntry::data(
                0,
                Vec::new(),
                br#"{"SomeVariantFromANewerBuild":{"whatever":1}}"#.to_vec(),
            );
            wal.append_sync(&entry).unwrap();
        }

        let outcome = MemoryStore::new(Some(dir.path()), durable_config());
        let Err(err) = outcome else {
            panic!(
                "recovery must not quietly skip a record it cannot decode: doing so leaves \
                 next_episode_id behind the log and the store reissues live ids"
            );
        };
        assert!(
            err.to_string().contains("could not be decoded"),
            "the operator must be told the log is not understood: {err}"
        );
    }

    #[test]
    fn concurrent_invalidations_leave_memory_and_log_agreeing() {
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let target = {
            let store = Arc::new(open(dir.path()));
            let ep = write(&store, "ns", "source episode");
            let target = store.add_fact("ns", fact(ep, "ana", "coffee")).unwrap();

            // Racing retractions at different times. Appends are serialized by
            // the log's writer lock; if the in-memory apply is ordered
            // independently, the last writer to memory need not be the last
            // writer to the log, and `valid_to` ends up different in each.
            let mut handles = Vec::new();
            for t in 1..=8u64 {
                let store = Arc::clone(&store);
                handles.push(std::thread::spawn(move || {
                    store.invalidate_fact("ns", target, t * 100).unwrap()
                }));
            }
            for h in handles {
                assert!(h.join().unwrap());
            }
            target
        };

        let live: Vec<bool> = (1..=9)
            .map(|t| {
                let s = open(dir.path());
                s.facts_valid_at("ns", t * 100 - 50)
                    .iter()
                    .any(|f| f.id == target)
            })
            .collect();
        let again: Vec<bool> = (1..=9)
            .map(|t| {
                let s = open(dir.path());
                s.facts_valid_at("ns", t * 100 - 50)
                    .iter()
                    .any(|f| f.id == target)
            })
            .collect();

        assert_eq!(
            live, again,
            "recovery must be deterministic: the same log replayed twice disagreed"
        );
    }

    /// Set env vars, run `f`, restore. Serialized against other users of this
    /// helper because the environment is process-global.
    fn temp_env(vars: &[(&str, Option<&str>)], f: impl FnOnce()) {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let saved: Vec<(String, Option<std::ffi::OsString>)> = vars
            .iter()
            .map(|(k, _)| (k.to_string(), std::env::var_os(k)))
            .collect();
        for (k, v) in vars {
            match v {
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
        f();
        for (k, v) in saved {
            match v {
                Some(v) => unsafe { std::env::set_var(&k, v) },
                None => unsafe { std::env::remove_var(&k) },
            }
        }
    }
}
