use crate::fact::FactEdge;
use crate::store::MemoryStore;
use parking_lot::Mutex;
use sochdb_query::memory_compaction::{
    ExtractiveSummarizer, HierarchicalMemory, MemoryCompactionConfig,
};
use sochdb_query::semantic_triggers::{SemanticTrigger, TriggerIndex};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    pub enrichment_poll_ms: u64,
    pub contradiction_bm25_threshold: f32,
    pub compaction: MemoryCompactionConfig,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            enrichment_poll_ms: 100,
            contradiction_bm25_threshold: 0.3,
            compaction: MemoryCompactionConfig::default(),
        }
    }
}

/// Background daemon: enrichment drain, contradiction pre-filter, compaction.
pub struct MemoryLifecycleDaemon {
    store: Arc<MemoryStore>,
    triggers: Arc<TriggerIndex>,
    compaction: Arc<Mutex<HierarchicalMemory<ExtractiveSummarizer>>>,
    running: Arc<AtomicBool>,
    handle: Mutex<Option<thread::JoinHandle<()>>>,
}

impl MemoryLifecycleDaemon {
    pub fn new(store: Arc<MemoryStore>, config: LifecycleConfig) -> Self {
        Self {
            store,
            triggers: Arc::new(TriggerIndex::new()),
            compaction: Arc::new(Mutex::new(HierarchicalMemory::new(
                config.compaction.clone(),
                Arc::new(ExtractiveSummarizer::default()),
            ))),
            running: Arc::new(AtomicBool::new(false)),
            handle: Mutex::new(None),
        }
    }

    pub fn register_trigger(&self, trigger: SemanticTrigger) {
        let _ = self.triggers.register_trigger(trigger);
    }

    pub fn start(&self, config: &LifecycleConfig) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let store = Arc::clone(&self.store);
        let running = Arc::clone(&self.running);
        let poll = Duration::from_millis(config.enrichment_poll_ms);
        let threshold = config.contradiction_bm25_threshold;

        let handle = thread::spawn(move || {
            // How many jobs one wake may drain before re-checking the shutdown
            // flag and the clock. Bounded so a saturated queue cannot keep this
            // thread inside the inner loop indefinitely.
            const DRAIN_BATCH: usize = 256;

            while running.load(Ordering::SeqCst) {
                let mut processed = 0usize;
                while processed < DRAIN_BATCH && running.load(Ordering::SeqCst) {
                    let Some(job) = store.enrichment_queue().pop() else {
                        break;
                    };
                    // Stage 1: embed episode + index in per-namespace HNSW
                    if let Err(e) = store.enrich_episode(&job) {
                        tracing::warn!(
                            namespace = %job.namespace,
                            episode_id = job.episode_id,
                            "enrichment failed: {e}"
                        );
                    }
                    // Stage 2: cheap lexical overlap pre-filter for contradiction candidates
                    let candidates = store.search_bm25(&job.namespace, &job.text, 8);
                    let _adjacent: Vec<_> = candidates
                        .into_iter()
                        .filter(|(_, score)| *score >= threshold)
                        .collect();
                    // Stage 3: LLM judge would run on |C| candidates only (not wired here)
                    store.enrichment_queue().mark_processed();
                    processed += 1;
                }

                // Yield only when there was nothing to do. Sleeping after every
                // *job* capped enrichment at one job per poll interval -- ten a
                // second by default -- no matter how deep the backlog, against
                // an ingest path that sustains orders of magnitude more. The
                // queue could therefore only grow to its cap and start
                // rejecting writes.
                if processed == 0 {
                    thread::sleep(poll);
                }
            }
        });
        *self.handle.lock() = Some(handle);
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(h) = self.handle.lock().take() {
            let _ = h.join();
        }
    }

    pub fn check_contradiction_candidates(
        &self,
        namespace: &str,
        new_fact_text: &str,
        threshold: f32,
    ) -> Vec<FactEdge> {
        let tau = u64::MAX;
        let facts = self.store.facts_valid_at(namespace, tau);
        let hits = self.store.search_bm25(namespace, new_fact_text, 16);
        let hit_ids: std::collections::HashSet<u64> = hits
            .into_iter()
            .filter(|(_, s)| *s >= threshold)
            .map(|(id, _)| id)
            .collect();
        facts
            .into_iter()
            .filter(|f| {
                hit_ids.contains(&f.episode_id)
                    || f.subject.contains(new_fact_text)
                    || f.object.contains(new_fact_text)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeWrite, MemoryStore};

    /// The daemon used to sleep for the poll interval after *every* job, so a
    /// backlog drained at one job per interval -- ten a second at the default
    /// 100ms -- regardless of depth. A write path that outruns that (and every
    /// realistic one does) could only grow the queue to its cap, after which
    /// enrichment requests are rejected outright.
    ///
    /// 50 jobs at the old rate needs ~5s; anything near the poll interval means
    /// the backlog is being drained in batches rather than one per nap.
    #[test]
    fn a_backlog_drains_in_batches_not_one_job_per_poll_interval() {
        let store = Arc::new(MemoryStore::with_defaults());
        for i in 0..50 {
            store
                .write_episode(EpisodeWrite {
                    namespace: "drain".into(),
                    text: format!("episode number {i} about sourdough and migrations"),
                    t_valid_from: None,
                    metadata: None,
                })
                .unwrap();
        }
        assert_eq!(store.enrichment_queue().depth_for("drain"), 50);

        let daemon = MemoryLifecycleDaemon::new(Arc::clone(&store), LifecycleConfig::default());
        let started = std::time::Instant::now();
        daemon.start(&LifecycleConfig::default());

        while store.enrichment_queue().depth_for("drain") > 0 {
            if started.elapsed() > Duration::from_secs(5) {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let elapsed = started.elapsed();
        daemon.stop();

        assert_eq!(
            store.enrichment_queue().depth_for("drain"),
            0,
            "queue still had work after {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "draining 50 jobs took {elapsed:?}; that is the one-job-per-poll behaviour"
        );
    }
}
