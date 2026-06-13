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
            while running.load(Ordering::SeqCst) {
                if let Some(job) = store.enrichment_queue().pop() {
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
                }
                thread::sleep(poll);
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
