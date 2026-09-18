use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct EnrichmentJob {
    pub namespace: String,
    pub episode_id: u64,
    pub text: String,
}

#[derive(Debug, Error)]
pub enum EnrichmentError {
    #[error("enrichment queue full (max {0})")]
    QueueFull(usize),
}

#[derive(Debug, Clone)]
pub struct EnrichmentQueueConfig {
    pub max_depth: usize,
}

/// Pending jobs plus the per-namespace counts that make a namespace-scoped
/// lookup O(1).
///
/// The counts live inside the same lock as the jobs because a count that can
/// disagree with the queue is worse than no count at all: a reader would either
/// skip work that is genuinely pending or wait for work that is not.
#[derive(Debug, Default)]
struct Pending {
    jobs: VecDeque<EnrichmentJob>,
    per_namespace: HashMap<String, usize>,
}

impl Pending {
    fn push(&mut self, job: EnrichmentJob) {
        *self.per_namespace.entry(job.namespace.clone()).or_insert(0) += 1;
        self.jobs.push_back(job);
    }

    fn note_removed(&mut self, namespace: &str) {
        if let Some(count) = self.per_namespace.get_mut(namespace) {
            *count -= 1;
            if *count == 0 {
                self.per_namespace.remove(namespace);
            }
        }
    }
}

/// Bounded async enrichment queue (embedding + fact extraction).
pub struct EnrichmentQueue {
    max_depth: usize,
    pending: Mutex<Pending>,
    processed: Mutex<u64>,
}

impl EnrichmentQueue {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            pending: Mutex::new(Pending::default()),
            processed: Mutex::new(0),
        }
    }

    pub fn depth(&self) -> usize {
        self.pending.lock().jobs.len()
    }

    /// Pending jobs for one namespace.
    ///
    /// This is the number a reader of that namespace has to care about. The
    /// total depth is not: it counts work for namespaces the reader will never
    /// look at, and treating that as a reason to wait is what lets one busy
    /// tenant add latency to every other tenant's queries.
    pub fn depth_for(&self, namespace: &str) -> usize {
        self.pending
            .lock()
            .per_namespace
            .get(namespace)
            .copied()
            .unwrap_or(0)
    }

    pub fn processed_count(&self) -> u64 {
        *self.processed.lock()
    }

    pub fn try_enqueue(&self, job: EnrichmentJob) -> Result<(), EnrichmentError> {
        let mut pending = self.pending.lock();
        if pending.jobs.len() >= self.max_depth {
            return Err(EnrichmentError::QueueFull(self.max_depth));
        }
        pending.push(job);
        Ok(())
    }

    pub fn pop(&self) -> Option<EnrichmentJob> {
        let mut pending = self.pending.lock();
        let job = pending.jobs.pop_front()?;
        pending.note_removed(&job.namespace);
        Some(job)
    }

    /// Remove and return every pending job for one namespace, in enqueue order,
    /// leaving every other namespace's work queued.
    ///
    /// The jobs are removed under the lock, so the background worker cannot
    /// also claim them: each job is enriched once, by whichever side took it.
    pub fn take_namespace(&self, namespace: &str) -> Vec<EnrichmentJob> {
        let mut pending = self.pending.lock();
        if !pending.per_namespace.contains_key(namespace) {
            return Vec::new();
        }
        let mut taken = Vec::new();
        let mut kept = VecDeque::with_capacity(pending.jobs.len());
        for job in pending.jobs.drain(..) {
            if job.namespace == namespace {
                taken.push(job);
            } else {
                kept.push_back(job);
            }
        }
        pending.jobs = kept;
        pending.per_namespace.remove(namespace);
        taken
    }

    pub fn mark_processed(&self) {
        *self.processed.lock() += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(namespace: &str, episode_id: u64) -> EnrichmentJob {
        EnrichmentJob {
            namespace: namespace.to_string(),
            episode_id,
            text: format!("episode {episode_id}"),
        }
    }

    #[test]
    fn take_namespace_leaves_other_namespaces_queued() {
        let q = EnrichmentQueue::new(16);
        q.try_enqueue(job("a", 1)).unwrap();
        q.try_enqueue(job("b", 2)).unwrap();
        q.try_enqueue(job("a", 3)).unwrap();
        q.try_enqueue(job("c", 4)).unwrap();

        let taken = q.take_namespace("a");
        assert_eq!(
            taken.iter().map(|j| j.episode_id).collect::<Vec<_>>(),
            vec![1, 3],
            "namespace jobs come back in enqueue order"
        );
        assert_eq!(q.depth(), 2, "b and c are untouched");
        assert_eq!(q.depth_for("a"), 0);
        assert_eq!(q.depth_for("b"), 1);
        assert_eq!(q.depth_for("c"), 1);

        // The retained jobs keep their relative order across the extraction.
        assert_eq!(q.pop().unwrap().episode_id, 2);
        assert_eq!(q.pop().unwrap().episode_id, 4);
        assert!(q.pop().is_none());
    }

    #[test]
    fn take_namespace_of_unknown_namespace_is_empty_and_harmless() {
        let q = EnrichmentQueue::new(16);
        q.try_enqueue(job("a", 1)).unwrap();
        assert!(q.take_namespace("missing").is_empty());
        assert_eq!(q.depth(), 1);
        assert_eq!(q.depth_for("a"), 1);
    }

    #[test]
    fn pop_keeps_per_namespace_counts_honest() {
        let q = EnrichmentQueue::new(16);
        q.try_enqueue(job("a", 1)).unwrap();
        q.try_enqueue(job("a", 2)).unwrap();
        assert_eq!(q.depth_for("a"), 2);
        q.pop().unwrap();
        assert_eq!(q.depth_for("a"), 1);
        q.pop().unwrap();
        assert_eq!(q.depth_for("a"), 0);
        assert_eq!(q.depth(), 0);
    }

    #[test]
    fn enqueue_past_max_depth_is_refused_without_counting_it() {
        let q = EnrichmentQueue::new(2);
        q.try_enqueue(job("a", 1)).unwrap();
        q.try_enqueue(job("a", 2)).unwrap();
        assert!(q.try_enqueue(job("a", 3)).is_err());
        assert_eq!(q.depth_for("a"), 2);
    }
}
