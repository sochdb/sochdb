use parking_lot::Mutex;
use std::collections::VecDeque;
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

/// Bounded async enrichment queue (embedding + fact extraction).
pub struct EnrichmentQueue {
    max_depth: usize,
    pending: Mutex<VecDeque<EnrichmentJob>>,
    processed: Mutex<u64>,
}

impl EnrichmentQueue {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            pending: Mutex::new(VecDeque::new()),
            processed: Mutex::new(0),
        }
    }

    pub fn depth(&self) -> usize {
        self.pending.lock().len()
    }

    pub fn processed_count(&self) -> u64 {
        *self.processed.lock()
    }

    pub fn try_enqueue(&self, job: EnrichmentJob) -> Result<(), EnrichmentError> {
        let mut q = self.pending.lock();
        if q.len() >= self.max_depth {
            return Err(EnrichmentError::QueueFull(self.max_depth));
        }
        q.push_back(job);
        Ok(())
    }

    pub fn pop(&self) -> Option<EnrichmentJob> {
        self.pending.lock().pop_front()
    }

    pub fn mark_processed(&self) {
        *self.processed.lock() += 1;
    }
}
