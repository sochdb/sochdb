// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Compaction Isolation
//!
//! This module provides per-shard compaction queues that allow background
//! index rebuilding without blocking readers.
//!
//! # Problem
//!
//! Global compaction blocks all shards:
//! - Lock contention during CSR rebuild
//! - Readers stall waiting for compaction
//! - Unpredictable latency spikes
//!
//! # Solution
//!
//! Per-shard compaction with version-based isolation:
//! 1. Each shard has independent compaction queue
//! 2. Build new CSR/AoSoA in background
//! 3. Atomic pointer swap when ready
//! 4. Old version kept until readers drain (epoch-based reclamation)
//!
//! # Invariants
//!
//! - Readers never block on compaction
//! - Writers queue updates for next compaction cycle
//! - At most one compaction per shard at a time
//! - Compaction preserves search correctness

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};

/// Shard identifier for compaction.
pub type ShardId = u32;

/// Version number for compacted data.
pub type Version = u64;

/// Compaction priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompactionPriority {
    /// Background compaction (lowest priority).
    Background = 0,
    /// Normal compaction.
    Normal = 1,
    /// High priority (e.g., after many deletes).
    High = 2,
    /// Urgent (e.g., space pressure).
    Urgent = 3,
}

/// Compaction task for a shard.
#[derive(Debug, Clone)]
pub struct CompactionTask {
    /// Target shard.
    pub shard_id: ShardId,
    /// Priority level.
    pub priority: CompactionPriority,
    /// Estimated work units.
    pub work_estimate: u64,
    /// Creation time.
    pub created_at: Instant,
    /// Task ID.
    pub task_id: u64,
}

impl CompactionTask {
    /// Create a new compaction task.
    pub fn new(shard_id: ShardId, priority: CompactionPriority) -> Self {
        static TASK_COUNTER: AtomicU64 = AtomicU64::new(0);
        Self {
            shard_id,
            priority,
            work_estimate: 0,
            created_at: Instant::now(),
            task_id: TASK_COUNTER.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Set work estimate.
    pub fn with_work_estimate(mut self, estimate: u64) -> Self {
        self.work_estimate = estimate;
        self
    }

    /// Time waiting in queue.
    pub fn queue_time(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Per-shard compaction state.
pub struct ShardCompactionState {
    /// Shard ID.
    shard_id: ShardId,
    /// Current version.
    current_version: AtomicU64,
    /// Is compaction in progress?
    compacting: AtomicBool,
    /// Pending tasks queue.
    pending_tasks: Mutex<VecDeque<CompactionTask>>,
    /// Active readers on each version.
    reader_counts: RwLock<Vec<(Version, u64)>>,
    /// Last compaction time.
    last_compaction: RwLock<Option<Instant>>,
    /// Statistics.
    stats: CompactionStats,
}

/// Compaction statistics.
#[derive(Debug, Default)]
pub struct CompactionStats {
    /// Total compactions completed.
    pub compactions_completed: AtomicU64,
    /// Total bytes reclaimed.
    pub bytes_reclaimed: AtomicU64,
    /// Total time spent compacting (ms).
    pub compaction_time_ms: AtomicU64,
    /// Maximum queue depth seen.
    pub max_queue_depth: AtomicU64,
}

impl ShardCompactionState {
    /// Create new shard state.
    pub fn new(shard_id: ShardId) -> Self {
        Self {
            shard_id,
            current_version: AtomicU64::new(1),
            compacting: AtomicBool::new(false),
            pending_tasks: Mutex::new(VecDeque::new()),
            reader_counts: RwLock::new(Vec::new()),
            last_compaction: RwLock::new(None),
            stats: CompactionStats::default(),
        }
    }

    /// Get current version.
    pub fn current_version(&self) -> Version {
        self.current_version.load(Ordering::Acquire)
    }

    /// Check if compaction is in progress.
    pub fn is_compacting(&self) -> bool {
        self.compacting.load(Ordering::Acquire)
    }

    /// Try to start compaction (returns false if already compacting).
    pub fn try_start_compaction(&self) -> bool {
        self.compacting
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Finish compaction and bump version.
    pub fn finish_compaction(&self, bytes_reclaimed: u64, duration: Duration) {
        let _new_version = self.current_version.fetch_add(1, Ordering::AcqRel) + 1;
        self.compacting.store(false, Ordering::Release);
        
        self.stats.compactions_completed.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_reclaimed.fetch_add(bytes_reclaimed, Ordering::Relaxed);
        self.stats.compaction_time_ms.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
        
        *self.last_compaction.write().unwrap() = Some(Instant::now());
    }

    /// Queue a compaction task.
    pub fn queue_task(&self, task: CompactionTask) {
        let mut queue = self.pending_tasks.lock().unwrap();
        queue.push_back(task);
        
        let depth = queue.len() as u64;
        let max = self.stats.max_queue_depth.load(Ordering::Relaxed);
        if depth > max {
            self.stats.max_queue_depth.store(depth, Ordering::Relaxed);
        }
    }

    /// Pop highest priority task.
    pub fn pop_task(&self) -> Option<CompactionTask> {
        let mut queue = self.pending_tasks.lock().unwrap();
        
        // Find highest priority task
        if queue.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_priority = queue[0].priority;
        
        for (i, task) in queue.iter().enumerate().skip(1) {
            if task.priority > best_priority {
                best_priority = task.priority;
                best_idx = i;
            }
        }

        Some(queue.remove(best_idx).unwrap())
    }

    /// Get pending task count.
    pub fn pending_count(&self) -> usize {
        self.pending_tasks.lock().unwrap().len()
    }

    /// Register a reader on current version.
    pub fn register_reader(&self) -> ReaderGuard {
        let version = self.current_version();
        
        {
            let mut counts = self.reader_counts.write().unwrap();
            if let Some(entry) = counts.iter_mut().find(|(v, _)| *v == version) {
                entry.1 += 1;
            } else {
                counts.push((version, 1));
            }
        }

        ReaderGuard {
            shard_id: self.shard_id,
            version,
        }
    }

    /// Unregister a reader (called by ReaderGuard drop).
    #[allow(dead_code)]
    fn unregister_reader(&self, version: Version) {
        let mut counts = self.reader_counts.write().unwrap();
        if let Some(entry) = counts.iter_mut().find(|(v, _)| *v == version) {
            entry.1 = entry.1.saturating_sub(1);
        }
        // Clean up old versions with no readers
        counts.retain(|(_, count)| *count > 0);
    }

    /// Check if any readers on old versions.
    pub fn has_old_readers(&self) -> bool {
        let current = self.current_version();
        let counts = self.reader_counts.read().unwrap();
        counts.iter().any(|(v, count)| *v < current && *count > 0)
    }

    /// Get time since last compaction.
    pub fn time_since_compaction(&self) -> Option<Duration> {
        self.last_compaction.read().unwrap().map(|t| t.elapsed())
    }

    /// Get statistics.
    pub fn stats(&self) -> &CompactionStats {
        &self.stats
    }
}

/// Guard that tracks reader lifetime.
pub struct ReaderGuard {
    shard_id: ShardId,
    version: Version,
}

impl ReaderGuard {
    /// Get version this reader is using.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Get shard ID.
    pub fn shard_id(&self) -> ShardId {
        self.shard_id
    }
}

// Note: In production, ReaderGuard would call unregister_reader on drop
// via a reference to ShardCompactionState. Here we show the pattern.

/// Compaction queue manager for multiple shards.
pub struct CompactionQueue {
    /// Per-shard state.
    shards: Vec<Arc<ShardCompactionState>>,
    /// Global shutdown flag.
    shutdown: AtomicBool,
    /// Configuration.
    config: CompactionConfig,
}

/// Compaction configuration.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Maximum concurrent compactions.
    pub max_concurrent: usize,
    /// Minimum interval between compactions per shard.
    pub min_interval: Duration,
    /// Work threshold to trigger compaction.
    pub work_threshold: u64,
    /// Enable background compaction.
    pub background_enabled: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            min_interval: Duration::from_secs(60),
            work_threshold: 10000,
            background_enabled: true,
        }
    }
}

impl CompactionQueue {
    /// Create a new compaction queue.
    pub fn new(num_shards: usize, config: CompactionConfig) -> Self {
        let shards = (0..num_shards)
            .map(|i| Arc::new(ShardCompactionState::new(i as ShardId)))
            .collect();

        Self {
            shards,
            shutdown: AtomicBool::new(false),
            config,
        }
    }

    /// Get shard state.
    pub fn shard(&self, shard_id: ShardId) -> Option<&Arc<ShardCompactionState>> {
        self.shards.get(shard_id as usize)
    }

    /// Schedule compaction for a shard.
    pub fn schedule(&self, shard_id: ShardId, priority: CompactionPriority) -> bool {
        if let Some(shard) = self.shard(shard_id) {
            // Check minimum interval
            if let Some(elapsed) = shard.time_since_compaction() {
                if elapsed < self.config.min_interval && priority < CompactionPriority::Urgent {
                    return false;
                }
            }

            let task = CompactionTask::new(shard_id, priority);
            shard.queue_task(task);
            true
        } else {
            false
        }
    }

    /// Get next task to process (from any shard).
    pub fn next_task(&self) -> Option<(ShardId, CompactionTask)> {
        // Find shard with highest priority task that isn't already compacting
        let mut best: Option<(ShardId, CompactionTask)> = None;
        
        for shard in &self.shards {
            if shard.is_compacting() {
                continue;
            }

            // Peek at next task
            let queue = shard.pending_tasks.lock().unwrap();
            if let Some(task) = queue.front() {
                let dominated = best.as_ref().map_or(false, |(_, best_task)| {
                    task.priority <= best_task.priority
                });
                
                if !dominated {
                    drop(queue);
                    if let Some(task) = shard.pop_task() {
                        best = Some((shard.shard_id, task));
                    }
                }
            }
        }

        best
    }

    /// Count active compactions.
    pub fn active_compactions(&self) -> usize {
        self.shards.iter().filter(|s| s.is_compacting()).count()
    }

    /// Check if more compactions can start.
    pub fn can_start_compaction(&self) -> bool {
        self.active_compactions() < self.config.max_concurrent
    }

    /// Total pending tasks across all shards.
    pub fn total_pending(&self) -> usize {
        self.shards.iter().map(|s| s.pending_count()).sum()
    }

    /// Signal shutdown.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Check if shutdown requested.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Shard that was compacted.
    pub shard_id: ShardId,
    /// New version after compaction.
    pub new_version: Version,
    /// Bytes reclaimed.
    pub bytes_reclaimed: u64,
    /// Duration of compaction.
    pub duration: Duration,
    /// Number of entries merged.
    pub entries_merged: u64,
    /// Success or failure.
    pub success: bool,
}

/// Trait for compaction executor.
pub trait CompactionExecutor: Send + Sync {
    /// Execute compaction for a shard.
    fn compact(&self, shard_id: ShardId) -> CompactionResult;
    
    /// Estimate work for compaction.
    fn estimate_work(&self, shard_id: ShardId) -> u64;
}

/// Simple in-memory compaction executor for testing.
pub struct MockCompactionExecutor {
    /// Simulated compaction time.
    compact_time: Duration,
    /// Simulated bytes reclaimed.
    bytes_per_compact: u64,
}

impl MockCompactionExecutor {
    /// Create a new mock executor.
    pub fn new(compact_time: Duration, bytes_per_compact: u64) -> Self {
        Self {
            compact_time,
            bytes_per_compact,
        }
    }
}

impl CompactionExecutor for MockCompactionExecutor {
    fn compact(&self, shard_id: ShardId) -> CompactionResult {
        // Simulate work
        std::thread::sleep(self.compact_time);

        CompactionResult {
            shard_id,
            new_version: 0, // Caller should update
            bytes_reclaimed: self.bytes_per_compact,
            duration: self.compact_time,
            entries_merged: 100,
            success: true,
        }
    }

    fn estimate_work(&self, _shard_id: ShardId) -> u64 {
        1000
    }
}

/// Background compaction worker.
pub struct CompactionWorker {
    /// Queue to process.
    queue: Arc<CompactionQueue>,
    /// Executor.
    executor: Arc<dyn CompactionExecutor>,
    /// Worker ID.
    worker_id: usize,
}

impl CompactionWorker {
    /// Create a new worker.
    pub fn new(
        queue: Arc<CompactionQueue>,
        executor: Arc<dyn CompactionExecutor>,
        worker_id: usize,
    ) -> Self {
        Self {
            queue,
            executor,
            worker_id,
        }
    }

    /// Run one compaction cycle.
    pub fn run_once(&self) -> Option<CompactionResult> {
        if self.queue.is_shutdown() {
            return None;
        }

        if !self.queue.can_start_compaction() {
            return None;
        }

        let (shard_id, _task) = self.queue.next_task()?;
        let shard = self.queue.shard(shard_id)?;

        if !shard.try_start_compaction() {
            return None;
        }

        let start = Instant::now();
        let mut result = self.executor.compact(shard_id);
        result.duration = start.elapsed();

        shard.finish_compaction(result.bytes_reclaimed, result.duration);
        result.new_version = shard.current_version();

        Some(result)
    }

    /// Get worker ID.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_state_version() {
        let state = ShardCompactionState::new(0);
        assert_eq!(state.current_version(), 1);
        
        state.try_start_compaction();
        state.finish_compaction(1000, Duration::from_millis(10));
        
        assert_eq!(state.current_version(), 2);
    }

    #[test]
    fn test_compaction_lock() {
        let state = ShardCompactionState::new(0);
        
        assert!(!state.is_compacting());
        assert!(state.try_start_compaction());
        assert!(state.is_compacting());
        assert!(!state.try_start_compaction()); // Already compacting
        
        state.finish_compaction(0, Duration::ZERO);
        assert!(!state.is_compacting());
    }

    #[test]
    fn test_task_queue() {
        let state = ShardCompactionState::new(0);

        state.queue_task(CompactionTask::new(0, CompactionPriority::Background));
        state.queue_task(CompactionTask::new(0, CompactionPriority::High));
        state.queue_task(CompactionTask::new(0, CompactionPriority::Normal));

        assert_eq!(state.pending_count(), 3);

        // Should get highest priority first
        let task = state.pop_task().unwrap();
        assert_eq!(task.priority, CompactionPriority::High);
    }

    #[test]
    fn test_compaction_queue() {
        let config = CompactionConfig {
            max_concurrent: 2,
            min_interval: Duration::ZERO,
            ..Default::default()
        };
        let queue = CompactionQueue::new(4, config);

        assert_eq!(queue.num_shards(), 4);
        assert_eq!(queue.active_compactions(), 0);

        queue.schedule(0, CompactionPriority::Normal);
        queue.schedule(1, CompactionPriority::High);

        assert_eq!(queue.total_pending(), 2);
    }

    #[test]
    fn test_next_task_priority() {
        let config = CompactionConfig {
            min_interval: Duration::ZERO,
            ..Default::default()
        };
        let queue = CompactionQueue::new(4, config);

        queue.schedule(0, CompactionPriority::Background);
        queue.schedule(1, CompactionPriority::Urgent);
        queue.schedule(2, CompactionPriority::Normal);

        // Should get urgent first
        let (shard_id, task) = queue.next_task().unwrap();
        assert_eq!(shard_id, 1);
        assert_eq!(task.priority, CompactionPriority::Urgent);
    }

    #[test]
    fn test_concurrent_limit() {
        let config = CompactionConfig {
            max_concurrent: 2,
            min_interval: Duration::ZERO,
            ..Default::default()
        };
        let queue = CompactionQueue::new(4, config);

        // Start two compactions
        queue.shard(0).unwrap().try_start_compaction();
        queue.shard(1).unwrap().try_start_compaction();

        assert_eq!(queue.active_compactions(), 2);
        assert!(!queue.can_start_compaction());

        // Finish one
        queue.shard(0).unwrap().finish_compaction(0, Duration::ZERO);
        assert!(queue.can_start_compaction());
    }

    #[test]
    fn test_reader_guard() {
        let state = ShardCompactionState::new(0);
        
        let guard = state.register_reader();
        assert_eq!(guard.version(), 1);
        assert_eq!(guard.shard_id(), 0);
    }

    #[test]
    fn test_stats_tracking() {
        let state = ShardCompactionState::new(0);

        state.try_start_compaction();
        state.finish_compaction(5000, Duration::from_millis(100));

        let stats = state.stats();
        assert_eq!(stats.compactions_completed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.bytes_reclaimed.load(Ordering::Relaxed), 5000);
        assert!(stats.compaction_time_ms.load(Ordering::Relaxed) >= 100);
    }

    #[test]
    fn test_mock_executor() {
        let executor = MockCompactionExecutor::new(
            Duration::from_millis(10),
            1000,
        );

        let result = executor.compact(0);
        assert!(result.success);
        assert_eq!(result.bytes_reclaimed, 1000);
    }

    #[test]
    fn test_worker_run_once() {
        let config = CompactionConfig {
            min_interval: Duration::ZERO,
            ..Default::default()
        };
        let queue = Arc::new(CompactionQueue::new(4, config));
        let executor = Arc::new(MockCompactionExecutor::new(
            Duration::from_millis(1),
            500,
        ));
        let worker = CompactionWorker::new(queue.clone(), executor, 0);

        // No tasks - should return None
        assert!(worker.run_once().is_none());

        // Add task
        queue.schedule(0, CompactionPriority::Normal);
        
        // Should run compaction
        let result = worker.run_once();
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert_eq!(result.shard_id, 0);
        assert!(result.success);
    }

    #[test]
    fn test_shutdown() {
        let queue = CompactionQueue::new(4, CompactionConfig::default());
        
        assert!(!queue.is_shutdown());
        queue.shutdown();
        assert!(queue.is_shutdown());
    }

    #[test]
    fn test_time_since_compaction() {
        let state = ShardCompactionState::new(0);
        
        // No compaction yet
        assert!(state.time_since_compaction().is_none());

        state.try_start_compaction();
        state.finish_compaction(0, Duration::ZERO);

        // Now should have a time
        let elapsed = state.time_since_compaction().unwrap();
        assert!(elapsed < Duration::from_secs(1));
    }
}
