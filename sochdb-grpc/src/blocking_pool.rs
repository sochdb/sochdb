// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! # Async Boundary Hardening
//!
//! Provides bounded blocking pools to prevent Tokio executor starvation when
//! the sync-first storage layer interacts with the async gRPC server.
//!
//! ## Problem Statement
//!
//! The codebase commits to sync-first storage (std::fs + std::thread + parking_lot).
//! When gRPC (Tokio/tonic) is exposed, we have a **two-scheduler system**:
//! - Tokio workers: optimized for short non-blocking tasks
//! - Storage: blocking I/O with page faults, fsync stalls, cache misses
//!
//! If blocking storage work runs on Tokio workers → scheduler inversion:
//! - Stalled workers reduce effective concurrency
//! - Tail latency spikes
//! - Potential deadlocks if background tasks depend on same executor
//!
//! ## Solution: Bounded Blocking Pools
//!
//! Separate pools for different workload patterns:
//! - **Request Pool**: Point reads, small writes (low latency priority)
//! - **Compaction Pool**: Sequential scans, merges (throughput priority)
//! - **Checkpoint Pool**: Snapshot creation, WAL archiving (background)
//!
//! ## Queueing Theory
//!
//! Model as M/M/c: bounded pools prevent `c` from being "stolen" by long tasks.
//! Pool sizing: `blocking_threads ≈ 2×cores` for mixed I/O+CPU, capped to
//! prevent memory blowups (each thread has stack + allocator footprint).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use parking_lot::{Condvar, Mutex};

/// Pool type for workload isolation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolType {
    /// Request path: point reads, small writes
    Request,
    /// Compaction/GC: sequential scans, merges
    Compaction,
    /// Checkpoint/backup: snapshot, WAL archiving
    Checkpoint,
}

impl PoolType {
    /// Get recommended pool size for this workload type
    pub fn default_size(&self) -> usize {
        let cores = num_cpus::get();
        match self {
            PoolType::Request => (cores * 2).clamp(4, 64),
            PoolType::Compaction => cores.clamp(2, 16),
            PoolType::Checkpoint => 2.max(cores / 4),
        }
    }

    /// Get queue depth for this pool
    pub fn default_queue_depth(&self) -> usize {
        match self {
            PoolType::Request => 1024,
            PoolType::Compaction => 64,
            PoolType::Checkpoint => 16,
        }
    }
}

/// Configuration for a blocking pool
#[derive(Debug, Clone)]
pub struct BlockingPoolConfig {
    /// Pool type
    pub pool_type: PoolType,
    /// Number of worker threads
    pub num_threads: usize,
    /// Maximum queue depth before backpressure
    pub queue_depth: usize,
    /// Stack size per thread (default 2MB)
    pub stack_size: usize,
    /// Thread name prefix
    pub name_prefix: String,
}

impl BlockingPoolConfig {
    /// Create config for request pool
    pub fn request() -> Self {
        Self {
            pool_type: PoolType::Request,
            num_threads: PoolType::Request.default_size(),
            queue_depth: PoolType::Request.default_queue_depth(),
            stack_size: 2 * 1024 * 1024,
            name_prefix: "sochdb-req".to_string(),
        }
    }

    /// Create config for compaction pool
    pub fn compaction() -> Self {
        Self {
            pool_type: PoolType::Compaction,
            num_threads: PoolType::Compaction.default_size(),
            queue_depth: PoolType::Compaction.default_queue_depth(),
            stack_size: 4 * 1024 * 1024, // Larger for compaction
            name_prefix: "sochdb-compact".to_string(),
        }
    }

    /// Create config for checkpoint pool
    pub fn checkpoint() -> Self {
        Self {
            pool_type: PoolType::Checkpoint,
            num_threads: PoolType::Checkpoint.default_size(),
            queue_depth: PoolType::Checkpoint.default_queue_depth(),
            stack_size: 2 * 1024 * 1024,
            name_prefix: "sochdb-ckpt".to_string(),
        }
    }
}

/// Task to execute on a blocking pool
type BlockingTask = Box<dyn FnOnce() + Send + 'static>;

/// Pool metrics for observability
#[derive(Debug, Default)]
pub struct PoolMetrics {
    /// Total tasks submitted
    pub tasks_submitted: AtomicU64,
    /// Tasks completed successfully
    pub tasks_completed: AtomicU64,
    /// Tasks rejected due to queue full
    pub tasks_rejected: AtomicU64,
    /// Current queue depth
    pub queue_depth: AtomicUsize,
    /// Active workers
    pub active_workers: AtomicUsize,
    /// Total execution time (microseconds)
    pub total_exec_time_us: AtomicU64,
    /// Maximum execution time seen (microseconds)
    pub max_exec_time_us: AtomicU64,
}

impl PoolMetrics {
    /// Record task execution
    pub fn record_execution(&self, duration: Duration) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
        let us = duration.as_micros() as u64;
        self.total_exec_time_us.fetch_add(us, Ordering::Relaxed);
        // Update max (lockless, may miss some updates)
        let _ = self.max_exec_time_us.fetch_max(us, Ordering::Relaxed);
    }

    /// Get average execution time
    pub fn avg_exec_time_us(&self) -> u64 {
        let completed = self.tasks_completed.load(Ordering::Relaxed);
        if completed == 0 {
            return 0;
        }
        self.total_exec_time_us.load(Ordering::Relaxed) / completed
    }
}

/// A bounded blocking thread pool
pub struct BlockingPool {
    config: BlockingPoolConfig,
    sender: Sender<BlockingTask>,
    metrics: Arc<PoolMetrics>,
    shutdown: Arc<(Mutex<bool>, Condvar)>,
}

impl BlockingPool {
    /// Create a new blocking pool
    pub fn new(config: BlockingPoolConfig) -> Self {
        let (sender, receiver) = bounded(config.queue_depth);
        let metrics = Arc::new(PoolMetrics::default());
        let shutdown = Arc::new((Mutex::new(false), Condvar::new()));

        // Spawn worker threads
        for i in 0..config.num_threads {
            let receiver = receiver.clone();
            let metrics = metrics.clone();
            let shutdown = shutdown.clone();
            let name = format!("{}-{}", config.name_prefix, i);

            thread::Builder::new()
                .name(name)
                .stack_size(config.stack_size)
                .spawn(move || {
                    Self::worker_loop(receiver, metrics, shutdown);
                })
                .expect("Failed to spawn blocking pool worker");
        }

        Self {
            config,
            sender,
            metrics,
            shutdown,
        }
    }

    /// Worker thread main loop
    fn worker_loop(
        receiver: Receiver<BlockingTask>,
        metrics: Arc<PoolMetrics>,
        shutdown: Arc<(Mutex<bool>, Condvar)>,
    ) {
        loop {
            // Check shutdown
            {
                let (lock, _) = &*shutdown;
                if *lock.lock() {
                    break;
                }
            }

            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(task) => {
                    metrics.active_workers.fetch_add(1, Ordering::Relaxed);
                    let start = Instant::now();

                    // Execute the task
                    task();

                    let elapsed = start.elapsed();
                    metrics.record_execution(elapsed);
                    metrics.active_workers.fetch_sub(1, Ordering::Relaxed);
                    metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Continue checking for shutdown
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
    }

    /// Submit a task to the pool
    ///
    /// Returns Err if the queue is full (backpressure)
    pub fn try_submit<F>(&self, task: F) -> Result<(), BlockingPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.metrics.tasks_submitted.fetch_add(1, Ordering::Relaxed);

        match self.sender.try_send(Box::new(task)) {
            Ok(()) => {
                self.metrics.queue_depth.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                self.metrics.tasks_rejected.fetch_add(1, Ordering::Relaxed);
                Err(BlockingPoolError::QueueFull)
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(BlockingPoolError::PoolShutdown)
            }
        }
    }

    /// Submit a task and block until queue has space
    pub fn submit<F>(&self, task: F) -> Result<(), BlockingPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.metrics.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        self.metrics.queue_depth.fetch_add(1, Ordering::Relaxed);

        self.sender
            .send(Box::new(task))
            .map_err(|_| BlockingPoolError::PoolShutdown)
    }

    /// Get pool metrics
    pub fn metrics(&self) -> &PoolMetrics {
        &self.metrics
    }

    /// Get pool type
    pub fn pool_type(&self) -> PoolType {
        self.config.pool_type
    }

    /// Shutdown the pool gracefully
    pub fn shutdown(&self) {
        let (lock, cvar) = &*self.shutdown;
        *lock.lock() = true;
        cvar.notify_all();
    }
}

impl Drop for BlockingPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Blocking pool error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockingPoolError {
    /// Queue is full, backpressure required
    QueueFull,
    /// Pool has been shut down
    PoolShutdown,
}

impl std::fmt::Display for BlockingPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockingPoolError::QueueFull => write!(f, "Blocking pool queue is full"),
            BlockingPoolError::PoolShutdown => write!(f, "Blocking pool has been shut down"),
        }
    }
}

impl std::error::Error for BlockingPoolError {}

/// Manager for multiple blocking pools with workload isolation
pub struct BlockingPoolManager {
    request_pool: BlockingPool,
    compaction_pool: BlockingPool,
    checkpoint_pool: BlockingPool,
}

impl BlockingPoolManager {
    /// Create with default configurations
    pub fn new() -> Self {
        Self {
            request_pool: BlockingPool::new(BlockingPoolConfig::request()),
            compaction_pool: BlockingPool::new(BlockingPoolConfig::compaction()),
            checkpoint_pool: BlockingPool::new(BlockingPoolConfig::checkpoint()),
        }
    }

    /// Create with custom configurations
    pub fn with_configs(
        request_config: BlockingPoolConfig,
        compaction_config: BlockingPoolConfig,
        checkpoint_config: BlockingPoolConfig,
    ) -> Self {
        Self {
            request_pool: BlockingPool::new(request_config),
            compaction_pool: BlockingPool::new(compaction_config),
            checkpoint_pool: BlockingPool::new(checkpoint_config),
        }
    }

    /// Get pool by type
    pub fn pool(&self, pool_type: PoolType) -> &BlockingPool {
        match pool_type {
            PoolType::Request => &self.request_pool,
            PoolType::Compaction => &self.compaction_pool,
            PoolType::Checkpoint => &self.checkpoint_pool,
        }
    }

    /// Submit to request pool
    pub fn submit_request<F>(&self, task: F) -> Result<(), BlockingPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.request_pool.try_submit(task)
    }

    /// Submit to compaction pool
    pub fn submit_compaction<F>(&self, task: F) -> Result<(), BlockingPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.compaction_pool.submit(task)
    }

    /// Submit to checkpoint pool
    pub fn submit_checkpoint<F>(&self, task: F) -> Result<(), BlockingPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.checkpoint_pool.submit(task)
    }

    /// Get all pool metrics
    pub fn all_metrics(&self) -> Vec<(PoolType, &PoolMetrics)> {
        vec![
            (PoolType::Request, self.request_pool.metrics()),
            (PoolType::Compaction, self.compaction_pool.metrics()),
            (PoolType::Checkpoint, self.checkpoint_pool.metrics()),
        ]
    }

    /// Shutdown all pools
    pub fn shutdown(&self) {
        self.request_pool.shutdown();
        self.compaction_pool.shutdown();
        self.checkpoint_pool.shutdown();
    }
}

impl Default for BlockingPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Async wrapper for blocking pool operations
/// 
/// Bridges the async gRPC layer with the sync storage layer by spawning
/// blocking work on dedicated pools and returning futures.
#[cfg(feature = "async")]
pub mod async_bridge {
    use super::*;
    use tokio::sync::oneshot;

    /// Execute a blocking operation on the specified pool, returning a future
    pub fn spawn_blocking<F, R>(
        pool: &BlockingPool,
        f: F,
    ) -> impl Future<Output = Result<R, BlockingPoolError>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        let result = pool.try_submit(move || {
            let result = f();
            let _ = tx.send(result);
        });

        async move {
            result?;
            rx.await.map_err(|_| BlockingPoolError::PoolShutdown)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_pool_basic_execution() {
        let pool = BlockingPool::new(BlockingPoolConfig::request());
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let counter = counter.clone();
            pool.submit(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();
        }

        // Wait for completion
        thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_pool_backpressure() {
        let config = BlockingPoolConfig {
            pool_type: PoolType::Request,
            num_threads: 1,
            queue_depth: 2,
            stack_size: 2 * 1024 * 1024,
            name_prefix: "test".to_string(),
        };
        let pool = BlockingPool::new(config);

        // Submit tasks that will block
        for _ in 0..2 {
            pool.try_submit(move || {
                thread::sleep(Duration::from_secs(1));
            })
            .unwrap();
        }

        // This should fail with QueueFull
        let result = pool.try_submit(|| {});
        assert!(matches!(result, Err(BlockingPoolError::QueueFull)));
    }

    #[test]
    fn test_pool_metrics() {
        let pool = BlockingPool::new(BlockingPoolConfig::request());

        pool.submit(|| {
            thread::sleep(Duration::from_millis(10));
        })
        .unwrap();

        thread::sleep(Duration::from_millis(50));

        assert!(pool.metrics().tasks_completed.load(Ordering::Relaxed) >= 1);
        assert!(pool.metrics().total_exec_time_us.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_pool_manager() {
        let manager = BlockingPoolManager::new();

        manager.submit_request(|| {}).unwrap();
        manager.submit_compaction(|| {}).unwrap();
        manager.submit_checkpoint(|| {}).unwrap();

        thread::sleep(Duration::from_millis(50));

        for (pool_type, metrics) in manager.all_metrics() {
            assert!(metrics.tasks_submitted.load(Ordering::Relaxed) >= 1);
        }
    }
}
