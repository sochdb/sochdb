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

//! Event-Driven Group Commit Buffer
//!
//! This module implements a proper group commit mechanism with:
//! - Event-driven wait using condition variables (not polling)
//! - Single fsync per batch with durability guarantee
//! - Adaptive batch sizing based on Little's Law
//!
//! ## Algorithm
//!
//! Group Commit Queueing Model:
//!
//! Little's Law: N = λ × W
//!   Where: N = avg number of requests in system
//!          λ = arrival rate (req/sec)
//!          W = avg time in system (sec)
//!
//! Optimal Batch Size: N* = sqrt(2 × L_fsync × λ / C_wait)
//!   Where: L_fsync = fsync latency
//!          C_wait = normalized waiting cost
//!
//! Example: L = 5ms, λ = 1000 req/s, C_wait = 1.0
//!   N* = sqrt(2 × 0.005 × 1000 / 1.0) ≈ 3.16 → 3 commits/batch
//!
//! ## Throughput Analysis
//!
//! Without group commit:
//!   Throughput = 1 / L = 200 commits/sec (for L = 5ms)
//!
//! With group commit (batch size N):
//!   Throughput = N / L = N × 200 commits/sec
//!
//! For N = 100:
//!   Throughput = 20,000 commits/sec
//!   Speedup = 100x

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Pending commit with notification channel
pub struct PendingCommitV2 {
    /// Transaction ID
    pub txn_id: u64,
    /// Enqueue timestamp
    pub enqueue_time: Instant,
    /// Notification channel (oneshot-style via Arc<Condvar>)
    pub notifier: Arc<(Mutex<CommitResult>, Condvar)>,
}

/// A submitted commit must always get an answer, however its flusher dies.
///
/// The committing thread blocks on `notifier` until the result leaves
/// `Pending`. Once flushing moved off that thread and onto a shared flusher,
/// nothing guaranteed anyone would ever write that result: `flush_batch`
/// drains the batch out of the queue before calling the flush callback, so if
/// that callback panics the batch is dropped during unwinding and every waiter
/// in it blocks forever on a condvar no one will signal again. A hung commit is
/// worse than a failed one -- it is indistinguishable from a slow disk.
///
/// Resolving on drop makes the guarantee structural rather than a property of
/// every path through the flusher. A commit that is still `Pending` when its
/// record is destroyed was, by definition, never durably written, so reporting
/// it as an error is also the truthful answer.
impl Drop for PendingCommitV2 {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.notifier;
        // A poisoned lock means the flusher panicked mid-notify. The waiter is
        // stuck either way, so take the guard and answer anyway.
        let mut result = match lock.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if matches!(*result, CommitResult::Pending) {
            *result = CommitResult::Error("group commit flusher dropped the batch".into());
            cvar.notify_all();
        }
    }
}

/// Clears the running flag when the flusher thread leaves, panic included.
struct FlusherExit(Arc<AtomicU64>);

impl Drop for FlusherExit {
    fn drop(&mut self) {
        self.0.store(0, Ordering::SeqCst);
    }
}

/// Result of a commit operation
#[derive(Debug, Clone)]
pub enum CommitResult {
    /// Commit pending (initial state)
    Pending,
    /// Commit succeeded with timestamp
    Success(u64),
    /// Commit failed with error message
    Error(String),
}

/// Event-driven group commit buffer with proper synchronization
#[allow(dead_code)]
pub struct EventDrivenGroupCommit {
    /// Pending commits queue
    pending: Mutex<VecDeque<PendingCommitV2>>,
    /// Signal that new commits are available
    commit_available: Condvar,
    /// Configuration
    config: GroupCommitConfig,
    /// Metrics
    metrics: GroupCommitMetrics,
    /// Flush callback (performs actual WAL fsync)
    #[allow(clippy::type_complexity)]
    flush_fn: Arc<dyn Fn(&[u64]) -> Result<u64, String> + Send + Sync>,
    /// Running flag
    ///
    /// Behind an `Arc` so the flusher thread can clear it on the way out
    /// without holding the whole committer alive; see `start_background`.
    running: Arc<AtomicU64>, // 1 = running, 0 = stopped
    /// Flush thread handle
    flush_thread: Mutex<Option<JoinHandle<()>>>,
}

/// Group commit configuration
#[derive(Clone)]
pub struct GroupCommitConfig {
    /// Minimum batch size before flush
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time before flush (microseconds)
    pub max_wait_us: u64,
    /// Initial fsync latency estimate (microseconds)
    pub fsync_latency_us: u64,
    /// Arrival rate EMA alpha (0.0-1.0)
    pub ema_alpha: f64,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 1000,
            max_wait_us: 10_000,     // 10ms max wait
            fsync_latency_us: 5_000, // 5ms default
            ema_alpha: 0.1,
        }
    }
}

/// Metrics for group commit monitoring
pub struct GroupCommitMetrics {
    /// Current adaptive batch size
    pub adaptive_batch_size: AtomicU64,
    /// Estimated arrival rate (req/s × 1000 for precision)
    pub arrival_rate_ema: AtomicU64,
    /// Estimated fsync latency (microseconds)
    pub fsync_latency_us: AtomicU64,
    /// Total commits processed
    pub total_commits: AtomicU64,
    /// Total batches processed
    pub total_batches: AtomicU64,
    /// Total fsync time (microseconds)
    pub total_fsync_time_us: AtomicU64,
    /// Last arrival timestamp (microseconds since epoch)
    pub last_arrival_us: AtomicU64,
}

impl Default for GroupCommitMetrics {
    fn default() -> Self {
        Self {
            adaptive_batch_size: AtomicU64::new(10),
            arrival_rate_ema: AtomicU64::new(100_000), // 100 req/s initial
            fsync_latency_us: AtomicU64::new(5_000),
            total_commits: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            total_fsync_time_us: AtomicU64::new(0),
            last_arrival_us: AtomicU64::new(0),
        }
    }
}

impl EventDrivenGroupCommit {
    /// Create a new event-driven group commit buffer
    ///
    /// # Arguments
    /// * `flush_fn` - Callback that performs WAL fsync. Takes list of txn_ids, returns commit timestamp.
    pub fn new<F>(flush_fn: F) -> Self
    where
        F: Fn(&[u64]) -> Result<u64, String> + Send + Sync + 'static,
    {
        Self::with_config(flush_fn, GroupCommitConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config<F>(flush_fn: F, config: GroupCommitConfig) -> Self
    where
        F: Fn(&[u64]) -> Result<u64, String> + Send + Sync + 'static,
    {
        let gc = Self {
            pending: Mutex::new(VecDeque::new()),
            commit_available: Condvar::new(),
            config,
            metrics: GroupCommitMetrics::default(),
            flush_fn: Arc::new(flush_fn),
            running: Arc::new(AtomicU64::new(0)),
            flush_thread: Mutex::new(None),
        };

        gc.metrics
            .fsync_latency_us
            .store(gc.config.fsync_latency_us, Ordering::Relaxed);
        gc
    }

    /// Start the background flush thread
    pub fn start(&self) -> Result<(), String> {
        if self
            .running
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err("Already running".into());
        }

        // Marks the committer as running without owning a thread. Callers that
        // hold an `Arc` should prefer `start_background`, which actually runs
        // the loop; this exists for owners that drive `flush_loop` themselves.
        Ok(())
    }

    /// Spawn the background flusher.
    ///
    /// Without a flusher thread `is_running()` is false, and `submit_and_wait`
    /// falls back to flushing inline on the committing thread. Every committer
    /// then issues its own barrier over whatever slice of the queue it happened
    /// to drain, so N concurrent committers produce N small serialized barriers
    /// instead of one that covers them all. With a single flusher, committers
    /// only enqueue and wait, and the barrier in progress becomes the batching
    /// window for everyone who arrives during it.
    ///
    /// The thread holds a `Weak`, not an `Arc`: an `Arc` would keep the group
    /// committer alive forever and the thread would never exit. Each iteration
    /// upgrades, does one step, and drops, so once the owner lets go the upgrade
    /// fails and the loop ends.
    pub fn start_background(self: &Arc<Self>) -> Result<(), String> {
        if self
            .running
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err("Already running".into());
        }

        let weak = Arc::downgrade(self);
        let running = Arc::clone(&self.running);
        match std::thread::Builder::new()
            .name("sochdb-group-commit".into())
            .spawn(move || {
                // Clears `running` however this thread leaves, including by
                // unwinding. `is_running()` is what `submit_and_wait` consults
                // to decide whether someone else will flush for it; if a panic
                // left the flag set with no thread behind it, every subsequent
                // commit would wait on a flusher that no longer exists. Clearing
                // it sends them back down the inline path, which is slower but
                // is exactly how the engine ran before there was a flusher.
                let _stopped = FlusherExit(running);
                while let Some(gc) = weak.upgrade() {
                    if !gc.is_running() {
                        break;
                    }
                    gc.flush_step();
                }
            }) {
            Ok(_) => Ok(()),
            Err(e) => {
                // Undo the flag. Left set with no thread behind it, every
                // committer would skip the inline flush and wait forever for a
                // flusher that does not exist.
                self.running.store(0, Ordering::SeqCst);
                Err(e.to_string())
            }
        }
    }

    /// Stop the flush thread
    pub fn stop(&self) {
        self.running.store(0, Ordering::SeqCst);

        // Wake up the flush thread
        let _lock = self.pending.lock().unwrap();
        self.commit_available.notify_all();
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst) == 1
    }

    /// Submit a commit and wait for it to complete
    ///
    /// This blocks until the transaction's batch has been fsynced.
    /// Returns the commit timestamp on success.
    pub fn submit_and_wait(&self, txn_id: u64) -> Result<u64, String> {
        // Update arrival rate
        self.update_arrival_rate();

        // Create notification channel
        let notifier = Arc::new((Mutex::new(CommitResult::Pending), Condvar::new()));
        let commit = PendingCommitV2 {
            txn_id,
            enqueue_time: Instant::now(),
            notifier: notifier.clone(),
        };

        // Enqueue and check if we should trigger flush
        let should_flush = {
            let mut pending = self.pending.lock().unwrap();
            pending.push_back(commit);

            let batch_size = self.optimal_batch_size();
            pending.len() >= batch_size
        };

        // Signal availability
        self.commit_available.notify_one();

        // If we should flush immediately and no background thread, flush inline
        if should_flush && !self.is_running() {
            self.flush_batch();
        }

        // Wait for result
        let (lock, cvar) = &*notifier;
        let mut result = lock.lock().unwrap();

        while matches!(*result, CommitResult::Pending) {
            // Wait with timeout for defensive programming
            let timeout = Duration::from_micros(self.config.max_wait_us * 2);
            let (new_result, timeout_result) = cvar.wait_timeout(result, timeout).unwrap();
            result = new_result;

            if timeout_result.timed_out() {
                // Timeout - try flushing ourselves if no background thread
                if !self.is_running() {
                    drop(result);
                    self.flush_batch();
                    result = lock.lock().unwrap();
                }
            }
        }

        match &*result {
            CommitResult::Success(ts) => Ok(*ts),
            CommitResult::Error(e) => Err(e.clone()),
            CommitResult::Pending => Err("Unexpected pending state".into()),
        }
    }

    /// Flush one batch of pending commits
    ///
    /// Called by background flush thread or inline when needed.
    pub fn flush_batch(&self) {
        let batch = {
            let mut pending = self.pending.lock().unwrap();
            if pending.is_empty() {
                return;
            }

            // Take EVERYONE who is already waiting, not `optimal_batch_size()`.
            //
            // A durability barrier is a fixed cost: measured on this machine it
            // is ~965 us whether it covers 8 records or 64 (see
            // examples/fsync_roofline.rs -- 121 us/record at batch=8, 15 us at
            // batch=64, i.e. the same barrier divided by more records). So a
            // commit that is already enqueued and already blocked rides along
            // for free, and excluding it does not make this barrier any cheaper
            // -- it just forces that commit to wait for an entire extra one.
            //
            // Little's Law still governs how long to WAIT for commits that have
            // not arrived yet, which is a real latency/throughput tradeoff and
            // is why `flush_loop` still consults `optimal_batch_size()`. It must
            // not also cap work that has already arrived. Capping it here held
            // the achieved batch to ~15 with 64 threads blocked, which put a
            // ceiling on durable throughput at roughly a quarter of the device.
            let batch_size = pending.len().min(self.config.max_batch_size);
            pending.drain(..batch_size).collect::<Vec<_>>()
        };

        if batch.is_empty() {
            return;
        }

        let txn_ids: Vec<_> = batch.iter().map(|c| c.txn_id).collect();
        let batch_size = batch.len();

        // Measure fsync time
        let start = Instant::now();
        let result = (self.flush_fn)(&txn_ids);
        let elapsed_us = start.elapsed().as_micros() as u64;

        // Update metrics
        self.update_fsync_latency(elapsed_us);
        self.metrics.total_batches.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_commits
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.metrics
            .total_fsync_time_us
            .fetch_add(elapsed_us, Ordering::Relaxed);

        // Notify all waiters
        for commit in batch {
            let (lock, cvar) = &*commit.notifier;
            let mut result_lock = lock.lock().unwrap();
            *result_lock = match &result {
                Ok(ts) => CommitResult::Success(*ts),
                Err(e) => CommitResult::Error(e.clone()),
            };
            cvar.notify_one();
        }
    }

    /// One iteration of the flush loop: wait for work if there is none, then
    /// flush everything that has arrived.
    fn flush_step(&self) {
        let should_flush = {
            let pending = self.pending.lock().unwrap();
            let batch_size = self.optimal_batch_size();

            if pending.len() >= batch_size {
                true
            } else if pending.is_empty() {
                // Wait for commits
                let _pending = self
                    .commit_available
                    .wait_timeout(pending, Duration::from_micros(self.config.max_wait_us))
                    .unwrap()
                    .0;
                false
            } else {
                // Have some commits, check if we should wait longer
                let oldest = pending
                    .front()
                    .map(|c| c.enqueue_time.elapsed().as_micros() as u64)
                    .unwrap_or(0);

                if oldest > self.config.max_wait_us {
                    true
                } else {
                    // Wait for more commits
                    let remaining =
                        Duration::from_micros(self.config.max_wait_us.saturating_sub(oldest));
                    let _pending = self
                        .commit_available
                        .wait_timeout(pending, remaining)
                        .unwrap()
                        .0;
                    true // Flush after wait
                }
            }
        };

        if should_flush {
            self.flush_batch();
        }
    }

    /// Background flush loop (call from owner thread)
    pub fn flush_loop(&self) {
        while self.is_running() {
            self.flush_step();
        }
    }

    /// Compute optimal batch size using Little's Law
    ///
    /// N* = sqrt(2 × L_fsync × λ / C_wait)
    fn optimal_batch_size(&self) -> usize {
        let lambda = self.metrics.arrival_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0;
        let l_fsync = self.metrics.fsync_latency_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let c_wait = 1.0; // Normalized waiting cost

        let n_opt = (2.0 * l_fsync * lambda / c_wait).sqrt();
        let batch_size = (n_opt as usize)
            .max(self.config.min_batch_size)
            .min(self.config.max_batch_size);

        self.metrics
            .adaptive_batch_size
            .store(batch_size as u64, Ordering::Relaxed);
        batch_size
    }

    /// Update arrival rate using exponential moving average
    fn update_arrival_rate(&self) {
        let now_us = Self::now_us();
        let last = self.metrics.last_arrival_us.swap(now_us, Ordering::Relaxed);

        if last > 0 {
            let delta_us = now_us.saturating_sub(last);
            if delta_us > 0 {
                // Rate = 1_000_000 / delta_us (requests per second)
                // Stored as rate × 1000 for precision
                let instant_rate = 1_000_000_000 / delta_us;

                let old_rate = self.metrics.arrival_rate_ema.load(Ordering::Relaxed);
                let alpha = (self.config.ema_alpha * 1000.0) as u64;
                let new_rate = (old_rate * (1000 - alpha) + instant_rate * alpha) / 1000;
                self.metrics
                    .arrival_rate_ema
                    .store(new_rate, Ordering::Relaxed);
            }
        }
    }

    /// Update fsync latency estimate
    fn update_fsync_latency(&self, latency_us: u64) {
        let old = self.metrics.fsync_latency_us.load(Ordering::Relaxed);
        let alpha = (self.config.ema_alpha * 1000.0) as u64;
        let new = (old * (1000 - alpha) + latency_us * alpha) / 1000;
        self.metrics.fsync_latency_us.store(new, Ordering::Relaxed);
    }

    /// Get current time in microseconds
    fn now_us() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Get statistics for monitoring
    pub fn stats(&self) -> GroupCommitStatsV2 {
        GroupCommitStatsV2 {
            adaptive_batch_size: self.metrics.adaptive_batch_size.load(Ordering::Relaxed) as usize,
            arrival_rate: self.metrics.arrival_rate_ema.load(Ordering::Relaxed) as f64 / 1000.0,
            fsync_latency_us: self.metrics.fsync_latency_us.load(Ordering::Relaxed),
            pending_count: self.pending.lock().unwrap().len(),
            total_commits: self.metrics.total_commits.load(Ordering::Relaxed),
            total_batches: self.metrics.total_batches.load(Ordering::Relaxed),
            avg_batch_size: {
                let batches = self.metrics.total_batches.load(Ordering::Relaxed);
                let commits = self.metrics.total_commits.load(Ordering::Relaxed);
                if batches > 0 {
                    commits as f64 / batches as f64
                } else {
                    0.0
                }
            },
            avg_fsync_time_us: {
                let batches = self.metrics.total_batches.load(Ordering::Relaxed);
                let time = self.metrics.total_fsync_time_us.load(Ordering::Relaxed);
                if batches > 0 { time / batches } else { 0 }
            },
        }
    }
}

/// Statistics for event-driven group commit
#[derive(Debug, Clone)]
pub struct GroupCommitStatsV2 {
    /// Current adaptive batch size
    pub adaptive_batch_size: usize,
    /// Estimated arrival rate (requests/second)
    pub arrival_rate: f64,
    /// Estimated fsync latency (microseconds)
    pub fsync_latency_us: u64,
    /// Current pending commit count
    pub pending_count: usize,
    /// Total commits processed
    pub total_commits: u64,
    /// Total batches processed
    pub total_batches: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average fsync time (microseconds)
    pub avg_fsync_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_single_commit() {
        let commit_ts = AtomicU64::new(100);
        let gc = EventDrivenGroupCommit::new(move |_txn_ids| {
            Ok(commit_ts.fetch_add(1, Ordering::SeqCst))
        });

        let result = gc.submit_and_wait(1);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 100);
    }

    #[test]
    fn test_batch_commit() {
        use parking_lot::RwLock;
        use std::sync::Arc;
        use std::thread;

        let _commit_ts = Arc::new(AtomicU64::new(100));
        let batch_sizes = Arc::new(RwLock::new(Vec::new()));
        let batch_sizes_clone = batch_sizes.clone();

        let gc = Arc::new(EventDrivenGroupCommit::with_config(
            move |txn_ids| {
                batch_sizes_clone.write().push(txn_ids.len());
                Ok(100)
            },
            GroupCommitConfig {
                min_batch_size: 3,
                max_wait_us: 1_000_000, // 1 second - long enough to batch
                ..Default::default()
            },
        ));

        // Submit 3 commits in parallel
        let mut handles = vec![];
        for i in 0..3 {
            let gc = Arc::clone(&gc);
            handles.push(thread::spawn(move || gc.submit_and_wait(i)));
        }

        // Wait for all
        for h in handles {
            assert!(h.join().unwrap().is_ok());
        }

        // Should have been batched
        let sizes = batch_sizes.read();
        assert!(!sizes.is_empty());
        // The 3 commits should have been batched together
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_adaptive_sizing() {
        let gc = EventDrivenGroupCommit::with_config(
            |_| Ok(1),
            GroupCommitConfig {
                fsync_latency_us: 5000, // 5ms
                ..Default::default()
            },
        );

        // Simulate high arrival rate (1000 req/s)
        gc.metrics
            .arrival_rate_ema
            .store(1_000_000, Ordering::Relaxed);

        let batch_size = gc.optimal_batch_size();

        // N* = sqrt(2 × 0.005 × 1000 / 1) ≈ 3.16
        assert!((3..=10).contains(&batch_size));
    }

    #[test]
    fn test_stats() {
        let gc = EventDrivenGroupCommit::new(|_| Ok(1));

        gc.metrics.total_commits.store(100, Ordering::Relaxed);
        gc.metrics.total_batches.store(10, Ordering::Relaxed);
        gc.metrics
            .total_fsync_time_us
            .store(50_000, Ordering::Relaxed);

        let stats = gc.stats();
        assert_eq!(stats.total_commits, 100);
        assert_eq!(stats.total_batches, 10);
        assert_eq!(stats.avg_batch_size, 10.0);
        assert_eq!(stats.avg_fsync_time_us, 5000);
    }

    /// The whole point of a background flusher: concurrent committers pay for
    /// one barrier between them, not one barrier each.
    ///
    /// Without a running flusher every committer flushes inline over whatever
    /// slice of the queue it drained, so this workload produced a run of tiny
    /// batches. Asserting on the batch count rather than on throughput keeps
    /// the test honest on a loaded or virtualised machine.
    #[test]
    fn a_running_flusher_groups_concurrent_commits_into_few_barriers() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let batches = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&batches);

        let gc = Arc::new(EventDrivenGroupCommit::with_config(
            move |txn_ids| {
                seen.lock().unwrap().push(txn_ids.len());
                // Stand in for a durability barrier, which is a fixed cost
                // regardless of how many records it covers.
                thread::sleep(Duration::from_millis(5));
                Ok(1)
            },
            GroupCommitConfig {
                min_batch_size: 1,
                ..Default::default()
            },
        ));
        gc.start_background().expect("flusher should start");
        assert!(gc.is_running());

        const COMMITTERS: u64 = 32;
        let handles: Vec<_> = (0..COMMITTERS)
            .map(|i| {
                let gc = Arc::clone(&gc);
                thread::spawn(move || gc.submit_and_wait(i))
            })
            .collect();
        for h in handles {
            assert!(h.join().unwrap().is_ok(), "every commit must be answered");
        }

        let sizes = batches.lock().unwrap();
        let total: usize = sizes.iter().sum();
        assert_eq!(total as u64, COMMITTERS, "no commit may be lost or doubled");
        assert!(
            sizes.len() < COMMITTERS as usize,
            "{COMMITTERS} commits took {} barriers; they were not grouped at all",
            sizes.len()
        );

        gc.stop();
    }

    /// A flusher that dies must not take every future commit down with it.
    ///
    /// `submit_and_wait` only flushes inline when `is_running()` is false, so a
    /// panicking flusher that left the flag set would leave the engine unable
    /// to commit anything ever again -- a hang, not an error, and so
    /// indistinguishable from a slow disk.
    #[test]
    fn a_panicking_flusher_falls_back_to_inline_commit() {
        use std::sync::Arc;

        let calls = Arc::new(AtomicU64::new(0));
        let seen = Arc::clone(&calls);
        let gc = Arc::new(EventDrivenGroupCommit::with_config(
            move |_txn_ids| {
                // Panic on the first flush only, so the fallback path has a
                // working callback to prove the engine still commits.
                if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                    panic!("flusher died");
                }
                Ok(7)
            },
            GroupCommitConfig {
                min_batch_size: 1,
                max_wait_us: 1_000,
                ..Default::default()
            },
        ));
        gc.start_background().expect("flusher should start");

        // Force the flusher to run a batch through the panicking callback. Its
        // waiter must be failed, not stranded.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        assert!(
            gc.submit_and_wait(1).is_err(),
            "the commit the flusher died on must report failure"
        );
        std::panic::set_hook(prev);
        // The exit guard must have cleared the flag on the way out, whether the
        // callback panicked or not.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while gc.is_running() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            !gc.is_running(),
            "a dead flusher must not leave the running flag set"
        );

        // And the engine still commits, inline, without hanging.
        assert_eq!(gc.submit_and_wait(2), Ok(7));
    }

    /// A submitted commit always gets an answer, even if its batch is destroyed
    /// before anyone writes a result into it.
    ///
    /// `flush_batch` removes a batch from the queue before invoking the flush
    /// callback, so an unwinding flusher drops those records on the floor. The
    /// waiters are blocked on a condvar nobody else can reach, so without a
    /// resolve-on-drop they wait forever.
    #[test]
    fn a_dropped_batch_fails_its_waiters_instead_of_stranding_them() {
        let notifier = Arc::new((Mutex::new(CommitResult::Pending), Condvar::new()));
        let commit = PendingCommitV2 {
            txn_id: 1,
            enqueue_time: Instant::now(),
            notifier: Arc::clone(&notifier),
        };

        drop(commit);

        let result = notifier.0.lock().unwrap();
        assert!(
            matches!(*result, CommitResult::Error(_)),
            "a commit that was never flushed must report failure, not stay pending"
        );
    }

    /// Resolving on drop must not overwrite a real result with a spurious
    /// error: `flush_batch` writes the result and then drops the record.
    #[test]
    fn resolving_on_drop_does_not_clobber_a_real_result() {
        let notifier = Arc::new((Mutex::new(CommitResult::Pending), Condvar::new()));
        let commit = PendingCommitV2 {
            txn_id: 1,
            enqueue_time: Instant::now(),
            notifier: Arc::clone(&notifier),
        };

        *notifier.0.lock().unwrap() = CommitResult::Success(42);
        drop(commit);

        assert!(matches!(
            *notifier.0.lock().unwrap(),
            CommitResult::Success(42)
        ));
    }
}
