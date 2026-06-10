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

//! Supervised background workers.
//!
//! Long-running detached workers (LSM compaction, GC, the event-driven flusher,
//! dirty-tracking aggregation, …) historically ran as a bare
//! [`std::thread::spawn`] containing a `loop { … }`. If any iteration panicked —
//! a poisoned lock, an arithmetic overflow in debug, an `unwrap()` on a
//! transient error — the thread unwound and **died silently**. Compaction would
//! stop, GC would stop, and the only symptom would be slowly growing disk usage
//! or unbounded version chains, with no signal that anything was wrong.
//!
//! [`Supervisor`] wraps a worker's per-iteration body in
//! [`std::panic::catch_unwind`] so a panic in one iteration is *contained*:
//! the panic is counted, the worker is marked unhealthy, a bounded backoff is
//! applied, and the loop is **restarted** rather than torn down. Callers can
//! observe liveness via [`WorkerHealth`].
//!
//! # Contract
//!
//! - The worker body is a closure returning [`WorkerStep`]. Returning
//!   [`WorkerStep::Continue`] runs another iteration; [`WorkerStep::Stop`] ends
//!   the worker cleanly.
//! - Shutdown is cooperative: callers flip the shared `running` flag (or return
//!   [`WorkerStep::Stop`] from the body) and then [`SupervisedWorker::join`].
//! - Backoff after a panic grows geometrically from `base_backoff` up to
//!   `max_backoff` and resets to zero after any successful (non-panicking)
//!   iteration. This prevents a tight panic loop from pinning a core.
//!
//! # Example
//!
//! ```ignore
//! let running = Arc::new(AtomicBool::new(true));
//! let worker = Supervisor::new("compaction")
//!     .spawn(running.clone(), move || {
//!         do_one_compaction_pass();
//!         WorkerStep::Continue
//!     });
//! // … later …
//! running.store(false, Ordering::SeqCst);
//! worker.join();
//! assert!(worker.health().panics() == 0);
//! ```

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// Outcome of a single worker iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStep {
    /// Run another iteration.
    Continue,
    /// Stop the worker cleanly (no further iterations).
    Stop,
}

/// Liveness and fault counters for a supervised worker.
///
/// Cheap to clone (`Arc` inside) and safe to read from any thread, so it can be
/// exported into a metrics registry or polled by a health endpoint.
#[derive(Debug, Clone)]
pub struct WorkerHealth {
    inner: Arc<WorkerHealthInner>,
}

#[derive(Debug)]
struct WorkerHealthInner {
    /// Total number of iterations that panicked.
    panics: AtomicU64,
    /// Total number of times the loop body was (re)entered after a panic.
    restarts: AtomicU64,
    /// Total successful (non-panicking) iterations completed.
    iterations: AtomicU64,
    /// `true` while the worker is making forward progress; flips to `false`
    /// immediately after a panic and back to `true` after the next success.
    healthy: AtomicBool,
    /// `true` once the worker loop has fully exited.
    finished: AtomicBool,
}

impl WorkerHealth {
    fn new() -> Self {
        Self {
            inner: Arc::new(WorkerHealthInner {
                panics: AtomicU64::new(0),
                restarts: AtomicU64::new(0),
                iterations: AtomicU64::new(0),
                healthy: AtomicBool::new(true),
                finished: AtomicBool::new(false),
            }),
        }
    }

    /// Number of iterations that panicked and were contained.
    #[inline]
    pub fn panics(&self) -> u64 {
        self.inner.panics.load(Ordering::Acquire)
    }

    /// Number of times the loop was restarted after a panic.
    #[inline]
    pub fn restarts(&self) -> u64 {
        self.inner.restarts.load(Ordering::Acquire)
    }

    /// Number of successful (non-panicking) iterations completed.
    #[inline]
    pub fn iterations(&self) -> u64 {
        self.inner.iterations.load(Ordering::Acquire)
    }

    /// Whether the worker is currently making forward progress.
    ///
    /// Returns `false` between a panic and the next successful iteration.
    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.inner.healthy.load(Ordering::Acquire)
    }

    /// Whether the worker loop has fully exited.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.inner.finished.load(Ordering::Acquire)
    }
}

/// Builder/configuration for a supervised worker.
pub struct Supervisor {
    name: String,
    base_backoff: Duration,
    max_backoff: Duration,
}

impl Supervisor {
    /// Create a supervisor for a worker with the given diagnostic name.
    ///
    /// Defaults: `base_backoff = 10ms`, `max_backoff = 1s`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_secs(1),
        }
    }

    /// Override the initial post-panic backoff.
    pub fn base_backoff(mut self, d: Duration) -> Self {
        self.base_backoff = d;
        self
    }

    /// Override the maximum post-panic backoff.
    pub fn max_backoff(mut self, d: Duration) -> Self {
        self.max_backoff = d;
        self
    }

    /// Spawn the supervised worker.
    ///
    /// The worker runs `body` repeatedly while `running` is `true` and the body
    /// keeps returning [`WorkerStep::Continue`]. Each call to `body` is isolated
    /// with [`catch_unwind`]; a panic is counted, the worker is marked unhealthy,
    /// a bounded backoff is applied, and the loop continues.
    pub fn spawn<F>(self, running: Arc<AtomicBool>, mut body: F) -> SupervisedWorker
    where
        F: FnMut() -> WorkerStep + Send + 'static,
    {
        let health = WorkerHealth::new();
        let thread_health = health.clone();
        let name = self.name.clone();
        let base = self.base_backoff;
        let max = self.max_backoff;

        let handle = std::thread::Builder::new()
            .name(format!("soch-sup-{name}"))
            .spawn(move || {
                let mut backoff = Duration::ZERO;
                while running.load(Ordering::Acquire) {
                    // Isolate one iteration. AssertUnwindSafe is sound here:
                    // on panic we do not observe any half-updated state owned by
                    // `body` — we simply re-enter the loop on the next iteration.
                    let result = catch_unwind(AssertUnwindSafe(&mut body));
                    match result {
                        Ok(WorkerStep::Continue) => {
                            thread_health
                                .inner
                                .iterations
                                .fetch_add(1, Ordering::AcqRel);
                            thread_health.inner.healthy.store(true, Ordering::Release);
                            backoff = Duration::ZERO;
                        }
                        Ok(WorkerStep::Stop) => break,
                        Err(_panic) => {
                            thread_health.inner.panics.fetch_add(1, Ordering::AcqRel);
                            thread_health.inner.restarts.fetch_add(1, Ordering::AcqRel);
                            thread_health.inner.healthy.store(false, Ordering::Release);

                            // Geometric backoff, clamped to `max`, to avoid a hot
                            // panic loop pinning a CPU. Reset on next success.
                            backoff = if backoff.is_zero() {
                                base
                            } else {
                                (backoff * 2).min(max)
                            };
                            if running.load(Ordering::Acquire) {
                                std::thread::sleep(backoff);
                            }
                        }
                    }
                }
                thread_health.inner.finished.store(true, Ordering::Release);
            })
            .expect("failed to spawn supervised worker thread");

        SupervisedWorker {
            handle: Some(handle),
            health,
        }
    }
}

/// Handle to a running supervised worker.
pub struct SupervisedWorker {
    handle: Option<JoinHandle<()>>,
    health: WorkerHealth,
}

impl SupervisedWorker {
    /// Health/liveness counters shared with the worker thread.
    #[inline]
    pub fn health(&self) -> WorkerHealth {
        self.health.clone()
    }

    /// Join the worker thread, blocking until it exits.
    ///
    /// The caller is responsible for first signalling shutdown (flipping the
    /// `running` flag passed to [`Supervisor::spawn`]) so the loop can observe
    /// it and exit; otherwise this blocks indefinitely.
    pub fn join(mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;

    #[test]
    fn test_runs_until_running_cleared() {
        let running = Arc::new(AtomicBool::new(true));
        let counter = Arc::new(AtomicU64::new(0));
        let c = counter.clone();
        let worker = Supervisor::new("count").spawn(running.clone(), move || {
            c.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(1));
            WorkerStep::Continue
        });

        // Let it run a bit, then stop.
        std::thread::sleep(Duration::from_millis(50));
        running.store(false, Ordering::SeqCst);
        let health = worker.health();
        worker.join();

        assert!(health.is_finished());
        assert!(counter.load(Ordering::Relaxed) > 0, "worker never ran");
        assert_eq!(health.panics(), 0);
    }

    #[test]
    fn test_stop_step_exits_cleanly() {
        let running = Arc::new(AtomicBool::new(true));
        let counter = Arc::new(AtomicU64::new(0));
        let c = counter.clone();
        let worker = Supervisor::new("stopper").spawn(running, move || {
            let n = c.fetch_add(1, Ordering::Relaxed);
            if n >= 2 {
                WorkerStep::Stop
            } else {
                WorkerStep::Continue
            }
        });

        let health = worker.health();
        worker.join();
        assert!(health.is_finished());
        assert_eq!(counter.load(Ordering::Relaxed), 3); // 0,1,2 -> stop at 2
    }

    #[test]
    fn test_panic_is_contained_and_loop_survives() {
        // The worker panics on the first iteration, then makes progress.
        // Without the supervisor the thread would die and `progress` stay 0.
        let running = Arc::new(AtomicBool::new(true));
        let attempts = Arc::new(AtomicU64::new(0));
        let progress = Arc::new(AtomicU64::new(0));
        let a = attempts.clone();
        let p = progress.clone();

        let worker = Supervisor::new("panicker")
            .base_backoff(Duration::from_millis(1))
            .max_backoff(Duration::from_millis(5))
            .spawn(running.clone(), move || {
                let n = a.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    panic!("boom on first iteration");
                }
                p.fetch_add(1, Ordering::SeqCst);
                WorkerStep::Continue
            });

        let health = worker.health();
        // Wait until the worker has recovered and made progress.
        let deadline = Instant::now() + Duration::from_secs(5);
        while progress.load(Ordering::SeqCst) == 0 {
            assert!(
                Instant::now() < deadline,
                "worker did not recover from panic"
            );
            std::thread::sleep(Duration::from_millis(2));
        }

        running.store(false, Ordering::SeqCst);
        worker.join();

        assert_eq!(health.panics(), 1, "panic should have been counted once");
        assert!(health.restarts() >= 1);
        assert!(
            progress.load(Ordering::SeqCst) > 0,
            "loop must survive the panic and keep working"
        );
        // After a successful iteration following the panic, it is healthy again.
        assert!(health.is_healthy());
    }

    #[test]
    fn test_health_unhealthy_immediately_after_panic() {
        // A worker that only ever panics must report unhealthy and keep counting.
        let running = Arc::new(AtomicBool::new(true));
        let worker = Supervisor::new("always-panic")
            .base_backoff(Duration::from_millis(1))
            .max_backoff(Duration::from_millis(2))
            .spawn(running.clone(), || {
                panic!("always");
            });

        let health = worker.health();
        let deadline = Instant::now() + Duration::from_secs(5);
        while health.panics() < 3 {
            assert!(Instant::now() < deadline, "panics were not counted");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(
            !health.is_healthy(),
            "a perpetually panicking worker is unhealthy"
        );

        running.store(false, Ordering::SeqCst);
        worker.join();
    }
}
