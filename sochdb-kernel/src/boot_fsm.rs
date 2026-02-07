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

//! # Deterministic Boot Finite State Machine
//!
//! Implements a production-grade boot sequence with:
//! - Well-defined states: `Init → Migrate → Recover → Warmup → Ready`
//! - Time budgets for each phase (for Kubernetes probe alignment)
//! - Progress reporting for external observability
//! - Recovery modes: Normal, ReadOnlyRecovery, ForceRecovery
//!
//! ## Kubernetes Integration
//!
//! The FSM exports progress metrics that align with K8s probe semantics:
//! - `startupProbe`: tolerates long recovery (uses recovery budget)
//! - `readinessProbe`: true only when FSM is in `Ready`
//! - `livenessProbe`: heartbeat-based (separate from FSM)
//!
//! ## Safety Property
//!
//! `Ready ⇒ (recovery_complete ∧ invariants_checked ∧ services_registered)`
//!
//! ## Complexity Bounds
//!
//! Recovery is O(|WAL| + |checkpoint|). The FSM tracks and exposes this
//! to allow operators to configure appropriate probe timeouts.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Boot phase states (DFA transitions)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BootPhase {
    /// Initial state before any boot activity
    Uninitialized = 0,
    /// Initializing core subsystems (allocator, config, logging)
    Init = 1,
    /// Running schema/format migrations
    Migrate = 2,
    /// Recovering from WAL (ARIES redo/undo)
    Recover = 3,
    /// Warming up caches and indexes
    Warmup = 4,
    /// Fully operational
    Ready = 5,
    /// Read-only recovery mode (for forensics)
    ReadOnlyRecovery = 6,
    /// Force recovery mode (skip some checks)
    ForceRecovery = 7,
    /// Graceful shutdown in progress
    ShuttingDown = 8,
    /// Boot failed (terminal state)
    Failed = 9,
}

impl BootPhase {
    /// Get human-readable phase name
    pub fn name(&self) -> &'static str {
        match self {
            BootPhase::Uninitialized => "uninitialized",
            BootPhase::Init => "init",
            BootPhase::Migrate => "migrate",
            BootPhase::Recover => "recover",
            BootPhase::Warmup => "warmup",
            BootPhase::Ready => "ready",
            BootPhase::ReadOnlyRecovery => "readonly_recovery",
            BootPhase::ForceRecovery => "force_recovery",
            BootPhase::ShuttingDown => "shutting_down",
            BootPhase::Failed => "failed",
        }
    }

    /// Check if this phase indicates the system is ready for traffic
    pub fn is_ready(&self) -> bool {
        matches!(self, BootPhase::Ready)
    }

    /// Check if this phase indicates the system is alive (not dead)
    pub fn is_alive(&self) -> bool {
        !matches!(self, BootPhase::Failed)
    }

    /// Check if boot is still in progress
    pub fn is_booting(&self) -> bool {
        matches!(
            self,
            BootPhase::Init
                | BootPhase::Migrate
                | BootPhase::Recover
                | BootPhase::Warmup
                | BootPhase::ReadOnlyRecovery
                | BootPhase::ForceRecovery
        )
    }
}

/// Progress information for a boot phase
#[derive(Debug, Clone)]
pub struct PhaseProgress {
    /// Current progress (0-100)
    pub percent: u8,
    /// Human-readable status message
    pub message: String,
    /// Items processed (e.g., WAL records replayed)
    pub items_processed: u64,
    /// Total items to process (0 if unknown)
    pub items_total: u64,
    /// Bytes processed
    pub bytes_processed: u64,
    /// Total bytes to process (0 if unknown)
    pub bytes_total: u64,
    /// Time spent in this phase
    pub elapsed: Duration,
}

impl Default for PhaseProgress {
    fn default() -> Self {
        Self {
            percent: 0,
            message: String::new(),
            items_processed: 0,
            items_total: 0,
            bytes_processed: 0,
            bytes_total: 0,
            elapsed: Duration::ZERO,
        }
    }
}

/// Time budget configuration for each boot phase
#[derive(Debug, Clone)]
pub struct BootBudgets {
    /// Maximum time for init phase
    pub init_budget: Duration,
    /// Maximum time for migration phase
    pub migrate_budget: Duration,
    /// Maximum time for recovery phase (WAL replay)
    pub recover_budget: Duration,
    /// Maximum time for warmup phase
    pub warmup_budget: Duration,
    /// Total boot timeout
    pub total_budget: Duration,
}

impl Default for BootBudgets {
    fn default() -> Self {
        Self {
            init_budget: Duration::from_secs(30),
            migrate_budget: Duration::from_secs(300),   // 5 min for migrations
            recover_budget: Duration::from_secs(1800),  // 30 min for WAL replay
            warmup_budget: Duration::from_secs(300),    // 5 min for cache warmup
            total_budget: Duration::from_secs(3600),    // 1 hour total
        }
    }
}

impl BootBudgets {
    /// Create budgets suitable for Kubernetes startupProbe
    /// 
    /// K8s startupProbe checks are: failureThreshold × periodSeconds
    /// These budgets should be less than that product.
    pub fn for_kubernetes(startup_probe_total_seconds: u64) -> Self {
        let total = Duration::from_secs(startup_probe_total_seconds);
        Self {
            init_budget: Duration::from_secs(startup_probe_total_seconds / 20),
            migrate_budget: Duration::from_secs(startup_probe_total_seconds / 5),
            recover_budget: Duration::from_secs(startup_probe_total_seconds * 3 / 5),
            warmup_budget: Duration::from_secs(startup_probe_total_seconds / 10),
            total_budget: total,
        }
    }
}

/// Recovery mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryMode {
    /// Normal recovery with full ARIES protocol
    Normal,
    /// Read-only mode for forensics (no WAL writes)
    ReadOnly,
    /// Force recovery (skip some consistency checks)
    Force,
}

/// Preload hints for deterministic warmup
#[derive(Debug, Clone, Default)]
pub struct PreloadHints {
    /// Specific pages to preload
    pub page_ids: Vec<u64>,
    /// Index names to preload
    pub indexes: Vec<String>,
    /// Estimated working set size in bytes
    pub working_set_bytes: u64,
}

/// Boot state machine with thread-safe state transitions
pub struct BootStateMachine {
    /// Current boot phase
    phase: RwLock<BootPhase>,
    /// Phase start time
    phase_start: RwLock<Instant>,
    /// Boot start time
    boot_start: RwLock<Option<Instant>>,
    /// Current phase progress
    progress: RwLock<PhaseProgress>,
    /// Time budgets
    budgets: BootBudgets,
    /// Recovery mode
    recovery_mode: RwLock<RecoveryMode>,
    /// Failure reason (if Failed)
    failure_reason: RwLock<Option<String>>,
    /// Preload hints for warmup
    preload_hints: RwLock<PreloadHints>,
    /// Metrics counters
    metrics: BootMetrics,
}

/// Boot metrics for observability
pub struct BootMetrics {
    /// Number of WAL records replayed
    pub wal_records_replayed: AtomicU64,
    /// Bytes of WAL data processed
    pub wal_bytes_processed: AtomicU64,
    /// Number of pages recovered
    pub pages_recovered: AtomicU64,
    /// Number of transactions rolled back
    pub txns_rolled_back: AtomicU64,
    /// Checkpoint scan bytes
    pub checkpoint_bytes_scanned: AtomicU64,
    /// Migration steps completed
    pub migration_steps_completed: AtomicU64,
    /// Cache hit rate during warmup (scaled by 1000)
    pub warmup_hit_rate_permille: AtomicU64,
}

impl Default for BootMetrics {
    fn default() -> Self {
        Self {
            wal_records_replayed: AtomicU64::new(0),
            wal_bytes_processed: AtomicU64::new(0),
            pages_recovered: AtomicU64::new(0),
            txns_rolled_back: AtomicU64::new(0),
            checkpoint_bytes_scanned: AtomicU64::new(0),
            migration_steps_completed: AtomicU64::new(0),
            warmup_hit_rate_permille: AtomicU64::new(0),
        }
    }
}

/// Error during boot
#[derive(Debug, Clone)]
pub struct BootError {
    pub phase: BootPhase,
    pub message: String,
    pub recoverable: bool,
}

impl std::fmt::Display for BootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Boot error in phase {}: {} (recoverable: {})",
            self.phase.name(),
            self.message,
            self.recoverable
        )
    }
}

impl std::error::Error for BootError {}

impl BootStateMachine {
    /// Create a new boot state machine
    pub fn new(budgets: BootBudgets) -> Self {
        Self {
            phase: RwLock::new(BootPhase::Uninitialized),
            phase_start: RwLock::new(Instant::now()),
            boot_start: RwLock::new(None),
            progress: RwLock::new(PhaseProgress::default()),
            budgets,
            recovery_mode: RwLock::new(RecoveryMode::Normal),
            failure_reason: RwLock::new(None),
            preload_hints: RwLock::new(PreloadHints::default()),
            metrics: BootMetrics::default(),
        }
    }

    /// Create with default budgets
    pub fn with_defaults() -> Self {
        Self::new(BootBudgets::default())
    }

    /// Get current boot phase
    pub fn current_phase(&self) -> BootPhase {
        *self.phase.read()
    }

    /// Get current progress
    pub fn current_progress(&self) -> PhaseProgress {
        let mut progress = self.progress.read().clone();
        progress.elapsed = self.phase_start.read().elapsed();
        progress
    }

    /// Check if system is ready for traffic
    pub fn is_ready(&self) -> bool {
        self.current_phase().is_ready()
    }

    /// Check if system is alive (for liveness probe)
    pub fn is_alive(&self) -> bool {
        self.current_phase().is_alive()
    }

    /// Get time remaining in current phase budget
    pub fn remaining_budget(&self) -> Duration {
        let phase = *self.phase.read();
        let elapsed = self.phase_start.read().elapsed();
        let budget = match phase {
            BootPhase::Init => self.budgets.init_budget,
            BootPhase::Migrate => self.budgets.migrate_budget,
            BootPhase::Recover | BootPhase::ReadOnlyRecovery | BootPhase::ForceRecovery => {
                self.budgets.recover_budget
            }
            BootPhase::Warmup => self.budgets.warmup_budget,
            _ => Duration::ZERO,
        };
        budget.saturating_sub(elapsed)
    }

    /// Get total boot elapsed time
    pub fn total_elapsed(&self) -> Duration {
        self.boot_start
            .read()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Start the boot sequence
    pub fn start_boot(&self, recovery_mode: RecoveryMode) -> Result<(), BootError> {
        let mut phase = self.phase.write();
        if *phase != BootPhase::Uninitialized {
            return Err(BootError {
                phase: *phase,
                message: "Boot already started".to_string(),
                recoverable: false,
            });
        }

        *self.boot_start.write() = Some(Instant::now());
        *self.recovery_mode.write() = recovery_mode;
        *phase = BootPhase::Init;
        *self.phase_start.write() = Instant::now();
        *self.progress.write() = PhaseProgress {
            message: "Initializing core subsystems".to_string(),
            ..Default::default()
        };

        Ok(())
    }

    /// Transition to next phase
    pub fn transition_to(&self, next_phase: BootPhase) -> Result<(), BootError> {
        let mut phase = self.phase.write();
        let current = *phase;

        // Validate transition
        let valid = match (current, next_phase) {
            (BootPhase::Uninitialized, BootPhase::Init) => true,
            (BootPhase::Init, BootPhase::Migrate) => true,
            (BootPhase::Init, BootPhase::Failed) => true,
            (BootPhase::Migrate, BootPhase::Recover) => true,
            (BootPhase::Migrate, BootPhase::ReadOnlyRecovery) => true,
            (BootPhase::Migrate, BootPhase::ForceRecovery) => true,
            (BootPhase::Migrate, BootPhase::Failed) => true,
            (BootPhase::Recover, BootPhase::Warmup) => true,
            (BootPhase::Recover, BootPhase::Ready) => true, // Skip warmup
            (BootPhase::Recover, BootPhase::Failed) => true,
            (BootPhase::ReadOnlyRecovery, BootPhase::Ready) => true,
            (BootPhase::ReadOnlyRecovery, BootPhase::Failed) => true,
            (BootPhase::ForceRecovery, BootPhase::Warmup) => true,
            (BootPhase::ForceRecovery, BootPhase::Ready) => true,
            (BootPhase::ForceRecovery, BootPhase::Failed) => true,
            (BootPhase::Warmup, BootPhase::Ready) => true,
            (BootPhase::Warmup, BootPhase::Failed) => true,
            (BootPhase::Ready, BootPhase::ShuttingDown) => true,
            (_, BootPhase::Failed) => true, // Can always fail
            _ => false,
        };

        if !valid {
            return Err(BootError {
                phase: current,
                message: format!(
                    "Invalid transition: {} -> {}",
                    current.name(),
                    next_phase.name()
                ),
                recoverable: false,
            });
        }

        // Check budget exceeded
        if self.remaining_budget() == Duration::ZERO && current.is_booting() {
            *phase = BootPhase::Failed;
            *self.failure_reason.write() = Some(format!(
                "Budget exceeded in phase {}",
                current.name()
            ));
            return Err(BootError {
                phase: current,
                message: "Phase budget exceeded".to_string(),
                recoverable: false,
            });
        }

        *phase = next_phase;
        *self.phase_start.write() = Instant::now();
        *self.progress.write() = PhaseProgress::default();

        Ok(())
    }

    /// Update progress within current phase
    pub fn update_progress(&self, progress: PhaseProgress) {
        *self.progress.write() = progress;
    }

    /// Mark boot as failed with reason
    pub fn fail(&self, reason: &str) {
        let current = *self.phase.read();
        *self.phase.write() = BootPhase::Failed;
        *self.failure_reason.write() = Some(format!(
            "Failed in {}: {}",
            current.name(),
            reason
        ));
    }

    /// Get failure reason if failed
    pub fn failure_reason(&self) -> Option<String> {
        self.failure_reason.read().clone()
    }

    /// Set preload hints for warmup phase
    pub fn set_preload_hints(&self, hints: PreloadHints) {
        *self.preload_hints.write() = hints;
    }

    /// Get preload hints
    pub fn preload_hints(&self) -> PreloadHints {
        self.preload_hints.read().clone()
    }

    /// Get boot metrics
    pub fn metrics(&self) -> &BootMetrics {
        &self.metrics
    }

    /// Record WAL replay progress
    pub fn record_wal_progress(&self, records: u64, bytes: u64) {
        self.metrics.wal_records_replayed.fetch_add(records, Ordering::Relaxed);
        self.metrics.wal_bytes_processed.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record page recovery
    pub fn record_page_recovered(&self, count: u64) {
        self.metrics.pages_recovered.fetch_add(count, Ordering::Relaxed);
    }

    /// Record transaction rollback
    pub fn record_txn_rollback(&self, count: u64) {
        self.metrics.txns_rolled_back.fetch_add(count, Ordering::Relaxed);
    }

    /// Generate health check response for Kubernetes probes
    pub fn health_status(&self) -> HealthStatus {
        let phase = self.current_phase();
        let progress = self.current_progress();

        HealthStatus {
            phase,
            phase_name: phase.name().to_string(),
            is_ready: phase.is_ready(),
            is_alive: phase.is_alive(),
            is_booting: phase.is_booting(),
            progress_percent: progress.percent,
            progress_message: progress.message,
            phase_elapsed_ms: progress.elapsed.as_millis() as u64,
            total_elapsed_ms: self.total_elapsed().as_millis() as u64,
            remaining_budget_ms: self.remaining_budget().as_millis() as u64,
            failure_reason: self.failure_reason(),
            wal_records_replayed: self.metrics.wal_records_replayed.load(Ordering::Relaxed),
            wal_bytes_processed: self.metrics.wal_bytes_processed.load(Ordering::Relaxed),
        }
    }
}

/// Health status for probes and observability
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub phase: BootPhase,
    pub phase_name: String,
    pub is_ready: bool,
    pub is_alive: bool,
    pub is_booting: bool,
    pub progress_percent: u8,
    pub progress_message: String,
    pub phase_elapsed_ms: u64,
    pub total_elapsed_ms: u64,
    pub remaining_budget_ms: u64,
    pub failure_reason: Option<String>,
    pub wal_records_replayed: u64,
    pub wal_bytes_processed: u64,
}

impl HealthStatus {
    /// Format as JSON for health endpoints
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"phase":"{}","is_ready":{},"is_alive":{},"is_booting":{},"progress_percent":{},"progress_message":"{}","phase_elapsed_ms":{},"total_elapsed_ms":{},"remaining_budget_ms":{},"failure_reason":{},"wal_records_replayed":{},"wal_bytes_processed":{}}}"#,
            self.phase_name,
            self.is_ready,
            self.is_alive,
            self.is_booting,
            self.progress_percent,
            self.progress_message.replace('"', "\\\""),
            self.phase_elapsed_ms,
            self.total_elapsed_ms,
            self.remaining_budget_ms,
            self.failure_reason.as_ref().map(|s| format!("\"{}\"", s.replace('"', "\\\""))).unwrap_or_else(|| "null".to_string()),
            self.wal_records_replayed,
            self.wal_bytes_processed,
        )
    }
}

/// Boot orchestrator that coordinates the full boot sequence
pub struct BootOrchestrator {
    fsm: Arc<BootStateMachine>,
}

impl BootOrchestrator {
    /// Create a new boot orchestrator
    pub fn new(budgets: BootBudgets) -> Self {
        Self {
            fsm: Arc::new(BootStateMachine::new(budgets)),
        }
    }

    /// Get the FSM for health checks
    pub fn fsm(&self) -> Arc<BootStateMachine> {
        self.fsm.clone()
    }

    /// Run the boot sequence with callbacks for each phase
    pub fn run_boot<I, M, R, W>(
        &self,
        recovery_mode: RecoveryMode,
        init_fn: I,
        migrate_fn: M,
        recover_fn: R,
        warmup_fn: W,
    ) -> Result<(), BootError>
    where
        I: FnOnce(&BootStateMachine) -> Result<(), BootError>,
        M: FnOnce(&BootStateMachine) -> Result<(), BootError>,
        R: FnOnce(&BootStateMachine) -> Result<PreloadHints, BootError>,
        W: FnOnce(&BootStateMachine, PreloadHints) -> Result<(), BootError>,
    {
        // Start boot
        self.fsm.start_boot(recovery_mode)?;

        // Init phase
        init_fn(&self.fsm)?;
        self.fsm.transition_to(BootPhase::Migrate)?;

        // Migrate phase
        migrate_fn(&self.fsm)?;
        let next_phase = match recovery_mode {
            RecoveryMode::Normal => BootPhase::Recover,
            RecoveryMode::ReadOnly => BootPhase::ReadOnlyRecovery,
            RecoveryMode::Force => BootPhase::ForceRecovery,
        };
        self.fsm.transition_to(next_phase)?;

        // Recover phase
        let hints = recover_fn(&self.fsm)?;
        self.fsm.set_preload_hints(hints.clone());

        // Warmup phase (optional skip)
        if hints.working_set_bytes > 0 || !hints.indexes.is_empty() {
            self.fsm.transition_to(BootPhase::Warmup)?;
            warmup_fn(&self.fsm, hints)?;
        }

        // Ready
        self.fsm.transition_to(BootPhase::Ready)?;

        Ok(())
    }

    /// Initiate graceful shutdown
    pub fn shutdown(&self) -> Result<(), BootError> {
        self.fsm.transition_to(BootPhase::ShuttingDown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_fsm_transitions() {
        let fsm = BootStateMachine::with_defaults();

        // Start boot
        assert!(fsm.start_boot(RecoveryMode::Normal).is_ok());
        assert_eq!(fsm.current_phase(), BootPhase::Init);

        // Progress through phases
        assert!(fsm.transition_to(BootPhase::Migrate).is_ok());
        assert!(fsm.transition_to(BootPhase::Recover).is_ok());
        assert!(fsm.transition_to(BootPhase::Warmup).is_ok());
        assert!(fsm.transition_to(BootPhase::Ready).is_ok());

        assert!(fsm.is_ready());
        assert!(fsm.is_alive());
    }

    #[test]
    fn test_invalid_transition() {
        let fsm = BootStateMachine::with_defaults();
        fsm.start_boot(RecoveryMode::Normal).unwrap();

        // Can't skip to Ready from Init
        assert!(fsm.transition_to(BootPhase::Ready).is_err());
    }

    #[test]
    fn test_health_status() {
        let fsm = BootStateMachine::with_defaults();
        fsm.start_boot(RecoveryMode::Normal).unwrap();

        let status = fsm.health_status();
        assert!(!status.is_ready);
        assert!(status.is_alive);
        assert!(status.is_booting);
        assert_eq!(status.phase_name, "init");
    }

    #[test]
    fn test_progress_tracking() {
        let fsm = BootStateMachine::with_defaults();
        fsm.start_boot(RecoveryMode::Normal).unwrap();

        fsm.record_wal_progress(100, 4096);
        assert_eq!(fsm.metrics().wal_records_replayed.load(Ordering::Relaxed), 100);
        assert_eq!(fsm.metrics().wal_bytes_processed.load(Ordering::Relaxed), 4096);
    }

    #[test]
    fn test_kubernetes_budgets() {
        let budgets = BootBudgets::for_kubernetes(600); // 10 minutes
        assert!(budgets.recover_budget >= Duration::from_secs(300));
        assert!(budgets.total_budget == Duration::from_secs(600));
    }
}
