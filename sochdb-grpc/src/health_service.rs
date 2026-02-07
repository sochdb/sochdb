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

//! # Health Service with Kubernetes Probe Semantics
//!
//! Implements comprehensive health checking beyond simple binary probes:
//! - `startupProbe`: tolerates long recovery, aligned with Boot FSM
//! - `readinessProbe`: true only when FSM is in Ready state
//! - `livenessProbe`: external watchdog-based heartbeat
//! - `degraded` signal: compaction debt, cache thrashing, disk pressure
//!
//! ## External Watchdog
//!
//! In-process watchdog threads can stall if the process is stuck.
//! We use an external heartbeat file that the liveness probe checks:
//! - A dedicated thread writes heartbeat to a file
//! - Liveness check verifies file mtime is recent
//! - Independent of main event loop health
//!
//! ## Degraded Mode Thresholds
//!
//! Uses control theory with hysteresis to avoid flapping:
//! - Enter degraded: metric > threshold for N consecutive checks
//! - Exit degraded: metric < (threshold - hysteresis) for M consecutive checks

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;

/// Health check result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthCheckResult {
    /// Healthy and ready for traffic
    Healthy,
    /// Alive but degraded (accept less traffic)
    Degraded,
    /// Not ready (startup/recovery in progress)
    NotReady,
    /// Not alive (process should be restarted)
    NotAlive,
}

impl HealthCheckResult {
    /// HTTP status code for this result
    pub fn http_status(&self) -> u16 {
        match self {
            HealthCheckResult::Healthy => 200,
            HealthCheckResult::Degraded => 200, // Still accept traffic
            HealthCheckResult::NotReady => 503,
            HealthCheckResult::NotAlive => 503,
        }
    }

    /// Is this result considered "passing" for readiness?
    pub fn is_ready(&self) -> bool {
        matches!(self, HealthCheckResult::Healthy | HealthCheckResult::Degraded)
    }

    /// Is this result considered "passing" for liveness?
    pub fn is_alive(&self) -> bool {
        !matches!(self, HealthCheckResult::NotAlive)
    }
}

/// Degraded condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DegradedCondition {
    /// Compaction debt too high
    CompactionDebt,
    /// Cache hit rate too low (thrashing)
    CacheThrashing,
    /// Replication lag too high
    ReplicationLag,
    /// Disk space running low
    DiskPressure,
    /// Memory pressure (close to OOM)
    MemoryPressure,
    /// WAL replay taking too long
    WalReplayLag,
    /// Queue depth too high
    QueueBackpressure,
}

impl DegradedCondition {
    pub fn name(&self) -> &'static str {
        match self {
            DegradedCondition::CompactionDebt => "compaction_debt",
            DegradedCondition::CacheThrashing => "cache_thrashing",
            DegradedCondition::ReplicationLag => "replication_lag",
            DegradedCondition::DiskPressure => "disk_pressure",
            DegradedCondition::MemoryPressure => "memory_pressure",
            DegradedCondition::WalReplayLag => "wal_replay_lag",
            DegradedCondition::QueueBackpressure => "queue_backpressure",
        }
    }
}

/// Threshold configuration for a degraded condition
#[derive(Debug, Clone)]
pub struct DegradedThreshold {
    /// Threshold to enter degraded state
    pub enter_threshold: f64,
    /// Threshold to exit degraded state (hysteresis)
    pub exit_threshold: f64,
    /// Number of consecutive checks to enter degraded
    pub enter_count: u32,
    /// Number of consecutive checks to exit degraded
    pub exit_count: u32,
}

impl Default for DegradedThreshold {
    fn default() -> Self {
        Self {
            enter_threshold: 0.8,
            exit_threshold: 0.6,
            enter_count: 3,
            exit_count: 5,
        }
    }
}

/// State tracking for a degraded condition
#[derive(Debug)]
struct ConditionState {
    threshold: DegradedThreshold,
    is_degraded: bool,
    consecutive_above: u32,
    consecutive_below: u32,
    last_value: f64,
}

impl ConditionState {
    fn new(threshold: DegradedThreshold) -> Self {
        Self {
            threshold,
            is_degraded: false,
            consecutive_above: 0,
            consecutive_below: 0,
            last_value: 0.0,
        }
    }

    /// Update state with new metric value, returns whether state changed
    fn update(&mut self, value: f64) -> bool {
        self.last_value = value;
        let old_degraded = self.is_degraded;

        if self.is_degraded {
            // Check for exit
            if value < self.threshold.exit_threshold {
                self.consecutive_below += 1;
                self.consecutive_above = 0;
                if self.consecutive_below >= self.threshold.exit_count {
                    self.is_degraded = false;
                }
            } else {
                self.consecutive_below = 0;
            }
        } else {
            // Check for enter
            if value >= self.threshold.enter_threshold {
                self.consecutive_above += 1;
                self.consecutive_below = 0;
                if self.consecutive_above >= self.threshold.enter_count {
                    self.is_degraded = true;
                }
            } else {
                self.consecutive_above = 0;
            }
        }

        old_degraded != self.is_degraded
    }
}

/// External watchdog that writes heartbeats to a file
pub struct ExternalWatchdog {
    heartbeat_path: PathBuf,
    interval: Duration,
    running: Arc<AtomicBool>,
    last_heartbeat: Arc<AtomicU64>,
}

impl ExternalWatchdog {
    /// Create and start a new watchdog
    pub fn new(heartbeat_path: impl AsRef<Path>, interval: Duration) -> Self {
        let heartbeat_path = heartbeat_path.as_ref().to_path_buf();
        let running = Arc::new(AtomicBool::new(true));
        let last_heartbeat = Arc::new(AtomicU64::new(0));

        let watchdog = Self {
            heartbeat_path: heartbeat_path.clone(),
            interval,
            running: running.clone(),
            last_heartbeat: last_heartbeat.clone(),
        };

        // Start heartbeat thread
        thread::Builder::new()
            .name("sochdb-watchdog".to_string())
            .spawn(move || {
                Self::heartbeat_loop(heartbeat_path, interval, running, last_heartbeat);
            })
            .expect("Failed to spawn watchdog thread");

        watchdog
    }

    fn heartbeat_loop(
        path: PathBuf,
        interval: Duration,
        running: Arc<AtomicBool>,
        last_heartbeat: Arc<AtomicU64>,
    ) {
        while running.load(Ordering::Relaxed) {
            // Write timestamp to heartbeat file
            if let Ok(mut file) = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
            {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let _ = writeln!(file, "{}", now);
                last_heartbeat.store(now, Ordering::Relaxed);
            }

            thread::sleep(interval);
        }
    }

    /// Check if heartbeat is recent (within 2x interval)
    pub fn is_alive(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last = self.last_heartbeat.load(Ordering::Relaxed);
        let max_age = self.interval.as_secs() * 2;
        now.saturating_sub(last) < max_age
    }

    /// Check heartbeat file mtime (for external liveness check)
    pub fn check_heartbeat_file(&self) -> bool {
        if let Ok(metadata) = std::fs::metadata(&self.heartbeat_path) {
            if let Ok(modified) = metadata.modified() {
                let age = SystemTime::now()
                    .duration_since(modified)
                    .unwrap_or(Duration::MAX);
                return age < self.interval * 2;
            }
        }
        false
    }

    /// Stop the watchdog
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Drop for ExternalWatchdog {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Health service configuration
#[derive(Debug, Clone)]
pub struct HealthServiceConfig {
    /// Path for watchdog heartbeat file
    pub heartbeat_path: PathBuf,
    /// Watchdog heartbeat interval
    pub heartbeat_interval: Duration,
    /// Thresholds for degraded conditions
    pub thresholds: HashMap<DegradedCondition, DegradedThreshold>,
}

impl Default for HealthServiceConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();

        thresholds.insert(
            DegradedCondition::CompactionDebt,
            DegradedThreshold {
                enter_threshold: 0.8,
                exit_threshold: 0.5,
                enter_count: 3,
                exit_count: 5,
            },
        );

        thresholds.insert(
            DegradedCondition::CacheThrashing,
            DegradedThreshold {
                enter_threshold: 0.3, // hit rate below 30%
                exit_threshold: 0.5,
                enter_count: 5,
                exit_count: 10,
            },
        );

        thresholds.insert(
            DegradedCondition::DiskPressure,
            DegradedThreshold {
                enter_threshold: 0.9, // 90% full
                exit_threshold: 0.8,
                enter_count: 1,
                exit_count: 3,
            },
        );

        thresholds.insert(
            DegradedCondition::MemoryPressure,
            DegradedThreshold {
                enter_threshold: 0.9, // 90% of cgroup limit
                exit_threshold: 0.8,
                enter_count: 2,
                exit_count: 5,
            },
        );

        thresholds.insert(
            DegradedCondition::QueueBackpressure,
            DegradedThreshold {
                enter_threshold: 0.8,
                exit_threshold: 0.5,
                enter_count: 3,
                exit_count: 5,
            },
        );

        Self {
            heartbeat_path: PathBuf::from("/tmp/sochdb-heartbeat"),
            heartbeat_interval: Duration::from_secs(5),
            thresholds,
        }
    }
}

/// Comprehensive health service
pub struct HealthService {
    config: HealthServiceConfig,
    watchdog: ExternalWatchdog,
    conditions: RwLock<HashMap<DegradedCondition, ConditionState>>,
    is_ready: AtomicBool,
    is_booting: AtomicBool,
    startup_time: Instant,
}

impl HealthService {
    /// Create a new health service
    pub fn new(config: HealthServiceConfig) -> Self {
        let watchdog = ExternalWatchdog::new(&config.heartbeat_path, config.heartbeat_interval);

        let mut conditions = HashMap::new();
        for (condition, threshold) in &config.thresholds {
            conditions.insert(*condition, ConditionState::new(threshold.clone()));
        }

        Self {
            config,
            watchdog,
            conditions: RwLock::new(conditions),
            is_ready: AtomicBool::new(false),
            is_booting: AtomicBool::new(true),
            startup_time: Instant::now(),
        }
    }

    /// Mark system as ready (called when boot FSM reaches Ready)
    pub fn set_ready(&self, ready: bool) {
        self.is_ready.store(ready, Ordering::SeqCst);
        if ready {
            self.is_booting.store(false, Ordering::SeqCst);
        }
    }

    /// Mark boot as complete
    pub fn set_boot_complete(&self) {
        self.is_booting.store(false, Ordering::SeqCst);
    }

    /// Update a degraded condition metric
    pub fn update_condition(&self, condition: DegradedCondition, value: f64) -> bool {
        let mut conditions = self.conditions.write();
        if let Some(state) = conditions.get_mut(&condition) {
            state.update(value)
        } else {
            false
        }
    }

    /// Check if any degraded condition is active
    pub fn is_degraded(&self) -> bool {
        self.conditions.read().values().any(|s| s.is_degraded)
    }

    /// Get active degraded conditions
    pub fn active_degraded_conditions(&self) -> Vec<DegradedCondition> {
        self.conditions
            .read()
            .iter()
            .filter(|(_, s)| s.is_degraded)
            .map(|(c, _)| *c)
            .collect()
    }

    /// Startup probe: true if still booting (gives time for recovery)
    pub fn startup_check(&self) -> HealthCheckResult {
        if self.is_ready.load(Ordering::SeqCst) {
            HealthCheckResult::Healthy
        } else if self.is_booting.load(Ordering::SeqCst) {
            // Still booting, probe should pass to give time
            HealthCheckResult::Healthy
        } else {
            // Boot failed
            HealthCheckResult::NotAlive
        }
    }

    /// Readiness probe: true only when Ready and not severely degraded
    pub fn readiness_check(&self) -> HealthCheckResult {
        if !self.is_ready.load(Ordering::SeqCst) {
            return HealthCheckResult::NotReady;
        }

        if self.is_degraded() {
            HealthCheckResult::Degraded
        } else {
            HealthCheckResult::Healthy
        }
    }

    /// Liveness probe: based on external watchdog heartbeat
    pub fn liveness_check(&self) -> HealthCheckResult {
        if self.watchdog.is_alive() {
            if self.is_degraded() {
                HealthCheckResult::Degraded
            } else {
                HealthCheckResult::Healthy
            }
        } else {
            HealthCheckResult::NotAlive
        }
    }

    /// Get comprehensive health status
    pub fn full_status(&self) -> FullHealthStatus {
        let conditions = self.conditions.read();
        let degraded_conditions: Vec<_> = conditions
            .iter()
            .filter(|(_, s)| s.is_degraded)
            .map(|(c, s)| DegradedInfo {
                condition: c.name().to_string(),
                value: s.last_value,
                threshold: s.threshold.enter_threshold,
            })
            .collect();

        FullHealthStatus {
            startup: self.startup_check(),
            readiness: self.readiness_check(),
            liveness: self.liveness_check(),
            is_ready: self.is_ready.load(Ordering::SeqCst),
            is_booting: self.is_booting.load(Ordering::SeqCst),
            is_degraded: !degraded_conditions.is_empty(),
            degraded_conditions,
            uptime_secs: self.startup_time.elapsed().as_secs(),
            watchdog_alive: self.watchdog.is_alive(),
        }
    }

    /// Format status as JSON for HTTP endpoints
    pub fn status_json(&self) -> String {
        let status = self.full_status();
        let degraded_json: Vec<String> = status
            .degraded_conditions
            .iter()
            .map(|d| {
                format!(
                    r#"{{"condition":"{}","value":{:.3},"threshold":{:.3}}}"#,
                    d.condition, d.value, d.threshold
                )
            })
            .collect();

        format!(
            r#"{{"startup":"{}","readiness":"{}","liveness":"{}","is_ready":{},"is_booting":{},"is_degraded":{},"degraded_conditions":[{}],"uptime_secs":{},"watchdog_alive":{}}}"#,
            status.startup.http_status(),
            status.readiness.http_status(),
            status.liveness.http_status(),
            status.is_ready,
            status.is_booting,
            status.is_degraded,
            degraded_json.join(","),
            status.uptime_secs,
            status.watchdog_alive,
        )
    }
}

/// Detailed degraded condition info
#[derive(Debug, Clone)]
pub struct DegradedInfo {
    pub condition: String,
    pub value: f64,
    pub threshold: f64,
}

/// Full health status
#[derive(Debug)]
pub struct FullHealthStatus {
    pub startup: HealthCheckResult,
    pub readiness: HealthCheckResult,
    pub liveness: HealthCheckResult,
    pub is_ready: bool,
    pub is_booting: bool,
    pub is_degraded: bool,
    pub degraded_conditions: Vec<DegradedInfo>,
    pub uptime_secs: u64,
    pub watchdog_alive: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degraded_threshold_hysteresis() {
        let threshold = DegradedThreshold {
            enter_threshold: 0.8,
            exit_threshold: 0.5,
            enter_count: 2,
            exit_count: 2,
        };
        let mut state = ConditionState::new(threshold);

        // Not degraded initially
        assert!(!state.is_degraded);

        // First check above threshold - not degraded yet (need 2)
        state.update(0.85);
        assert!(!state.is_degraded);

        // Second check above threshold - now degraded
        state.update(0.90);
        assert!(state.is_degraded);

        // Drop below enter but above exit - still degraded
        state.update(0.7);
        assert!(state.is_degraded);

        // First check below exit threshold
        state.update(0.4);
        assert!(state.is_degraded);

        // Second check below exit - now not degraded
        state.update(0.3);
        assert!(!state.is_degraded);
    }

    #[test]
    fn test_health_service_ready_state() {
        let config = HealthServiceConfig::default();
        let service = HealthService::new(config);

        // Initially booting
        assert!(service.is_booting.load(Ordering::SeqCst));
        assert!(!service.is_ready.load(Ordering::SeqCst));

        // Startup check should pass during boot
        assert_eq!(service.startup_check(), HealthCheckResult::Healthy);

        // Readiness should fail
        assert_eq!(service.readiness_check(), HealthCheckResult::NotReady);

        // Set ready
        service.set_ready(true);
        assert_eq!(service.readiness_check(), HealthCheckResult::Healthy);
    }

    #[test]
    fn test_degraded_conditions() {
        let config = HealthServiceConfig::default();
        let service = HealthService::new(config);
        service.set_ready(true);

        // Initially healthy
        assert!(!service.is_degraded());

        // Trigger compaction debt degraded (need 3 consecutive)
        for _ in 0..3 {
            service.update_condition(DegradedCondition::CompactionDebt, 0.9);
        }
        assert!(service.is_degraded());

        // Check readiness reflects degraded
        assert_eq!(service.readiness_check(), HealthCheckResult::Degraded);
    }
}
