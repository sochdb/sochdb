// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Regression gate evaluation for CI/CD integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gate result for a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "lowercase")]
pub enum GateResult {
    /// Metric is within acceptable bounds
    Pass,
    /// Metric shows improvement
    Improvement {
        metric: String,
        delta_pct: f64,
    },
    /// Metric is close to threshold (warning)
    Warning {
        metric: String,
        delta_pct: f64,
        threshold_pct: f64,
    },
    /// Metric exceeds threshold (failure)
    Fail {
        metric: String,
        delta_pct: f64,
        threshold_pct: f64,
    },
}

impl GateResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, GateResult::Pass | GateResult::Improvement { .. })
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, GateResult::Fail { .. })
    }
}

/// Regression gate configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionGates {
    // Performance gates
    /// Maximum allowed QPS regression (e.g., -5% = -0.05)
    #[serde(default = "default_qps_threshold")]
    pub qps_regression_threshold: f64,
    
    /// Maximum allowed p95 latency increase (e.g., +5% = 0.05)
    #[serde(default = "default_p95_threshold")]
    pub p95_latency_threshold: f64,
    
    /// Maximum allowed p99 latency increase
    #[serde(default = "default_p99_threshold")]
    pub p99_latency_threshold: f64,
    
    /// Maximum allowed p99.9 latency increase
    #[serde(default = "default_p999_threshold")]
    pub p999_latency_threshold: f64,

    // Quality gates
    /// Maximum allowed recall regression (absolute, e.g., -0.002 = -0.2%)
    #[serde(default = "default_recall_threshold")]
    pub recall_regression_threshold: f64,

    // PMU gates (Linux only)
    /// Maximum allowed branch miss rate increase
    #[serde(default = "default_branch_miss_threshold")]
    pub branch_miss_rate_threshold: f64,
    
    /// Maximum allowed L1D miss rate increase
    #[serde(default = "default_l1d_miss_threshold")]
    pub l1d_miss_rate_threshold: f64,
    
    /// Maximum allowed IPC drop
    #[serde(default = "default_ipc_threshold")]
    pub ipc_drop_threshold: f64,

    // Warning thresholds (50% of fail thresholds by default)
    #[serde(default)]
    pub warning_factor: f64,
}

fn default_qps_threshold() -> f64 { -0.05 }
fn default_p95_threshold() -> f64 { 0.05 }
fn default_p99_threshold() -> f64 { 0.10 }
fn default_p999_threshold() -> f64 { 0.15 }
fn default_recall_threshold() -> f64 { -0.002 }
fn default_branch_miss_threshold() -> f64 { 0.10 }
fn default_l1d_miss_threshold() -> f64 { 0.20 }
fn default_ipc_threshold() -> f64 { -0.10 }

impl Default for RegressionGates {
    fn default() -> Self {
        Self {
            qps_regression_threshold: default_qps_threshold(),
            p95_latency_threshold: default_p95_threshold(),
            p99_latency_threshold: default_p99_threshold(),
            p999_latency_threshold: default_p999_threshold(),
            recall_regression_threshold: default_recall_threshold(),
            branch_miss_rate_threshold: default_branch_miss_threshold(),
            l1d_miss_rate_threshold: default_l1d_miss_threshold(),
            ipc_drop_threshold: default_ipc_threshold(),
            warning_factor: 0.5,
        }
    }
}

impl RegressionGates {
    /// Create strict gates for PR checks.
    pub fn pr_strict() -> Self {
        Self {
            qps_regression_threshold: -0.03,
            p95_latency_threshold: 0.03,
            p99_latency_threshold: 0.05,
            p999_latency_threshold: 0.10,
            recall_regression_threshold: -0.001,
            ..Default::default()
        }
    }

    /// Create relaxed gates for nightly runs (allow more variance).
    pub fn nightly_relaxed() -> Self {
        Self {
            qps_regression_threshold: -0.08,
            p95_latency_threshold: 0.08,
            p99_latency_threshold: 0.12,
            p999_latency_threshold: 0.20,
            recall_regression_threshold: -0.003,
            ..Default::default()
        }
    }

    /// Evaluate a higher-is-better metric (e.g., QPS).
    pub fn check_higher_is_better(
        &self,
        name: &str,
        baseline: f64,
        current: f64,
        threshold: f64,
    ) -> GateResult {
        if baseline == 0.0 {
            return GateResult::Pass;
        }

        let delta_pct = (current - baseline) / baseline;

        if delta_pct >= 0.0 {
            // Improvement or no change
            if delta_pct > 0.05 {
                GateResult::Improvement {
                    metric: name.to_string(),
                    delta_pct: delta_pct * 100.0,
                }
            } else {
                GateResult::Pass
            }
        } else if delta_pct < threshold {
            // Regression beyond threshold
            GateResult::Fail {
                metric: name.to_string(),
                delta_pct: delta_pct * 100.0,
                threshold_pct: threshold * 100.0,
            }
        } else if delta_pct < threshold * self.warning_factor {
            // Warning zone
            GateResult::Warning {
                metric: name.to_string(),
                delta_pct: delta_pct * 100.0,
                threshold_pct: threshold * 100.0,
            }
        } else {
            GateResult::Pass
        }
    }

    /// Evaluate a lower-is-better metric (e.g., latency).
    pub fn check_lower_is_better(
        &self,
        name: &str,
        baseline: f64,
        current: f64,
        threshold: f64,
    ) -> GateResult {
        if baseline == 0.0 {
            return GateResult::Pass;
        }

        let delta_pct = (current - baseline) / baseline;

        if delta_pct <= 0.0 {
            // Improvement or no change
            if delta_pct < -0.05 {
                GateResult::Improvement {
                    metric: name.to_string(),
                    delta_pct: delta_pct * 100.0,
                }
            } else {
                GateResult::Pass
            }
        } else if delta_pct > threshold {
            // Regression beyond threshold
            GateResult::Fail {
                metric: name.to_string(),
                delta_pct: delta_pct * 100.0,
                threshold_pct: threshold * 100.0,
            }
        } else if delta_pct > threshold * self.warning_factor {
            // Warning zone
            GateResult::Warning {
                metric: name.to_string(),
                delta_pct: delta_pct * 100.0,
                threshold_pct: threshold * 100.0,
            }
        } else {
            GateResult::Pass
        }
    }

    /// Evaluate recall metric (absolute threshold).
    pub fn check_recall(&self, name: &str, baseline: f64, current: f64) -> GateResult {
        let delta = current - baseline;

        if delta >= 0.0 {
            // Improvement or no change
            if delta > 0.01 {
                GateResult::Improvement {
                    metric: name.to_string(),
                    delta_pct: delta * 100.0,
                }
            } else {
                GateResult::Pass
            }
        } else if delta < self.recall_regression_threshold {
            GateResult::Fail {
                metric: name.to_string(),
                delta_pct: delta * 100.0,
                threshold_pct: self.recall_regression_threshold * 100.0,
            }
        } else if delta < self.recall_regression_threshold * self.warning_factor {
            GateResult::Warning {
                metric: name.to_string(),
                delta_pct: delta * 100.0,
                threshold_pct: self.recall_regression_threshold * 100.0,
            }
        } else {
            GateResult::Pass
        }
    }
}

/// Complete gate evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvaluation {
    pub overall_pass: bool,
    pub results: Vec<GateResult>,
    pub summary: String,
}

impl GateEvaluation {
    /// Evaluate all metrics against a baseline.
    pub fn evaluate(
        gates: &RegressionGates,
        baseline: &MetricsSnapshot,
        current: &MetricsSnapshot,
    ) -> Self {
        let mut results = Vec::new();

        // QPS gate
        if let (Some(b), Some(c)) = (baseline.qps, current.qps) {
            results.push(gates.check_higher_is_better(
                "qps",
                b,
                c,
                gates.qps_regression_threshold,
            ));
        }

        // Latency gates
        if let (Some(b), Some(c)) = (baseline.p95_latency_ns, current.p95_latency_ns) {
            results.push(gates.check_lower_is_better(
                "p95_latency",
                b as f64,
                c as f64,
                gates.p95_latency_threshold,
            ));
        }

        if let (Some(b), Some(c)) = (baseline.p99_latency_ns, current.p99_latency_ns) {
            results.push(gates.check_lower_is_better(
                "p99_latency",
                b as f64,
                c as f64,
                gates.p99_latency_threshold,
            ));
        }

        if let (Some(b), Some(c)) = (baseline.p999_latency_ns, current.p999_latency_ns) {
            results.push(gates.check_lower_is_better(
                "p999_latency",
                b as f64,
                c as f64,
                gates.p999_latency_threshold,
            ));
        }

        // Recall gates
        for k in &[10, 20, 50] {
            let key = format!("recall@{}", k);
            if let (Some(b), Some(c)) = (
                baseline.recall.get(&key),
                current.recall.get(&key),
            ) {
                results.push(gates.check_recall(&key, *b, *c));
            }
        }

        // PMU gates
        if let (Some(b), Some(c)) = (baseline.branch_miss_rate, current.branch_miss_rate) {
            results.push(gates.check_lower_is_better(
                "branch_miss_rate",
                b,
                c,
                gates.branch_miss_rate_threshold,
            ));
        }

        if let (Some(b), Some(c)) = (baseline.l1d_miss_rate, current.l1d_miss_rate) {
            results.push(gates.check_lower_is_better(
                "l1d_miss_rate",
                b,
                c,
                gates.l1d_miss_rate_threshold,
            ));
        }

        if let (Some(b), Some(c)) = (baseline.ipc, current.ipc) {
            results.push(gates.check_higher_is_better(
                "ipc",
                b,
                c,
                gates.ipc_drop_threshold,
            ));
        }

        let failures: Vec<_> = results.iter().filter(|r| r.is_fail()).collect();
        let overall_pass = failures.is_empty();

        let summary = if overall_pass {
            "All gates passed".to_string()
        } else {
            format!("{} gate(s) failed", failures.len())
        };

        Self {
            overall_pass,
            results,
            summary,
        }
    }

    /// Get exit code for CI (0 = pass, 1 = fail).
    pub fn exit_code(&self) -> i32 {
        if self.overall_pass { 0 } else { 1 }
    }
}

/// Metrics snapshot for comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub qps: Option<f64>,
    pub insert_rate: Option<f64>,
    pub p50_latency_ns: Option<u64>,
    pub p95_latency_ns: Option<u64>,
    pub p99_latency_ns: Option<u64>,
    pub p999_latency_ns: Option<u64>,
    pub recall: HashMap<String, f64>,
    pub branch_miss_rate: Option<f64>,
    pub l1d_miss_rate: Option<f64>,
    pub ipc: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qps_gate() {
        let gates = RegressionGates::default();

        // Improvement
        let result = gates.check_higher_is_better("qps", 1000.0, 1100.0, -0.05);
        assert!(matches!(result, GateResult::Improvement { .. }));

        // Pass (within threshold)
        let result = gates.check_higher_is_better("qps", 1000.0, 980.0, -0.05);
        assert!(result.is_pass());

        // Fail (beyond threshold)
        let result = gates.check_higher_is_better("qps", 1000.0, 900.0, -0.05);
        assert!(result.is_fail());
    }

    #[test]
    fn test_latency_gate() {
        let gates = RegressionGates::default();

        // Improvement (lower latency)
        let result = gates.check_lower_is_better("p95", 1000.0, 900.0, 0.05);
        assert!(matches!(result, GateResult::Improvement { .. }));

        // Fail (higher latency beyond threshold)
        let result = gates.check_lower_is_better("p95", 1000.0, 1100.0, 0.05);
        assert!(result.is_fail());
    }

    #[test]
    fn test_recall_gate() {
        let gates = RegressionGates::default();

        // Pass (no change)
        let result = gates.check_recall("recall@10", 0.95, 0.95);
        assert!(result.is_pass());

        // Fail (regression beyond threshold)
        let result = gates.check_recall("recall@10", 0.95, 0.944);
        assert!(result.is_fail());
    }
}
