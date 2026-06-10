//! Discrete-event simulation engine.

use crate::calibration::CalibrationTable;
use crate::component::{Component, SimEnvironment};
use crate::scenario::{Scenario, ScenarioOp};
use crate::topology::{Operation, Topology};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Result of simulating a single operation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpResult {
    pub workload: String,
    pub topology: String,
    pub ops: u64,
    pub mean_latency_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub throughput_ops_sec: f64,
    pub bottleneck: String,
    pub component_breakdown: Vec<ComponentLatency>,
    /// Quality metrics (recall, MRR, etc.) when applicable.
    pub quality_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentLatency {
    pub component: String,
    pub latency_us: f64,
    pub pct_of_total: f64,
}

/// Full scenario simulation output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub scenario_name: String,
    pub topology: String,
    pub operations: Vec<OpResult>,
    pub total_simulated_ops: u64,
}

pub struct SimulationEngine {
    rng: ChaCha8Rng,
    /// Jitter factor for latency variance (0.0 = deterministic).
    jitter: f64,
    calibration: CalibrationTable,
    /// Use benchmark-calibrated profiles (default: true).
    calibrated: bool,
}

impl SimulationEngine {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            jitter: 0.10,
            calibration: CalibrationTable::default(),
            calibrated: true,
        }
    }

    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    pub fn run_scenario(&mut self, scenario: &Scenario) -> ScenarioResult {
        let mut results = Vec::new();
        let mut total_ops = 0u64;

        for op_spec in &scenario.operations {
            if op_spec.read_ratio.is_some() {
                let result = if op_spec.workload_override.is_some() {
                    self.simulate_operation_with_workload(
                        scenario.topology,
                        op_spec.operation,
                        op_spec.ops,
                        &scenario.environment,
                        op_spec.workload_override.as_deref(),
                    )
                } else {
                    self.simulate_mixed(scenario.topology, op_spec, &scenario.environment)
                };
                total_ops += op_spec.ops;
                results.push(result);
            } else {
                let result = self.simulate_operation_with_workload(
                    scenario.topology,
                    op_spec.operation,
                    op_spec.ops,
                    &scenario.environment,
                    op_spec.workload_override.as_deref(),
                );
                total_ops += op_spec.ops;
                results.push(result);
            }
        }

        ScenarioResult {
            scenario_id: scenario.id.clone(),
            scenario_name: scenario.name.clone(),
            topology: scenario.topology.name().into(),
            operations: results,
            total_simulated_ops: total_ops,
        }
    }

    pub fn simulate_operation(
        &mut self,
        topology: Topology,
        operation: Operation,
        ops: u64,
        env: &SimEnvironment,
    ) -> OpResult {
        self.simulate_operation_with_workload(topology, operation, ops, env, None)
    }

    pub fn simulate_operation_with_workload(
        &mut self,
        topology: Topology,
        operation: Operation,
        ops: u64,
        env: &SimEnvironment,
        workload_override: Option<&str>,
    ) -> OpResult {
        let path = operation.component_path(topology);
        let breakdown = self.component_breakdown(&path, env);
        let is_distributed = topology == Topology::Distributed;

        let (mean, p50, p99, throughput, bottleneck) = if self.calibrated {
            let workload = workload_override.unwrap_or_else(|| operation.workload_id());
            let profile = self
                .calibration
                .profile_for(workload, is_distributed)
                .or_else(|| {
                    // Fall back: project from standalone measurement
                    self.calibration
                        .profile_for(workload, false)
                        .map(|p| {
                            if is_distributed {
                                self.calibration.project_distributed(p)
                            } else {
                                p
                            }
                        })
                });

            if let Some(base) = profile {
                let exact = self.calibration.profile_for(workload, is_distributed).is_some();
                let scaled = if exact {
                    base
                } else {
                    self.calibration.scale_profile(
                        base,
                        operation,
                        env.vector_count,
                        env.concurrent_clients,
                        env.network_rtt_us,
                        is_distributed,
                    )
                };
                let samples =
                    self.monte_carlo_samples(scaled.mean_us, ops.min(10_000) as usize);
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                (
                    mean,
                    scaled.p50_us,
                    scaled.p99_us,
                    scaled.throughput_ops_sec,
                    self.identify_bottleneck(&path, env),
                )
            } else {
                self.analytical_latency(&breakdown, &path, env, ops)
            }
        } else {
            self.analytical_latency(&breakdown, &path, env, ops)
        };

        let quality_score = if operation == Operation::ContextQuery {
            Some(self.simulate_retrieval_quality(env))
        } else {
            None
        };

        OpResult {
            workload: workload_override
                .unwrap_or_else(|| operation.workload_id())
                .into(),
            topology: topology.name().into(),
            ops,
            mean_latency_us: mean,
            p50_us: p50,
            p99_us: p99,
            throughput_ops_sec: throughput,
            bottleneck,
            component_breakdown: breakdown,
            quality_score,
        }
    }

    fn analytical_latency(
        &mut self,
        breakdown: &[ComponentLatency],
        path: &[Component],
        env: &SimEnvironment,
        ops: u64,
    ) -> (f64, f64, f64, f64, String) {
        let total_latency = breakdown.iter().map(|c| c.latency_us).sum::<f64>();
        let bottleneck = self.identify_bottleneck(path, env);
        let bottleneck_throughput = path
            .iter()
            .map(|c| env.effective_throughput(*c))
            .fold(f64::INFINITY, f64::min);
        let samples = self.monte_carlo_samples(total_latency, ops.min(10_000) as usize);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let throughput = if total_latency > 0.0 {
            (1_000_000.0 / total_latency).min(bottleneck_throughput)
        } else {
            bottleneck_throughput
        };
        (
            mean,
            percentile(&samples, 0.50),
            percentile(&samples, 0.99),
            throughput,
            bottleneck,
        )
    }

    fn identify_bottleneck(&self, path: &[Component], env: &SimEnvironment) -> String {
        path.iter()
            .min_by(|a, b| {
                env.effective_throughput(**a)
                    .partial_cmp(&env.effective_throughput(**b))
                    .unwrap()
            })
            .map(|c| c.display_name().to_string())
            .unwrap_or_else(|| "unknown".into())
    }

    fn simulate_mixed(
        &mut self,
        topology: Topology,
        op_spec: &ScenarioOp,
        env: &SimEnvironment,
    ) -> OpResult {
        let read_ratio = op_spec.read_ratio.unwrap_or(0.8);
        let read_ops = (op_spec.ops as f64 * read_ratio) as u64;
        let write_ops = op_spec.ops - read_ops;

        let read_result = self.simulate_operation(topology, Operation::PointRead, read_ops, env);
        let write_result = self.simulate_operation(topology, Operation::PointWrite, write_ops, env);

        let weighted_latency = read_result.mean_latency_us * read_ratio
            + write_result.mean_latency_us * (1.0 - read_ratio);

        let throughput = op_spec.ops as f64 / (op_spec.ops as f64 / read_result.throughput_ops_sec
            + op_spec.ops as f64 / write_result.throughput_ops_sec);

        OpResult {
            workload: "mixed_80r_20w".into(),
            topology: topology.name().into(),
            ops: op_spec.ops,
            mean_latency_us: weighted_latency,
            p50_us: read_result.p50_us * read_ratio + write_result.p50_us * (1.0 - read_ratio),
            p99_us: write_result.p99_us.max(read_result.p99_us),
            throughput_ops_sec: throughput,
            bottleneck: format!(
                "read:{} write:{}",
                read_result.bottleneck, write_result.bottleneck
            ),
            component_breakdown: read_result.component_breakdown,
            quality_score: None,
        }
    }

    fn component_breakdown(&mut self, path: &[Component], env: &SimEnvironment) -> Vec<ComponentLatency> {
        let total: f64 = path.iter().map(|c| env.effective_latency_us(*c)).sum();
        path.iter()
            .map(|c| {
                let lat = env.effective_latency_us(*c);
                ComponentLatency {
                    component: c.display_name().into(),
                    latency_us: lat,
                    pct_of_total: if total > 0.0 { lat / total * 100.0 } else { 0.0 },
                }
            })
            .collect()
    }

    fn monte_carlo_samples(&mut self, base_latency: f64, n: usize) -> Vec<f64> {
        (0..n.max(100))
            .map(|_| {
                let jitter = 1.0 + self.rng.gen_range(-self.jitter..self.jitter);
                // Log-normal tail for p99
                let tail = if self.rng.gen::<f64>() > 0.99 {
                    1.0 + self.rng.gen_range(0.0..2.0)
                } else {
                    1.0
                };
                base_latency * jitter * tail
            })
            .collect()
    }

    /// Model retrieval quality from HNSW parameters (matches SciFact sweep).
    fn simulate_retrieval_quality(&mut self, env: &SimEnvironment) -> f64 {
        let base_recall = 0.80;
        let m_bonus = if env.ef_search >= 128 { 0.004 } else { 0.0 };
        let ef_bonus = (env.ef_search as f64 / 128.0) * 0.002;
        let scale_penalty = (env.vector_count as f64 / 100_000.0).min(0.02);
        let noise = self.rng.gen_range(-0.005..0.005);
        (base_recall + m_bonus + ef_bonus - scale_penalty + noise).clamp(0.0, 1.0)
    }
}

fn percentile(sorted_samples: &[f64], p: f64) -> f64 {
    let mut s = sorted_samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((s.len() as f64 * p) as usize).min(s.len().saturating_sub(1));
    s[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_read_faster_than_distributed() {
        let mut engine = SimulationEngine::new(42);
        let env = SimEnvironment::default();

        let standalone = engine.simulate_operation(
            Topology::Standalone,
            Operation::PointRead,
            1000,
            &env,
        );
        let distributed = engine.simulate_operation(
            Topology::Distributed,
            Operation::GrpcKvGet,
            1000,
            &env,
        );

        assert!(standalone.mean_latency_us < distributed.mean_latency_us);
        assert!(standalone.throughput_ops_sec > distributed.throughput_ops_sec);
    }

    #[test]
    fn hnsw_latency_scales_with_vector_count() {
        let mut engine = SimulationEngine::new(42);
        let small = SimEnvironment { vector_count: 10_000, ..Default::default() };
        let large = SimEnvironment { vector_count: 3_500_000, ef_search: 64, ..Default::default() };

        let small_r = engine.simulate_operation_with_workload(
            Topology::Distributed,
            Operation::GrpcHnswSearch,
            100,
            &small,
            Some("grpc_hnsw_search_10k"),
        );
        let large_r = engine.simulate_operation_with_workload(
            Topology::Distributed,
            Operation::GrpcHnswSearch,
            100,
            &large,
            Some("grpc_hnsw_search_3_5m"),
        );

        assert!(large_r.mean_latency_us > small_r.mean_latency_us);
        assert!(large_r.throughput_ops_sec < small_r.throughput_ops_sec);
    }
}