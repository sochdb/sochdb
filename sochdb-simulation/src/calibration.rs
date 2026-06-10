//! Benchmark-calibrated latency profiles.
//!
//! Base latencies are derived from sochdb-bench results (2026-06-08, 10K scale).
//! Distributed overhead is added on top for gRPC paths.

use crate::topology::Operation;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct LatencyProfile {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub throughput_ops_sec: f64,
}

/// Standalone embedded measurements (macOS aarch64, throughput_optimized).
static CALIBRATED_STANDALONE: &[(&str, LatencyProfile)] = &[
    ("oltp_seq_write", lp(5.63, 5.34, 9.54, 177_772.0)),
    ("oltp_seq_read", lp(0.11, 0.08, 0.29, 9_046_131.0)),
    ("oltp_rand_read", lp(0.18, 0.17, 0.46, 5_520_446.0)),
    ("oltp_batch_write", lp(0.43, 0.42, 0.55, 2_320_432.0)),
    ("oltp_delete", lp(4.70, 4.63, 6.58, 212_694.0)),
    ("analytics_bulk_insert", lp(0.84, 0.80, 0.99, 1_185_911.0)),
    ("analytics_queries", lp(28.38, 4.0, 2054.0, 35_240.0)),
    ("vector_insert", lp(0.62, 0.56, 1.35, 1_600_929.0)),
    ("vector_search", lp(314.29, 298.0, 317.0, 3_182.0)),
    ("mixed_80r_20w", lp(1.20, 0.33, 5.17, 831_143.0)),
];

/// Distributed gRPC measurements (Hetzner AX41).
static CALIBRATED_DISTRIBUTED: &[(&str, LatencyProfile)] = &[
    ("grpc_kv_put", lp(5000.0, 5000.0, 15_000.0, 10_000.0)),
    ("grpc_kv_get", lp(500.0, 500.0, 10_000.0, 50_000.0)),
    ("grpc_vector_insert_10k", lp(200.0, 200.0, 500.0, 5_033.0)),
    ("grpc_hnsw_search_10k", lp(1720.0, 1700.0, 6250.0, 581.0)),
    ("grpc_hnsw_search_50k", lp(2670.0, 2600.0, 8000.0, 374.0)),
    ("grpc_hnsw_search_3_5m", lp(1970.0, 1870.0, 6250.0, 507.0)),
    ("grpc_hnsw_search_concurrent_8", lp(4000.0, 4000.0, 12_000.0, 1_964.0)),
    ("grpc_hnsw_search_concurrent_32", lp(15_000.0, 15_000.0, 30_000.0, 2_072.0)),
];

const fn lp(mean: f64, p50: f64, p99: f64, throughput: f64) -> LatencyProfile {
    LatencyProfile {
        mean_us: mean,
        p50_us: p50,
        p99_us: p99,
        throughput_ops_sec: throughput,
    }
}

pub struct CalibrationTable {
    standalone: HashMap<String, LatencyProfile>,
    distributed: HashMap<String, LatencyProfile>,
    /// Fixed gRPC stack overhead added when projecting standalone → distributed.
    pub grpc_overhead_us: f64,
}

impl Default for CalibrationTable {
    fn default() -> Self {
        let mut standalone = HashMap::new();
        for (k, v) in CALIBRATED_STANDALONE {
            standalone.insert(k.to_string(), *v);
        }
        let mut distributed = HashMap::new();
        for (k, v) in CALIBRATED_DISTRIBUTED {
            distributed.insert(k.to_string(), *v);
        }
        Self {
            standalone,
            distributed,
            grpc_overhead_us: 315.0, // measured: distributed_kv_get - standalone_read
        }
    }
}

impl CalibrationTable {
    pub fn profile_for(&self, workload: &str, is_distributed: bool) -> Option<LatencyProfile> {
        if is_distributed {
            self.distributed.get(workload).copied()
        } else {
            self.standalone.get(workload).copied()
        }
    }

    /// Scale a profile for environment changes (vector count, concurrency, network).
    pub fn scale_profile(
        &self,
        base: LatencyProfile,
        operation: Operation,
        vector_count: u64,
        concurrent_clients: u32,
        network_rtt_us: f64,
        is_distributed: bool,
    ) -> LatencyProfile {
        let mut mean = base.mean_us;
        let mut p50 = base.p50_us;
        let mut p99 = base.p99_us;
        let mut throughput = base.throughput_ops_sec;

        // Vector count scaling for search operations
        if matches!(
            operation,
            Operation::VectorSearchBruteForce
                | Operation::VectorSearchHnsw
                | Operation::GrpcHnswSearch
        ) {
            let ref_count = match operation {
                Operation::GrpcHnswSearch => 10_000.0,
                _ => 10_000.0,
            };
            let scale = (vector_count as f64 / ref_count).max(1.0);
            if operation == Operation::VectorSearchBruteForce {
                mean *= scale;
                p50 *= scale;
                p99 *= scale;
                throughput /= scale;
            } else {
                let log_scale = (vector_count as f64).log2() / ref_count.log2();
                mean *= log_scale.max(1.0);
                p50 *= log_scale.max(1.0);
                p99 *= log_scale.max(1.0);
                throughput /= log_scale.max(1.0);
            }
        }

        // Concurrent clients boost throughput for search (diminishing returns)
        if concurrent_clients > 1
            && matches!(
                operation,
                Operation::GrpcHnswSearch | Operation::VectorSearchHnsw
            )
        {
            let boost = (concurrent_clients as f64 * 0.7).min(4.0);
            throughput *= boost;
            mean *= 1.0 + (concurrent_clients as f64 - 1.0) * 0.1;
            p99 *= 1.0 + (concurrent_clients as f64 - 1.0) * 0.3;
        }

        // Network RTT adjustment for distributed (baseline 50μs localhost)
        if is_distributed {
            let rtt_delta = network_rtt_us - 50.0;
            mean += rtt_delta * 2.0;
            p50 += rtt_delta * 2.0;
            p99 += rtt_delta * 2.0;
            if rtt_delta > 0.0 {
                throughput *= 50.0 / network_rtt_us.max(1.0);
            }
        }

        LatencyProfile {
            mean_us: mean,
            p50_us: p50,
            p99_us: p99,
            throughput_ops_sec: throughput,
        }
    }

    /// Project standalone operation to distributed by adding gRPC overhead.
    pub fn project_distributed(&self, standalone: LatencyProfile) -> LatencyProfile {
        LatencyProfile {
            mean_us: standalone.mean_us + self.grpc_overhead_us,
            p50_us: standalone.p50_us + self.grpc_overhead_us,
            p99_us: standalone.p99_us + self.grpc_overhead_us * 1.2,
            throughput_ops_sec: (1_000_000.0
                / (1_000_000.0 / standalone.throughput_ops_sec + self.grpc_overhead_us))
                .max(100.0),
        }
    }
}