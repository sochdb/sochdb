//! Simulation scenarios — workload definitions and full-stack coverage.

use crate::component::SimEnvironment;
use crate::topology::{Operation, Topology};
use serde::{Deserialize, Serialize};

/// A runnable simulation scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub id: String,
    pub name: String,
    pub topology: Topology,
    pub operations: Vec<ScenarioOp>,
    pub environment: SimEnvironment,
    pub description: String,
}

/// Single operation within a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioOp {
    pub operation: Operation,
    pub ops: u64,
    /// Read ratio for mixed workloads (0.0–1.0).
    pub read_ratio: Option<f64>,
    /// Override workload ID for expected-score lookup (e.g. grpc_hnsw_search_3_5m).
    pub workload_override: Option<String>,
}

impl Scenario {
    pub fn standalone_full_stack() -> Self {
        Self {
            id: "standalone_full".into(),
            name: "Standalone Full Stack".into(),
            topology: Topology::Standalone,
            description: "All embedded workloads matching sochdb-bench 10K scale".into(),
            environment: SimEnvironment::default(),
            operations: vec![
                ScenarioOp {
                    operation: Operation::PointWrite,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::PointRead,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::BatchWrite,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::Delete,
                    ops: 2_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::AnalyticsInsert,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::AnalyticsQuery,
                    ops: 80,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::VectorInsert,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::VectorSearchBruteForce,
                    ops: 200,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::ContextQuery,
                    ops: 100,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::HybridFusion,
                    ops: 100,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::McpToolCall,
                    ops: 50,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::TemporalGraphQuery,
                    ops: 200,
                    read_ratio: None,
                    workload_override: None,
                },
            ],
        }
    }

    pub fn distributed_grpc() -> Self {
        Self {
            id: "distributed_grpc".into(),
            name: "Distributed gRPC Cluster".into(),
            topology: Topology::Distributed,
            description: "gRPC server workloads from Hetzner AX41 benchmarks".into(),
            environment: SimEnvironment {
                network_rtt_us: 50.0,
                replication_factor: 1,
                concurrent_clients: 1,
                vector_count: 10_000,
                vector_dim: 768,
                ef_search: 64,
                cache_warm: true,
                group_commit: true,
            },
            operations: vec![
                ScenarioOp {
                    operation: Operation::GrpcKvPut,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::GrpcKvGet,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::GrpcVectorInsert,
                    ops: 10_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::GrpcHnswSearch,
                    ops: 1_000,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::ContextQuery,
                    ops: 300,
                    read_ratio: None,
                    workload_override: None,
                },
                ScenarioOp {
                    operation: Operation::McpToolCall,
                    ops: 100,
                    read_ratio: None,
                    workload_override: None,
                },
            ],
        }
    }

    pub fn distributed_large_scale() -> Self {
        let mut s = Self::distributed_grpc();
        s.id = "distributed_large".into();
        s.name = "Distributed Large-Scale HNSW".into();
        s.description = "3.5M vectors × 768D on commodity server".into();
        s.environment.vector_count = 3_495_253;
        s.environment.ef_search = 64;
        s.operations = vec![
            ScenarioOp {
                operation: Operation::GrpcHnswSearch,
                ops: 1_000,
                read_ratio: None,
                workload_override: Some("grpc_hnsw_search_3_5m".into()),
            },
            ScenarioOp {
                operation: Operation::GrpcVectorInsert,
                ops: 100_000,
                read_ratio: None,
                workload_override: None,
            },
        ];
        s
    }

    pub fn distributed_concurrent() -> Self {
        let mut s = Self::distributed_grpc();
        s.id = "distributed_concurrent".into();
        s.name = "Distributed Concurrent Search".into();
        s.description = "Parallel gRPC streams (c=8, c=32)".into();
        s.environment.concurrent_clients = 8;
        s.operations = vec![ScenarioOp {
            operation: Operation::GrpcHnswSearch,
            ops: 1_000,
            read_ratio: None,
            workload_override: Some("grpc_hnsw_search_concurrent_8".into()),
        }];
        s
    }

    pub fn retrieval_quality() -> Self {
        Self {
            id: "retrieval_quality".into(),
            name: "Retrieval Quality (SciFact)".into(),
            topology: Topology::Distributed,
            description: "BEIR SciFact recall/MRR/nDCG targets".into(),
            environment: SimEnvironment {
                vector_count: 5_183,
                vector_dim: 768,
                ef_search: 128,
                concurrent_clients: 1,
                ..SimEnvironment::default()
            },
            operations: vec![ScenarioOp {
                operation: Operation::ContextQuery,
                ops: 300,
                read_ratio: None,
                workload_override: None,
            }],
        }
    }

    pub fn mixed_workload() -> Self {
        Self {
            id: "mixed_80r_20w".into(),
            name: "Mixed 80% Read / 20% Write".into(),
            topology: Topology::Standalone,
            description: "SLO mixed workload target".into(),
            environment: SimEnvironment::default(),
            operations: vec![ScenarioOp {
                operation: Operation::PointRead,
                ops: 10_000,
                read_ratio: Some(0.8),
                workload_override: Some("mixed_80r_20w".into()),
            }],
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            Self::standalone_full_stack(),
            Self::distributed_grpc(),
            Self::distributed_large_scale(),
            Self::distributed_concurrent(),
            Self::retrieval_quality(),
            Self::mixed_workload(),
        ]
    }

    pub fn by_id(id: &str) -> Option<Self> {
        Self::all().into_iter().find(|s| s.id == id)
    }
}
