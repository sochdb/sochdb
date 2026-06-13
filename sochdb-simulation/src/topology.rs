//! Deployment topologies: standalone (embedded) vs distributed (gRPC cluster).

use crate::component::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Topology {
    /// Single-process embedded: Client → Storage directly.
    Standalone,
    /// Thick-server: Client → gRPC → Services → Storage.
    Distributed,
}

impl Topology {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Standalone => "standalone",
            Self::Distributed => "distributed",
        }
    }

    /// Common prefix for all operations in this topology.
    pub fn ingress_path(&self) -> Vec<Component> {
        match self {
            Self::Standalone => vec![Component::EmbeddedConnection],
            Self::Distributed => vec![
                Component::GrpcClient,
                Component::NetworkRoundTrip,
                Component::LoadBalancer,
                Component::GrpcServer,
                Component::AuthInterceptor,
                Component::ProtobufCodec,
            ],
        }
    }

    /// Egress path (response serialization + network return).
    pub fn egress_path(&self) -> Vec<Component> {
        match self {
            Self::Standalone => vec![],
            Self::Distributed => vec![
                Component::ProtobufCodec,
                Component::NetworkRoundTrip,
                Component::GrpcClient,
            ],
        }
    }
}

/// Operation-specific component chain through the SochDB stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation {
    // ── KV / OLTP ──
    PointRead,
    PointWrite,
    BatchWrite,
    Delete,

    // ── Analytics ──
    AnalyticsInsert,
    AnalyticsQuery,

    // ── Vector ──
    VectorInsert,
    VectorSearchBruteForce,
    VectorSearchHnsw,

    // ── AI-native ──
    ContextQuery,
    HybridFusion,
    McpToolCall,
    TemporalGraphQuery,

    // ── Distributed-specific ──
    GrpcKvPut,
    GrpcKvGet,
    GrpcVectorInsert,
    GrpcHnswSearch,
}

impl Operation {
    pub fn workload_id(&self) -> &'static str {
        match self {
            Self::PointRead => "oltp_seq_read",
            Self::PointWrite => "oltp_seq_write",
            Self::BatchWrite => "oltp_batch_write",
            Self::Delete => "oltp_delete",
            Self::AnalyticsInsert => "analytics_bulk_insert",
            Self::AnalyticsQuery => "analytics_queries",
            Self::VectorInsert => "vector_insert",
            Self::VectorSearchBruteForce => "vector_search",
            Self::VectorSearchHnsw => "grpc_hnsw_search_10k",
            Self::ContextQuery => "context_query",
            Self::HybridFusion => "hybrid_fusion",
            Self::McpToolCall => "mcp_tool_call",
            Self::TemporalGraphQuery => "temporal_graph_query",
            Self::GrpcKvPut => "grpc_kv_put",
            Self::GrpcKvGet => "grpc_kv_get",
            Self::GrpcVectorInsert => "grpc_vector_insert_10k",
            Self::GrpcHnswSearch => "grpc_hnsw_search_10k",
        }
    }

    /// Build the full component path for this operation in the given topology.
    pub fn component_path(&self, topology: Topology) -> Vec<Component> {
        let mut path = topology.ingress_path();
        path.extend(self.core_path());
        path.extend(topology.egress_path());
        path
    }

    fn core_path(&self) -> Vec<Component> {
        match self {
            Self::PointRead => vec![Component::MvccCoordinator, Component::MemtableLookup],
            Self::PointWrite => vec![
                Component::MvccCoordinator,
                Component::WalWriter,
                Component::MemtableLookup,
            ],
            Self::BatchWrite => vec![Component::MvccCoordinator, Component::WalWriter],
            Self::Delete => vec![
                Component::MvccCoordinator,
                Component::WalWriter,
                Component::MemtableLookup,
            ],
            Self::AnalyticsInsert => vec![
                Component::MvccCoordinator,
                Component::WalWriter,
                Component::ColumnarCache,
            ],
            Self::AnalyticsQuery => vec![
                Component::SochQlParser,
                Component::QueryPlanner,
                Component::ColumnarCache,
                Component::CostOptimizer,
            ],
            Self::VectorInsert => vec![
                Component::MvccCoordinator,
                Component::WalWriter,
                Component::VectorCache,
            ],
            Self::VectorSearchBruteForce => vec![Component::VectorCache, Component::BruteForceScan],
            Self::VectorSearchHnsw | Self::GrpcHnswSearch => vec![Component::HnswIndex],
            Self::ContextQuery => vec![
                Component::SochQlParser,
                Component::QueryPlanner,
                Component::FusionPipeline,
                Component::HnswIndex,
                Component::Bm25Filter,
                Component::ContextBuilder,
                Component::TokenBudgetEngine,
                Component::ToonEncoder,
            ],
            Self::HybridFusion => vec![
                Component::FusionPipeline,
                Component::HnswIndex,
                Component::Bm25Filter,
                Component::MetadataIndex,
            ],
            Self::McpToolCall => vec![
                Component::McpServer,
                Component::ContextBuilder,
                Component::TokenBudgetEngine,
                Component::ToonEncoder,
            ],
            Self::TemporalGraphQuery => vec![
                Component::TemporalGraph,
                Component::MvccCoordinator,
                Component::MemtableLookup,
            ],
            Self::GrpcKvPut => vec![Component::MvccCoordinator, Component::WalWriter],
            Self::GrpcKvGet => vec![Component::MvccCoordinator, Component::MemtableLookup],
            Self::GrpcVectorInsert => vec![
                Component::MvccCoordinator,
                Component::WalWriter,
                Component::HnswIndex,
            ],
        }
    }
}

/// Map benchmark workload names to operations.
pub fn workload_to_operation(workload: &str) -> Option<Operation> {
    match workload {
        "oltp_seq_read" => Some(Operation::PointRead),
        "oltp_rand_read" => Some(Operation::PointRead),
        "oltp_seq_write" => Some(Operation::PointWrite),
        "oltp_batch_write" => Some(Operation::BatchWrite),
        "oltp_delete" => Some(Operation::Delete),
        "analytics_bulk_insert" => Some(Operation::AnalyticsInsert),
        "analytics_queries" => Some(Operation::AnalyticsQuery),
        "vector_insert" => Some(Operation::VectorInsert),
        "vector_search" => Some(Operation::VectorSearchBruteForce),
        "mixed_80r_20w" => None, // composite — handled separately
        "grpc_kv_put" => Some(Operation::GrpcKvPut),
        "grpc_kv_get" => Some(Operation::GrpcKvGet),
        "grpc_vector_insert_10k" => Some(Operation::GrpcVectorInsert),
        "grpc_hnsw_search_10k" | "grpc_hnsw_search_50k" | "grpc_hnsw_search_3_5m" => {
            Some(Operation::GrpcHnswSearch)
        }
        "grpc_hnsw_search_concurrent_8" | "grpc_hnsw_search_concurrent_32" => {
            Some(Operation::GrpcHnswSearch)
        }
        "context_query" => Some(Operation::ContextQuery),
        "hybrid_fusion" => Some(Operation::HybridFusion),
        "mcp_tool_call" => Some(Operation::McpToolCall),
        "retrieval_recall_at_5" | "retrieval_mrr" | "retrieval_ndcg_at_5" | "retrieval_p50_ms" => {
            Some(Operation::ContextQuery)
        }
        _ => None,
    }
}
