//! SochDB subsystem component models.
//!
//! Each component carries measured or modeled latency/throughput from benchmarks
//! and architecture docs. The simulation engine composes these into request paths.

use serde::{Deserialize, Serialize};

/// A SochDB subsystem with performance characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Component {
    // ── Client / connection layer ──
    EmbeddedConnection,
    GrpcClient,
    IpcClient,

    // ── Server / protocol layer ──
    GrpcServer,
    IpcServer,
    AuthInterceptor,
    ProtobufCodec,
    McpServer,

    // ── Storage engine ──
    MvccCoordinator,
    WalWriter,
    MemtableLookup,
    TchPathResolver,
    ColumnarCache,
    SstableReader,
    CompactionWorker,

    // ── Vector index ──
    HnswIndex,
    VamanaIndex,
    VectorCache,
    BruteForceScan,

    // ── Query / fusion ──
    SochQlParser,
    QueryPlanner,
    CostOptimizer,
    FusionPipeline,
    Bm25Filter,
    MetadataIndex,

    // ── AI-native features ──
    ContextBuilder,
    TokenBudgetEngine,
    ToonEncoder,
    SemanticCache,
    TemporalGraph,

    // ── Infrastructure ──
    NetworkRoundTrip,
    LoadBalancer,
    CheckpointRecovery,
}

impl Component {
    /// Base latency in microseconds (p50-ish, standalone embedded path).
    pub fn base_latency_us(&self) -> f64 {
        match self {
            Self::EmbeddedConnection => 0.05,
            Self::GrpcClient => 20.0,
            Self::IpcClient => 5.0,
            Self::GrpcServer => 30.0,
            Self::IpcServer => 10.0,
            Self::AuthInterceptor => 10.0,
            Self::ProtobufCodec => 15.0,
            Self::McpServer => 40.0,
            Self::MvccCoordinator => 0.5,
            Self::WalWriter => 5.0,
            Self::MemtableLookup => 0.1,
            Self::TchPathResolver => 0.2,
            Self::ColumnarCache => 2.0,
            Self::SstableReader => 5.0,
            Self::CompactionWorker => 0.0, // background, not on hot path
            Self::HnswIndex => 1500.0,
            Self::VamanaIndex => 800.0,
            Self::VectorCache => 50.0,
            Self::BruteForceScan => 300.0,
            Self::SochQlParser => 10.0,
            Self::QueryPlanner => 15.0,
            Self::CostOptimizer => 8.0,
            Self::FusionPipeline => 50.0,
            Self::Bm25Filter => 30.0,
            Self::MetadataIndex => 5.0,
            Self::ContextBuilder => 100.0,
            Self::TokenBudgetEngine => 20.0,
            Self::ToonEncoder => 25.0,
            Self::SemanticCache => 15.0,
            Self::TemporalGraph => 40.0,
            Self::NetworkRoundTrip => 100.0,
            Self::LoadBalancer => 5.0,
            Self::CheckpointRecovery => 0.0,
        }
    }

    /// Max sustainable throughput for this component (ops/sec).
    pub fn max_throughput(&self) -> f64 {
        match self {
            Self::EmbeddedConnection => 10_000_000.0,
            Self::GrpcClient => 200_000.0,
            Self::IpcClient => 500_000.0,
            Self::GrpcServer => 150_000.0,
            Self::IpcServer => 400_000.0,
            Self::AuthInterceptor => 300_000.0,
            Self::ProtobufCodec => 250_000.0,
            Self::McpServer => 50_000.0,
            Self::MvccCoordinator => 2_000_000.0,
            Self::WalWriter => 200_000.0,
            Self::MemtableLookup => 9_000_000.0,
            Self::TchPathResolver => 5_000_000.0,
            Self::ColumnarCache => 500_000.0,
            Self::SstableReader => 1_000_000.0,
            Self::CompactionWorker => 10_000.0,
            Self::HnswIndex => 600.0,
            Self::VamanaIndex => 1_200.0,
            Self::VectorCache => 50_000.0,
            Self::BruteForceScan => 3_200.0,
            Self::SochQlParser => 100_000.0,
            Self::QueryPlanner => 80_000.0,
            Self::CostOptimizer => 100_000.0,
            Self::FusionPipeline => 20_000.0,
            Self::Bm25Filter => 30_000.0,
            Self::MetadataIndex => 200_000.0,
            Self::ContextBuilder => 10_000.0,
            Self::TokenBudgetEngine => 50_000.0,
            Self::ToonEncoder => 40_000.0,
            Self::SemanticCache => 100_000.0,
            Self::TemporalGraph => 25_000.0,
            Self::NetworkRoundTrip => 100_000.0,
            Self::LoadBalancer => 500_000.0,
            Self::CheckpointRecovery => 1_000.0,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::EmbeddedConnection => "Embedded Connection",
            Self::GrpcClient => "gRPC Client",
            Self::IpcClient => "IPC Client",
            Self::GrpcServer => "gRPC Server",
            Self::IpcServer => "IPC Server",
            Self::AuthInterceptor => "Auth Interceptor",
            Self::ProtobufCodec => "Protobuf Codec",
            Self::McpServer => "MCP Server",
            Self::MvccCoordinator => "MVCC Coordinator",
            Self::WalWriter => "WAL Writer",
            Self::MemtableLookup => "Memtable Lookup",
            Self::TchPathResolver => "TCH Path Resolver",
            Self::ColumnarCache => "Columnar Cache",
            Self::SstableReader => "SSTable Reader",
            Self::CompactionWorker => "Compaction Worker",
            Self::HnswIndex => "HNSW Index",
            Self::VamanaIndex => "Vamana Index",
            Self::VectorCache => "Vector Cache",
            Self::BruteForceScan => "Brute-Force Scan",
            Self::SochQlParser => "SochQL Parser",
            Self::QueryPlanner => "Query Planner",
            Self::CostOptimizer => "Cost Optimizer",
            Self::FusionPipeline => "Fusion Pipeline",
            Self::Bm25Filter => "BM25 Filter",
            Self::MetadataIndex => "Metadata Index",
            Self::ContextBuilder => "Context Builder",
            Self::TokenBudgetEngine => "Token Budget Engine",
            Self::ToonEncoder => "TOON Encoder",
            Self::SemanticCache => "Semantic Cache",
            Self::TemporalGraph => "Temporal Graph",
            Self::NetworkRoundTrip => "Network RTT",
            Self::LoadBalancer => "Load Balancer",
            Self::CheckpointRecovery => "Checkpoint Recovery",
        }
    }
}

/// Tunable environment parameters affecting component behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimEnvironment {
    /// Network round-trip latency override (μs). Default: 100 (localhost).
    pub network_rtt_us: f64,
    /// Number of storage replicas (distributed writes).
    pub replication_factor: u32,
    /// Concurrent client streams (distributed search scaling).
    pub concurrent_clients: u32,
    /// Vector count in index (affects HNSW/Vamana latency).
    pub vector_count: u64,
    /// Vector dimension.
    pub vector_dim: usize,
    /// HNSW ef_search parameter.
    pub ef_search: u32,
    /// Working set fits in memory (fast path).
    pub cache_warm: bool,
    /// Group commit enabled (adds write batching latency).
    pub group_commit: bool,
}

impl Default for SimEnvironment {
    fn default() -> Self {
        Self {
            network_rtt_us: 100.0,
            replication_factor: 1,
            concurrent_clients: 1,
            vector_count: 10_000,
            vector_dim: 128,
            ef_search: 50,
            cache_warm: true,
            group_commit: false,
        }
    }
}

impl SimEnvironment {
    /// Effective latency for a component given environment state.
    pub fn effective_latency_us(&self, component: Component) -> f64 {
        let base = match component {
            Component::NetworkRoundTrip => self.network_rtt_us,
            Component::HnswIndex => {
                let scale = (self.vector_count as f64).log2().max(1.0);
                let ef_factor = self.ef_search as f64 / 50.0;
                component.base_latency_us() * scale * 0.3 * ef_factor
            }
            Component::VamanaIndex => {
                let scale = (self.vector_count as f64 / 1_000_000.0).max(0.1);
                component.base_latency_us() * scale.sqrt()
            }
            Component::BruteForceScan => {
                component.base_latency_us() * (self.vector_count as f64 / 10_000.0)
            }
            Component::WalWriter if self.group_commit => component.base_latency_us() + 5000.0,
            Component::SstableReader if !self.cache_warm => component.base_latency_us() * 10.0,
            _ => component.base_latency_us(),
        };

        // Replication adds quorum write latency
        if component == Component::WalWriter && self.replication_factor > 1 {
            base * self.replication_factor as f64 * 0.7
        } else {
            base
        }
    }

    /// Effective throughput considering concurrency and bottlenecks.
    pub fn effective_throughput(&self, component: Component) -> f64 {
        let base = component.max_throughput();
        match component {
            Component::HnswIndex | Component::VamanaIndex | Component::BruteForceScan => {
                base * self.concurrent_clients as f64 * 0.85
            }
            Component::GrpcServer | Component::NetworkRoundTrip => {
                base.min(200_000.0 * self.concurrent_clients as f64)
            }
            _ => base,
        }
    }
}
