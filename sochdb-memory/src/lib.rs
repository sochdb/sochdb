//! Bi-temporal agent memory with write-time lexical recall.
//!
//! Store-first, enrich-async pipeline:
//! episode write → WAL → lexical index (retrievable immediately) → async enrichment.

pub mod embedding;
pub mod enrichment;
pub mod episode;
pub mod fact;
pub mod lifecycle;
pub mod provenance;
pub mod query;
pub mod store;

pub use enrichment::{EnrichmentJob, EnrichmentQueue, EnrichmentQueueConfig};
pub use episode::{ConversationTurn, Episode, EpisodeId, EpisodeWrite};
pub use fact::{FactEdge, FactId, FactKind};
pub use lifecycle::{LifecycleConfig, MemoryLifecycleDaemon};
pub use provenance::{ProvenanceBundle, TrustScore, TrustScoreConfig};
pub use query::{Lane, MemoryHit, MemoryQuery, MemoryQueryResult, QueryLanes};
pub use store::{MemoryStore, MemoryStoreConfig, WriteResult};

// Re-export embedding provider for custom MemoryStore construction.
pub use sochdb_query::{EmbeddingProvider, MockEmbeddingProvider};

#[cfg(test)]
mod tests;
