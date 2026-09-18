//! Bi-temporal agent memory with write-time lexical recall.
//!
//! Store-first, enrich-async pipeline:
//! episode write → WAL → lexical index (retrievable immediately) → async enrichment.
//!
//! The WAL step is active only for a store opened with a data directory; see
//! [`Durability`] and [`MemoryStore::new`]. Without one the store is purely
//! in-memory and does not survive a restart.

pub mod durability;
pub mod embedding;
pub mod enrichment;
pub mod episode;
pub mod fact;
#[cfg(any(test, feature = "fault-injection"))]
pub mod fault;
pub mod lifecycle;
pub mod provenance;
pub mod query;
pub mod store;
mod topk;

pub use durability::Durability;
pub use enrichment::{EnrichmentJob, EnrichmentQueue, EnrichmentQueueConfig};
pub use episode::{ConversationTurn, Episode, EpisodeId, EpisodeWrite};
pub use fact::{FactEdge, FactId, FactKind};
pub use lifecycle::{LifecycleConfig, MemoryLifecycleDaemon};
pub use provenance::{ProvenanceBundle, TrustScore, TrustScoreConfig};
pub use query::{Lane, MemoryHit, MemoryQuery, MemoryQueryResult, QueryLanes};
pub use store::{
    MemoryError, MemoryResult, MemoryStore, MemoryStoreConfig, RecoveryReport, WriteResult,
};

// Re-export embedding provider for custom MemoryStore construction.
pub use sochdb_query::{EmbeddingProvider, MockEmbeddingProvider};

#[cfg(test)]
mod tests;
