use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique episode identifier within a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpisodeId(pub u64);

impl fmt::Display for EpisodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ep:{}", self.0)
    }
}

/// Raw episode write — no LLM extraction on the hot path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeWrite {
    pub namespace: String,
    pub text: String,
    /// Optional world-time valid-from (microseconds). Defaults to write time.
    pub t_valid_from: Option<u64>,
    pub metadata: Option<serde_json::Value>,
}

/// Stored episode record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: EpisodeId,
    pub namespace: String,
    pub text: String,
    pub t_created: u64,
    pub t_valid_from: u64,
    pub enriched: bool,
    pub metadata: Option<serde_json::Value>,
}