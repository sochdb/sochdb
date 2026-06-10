use serde::{Deserialize, Serialize};
use sochdb_core::knowledge_object::BitemporalCoord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FactId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactKind {
    Extracted,
    Inferred,
    UserAsserted,
}

/// Fact edge with bi-temporal coordinates. Invalidation closes `valid_to`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactEdge {
    pub id: FactId,
    pub episode_id: u64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub kind: FactKind,
    pub temporal: BitemporalCoord,
}

impl FactEdge {
    pub fn is_valid_at(&self, tau: u64) -> bool {
        self.temporal.valid_at(tau)
    }

    pub fn invalidate(&mut self, t_invalid: u64) {
        self.temporal.close_valid_time(t_invalid);
    }
}