use serde::{Deserialize, Serialize};

/// Transparent trust score: weighted sum, not learned.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrustScore {
    pub value: f32,
    pub source_count: u32,
    pub recency_factor: f32,
    pub contradiction_penalty: f32,
}

#[derive(Debug, Clone)]
pub struct TrustScoreConfig {
    pub source_weight: f32,
    pub recency_weight: f32,
    pub contradiction_weight: f32,
    pub half_life_secs: f64,
}

impl Default for TrustScoreConfig {
    fn default() -> Self {
        Self {
            source_weight: 0.4,
            recency_weight: 0.4,
            contradiction_weight: 0.2,
            half_life_secs: 86_400.0 * 30.0,
        }
    }
}

impl TrustScore {
    pub fn compute(
        cfg: &TrustScoreConfig,
        source_count: u32,
        t_created: u64,
        contradiction_count: u32,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let age_secs = now.saturating_sub(t_created / 1_000_000) as f64;
        let recency = (-age_secs / cfg.half_life_secs).exp() as f32;
        let source_factor = (source_count as f32).ln_1p().min(3.0) / 3.0;
        let contradiction_penalty = (contradiction_count as f32 * 0.25).min(1.0);
        let value = (cfg.source_weight * source_factor + cfg.recency_weight * recency
            - cfg.contradiction_weight * contradiction_penalty)
            .clamp(0.0, 1.0);
        Self {
            value,
            source_count,
            recency_factor: recency,
            contradiction_penalty,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBundle {
    pub episode_id: u64,
    pub t_valid_from: u64,
    pub t_valid_to: u64,
    pub trust: TrustScore,
}
