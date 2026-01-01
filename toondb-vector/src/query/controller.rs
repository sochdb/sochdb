//! Adaptive recall controller.
//!
//! Monitors query confidence signals and adjusts search parameters dynamically.

use crate::types::*;
use crate::config::QueryConfig;

/// Adaptive controller for recall optimization
pub struct AdaptiveController {
    config: QueryConfig,
}

impl AdaptiveController {
    /// Create a new adaptive controller
    pub fn new(config: QueryConfig) -> Self {
        Self { config }
    }

    /// Compute confidence score based on query results
    pub fn compute_confidence(&self, results: &[ScoredCandidate], k: usize) -> ConfidenceSignals {
        if results.is_empty() {
            return ConfidenceSignals {
                score_gap: 0.0,
                entropy: 0.0,
                coverage: 0.0,
                confidence: 0.0,
            };
        }

        // Score gap: difference between k-th and 2k-th result
        let score_gap = if results.len() > k {
            let k_score = results.get(k - 1).map(|c| c.score).unwrap_or(0.0);
            let two_k_score = results.get(2 * k - 1).map(|c| c.score).unwrap_or(0.0);
            (k_score - two_k_score).abs()
        } else {
            // Not enough results, low confidence
            0.0
        };

        // Score entropy (flatness of score distribution)
        let entropy = self.compute_entropy(&results[..k.min(results.len())]);

        // Coverage: did we find enough candidates?
        let coverage = (results.len() as f32) / (k as f32).max(1.0);

        // Combined confidence
        let confidence = self.combine_signals(score_gap, entropy, coverage);

        ConfidenceSignals {
            score_gap,
            entropy,
            coverage,
            confidence,
        }
    }

    /// Compute entropy of score distribution
    fn compute_entropy(&self, results: &[ScoredCandidate]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let scores: Vec<f32> = results.iter().map(|c| c.score.max(0.001)).collect();
        let sum: f32 = scores.iter().sum();
        
        if sum <= 0.0 {
            return 0.0;
        }

        let probs: Vec<f32> = scores.iter().map(|s| s / sum).collect();
        let entropy: f32 = probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        // Normalize by max entropy
        let max_entropy = (results.len() as f32).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Combine signals into overall confidence
    fn combine_signals(&self, score_gap: f32, entropy: f32, coverage: f32) -> f32 {
        // High score gap = high confidence (clear separation)
        let gap_signal = (score_gap / self.config.score_gap_threshold).min(1.0);
        
        // Low entropy = high confidence (peaked distribution)
        let entropy_signal = 1.0 - entropy;
        
        // High coverage = high confidence
        let coverage_signal = coverage.min(1.0);

        // Weighted combination
        0.4 * gap_signal + 0.3 * entropy_signal + 0.3 * coverage_signal
    }

    /// Determine if widening is needed
    pub fn should_widen(&self, confidence: f32) -> bool {
        confidence < 0.5 // Threshold for widening
    }

    /// Compute widening parameters
    pub fn compute_widening(&self, signals: &ConfidenceSignals, _params: &QueryParams) -> WideningParams {
        if signals.confidence >= 0.5 {
            return WideningParams::none();
        }

        let factor = self.config.widening_factor;
        
        // Progressive widening strategy
        if signals.confidence < 0.2 {
            // Very low confidence: widen everything
            WideningParams {
                l_a_factor: factor * 2.0,
                l_b_factor: factor * 2.0,
                r_factor: factor,
                router_probes_factor: 2.0,
            }
        } else if signals.confidence < 0.35 {
            // Low confidence: widen BPS first (cheaper)
            WideningParams {
                l_a_factor: factor,
                l_b_factor: factor * 1.5,
                r_factor: 1.0,
                router_probes_factor: 1.5,
            }
        } else {
            // Moderate confidence: slight widening
            WideningParams {
                l_a_factor: 1.0,
                l_b_factor: factor,
                r_factor: 1.0,
                router_probes_factor: 1.0,
            }
        }
    }

    /// Apply filter-aware widening
    pub fn apply_filter_widening(
        &self,
        params: &mut QueryParams,
        selectivity: f32,
        max_factor: f32,
    ) {
        if selectivity >= 1.0 || selectivity <= 0.0 {
            return;
        }

        let factor = (1.0 / selectivity).min(max_factor);
        params.l_a = ((params.l_a as f32) * factor) as usize;
        params.l_b = ((params.l_b as f32) * factor) as usize;
    }
}

/// Confidence signals from query execution
#[derive(Debug, Clone)]
pub struct ConfidenceSignals {
    /// Score gap between k and 2k results
    pub score_gap: f32,
    /// Entropy/flatness of score distribution
    pub entropy: f32,
    /// Coverage (fraction of requested candidates found)
    pub coverage: f32,
    /// Combined confidence score (0-1)
    pub confidence: f32,
}

/// Widening parameters
#[derive(Debug, Clone)]
pub struct WideningParams {
    pub l_a_factor: f32,
    pub l_b_factor: f32,
    pub r_factor: f32,
    pub router_probes_factor: f32,
}

impl WideningParams {
    pub fn none() -> Self {
        Self {
            l_a_factor: 1.0,
            l_b_factor: 1.0,
            r_factor: 1.0,
            router_probes_factor: 1.0,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.l_a_factor == 1.0
            && self.l_b_factor == 1.0
            && self.r_factor == 1.0
            && self.router_probes_factor == 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_high() {
        let config = QueryConfig::default();
        let controller = AdaptiveController::new(config);

        // Clear separation between results
        let results: Vec<ScoredCandidate> = (0..20)
            .map(|i| ScoredCandidate {
                id: i as u32,
                score: 1.0 - (i as f32) * 0.05,
            })
            .collect();

        let signals = controller.compute_confidence(&results, 10);
        assert!(signals.confidence > 0.5, "Expected high confidence: {}", signals.confidence);
        assert!(!controller.should_widen(signals.confidence));
    }

    #[test]
    fn test_confidence_low() {
        let config = QueryConfig::default();
        let controller = AdaptiveController::new(config);

        // Flat score distribution
        let results: Vec<ScoredCandidate> = (0..20)
            .map(|i| ScoredCandidate {
                id: i as u32,
                score: 1.0, // All same score
            })
            .collect();

        let signals = controller.compute_confidence(&results, 10);
        // Should have high entropy, lower confidence
        assert!(signals.entropy > 0.8, "Expected high entropy: {}", signals.entropy);
    }

    #[test]
    fn test_widening_params() {
        let config = QueryConfig::default();
        let controller = AdaptiveController::new(config);
        let params = QueryParams::default();

        let low_confidence = ConfidenceSignals {
            score_gap: 0.01,
            entropy: 0.9,
            coverage: 0.5,
            confidence: 0.2,
        };

        let widening = controller.compute_widening(&low_confidence, &params);
        assert!(widening.l_b_factor > 1.0);
        assert!(widening.l_a_factor > 1.0);
    }

    #[test]
    fn test_filter_widening() {
        let config = QueryConfig::default();
        let controller = AdaptiveController::new(config);

        let mut params = QueryParams {
            k: 10,
            l_a: 1000,
            l_b: 2000,
            ..Default::default()
        };

        // 10% selectivity should ~10x widen (capped)
        controller.apply_filter_widening(&mut params, 0.1, 5.0);
        assert_eq!(params.l_a, 5000);
        assert_eq!(params.l_b, 10000);
    }
}
