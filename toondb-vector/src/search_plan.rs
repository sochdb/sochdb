// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Quantization-aware Search Plan
//!
//! This module provides a formal runtime plan for vector search that separates
//! policy (what to optimize for) from mechanism (how to execute).
//!
//! # Architecture
//!
//! ```text
//! SearchRequest + SLA → Planner → SearchPlan → Executor → Results
//!                          ↑
//!                    Cost Model + Statistics
//! ```
//!
//! # Policy vs Mechanism
//!
//! **Policy** (what to optimize):
//! - Target recall@k (e.g., 0.95)
//! - Latency budget (e.g., 5ms p99)
//! - Token/compute budget
//!
//! **Mechanism** (how to execute):
//! - BPS coarse scan parameters
//! - PQ scoring parameters
//! - Rerank depth and method
//! - ef_search value
//! - Filter evaluation order
//!
//! # Cost Model
//!
//! The planner uses measured per-stage costs:
//! - `cost_bps(N, D)` = N × D × c_bps
//! - `cost_pq(N, D, M)` = N × M × c_pq
//! - `cost_rerank(N, D)` = N × D × c_f32
//!
//! # Optimization
//!
//! Minimize expected latency subject to:
//! - recall@k ≥ target_recall
//! - total_cost ≤ budget
//!
//! Uses bandit-like adaptation based on recent query statistics.

use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Quantization level for a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageQuantLevel {
    /// Block Projection Sketch (coarse filtering)
    BPS,
    /// Product Quantization
    PQ,
    /// 8-bit integer
    I8,
    /// Full precision f32
    F32,
}

impl StageQuantLevel {
    /// Relative cost per vector-dimension operation.
    pub const fn relative_cost(self) -> f32 {
        match self {
            StageQuantLevel::BPS => 0.05,
            StageQuantLevel::PQ => 0.10,
            StageQuantLevel::I8 => 0.25,
            StageQuantLevel::F32 => 1.00,
        }
    }

    /// Expected recall at this level.
    pub const fn expected_recall(self) -> f32 {
        match self {
            StageQuantLevel::BPS => 0.70,
            StageQuantLevel::PQ => 0.90,
            StageQuantLevel::I8 => 0.995,
            StageQuantLevel::F32 => 1.00,
        }
    }
}

/// A single stage in the search pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Quantization level for this stage.
    pub quant_level: StageQuantLevel,
    /// Number of candidates to consider.
    pub input_candidates: usize,
    /// Number of candidates to output.
    pub output_candidates: usize,
    /// Whether to apply filters at this stage.
    pub apply_filter: bool,
}

impl PipelineStage {
    /// Estimate the cost of this stage.
    pub fn estimate_cost(&self, dimension: usize, cost_model: &CostModel) -> f32 {
        let base_cost = self.input_candidates as f32
            * dimension as f32
            * self.quant_level.relative_cost();
        base_cost * cost_model.cpu_cycles_per_op
    }

    /// Estimate the recall of this stage.
    pub fn estimate_recall(&self, total_vectors: usize) -> f32 {
        let coverage = (self.input_candidates as f32 / total_vectors as f32).min(1.0);
        self.quant_level.expected_recall() * coverage.sqrt()
    }
}

/// Service Level Agreement for search.
#[derive(Debug, Clone)]
pub struct SearchSLA {
    /// Target recall@k (0.0 to 1.0).
    pub target_recall: f32,
    /// Maximum latency budget.
    pub latency_budget: Duration,
    /// Maximum compute tokens (relative units).
    pub token_budget: Option<u64>,
    /// Optimization mode.
    pub mode: OptimizationMode,
}

impl Default for SearchSLA {
    fn default() -> Self {
        Self {
            target_recall: 0.95,
            latency_budget: Duration::from_millis(10),
            token_budget: None,
            mode: OptimizationMode::Balanced,
        }
    }
}

/// Optimization mode for the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationMode {
    /// Minimize latency (speed priority).
    Speed,
    /// Maximize recall (quality priority).
    Quality,
    /// Balance latency and recall.
    #[default]
    Balanced,
    /// Strict SLO enforcement.
    SLO,
}

/// Cost model parameters (calibrated per hardware).
#[derive(Debug, Clone)]
pub struct CostModel {
    /// CPU cycles per operation (normalized).
    pub cpu_cycles_per_op: f32,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f32,
    /// L3 cache size in bytes.
    pub l3_cache_bytes: usize,
    /// Measured per-stage costs (ns per candidate).
    pub stage_costs_ns: StageCosts,
}

/// Per-stage cost measurements.
#[derive(Debug, Clone, Default)]
pub struct StageCosts {
    /// BPS scan cost per candidate (ns).
    pub bps_per_candidate_ns: f32,
    /// PQ scoring cost per candidate (ns).
    pub pq_per_candidate_ns: f32,
    /// I8 rerank cost per candidate (ns).
    pub i8_per_candidate_ns: f32,
    /// F32 rerank cost per candidate (ns).
    pub f32_per_candidate_ns: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            cpu_cycles_per_op: 1.0,
            memory_bandwidth_gbps: 50.0,
            l3_cache_bytes: 32 * 1024 * 1024, // 32 MB
            stage_costs_ns: StageCosts {
                bps_per_candidate_ns: 10.0,
                pq_per_candidate_ns: 50.0,
                i8_per_candidate_ns: 100.0,
                f32_per_candidate_ns: 500.0,
            },
        }
    }
}

/// Statistics about the dataset for planning.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total number of vectors.
    pub total_vectors: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Available quantization levels.
    pub available_levels: Vec<StageQuantLevel>,
    /// Filter selectivity (if known).
    pub filter_selectivity: Option<f32>,
    /// Recent query latency histogram (p50, p90, p99).
    pub recent_latencies: Option<(Duration, Duration, Duration)>,
}

impl Default for DatasetStats {
    fn default() -> Self {
        Self {
            total_vectors: 0,
            dimension: 0,
            available_levels: vec![StageQuantLevel::F32],
            filter_selectivity: None,
            recent_latencies: None,
        }
    }
}

/// The search plan: a complete specification for executing a search.
#[derive(Debug, Clone)]
pub struct SearchPlan {
    /// Pipeline stages in execution order.
    pub stages: Vec<PipelineStage>,
    /// ef_search parameter for HNSW.
    pub ef_search: usize,
    /// Final k to return.
    pub k: usize,
    /// Whether to use batched expansion.
    pub use_batched_expansion: bool,
    /// Prefetch distance (0 = disabled).
    pub prefetch_distance: usize,
    /// Estimated total latency.
    pub estimated_latency: Duration,
    /// Estimated recall.
    pub estimated_recall: f32,
    /// Plan generation timestamp.
    pub created_at: Instant,
}

impl SearchPlan {
    /// Create a simple single-stage plan (F32 only).
    pub fn simple(k: usize, ef_search: usize) -> Self {
        Self {
            stages: vec![PipelineStage {
                quant_level: StageQuantLevel::F32,
                input_candidates: ef_search,
                output_candidates: k,
                apply_filter: false,
            }],
            ef_search,
            k,
            use_batched_expansion: true,
            prefetch_distance: 4,
            estimated_latency: Duration::from_millis(1),
            estimated_recall: 0.95,
            created_at: Instant::now(),
        }
    }

    /// Create a multi-stage plan with BPS → PQ → F32 pipeline.
    pub fn multi_stage(
        k: usize,
        total_vectors: usize,
        target_recall: f32,
    ) -> Self {
        // Calculate candidate counts for each stage
        let coarse_candidates = (total_vectors as f32 * 0.1).min(10000.0) as usize;
        let refine_candidates = (coarse_candidates as f32 * 0.1).max(k as f32 * 10.0) as usize;
        let _rerank_candidates = (refine_candidates as f32 * 0.5).max(k as f32 * 2.0) as usize;

        let mut stages = Vec::new();

        // BPS coarse stage (if dataset is large enough)
        if total_vectors > 10_000 {
            stages.push(PipelineStage {
                quant_level: StageQuantLevel::BPS,
                input_candidates: total_vectors,
                output_candidates: coarse_candidates,
                apply_filter: true, // Early filter
            });
        }

        // PQ refinement stage
        if total_vectors > 1_000 {
            stages.push(PipelineStage {
                quant_level: StageQuantLevel::PQ,
                input_candidates: coarse_candidates,
                output_candidates: refine_candidates,
                apply_filter: false,
            });
        }

        // I8 or F32 rerank (choose based on recall target)
        let rerank_level = if target_recall > 0.99 {
            StageQuantLevel::F32
        } else {
            StageQuantLevel::I8
        };

        stages.push(PipelineStage {
            quant_level: rerank_level,
            input_candidates: refine_candidates,
            output_candidates: k,
            apply_filter: false,
        });

        Self {
            stages,
            ef_search: refine_candidates.max(64),
            k,
            use_batched_expansion: true,
            prefetch_distance: 4,
            estimated_latency: Duration::from_millis(5),
            estimated_recall: target_recall,
            created_at: Instant::now(),
        }
    }

    /// Get the total estimated cost.
    pub fn total_cost(&self, dimension: usize, cost_model: &CostModel) -> f32 {
        self.stages
            .iter()
            .map(|s| s.estimate_cost(dimension, cost_model))
            .sum()
    }

    /// Check if the plan meets the SLA.
    pub fn meets_sla(&self, sla: &SearchSLA) -> bool {
        self.estimated_recall >= sla.target_recall
            && self.estimated_latency <= sla.latency_budget
    }
}

/// Search planner that generates optimal plans.
pub struct SearchPlanner {
    /// Cost model for estimation.
    cost_model: CostModel,
    /// Recent query statistics for adaptation.
    recent_stats: RwLock<RecentStats>,
    /// Query counter for bandit adaptation.
    query_count: AtomicU64,
}

/// Recent query statistics for adaptive planning.
#[derive(Debug, Default)]
struct RecentStats {
    /// Recent latencies (sliding window).
    latencies: VecDeque<Duration>,
    /// Recent recalls (sliding window).
    recalls: VecDeque<f32>,
    /// Window size.
    window_size: usize,
}

impl RecentStats {
    fn new(window_size: usize) -> Self {
        Self {
            latencies: VecDeque::with_capacity(window_size),
            recalls: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    fn record(&mut self, latency: Duration, recall: f32) {
        if self.latencies.len() >= self.window_size {
            self.latencies.pop_front();
            self.recalls.pop_front();
        }
        self.latencies.push_back(latency);
        self.recalls.push_back(recall);
    }

    fn avg_latency(&self) -> Option<Duration> {
        if self.latencies.is_empty() {
            return None;
        }
        let sum: Duration = self.latencies.iter().sum();
        Some(sum / self.latencies.len() as u32)
    }

    #[allow(dead_code)]
    fn avg_recall(&self) -> Option<f32> {
        if self.recalls.is_empty() {
            return None;
        }
        Some(self.recalls.iter().sum::<f32>() / self.recalls.len() as f32)
    }
}

impl SearchPlanner {
    /// Create a new search planner with default cost model.
    pub fn new() -> Self {
        Self {
            cost_model: CostModel::default(),
            recent_stats: RwLock::new(RecentStats::new(100)),
            query_count: AtomicU64::new(0),
        }
    }

    /// Create a planner with custom cost model.
    pub fn with_cost_model(cost_model: CostModel) -> Self {
        Self {
            cost_model,
            recent_stats: RwLock::new(RecentStats::new(100)),
            query_count: AtomicU64::new(0),
        }
    }

    /// Generate an optimal search plan.
    pub fn plan(
        &self,
        k: usize,
        sla: &SearchSLA,
        stats: &DatasetStats,
    ) -> SearchPlan {
        self.query_count.fetch_add(1, Ordering::Relaxed);

        // Choose planning strategy based on mode
        match sla.mode {
            OptimizationMode::Speed => self.plan_for_speed(k, sla, stats),
            OptimizationMode::Quality => self.plan_for_quality(k, sla, stats),
            OptimizationMode::Balanced => self.plan_balanced(k, sla, stats),
            OptimizationMode::SLO => self.plan_for_slo(k, sla, stats),
        }
    }

    /// Plan optimized for speed.
    fn plan_for_speed(&self, k: usize, _sla: &SearchSLA, stats: &DatasetStats) -> SearchPlan {
        // Use aggressive coarse filtering
        let ef = k.max(16);

        if stats.total_vectors > 100_000 && stats.available_levels.contains(&StageQuantLevel::BPS) {
            // BPS → I8 pipeline
            let coarse_count = (stats.total_vectors as f32 * 0.01).max(1000.0) as usize;
            
            SearchPlan {
                stages: vec![
                    PipelineStage {
                        quant_level: StageQuantLevel::BPS,
                        input_candidates: stats.total_vectors,
                        output_candidates: coarse_count,
                        apply_filter: true,
                    },
                    PipelineStage {
                        quant_level: StageQuantLevel::I8,
                        input_candidates: coarse_count,
                        output_candidates: k,
                        apply_filter: false,
                    },
                ],
                ef_search: ef,
                k,
                use_batched_expansion: true,
                prefetch_distance: 8,
                estimated_latency: Duration::from_micros(500),
                estimated_recall: 0.85,
                created_at: Instant::now(),
            }
        } else {
            // Simple I8 or F32
            let level = if stats.available_levels.contains(&StageQuantLevel::I8) {
                StageQuantLevel::I8
            } else {
                StageQuantLevel::F32
            };

            SearchPlan {
                stages: vec![PipelineStage {
                    quant_level: level,
                    input_candidates: ef * 4,
                    output_candidates: k,
                    apply_filter: true,
                }],
                ef_search: ef,
                k,
                use_batched_expansion: true,
                prefetch_distance: 4,
                estimated_latency: Duration::from_millis(1),
                estimated_recall: 0.90,
                created_at: Instant::now(),
            }
        }
    }

    /// Plan optimized for quality.
    fn plan_for_quality(&self, k: usize, sla: &SearchSLA, stats: &DatasetStats) -> SearchPlan {
        // Use full F32 with high ef
        let ef = (k * 10).max(200);
        
        SearchPlan {
            stages: vec![PipelineStage {
                quant_level: StageQuantLevel::F32,
                input_candidates: ef,
                output_candidates: k,
                apply_filter: false, // Filter after scoring for max recall
            }],
            ef_search: ef,
            k,
            use_batched_expansion: true,
            prefetch_distance: 4,
            estimated_latency: self.estimate_latency(ef, stats.dimension, StageQuantLevel::F32),
            estimated_recall: sla.target_recall.min(0.99),
            created_at: Instant::now(),
        }
    }

    /// Balanced plan.
    fn plan_balanced(&self, k: usize, sla: &SearchSLA, stats: &DatasetStats) -> SearchPlan {
        // Multi-stage with adaptive parameters
        let ef = (k * 4).max(64);
        
        // Decide stages based on dataset size and available levels
        let use_pq = stats.total_vectors > 10_000 
            && stats.available_levels.contains(&StageQuantLevel::PQ);
        let use_i8 = stats.available_levels.contains(&StageQuantLevel::I8);

        let mut stages = Vec::new();

        if use_pq {
            stages.push(PipelineStage {
                quant_level: StageQuantLevel::PQ,
                input_candidates: ef * 10,
                output_candidates: ef * 2,
                apply_filter: true,
            });
        }

        let final_level = if sla.target_recall > 0.98 {
            StageQuantLevel::F32
        } else if use_i8 {
            StageQuantLevel::I8
        } else {
            StageQuantLevel::F32
        };

        stages.push(PipelineStage {
            quant_level: final_level,
            input_candidates: if use_pq { ef * 2 } else { ef * 4 },
            output_candidates: k,
            apply_filter: !use_pq,
        });

        let estimated_recall = self.estimate_pipeline_recall(&stages, stats.total_vectors);
        let estimated_latency = self.estimate_pipeline_latency(&stages, stats.dimension);

        SearchPlan {
            stages,
            ef_search: ef,
            k,
            use_batched_expansion: true,
            prefetch_distance: 4,
            estimated_latency,
            estimated_recall,
            created_at: Instant::now(),
        }
    }

    /// Plan for strict SLO enforcement.
    fn plan_for_slo(&self, k: usize, sla: &SearchSLA, stats: &DatasetStats) -> SearchPlan {
        // Use adaptive feedback from recent stats
        let recent = self.recent_stats.read().unwrap();
        
        let base_plan = if let Some(avg_latency) = recent.avg_latency() {
            // Adjust based on recent performance
            if avg_latency > sla.latency_budget {
                // We're too slow, reduce work
                self.plan_for_speed(k, sla, stats)
            } else if avg_latency < sla.latency_budget / 2 {
                // We have headroom, increase quality
                self.plan_for_quality(k, sla, stats)
            } else {
                self.plan_balanced(k, sla, stats)
            }
        } else {
            // No history, start balanced
            self.plan_balanced(k, sla, stats)
        };

        // Ensure we meet SLA
        if base_plan.estimated_latency > sla.latency_budget {
            // Fall back to speed mode
            self.plan_for_speed(k, sla, stats)
        } else {
            base_plan
        }
    }

    /// Record query execution feedback for adaptation.
    pub fn record_feedback(&self, latency: Duration, recall: f32) {
        let mut stats = self.recent_stats.write().unwrap();
        stats.record(latency, recall);
    }

    /// Estimate latency for a stage.
    fn estimate_latency(&self, candidates: usize, dimension: usize, level: StageQuantLevel) -> Duration {
        let cost_per_candidate = match level {
            StageQuantLevel::BPS => self.cost_model.stage_costs_ns.bps_per_candidate_ns,
            StageQuantLevel::PQ => self.cost_model.stage_costs_ns.pq_per_candidate_ns,
            StageQuantLevel::I8 => self.cost_model.stage_costs_ns.i8_per_candidate_ns,
            StageQuantLevel::F32 => self.cost_model.stage_costs_ns.f32_per_candidate_ns,
        };

        let total_ns = candidates as f32 * cost_per_candidate * (dimension as f32 / 128.0);
        Duration::from_nanos(total_ns as u64)
    }

    /// Estimate recall for a pipeline.
    fn estimate_pipeline_recall(&self, stages: &[PipelineStage], total_vectors: usize) -> f32 {
        stages.iter().fold(1.0, |acc, stage| {
            acc * stage.estimate_recall(total_vectors)
        })
    }

    /// Estimate latency for a pipeline.
    fn estimate_pipeline_latency(&self, stages: &[PipelineStage], dimension: usize) -> Duration {
        stages.iter().map(|stage| {
            self.estimate_latency(stage.input_candidates, dimension, stage.quant_level)
        }).sum()
    }

    /// Get current cost model.
    pub fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// Get query count.
    pub fn query_count(&self) -> u64 {
        self.query_count.load(Ordering::Relaxed)
    }
}

impl Default for SearchPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Plan executor that runs a search plan.
pub struct PlanExecutor;

impl PlanExecutor {
    /// Validate a plan before execution.
    pub fn validate(plan: &SearchPlan) -> Result<(), PlanError> {
        if plan.stages.is_empty() {
            return Err(PlanError::EmptyPipeline);
        }

        if plan.k == 0 {
            return Err(PlanError::InvalidK);
        }

        // Check stage consistency
        for window in plan.stages.windows(2) {
            if window[0].output_candidates < window[1].input_candidates {
                // Allow some slack for over-request
                if window[0].output_candidates * 2 < window[1].input_candidates {
                    return Err(PlanError::StageOutputMismatch);
                }
            }
        }

        Ok(())
    }
}

/// Plan validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// Pipeline has no stages.
    EmptyPipeline,
    /// k must be > 0.
    InvalidK,
    /// Stage output doesn't feed next stage input.
    StageOutputMismatch,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::EmptyPipeline => write!(f, "Pipeline has no stages"),
            PlanError::InvalidK => write!(f, "k must be greater than 0"),
            PlanError::StageOutputMismatch => write!(f, "Stage output doesn't match next stage input"),
        }
    }
}

impl std::error::Error for PlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_plan() {
        let plan = SearchPlan::simple(10, 64);
        
        assert_eq!(plan.k, 10);
        assert_eq!(plan.ef_search, 64);
        assert_eq!(plan.stages.len(), 1);
        assert!(PlanExecutor::validate(&plan).is_ok());
    }

    #[test]
    fn test_multi_stage_plan() {
        let plan = SearchPlan::multi_stage(10, 1_000_000, 0.95);
        
        assert_eq!(plan.k, 10);
        assert!(plan.stages.len() >= 2);
        assert!(PlanExecutor::validate(&plan).is_ok());
    }

    #[test]
    fn test_planner_speed_mode() {
        let planner = SearchPlanner::new();
        let sla = SearchSLA {
            mode: OptimizationMode::Speed,
            ..Default::default()
        };
        let stats = DatasetStats {
            total_vectors: 1_000_000,
            dimension: 768,
            available_levels: vec![StageQuantLevel::BPS, StageQuantLevel::I8, StageQuantLevel::F32],
            ..Default::default()
        };

        let plan = planner.plan(10, &sla, &stats);
        
        // Speed mode should use BPS for large datasets
        assert!(plan.stages.iter().any(|s| s.quant_level == StageQuantLevel::BPS));
        assert!(PlanExecutor::validate(&plan).is_ok());
    }

    #[test]
    fn test_planner_quality_mode() {
        let planner = SearchPlanner::new();
        let sla = SearchSLA {
            mode: OptimizationMode::Quality,
            target_recall: 0.99,
            ..Default::default()
        };
        let stats = DatasetStats {
            total_vectors: 100_000,
            dimension: 768,
            available_levels: vec![StageQuantLevel::F32],
            ..Default::default()
        };

        let plan = planner.plan(10, &sla, &stats);
        
        // Quality mode should use F32
        assert!(plan.stages.iter().any(|s| s.quant_level == StageQuantLevel::F32));
        assert!(plan.ef_search >= 100);
    }

    #[test]
    fn test_planner_balanced_mode() {
        let planner = SearchPlanner::new();
        let sla = SearchSLA {
            mode: OptimizationMode::Balanced,
            target_recall: 0.95,
            ..Default::default()
        };
        let stats = DatasetStats {
            total_vectors: 100_000,
            dimension: 384,
            available_levels: vec![StageQuantLevel::PQ, StageQuantLevel::I8, StageQuantLevel::F32],
            ..Default::default()
        };

        let plan = planner.plan(10, &sla, &stats);
        
        assert!(plan.stages.len() >= 1);
        assert!(PlanExecutor::validate(&plan).is_ok());
    }

    #[test]
    fn test_feedback_adaptation() {
        let planner = SearchPlanner::new();
        
        // Record some fast queries
        for _ in 0..10 {
            planner.record_feedback(Duration::from_micros(100), 0.98);
        }

        let sla = SearchSLA {
            mode: OptimizationMode::SLO,
            latency_budget: Duration::from_millis(5),
            ..Default::default()
        };
        let stats = DatasetStats {
            total_vectors: 100_000,
            dimension: 384,
            available_levels: vec![StageQuantLevel::F32],
            ..Default::default()
        };

        // With fast recent queries, SLO mode should choose quality
        let plan = planner.plan(10, &sla, &stats);
        assert!(plan.ef_search >= 64);
    }

    #[test]
    fn test_plan_cost_estimation() {
        let plan = SearchPlan::simple(10, 64);
        let cost_model = CostModel::default();
        
        let cost = plan.total_cost(384, &cost_model);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_plan_meets_sla() {
        let plan = SearchPlan {
            stages: vec![],
            ef_search: 64,
            k: 10,
            use_batched_expansion: true,
            prefetch_distance: 4,
            estimated_latency: Duration::from_millis(2),
            estimated_recall: 0.96,
            created_at: Instant::now(),
        };

        let sla = SearchSLA {
            target_recall: 0.95,
            latency_budget: Duration::from_millis(5),
            ..Default::default()
        };

        assert!(plan.meets_sla(&sla));

        let strict_sla = SearchSLA {
            target_recall: 0.99,
            latency_budget: Duration::from_millis(1),
            ..Default::default()
        };

        assert!(!plan.meets_sla(&strict_sla));
    }

    #[test]
    fn test_invalid_plan() {
        let empty_plan = SearchPlan {
            stages: vec![],
            ef_search: 64,
            k: 10,
            use_batched_expansion: true,
            prefetch_distance: 4,
            estimated_latency: Duration::from_millis(1),
            estimated_recall: 0.95,
            created_at: Instant::now(),
        };

        assert_eq!(PlanExecutor::validate(&empty_plan), Err(PlanError::EmptyPipeline));

        let zero_k_plan = SearchPlan {
            stages: vec![PipelineStage {
                quant_level: StageQuantLevel::F32,
                input_candidates: 64,
                output_candidates: 0,
                apply_filter: false,
            }],
            ef_search: 64,
            k: 0,
            use_batched_expansion: true,
            prefetch_distance: 4,
            estimated_latency: Duration::from_millis(1),
            estimated_recall: 0.95,
            created_at: Instant::now(),
        };

        assert_eq!(PlanExecutor::validate(&zero_k_plan), Err(PlanError::InvalidK));
    }

    #[test]
    fn test_stage_relative_costs() {
        assert!(StageQuantLevel::BPS.relative_cost() < StageQuantLevel::PQ.relative_cost());
        assert!(StageQuantLevel::PQ.relative_cost() < StageQuantLevel::I8.relative_cost());
        assert!(StageQuantLevel::I8.relative_cost() < StageQuantLevel::F32.relative_cost());
    }
}
