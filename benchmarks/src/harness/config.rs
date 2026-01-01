// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Workload specification parsing and configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Dataset tier for benchmark sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetTier {
    /// Small dataset for quick tests (100K vectors typically)
    Small,
    /// Medium dataset for nightly runs (5M vectors typically)
    Medium,
    /// Large dataset for comprehensive testing (50M+ vectors)
    Large,
}

impl Default for DatasetTier {
    fn default() -> Self {
        Self::Small
    }
}

impl std::fmt::Display for DatasetTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
        }
    }
}

/// Query pattern types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryPattern {
    /// Pure vector search
    VectorSearch {
        top_k: Vec<usize>,
        #[serde(default)]
        filters: Vec<FilterSpec>,
        #[serde(default)]
        time_decay: bool,
    },
    /// Time range scan
    TimeRangeScan {
        #[serde(default)]
        with_semantic: bool,
    },
    /// Key-value operations
    KeyValue {
        read_ratio: f64,
        write_ratio: f64,
    },
}

/// Filter specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterSpec {
    pub field: String,
    pub op: FilterOp,
    #[serde(default)]
    pub optional: bool,
}

/// Filter operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Contains,
}

/// Schema field definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: FieldType,
    #[serde(default)]
    pub dimension: Option<usize>,
    #[serde(default)]
    pub optional: bool,
}

/// Field types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Vector,
    Timestamp,
    Uint32,
    Uint64,
    Int32,
    Int64,
    Float32,
    Float64,
    String,
    StringArray,
    Bool,
    Bytes,
    Uint8,
}

/// Dataset tier configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    #[serde(default)]
    pub vectors: Option<usize>,
    #[serde(default)]
    pub agents: Option<usize>,
    #[serde(default)]
    pub memories_per_agent: Option<usize>,
    #[serde(default)]
    pub chunks: Option<usize>,
    #[serde(default)]
    pub events: Option<usize>,
    #[serde(default)]
    pub total: Option<usize>,
    pub seed: u64,
}

/// Query configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    pub count: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub include_near_duplicates: bool,
    #[serde(default)]
    pub include_topic_shifts: bool,
    #[serde(default)]
    pub stratified_by_selectivity: bool,
    #[serde(default)]
    pub ood_queries_percent: f64,
}

/// Quality metrics to collect.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityMetric {
    #[serde(rename = "recall@10")]
    RecallAt10,
    #[serde(rename = "recall@20")]
    RecallAt20,
    #[serde(rename = "recall@50")]
    RecallAt50,
    #[serde(rename = "ndcg@10")]
    NdcgAt10,
    Checksum,
}

/// Performance targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfTargets {
    #[serde(default)]
    pub p95_latency_ms: Option<f64>,
    #[serde(default)]
    pub min_qps: Option<f64>,
    #[serde(default)]
    pub max_error_rate: Option<f64>,
}

/// Concurrency configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    pub levels: Vec<usize>,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self { levels: vec![1, 4, 16, 64] }
    }
}

/// Filter selectivity ranges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectivityRange {
    pub min: f64,
    pub max: f64,
}

/// Workload metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMeta {
    pub name: String,
    pub description: String,
}

/// Data model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataModel {
    pub schema: Vec<SchemaField>,
}

/// Complete workload specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub workload: WorkloadMeta,
    #[serde(default)]
    pub data_model: Option<DataModel>,
    #[serde(default)]
    pub query_patterns: Vec<QueryPattern>,
    pub datasets: HashMap<String, DatasetConfig>,
    #[serde(default)]
    pub queries: Option<QueryConfig>,
    #[serde(default)]
    pub quality_metrics: Vec<QualityMetric>,
    #[serde(default)]
    pub perf_targets: Option<PerfTargets>,
    #[serde(default)]
    pub concurrency: ConcurrencyConfig,
}

impl WorkloadSpec {
    /// Load workload specification from a YAML file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read workload spec: {}", e))?;
        
        serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse workload YAML: {}", e))
    }

    /// Load workload specification from a TOML file.
    pub fn load_toml(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read workload spec: {}", e))?;
        
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse workload TOML: {}", e))
    }

    /// Get dataset configuration for a tier.
    pub fn dataset_config(&self, tier: DatasetTier) -> Option<&DatasetConfig> {
        self.datasets.get(&tier.to_string())
    }

    /// Get the total record count for a tier.
    pub fn record_count(&self, tier: DatasetTier) -> Option<usize> {
        let config = self.dataset_config(tier)?;
        
        // Check various count fields
        config.total
            .or(config.vectors)
            .or(config.chunks)
            .or(config.events)
            .or_else(|| {
                match (config.agents, config.memories_per_agent) {
                    (Some(a), Some(m)) => Some(a * m),
                    _ => None,
                }
            })
    }

    /// Check if workload requires vector search.
    pub fn requires_vector_search(&self) -> bool {
        self.query_patterns.iter().any(|p| matches!(p, QueryPattern::VectorSearch { .. }))
    }

    /// Check if workload requires time-range queries.
    pub fn requires_time_range(&self) -> bool {
        self.query_patterns.iter().any(|p| matches!(p, QueryPattern::TimeRangeScan { .. }))
    }

    /// Get vector dimension from schema.
    pub fn vector_dimension(&self) -> Option<usize> {
        self.data_model.as_ref()?.schema.iter()
            .find(|f| matches!(f.field_type, FieldType::Vector))
            .and_then(|f| f.dimension)
    }
}

/// Run configuration for a benchmark.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub workload: WorkloadSpec,
    pub tier: DatasetTier,
    pub concurrency: usize,
    pub duration_secs: u64,
    pub warmup_secs: u64,
    pub k_values: Vec<usize>,
}

impl RunConfig {
    /// Create a quick test configuration.
    pub fn quick_test(workload: WorkloadSpec) -> Self {
        Self {
            workload,
            tier: DatasetTier::Small,
            concurrency: 1,
            duration_secs: 30,
            warmup_secs: 5,
            k_values: vec![10],
        }
    }

    /// Create a PR gate configuration.
    pub fn pr_gate(workload: WorkloadSpec) -> Self {
        Self {
            workload,
            tier: DatasetTier::Small,
            concurrency: 4,
            duration_secs: 180,
            warmup_secs: 30,
            k_values: vec![10, 20],
        }
    }

    /// Create a nightly gate configuration.
    pub fn nightly_gate(workload: WorkloadSpec, tier: DatasetTier) -> Self {
        Self {
            workload,
            tier,
            concurrency: 16,
            duration_secs: 300,
            warmup_secs: 60,
            k_values: vec![10, 20, 50],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_tier_display() {
        assert_eq!(DatasetTier::Small.to_string(), "small");
        assert_eq!(DatasetTier::Medium.to_string(), "medium");
        assert_eq!(DatasetTier::Large.to_string(), "large");
    }

    #[test]
    fn test_parse_workload_yaml() {
        let yaml = r#"
workload:
  name: test_workload
  description: Test workload for unit tests

query_patterns:
  - type: vector_search
    top_k: [10, 20]
    time_decay: true

datasets:
  small:
    vectors: 100000
    seed: 42
  medium:
    vectors: 5000000
    seed: 42

quality_metrics:
  - recall@10
  - recall@20

concurrency:
  levels: [1, 4, 16]
"#;

        let spec: WorkloadSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.workload.name, "test_workload");
        assert!(spec.requires_vector_search());
        assert_eq!(spec.record_count(DatasetTier::Small), Some(100000));
    }
}
