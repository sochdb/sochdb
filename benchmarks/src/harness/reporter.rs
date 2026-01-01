// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Run report generation with comprehensive output.

use super::config::DatasetTier;
use super::environment::EnvironmentInfo;
use super::gates::{GateEvaluation, MetricsSnapshot};
use super::metrics::MetricsSummary;
use super::recall::RecallMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Schema version for the run report.
const SCHEMA_VERSION: &str = "2.0";

/// Complete run report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    /// Schema version for compatibility
    pub schema_version: String,
    /// Unique run identifier
    pub run_id: String,
    /// UTC timestamp
    pub timestamp_utc: String,
    /// Build information
    pub build: BuildInfo,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Workload configuration
    pub workload: WorkloadInfo,
    /// Dataset information
    pub dataset: DatasetInfo,
    /// Performance metrics
    pub metrics: MetricsSummary,
    /// Quality metrics (recall)
    pub quality: QualityMetrics,
    /// PMU counters (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pmu: Option<PmuMetrics>,
    /// Gate evaluation results (if baseline provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gates: Option<GateEvaluation>,
    /// Paths to artifacts
    pub artifacts: ArtifactPaths,
}

/// Build information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub git_sha: String,
    pub git_dirty: bool,
    pub rustc_version: String,
    pub build_profile: String,
    pub features: Vec<String>,
}

/// Workload information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadInfo {
    pub name: String,
    pub description: String,
    pub config_hash: String,
    pub params: HashMap<String, serde_json::Value>,
}

/// Dataset information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub tier: String,
    pub record_count: usize,
    pub hash: String,
}

/// Quality metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_10: Option<RecallMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_20: Option<RecallMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_50: Option<RecallMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ndcg_at_10: Option<f64>,
}

/// PMU/perf counter metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmuMetrics {
    pub cycles: u64,
    pub instructions: u64,
    pub ipc: f64,
    pub branch_instructions: u64,
    pub branch_misses: u64,
    pub branch_miss_rate: f64,
    pub l1d_loads: u64,
    pub l1d_load_misses: u64,
    pub l1d_miss_rate: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llc_loads: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llc_load_misses: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llc_miss_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtlb_load_misses: Option<u64>,
    pub context_switches: u64,
    pub page_faults: u64,
}

/// Paths to artifact files.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactPaths {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perf_stat: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flamegraph: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram_hdr: Option<String>,
}

impl RunReport {
    /// Create a new run report builder.
    pub fn builder() -> RunReportBuilder {
        RunReportBuilder::new()
    }

    /// Save report to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize report: {}", e))?;
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create report directory: {}", e))?;
        }

        fs::write(path, json)
            .map_err(|e| format!("Failed to write report: {}", e))?;

        Ok(())
    }

    /// Load report from JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read report: {}", e))?;
        
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse report: {}", e))
    }

    /// Convert to metrics snapshot for gate evaluation.
    pub fn to_snapshot(&self) -> MetricsSnapshot {
        let mut recall = HashMap::new();
        
        if let Some(ref r) = self.quality.recall_at_10 {
            recall.insert("recall@10".to_string(), r.mean);
        }
        if let Some(ref r) = self.quality.recall_at_20 {
            recall.insert("recall@20".to_string(), r.mean);
        }
        if let Some(ref r) = self.quality.recall_at_50 {
            recall.insert("recall@50".to_string(), r.mean);
        }

        MetricsSnapshot {
            qps: Some(self.metrics.qps),
            insert_rate: Some(self.metrics.insert_rate),
            p50_latency_ns: Some(self.metrics.query_latency.p50_ns),
            p95_latency_ns: Some(self.metrics.query_latency.p95_ns),
            p99_latency_ns: Some(self.metrics.query_latency.p99_ns),
            p999_latency_ns: Some(self.metrics.query_latency.p999_ns),
            recall,
            branch_miss_rate: self.pmu.as_ref().map(|p| p.branch_miss_rate),
            l1d_miss_rate: self.pmu.as_ref().map(|p| p.l1d_miss_rate),
            ipc: self.pmu.as_ref().map(|p| p.ipc),
        }
    }

    /// Print summary to stdout.
    pub fn print_summary(&self) {
        println!("\n========================================");
        println!("  Benchmark Results: {}", self.workload.name);
        println!("========================================");
        println!("Run ID: {}", self.run_id);
        println!("Duration: {:.2}s", self.metrics.duration_s);
        println!();

        println!("Performance:");
        println!("  QPS: {:.2}", self.metrics.qps);
        println!("  Insert Rate: {:.2} ops/s", self.metrics.insert_rate);
        println!();

        println!("Latency:");
        println!("  p50:   {:.3} ms", self.metrics.query_latency.p50_ms());
        println!("  p95:   {:.3} ms", self.metrics.query_latency.p95_ms());
        println!("  p99:   {:.3} ms", self.metrics.query_latency.p99_ms());
        println!("  p99.9: {:.3} ms", self.metrics.query_latency.p999_ms());
        println!();

        if self.quality.recall_at_10.is_some() {
            println!("Quality:");
            if let Some(ref r) = self.quality.recall_at_10 {
                println!("  Recall@10: {:.4}", r.mean);
            }
            if let Some(ref r) = self.quality.recall_at_20 {
                println!("  Recall@20: {:.4}", r.mean);
            }
            println!();
        }

        if let Some(ref pmu) = self.pmu {
            println!("PMU Counters:");
            println!("  IPC: {:.2}", pmu.ipc);
            println!("  Branch miss rate: {:.2}%", pmu.branch_miss_rate * 100.0);
            println!("  L1D miss rate: {:.2}%", pmu.l1d_miss_rate * 100.0);
            println!();
        }

        if let Some(ref gates) = self.gates {
            println!("Gate Evaluation: {}", if gates.overall_pass { "PASS ✅" } else { "FAIL ❌" });
            for result in &gates.results {
                match result {
                    super::gates::GateResult::Pass => {}
                    super::gates::GateResult::Improvement { metric, delta_pct } => {
                        println!("  ✅ {}: +{:.1}% (improvement)", metric, delta_pct);
                    }
                    super::gates::GateResult::Warning { metric, delta_pct, threshold_pct } => {
                        println!("  ⚠️  {}: {:.1}% (threshold: {}%)", metric, delta_pct, threshold_pct);
                    }
                    super::gates::GateResult::Fail { metric, delta_pct, threshold_pct } => {
                        println!("  ❌ {}: {:.1}% (threshold: {}%)", metric, delta_pct, threshold_pct);
                    }
                }
            }
            println!();
        }

        if let Some(ref report_path) = self.artifacts.report_json {
            println!("Saved to: {}", report_path);
        }
    }
}

/// Builder for RunReport.
#[derive(Clone)]
pub struct RunReportBuilder {
    run_id: Option<String>,
    timestamp: Option<String>,
    environment: Option<EnvironmentInfo>,
    workload_name: Option<String>,
    workload_desc: Option<String>,
    dataset_name: Option<String>,
    dataset_tier: Option<DatasetTier>,
    record_count: Option<usize>,
    metrics: Option<MetricsSummary>,
    quality: QualityMetrics,
    pmu: Option<PmuMetrics>,
    gates: Option<GateEvaluation>,
    artifacts: ArtifactPaths,
}

impl RunReportBuilder {
    pub fn new() -> Self {
        Self {
            run_id: None,
            timestamp: None,
            environment: None,
            workload_name: None,
            workload_desc: None,
            dataset_name: None,
            dataset_tier: None,
            record_count: None,
            metrics: None,
            quality: QualityMetrics::default(),
            pmu: None,
            gates: None,
            artifacts: ArtifactPaths::default(),
        }
    }

    pub fn run_id(mut self, id: &str) -> Self {
        self.run_id = Some(id.to_string());
        self
    }

    pub fn timestamp(mut self, ts: &str) -> Self {
        self.timestamp = Some(ts.to_string());
        self
    }

    pub fn environment(mut self, env: EnvironmentInfo) -> Self {
        self.environment = Some(env);
        self
    }

    pub fn workload(mut self, name: &str, description: &str) -> Self {
        self.workload_name = Some(name.to_string());
        self.workload_desc = Some(description.to_string());
        self
    }

    pub fn dataset(mut self, name: &str, tier: DatasetTier, count: usize) -> Self {
        self.dataset_name = Some(name.to_string());
        self.dataset_tier = Some(tier);
        self.record_count = Some(count);
        self
    }

    pub fn metrics(mut self, m: MetricsSummary) -> Self {
        self.metrics = Some(m);
        self
    }

    pub fn recall_at_10(mut self, r: RecallMetrics) -> Self {
        self.quality.recall_at_10 = Some(r);
        self
    }

    pub fn recall_at_20(mut self, r: RecallMetrics) -> Self {
        self.quality.recall_at_20 = Some(r);
        self
    }

    pub fn recall_at_50(mut self, r: RecallMetrics) -> Self {
        self.quality.recall_at_50 = Some(r);
        self
    }

    pub fn pmu(mut self, p: PmuMetrics) -> Self {
        self.pmu = Some(p);
        self
    }

    pub fn gates(mut self, g: GateEvaluation) -> Self {
        self.gates = Some(g);
        self
    }

    pub fn artifact_report(mut self, path: &str) -> Self {
        self.artifacts.report_json = Some(path.to_string());
        self
    }

    pub fn artifact_flamegraph(mut self, path: &str) -> Self {
        self.artifacts.flamegraph = Some(path.to_string());
        self
    }

    pub fn build(self) -> Result<RunReport, String> {
        let env = self.environment.ok_or("Environment info required")?;
        let metrics = self.metrics.ok_or("Metrics required")?;

        Ok(RunReport {
            schema_version: SCHEMA_VERSION.to_string(),
            run_id: self.run_id.unwrap_or_else(|| "unknown".to_string()),
            timestamp_utc: self.timestamp.unwrap_or_else(|| "unknown".to_string()),
            build: BuildInfo {
                git_sha: env.git_sha.clone(),
                git_dirty: env.git_dirty,
                rustc_version: env.rustc_version.clone(),
                build_profile: env.build_profile.clone(),
                features: env.features.clone(),
            },
            environment: env,
            workload: WorkloadInfo {
                name: self.workload_name.unwrap_or_else(|| "unknown".to_string()),
                description: self.workload_desc.unwrap_or_default(),
                config_hash: "pending".to_string(),
                params: HashMap::new(),
            },
            dataset: DatasetInfo {
                name: self.dataset_name.unwrap_or_else(|| "default".to_string()),
                tier: self.dataset_tier.map(|t| t.to_string()).unwrap_or_else(|| "small".to_string()),
                record_count: self.record_count.unwrap_or(0),
                hash: "pending".to_string(),
            },
            metrics,
            quality: self.quality,
            pmu: self.pmu,
            gates: self.gates,
            artifacts: self.artifacts,
        })
    }
}

impl Default for RunReportBuilder {
    fn default() -> Self {
        Self::new()
    }
}
