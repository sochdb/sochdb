//! Expected score targets from benchmarks, SLOs, and retrieval evaluations.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedTargetsFile {
    pub source: String,
    pub topology: String,
    pub targets: Vec<ExpectedTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedTarget {
    pub workload: String,
    #[serde(default)]
    pub throughput_ops_sec: Option<f64>,
    #[serde(default)]
    pub p50_us: Option<f64>,
    #[serde(default)]
    pub p99_us: Option<f64>,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default = "default_tolerance")]
    pub tolerance_pct: f64,
    #[serde(default)]
    pub unit: Option<String>,
}

fn default_tolerance() -> f64 {
    25.0
}

impl ExpectedTarget {
    pub fn unit_kind(&self) -> TargetUnit {
        match self.unit.as_deref() {
            Some("ratio") => TargetUnit::Ratio,
            Some("ms") => TargetUnit::Milliseconds,
            Some("latency_ceiling") => TargetUnit::LatencyCeiling,
            Some("throughput_floor") => TargetUnit::ThroughputFloor,
            Some("ratio_floor") => TargetUnit::RatioFloor,
            _ if self.score.is_some() => TargetUnit::Ratio,
            _ => TargetUnit::Throughput,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetUnit {
    Throughput,
    Ratio,
    Milliseconds,
    LatencyCeiling,
    ThroughputFloor,
    RatioFloor,
}

pub struct ExpectedStore {
    files: Vec<ExpectedTargetsFile>,
}

impl ExpectedStore {
    pub fn load_defaults() -> Self {
        let manifests = [
            include_str!("../expected/standalone_10k.json"),
            include_str!("../expected/distributed_grpc.json"),
            include_str!("../expected/retrieval_quality.json"),
            include_str!("../expected/slos.json"),
        ];
        let files = manifests
            .iter()
            .map(|m| serde_json::from_str(m).expect("valid embedded expected targets"))
            .collect();
        Self { files }
    }

    pub fn load_dir(dir: &Path) -> Result<Self, std::io::Error> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                let content = std::fs::read_to_string(&path)?;
                let file: ExpectedTargetsFile = serde_json::from_str(&content)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                files.push(file);
            }
        }
        Ok(Self { files })
    }

    pub fn targets_for_topology(&self, topology: &str) -> Vec<&ExpectedTarget> {
        self.files
            .iter()
            .filter(|f| f.topology == topology || f.topology == "any")
            .flat_map(|f| f.targets.iter())
            .collect()
    }

    pub fn target_by_workload(&self, workload: &str) -> Option<&ExpectedTarget> {
        self.files
            .iter()
            .flat_map(|f| f.targets.iter())
            .find(|t| t.workload == workload)
    }

    pub fn all_files(&self) -> &[ExpectedTargetsFile] {
        &self.files
    }
}
