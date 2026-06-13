//! Release gate definitions.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateCategory {
    StorageCorrectness,
    ConcurrencyMvcc,
    FfiSafety,
    Packaging,
    CiQuality,
    Performance,
    BackwardCompatibility,
    Security,
    ReleaseOps,
}

impl GateCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::StorageCorrectness => "storage_correctness",
            Self::ConcurrencyMvcc => "concurrency_mvcc",
            Self::FfiSafety => "ffi_safety",
            Self::Packaging => "packaging",
            Self::CiQuality => "ci_quality",
            Self::Performance => "performance",
            Self::BackwardCompatibility => "backward_compatibility",
            Self::Security => "security",
            Self::ReleaseOps => "release_ops",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::StorageCorrectness => "1. Storage Correctness",
            Self::ConcurrencyMvcc => "2. Concurrency & MVCC",
            Self::FfiSafety => "3. FFI & SDK Safety",
            Self::Packaging => "4. Packaging Quality",
            Self::CiQuality => "5. CI / Release Gate",
            Self::Performance => "6. Performance Thresholds",
            Self::BackwardCompatibility => "7. Backward Compatibility",
            Self::Security => "8. Security & Supply Chain",
            Self::ReleaseOps => "9. Release Operations",
        }
    }

    pub fn all() -> &'static [GateCategory] {
        &[
            Self::StorageCorrectness,
            Self::ConcurrencyMvcc,
            Self::FfiSafety,
            Self::Packaging,
            Self::CiQuality,
            Self::Performance,
            Self::BackwardCompatibility,
            Self::Security,
            Self::ReleaseOps,
        ]
    }
}

impl std::str::FromStr for GateCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "storage_correctness" | "storage" => Ok(Self::StorageCorrectness),
            "concurrency_mvcc" | "concurrency" => Ok(Self::ConcurrencyMvcc),
            "ffi_safety" | "ffi" => Ok(Self::FfiSafety),
            "packaging" => Ok(Self::Packaging),
            "ci_quality" | "ci" => Ok(Self::CiQuality),
            "performance" | "perf" => Ok(Self::Performance),
            "backward_compatibility" | "compat" => Ok(Self::BackwardCompatibility),
            "security" => Ok(Self::Security),
            "release_ops" | "release" => Ok(Self::ReleaseOps),
            other => Err(format!("unknown category: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatePriority {
    Blocker,
    Warning,
    Advisory,
}

impl GatePriority {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Blocker => "blocker",
            Self::Warning => "warning",
            Self::Advisory => "advisory",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    LiveTest,
    LiveCommand,
    LoomTest,
    StaticFile,
    StaticGrep,
    StaticGrepAbsence,
    StaticMultiFile,
    Simulated,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseGate {
    pub id: String,
    pub category: String,
    pub priority: GatePriority,
    pub title: String,
    pub description: String,
    pub kind: GateKind,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub paths: Vec<String>,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub min_matches: Option<usize>,
    #[serde(default)]
    pub scenario: Option<String>,
    pub expected: String,
}

impl ReleaseGate {
    pub fn category_enum(&self) -> GateCategory {
        self.category.parse().unwrap_or(GateCategory::ReleaseOps)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseGateFile {
    pub source: String,
    pub gates: Vec<ReleaseGate>,
}
