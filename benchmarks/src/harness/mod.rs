// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Performance test harness for ToonDB
//!
//! This module provides:
//! - HDRHistogram-based latency measurement
//! - Recall@k evaluation with ground truth
//! - Workload specification parsing
//! - PMU/perf counter integration (Linux)
//! - Regression gate evaluation

pub mod config;
pub mod environment;
pub mod gates;
pub mod metrics;
pub mod pmu;
pub mod recall;
pub mod reporter;

pub use config::{WorkloadSpec, DatasetTier, QueryPattern};
pub use environment::EnvironmentInfo;
pub use gates::{RegressionGates, GateResult};
pub use metrics::{MetricsCollector, LatencyHistogram};
pub use pmu::{PmuCollector, PmuSnapshot};
pub use recall::RecallEvaluator;
pub use reporter::RunReport;
