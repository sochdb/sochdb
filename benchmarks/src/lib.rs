// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! ToonDB Benchmarks Library
//!
//! This crate provides:
//! - Performance test harness with HDRHistogram metrics
//! - Recall@k evaluation with ground truth
//! - Workload specification parsing
//! - Regression gate evaluation

pub mod harness;

pub use harness::*;
