// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Probabilistic Data Structures for Streaming Analytics
//!
//! This module provides memory-efficient streaming algorithms for:
//! - DDSketch: O(1) percentile queries (P50, P90, P95, P99)
//! - HyperLogLog: Cardinality estimation (unique counts)
//! - ExponentialHistogram: Mergeable histograms for rollups
//! - CountMinSketch: Frequency estimation for top-K queries
//! - AdaptiveSketch: Memory-efficient latency tracking (sparse → dense)

pub mod adaptive_sketch;
pub mod count_min_sketch;
pub mod ddsketch;
pub mod exponential_histogram;
pub mod hyperloglog;

pub use adaptive_sketch::{AdaptiveSketch, SketchPercentiles, SparseBuffer};
pub use count_min_sketch::CountMinSketch;
pub use ddsketch::DDSketch;
pub use exponential_histogram::ExponentialHistogram;
pub use hyperloglog::HyperLogLog;
