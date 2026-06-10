// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # Unified Volcano Query Executor (v1.0)
//!
//! Single pipeline for all SQL execution:
//!
//! ```text
//! SQL Text → Parser → AST → Planner → Volcano Operator Tree → Row-at-a-time → Result
//! ```
//!
//! ## Architecture
//!
//! All operators implement the [`PlanNode`] trait (Volcano iterator model):
//!
//! ```text
//! trait PlanNode {
//!     fn schema(&self) -> &Schema;
//!     fn next(&mut self) -> Result<Option<Row>>;
//! }
//! ```
//!
//! Operators form a tree: each `next()` call pulls one row from its children,
//! processes it, and returns the result. This enables streaming execution
//! with minimal memory footprint.
//!
//! ## Operators
//!
//! | Operator       | Description                                |
//! |----------------|--------------------------------------------|
//! | SeqScan        | Full table scan via StorageBackend         |
//! | IndexSeek      | Index-based lookup via StorageBackend      |
//! | Filter         | Predicate evaluation (WHERE)               |
//! | Project        | Column selection + expression eval         |
//! | Sort           | In-memory sort (materializing)             |
//! | Limit          | LIMIT + OFFSET                             |
//! | HashJoin       | Hash-based equi-join                       |
//! | NestedLoopJoin | Nested loop join (theta joins)             |
//! | MergeJoin      | Merge join on sorted inputs                |
//! | HashAggregate  | GROUP BY + aggregate functions              |
//! | Explain        | EXPLAIN plan output                        |
//! | Values         | Inline VALUES (...) rows                   |
//! | Empty          | Returns no rows                            |

pub mod aggregate;
pub mod eval;
pub mod explain;
pub mod filter;
pub mod join;
pub mod limit;
pub mod node;
pub mod pipeline;
pub mod planner;
pub mod project;
pub mod scan;
pub mod sort;
pub mod types;

#[cfg(test)]
mod tests;

// Re-exports
pub use aggregate::HashAggregateNode;
pub use eval::{eval_expr, eval_predicate};
pub use explain::ExplainNode;
pub use filter::FilterNode;
pub use join::{HashJoinNode, MergeJoinNode, NestedLoopJoinNode};
pub use limit::LimitNode;
pub use node::PlanNode;
pub use pipeline::{ExecutorConfig, execute_sql, execute_statement};
pub use planner::QueryPlanner;
pub use project::ProjectNode;
pub use scan::{IndexSeekNode, SeqScanNode};
pub use sort::SortNode;
pub use types::{ColumnMeta, Row, Schema};
