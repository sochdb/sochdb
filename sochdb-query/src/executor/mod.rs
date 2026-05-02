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

pub mod types;
pub mod node;
pub mod eval;
pub mod scan;
pub mod filter;
pub mod project;
pub mod sort;
pub mod limit;
pub mod join;
pub mod aggregate;
pub mod explain;
pub mod planner;
pub mod pipeline;

#[cfg(test)]
mod tests;

// Re-exports
pub use types::{Row, Schema, ColumnMeta};
pub use node::PlanNode;
pub use eval::{eval_expr, eval_predicate};
pub use scan::{SeqScanNode, IndexSeekNode};
pub use filter::FilterNode;
pub use project::ProjectNode;
pub use sort::SortNode;
pub use limit::LimitNode;
pub use join::{HashJoinNode, NestedLoopJoinNode, MergeJoinNode};
pub use aggregate::HashAggregateNode;
pub use explain::ExplainNode;
pub use planner::QueryPlanner;
pub use pipeline::{execute_sql, execute_statement, ExecutorConfig};
