// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Query Optimizer with Cost-Based Planning
//!
//! Provides cost-based query optimization for SOCH-QL.
//! Selects index, estimates cardinality, and chooses the cheapest plan.

use std::collections::HashMap;

/// Query optimizer with cost-based planning
#[derive(Debug, Default)]
pub struct QueryOptimizer {
    cost_model: CostModel,
    cardinality_hints: HashMap<String, CardinalityHint>,
    total_edges: usize,
}

/// Cost model configuration
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Cost of reading a single row
    pub row_read_cost: f64,
    /// Cost of index lookup
    pub index_lookup_cost: f64,
    /// Cost of sequential scan per row
    pub seq_scan_cost: f64,
    /// Cost of vector search per vector
    pub vector_search_cost: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            row_read_cost: 1.0,
            index_lookup_cost: 0.5,
            seq_scan_cost: 0.1,
            vector_search_cost: 10.0,
        }
    }
}

/// Cardinality hint for a column or index
#[derive(Debug, Clone)]
pub struct CardinalityHint {
    /// Estimated cardinality
    pub cardinality: usize,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Source of the estimate
    pub source: CardinalitySource,
}

/// Source of cardinality estimate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardinalitySource {
    /// Exact count
    Exact,
    /// HyperLogLog estimate
    HyperLogLog,
    /// Histogram estimate
    Histogram,
    /// Default/unknown
    Default,
}

/// Query predicate for optimization
#[derive(Debug, Clone)]
pub enum QueryPredicate {
    /// Equality predicate
    Eq(String, String),
    /// Range predicate
    Range {
        column: String,
        min: Option<String>,
        max: Option<String>,
    },
    /// In list predicate
    In(String, Vec<String>),
    /// Prefix match
    Prefix(String, String),
    /// Time range predicate
    TimeRange(u64, u64),
    /// Project filter
    Project(u16),
    /// Tenant filter
    Tenant(u32),
    /// Span type filter
    SpanType(String),
}

/// Query operation types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryOperation {
    /// Point lookup
    PointLookup,
    /// Range scan
    RangeScan,
    /// Full scan
    FullScan,
    /// Vector search
    VectorSearch { k: usize },
    /// LSM range scan
    LsmRangeScan { start_us: u64, end_us: u64 },
    /// Graph traversal
    GraphTraversal {
        direction: TraversalDirection,
        max_depth: usize,
    },
}

/// Index selection recommendation
#[derive(Debug, Clone)]
pub enum IndexSelection {
    /// Use primary key index
    PrimaryKey,
    /// Use secondary index
    Secondary(String),
    /// Use time-based index
    TimeIndex,
    /// Use vector index
    VectorIndex,
    /// Full table scan / LSM scan
    FullScan,
    /// LSM scan
    LsmScan,
    /// Causal index
    CausalIndex,
    /// Project index
    ProjectIndex,
    /// Multi-index intersection
    MultiIndex(Vec<IndexSelection>),
}

/// Query cost estimate
#[derive(Debug, Clone)]
pub struct QueryCost {
    /// Estimated cost
    pub estimated_cost: f64,
    /// Total cost (alias for compatibility)
    pub total_cost: f64,
    /// Estimated rows
    pub estimated_rows: usize,
    /// Records returned (alias)
    pub records_returned: usize,
    /// Selected index
    pub index: IndexSelection,
    /// Operation type
    pub operation: QueryOperation,
    /// Cost breakdown by operation
    pub breakdown: Vec<(QueryOperation, f64)>,
}

/// Traversal direction for range queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    Forward,
    Backward,
}

/// Query plan from optimizer
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Cost estimate
    pub cost: QueryCost,
    /// Predicates to apply
    pub predicates: Vec<QueryPredicate>,
    /// Traversal direction
    pub direction: TraversalDirection,
    /// Limit
    pub limit: Option<usize>,
    /// Index selection
    pub index_selection: IndexSelection,
    /// Operations to execute
    pub operations: Vec<QueryOperation>,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom cost model
    pub fn with_cost_model(cost_model: CostModel) -> Self {
        Self {
            cost_model,
            ..Default::default()
        }
    }

    /// Update total edge count for cardinality estimation
    pub fn update_total_edges(&mut self, count: usize, _source: CardinalitySource) {
        self.total_edges = count;
    }

    /// Set cardinality hint for a column
    pub fn set_cardinality_hint(&mut self, column: &str, hint: CardinalityHint) {
        self.cardinality_hints.insert(column.to_string(), hint);
    }

    /// Plan a query using cost-based index selection
    ///
    /// Evaluates each candidate index against a full scan, choosing the
    /// cheapest access path based on selectivity × row count.
    pub fn plan_query(&self, predicates: &[QueryPredicate], limit: Option<usize>) -> QueryPlan {
        if predicates.is_empty() {
            return self.build_plan(
                IndexSelection::FullScan,
                QueryOperation::FullScan,
                predicates,
                limit,
            );
        }

        // Score each candidate index
        let mut candidates: Vec<(IndexSelection, QueryOperation, f64, usize)> = Vec::new();

        for pred in predicates {
            let sel = self.estimate_selectivity(pred);
            let est_rows = (self.total_edges as f64 * sel).max(1.0) as usize;

            match pred {
                QueryPredicate::Eq(_, _) => {
                    let cost = self.cost_model.index_lookup_cost;
                    candidates.push((IndexSelection::PrimaryKey, QueryOperation::PointLookup, cost, 1));
                }
                QueryPredicate::TimeRange(start, end) => {
                    let cost = est_rows as f64 * self.cost_model.row_read_cost;
                    candidates.push((
                        IndexSelection::TimeIndex,
                        QueryOperation::LsmRangeScan {
                            start_us: *start,
                            end_us: *end,
                        },
                        cost,
                        est_rows,
                    ));
                }
                QueryPredicate::Range { .. } | QueryPredicate::Prefix(_, _) => {
                    let cost = est_rows as f64 * self.cost_model.row_read_cost;
                    candidates.push((
                        IndexSelection::LsmScan,
                        QueryOperation::RangeScan,
                        cost,
                        est_rows,
                    ));
                }
                QueryPredicate::Project(_) => {
                    let cost = est_rows as f64 * self.cost_model.index_lookup_cost;
                    candidates.push((
                        IndexSelection::ProjectIndex,
                        QueryOperation::RangeScan,
                        cost,
                        est_rows,
                    ));
                }
                QueryPredicate::Tenant(_) | QueryPredicate::SpanType(_) | QueryPredicate::In(_, _) => {
                    let cost = est_rows as f64 * self.cost_model.row_read_cost;
                    candidates.push((
                        IndexSelection::FullScan,
                        QueryOperation::RangeScan,
                        cost,
                        est_rows,
                    ));
                }
            }
        }

        // Full scan baseline
        let full_scan_cost = self.total_edges.max(1) as f64 * self.cost_model.seq_scan_cost;
        candidates.push((
            IndexSelection::FullScan,
            QueryOperation::FullScan,
            full_scan_cost,
            self.total_edges.max(1),
        ));

        // Choose cheapest
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let (index, operation, _cost, _rows) = candidates.into_iter().next().unwrap();

        self.build_plan(index, operation, predicates, limit)
    }

    /// Build a QueryPlan from chosen access path
    fn build_plan(
        &self,
        index: IndexSelection,
        operation: QueryOperation,
        predicates: &[QueryPredicate],
        limit: Option<usize>,
    ) -> QueryPlan {
        let estimated_rows = match &index {
            IndexSelection::PrimaryKey => 1,
            IndexSelection::FullScan => self.total_edges.max(1),
            _ => {
                let sel: f64 = predicates
                    .iter()
                    .map(|p| self.estimate_selectivity(p))
                    .product();
                (self.total_edges as f64 * sel).max(1.0) as usize
            }
        };

        let estimated_cost = match &operation {
            QueryOperation::PointLookup => self.cost_model.index_lookup_cost,
            QueryOperation::RangeScan => estimated_rows as f64 * self.cost_model.row_read_cost,
            QueryOperation::FullScan => self.total_edges as f64 * self.cost_model.seq_scan_cost,
            QueryOperation::VectorSearch { .. } => self.cost_model.vector_search_cost,
            QueryOperation::LsmRangeScan { .. } => estimated_rows as f64 * self.cost_model.row_read_cost,
            QueryOperation::GraphTraversal { .. } => estimated_rows as f64 * self.cost_model.row_read_cost,
        };

        QueryPlan {
            cost: QueryCost {
                estimated_cost,
                total_cost: estimated_cost,
                estimated_rows,
                records_returned: estimated_rows,
                index: index.clone(),
                operation: operation.clone(),
                breakdown: vec![(operation.clone(), estimated_cost)],
            },
            predicates: predicates.to_vec(),
            direction: TraversalDirection::Forward,
            limit,
            index_selection: index,
            operations: vec![operation],
        }
    }

    /// Estimate selectivity of a predicate
    pub fn estimate_selectivity(&self, predicate: &QueryPredicate) -> f64 {
        match predicate {
            QueryPredicate::Eq(col, _) => {
                if let Some(hint) = self.cardinality_hints.get(col) {
                    1.0 / hint.cardinality.max(1) as f64
                } else {
                    0.1 // Default 10% selectivity
                }
            }
            QueryPredicate::Range { .. } => 0.25, // Default 25% for range
            QueryPredicate::In(_, values) => (values.len() as f64 * 0.1).min(0.5),
            QueryPredicate::Prefix(_, _) => 0.15, // Default 15% for prefix
            QueryPredicate::TimeRange(_, _) => 0.2,
            QueryPredicate::Project(_) => 0.1,
            QueryPredicate::Tenant(_) => 0.05,
            QueryPredicate::SpanType(_) => 0.15,
        }
    }
}
