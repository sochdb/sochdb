// SPDX-License-Identifier: AGPL-3.0-or-later

//! Query planner: SQL AST → Volcano operator tree.
//!
//! Converts a parsed SQL `SelectStmt` into a tree of physical operators
//! that can be executed row-at-a-time via the Volcano model.
//!
//! ```text
//! SelectStmt
//!   → FROM → SeqScan / Join tree
//!   → WHERE → Filter
//!   → GROUP BY + aggregates → HashAggregate
//!   → HAVING → Filter
//!   → SELECT → Project
//!   → ORDER BY → Sort
//!   → LIMIT/OFFSET → Limit
//! ```

use crate::optimizer_integration::StorageBackend;
use crate::sql::ast::*;
use super::aggregate::{AggDef, AggFunc, HashAggregateNode};
use super::filter::FilterNode;
use super::join::{HashJoinNode, NestedLoopJoinNode};
use super::limit::LimitNode;
use super::node::PlanNode;
use super::project::{ProjectExpr, ProjectNode};
use super::scan::{EmptyNode, SeqScanNode};
use super::sort::{SortKey, SortNode};
use super::types::Schema;
use sochdb_core::Result;
use std::sync::Arc;

/// Query planner that converts SQL AST to Volcano operator trees.
pub struct QueryPlanner {
    storage: Arc<dyn StorageBackend>,
}

impl QueryPlanner {
    pub fn new(storage: Arc<dyn StorageBackend>) -> Self {
        Self { storage }
    }

    /// Plan a SELECT statement into an operator tree.
    pub fn plan_select(&self, select: &SelectStmt) -> Result<Box<dyn PlanNode>> {
        // 1. FROM clause → base scan / join tree
        let mut node = self.plan_from(&select.from)?;

        // 2. WHERE clause → Filter
        if let Some(where_expr) = &select.where_clause {
            node = Box::new(FilterNode::new(node, where_expr.clone()));
        }

        // 3. Detect aggregates in SELECT list
        let has_aggregates = self.has_aggregate_in_select(&select.columns);
        let has_group_by = !select.group_by.is_empty();

        if has_aggregates || has_group_by {
            // GROUP BY + aggregates → HashAggregate
            let (agg_defs, group_by_exprs) =
                self.extract_aggregates(&select.columns, &select.group_by)?;
            node = Box::new(HashAggregateNode::new(node, group_by_exprs, agg_defs));

            // HAVING → Filter (operates on aggregate output)
            if let Some(having) = &select.having {
                node = Box::new(FilterNode::new(node, having.clone()));
            }
        } else {
            // 4. SELECT → Project (non-aggregate case)
            let needs_projection = !self.is_wildcard_only(&select.columns);
            if needs_projection {
                let exprs = self.plan_select_exprs(&select.columns, node.schema())?;
                if !exprs.is_empty() {
                    node = Box::new(ProjectNode::new(node, exprs));
                }
            }
        }

        // 5. DISTINCT — implement as a sort + dedup or hash-based
        // (simplified: not yet implemented, would need a DistinctNode)

        // 6. ORDER BY → Sort
        if !select.order_by.is_empty() {
            let sort_keys = self.plan_order_by(&select.order_by)?;
            node = Box::new(SortNode::new(node, sort_keys));
        }

        // 7. LIMIT / OFFSET → Limit
        let limit = self.extract_usize(&select.limit)?;
        let offset = self.extract_usize(&select.offset)?.unwrap_or(0);
        if limit.is_some() || offset > 0 {
            node = Box::new(LimitNode::new(node, limit, offset));
        }

        Ok(node)
    }

    // ========================================================================
    // FROM clause planning
    // ========================================================================

    fn plan_from(&self, from: &Option<FromClause>) -> Result<Box<dyn PlanNode>> {
        let from = match from {
            Some(f) => f,
            None => {
                // No FROM: return a single empty row (for SELECT 1+1, etc.)
                return Ok(Box::new(super::scan::ValuesNode::new(
                    Schema::empty(),
                    vec![vec![]],
                )));
            }
        };

        if from.tables.is_empty() {
            return Ok(Box::new(EmptyNode::new(Schema::empty())));
        }

        // Plan first table
        let mut node = self.plan_table_ref(&from.tables[0])?;

        // Implicit cross join for multiple tables in FROM
        for table_ref in from.tables.iter().skip(1) {
            let right = self.plan_table_ref(table_ref)?;
            node = Box::new(NestedLoopJoinNode::new(
                node,
                right,
                None, // CROSS JOIN
                JoinType::Cross,
            ));
        }

        Ok(node)
    }

    fn plan_table_ref(&self, table_ref: &TableRef) -> Result<Box<dyn PlanNode>> {
        match table_ref {
            TableRef::Table { name, alias } => {
                let table_name = name.name().to_string();
                // Start with wildcard scan; projection will be added later
                Ok(Box::new(SeqScanNode::new(
                    self.storage.clone(),
                    table_name,
                    vec!["*".to_string()],
                    alias.as_deref(),
                )))
            }

            TableRef::Join {
                left,
                join_type,
                right,
                condition,
            } => self.plan_join(left, *join_type, right, condition),

            TableRef::Subquery { query, alias: _ } => self.plan_select(query),

            TableRef::Function { .. } => Err(sochdb_core::SochDBError::Internal(
                "Table-valued functions not yet supported".into(),
            )),
        }
    }

    fn plan_join(
        &self,
        left_ref: &TableRef,
        join_type: JoinType,
        right_ref: &TableRef,
        condition: &Option<JoinCondition>,
    ) -> Result<Box<dyn PlanNode>> {
        let left = self.plan_table_ref(left_ref)?;
        let right = self.plan_table_ref(right_ref)?;

        match condition {
            Some(JoinCondition::On(expr)) => {
                // Try to detect equi-join for HashJoin optimization
                if let Some((left_key, right_key)) = self.extract_equi_keys(expr) {
                    Ok(Box::new(HashJoinNode::new(
                        left, right, left_key, right_key, join_type,
                    )))
                } else {
                    // Theta join — use nested loop
                    Ok(Box::new(NestedLoopJoinNode::new(
                        left,
                        right,
                        Some(expr.clone()),
                        join_type,
                    )))
                }
            }
            Some(JoinCondition::Using(columns)) => {
                // USING(col) → equi-join on col = col
                if let Some(col) = columns.first() {
                    let left_key = Expr::Column(ColumnRef::new(col.clone()));
                    let right_key = Expr::Column(ColumnRef::new(col.clone()));
                    Ok(Box::new(HashJoinNode::new(
                        left, right, left_key, right_key, join_type,
                    )))
                } else {
                    Ok(Box::new(NestedLoopJoinNode::new(
                        left, right, None, JoinType::Cross,
                    )))
                }
            }
            Some(JoinCondition::Natural) | None => {
                if join_type == JoinType::Cross {
                    Ok(Box::new(NestedLoopJoinNode::new(
                        left, right, None, JoinType::Cross,
                    )))
                } else {
                    // Natural join — would need schema introspection to find common columns
                    // For now, fall back to cross join
                    Ok(Box::new(NestedLoopJoinNode::new(
                        left, right, None, JoinType::Cross,
                    )))
                }
            }
        }
    }

    /// Try to extract equi-join keys from an ON expression.
    /// Returns (left_key_expr, right_key_expr) if the expression is `a.x = b.y`.
    fn extract_equi_keys(&self, expr: &Expr) -> Option<(Expr, Expr)> {
        match expr {
            Expr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => Some((*left.clone(), *right.clone())),
            _ => None,
        }
    }

    // ========================================================================
    // SELECT list / Projection
    // ========================================================================

    fn is_wildcard_only(&self, items: &[SelectItem]) -> bool {
        items.len() == 1 && matches!(&items[0], SelectItem::Wildcard)
    }

    fn plan_select_exprs(
        &self,
        items: &[SelectItem],
        _input_schema: &Schema,
    ) -> Result<Vec<ProjectExpr>> {
        let mut exprs = Vec::new();

        for item in items {
            match item {
                SelectItem::Wildcard => {
                    // Wildcard — pass-through handled separately
                    return Ok(vec![]);
                }
                SelectItem::QualifiedWildcard(_table) => {
                    // table.* — would need schema lookup
                    return Ok(vec![]);
                }
                SelectItem::Expr { expr, alias } => {
                    let name = alias.clone().unwrap_or_else(|| match expr {
                        Expr::Column(col) => col.column.clone(),
                        Expr::Function(func) => {
                            let args_str = if func.args.is_empty() {
                                "*".to_string()
                            } else {
                                "...".to_string()
                            };
                            format!("{}({})", func.name.name(), args_str)
                        }
                        _ => "?column?".to_string(),
                    });
                    exprs.push(ProjectExpr {
                        expr: expr.clone(),
                        alias: name,
                    });
                }
            }
        }

        Ok(exprs)
    }

    // ========================================================================
    // Aggregate detection and extraction
    // ========================================================================

    fn has_aggregate_in_select(&self, items: &[SelectItem]) -> bool {
        for item in items {
            if let SelectItem::Expr { expr, .. } = item {
                if self.expr_has_aggregate(expr) {
                    return true;
                }
            }
        }
        false
    }

    fn expr_has_aggregate(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Function(func) => {
                let name = func.name.name().to_uppercase();
                matches!(
                    name.as_str(),
                    "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"
                )
            }
            Expr::BinaryOp { left, right, .. } => {
                self.expr_has_aggregate(left) || self.expr_has_aggregate(right)
            }
            Expr::UnaryOp { expr, .. } => self.expr_has_aggregate(expr),
            _ => false,
        }
    }

    fn extract_aggregates(
        &self,
        items: &[SelectItem],
        group_by: &[Expr],
    ) -> Result<(Vec<AggDef>, Vec<Expr>)> {
        let mut agg_defs = Vec::new();

        for item in items {
            if let SelectItem::Expr { expr, alias } = item {
                if let Some(agg_def) = self.try_extract_agg(expr, alias)? {
                    agg_defs.push(agg_def);
                }
                // Group-by columns are handled automatically by HashAggregateNode
            }
        }

        Ok((agg_defs, group_by.to_vec()))
    }

    fn try_extract_agg(
        &self,
        expr: &Expr,
        alias: &Option<String>,
    ) -> Result<Option<AggDef>> {
        match expr {
            Expr::Function(func) => {
                let name = func.name.name().to_uppercase();
                let func_type = match name.as_str() {
                    "COUNT" => {
                        if func.distinct {
                            Some(AggFunc::CountDistinct)
                        } else {
                            Some(AggFunc::Count)
                        }
                    }
                    "SUM" => Some(AggFunc::Sum),
                    "AVG" => Some(AggFunc::Avg),
                    "MIN" => Some(AggFunc::Min),
                    "MAX" => Some(AggFunc::Max),
                    _ => None,
                };

                if let Some(func_type) = func_type {
                    let agg_expr = if func.args.is_empty()
                        || (func.args.len() == 1
                            && matches!(&func.args[0], Expr::Column(c) if c.column == "*"))
                    {
                        None // COUNT(*)
                    } else {
                        Some(func.args[0].clone())
                    };

                    let output_name = alias.clone().unwrap_or_else(|| {
                        let args_str = if func.args.is_empty() {
                            "*".to_string()
                        } else {
                            match &func.args[0] {
                                Expr::Column(c) => c.column.clone(),
                                _ => "expr".to_string(),
                            }
                        };
                        format!("{}({})", name.to_lowercase(), args_str)
                    });

                    Ok(Some(AggDef {
                        func: func_type,
                        expr: agg_expr,
                        alias: output_name,
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    // ========================================================================
    // ORDER BY
    // ========================================================================

    fn plan_order_by(&self, items: &[OrderByItem]) -> Result<Vec<SortKey>> {
        Ok(items
            .iter()
            .map(|item| SortKey {
                expr: item.expr.clone(),
                ascending: item.asc,
                nulls_first: item.nulls_first.unwrap_or(!item.asc),
            })
            .collect())
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    fn extract_usize(&self, expr: &Option<Expr>) -> Result<Option<usize>> {
        match expr {
            Some(Expr::Literal(Literal::Integer(n))) => Ok(Some(*n as usize)),
            Some(_) => Err(sochdb_core::SochDBError::Internal(
                "LIMIT/OFFSET must be an integer literal".into(),
            )),
            None => Ok(None),
        }
    }
}

/// Generate a textual EXPLAIN representation for a SELECT statement.
pub fn explain_select(select: &SelectStmt, _storage: &Arc<dyn StorageBackend>) -> String {
    let mut lines = Vec::new();

    // Simplified EXPLAIN output
    if let Some(from) = &select.from {
        for table_ref in &from.tables {
            explain_table_ref(table_ref, &mut lines, 0);
        }
    }

    if select.where_clause.is_some() {
        lines.push("  Filter (WHERE)".to_string());
    }

    if !select.group_by.is_empty() {
        let cols: Vec<String> = select.group_by.iter().map(|e| format!("{:?}", e)).collect();
        lines.push(format!("  HashAggregate [group_by={}]", cols.join(", ")));
    }

    if select.having.is_some() {
        lines.push("  Filter (HAVING)".to_string());
    }

    // Check for aggregates in SELECT
    let has_agg = select.columns.iter().any(|item| {
        if let SelectItem::Expr { expr, .. } = item {
            matches!(expr, Expr::Function(f) if {
                let n = f.name.name().to_uppercase();
                matches!(n.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX")
            })
        } else {
            false
        }
    });
    if has_agg && select.group_by.is_empty() {
        lines.push("  HashAggregate [global]".to_string());
    }

    let col_names: Vec<String> = select.columns.iter().map(|item| {
        match item {
            SelectItem::Wildcard => "*".to_string(),
            SelectItem::QualifiedWildcard(t) => format!("{}.*", t),
            SelectItem::Expr { expr, alias } => {
                alias.clone().unwrap_or_else(|| format!("{:?}", expr))
            }
        }
    }).collect();
    lines.push(format!("  Project [{}]", col_names.join(", ")));

    if !select.order_by.is_empty() {
        let orders: Vec<String> = select.order_by.iter().map(|o| {
            let dir = if o.asc { "ASC" } else { "DESC" };
            format!("{:?} {}", o.expr, dir)
        }).collect();
        lines.push(format!("  Sort [{}]", orders.join(", ")));
    }

    if select.limit.is_some() || select.offset.is_some() {
        lines.push(format!(
            "  Limit [limit={:?}, offset={:?}]",
            select.limit, select.offset
        ));
    }

    lines.join("\n")
}

fn explain_table_ref(table_ref: &TableRef, lines: &mut Vec<String>, depth: usize) {
    let indent = "  ".repeat(depth);
    match table_ref {
        TableRef::Table { name, alias } => {
            let alias_str = alias.as_ref().map_or(String::new(), |a| format!(" AS {}", a));
            lines.push(format!("{}SeqScan [table={}{}]", indent, name, alias_str));
        }
        TableRef::Join {
            left,
            join_type,
            right,
            condition,
        } => {
            let jt = match join_type {
                JoinType::Inner => "INNER",
                JoinType::Left => "LEFT",
                JoinType::Right => "RIGHT",
                JoinType::Full => "FULL",
                JoinType::Cross => "CROSS",
            };
            let cond_str = match condition {
                Some(JoinCondition::On(expr)) => format!(" ON {:?}", expr),
                Some(JoinCondition::Using(cols)) => format!(" USING({})", cols.join(", ")),
                Some(JoinCondition::Natural) => " NATURAL".to_string(),
                None => String::new(),
            };
            lines.push(format!("{}{} JOIN{}", indent, jt, cond_str));
            explain_table_ref(left, lines, depth + 1);
            explain_table_ref(right, lines, depth + 1);
        }
        TableRef::Subquery { alias, .. } => {
            lines.push(format!("{}Subquery [alias={}]", indent, alias));
        }
        TableRef::Function { name, .. } => {
            lines.push(format!("{}Function [{}]", indent, name));
        }
    }
}
