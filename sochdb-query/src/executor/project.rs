// SPDX-License-Identifier: AGPL-3.0-or-later

//! Project operator (column selection and expression evaluation).

use crate::sql::ast::Expr;
use super::eval::eval_expr;
use super::node::PlanNode;
use super::types::{Row, Schema, ColumnMeta};
use sochdb_core::Result;

/// Projection expression: an expression and its output alias.
pub struct ProjectExpr {
    pub expr: Expr,
    pub alias: String,
}

/// Project operator: evaluates expressions to produce a new row shape.
///
/// ```text
/// Project(exprs=[a, b+1 AS c])
///   └── input
/// ```
pub struct ProjectNode {
    input: Box<dyn PlanNode>,
    exprs: Vec<ProjectExpr>,
    output_schema: Schema,
}

impl ProjectNode {
    pub fn new(input: Box<dyn PlanNode>, exprs: Vec<ProjectExpr>) -> Self {
        let output_schema = Schema::new(
            exprs.iter().map(|e| ColumnMeta::new(e.alias.clone())).collect(),
        );
        Self { input, exprs, output_schema }
    }

    /// Create a simple column-selection projection (no expressions).
    pub fn columns(input: Box<dyn PlanNode>, columns: Vec<String>) -> Self {
        let input_schema = input.schema().clone();
        let exprs: Vec<ProjectExpr> = columns
            .iter()
            .map(|c| {
                ProjectExpr {
                    expr: Expr::Column(crate::sql::ast::ColumnRef::new(c.clone())),
                    alias: c.clone(),
                }
            })
            .collect();
        let output_schema = Schema::new(
            columns.iter().map(|c| {
                // Preserve table qualification from input schema
                input_schema
                    .columns
                    .iter()
                    .find(|cm| cm.name == *c)
                    .cloned()
                    .unwrap_or_else(|| ColumnMeta::new(c.clone()))
            }).collect()
        );
        Self { input, exprs, output_schema }
    }
}

impl PlanNode for ProjectNode {
    fn schema(&self) -> &Schema {
        &self.output_schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        match self.input.next()? {
            Some(row) => {
                let input_schema = self.input.schema();
                let mut output = Vec::with_capacity(self.exprs.len());
                for pe in &self.exprs {
                    let val = eval_expr(&pe.expr, &row, input_schema)?;
                    output.push(val);
                }
                Ok(Some(output))
            }
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.input.reset()
    }
}

/// Wildcard expansion: pass-through all columns.
pub struct PassThroughNode {
    input: Box<dyn PlanNode>,
}

impl PassThroughNode {
    pub fn new(input: Box<dyn PlanNode>) -> Self {
        Self { input }
    }
}

impl PlanNode for PassThroughNode {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn next(&mut self) -> Result<Option<Row>> {
        self.input.next()
    }

    fn reset(&mut self) -> Result<()> {
        self.input.reset()
    }
}
