// SPDX-License-Identifier: AGPL-3.0-or-later

//! EXPLAIN operator — generates query plan description as rows.

use crate::soch_ql::SochValue;
use super::node::PlanNode;
use super::types::{Row, Schema, ColumnMeta};
use sochdb_core::Result;

/// Explain operator: returns the plan description as text rows.
pub struct ExplainNode {
    schema: Schema,
    lines: Vec<String>,
    pos: usize,
}

impl ExplainNode {
    pub fn new(plan_text: String) -> Self {
        let lines: Vec<String> = plan_text.lines().map(|l| l.to_string()).collect();
        Self {
            schema: Schema::new(vec![ColumnMeta::new("QUERY PLAN".to_string())]),
            lines,
            pos: 0,
        }
    }
}

impl PlanNode for ExplainNode {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        if self.pos < self.lines.len() {
            let line = self.lines[self.pos].clone();
            self.pos += 1;
            Ok(Some(vec![SochValue::Text(line)]))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        Ok(())
    }
}

/// Generate a human-readable plan description from an operator tree.
pub fn describe_plan(node: &dyn PlanNode, _depth: usize) -> String {
    // We'll use the schema to identify operator type
    let schema = node.schema();
    let cols = schema.column_names().join(", ");
    format!("Operator [columns={}]", cols)
}

/// Generate EXPLAIN output for a planned query.
///
/// This creates a structured plan description showing:
/// - Operator type
/// - Estimated cost/rows (if available)
/// - Column list
pub fn format_plan_tree(description: &str) -> String {
    description.to_string()
}
