// SPDX-License-Identifier: AGPL-3.0-or-later

//! Filter operator (WHERE clause).

use crate::sql::ast::Expr;
use super::eval::eval_predicate;
use super::node::PlanNode;
use super::types::{Row, Schema};
use sochdb_core::Result;

/// Filter operator: passes through only rows satisfying the predicate.
///
/// ```text
/// Filter(predicate)
///   └── input
/// ```
pub struct FilterNode {
    input: Box<dyn PlanNode>,
    predicate: Expr,
}

impl FilterNode {
    pub fn new(input: Box<dyn PlanNode>, predicate: Expr) -> Self {
        Self { input, predicate }
    }
}

impl PlanNode for FilterNode {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn next(&mut self) -> Result<Option<Row>> {
        loop {
            match self.input.next()? {
                Some(row) => {
                    if eval_predicate(&self.predicate, &row, self.input.schema())? {
                        return Ok(Some(row));
                    }
                    // Row didn't match predicate, continue to next
                }
                None => return Ok(None),
            }
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.input.reset()
    }
}
