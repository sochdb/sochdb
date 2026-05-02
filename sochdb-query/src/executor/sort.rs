// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sort operator.

use super::eval::{eval_expr, compare_values};
use super::node::PlanNode;
use super::types::{Row, Schema};
use crate::sql::ast::Expr;
use sochdb_core::Result;

/// Sort key: expression + direction.
pub struct SortKey {
    pub expr: Expr,
    /// true = ascending, false = descending
    pub ascending: bool,
    /// true = nulls sort first
    pub nulls_first: bool,
}

/// Sort operator: materializes all input rows, sorts them, then iterates.
///
/// This is a *blocking* operator — it must consume all input before
/// producing any output.
///
/// ```text
/// Sort(order_by=[age DESC, name ASC])
///   └── input
/// ```
pub struct SortNode {
    input: Box<dyn PlanNode>,
    sort_keys: Vec<SortKey>,
    /// Materialized and sorted rows.
    buffer: Option<Vec<Row>>,
    pos: usize,
}

impl SortNode {
    pub fn new(input: Box<dyn PlanNode>, sort_keys: Vec<SortKey>) -> Self {
        Self {
            input,
            sort_keys,
            buffer: None,
            pos: 0,
        }
    }

    fn materialize_and_sort(&mut self) -> Result<()> {
        if self.buffer.is_some() {
            return Ok(());
        }

        // Consume all input
        let mut rows = self.input.collect_all()?;
        let schema = self.input.schema().clone();

        // Pre-evaluate sort keys for each row (avoids re-evaluation during sort)
        let mut keyed: Vec<(Vec<crate::soch_ql::SochValue>, Row)> = rows
            .drain(..)
            .map(|row| {
                let keys: Vec<crate::soch_ql::SochValue> = self
                    .sort_keys
                    .iter()
                    .map(|sk| eval_expr(&sk.expr, &row, &schema).unwrap_or(crate::soch_ql::SochValue::Null))
                    .collect();
                (keys, row)
            })
            .collect();

        // Sort using pre-evaluated keys
        let sort_keys = &self.sort_keys;
        keyed.sort_by(|(ka, _), (kb, _)| {
            for (i, sk) in sort_keys.iter().enumerate() {
                let a = &ka[i];
                let b = &kb[i];

                // Handle NULLs
                let a_null = matches!(a, crate::soch_ql::SochValue::Null);
                let b_null = matches!(b, crate::soch_ql::SochValue::Null);

                if a_null && b_null {
                    continue;
                }
                if a_null {
                    return if sk.nulls_first {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    };
                }
                if b_null {
                    return if sk.nulls_first {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    };
                }

                if let Some(ord) = compare_values(a, b) {
                    let ord = if sk.ascending { ord } else { ord.reverse() };
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
            }
            std::cmp::Ordering::Equal
        });

        self.buffer = Some(keyed.into_iter().map(|(_, row)| row).collect());
        Ok(())
    }
}

impl PlanNode for SortNode {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn next(&mut self) -> Result<Option<Row>> {
        self.materialize_and_sort()?;

        if let Some(buf) = &self.buffer {
            if self.pos < buf.len() {
                let row = buf[self.pos].clone();
                self.pos += 1;
                Ok(Some(row))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        // Don't re-materialize; sorted results are still valid
        Ok(())
    }
}
