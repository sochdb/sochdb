// SPDX-License-Identifier: AGPL-3.0-or-later

//! Volcano iterator model trait.

use super::types::{Row, Schema};
use sochdb_core::Result;

/// Volcano-model plan node.
///
/// Each operator implements `next()` which returns one row at a time,
/// pulling from child operators on demand. This enables streaming
/// execution with predictable memory usage.
///
/// ```text
/// while let Some(row) = node.next()? {
///     // process row
/// }
/// ```
pub trait PlanNode {
    /// Schema of rows produced by this node.
    fn schema(&self) -> &Schema;

    /// Return the next row, or `None` when exhausted.
    fn next(&mut self) -> Result<Option<Row>>;

    /// Reset the operator for re-execution (e.g., inner side of nested loop join).
    fn reset(&mut self) -> Result<()> {
        Ok(()) // Default: no-op (many operators don't need reset)
    }

    /// Collect all remaining rows (convenience method).
    fn collect_all(&mut self) -> Result<Vec<Row>> {
        let mut rows = Vec::new();
        while let Some(row) = self.next()? {
            rows.push(row);
        }
        Ok(rows)
    }
}

/// Blanket impl so `Box<dyn PlanNode>` itself implements `PlanNode`.
impl PlanNode for Box<dyn PlanNode> {
    fn schema(&self) -> &Schema {
        (**self).schema()
    }

    fn next(&mut self) -> Result<Option<Row>> {
        (**self).next()
    }

    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
}
