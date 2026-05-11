// SPDX-License-Identifier: AGPL-3.0-or-later

//! Limit + Offset operator.

use super::node::PlanNode;
use super::types::{Row, Schema};
use sochdb_core::Result;

/// Limit operator: skips `offset` rows, then returns at most `limit` rows.
///
/// ```text
/// Limit(limit=10, offset=5)
///   └── input
/// ```
pub struct LimitNode {
    input: Box<dyn PlanNode>,
    limit: Option<usize>,
    offset: usize,
    /// Rows emitted so far (after offset).
    emitted: usize,
    /// Rows skipped so far (for offset).
    skipped: usize,
}

impl LimitNode {
    pub fn new(input: Box<dyn PlanNode>, limit: Option<usize>, offset: usize) -> Self {
        Self {
            input,
            limit,
            offset,
            emitted: 0,
            skipped: 0,
        }
    }
}

impl PlanNode for LimitNode {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn next(&mut self) -> Result<Option<Row>> {
        // Skip offset rows
        while self.skipped < self.offset {
            match self.input.next()? {
                Some(_) => self.skipped += 1,
                None => return Ok(None),
            }
        }

        // Check limit
        if let Some(limit) = self.limit {
            if self.emitted >= limit {
                return Ok(None);
            }
        }

        match self.input.next()? {
            Some(row) => {
                self.emitted += 1;
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.emitted = 0;
        self.skipped = 0;
        self.input.reset()
    }
}
