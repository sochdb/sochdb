// SPDX-License-Identifier: AGPL-3.0-or-later

//! Table scan and index seek operators.

use crate::optimizer_integration::StorageBackend;
use crate::soch_ql::SochValue;
use super::types::{Row, Schema, ColumnMeta};
use super::node::PlanNode;
use sochdb_core::Result;
use std::sync::Arc;

// ============================================================================
// SeqScanNode — Full table scan
// ============================================================================

/// Full sequential table scan operator.
///
/// Materializes all rows from `StorageBackend::table_scan()` on first call,
/// then returns one row at a time via `next()`.
pub struct SeqScanNode {
    schema: Schema,
    storage: Arc<dyn StorageBackend>,
    table: String,
    columns: Vec<String>,
    /// Materialized row buffer (lazily populated).
    buffer: Option<Vec<Row>>,
    /// Current position in buffer.
    pos: usize,
}

impl SeqScanNode {
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        table: String,
        columns: Vec<String>,
        table_alias: Option<&str>,
    ) -> Self {
        let tbl = table_alias.unwrap_or(&table);
        let schema = Schema::new(
            columns
                .iter()
                .map(|c| {
                    if c == "*" {
                        ColumnMeta::new(c.clone())
                    } else {
                        ColumnMeta::qualified(tbl.to_string(), c.clone())
                    }
                })
                .collect(),
        );
        Self {
            schema,
            storage,
            table,
            columns,
            buffer: None,
            pos: 0,
        }
    }

    fn materialize(&mut self) -> Result<()> {
        if self.buffer.is_some() {
            return Ok(());
        }

        let raw_rows = self.storage.table_scan(&self.table, &self.columns, None)?;

        // If columns is ["*"], discover actual columns from first row
        if self.columns.len() == 1 && self.columns[0] == "*" {
            if let Some(first) = raw_rows.first() {
                let mut col_names: Vec<String> = first.keys().cloned().collect();
                col_names.sort(); // Deterministic column order
                let tbl = self.schema.columns.first()
                    .and_then(|c| c.table.clone())
                    .unwrap_or_else(|| self.table.clone());
                self.schema = Schema::new(
                    col_names.iter().map(|c| ColumnMeta::qualified(tbl.clone(), c.clone())).collect(),
                );
                self.columns = col_names;
            }
        }

        let rows = raw_rows
            .into_iter()
            .map(|row_map| self.row_from_map(row_map))
            .collect();

        self.buffer = Some(rows);
        Ok(())
    }

    fn row_from_map(&self, map: std::collections::HashMap<String, SochValue>) -> Row {
        self.columns
            .iter()
            .map(|col| map.get(col).cloned().unwrap_or(SochValue::Null))
            .collect()
    }
}

impl PlanNode for SeqScanNode {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        self.materialize()?;

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
        Ok(())
    }
}

// ============================================================================
// IndexSeekNode — Index-based lookup
// ============================================================================

/// Index-based seek operator.
///
/// Uses `StorageBackend::secondary_index_seek()` to retrieve rows matching
/// a specific key, then iterates over results.
pub struct IndexSeekNode {
    schema: Schema,
    storage: Arc<dyn StorageBackend>,
    table: String,
    index: String,
    key: SochValue,
    columns: Vec<String>,
    buffer: Option<Vec<Row>>,
    pos: usize,
}

impl IndexSeekNode {
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        table: String,
        index: String,
        key: SochValue,
        columns: Vec<String>,
    ) -> Self {
        let schema = Schema::new(
            columns.iter().map(|c| ColumnMeta::qualified(table.clone(), c.clone())).collect(),
        );
        Self {
            schema,
            storage,
            table,
            index,
            key,
            columns,
            buffer: None,
            pos: 0,
        }
    }

    fn materialize(&mut self) -> Result<()> {
        if self.buffer.is_some() {
            return Ok(());
        }

        let raw_rows = self.storage.secondary_index_seek(&self.table, &self.index, &self.key)?;
        let rows = raw_rows
            .into_iter()
            .map(|row_map| {
                self.columns
                    .iter()
                    .map(|col| row_map.get(col).cloned().unwrap_or(SochValue::Null))
                    .collect()
            })
            .collect();

        self.buffer = Some(rows);
        Ok(())
    }
}

impl PlanNode for IndexSeekNode {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        self.materialize()?;

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
        Ok(())
    }
}

// ============================================================================
// ValuesNode — Inline VALUES rows
// ============================================================================

/// Returns pre-computed rows (for INSERT ... VALUES or subquery materialization).
pub struct ValuesNode {
    schema: Schema,
    rows: Vec<Row>,
    pos: usize,
}

impl ValuesNode {
    pub fn new(schema: Schema, rows: Vec<Row>) -> Self {
        Self { schema, rows, pos: 0 }
    }
}

impl PlanNode for ValuesNode {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        if self.pos < self.rows.len() {
            let row = self.rows[self.pos].clone();
            self.pos += 1;
            Ok(Some(row))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        Ok(())
    }
}

/// Empty node that produces no rows.
pub struct EmptyNode {
    schema: Schema,
}

impl EmptyNode {
    pub fn new(schema: Schema) -> Self {
        Self { schema }
    }
}

impl PlanNode for EmptyNode {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn next(&mut self) -> Result<Option<Row>> {
        Ok(None)
    }
}
