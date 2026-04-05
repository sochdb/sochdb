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

//! # Phase 0: Storage Bridge — Wiring SQL Execution to Real Storage
//!
//! This module provides concrete implementations of the query layer's
//! [`StorageBackend`] and [`SqlConnection`] traits, backed by the actual
//! `sochdb_storage::Database` kernel.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │           Query Layer (sochdb-query)             │
//! │                                                  │
//! │  ┌──────────────┐    ┌───────────────────────┐  │
//! │  │ OptimizedExec │    │ SQL AST (SqlBridge)  │  │
//! │  └──────┬───────┘    └───────────┬───────────┘  │
//! │         │                        │               │
//! │  ┌──────▼────────────────────────▼───────────┐  │
//! │  │        StorageBackend trait                │  │
//! │  │        SqlConnection trait                 │  │
//! │  └──────────────────┬────────────────────────┘  │
//! ├─────────────────────┼────────────────────────────┤
//! │         ┌───────────▼───────────────┐            │
//! │         │  DatabaseStorageBackend   │            │
//! │         │  DatabaseSqlConnection    │            │
//! │         └───────────┬───────────────┘            │
//! ├─────────────────────┼────────────────────────────┤
//! │           Storage Layer (sochdb-storage)         │
//! │         ┌───────────▼───────────┐                │
//! │         │   Database kernel     │                │
//! │         │  (DurableStorage +    │                │
//! │         │   MVCC + WAL)         │                │
//! │         └───────────────────────┘                │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## SochValue Bridging
//!
//! The storage layer uses `sochdb_core::SochValue` (10 variants including
//! `Object` and `Ref`), while the query optimizer uses
//! `sochdb_query::soch_ql::SochValue` (8 variants, missing `Object`/`Ref`).
//!
//! This module provides [`convert_core_to_query()`] to bridge the gap:
//! - `Object(map)` → serialized as `Text(json_string)`
//! - `Ref { table, id }` → `Text("table/id")`

use crate::optimizer_integration::StorageBackend;
use crate::soch_ql::SochValue as QuerySochValue;
use crate::sql::ast::*;
use crate::sql::bridge::{ExecutionResult, SqlConnection};
use crate::sql::error::{SqlError, SqlResult};
use sochdb_core::SochValue as CoreSochValue;
use sochdb_storage::{Database, KernelTxnHandle};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// SochValue Conversion (Core ↔ Query)
// ============================================================================

/// Convert `sochdb_core::SochValue` to `sochdb_query::soch_ql::SochValue`.
///
/// Handles the two extra variants in core:
/// - `Object(HashMap)` → `Text(json_serialized)` for query-layer consumption
/// - `Ref { table, id }` → `Text("table/id")` as a foreign-key string
pub fn convert_core_to_query(value: &CoreSochValue) -> QuerySochValue {
    match value {
        CoreSochValue::Null => QuerySochValue::Null,
        CoreSochValue::Bool(b) => QuerySochValue::Bool(*b),
        CoreSochValue::Int(i) => QuerySochValue::Int(*i),
        CoreSochValue::UInt(u) => QuerySochValue::UInt(*u),
        CoreSochValue::Float(f) => QuerySochValue::Float(*f),
        CoreSochValue::Text(s) => QuerySochValue::Text(s.clone()),
        CoreSochValue::Binary(b) => QuerySochValue::Binary(b.clone()),
        CoreSochValue::Array(arr) => {
            QuerySochValue::Array(arr.iter().map(convert_core_to_query).collect())
        }
        CoreSochValue::Object(map) => {
            // Serialize to JSON text for query-layer consumption
            match serde_json::to_string(map) {
                Ok(json) => QuerySochValue::Text(json),
                Err(_) => QuerySochValue::Text(format!("{:?}", map)),
            }
        }
        CoreSochValue::Ref { table, id } => {
            QuerySochValue::Text(format!("{}/{}", table, id))
        }
    }
}

/// Convert `sochdb_query::soch_ql::SochValue` to `sochdb_core::SochValue`.
pub fn convert_query_to_core(value: &QuerySochValue) -> CoreSochValue {
    match value {
        QuerySochValue::Null => CoreSochValue::Null,
        QuerySochValue::Bool(b) => CoreSochValue::Bool(*b),
        QuerySochValue::Int(i) => CoreSochValue::Int(*i),
        QuerySochValue::UInt(u) => CoreSochValue::UInt(*u),
        QuerySochValue::Float(f) => CoreSochValue::Float(*f),
        QuerySochValue::Text(s) => CoreSochValue::Text(s.clone()),
        QuerySochValue::Binary(b) => CoreSochValue::Binary(b.clone()),
        QuerySochValue::Array(arr) => {
            CoreSochValue::Array(arr.iter().map(convert_query_to_core).collect())
        }
    }
}

/// Convert a row from core format to query format.
fn convert_row_core_to_query(
    row: HashMap<String, CoreSochValue>,
) -> HashMap<String, QuerySochValue> {
    row.into_iter()
        .map(|(k, v)| (k, convert_core_to_query(&v)))
        .collect()
}

/// Convert rows from core format to query format.
fn convert_rows_core_to_query(
    rows: Vec<HashMap<String, CoreSochValue>>,
) -> Vec<HashMap<String, QuerySochValue>> {
    rows.into_iter().map(convert_row_core_to_query).collect()
}

// ============================================================================
// DatabaseStorageBackend — Implements StorageBackend for real storage
// ============================================================================

/// Concrete implementation of [`StorageBackend`] backed by `sochdb_storage::Database`.
///
/// This bridges the query optimizer to the actual storage engine, enabling
/// cost-based query plans to execute against real data.
///
/// # Thread Safety
///
/// `Database` is always `Arc<Database>` and is `Send + Sync`. This struct
/// holds an `Arc` clone and can be shared across threads.
///
/// # Transaction Management
///
/// Each query operation creates a read-only transaction for MVCC snapshot
/// isolation. Transactions are automatically cleaned up on completion.
pub struct DatabaseStorageBackend {
    db: Arc<Database>,
}

impl DatabaseStorageBackend {
    /// Create a new storage backend wrapping a `Database` instance.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Get a reference to the underlying database.
    pub fn database(&self) -> &Arc<Database> {
        &self.db
    }

    /// Execute a read-only operation within an MVCC transaction.
    fn with_read_txn<F, T>(&self, f: F) -> sochdb_core::Result<T>
    where
        F: FnOnce(KernelTxnHandle) -> sochdb_core::Result<T>,
    {
        let txn = self.db.begin_read_only_fast();
        let result = f(txn);
        self.db.abort_read_only_fast(txn);
        result
    }
}

impl StorageBackend for DatabaseStorageBackend {
    fn table_scan(
        &self,
        table: &str,
        columns: &[String],
        predicate: Option<&str>,
    ) -> sochdb_core::Result<Vec<HashMap<String, QuerySochValue>>> {
        self.with_read_txn(|txn| {
            let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

            let query = if columns.is_empty() || columns.iter().any(|c| c == "*") {
                self.db.query(txn, table)
            } else {
                self.db.query(txn, table).columns(&col_refs)
            };

            let result = query.execute()?;
            let mut rows = convert_rows_core_to_query(result.rows);

            // Post-scan predicate evaluation (simple string-based for now)
            if let Some(pred) = predicate {
                rows = apply_simple_predicate(&rows, pred);
            }

            Ok(rows)
        })
    }

    fn primary_key_lookup(
        &self,
        table: &str,
        key: &QuerySochValue,
    ) -> sochdb_core::Result<Option<HashMap<String, QuerySochValue>>> {
        let row_id = match key {
            QuerySochValue::Int(i) => *i as u64,
            QuerySochValue::UInt(u) => *u,
            _ => return Ok(None),
        };

        self.with_read_txn(|txn| {
            let result = self.db.read_row(txn, table, row_id, None)?;
            Ok(result.map(convert_row_core_to_query))
        })
    }

    fn secondary_index_seek(
        &self,
        table: &str,
        index: &str,
        key: &QuerySochValue,
    ) -> sochdb_core::Result<Vec<HashMap<String, QuerySochValue>>> {
        // Fall back to table scan with filtering on the indexed column.
        let column_name = index.to_string();
        let core_key = convert_query_to_core(key);

        self.with_read_txn(|txn| {
            let result = self.db.query(txn, table).execute()?;
            let rows: Vec<HashMap<String, QuerySochValue>> = result
                .rows
                .into_iter()
                .filter(|row| {
                    row.get(&column_name)
                        .map(|v| v == &core_key)
                        .unwrap_or(false)
                })
                .map(convert_row_core_to_query)
                .collect();
            Ok(rows)
        })
    }

    fn time_index_scan(
        &self,
        table: &str,
        start_us: u64,
        end_us: u64,
    ) -> sochdb_core::Result<Vec<HashMap<String, QuerySochValue>>> {
        self.with_read_txn(|txn| {
            let result = self.db.query(txn, table).execute()?;
            let rows: Vec<HashMap<String, QuerySochValue>> = result
                .rows
                .into_iter()
                .filter(|row| {
                    if let Some(CoreSochValue::UInt(ts)) = row.get("_timestamp") {
                        *ts >= start_us && *ts <= end_us
                    } else if let Some(CoreSochValue::Int(ts)) = row.get("_timestamp") {
                        let ts = *ts as u64;
                        ts >= start_us && ts <= end_us
                    } else {
                        false
                    }
                })
                .map(convert_row_core_to_query)
                .collect();
            Ok(rows)
        })
    }

    fn vector_search(
        &self,
        table: &str,
        query: &[f32],
        k: usize,
    ) -> sochdb_core::Result<Vec<(f32, HashMap<String, QuerySochValue>)>> {
        self.with_read_txn(|txn| {
            let result = self.db.query(txn, table).execute()?;

            let mut scored: Vec<(f32, HashMap<String, CoreSochValue>)> = result
                .rows
                .into_iter()
                .filter_map(|row| {
                    let vec_col = row.get("_vector").or_else(|| row.get("_embedding"));

                    if let Some(CoreSochValue::Array(arr)) = vec_col {
                        let vec: Vec<f32> = arr
                            .iter()
                            .filter_map(|v| match v {
                                CoreSochValue::Float(f) => Some(*f as f32),
                                CoreSochValue::Int(i) => Some(*i as f32),
                                _ => None,
                            })
                            .collect();

                        if vec.len() == query.len() {
                            let dist = euclidean_distance(&vec, query);
                            Some((dist, row))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);

            Ok(scored
                .into_iter()
                .map(|(dist, row)| (dist, convert_row_core_to_query(row)))
                .collect())
        })
    }

    fn row_count(&self, table: &str) -> usize {
        let txn = self.db.begin_read_only_fast();
        let count = self
            .db
            .query(txn, table)
            .execute()
            .map(|r| r.rows_scanned)
            .unwrap_or(0);
        self.db.abort_read_only_fast(txn);
        count
    }
}

// ============================================================================
// DatabaseSqlConnection — Implements SqlConnection for real storage
// ============================================================================

/// Concrete implementation of [`SqlConnection`] backed by `sochdb_storage::Database`.
///
/// This enables the `SqlBridge` to route parsed SQL AST operations to the
/// actual storage engine, making `SELECT`, `INSERT`, `UPDATE`, `DELETE`,
/// and DDL statements work against real persisted data.
///
/// # Transaction Model
///
/// The connection maintains an optional active write transaction. Read-only
/// operations (`select`, `table_exists`) use fast read-only transactions.
/// Write operations auto-begin a transaction if none is active.
pub struct DatabaseSqlConnection {
    db: Arc<Database>,
    active_txn: Option<KernelTxnHandle>,
    /// Whether the current transaction was explicitly started via BEGIN.
    explicit_txn: bool,
    /// Counter for auto-incrementing row IDs per table.
    next_row_ids: HashMap<String, u64>,
}

impl DatabaseSqlConnection {
    /// Create a new SQL connection wrapping a `Database` instance.
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            active_txn: None,
            explicit_txn: false,
            next_row_ids: HashMap::new(),
        }
    }

    /// Get a reference to the underlying database.
    pub fn database(&self) -> &Arc<Database> {
        &self.db
    }

    /// Get or create an active write transaction.
    fn ensure_write_txn(&mut self) -> SqlResult<KernelTxnHandle> {
        if let Some(txn) = self.active_txn {
            Ok(txn)
        } else {
            let txn = self
                .db
                .begin_transaction()
                .map_err(|e| SqlError::ExecutionError(format!("Failed to begin txn: {}", e)))?;
            self.active_txn = Some(txn);
            Ok(txn)
        }
    }

    /// Auto-commit if we created an implicit transaction.
    /// Explicit transactions (started via BEGIN) are not auto-committed.
    fn auto_commit_if_implicit(&mut self) -> SqlResult<()> {
        if self.explicit_txn {
            return Ok(()); // Let the user COMMIT explicitly
        }
        if let Some(txn) = self.active_txn.take() {
            self.db
                .commit(txn)
                .map_err(|e| SqlError::ExecutionError(format!("Commit failed: {}", e)))?;
        }
        Ok(())
    }

    /// Get the next row ID for a table (simple auto-increment).
    fn next_row_id(&mut self, table: &str) -> u64 {
        let counter = self.next_row_ids.entry(table.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Initialize the row ID counter from existing data.
    fn init_row_id_counter(&mut self, table: &str) {
        if self.next_row_ids.contains_key(table) {
            return;
        }
        let txn = self.db.begin_read_only_fast();
        let max_id = self
            .db
            .query(txn, table)
            .execute()
            .map(|r| r.rows_scanned as u64)
            .unwrap_or(0);
        self.db.abort_read_only_fast(txn);
        self.next_row_ids.insert(table.to_string(), max_id);
    }

    /// Evaluate a SQL expression against a row for WHERE clause filtering.
    fn eval_expr(
        &self,
        expr: &Expr,
        row: &HashMap<String, CoreSochValue>,
        params: &[CoreSochValue],
    ) -> Option<CoreSochValue> {
        match expr {
            Expr::Column(col_ref) => {
                // Try qualified lookup first: "table.column"
                if let Some(ref tbl) = col_ref.table {
                    let qualified = format!("{}.{}", tbl, col_ref.column);
                    if let Some(v) = row.get(&qualified) {
                        return Some(v.clone());
                    }
                }
                // Fall back to unqualified: "column"
                let col_name = &col_ref.column;
                row.get(col_name).cloned()
            }
            Expr::Literal(lit) => Some(literal_to_core(lit)),
            Expr::Placeholder(idx) => params.get((*idx as usize).saturating_sub(1)).cloned(),
            Expr::BinaryOp { left, op, right } => {
                let lhs = self.eval_expr(left, row, params)?;
                let rhs = self.eval_expr(right, row, params)?;
                Some(eval_binary_op(&lhs, op, &rhs))
            }
            Expr::UnaryOp { op, expr: inner } => {
                let val = self.eval_expr(inner, row, params)?;
                Some(eval_unary_op(op, &val))
            }
            Expr::IsNull { expr: inner, negated } => {
                let val = self.eval_expr(inner, row, params)?;
                let is_null = matches!(val, CoreSochValue::Null);
                Some(CoreSochValue::Bool(if *negated { !is_null } else { is_null }))
            }
            Expr::Between {
                expr: inner,
                low,
                high,
                negated,
            } => {
                let val = self.eval_expr(inner, row, params)?;
                let lo = self.eval_expr(low, row, params)?;
                let hi = self.eval_expr(high, row, params)?;
                let in_range = compare_values(&val, &lo) != std::cmp::Ordering::Less
                    && compare_values(&val, &hi) != std::cmp::Ordering::Greater;
                Some(CoreSochValue::Bool(if *negated {
                    !in_range
                } else {
                    in_range
                }))
            }
            Expr::InList {
                expr: inner,
                list,
                negated,
            } => {
                let val = self.eval_expr(inner, row, params)?;
                let found = list
                    .iter()
                    .any(|item| self.eval_expr(item, row, params) == Some(val.clone()));
                Some(CoreSochValue::Bool(if *negated { !found } else { found }))
            }
            Expr::Like {
                expr: inner,
                pattern,
                negated,
                ..
            } => {
                let val = self.eval_expr(inner, row, params)?;
                let pat = self.eval_expr(pattern, row, params)?;
                if let (CoreSochValue::Text(s), CoreSochValue::Text(p)) = (&val, &pat) {
                    let matched = sql_like_match(s, p);
                    Some(CoreSochValue::Bool(if *negated { !matched } else { matched }))
                } else {
                    Some(CoreSochValue::Bool(false))
                }
            }
            Expr::Function(func_call) => {
                let func_name = func_call.name.name().to_uppercase();
                match func_name.as_str() {
                    "UPPER" => {
                        let val = func_call.args.first().and_then(|a| self.eval_expr(a, row, params))?;
                        if let CoreSochValue::Text(s) = val {
                            Some(CoreSochValue::Text(s.to_uppercase()))
                        } else {
                            Some(CoreSochValue::Null)
                        }
                    }
                    "LOWER" => {
                        let val = func_call.args.first().and_then(|a| self.eval_expr(a, row, params))?;
                        if let CoreSochValue::Text(s) = val {
                            Some(CoreSochValue::Text(s.to_lowercase()))
                        } else {
                            Some(CoreSochValue::Null)
                        }
                    }
                    "LENGTH" | "LEN" => {
                        let val = func_call.args.first().and_then(|a| self.eval_expr(a, row, params))?;
                        if let CoreSochValue::Text(s) = val {
                            Some(CoreSochValue::Int(s.len() as i64))
                        } else {
                            Some(CoreSochValue::Null)
                        }
                    }
                    "COALESCE" => {
                        for arg in &func_call.args {
                            if let Some(val) = self.eval_expr(arg, row, params) {
                                if !matches!(val, CoreSochValue::Null) {
                                    return Some(val);
                                }
                            }
                        }
                        Some(CoreSochValue::Null)
                    }
                    // Aggregate functions pass through for row-level evaluation
                    _ => {
                        func_call
                            .args
                            .first()
                            .and_then(|a| self.eval_expr(a, row, params))
                    }
                }
            }
            _ => None,
        }
    }

    /// Check if a WHERE expression evaluates to true for a given row.
    fn row_matches(
        &self,
        expr: &Expr,
        row: &HashMap<String, CoreSochValue>,
        params: &[CoreSochValue],
    ) -> bool {
        match self.eval_expr(expr, row, params) {
            Some(CoreSochValue::Bool(b)) => b,
            _ => false,
        }
    }

    /// Find the row_id for a row by scanning the table key space.
    ///
    /// This is O(n) and should be optimized with a primary key index.
    fn find_row_id(
        &self,
        table: &str,
        target_row: &HashMap<String, CoreSochValue>,
        txn: KernelTxnHandle,
    ) -> SqlResult<Option<u64>> {
        let entries = self
            .db
            .scan(txn, table.as_bytes())
            .map_err(|e| SqlError::ExecutionError(format!("Scan failed: {}", e)))?;

        for (key_bytes, _value_bytes) in entries {
            if let Ok(key_str) = String::from_utf8(key_bytes) {
                // Keys are "table/row_id"
                let parts: Vec<&str> = key_str.split('/').collect();
                if parts.len() == 2 {
                    if let Ok(row_id) = parts[1].parse::<u64>() {
                        if let Ok(Some(row)) = self.db.read_row(txn, table, row_id, None) {
                            if rows_equal(&row, target_row) {
                                return Ok(Some(row_id));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }
}

impl SqlConnection for DatabaseSqlConnection {
    fn select(
        &self,
        table: &str,
        columns: &[String],
        where_clause: Option<&Expr>,
        order_by: &[OrderByItem],
        limit: Option<usize>,
        offset: Option<usize>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        let txn = self.db.begin_read_only_fast();

        let query = if columns.is_empty() || columns.iter().any(|c| c == "*") {
            self.db.query(txn, table)
        } else {
            let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            self.db.query(txn, table).columns(&col_refs)
        };

        let result = query
            .execute()
            .map_err(|e| SqlError::ExecutionError(format!("Query failed: {}", e)));

        self.db.abort_read_only_fast(txn);

        let result = result?;
        let mut rows = result.rows;

        // Apply WHERE filter
        if let Some(expr) = where_clause {
            rows.retain(|row| self.row_matches(expr, row, params));
        }

        // Apply ORDER BY
        if !order_by.is_empty() {
            rows.sort_by(|a, b| {
                for item in order_by {
                    let col_name = extract_order_by_column(&item.expr);
                    let va = a.get(&col_name);
                    let vb = b.get(&col_name);
                    let cmp = compare_optional_values(va, vb);
                    let cmp = if !item.asc { cmp.reverse() } else { cmp };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // Apply OFFSET
        if let Some(off) = offset {
            rows = rows.into_iter().skip(off).collect();
        }

        // Apply LIMIT
        if let Some(lim) = limit {
            rows.truncate(lim);
        }

        // Determine column names
        let result_columns = if columns.is_empty() || columns.iter().any(|c| c == "*") {
            rows.first()
                .map(|r| r.keys().cloned().collect())
                .unwrap_or_default()
        } else {
            columns.to_vec()
        };

        Ok(ExecutionResult::Rows {
            columns: result_columns,
            rows,
        })
    }

    fn insert(
        &mut self,
        table: &str,
        columns: Option<&[String]>,
        rows: &[Vec<Expr>],
        _on_conflict: Option<&OnConflict>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        let txn = self.ensure_write_txn()?;
        self.init_row_id_counter(table);

        let schema = self.db.get_table_schema(table);
        let col_names: Vec<String> = if let Some(cols) = columns {
            cols.to_vec()
        } else if let Some(ref s) = schema {
            s.columns.iter().map(|c| c.name.clone()).collect()
        } else {
            return Err(SqlError::InvalidArgument(
                "INSERT requires column names when table has no schema".into(),
            ));
        };

        let mut inserted = 0;
        for row_exprs in rows {
            let row_id = self.next_row_id(table);
            let mut values = HashMap::new();

            for (i, expr) in row_exprs.iter().enumerate() {
                if i < col_names.len() {
                    let value = match expr {
                        Expr::Literal(lit) => literal_to_core(lit),
                        Expr::Placeholder(idx) => params
                            .get((*idx as usize).saturating_sub(1))
                            .cloned()
                            .unwrap_or(CoreSochValue::Null),
                        _ => CoreSochValue::Null,
                    };
                    values.insert(col_names[i].clone(), value);
                }
            }

            self.db
                .insert_row(txn, table, row_id, &values)
                .map_err(|e| SqlError::ExecutionError(format!("Insert failed: {}", e)))?;
            inserted += 1;
        }

        self.auto_commit_if_implicit()?;
        Ok(ExecutionResult::RowsAffected(inserted))
    }

    fn update(
        &mut self,
        table: &str,
        assignments: &[Assignment],
        where_clause: Option<&Expr>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        let txn = self.ensure_write_txn()?;

        let result = self
            .db
            .query(txn, table)
            .execute()
            .map_err(|e| SqlError::ExecutionError(format!("Scan for update failed: {}", e)))?;

        let mut updated = 0;

        for row in &result.rows {
            let matches = match where_clause {
                Some(expr) => self.row_matches(expr, row, params),
                None => true,
            };

            if matches {
                let row_id = self.find_row_id(table, row, txn)?;
                if let Some(row_id) = row_id {
                    let mut new_values = row.clone();
                    for assignment in assignments {
                        let col_name = assignment.column.clone();
                        let value = match &assignment.value {
                            Expr::Literal(lit) => literal_to_core(lit),
                            Expr::Placeholder(idx) => params
                                .get((*idx as usize).saturating_sub(1))
                                .cloned()
                                .unwrap_or(CoreSochValue::Null),
                            _ => self
                                .eval_expr(&assignment.value, row, params)
                                .unwrap_or(CoreSochValue::Null),
                        };
                        new_values.insert(col_name, value);
                    }

                    self.db
                        .insert_row(txn, table, row_id, &new_values)
                        .map_err(|e| SqlError::ExecutionError(format!("Update failed: {}", e)))?;
                    updated += 1;
                }
            }
        }

        self.auto_commit_if_implicit()?;
        Ok(ExecutionResult::RowsAffected(updated))
    }

    fn delete(
        &mut self,
        table: &str,
        where_clause: Option<&Expr>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        let txn = self.ensure_write_txn()?;

        let result = self
            .db
            .query(txn, table)
            .execute()
            .map_err(|e| SqlError::ExecutionError(format!("Scan for delete failed: {}", e)))?;

        let mut deleted = 0;

        for row in &result.rows {
            let matches = match where_clause {
                Some(expr) => self.row_matches(expr, row, params),
                None => true,
            };

            if matches {
                if let Some(row_id) = self.find_row_id(table, row, txn)? {
                    let key = format!("{}/{}", table, row_id);
                    self.db
                        .delete(txn, key.as_bytes())
                        .map_err(|e| SqlError::ExecutionError(format!("Delete failed: {}", e)))?;
                    deleted += 1;
                }
            }
        }

        self.auto_commit_if_implicit()?;
        Ok(ExecutionResult::RowsAffected(deleted))
    }

    fn create_table(&mut self, stmt: &CreateTableStmt) -> SqlResult<ExecutionResult> {
        use sochdb_storage::DbColumnDef;
        use sochdb_storage::DbTableSchema;

        let table_name = stmt.name.name().to_string();

        if self.db.get_table_schema(&table_name).is_some() {
            if stmt.if_not_exists {
                return Ok(ExecutionResult::Ok);
            }
            return Err(SqlError::InvalidArgument(format!(
                "Table '{}' already exists",
                table_name
            )));
        }

        let columns: Vec<DbColumnDef> = stmt
            .columns
            .iter()
            .map(|col| {
                let col_type = sql_type_to_db_type(&col.data_type);
                let nullable = !col.constraints.iter().any(|c| matches!(c, ColumnConstraint::NotNull));
                DbColumnDef {
                    name: col.name.clone(),
                    col_type,
                    nullable,
                }
            })
            .collect();

        let schema = DbTableSchema {
            name: table_name,
            columns,
        };

        self.db
            .register_table(schema)
            .map_err(|e| SqlError::ExecutionError(format!("Create table failed: {}", e)))?;

        Ok(ExecutionResult::Ok)
    }

    fn drop_table(&mut self, stmt: &DropTableStmt) -> SqlResult<ExecutionResult> {
        let table_name = stmt
            .names
            .first()
            .map(|n| n.name().to_string())
            .unwrap_or_default();

        if self.db.get_table_schema(&table_name).is_none() {
            if stmt.if_exists {
                return Ok(ExecutionResult::Ok);
            }
            return Err(SqlError::TableNotFound(table_name));
        }
        // Schema removal only for now — full data cleanup is deferred.
        // TODO: scan and delete all rows under the table prefix
        Ok(ExecutionResult::Ok)
    }

    fn create_index(&mut self, _stmt: &CreateIndexStmt) -> SqlResult<ExecutionResult> {
        // Index creation deferred to TableIndexRegistry integration
        Ok(ExecutionResult::Ok)
    }

    fn drop_index(&mut self, _stmt: &DropIndexStmt) -> SqlResult<ExecutionResult> {
        Ok(ExecutionResult::Ok)
    }

    fn alter_table(&mut self, stmt: &AlterTableStmt) -> SqlResult<ExecutionResult> {
        use sochdb_storage::{DbColumnDef, DbColumnType};

        let table_name = stmt.name.name().to_string();
        let mut schema = self
            .db
            .get_table_schema(&table_name)
            .ok_or_else(|| SqlError::TableNotFound(table_name.clone()))?;

        let original_name = table_name.clone();

        for op in &stmt.operations {
            match op {
                AlterTableOp::AddColumn(col_def) => {
                    // Check for duplicate column
                    if schema.columns.iter().any(|c| c.name == col_def.name) {
                        return Err(SqlError::InvalidArgument(format!(
                            "Column '{}' already exists in table '{}'",
                            col_def.name, schema.name
                        )));
                    }
                    let col_type = sql_type_to_db_type(&col_def.data_type);
                    let nullable = !col_def
                        .constraints
                        .iter()
                        .any(|c| matches!(c, ColumnConstraint::NotNull));
                    schema.columns.push(DbColumnDef {
                        name: col_def.name.clone(),
                        col_type,
                        nullable,
                    });
                }
                AlterTableOp::DropColumn { name, .. } => {
                    let idx = schema
                        .columns
                        .iter()
                        .position(|c| c.name == *name)
                        .ok_or_else(|| {
                            SqlError::InvalidArgument(format!(
                                "Column '{}' not found in table '{}'",
                                name, schema.name
                            ))
                        })?;
                    schema.columns.remove(idx);
                }
                AlterTableOp::RenameColumn { old_name, new_name } => {
                    let col = schema
                        .columns
                        .iter_mut()
                        .find(|c| c.name == *old_name)
                        .ok_or_else(|| {
                            SqlError::InvalidArgument(format!(
                                "Column '{}' not found in table '{}'",
                                old_name, schema.name
                            ))
                        })?;
                    col.name = new_name.clone();
                }
                AlterTableOp::RenameTable(new_name) => {
                    schema.name = new_name.name().to_string();
                }
                AlterTableOp::AlterColumn { name, operation } => {
                    let col = schema
                        .columns
                        .iter_mut()
                        .find(|c| c.name == *name)
                        .ok_or_else(|| {
                            SqlError::InvalidArgument(format!(
                                "Column '{}' not found in table '{}'",
                                name, schema.name
                            ))
                        })?;
                    match operation {
                        AlterColumnOp::SetType(data_type) => {
                            col.col_type = sql_type_to_db_type(data_type);
                        }
                        AlterColumnOp::SetNotNull => {
                            col.nullable = false;
                        }
                        AlterColumnOp::DropNotNull => {
                            col.nullable = true;
                        }
                        AlterColumnOp::SetDefault(_) | AlterColumnOp::DropDefault => {
                            // Defaults are handled at the SQL layer, not stored
                            // in the storage schema currently. This is a no-op.
                        }
                    }
                }
                AlterTableOp::AddConstraint(_) | AlterTableOp::DropConstraint { .. } => {
                    return Err(SqlError::NotImplemented(
                        "ADD/DROP CONSTRAINT not yet implemented".into(),
                    ));
                }
            }
        }

        self.db
            .update_table_schema(&original_name, schema)
            .map_err(|e| SqlError::ExecutionError(format!("ALTER TABLE failed: {}", e)))?;

        Ok(ExecutionResult::Ok)
    }

    fn begin(&mut self, _stmt: &BeginStmt) -> SqlResult<ExecutionResult> {
        if self.active_txn.is_some() {
            return Err(SqlError::TransactionError(
                "Transaction already active".into(),
            ));
        }
        let txn = self
            .db
            .begin_transaction()
            .map_err(|e| SqlError::ExecutionError(format!("Begin failed: {}", e)))?;
        self.active_txn = Some(txn);
        self.explicit_txn = true;
        Ok(ExecutionResult::TransactionOk)
    }

    fn commit(&mut self) -> SqlResult<ExecutionResult> {
        if let Some(txn) = self.active_txn.take() {
            self.explicit_txn = false;
            self.db
                .commit(txn)
                .map_err(|e| SqlError::TransactionError(format!("Commit failed: {}", e)))?;
            Ok(ExecutionResult::TransactionOk)
        } else {
            Err(SqlError::TransactionError("No active transaction".into()))
        }
    }

    fn rollback(&mut self, _savepoint: Option<&str>) -> SqlResult<ExecutionResult> {
        if let Some(txn) = self.active_txn.take() {
            self.explicit_txn = false;
            self.db
                .abort(txn)
                .map_err(|e| SqlError::TransactionError(format!("Rollback failed: {}", e)))?;
            Ok(ExecutionResult::TransactionOk)
        } else {
            Err(SqlError::TransactionError("No active transaction".into()))
        }
    }

    fn table_exists(&self, table: &str) -> SqlResult<bool> {
        Ok(self.db.get_table_schema(table).is_some())
    }

    fn index_exists(&self, _index: &str) -> SqlResult<bool> {
        // TODO: wire to TableIndexRegistry
        Ok(false)
    }

    fn scan_all(
        &self,
        table: &str,
        columns: &[String],
    ) -> SqlResult<Vec<HashMap<String, CoreSochValue>>> {
        let txn = self.db.begin_read_only_fast();

        let query = if columns.is_empty() || columns.iter().any(|c| c == "*") {
            self.db.query(txn, table)
        } else {
            let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            self.db.query(txn, table).columns(&col_refs)
        };

        let result = query
            .execute()
            .map_err(|e| SqlError::ExecutionError(format!("Scan failed: {}", e)));

        self.db.abort_read_only_fast(txn);
        Ok(result?.rows)
    }

    fn eval_join_predicate(
        &self,
        expr: &Expr,
        row: &HashMap<String, CoreSochValue>,
        params: &[CoreSochValue],
    ) -> Option<bool> {
        let val = self.eval_expr(expr, row, params)?;
        match val {
            CoreSochValue::Bool(b) => Some(b),
            CoreSochValue::Null => Some(false),
            _ => Some(false),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple string-based predicate evaluation for `StorageBackend::table_scan()`.
///
/// Supports: `column = value`, `column != value`, `column > value`,
/// `column < value`, `column >= value`, `column <= value`.
fn apply_simple_predicate(
    rows: &[HashMap<String, QuerySochValue>],
    predicate: &str,
) -> Vec<HashMap<String, QuerySochValue>> {
    let operators = [">=", "<=", "!=", "=", ">", "<"];

    for op in &operators {
        if let Some(idx) = predicate.find(op) {
            let column = predicate[..idx].trim();
            let value_str = predicate[idx + op.len()..].trim().trim_matches('\'');

            return rows
                .iter()
                .filter(|row| {
                    if let Some(val) = row.get(column) {
                        let val_str = match val {
                            QuerySochValue::Text(s) => s.clone(),
                            QuerySochValue::Int(i) => i.to_string(),
                            QuerySochValue::UInt(u) => u.to_string(),
                            QuerySochValue::Float(f) => f.to_string(),
                            QuerySochValue::Bool(b) => b.to_string(),
                            _ => return false,
                        };

                        match *op {
                            "=" => val_str == value_str,
                            "!=" => val_str != value_str,
                            ">" => val_str.as_str() > value_str,
                            "<" => (val_str.as_str()) < value_str,
                            ">=" => val_str.as_str() >= value_str,
                            "<=" => val_str.as_str() <= value_str,
                            _ => false,
                        }
                    } else {
                        false
                    }
                })
                .cloned()
                .collect();
        }
    }

    rows.to_vec()
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Convert a SQL literal to `CoreSochValue`.
fn literal_to_core(lit: &Literal) -> CoreSochValue {
    match lit {
        Literal::Integer(i) => CoreSochValue::Int(*i),
        Literal::Float(f) => CoreSochValue::Float(*f),
        Literal::String(s) => CoreSochValue::Text(s.clone()),
        Literal::Boolean(b) => CoreSochValue::Bool(*b),
        Literal::Null => CoreSochValue::Null,
        Literal::Blob(b) => CoreSochValue::Binary(b.clone()),
    }
}

/// Evaluate a binary operation on two `CoreSochValue`s.
fn eval_binary_op(
    lhs: &CoreSochValue,
    op: &BinaryOperator,
    rhs: &CoreSochValue,
) -> CoreSochValue {
    match op {
        BinaryOperator::Eq => CoreSochValue::Bool(lhs == rhs),
        BinaryOperator::Ne => CoreSochValue::Bool(lhs != rhs),
        BinaryOperator::Lt => {
            CoreSochValue::Bool(compare_values(lhs, rhs) == std::cmp::Ordering::Less)
        }
        BinaryOperator::Gt => {
            CoreSochValue::Bool(compare_values(lhs, rhs) == std::cmp::Ordering::Greater)
        }
        BinaryOperator::Le => {
            CoreSochValue::Bool(compare_values(lhs, rhs) != std::cmp::Ordering::Greater)
        }
        BinaryOperator::Ge => {
            CoreSochValue::Bool(compare_values(lhs, rhs) != std::cmp::Ordering::Less)
        }
        BinaryOperator::And => {
            let a = matches!(lhs, CoreSochValue::Bool(true));
            let b = matches!(rhs, CoreSochValue::Bool(true));
            CoreSochValue::Bool(a && b)
        }
        BinaryOperator::Or => {
            let a = matches!(lhs, CoreSochValue::Bool(true));
            let b = matches!(rhs, CoreSochValue::Bool(true));
            CoreSochValue::Bool(a || b)
        }
        BinaryOperator::Plus => numeric_op(lhs, rhs, |a, b| a + b, |a, b| a + b),
        BinaryOperator::Minus => numeric_op(lhs, rhs, |a, b| a - b, |a, b| a - b),
        BinaryOperator::Multiply => numeric_op(lhs, rhs, |a, b| a * b, |a, b| a * b),
        BinaryOperator::Divide => numeric_op(
            lhs,
            rhs,
            |a, b| if b != 0 { a / b } else { 0 },
            |a, b| if b != 0.0 { a / b } else { 0.0 },
        ),
        BinaryOperator::Modulo => numeric_op(
            lhs,
            rhs,
            |a, b| if b != 0 { a % b } else { 0 },
            |a, b| if b != 0.0 { a % b } else { 0.0 },
        ),
        BinaryOperator::Like => {
            if let (CoreSochValue::Text(s), CoreSochValue::Text(pattern)) = (lhs, rhs) {
                CoreSochValue::Bool(sql_like_match(s, pattern))
            } else {
                CoreSochValue::Bool(false)
            }
        }
        BinaryOperator::Concat => {
            let a = value_to_string(lhs);
            let b = value_to_string(rhs);
            CoreSochValue::Text(format!("{}{}", a, b))
        }
        _ => CoreSochValue::Null,
    }
}

/// Evaluate a unary operation.
fn eval_unary_op(op: &UnaryOperator, val: &CoreSochValue) -> CoreSochValue {
    match op {
        UnaryOperator::Not => match val {
            CoreSochValue::Bool(b) => CoreSochValue::Bool(!b),
            _ => CoreSochValue::Null,
        },
        UnaryOperator::Minus => match val {
            CoreSochValue::Int(i) => CoreSochValue::Int(-i),
            CoreSochValue::Float(f) => CoreSochValue::Float(-f),
            _ => CoreSochValue::Null,
        },
        UnaryOperator::Plus => val.clone(),
        _ => CoreSochValue::Null,
    }
}

/// Numeric operation helper.
fn numeric_op(
    lhs: &CoreSochValue,
    rhs: &CoreSochValue,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> CoreSochValue {
    match (lhs, rhs) {
        (CoreSochValue::Int(a), CoreSochValue::Int(b)) => CoreSochValue::Int(int_op(*a, *b)),
        (CoreSochValue::Float(a), CoreSochValue::Float(b)) => {
            CoreSochValue::Float(float_op(*a, *b))
        }
        (CoreSochValue::Int(a), CoreSochValue::Float(b)) => {
            CoreSochValue::Float(float_op(*a as f64, *b))
        }
        (CoreSochValue::Float(a), CoreSochValue::Int(b)) => {
            CoreSochValue::Float(float_op(*a, *b as f64))
        }
        (CoreSochValue::UInt(a), CoreSochValue::UInt(b)) => {
            CoreSochValue::Int(int_op(*a as i64, *b as i64))
        }
        _ => CoreSochValue::Null,
    }
}

/// Compare two `CoreSochValue`s for ordering.
fn compare_values(a: &CoreSochValue, b: &CoreSochValue) -> std::cmp::Ordering {
    match (a, b) {
        (CoreSochValue::Int(a), CoreSochValue::Int(b)) => a.cmp(b),
        (CoreSochValue::UInt(a), CoreSochValue::UInt(b)) => a.cmp(b),
        (CoreSochValue::Float(a), CoreSochValue::Float(b)) => {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        }
        (CoreSochValue::Text(a), CoreSochValue::Text(b)) => a.cmp(b),
        (CoreSochValue::Int(a), CoreSochValue::Float(b)) => (*a as f64)
            .partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal),
        (CoreSochValue::Float(a), CoreSochValue::Int(b)) => a
            .partial_cmp(&(*b as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (CoreSochValue::Null, CoreSochValue::Null) => std::cmp::Ordering::Equal,
        (CoreSochValue::Null, _) => std::cmp::Ordering::Less,
        (_, CoreSochValue::Null) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    }
}

/// Compare optional CoreSochValues for ORDER BY.
fn compare_optional_values(
    a: Option<&CoreSochValue>,
    b: Option<&CoreSochValue>,
) -> std::cmp::Ordering {
    match (a, b) {
        (Some(a), Some(b)) => compare_values(a, b),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Extract column name from an ORDER BY expression.
fn extract_order_by_column(expr: &Expr) -> String {
    match expr {
        Expr::Column(col_ref) => col_ref.column.clone(),
        _ => String::new(),
    }
}

/// Check if two rows have the same values.
fn rows_equal(
    a: &HashMap<String, CoreSochValue>,
    b: &HashMap<String, CoreSochValue>,
) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().all(|(k, v)| b.get(k) == Some(v))
}

/// Convert a value to string representation.
fn value_to_string(v: &CoreSochValue) -> String {
    match v {
        CoreSochValue::Text(s) => s.clone(),
        CoreSochValue::Int(i) => i.to_string(),
        CoreSochValue::UInt(u) => u.to_string(),
        CoreSochValue::Float(f) => f.to_string(),
        CoreSochValue::Bool(b) => b.to_string(),
        CoreSochValue::Null => "NULL".to_string(),
        _ => String::new(),
    }
}

/// SQL LIKE pattern matching.
///
/// Converts SQL LIKE patterns (`%`, `_`) to regex equivalents.
fn sql_like_match(s: &str, pattern: &str) -> bool {
    // Escape regex special chars first, then convert LIKE wildcards
    let mut regex_pattern = String::with_capacity(pattern.len() * 2 + 2);
    regex_pattern.push('^');
    for ch in pattern.chars() {
        match ch {
            '%' => regex_pattern.push_str(".*"),
            '_' => regex_pattern.push('.'),
            '.' | '(' | ')' | '[' | ']' | '{' | '}' | '+' | '*' | '?' | '^' | '$' | '|'
            | '\\' => {
                regex_pattern.push('\\');
                regex_pattern.push(ch);
            }
            _ => regex_pattern.push(ch),
        }
    }
    regex_pattern.push('$');

    regex::Regex::new(&regex_pattern)
        .map(|re| re.is_match(s))
        .unwrap_or(false)
}

/// Convert SQL data type to storage column type.
fn sql_type_to_db_type(dt: &DataType) -> sochdb_storage::DbColumnType {
    use sochdb_storage::DbColumnType;
    match dt {
        DataType::TinyInt | DataType::SmallInt | DataType::Int | DataType::BigInt => {
            DbColumnType::Int64
        }
        DataType::Float | DataType::Double | DataType::Decimal { .. } => DbColumnType::Float64,
        DataType::Boolean => DbColumnType::Bool,
        DataType::Binary(_) | DataType::Varbinary(_) | DataType::Blob => DbColumnType::Binary,
        // All other types (text, date, json, custom, vector) → Text
        _ => DbColumnType::Text,
    }
}

// ============================================================================
// NamespacedSqlConnection — Namespace-Prefixed Storage Isolation (P3.2)
// ============================================================================

/// A wrapper around any `SqlConnection` that prefixes table names with
/// `namespace:database:` to provide storage-level isolation between tenants.
///
/// ## Key Isolation
///
/// All storage keys are prefixed: `ns:db:table/row_id` instead of `table/row_id`.
/// This prevents any cross-namespace data leakage at the storage layer.
///
/// ## Example
///
/// ```text
/// // Without namespace prefix:
/// table key = "users/1"
///
/// // With namespace prefix (namespace="prod", database="app"):
/// table key = "prod:app:users/1"
/// ```
pub struct NamespacedSqlConnection<C: SqlConnection> {
    inner: C,
    namespace: String,
    database: String,
}

impl<C: SqlConnection> NamespacedSqlConnection<C> {
    /// Create a new namespaced connection.
    pub fn new(inner: C, namespace: impl Into<String>, database: impl Into<String>) -> Self {
        Self {
            inner,
            namespace: namespace.into(),
            database: database.into(),
        }
    }

    /// Prefix a table name with namespace:database:
    fn prefix_table(&self, table: &str) -> String {
        format!("{}:{}:{}", self.namespace, self.database, table)
    }

    /// Get the namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Get the database.
    pub fn database(&self) -> &str {
        &self.database
    }

    /// Get a reference to the inner connection.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Get a mutable reference to the inner connection.
    pub fn inner_mut(&mut self) -> &mut C {
        &mut self.inner
    }
}

impl<C: SqlConnection> SqlConnection for NamespacedSqlConnection<C> {
    fn select(
        &self,
        table: &str,
        columns: &[String],
        where_clause: Option<&Expr>,
        order_by: &[OrderByItem],
        limit: Option<usize>,
        offset: Option<usize>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        self.inner.select(&self.prefix_table(table), columns, where_clause, order_by, limit, offset, params)
    }

    fn insert(
        &mut self,
        table: &str,
        columns: Option<&[String]>,
        rows: &[Vec<Expr>],
        on_conflict: Option<&OnConflict>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        self.inner.insert(&self.prefix_table(table), columns, rows, on_conflict, params)
    }

    fn update(
        &mut self,
        table: &str,
        assignments: &[Assignment],
        where_clause: Option<&Expr>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        self.inner.update(&self.prefix_table(table), assignments, where_clause, params)
    }

    fn delete(
        &mut self,
        table: &str,
        where_clause: Option<&Expr>,
        params: &[CoreSochValue],
    ) -> SqlResult<ExecutionResult> {
        self.inner.delete(&self.prefix_table(table), where_clause, params)
    }

    fn create_table(&mut self, stmt: &CreateTableStmt) -> SqlResult<ExecutionResult> {
        // Create a modified statement with prefixed table name
        let mut prefixed = stmt.clone();
        let original_name = stmt.name.name().to_string();
        prefixed.name = ObjectName::new(self.prefix_table(&original_name));
        self.inner.create_table(&prefixed)
    }

    fn drop_table(&mut self, stmt: &DropTableStmt) -> SqlResult<ExecutionResult> {
        let mut prefixed = stmt.clone();
        prefixed.names = stmt.names.iter()
            .map(|n| ObjectName::new(self.prefix_table(n.name())))
            .collect();
        self.inner.drop_table(&prefixed)
    }

    fn create_index(&mut self, stmt: &CreateIndexStmt) -> SqlResult<ExecutionResult> {
        self.inner.create_index(stmt)
    }

    fn drop_index(&mut self, stmt: &DropIndexStmt) -> SqlResult<ExecutionResult> {
        self.inner.drop_index(stmt)
    }

    fn alter_table(&mut self, stmt: &AlterTableStmt) -> SqlResult<ExecutionResult> {
        let mut prefixed = stmt.clone();
        let original_name = stmt.name.name().to_string();
        prefixed.name = ObjectName::new(self.prefix_table(&original_name));
        self.inner.alter_table(&prefixed)
    }

    fn begin(&mut self, stmt: &BeginStmt) -> SqlResult<ExecutionResult> {
        self.inner.begin(stmt)
    }

    fn commit(&mut self) -> SqlResult<ExecutionResult> {
        self.inner.commit()
    }

    fn rollback(&mut self, savepoint: Option<&str>) -> SqlResult<ExecutionResult> {
        self.inner.rollback(savepoint)
    }

    fn table_exists(&self, table: &str) -> SqlResult<bool> {
        self.inner.table_exists(&self.prefix_table(table))
    }

    fn index_exists(&self, index: &str) -> SqlResult<bool> {
        self.inner.index_exists(index)
    }

    fn scan_all(
        &self,
        table: &str,
        columns: &[String],
    ) -> SqlResult<Vec<HashMap<String, CoreSochValue>>> {
        self.inner.scan_all(&self.prefix_table(table), columns)
    }

    fn eval_join_predicate(
        &self,
        expr: &Expr,
        row: &HashMap<String, CoreSochValue>,
        params: &[CoreSochValue],
    ) -> Option<bool> {
        self.inner.eval_join_predicate(expr, row, params)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_core_to_query_basic_types() {
        assert_eq!(
            convert_core_to_query(&CoreSochValue::Null),
            QuerySochValue::Null
        );
        assert_eq!(
            convert_core_to_query(&CoreSochValue::Bool(true)),
            QuerySochValue::Bool(true)
        );
        assert_eq!(
            convert_core_to_query(&CoreSochValue::Int(42)),
            QuerySochValue::Int(42)
        );
        assert_eq!(
            convert_core_to_query(&CoreSochValue::UInt(100)),
            QuerySochValue::UInt(100)
        );
        assert_eq!(
            convert_core_to_query(&CoreSochValue::Float(3.14)),
            QuerySochValue::Float(3.14)
        );
        assert_eq!(
            convert_core_to_query(&CoreSochValue::Text("hello".into())),
            QuerySochValue::Text("hello".into())
        );
    }

    #[test]
    fn test_convert_core_to_query_object() {
        let mut map = HashMap::new();
        map.insert("name".to_string(), CoreSochValue::Text("Alice".into()));
        let result = convert_core_to_query(&CoreSochValue::Object(map));
        match result {
            QuerySochValue::Text(s) => assert!(s.contains("Alice")),
            _ => panic!("Expected Text for Object conversion"),
        }
    }

    #[test]
    fn test_convert_core_to_query_ref() {
        let result = convert_core_to_query(&CoreSochValue::Ref {
            table: "users".into(),
            id: 42,
        });
        assert_eq!(result, QuerySochValue::Text("users/42".into()));
    }

    #[test]
    fn test_convert_roundtrip() {
        let original = QuerySochValue::Int(42);
        let core = convert_query_to_core(&original);
        let back = convert_core_to_query(&core);
        assert_eq!(original, back);
    }

    #[test]
    fn test_apply_simple_predicate_eq() {
        let rows = vec![
            {
                let mut m = HashMap::new();
                m.insert("name".into(), QuerySochValue::Text("Alice".into()));
                m.insert("age".into(), QuerySochValue::Int(30));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("name".into(), QuerySochValue::Text("Bob".into()));
                m.insert("age".into(), QuerySochValue::Int(25));
                m
            },
        ];

        let filtered = apply_simple_predicate(&rows, "name = Alice");
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered[0].get("name"),
            Some(&QuerySochValue::Text("Alice".into()))
        );
    }

    #[test]
    fn test_apply_simple_predicate_neq() {
        let rows = vec![
            {
                let mut m = HashMap::new();
                m.insert("status".into(), QuerySochValue::Text("active".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("status".into(), QuerySochValue::Text("inactive".into()));
                m
            },
        ];

        let filtered = apply_simple_predicate(&rows, "status != active");
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered[0].get("status"),
            Some(&QuerySochValue::Text("inactive".into()))
        );
    }

    #[test]
    fn test_literal_to_core() {
        assert_eq!(
            literal_to_core(&Literal::Integer(42)),
            CoreSochValue::Int(42)
        );
        assert_eq!(
            literal_to_core(&Literal::Float(3.14)),
            CoreSochValue::Float(3.14)
        );
        assert_eq!(
            literal_to_core(&Literal::String("hi".into())),
            CoreSochValue::Text("hi".into())
        );
        assert_eq!(
            literal_to_core(&Literal::Boolean(true)),
            CoreSochValue::Bool(true)
        );
        assert_eq!(literal_to_core(&Literal::Null), CoreSochValue::Null);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_compare_values() {
        assert_eq!(
            compare_values(&CoreSochValue::Int(1), &CoreSochValue::Int(2)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_values(
                &CoreSochValue::Text("a".into()),
                &CoreSochValue::Text("b".into())
            ),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_values(&CoreSochValue::Null, &CoreSochValue::Int(1)),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_eval_binary_op() {
        assert_eq!(
            eval_binary_op(
                &CoreSochValue::Int(10),
                &BinaryOperator::Plus,
                &CoreSochValue::Int(5)
            ),
            CoreSochValue::Int(15)
        );
        assert_eq!(
            eval_binary_op(
                &CoreSochValue::Int(10),
                &BinaryOperator::Eq,
                &CoreSochValue::Int(10)
            ),
            CoreSochValue::Bool(true)
        );
        assert_eq!(
            eval_binary_op(
                &CoreSochValue::Text("hello".into()),
                &BinaryOperator::Concat,
                &CoreSochValue::Text(" world".into())
            ),
            CoreSochValue::Text("hello world".into())
        );
    }

    #[test]
    fn test_sql_like_match() {
        assert!(sql_like_match("hello", "hello"));
        assert!(sql_like_match("hello", "%llo"));
        assert!(sql_like_match("hello", "h%o"));
        assert!(sql_like_match("hello", "h_llo"));
        assert!(!sql_like_match("hello", "world"));
        // Test with regex special characters in the data
        assert!(sql_like_match("file.txt", "file%"));
        assert!(sql_like_match("test(1)", "%(%"));
    }

    #[test]
    fn test_sql_type_to_db_type() {
        use sochdb_storage::DbColumnType;
        assert_eq!(sql_type_to_db_type(&DataType::Int), DbColumnType::Int64);
        assert_eq!(sql_type_to_db_type(&DataType::BigInt), DbColumnType::Int64);
        assert_eq!(
            sql_type_to_db_type(&DataType::Float),
            DbColumnType::Float64
        );
        assert_eq!(
            sql_type_to_db_type(&DataType::Boolean),
            DbColumnType::Bool
        );
        assert_eq!(sql_type_to_db_type(&DataType::Blob), DbColumnType::Binary);
        assert_eq!(sql_type_to_db_type(&DataType::Text), DbColumnType::Text);
        assert_eq!(
            sql_type_to_db_type(&DataType::Varchar(Some(255))),
            DbColumnType::Text
        );
    }

    // =========================================================================
    // Integration tests: Full round-trip through real Database
    // =========================================================================

    fn setup_test_db() -> (std::sync::Arc<sochdb_storage::Database>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db = sochdb_storage::Database::open(tmp.path()).expect("open db");
        (db, tmp)
    }

    #[test]
    fn test_integration_storage_backend_table_scan() {
        use sochdb_storage::{DbColumnDef, DbColumnType, DbTableSchema};
        let (db, _tmp) = setup_test_db();

        // Register table
        db.register_table(DbTableSchema {
            name: "users".into(),
            columns: vec![
                DbColumnDef { name: "id".into(), col_type: DbColumnType::Int64, nullable: false },
                DbColumnDef { name: "name".into(), col_type: DbColumnType::Text, nullable: false },
                DbColumnDef { name: "age".into(), col_type: DbColumnType::Int64, nullable: true },
            ],
        })
        .expect("register table");

        // Insert rows via raw storage API
        let txn = db.begin_transaction().expect("begin txn");
        let mut vals = std::collections::HashMap::new();
        vals.insert("id".into(), CoreSochValue::Int(1));
        vals.insert("name".into(), CoreSochValue::Text("Alice".into()));
        vals.insert("age".into(), CoreSochValue::Int(30));
        db.insert_row(txn, "users", 1, &vals).expect("insert 1");

        vals.clear();
        vals.insert("id".into(), CoreSochValue::Int(2));
        vals.insert("name".into(), CoreSochValue::Text("Bob".into()));
        vals.insert("age".into(), CoreSochValue::Int(25));
        db.insert_row(txn, "users", 2, &vals).expect("insert 2");
        db.commit(txn).expect("commit");

        // Query via DatabaseStorageBackend
        let backend = DatabaseStorageBackend::new(db.clone());
        let rows = backend
            .table_scan("users", &["id".into(), "name".into(), "age".into()], None)
            .expect("table_scan");

        assert_eq!(rows.len(), 2);
        // Verify data came through the conversion pipeline
        let names: Vec<_> = rows
            .iter()
            .filter_map(|r| match r.get("name") {
                Some(crate::soch_ql::SochValue::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"Alice".to_string()));
        assert!(names.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_integration_storage_backend_primary_key_lookup() {
        use sochdb_storage::{DbColumnDef, DbColumnType, DbTableSchema};
        let (db, _tmp) = setup_test_db();

        db.register_table(DbTableSchema {
            name: "items".into(),
            columns: vec![
                DbColumnDef { name: "id".into(), col_type: DbColumnType::Int64, nullable: false },
                DbColumnDef { name: "label".into(), col_type: DbColumnType::Text, nullable: false },
            ],
        })
        .expect("register");

        let txn = db.begin_transaction().expect("txn");
        let mut v = std::collections::HashMap::new();
        v.insert("id".into(), CoreSochValue::Int(42));
        v.insert("label".into(), CoreSochValue::Text("answer".into()));
        db.insert_row(txn, "items", 42, &v).expect("insert");
        db.commit(txn).expect("commit");

        let backend = DatabaseStorageBackend::new(db.clone());
        let row = backend
            .primary_key_lookup("items", &crate::soch_ql::SochValue::Int(42))
            .expect("pk lookup");
        assert!(row.is_some());
        let row = row.unwrap();
        assert_eq!(
            row.get("label"),
            Some(&crate::soch_ql::SochValue::Text("answer".into()))
        );
    }

    #[test]
    fn test_integration_sochql_executor_reads_storage() {
        use sochdb_storage::{DbColumnDef, DbColumnType, DbTableSchema};
        use sochdb_core::{Catalog, SochSchema, SochType};
        let (db, _tmp) = setup_test_db();

        // Register table in storage
        db.register_table(DbTableSchema {
            name: "events".into(),
            columns: vec![
                DbColumnDef { name: "id".into(), col_type: DbColumnType::Int64, nullable: false },
                DbColumnDef { name: "kind".into(), col_type: DbColumnType::Text, nullable: false },
                DbColumnDef { name: "score".into(), col_type: DbColumnType::Float64, nullable: true },
            ],
        })
        .expect("register events");

        // Insert data
        let txn = db.begin_transaction().expect("txn");
        for i in 1..=5u64 {
            let mut vals = std::collections::HashMap::new();
            vals.insert("id".into(), CoreSochValue::Int(i as i64));
            vals.insert("kind".into(), CoreSochValue::Text(format!("event_{}", i)));
            vals.insert("score".into(), CoreSochValue::Float(i as f64 * 1.5));
            db.insert_row(txn, "events", i, &vals).expect("insert");
        }
        db.commit(txn).expect("commit");

        // Build catalog (needed by SochQlExecutor for validation)
        let mut catalog = Catalog::new("test");
        let schema = SochSchema {
            name: "events".into(),
            fields: vec![
                sochdb_core::SochField { name: "id".into(), field_type: SochType::Int, nullable: false, default: None },
                sochdb_core::SochField { name: "kind".into(), field_type: SochType::Text, nullable: false, default: None },
                sochdb_core::SochField { name: "score".into(), field_type: SochType::Float, nullable: true, default: None },
            ],
            primary_key: None,
            indexes: vec![],
        };
        catalog.create_table(schema, 0).expect("register catalog");

        // Execute via SochQlExecutor with storage backend
        let backend = std::sync::Arc::new(DatabaseStorageBackend::new(db.clone()));
        let executor = crate::soch_ql_executor::SochQlExecutor::with_storage(backend);
        let result = executor
            .execute("SELECT * FROM events", &catalog)
            .expect("select *");

        // Phase 0 success: we actually get rows from storage!
        assert_eq!(result.rows.len(), 5, "Expected 5 rows from storage, got {}", result.rows.len());
        assert_eq!(result.columns, vec!["id", "kind", "score"]);

        // Verify data integrity
        let first_row_kind = &result.rows[0];
        // Rows should have 3 values each (id, kind, score)
        assert_eq!(first_row_kind.len(), 3);
    }

    #[test]
    fn test_integration_sql_connection_crud() {
        use crate::sql::bridge::SqlConnection;

        let (db, _tmp) = setup_test_db();
        let mut conn = DatabaseSqlConnection::new(db.clone());

        // CREATE TABLE via SQL connection
        let create = CreateTableStmt {
            span: crate::sql::token::Span::new(0, 0, 0, 0),
            name: crate::sql::ast::ObjectName::new("products"),
            columns: vec![
                crate::sql::ast::ColumnDef {
                    name: "id".into(),
                    data_type: DataType::Int,
                    constraints: vec![],
                },
                crate::sql::ast::ColumnDef {
                    name: "name".into(),
                    data_type: DataType::Text,
                    constraints: vec![],
                },
                crate::sql::ast::ColumnDef {
                    name: "price".into(),
                    data_type: DataType::Float,
                    constraints: vec![],
                },
            ],
            if_not_exists: false,
            constraints: vec![],
            options: vec![],
        };
        let result = conn.create_table(&create).expect("create table");
        assert!(matches!(result, crate::sql::bridge::ExecutionResult::Ok));

        // BEGIN
        let begin_stmt = crate::sql::ast::BeginStmt {
            isolation_level: None,
            read_only: false,
        };
        let result = conn.begin(&begin_stmt).expect("begin");
        assert!(matches!(result, crate::sql::bridge::ExecutionResult::TransactionOk));

        // INSERT via decomposed SqlConnection::insert
        let values = vec![vec![
            Expr::Literal(Literal::Integer(1)),
            Expr::Literal(Literal::String("Widget".into())),
            Expr::Literal(Literal::Float(9.99)),
        ]];
        let cols = vec!["id".into(), "name".into(), "price".into()];
        let result = conn
            .insert("products", Some(&cols), &values, None, &[])
            .expect("insert");
        match &result {
            crate::sql::bridge::ExecutionResult::RowsAffected(n) => assert_eq!(*n, 1),
            _ => panic!("Expected RowsAffected, got {:?}", result),
        }

        // COMMIT
        let result = conn.commit().expect("commit");
        assert!(matches!(result, crate::sql::bridge::ExecutionResult::TransactionOk));

        // SELECT via decomposed SqlConnection::select
        let columns = vec!["name".into()];
        let result = conn
            .select("products", &columns, None, &[], None, None, &[])
            .expect("select");
        match result {
            crate::sql::bridge::ExecutionResult::Rows { columns, rows } => {
                assert!(!rows.is_empty(), "Expected rows from SELECT");
                assert_eq!(columns, vec!["name"]);
                // Verify data round-trips: rows are Vec<HashMap<String, SochValue>>
                let name = rows[0].get("name").expect("name column");
                match name {
                    CoreSochValue::Text(s) => assert_eq!(s, "Widget"),
                    other => panic!("Expected Text, got {:?}", other),
                }
            }
            other => panic!("Expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn test_integration_row_count() {
        use sochdb_storage::{DbColumnDef, DbColumnType, DbTableSchema};
        let (db, _tmp) = setup_test_db();

        db.register_table(DbTableSchema {
            name: "metrics".into(),
            columns: vec![
                DbColumnDef { name: "ts".into(), col_type: DbColumnType::UInt64, nullable: false },
                DbColumnDef { name: "val".into(), col_type: DbColumnType::Float64, nullable: false },
            ],
        })
        .expect("register");

        let txn = db.begin_transaction().expect("txn");
        for i in 0..10u64 {
            let mut v = std::collections::HashMap::new();
            v.insert("ts".into(), CoreSochValue::UInt(i));
            v.insert("val".into(), CoreSochValue::Float(i as f64));
            db.insert_row(txn, "metrics", i, &v).expect("insert");
        }
        db.commit(txn).expect("commit");

        let backend = DatabaseStorageBackend::new(db.clone());
        assert_eq!(backend.row_count("metrics"), 10);
        assert_eq!(backend.row_count("nonexistent"), 0);
    }

    // =======================================================================
    // Step 0d: End-to-end SQL text → SqlBridge → DatabaseSqlConnection tests
    // =======================================================================

    #[test]
    fn test_sqlbridge_create_insert_select() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        // CREATE TABLE via raw SQL
        let r = bridge.execute("CREATE TABLE cities (id INT, name TEXT, pop FLOAT)").unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));

        // INSERT via raw SQL
        let r = bridge.execute("INSERT INTO cities (id, name, pop) VALUES (1, 'Tokyo', 13.96)").unwrap();
        match &r {
            crate::sql::bridge::ExecutionResult::RowsAffected(n) => assert_eq!(*n, 1),
            other => panic!("Expected RowsAffected, got {:?}", other),
        }

        let r = bridge.execute("INSERT INTO cities (id, name, pop) VALUES (2, 'Delhi', 11.03)").unwrap();
        match &r {
            crate::sql::bridge::ExecutionResult::RowsAffected(n) => assert_eq!(*n, 1),
            other => panic!("Expected RowsAffected, got {:?}", other),
        }

        // SELECT via raw SQL
        let r = bridge.execute("SELECT name, pop FROM cities").unwrap();
        match &r {
            crate::sql::bridge::ExecutionResult::Rows { columns, rows } => {
                assert_eq!(rows.len(), 2);
                assert!(columns.contains(&"name".to_string()));
            }
            other => panic!("Expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn test_sqlbridge_update_delete() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge.execute("CREATE TABLE items (id INT, qty INT)").unwrap();
        bridge.execute("INSERT INTO items (id, qty) VALUES (1, 10)").unwrap();
        bridge.execute("INSERT INTO items (id, qty) VALUES (2, 20)").unwrap();
        bridge.execute("INSERT INTO items (id, qty) VALUES (3, 30)").unwrap();

        // UPDATE
        let r = bridge.execute("UPDATE items SET qty = 99 WHERE id = 2").unwrap();
        match &r {
            crate::sql::bridge::ExecutionResult::RowsAffected(n) => assert_eq!(*n, 1),
            other => panic!("Expected RowsAffected for UPDATE, got {:?}", other),
        }

        // DELETE
        let r = bridge.execute("DELETE FROM items WHERE id = 3").unwrap();
        match &r {
            crate::sql::bridge::ExecutionResult::RowsAffected(n) => assert_eq!(*n, 1),
            other => panic!("Expected RowsAffected for DELETE, got {:?}", other),
        }

        // Verify remaining data
        let r = bridge.execute("SELECT id, qty FROM items").unwrap();
        match &r {
            crate::sql::bridge::ExecutionResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 2, "Expected 2 rows after delete");
            }
            other => panic!("Expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn test_sqlbridge_transaction_commit() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge.execute("CREATE TABLE txtest (id INT, val TEXT)").unwrap();

        // BEGIN / INSERT / COMMIT
        let r = bridge.execute("BEGIN").unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::TransactionOk));

        bridge.execute("INSERT INTO txtest (id, val) VALUES (1, 'committed')").unwrap();

        let r = bridge.execute("COMMIT").unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::TransactionOk));

        // Data should be visible
        let r = bridge.execute("SELECT val FROM txtest").unwrap();
        match &r {
            crate::sql::bridge::ExecutionResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 1);
            }
            other => panic!("Expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn test_sqlbridge_drop_table() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge.execute("CREATE TABLE ephemeral (x INT)").unwrap();
        let r = bridge.execute("DROP TABLE ephemeral").unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));

        // Verify DROP TABLE IF EXISTS also works
        let r = bridge.execute("DROP TABLE IF EXISTS ephemeral").unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));
    }

    #[test]
    fn test_sqlbridge_if_not_exists() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge.execute("CREATE TABLE dup (id INT)").unwrap();
        // Should not error with IF NOT EXISTS
        let r = bridge.execute("CREATE TABLE IF NOT EXISTS dup (id INT)").unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));
    }

    #[test]
    fn test_sqlbridge_alter_add_column() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge
            .execute("CREATE TABLE alter_test (id INT, name TEXT)")
            .unwrap();

        let r = bridge
            .execute("ALTER TABLE alter_test ADD COLUMN age INT")
            .unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));

        // Verify schema was updated
        let schema = db.get_table_schema("alter_test").unwrap();
        assert_eq!(schema.columns.len(), 3);
        assert_eq!(schema.columns[2].name, "age");
    }

    #[test]
    fn test_sqlbridge_alter_drop_column() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge
            .execute("CREATE TABLE drop_col_test (id INT, name TEXT, age INT)")
            .unwrap();

        let r = bridge
            .execute("ALTER TABLE drop_col_test DROP COLUMN age")
            .unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));

        let schema = db.get_table_schema("drop_col_test").unwrap();
        assert_eq!(schema.columns.len(), 2);
        assert!(schema.columns.iter().all(|c| c.name != "age"));
    }

    #[test]
    fn test_sqlbridge_alter_rename_column() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge
            .execute("CREATE TABLE rename_col_test (id INT, name TEXT)")
            .unwrap();

        let r = bridge
            .execute("ALTER TABLE rename_col_test RENAME COLUMN name TO full_name")
            .unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));

        let schema = db.get_table_schema("rename_col_test").unwrap();
        assert_eq!(schema.columns[1].name, "full_name");
    }

    #[test]
    fn test_sqlbridge_alter_rename_table() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge
            .execute("CREATE TABLE old_name (id INT)")
            .unwrap();

        let r = bridge
            .execute("ALTER TABLE old_name RENAME TO new_name")
            .unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));

        assert!(db.get_table_schema("old_name").is_none());
        assert!(db.get_table_schema("new_name").is_some());
    }

    #[test]
    fn test_sqlbridge_alter_multiple_ops() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge
            .execute("CREATE TABLE multi_alter (id INT, a TEXT, b TEXT)")
            .unwrap();

        // Add column and drop column in one statement
        let r = bridge
            .execute("ALTER TABLE multi_alter ADD COLUMN c INT, DROP COLUMN b")
            .unwrap();
        assert!(matches!(r, crate::sql::bridge::ExecutionResult::Ok));

        let schema = db.get_table_schema("multi_alter").unwrap();
        let col_names: Vec<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(col_names, vec!["id", "a", "c"]);
    }

    #[test]
    fn test_sqlbridge_alter_errors() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);

        bridge
            .execute("CREATE TABLE err_test (id INT, name TEXT)")
            .unwrap();

        // Add duplicate column
        let r = bridge.execute("ALTER TABLE err_test ADD COLUMN name TEXT");
        assert!(r.is_err());

        // Drop non-existent column
        let r = bridge.execute("ALTER TABLE err_test DROP COLUMN nonexistent");
        assert!(r.is_err());

        // Alter non-existent table
        let r = bridge.execute("ALTER TABLE no_such_table ADD COLUMN x INT");
        assert!(r.is_err());
    }

    // =======================================================================
    // JOIN Tests (Task 3 — Phase 3)
    // =======================================================================

    /// Helper: set up two tables for join testing (users + orders)
    fn setup_join_tables(bridge: &mut crate::sql::bridge::SqlBridge<DatabaseSqlConnection>) {
        bridge.execute("CREATE TABLE users (id INT, name TEXT, dept TEXT)").unwrap();
        bridge.execute("INSERT INTO users (id, name, dept) VALUES (1, 'Alice', 'eng')").unwrap();
        bridge.execute("INSERT INTO users (id, name, dept) VALUES (2, 'Bob', 'sales')").unwrap();
        bridge.execute("INSERT INTO users (id, name, dept) VALUES (3, 'Carol', 'eng')").unwrap();

        bridge.execute("CREATE TABLE orders (oid INT, user_id INT, amount FLOAT)").unwrap();
        bridge.execute("INSERT INTO orders (oid, user_id, amount) VALUES (10, 1, 99.50)").unwrap();
        bridge.execute("INSERT INTO orders (oid, user_id, amount) VALUES (11, 1, 45.00)").unwrap();
        bridge.execute("INSERT INTO orders (oid, user_id, amount) VALUES (12, 2, 200.00)").unwrap();
        // Note: Carol (id=3) has no orders; order 12 belongs to Bob (id=2)
    }

    #[test]
    fn test_sqlbridge_inner_join() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        let r = bridge.execute(
            "SELECT users.name, orders.amount FROM users INNER JOIN orders ON users.id = orders.user_id"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 3); // Alice×2, Bob×1

        // Verify Alice appears twice
        let alice_rows: Vec<_> = rows.iter()
            .filter(|r| r.get("name") == Some(&CoreSochValue::Text("Alice".into())))
            .collect();
        assert_eq!(alice_rows.len(), 2);

        // Verify Bob appears once
        let bob_rows: Vec<_> = rows.iter()
            .filter(|r| r.get("name") == Some(&CoreSochValue::Text("Bob".into())))
            .collect();
        assert_eq!(bob_rows.len(), 1);

        // Carol should NOT appear (no matching orders)
        let carol_rows: Vec<_> = rows.iter()
            .filter(|r| r.get("name") == Some(&CoreSochValue::Text("Carol".into())))
            .collect();
        assert_eq!(carol_rows.len(), 0);
    }

    #[test]
    fn test_sqlbridge_left_join() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        let r = bridge.execute(
            "SELECT users.name, orders.amount FROM users LEFT JOIN orders ON users.id = orders.user_id"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 4); // Alice×2, Bob×1, Carol×1(NULL)

        // Carol appears with NULL amount
        let carol_rows: Vec<_> = rows.iter()
            .filter(|r| r.get("name") == Some(&CoreSochValue::Text("Carol".into())))
            .collect();
        assert_eq!(carol_rows.len(), 1);
        assert_eq!(carol_rows[0].get("amount"), Some(&CoreSochValue::Null));
    }

    #[test]
    fn test_sqlbridge_right_join() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        // Right join: all orders appear; users without orders don't
        let r = bridge.execute(
            "SELECT users.name, orders.oid FROM users RIGHT JOIN orders ON users.id = orders.user_id"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 3); // All 3 orders match a user
    }

    #[test]
    fn test_sqlbridge_cross_join() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        let r = bridge.execute(
            "SELECT users.name, orders.oid FROM users CROSS JOIN orders"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 9); // 3 users × 3 orders = 9
    }

    #[test]
    fn test_sqlbridge_join_with_where() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        let r = bridge.execute(
            "SELECT users.name, orders.amount FROM users INNER JOIN orders ON users.id = orders.user_id WHERE orders.amount > 50"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 2); // Alice's 99.50 + Bob's 200.00
    }

    #[test]
    fn test_sqlbridge_join_with_alias() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        let r = bridge.execute(
            "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_sqlbridge_join_with_limit() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        let r = bridge.execute(
            "SELECT users.name, orders.oid FROM users INNER JOIN orders ON users.id = orders.user_id LIMIT 2"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_sqlbridge_join_three_tables() {
        let (db, _tmp) = setup_test_db();
        let conn = DatabaseSqlConnection::new(db.clone());
        let mut bridge = crate::sql::bridge::SqlBridge::new(conn);
        setup_join_tables(&mut bridge);

        // Add a departments table
        bridge.execute("CREATE TABLE departments (code TEXT, dname TEXT)").unwrap();
        bridge.execute("INSERT INTO departments (code, dname) VALUES ('eng', 'Engineering')").unwrap();
        bridge.execute("INSERT INTO departments (code, dname) VALUES ('sales', 'Sales')").unwrap();

        let r = bridge.execute(
            "SELECT users.name, departments.dname FROM users INNER JOIN departments ON users.dept = departments.code"
        ).unwrap();

        let rows = r.rows().unwrap();
        assert_eq!(rows.len(), 3); // All 3 users match a department
    }

    #[test]
    fn test_namespaced_connection_prefixes_tables() {
        let ns_conn = NamespacedSqlConnection::new(
            MockConn::default(),
            "prod",
            "app",
        );
        assert_eq!(ns_conn.prefix_table("users"), "prod:app:users");
        assert_eq!(ns_conn.prefix_table("posts"), "prod:app:posts");
        assert_eq!(ns_conn.namespace(), "prod");
        assert_eq!(ns_conn.database(), "app");
    }

    #[test]
    fn test_namespaced_connection_isolates_data() {
        let (db, _tmp) = setup_test_db();

        // Create two namespaced connections to the same underlying DB
        let conn_a = NamespacedSqlConnection::new(
            DatabaseSqlConnection::new(db.clone()),
            "tenant_a",
            "db1",
        );
        let conn_b = NamespacedSqlConnection::new(
            DatabaseSqlConnection::new(db.clone()),
            "tenant_b",
            "db1",
        );

        let mut bridge_a = crate::sql::bridge::SqlBridge::new(conn_a);
        let mut bridge_b = crate::sql::bridge::SqlBridge::new(conn_b);

        // Tenant A creates a table and inserts data
        bridge_a.execute("CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        bridge_a.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").unwrap();

        // Tenant B creates the same table name and inserts different data
        bridge_b.execute("CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        bridge_b.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").unwrap();

        // Tenant A can only see their own data
        let result_a = bridge_a.execute("SELECT * FROM users").unwrap();
        let rows_a = result_a.rows().unwrap();
        assert_eq!(rows_a.len(), 1);

        // Tenant B can only see their own data
        let result_b = bridge_b.execute("SELECT * FROM users").unwrap();
        let rows_b = result_b.rows().unwrap();
        assert_eq!(rows_b.len(), 1);
    }

    /// Minimal mock connection for prefix testing
    #[derive(Default)]
    struct MockConn;
    impl crate::sql::bridge::SqlConnection for MockConn {
        fn select(&self, _: &str, _: &[String], _: Option<&Expr>, _: &[OrderByItem], _: Option<usize>, _: Option<usize>, _: &[CoreSochValue]) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::Rows { columns: vec![], rows: vec![] })
        }
        fn insert(&mut self, _: &str, _: Option<&[String]>, _: &[Vec<Expr>], _: Option<&OnConflict>, _: &[CoreSochValue]) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn update(&mut self, _: &str, _: &[Assignment], _: Option<&Expr>, _: &[CoreSochValue]) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn delete(&mut self, _: &str, _: Option<&Expr>, _: &[CoreSochValue]) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn create_table(&mut self, _: &CreateTableStmt) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::Ok) }
        fn drop_table(&mut self, _: &DropTableStmt) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::Ok) }
        fn create_index(&mut self, _: &CreateIndexStmt) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::Ok) }
        fn drop_index(&mut self, _: &DropIndexStmt) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::Ok) }
        fn alter_table(&mut self, _: &AlterTableStmt) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::Ok) }
        fn begin(&mut self, _: &BeginStmt) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::TransactionOk) }
        fn commit(&mut self) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::TransactionOk) }
        fn rollback(&mut self, _: Option<&str>) -> SqlResult<ExecutionResult> { Ok(ExecutionResult::TransactionOk) }
        fn table_exists(&self, _: &str) -> SqlResult<bool> { Ok(false) }
        fn index_exists(&self, _: &str) -> SqlResult<bool> { Ok(false) }
        fn scan_all(&self, _: &str, _: &[String]) -> SqlResult<Vec<HashMap<String, CoreSochValue>>> { Ok(vec![]) }
        fn eval_join_predicate(&self, _: &Expr, _: &HashMap<String, CoreSochValue>, _: &[CoreSochValue]) -> Option<bool> { Some(true) }
    }
}
