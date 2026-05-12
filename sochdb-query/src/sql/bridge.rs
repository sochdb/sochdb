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

//! # SQL Execution Bridge
//!
//! Unified SQL execution pipeline that routes all SQL through a single AST.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │   SQL Text  │ --> │   Lexer     │ --> │   Parser    │ --> │    AST      │
//! └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
//!                                                                    │
//!                     ┌──────────────────────────────────────────────┘
//!                     │
//!                     v
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  Executor   │ <-- │  Planner    │ <-- │  Validator  │
//! └─────────────┘     └─────────────┘     └─────────────┘
//!       │
//!       v
//! ┌─────────────┐
//! │   Result    │
//! └─────────────┘
//! ```
//!
//! ## Benefits
//!
//! 1. **Single parser**: All SQL goes through one lexer/parser
//! 2. **Type-safe AST**: Structured representation of all queries
//! 3. **Dialect normalization**: MySQL/PostgreSQL/SQLite → canonical AST
//! 4. **Extensible**: Add new features by extending AST, not string parsing

use super::ast::*;
use super::compatibility::SqlDialect;
use super::error::{SqlError, SqlResult};
use super::parser::Parser;
use std::collections::HashMap;
use sochdb_core::SochValue;

/// Execution result types
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// SELECT query result
    Rows {
        columns: Vec<String>,
        rows: Vec<HashMap<String, SochValue>>,
    },
    /// DML result (INSERT/UPDATE/DELETE)
    RowsAffected(usize),
    /// DDL result (CREATE/DROP/ALTER)
    Ok,
    /// Transaction control result
    TransactionOk,
}

impl ExecutionResult {
    /// Get rows if this is a SELECT result
    pub fn rows(&self) -> Option<&Vec<HashMap<String, SochValue>>> {
        match self {
            ExecutionResult::Rows { rows, .. } => Some(rows),
            _ => None,
        }
    }

    /// Get column names if this is a SELECT result
    pub fn columns(&self) -> Option<&Vec<String>> {
        match self {
            ExecutionResult::Rows { columns, .. } => Some(columns),
            _ => None,
        }
    }

    /// Get affected row count
    pub fn rows_affected(&self) -> usize {
        match self {
            ExecutionResult::RowsAffected(n) => *n,
            ExecutionResult::Rows { rows, .. } => rows.len(),
            _ => 0,
        }
    }
}

/// Storage connection trait for executing SQL against actual storage
///
/// Implementations of this trait provide the bridge between parsed SQL
/// and the underlying storage engine.
pub trait SqlConnection {
    /// Execute a SELECT query
    fn select(
        &self,
        table: &str,
        columns: &[String],
        where_clause: Option<&Expr>,
        order_by: &[OrderByItem],
        limit: Option<usize>,
        offset: Option<usize>,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult>;

    /// Execute an INSERT
    fn insert(
        &mut self,
        table: &str,
        columns: Option<&[String]>,
        rows: &[Vec<Expr>],
        on_conflict: Option<&OnConflict>,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult>;

    /// Execute an UPDATE
    fn update(
        &mut self,
        table: &str,
        assignments: &[Assignment],
        where_clause: Option<&Expr>,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult>;

    /// Execute a DELETE
    fn delete(
        &mut self,
        table: &str,
        where_clause: Option<&Expr>,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult>;

    /// Create a table
    fn create_table(&mut self, stmt: &CreateTableStmt) -> SqlResult<ExecutionResult>;

    /// Drop a table
    fn drop_table(&mut self, stmt: &DropTableStmt) -> SqlResult<ExecutionResult>;

    /// Create an index
    fn create_index(&mut self, stmt: &CreateIndexStmt) -> SqlResult<ExecutionResult>;

    /// Drop an index
    fn drop_index(&mut self, stmt: &DropIndexStmt) -> SqlResult<ExecutionResult>;

    /// Alter a table (add/drop/rename columns, rename table)
    fn alter_table(&mut self, stmt: &AlterTableStmt) -> SqlResult<ExecutionResult>;

    /// Begin transaction
    fn begin(&mut self, stmt: &BeginStmt) -> SqlResult<ExecutionResult>;

    /// Commit transaction
    fn commit(&mut self) -> SqlResult<ExecutionResult>;

    /// Rollback transaction
    fn rollback(&mut self, savepoint: Option<&str>) -> SqlResult<ExecutionResult>;

    /// Check if table exists
    fn table_exists(&self, table: &str) -> SqlResult<bool>;

    /// Check if index exists
    fn index_exists(&self, index: &str) -> SqlResult<bool>;

    /// Scan all rows from a table (no filter, no ordering).
    /// Used for JOIN processing — each leaf table is scanned once,
    /// then join logic is applied in-memory.
    fn scan_all(
        &self,
        table: &str,
        columns: &[String],
    ) -> SqlResult<Vec<HashMap<String, SochValue>>>;

    /// Evaluate an expression against a merged row (used for JOIN ON conditions).
    /// Returns true/false, or None if evaluation fails.
    fn eval_join_predicate(
        &self,
        expr: &Expr,
        row: &HashMap<String, SochValue>,
        params: &[SochValue],
    ) -> Option<bool>;
}

/// A stored scope definition (from DEFINE SCOPE).
#[derive(Debug, Clone)]
pub struct ScopeDefinition {
    /// Scope name
    pub name: String,
    /// Session duration in seconds
    pub session_duration_secs: Option<u64>,
    /// SIGNIN expression (stored as AST)
    pub signin: Option<Box<Expr>>,
    /// SIGNUP expression (stored as AST)
    pub signup: Option<Box<Expr>>,
}

/// Stored per-table permission rules (from DEFINE TABLE ... PERMISSIONS).
#[derive(Debug, Clone)]
pub struct StoredTablePermissions {
    /// Table name
    pub table: String,
    /// Permission rules keyed by operation
    pub permissions: Vec<TablePermission>,
}

/// Unified SQL executor that routes through AST
pub struct SqlBridge<C: SqlConnection> {
    conn: C,
    /// Scope definitions (DEFINE SCOPE)
    scope_definitions: HashMap<String, ScopeDefinition>,
    /// Per-table permission rules (DEFINE TABLE ... PERMISSIONS)
    table_permissions: HashMap<String, StoredTablePermissions>,
}

impl<C: SqlConnection> SqlBridge<C> {
    /// Create a new SQL bridge with the given connection
    pub fn new(conn: C) -> Self {
        Self {
            conn,
            scope_definitions: HashMap::new(),
            table_permissions: HashMap::new(),
        }
    }

    /// Get a scope definition by name.
    pub fn get_scope(&self, name: &str) -> Option<&ScopeDefinition> {
        self.scope_definitions.get(name)
    }

    /// Get the table permission rules for a table.
    pub fn get_table_permissions(&self, table: &str) -> Option<&StoredTablePermissions> {
        self.table_permissions.get(table)
    }

    /// Check if the given operation is permitted on the table.
    /// If no permissions are defined for the table, all operations are allowed.
    /// Returns Ok(()) if allowed, Err if denied.
    pub fn check_table_permission(&self, table: &str, op: PermissionOp) -> SqlResult<()> {
        if let Some(perms) = self.table_permissions.get(table) {
            // Find the rule matching the operation
            let rule = perms.permissions.iter().find(|p| p.operation == op);
            match rule {
                Some(perm) => {
                    // Static evaluation: if condition is a literal `true` or `false`,
                    // we can decide immediately. More complex conditions involving
                    // $auth would need runtime context in a full implementation.
                    match &perm.condition {
                        Expr::Literal(Literal::Boolean(true)) => Ok(()),
                        Expr::Literal(Literal::Boolean(false)) => {
                            Err(SqlError::PermissionDenied(format!(
                                "{:?} denied on table '{}' by table permission rule",
                                op, table
                            )))
                        }
                        // For non-trivial expressions, allow by default.
                        // A full implementation would evaluate the expression
                        // against $auth context at runtime.
                        _ => Ok(()),
                    }
                }
                // No explicit rule for this operation — deny by default when
                // permissions are defined (secure-by-default).
                None => Err(SqlError::PermissionDenied(format!(
                    "{:?} not permitted on table '{}' (no matching permission rule)",
                    op, table
                ))),
            }
        } else {
            // No permissions defined — all operations allowed
            Ok(())
        }
    }

    /// Execute a SQL statement
    pub fn execute(&mut self, sql: &str) -> SqlResult<ExecutionResult> {
        self.execute_with_params(sql, &[])
    }

    /// Execute a SQL statement with parameters
    pub fn execute_with_params(
        &mut self,
        sql: &str,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        // Detect dialect for better error messages
        let _dialect = SqlDialect::detect(sql);

        // Parse SQL into AST
        let stmt = Parser::parse(sql).map_err(SqlError::from_parse_errors)?;

        // Validate placeholder count
        let max_placeholder = self.find_max_placeholder(&stmt);
        if max_placeholder as usize > params.len() {
            return Err(SqlError::InvalidArgument(format!(
                "Query contains {} placeholders but only {} parameters provided",
                max_placeholder,
                params.len()
            )));
        }

        // Execute statement
        self.execute_statement(&stmt, params)
    }

    /// Execute a parsed statement
    pub fn execute_statement(
        &mut self,
        stmt: &Statement,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        match stmt {
            Statement::Select(select) => self.execute_select(select, params),
            Statement::Insert(insert) => self.execute_insert(insert, params),
            Statement::Update(update) => self.execute_update(update, params),
            Statement::Delete(delete) => self.execute_delete(delete, params),
            Statement::CreateTable(create) => self.execute_create_table(create),
            Statement::DropTable(drop) => self.execute_drop_table(drop),
            Statement::CreateIndex(create) => self.execute_create_index(create),
            Statement::DropIndex(drop) => self.execute_drop_index(drop),
            Statement::AlterTable(alter) => self.execute_alter_table(alter),
            Statement::Begin(begin) => self.conn.begin(begin),
            Statement::Commit => self.conn.commit(),
            Statement::Rollback(savepoint) => self.conn.rollback(savepoint.as_deref()),
            Statement::Savepoint(_name) => Err(SqlError::NotImplemented(
                "SAVEPOINT not yet implemented".into(),
            )),
            Statement::Release(_name) => Err(SqlError::NotImplemented(
                "RELEASE SAVEPOINT not yet implemented".into(),
            )),
            Statement::Explain(_stmt) => Err(SqlError::NotImplemented(
                "EXPLAIN not yet implemented".into(),
            )),
            Statement::DefineScope(def) => {
                self.scope_definitions.insert(def.name.clone(), ScopeDefinition {
                    name: def.name.clone(),
                    session_duration_secs: def.session_duration_secs,
                    signin: def.signin.clone(),
                    signup: def.signup.clone(),
                });
                Ok(ExecutionResult::Ok)
            },
            Statement::DefineTablePermissions(def) => {
                let table_name = def.table.name().to_string();
                self.table_permissions.insert(table_name.clone(), StoredTablePermissions {
                    table: table_name,
                    permissions: def.permissions.clone(),
                });
                Ok(ExecutionResult::Ok)
            },
            Statement::RemoveScope(name) => {
                self.scope_definitions.remove(name);
                Ok(ExecutionResult::Ok)
            },
            Statement::Relate(_) => Err(SqlError::NotImplemented(
                "RELATE not yet implemented — graph execution engine required".into(),
            )),
            Statement::LiveSelect(_) => Err(SqlError::NotImplemented(
                "LIVE SELECT not yet implemented — CDC subscription engine required".into(),
            )),
            Statement::DefineEvent(_) => Err(SqlError::NotImplemented(
                "DEFINE EVENT not yet implemented — event trigger engine required".into(),
            )),
        }
    }

    fn execute_select(
        &self,
        select: &SelectStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        // Get table from FROM clause
        let from = select
            .from
            .as_ref()
            .ok_or_else(|| SqlError::InvalidArgument("SELECT requires FROM clause".into()))?;

        if from.tables.len() != 1 {
            return Err(SqlError::NotImplemented(
                "Multi-table queries (comma-separated) not yet supported".into(),
            ));
        }

        // Check if this is a JOIN query
        let table_ref = &from.tables[0];
        if self.contains_join(table_ref) {
            return self.execute_join_select(select, table_ref, params);
        }

        // Simple single-table SELECT  
        let table_name = match table_ref {
            TableRef::Table { name, .. } => name.name().to_string(),
            TableRef::Subquery { .. } => {
                return Err(SqlError::NotImplemented(
                    "Subqueries not yet supported".into(),
                ));
            }
            TableRef::Function { .. } => {
                return Err(SqlError::NotImplemented(
                    "Table functions not yet supported".into(),
                ));
            }
            TableRef::Join { .. } => unreachable!("handled above"),
        };

        // Check table-level permission for SELECT
        self.check_table_permission(&table_name, PermissionOp::Select)?;

        // Extract column names
        let columns = self.extract_select_columns(&select.columns)?;

        // Extract LIMIT/OFFSET
        let limit = self.extract_limit(&select.limit)?;
        let offset = self.extract_limit(&select.offset)?;

        self.conn.select(
            &table_name,
            &columns,
            select.where_clause.as_ref(),
            &select.order_by,
            limit,
            offset,
            params,
        )
    }

    /// Check if a table reference contains any JOIN.
    fn contains_join(&self, table_ref: &TableRef) -> bool {
        matches!(table_ref, TableRef::Join { .. })
    }

    /// Execute a SELECT query that involves JOINs.
    ///
    /// Strategy: resolve the join tree into a flat row set, then apply
    /// WHERE, ORDER BY, LIMIT, OFFSET, and column projection.
    fn execute_join_select(
        &self,
        select: &SelectStmt,
        table_ref: &TableRef,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        // Resolve the join tree into merged rows.
        // Each row is a HashMap<String, SochValue> with keys like "table.col"
        // for qualified access and "col" for unqualified access (when unambiguous).
        let mut rows = self.resolve_table_ref(table_ref, params)?;

        // Apply WHERE filter
        if let Some(ref expr) = select.where_clause {
            rows.retain(|row| {
                self.conn
                    .eval_join_predicate(expr, row, params)
                    .unwrap_or(false)
            });
        }

        // Apply ORDER BY
        if !select.order_by.is_empty() {
            rows.sort_by(|a, b| {
                for item in &select.order_by {
                    let col = Self::extract_order_column(&item.expr);
                    let va = a.get(&col);
                    let vb = b.get(&col);
                    let cmp = Self::compare_optional_values(va, vb);
                    let cmp = if !item.asc { cmp.reverse() } else { cmp };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // Apply OFFSET
        let offset = self.extract_limit(&select.offset)?;
        if let Some(off) = offset {
            rows = rows.into_iter().skip(off).collect();
        }

        // Apply LIMIT
        let limit = self.extract_limit(&select.limit)?;
        if let Some(lim) = limit {
            rows.truncate(lim);
        }

        // Project columns
        let select_columns = self.extract_select_columns(&select.columns)?;
        let (result_columns, projected_rows) =
            self.project_join_rows(&select_columns, &rows)?;

        Ok(ExecutionResult::Rows {
            columns: result_columns,
            rows: projected_rows,
        })
    }

    /// Recursively resolve a TableRef into a row set.
    ///
    /// - Simple table: scan all rows, prefix each key with "table."
    /// - Join: resolve left and right, then merge via nested-loop join
    fn resolve_table_ref(
        &self,
        table_ref: &TableRef,
        params: &[SochValue],
    ) -> SqlResult<Vec<HashMap<String, SochValue>>> {
        match table_ref {
            TableRef::Table { name, alias } => {
                let table_name = name.name().to_string();
                let prefix = alias.as_deref().unwrap_or(&table_name);
                let raw_rows = self.conn.scan_all(&table_name, &[])?;

                // Prefix each column with "table." and keep unqualified copy
                let mut result = Vec::with_capacity(raw_rows.len());
                for row in raw_rows {
                    let mut merged = HashMap::new();
                    for (k, v) in &row {
                        merged.insert(format!("{}.{}", prefix, k), v.clone());
                        // Also insert unqualified name (may be overwritten
                        // later if ambiguous in multi-table scenarios)
                        merged.insert(k.clone(), v.clone());
                    }
                    result.push(merged);
                }
                Ok(result)
            }
            TableRef::Join {
                left,
                join_type,
                right,
                condition,
            } => {
                let left_rows = self.resolve_table_ref(left, params)?;
                let right_rows = self.resolve_table_ref(right, params)?;
                self.execute_join(&left_rows, &right_rows, *join_type, condition.as_ref(), params)
            }
            TableRef::Subquery { .. } => Err(SqlError::NotImplemented(
                "Subqueries in FROM not yet supported".into(),
            )),
            TableRef::Function { .. } => Err(SqlError::NotImplemented(
                "Table functions not yet supported".into(),
            )),
        }
    }

    /// Execute a join between two resolved row sets.
    ///
    /// Uses nested-loop join with optional hash optimization for equi-joins.
    fn execute_join(
        &self,
        left_rows: &[HashMap<String, SochValue>],
        right_rows: &[HashMap<String, SochValue>],
        join_type: JoinType,
        condition: Option<&JoinCondition>,
        params: &[SochValue],
    ) -> SqlResult<Vec<HashMap<String, SochValue>>> {
        // Extract the ON expression or USING columns
        let (on_expr, using_cols) = match condition {
            Some(JoinCondition::On(expr)) => (Some(expr), None),
            Some(JoinCondition::Using(cols)) => (None, Some(cols.as_slice())),
            Some(JoinCondition::Natural) => {
                return Err(SqlError::NotImplemented(
                    "NATURAL JOIN not yet supported".into(),
                ))
            }
            None => (None, None), // CROSS JOIN — no condition
        };

        // Try hash join for simple equi-conditions
        if let Some(expr) = on_expr {
            if let Some((left_key, right_key)) = Self::extract_equi_join_keys(expr) {
                return self.hash_join(
                    left_rows, right_rows, &left_key, &right_key, join_type, params,
                );
            }
        }

        // Fall back to nested-loop join
        let mut result = Vec::new();
        let null_right: HashMap<String, SochValue> = Self::null_row(right_rows);
        let null_left: HashMap<String, SochValue> = Self::null_row(left_rows);

        let mut right_matched = vec![false; right_rows.len()];

        for left in left_rows {
            let mut found_match = false;

            for (ri, right) in right_rows.iter().enumerate() {
                let merged = Self::merge_rows(left, right);
                let matches = match (on_expr, using_cols) {
                    (Some(expr), _) => self
                        .conn
                        .eval_join_predicate(expr, &merged, params)
                        .unwrap_or(false),
                    (_, Some(cols)) => Self::using_matches(left, right, cols),
                    (None, None) => true, // CROSS JOIN
                };

                if matches {
                    result.push(merged);
                    found_match = true;
                    right_matched[ri] = true;
                }
            }

            // LEFT / FULL: emit left + NULLs if no match
            if !found_match
                && matches!(join_type, JoinType::Left | JoinType::Full)
            {
                result.push(Self::merge_rows(left, &null_right));
            }
        }

        // RIGHT / FULL: emit NULLs + right for unmatched right rows
        if matches!(join_type, JoinType::Right | JoinType::Full) {
            for (ri, right) in right_rows.iter().enumerate() {
                if !right_matched[ri] {
                    result.push(Self::merge_rows(&null_left, right));
                }
            }
        }

        Ok(result)
    }

    /// Hash join for equi-join conditions (O(n+m) instead of O(n*m)).
    fn hash_join(
        &self,
        left_rows: &[HashMap<String, SochValue>],
        right_rows: &[HashMap<String, SochValue>],
        left_key: &str,
        right_key: &str,
        join_type: JoinType,
        _params: &[SochValue],
    ) -> SqlResult<Vec<HashMap<String, SochValue>>> {
        // Build phase: index the smaller side (right) by join key
        let mut hash_table: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, row) in right_rows.iter().enumerate() {
            if let Some(val) = row.get(right_key) {
                let key = Self::value_to_hash_key(val);
                hash_table.entry(key).or_default().push(i);
            }
        }

        let null_right = Self::null_row(right_rows);
        let null_left = Self::null_row(left_rows);
        let mut right_matched = vec![false; right_rows.len()];
        let mut result = Vec::new();

        // Probe phase
        for left in left_rows {
            let mut found_match = false;
            if let Some(val) = left.get(left_key) {
                let key = Self::value_to_hash_key(val);
                if let Some(indices) = hash_table.get(&key) {
                    for &ri in indices {
                        result.push(Self::merge_rows(left, &right_rows[ri]));
                        found_match = true;
                        right_matched[ri] = true;
                    }
                }
            }
            if !found_match && matches!(join_type, JoinType::Left | JoinType::Full) {
                result.push(Self::merge_rows(left, &null_right));
            }
        }

        if matches!(join_type, JoinType::Right | JoinType::Full) {
            for (ri, right) in right_rows.iter().enumerate() {
                if !right_matched[ri] {
                    result.push(Self::merge_rows(&null_left, right));
                }
            }
        }

        Ok(result)
    }

    /// Extract equi-join keys from a simple `a.col = b.col` expression.
    /// Returns (left_key, right_key) as qualified column names.
    fn extract_equi_join_keys(expr: &Expr) -> Option<(String, String)> {
        if let Expr::BinaryOp { left, op, right } = expr {
            if *op == BinaryOperator::Eq {
                if let (Expr::Column(l), Expr::Column(r)) = (left.as_ref(), right.as_ref()) {
                    let lk = if let Some(ref t) = l.table {
                        format!("{}.{}", t, l.column)
                    } else {
                        l.column.clone()
                    };
                    let rk = if let Some(ref t) = r.table {
                        format!("{}.{}", t, r.column)
                    } else {
                        r.column.clone()
                    };
                    return Some((lk, rk));
                }
            }
        }
        None
    }

    /// Merge two rows into one (left keys + right keys).
    fn merge_rows(
        left: &HashMap<String, SochValue>,
        right: &HashMap<String, SochValue>,
    ) -> HashMap<String, SochValue> {
        let mut merged = left.clone();
        for (k, v) in right {
            // Don't overwrite left's unqualified columns with right's
            // (prefer left-side for ambiguous unqualified names)
            if !merged.contains_key(k) || k.contains('.') {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    /// Build a NULL-valued row with the same keys as the sample rows.
    fn null_row(rows: &[HashMap<String, SochValue>]) -> HashMap<String, SochValue> {
        if let Some(sample) = rows.first() {
            sample
                .keys()
                .map(|k| (k.clone(), SochValue::Null))
                .collect()
        } else {
            HashMap::new()
        }
    }

    /// Check USING condition: columns with same name must be equal.
    fn using_matches(
        left: &HashMap<String, SochValue>,
        right: &HashMap<String, SochValue>,
        cols: &[String],
    ) -> bool {
        cols.iter().all(|col| {
            let lv = left.get(col);
            let rv = right.get(col);
            match (lv, rv) {
                (Some(l), Some(r)) => l == r,
                _ => false,
            }
        })
    }

    /// Convert a value to a string hash key for hash join.
    fn value_to_hash_key(val: &SochValue) -> String {
        format!("{:?}", val)
    }

    /// Extract column name for ORDER BY (handles qualified and unqualified).
    fn extract_order_column(expr: &Expr) -> String {
        match expr {
            Expr::Column(col) => {
                if let Some(ref t) = col.table {
                    format!("{}.{}", t, col.column)
                } else {
                    col.column.clone()
                }
            }
            _ => String::new(),
        }
    }

    /// Compare two optional SochValues for ordering.
    fn compare_optional_values(
        a: Option<&SochValue>,
        b: Option<&SochValue>,
    ) -> std::cmp::Ordering {
        match (a, b) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(va), Some(vb)) => Self::compare_values(va, vb),
        }
    }

    /// Compare two SochValues for ordering.
    fn compare_values(a: &SochValue, b: &SochValue) -> std::cmp::Ordering {
        match (a, b) {
            (SochValue::Int(a), SochValue::Int(b)) => a.cmp(b),
            (SochValue::Float(a), SochValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
            (SochValue::Text(a), SochValue::Text(b)) => a.cmp(b),
            (SochValue::Bool(a), SochValue::Bool(b)) => a.cmp(b),
            (SochValue::Null, SochValue::Null) => std::cmp::Ordering::Equal,
            (SochValue::Null, _) => std::cmp::Ordering::Less,
            (_, SochValue::Null) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Project join result rows to the requested columns.
    fn project_join_rows(
        &self,
        select_columns: &[String],
        rows: &[HashMap<String, SochValue>],
    ) -> SqlResult<(Vec<String>, Vec<HashMap<String, SochValue>>)> {
        // If SELECT *, return all columns
        if select_columns.is_empty() || select_columns.iter().any(|c| c == "*") {
            let all_cols: Vec<String> = rows
                .first()
                .map(|r| {
                    // Return only qualified columns (containing '.') for clarity
                    let mut cols: Vec<String> = r
                        .keys()
                        .filter(|k| k.contains('.'))
                        .cloned()
                        .collect();
                    cols.sort();
                    if cols.is_empty() {
                        // Fallback: return all keys
                        cols = r.keys().cloned().collect();
                        cols.sort();
                    }
                    cols
                })
                .unwrap_or_default();

            let projected: Vec<HashMap<String, SochValue>> = rows
                .iter()
                .map(|row| {
                    all_cols
                        .iter()
                        .map(|c| {
                            let short = c.rsplit('.').next().unwrap_or(c);
                            (short.to_string(), row.get(c).cloned().unwrap_or(SochValue::Null))
                        })
                        .collect()
                })
                .collect();
            let short_cols: Vec<String> = all_cols
                .iter()
                .map(|c| c.rsplit('.').next().unwrap_or(c).to_string())
                .collect();
            return Ok((short_cols, projected));
        }

        // Specific columns requested — resolve each
        let mut result_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let mut projected = HashMap::new();
            for col in select_columns {
                // Try exact match first, then qualified variations
                let val = row
                    .get(col)
                    .or_else(|| {
                        // Try all qualified versions: "anything.col"
                        row.iter()
                            .find(|(k, _)| {
                                k.ends_with(&format!(".{}", col))
                                    || k.as_str() == col
                            })
                            .map(|(_, v)| v)
                    })
                    .cloned()
                    .unwrap_or(SochValue::Null);
                projected.insert(col.clone(), val);
            }
            result_rows.push(projected);
        }

        Ok((select_columns.to_vec(), result_rows))
    }

    fn execute_insert(
        &mut self,
        insert: &InsertStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = insert.table.name();

        // Check table-level permission for CREATE (INSERT maps to CREATE)
        self.check_table_permission(table_name, PermissionOp::Create)?;

        let rows = match &insert.source {
            InsertSource::Values(values) => values,
            InsertSource::Query(_) => {
                return Err(SqlError::NotImplemented(
                    "INSERT ... SELECT not yet supported".into(),
                ));
            }
            InsertSource::Default => {
                return Err(SqlError::NotImplemented(
                    "INSERT DEFAULT VALUES not yet supported".into(),
                ));
            }
        };

        self.conn.insert(
            table_name,
            insert.columns.as_deref(),
            rows,
            insert.on_conflict.as_ref(),
            params,
        )
    }

    fn execute_update(
        &mut self,
        update: &UpdateStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = update.table.name();

        // Check table-level permission for UPDATE
        self.check_table_permission(table_name, PermissionOp::Update)?;

        self.conn.update(
            table_name,
            &update.assignments,
            update.where_clause.as_ref(),
            params,
        )
    }

    fn execute_delete(
        &mut self,
        delete: &DeleteStmt,
        params: &[SochValue],
    ) -> SqlResult<ExecutionResult> {
        let table_name = delete.table.name();

        // Check table-level permission for DELETE
        self.check_table_permission(table_name, PermissionOp::Delete)?;

        self.conn.delete(
            table_name,
            delete.where_clause.as_ref(),
            params,
        )
    }

    fn execute_create_table(&mut self, stmt: &CreateTableStmt) -> SqlResult<ExecutionResult> {
        // Handle IF NOT EXISTS
        if stmt.if_not_exists {
            let table_name = stmt.name.name();
            if self.conn.table_exists(table_name)? {
                return Ok(ExecutionResult::Ok);
            }
        }

        self.conn.create_table(stmt)
    }

    fn execute_drop_table(&mut self, stmt: &DropTableStmt) -> SqlResult<ExecutionResult> {
        // Handle IF EXISTS
        if stmt.if_exists {
            for name in &stmt.names {
                if !self.conn.table_exists(name.name())? {
                    return Ok(ExecutionResult::Ok);
                }
            }
        }

        self.conn.drop_table(stmt)
    }

    fn execute_create_index(&mut self, stmt: &CreateIndexStmt) -> SqlResult<ExecutionResult> {
        // Handle IF NOT EXISTS
        if stmt.if_not_exists {
            if self.conn.index_exists(&stmt.name)? {
                return Ok(ExecutionResult::Ok);
            }
        }

        self.conn.create_index(stmt)
    }

    fn execute_drop_index(&mut self, stmt: &DropIndexStmt) -> SqlResult<ExecutionResult> {
        // Handle IF EXISTS
        if stmt.if_exists {
            if !self.conn.index_exists(&stmt.name)? {
                return Ok(ExecutionResult::Ok);
            }
        }

        self.conn.drop_index(stmt)
    }

    fn execute_alter_table(&mut self, stmt: &AlterTableStmt) -> SqlResult<ExecutionResult> {
        self.conn.alter_table(stmt)
    }

    /// Extract column names from SELECT list
    fn extract_select_columns(&self, items: &[SelectItem]) -> SqlResult<Vec<String>> {
        let mut columns = Vec::new();

        for item in items {
            match item {
                SelectItem::Wildcard => columns.push("*".to_string()),
                SelectItem::QualifiedWildcard(table) => columns.push(format!("{}.*", table)),
                SelectItem::Expr { expr, alias } => {
                    let name = alias.clone().unwrap_or_else(|| match expr {
                        Expr::Column(col) => col.column.clone(),
                        Expr::Function(func) => format!("{}()", func.name.name()),
                        _ => "?column?".to_string(),
                    });
                    columns.push(name);
                }
            }
        }

        Ok(columns)
    }

    /// Extract LIMIT/OFFSET value
    fn extract_limit(&self, expr: &Option<Expr>) -> SqlResult<Option<usize>> {
        match expr {
            Some(Expr::Literal(Literal::Integer(n))) => Ok(Some(*n as usize)),
            Some(_) => Err(SqlError::InvalidArgument(
                "LIMIT/OFFSET must be an integer literal".into(),
            )),
            None => Ok(None),
        }
    }

    /// Find the maximum placeholder index in a statement
    fn find_max_placeholder(&self, stmt: &Statement) -> u32 {
        let mut visitor = PlaceholderVisitor::new();
        visitor.visit_statement(stmt);
        visitor.max_placeholder
    }
}

/// Visitor to find maximum placeholder index
struct PlaceholderVisitor {
    max_placeholder: u32,
}

impl PlaceholderVisitor {
    fn new() -> Self {
        Self { max_placeholder: 0 }
    }

    fn visit_statement(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Select(s) => self.visit_select(s),
            Statement::Insert(i) => self.visit_insert(i),
            Statement::Update(u) => self.visit_update(u),
            Statement::Delete(d) => self.visit_delete(d),
            _ => {}
        }
    }

    fn visit_select(&mut self, select: &SelectStmt) {
        for item in &select.columns {
            if let SelectItem::Expr { expr, .. } = item {
                self.visit_expr(expr);
            }
        }
        if let Some(where_clause) = &select.where_clause {
            self.visit_expr(where_clause);
        }
        if let Some(having) = &select.having {
            self.visit_expr(having);
        }
        for order in &select.order_by {
            self.visit_expr(&order.expr);
        }
        if let Some(limit) = &select.limit {
            self.visit_expr(limit);
        }
        if let Some(offset) = &select.offset {
            self.visit_expr(offset);
        }
    }

    fn visit_insert(&mut self, insert: &InsertStmt) {
        if let InsertSource::Values(rows) = &insert.source {
            for row in rows {
                for expr in row {
                    self.visit_expr(expr);
                }
            }
        }
    }

    fn visit_update(&mut self, update: &UpdateStmt) {
        for assign in &update.assignments {
            self.visit_expr(&assign.value);
        }
        if let Some(where_clause) = &update.where_clause {
            self.visit_expr(where_clause);
        }
    }

    fn visit_delete(&mut self, delete: &DeleteStmt) {
        if let Some(where_clause) = &delete.where_clause {
            self.visit_expr(where_clause);
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Placeholder(n) => {
                self.max_placeholder = self.max_placeholder.max(*n);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            Expr::UnaryOp { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::Function(func) => {
                for arg in &func.args {
                    self.visit_expr(arg);
                }
            }
            Expr::Case { operand, conditions, else_result } => {
                if let Some(op) = operand {
                    self.visit_expr(op);
                }
                for (when, then) in conditions {
                    self.visit_expr(when);
                    self.visit_expr(then);
                }
                if let Some(else_expr) = else_result {
                    self.visit_expr(else_expr);
                }
            }
            Expr::InList { expr, list, .. } => {
                self.visit_expr(expr);
                for item in list {
                    self.visit_expr(item);
                }
            }
            Expr::Between { expr, low, high, .. } => {
                self.visit_expr(expr);
                self.visit_expr(low);
                self.visit_expr(high);
            }
            Expr::Cast { expr, .. } => {
                self.visit_expr(expr);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder_visitor() {
        let stmt = Parser::parse("SELECT * FROM users WHERE id = $1 AND name = $2").unwrap();
        let mut visitor = PlaceholderVisitor::new();
        visitor.visit_statement(&stmt);
        assert_eq!(visitor.max_placeholder, 2);
    }

    #[test]
    fn test_question_mark_placeholders() {
        let stmt = Parser::parse("SELECT * FROM users WHERE id = ? AND name = ?").unwrap();
        let mut visitor = PlaceholderVisitor::new();
        visitor.visit_statement(&stmt);
        assert_eq!(visitor.max_placeholder, 2);
    }

    #[test]
    fn test_dialect_detection() {
        assert_eq!(SqlDialect::detect("SELECT * FROM users"), SqlDialect::Standard);
        assert_eq!(
            SqlDialect::detect("INSERT IGNORE INTO users VALUES (1)"),
            SqlDialect::MySQL
        );
        assert_eq!(
            SqlDialect::detect("INSERT OR IGNORE INTO users VALUES (1)"),
            SqlDialect::SQLite
        );
    }

    #[test]
    fn test_define_scope_stores_definition() {
        use crate::sql::bridge::tests::make_mock_bridge;
        let mut bridge = make_mock_bridge();
        // Use integer duration (seconds) since lexer splits "24h" into separate tokens
        let result = bridge.execute("DEFINE SCOPE user_scope SESSION 86400");
        result.unwrap();
        let scope = bridge.get_scope("user_scope");
        assert!(scope.is_some(), "Scope not stored");
        let scope = scope.unwrap();
        assert_eq!(scope.name, "user_scope");
        assert_eq!(scope.session_duration_secs, Some(86400));
    }

    #[test]
    fn test_remove_scope_deletes_definition() {
        use crate::sql::bridge::tests::make_mock_bridge;
        let mut bridge = make_mock_bridge();
        bridge.execute("DEFINE SCOPE temp_scope SESSION 3600").unwrap();
        assert!(bridge.get_scope("temp_scope").is_some());
        bridge.execute("REMOVE SCOPE temp_scope").unwrap();
        assert!(bridge.get_scope("temp_scope").is_none());
    }

    #[test]
    fn test_define_table_permissions_stores_rules() {
        use crate::sql::bridge::tests::make_mock_bridge;
        let mut bridge = make_mock_bridge();
        let result = bridge.execute(
            "DEFINE TABLE posts PERMISSIONS FOR select WHERE true FOR delete WHERE false"
        );
        assert!(result.is_ok());
        let perms = bridge.get_table_permissions("posts");
        assert!(perms.is_some());
        assert_eq!(perms.unwrap().permissions.len(), 2);
    }

    #[test]
    fn test_table_permission_check_allows_matching_true() {
        use crate::sql::bridge::tests::make_mock_bridge;
        let mut bridge = make_mock_bridge();
        bridge.execute(
            "DEFINE TABLE docs PERMISSIONS FOR select WHERE true FOR insert WHERE true FOR update WHERE true FOR delete WHERE true"
        ).unwrap();
        assert!(bridge.check_table_permission("docs", PermissionOp::Select).is_ok());
        assert!(bridge.check_table_permission("docs", PermissionOp::Create).is_ok());
        assert!(bridge.check_table_permission("docs", PermissionOp::Update).is_ok());
        assert!(bridge.check_table_permission("docs", PermissionOp::Delete).is_ok());
    }

    #[test]
    fn test_table_permission_check_denies_matching_false() {
        use crate::sql::bridge::tests::make_mock_bridge;
        let mut bridge = make_mock_bridge();
        bridge.execute(
            "DEFINE TABLE secrets PERMISSIONS FOR select WHERE false FOR delete WHERE false"
        ).unwrap();
        let err = bridge.check_table_permission("secrets", PermissionOp::Select);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("Permission denied"));
    }

    #[test]
    fn test_table_permission_denies_undefined_op_when_rules_exist() {
        use crate::sql::bridge::tests::make_mock_bridge;
        let mut bridge = make_mock_bridge();
        // Only define SELECT — UPDATE should be denied since other rules exist
        bridge.execute(
            "DEFINE TABLE restricted PERMISSIONS FOR select WHERE true"
        ).unwrap();
        assert!(bridge.check_table_permission("restricted", PermissionOp::Select).is_ok());
        let err = bridge.check_table_permission("restricted", PermissionOp::Update);
        assert!(err.is_err());
    }

    #[test]
    fn test_no_permissions_allows_all() {
        use crate::sql::bridge::tests::make_mock_bridge;
        let bridge = make_mock_bridge();
        // No permissions defined = everything allowed
        assert!(bridge.check_table_permission("any_table", PermissionOp::Select).is_ok());
        assert!(bridge.check_table_permission("any_table", PermissionOp::Delete).is_ok());
    }

    // Helper: create a SqlBridge with a mock connection for permission tests
    fn make_mock_bridge() -> SqlBridge<MockPermConn> {
        SqlBridge::new(MockPermConn)
    }

    /// Minimal mock connection that just returns Ok for everything
    struct MockPermConn;

    impl SqlConnection for MockPermConn {
        fn select(&self, _: &str, _: &[String], _: Option<&Expr>, _: &[OrderByItem], _: Option<usize>, _: Option<usize>, _: &[SochValue]) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::Rows { columns: vec![], rows: vec![] })
        }
        fn insert(&mut self, _: &str, _: Option<&[String]>, _: &[Vec<Expr>], _: Option<&OnConflict>, _: &[SochValue]) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn update(&mut self, _: &str, _: &[Assignment], _: Option<&Expr>, _: &[SochValue]) -> SqlResult<ExecutionResult> {
            Ok(ExecutionResult::RowsAffected(0))
        }
        fn delete(&mut self, _: &str, _: Option<&Expr>, _: &[SochValue]) -> SqlResult<ExecutionResult> {
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
        fn table_exists(&self, _: &str) -> SqlResult<bool> { Ok(true) }
        fn index_exists(&self, _: &str) -> SqlResult<bool> { Ok(false) }
        fn scan_all(&self, _: &str, _: &[String]) -> SqlResult<Vec<HashMap<String, SochValue>>> { Ok(vec![]) }
        fn eval_join_predicate(&self, _: &Expr, _: &HashMap<String, SochValue>, _: &[SochValue]) -> Option<bool> { Some(true) }
    }
}
