// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified execution pipeline: SQL text → result.
//!
//! This is the top-level entry point that replaces the disconnected
//! three-way split between `SqlBridge`, `SqlExecutor`, and `SochQlExecutor`.
//!
//! ```text
//! SQL Text → Parser → AST → Planner → Volcano Operator Tree → ExecutionResult
//! ```

use crate::optimizer_integration::StorageBackend;
use crate::soch_ql::SochValue;
use crate::storage_bridge::convert_query_to_core;
use crate::sql::bridge::ExecutionResult;
use crate::sql::ast::*;
use crate::sql::error::{SqlError, SqlResult};
use crate::sql::parser::Parser;
use super::explain::ExplainNode;
use super::node::PlanNode;
use super::planner::{QueryPlanner, explain_select};
use std::collections::HashMap;
use std::sync::Arc;

/// Executor configuration.
pub struct ExecutorConfig {
    /// Maximum rows to return (safety limit).
    pub max_rows: usize,
    /// Enable EXPLAIN output.
    pub explain_mode: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            explain_mode: false,
        }
    }
}

/// Execute a SQL string against a storage backend, returning the result.
///
/// This is the unified entry point for all SQL execution:
///
/// ```text
/// let result = execute_sql("SELECT * FROM users WHERE age > 21", &storage)?;
/// ```
pub fn execute_sql(
    sql: &str,
    storage: &Arc<dyn StorageBackend>,
) -> SqlResult<ExecutionResult> {
    let stmt = Parser::parse(sql).map_err(SqlError::from_parse_errors)?;
    execute_statement(&stmt, storage)
}

/// Execute a parsed SQL statement against a storage backend.
pub fn execute_statement(
    stmt: &Statement,
    storage: &Arc<dyn StorageBackend>,
) -> SqlResult<ExecutionResult> {
    match stmt {
        Statement::Select(select) => execute_select(select, storage),

        Statement::Explain(inner) => {
            match inner.as_ref() {
                Statement::Select(select) => {
                    let plan_text = explain_select(select, storage);
                    let mut node = ExplainNode::new(plan_text);
                    collect_rows_from_node(&mut node)
                }
                _ => Err(SqlError::NotImplemented(
                    "EXPLAIN only supported for SELECT statements".into(),
                )),
            }
        }

        // DML / DDL — these still need to go through the SqlConnection/storage bridge
        // The Volcano executor handles SELECT; mutations go through the existing path.
        Statement::Insert(_)
        | Statement::Update(_)
        | Statement::Delete(_)
        | Statement::CreateTable(_)
        | Statement::DropTable(_)
        | Statement::CreateIndex(_)
        | Statement::DropIndex(_)
        | Statement::AlterTable(_)
        | Statement::Begin(_)
        | Statement::Commit
        | Statement::Rollback(_)
        | Statement::Savepoint(_)
        | Statement::Release(_)
        | Statement::DefineScope(_)
        | Statement::DefineTablePermissions(_)
        | Statement::RemoveScope(_)
        | Statement::Relate(_)
        | Statement::LiveSelect(_)
        | Statement::DefineEvent(_) => Err(SqlError::NotImplemented(
            "DML/DDL statements should be routed through SqlBridge".into(),
        )),
    }
}

/// Execute a SELECT statement and collect results.
fn execute_select(
    select: &SelectStmt,
    storage: &Arc<dyn StorageBackend>,
) -> SqlResult<ExecutionResult> {
    let planner = QueryPlanner::new(storage.clone());
    let mut node = planner
        .plan_select(select)
        .map_err(|e| SqlError::ExecutionError(e.to_string()))?;

    collect_rows_from_node(&mut node)
}

/// Collect all rows from a PlanNode into an ExecutionResult.
fn collect_rows_from_node(node: &mut dyn PlanNode) -> SqlResult<ExecutionResult> {
    let schema = node.schema().clone();
    let columns = schema.column_names();

    let mut rows: Vec<HashMap<String, sochdb_core::SochValue>> = Vec::new();
    loop {
        match node.next() {
            Ok(Some(row)) => {
                let mut row_map = HashMap::new();
                for (i, val) in row.into_iter().enumerate() {
                    let col_name = columns
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("col{}", i));
                    row_map.insert(col_name, convert_query_to_core(&val));
                }
                rows.push(row_map);
            }
            Ok(None) => break,
            Err(e) => return Err(SqlError::ExecutionError(e.to_string())),
        }
    }

    Ok(ExecutionResult::Rows { columns, rows })
}

// ============================================================================
// Convenience: execute SQL and get results as vectors
// ============================================================================

/// Execute SQL and return rows as Vec<Vec<SochValue>> (positional).
pub fn execute_sql_rows(
    sql: &str,
    storage: &Arc<dyn StorageBackend>,
) -> SqlResult<(Vec<String>, Vec<Vec<SochValue>>)> {
    let result = execute_sql(sql, storage)?;
    match result {
        ExecutionResult::Rows { columns, rows } => {
            let typed_rows: Vec<Vec<SochValue>> = rows
                .into_iter()
                .map(|row_map| {
                    columns
                        .iter()
                        .map(|col| {
                            row_map
                                .get(col)
                                .map(|v| crate::storage_bridge::convert_core_to_query(v))
                                .unwrap_or(SochValue::Null)
                        })
                        .collect()
                })
                .collect();
            Ok((columns, typed_rows))
        }
        _ => Ok((vec![], vec![])),
    }
}
