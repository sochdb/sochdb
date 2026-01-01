// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! TOON-QL Query Executor
//!
//! SQL-like query interface with TOON output format.

use crate::connection::ToonConnection;
use crate::crud::{DeleteResult, InsertResult, UpdateResult};
use crate::error::{ClientError, Result};
use crate::schema::{CreateIndexResult, CreateTableResult, DropTableResult, SchemaBuilder};
use std::collections::HashMap;

use toondb_core::toon::{ToonType, ToonValue};

/// Query execution result
#[derive(Debug)]
pub enum QueryResult {
    /// SELECT result with rows
    Select(Vec<HashMap<String, ToonValue>>),
    /// INSERT result
    Insert(InsertResult),
    /// UPDATE result
    Update(UpdateResult),
    /// DELETE result
    Delete(DeleteResult),
    /// CREATE TABLE result
    CreateTable(CreateTableResult),
    /// DROP TABLE result
    DropTable(DropTableResult),
    /// CREATE INDEX result
    CreateIndex(CreateIndexResult),
    /// Empty result (e.g., SET, BEGIN, COMMIT)
    Empty,
}

/// Parsed query type
#[derive(Debug, Clone, PartialEq)]
enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
    CreateTable,
    DropTable,
    CreateIndex,
    DropIndex,
    Begin,
    Commit,
    Rollback,
}

/// TOON-QL query executor
pub struct QueryExecutor<'a> {
    conn: &'a ToonConnection,
}

impl<'a> QueryExecutor<'a> {
    /// Create new query executor
    pub fn new(conn: &'a ToonConnection) -> Self {
        Self { conn }
    }

    /// Execute a TOON-QL query
    pub fn execute(&self, sql: &str) -> Result<QueryResult> {
        let sql = sql.trim();
        let query_type = self.parse_query_type(sql)?;

        match query_type {
            QueryType::Select => self.execute_select(sql),
            QueryType::Insert => self.execute_insert(sql),
            QueryType::Update => self.execute_update(sql),
            QueryType::Delete => self.execute_delete(sql),
            QueryType::CreateTable => self.execute_create_table(sql),
            QueryType::DropTable => self.execute_drop_table(sql),
            QueryType::CreateIndex => self.execute_create_index(sql),
            QueryType::DropIndex => self.execute_drop_index(sql),
            QueryType::Begin => {
                // Transaction is handled at connection level
                Ok(QueryResult::Empty)
            }
            QueryType::Commit => Ok(QueryResult::Empty),
            QueryType::Rollback => Ok(QueryResult::Empty),
        }
    }

    /// Parse query type from SQL
    fn parse_query_type(&self, sql: &str) -> Result<QueryType> {
        let upper = sql.to_uppercase();
        let first_word = upper
            .split_whitespace()
            .next()
            .ok_or_else(|| ClientError::Parse("Empty query".to_string()))?;

        match first_word {
            "SELECT" => Ok(QueryType::Select),
            "INSERT" => Ok(QueryType::Insert),
            "UPDATE" => Ok(QueryType::Update),
            "DELETE" => Ok(QueryType::Delete),
            "CREATE" => {
                if upper.contains("TABLE") {
                    Ok(QueryType::CreateTable)
                } else if upper.contains("INDEX") {
                    Ok(QueryType::CreateIndex)
                } else {
                    Err(ClientError::Parse("Unknown CREATE statement".to_string()))
                }
            }
            "DROP" => {
                if upper.contains("TABLE") {
                    Ok(QueryType::DropTable)
                } else if upper.contains("INDEX") {
                    Ok(QueryType::DropIndex)
                } else {
                    Err(ClientError::Parse("Unknown DROP statement".to_string()))
                }
            }
            "BEGIN" => Ok(QueryType::Begin),
            "COMMIT" => Ok(QueryType::Commit),
            "ROLLBACK" => Ok(QueryType::Rollback),
            _ => Err(ClientError::Parse(format!(
                "Unknown query type: {}",
                first_word
            ))),
        }
    }

    /// Execute SELECT query
    fn execute_select(&self, sql: &str) -> Result<QueryResult> {
        // Parse: SELECT cols FROM table [WHERE ...] [ORDER BY ...] [LIMIT ...]
        let parsed = self.parse_select(sql)?;

        let mut builder = self.conn.find(&parsed.table);

        if !parsed.columns.is_empty() && parsed.columns[0] != "*" {
            builder = builder.select(
                &parsed
                    .columns
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            );
        }

        if let Some((field, op, value)) = &parsed.where_clause {
            use crate::crud::CompareOp;
            let compare_op = match op.as_str() {
                "=" => CompareOp::Eq,
                "!=" | "<>" => CompareOp::Ne,
                ">" => CompareOp::Gt,
                ">=" => CompareOp::Ge,
                "<" => CompareOp::Lt,
                "<=" => CompareOp::Le,
                _ => CompareOp::Eq,
            };
            builder = builder.where_cond(field, compare_op, value.clone());
        }

        if let Some((col, asc)) = &parsed.order_by {
            builder = builder.order_by(col, *asc);
        }

        if let Some(limit) = parsed.limit {
            builder = builder.limit(limit);
        }

        if let Some(offset) = parsed.offset {
            builder = builder.offset(offset);
        }

        let result = builder.execute()?;
        Ok(QueryResult::Select(result))
    }

    /// Execute INSERT query
    fn execute_insert(&self, sql: &str) -> Result<QueryResult> {
        // Parse: INSERT INTO table (cols) VALUES (vals)
        let parsed = self.parse_insert(sql)?;

        let mut builder = self.conn.insert_into(&parsed.table);
        for (col, val) in parsed.columns.iter().zip(parsed.values.iter()) {
            builder = builder.set(col, val.clone());
        }

        let result = builder.execute()?;
        Ok(QueryResult::Insert(result))
    }

    /// Execute UPDATE query
    fn execute_update(&self, sql: &str) -> Result<QueryResult> {
        // Parse: UPDATE table SET col=val [WHERE ...]
        let parsed = self.parse_update(sql)?;

        let mut builder = self.conn.update(&parsed.table);
        for (col, val) in &parsed.updates {
            builder = builder.set(col, val.clone());
        }

        if let Some((field, _, value)) = &parsed.where_clause {
            builder = builder.where_eq(field, value.clone());
        }

        let result = builder.execute()?;
        Ok(QueryResult::Update(result))
    }

    /// Execute DELETE query
    fn execute_delete(&self, sql: &str) -> Result<QueryResult> {
        // Parse: DELETE FROM table [WHERE ...]
        let parsed = self.parse_delete(sql)?;

        let mut builder = self.conn.delete_from(&parsed.table);

        if let Some((field, _, value)) = &parsed.where_clause {
            builder = builder.where_eq(field, value.clone());
        }

        let result = builder.execute()?;
        Ok(QueryResult::Delete(result))
    }

    /// Execute CREATE TABLE
    fn execute_create_table(&self, sql: &str) -> Result<QueryResult> {
        // Parse: CREATE TABLE name (col type, ...)
        let parsed = self.parse_create_table(sql)?;

        let mut schema = SchemaBuilder::table(&parsed.table);
        for (name, typ) in &parsed.columns {
            schema = schema.field(name, typ.clone()).not_null().builder;
        }

        if let Some(pk) = &parsed.primary_key {
            schema = schema.primary_key(pk);
        }

        let result = self.conn.create_table(schema.build())?;
        Ok(QueryResult::CreateTable(result))
    }

    /// Execute DROP TABLE
    fn execute_drop_table(&self, sql: &str) -> Result<QueryResult> {
        // Parse: DROP TABLE name
        let table = self.parse_drop_table(sql)?;
        let result = self.conn.drop_table(&table)?;
        Ok(QueryResult::DropTable(result))
    }

    /// Execute CREATE INDEX
    fn execute_create_index(&self, sql: &str) -> Result<QueryResult> {
        // Parse: CREATE [UNIQUE] INDEX name ON table (cols)
        let parsed = self.parse_create_index(sql)?;
        let cols: Vec<&str> = parsed.columns.iter().map(|s| s.as_str()).collect();
        let result = self
            .conn
            .create_index(&parsed.name, &parsed.table, &cols, parsed.unique)?;
        Ok(QueryResult::CreateIndex(result))
    }

    /// Execute DROP INDEX
    fn execute_drop_index(&self, sql: &str) -> Result<QueryResult> {
        // Parse: DROP INDEX name
        let name = self.parse_drop_index(sql)?;
        self.conn.drop_index(&name)?;
        Ok(QueryResult::Empty)
    }

    // ===== Parsing helpers =====

    fn parse_select(&self, sql: &str) -> Result<ParsedSelect> {
        // Simple parser for: SELECT cols FROM table [WHERE ...] [ORDER BY ...] [LIMIT ...]
        let upper = sql.to_uppercase();

        // Extract columns
        let select_end = upper
            .find("FROM")
            .ok_or_else(|| ClientError::Parse("Missing FROM".to_string()))?;
        let cols_str = sql[6..select_end].trim();
        let columns: Vec<String> = cols_str.split(',').map(|s| s.trim().to_string()).collect();

        // Extract table name - skip FROM and leading whitespace
        let from_start = select_end + 4;
        let remaining = sql[from_start..].trim_start();
        let table_end = remaining
            .find(char::is_whitespace)
            .unwrap_or(remaining.len());
        let table = remaining[..table_end].to_string();

        // Parse WHERE clause (simplified)
        let where_clause = if let Some(where_pos) = upper.find("WHERE") {
            let where_start = where_pos + 5;
            let where_str = &sql[where_start..];
            // Find end of where clause
            let end = where_str
                .to_uppercase()
                .find("ORDER")
                .or_else(|| where_str.to_uppercase().find("LIMIT"))
                .unwrap_or(where_str.len());
            let clause = where_str[..end].trim();

            // Parse field op value
            self.parse_where_clause(clause)?
        } else {
            None
        };

        // Parse ORDER BY
        let order_by = if let Some(order_pos) = upper.find("ORDER BY") {
            let order_start = order_pos + 8;
            let order_str = &sql[order_start..];
            let end = order_str.find("LIMIT").unwrap_or(order_str.len());
            let clause = order_str[..end].trim();

            let asc = !clause.to_uppercase().contains("DESC");
            let col = clause.split_whitespace().next().unwrap_or("").to_string();
            Some((col, asc))
        } else {
            None
        };

        // Parse LIMIT
        let limit = if let Some(limit_pos) = upper.find("LIMIT") {
            let limit_start = limit_pos + 5;
            let limit_str = &sql[limit_start..];
            let end = limit_str.find("OFFSET").unwrap_or(limit_str.len());
            let num_str = limit_str[..end].trim();
            num_str.parse().ok()
        } else {
            None
        };

        // Parse OFFSET
        let offset = if let Some(offset_pos) = upper.find("OFFSET") {
            let offset_start = offset_pos + 6;
            let offset_str = sql[offset_start..].trim();
            offset_str
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
        } else {
            None
        };

        Ok(ParsedSelect {
            columns,
            table,
            where_clause,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_where_clause(
        &self,
        clause: &str,
    ) -> Result<Option<(String, String, toondb_core::toon::ToonValue)>> {
        // Parse: field op value
        let ops = [">=", "<=", "!=", "<>", "=", ">", "<"];
        for op in ops {
            if let Some(pos) = clause.find(op) {
                let field = clause[..pos].trim().to_string();
                let value_str = clause[pos + op.len()..].trim();
                let value = self.parse_value(value_str);
                return Ok(Some((field, op.to_string(), value)));
            }
        }
        Ok(None)
    }

    fn parse_value(&self, s: &str) -> toondb_core::toon::ToonValue {
        let s = s.trim();

        // String literal
        if s.starts_with('\'') && s.ends_with('\'') {
            return ToonValue::Text(s[1..s.len() - 1].to_string());
        }
        if s.starts_with('"') && s.ends_with('"') {
            return ToonValue::Text(s[1..s.len() - 1].to_string());
        }

        // Boolean
        if s.eq_ignore_ascii_case("true") {
            return ToonValue::Bool(true);
        }
        if s.eq_ignore_ascii_case("false") {
            return ToonValue::Bool(false);
        }

        // Null
        if s.eq_ignore_ascii_case("null") {
            return ToonValue::Null;
        }

        // Number
        if let Ok(i) = s.parse::<i64>() {
            return ToonValue::Int(i);
        }
        if let Ok(f) = s.parse::<f64>() {
            return ToonValue::Float(f);
        }

        // Default to text
        ToonValue::Text(s.to_string())
    }

    fn parse_insert(&self, sql: &str) -> Result<ParsedInsert> {
        // Parse: INSERT INTO table (cols) VALUES (vals)
        let upper = sql.to_uppercase();

        // Find table name
        let into_pos = upper
            .find("INTO")
            .ok_or_else(|| ClientError::Parse("Missing INTO".to_string()))?
            + 4;
        let paren_pos = sql
            .find('(')
            .ok_or_else(|| ClientError::Parse("Missing columns".to_string()))?;
        let table = sql[into_pos..paren_pos].trim().to_string();

        // Find columns
        let col_end = sql
            .find(')')
            .ok_or_else(|| ClientError::Parse("Missing )".to_string()))?;
        let cols_str = &sql[paren_pos + 1..col_end];
        let columns: Vec<String> = cols_str.split(',').map(|s| s.trim().to_string()).collect();

        // Find VALUES
        let values_pos = upper
            .find("VALUES")
            .ok_or_else(|| ClientError::Parse("Missing VALUES".to_string()))?
            + 6;
        let val_start = sql[values_pos..]
            .find('(')
            .ok_or_else(|| ClientError::Parse("Missing values".to_string()))?
            + values_pos
            + 1;
        let val_end = sql[val_start..].find(')').unwrap_or(sql.len() - val_start) + val_start;
        let vals_str = &sql[val_start..val_end];
        let values: Vec<ToonValue> = vals_str
            .split(',')
            .map(|s| self.parse_value(s.trim()))
            .collect();

        Ok(ParsedInsert {
            table,
            columns,
            values,
        })
    }

    fn parse_update(&self, sql: &str) -> Result<ParsedUpdate> {
        // Parse: UPDATE table SET col=val, ... [WHERE ...]
        let upper = sql.to_uppercase();

        // Find table name
        let update_end = 6; // "UPDATE"
        let set_pos = upper
            .find("SET")
            .ok_or_else(|| ClientError::Parse("Missing SET".to_string()))?;
        let table = sql[update_end..set_pos].trim().to_string();

        // Find SET clause
        let set_start = set_pos + 3;
        let where_pos = upper.find("WHERE").unwrap_or(sql.len());
        let set_str = &sql[set_start..where_pos];

        let updates: Vec<(String, ToonValue)> = set_str
            .split(',')
            .filter_map(|s| {
                let parts: Vec<&str> = s.split('=').collect();
                if parts.len() == 2 {
                    Some((parts[0].trim().to_string(), self.parse_value(parts[1])))
                } else {
                    None
                }
            })
            .collect();

        // Parse WHERE
        let where_clause = if where_pos < sql.len() {
            let where_str = &sql[where_pos + 5..];
            self.parse_where_clause(where_str)?
        } else {
            None
        };

        Ok(ParsedUpdate {
            table,
            updates,
            where_clause,
        })
    }

    fn parse_delete(&self, sql: &str) -> Result<ParsedDelete> {
        // Parse: DELETE FROM table [WHERE ...]
        let upper = sql.to_uppercase();

        // Find table name
        let from_pos = upper
            .find("FROM")
            .ok_or_else(|| ClientError::Parse("Missing FROM".to_string()))?
            + 4;
        let where_pos = upper.find("WHERE").unwrap_or(sql.len());
        let table = sql[from_pos..where_pos].trim().to_string();

        // Parse WHERE
        let where_clause = if where_pos < sql.len() {
            let where_str = &sql[where_pos + 5..];
            self.parse_where_clause(where_str)?
        } else {
            None
        };

        Ok(ParsedDelete {
            table,
            where_clause,
        })
    }

    fn parse_create_table(&self, sql: &str) -> Result<ParsedCreateTable> {
        // Parse: CREATE TABLE name (col type, ...)
        let upper = sql.to_uppercase();

        // Find table name
        let table_pos = upper
            .find("TABLE")
            .ok_or_else(|| ClientError::Parse("Missing TABLE".to_string()))?
            + 5;
        let paren_pos = sql
            .find('(')
            .ok_or_else(|| ClientError::Parse("Missing columns".to_string()))?;
        let table = sql[table_pos..paren_pos].trim().to_string();

        // Find columns
        let col_start = paren_pos + 1;
        let col_end = sql
            .rfind(')')
            .ok_or_else(|| ClientError::Parse("Missing )".to_string()))?;
        let cols_str = &sql[col_start..col_end];

        let mut columns = Vec::new();
        let mut primary_key = None;

        for col_def in cols_str.split(',') {
            let col_def = col_def.trim();
            if col_def.to_uppercase().starts_with("PRIMARY KEY") {
                // Extract PK column
                if let Some(start) = col_def.find('(')
                    && let Some(end) = col_def.find(')')
                {
                    primary_key = Some(col_def[start + 1..end].trim().to_string());
                }
                continue;
            }

            let parts: Vec<&str> = col_def.split_whitespace().collect();
            if parts.len() >= 2 {
                let name = parts[0].to_string();
                let typ = self.parse_type(parts[1]);
                columns.push((name, typ));
            }
        }

        Ok(ParsedCreateTable {
            table,
            columns,
            primary_key,
        })
    }

    fn parse_type(&self, s: &str) -> ToonType {
        match s.to_uppercase().as_str() {
            "INT" | "INTEGER" | "BIGINT" => ToonType::Int,
            "UINT" | "UNSIGNED" => ToonType::UInt,
            "FLOAT" | "DOUBLE" | "REAL" | "DECIMAL" => ToonType::Float,
            "TEXT" | "VARCHAR" | "STRING" | "CHAR" => ToonType::Text,
            "BOOL" | "BOOLEAN" => ToonType::Bool,
            "BLOB" | "BYTES" | "BINARY" => ToonType::Binary,
            _ => ToonType::Text,
        }
    }

    fn parse_drop_table(&self, sql: &str) -> Result<String> {
        let upper = sql.to_uppercase();
        let table_pos = upper
            .find("TABLE")
            .ok_or_else(|| ClientError::Parse("Missing TABLE".to_string()))?
            + 5;
        Ok(sql[table_pos..].trim().to_string())
    }

    fn parse_create_index(&self, sql: &str) -> Result<ParsedCreateIndex> {
        // Parse: CREATE [UNIQUE] INDEX name ON table (cols)
        let upper = sql.to_uppercase();
        let unique = upper.contains("UNIQUE");

        // Find index name
        let index_pos = upper
            .find("INDEX")
            .ok_or_else(|| ClientError::Parse("Missing INDEX".to_string()))?
            + 5;
        let on_pos = upper
            .find("ON")
            .ok_or_else(|| ClientError::Parse("Missing ON".to_string()))?;
        let name = sql[index_pos..on_pos].trim().to_string();

        // Find table name
        let table_start = on_pos + 2;
        let paren_pos = sql
            .find('(')
            .ok_or_else(|| ClientError::Parse("Missing columns".to_string()))?;
        let table = sql[table_start..paren_pos].trim().to_string();

        // Find columns
        let col_start = paren_pos + 1;
        let col_end = sql
            .rfind(')')
            .ok_or_else(|| ClientError::Parse("Missing )".to_string()))?;
        let columns: Vec<String> = sql[col_start..col_end]
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        Ok(ParsedCreateIndex {
            name,
            table,
            columns,
            unique,
        })
    }

    fn parse_drop_index(&self, sql: &str) -> Result<String> {
        let upper = sql.to_uppercase();
        let index_pos = upper
            .find("INDEX")
            .ok_or_else(|| ClientError::Parse("Missing INDEX".to_string()))?
            + 5;
        Ok(sql[index_pos..].trim().to_string())
    }
}

// Parsed query structures
struct ParsedSelect {
    columns: Vec<String>,
    table: String,
    where_clause: Option<(String, String, toondb_core::toon::ToonValue)>,
    order_by: Option<(String, bool)>,
    limit: Option<usize>,
    offset: Option<usize>,
}

struct ParsedInsert {
    table: String,
    columns: Vec<String>,
    values: Vec<toondb_core::toon::ToonValue>,
}

struct ParsedUpdate {
    table: String,
    updates: Vec<(String, toondb_core::toon::ToonValue)>,
    where_clause: Option<(String, String, toondb_core::toon::ToonValue)>,
}

struct ParsedDelete {
    table: String,
    where_clause: Option<(String, String, toondb_core::toon::ToonValue)>,
}

struct ParsedCreateTable {
    table: String,
    columns: Vec<(String, ToonType)>,
    primary_key: Option<String>,
}

struct ParsedCreateIndex {
    name: String,
    table: String,
    columns: Vec<String>,
    unique: bool,
}

/// SQL query methods on connection
impl ToonConnection {
    /// Execute TOON-QL query
    pub fn query_sql(&self, sql: &str) -> Result<QueryResult> {
        QueryExecutor::new(self).execute(sql)
    }

    /// Execute and get rows (for SELECT)
    pub fn query_rows(&self, sql: &str) -> Result<Vec<HashMap<String, ToonValue>>> {
        match self.query_sql(sql)? {
            QueryResult::Select(result) => Ok(result),
            _ => Err(ClientError::Parse("Expected SELECT query".to_string())),
        }
    }

    /// Execute non-query SQL
    pub fn execute_sql(&self, sql: &str) -> Result<u64> {
        match self.query_sql(sql)? {
            QueryResult::Insert(r) => Ok(r.rows_inserted as u64),
            QueryResult::Update(r) => Ok(r.rows_updated as u64),
            QueryResult::Delete(r) => Ok(r.rows_deleted as u64),
            QueryResult::CreateTable(_) => Ok(0),
            QueryResult::DropTable(_) => Ok(0),
            QueryResult::CreateIndex(_) => Ok(0),
            QueryResult::Empty => Ok(0),
            QueryResult::Select(_) => Err(ClientError::Parse(
                "Use query_rows() for SELECT".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_type_parsing() {
        let conn = ToonConnection::open("./test").unwrap();
        let executor = QueryExecutor::new(&conn);

        assert_eq!(
            executor.parse_query_type("SELECT * FROM users").unwrap(),
            QueryType::Select
        );
        assert_eq!(
            executor.parse_query_type("INSERT INTO users").unwrap(),
            QueryType::Insert
        );
        assert_eq!(
            executor.parse_query_type("UPDATE users SET").unwrap(),
            QueryType::Update
        );
        assert_eq!(
            executor.parse_query_type("DELETE FROM users").unwrap(),
            QueryType::Delete
        );
        assert_eq!(
            executor.parse_query_type("CREATE TABLE foo").unwrap(),
            QueryType::CreateTable
        );
        assert_eq!(
            executor.parse_query_type("DROP TABLE foo").unwrap(),
            QueryType::DropTable
        );
    }

    #[test]
    fn test_select_parsing() {
        let conn = ToonConnection::open("./test").unwrap();
        let executor = QueryExecutor::new(&conn);

        let parsed = executor
            .parse_select("SELECT id, name FROM users WHERE active = true ORDER BY name LIMIT 10")
            .unwrap();
        assert_eq!(parsed.table, "users");
        assert_eq!(parsed.columns, vec!["id", "name"]);
        assert!(parsed.where_clause.is_some());
        assert!(parsed.order_by.is_some());
        assert_eq!(parsed.limit, Some(10));
    }

    #[test]
    fn test_insert_parsing() {
        let conn = ToonConnection::open("./test").unwrap();
        let executor = QueryExecutor::new(&conn);

        let parsed = executor
            .parse_insert("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        assert_eq!(parsed.table, "users");
        assert_eq!(parsed.columns, vec!["id", "name"]);
        assert_eq!(parsed.values.len(), 2);
    }

    #[test]
    fn test_create_table_parsing() {
        let conn = ToonConnection::open("./test").unwrap();
        let executor = QueryExecutor::new(&conn);

        let parsed = executor
            .parse_create_table("CREATE TABLE users (id INT, name TEXT, PRIMARY KEY (id))")
            .unwrap();
        assert_eq!(parsed.table, "users");
        assert_eq!(parsed.columns.len(), 2);
        assert_eq!(parsed.primary_key, Some("id".to_string()));
    }

    #[test]
    fn test_value_parsing() {
        let conn = ToonConnection::open("./test").unwrap();
        let executor = QueryExecutor::new(&conn);

        use toondb_core::toon::ToonValue;

        match executor.parse_value("123") {
            ToonValue::Int(i) => assert_eq!(i, 123),
            _ => panic!("Expected Int"),
        }

        match executor.parse_value("'hello'") {
            ToonValue::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("Expected Text"),
        }

        match executor.parse_value("true") {
            ToonValue::Bool(b) => assert!(b),
            _ => panic!("Expected Bool"),
        }
    }
}
