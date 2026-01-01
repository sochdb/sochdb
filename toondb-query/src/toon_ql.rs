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

//! TOON Query Language (TOON-QL)
//!
//! SQL-like query language for TOON-native data.
//!
//! ## Query Syntax
//!
//! ```text
//! -- Schema Definition (DDL)
//! CREATE TABLE users {
//!   id: u64 PRIMARY KEY,
//!   name: text NOT NULL,
//!   email: text UNIQUE,
//!   score: f64 DEFAULT 0.0
//! }
//!
//! -- Data Manipulation (DML) - TOON in, TOON out
//! INSERT users:
//! id: 1
//! name: Alice
//! email: alice@example.com
//!
//! -- Queries return TOON
//! SELECT id,name FROM users WHERE score > 80
//! → users[2]{id,name}:
//!   1,Alice
//!   3,Charlie
//! ```

/// A parsed TOON-QL query
#[derive(Debug, Clone)]
pub enum ToonQuery {
    /// SELECT query
    Select(SelectQuery),
    /// INSERT query  
    Insert(InsertQuery),
    /// CREATE TABLE query
    CreateTable(CreateTableQuery),
    /// DROP TABLE query
    DropTable { table: String },
}

/// SELECT query
#[derive(Debug, Clone)]
pub struct SelectQuery {
    /// Columns to select (* means all)
    pub columns: Vec<String>,
    /// Table to query
    pub table: String,
    /// WHERE clause conditions
    pub where_clause: Option<WhereClause>,
    /// ORDER BY clause
    pub order_by: Option<OrderBy>,
    /// LIMIT clause
    pub limit: Option<usize>,
    /// OFFSET clause
    pub offset: Option<usize>,
}

/// INSERT query
#[derive(Debug, Clone)]
pub struct InsertQuery {
    /// Target table
    pub table: String,
    /// Columns to insert
    pub columns: Vec<String>,
    /// Rows of values
    pub rows: Vec<Vec<ToonValue>>,
}

/// CREATE TABLE query
#[derive(Debug, Clone)]
pub struct CreateTableQuery {
    /// Table name
    pub table: String,
    /// Column definitions
    pub columns: Vec<ColumnDef>,
    /// Primary key column
    pub primary_key: Option<String>,
}

/// Column definition for CREATE TABLE
#[derive(Debug, Clone)]
pub struct ColumnDef {
    /// Column name
    pub name: String,
    /// Column type
    pub col_type: ColumnType,
    /// NOT NULL constraint
    pub not_null: bool,
    /// UNIQUE constraint
    pub unique: bool,
    /// Default value
    pub default: Option<ToonValue>,
}

/// Column type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Bool,
    Int64,
    UInt64,
    Float64,
    Text,
    Binary,
    Timestamp,
}

/// A value in TOON format
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ToonValue {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
    Binary(Vec<u8>),
    Array(Vec<ToonValue>),
}

impl ToonValue {
    /// Format as TOON string
    pub fn to_toon_string(&self) -> String {
        match self {
            ToonValue::Null => "null".to_string(),
            ToonValue::Bool(b) => b.to_string(),
            ToonValue::Int(i) => i.to_string(),
            ToonValue::UInt(u) => u.to_string(),
            ToonValue::Float(f) => format!("{:.2}", f),
            ToonValue::Text(s) => s.clone(),
            ToonValue::Binary(b) => {
                let hex_str: String = b.iter().map(|byte| format!("{:02x}", byte)).collect();
                format!("0x{}", hex_str)
            }
            ToonValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_toon_string()).collect();
                format!("[{}]", items.join(","))
            }
        }
    }
}

/// WHERE clause
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WhereClause {
    pub conditions: Vec<Condition>,
    pub operator: LogicalOp,
}

/// Logical operator
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum LogicalOp {
    And,
    Or,
}

/// A condition in WHERE clause
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Condition {
    pub column: String,
    pub operator: ComparisonOp,
    pub value: ToonValue,
}

/// Comparison operator
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Like,
    In,
}

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderBy {
    pub column: String,
    pub direction: SortDirection,
}

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Query result in TOON format
#[derive(Debug, Clone)]
pub struct ToonResult {
    /// Table name
    pub table: String,
    /// Column names
    pub columns: Vec<String>,
    /// Rows of values
    pub rows: Vec<Vec<ToonValue>>,
}

impl ToonResult {
    /// Format as TOON string
    ///
    /// Example output:
    /// ```text
    /// users[2]{id,name}:
    /// 1,Alice
    /// 2,Bob
    /// ```
    pub fn to_toon_string(&self) -> String {
        let mut result = String::new();

        // Header: table[row_count]{columns}:
        result.push_str(&format!(
            "{}[{}]{{{}}}:\n",
            self.table,
            self.rows.len(),
            self.columns.join(",")
        ));

        // Data rows
        for row in &self.rows {
            let values: Vec<String> = row.iter().map(|v| v.to_toon_string()).collect();
            result.push_str(&values.join(","));
            result.push('\n');
        }

        result
    }

    /// Number of rows
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Get a value by row and column index
    pub fn get(&self, row: usize, col: usize) -> Option<&ToonValue> {
        self.rows.get(row)?.get(col)
    }
}

/// Simple TOON-QL parser
pub struct ToonQlParser;

impl ToonQlParser {
    /// Parse a TOON-QL query string
    pub fn parse(query: &str) -> Result<ToonQuery, ParseError> {
        let query = query.trim();

        if query.to_uppercase().starts_with("SELECT") {
            Self::parse_select(query)
        } else if query.to_uppercase().starts_with("INSERT") {
            Self::parse_insert(query)
        } else if query.to_uppercase().starts_with("CREATE TABLE") {
            Self::parse_create_table(query)
        } else if query.to_uppercase().starts_with("DROP TABLE") {
            Self::parse_drop_table(query)
        } else {
            Err(ParseError::UnknownStatement)
        }
    }

    fn parse_select(query: &str) -> Result<ToonQuery, ParseError> {
        // Simple SELECT parser
        // SELECT col1, col2 FROM table WHERE condition ORDER BY col LIMIT n
        let query_upper = query.to_uppercase();

        // Extract columns
        let from_idx = query_upper.find("FROM").ok_or(ParseError::MissingFrom)?;
        let columns_str = &query[6..from_idx].trim();
        let columns: Vec<String> = if columns_str == &"*" {
            vec!["*".to_string()]
        } else {
            columns_str
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        };

        // Extract table name
        let after_from = &query[from_idx + 4..].trim();
        let table_end = after_from
            .find(|c: char| c.is_whitespace())
            .unwrap_or(after_from.len());
        let table = after_from[..table_end].to_string();

        // Parse WHERE clause
        let where_clause = Self::parse_where_clause(&query_upper, query)?;

        let order_by = None;
        let limit = None;
        let offset = None;

        Ok(ToonQuery::Select(SelectQuery {
            columns,
            table,
            where_clause,
            order_by,
            limit,
            offset,
        }))
    }

    /// Parse WHERE clause from query
    fn parse_where_clause(
        query_upper: &str,
        original: &str,
    ) -> Result<Option<WhereClause>, ParseError> {
        let where_idx = match query_upper.find("WHERE") {
            Some(idx) => idx,
            None => return Ok(None),
        };

        // Find the end of WHERE clause (ORDER BY, LIMIT, or end of string)
        let after_where = &original[where_idx + 5..].trim();
        let clause_end = after_where
            .to_uppercase()
            .find("ORDER BY")
            .or_else(|| after_where.to_uppercase().find("LIMIT"))
            .unwrap_or(after_where.len());

        let condition_str = after_where[..clause_end].trim();

        // Parse condition: column op value
        // Supported operators: =, !=, <, <=, >, >=, LIKE, IN
        let (column, operator, value) = Self::parse_condition(condition_str)?;

        Ok(Some(WhereClause {
            conditions: vec![Condition {
                column,
                operator,
                value,
            }],
            operator: LogicalOp::And, // Default to AND for single condition
        }))
    }

    /// Parse a single condition: field op value
    fn parse_condition(condition: &str) -> Result<(String, ComparisonOp, ToonValue), ParseError> {
        let condition_upper = condition.to_uppercase();

        // Check for IN operator first (contains space)
        if let Some(in_idx) = condition_upper.find(" IN ") {
            let field = condition[..in_idx].trim().to_string();
            let values_str = condition[in_idx + 4..].trim();
            // Parse (val1, val2, val3)
            let values = Self::parse_in_values(values_str)?;
            return Ok((field, ComparisonOp::In, values));
        }

        // Check for LIKE operator
        if let Some(like_idx) = condition_upper.find(" LIKE ") {
            let field = condition[..like_idx].trim().to_string();
            let pattern = condition[like_idx + 6..].trim();
            let value = Self::parse_value(pattern)?;
            return Ok((field, ComparisonOp::Like, value));
        }

        // Check comparison operators (in order of length)
        let operators = [
            ("!=", ComparisonOp::Ne),
            ("<=", ComparisonOp::Le),
            (">=", ComparisonOp::Ge),
            ("<>", ComparisonOp::Ne),
            ("=", ComparisonOp::Eq),
            ("<", ComparisonOp::Lt),
            (">", ComparisonOp::Gt),
        ];

        for (op_str, op) in operators {
            if let Some(op_idx) = condition.find(op_str) {
                let field = condition[..op_idx].trim().to_string();
                let value_str = condition[op_idx + op_str.len()..].trim();
                let value = Self::parse_value(value_str)?;
                return Ok((field, op, value));
            }
        }

        Err(ParseError::InvalidSyntax)
    }

    /// Parse IN clause values: (val1, val2, val3)
    fn parse_in_values(values_str: &str) -> Result<ToonValue, ParseError> {
        let trimmed = values_str.trim();
        if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
            return Err(ParseError::InvalidSyntax);
        }

        let inner = &trimmed[1..trimmed.len() - 1];
        let values: Result<Vec<ToonValue>, ParseError> = inner
            .split(',')
            .map(|v| Self::parse_value(v.trim()))
            .collect();

        // Return as an array ToonValue
        Ok(ToonValue::Array(values?))
    }

    /// Parse a single value
    fn parse_value(value_str: &str) -> Result<ToonValue, ParseError> {
        let trimmed = value_str.trim();

        // String literal
        if (trimmed.starts_with('\'') && trimmed.ends_with('\''))
            || (trimmed.starts_with('"') && trimmed.ends_with('"'))
        {
            let inner = &trimmed[1..trimmed.len() - 1];
            return Ok(ToonValue::Text(inner.to_string()));
        }

        // Boolean
        if trimmed.eq_ignore_ascii_case("true") {
            return Ok(ToonValue::Bool(true));
        }
        if trimmed.eq_ignore_ascii_case("false") {
            return Ok(ToonValue::Bool(false));
        }

        // NULL
        if trimmed.eq_ignore_ascii_case("null") {
            return Ok(ToonValue::Null);
        }

        // Float (contains decimal point)
        if trimmed.contains('.')
            && let Ok(f) = trimmed.parse::<f64>()
        {
            return Ok(ToonValue::Float(f));
        }

        // Integer
        if let Ok(i) = trimmed.parse::<i64>() {
            return Ok(ToonValue::Int(i));
        }

        // Unsigned integer (if positive and within range)
        if let Ok(u) = trimmed.parse::<u64>() {
            return Ok(ToonValue::UInt(u));
        }

        // If nothing else, treat as unquoted string (column name or identifier)
        Ok(ToonValue::Text(trimmed.to_string()))
    }

    fn parse_insert(query: &str) -> Result<ToonQuery, ParseError> {
        // Simple INSERT parser
        // INSERT table: col1: val1 col2: val2
        // or INSERT table[n]{cols}: val1,val2 val3,val4
        let after_insert = query[6..].trim();

        // Find table name
        let table_end = after_insert
            .find([':', '['])
            .ok_or(ParseError::InvalidSyntax)?;
        let table = after_insert[..table_end].trim().to_string();

        // Simplified: just return structure
        Ok(ToonQuery::Insert(InsertQuery {
            table,
            columns: Vec::new(),
            rows: Vec::new(),
        }))
    }

    fn parse_create_table(query: &str) -> Result<ToonQuery, ParseError> {
        // CREATE TABLE name { ... }
        let after_create = &query[12..].trim();
        let brace_idx = after_create.find('{').ok_or(ParseError::InvalidSyntax)?;
        let table = after_create[..brace_idx].trim().to_string();

        Ok(ToonQuery::CreateTable(CreateTableQuery {
            table,
            columns: Vec::new(),
            primary_key: None,
        }))
    }

    fn parse_drop_table(query: &str) -> Result<ToonQuery, ParseError> {
        let table = query[10..].trim().to_string();
        Ok(ToonQuery::DropTable { table })
    }
}

/// Parse error
#[derive(Debug, Clone)]
pub enum ParseError {
    UnknownStatement,
    MissingFrom,
    InvalidSyntax,
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnknownStatement => write!(f, "Unknown statement"),
            ParseError::MissingFrom => write!(f, "Missing FROM clause"),
            ParseError::InvalidSyntax => write!(f, "Invalid syntax"),
            ParseError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select() {
        let query = "SELECT id, name FROM users";
        let result = ToonQlParser::parse(query).unwrap();

        match result {
            ToonQuery::Select(select) => {
                assert_eq!(select.table, "users");
                assert_eq!(select.columns, vec!["id", "name"]);
            }
            _ => panic!("Expected SELECT query"),
        }
    }

    #[test]
    fn test_parse_select_star() {
        let query = "SELECT * FROM users";
        let result = ToonQlParser::parse(query).unwrap();

        match result {
            ToonQuery::Select(select) => {
                assert_eq!(select.table, "users");
                assert_eq!(select.columns, vec!["*"]);
            }
            _ => panic!("Expected SELECT query"),
        }
    }

    #[test]
    fn test_parse_create_table() {
        let query = "CREATE TABLE users { id: u64, name: text }";
        let result = ToonQlParser::parse(query).unwrap();

        match result {
            ToonQuery::CreateTable(ct) => {
                assert_eq!(ct.table, "users");
            }
            _ => panic!("Expected CREATE TABLE query"),
        }
    }

    #[test]
    fn test_parse_drop_table() {
        let query = "DROP TABLE users";
        let result = ToonQlParser::parse(query).unwrap();

        match result {
            ToonQuery::DropTable { table } => {
                assert_eq!(table, "users");
            }
            _ => panic!("Expected DROP TABLE query"),
        }
    }

    #[test]
    fn test_toon_result_format() {
        let result = ToonResult {
            table: "users".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![
                vec![ToonValue::UInt(1), ToonValue::Text("Alice".to_string())],
                vec![ToonValue::UInt(2), ToonValue::Text("Bob".to_string())],
            ],
        };

        let formatted = result.to_toon_string();
        assert!(formatted.contains("users[2]{id,name}:"));
        assert!(formatted.contains("1,Alice"));
        assert!(formatted.contains("2,Bob"));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_toon_value_format() {
        assert_eq!(ToonValue::Null.to_toon_string(), "null");
        assert_eq!(ToonValue::Bool(true).to_toon_string(), "true");
        assert_eq!(ToonValue::Int(-42).to_toon_string(), "-42");
        assert_eq!(ToonValue::UInt(100).to_toon_string(), "100");
        assert_eq!(ToonValue::Float(3.14).to_toon_string(), "3.14");
        assert_eq!(
            ToonValue::Text("hello".to_string()).to_toon_string(),
            "hello"
        );
    }
}
