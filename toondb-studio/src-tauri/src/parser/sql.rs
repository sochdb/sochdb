//! SQL parser for ToonDB Studio
//!
//! Provides a complete SQL parser for DML/DDL statements with proper AST generation.
//! This is the unified parser used across Studio, MCP, and query engine.

use serde::{Deserialize, Serialize};

/// SQL statement type
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlStatement {
    Select {
        table: String,
        columns: Vec<String>,
        where_clause: Option<WhereExpr>,
        limit: Option<usize>,
        offset: Option<usize>,
        order_by: Option<OrderBy>,
    },
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Vec<SqlValue>>,
    },
    Update {
        table: String,
        set_clause: Vec<(String, SqlValue)>,
        where_clause: Option<WhereExpr>,
    },
    Delete {
        table: String,
        where_clause: Option<WhereExpr>,
    },
    CreateTable {
        name: String,
        columns: Vec<ColumnDef>,
        if_not_exists: bool,
    },
    DropTable {
        name: String,
        if_exists: bool,
    },
    CreateIndex {
        name: String,
        table: String,
        columns: Vec<String>,
        unique: bool,
    },
    /// Unsupported statement with explanation
    Unsupported {
        statement: String,
        reason: String,
    },
}

/// Column definition for CREATE TABLE
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub default: Option<SqlValue>,
}

/// SQL data types
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    BigInt,
    Float,
    Double,
    Text,
    Varchar(usize),
    Boolean,
    Timestamp,
    Blob,
    Vector(usize), // Vector with dimension
    Json,
}

/// SQL value
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Placeholder(String), // For prepared statements: $1, ?
}

/// WHERE expression (supports full boolean logic)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhereExpr {
    Comparison {
        column: String,
        op: CompareOp,
        value: SqlValue,
    },
    In {
        column: String,
        values: Vec<SqlValue>,
        negated: bool,
    },
    Like {
        column: String,
        pattern: String,
        negated: bool,
    },
    IsNull {
        column: String,
        negated: bool,
    },
    Between {
        column: String,
        low: SqlValue,
        high: SqlValue,
    },
    And(Vec<WhereExpr>),
    Or(Vec<WhereExpr>),
    Not(Box<WhereExpr>),
}

/// Comparison operators
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// ORDER BY clause
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBy {
    pub columns: Vec<(String, SortOrder)>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Asc,
    Desc,
}

/// SQL Parser with tokenizer
#[allow(dead_code)]
pub struct SqlParser<'a> {
    input: &'a str,
    pos: usize,
    tokens: Vec<Token>,
    token_pos: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Keyword(String),
    Identifier(String),
    String(String),
    Number(String),
    Operator(String),
    Comma,
    OpenParen,
    CloseParen,
    Semicolon,
    Star,
    Dot,
    Eof,
}

impl<'a> SqlParser<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut parser = Self {
            input,
            pos: 0,
            tokens: Vec::new(),
            token_pos: 0,
        };
        parser.tokenize();
        parser
    }

    fn tokenize(&mut self) {
        let chars: Vec<char> = self.input.chars().collect();
        let len = chars.len();
        self.pos = 0;

        while self.pos < len {
            let c = chars[self.pos];

            // Skip whitespace
            if c.is_whitespace() {
                self.pos += 1;
                continue;
            }

            // String literal
            if c == '\'' || c == '"' {
                let quote = c;
                self.pos += 1;
                let start = self.pos;
                while self.pos < len && chars[self.pos] != quote {
                    if chars[self.pos] == '\\' && self.pos + 1 < len {
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
                let s: String = chars[start..self.pos].iter().collect();
                self.tokens.push(Token::String(s));
                self.pos += 1; // Skip closing quote
                continue;
            }

            // Number
            if c.is_ascii_digit() || (c == '-' && self.pos + 1 < len && chars[self.pos + 1].is_ascii_digit()) {
                let start = self.pos;
                if c == '-' {
                    self.pos += 1;
                }
                while self.pos < len && (chars[self.pos].is_ascii_digit() || chars[self.pos] == '.') {
                    self.pos += 1;
                }
                let s: String = chars[start..self.pos].iter().collect();
                self.tokens.push(Token::Number(s));
                continue;
            }

            // Identifier or keyword
            if c.is_alphabetic() || c == '_' {
                let start = self.pos;
                while self.pos < len && (chars[self.pos].is_alphanumeric() || chars[self.pos] == '_') {
                    self.pos += 1;
                }
                let s: String = chars[start..self.pos].iter().collect();
                let upper = s.to_uppercase();
                if is_keyword(&upper) {
                    self.tokens.push(Token::Keyword(upper));
                } else {
                    self.tokens.push(Token::Identifier(s));
                }
                continue;
            }

            // Operators and punctuation
            match c {
                ',' => {
                    self.tokens.push(Token::Comma);
                    self.pos += 1;
                }
                '(' => {
                    self.tokens.push(Token::OpenParen);
                    self.pos += 1;
                }
                ')' => {
                    self.tokens.push(Token::CloseParen);
                    self.pos += 1;
                }
                ';' => {
                    self.tokens.push(Token::Semicolon);
                    self.pos += 1;
                }
                '*' => {
                    self.tokens.push(Token::Star);
                    self.pos += 1;
                }
                '.' => {
                    self.tokens.push(Token::Dot);
                    self.pos += 1;
                }
                '=' => {
                    self.tokens.push(Token::Operator("=".to_string()));
                    self.pos += 1;
                }
                '<' => {
                    if self.pos + 1 < len && chars[self.pos + 1] == '=' {
                        self.tokens.push(Token::Operator("<=".to_string()));
                        self.pos += 2;
                    } else if self.pos + 1 < len && chars[self.pos + 1] == '>' {
                        self.tokens.push(Token::Operator("<>".to_string()));
                        self.pos += 2;
                    } else {
                        self.tokens.push(Token::Operator("<".to_string()));
                        self.pos += 1;
                    }
                }
                '>' => {
                    if self.pos + 1 < len && chars[self.pos + 1] == '=' {
                        self.tokens.push(Token::Operator(">=".to_string()));
                        self.pos += 2;
                    } else {
                        self.tokens.push(Token::Operator(">".to_string()));
                        self.pos += 1;
                    }
                }
                '!' => {
                    if self.pos + 1 < len && chars[self.pos + 1] == '=' {
                        self.tokens.push(Token::Operator("!=".to_string()));
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
                '$' | '?' => {
                    // Placeholder
                    let start = self.pos;
                    self.pos += 1;
                    while self.pos < len && chars[self.pos].is_ascii_digit() {
                        self.pos += 1;
                    }
                    let s: String = chars[start..self.pos].iter().collect();
                    self.tokens.push(Token::Identifier(s));
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
        self.tokens.push(Token::Eof);
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.token_pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.token_pos < self.tokens.len() {
            self.token_pos += 1;
        }
    }

    fn expect_keyword(&mut self, kw: &str) -> Result<(), String> {
        match self.current() {
            Token::Keyword(k) if k == kw => {
                self.advance();
                Ok(())
            }
            other => Err(format!("Expected keyword '{}', found {:?}", kw, other)),
        }
    }

    fn expect_identifier(&mut self) -> Result<String, String> {
        match self.current().clone() {
            Token::Identifier(s) => {
                self.advance();
                Ok(s)
            }
            Token::Keyword(s) => {
                // Allow keywords as identifiers in some contexts
                self.advance();
                Ok(s.to_lowercase())
            }
            other => Err(format!("Expected identifier, found {:?}", other)),
        }
    }

    fn parse_value(&mut self) -> Result<SqlValue, String> {
        match self.current().clone() {
            Token::String(s) => {
                self.advance();
                Ok(SqlValue::Text(s))
            }
            Token::Number(s) => {
                self.advance();
                if s.contains('.') {
                    Ok(SqlValue::Float(s.parse().map_err(|_| "Invalid float")?))
                } else {
                    Ok(SqlValue::Integer(s.parse().map_err(|_| "Invalid integer")?))
                }
            }
            Token::Keyword(k) if k == "NULL" => {
                self.advance();
                Ok(SqlValue::Null)
            }
            Token::Keyword(k) if k == "TRUE" => {
                self.advance();
                Ok(SqlValue::Boolean(true))
            }
            Token::Keyword(k) if k == "FALSE" => {
                self.advance();
                Ok(SqlValue::Boolean(false))
            }
            Token::Identifier(s) if s.starts_with('$') || s.starts_with('?') => {
                self.advance();
                Ok(SqlValue::Placeholder(s))
            }
            other => Err(format!("Expected value, found {:?}", other)),
        }
    }
}

#[allow(dead_code)]
fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "SELECT" | "FROM" | "WHERE" | "AND" | "OR" | "NOT" | "IN" | "LIKE" |
        "IS" | "NULL" | "BETWEEN" | "ORDER" | "BY" | "ASC" | "DESC" |
        "LIMIT" | "OFFSET" | "INSERT" | "INTO" | "VALUES" | "UPDATE" |
        "SET" | "DELETE" | "CREATE" | "TABLE" | "DROP" | "INDEX" |
        "IF" | "EXISTS" | "PRIMARY" | "KEY" | "UNIQUE" |
        "DEFAULT" | "ON" | "TRUE" | "FALSE" | "INTEGER" | "INT" |
        "BIGINT" | "FLOAT" | "DOUBLE" | "TEXT" | "VARCHAR" | "BOOLEAN" |
        "TIMESTAMP" | "BLOB" | "VECTOR" | "JSON"
    )
}

/// Parse a SQL query string
#[allow(dead_code)]
pub fn parse_sql(query: &str) -> Result<SqlStatement, String> {
    let mut parser = SqlParser::new(query.trim());
    
    match parser.current() {
        Token::Keyword(k) => match k.as_str() {
            "SELECT" => parse_select_stmt(&mut parser),
            "INSERT" => parse_insert_stmt(&mut parser),
            "UPDATE" => parse_update_stmt(&mut parser),
            "DELETE" => parse_delete_stmt(&mut parser),
            "CREATE" => parse_create_stmt(&mut parser),
            "DROP" => parse_drop_stmt(&mut parser),
            _ => Ok(SqlStatement::Unsupported {
                statement: query.to_string(),
                reason: format!("Unsupported statement type: {}", k),
            }),
        },
        _ => Err("Expected SQL keyword at start of statement".to_string()),
    }
}

#[allow(dead_code)]
fn parse_select_stmt(parser: &mut SqlParser) -> Result<SqlStatement, String> {
    parser.expect_keyword("SELECT")?;
    
    // Parse columns
    let mut columns = Vec::new();
    loop {
        match parser.current() {
            Token::Star => {
                parser.advance();
                columns.push("*".to_string());
            }
            Token::Identifier(_) | Token::Keyword(_) => {
                let col = parser.expect_identifier()?;
                columns.push(col);
            }
            _ => break,
        }
        if matches!(parser.current(), Token::Comma) {
            parser.advance();
        } else {
            break;
        }
    }

    // FROM clause
    parser.expect_keyword("FROM")?;
    let table = parser.expect_identifier()?;

    // WHERE clause (optional)
    let where_clause = if matches!(parser.current(), Token::Keyword(k) if k == "WHERE") {
        parser.advance();
        Some(parse_where_expr(parser)?)
    } else {
        None
    };

    // ORDER BY (optional)
    let order_by = if matches!(parser.current(), Token::Keyword(k) if k == "ORDER") {
        parser.advance();
        parser.expect_keyword("BY")?;
        let mut order_cols = Vec::new();
        loop {
            let col = parser.expect_identifier()?;
            let order = if matches!(parser.current(), Token::Keyword(k) if k == "DESC") {
                parser.advance();
                SortOrder::Desc
            } else {
                if matches!(parser.current(), Token::Keyword(k) if k == "ASC") {
                    parser.advance();
                }
                SortOrder::Asc
            };
            order_cols.push((col, order));
            if matches!(parser.current(), Token::Comma) {
                parser.advance();
            } else {
                break;
            }
        }
        Some(OrderBy { columns: order_cols })
    } else {
        None
    };

    // LIMIT (optional)
    let limit = if matches!(parser.current(), Token::Keyword(k) if k == "LIMIT") {
        parser.advance();
        match parser.current() {
            Token::Number(s) => {
                let n = s.parse().ok();
                parser.advance();
                n
            }
            _ => None,
        }
    } else {
        None
    };

    // OFFSET (optional)
    let offset = if matches!(parser.current(), Token::Keyword(k) if k == "OFFSET") {
        parser.advance();
        match parser.current() {
            Token::Number(s) => {
                let n = s.parse().ok();
                parser.advance();
                n
            }
            _ => None,
        }
    } else {
        None
    };

    Ok(SqlStatement::Select {
        table,
        columns,
        where_clause,
        limit,
        offset,
        order_by,
    })
}

#[allow(dead_code)]
fn parse_where_expr(parser: &mut SqlParser) -> Result<WhereExpr, String> {
    parse_or_expr(parser)
}

#[allow(dead_code)]
fn parse_or_expr(parser: &mut SqlParser) -> Result<WhereExpr, String> {
    let mut left = parse_and_expr(parser)?;
    
    while matches!(parser.current(), Token::Keyword(k) if k == "OR") {
        parser.advance();
        let right = parse_and_expr(parser)?;
        left = WhereExpr::Or(vec![left, right]);
    }
    
    Ok(left)
}

#[allow(dead_code)]
fn parse_and_expr(parser: &mut SqlParser) -> Result<WhereExpr, String> {
    let mut left = parse_comparison(parser)?;
    
    while matches!(parser.current(), Token::Keyword(k) if k == "AND") {
        parser.advance();
        let right = parse_comparison(parser)?;
        left = WhereExpr::And(vec![left, right]);
    }
    
    Ok(left)
}

#[allow(dead_code)]
fn parse_comparison(parser: &mut SqlParser) -> Result<WhereExpr, String> {
    // Handle NOT
    if matches!(parser.current(), Token::Keyword(k) if k == "NOT") {
        parser.advance();
        let inner = parse_comparison(parser)?;
        return Ok(WhereExpr::Not(Box::new(inner)));
    }

    // Handle parentheses
    if matches!(parser.current(), Token::OpenParen) {
        parser.advance();
        let expr = parse_where_expr(parser)?;
        if matches!(parser.current(), Token::CloseParen) {
            parser.advance();
        }
        return Ok(expr);
    }

    // Column name
    let column = parser.expect_identifier()?;

    // Check for IS NULL / IS NOT NULL
    if matches!(parser.current(), Token::Keyword(k) if k == "IS") {
        parser.advance();
        let negated = if matches!(parser.current(), Token::Keyword(k) if k == "NOT") {
            parser.advance();
            true
        } else {
            false
        };
        parser.expect_keyword("NULL")?;
        return Ok(WhereExpr::IsNull { column, negated });
    }

    // Check for IN
    if matches!(parser.current(), Token::Keyword(k) if k == "IN" || k == "NOT") {
        let negated = if matches!(parser.current(), Token::Keyword(k) if k == "NOT") {
            parser.advance();
            parser.expect_keyword("IN")?;
            true
        } else {
            parser.advance(); // consume IN
            false
        };
        
        if !matches!(parser.current(), Token::OpenParen) {
            return Err("Expected '(' after IN".to_string());
        }
        parser.advance();
        
        let mut values = Vec::new();
        loop {
            values.push(parser.parse_value()?);
            if matches!(parser.current(), Token::Comma) {
                parser.advance();
            } else {
                break;
            }
        }
        
        if matches!(parser.current(), Token::CloseParen) {
            parser.advance();
        }
        
        return Ok(WhereExpr::In { column, values, negated });
    }

    // Check for LIKE
    if matches!(parser.current(), Token::Keyword(k) if k == "LIKE" || k == "NOT") {
        let negated = if matches!(parser.current(), Token::Keyword(k) if k == "NOT") {
            parser.advance();
            parser.expect_keyword("LIKE")?;
            true
        } else {
            parser.advance(); // consume LIKE
            false
        };
        
        let pattern = match parser.current() {
            Token::String(s) => {
                let p = s.clone();
                parser.advance();
                p
            }
            _ => return Err("Expected string pattern after LIKE".to_string()),
        };
        
        return Ok(WhereExpr::Like { column, pattern, negated });
    }

    // Check for BETWEEN
    if matches!(parser.current(), Token::Keyword(k) if k == "BETWEEN") {
        parser.advance();
        let low = parser.parse_value()?;
        parser.expect_keyword("AND")?;
        let high = parser.parse_value()?;
        return Ok(WhereExpr::Between { column, low, high });
    }

    // Standard comparison
    let op = match parser.current() {
        Token::Operator(s) => match s.as_str() {
            "=" => CompareOp::Eq,
            "!=" | "<>" => CompareOp::Ne,
            "<" => CompareOp::Lt,
            "<=" => CompareOp::Le,
            ">" => CompareOp::Gt,
            ">=" => CompareOp::Ge,
            _ => return Err(format!("Unknown operator: {}", s)),
        },
        _ => return Err("Expected comparison operator".to_string()),
    };
    parser.advance();

    let value = parser.parse_value()?;

    Ok(WhereExpr::Comparison { column, op, value })
}

#[allow(dead_code)]
fn parse_insert_stmt(parser: &mut SqlParser) -> Result<SqlStatement, String> {
    parser.expect_keyword("INSERT")?;
    parser.expect_keyword("INTO")?;
    
    let table = parser.expect_identifier()?;
    
    // Optional column list
    let columns = if matches!(parser.current(), Token::OpenParen) {
        parser.advance();
        let mut cols = Vec::new();
        loop {
            cols.push(parser.expect_identifier()?);
            if matches!(parser.current(), Token::Comma) {
                parser.advance();
            } else {
                break;
            }
        }
        if matches!(parser.current(), Token::CloseParen) {
            parser.advance();
        }
        cols
    } else {
        Vec::new()
    };
    
    parser.expect_keyword("VALUES")?;
    
    // Parse value rows
    let mut values = Vec::new();
    loop {
        if !matches!(parser.current(), Token::OpenParen) {
            break;
        }
        parser.advance();
        
        let mut row = Vec::new();
        loop {
            row.push(parser.parse_value()?);
            if matches!(parser.current(), Token::Comma) {
                parser.advance();
            } else {
                break;
            }
        }
        
        if matches!(parser.current(), Token::CloseParen) {
            parser.advance();
        }
        values.push(row);
        
        if matches!(parser.current(), Token::Comma) {
            parser.advance();
        } else {
            break;
        }
    }
    
    Ok(SqlStatement::Insert { table, columns, values })
}

#[allow(dead_code)]
fn parse_update_stmt(parser: &mut SqlParser) -> Result<SqlStatement, String> {
    parser.expect_keyword("UPDATE")?;
    let table = parser.expect_identifier()?;
    parser.expect_keyword("SET")?;
    
    // Parse SET clause
    let mut set_clause = Vec::new();
    loop {
        let col = parser.expect_identifier()?;
        if !matches!(parser.current(), Token::Operator(s) if s == "=") {
            return Err("Expected '=' in SET clause".to_string());
        }
        parser.advance();
        let val = parser.parse_value()?;
        set_clause.push((col, val));
        
        if matches!(parser.current(), Token::Comma) {
            parser.advance();
        } else {
            break;
        }
    }
    
    // WHERE clause (optional)
    let where_clause = if matches!(parser.current(), Token::Keyword(k) if k == "WHERE") {
        parser.advance();
        Some(parse_where_expr(parser)?)
    } else {
        None
    };
    
    Ok(SqlStatement::Update { table, set_clause, where_clause })
}

#[allow(dead_code)]
fn parse_delete_stmt(parser: &mut SqlParser) -> Result<SqlStatement, String> {
    parser.expect_keyword("DELETE")?;
    parser.expect_keyword("FROM")?;
    let table = parser.expect_identifier()?;
    
    // WHERE clause (optional)
    let where_clause = if matches!(parser.current(), Token::Keyword(k) if k == "WHERE") {
        parser.advance();
        Some(parse_where_expr(parser)?)
    } else {
        None
    };
    
    Ok(SqlStatement::Delete { table, where_clause })
}

#[allow(dead_code)]
fn parse_create_stmt(parser: &mut SqlParser) -> Result<SqlStatement, String> {
    parser.expect_keyword("CREATE")?;
    
    match parser.current() {
        Token::Keyword(k) if k == "TABLE" => {
            parser.advance();
            
            let if_not_exists = if matches!(parser.current(), Token::Keyword(k) if k == "IF") {
                parser.advance();
                parser.expect_keyword("NOT")?;
                parser.expect_keyword("EXISTS")?;
                true
            } else {
                false
            };
            
            let name = parser.expect_identifier()?;
            
            if !matches!(parser.current(), Token::OpenParen) {
                return Err("Expected '(' after table name".to_string());
            }
            parser.advance();
            
            let mut columns = Vec::new();
            loop {
                let col_name = parser.expect_identifier()?;
                let data_type = parse_data_type(parser)?;
                
                let mut nullable = true;
                let mut primary_key = false;
                let mut default = None;
                
                // Parse column constraints
                loop {
                    match parser.current() {
                        Token::Keyword(k) if k == "NOT" => {
                            parser.advance();
                            parser.expect_keyword("NULL")?;
                            nullable = false;
                        }
                        Token::Keyword(k) if k == "NULL" => {
                            parser.advance();
                            nullable = true;
                        }
                        Token::Keyword(k) if k == "PRIMARY" => {
                            parser.advance();
                            parser.expect_keyword("KEY")?;
                            primary_key = true;
                            nullable = false;
                        }
                        Token::Keyword(k) if k == "DEFAULT" => {
                            parser.advance();
                            default = Some(parser.parse_value()?);
                        }
                        _ => break,
                    }
                }
                
                columns.push(ColumnDef {
                    name: col_name,
                    data_type,
                    nullable,
                    primary_key,
                    default,
                });
                
                if matches!(parser.current(), Token::Comma) {
                    parser.advance();
                } else {
                    break;
                }
            }
            
            if matches!(parser.current(), Token::CloseParen) {
                parser.advance();
            }
            
            Ok(SqlStatement::CreateTable { name, columns, if_not_exists })
        }
        Token::Keyword(k) if k == "INDEX" || k == "UNIQUE" => {
            let unique = k == "UNIQUE";
            if unique {
                parser.advance();
            }
            parser.expect_keyword("INDEX")?;
            
            let name = parser.expect_identifier()?;
            parser.expect_keyword("ON")?;
            let table = parser.expect_identifier()?;
            
            if !matches!(parser.current(), Token::OpenParen) {
                return Err("Expected '(' after table name".to_string());
            }
            parser.advance();
            
            let mut index_columns = Vec::new();
            loop {
                index_columns.push(parser.expect_identifier()?);
                if matches!(parser.current(), Token::Comma) {
                    parser.advance();
                } else {
                    break;
                }
            }
            
            if matches!(parser.current(), Token::CloseParen) {
                parser.advance();
            }
            
            Ok(SqlStatement::CreateIndex { name, table, columns: index_columns, unique })
        }
        _ => Ok(SqlStatement::Unsupported {
            statement: parser.input.to_string(),
            reason: "Only CREATE TABLE and CREATE INDEX are supported".to_string(),
        }),
    }
}

#[allow(dead_code)]
fn parse_data_type(parser: &mut SqlParser) -> Result<DataType, String> {
    match parser.current() {
        Token::Keyword(k) => {
            let dt = match k.as_str() {
                "INTEGER" | "INT" => DataType::Integer,
                "BIGINT" => DataType::BigInt,
                "FLOAT" => DataType::Float,
                "DOUBLE" => DataType::Double,
                "TEXT" => DataType::Text,
                "BOOLEAN" | "BOOL" => DataType::Boolean,
                "TIMESTAMP" => DataType::Timestamp,
                "BLOB" => DataType::Blob,
                "JSON" => DataType::Json,
                "VARCHAR" => {
                    parser.advance();
                    if matches!(parser.current(), Token::OpenParen) {
                        parser.advance();
                        let size = match parser.current() {
                            Token::Number(s) => s.parse().unwrap_or(255),
                            _ => 255,
                        };
                        parser.advance();
                        if matches!(parser.current(), Token::CloseParen) {
                            parser.advance();
                        }
                        return Ok(DataType::Varchar(size));
                    }
                    return Ok(DataType::Varchar(255));
                }
                "VECTOR" => {
                    parser.advance();
                    if matches!(parser.current(), Token::OpenParen) {
                        parser.advance();
                        let dim = match parser.current() {
                            Token::Number(s) => s.parse().unwrap_or(768),
                            _ => 768,
                        };
                        parser.advance();
                        if matches!(parser.current(), Token::CloseParen) {
                            parser.advance();
                        }
                        return Ok(DataType::Vector(dim));
                    }
                    return Ok(DataType::Vector(768));
                }
                _ => return Err(format!("Unknown data type: {}", k)),
            };
            parser.advance();
            Ok(dt)
        }
        _ => Err("Expected data type".to_string()),
    }
}

#[allow(dead_code)]
fn parse_drop_stmt(parser: &mut SqlParser) -> Result<SqlStatement, String> {
    parser.expect_keyword("DROP")?;
    parser.expect_keyword("TABLE")?;
    
    let if_exists = if matches!(parser.current(), Token::Keyword(k) if k == "IF") {
        parser.advance();
        parser.expect_keyword("EXISTS")?;
        true
    } else {
        false
    };
    
    let name = parser.expect_identifier()?;
    
    Ok(SqlStatement::DropTable { name, if_exists })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select() {
        let result = parse_sql("SELECT * FROM users LIMIT 100");
        assert!(result.is_ok());
        
        if let SqlStatement::Select { table, columns, limit, .. } = result.unwrap() {
            assert_eq!(table, "users");
            assert_eq!(columns, vec!["*"]);
            assert_eq!(limit, Some(100));
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn test_parse_select_with_where() {
        let result = parse_sql("SELECT id, name FROM users WHERE age > 18 AND status = 'active'");
        assert!(result.is_ok());
        
        if let SqlStatement::Select { table, columns, where_clause, .. } = result.unwrap() {
            assert_eq!(table, "users");
            assert_eq!(columns, vec!["id", "name"]);
            assert!(where_clause.is_some());
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn test_parse_insert() {
        let result = parse_sql("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')");
        assert!(result.is_ok());
        
        if let SqlStatement::Insert { table, columns, values } = result.unwrap() {
            assert_eq!(table, "users");
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(values.len(), 2);
        } else {
            panic!("Expected INSERT statement");
        }
    }

    #[test]
    fn test_parse_update() {
        let result = parse_sql("UPDATE users SET name = 'Charlie', age = 30 WHERE id = 1");
        assert!(result.is_ok());
        
        if let SqlStatement::Update { table, set_clause, where_clause } = result.unwrap() {
            assert_eq!(table, "users");
            assert_eq!(set_clause.len(), 2);
            assert!(where_clause.is_some());
        } else {
            panic!("Expected UPDATE statement");
        }
    }

    #[test]
    fn test_parse_delete() {
        let result = parse_sql("DELETE FROM users WHERE id = 1");
        assert!(result.is_ok());
        
        if let SqlStatement::Delete { table, where_clause } = result.unwrap() {
            assert_eq!(table, "users");
            assert!(where_clause.is_some());
        } else {
            panic!("Expected DELETE statement");
        }
    }

    #[test]
    fn test_parse_create_table() {
        let result = parse_sql("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, embedding VECTOR(768))");
        assert!(result.is_ok());
        
        if let SqlStatement::CreateTable { name, columns, if_not_exists } = result.unwrap() {
            assert_eq!(name, "users");
            assert!(if_not_exists);
            assert_eq!(columns.len(), 3);
            assert!(columns[0].primary_key);
            assert!(!columns[1].nullable);
            assert!(matches!(columns[2].data_type, DataType::Vector(768)));
        } else {
            panic!("Expected CREATE TABLE statement");
        }
    }

    #[test]
    fn test_parse_in_clause() {
        let result = parse_sql("SELECT * FROM users WHERE status IN ('active', 'pending')");
        assert!(result.is_ok());
        
        if let SqlStatement::Select { where_clause: Some(w), .. } = result.unwrap() {
            if let WhereExpr::In { column, values, negated } = w {
                assert_eq!(column, "status");
                assert_eq!(values.len(), 2);
                assert!(!negated);
            } else {
                panic!("Expected IN clause");
            }
        } else {
            panic!("Expected SELECT statement with WHERE");
        }
    }

    #[test]
    fn test_parse_complex_where() {
        let result = parse_sql("SELECT * FROM users WHERE (age > 18 AND age < 65) OR status = 'vip'");
        assert!(result.is_ok());
    }
}
