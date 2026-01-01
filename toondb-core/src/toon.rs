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

//! TOON (Tabular Object-Oriented Notation) - Native Data Format for ToonDB
//!
//! TOON is a compact, schema-aware data format optimized for LLMs and databases.
//! It's the native format for ToonDB, like JSON is for MongoDB.
//!
//! Format: `name[count]{fields}:\nrow1\nrow2\n...`
//!
//! Example:
//! ```text
//! users[3]{id,name,email}:
//! 1,Alice,alice@example.com
//! 2,Bob,bob@example.com
//! 3,Charlie,charlie@example.com
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// TOON Value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToonValue {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
    Binary(Vec<u8>),
    Array(Vec<ToonValue>),
    Object(HashMap<String, ToonValue>),
    /// Reference to another table row: ref(table_name, id)
    Ref {
        table: String,
        id: u64,
    },
}

impl ToonValue {
    pub fn is_null(&self) -> bool {
        matches!(self, ToonValue::Null)
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            ToonValue::Int(v) => Some(*v),
            ToonValue::UInt(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_uint(&self) -> Option<u64> {
        match self {
            ToonValue::UInt(v) => Some(*v),
            ToonValue::Int(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            ToonValue::Float(v) => Some(*v),
            ToonValue::Int(v) => Some(*v as f64),
            ToonValue::UInt(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            ToonValue::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ToonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

fn needs_quoting(s: &str) -> bool {
    if s.is_empty() { return true; }
    if s.starts_with(' ') || s.ends_with(' ') { return true; }
    if matches!(s, "true" | "false" | "null") { return true; }
    
    // Check for number-like patterns
    if s.parse::<f64>().is_ok() { return true; }
    if s == "-" || s.starts_with('-') { return true; }
    // Leading zeros check (e.g. 05 usually treated as number in some contexts or invalid)
    if s.len() > 1 && s.starts_with('0') && s.chars().nth(1).map_or(false, |c| c.is_ascii_digit()) && !s.contains('.') {
        return true;
    }

    // Check for special chars or delimiter (comma)
    // Spec §7.3: :, ", \, [, ], {, }, newline, return, tab, delimiter
    s.contains(|c| matches!(c, ':' | '"' | '\\' | '[' | ']' | '{' | '}' | '\n' | '\r' | '\t' | ','))
}

impl fmt::Display for ToonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToonValue::Null => write!(f, "null"),
            ToonValue::Bool(b) => write!(f, "{}", b),
            ToonValue::Int(i) => write!(f, "{}", i),
            ToonValue::UInt(u) => write!(f, "{}", u),
            ToonValue::Float(fl) => write!(f, "{}", fl),
            ToonValue::Text(s) => {
                if needs_quoting(s) {
                    write!(f, "\"")?;
                    for c in s.chars() {
                        match c {
                            '"' => write!(f, "\\\"")?,
                            '\\' => write!(f, "\\\\")?,
                            '\n' => write!(f, "\\n")?,
                            '\r' => write!(f, "\\r")?,
                            '\t' => write!(f, "\\t")?,
                            c => write!(f, "{}", c)?,
                        }
                    }
                    write!(f, "\"")
                } else {
                    write!(f, "{}", s)
                }
            }
            ToonValue::Binary(b) => write!(f, "0x{}", hex::encode(b)),
            ToonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ";")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ToonValue::Object(obj) => {
                write!(f, "{{")?;
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ";")?;
                    }
                    write!(f, "{}:{}", k, v)?;
                }
                write!(f, "}}")
            }
            ToonValue::Ref { table, id } => write!(f, "@{}:{}", table, id),
        }
    }
}

/// Field type in a TOON schema
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToonType {
    Null,
    Bool,
    Int,
    UInt,
    Float,
    Text,
    Binary,
    Array(Box<ToonType>),
    Object(Vec<(String, ToonType)>),
    Ref(String), // Reference to table name
    /// Union of types (for nullable fields)
    Optional(Box<ToonType>),
}

impl ToonType {
    /// Check if a value matches this type
    pub fn matches(&self, value: &ToonValue) -> bool {
        match (self, value) {
            (ToonType::Null, ToonValue::Null) => true,
            (ToonType::Bool, ToonValue::Bool(_)) => true,
            (ToonType::Int, ToonValue::Int(_)) => true,
            (ToonType::UInt, ToonValue::UInt(_)) => true,
            (ToonType::Float, ToonValue::Float(_)) => true,
            (ToonType::Text, ToonValue::Text(_)) => true,
            (ToonType::Binary, ToonValue::Binary(_)) => true,
            (ToonType::Array(inner), ToonValue::Array(arr)) => arr.iter().all(|v| inner.matches(v)),
            (ToonType::Ref(table), ToonValue::Ref { table: t, .. }) => table == t,
            (ToonType::Optional(inner), value) => value.is_null() || inner.matches(value),
            _ => false,
        }
    }

    /// Parse type from string notation
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        match s {
            "null" => Some(ToonType::Null),
            "bool" => Some(ToonType::Bool),
            "int" | "i64" => Some(ToonType::Int),
            "uint" | "u64" => Some(ToonType::UInt),
            "float" | "f64" => Some(ToonType::Float),
            "text" | "string" => Some(ToonType::Text),
            "binary" | "bytes" => Some(ToonType::Binary),
            _ if s.starts_with("ref(") && s.ends_with(')') => {
                let table = &s[4..s.len() - 1];
                Some(ToonType::Ref(table.to_string()))
            }
            _ if s.starts_with("array(") && s.ends_with(')') => {
                let inner = &s[6..s.len() - 1];
                ToonType::parse(inner).map(|t| ToonType::Array(Box::new(t)))
            }
            _ if s.ends_with('?') => {
                let inner = &s[..s.len() - 1];
                ToonType::parse(inner).map(|t| ToonType::Optional(Box::new(t)))
            }
            _ => None,
        }
    }
}

impl fmt::Display for ToonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToonType::Null => write!(f, "null"),
            ToonType::Bool => write!(f, "bool"),
            ToonType::Int => write!(f, "int"),
            ToonType::UInt => write!(f, "uint"),
            ToonType::Float => write!(f, "float"),
            ToonType::Text => write!(f, "text"),
            ToonType::Binary => write!(f, "binary"),
            ToonType::Array(inner) => write!(f, "array({})", inner),
            ToonType::Object(fields) => {
                write!(f, "{{")?;
                for (i, (name, ty)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}:{}", name, ty)?;
                }
                write!(f, "}}")
            }
            ToonType::Ref(table) => write!(f, "ref({})", table),
            ToonType::Optional(inner) => write!(f, "{}?", inner),
        }
    }
}

/// A TOON schema definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToonSchema {
    /// Schema name (table name)
    pub name: String,
    /// Field definitions
    pub fields: Vec<ToonField>,
    /// Primary key field name
    pub primary_key: Option<String>,
    /// Indexes on this schema
    pub indexes: Vec<ToonIndex>,
}

/// A field in a TOON schema
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToonField {
    pub name: String,
    pub field_type: ToonType,
    pub nullable: bool,
    pub default: Option<String>, // Default value as TOON string
}

/// An index definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToonIndex {
    pub name: String,
    pub fields: Vec<String>,
    pub unique: bool,
}

impl ToonSchema {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            primary_key: None,
            indexes: Vec::new(),
        }
    }

    pub fn field(mut self, name: impl Into<String>, field_type: ToonType) -> Self {
        self.fields.push(ToonField {
            name: name.into(),
            field_type,
            nullable: false,
            default: None,
        });
        self
    }

    pub fn nullable_field(mut self, name: impl Into<String>, field_type: ToonType) -> Self {
        self.fields.push(ToonField {
            name: name.into(),
            field_type,
            nullable: true,
            default: None,
        });
        self
    }

    pub fn primary_key(mut self, field: impl Into<String>) -> Self {
        self.primary_key = Some(field.into());
        self
    }

    pub fn index(mut self, name: impl Into<String>, fields: Vec<String>, unique: bool) -> Self {
        self.indexes.push(ToonIndex {
            name: name.into(),
            fields,
            unique,
        });
        self
    }

    /// Get field names for header
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Format schema header: name[0]{field1,field2,...}:
    pub fn format_header(&self) -> String {
        let fields: Vec<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();
        format!("{}[0]{{{}}}:", self.name, fields.join(","))
    }
}

/// A TOON row - values for a single record
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToonRow {
    pub values: Vec<ToonValue>,
}

impl ToonRow {
    pub fn new(values: Vec<ToonValue>) -> Self {
        Self { values }
    }

    /// Get value by index
    pub fn get(&self, index: usize) -> Option<&ToonValue> {
        self.values.get(index)
    }

    /// Format row as TOON line
    pub fn format(&self) -> String {
        self.values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Parse row from TOON line
    pub fn parse(line: &str, schema: &ToonSchema) -> Result<Self, String> {
        let mut values = Vec::with_capacity(schema.fields.len());
        let mut chars = line.chars().peekable();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut field_idx = 0;

        while let Some(ch) = chars.next() {
            match ch {
                '"' if !in_quotes => {
                    in_quotes = true;
                }
                '"' if in_quotes => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        current.push('"');
                    } else {
                        in_quotes = false;
                    }
                }
                ',' if !in_quotes => {
                    let value = Self::parse_value(&current, field_idx, schema)?;
                    values.push(value);
                    current.clear();
                    field_idx += 1;
                }
                _ => {
                    current.push(ch);
                }
            }
        }

        // Last field
        if !current.is_empty() || field_idx < schema.fields.len() {
            let value = Self::parse_value(&current, field_idx, schema)?;
            values.push(value);
        }

        Ok(Self { values })
    }

    fn parse_value(s: &str, field_idx: usize, schema: &ToonSchema) -> Result<ToonValue, String> {
        let s = s.trim();

        if s.is_empty() || s == "null" {
            return Ok(ToonValue::Null);
        }

        let field = schema
            .fields
            .get(field_idx)
            .ok_or_else(|| format!("Field index {} out of bounds", field_idx))?;

        match &field.field_type {
            ToonType::Bool => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" => Ok(ToonValue::Bool(true)),
                "false" | "0" | "no" => Ok(ToonValue::Bool(false)),
                _ => Err(format!("Invalid bool: {}", s)),
            },
            ToonType::Int => s
                .parse::<i64>()
                .map(ToonValue::Int)
                .map_err(|e| format!("Invalid int: {}", e)),
            ToonType::UInt => s
                .parse::<u64>()
                .map(ToonValue::UInt)
                .map_err(|e| format!("Invalid uint: {}", e)),
            ToonType::Float => s
                .parse::<f64>()
                .map(ToonValue::Float)
                .map_err(|e| format!("Invalid float: {}", e)),
            ToonType::Text => Ok(ToonValue::Text(s.to_string())),
            ToonType::Binary => {
                if let Some(hex_str) = s.strip_prefix("0x") {
                    hex::decode(hex_str)
                        .map(ToonValue::Binary)
                        .map_err(|e| format!("Invalid hex: {}", e))
                } else {
                    Err("Binary must start with 0x".to_string())
                }
            }
            ToonType::Ref(table) => {
                // Format: @table:id or just id
                if let Some(ref_str) = s.strip_prefix('@') {
                    let parts: Vec<&str> = ref_str.split(':').collect();
                    if parts.len() == 2 {
                        let id = parts[1]
                            .parse::<u64>()
                            .map_err(|e| format!("Invalid ref id: {}", e))?;
                        Ok(ToonValue::Ref {
                            table: parts[0].to_string(),
                            id,
                        })
                    } else {
                        Err(format!("Invalid ref format: {}", s))
                    }
                } else {
                    let id = s
                        .parse::<u64>()
                        .map_err(|e| format!("Invalid ref id: {}", e))?;
                    Ok(ToonValue::Ref {
                        table: table.clone(),
                        id,
                    })
                }
            }
            ToonType::Optional(inner) => {
                // Try to parse as inner type
                let temp_field = ToonField {
                    name: field.name.clone(),
                    field_type: (**inner).clone(),
                    nullable: true,
                    default: None,
                };
                let temp_schema = ToonSchema {
                    name: schema.name.clone(),
                    fields: vec![temp_field],
                    primary_key: None,
                    indexes: vec![],
                };
                Self::parse_value(s, 0, &temp_schema)
            }
            _ => Ok(ToonValue::Text(s.to_string())),
        }
    }
}

/// A complete TOON table (header + rows)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToonTable {
    pub schema: ToonSchema,
    pub rows: Vec<ToonRow>,
}

impl ToonTable {
    pub fn new(schema: ToonSchema) -> Self {
        Self {
            schema,
            rows: Vec::new(),
        }
    }

    pub fn with_rows(schema: ToonSchema, rows: Vec<ToonRow>) -> Self {
        Self { schema, rows }
    }

    pub fn push(&mut self, row: ToonRow) {
        self.rows.push(row);
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Format as TOON string
    pub fn format(&self) -> String {
        let fields: Vec<&str> = self.schema.fields.iter().map(|f| f.name.as_str()).collect();
        let header = format!(
            "{}[{}]{{{}}}:",
            self.schema.name,
            self.rows.len(),
            fields.join(",")
        );

        let mut output = header;
        for row in &self.rows {
            output.push('\n');
            output.push_str(&row.format());
        }
        output
    }

    /// Parse TOON string to table
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut lines = input.lines();

        // Parse header: name[count]{field1,field2,...}:
        let header = lines.next().ok_or("Empty input")?;
        let (schema, _count) = Self::parse_header(header)?;

        // Parse rows
        let mut rows = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let row = ToonRow::parse(line, &schema)?;
            rows.push(row);
        }

        Ok(Self { schema, rows })
    }

    fn parse_header(header: &str) -> Result<(ToonSchema, usize), String> {
        // name[count]{field1,field2,...}:
        let header = header.trim_end_matches(':');

        let bracket_start = header.find('[').ok_or("Missing [")?;
        let bracket_end = header.find(']').ok_or("Missing ]")?;
        let brace_start = header.find('{').ok_or("Missing {")?;
        let brace_end = header.find('}').ok_or("Missing }")?;

        let name = &header[..bracket_start];
        let count_str = &header[bracket_start + 1..bracket_end];
        let fields_str = &header[brace_start + 1..brace_end];

        let count = count_str
            .parse::<usize>()
            .map_err(|e| format!("Invalid count: {}", e))?;

        let field_names: Vec<&str> = fields_str.split(',').map(|s| s.trim()).collect();

        let mut schema = ToonSchema::new(name);
        for field_name in field_names {
            // Check if type is specified: field_name:type
            if let Some(colon_pos) = field_name.find(':') {
                let fname = &field_name[..colon_pos];
                let ftype_str = &field_name[colon_pos + 1..];
                let ftype = ToonType::parse(ftype_str).unwrap_or(ToonType::Text);
                schema.fields.push(ToonField {
                    name: fname.to_string(),
                    field_type: ftype,
                    nullable: false,
                    default: None,
                });
            } else {
                // Default to text type
                schema.fields.push(ToonField {
                    name: field_name.to_string(),
                    field_type: ToonType::Text,
                    nullable: false,
                    default: None,
                });
            }
        }

        Ok((schema, count))
    }
}

impl fmt::Display for ToonTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Trait for accessing columnar data without allocation
pub trait ColumnAccess {
    fn row_count(&self) -> usize;
    fn col_count(&self) -> usize;
    fn field_names(&self) -> Vec<&str>;
    fn write_value(
        &self,
        col_idx: usize,
        row_idx: usize,
        f: &mut dyn std::fmt::Write,
    ) -> std::fmt::Result;
}

/// Cursor for iterating over columnar data and emitting TOON format
pub struct ToonCursor<'a, C: ColumnAccess> {
    access: &'a C,
    current_row: usize,
    header_emitted: bool,
    schema_name: String,
}

impl<'a, C: ColumnAccess> ToonCursor<'a, C> {
    pub fn new(access: &'a C, schema_name: String) -> Self {
        Self {
            access,
            current_row: 0,
            header_emitted: false,
            schema_name,
        }
    }
}

impl<'a, C: ColumnAccess> Iterator for ToonCursor<'a, C> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.header_emitted {
            self.header_emitted = true;
            let fields = self.access.field_names().join(",");
            return Some(format!(
                "{}[{}]{{{}}}:",
                self.schema_name,
                self.access.row_count(),
                fields
            ));
        }

        if self.current_row >= self.access.row_count() {
            return None;
        }

        let mut row_str = String::new();
        for col_idx in 0..self.access.col_count() {
            if col_idx > 0 {
                row_str.push(',');
            }
            // We ignore write errors here as String write shouldn't fail
            let _ = self
                .access
                .write_value(col_idx, self.current_row, &mut row_str);
        }

        self.current_row += 1;
        Some(row_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toon_value_display() {
        assert_eq!(ToonValue::Int(42).to_string(), "42");
        assert_eq!(ToonValue::Text("hello".into()).to_string(), "hello");
        assert_eq!(
            ToonValue::Text("hello, world".into()).to_string(),
            "\"hello, world\""
        );
        assert_eq!(ToonValue::Bool(true).to_string(), "true");
        assert_eq!(ToonValue::Null.to_string(), "null");
    }

    #[test]
    fn test_toon_schema() {
        let schema = ToonSchema::new("users")
            .field("id", ToonType::UInt)
            .field("name", ToonType::Text)
            .field("email", ToonType::Text)
            .primary_key("id");

        assert_eq!(schema.name, "users");
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.primary_key, Some("id".to_string()));
    }

    #[test]
    fn test_toon_table_format() {
        let schema = ToonSchema::new("users")
            .field("id", ToonType::UInt)
            .field("name", ToonType::Text)
            .field("email", ToonType::Text);

        let mut table = ToonTable::new(schema);
        table.push(ToonRow::new(vec![
            ToonValue::UInt(1),
            ToonValue::Text("Alice".into()),
            ToonValue::Text("alice@example.com".into()),
        ]));
        table.push(ToonRow::new(vec![
            ToonValue::UInt(2),
            ToonValue::Text("Bob".into()),
            ToonValue::Text("bob@example.com".into()),
        ]));

        let formatted = table.format();
        assert!(formatted.contains("users[2]{id,name,email}:"));
        assert!(formatted.contains("1,Alice,alice@example.com"));
        assert!(formatted.contains("2,Bob,bob@example.com"));
    }

    #[test]
    fn test_toon_table_parse() {
        let input = r#"users[2]{id,name,email}:
1,Alice,alice@example.com
2,Bob,bob@example.com"#;

        let table = ToonTable::parse(input).unwrap();
        assert_eq!(table.schema.name, "users");
        assert_eq!(table.rows.len(), 2);
    }

    #[test]
    fn test_toon_type_parse() {
        assert_eq!(ToonType::parse("int"), Some(ToonType::Int));
        assert_eq!(ToonType::parse("text"), Some(ToonType::Text));
        assert_eq!(
            ToonType::parse("ref(users)"),
            Some(ToonType::Ref("users".into()))
        );
        assert_eq!(
            ToonType::parse("int?"),
            Some(ToonType::Optional(Box::new(ToonType::Int)))
        );
    }
}
