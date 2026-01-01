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

//! Schema Bridge - TOON to Columnar Mapping
//!
//! This module provides bidirectional mapping between TOON document format
//! and columnar storage format for efficient analytical queries.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Schema Bridge                               │
//! │                                                                 │
//! │  TOON Document Format          Columnar Format                  │
//! │  ┌──────────────────┐          ┌──────────────────┐            │
//! │  │ users[3]{id,name}│          │ Column: id       │            │
//! │  │ 1,Alice          │   ←───→  │ [1, 2, 3]        │            │
//! │  │ 2,Bob            │          │                  │            │
//! │  │ 3,Carol          │          │ Column: name     │            │
//! │  └──────────────────┘          │ ["Alice","Bob",  │            │
//! │                                │  "Carol"]        │            │
//! │                                └──────────────────┘            │
//! │                                                                 │
//! │  Mapping Strategy:                                              │
//! │  • Primitive fields → Direct column mapping                     │
//! │  • Nested objects → Flattened with dot notation                │
//! │  • Arrays → Repeated columns with indices                       │
//! │  • Nulls → Validity bitmap                                      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Benefits
//!
//! - **Vectorized Operations**: SIMD-friendly columnar data
//! - **Compression**: Better compression ratios for homogeneous data
//! - **Cache Efficiency**: Access only needed columns
//! - **Predicate Pushdown**: Filter before materialization

use std::collections::HashMap;
use std::sync::Arc;

use crate::toon::{ToonRow, ToonSchema, ToonTable, ToonType, ToonValue};
use crate::{Result, ToonDBError};

/// Column data type for columnar storage
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    String,
    Binary,
    /// Nested structure (flattened)
    Struct(Vec<(String, Box<ColumnType>)>),
    /// List of elements
    List(Box<ColumnType>),
}

impl ColumnType {
    /// Convert from ToonType
    pub fn from_toon_type(toon_type: &ToonType) -> Self {
        match toon_type {
            ToonType::Bool => ColumnType::Bool,
            ToonType::Int => ColumnType::Int64,
            ToonType::UInt => ColumnType::UInt64,
            ToonType::Float => ColumnType::Float64,
            ToonType::Text => ColumnType::String,
            ToonType::Binary => ColumnType::Binary,
            ToonType::Array(inner) => ColumnType::List(Box::new(Self::from_toon_type(inner))),
            ToonType::Object(fields) => {
                let struct_fields: Vec<_> = fields
                    .iter()
                    .map(|(name, ty)| (name.clone(), Box::new(Self::from_toon_type(ty))))
                    .collect();
                ColumnType::Struct(struct_fields)
            }
            ToonType::Null => ColumnType::Int64, // Null represented as nullable int64
            ToonType::Ref(_) => ColumnType::String, // References stored as strings
            ToonType::Optional(inner) => Self::from_toon_type(inner), // Optional uses nullable columns
        }
    }

    /// Byte size per element (for fixed-size types)
    pub fn byte_size(&self) -> Option<usize> {
        match self {
            ColumnType::Bool => Some(1),
            ColumnType::Int8 | ColumnType::UInt8 => Some(1),
            ColumnType::Int16 | ColumnType::UInt16 => Some(2),
            ColumnType::Int32 | ColumnType::UInt32 | ColumnType::Float32 => Some(4),
            ColumnType::Int64 | ColumnType::UInt64 | ColumnType::Float64 => Some(8),
            _ => None, // Variable-size types
        }
    }
}

/// Columnar storage for a single column
#[derive(Debug, Clone)]
pub struct Column {
    /// Column name (may include dot notation for nested)
    pub name: String,
    /// Column data type
    pub dtype: ColumnType,
    /// Raw data buffer
    pub data: ColumnData,
    /// Validity bitmap (1 bit per value, 1 = valid, 0 = null)
    pub validity: Option<Vec<u8>>,
    /// Number of values
    pub len: usize,
}

/// Column data storage
#[derive(Debug, Clone)]
pub enum ColumnData {
    Bool(Vec<bool>),
    Int64(Vec<i64>),
    UInt64(Vec<u64>),
    Float64(Vec<f64>),
    String(Vec<String>),
    Binary(Vec<Vec<u8>>),
    /// Offsets for nested/list data
    Offsets(Vec<u32>),
}

impl Column {
    /// Create new empty column
    pub fn new(name: impl Into<String>, dtype: ColumnType) -> Self {
        let data = match &dtype {
            ColumnType::Bool => ColumnData::Bool(Vec::new()),
            ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64 => {
                ColumnData::Int64(Vec::new())
            }
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                ColumnData::UInt64(Vec::new())
            }
            ColumnType::Float32 | ColumnType::Float64 => ColumnData::Float64(Vec::new()),
            ColumnType::String => ColumnData::String(Vec::new()),
            ColumnType::Binary => ColumnData::Binary(Vec::new()),
            ColumnType::Struct(_) | ColumnType::List(_) => ColumnData::Offsets(Vec::new()),
        };

        Self {
            name: name.into(),
            dtype,
            data,
            validity: None,
            len: 0,
        }
    }

    /// Append a value
    pub fn push(&mut self, value: &ToonValue) {
        match (&mut self.data, value) {
            (ColumnData::Bool(v), ToonValue::Bool(b)) => v.push(*b),
            (ColumnData::Int64(v), ToonValue::Int(i)) => v.push(*i),
            (ColumnData::UInt64(v), ToonValue::UInt(u)) => v.push(*u),
            (ColumnData::Float64(v), ToonValue::Float(f)) => v.push(*f),
            (ColumnData::String(v), ToonValue::Text(s)) => v.push(s.clone()),
            (ColumnData::Binary(v), ToonValue::Binary(b)) => v.push(b.clone()),
            (ColumnData::Int64(v), ToonValue::Null) => {
                v.push(0);
                self.set_null(self.len);
            }
            (ColumnData::UInt64(v), ToonValue::Null) => {
                v.push(0);
                self.set_null(self.len);
            }
            (ColumnData::Float64(v), ToonValue::Null) => {
                v.push(0.0);
                self.set_null(self.len);
            }
            (ColumnData::String(v), ToonValue::Null) => {
                v.push(String::new());
                self.set_null(self.len);
            }
            _ => {} // Type mismatch - skip
        }
        self.len += 1;
    }

    /// Set a value as null
    fn set_null(&mut self, idx: usize) {
        if self.validity.is_none() {
            // Initialize validity bitmap with all valid (1s)
            let bytes_needed = (self.len + 8) / 8;
            self.validity = Some(vec![0xFF; bytes_needed]);
        }

        if let Some(ref mut bitmap) = self.validity {
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;

            // Ensure bitmap is large enough
            while bitmap.len() <= byte_idx {
                bitmap.push(0xFF);
            }

            // Clear the bit (set to null)
            bitmap[byte_idx] &= !(1 << bit_idx);
        }
    }

    /// Check if value at index is null
    pub fn is_null(&self, idx: usize) -> bool {
        match &self.validity {
            None => false,
            Some(bitmap) => {
                let byte_idx = idx / 8;
                let bit_idx = idx % 8;
                if byte_idx >= bitmap.len() {
                    false
                } else {
                    (bitmap[byte_idx] & (1 << bit_idx)) == 0
                }
            }
        }
    }

    /// Get value at index as ToonValue
    pub fn get(&self, idx: usize) -> Option<ToonValue> {
        if idx >= self.len {
            return None;
        }

        if self.is_null(idx) {
            return Some(ToonValue::Null);
        }

        match &self.data {
            ColumnData::Bool(v) => v.get(idx).map(|b| ToonValue::Bool(*b)),
            ColumnData::Int64(v) => v.get(idx).map(|i| ToonValue::Int(*i)),
            ColumnData::UInt64(v) => v.get(idx).map(|u| ToonValue::UInt(*u)),
            ColumnData::Float64(v) => v.get(idx).map(|f| ToonValue::Float(*f)),
            ColumnData::String(v) => v.get(idx).map(|s| ToonValue::Text(s.clone())),
            ColumnData::Binary(v) => v.get(idx).map(|b| ToonValue::Binary(b.clone())),
            ColumnData::Offsets(_) => None, // Nested types need special handling
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let data_size = match &self.data {
            ColumnData::Bool(v) => v.len(),
            ColumnData::Int64(v) => v.len() * 8,
            ColumnData::UInt64(v) => v.len() * 8,
            ColumnData::Float64(v) => v.len() * 8,
            ColumnData::String(v) => v.iter().map(|s| s.len()).sum(),
            ColumnData::Binary(v) => v.iter().map(|b| b.len()).sum(),
            ColumnData::Offsets(v) => v.len() * 4,
        };

        let validity_size = self.validity.as_ref().map_or(0, |v| v.len());
        data_size + validity_size
    }
}

/// Columnar representation of a TOON table
#[derive(Debug, Clone)]
pub struct ColumnarTable {
    /// Table name
    pub name: String,
    /// Columns by name
    pub columns: HashMap<String, Column>,
    /// Column order (for reconstruction)
    pub column_order: Vec<String>,
    /// Number of rows
    pub row_count: usize,
}

impl ColumnarTable {
    /// Create empty columnar table from schema
    pub fn from_schema(schema: &ToonSchema) -> Self {
        let mut columns = HashMap::new();
        let mut column_order = Vec::new();

        for field in &schema.fields {
            let dtype = ColumnType::from_toon_type(&field.field_type);
            let column = Column::new(&field.name, dtype);
            column_order.push(field.name.clone());
            columns.insert(field.name.clone(), column);
        }

        Self {
            name: schema.name.clone(),
            columns,
            column_order,
            row_count: 0,
        }
    }

    /// Append a row
    pub fn push_row(&mut self, row: &ToonRow) {
        for (i, col_name) in self.column_order.iter().enumerate() {
            if let Some(column) = self.columns.get_mut(col_name) {
                if let Some(value) = row.values.get(i) {
                    column.push(value);
                } else {
                    column.push(&ToonValue::Null);
                }
            }
        }
        self.row_count += 1;
    }

    /// Get a row by index
    pub fn get_row(&self, idx: usize) -> Option<ToonRow> {
        if idx >= self.row_count {
            return None;
        }

        let values: Vec<ToonValue> = self
            .column_order
            .iter()
            .filter_map(|col_name| self.columns.get(col_name)?.get(idx))
            .collect();

        Some(ToonRow::new(values))
    }

    /// Get column by name
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    /// Total memory usage
    pub fn memory_usage(&self) -> usize {
        self.columns.values().map(|c| c.memory_usage()).sum()
    }
}

/// Schema bridge for converting between TOON and columnar formats
pub struct SchemaBridge {
    /// Cached schema mappings
    schema_cache: HashMap<String, Arc<ColumnMapping>>,
}

/// Mapping between TOON schema and columnar schema
#[derive(Debug, Clone)]
pub struct ColumnMapping {
    /// Source TOON schema
    pub source_schema: ToonSchema,
    /// Column types for each field
    pub column_types: Vec<(String, ColumnType)>,
    /// Nested field mappings (for flattening)
    pub nested_mappings: HashMap<String, Vec<String>>,
}

impl ColumnMapping {
    /// Create mapping from TOON schema
    pub fn from_schema(schema: &ToonSchema) -> Self {
        let mut column_types = Vec::new();
        let mut nested_mappings = HashMap::new();

        for field in &schema.fields {
            let dtype = ColumnType::from_toon_type(&field.field_type);

            // Handle nested structures by flattening
            if let ColumnType::Struct(fields) = &dtype {
                let mut nested_cols = Vec::new();
                for (nested_name, nested_type) in fields {
                    let full_name = format!("{}.{}", field.name, nested_name);
                    column_types.push((full_name.clone(), (**nested_type).clone()));
                    nested_cols.push(full_name);
                }
                nested_mappings.insert(field.name.clone(), nested_cols);
            } else {
                column_types.push((field.name.clone(), dtype));
            }
        }

        Self {
            source_schema: schema.clone(),
            column_types,
            nested_mappings,
        }
    }

    /// Get flattened column names
    pub fn column_names(&self) -> Vec<&str> {
        self.column_types.iter().map(|(n, _)| n.as_str()).collect()
    }
}

impl SchemaBridge {
    /// Create new schema bridge
    pub fn new() -> Self {
        Self {
            schema_cache: HashMap::new(),
        }
    }

    /// Register a schema and get its mapping
    pub fn register_schema(&mut self, schema: &ToonSchema) -> Arc<ColumnMapping> {
        if let Some(existing) = self.schema_cache.get(&schema.name) {
            return Arc::clone(existing);
        }

        let mapping = Arc::new(ColumnMapping::from_schema(schema));
        self.schema_cache
            .insert(schema.name.clone(), Arc::clone(&mapping));
        mapping
    }

    /// Convert TOON table to columnar format
    pub fn to_columnar(&self, table: &ToonTable) -> Result<ColumnarTable> {
        let mut columnar = ColumnarTable::from_schema(&table.schema);

        for row in &table.rows {
            columnar.push_row(row);
        }

        Ok(columnar)
    }

    /// Convert columnar table back to TOON format
    pub fn from_columnar(
        &self,
        columnar: &ColumnarTable,
        schema: &ToonSchema,
    ) -> Result<ToonTable> {
        let mut table = ToonTable::new(schema.clone());

        for i in 0..columnar.row_count {
            if let Some(row) = columnar.get_row(i) {
                table.push(row);
            }
        }

        Ok(table)
    }

    /// Project specific columns from columnar table
    pub fn project(&self, columnar: &ColumnarTable, columns: &[&str]) -> Result<ColumnarTable> {
        let mut projected = ColumnarTable {
            name: columnar.name.clone(),
            columns: HashMap::new(),
            column_order: Vec::new(),
            row_count: columnar.row_count,
        };

        for col_name in columns {
            if let Some(column) = columnar.columns.get(*col_name) {
                projected
                    .columns
                    .insert(col_name.to_string(), column.clone());
                projected.column_order.push(col_name.to_string());
            } else {
                return Err(ToonDBError::InvalidArgument(format!(
                    "Column '{}' not found",
                    col_name
                )));
            }
        }

        Ok(projected)
    }

    /// Filter columnar table by predicate on a column
    pub fn filter<F>(
        &self,
        columnar: &ColumnarTable,
        column: &str,
        predicate: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(&ToonValue) -> bool,
    {
        let col = columnar.columns.get(column).ok_or_else(|| {
            ToonDBError::InvalidArgument(format!("Column '{}' not found", column))
        })?;

        let mut matching_indices = Vec::new();
        for i in 0..col.len {
            if let Some(value) = col.get(i)
                && predicate(&value)
            {
                matching_indices.push(i);
            }
        }

        Ok(matching_indices)
    }
}

impl Default for SchemaBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for columnar operations
#[derive(Debug, Default)]
pub struct ColumnarStats {
    pub tables_converted: u64,
    pub rows_processed: u64,
    pub columns_projected: u64,
    pub filters_applied: u64,
    pub bytes_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema() -> ToonSchema {
        ToonSchema::new("users")
            .field("id", ToonType::UInt)
            .field("name", ToonType::Text)
            .field("age", ToonType::Int)
    }

    fn create_test_table() -> ToonTable {
        let schema = create_test_schema();
        let mut table = ToonTable::new(schema);

        table.push(ToonRow::new(vec![
            ToonValue::UInt(1),
            ToonValue::Text("Alice".into()),
            ToonValue::Int(30),
        ]));
        table.push(ToonRow::new(vec![
            ToonValue::UInt(2),
            ToonValue::Text("Bob".into()),
            ToonValue::Int(25),
        ]));
        table.push(ToonRow::new(vec![
            ToonValue::UInt(3),
            ToonValue::Text("Carol".into()),
            ToonValue::Int(35),
        ]));

        table
    }

    #[test]
    fn test_column_type_conversion() {
        assert_eq!(
            ColumnType::from_toon_type(&ToonType::Int),
            ColumnType::Int64
        );
        assert_eq!(
            ColumnType::from_toon_type(&ToonType::Text),
            ColumnType::String
        );
        assert_eq!(
            ColumnType::from_toon_type(&ToonType::Bool),
            ColumnType::Bool
        );
    }

    #[test]
    fn test_column_push_and_get() {
        let mut col = Column::new("test", ColumnType::Int64);

        col.push(&ToonValue::Int(10));
        col.push(&ToonValue::Int(20));
        col.push(&ToonValue::Int(30));

        assert_eq!(col.len, 3);
        assert_eq!(col.get(0), Some(ToonValue::Int(10)));
        assert_eq!(col.get(1), Some(ToonValue::Int(20)));
        assert_eq!(col.get(2), Some(ToonValue::Int(30)));
        assert_eq!(col.get(3), None);
    }

    #[test]
    fn test_column_null_handling() {
        let mut col = Column::new("test", ColumnType::Int64);

        col.push(&ToonValue::Int(10));
        col.push(&ToonValue::Null);
        col.push(&ToonValue::Int(30));

        assert!(!col.is_null(0));
        assert!(col.is_null(1));
        assert!(!col.is_null(2));

        assert_eq!(col.get(0), Some(ToonValue::Int(10)));
        assert_eq!(col.get(1), Some(ToonValue::Null));
        assert_eq!(col.get(2), Some(ToonValue::Int(30)));
    }

    #[test]
    fn test_columnar_table_from_schema() {
        let schema = create_test_schema();
        let columnar = ColumnarTable::from_schema(&schema);

        assert_eq!(columnar.name, "users");
        assert_eq!(columnar.columns.len(), 3);
        assert!(columnar.columns.contains_key("id"));
        assert!(columnar.columns.contains_key("name"));
        assert!(columnar.columns.contains_key("age"));
    }

    #[test]
    fn test_toon_to_columnar_conversion() {
        let table = create_test_table();
        let bridge = SchemaBridge::new();

        let columnar = bridge.to_columnar(&table).unwrap();

        assert_eq!(columnar.row_count, 3);

        let id_col = columnar.column("id").unwrap();
        assert_eq!(id_col.get(0), Some(ToonValue::UInt(1)));
        assert_eq!(id_col.get(1), Some(ToonValue::UInt(2)));
        assert_eq!(id_col.get(2), Some(ToonValue::UInt(3)));

        let name_col = columnar.column("name").unwrap();
        assert_eq!(name_col.get(0), Some(ToonValue::Text("Alice".into())));
    }

    #[test]
    fn test_columnar_to_toon_roundtrip() {
        let original = create_test_table();
        let bridge = SchemaBridge::new();

        let columnar = bridge.to_columnar(&original).unwrap();
        let restored = bridge.from_columnar(&columnar, &original.schema).unwrap();

        assert_eq!(restored.rows.len(), original.rows.len());

        for (i, row) in restored.rows.iter().enumerate() {
            assert_eq!(row.values, original.rows[i].values);
        }
    }

    #[test]
    fn test_column_projection() {
        let table = create_test_table();
        let bridge = SchemaBridge::new();

        let columnar = bridge.to_columnar(&table).unwrap();
        let projected = bridge.project(&columnar, &["id", "name"]).unwrap();

        assert_eq!(projected.columns.len(), 2);
        assert!(projected.columns.contains_key("id"));
        assert!(projected.columns.contains_key("name"));
        assert!(!projected.columns.contains_key("age"));
    }

    #[test]
    fn test_column_filter() {
        let table = create_test_table();
        let bridge = SchemaBridge::new();

        let columnar = bridge.to_columnar(&table).unwrap();

        // Filter for age > 28
        let matches = bridge
            .filter(&columnar, "age", |v| match v {
                ToonValue::Int(age) => *age > 28,
                _ => false,
            })
            .unwrap();

        assert_eq!(matches, vec![0, 2]); // Alice (30) and Carol (35)
    }

    #[test]
    fn test_schema_mapping() {
        let schema = create_test_schema();
        let mapping = ColumnMapping::from_schema(&schema);

        assert_eq!(mapping.column_types.len(), 3);
        assert_eq!(mapping.column_names(), vec!["id", "name", "age"]);
    }

    #[test]
    fn test_memory_usage() {
        let table = create_test_table();
        let bridge = SchemaBridge::new();

        let columnar = bridge.to_columnar(&table).unwrap();
        let usage = columnar.memory_usage();

        // Should have some memory allocated
        assert!(usage > 0);
    }

    #[test]
    fn test_get_row() {
        let table = create_test_table();
        let bridge = SchemaBridge::new();

        let columnar = bridge.to_columnar(&table).unwrap();

        let row0 = columnar.get_row(0).unwrap();
        assert_eq!(row0.values[0], ToonValue::UInt(1));
        assert_eq!(row0.values[1], ToonValue::Text("Alice".into()));
        assert_eq!(row0.values[2], ToonValue::Int(30));

        assert!(columnar.get_row(100).is_none());
    }

    #[test]
    fn test_column_type_byte_size() {
        assert_eq!(ColumnType::Bool.byte_size(), Some(1));
        assert_eq!(ColumnType::Int64.byte_size(), Some(8));
        assert_eq!(ColumnType::Float64.byte_size(), Some(8));
        assert_eq!(ColumnType::String.byte_size(), None);
    }

    #[test]
    fn test_schema_bridge_caching() {
        let schema = create_test_schema();
        let mut bridge = SchemaBridge::new();

        let mapping1 = bridge.register_schema(&schema);
        let mapping2 = bridge.register_schema(&schema);

        // Should return same Arc (cached)
        assert!(Arc::ptr_eq(&mapping1, &mapping2));
    }

    #[test]
    fn test_invalid_column_projection() {
        let table = create_test_table();
        let bridge = SchemaBridge::new();

        let columnar = bridge.to_columnar(&table).unwrap();
        let result = bridge.project(&columnar, &["nonexistent"]);

        assert!(result.is_err());
    }
}
