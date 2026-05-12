// SPDX-License-Identifier: AGPL-3.0-or-later

//! Core types for the Volcano executor.

use crate::soch_ql::SochValue;

/// A single row: positional values matching the schema column order.
pub type Row = Vec<SochValue>;

/// Column metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMeta {
    /// Column name.
    pub name: String,
    /// Source table (if known).
    pub table: Option<String>,
}

impl ColumnMeta {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), table: None }
    }

    pub fn qualified(table: impl Into<String>, name: impl Into<String>) -> Self {
        Self { name: name.into(), table: Some(table.into()) }
    }
}

/// Schema: ordered list of columns defining the row layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub columns: Vec<ColumnMeta>,
}

impl Schema {
    pub fn new(columns: Vec<ColumnMeta>) -> Self {
        Self { columns }
    }

    pub fn empty() -> Self {
        Self { columns: vec![] }
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Find column index by name (unqualified lookup).
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Find column index by qualified name (table.column).
    pub fn index_of_qualified(&self, table: Option<&str>, name: &str) -> Option<usize> {
        match table {
            Some(t) => self.columns.iter().position(|c| {
                c.name == name && c.table.as_deref() == Some(t)
            }),
            None => self.index_of(name),
        }
    }

    /// Column names as strings.
    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    /// Merge two schemas (for joins).
    pub fn merge(&self, other: &Schema) -> Schema {
        let mut cols = self.columns.clone();
        cols.extend(other.columns.iter().cloned());
        Schema::new(cols)
    }
}
