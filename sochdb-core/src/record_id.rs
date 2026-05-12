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

//! Record ID — the universal document/record identifier for SochDB's multi-model layer.
//!
//! A `RecordId` combines a table name with a unique identifier within that table,
//! modeled after SurrealDB's `thing` concept (`table:id`).
//!
//! # Binary Key Encoding
//!
//! ```text
//! [table_id: u32 BE][id_bytes: variable]
//! ```
//!
//! - `table_id` is a 4-byte big-endian table hash (FNV-1a) for sort-ordered prefix scans.
//! - `id_bytes` is the raw identifier — either a big-endian u64 for integer IDs, or
//!   UTF-8 bytes for string IDs.
//!
//! Big-endian encoding ensures lexicographic byte ordering matches numeric ordering,
//! enabling efficient range scans on the underlying KV store.
//!
//! # Display Format
//!
//! `table:id` — e.g. `person:1`, `post:abc`, `user:⟨uuid⟩`
//!
//! # Examples
//!
//! ```
//! use sochdb_core::record_id::RecordId;
//!
//! // Integer ID
//! let rid = RecordId::new("person", 42u64);
//! assert_eq!(rid.table(), "person");
//! assert_eq!(rid.to_string(), "person:42");
//!
//! // String ID
//! let rid = RecordId::from_string("post", "hello-world");
//! assert_eq!(rid.to_string(), "post:hello-world");
//!
//! // Round-trip through binary key
//! let key = rid.to_key();
//! let decoded = RecordId::from_key_with_table(&key, "post").unwrap();
//! assert_eq!(rid, decoded);
//! ```

use std::fmt;

/// The identifier part of a RecordId.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IdValue {
    /// Integer identifier (stored as big-endian u64).
    Integer(u64),
    /// String identifier (stored as UTF-8 bytes).
    String(String),
}

impl IdValue {
    /// Encode the id value to bytes (for use in storage keys).
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            IdValue::Integer(n) => n.to_be_bytes().to_vec(),
            IdValue::String(s) => s.as_bytes().to_vec(),
        }
    }
}

impl fmt::Display for IdValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdValue::Integer(n) => write!(f, "{}", n),
            IdValue::String(s) => write!(f, "{}", s),
        }
    }
}

/// Tag byte prefixed to id_bytes in the binary key to distinguish integer vs string.
const ID_TAG_INTEGER: u8 = 0x01;
const ID_TAG_STRING: u8 = 0x02;

/// A `RecordId` is a `(table, id)` pair that uniquely identifies a record
/// across SochDB's multi-model storage.
///
/// It replaces the previous string-based `node_id` / `from_id` / `to_id` pattern
/// used in the graph overlay, providing:
/// - Type-safe table scoping
/// - Compact binary keys for storage
/// - Sort-ordered prefix scans per table
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordId {
    table: String,
    id: IdValue,
}

impl RecordId {
    /// Create a RecordId with an integer identifier.
    pub fn new(table: impl Into<String>, id: u64) -> Self {
        Self {
            table: table.into(),
            id: IdValue::Integer(id),
        }
    }

    /// Create a RecordId with a string identifier.
    pub fn from_string(table: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            id: IdValue::String(id.into()),
        }
    }

    /// Create a RecordId from an IdValue.
    pub fn with_id(table: impl Into<String>, id: IdValue) -> Self {
        Self {
            table: table.into(),
            id,
        }
    }

    /// Table name.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// The identifier value.
    pub fn id(&self) -> &IdValue {
        &self.id
    }

    /// Compute the FNV-1a hash of the table name (used as table_id in binary keys).
    fn table_hash(table: &str) -> u32 {
        // FNV-1a 32-bit
        let mut hash: u32 = 0x811c9dc5;
        for byte in table.as_bytes() {
            hash ^= *byte as u32;
            hash = hash.wrapping_mul(0x01000193);
        }
        hash
    }

    /// Encode to a binary storage key.
    ///
    /// Format: `[table_id: u32 BE][tag: u8][id_bytes]`
    ///
    /// The table_id is a FNV-1a hash, ensuring records of the same table
    /// cluster together in lexicographic key order.
    pub fn to_key(&self) -> Vec<u8> {
        let table_id = Self::table_hash(&self.table);
        let id_bytes = self.id.to_bytes();
        let tag = match &self.id {
            IdValue::Integer(_) => ID_TAG_INTEGER,
            IdValue::String(_) => ID_TAG_STRING,
        };
        let mut key = Vec::with_capacity(4 + 1 + id_bytes.len());
        key.extend_from_slice(&table_id.to_be_bytes());
        key.push(tag);
        key.extend_from_slice(&id_bytes);
        key
    }

    /// Decode from a binary storage key.
    ///
    /// Note: The table name is NOT recoverable from the key alone (only the hash is stored).
    /// Use `from_key_with_table` if you know the table name, or `from_key` for a lossy decode.
    pub fn from_key(key: &[u8]) -> Option<Self> {
        if key.len() < 6 {
            // 4 (table_id) + 1 (tag) + 1 (min id)
            return None;
        }
        let _table_id = u32::from_be_bytes([key[0], key[1], key[2], key[3]]);
        let tag = key[4];
        let id_bytes = &key[5..];

        let id = match tag {
            ID_TAG_INTEGER => {
                if id_bytes.len() != 8 {
                    return None;
                }
                let n = u64::from_be_bytes([
                    id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3],
                    id_bytes[4], id_bytes[5], id_bytes[6], id_bytes[7],
                ]);
                IdValue::Integer(n)
            }
            ID_TAG_STRING => {
                let s = std::str::from_utf8(id_bytes).ok()?;
                IdValue::String(s.to_string())
            }
            _ => return None,
        };

        // Table name is lost in key encoding — use hash placeholder
        Some(RecordId {
            table: format!("#{:08x}", _table_id),
            id,
        })
    }

    /// Decode from a binary key when the table name is known.
    pub fn from_key_with_table(key: &[u8], table: &str) -> Option<Self> {
        if key.len() < 6 {
            return None;
        }
        let stored_hash = u32::from_be_bytes([key[0], key[1], key[2], key[3]]);
        if stored_hash != Self::table_hash(table) {
            return None; // Hash mismatch
        }
        let tag = key[4];
        let id_bytes = &key[5..];

        let id = match tag {
            ID_TAG_INTEGER => {
                if id_bytes.len() != 8 {
                    return None;
                }
                let n = u64::from_be_bytes([
                    id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3],
                    id_bytes[4], id_bytes[5], id_bytes[6], id_bytes[7],
                ]);
                IdValue::Integer(n)
            }
            ID_TAG_STRING => {
                let s = std::str::from_utf8(id_bytes).ok()?;
                IdValue::String(s.to_string())
            }
            _ => return None,
        };

        Some(RecordId {
            table: table.to_string(),
            id,
        })
    }

    /// Generate the key prefix for all records in a given table.
    ///
    /// Useful for prefix scans: `storage.scan(RecordId::table_prefix("person"))`.
    pub fn table_prefix(table: &str) -> Vec<u8> {
        Self::table_hash(table).to_be_bytes().to_vec()
    }

    /// Parse from `table:id` string format.
    ///
    /// Supports:
    /// - `person:42` → integer ID
    /// - `post:hello-world` → string ID
    pub fn parse(s: &str) -> Option<Self> {
        let colon_pos = s.find(':')?;
        if colon_pos == 0 || colon_pos == s.len() - 1 {
            return None;
        }
        let table = &s[..colon_pos];
        let id_str = &s[colon_pos + 1..];

        let id = if let Ok(n) = id_str.parse::<u64>() {
            IdValue::Integer(n)
        } else {
            IdValue::String(id_str.to_string())
        };

        Some(RecordId {
            table: table.to_string(),
            id,
        })
    }
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.table, self.id)
    }
}

impl PartialOrd for RecordId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecordId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by binary key for consistent ordering with storage
        self.to_key().cmp(&other.to_key())
    }
}

// ============================================================================
// Serde support (feature-gated for optional use)
// ============================================================================

impl serde::Serialize for RecordId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for RecordId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        RecordId::parse(&s).ok_or_else(|| serde::de::Error::custom(
            format!("invalid RecordId: '{}' (expected table:id)", s),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_id_integer() {
        let rid = RecordId::new("person", 42);
        assert_eq!(rid.table(), "person");
        assert_eq!(rid.to_string(), "person:42");
        assert!(matches!(rid.id(), IdValue::Integer(42)));
    }

    #[test]
    fn test_record_id_string() {
        let rid = RecordId::from_string("post", "hello-world");
        assert_eq!(rid.table(), "post");
        assert_eq!(rid.to_string(), "post:hello-world");
    }

    #[test]
    fn test_record_id_binary_key_roundtrip_integer() {
        let rid = RecordId::new("person", 42);
        let key = rid.to_key();
        assert_eq!(key.len(), 4 + 1 + 8); // table_id + tag + u64
        let decoded = RecordId::from_key_with_table(&key, "person").unwrap();
        assert_eq!(rid, decoded);
    }

    #[test]
    fn test_record_id_binary_key_roundtrip_string() {
        let rid = RecordId::from_string("post", "abc");
        let key = rid.to_key();
        assert_eq!(key.len(), 4 + 1 + 3); // table_id + tag + "abc"
        let decoded = RecordId::from_key_with_table(&key, "post").unwrap();
        assert_eq!(rid, decoded);
    }

    #[test]
    fn test_record_id_table_prefix() {
        let rid1 = RecordId::new("person", 1);
        let rid2 = RecordId::new("person", 999);
        let prefix = RecordId::table_prefix("person");

        let key1 = rid1.to_key();
        let key2 = rid2.to_key();

        assert_eq!(&key1[..4], &prefix);
        assert_eq!(&key2[..4], &prefix);
    }

    #[test]
    fn test_record_id_ordering() {
        let r1 = RecordId::new("person", 1);
        let r2 = RecordId::new("person", 2);
        let r3 = RecordId::new("person", 100);

        // Same table: ordered by ID
        assert!(r1 < r2);
        assert!(r2 < r3);
    }

    #[test]
    fn test_record_id_parse() {
        let rid = RecordId::parse("person:42").unwrap();
        assert_eq!(rid.table(), "person");
        assert!(matches!(rid.id(), IdValue::Integer(42)));

        let rid = RecordId::parse("post:hello-world").unwrap();
        assert_eq!(rid.table(), "post");
        assert!(matches!(rid.id(), IdValue::String(s) if s == "hello-world"));

        assert!(RecordId::parse("").is_none());
        assert!(RecordId::parse(":42").is_none());
        assert!(RecordId::parse("person:").is_none());
    }

    #[test]
    fn test_record_id_serde_roundtrip() {
        let rid = RecordId::new("person", 42);
        let json = serde_json::to_string(&rid).unwrap();
        assert_eq!(json, "\"person:42\"");
        let decoded: RecordId = serde_json::from_str(&json).unwrap();
        assert_eq!(rid, decoded);
    }

    #[test]
    fn test_record_id_hash_mismatch() {
        let rid = RecordId::new("person", 42);
        let key = rid.to_key();
        // Try decoding with wrong table name
        assert!(RecordId::from_key_with_table(&key, "animal").is_none());
    }

    #[test]
    fn test_record_id_different_tables_cluster() {
        let person_prefix = RecordId::table_prefix("person");
        let post_prefix = RecordId::table_prefix("post");
        // Different tables have different prefixes (extremely high probability)
        assert_ne!(person_prefix, post_prefix);
    }

    #[test]
    fn test_record_id_from_key_lossy() {
        let rid = RecordId::new("person", 42);
        let key = rid.to_key();
        let decoded = RecordId::from_key(&key).unwrap();
        // Table name is lost — replaced with hash placeholder
        assert!(decoded.table().starts_with('#'));
        assert!(matches!(decoded.id(), IdValue::Integer(42)));
    }
}
