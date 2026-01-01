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

//! TOON Format Codec
//!
//! This module implements the TOON (Token-Optimized Object Notation) format
//! specification using the official `toon-format` crate.
//!
//! ## TOON Format Grammar (Simplified)
//!
//! ```text
//! document     ::= top_level_value
//! value        ::= simple_object | array | primitive
//! simple_object::= (key ":" value newline)+ 
//! array        ::= header newline item*
//! header       ::= name "[" count "]" ( "{" fields "}" )? ":"
//! item         ::= "-" value newline | row newline
//! ```

use crate::toon::{ToonValue}; // Use shared types from toon.rs
use std::collections::HashMap;
use toon_format::{self, EncodeOptions, DecodeOptions, Delimiter, Indent};
use toon_format::types::KeyFoldingMode;

// ============================================================================
// TOON Value Types
// ============================================================================

/// TOON value type tags for binary encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ToonTypeTag {
    /// Null value
    Null = 0x00,
    /// Boolean false
    False = 0x01,
    /// Boolean true  
    True = 0x02,
    /// Positive fixint (0-15, embedded in lower nibble: 0x10-0x1F)
    PosFixint = 0x10,
    /// Negative fixint (-16 to -1, embedded: 0x20-0x2F)
    NegFixint = 0x20,
    /// 8-bit signed integer
    Int8 = 0x30,
    /// 16-bit signed integer
    Int16 = 0x31,
    /// 32-bit signed integer
    Int32 = 0x32,
    /// 64-bit signed integer
    Int64 = 0x33,
    /// 32-bit float
    Float32 = 0x40,
    /// 64-bit float
    Float64 = 0x41,
    /// Fixed-length string (length in lower 4 bits: 0x50-0x5F, 0-15 chars)
    FixStr = 0x50,
    /// String with 8-bit length prefix
    Str8 = 0x60,
    /// String with 16-bit length prefix
    Str16 = 0x61,
    /// String with 32-bit length prefix
    Str32 = 0x62,
    /// Array
    Array = 0x70,
    /// Reference to another table row
    Ref = 0x80,
    /// Object (Map)
    Object = 0x90,
    /// Binary data
    Binary = 0xA0,
    /// Unsigned Integer (varint)
    UInt = 0xB0,
}

// ============================================================================
// TOON Document Structure
// ============================================================================

/// TOON document
#[derive(Debug, Clone)]
pub struct ToonDocument {
    /// Root value
    pub root: ToonValue,
    /// Schema version
    pub version: u32,
}

impl ToonDocument {
    /// Create a new TOON document from a value
    pub fn new(root: ToonValue) -> Self {
        Self {
            root,
            version: 1,
        }
    }

    /// Create a table-like document (legacy helper)
    pub fn new_table(_name: impl Into<String>, fields: Vec<String>, rows: Vec<Vec<ToonValue>>) -> Self {
        // Convert to Array of Objects for canonical representation
        let fields_str: Vec<String> = fields;
        let mut array = Vec::new();
        for row in rows {
            let mut obj = HashMap::new();
            for (i, val) in row.into_iter().enumerate() {
                if i < fields_str.len() {
                    obj.insert(fields_str[i].clone(), val);
                }
            }
            array.push(ToonValue::Object(obj));
        }
        
        Self {
            root: ToonValue::Array(array),
            version: 1,
        }
    }
}

// ============================================================================
// Text Format (Human-Readable)
// ============================================================================

/// TOON text format encoder (wraps toon-format crate)
pub struct ToonTextEncoder;

impl ToonTextEncoder {
    /// Encode a document to TOON text format
    pub fn encode(doc: &ToonDocument) -> String {
        // Use default options for now, can be sophisticated later
        let options = EncodeOptions::new()
            .with_indent(Indent::Spaces(2))
            .with_delimiter(Delimiter::Comma)
            .with_key_folding(KeyFoldingMode::Safe);
        
        // Use toon_format to encode the ToonValue
        // ToonValue implements Serialize, so this works directly.
        toon_format::encode(&doc.root, &options).unwrap_or_else(|e| format!("Error encoding TOON: {}", e))
    }
}

/// TOON text format decoder/parser (wraps toon-format crate)
pub struct ToonTextParser;

impl ToonTextParser {
    pub fn parse(input: &str) -> Result<ToonDocument, ToonParseError> {
         Self::parse_with_options(input, DecodeOptions::default())
    }
    
    pub fn parse_with_options(input: &str, options: DecodeOptions) -> Result<ToonDocument, ToonParseError> {
        let root: ToonValue = toon_format::decode(input, &options)
            .map_err(|e| ToonParseError::RowError { line: 0, cause: e.to_string() })?;
            
        Ok(ToonDocument::new(root))
    }
    
    // Legacy helper kept for compatibility if needed, but useless now
    pub fn parse_header(_line: &str) -> Result<(String, usize, Vec<String>), ToonParseError> {
        Err(ToonParseError::InvalidHeader)
    }
}

/// Token counter (dummy implementation for now)
pub struct ToonTokenCounter;
impl ToonTokenCounter {
    pub fn count(_doc: &ToonDocument) -> usize {
        0
    }
}


/// Parse error types
#[derive(Debug, Clone)]
pub enum ToonParseError {
    EmptyInput,
    InvalidHeader,
    InvalidRowCount,
    InvalidValue,
    RowCountMismatch { expected: usize, actual: usize },
    FieldCountMismatch { expected: usize, actual: usize },
    RowError { line: usize, cause: String },
}

impl std::fmt::Display for ToonParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for ToonParseError {}


// ============================================================================
// Binary Format (Compact)
// ============================================================================

/// TOON binary format magic bytes
pub const TOON_MAGIC: [u8; 4] = [0x54, 0x4F, 0x4F, 0x4E]; // "TOON"

/// TOON binary codec (Renamed from ToonBinaryCodec to ToonDbBinaryCodec)
pub struct ToonDbBinaryCodec;

impl ToonDbBinaryCodec {
    /// Encode a document to binary format
    pub fn encode(doc: &ToonDocument) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&TOON_MAGIC);
        // Version
        Self::write_varint(&mut buf, doc.version as u64);
        // Root value
        Self::write_value(&mut buf, &doc.root);
        // Checksum
        let checksum = crc32fast::hash(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Decode binary format to document
    pub fn decode(data: &[u8]) -> Result<ToonDocument, ToonParseError> {
        if data.len() < 8 { return Err(ToonParseError::InvalidHeader); }
        if data[0..4] != TOON_MAGIC { return Err(ToonParseError::InvalidHeader); }
        
        // Verify checksum
        let stored_checksum = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        let computed_checksum = crc32fast::hash(&data[..data.len() - 4]);
        if stored_checksum != computed_checksum { return Err(ToonParseError::InvalidValue); }
        
        let data = &data[..data.len() - 4];
        let mut cursor = 4;
        
        let (version, bytes) = Self::read_varint(&data[cursor..])?;
        cursor += bytes;
        
        let (root, _) = Self::read_value(&data[cursor..])?;
        
        Ok(ToonDocument {
            root,
            version: version as u32,
        })
    }
    
    fn write_varint(buf: &mut Vec<u8>, mut n: u64) {
        while n > 127 {
            buf.push((n as u8 & 0x7F) | 0x80);
            n >>= 7;
        }
        buf.push(n as u8 & 0x7F);
    }
    
    fn read_varint(data: &[u8]) -> Result<(u64, usize), ToonParseError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut i = 0;
        while i < data.len() {
            let byte = data[i];
            result |= ((byte & 0x7F) as u64) << shift;
            i += 1;
            if byte & 0x80 == 0 { return Ok((result, i)); }
            shift += 7;
        }
        Err(ToonParseError::InvalidValue)
    }

    fn read_string(data: &[u8]) -> Result<(String, usize), ToonParseError> {
        let (len, varint_bytes) = Self::read_varint(data)?;
        let len = len as usize;
        if data.len() < varint_bytes + len { return Err(ToonParseError::InvalidValue); }
        let s = std::str::from_utf8(&data[varint_bytes..varint_bytes+len]).map_err(|_| ToonParseError::InvalidValue)?.to_string();
        Ok((s, varint_bytes + len))
    }
    
    fn write_value(buf: &mut Vec<u8>, value: &ToonValue) {
        match value {
            ToonValue::Null => buf.push(ToonTypeTag::Null as u8),
            ToonValue::Bool(true) => buf.push(ToonTypeTag::True as u8),
            ToonValue::Bool(false) => buf.push(ToonTypeTag::False as u8),
            ToonValue::Int(n) => {
                 // Optimization: FixInts
                 buf.push(ToonTypeTag::Int64 as u8);
                 buf.extend_from_slice(&n.to_le_bytes());
            },
            ToonValue::UInt(n) => {
                 buf.push(ToonTypeTag::UInt as u8);
                 Self::write_varint(buf, *n);
            },
            ToonValue::Float(f) => {
                 buf.push(ToonTypeTag::Float64 as u8);
                 buf.extend_from_slice(&f.to_le_bytes());
            },
            ToonValue::Text(s) => {
                 buf.push(ToonTypeTag::Str32 as u8);
                 let bytes = s.as_bytes();
                 buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                 buf.extend_from_slice(bytes);
            },
            ToonValue::Binary(b) => {
                 buf.push(ToonTypeTag::Binary as u8);
                 Self::write_varint(buf, b.len() as u64);
                 buf.extend_from_slice(b);
            },
            ToonValue::Array(arr) => {
                 buf.push(ToonTypeTag::Array as u8);
                 Self::write_varint(buf, arr.len() as u64);
                 for item in arr { Self::write_value(buf, item); }
            },
            ToonValue::Object(map) => {
                 buf.push(ToonTypeTag::Object as u8);
                 Self::write_varint(buf, map.len() as u64);
                 for (k, v) in map {
                     // Key string
                     let k_bytes = k.as_bytes();
                     Self::write_varint(buf, k_bytes.len() as u64);
                     buf.extend_from_slice(k_bytes);
                     // Value
                     Self::write_value(buf, v);
                 }
            },
            ToonValue::Ref { table, id } => {
                 buf.push(ToonTypeTag::Ref as u8);
                 // table name
                 let t_bytes = table.as_bytes();
                 Self::write_varint(buf, t_bytes.len() as u64);
                 buf.extend_from_slice(t_bytes);
                 // id
                 Self::write_varint(buf, *id);
            }
        }
    }
    
    fn read_value(data: &[u8]) -> Result<(ToonValue, usize), ToonParseError> {
        if data.is_empty() { return Err(ToonParseError::InvalidValue); }
        let tag = data[0];
        let mut cursor = 1;
        
        match tag {
            0x00 => Ok((ToonValue::Null, 1)),
            0x01 => Ok((ToonValue::Bool(false), 1)),
            0x02 => Ok((ToonValue::Bool(true), 1)),
            0x33 => { // Int64
                 if data.len() < cursor + 8 { return Err(ToonParseError::InvalidValue); }
                 let n = i64::from_le_bytes(data[cursor..cursor+8].try_into().unwrap());
                 Ok((ToonValue::Int(n), cursor+8))
            },
            0x41 => { // Float64
                 if data.len() < cursor + 8 { return Err(ToonParseError::InvalidValue); }
                 let f = f64::from_le_bytes(data[cursor..cursor+8].try_into().unwrap());
                 Ok((ToonValue::Float(f), cursor+8))
            },
            0x62 => { // Str32
                 if data.len() < cursor + 4 { return Err(ToonParseError::InvalidValue); }
                 let len = u32::from_le_bytes(data[cursor..cursor+4].try_into().unwrap()) as usize;
                 cursor += 4;
                 if data.len() < cursor + len { return Err(ToonParseError::InvalidValue); }
                 let s = std::str::from_utf8(&data[cursor..cursor+len]).unwrap().to_string();
                 Ok((ToonValue::Text(s), cursor+len))
            },
            0x70 => { // Array
                 let (len, bytes) = Self::read_varint(&data[cursor..])?;
                 cursor += bytes;
                 let mut arr = Vec::new();
                 for _ in 0..len {
                     let (val, bytes_read) = Self::read_value(&data[cursor..])?;
                     cursor += bytes_read;
                     arr.push(val);
                 }
                 Ok((ToonValue::Array(arr), cursor))
            },
            0xB0 => { // UInt
                 let (n, bytes) = Self::read_varint(&data[cursor..])?;
                 Ok((ToonValue::UInt(n), cursor+bytes))
            },
            0x80 => { // Ref
                 let (table, table_bytes) = Self::read_string(&data[cursor..])?;
                 cursor += table_bytes;
                 let (id, id_bytes) = Self::read_varint(&data[cursor..])?;
                 Ok((ToonValue::Ref { table, id }, cursor+id_bytes))
            },
            0x90 => { // Object
                 let (len, bytes_read) = Self::read_varint(&data[cursor..])?;
                 cursor += bytes_read;
                 let mut map = HashMap::new();
                 for _ in 0..len {
                     let (k, k_bytes) = Self::read_string(&data[cursor..])?;
                     cursor += k_bytes;
                     let (v, v_bytes) = Self::read_value(&data[cursor..])?;
                     cursor += v_bytes;
                     map.insert(k, v);
                 }
                 Ok((ToonValue::Object(map), cursor))
            },
            // Add other cases as needed
            _ => Err(ToonParseError::InvalidValue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_object() {
        let mut obj = HashMap::new();
        obj.insert("id".to_string(), ToonValue::Int(1));
        obj.insert("name".to_string(), ToonValue::Text("Alice".to_string()));
        let doc = ToonDocument::new(ToonValue::Object(obj));
        
        // This test now uses canonical encoder
        let encoded = ToonTextEncoder::encode(&doc);
        // Canonical output might differ slightly (e.g. sorting), but should contain keys
        assert!(encoded.contains("id"));
        assert!(encoded.contains("1"));
        assert!(encoded.contains("name"));
        assert!(encoded.contains("Alice"));
        
        // Roundtrip binary with new codec name
        let bin = ToonDbBinaryCodec::encode(&doc);
        let decoded = ToonDbBinaryCodec::decode(&bin).unwrap();
        if let ToonValue::Object(map) = decoded.root {
             // Accessing values. Note: ToonValue doesn't impl PartialEq against literal ints easily matching on variant needed
             // Use string representation or direct match
            assert_eq!(map.get("id"), Some(&ToonValue::Int(1)));
            assert_eq!(map.get("name"), Some(&ToonValue::Text("Alice".to_string())));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_array() {
        let arr = vec![
            ToonValue::Int(1),
            ToonValue::Int(2),
        ];
        let doc = ToonDocument::new(ToonValue::Array(arr));
        
        let encoded = ToonTextEncoder::encode(&doc);
        // Should contain values
        assert!(encoded.contains("1"));
        assert!(encoded.contains("2"));
        
        let bin = ToonDbBinaryCodec::encode(&doc);
        let decoded = ToonDbBinaryCodec::decode(&bin).unwrap();
        if let ToonValue::Array(arr) = decoded.root {
             assert_eq!(arr.len(), 2);
             assert_eq!(arr[0], ToonValue::Int(1));
        } else {
            panic!("Expected array");
        }
    }
}
