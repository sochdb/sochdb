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

//! Binary edge key encoding for the graph overlay.
//!
//! Replaces the previous UTF-8 string key format with compact binary keys
//! that support efficient prefix scans and maintain lexicographic ordering.
//!
//! # Key Layout
//!
//! All keys start with a one-byte tag that identifies the key type:
//!
//! | Tag  | Kind          | Format                                                        |
//! |------|---------------|---------------------------------------------------------------|
//! | 0x01 | Node          | `[0x01][ns_hash: 4B][table_hash: 4B][tag: 1B][id_bytes]`     |
//! | 0x02 | Edge          | `[0x02][ns_hash: 4B][from_key...][et_hash: 4B][to_key...]`   |
//! | 0x03 | Reverse Index | `[0x03][ns_hash: 4B][et_hash: 4B][to_key...][from_key...]`   |
//!
//! Where `from_key` / `to_key` are `[table_hash: 4B][tag: 1B][id_bytes]` (the RecordId key).
//!
//! The namespace hash ensures graphs in different namespaces never collide.
//! All hashes are FNV-1a 32-bit, big-endian encoded for sort ordering.

use crate::record_id::RecordId;
use crate::soch::SochValue;

/// Tag bytes for key type discrimination.
const TAG_NODE: u8 = 0x01;
const TAG_EDGE: u8 = 0x02;
const TAG_REVERSE: u8 = 0x03;

/// Length-prefixed sub-key: `[len: u16 BE][bytes]`.
/// Used to delimit variable-length RecordId keys within edge keys.
fn write_length_prefixed(buf: &mut Vec<u8>, data: &[u8]) {
    assert!(data.len() <= u16::MAX as usize, "sub-key too long");
    buf.extend_from_slice(&(data.len() as u16).to_be_bytes());
    buf.extend_from_slice(data);
}

/// Read a length-prefixed sub-key, returning `(bytes, rest)`.
fn read_length_prefixed(data: &[u8]) -> Option<(&[u8], &[u8])> {
    if data.len() < 2 {
        return None;
    }
    let len = u16::from_be_bytes([data[0], data[1]]) as usize;
    let rest = &data[2..];
    if rest.len() < len {
        return None;
    }
    Some((&rest[..len], &rest[len..]))
}

/// FNV-1a 32-bit hash (same as RecordId::table_hash, duplicated to avoid coupling).
fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    for &b in bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// Build a node storage key.
///
/// Format: `[0x01][ns_hash: 4B][record_id_key]`
pub fn node_key(namespace: &str, record_id: &RecordId) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let rid_key = record_id.to_key();
    let mut key = Vec::with_capacity(1 + 4 + rid_key.len());
    key.push(TAG_NODE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    key.extend_from_slice(&rid_key);
    key
}

/// Build a node prefix for scanning all nodes in a namespace.
///
/// Format: `[0x01][ns_hash: 4B]`
pub fn node_prefix(namespace: &str) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let mut key = Vec::with_capacity(5);
    key.push(TAG_NODE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    key
}

/// Build a node prefix for scanning all nodes of a specific table in a namespace.
///
/// Format: `[0x01][ns_hash: 4B][table_hash: 4B]`
pub fn node_table_prefix(namespace: &str, table: &str) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let tbl_prefix = RecordId::table_prefix(table);
    let mut key = Vec::with_capacity(1 + 4 + tbl_prefix.len());
    key.push(TAG_NODE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    key.extend_from_slice(&tbl_prefix);
    key
}

/// Build an edge storage key.
///
/// Format: `[0x02][ns_hash: 4B][from_key_len: 2B][from_key][et_hash: 4B][to_key_len: 2B][to_key]`
pub fn edge_key(
    namespace: &str,
    from_id: &RecordId,
    edge_type: &str,
    to_id: &RecordId,
) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let et_hash = fnv1a_32(edge_type.as_bytes());
    let from_key = from_id.to_key();
    let to_key = to_id.to_key();
    let mut key = Vec::with_capacity(1 + 4 + 2 + from_key.len() + 4 + 2 + to_key.len());
    key.push(TAG_EDGE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    write_length_prefixed(&mut key, &from_key);
    key.extend_from_slice(&et_hash.to_be_bytes());
    write_length_prefixed(&mut key, &to_key);
    key
}

/// Build an edge prefix for scanning all edges from a node.
///
/// Format: `[0x02][ns_hash: 4B][from_key_len: 2B][from_key]`
pub fn edge_from_prefix(namespace: &str, from_id: &RecordId) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let from_key = from_id.to_key();
    let mut key = Vec::with_capacity(1 + 4 + 2 + from_key.len());
    key.push(TAG_EDGE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    write_length_prefixed(&mut key, &from_key);
    key
}

/// Build an edge prefix for scanning edges of a specific type from a node.
///
/// Format: `[0x02][ns_hash: 4B][from_key_len: 2B][from_key][et_hash: 4B]`
pub fn edge_from_type_prefix(
    namespace: &str,
    from_id: &RecordId,
    edge_type: &str,
) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let et_hash = fnv1a_32(edge_type.as_bytes());
    let from_key = from_id.to_key();
    let mut key = Vec::with_capacity(1 + 4 + 2 + from_key.len() + 4);
    key.push(TAG_EDGE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    write_length_prefixed(&mut key, &from_key);
    key.extend_from_slice(&et_hash.to_be_bytes());
    key
}

/// Build an edge prefix for scanning all edges in a namespace.
///
/// Format: `[0x02][ns_hash: 4B]`
pub fn edge_prefix(namespace: &str) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let mut key = Vec::with_capacity(5);
    key.push(TAG_EDGE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    key
}

/// Build a reverse index key.
///
/// Format: `[0x03][ns_hash: 4B][et_hash: 4B][to_key_len: 2B][to_key][from_key_len: 2B][from_key]`
pub fn reverse_key(
    namespace: &str,
    edge_type: &str,
    to_id: &RecordId,
    from_id: &RecordId,
) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let et_hash = fnv1a_32(edge_type.as_bytes());
    let to_key = to_id.to_key();
    let from_key = from_id.to_key();
    let mut key = Vec::with_capacity(1 + 4 + 4 + 2 + to_key.len() + 2 + from_key.len());
    key.push(TAG_REVERSE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    key.extend_from_slice(&et_hash.to_be_bytes());
    write_length_prefixed(&mut key, &to_key);
    write_length_prefixed(&mut key, &from_key);
    key
}

/// Build a reverse index prefix for all edges of a given type pointing to a node.
///
/// Format: `[0x03][ns_hash: 4B][et_hash: 4B][to_key_len: 2B][to_key]`
pub fn reverse_type_to_prefix(
    namespace: &str,
    edge_type: &str,
    to_id: &RecordId,
) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let et_hash = fnv1a_32(edge_type.as_bytes());
    let to_key = to_id.to_key();
    let mut key = Vec::with_capacity(1 + 4 + 4 + 2 + to_key.len());
    key.push(TAG_REVERSE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    key.extend_from_slice(&et_hash.to_be_bytes());
    write_length_prefixed(&mut key, &to_key);
    key
}

/// Build a reverse index prefix for all reverse entries in a namespace.
///
/// Format: `[0x03][ns_hash: 4B]`
pub fn reverse_prefix(namespace: &str) -> Vec<u8> {
    let ns_hash = fnv1a_32(namespace.as_bytes());
    let mut key = Vec::with_capacity(5);
    key.push(TAG_REVERSE);
    key.extend_from_slice(&ns_hash.to_be_bytes());
    key
}

/// Decoded edge key components.
#[derive(Debug, Clone)]
pub struct DecodedEdgeKey {
    pub from_key: Vec<u8>,
    pub edge_type_hash: u32,
    pub to_key: Vec<u8>,
}

/// Decode an edge key (tag 0x02) into its components.
///
/// Returns None if the key is malformed or not an edge key.
pub fn decode_edge_key(key: &[u8]) -> Option<DecodedEdgeKey> {
    if key.is_empty() || key[0] != TAG_EDGE {
        return None;
    }
    let rest = &key[1..]; // skip tag
    if rest.len() < 4 {
        return None;
    }
    let rest = &rest[4..]; // skip ns_hash

    let (from_key, rest) = read_length_prefixed(rest)?;
    if rest.len() < 4 {
        return None;
    }
    let et_hash = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
    let rest = &rest[4..];
    let (to_key, _rest) = read_length_prefixed(rest)?;

    Some(DecodedEdgeKey {
        from_key: from_key.to_vec(),
        edge_type_hash: et_hash,
        to_key: to_key.to_vec(),
    })
}

/// Decoded reverse index key components.
#[derive(Debug, Clone)]
pub struct DecodedReverseKey {
    pub edge_type_hash: u32,
    pub to_key: Vec<u8>,
    pub from_key: Vec<u8>,
}

/// Decode a reverse index key (tag 0x03) into its components.
pub fn decode_reverse_key(key: &[u8]) -> Option<DecodedReverseKey> {
    if key.is_empty() || key[0] != TAG_REVERSE {
        return None;
    }
    let rest = &key[1..];
    if rest.len() < 4 {
        return None;
    }
    let rest = &rest[4..]; // skip ns_hash
    if rest.len() < 4 {
        return None;
    }
    let et_hash = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
    let rest = &rest[4..];
    let (to_key, rest) = read_length_prefixed(rest)?;
    let (from_key, _rest) = read_length_prefixed(rest)?;

    Some(DecodedReverseKey {
        edge_type_hash: et_hash,
        to_key: to_key.to_vec(),
        from_key: from_key.to_vec(),
    })
}

/// Encode a `HashMap<String, SochValue>` as a compact binary value.
///
/// Format: `[num_entries: u32 BE] { [key_len: u16 BE][key_utf8][value_json_len: u32 BE][value_json] }*`
///
/// Uses JSON for individual SochValues as a pragmatic choice — the hot path is keys,
/// and values are typically small property bags. A future optimization can replace this
/// with PackedRow encoding without changing the key format.
pub fn encode_properties(props: &std::collections::HashMap<String, SochValue>) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(props.len() as u32).to_be_bytes());
    for (k, v) in props {
        let key_bytes = k.as_bytes();
        buf.extend_from_slice(&(key_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(key_bytes);
        // Encode SochValue as JSON for now (pragmatic; PackedRow upgrade later)
        let val_json = serde_json::to_vec(v).unwrap_or_default();
        buf.extend_from_slice(&(val_json.len() as u32).to_be_bytes());
        buf.extend_from_slice(&val_json);
    }
    buf
}

/// Decode a `HashMap<String, SochValue>` from compact binary encoding.
pub fn decode_properties(data: &[u8]) -> Option<std::collections::HashMap<String, SochValue>> {
    if data.len() < 4 {
        return None;
    }
    let num = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut offset = 4;
    let mut map = std::collections::HashMap::with_capacity(num);

    for _ in 0..num {
        if offset + 2 > data.len() {
            return None;
        }
        let key_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + key_len > data.len() {
            return None;
        }
        let key = std::str::from_utf8(&data[offset..offset + key_len]).ok()?.to_string();
        offset += key_len;

        if offset + 4 > data.len() {
            return None;
        }
        let val_len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
        offset += 4;
        if offset + val_len > data.len() {
            return None;
        }
        let val: SochValue = serde_json::from_slice(&data[offset..offset + val_len]).ok()?;
        offset += val_len;

        map.insert(key, val);
    }

    Some(map)
}

/// Encode a node value (node_type + properties) as binary.
///
/// Format: `[type_len: u16 BE][type_utf8][properties_bytes]`
pub fn encode_node_value(node_type: &str, props: &std::collections::HashMap<String, SochValue>) -> Vec<u8> {
    let type_bytes = node_type.as_bytes();
    let props_bytes = encode_properties(props);
    let mut buf = Vec::with_capacity(2 + type_bytes.len() + props_bytes.len());
    buf.extend_from_slice(&(type_bytes.len() as u16).to_be_bytes());
    buf.extend_from_slice(type_bytes);
    buf.extend_from_slice(&props_bytes);
    buf
}

/// Decode a node value into (node_type, properties).
pub fn decode_node_value(data: &[u8]) -> Option<(String, std::collections::HashMap<String, SochValue>)> {
    if data.len() < 2 {
        return None;
    }
    let type_len = u16::from_be_bytes([data[0], data[1]]) as usize;
    if data.len() < 2 + type_len {
        return None;
    }
    let node_type = std::str::from_utf8(&data[2..2 + type_len]).ok()?.to_string();
    let props = decode_properties(&data[2 + type_len..])?;
    Some((node_type, props))
}

/// Encode an edge value (from_table, from_id_display, edge_type, to_table, to_id_display + properties).
///
/// Format: `[edge_type_len: u16 BE][edge_type_utf8][from_rid_str_len: u16 BE][from_rid_str][to_rid_str_len: u16 BE][to_rid_str][properties_bytes]`
///
/// We store the full RecordId display strings so we can reconstitute them on read
/// without needing another lookup.
pub fn encode_edge_value(
    from_id: &RecordId,
    edge_type: &str,
    to_id: &RecordId,
    props: &std::collections::HashMap<String, SochValue>,
) -> Vec<u8> {
    let et_bytes = edge_type.as_bytes();
    let from_str = from_id.to_string();
    let from_bytes = from_str.as_bytes();
    let to_str = to_id.to_string();
    let to_bytes = to_str.as_bytes();
    let props_bytes = encode_properties(props);

    let mut buf = Vec::with_capacity(2 + et_bytes.len() + 2 + from_bytes.len() + 2 + to_bytes.len() + props_bytes.len());
    buf.extend_from_slice(&(et_bytes.len() as u16).to_be_bytes());
    buf.extend_from_slice(et_bytes);
    buf.extend_from_slice(&(from_bytes.len() as u16).to_be_bytes());
    buf.extend_from_slice(from_bytes);
    buf.extend_from_slice(&(to_bytes.len() as u16).to_be_bytes());
    buf.extend_from_slice(to_bytes);
    buf.extend_from_slice(&props_bytes);
    buf
}

/// Edge value decoded components.
#[derive(Debug, Clone)]
pub struct DecodedEdgeValue {
    pub edge_type: String,
    pub from_id: RecordId,
    pub to_id: RecordId,
    pub properties: std::collections::HashMap<String, SochValue>,
}

/// Decode an edge value.
pub fn decode_edge_value(data: &[u8]) -> Option<DecodedEdgeValue> {
    let mut offset = 0;

    // edge_type
    if offset + 2 > data.len() { return None; }
    let et_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    offset += 2;
    if offset + et_len > data.len() { return None; }
    let edge_type = std::str::from_utf8(&data[offset..offset + et_len]).ok()?.to_string();
    offset += et_len;

    // from_id
    if offset + 2 > data.len() { return None; }
    let from_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    offset += 2;
    if offset + from_len > data.len() { return None; }
    let from_str = std::str::from_utf8(&data[offset..offset + from_len]).ok()?;
    let from_id = RecordId::parse(from_str)?;
    offset += from_len;

    // to_id
    if offset + 2 > data.len() { return None; }
    let to_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    offset += 2;
    if offset + to_len > data.len() { return None; }
    let to_str = std::str::from_utf8(&data[offset..offset + to_len]).ok()?;
    let to_id = RecordId::parse(to_str)?;
    offset += to_len;

    // properties
    let props = decode_properties(&data[offset..])?;

    Some(DecodedEdgeValue {
        edge_type,
        from_id,
        to_id,
        properties: props,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // SochValue is already imported via `use crate::soch::SochValue` in the parent module

    #[test]
    fn test_node_key_format() {
        let rid = RecordId::new("person", 42);
        let key = node_key("agent_001", &rid);
        assert_eq!(key[0], TAG_NODE);
        // Namespace hash occupies bytes 1..5
        // RecordId key follows
        assert!(key.len() > 5);
    }

    #[test]
    fn test_node_prefix_is_prefix_of_node_key() {
        let rid = RecordId::new("person", 42);
        let key = node_key("agent_001", &rid);
        let prefix = node_prefix("agent_001");
        assert!(key.starts_with(&prefix));
    }

    #[test]
    fn test_edge_key_roundtrip() {
        let from = RecordId::new("user", 1);
        let to = RecordId::new("conv", 100);
        let key = edge_key("ns", &from, "STARTED", &to);
        let decoded = decode_edge_key(&key).unwrap();
        assert_eq!(decoded.from_key, from.to_key());
        assert_eq!(decoded.to_key, to.to_key());
        assert_eq!(decoded.edge_type_hash, fnv1a_32(b"STARTED"));
    }

    #[test]
    fn test_edge_from_prefix_is_prefix() {
        let from = RecordId::new("user", 1);
        let to = RecordId::new("conv", 100);
        let key = edge_key("ns", &from, "SENT", &to);
        let prefix = edge_from_prefix("ns", &from);
        assert!(key.starts_with(&prefix));
    }

    #[test]
    fn test_edge_from_type_prefix_is_prefix() {
        let from = RecordId::new("user", 1);
        let to = RecordId::new("conv", 100);
        let key = edge_key("ns", &from, "SENT", &to);
        let prefix = edge_from_type_prefix("ns", &from, "SENT");
        assert!(key.starts_with(&prefix));
    }

    #[test]
    fn test_reverse_key_roundtrip() {
        let from = RecordId::new("user", 1);
        let to = RecordId::new("msg", 42);
        let key = reverse_key("ns", "SENT", &to, &from);
        let decoded = decode_reverse_key(&key).unwrap();
        assert_eq!(decoded.from_key, from.to_key());
        assert_eq!(decoded.to_key, to.to_key());
        assert_eq!(decoded.edge_type_hash, fnv1a_32(b"SENT"));
    }

    #[test]
    fn test_reverse_prefix_is_prefix() {
        let from = RecordId::new("user", 1);
        let to = RecordId::new("msg", 42);
        let key = reverse_key("ns", "SENT", &to, &from);
        let prefix = reverse_type_to_prefix("ns", "SENT", &to);
        assert!(key.starts_with(&prefix));
    }

    #[test]
    fn test_encode_decode_properties() {
        let mut props = std::collections::HashMap::new();
        props.insert("name".to_string(), SochValue::Text("Alice".to_string()));
        props.insert("age".to_string(), SochValue::Int(30));
        props.insert("active".to_string(), SochValue::Bool(true));

        let encoded = encode_properties(&props);
        let decoded = decode_properties(&encoded).unwrap();

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded.get("name"), Some(&SochValue::Text("Alice".to_string())));
        assert_eq!(decoded.get("age"), Some(&SochValue::Int(30)));
        assert_eq!(decoded.get("active"), Some(&SochValue::Bool(true)));
    }

    #[test]
    fn test_encode_decode_node_value() {
        let mut props = std::collections::HashMap::new();
        props.insert("email".to_string(), SochValue::Text("a@b.com".to_string()));

        let encoded = encode_node_value("User", &props);
        let (node_type, decoded_props) = decode_node_value(&encoded).unwrap();
        assert_eq!(node_type, "User");
        assert_eq!(decoded_props.get("email"), Some(&SochValue::Text("a@b.com".to_string())));
    }

    #[test]
    fn test_encode_decode_edge_value() {
        let from = RecordId::new("user", 1);
        let to = RecordId::from_string("conv", "abc");
        let mut props = std::collections::HashMap::new();
        props.insert("weight".to_string(), SochValue::Float(0.95));

        let encoded = encode_edge_value(&from, "STARTED", &to, &props);
        let decoded = decode_edge_value(&encoded).unwrap();
        assert_eq!(decoded.edge_type, "STARTED");
        assert_eq!(decoded.from_id, from);
        assert_eq!(decoded.to_id, to);
        assert_eq!(decoded.properties.get("weight"), Some(&SochValue::Float(0.95)));
    }

    #[test]
    fn test_empty_properties() {
        let props = std::collections::HashMap::new();
        let encoded = encode_properties(&props);
        let decoded = decode_properties(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_different_namespaces_produce_different_keys() {
        let rid = RecordId::new("person", 1);
        let key1 = node_key("ns_a", &rid);
        let key2 = node_key("ns_b", &rid);
        assert_ne!(key1, key2);
    }
}
