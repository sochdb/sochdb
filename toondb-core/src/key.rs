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

//! Key types for ToonDB indexing

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::cmp::Ordering;
use std::io::Result as IoResult;

/// Composite key for temporal ordering
///
/// Primary: timestamp_us (microseconds for temporal queries)
/// Secondary: record_id (for uniqueness within same timestamp)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporalKey {
    pub timestamp_us: u64,
    pub edge_id: u128,
}

impl TemporalKey {
    pub fn new(timestamp_us: u64, edge_id: u128) -> Self {
        Self {
            timestamp_us,
            edge_id,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.write_u64::<LittleEndian>(self.timestamp_us).unwrap();
        buf.write_u128::<LittleEndian>(self.edge_id).unwrap();
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> IoResult<Self> {
        let mut cursor = bytes;
        let timestamp_us = cursor.read_u64::<LittleEndian>()?;
        let edge_id = cursor.read_u128::<LittleEndian>()?;
        Ok(Self {
            timestamp_us,
            edge_id,
        })
    }
}

impl PartialOrd for TemporalKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TemporalKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp_us
            .cmp(&other.timestamp_us)
            .then_with(|| self.edge_id.cmp(&other.edge_id))
    }
}

/// Causal key for graph traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalKey {
    pub parent_id: u128,
    pub child_id: u128,
}

impl CausalKey {
    pub fn new(parent_id: u128, child_id: u128) -> Self {
        Self {
            parent_id,
            child_id,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.write_u128::<LittleEndian>(self.parent_id).unwrap();
        buf.write_u128::<LittleEndian>(self.child_id).unwrap();
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> IoResult<Self> {
        let mut cursor = bytes;
        let parent_id = cursor.read_u128::<LittleEndian>()?;
        let child_id = cursor.read_u128::<LittleEndian>()?;
        Ok(Self {
            parent_id,
            child_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_key_ordering() {
        let k1 = TemporalKey::new(100, 1);
        let k2 = TemporalKey::new(100, 2);
        let k3 = TemporalKey::new(200, 1);

        assert!(k1 < k2);
        assert!(k2 < k3);
        assert!(k1 < k3);
    }

    #[test]
    fn test_temporal_key_serialization() {
        let key = TemporalKey::new(12345, 67890);
        let bytes = key.to_bytes();
        let decoded = TemporalKey::from_bytes(&bytes).unwrap();
        assert_eq!(key, decoded);
    }
}
