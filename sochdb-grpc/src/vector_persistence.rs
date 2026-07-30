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

//! Gives the served vector indexes somewhere to survive.
//!
//! Without this, `VectorIndexServer` keeps every index in a process map, so a
//! restart silently discards them. This binds each served index to a
//! [`GenerationStore`], restores them at startup, and publishes a new
//! generation once enough vectors have accumulated.
//!
//! # What this does and does not promise
//!
//! Recovery is to the **last published generation**, not to the last
//! acknowledged insert. Vectors inserted after a checkpoint are lost on an
//! unclean stop. That is a real limit and it is stated in the capability
//! manifest rather than glossed: reaching a zero recovery point needs a
//! write-ahead log on the insert path, which is separate work. What is promised
//! here is that a restart resumes from a complete, verified, self-describing
//! generation instead of from nothing.
//!
//! Checkpointing republishes the whole index, so it is deliberately not done
//! per batch. A generation is immutable by construction, which is what makes
//! recovery trustworthy, but it also means the cost of publishing scales with
//! the index rather than with the change. Incremental delta segments are the
//! answer to that and are not implemented here.

use sochdb_index::generation::{GenerationManifest, GenerationStore};
use sochdb_index::hnsw::HnswIndex;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::proto::HnswConfig as ProtoHnswConfig;

const METRIC_KEY: &str = "metric";
const MAX_CONNECTIONS_KEY: &str = "max_connections";
const MAX_CONNECTIONS_L0_KEY: &str = "max_connections_layer0";
const EF_CONSTRUCTION_KEY: &str = "ef_construction";
const EF_SEARCH_KEY: &str = "ef_search";
const CREATED_AT_KEY: &str = "created_at";

/// How an index describes itself when it is checkpointed.
pub struct IndexDescriptor<'a> {
    /// The server's index key, `namespace:name` or bare `name`.
    pub key: &'a str,
    pub dimension: u32,
    /// `proto::DistanceMetric` as its wire integer.
    pub metric: i32,
    pub config: &'a ProtoHnswConfig,
    pub created_at: u64,
}

/// An index brought back from disk at startup.
pub struct RestoredIndex {
    pub key: String,
    pub index: HnswIndex,
    pub dimension: u32,
    pub metric: i32,
    pub config: ProtoHnswConfig,
    pub generation: u64,
    pub created_at: u64,
}

/// Durable homes for the served indexes.
pub struct VectorPersistence {
    root: PathBuf,
    checkpoint_every: u64,
}

impl VectorPersistence {
    /// Open (creating if needed) a directory holding one generation store per
    /// index.
    ///
    /// `checkpoint_every` is the number of inserted vectors after which a new
    /// generation is published. Zero means never checkpoint automatically,
    /// which is useful for a caller driving publication itself.
    pub fn open(root: impl Into<PathBuf>, checkpoint_every: u64) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            checkpoint_every,
        })
    }

    /// Generation currently published for an index, if any.
    pub fn active_generation(&self, key: &str) -> Result<Option<u64>, String> {
        let dir = self.index_dir(key);
        if !dir.exists() {
            return Ok(None);
        }
        let store = GenerationStore::open(&dir)
            .map_err(|e| format!("cannot open generation store: {e}"))?;
        store
            .active_generation()
            .map_err(|e| format!("cannot read active generation: {e}"))
    }

    pub fn checkpoint_every(&self) -> u64 {
        self.checkpoint_every
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory for one index.
    ///
    /// Index keys contain a colon and are otherwise caller-supplied, so they
    /// are hex encoded rather than used as path components. That removes any
    /// question of separators, case-insensitive collisions, reserved names, or
    /// traversal, and it is reversible. The readable name is preserved inside
    /// the manifest for anyone inspecting the directory by hand.
    fn index_dir(&self, key: &str) -> PathBuf {
        let mut encoded = String::with_capacity(key.len() * 2);
        for b in key.as_bytes() {
            encoded.push_str(&format!("{b:02x}"));
        }
        self.root.join(encoded)
    }

    fn decode_key(encoded: &str) -> Option<String> {
        if encoded.len() % 2 != 0 {
            return None;
        }
        let mut bytes = Vec::with_capacity(encoded.len() / 2);
        for pair in encoded.as_bytes().chunks(2) {
            let s = std::str::from_utf8(pair).ok()?;
            bytes.push(u8::from_str_radix(s, 16).ok()?);
        }
        String::from_utf8(bytes).ok()
    }

    /// Publish a new generation for one index.
    ///
    /// Blocking: callers on an async path should run this on a blocking thread.
    pub fn checkpoint(
        &self,
        descriptor: &IndexDescriptor<'_>,
        index: &HnswIndex,
    ) -> Result<u64, String> {
        let store = GenerationStore::open(self.index_dir(descriptor.key))
            .map_err(|e| format!("cannot open generation store: {e}"))?;

        // Generations are immutable, so the next one is always one past
        // whatever is currently active -- never a reuse of that number.
        let next = store
            .active_generation()
            .map_err(|e| format!("cannot read active generation: {e}"))?
            .map(|g| g + 1)
            .unwrap_or(1);

        let mut manifest = GenerationManifest::new(
            next,
            descriptor.key,
            descriptor.dimension,
            metric_label(descriptor.metric),
        );
        manifest.searchable_watermark = index.len() as u64;
        manifest.build_parameters = build_parameters(descriptor);

        store
            .publish_hnsw(&manifest, index)
            .map_err(|e| format!("cannot publish generation {next}: {e}"))?;
        Ok(next)
    }

    /// Forget an index entirely. Used when it is dropped.
    pub fn forget(&self, key: &str) -> io::Result<()> {
        let dir = self.index_dir(key);
        if dir.exists() {
            fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    /// Restore every index that has a published generation.
    ///
    /// Returns the indexes that loaded and, separately, the ones that did not.
    /// A store that fails verification is reported rather than skipped
    /// silently: coming back up quietly missing an index looks identical to
    /// coming back up correctly, and an operator would have no reason to look.
    #[allow(clippy::type_complexity)]
    pub fn restore_all(&self) -> io::Result<(Vec<RestoredIndex>, Vec<(String, String)>)> {
        let mut restored = Vec::new();
        let mut failed = Vec::new();

        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let raw = entry.file_name();
            let Some(encoded) = raw.to_str() else {
                continue;
            };
            let Some(key) = Self::decode_key(encoded) else {
                failed.push((
                    encoded.to_string(),
                    "directory name is not an encoded index key".to_string(),
                ));
                continue;
            };

            match self.restore_one(&key, entry.path()) {
                Ok(Some(index)) => restored.push(index),
                Ok(None) => {}
                Err(e) => failed.push((key, e)),
            }
        }

        restored.sort_by(|a, b| a.key.cmp(&b.key));
        failed.sort();
        Ok((restored, failed))
    }

    fn restore_one(&self, key: &str, dir: PathBuf) -> Result<Option<RestoredIndex>, String> {
        let store =
            GenerationStore::open(dir).map_err(|e| format!("cannot open generation store: {e}"))?;

        // Debris from a publication that never became visible is safe to remove
        // and would otherwise accumulate across every unclean stop.
        if let Err(e) = store.discard_incomplete() {
            return Err(format!("cannot clear incomplete generations: {e}"));
        }

        let Some((manifest, index)) = store
            .load_active_hnsw()
            .map_err(|e| format!("cannot load active generation: {e}"))?
        else {
            return Ok(None);
        };

        let params = &manifest.build_parameters;
        Ok(Some(RestoredIndex {
            key: key.to_string(),
            index,
            dimension: manifest.dimension,
            metric: parse_param(params, METRIC_KEY).unwrap_or(0),
            config: ProtoHnswConfig {
                max_connections: parse_param(params, MAX_CONNECTIONS_KEY).unwrap_or(0),
                max_connections_layer0: parse_param(params, MAX_CONNECTIONS_L0_KEY).unwrap_or(0),
                ef_construction: parse_param(params, EF_CONSTRUCTION_KEY).unwrap_or(0),
                ef_search: parse_param(params, EF_SEARCH_KEY).unwrap_or(0),
            },
            generation: manifest.generation,
            created_at: parse_param(params, CREATED_AT_KEY).unwrap_or(0),
        }))
    }
}

fn parse_param<T: std::str::FromStr>(params: &BTreeMap<String, String>, key: &str) -> Option<T> {
    params.get(key).and_then(|v| v.parse().ok())
}

fn build_parameters(d: &IndexDescriptor<'_>) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(METRIC_KEY.to_string(), d.metric.to_string());
    p.insert(
        MAX_CONNECTIONS_KEY.to_string(),
        d.config.max_connections.to_string(),
    );
    p.insert(
        MAX_CONNECTIONS_L0_KEY.to_string(),
        d.config.max_connections_layer0.to_string(),
    );
    p.insert(
        EF_CONSTRUCTION_KEY.to_string(),
        d.config.ef_construction.to_string(),
    );
    p.insert(EF_SEARCH_KEY.to_string(), d.config.ef_search.to_string());
    p.insert(CREATED_AT_KEY.to_string(), d.created_at.to_string());
    p
}

fn metric_label(metric: i32) -> &'static str {
    match crate::proto::DistanceMetric::try_from(metric) {
        Ok(crate::proto::DistanceMetric::L2) => "euclidean",
        Ok(crate::proto::DistanceMetric::Cosine) => "cosine",
        Ok(crate::proto::DistanceMetric::DotProduct) => "dot_product",
        _ => "unspecified",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sochdb_index::hnsw::HnswConfig;
    use tempfile::TempDir;

    fn config() -> ProtoHnswConfig {
        ProtoHnswConfig {
            max_connections: 16,
            max_connections_layer0: 32,
            ef_construction: 200,
            ef_search: 50,
        }
    }

    fn seeded(dimension: usize, count: u128) -> HnswIndex {
        let index = HnswIndex::new(dimension, HnswConfig::default());
        for id in 0..count {
            let v: Vec<f32> = (0..dimension)
                .map(|d| id as f32 + d as f32 * 0.01)
                .collect();
            index.insert(id, v).expect("insert");
        }
        index
    }

    fn descriptor<'a>(key: &'a str, config: &'a ProtoHnswConfig) -> IndexDescriptor<'a> {
        IndexDescriptor {
            key,
            dimension: 8,
            metric: crate::proto::DistanceMetric::Cosine as i32,
            config,
            created_at: 1_700_000_000,
        }
    }

    #[test]
    fn a_checkpointed_index_comes_back_with_its_settings_intact() {
        let dir = TempDir::new().unwrap();
        let p = VectorPersistence::open(dir.path(), 100).unwrap();
        let cfg = config();
        let index = seeded(8, 32);

        assert_eq!(
            p.checkpoint(&descriptor("ns:docs", &cfg), &index).unwrap(),
            1
        );

        let (restored, failed) = p.restore_all().unwrap();
        assert!(failed.is_empty(), "unexpected failures: {failed:?}");
        assert_eq!(restored.len(), 1);
        let r = &restored[0];
        assert_eq!(r.key, "ns:docs", "the readable key must survive encoding");
        assert_eq!(r.dimension, 8);
        assert_eq!(r.metric, crate::proto::DistanceMetric::Cosine as i32);
        assert_eq!(r.config.ef_construction, 200);
        assert_eq!(r.config.max_connections_layer0, 32);
        assert_eq!(r.created_at, 1_700_000_000);
        assert_eq!(r.generation, 1);
        assert_eq!(r.index.len(), 32);
    }

    /// The point of restoring at all: the recovered index must answer the same
    /// questions the same way.
    #[test]
    fn a_restored_index_answers_identically() {
        let dir = TempDir::new().unwrap();
        let p = VectorPersistence::open(dir.path(), 100).unwrap();
        let cfg = config();
        let index = seeded(8, 48);

        let query: Vec<f32> = (0..8).map(|d| 12.0 + d as f32 * 0.01).collect();
        let before: Vec<u128> = index
            .search(&query, 8)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        p.checkpoint(&descriptor("ns:docs", &cfg), &index).unwrap();
        drop(index);

        let (restored, _) = p.restore_all().unwrap();
        let after: Vec<u128> = restored[0]
            .index
            .search(&query, 8)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(after, before, "a restart changed this query's answer");
    }

    /// Generations are immutable, so a second checkpoint must take the next
    /// number rather than overwrite the live one.
    #[test]
    fn checkpointing_again_advances_the_generation() {
        let dir = TempDir::new().unwrap();
        let p = VectorPersistence::open(dir.path(), 100).unwrap();
        let cfg = config();

        let index = seeded(8, 4);
        assert_eq!(p.checkpoint(&descriptor("docs", &cfg), &index).unwrap(), 1);
        for id in 4..12u128 {
            index
                .insert(id, (0..8).map(|d| id as f32 + d as f32 * 0.01).collect())
                .unwrap();
        }
        assert_eq!(p.checkpoint(&descriptor("docs", &cfg), &index).unwrap(), 2);

        let (restored, _) = p.restore_all().unwrap();
        assert_eq!(restored[0].generation, 2);
        assert_eq!(
            restored[0].index.len(),
            12,
            "the newer generation must be the one restored"
        );
    }

    #[test]
    fn a_dropped_index_does_not_come_back() {
        let dir = TempDir::new().unwrap();
        let p = VectorPersistence::open(dir.path(), 100).unwrap();
        let cfg = config();
        let index = seeded(8, 4);
        p.checkpoint(&descriptor("gone", &cfg), &index).unwrap();
        p.checkpoint(&descriptor("kept", &cfg), &index).unwrap();

        p.forget("gone").unwrap();

        let (restored, failed) = p.restore_all().unwrap();
        assert!(failed.is_empty());
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].key, "kept");
    }

    /// Keys are caller-supplied and contain separators, so they must never be
    /// used as path components directly.
    #[test]
    fn hostile_index_keys_cannot_escape_the_root() {
        let dir = TempDir::new().unwrap();
        let p = VectorPersistence::open(dir.path(), 100).unwrap();
        let cfg = config();
        let index = seeded(8, 2);

        for key in ["../escape", "a/b/c", "ns:with spaces", "..", "CON"] {
            p.checkpoint(&descriptor(key, &cfg), &index)
                .unwrap_or_else(|e| panic!("checkpoint {key}: {e}"));
            assert!(
                p.index_dir(key).starts_with(dir.path()),
                "`{key}` produced a path outside the root"
            );
        }

        let (restored, failed) = p.restore_all().unwrap();
        assert!(failed.is_empty(), "unexpected failures: {failed:?}");
        let keys: Vec<&str> = restored.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["..", "../escape", "CON", "a/b/c", "ns:with spaces"]
        );
    }

    /// A store that cannot be verified must be reported. Coming back up quietly
    /// missing an index is indistinguishable from coming back up correctly.
    #[test]
    fn a_corrupt_store_is_reported_rather_than_silently_skipped() {
        let dir = TempDir::new().unwrap();
        let p = VectorPersistence::open(dir.path(), 100).unwrap();
        let cfg = config();
        let index = seeded(8, 4);
        p.checkpoint(&descriptor("good", &cfg), &index).unwrap();
        p.checkpoint(&descriptor("bad", &cfg), &index).unwrap();

        let seg = p
            .index_dir("bad")
            .join("generations")
            .join("00000000000000000001")
            .join("index.hnsw");
        let mut bytes = fs::read(&seg).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        fs::write(&seg, &bytes).unwrap();

        let (restored, failed) = p.restore_all().unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].key, "good");
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "bad");
        assert!(
            failed[0].1.contains("checksum"),
            "the failure must say what was wrong, got: {}",
            failed[0].1
        );
    }

    #[test]
    fn an_empty_root_restores_nothing_without_erroring() {
        let dir = TempDir::new().unwrap();
        let p = VectorPersistence::open(dir.path(), 100).unwrap();
        let (restored, failed) = p.restore_all().unwrap();
        assert!(restored.is_empty());
        assert!(failed.is_empty());
    }
}
