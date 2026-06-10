//! Object-storage-native cold tier for immutable index segments.
//!
//! Delta (in-memory mutable) + sealed immutable segments in S3/GCS with local
//! NVMe cache. Restart reads manifest only — no full corpus rebuild.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObjectStoreError {
    #[error("segment not found: {0}")]
    SegmentNotFound(String),
    #[error("manifest error: {0}")]
    Manifest(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentDescriptor {
    pub segment_id: String,
    pub object_uri: String,
    pub checksum: u64,
    pub doc_count: u64,
    pub byte_size: u64,
    pub sealed_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStoreTierConfig {
    pub bucket_uri: String,
    pub local_cache_dir: PathBuf,
    pub seal_threshold_docs: usize,
    pub max_cache_segments: usize,
}

impl Default for ObjectStoreTierConfig {
    fn default() -> Self {
        Self {
            bucket_uri: "s3://sochdb-segments".to_string(),
            local_cache_dir: PathBuf::from("./sochdb-cache"),
            seal_threshold_docs: 10_000,
            max_cache_segments: 64,
        }
    }
}

/// LSM-style tier: in-memory delta + immutable object-store segments.
pub struct ObjectStoreTier {
    config: ObjectStoreTierConfig,
    delta_docs: HashMap<u64, String>,
    sealed_segments: Vec<SegmentDescriptor>,
    cache_hits: u64,
    cache_misses: u64,
}

impl ObjectStoreTier {
    pub fn new(config: ObjectStoreTierConfig) -> Self {
        Self {
            config,
            delta_docs: HashMap::new(),
            sealed_segments: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn insert_delta(&mut self, doc_id: u64, text: String) {
        self.delta_docs.insert(doc_id, text);
        if self.delta_docs.len() >= self.config.seal_threshold_docs {
            let _ = self.seal_current_delta();
        }
    }

    pub fn seal_current_delta(&mut self) -> Result<SegmentDescriptor, ObjectStoreError> {
        if self.delta_docs.is_empty() {
            return Err(ObjectStoreError::SegmentNotFound("empty delta".into()));
        }
        let segment_id = format!("seg-{}", self.sealed_segments.len());
        let doc_count = self.delta_docs.len() as u64;
        let byte_size: u64 = self.delta_docs.values().map(|t| t.len() as u64).sum();
        let mut payload_bytes = Vec::new();
        for (k, v) in &self.delta_docs {
            payload_bytes.extend_from_slice(format!("{k}:{v}").as_bytes());
        }
        let checksum = crc32fast::hash(&payload_bytes) as u64;

        let desc = SegmentDescriptor {
            segment_id: segment_id.clone(),
            object_uri: format!("{}/{}", self.config.bucket_uri, segment_id),
            checksum,
            doc_count,
            byte_size,
            sealed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        // Persist to local cache (object upload would happen async in production)
        std::fs::create_dir_all(&self.config.local_cache_dir)?;
        let cache_path = self
            .config
            .local_cache_dir
            .join(format!("{segment_id}.seg"));
        let payload = serde_json::to_string(&self.delta_docs).map_err(|e| {
            ObjectStoreError::Manifest(e.to_string())
        })?;
        std::fs::write(&cache_path, payload)?;

        self.sealed_segments.push(desc.clone());
        self.delta_docs.clear();
        Ok(desc)
    }

    pub fn hydrate_from_manifest(&mut self, manifest_path: &Path) -> Result<(), ObjectStoreError> {
        if !manifest_path.exists() {
            return Ok(());
        }
        let data = std::fs::read_to_string(manifest_path)?;
        let segments: Vec<SegmentDescriptor> = serde_json::from_str(&data)
            .map_err(|e| ObjectStoreError::Manifest(e.to_string()))?;
        self.sealed_segments = segments;
        Ok(())
    }

    pub fn lookup(&mut self, doc_id: u64) -> Option<String> {
        if let Some(t) = self.delta_docs.get(&doc_id) {
            self.cache_hits += 1;
            return Some(t.clone());
        }
        for seg in &self.sealed_segments {
            let cache_path = self
                .config
                .local_cache_dir
                .join(format!("{}.seg", seg.segment_id));
            if cache_path.exists() {
                if let Ok(data) = std::fs::read_to_string(&cache_path) {
                    if let Ok(map) = serde_json::from_str::<HashMap<u64, String>>(&data) {
                        if let Some(t) = map.get(&doc_id) {
                            self.cache_hits += 1;
                            return Some(t.clone());
                        }
                    }
                }
            }
            self.cache_misses += 1;
        }
        None
    }

    pub fn stats(&self) -> (u64, u64, usize, usize) {
        (
            self.cache_hits,
            self.cache_misses,
            self.delta_docs.len(),
            self.sealed_segments.len(),
        )
    }
}