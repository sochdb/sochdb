//! ToonDB-based catalog implementation.
//!
//! Uses ToonDB's durable storage for unified transaction semantics.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use toondb_storage::database::{Database, DatabaseConfig};

use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::types::*;

/// Catalog for managing segment metadata using ToonDB storage
pub struct Catalog {
    db: Arc<Database>,
}

impl Catalog {
    /// Open or create a catalog database
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = DatabaseConfig {
            group_commit: true,
            ..Default::default()
        };
        let db = Database::open_with_config(path, config)
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(Self { db })
    }

    /// Open an in-memory catalog (for testing)
    pub fn open_memory() -> Result<Self> {
        let temp_dir = tempfile::tempdir().map_err(|e| Error::Storage(e.to_string()))?;
        Self::open(temp_dir.path())
    }

    /// Get current timestamp in seconds
    fn now_secs() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }

    /// Create a new collection
    pub fn create_collection(&self, name: &str, config: &EngineConfig) -> Result<i64> {
        let config_json =
            serde_json::to_string(config).map_err(|e| Error::Serialization(e.to_string()))?;

        let metric = match config.metric {
            Metric::DotProduct => "dot_product",
            Metric::Cosine => "cosine",
        };

        let txn = self.db
            .begin_transaction()
            .map_err(|e| Error::Storage(e.to_string()))?;

        // Use timestamp as ID for simplicity
        let id = Self::now_secs();

        let key = format!("collections/{}", name);
        let value = serde_json::json!({
            "id": id,
            "name": name,
            "dim": config.dim,
            "metric": metric,
            "config_json": config_json,
            "created_at": Self::now_secs()
        });

        self.db.put(txn, key.as_bytes(), value.to_string().as_bytes())
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.db.commit(txn)
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(id)
    }

    /// Get collection by name
    pub fn get_collection(&self, name: &str) -> Result<CollectionInfo> {
        let key = format!("collections/{}", name);

        let txn = self.db
            .begin_transaction()
            .map_err(|e| Error::Storage(e.to_string()))?;
        let value = self.db
            .get(txn, key.as_bytes())
            .map_err(|e| Error::Storage(e.to_string()))?
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))?;
        self.db.commit(txn)
            .map_err(|e| Error::Storage(e.to_string()))?;

        let json: serde_json::Value =
            serde_json::from_slice(&value).map_err(|e| Error::Serialization(e.to_string()))?;

        Ok(CollectionInfo {
            id: json["id"].as_i64().unwrap_or(0),
            name: json["name"].as_str().unwrap_or("").to_string(),
            dim: json["dim"].as_u64().unwrap_or(0) as u32,
            metric: json["metric"].as_str().unwrap_or("dot_product").to_string(),
            config_json: json["config_json"].as_str().unwrap_or("{}").to_string(),
        })
    }

    /// List all collections
    pub fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        let txn = self.db
            .begin_transaction()
            .map_err(|e| Error::Storage(e.to_string()))?;

        let prefix = b"collections/";
        let entries = self.db
            .scan(txn, prefix)
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.db.commit(txn)
            .map_err(|e| Error::Storage(e.to_string()))?;

        let mut collections = Vec::new();
        for (_key, value) in entries {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&value) {
                collections.push(CollectionInfo {
                    id: json["id"].as_i64().unwrap_or(0),
                    name: json["name"].as_str().unwrap_or("").to_string(),
                    dim: json["dim"].as_u64().unwrap_or(0) as u32,
                    metric: json["metric"].as_str().unwrap_or("dot_product").to_string(),
                    config_json: json["config_json"].as_str().unwrap_or("{}").to_string(),
                });
            }
        }

        Ok(collections)
    }

    /// Register a new segment
    pub fn add_segment(&self, collection_id: i64, segment: &SegmentInfo) -> Result<()> {
        let txn = self.db
            .begin_transaction()
            .map_err(|e| Error::Storage(e.to_string()))?;

        let key = format!("segments/{}/{}", collection_id, segment.id);
        let value = serde_json::json!({
            "id": segment.id,
            "collection_id": collection_id,
            "path": segment.path,
            "state": segment.state.to_string(),
            "n_vec": segment.n_vec,
            "min_vec_id": segment.min_vec_id,
            "max_vec_id": segment.max_vec_id,
            "created_at": Self::now_secs()
        });

        self.db.put(txn, key.as_bytes(), value.to_string().as_bytes())
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.db.commit(txn)
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    /// Get all active segments for a collection
    pub fn get_segments(&self, collection_id: i64) -> Result<Vec<SegmentInfo>> {
        let txn = self.db
            .begin_transaction()
            .map_err(|e| Error::Storage(e.to_string()))?;

        let prefix = format!("segments/{}/", collection_id);
        let entries = self.db
            .scan(txn, prefix.as_bytes())
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.db.commit(txn)
            .map_err(|e| Error::Storage(e.to_string()))?;

        let mut segments = Vec::new();
        for (_key, value) in entries {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&value) {
                let state_str = json["state"].as_str().unwrap_or("sealed");
                if state_str != "deleted" {
                    segments.push(SegmentInfo {
                        id: json["id"].as_u64().unwrap_or(0),
                        path: json["path"].as_str().unwrap_or("").to_string(),
                        state: SegmentState::from_str(state_str),
                        n_vec: json["n_vec"].as_u64().unwrap_or(0) as u32,
                        min_vec_id: json["min_vec_id"].as_u64().map(|v| v as u32),
                        max_vec_id: json["max_vec_id"].as_u64().map(|v| v as u32),
                    });
                }
            }
        }

        // Sort by ID descending (newest first)
        segments.sort_by(|a, b| b.id.cmp(&a.id));
        Ok(segments)
    }

    /// Update segment state
    pub fn update_segment_state(&self, _segment_id: u64, _state: SegmentState) -> Result<()> {
        // TODO: Implement key scanning to find and update the segment
        Ok(())
    }

    /// Add a tombstone
    pub fn add_tombstone(&self, collection_id: i64, segment_id: u64, vec_id: u32) -> Result<()> {
        let txn = self.db
            .begin_transaction()
            .map_err(|e| Error::Storage(e.to_string()))?;

        let key = format!("tombstones/{}/{}/{}", collection_id, segment_id, vec_id);
        let value = serde_json::json!({
            "collection_id": collection_id,
            "segment_id": segment_id,
            "vec_id": vec_id,
            "created_at": Self::now_secs()
        });

        self.db.put(txn, key.as_bytes(), value.to_string().as_bytes())
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.db.commit(txn)
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    /// Get tombstones for a segment
    pub fn get_tombstones(&self, segment_id: u64) -> Result<Vec<u32>> {
        let txn = self.db
            .begin_transaction()
            .map_err(|e| Error::Storage(e.to_string()))?;

        // Scan all tombstones and filter by segment_id
        let prefix = b"tombstones/";
        let entries = self.db
            .scan(txn, prefix)
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.db.commit(txn)
            .map_err(|e| Error::Storage(e.to_string()))?;

        let mut tombstones = Vec::new();
        for (_key, value) in entries {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&value) {
                if json["segment_id"].as_u64() == Some(segment_id) {
                    if let Some(vec_id) = json["vec_id"].as_u64() {
                        tombstones.push(vec_id as u32);
                    }
                }
            }
        }

        tombstones.sort();
        Ok(tombstones)
    }

    /// Delete tombstones for a segment (after compaction)
    pub fn clear_tombstones(&self, _segment_id: u64) -> Result<()> {
        // TODO: Implement tombstone clearing
        Ok(())
    }

    /// Get total vector count for collection
    pub fn get_vector_count(&self, collection_id: i64) -> Result<u64> {
        let segments = self.get_segments(collection_id)?;
        let total: u64 = segments.iter().map(|s| s.n_vec as u64).sum();
        Ok(total)
    }

    /// Begin a transaction
    pub fn begin_transaction(&self) -> Result<()> {
        // ToonDB handles transactions internally
        Ok(())
    }

    /// Commit transaction
    pub fn commit(&self) -> Result<()> {
        // ToonDB handles transactions internally
        Ok(())
    }

    /// Rollback transaction
    pub fn rollback(&self) -> Result<()> {
        // ToonDB handles transactions internally
        Ok(())
    }

    /// Execute checkpoint
    pub fn checkpoint(&self) -> Result<()> {
        // ToonDB handles checkpointing internally
        Ok(())
    }
}

/// Collection info from catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub id: i64,
    pub name: String,
    pub dim: u32,
    pub metric: String,
    pub config_json: String,
}

impl CollectionInfo {
    /// Parse config from JSON
    pub fn config(&self) -> Result<EngineConfig> {
        serde_json::from_str(&self.config_json).map_err(|e| Error::Serialization(e.to_string()))
    }
}

/// Segment info from catalog
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    pub id: u64,
    pub path: String,
    pub state: SegmentState,
    pub n_vec: u32,
    pub min_vec_id: Option<u32>,
    pub max_vec_id: Option<u32>,
}

impl SegmentState {
    fn to_string(&self) -> &'static str {
        match self {
            SegmentState::Mutable => "mutable",
            SegmentState::Sealing => "sealing",
            SegmentState::Sealed => "sealed",
            SegmentState::Compacting => "compacting",
            SegmentState::Deleted => "deleted",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "mutable" => SegmentState::Mutable,
            "sealing" => SegmentState::Sealing,
            "sealed" => SegmentState::Sealed,
            "compacting" => SegmentState::Compacting,
            "deleted" => SegmentState::Deleted,
            _ => SegmentState::Sealed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_operations() {
        let catalog = Catalog::open_memory().unwrap();

        // Create collection
        let config = EngineConfig::with_dim(768);
        let collection_id = catalog.create_collection("test", &config).unwrap();
        assert!(collection_id > 0);

        // Get collection
        let info = catalog.get_collection("test").unwrap();
        assert_eq!(info.dim, 768);

        // Add segment
        let segment = SegmentInfo {
            id: 1,
            path: "/data/segment_1.seg".to_string(),
            state: SegmentState::Sealed,
            n_vec: 10000,
            min_vec_id: Some(0),
            max_vec_id: Some(9999),
        };
        catalog.add_segment(collection_id, &segment).unwrap();

        // Get segments
        let segments = catalog.get_segments(collection_id).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].n_vec, 10000);

        // Add tombstone
        catalog.add_tombstone(collection_id, 1, 500).unwrap();
        let tombstones = catalog.get_tombstones(1).unwrap();
        assert_eq!(tombstones, vec![500]);
    }

    #[test]
    fn test_collection_not_found() {
        let catalog = Catalog::open_memory().unwrap();
        let result = catalog.get_collection("nonexistent");
        assert!(matches!(result, Err(Error::CollectionNotFound(_))));
    }
}
