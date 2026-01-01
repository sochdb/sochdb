//! LSM-style segment management.
//!
//! Handles mutable mem-segments, immutable sealed segments,
//! tombstones, and compaction.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;

use crate::catalog::Catalog;
use crate::config::EngineConfig;
use crate::error::Result;
use crate::segment::{Segment, SegmentWriter};
use crate::types::*;

/// LSM manager for segment lifecycle
pub struct LsmManager {
    config: EngineConfig,
    data_dir: PathBuf,
    /// Mutable segment currently accepting writes
    mutable_segment: RwLock<Option<MutableSegment>>,
    /// Sealed immutable segments (newest first)
    sealed_segments: RwLock<Vec<Arc<Segment>>>,
    /// Next segment ID
    next_segment_id: AtomicU64,
    /// Tombstones pending compaction
    tombstones: RwLock<HashSet<(SegmentId, VectorId)>>,
}

impl LsmManager {
    /// Create a new LSM manager
    pub fn new(config: EngineConfig, data_dir: PathBuf) -> Self {
        Self {
            config,
            data_dir,
            mutable_segment: RwLock::new(None),
            sealed_segments: RwLock::new(Vec::new()),
            next_segment_id: AtomicU64::new(1),
            tombstones: RwLock::new(HashSet::new()),
        }
    }

    /// Load existing segments from catalog
    pub fn load_from_catalog(&self, catalog: &Catalog, collection_id: i64) -> Result<()> {
        let segment_infos = catalog.get_segments(collection_id)?;
        
        let mut sealed = self.sealed_segments.write();
        for info in segment_infos {
            if info.state == SegmentState::Sealed {
                let segment = Segment::open(&info.path)?;
                sealed.push(Arc::new(segment));
            }
            
            // Update next segment ID
            let current_max = self.next_segment_id.load(Ordering::SeqCst);
            if info.id >= current_max {
                self.next_segment_id.store(info.id + 1, Ordering::SeqCst);
            }
        }
        
        // Load tombstones
        for info in catalog.get_segments(collection_id)? {
            let tombstone_ids = catalog.get_tombstones(info.id)?;
            let mut tombstones = self.tombstones.write();
            for vid in tombstone_ids {
                tombstones.insert((info.id, vid));
            }
        }
        
        Ok(())
    }

    /// Insert a vector
    pub fn insert(&self, vector: &[f32]) -> Result<(SegmentId, VectorId)> {
        let mut mutable = self.mutable_segment.write();
        
        // Create mutable segment if needed
        if mutable.is_none() {
            let seg_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);
            let writer = SegmentWriter::new(self.config.clone())?;
            *mutable = Some(MutableSegment {
                id: seg_id,
                writer,
            });
        }
        
        let seg = mutable.as_mut().unwrap();
        let vid = seg.writer.add(vector)?;
        let current_seg_id = seg.id;
        
        // Check if we need to seal
        let should_seal = seg.writer.len() >= self.config.lsm.max_mutable_size;
        
        if should_seal {
            let mutable_seg = mutable.take().unwrap();
            let sealed_seg = self.seal_mutable(mutable_seg)?;
            drop(mutable); // Release the lock before acquiring sealed lock
            
            let mut sealed = self.sealed_segments.write();
            sealed.insert(0, sealed_seg); // Newest first
            
            // Trigger compaction if needed
            if sealed.len() > self.config.lsm.max_segments {
                drop(sealed);
                self.trigger_compaction()?;
            }
        }
        
        Ok((current_seg_id, vid))
    }

    /// Delete a vector
    pub fn delete(&self, segment_id: SegmentId, vec_id: VectorId) -> Result<()> {
        let mut tombstones = self.tombstones.write();
        tombstones.insert((segment_id, vec_id));
        Ok(())
    }

    /// Seal the mutable segment and write to disk
    fn seal_mutable(&self, mutable: MutableSegment) -> Result<Arc<Segment>> {
        let path = self.segment_path(mutable.id);
        mutable.writer.build(&path)?;
        
        let segment = Segment::open(&path)?;
        Ok(Arc::new(segment))
    }

    /// Generate segment file path
    fn segment_path(&self, seg_id: SegmentId) -> PathBuf {
        self.data_dir.join(format!("segment_{:016x}.seg", seg_id))
    }

    /// Force seal current mutable segment
    pub fn flush(&self) -> Result<Option<Arc<Segment>>> {
        let mut mutable = self.mutable_segment.write();
        
        if let Some(seg) = mutable.take() {
            if seg.writer.len() > 0 {
                let sealed = self.seal_mutable(seg)?;
                let mut sealed_list = self.sealed_segments.write();
                sealed_list.insert(0, Arc::clone(&sealed));
                return Ok(Some(sealed));
            }
        }
        
        Ok(None)
    }

    /// Get all segments for querying (newest first)
    pub fn get_query_segments(&self) -> Vec<Arc<Segment>> {
        self.sealed_segments.read().clone()
    }

    /// Check if a vector is tombstoned
    pub fn is_tombstoned(&self, segment_id: SegmentId, vec_id: VectorId) -> bool {
        self.tombstones.read().contains(&(segment_id, vec_id))
    }

    /// Trigger compaction
    fn trigger_compaction(&self) -> Result<()> {
        // TODO: Implement background compaction
        // For now, just log
        tracing::info!("Compaction triggered (not implemented)");
        Ok(())
    }

    /// Run compaction (merge multiple segments into one)
    pub fn compact(&self, catalog: &Catalog, _collection_id: i64) -> Result<()> {
        let mut sealed = self.sealed_segments.write();
        
        if sealed.len() < self.config.lsm.compaction_ratio {
            return Ok(());
        }
        
        // Take oldest N segments for compaction
        let num_to_compact = self.config.lsm.compaction_ratio;
        let start_idx = sealed.len() - num_to_compact;
        let to_compact: Vec<Arc<Segment>> = sealed.drain(start_idx..).collect();
        
        drop(sealed);
        
        // Create new merged segment
        let new_seg_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);
        let mut writer = SegmentWriter::new(self.config.clone())?;
        
        for old_seg in &to_compact {
            if let Some(fp32) = old_seg.fp32_data() {
                let dim = old_seg.dim() as usize;
                for vid in 0..old_seg.num_vectors() {
                    // Skip tombstoned vectors
                    let old_seg_id = new_seg_id - 1; // Approximate
                    if self.is_tombstoned(old_seg_id, vid) {
                        continue;
                    }
                    
                    let offset = vid as usize * dim;
                    let vec = &fp32[offset..offset + dim];
                    writer.add(vec)?;
                }
            }
        }
        
        // Write new segment
        let new_path = self.segment_path(new_seg_id);
        if writer.len() > 0 {
            writer.build(&new_path)?;
            let new_segment = Segment::open(&new_path)?;
            
            // Update sealed list
            let mut sealed = self.sealed_segments.write();
            sealed.push(Arc::new(new_segment));
        }
        
        // Mark old segments as deleted in catalog
        for old_seg in &to_compact {
            // Extract segment ID from path (simplified)
            let path = old_seg.path();
            if let Some(seg_id_str) = path.split("segment_").last() {
                if let Some(id_hex) = seg_id_str.strip_suffix(".seg") {
                    if let Ok(seg_id) = u64::from_str_radix(id_hex, 16) {
                        catalog.update_segment_state(seg_id, SegmentState::Deleted)?;
                        catalog.clear_tombstones(seg_id)?;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Get total vector count
    pub fn vector_count(&self) -> u32 {
        let mutable_count = self.mutable_segment
            .read()
            .as_ref()
            .map(|s| s.writer.len() as u32)
            .unwrap_or(0);
        
        let sealed_count: u32 = self.sealed_segments
            .read()
            .iter()
            .map(|s| s.num_vectors())
            .sum();
        
        let tombstone_count = self.tombstones.read().len() as u32;
        
        mutable_count + sealed_count - tombstone_count
    }
}

/// Mutable segment accepting writes
struct MutableSegment {
    id: SegmentId,
    writer: SegmentWriter,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_lsm_insert_flush() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_dim(64);
        let lsm = LsmManager::new(config, dir.path().to_path_buf());

        // Insert vectors
        for i in 0..100 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            lsm.insert(&vec).unwrap();
        }

        // Flush
        let flushed = lsm.flush().unwrap();
        assert!(flushed.is_some());

        // Should have one sealed segment
        let segments = lsm.get_query_segments();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].num_vectors(), 100);
    }

    #[test]
    fn test_lsm_tombstones() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_dim(64);
        let lsm = LsmManager::new(config, dir.path().to_path_buf());

        // Insert and flush
        let vec: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let (seg_id, vid) = lsm.insert(&vec).unwrap();
        lsm.flush().unwrap();

        // Delete
        lsm.delete(1, 0).unwrap();
        assert!(lsm.is_tombstoned(1, 0));
    }
}
