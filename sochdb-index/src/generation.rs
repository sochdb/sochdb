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

//! Durable, immutable index generations published by atomic manifest commit.
//!
//! An index that exists only in a process map is not something a caller can
//! build guarantees on: it disappears on restart, and worse, it can come back
//! describing a different state of the source table without anything saying so.
//! This module gives an index a durable identity — a numbered generation, whose
//! contents are fixed at publication and whose manifest records exactly which
//! source snapshot it was built from.
//!
//! # The property this module exists to provide
//!
//! *A generation becomes active only if it is completely and verifiably on
//! disk.* Every partial state a crash can leave behind must resolve to the
//! previously active generation, never to a half-written one, and never to a
//! plausible-looking manifest whose segments do not match it.
//!
//! Two decisions follow from that, and they are the parts worth reviewing:
//!
//! **Publication is a rename, not a write.** A generation is assembled in a
//! `.tmp` directory that nothing reads. Only once every segment and the
//! manifest are written and flushed does a single `rename` make the directory
//! visible, and only then does a second rename move the `CURRENT` pointer. A
//! crash at any point leaves either the old pointer or the new one; there is no
//! instant at which `CURRENT` names a directory that is still being filled.
//!
//! **Loading verifies before it activates.** The manifest carries a checksum of
//! its own body and a checksum of every segment. A manifest that does not hash
//! to its recorded value, or that names a segment whose bytes have changed, is
//! refused. It is not repaired and it does not silently fall back to an older
//! generation: quietly serving generation 6 when the catalog believes 7 is
//! active would break the one thing this module promises, which is that the
//! active generation names one exact source snapshot.

use crate::hnsw::HnswIndex;
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// On-disk layout version. A reader refuses anything it does not recognise
/// rather than guessing at field meanings.
pub const MANIFEST_FORMAT_VERSION: u32 = 1;

const CURRENT_POINTER: &str = "CURRENT";
const GENERATIONS_DIR: &str = "generations";
const MANIFEST_FILE: &str = "manifest.json";
const TMP_SUFFIX: &str = ".tmp";

/// One immutable file belonging to a generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRef {
    /// Path relative to the generation directory. Relative so a generation can
    /// be copied or restored under a different root without rewriting it.
    pub file: String,
    pub bytes: u64,
    /// BLAKE3 of the file contents, hex encoded.
    pub checksum: String,
}

/// Everything needed to identify what a generation contains and where it came
/// from.
///
/// The provenance fields are not decoration. An index that cannot say which
/// source snapshot it was built from cannot be reasoned about after a restart:
/// it looks identical to one built from a different state of the table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationManifest {
    pub format_version: u32,
    pub generation: u64,
    pub index_name: String,
    pub dimension: u32,
    pub metric: String,
    pub vector_count: u64,
    /// The highest source position included in this generation. A reader may
    /// treat everything at or below it as searchable and nothing above it.
    pub searchable_watermark: u64,
    /// Identifier of the exact source snapshot this generation was built from.
    pub source_snapshot_id: Option<String>,
    pub embedding_model_id: Option<String>,
    pub embedding_model_revision: Option<String>,
    /// Build parameters, ordered so the manifest serializes deterministically.
    pub build_parameters: BTreeMap<String, String>,
    pub segments: Vec<SegmentRef>,
    pub created_at_unix_s: u64,
    pub builder: String,
}

impl GenerationManifest {
    pub fn new(
        generation: u64,
        index_name: impl Into<String>,
        dimension: u32,
        metric: impl Into<String>,
    ) -> Self {
        Self {
            format_version: MANIFEST_FORMAT_VERSION,
            generation,
            index_name: index_name.into(),
            dimension,
            metric: metric.into(),
            vector_count: 0,
            searchable_watermark: 0,
            source_snapshot_id: None,
            embedding_model_id: None,
            embedding_model_revision: None,
            build_parameters: BTreeMap::new(),
            segments: Vec::new(),
            created_at_unix_s: now_unix_s(),
            builder: format!("sochdb-index/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

/// What is actually written to `manifest.json`.
///
/// The checksum lives outside the body so it can cover the body's exact bytes.
#[derive(Debug, Serialize, Deserialize)]
struct ManifestEnvelope {
    checksum: String,
    manifest: GenerationManifest,
}

/// A reason a generation could not be published or trusted.
#[derive(Debug)]
pub enum GenerationError {
    Io(io::Error),
    /// The manifest could not be parsed at all — usually a truncated write.
    Malformed(String),
    /// The manifest parsed but does not hash to its recorded checksum.
    ManifestChecksumMismatch {
        generation: u64,
    },
    /// A segment's bytes are not what the manifest says they are.
    SegmentChecksumMismatch {
        generation: u64,
        file: String,
    },
    /// A segment named by the manifest is not on disk.
    SegmentMissing {
        generation: u64,
        file: String,
    },
    SegmentSizeMismatch {
        generation: u64,
        file: String,
        expected: u64,
        found: u64,
    },
    /// Written by a version that structures the directory differently.
    UnsupportedFormat {
        found: u32,
        supported: u32,
    },
    /// `CURRENT` names a generation that is not there.
    ActiveGenerationMissing {
        generation: u64,
    },
    /// The manifest inside the directory disagrees with the directory's name.
    GenerationMismatch {
        expected: u64,
        found: u64,
    },
    /// Refusing to overwrite an already-published generation, which is
    /// immutable by construction.
    AlreadyPublished {
        generation: u64,
    },
    /// The index could not be encoded or rebuilt.
    Index(String),
    /// The manifest describes an index that is not the one being published.
    DimensionMismatch {
        manifest: u32,
        index: u32,
    },
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenerationError::Io(e) => write!(f, "io error: {e}"),
            GenerationError::Malformed(m) => write!(f, "manifest is not readable: {m}"),
            GenerationError::ManifestChecksumMismatch { generation } => write!(
                f,
                "manifest for generation {generation} does not match its own checksum; \
                 it was modified or partially written"
            ),
            GenerationError::SegmentChecksumMismatch { generation, file } => write!(
                f,
                "segment `{file}` in generation {generation} does not match the checksum \
                 recorded when it was published"
            ),
            GenerationError::SegmentMissing { generation, file } => write!(
                f,
                "segment `{file}` is named by generation {generation} but is not on disk"
            ),
            GenerationError::SegmentSizeMismatch {
                generation,
                file,
                expected,
                found,
            } => write!(
                f,
                "segment `{file}` in generation {generation} is {found} bytes but the \
                 manifest records {expected}"
            ),
            GenerationError::UnsupportedFormat { found, supported } => write!(
                f,
                "manifest format version {found} is not readable by this build, which \
                 supports {supported}"
            ),
            GenerationError::ActiveGenerationMissing { generation } => write!(
                f,
                "CURRENT names generation {generation}, which is not present"
            ),
            GenerationError::GenerationMismatch { expected, found } => write!(
                f,
                "directory for generation {expected} contains a manifest for {found}"
            ),
            GenerationError::AlreadyPublished { generation } => write!(
                f,
                "generation {generation} is already published and generations are immutable"
            ),
            GenerationError::Index(m) => write!(f, "index could not be encoded or rebuilt: {m}"),
            GenerationError::DimensionMismatch { manifest, index } => write!(
                f,
                "manifest declares dimension {manifest} but the index has {index}; \
                 publishing would record provenance that does not describe the data"
            ),
        }
    }
}

impl std::error::Error for GenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GenerationError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for GenerationError {
    fn from(e: io::Error) -> Self {
        GenerationError::Io(e)
    }
}

type Result<T> = std::result::Result<T, GenerationError>;

/// A durable home for an index's published generations.
pub struct GenerationStore {
    root: PathBuf,
}

impl GenerationStore {
    /// Open or create a store rooted at `root`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join(GENERATIONS_DIR))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn generation_dir(&self, generation: u64) -> PathBuf {
        // Zero-padded so a directory listing sorts in generation order, which
        // matters when someone is reading it by hand during an incident.
        self.root
            .join(GENERATIONS_DIR)
            .join(format!("{generation:020}"))
    }

    fn current_path(&self) -> PathBuf {
        self.root.join(CURRENT_POINTER)
    }

    /// Publish a generation and make it active.
    ///
    /// `segments` supplies the bytes of each file; their checksums and sizes are
    /// computed here rather than trusted from the caller, because a checksum the
    /// writer supplies proves nothing about what actually reached the disk.
    ///
    /// The generation is assembled under a `.tmp` name and made visible with a
    /// single rename, so a crash cannot expose a partially written generation.
    pub fn publish(
        &self,
        manifest: &GenerationManifest,
        segments: &[(&str, &[u8])],
    ) -> Result<GenerationManifest> {
        let generation = manifest.generation;
        let final_dir = self.generation_dir(generation);
        if final_dir.exists() {
            return Err(GenerationError::AlreadyPublished { generation });
        }

        let staging = final_dir.with_extension(TMP_SUFFIX.trim_start_matches('.'));
        if staging.exists() {
            // Left by an earlier crash. It was never visible, so discarding it
            // loses nothing that anyone was allowed to read.
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;

        let mut refs = Vec::with_capacity(segments.len());
        for (name, bytes) in segments {
            let path = staging.join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            write_and_sync(&path, bytes)?;
            refs.push(SegmentRef {
                file: (*name).to_string(),
                bytes: bytes.len() as u64,
                checksum: checksum_hex(bytes),
            });
        }

        let mut published = manifest.clone();
        published.format_version = MANIFEST_FORMAT_VERSION;
        published.segments = refs;

        let body = serde_json::to_vec(&published)
            .map_err(|e| GenerationError::Malformed(format!("cannot encode manifest: {e}")))?;
        let envelope = ManifestEnvelope {
            checksum: checksum_hex(&body),
            manifest: published.clone(),
        };
        let encoded = serde_json::to_vec_pretty(&envelope)
            .map_err(|e| GenerationError::Malformed(format!("cannot encode manifest: {e}")))?;
        write_and_sync(&staging.join(MANIFEST_FILE), &encoded)?;

        // Flush the staging directory itself before renaming it: on POSIX the
        // file contents being durable does not imply the directory entries
        // naming them are.
        sync_dir(&staging)?;
        fs::rename(&staging, &final_dir)?;
        sync_dir(&self.root.join(GENERATIONS_DIR))?;

        self.set_current(generation)?;
        Ok(published)
    }

    /// Point `CURRENT` at a generation, atomically.
    fn set_current(&self, generation: u64) -> Result<()> {
        let tmp = self.root.join(format!("{CURRENT_POINTER}{TMP_SUFFIX}"));
        write_and_sync(&tmp, generation.to_string().as_bytes())?;
        fs::rename(&tmp, self.current_path())?;
        sync_dir(&self.root)?;
        Ok(())
    }

    /// The generation `CURRENT` names, if the store has ever published one.
    pub fn active_generation(&self) -> Result<Option<u64>> {
        match fs::read_to_string(self.current_path()) {
            Ok(s) => {
                let trimmed = s.trim();
                trimmed
                    .parse::<u64>()
                    .map(Some)
                    .map_err(|_| GenerationError::Malformed(format!("CURRENT reads `{trimmed}`")))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Load and fully verify the active generation.
    ///
    /// Returns `Ok(None)` only when nothing has ever been published. Anything
    /// else that goes wrong is an error, deliberately: falling back to an older
    /// generation would silently answer queries from a different source
    /// snapshot than the caller believes is active.
    pub fn load_active(&self) -> Result<Option<GenerationManifest>> {
        let Some(generation) = self.active_generation()? else {
            return Ok(None);
        };
        let dir = self.generation_dir(generation);
        if !dir.exists() {
            return Err(GenerationError::ActiveGenerationMissing { generation });
        }
        self.load_generation(generation).map(Some)
    }

    /// Load and fully verify one specific generation.
    pub fn load_generation(&self, generation: u64) -> Result<GenerationManifest> {
        let dir = self.generation_dir(generation);
        let encoded = fs::read(dir.join(MANIFEST_FILE)).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                GenerationError::ActiveGenerationMissing { generation }
            } else {
                GenerationError::Io(e)
            }
        })?;

        let envelope: ManifestEnvelope = serde_json::from_slice(&encoded)
            .map_err(|e| GenerationError::Malformed(format!("generation {generation}: {e}")))?;

        // Re-encode the body and compare. This catches a manifest that was
        // edited in place as well as one whose write was cut short.
        let body = serde_json::to_vec(&envelope.manifest)
            .map_err(|e| GenerationError::Malformed(format!("cannot re-encode manifest: {e}")))?;
        if checksum_hex(&body) != envelope.checksum {
            return Err(GenerationError::ManifestChecksumMismatch { generation });
        }

        let manifest = envelope.manifest;
        if manifest.format_version != MANIFEST_FORMAT_VERSION {
            return Err(GenerationError::UnsupportedFormat {
                found: manifest.format_version,
                supported: MANIFEST_FORMAT_VERSION,
            });
        }
        if manifest.generation != generation {
            return Err(GenerationError::GenerationMismatch {
                expected: generation,
                found: manifest.generation,
            });
        }

        for seg in &manifest.segments {
            let path = dir.join(&seg.file);
            let bytes = fs::read(&path).map_err(|e| {
                if e.kind() == io::ErrorKind::NotFound {
                    GenerationError::SegmentMissing {
                        generation,
                        file: seg.file.clone(),
                    }
                } else {
                    GenerationError::Io(e)
                }
            })?;
            if bytes.len() as u64 != seg.bytes {
                return Err(GenerationError::SegmentSizeMismatch {
                    generation,
                    file: seg.file.clone(),
                    expected: seg.bytes,
                    found: bytes.len() as u64,
                });
            }
            if checksum_hex(&bytes) != seg.checksum {
                return Err(GenerationError::SegmentChecksumMismatch {
                    generation,
                    file: seg.file.clone(),
                });
            }
        }

        Ok(manifest)
    }

    /// Read one segment of a verified generation.
    pub fn read_segment(&self, generation: u64, file: &str) -> Result<Vec<u8>> {
        let path = self.generation_dir(generation).join(file);
        Ok(fs::read(path)?)
    }

    /// Every fully published generation, ascending. Staging directories are
    /// skipped: they were never visible and are not generations.
    pub fn list_generations(&self) -> Result<Vec<u64>> {
        let dir = self.root.join(GENERATIONS_DIR);
        let mut out = Vec::new();
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.ends_with(TMP_SUFFIX) {
                continue;
            }
            if let Ok(g) = name.parse::<u64>() {
                out.push(g);
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    /// Delete generations older than `keep_from`, never touching the active one.
    ///
    /// Retention is the caller's policy, but "do not delete what is being
    /// served" is not, so it is enforced here rather than trusted.
    pub fn prune_below(&self, keep_from: u64) -> Result<Vec<u64>> {
        let active = self.active_generation()?;
        let mut removed = Vec::new();
        for g in self.list_generations()? {
            if g >= keep_from || Some(g) == active {
                continue;
            }
            fs::remove_dir_all(self.generation_dir(g))?;
            removed.push(g);
        }
        if !removed.is_empty() {
            sync_dir(&self.root.join(GENERATIONS_DIR))?;
        }
        Ok(removed)
    }

    /// Remove staging directories left behind by a crash.
    ///
    /// Safe at any time: a staging directory is by definition one that no
    /// reader has ever been able to see.
    pub fn discard_incomplete(&self) -> Result<usize> {
        let dir = self.root.join(GENERATIONS_DIR);
        let mut n = 0;
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.ends_with(TMP_SUFFIX) {
                fs::remove_dir_all(entry.path())?;
                n += 1;
            }
        }
        if n > 0 {
            sync_dir(&dir)?;
        }
        Ok(n)
    }
}

fn checksum_hex(bytes: &[u8]) -> String {
    let mut h = Hasher::new();
    h.update(bytes);
    h.finalize().to_hex().to_string()
}

fn now_unix_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Write a file and flush it to the device before returning.
fn write_and_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

/// Flush a directory's entries.
///
/// On POSIX, a file's contents being durable does not imply that the directory
/// entry naming it is, so a crash can leave a fully written file that no longer
/// appears in its directory. Windows has no equivalent operation and does not
/// need one: `rename` there is already ordered with respect to the metadata.
fn sync_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// The segment an HNSW index is stored under inside a generation.
pub const HNSW_SEGMENT: &str = "index.hnsw";

impl GenerationStore {
    /// Publish an HNSW index as a durable generation.
    ///
    /// The manifest's dimension is checked against the index rather than
    /// trusted. A generation whose recorded provenance does not describe its own
    /// contents is worse than no provenance, because everything downstream is
    /// entitled to rely on it.
    pub fn publish_hnsw(
        &self,
        manifest: &GenerationManifest,
        index: &HnswIndex,
    ) -> Result<GenerationManifest> {
        let index_dimension = index.dimension as u32;
        if manifest.dimension != index_dimension {
            return Err(GenerationError::DimensionMismatch {
                manifest: manifest.dimension,
                index: index_dimension,
            });
        }

        let mut encoded = Vec::new();
        index
            .save_to_writer(&mut encoded)
            .map_err(GenerationError::Index)?;

        let mut with_count = manifest.clone();
        with_count.vector_count = index.len() as u64;
        self.publish(&with_count, &[(HNSW_SEGMENT, &encoded)])
    }

    /// Load the active generation and rebuild its index.
    ///
    /// The generation is verified in full before a single byte is handed to the
    /// index decoder, so a corrupt segment is reported as corruption rather than
    /// as a confusing deserialization failure.
    pub fn load_active_hnsw(&self) -> Result<Option<(GenerationManifest, HnswIndex)>> {
        let Some(manifest) = self.load_active()? else {
            return Ok(None);
        };
        let generation = manifest.generation;
        let bytes = self.read_segment(generation, HNSW_SEGMENT)?;
        let index =
            HnswIndex::load_from_reader(&mut bytes.as_slice()).map_err(GenerationError::Index)?;
        Ok(Some((manifest, index)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn store() -> (TempDir, GenerationStore) {
        let dir = TempDir::new().expect("tempdir");
        let store = GenerationStore::open(dir.path()).expect("open");
        (dir, store)
    }

    fn manifest(generation: u64) -> GenerationManifest {
        let mut m = GenerationManifest::new(generation, "docs", 4, "cosine");
        m.vector_count = 3;
        m.searchable_watermark = 100 + generation;
        m.source_snapshot_id = Some(format!("snapshot-{generation}"));
        m
    }

    fn publish(store: &GenerationStore, generation: u64) -> GenerationManifest {
        store
            .publish(
                &manifest(generation),
                &[("graph.bin", b"graph-bytes"), ("vectors.bin", b"vecs")],
            )
            .expect("publish")
    }

    #[test]
    fn a_published_generation_is_readable_after_reopening_the_store() {
        let (dir, store) = store();
        publish(&store, 1);
        drop(store);

        // Reopening is the closest in-process equivalent of a restart: nothing
        // is carried over except what reached the disk.
        let reopened = GenerationStore::open(dir.path()).expect("reopen");
        let loaded = reopened.load_active().expect("load").expect("some");
        assert_eq!(loaded.generation, 1);
        assert_eq!(loaded.searchable_watermark, 101);
        assert_eq!(loaded.source_snapshot_id.as_deref(), Some("snapshot-1"));
        assert_eq!(
            reopened.read_segment(1, "graph.bin").unwrap(),
            b"graph-bytes"
        );
    }

    #[test]
    fn an_empty_store_reports_nothing_active_rather_than_failing() {
        let (_dir, store) = store();
        assert!(store.active_generation().unwrap().is_none());
        assert!(store.load_active().unwrap().is_none());
    }

    #[test]
    fn publishing_advances_the_active_generation() {
        let (_dir, store) = store();
        publish(&store, 1);
        assert_eq!(store.active_generation().unwrap(), Some(1));
        publish(&store, 2);
        assert_eq!(store.active_generation().unwrap(), Some(2));
        assert_eq!(store.load_active().unwrap().unwrap().generation, 2);
        assert_eq!(store.list_generations().unwrap(), vec![1, 2]);
    }

    /// Generations are immutable once visible. Overwriting one would change
    /// what a reader that already pinned it sees.
    #[test]
    fn republishing_a_generation_is_refused() {
        let (_dir, store) = store();
        publish(&store, 1);
        let err = store
            .publish(&manifest(1), &[("graph.bin", b"different")])
            .unwrap_err();
        assert!(matches!(
            err,
            GenerationError::AlreadyPublished { generation: 1 }
        ));
        assert_eq!(
            store.read_segment(1, "graph.bin").unwrap(),
            b"graph-bytes",
            "the original contents must be untouched"
        );
    }

    /// The central crash case. A publication interrupted before the final
    /// rename leaves a staging directory; the previously active generation must
    /// still be the one that loads, and the debris must not be mistaken for a
    /// generation.
    #[test]
    fn a_publication_interrupted_before_its_rename_leaves_the_old_generation_active() {
        let (dir, store) = store();
        publish(&store, 1);

        // Reconstruct precisely what a crash mid-publish leaves behind: a fully
        // written staging directory that was never renamed.
        let staging = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000002.tmp");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("graph.bin"), b"half-written").unwrap();

        assert_eq!(
            store.active_generation().unwrap(),
            Some(1),
            "CURRENT is only moved after the rename, so it still names 1"
        );
        assert_eq!(store.load_active().unwrap().unwrap().generation, 1);
        assert_eq!(
            store.list_generations().unwrap(),
            vec![1],
            "a staging directory is not a generation"
        );

        assert_eq!(store.discard_incomplete().unwrap(), 1);
        assert!(!staging.exists());
        assert_eq!(store.load_active().unwrap().unwrap().generation, 1);
    }

    /// A staging directory left by an earlier crash must not contaminate a
    /// later attempt at the same generation number.
    #[test]
    fn stale_staging_debris_does_not_leak_into_a_later_publication() {
        let (dir, store) = store();
        let staging = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000001.tmp");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("stale.bin"), b"from a crashed run").unwrap();

        let published = publish(&store, 1);
        assert!(
            published.segments.iter().all(|s| s.file != "stale.bin"),
            "a file from the abandoned attempt must not appear in the manifest"
        );
        assert!(
            !dir.path()
                .join(GENERATIONS_DIR)
                .join("00000000000000000001")
                .join("stale.bin")
                .exists(),
            "and must not be on disk in the published generation"
        );
        store
            .load_active()
            .expect("the published generation verifies");
    }

    #[test]
    fn a_truncated_manifest_is_refused_rather_than_partially_believed() {
        let (dir, store) = store();
        publish(&store, 1);
        let manifest_path = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000001")
            .join(MANIFEST_FILE);
        let good = fs::read(&manifest_path).unwrap();
        fs::write(&manifest_path, &good[..good.len() / 2]).unwrap();

        assert!(matches!(
            store.load_active().unwrap_err(),
            GenerationError::Malformed(_)
        ));
    }

    /// An edit that leaves valid JSON is the dangerous case: nothing about the
    /// file looks wrong, so only the checksum catches it.
    #[test]
    fn a_manifest_edited_in_place_is_caught_by_its_own_checksum() {
        let (dir, store) = store();
        publish(&store, 1);
        let manifest_path = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000001")
            .join(MANIFEST_FILE);

        let mut envelope: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        envelope["manifest"]["searchable_watermark"] = serde_json::json!(999_999);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&envelope).unwrap(),
        )
        .unwrap();

        assert!(matches!(
            store.load_active().unwrap_err(),
            GenerationError::ManifestChecksumMismatch { generation: 1 }
        ));
    }

    #[test]
    fn a_corrupted_segment_is_caught_even_though_the_manifest_is_intact() {
        let (dir, store) = store();
        publish(&store, 1);
        let seg = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000001")
            .join("graph.bin");
        // Same length, different bytes: a size check alone would not notice.
        fs::write(&seg, b"graph-bytez").unwrap();

        match store.load_active().unwrap_err() {
            GenerationError::SegmentChecksumMismatch { generation, file } => {
                assert_eq!(generation, 1);
                assert_eq!(file, "graph.bin");
            }
            other => panic!("expected a checksum mismatch, got {other}"),
        }
    }

    #[test]
    fn a_missing_segment_is_refused() {
        let (dir, store) = store();
        publish(&store, 1);
        fs::remove_file(
            dir.path()
                .join(GENERATIONS_DIR)
                .join("00000000000000000001")
                .join("vectors.bin"),
        )
        .unwrap();

        assert!(matches!(
            store.load_active().unwrap_err(),
            GenerationError::SegmentMissing { generation: 1, .. }
        ));
    }

    #[test]
    fn a_truncated_segment_is_refused() {
        let (dir, store) = store();
        publish(&store, 1);
        let seg = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000001")
            .join("graph.bin");
        fs::write(&seg, b"graph").unwrap();

        assert!(matches!(
            store.load_active().unwrap_err(),
            GenerationError::SegmentSizeMismatch { generation: 1, .. }
        ));
    }

    /// Loading must not quietly answer from an older generation. Serving
    /// generation 1 while the catalog believes 2 is active would break the
    /// promise that the active generation names one exact source snapshot.
    #[test]
    fn a_dangling_current_pointer_fails_rather_than_falling_back() {
        let (dir, store) = store();
        publish(&store, 1);
        publish(&store, 2);
        fs::remove_dir_all(
            dir.path()
                .join(GENERATIONS_DIR)
                .join("00000000000000000002"),
        )
        .unwrap();

        assert!(matches!(
            store.load_active().unwrap_err(),
            GenerationError::ActiveGenerationMissing { generation: 2 }
        ));
    }

    #[test]
    fn a_manifest_from_an_unknown_format_version_is_refused() {
        let (dir, store) = store();
        // Publish normally, then rewrite the envelope so the recorded checksum
        // still matches: the format check must stand on its own rather than
        // being reached only because the checksum already failed.
        store.publish(&manifest(1), &[("a.bin", b"a")]).unwrap();
        let path = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000001")
            .join(MANIFEST_FILE);
        let envelope: ManifestEnvelope = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let mut inner = envelope.manifest;
        inner.format_version = MANIFEST_FORMAT_VERSION + 7;
        let body = serde_json::to_vec(&inner).unwrap();
        let rewritten = ManifestEnvelope {
            checksum: checksum_hex(&body),
            manifest: inner,
        };
        fs::write(&path, serde_json::to_vec_pretty(&rewritten).unwrap()).unwrap();

        assert!(matches!(
            store.load_active().unwrap_err(),
            GenerationError::UnsupportedFormat { .. }
        ));
    }

    #[test]
    fn pruning_never_removes_the_generation_being_served() {
        let (_dir, store) = store();
        publish(&store, 1);
        publish(&store, 2);
        publish(&store, 3);

        // Ask for something the caller should not get: dropping everything
        // below 9 would include the active generation.
        let removed = store.prune_below(9).unwrap();
        assert_eq!(removed, vec![1, 2]);
        assert_eq!(store.list_generations().unwrap(), vec![3]);
        assert_eq!(store.load_active().unwrap().unwrap().generation, 3);
    }

    #[test]
    fn segment_checksums_are_computed_from_what_was_written_not_from_the_caller() {
        let (_dir, store) = store();
        let published = store
            .publish(&manifest(1), &[("only.bin", b"exact contents")])
            .unwrap();
        let seg = &published.segments[0];
        assert_eq!(seg.bytes, 14);
        assert_eq!(seg.checksum, checksum_hex(b"exact contents"));
    }

    fn seeded_index(dimension: usize, count: u128) -> HnswIndex {
        let index = HnswIndex::new(dimension, crate::hnsw::HnswConfig::default());
        for id in 0..count {
            // Deterministic, well-separated vectors so nearest-neighbour order
            // is unambiguous and a difference after recovery means a real
            // difference rather than a tie broken differently.
            let base = id as f32;
            let v: Vec<f32> = (0..dimension).map(|d| base + d as f32 * 0.01).collect();
            index.insert(id, v).expect("insert");
        }
        index
    }

    /// The criterion that makes durability worth having: an index rebuilt from
    /// a published generation must answer identically to the one that was
    /// published. A recovered index that merely *works* is not enough -- if it
    /// returns different neighbours, a restart silently changes query results.
    #[test]
    fn an_index_recovered_from_a_generation_answers_identically() {
        let (dir, store) = store();
        let index = seeded_index(8, 64);

        let queries: Vec<Vec<f32>> = (0..5)
            .map(|q| (0..8).map(|d| q as f32 * 7.0 + d as f32 * 0.01).collect())
            .collect();
        let before: Vec<Vec<(u128, f32)>> = queries
            .iter()
            .map(|q| index.search(q, 10).expect("search"))
            .collect();

        let mut m = GenerationManifest::new(1, "docs", 8, "cosine");
        m.source_snapshot_id = Some("snapshot-a".into());
        let published = store.publish_hnsw(&m, &index).expect("publish");
        assert_eq!(published.vector_count, 64, "count is taken from the index");
        drop(index);
        drop(store);

        let reopened = GenerationStore::open(dir.path()).expect("reopen");
        let (manifest, recovered) = reopened
            .load_active_hnsw()
            .expect("load")
            .expect("a generation is active");
        assert_eq!(manifest.generation, 1);
        assert_eq!(manifest.source_snapshot_id.as_deref(), Some("snapshot-a"));
        assert_eq!(recovered.len(), 64);

        for (q, expected) in queries.iter().zip(&before) {
            let actual = recovered.search(q, 10).expect("search after recovery");
            let expected_ids: Vec<u128> = expected.iter().map(|(id, _)| *id).collect();
            let actual_ids: Vec<u128> = actual.iter().map(|(id, _)| *id).collect();
            assert_eq!(
                actual_ids, expected_ids,
                "a restart changed which vectors this query matches"
            );
            for ((_, a), (_, b)) in actual.iter().zip(expected) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "distances changed across recovery: {a} vs {b}"
                );
            }
        }
    }

    /// A manifest that misdescribes its own index would make every downstream
    /// consumer of that provenance wrong, so it is refused at publication
    /// rather than discovered later.
    #[test]
    fn publishing_a_manifest_that_misdescribes_the_index_is_refused() {
        let (_dir, store) = store();
        let index = seeded_index(8, 4);
        let m = GenerationManifest::new(1, "docs", 16, "cosine");

        let err = store.publish_hnsw(&m, &index).unwrap_err();
        assert!(matches!(
            err,
            GenerationError::DimensionMismatch {
                manifest: 16,
                index: 8
            }
        ));
        assert!(
            store.active_generation().unwrap().is_none(),
            "a refused publication must not have moved the active pointer"
        );
    }

    /// Corruption must be reported as corruption. Handing damaged bytes to the
    /// index decoder would surface as an opaque deserialization message, and an
    /// operator would spend the incident looking at the wrong layer.
    #[test]
    fn a_corrupted_index_segment_is_reported_as_corruption() {
        let (dir, store) = store();
        let index = seeded_index(8, 8);
        let m = GenerationManifest::new(1, "docs", 8, "cosine");
        store.publish_hnsw(&m, &index).expect("publish");

        let seg = dir
            .path()
            .join(GENERATIONS_DIR)
            .join("00000000000000000001")
            .join(HNSW_SEGMENT);
        let mut bytes = fs::read(&seg).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        fs::write(&seg, &bytes).unwrap();

        match store.load_active_hnsw() {
            Err(GenerationError::SegmentChecksumMismatch { generation, file }) => {
                assert_eq!(generation, 1);
                assert_eq!(file, HNSW_SEGMENT);
            }
            Err(other) => panic!("expected corruption to be named as such, got {other}"),
            Ok(_) => panic!("a corrupted segment must not load"),
        }
    }

    #[test]
    fn provenance_survives_a_round_trip() {
        let (_dir, store) = store();
        let mut m = manifest(1);
        m.embedding_model_id = Some("text-embed-3".into());
        m.embedding_model_revision = Some("2026-01-11".into());
        m.build_parameters
            .insert("ef_construction".into(), "200".into());
        m.build_parameters.insert("m".into(), "16".into());
        store.publish(&m, &[("a.bin", b"a")]).unwrap();

        let loaded = store.load_active().unwrap().unwrap();
        assert_eq!(loaded.embedding_model_id.as_deref(), Some("text-embed-3"));
        assert_eq!(
            loaded.embedding_model_revision.as_deref(),
            Some("2026-01-11")
        );
        assert_eq!(loaded.build_parameters["ef_construction"], "200");
        assert_eq!(loaded.build_parameters["m"], "16");
        assert_eq!(loaded.source_snapshot_id.as_deref(), Some("snapshot-1"));
    }
}
