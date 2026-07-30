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

//! Lineage and idempotency for embedding maintenance.
//!
//! An embedding is derived data. It is only meaningful relative to the content
//! it came from, the chunking that produced it, and the model that computed it.
//! An index that stores vectors without recording those three things cannot
//! answer the two questions that matter most in operation: *is this still
//! current*, and *can this be compared to that*.
//!
//! The second question is the sharper one. Vectors from two different models
//! occupy unrelated spaces. Mixing them produces distances that are arithmetic
//! but not meaningful -- the search returns results, ranked, with no error and
//! no warning, and the ranking is noise. This module makes that state
//! unreachable: a change of model or chunking policy yields
//! [`MaintenancePlan::Rebuild`], never an incremental update into the existing
//! generation.
//!
//! Everything here is deterministic and free of I/O so that the decisions --
//! what to re-embed, what to reuse, what to delete, and when a rebuild is
//! mandatory -- can be tested directly rather than inferred from the behaviour
//! of a pipeline that also does network calls.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A model identified the way it must be identified: by provider, name and an
/// exact revision.
///
/// A bare marketing name is refused. Names like `text-embedding-3-large` are
/// reused across silently updated deployments, so an index built against "the"
/// model has no way to detect that the model changed underneath it -- which is
/// exactly the mixed-space failure above, arriving without anyone acting.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider: String,
    pub name: String,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRefError {
    MissingProvider,
    MissingRevision,
    EmptyComponent(&'static str),
}

impl std::fmt::Display for ModelRefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingProvider => write!(
                f,
                "model reference must be qualified by a provider, as in \
                 'openai:text-embedding-3-large@2024-01'"
            ),
            Self::MissingRevision => write!(
                f,
                "model reference must pin an exact revision with '@'; an \
                 unpinned name can change without the index noticing"
            ),
            Self::EmptyComponent(which) => write!(f, "model reference has an empty {which}"),
        }
    }
}

impl std::error::Error for ModelRefError {}

impl ModelRef {
    /// Parse `provider:name@revision`.
    pub fn parse(reference: &str) -> Result<Self, ModelRefError> {
        let (provider, rest) = reference
            .split_once(':')
            .ok_or(ModelRefError::MissingProvider)?;
        // Split from the right: a name may contain '@' in principle, but the
        // revision never contains one, so the last '@' is the separator.
        let (name, revision) = rest
            .rsplit_once('@')
            .ok_or(ModelRefError::MissingRevision)?;

        if provider.is_empty() {
            return Err(ModelRefError::EmptyComponent("provider"));
        }
        if name.is_empty() {
            return Err(ModelRefError::EmptyComponent("name"));
        }
        if revision.is_empty() {
            return Err(ModelRefError::EmptyComponent("revision"));
        }

        Ok(Self {
            provider: provider.to_string(),
            name: name.to_string(),
            revision: revision.to_string(),
        })
    }
}

impl std::fmt::Display for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}@{}", self.provider, self.name, self.revision)
    }
}

/// The parameters that decide how a document becomes chunks.
///
/// `revision` exists because a change to chunking invalidates every embedding
/// derived under the old one, even when the source content is byte-identical.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkPolicy {
    pub revision: String,
    pub max_tokens: u32,
    pub overlap_tokens: u32,
}

/// Everything that must be true for a stored embedding to still be valid.
///
/// Grouped into one type deliberately: these three always travel together, and
/// separating them is how a comparison ends up checking two of the three.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingContext {
    pub model: ModelRef,
    pub chunk_policy: ChunkPolicy,
    pub dimension: u32,
}

impl EmbeddingContext {
    /// Whether embeddings produced under `other` may share an index with
    /// embeddings produced under `self`.
    ///
    /// Anything but an exact match is incompatible. This is intentionally not
    /// a "close enough" comparison: two revisions of one model produce vectors
    /// in different spaces just as surely as two different models do.
    pub fn is_compatible_with(&self, other: &EmbeddingContext) -> bool {
        self == other
    }
}

/// A chunk of source content, ready to be embedded or skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChunk {
    /// Identity of the row this came from, stable across edits.
    pub source_row_id: String,
    /// Position of the chunk within the row.
    pub chunk_ordinal: u32,
    /// Digest of the chunk's content after normalisation.
    pub content_digest: String,
}

impl SourceChunk {
    /// Stable identity of a chunk within its source row.
    pub fn chunk_id(&self) -> String {
        format!("{}#{}", self.source_row_id, self.chunk_ordinal)
    }
}

/// What is recorded alongside every indexed vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub source_row_id: String,
    pub chunk_ordinal: u32,
    pub content_digest: String,
    pub chunk_policy_revision: String,
    pub model: ModelRef,
    pub dimension: u32,
    pub generated_at_unix_ms: u64,
    /// The content-addressed key. Two chunks with this key are the same
    /// embedding and only one needs computing.
    pub cache_key: String,
}

impl EmbeddingRecord {
    pub fn chunk_id(&self) -> String {
        format!("{}#{}", self.source_row_id, self.chunk_ordinal)
    }
}

/// Digest of normalised content.
pub fn content_digest(normalized: &str) -> String {
    blake3::hash(normalized.as_bytes()).to_hex().to_string()
}

/// The content-addressed key from the plan:
/// `hash(normalized_content, chunk_policy_revision, model_revision)`.
///
/// Two chunks anywhere in the corpus that agree on all three are the same
/// embedding, so an identical paragraph repeated across a thousand documents
/// costs one call rather than a thousand.
///
/// Fields are length-prefixed. Without that, a content digest ending in the
/// first characters of a policy revision could produce the same key as a
/// different pair, and a collision here means a stale vector is served as
/// current.
pub fn embedding_cache_key(
    content_digest: &str,
    chunk_policy_revision: &str,
    model: &ModelRef,
) -> String {
    let mut hasher = blake3::Hasher::new();
    for field in [
        content_digest,
        chunk_policy_revision,
        model.provider.as_str(),
        model.name.as_str(),
        model.revision.as_str(),
    ] {
        hasher.update(&(field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// Why a full rebuild is required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildReason {
    ModelChanged { from: ModelRef, to: ModelRef },
    ChunkPolicyChanged { from: String, to: String },
    DimensionChanged { from: u32, to: u32 },
}

impl std::fmt::Display for RebuildReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelChanged { from, to } => write!(
                f,
                "model changed from {from} to {to}; vectors from different \
                 models are not comparable"
            ),
            Self::ChunkPolicyChanged { from, to } => write!(
                f,
                "chunking policy changed from revision {from} to {to}; every \
                 existing embedding was derived under the old one"
            ),
            Self::DimensionChanged { from, to } => {
                write!(f, "embedding dimension changed from {from} to {to}")
            }
        }
    }
}

/// One unit of work, named so a retry is recognisable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedTask {
    pub chunk: SourceChunk,
    pub cache_key: String,
    /// Set when another chunk in this same plan has the same cache key, so
    /// this one can copy that result instead of calling the provider.
    pub duplicate_of: Option<String>,
}

/// What maintenance should do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaintenancePlan {
    /// The existing generation can be updated in place.
    Incremental {
        /// Chunks whose embeddings must be computed.
        embed: Vec<EmbedTask>,
        /// Chunk ids already correct, needing nothing.
        reuse: Vec<String>,
        /// Chunk ids present in the index whose source is gone.
        delete: Vec<String>,
    },
    /// The existing generation cannot be updated and must be replaced. Every
    /// chunk is re-embedded into a new generation; the old one keeps serving
    /// until the new one is published.
    Rebuild {
        reason: RebuildReason,
        embed: Vec<EmbedTask>,
    },
}

impl MaintenancePlan {
    /// Whether this plan would change anything.
    pub fn is_noop(&self) -> bool {
        match self {
            Self::Incremental {
                embed,
                delete,
                reuse: _,
            } => embed.is_empty() && delete.is_empty(),
            Self::Rebuild { .. } => false,
        }
    }

    /// Chunks that require an actual provider call, after in-plan duplicates
    /// have been collapsed.
    pub fn provider_calls(&self) -> usize {
        let tasks = match self {
            Self::Incremental { embed, .. } => embed,
            Self::Rebuild { embed, .. } => embed,
        };
        tasks
            .iter()
            .filter(|t| t.duplicate_of.is_none())
            .map(|t| t.cache_key.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }
}

/// Decide what maintenance is required.
///
/// `existing` is what the index currently holds; `desired` is what the source
/// currently says. Both are keyed by chunk id.
///
/// The incompatibility check runs first and is unconditional. Deciding
/// chunk-by-chunk whether an embedding is current would, for a document whose
/// content had not changed, conclude that its old-model vector was still fine
/// -- and leave it in an index the rest of which had moved to a new model.
pub fn plan_maintenance(
    existing: &[EmbeddingRecord],
    desired: &[SourceChunk],
    context: &EmbeddingContext,
) -> MaintenancePlan {
    if let Some(reason) = incompatibility(existing, context) {
        return MaintenancePlan::Rebuild {
            reason,
            embed: build_tasks(desired, context),
        };
    }

    let by_chunk: BTreeMap<String, &EmbeddingRecord> =
        existing.iter().map(|r| (r.chunk_id(), r)).collect();
    let desired_ids: BTreeSet<String> = desired.iter().map(|c| c.chunk_id()).collect();

    let mut stale = Vec::new();
    let mut reuse = Vec::new();

    for chunk in desired {
        let id = chunk.chunk_id();
        match by_chunk.get(&id) {
            // Same chunk, same content: the stored embedding is still exactly
            // what this chunk would produce.
            Some(record) if record.content_digest == chunk.content_digest => reuse.push(id),
            _ => stale.push(chunk.clone()),
        }
    }

    let delete = by_chunk
        .keys()
        .filter(|id| !desired_ids.contains(*id))
        .cloned()
        .collect();

    MaintenancePlan::Incremental {
        embed: build_tasks(&stale, context),
        reuse,
        delete,
    }
}

fn incompatibility(
    existing: &[EmbeddingRecord],
    context: &EmbeddingContext,
) -> Option<RebuildReason> {
    for record in existing {
        if record.model != context.model {
            return Some(RebuildReason::ModelChanged {
                from: record.model.clone(),
                to: context.model.clone(),
            });
        }
        if record.chunk_policy_revision != context.chunk_policy.revision {
            return Some(RebuildReason::ChunkPolicyChanged {
                from: record.chunk_policy_revision.clone(),
                to: context.chunk_policy.revision.clone(),
            });
        }
        if record.dimension != context.dimension {
            return Some(RebuildReason::DimensionChanged {
                from: record.dimension,
                to: context.dimension,
            });
        }
    }
    None
}

fn build_tasks(chunks: &[SourceChunk], context: &EmbeddingContext) -> Vec<EmbedTask> {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut tasks = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let cache_key = embedding_cache_key(
            &chunk.content_digest,
            &context.chunk_policy.revision,
            &context.model,
        );
        let duplicate_of = seen.get(&cache_key).cloned();
        if duplicate_of.is_none() {
            seen.insert(cache_key.clone(), chunk.chunk_id());
        }
        tasks.push(EmbedTask {
            chunk: chunk.clone(),
            cache_key,
            duplicate_of,
        });
    }
    tasks
}

/// Materialise the records a plan's embed list would produce.
///
/// Separated from planning so that applying a plan is a pure function of the
/// plan: replaying it cannot depend on anything that changed in between.
pub fn records_for(
    tasks: &[EmbedTask],
    context: &EmbeddingContext,
    generated_at_unix_ms: u64,
) -> Vec<EmbeddingRecord> {
    tasks
        .iter()
        .map(|task| EmbeddingRecord {
            source_row_id: task.chunk.source_row_id.clone(),
            chunk_ordinal: task.chunk.chunk_ordinal,
            content_digest: task.chunk.content_digest.clone(),
            chunk_policy_revision: context.chunk_policy.revision.clone(),
            model: context.model.clone(),
            dimension: context.dimension,
            generated_at_unix_ms,
            cache_key: task.cache_key.clone(),
        })
        .collect()
}

/// Apply a plan to a set of records, returning the resulting set.
///
/// Insertion is keyed by chunk id, so applying the same plan twice produces
/// the same set rather than two copies of every chunk. This is the property
/// that makes a retried pipeline run safe.
pub fn apply(
    existing: &[EmbeddingRecord],
    plan: &MaintenancePlan,
    context: &EmbeddingContext,
    generated_at_unix_ms: u64,
) -> Vec<EmbeddingRecord> {
    let mut state: BTreeMap<String, EmbeddingRecord> = match plan {
        // A rebuild starts from nothing: that is what makes it a rebuild
        // rather than an update, and what guarantees no old-model vector
        // survives into the new generation.
        MaintenancePlan::Rebuild { .. } => BTreeMap::new(),
        MaintenancePlan::Incremental { delete, .. } => {
            let removed: BTreeSet<&String> = delete.iter().collect();
            existing
                .iter()
                .filter(|r| !removed.contains(&r.chunk_id()))
                .map(|r| (r.chunk_id(), r.clone()))
                .collect()
        }
    };

    let tasks = match plan {
        MaintenancePlan::Incremental { embed, .. } => embed,
        MaintenancePlan::Rebuild { embed, .. } => embed,
    };

    for record in records_for(tasks, context, generated_at_unix_ms) {
        state.insert(record.chunk_id(), record);
    }

    state.into_values().collect()
}

/// A durable record of work that must happen, written in the same commit as
/// the change that caused it.
///
/// The pattern exists because a source commit and an indexing job are two
/// different systems. Writing the job after the commit loses it if the process
/// dies in between; writing it before means it can name a commit that never
/// happened. Writing it *with* the commit is the only ordering that cannot
/// drop work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutboxEntry {
    pub sequence: u64,
    pub source_row_id: String,
    pub content_digest: String,
    /// False when the source row was deleted.
    pub present: bool,
}

/// An at-least-once work queue with a durable position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outbox {
    entries: Vec<OutboxEntry>,
    /// Highest sequence known to be durably applied. An absolute sequence,
    /// not an index: compaction removes entries from the front, so any
    /// position expressed as an offset into `entries` silently changes
    /// meaning the moment old work is dropped.
    checkpoint: u64,
    /// Next sequence to issue. Derived from a counter rather than from
    /// `entries.len()` for the same reason.
    next_sequence: u64,
}

impl Outbox {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            checkpoint: 0,
            next_sequence: 1,
        }
    }

    fn highest_issued(&self) -> u64 {
        self.next_sequence.saturating_sub(1)
    }

    /// Append a change. The sequence is assigned here so it is always dense
    /// and monotonic; a caller-supplied sequence could skip or repeat.
    pub fn append(
        &mut self,
        source_row_id: impl Into<String>,
        content_digest: impl Into<String>,
        present: bool,
    ) -> u64 {
        // A default-constructed Outbox has next_sequence 0; treat that as 1
        // so a value deserialized from an older shape still issues sequences
        // above the checkpoint rather than colliding with it.
        let sequence = self.next_sequence.max(1);
        self.next_sequence = sequence + 1;
        self.entries.push(OutboxEntry {
            sequence,
            source_row_id: source_row_id.into(),
            content_digest: content_digest.into(),
            present,
        });
        sequence
    }

    /// Work not yet acknowledged.
    pub fn pending(&self) -> &[OutboxEntry] {
        // Entries are appended in sequence order, so the acknowledged prefix
        // is contiguous and can be located without scanning.
        let start = self
            .entries
            .partition_point(|e| e.sequence <= self.checkpoint);
        &self.entries[start..]
    }

    /// Record that everything up to and including `sequence` is durably
    /// applied.
    ///
    /// The checkpoint only ever moves forward. A late acknowledgement for
    /// earlier work must not rewind it and cause completed work to be replayed
    /// -- and, worse, cause work completed *after* it to be skipped on the
    /// next pass because it is now ahead of the mark.
    pub fn acknowledge(&mut self, sequence: u64) {
        // Clamped to what has actually been issued. An acknowledgement for a
        // sequence that does not exist yet would put the checkpoint ahead of
        // future work, and that work would never be seen as pending.
        self.checkpoint = self.checkpoint.max(sequence.min(self.highest_issued()));
    }

    pub fn checkpoint(&self) -> u64 {
        self.checkpoint
    }

    /// Drop acknowledged entries, renumbering nothing.
    pub fn compact(&mut self) {
        let checkpoint = self.checkpoint;
        self.entries.retain(|e| e.sequence > checkpoint);
    }
}

impl Default for Outbox {
    fn default() -> Self {
        Self::new()
    }
}

/// Freshness: how far the searchable state lags the source.
///
/// Reported as a lag rather than a boolean because "is it fresh" has no
/// answer that is true for more than an instant, whereas "how far behind" is
/// something an SLO can be written against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Freshness {
    pub source_sequence: u64,
    pub searchable_sequence: u64,
}

impl Freshness {
    pub fn lag(&self) -> u64 {
        self.source_sequence
            .saturating_sub(self.searchable_sequence)
    }

    pub fn is_current(&self) -> bool {
        self.lag() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelRef {
        ModelRef::parse("openai:text-embedding-3-large@2024-01").unwrap()
    }

    fn context() -> EmbeddingContext {
        EmbeddingContext {
            model: model(),
            chunk_policy: ChunkPolicy {
                revision: "chunk-v1".to_string(),
                max_tokens: 512,
                overlap_tokens: 64,
            },
            dimension: 1024,
        }
    }

    fn chunk(row: &str, ordinal: u32, content: &str) -> SourceChunk {
        SourceChunk {
            source_row_id: row.to_string(),
            chunk_ordinal: ordinal,
            content_digest: content_digest(content),
        }
    }

    fn seed(chunks: &[SourceChunk], ctx: &EmbeddingContext) -> Vec<EmbeddingRecord> {
        let plan = plan_maintenance(&[], chunks, ctx);
        apply(&[], &plan, ctx, 1_000)
    }

    /// A bare marketing name is not an identity. Providers ship updates under
    /// the same name, and an index built against "the" model cannot detect
    /// that -- which produces mixed vector spaces with nobody having acted.
    #[test]
    fn an_unpinned_model_name_is_refused() {
        assert_eq!(
            ModelRef::parse("text-embedding-3-large"),
            Err(ModelRefError::MissingProvider)
        );
        assert_eq!(
            ModelRef::parse("openai:text-embedding-3-large"),
            Err(ModelRefError::MissingRevision)
        );
        assert_eq!(
            ModelRef::parse(":name@rev"),
            Err(ModelRefError::EmptyComponent("provider"))
        );
        assert_eq!(
            ModelRef::parse("openai:@rev"),
            Err(ModelRefError::EmptyComponent("name"))
        );
        assert_eq!(
            ModelRef::parse("openai:name@"),
            Err(ModelRefError::EmptyComponent("revision"))
        );
        assert!(
            ModelRef::parse("openai:text-embedding-3-large")
                .unwrap_err()
                .to_string()
                .contains("revision")
        );
    }

    #[test]
    fn a_qualified_model_reference_round_trips() {
        for reference in [
            "openai:text-embedding-3-large@2024-01",
            "local:bge-m3@sha256-abc",
            "internal:support-embedding-v2@d34db33f",
        ] {
            let parsed = ModelRef::parse(reference).unwrap();
            assert_eq!(parsed.to_string(), reference);
        }
    }

    /// The property that makes a retried pipeline run safe. Replaying a run
    /// must not create a second copy of anything.
    #[test]
    fn replaying_a_run_creates_no_duplicates() {
        let ctx = context();
        let chunks = vec![chunk("row-1", 0, "alpha"), chunk("row-1", 1, "beta")];

        let plan = plan_maintenance(&[], &chunks, &ctx);
        let once = apply(&[], &plan, &ctx, 1_000);
        assert_eq!(once.len(), 2);

        // The same plan applied again, as a crashed-then-retried run would.
        let twice = apply(&[], &plan, &ctx, 2_000);
        assert_eq!(twice.len(), 2, "the replay duplicated every chunk");

        // And a fresh plan computed from the resulting state does nothing.
        let after = plan_maintenance(&once, &chunks, &ctx);
        assert!(after.is_noop(), "a converged pipeline still wanted to work");
    }

    /// Only what changed is re-embedded. This is the whole economic argument
    /// for incremental maintenance: O(changed) rather than O(corpus).
    #[test]
    fn only_changed_content_is_re_embedded() {
        let ctx = context();
        let before = vec![
            chunk("row-1", 0, "alpha"),
            chunk("row-2", 0, "beta"),
            chunk("row-3", 0, "gamma"),
        ];
        let existing = seed(&before, &ctx);

        let after = vec![
            chunk("row-1", 0, "alpha"),
            chunk("row-2", 0, "beta CHANGED"),
            chunk("row-3", 0, "gamma"),
        ];
        let plan = plan_maintenance(&existing, &after, &ctx);

        match &plan {
            MaintenancePlan::Incremental {
                embed,
                reuse,
                delete,
            } => {
                assert_eq!(embed.len(), 1);
                assert_eq!(embed[0].chunk.source_row_id, "row-2");
                assert_eq!(reuse.len(), 2);
                assert!(delete.is_empty());
            }
            other => panic!("expected incremental, got {other:?}"),
        }
    }

    /// A source row that disappeared must have its vectors removed. Leaving
    /// them makes deleted content permanently retrievable, which is a data
    /// deletion failure rather than a staleness one.
    #[test]
    fn vectors_for_a_deleted_row_are_removed() {
        let ctx = context();
        let existing = seed(&[chunk("row-1", 0, "a"), chunk("row-2", 0, "b")], &ctx);

        let plan = plan_maintenance(&existing, &[chunk("row-1", 0, "a")], &ctx);
        match &plan {
            MaintenancePlan::Incremental { delete, .. } => {
                assert_eq!(delete, &vec!["row-2#0".to_string()]);
            }
            other => panic!("expected incremental, got {other:?}"),
        }

        let after = apply(&existing, &plan, &ctx, 2_000);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].source_row_id, "row-1");
    }

    /// A row that loses chunks -- because it got shorter -- must lose the
    /// trailing vectors too, or a search still returns text that is no longer
    /// in the document.
    #[test]
    fn a_shortened_row_loses_its_trailing_chunks() {
        let ctx = context();
        let existing = seed(
            &[
                chunk("row-1", 0, "a"),
                chunk("row-1", 1, "b"),
                chunk("row-1", 2, "c"),
            ],
            &ctx,
        );
        let plan = plan_maintenance(&existing, &[chunk("row-1", 0, "a")], &ctx);
        match &plan {
            MaintenancePlan::Incremental { delete, .. } => {
                assert_eq!(delete, &vec!["row-1#1".to_string(), "row-1#2".to_string()]);
            }
            other => panic!("expected incremental, got {other:?}"),
        }
    }

    /// The failure this module exists to prevent. Vectors from two models are
    /// not comparable; mixing them yields a ranking that is arithmetic but
    /// meaningless, with no error anywhere. A model change must therefore
    /// force a rebuild, not an incremental update -- *especially* for
    /// documents whose content did not change, since those are exactly the
    /// ones an incremental check would decide were still fine.
    #[test]
    fn a_model_change_forces_a_rebuild_even_when_no_content_changed() {
        let ctx = context();
        let chunks = vec![chunk("row-1", 0, "alpha"), chunk("row-2", 0, "beta")];
        let existing = seed(&chunks, &ctx);

        let mut newer = context();
        newer.model = ModelRef::parse("openai:text-embedding-3-large@2024-06").unwrap();

        // Identical content, only the model revision moved.
        let plan = plan_maintenance(&existing, &chunks, &newer);
        match &plan {
            MaintenancePlan::Rebuild { reason, embed } => {
                assert!(matches!(reason, RebuildReason::ModelChanged { .. }));
                assert_eq!(embed.len(), 2, "a rebuild must re-embed everything");
                assert!(reason.to_string().contains("not comparable"));
            }
            other => panic!("a model change produced {other:?} instead of a rebuild"),
        }

        // And applying it leaves nothing from the old model behind.
        let after = apply(&existing, &plan, &newer, 2_000);
        assert_eq!(after.len(), 2);
        assert!(
            after.iter().all(|r| r.model == newer.model),
            "an old-model vector survived the rebuild"
        );
    }

    #[test]
    fn a_chunk_policy_change_forces_a_rebuild() {
        let ctx = context();
        let chunks = vec![chunk("row-1", 0, "alpha")];
        let existing = seed(&chunks, &ctx);

        let mut newer = context();
        newer.chunk_policy.revision = "chunk-v2".to_string();

        match plan_maintenance(&existing, &chunks, &newer) {
            MaintenancePlan::Rebuild { reason, .. } => {
                assert!(matches!(reason, RebuildReason::ChunkPolicyChanged { .. }));
            }
            other => panic!("expected a rebuild, got {other:?}"),
        }
    }

    #[test]
    fn a_dimension_change_forces_a_rebuild() {
        let ctx = context();
        let chunks = vec![chunk("row-1", 0, "alpha")];
        let existing = seed(&chunks, &ctx);

        let mut newer = context();
        newer.dimension = 512;

        match plan_maintenance(&existing, &chunks, &newer) {
            MaintenancePlan::Rebuild { reason, .. } => {
                assert!(matches!(reason, RebuildReason::DimensionChanged { .. }));
            }
            other => panic!("expected a rebuild, got {other:?}"),
        }
    }

    #[test]
    fn compatibility_requires_an_exact_match() {
        let a = context();
        assert!(a.is_compatible_with(&context()));

        let mut b = context();
        b.model.revision = "2024-06".to_string();
        assert!(
            !a.is_compatible_with(&b),
            "two revisions of one model were treated as interchangeable"
        );
    }

    /// The cache key is what turns a repeated paragraph into one provider
    /// call instead of many.
    #[test]
    fn identical_content_shares_one_cache_key() {
        let ctx = context();
        let a = embedding_cache_key(
            &content_digest("shared boilerplate"),
            &ctx.chunk_policy.revision,
            &ctx.model,
        );
        let b = embedding_cache_key(
            &content_digest("shared boilerplate"),
            &ctx.chunk_policy.revision,
            &ctx.model,
        );
        assert_eq!(a, b);
    }

    /// ...and every input that would change the embedding changes the key,
    /// because reusing a vector across any of these is serving a stale answer.
    #[test]
    fn every_input_that_changes_the_embedding_changes_the_key() {
        let ctx = context();
        let base = embedding_cache_key(&content_digest("x"), "chunk-v1", &ctx.model);

        assert_ne!(
            base,
            embedding_cache_key(&content_digest("y"), "chunk-v1", &ctx.model)
        );
        assert_ne!(
            base,
            embedding_cache_key(&content_digest("x"), "chunk-v2", &ctx.model)
        );
        assert_ne!(
            base,
            embedding_cache_key(
                &content_digest("x"),
                "chunk-v1",
                &ModelRef::parse("openai:text-embedding-3-large@2024-06").unwrap()
            )
        );
        assert_ne!(
            base,
            embedding_cache_key(
                &content_digest("x"),
                "chunk-v1",
                &ModelRef::parse("local:text-embedding-3-large@2024-01").unwrap()
            )
        );
    }

    /// Length prefixes stop a boundary between two fields from moving. A
    /// collision here serves a stale vector as current.
    #[test]
    fn key_field_boundaries_cannot_shift() {
        let m = model();
        assert_ne!(
            embedding_cache_key("ab", "c", &m),
            embedding_cache_key("a", "bc", &m)
        );
    }

    /// Duplicate content within one batch collapses to a single provider call.
    #[test]
    fn duplicate_chunks_in_one_batch_cost_one_call() {
        let ctx = context();
        let chunks = vec![
            chunk("row-1", 0, "shared"),
            chunk("row-2", 0, "shared"),
            chunk("row-3", 0, "shared"),
            chunk("row-4", 0, "different"),
        ];
        let plan = plan_maintenance(&[], &chunks, &ctx);

        assert_eq!(
            plan.provider_calls(),
            2,
            "identical content was embedded more than once"
        );
        match &plan {
            MaintenancePlan::Incremental { embed, .. } => {
                assert_eq!(embed.len(), 4, "every chunk still needs a vector");
                assert_eq!(embed[0].duplicate_of, None);
                assert_eq!(embed[1].duplicate_of.as_deref(), Some("row-1#0"));
                assert_eq!(embed[2].duplicate_of.as_deref(), Some("row-1#0"));
                assert_eq!(embed[3].duplicate_of, None);
            }
            other => panic!("expected incremental, got {other:?}"),
        }
    }

    #[test]
    fn every_record_carries_its_full_lineage() {
        let ctx = context();
        let records = seed(&[chunk("row-1", 3, "alpha")], &ctx);
        let r = &records[0];
        assert_eq!(r.source_row_id, "row-1");
        assert_eq!(r.chunk_ordinal, 3);
        assert_eq!(r.content_digest, content_digest("alpha"));
        assert_eq!(r.chunk_policy_revision, "chunk-v1");
        assert_eq!(r.model, model());
        assert_eq!(r.dimension, 1024);
        assert_eq!(r.generated_at_unix_ms, 1_000);
        assert_eq!(r.chunk_id(), "row-1#3");
        assert!(!r.cache_key.is_empty());
    }

    /// Work is only forgotten once it is acknowledged, so a crash mid-run
    /// replays rather than skips.
    #[test]
    fn unacknowledged_work_survives_and_is_replayed() {
        let mut outbox = Outbox::new();
        outbox.append("row-1", "d1", true);
        outbox.append("row-2", "d2", true);
        outbox.append("row-3", "d3", false);

        assert_eq!(outbox.pending().len(), 3);
        outbox.acknowledge(1);
        assert_eq!(outbox.pending().len(), 2);
        assert_eq!(outbox.pending()[0].source_row_id, "row-2");

        // A crash before acknowledging the rest leaves them pending.
        let recovered: Outbox =
            serde_json::from_str(&serde_json::to_string(&outbox).unwrap()).unwrap();
        assert_eq!(recovered.pending().len(), 2);
        assert_eq!(recovered.checkpoint(), 1);
    }

    /// A late acknowledgement must not rewind the checkpoint. If it did,
    /// completed work would replay and -- worse -- work finished after the
    /// late one would be skipped on the next pass.
    #[test]
    fn the_checkpoint_never_moves_backwards() {
        let mut outbox = Outbox::new();
        outbox.append("row-1", "d1", true);
        outbox.append("row-2", "d2", true);
        outbox.append("row-3", "d3", true);

        outbox.acknowledge(3);
        assert_eq!(outbox.checkpoint(), 3);
        outbox.acknowledge(1);
        assert_eq!(outbox.checkpoint(), 3, "a stale ack rewound the checkpoint");
        assert!(outbox.pending().is_empty());
    }

    /// An acknowledgement beyond what exists must not push the checkpoint past
    /// real work, or entries appended afterwards are never processed.
    #[test]
    fn an_acknowledgement_cannot_skip_work_that_does_not_exist_yet() {
        let mut outbox = Outbox::new();
        outbox.append("row-1", "d1", true);
        outbox.acknowledge(999);
        assert_eq!(outbox.checkpoint(), 1);

        outbox.append("row-2", "d2", true);
        assert_eq!(
            outbox.pending().len(),
            1,
            "work appended after an over-large ack was skipped"
        );
        assert_eq!(outbox.pending()[0].source_row_id, "row-2");
    }

    /// Compaction drops acknowledged entries. Positions expressed as offsets
    /// into the entry list silently change meaning when that happens, which is
    /// why the checkpoint is an absolute sequence -- an earlier version of this
    /// tracked an index and lost the remaining work on the first compaction.
    #[test]
    fn compaction_keeps_unacknowledged_work() {
        let mut outbox = Outbox::new();
        outbox.append("row-1", "d1", true);
        outbox.append("row-2", "d2", true);
        outbox.acknowledge(1);
        outbox.compact();
        assert_eq!(outbox.pending().len(), 1);
        assert_eq!(outbox.pending()[0].source_row_id, "row-2");

        // Sequences issued after a compaction must still be above the
        // checkpoint, or newly appended work would look already acknowledged.
        let next = outbox.append("row-3", "d3", true);
        assert!(next > outbox.checkpoint());
        assert_eq!(outbox.pending().len(), 2);

        outbox.acknowledge(next);
        outbox.compact();
        assert!(outbox.pending().is_empty());
    }

    #[test]
    fn freshness_is_reported_as_a_lag() {
        let f = Freshness {
            source_sequence: 100,
            searchable_sequence: 93,
        };
        assert_eq!(f.lag(), 7);
        assert!(!f.is_current());

        assert!(
            Freshness {
                source_sequence: 100,
                searchable_sequence: 100
            }
            .is_current()
        );

        // A searchable position ahead of the source is nonsense, but it must
        // not underflow into an enormous lag.
        assert_eq!(
            Freshness {
                source_sequence: 5,
                searchable_sequence: 9
            }
            .lag(),
            0
        );
    }
}
