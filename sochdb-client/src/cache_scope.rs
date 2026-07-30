// SPDX-License-Identifier: AGPL-3.0-or-later
//
// SochDB - A unified database for AI-native applications
// Copyright (C) 2025 Sushanth Reddy Vanagala
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Policy-safe scoping for the semantic cache.
//!
//! A cache is a channel. Anything that can be reached by a lookup can be read,
//! so the question a cache has to answer is not "do I have an answer for this
//! query" but "is this requester entitled to the answer I have, computed under
//! the rules that apply right now". [`semantic_cache`](crate::semantic_cache)
//! keys on a query hash, a namespace and a hash of the allowed id set. That
//! stops one obvious leak -- two users with different visible documents do not
//! share entries -- and misses several others.
//!
//! # What an allowed-set hash does not cover
//!
//! An authorization decision is more than a set of ids. Column masking, row
//! filters, redaction rules and purpose limitations all change the *content* of
//! an answer without changing which objects were readable. Revoke a user's
//! right to see unmasked salaries and their allowed set does not move, so the
//! answer computed before the change is still reachable afterwards.
//!
//! Nor does it cover the machinery that produced the answer. A cached response
//! depends on the embedding model that found the evidence, the retrieval
//! profile that ranked it, the generation model that wrote it and the prompt
//! that framed it. Change any of them and the entry is no longer an answer to
//! the current question, but nothing in the old key notices.
//!
//! Nor does it cover the data. Provenance is recorded as a list of document
//! ids, and ids do not change when documents do.
//!
//! # The shape of the fix
//!
//! Every dimension that can change an answer is part of the key, and every one
//! of them is compared exactly. Similarity is allowed to be approximate about
//! *what was asked*; it is never allowed to be approximate about who is asking,
//! under what rules, or with which models. That is the condition the plan
//! states and it is enforced here as a type: you cannot build a
//! [`ScopedCacheKey`] without supplying a whole [`ScopeBinding`].
//!
//! Verification returns a [`Verdict`], not a boolean. A boolean invites the
//! caller to treat "I could not tell" as "yes", which is the failure mode this
//! module exists to prevent. Every path that is not a proven-fresh hit is a
//! named refusal or a named staleness, and both are misses.
//!
//! # Cache classes
//!
//! The four classes carry genuinely different risk, so they are separate types
//! of thing rather than a label on one:
//!
//! - [`CacheClass::CatalogHelp`] -- schema and documentation lookups over
//!   non-sensitive metadata. Safe to roll out first, and the only class where a
//!   near-miss on the query is cheap to be wrong about.
//! - [`CacheClass::RetrievalResult`] -- a set of document references. A hit
//!   still has to be reauthorized because the references are the sensitive
//!   part.
//! - [`CacheClass::ContextAssembly`] -- an assembled prompt context. Refuses
//!   semantic matching outright: the assembly is fitted to a specific query
//!   under a specific budget, and a merely similar query gets context that was
//!   packed for a different one.
//! - [`CacheClass::GeneratedResponse`] -- model output. Rolled out last, and
//!   never returned without its evidence, so the caller can reauthorize the
//!   sources rather than trusting the text.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// How large a similarity threshold has to be before a class will consider a
/// non-identical query at all.
///
/// This is a floor, not a default. A caller may be stricter. The floor exists
/// because a low threshold turns "answer to a similar question" into "answer to
/// a different question", and no amount of correct authorization makes that
/// answer right.
pub const MIN_SEMANTIC_THRESHOLD: f32 = 0.90;

/// The class of thing being cached.
///
/// Classes differ in what invalidates them and in how much latitude a lookup
/// gets, so they are enumerated rather than left to a string label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CacheClass {
    /// Schema, catalog and documentation lookups over non-sensitive metadata.
    CatalogHelp,
    /// A ranked set of document references produced by retrieval.
    RetrievalResult,
    /// A prompt context assembled from retrieved evidence.
    ContextAssembly,
    /// Text produced by a generation model.
    GeneratedResponse,
}

impl CacheClass {
    /// A stable byte tag for digesting.
    ///
    /// Explicit rather than derived from the discriminant, so reordering the
    /// enum cannot silently make one class's entries readable as another's.
    pub const fn tag(self) -> u8 {
        match self {
            CacheClass::CatalogHelp => 1,
            CacheClass::RetrievalResult => 2,
            CacheClass::ContextAssembly => 3,
            CacheClass::GeneratedResponse => 4,
        }
    }

    /// Whether this class may be served on a merely similar query.
    ///
    /// Context assembly may not. The assembled context is packed to a token
    /// budget around one specific query -- which passages were included, in
    /// what order, truncated where -- and a similar query is not served by a
    /// context fitted to a different one. The failure is quiet: the model
    /// receives plausible context that omits the part that mattered.
    pub const fn allows_semantic_match(self) -> bool {
        !matches!(self, CacheClass::ContextAssembly)
    }

    /// Whether a hit must carry source evidence the caller can reauthorize.
    ///
    /// Everything derived from governed data must. Catalog help need not,
    /// because it is not derived from governed data -- and if a deployment
    /// treats its schema as sensitive, that content belongs in another class.
    pub const fn requires_evidence(self) -> bool {
        !matches!(self, CacheClass::CatalogHelp)
    }
}

/// How the result is rendered back to the caller.
///
/// Part of the key because the same evidence rendered as prose and as a table
/// are different answers, and because a caller that asked for one and received
/// the other has been given something it did not request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OutputMode {
    /// Free text.
    Prose,
    /// Structured rows.
    Table,
    /// A machine-readable document.
    Json,
    /// A SQL statement.
    Sql,
}

impl OutputMode {
    /// A stable byte tag for digesting.
    pub const fn tag(self) -> u8 {
        match self {
            OutputMode::Prose => 1,
            OutputMode::Table => 2,
            OutputMode::Json => 3,
            OutputMode::Sql => 4,
        }
    }
}

/// A component that can change an answer and therefore has to be pinned.
///
/// Kept as an explicit enum so that adding a new one is a compile error at
/// every construction site rather than a field somebody forgets to fill in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Component {
    /// The model that produced the query and document embeddings.
    EmbeddingModel,
    /// The profile that fused and ranked retrieval lanes.
    RetrievalProfile,
    /// The model that generated text.
    GenerationModel,
    /// The prompt or template that framed the generation.
    PromptTemplate,
}

impl Component {
    /// A stable byte tag for digesting.
    pub const fn tag(self) -> u8 {
        match self {
            Component::EmbeddingModel => 1,
            Component::RetrievalProfile => 2,
            Component::GenerationModel => 3,
            Component::PromptTemplate => 4,
        }
    }

    /// A human-readable name, for refusal reasons.
    pub const fn name(self) -> &'static str {
        match self {
            Component::EmbeddingModel => "embedding model",
            Component::RetrievalProfile => "retrieval profile",
            Component::GenerationModel => "generation model",
            Component::PromptTemplate => "prompt template",
        }
    }
}

/// The exact-match dimensions of a cache lookup.
///
/// Everything here is compared byte for byte. Nothing here is approximate.
///
/// The type has no `Default`. A default binding would be a binding somebody did
/// not think about, and the whole point is that each field is a deliberate
/// statement about the request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeBinding {
    /// The isolation boundary. Entries never cross it, whatever else matches.
    pub tenant: String,
    /// A digest of the effective authorization decision: which objects are
    /// readable *and* under what masking, filtering and redaction.
    ///
    /// This is the field the old allowed-set hash was trying to be. It is a
    /// digest of the decision, not of the object list, so a policy change that
    /// leaves the object list alone still moves it.
    pub authorization_digest: [u8; 32],
    /// The revision of the policy that produced that decision.
    ///
    /// Carried separately from the digest so that a refusal can say *why*: a
    /// policy that was rewritten is a different situation from a user whose
    /// grants changed, and an operator reading the counters needs to tell them
    /// apart.
    pub policy_revision: u64,
    /// Revisions of every component that shaped the answer.
    ///
    /// A `BTreeMap` so iteration order is the key order and the digest does not
    /// depend on insertion sequence.
    pub components: BTreeMap<Component, String>,
    /// How the answer is rendered.
    pub output_mode: OutputMode,
}

impl ScopeBinding {
    /// Start a binding for a tenant under a given authorization decision.
    pub fn new(
        tenant: impl Into<String>,
        authorization_digest: [u8; 32],
        policy_revision: u64,
        output_mode: OutputMode,
    ) -> Self {
        Self {
            tenant: tenant.into(),
            authorization_digest,
            policy_revision,
            components: BTreeMap::new(),
            output_mode,
        }
    }

    /// Pin a component revision.
    #[must_use]
    pub fn with_component(mut self, component: Component, revision: impl Into<String>) -> Self {
        self.components.insert(component, revision.into());
        self
    }

    /// The components this class requires to be pinned.
    ///
    /// A retrieval result depends on the embedding model and the ranking
    /// profile. A generated response depends on those *and* on the model and
    /// prompt that wrote it -- the evidence was still retrieved, so the
    /// retrieval components do not stop mattering just because a model has been
    /// layered on top.
    pub fn required_components(class: CacheClass) -> &'static [Component] {
        match class {
            CacheClass::CatalogHelp => &[],
            CacheClass::RetrievalResult => {
                &[Component::EmbeddingModel, Component::RetrievalProfile]
            }
            CacheClass::ContextAssembly => {
                &[Component::EmbeddingModel, Component::RetrievalProfile]
            }
            CacheClass::GeneratedResponse => &[
                Component::EmbeddingModel,
                Component::RetrievalProfile,
                Component::GenerationModel,
                Component::PromptTemplate,
            ],
        }
    }

    /// Check that every component this class depends on has been pinned.
    ///
    /// An unpinned component is not a missing optimisation, it is an entry that
    /// will survive a change it should not survive. So this is checked before a
    /// key can be built rather than left to the caller.
    pub fn check_complete(&self, class: CacheClass) -> Result<(), ScopeError> {
        if self.tenant.is_empty() {
            return Err(ScopeError::MissingTenant);
        }
        for required in Self::required_components(class) {
            match self.components.get(required) {
                None => return Err(ScopeError::UnpinnedComponent(*required)),
                Some(revision) if revision.is_empty() => {
                    return Err(ScopeError::UnpinnedComponent(*required));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// A canonical, unambiguous encoding of the binding.
    ///
    /// Every variable-length field is length-prefixed. Concatenating
    /// `tenant = "a"` with an object named `"b:c"` and `tenant = "a:b"` with an
    /// object named `"c"` would otherwise produce the same bytes, and one
    /// tenant's entries would be reachable by another.
    fn canonical(&self, class: CacheClass) -> Vec<u8> {
        let mut out = Vec::with_capacity(128);
        out.push(class.tag());
        out.push(self.output_mode.tag());
        push_field(&mut out, self.tenant.as_bytes());
        push_field(&mut out, &self.authorization_digest);
        out.extend_from_slice(&self.policy_revision.to_le_bytes());
        out.extend_from_slice(&(self.components.len() as u32).to_le_bytes());
        for (component, revision) in &self.components {
            out.push(component.tag());
            push_field(&mut out, revision.as_bytes());
        }
        out
    }
}

/// Length-prefix a field before it joins the digest input.
fn push_field(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// A scope could not be used to build a key.
#[derive(Debug, Clone, PartialEq)]
pub enum ScopeError {
    /// No tenant was named, so there is no isolation boundary.
    MissingTenant,
    /// A component this class depends on was left unpinned.
    UnpinnedComponent(Component),
    /// A similarity threshold below what any class permits.
    ThresholdTooLow { requested: f32, floor: f32 },
}

impl std::fmt::Display for ScopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopeError::MissingTenant => {
                write!(
                    f,
                    "cache scope has no tenant, so it has no isolation boundary"
                )
            }
            ScopeError::UnpinnedComponent(c) => write!(
                f,
                "cache scope leaves the {} unpinned, so a change to it would not invalidate the entry",
                c.name()
            ),
            ScopeError::ThresholdTooLow { requested, floor } => write!(
                f,
                "similarity threshold {requested} is below the floor {floor}"
            ),
        }
    }
}

impl std::error::Error for ScopeError {}

/// A cache key that cannot be built without a complete scope.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ScopedCacheKey {
    /// The class, kept in the clear so entries can be swept per class.
    pub class: CacheClass,
    /// Digest over the class, the binding and the query.
    pub digest: [u8; 32],
}

impl ScopedCacheKey {
    /// Build a key, or refuse because the scope is incomplete.
    ///
    /// The query is normalised the way the existing cache normalises it --
    /// trimmed and lowercased -- so that the two agree about what counts as the
    /// same question. Normalisation stops there. Collapsing whitespace or
    /// stripping punctuation would make `DROP TABLE x` and `drop table  x`
    /// identical, which is fine, but would also start making decisions about
    /// meaning that belong to the semantic layer.
    pub fn new(class: CacheClass, binding: &ScopeBinding, query: &str) -> Result<Self, ScopeError> {
        binding.check_complete(class)?;
        let mut hasher = blake3::Hasher::new();
        let canonical = binding.canonical(class);
        hasher.update(&(canonical.len() as u32).to_le_bytes());
        hasher.update(&canonical);
        let normalized = query.trim().to_lowercase();
        hasher.update(&(normalized.len() as u32).to_le_bytes());
        hasher.update(normalized.as_bytes());
        Ok(Self {
            class,
            digest: *hasher.finalize().as_bytes(),
        })
    }

    /// The storage key for this entry.
    ///
    /// Hex-encoded, so no part of it can contain a separator. The old key
    /// interpolated a caller-supplied namespace straight into a slash-delimited
    /// path; nothing here is caller-supplied by the time it reaches the path.
    pub fn to_storage_key(&self) -> Vec<u8> {
        let mut key = format!("_cache/v2/{}/", self.class.tag()).into_bytes();
        key.reserve(64);
        for byte in self.digest {
            key.extend_from_slice(format!("{byte:02x}").as_bytes());
        }
        key
    }
}

/// A source object as it stood when an entry was built.
///
/// The version is what makes this different from the existing `source_docs`
/// list. An id says which object the answer came from; only a version says
/// which *state* of it, and only a version lets a reader notice that the object
/// has moved on.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceSnapshot {
    /// The object the answer drew on.
    pub object_id: String,
    /// The version of that object at the time.
    pub version: u64,
}

impl SourceSnapshot {
    /// Record an object at a version.
    pub fn new(object_id: impl Into<String>, version: u64) -> Self {
        Self {
            object_id: object_id.into(),
            version,
        }
    }
}

/// What was cached, and everything needed to decide whether it may be returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScopedEntry {
    /// The key this was stored under.
    pub key: ScopedCacheKey,
    /// The binding it was built under, kept so a hit can be checked against the
    /// binding of the request rather than trusted because the key matched.
    ///
    /// The key already covers the binding, so a mismatch here would mean a
    /// digest collision or a corrupted store. Checking anyway costs a
    /// comparison and turns a silent cross-scope read into a named refusal.
    pub binding: ScopeBinding,
    /// The query as asked, for semantic comparison and for auditing.
    pub query: String,
    /// The cached payload.
    pub payload: Vec<u8>,
    /// The sources the payload was derived from, with versions.
    pub sources: Vec<SourceSnapshot>,
    /// When it was written, in milliseconds since the epoch.
    pub created_at_ms: u64,
    /// When it stops being usable, in milliseconds since the epoch.
    pub expires_at_ms: u64,
}

impl ScopedEntry {
    /// Build an entry, refusing one that cannot be reauthorized later.
    ///
    /// A generated response with no recorded evidence is exactly the artifact
    /// this design exists to prevent: text that has to be trusted because there
    /// is nothing left to check. So it is refused at write time, where the
    /// caller still has the evidence, rather than at read time when it is gone.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key: ScopedCacheKey,
        binding: ScopeBinding,
        query: impl Into<String>,
        payload: Vec<u8>,
        sources: Vec<SourceSnapshot>,
        created_at_ms: u64,
        ttl_ms: u64,
    ) -> Result<Self, EntryError> {
        if key.class.requires_evidence() && sources.is_empty() {
            return Err(EntryError::NoEvidence(key.class));
        }
        let mut sources = sources;
        sources.sort();
        sources.dedup();
        Ok(Self {
            key,
            binding,
            query: query.into(),
            payload,
            sources,
            created_at_ms,
            expires_at_ms: created_at_ms.saturating_add(ttl_ms),
        })
    }
}

/// An entry could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryError {
    /// A class that requires reauthorizable evidence was given none.
    NoEvidence(CacheClass),
}

impl std::fmt::Display for EntryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntryError::NoEvidence(class) => write!(
                f,
                "{class:?} entries must record their sources so a hit can be reauthorized"
            ),
        }
    }
}

impl std::error::Error for EntryError {}

/// Why a candidate was not returned.
///
/// Named rather than counted so that the reason reaches the operator. "Cache
/// miss" hides the difference between a cold cache, a policy rewrite and an
/// attempted cross-tenant read, and only one of those is routine.
#[derive(Debug, Clone, PartialEq)]
pub enum RefusalReason {
    /// The request and the entry belong to different tenants.
    TenantMismatch,
    /// The policy has been rewritten since the entry was built.
    PolicyRevisionChanged { cached: u64, current: u64 },
    /// The effective authorization decision differs.
    AuthorizationChanged,
    /// A component was upgraded, retuned or replaced.
    ComponentChanged(Component),
    /// The requested rendering differs.
    OutputModeChanged,
    /// The class differs, which should be impossible and therefore matters.
    ClassMismatch,
    /// The entry's own key does not match its contents.
    KeyMismatch,
    /// The class refuses to be served on a merely similar query.
    SemanticMatchNotAllowed(CacheClass),
    /// The similarity offered is below the configured threshold.
    BelowThreshold { similarity: f32, threshold: f32 },
}

impl std::fmt::Display for RefusalReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefusalReason::TenantMismatch => write!(f, "entry belongs to a different tenant"),
            RefusalReason::PolicyRevisionChanged { cached, current } => write!(
                f,
                "policy moved from revision {cached} to {current} since the entry was written"
            ),
            RefusalReason::AuthorizationChanged => {
                write!(f, "the effective authorization decision has changed")
            }
            RefusalReason::ComponentChanged(c) => write!(f, "the {} has changed", c.name()),
            RefusalReason::OutputModeChanged => write!(f, "a different output mode was requested"),
            RefusalReason::ClassMismatch => write!(f, "entry is of a different cache class"),
            RefusalReason::KeyMismatch => write!(f, "entry does not match the key it was found at"),
            RefusalReason::SemanticMatchNotAllowed(c) => {
                write!(f, "{c:?} entries are only served on an identical query")
            }
            RefusalReason::BelowThreshold {
                similarity,
                threshold,
            } => write!(
                f,
                "similarity {similarity} is below the threshold {threshold}"
            ),
        }
    }
}

/// Why an otherwise-authorized entry is out of date.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaleReason {
    /// The entry's time to live has elapsed.
    Expired { expired_at_ms: u64, now_ms: u64 },
    /// A source object has advanced past the version the answer was built on.
    SourceAdvanced {
        object_id: String,
        cached_version: u64,
        current_version: u64,
    },
    /// A source object has gone, so the answer rests on something that no
    /// longer exists.
    SourceMissing { object_id: String },
}

impl std::fmt::Display for StaleReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaleReason::Expired {
                expired_at_ms,
                now_ms,
            } => write!(f, "entry expired at {expired_at_ms}, now {now_ms}"),
            StaleReason::SourceAdvanced {
                object_id,
                cached_version,
                current_version,
            } => write!(
                f,
                "source {object_id} moved from version {cached_version} to {current_version}"
            ),
            StaleReason::SourceMissing { object_id } => {
                write!(f, "source {object_id} no longer exists")
            }
        }
    }
}

/// The outcome of checking a candidate.
///
/// Three outcomes, not two, because the operator response differs: a refusal
/// may be an attack and is always worth counting separately, while staleness is
/// ordinary cache behaviour that should trigger a rewrite.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The entry may be returned, together with the evidence to reauthorize.
    Fresh { evidence: Vec<SourceSnapshot> },
    /// The entry is authorized but out of date. Recompute and replace it.
    Stale(StaleReason),
    /// The entry must not be returned to this requester at all.
    Refused(RefusalReason),
}

impl Verdict {
    /// Whether this verdict permits returning the payload.
    ///
    /// The only affirmative constructor is [`Verdict::Fresh`], and it carries
    /// the evidence, so a caller that wants the payload has to hold the
    /// evidence too.
    pub fn is_hit(&self) -> bool {
        matches!(self, Verdict::Fresh { .. })
    }

    /// A short label for metrics.
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::Fresh { .. } => "fresh",
            Verdict::Stale(_) => "stale",
            Verdict::Refused(_) => "refused",
        }
    }
}

/// The current version of a source object, as the caller sees it now.
///
/// `None` means the object is gone or is no longer visible; both are treated as
/// staleness rather than as "unchanged", because an answer resting on something
/// the requester cannot currently see is not an answer they may have.
pub type SourceVersionLookup<'a> = &'a dyn Fn(&str) -> Option<u64>;

/// Decide whether a candidate entry may be returned.
///
/// Order matters. Authorization is checked before freshness, so that a refusal
/// never leaks through as staleness -- a caller told "stale" learns that an
/// entry exists, and for a cross-tenant probe that is already more than it
/// should learn. Freshness is checked before similarity, because a stale entry
/// should be rewritten regardless of how close the query was.
pub fn verify(
    entry: &ScopedEntry,
    request: &ScopeBinding,
    class: CacheClass,
    now_ms: u64,
    current_version: SourceVersionLookup<'_>,
) -> Verdict {
    if entry.key.class != class {
        return Verdict::Refused(RefusalReason::ClassMismatch);
    }

    // Rebuilding the key proves the entry has not been moved or edited under a
    // binding it was not written with. Without it, a store that can be written
    // to out of band could park an entry at a key whose binding it does not
    // actually satisfy, and every later check would compare against the
    // attacker's binding rather than the one the data was produced under.
    match ScopedCacheKey::new(class, &entry.binding, &entry.query) {
        Ok(rebuilt) if rebuilt == entry.key => {}
        _ => return Verdict::Refused(RefusalReason::KeyMismatch),
    }

    if entry.binding.tenant != request.tenant {
        return Verdict::Refused(RefusalReason::TenantMismatch);
    }
    if entry.binding.authorization_digest != request.authorization_digest {
        return Verdict::Refused(RefusalReason::AuthorizationChanged);
    }
    if entry.binding.policy_revision != request.policy_revision {
        return Verdict::Refused(RefusalReason::PolicyRevisionChanged {
            cached: entry.binding.policy_revision,
            current: request.policy_revision,
        });
    }
    if entry.binding.output_mode != request.output_mode {
        return Verdict::Refused(RefusalReason::OutputModeChanged);
    }

    // Compare the union of both sides' components, not just the request's. A
    // component present in the entry and absent from the request is a change:
    // the answer was shaped by something the current request does not use.
    for component in entry
        .binding
        .components
        .keys()
        .chain(request.binding_component_keys())
    {
        if entry.binding.components.get(component) != request.components.get(component) {
            return Verdict::Refused(RefusalReason::ComponentChanged(*component));
        }
    }

    if entry.expires_at_ms <= now_ms {
        return Verdict::Stale(StaleReason::Expired {
            expired_at_ms: entry.expires_at_ms,
            now_ms,
        });
    }

    for source in &entry.sources {
        match current_version(&source.object_id) {
            None => {
                return Verdict::Stale(StaleReason::SourceMissing {
                    object_id: source.object_id.clone(),
                });
            }
            Some(current) if current != source.version => {
                return Verdict::Stale(StaleReason::SourceAdvanced {
                    object_id: source.object_id.clone(),
                    cached_version: source.version,
                    current_version: current,
                });
            }
            Some(_) => {}
        }
    }

    Verdict::Fresh {
        evidence: entry.sources.clone(),
    }
}

impl ScopeBinding {
    /// The component keys of this binding, for union comparison.
    fn binding_component_keys(&self) -> impl Iterator<Item = &Component> {
        self.components.keys()
    }
}

/// Decide whether a semantically similar entry may be considered at all.
///
/// This is the second half of the hit condition. The first half -- that every
/// exact dimension matches -- is [`verify`], and it runs first. Similarity
/// never substitutes for it: a very similar query under a different policy is
/// still refused, and an identical query under a changed model is still
/// refused.
pub fn admit_semantic(
    class: CacheClass,
    similarity: f32,
    threshold: f32,
) -> Result<Option<RefusalReason>, ScopeError> {
    if threshold < MIN_SEMANTIC_THRESHOLD {
        return Err(ScopeError::ThresholdTooLow {
            requested: threshold,
            floor: MIN_SEMANTIC_THRESHOLD,
        });
    }
    if !class.allows_semantic_match() {
        return Ok(Some(RefusalReason::SemanticMatchNotAllowed(class)));
    }
    // Written as an explicit comparison rather than as `similarity < threshold`
    // because the two differ exactly where it matters. `NaN < threshold` is
    // false, so the simpler form would *admit* an unorderable similarity.
    // `partial_cmp` returns `None` for NaN, which lands in the refusal branch.
    let admitted = matches!(
        similarity.partial_cmp(&threshold),
        Some(Ordering::Greater | Ordering::Equal)
    );
    if !admitted {
        return Ok(Some(RefusalReason::BelowThreshold {
            similarity,
            threshold,
        }));
    }
    Ok(None)
}

/// Counters for the decisions this module makes.
///
/// Refusals are counted by reason rather than lumped into a miss rate. A rising
/// `TenantMismatch` count is a probe; a rising `PolicyRevisionChanged` count is
/// a policy that is being edited too often to cache against. A single "miss"
/// number cannot tell an operator either thing.
#[derive(Debug, Clone, Default)]
pub struct CacheOutcomeCounters {
    /// Hits, by class.
    pub fresh: BTreeMap<CacheClass, u64>,
    /// Stale entries found, by class.
    pub stale: BTreeMap<CacheClass, u64>,
    /// Refusals, by class and reason label.
    pub refused: BTreeMap<(CacheClass, &'static str), u64>,
}

impl CacheOutcomeCounters {
    /// Record one verdict.
    pub fn record(&mut self, class: CacheClass, verdict: &Verdict) {
        match verdict {
            Verdict::Fresh { .. } => *self.fresh.entry(class).or_default() += 1,
            Verdict::Stale(_) => *self.stale.entry(class).or_default() += 1,
            Verdict::Refused(reason) => {
                *self
                    .refused
                    .entry((class, refusal_label(reason)))
                    .or_default() += 1;
            }
        }
    }

    /// Hit rate for a class, or `None` if it has been asked nothing.
    ///
    /// Refusals are in the denominator. Excluding them would let a cache that
    /// refuses almost everything report a perfect hit rate, which is the number
    /// an operator would least like to be reassured by.
    pub fn hit_rate(&self, class: CacheClass) -> Option<f64> {
        let fresh = self.fresh.get(&class).copied().unwrap_or(0);
        let stale = self.stale.get(&class).copied().unwrap_or(0);
        let refused: u64 = self
            .refused
            .iter()
            .filter(|((c, _), _)| *c == class)
            .map(|(_, n)| *n)
            .sum();
        let total = fresh + stale + refused;
        if total == 0 {
            None
        } else {
            Some(fresh as f64 / total as f64)
        }
    }

    /// How many cross-scope lookups were refused for this class.
    pub fn cross_scope_refusals(&self, class: CacheClass) -> u64 {
        self.refused
            .iter()
            .filter(|((c, label), _)| {
                *c == class && matches!(*label, "tenant_mismatch" | "authorization_changed")
            })
            .map(|(_, n)| *n)
            .sum()
    }
}

/// A stable metric label for a refusal.
fn refusal_label(reason: &RefusalReason) -> &'static str {
    match reason {
        RefusalReason::TenantMismatch => "tenant_mismatch",
        RefusalReason::PolicyRevisionChanged { .. } => "policy_revision_changed",
        RefusalReason::AuthorizationChanged => "authorization_changed",
        RefusalReason::ComponentChanged(_) => "component_changed",
        RefusalReason::OutputModeChanged => "output_mode_changed",
        RefusalReason::ClassMismatch => "class_mismatch",
        RefusalReason::KeyMismatch => "key_mismatch",
        RefusalReason::SemanticMatchNotAllowed(_) => "semantic_not_allowed",
        RefusalReason::BelowThreshold { .. } => "below_threshold",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(tenant: &str, auth: u8, policy: u64) -> ScopeBinding {
        ScopeBinding::new(tenant, [auth; 32], policy, OutputMode::Prose)
            .with_component(Component::EmbeddingModel, "openai:text-3@1")
            .with_component(Component::RetrievalProfile, "hybrid@7")
            .with_component(Component::GenerationModel, "anthropic:sonnet@5")
            .with_component(Component::PromptTemplate, "answer@3")
    }

    fn entry(b: &ScopeBinding, query: &str) -> ScopedEntry {
        let key = ScopedCacheKey::new(CacheClass::GeneratedResponse, b, query).unwrap();
        ScopedEntry::new(
            key,
            b.clone(),
            query,
            b"cached answer".to_vec(),
            vec![SourceSnapshot::new("doc-1", 4)],
            1_000,
            60_000,
        )
        .unwrap()
    }

    fn versions_unchanged(id: &str) -> Option<u64> {
        match id {
            "doc-1" => Some(4),
            _ => None,
        }
    }

    /// The defect this module exists for, stated against the code it replaces.
    ///
    /// The existing key is (query, namespace, allowed-set hash). A policy
    /// change that alters masking or redaction without changing which objects
    /// are readable leaves all three identical, so the answer computed under
    /// the old rules stays reachable under the new ones. This asserts that
    /// collision directly rather than describing it.
    #[test]
    fn the_old_key_cannot_see_a_policy_change_that_leaves_the_object_set_alone() {
        use crate::semantic_cache::CacheKey;

        let allowed = 0xdead_beef_u64;
        let before = CacheKey::new("what is the average salary", "hr", allowed);
        let after = CacheKey::new("what is the average salary", "hr", allowed);
        assert_eq!(
            before.to_storage_key(),
            after.to_storage_key(),
            "same objects readable, so the old key is identical across the policy change"
        );

        // The same two situations under a scope binding are different keys,
        // because the policy revision is part of the key.
        let b1 = binding("hr", 1, 41);
        let b2 = binding("hr", 1, 42);
        let k1 = ScopedCacheKey::new(
            CacheClass::GeneratedResponse,
            &b1,
            "what is the average salary",
        )
        .unwrap();
        let k2 = ScopedCacheKey::new(
            CacheClass::GeneratedResponse,
            &b2,
            "what is the average salary",
        )
        .unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn a_policy_revision_change_refuses_the_entry() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("acme", 1, 8),
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(
            verdict,
            Verdict::Refused(RefusalReason::PolicyRevisionChanged {
                cached: 7,
                current: 8
            })
        );
    }

    #[test]
    fn another_tenant_cannot_read_the_entry() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("globex", 1, 7),
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(verdict, Verdict::Refused(RefusalReason::TenantMismatch));
        assert!(!verdict.is_hit());
    }

    /// The case the old allowed-set hash was built for, kept because it must
    /// keep working: same tenant, same policy revision, different effective
    /// decision.
    #[test]
    fn a_changed_authorization_decision_refuses_the_entry() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("acme", 2, 7),
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(
            verdict,
            Verdict::Refused(RefusalReason::AuthorizationChanged)
        );
    }

    /// Authorization is checked before freshness so a probe cannot learn from
    /// the difference. An expired entry belonging to another tenant must be
    /// refused, not reported stale -- "stale" would confirm the entry exists.
    #[test]
    fn a_cross_tenant_probe_is_refused_rather_than_reported_stale() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("globex", 1, 7),
            CacheClass::GeneratedResponse,
            9_999_999,
            &versions_unchanged,
        );
        assert_eq!(verdict, Verdict::Refused(RefusalReason::TenantMismatch));
    }

    #[test]
    fn a_new_embedding_model_refuses_the_entry() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let mut request = binding("acme", 1, 7);
        request
            .components
            .insert(Component::EmbeddingModel, "openai:text-3@2".into());
        let verdict = verify(
            &cached,
            &request,
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(
            verdict,
            Verdict::Refused(RefusalReason::ComponentChanged(Component::EmbeddingModel))
        );
    }

    /// A component the entry was built with and the request no longer pins is
    /// still a change. Dropping a prompt template does not make the cached
    /// answer template-independent.
    #[test]
    fn dropping_a_component_is_also_a_change() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let mut request = binding("acme", 1, 7);
        request.components.remove(&Component::PromptTemplate);
        let verdict = verify(
            &cached,
            &request,
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(
            verdict,
            Verdict::Refused(RefusalReason::ComponentChanged(Component::PromptTemplate))
        );
    }

    #[test]
    fn a_different_output_mode_refuses_the_entry() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let mut request = binding("acme", 1, 7);
        request.output_mode = OutputMode::Table;
        let verdict = verify(
            &cached,
            &request,
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(verdict, Verdict::Refused(RefusalReason::OutputModeChanged));
    }

    #[test]
    fn an_advanced_source_makes_the_entry_stale() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("acme", 1, 7),
            CacheClass::GeneratedResponse,
            2_000,
            &|id| if id == "doc-1" { Some(5) } else { None },
        );
        assert_eq!(
            verdict,
            Verdict::Stale(StaleReason::SourceAdvanced {
                object_id: "doc-1".into(),
                cached_version: 4,
                current_version: 5,
            })
        );
    }

    /// A source that has vanished is staleness, not freshness. Treating an
    /// unresolvable source as unchanged would keep serving an answer built on
    /// something that is no longer there -- and a source that is merely no
    /// longer visible to this requester lands here too, which is the safe way
    /// round.
    #[test]
    fn a_missing_source_makes_the_entry_stale() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("acme", 1, 7),
            CacheClass::GeneratedResponse,
            2_000,
            &|_| None,
        );
        assert_eq!(
            verdict,
            Verdict::Stale(StaleReason::SourceMissing {
                object_id: "doc-1".into()
            })
        );
    }

    #[test]
    fn an_expired_entry_is_stale() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("acme", 1, 7),
            CacheClass::GeneratedResponse,
            999_999,
            &versions_unchanged,
        );
        assert!(matches!(
            verdict,
            Verdict::Stale(StaleReason::Expired { .. })
        ));
    }

    #[test]
    fn a_fresh_hit_carries_the_evidence_to_reauthorize() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("acme", 1, 7),
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        match verdict {
            Verdict::Fresh { evidence } => {
                assert_eq!(evidence, vec![SourceSnapshot::new("doc-1", 4)]);
            }
            other => panic!("expected a fresh hit, got {other:?}"),
        }
    }

    /// An entry parked at a key whose binding it does not satisfy is refused.
    /// Without this check every later comparison would be against the forged
    /// binding rather than the one the answer was produced under.
    #[test]
    fn an_entry_that_does_not_match_its_key_is_refused() {
        let mut cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        cached.binding.tenant = "globex".into();
        let verdict = verify(
            &cached,
            &binding("globex", 1, 7),
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(verdict, Verdict::Refused(RefusalReason::KeyMismatch));
    }

    #[test]
    fn an_entry_of_another_class_is_refused() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        let verdict = verify(
            &cached,
            &binding("acme", 1, 7),
            CacheClass::RetrievalResult,
            2_000,
            &versions_unchanged,
        );
        assert_eq!(verdict, Verdict::Refused(RefusalReason::ClassMismatch));
    }

    /// Two classes over the same query and scope must not share an entry.
    #[test]
    fn classes_do_not_share_keys() {
        let b = binding("acme", 1, 7);
        let a = ScopedCacheKey::new(CacheClass::RetrievalResult, &b, "q").unwrap();
        let c = ScopedCacheKey::new(CacheClass::ContextAssembly, &b, "q").unwrap();
        assert_ne!(a.digest, c.digest);
        assert_ne!(a.to_storage_key(), c.to_storage_key());
    }

    /// Length-prefixing is what stops one tenant's key from being another's.
    /// Without it `"a" + "b:c"` and `"a:b" + "c"` digest identically.
    #[test]
    fn field_boundaries_cannot_be_shifted_between_tenant_and_revision() {
        let mut left = ScopeBinding::new("acme", [0; 32], 1, OutputMode::Prose);
        left = left.with_component(Component::EmbeddingModel, "model-x");
        let mut right = ScopeBinding::new("acmemodel", [0; 32], 1, OutputMode::Prose);
        right = right.with_component(Component::EmbeddingModel, "-x");
        let a = ScopedCacheKey::new(CacheClass::CatalogHelp, &left, "q").unwrap();
        let b = ScopedCacheKey::new(CacheClass::CatalogHelp, &right, "q").unwrap();
        assert_ne!(a.digest, b.digest);
    }

    /// A key cannot be built at all if a component this class depends on is
    /// unpinned, so the unsafe entry never gets written.
    #[test]
    fn a_key_cannot_be_built_with_an_unpinned_component() {
        let partial = ScopeBinding::new("acme", [0; 32], 1, OutputMode::Prose)
            .with_component(Component::EmbeddingModel, "m@1");
        let err =
            ScopedCacheKey::new(CacheClass::GeneratedResponse, &partial, "q").expect_err("refused");
        assert_eq!(
            err,
            ScopeError::UnpinnedComponent(Component::RetrievalProfile)
        );
    }

    #[test]
    fn a_key_cannot_be_built_without_a_tenant() {
        let anonymous = ScopeBinding::new("", [0; 32], 1, OutputMode::Prose);
        let err =
            ScopedCacheKey::new(CacheClass::CatalogHelp, &anonymous, "q").expect_err("refused");
        assert_eq!(err, ScopeError::MissingTenant);
    }

    /// Catalog help pins nothing, because it is not derived from a model or
    /// from governed data. That is the class the rollout starts with.
    #[test]
    fn catalog_help_needs_no_pinned_components() {
        let plain = ScopeBinding::new("acme", [0; 32], 1, OutputMode::Table);
        assert!(ScopedCacheKey::new(CacheClass::CatalogHelp, &plain, "list tables").is_ok());
    }

    /// A generated response with nothing to reauthorize is refused at write
    /// time, where the caller still has the evidence.
    #[test]
    fn a_response_without_evidence_cannot_be_stored() {
        let b = binding("acme", 1, 7);
        let key = ScopedCacheKey::new(CacheClass::GeneratedResponse, &b, "q").unwrap();
        let err =
            ScopedEntry::new(key, b, "q", b"text".to_vec(), vec![], 0, 1_000).expect_err("refused");
        assert_eq!(err, EntryError::NoEvidence(CacheClass::GeneratedResponse));
    }

    #[test]
    fn catalog_help_may_be_stored_without_evidence() {
        let b = ScopeBinding::new("acme", [0; 32], 1, OutputMode::Table);
        let key = ScopedCacheKey::new(CacheClass::CatalogHelp, &b, "list tables").unwrap();
        assert!(ScopedEntry::new(key, b, "list tables", b"[]".to_vec(), vec![], 0, 1_000).is_ok());
    }

    /// Assembled context is packed to a budget around one specific query, so a
    /// merely similar query gets context fitted to a different one.
    #[test]
    fn context_assembly_refuses_a_similar_query() {
        let refusal = admit_semantic(CacheClass::ContextAssembly, 0.999, 0.95).unwrap();
        assert_eq!(
            refusal,
            Some(RefusalReason::SemanticMatchNotAllowed(
                CacheClass::ContextAssembly
            ))
        );
    }

    #[test]
    fn a_similar_enough_query_is_admitted_for_a_class_that_allows_it() {
        assert_eq!(
            admit_semantic(CacheClass::GeneratedResponse, 0.97, 0.95).unwrap(),
            None
        );
    }

    #[test]
    fn a_threshold_below_the_floor_is_refused_outright() {
        let err = admit_semantic(CacheClass::GeneratedResponse, 1.0, 0.5).expect_err("refused");
        assert_eq!(
            err,
            ScopeError::ThresholdTooLow {
                requested: 0.5,
                floor: MIN_SEMANTIC_THRESHOLD
            }
        );
    }

    /// A similarity that is not a number is not "at least the threshold". The
    /// comparison is written so that NaN falls to the refusal branch rather
    /// than to the admitting one.
    #[test]
    fn an_unorderable_similarity_is_refused() {
        let refusal = admit_semantic(CacheClass::GeneratedResponse, f32::NAN, 0.95).unwrap();
        assert!(matches!(
            refusal,
            Some(RefusalReason::BelowThreshold { .. })
        ));
    }

    /// Exactly at the threshold is admitted; a hair under is not.
    #[test]
    fn the_threshold_boundary_is_inclusive() {
        assert_eq!(
            admit_semantic(CacheClass::RetrievalResult, 0.95, 0.95).unwrap(),
            None
        );
        assert!(
            admit_semantic(CacheClass::RetrievalResult, 0.95 - f32::EPSILON, 0.95)
                .unwrap()
                .is_some()
        );
    }

    /// Similarity never substitutes for the exact dimensions. Even an identical
    /// query under a changed policy is refused by `verify`, which runs first.
    #[test]
    fn similarity_cannot_rescue_a_changed_policy() {
        let cached = entry(&binding("acme", 1, 7), "how many orders shipped");
        assert_eq!(
            admit_semantic(CacheClass::GeneratedResponse, 1.0, 0.95).unwrap(),
            None
        );
        let verdict = verify(
            &cached,
            &binding("acme", 1, 8),
            CacheClass::GeneratedResponse,
            2_000,
            &versions_unchanged,
        );
        assert!(!verdict.is_hit());
    }

    /// Refusals are in the hit-rate denominator, so a cache that refuses
    /// everything cannot report success.
    #[test]
    fn refusals_count_against_the_hit_rate() {
        let mut counters = CacheOutcomeCounters::default();
        counters.record(
            CacheClass::GeneratedResponse,
            &Verdict::Fresh { evidence: vec![] },
        );
        counters.record(
            CacheClass::GeneratedResponse,
            &Verdict::Refused(RefusalReason::TenantMismatch),
        );
        counters.record(
            CacheClass::GeneratedResponse,
            &Verdict::Refused(RefusalReason::AuthorizationChanged),
        );
        counters.record(
            CacheClass::GeneratedResponse,
            &Verdict::Stale(StaleReason::SourceMissing {
                object_id: "d".into(),
            }),
        );
        assert_eq!(
            counters.hit_rate(CacheClass::GeneratedResponse),
            Some(0.25),
            "one fresh out of four decisions"
        );
        assert_eq!(
            counters.cross_scope_refusals(CacheClass::GeneratedResponse),
            2
        );
        assert_eq!(counters.hit_rate(CacheClass::CatalogHelp), None);
    }

    /// Sources are sorted and deduplicated, so the same evidence recorded in a
    /// different order produces the same entry.
    #[test]
    fn evidence_order_does_not_change_the_entry() {
        let b = binding("acme", 1, 7);
        let key = ScopedCacheKey::new(CacheClass::RetrievalResult, &b, "q").unwrap();
        let one = ScopedEntry::new(
            key.clone(),
            b.clone(),
            "q",
            vec![],
            vec![SourceSnapshot::new("b", 1), SourceSnapshot::new("a", 2)],
            0,
            1_000,
        )
        .unwrap();
        let two = ScopedEntry::new(
            key,
            b,
            "q",
            vec![],
            vec![
                SourceSnapshot::new("a", 2),
                SourceSnapshot::new("b", 1),
                SourceSnapshot::new("a", 2),
            ],
            0,
            1_000,
        )
        .unwrap();
        assert_eq!(one.sources, two.sources);
    }

    /// The same object at two versions is two snapshots, not a duplicate. If
    /// dedup collapsed them the entry would claim a consistency it does not
    /// have.
    #[test]
    fn two_versions_of_one_object_are_two_snapshots() {
        let b = binding("acme", 1, 7);
        let key = ScopedCacheKey::new(CacheClass::RetrievalResult, &b, "q").unwrap();
        let e = ScopedEntry::new(
            key,
            b,
            "q",
            vec![],
            vec![SourceSnapshot::new("a", 1), SourceSnapshot::new("a", 2)],
            0,
            1_000,
        )
        .unwrap();
        assert_eq!(e.sources.len(), 2);
    }

    /// A ttl that would overflow saturates rather than wrapping to a moment in
    /// the past, which would make a brand new entry instantly stale.
    #[test]
    fn an_enormous_ttl_does_not_wrap_into_the_past() {
        let b = ScopeBinding::new("acme", [0; 32], 1, OutputMode::Prose);
        let key = ScopedCacheKey::new(CacheClass::CatalogHelp, &b, "q").unwrap();
        let e = ScopedEntry::new(key, b, "q", vec![], vec![], u64::MAX - 1, u64::MAX).unwrap();
        assert_eq!(e.expires_at_ms, u64::MAX);
    }

    #[test]
    fn query_normalization_agrees_with_the_existing_cache() {
        let b = binding("acme", 1, 7);
        let a =
            ScopedCacheKey::new(CacheClass::GeneratedResponse, &b, "  How Many Orders  ").unwrap();
        let c = ScopedCacheKey::new(CacheClass::GeneratedResponse, &b, "how many orders").unwrap();
        assert_eq!(a, c);
    }

    /// The storage key is hex, so nothing in it can be read as a path
    /// separator. The old key interpolated a caller-supplied namespace into a
    /// slash-delimited path.
    #[test]
    fn the_storage_key_contains_no_caller_supplied_bytes() {
        let b = binding("acme/../globex", 1, 7);
        let key = ScopedCacheKey::new(CacheClass::GeneratedResponse, &b, "q").unwrap();
        let stored = String::from_utf8(key.to_storage_key()).unwrap();
        assert!(!stored.contains(".."), "{stored}");
        // `_cache` / `v2` / class tag / hex digest -- three separators, all
        // of them ours, none of them from the binding.
        assert_eq!(stored.matches('/').count(), 3, "{stored}");
        assert!(!stored.contains("globex"), "{stored}");
    }
}
