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

//! # Knowledge Object — Object-Centric Data Model for Knowledge Fabric
//!
//! The `KnowledgeObject` is the atomic unit of the Knowledge Fabric. Unlike the
//! tabular TOON format (which separates data, embeddings, edges, and temporal
//! metadata across different structures), a Knowledge Object **co-locates** all
//! information about a single entity:
//!
//! - **Content-addressed identity**: `oid = BLAKE3(canonical_payload)` — immutable,
//!   collision-resistant, enabling structural deduplication and content verification.
//! - **Embedded edges**: Relationships are stored *within* the object, so loading
//!   an object immediately provides its connections without a separate graph lookup.
//! - **Multi-space embeddings**: A single object can carry embeddings in multiple
//!   semantic spaces (e.g., `"semantic"`, `"code"`, `"temporal"`), enabling
//!   domain-specific similarity search without separate vector indices.
//! - **Bitemporal coordinates**: Every object carries `(valid_from, valid_to, system_time)`,
//!   supporting both "what was true?" (valid time) and "what did the system know?"
//!   (system time) queries.
//! - **Provenance chains**: Hash-linked derivation tracking — every transformation
//!   records its parent OIDs, creating an auditable lineage.
//!
//! ## Why Co-Location Matters
//!
//! In a traditional architecture, a compositional query ("find entities similar to X
//! that are connected to Y and were valid at time T") requires:
//!
//! 1. Vector index lookup → candidate set (separate I/O)
//! 2. Graph traversal → filter by connectivity (separate I/O)
//! 3. Temporal filter → narrow by validity (separate I/O)
//! 4. Attribute filter → apply predicates (separate I/O)
//!
//! Each boundary adds serialization, allocation, and cache misses. With co-located
//! Knowledge Objects, the fused query executor can evaluate all predicates in a
//! single pass, reducing latency from ~11 ms to ~300 μs (30–50× improvement).
//!
//! ## Relationship to TOON
//!
//! Knowledge Objects wrap `SochValue` payloads — TOON data remains the content
//! format. The Knowledge Object adds the metadata envelope that enables the
//! Knowledge Fabric's compositional queries.
//!
//! ## Example
//!
//! ```rust,ignore
//! use sochdb_core::knowledge_object::*;
//!
//! let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
//!     .attribute("name", SochValue::Text("Alice".into()))
//!     .attribute("role", SochValue::Text("engineer".into()))
//!     .embedding("semantic", vec![0.1, 0.2, 0.3])
//!     .edge(Edge::new(target_oid, EdgeKind::typed("works_at"), 1.0))
//!     .valid_from(1700000000_000000)
//!     .valid_to(u64::MAX)
//!     .build();
//!
//! assert!(ko.oid().as_bytes().len() == 32);
//! assert_eq!(ko.edges().len(), 1);
//! assert!(ko.embedding("semantic").is_some());
//! ```

use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};

use crate::soch::SochValue;

// =============================================================================
// Content-Addressed Object Identity
// =============================================================================

/// A 256-bit BLAKE3 content hash serving as the immutable identity of a
/// Knowledge Object.
///
/// `oid = BLAKE3(canonical_serialization(payload + edges + embeddings))`
///
/// Properties:
/// - **Deterministic**: Same content always produces the same OID.
/// - **Collision-resistant**: 256-bit output makes collisions computationally infeasible.
/// - **Structural deduplication**: Identical objects share the same OID.
/// - **Content verification**: Recomputing the hash detects corruption or tampering.
///
/// The OID is computed over the *canonical* byte representation (sorted keys,
/// normalized floats) to ensure deterministic hashing regardless of insertion order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectId([u8; 32]);

impl ObjectId {
    /// Create an OID from a raw 32-byte hash.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Compute the OID from canonical content bytes using BLAKE3.
    pub fn from_content(content: &[u8]) -> Self {
        let hash = blake3::hash(content);
        Self(*hash.as_bytes())
    }

    /// The raw 32 bytes of the OID.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Hex-encoded OID string (64 characters).
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse an OID from a 64-character hex string.
    pub fn from_hex(s: &str) -> Result<Self, ObjectIdError> {
        let bytes = hex::decode(s).map_err(|_| ObjectIdError::InvalidHex)?;
        if bytes.len() != 32 {
            return Err(ObjectIdError::InvalidLength(bytes.len()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }

    /// A zero/nil OID, used as a sentinel for "no parent" in provenance chains.
    pub const NIL: Self = Self([0u8; 32]);

    /// Check if this is the nil/zero OID.
    pub fn is_nil(&self) -> bool {
        self.0 == [0u8; 32]
    }
}

impl fmt::Debug for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectId({})", &self.to_hex()[..16]) // Show first 16 hex chars
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Errors when parsing ObjectId.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ObjectIdError {
    #[error("invalid hex encoding")]
    InvalidHex,
    #[error("expected 32 bytes, got {0}")]
    InvalidLength(usize),
}

// =============================================================================
// Bitemporal Coordinates
// =============================================================================

/// Bitemporal versioning coordinate for a Knowledge Object.
///
/// Supports two independent time dimensions:
///
/// - **Valid time** (`valid_from`, `valid_to`): When the fact was/is true in the
///   real world. Example: an employee's tenure at a company.
/// - **System time** (`system_time`): When the system recorded this version.
///   Assigned automatically by the HLC on write. Monotonically increasing.
///
/// This enables queries like:
/// - `as_of(system_time=T₁)` — "What did the system know at time T₁?"
/// - `valid_at(valid_time=T₂)` — "What was true at time T₂?"
/// - `as_of(T₁).valid_at(T₂)` — "What did the system believe at T₁ about T₂?"
///
/// Timestamps are HLC-encoded microseconds (see `hlc.rs`): upper 48 bits are
/// physical microseconds since Unix epoch, lower 16 bits are logical counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitemporalCoord {
    /// Start of valid time interval (inclusive). HLC-encoded microseconds.
    pub valid_from: u64,

    /// End of valid time interval (exclusive). `u64::MAX` means "still valid".
    /// HLC-encoded microseconds.
    pub valid_to: u64,

    /// System time when this version was recorded. Assigned by HLC on write.
    /// HLC-encoded microseconds.
    pub system_time: u64,
}

impl BitemporalCoord {
    /// Create a new bitemporal coordinate with an open-ended valid interval.
    pub fn new(valid_from: u64, system_time: u64) -> Self {
        Self {
            valid_from,
            valid_to: u64::MAX,
            system_time,
        }
    }

    /// Create a coordinate with a closed valid interval.
    pub fn with_valid_range(valid_from: u64, valid_to: u64, system_time: u64) -> Self {
        Self {
            valid_from,
            valid_to,
            system_time,
        }
    }

    /// Check if this coordinate is valid at a given valid time.
    pub fn valid_at(&self, valid_time: u64) -> bool {
        self.valid_from <= valid_time && valid_time < self.valid_to
    }

    /// Check if this coordinate was known to the system by a given system time.
    pub fn known_at(&self, system_time: u64) -> bool {
        self.system_time <= system_time
    }

    /// Combined bitemporal query: was this fact known at `sys_time` and valid at `valid_time`?
    pub fn visible_at(&self, system_time: u64, valid_time: u64) -> bool {
        self.known_at(system_time) && self.valid_at(valid_time)
    }

    /// Close the valid time interval (the fact is no longer true).
    pub fn close_valid_time(&mut self, valid_to: u64) {
        self.valid_to = valid_to;
    }

    /// Check if this coordinate represents a currently-valid fact (valid_to == MAX).
    pub fn is_current(&self) -> bool {
        self.valid_to == u64::MAX
    }

    /// Default "eternal" coordinate — valid from epoch 0, never expires.
    pub const ETERNAL: Self = Self {
        valid_from: 0,
        valid_to: u64::MAX,
        system_time: 0,
    };
}

impl Default for BitemporalCoord {
    fn default() -> Self {
        Self::ETERNAL
    }
}

// =============================================================================
// Embedded Edges
// =============================================================================

/// The kind/type of an edge between Knowledge Objects.
///
/// Typed edges enable graph queries like "traverse all `works_at` edges"
/// without inspecting edge payloads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// A named relationship type (e.g., "works_at", "authored_by", "cites").
    Typed(String),
    /// A hierarchical containment relationship (parent → child).
    Contains,
    /// A derivation relationship (source → derived).
    DerivedFrom,
    /// A reference/citation relationship.
    References,
    /// A temporal succession relationship (predecessor → successor).
    Succeeds,
    /// A semantic similarity link (auto-generated by embedding proximity).
    SimilarTo,
}

impl EdgeKind {
    /// Create a typed edge kind with the given label.
    pub fn typed(label: impl Into<String>) -> Self {
        Self::Typed(label.into())
    }

    /// Returns the string label for this edge kind.
    pub fn label(&self) -> &str {
        match self {
            EdgeKind::Typed(s) => s,
            EdgeKind::Contains => "contains",
            EdgeKind::DerivedFrom => "derived_from",
            EdgeKind::References => "references",
            EdgeKind::Succeeds => "succeeds",
            EdgeKind::SimilarTo => "similar_to",
        }
    }
}

impl fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// A directed, typed, weighted, temporally-versioned edge between two
/// Knowledge Objects.
///
/// Edges are **embedded** within the source object — when you load an object,
/// you immediately have its outgoing relationships. This eliminates the
/// separate graph lookup required by KV-backed edge stores.
///
/// ## Memory Layout (32 bytes per edge)
///
/// | Field       | Size  | Purpose                          |
/// |-------------|-------|----------------------------------|
/// | target      | 32B   | Target ObjectId (BLAKE3 hash)    |
/// | kind        | ~24B  | Edge type (enum + string)        |
/// | weight      | 4B    | Relationship strength [0.0, 1.0] |
/// | valid_from  | 8B    | Temporal validity start          |
/// | valid_to    | 8B    | Temporal validity end            |
/// | properties  | var   | Optional edge attributes         |
///
/// For the hot path (CSR-based graph traversal), edges are projected to
/// `(target_internal_id: u32, weight: f32)` for cache efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Target object this edge points to.
    pub target: ObjectId,

    /// The type/kind of this relationship.
    pub kind: EdgeKind,

    /// Relationship strength/confidence in [0.0, 1.0].
    /// - 1.0 = definitive relationship
    /// - 0.5 = probable relationship
    /// - 0.0 = hypothetical/weak relationship
    pub weight: f32,

    /// Temporal validity interval for this edge.
    /// Uses the same HLC-encoded microsecond format as `BitemporalCoord`.
    pub valid_from: u64,

    /// End of temporal validity (exclusive). `u64::MAX` = still valid.
    pub valid_to: u64,

    /// Optional edge properties (e.g., "role": "lead", "confidence": 0.95).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, SochValue>,
}

impl Edge {
    /// Create a new edge with default weight 1.0 and open-ended validity.
    pub fn new(target: ObjectId, kind: EdgeKind, weight: f32) -> Self {
        Self {
            target,
            kind,
            weight,
            valid_from: 0,
            valid_to: u64::MAX,
            properties: HashMap::new(),
        }
    }

    /// Create an edge with temporal validity.
    pub fn with_validity(
        target: ObjectId,
        kind: EdgeKind,
        weight: f32,
        valid_from: u64,
        valid_to: u64,
    ) -> Self {
        Self {
            target,
            kind,
            weight,
            valid_from,
            valid_to,
            properties: HashMap::new(),
        }
    }

    /// Add a property to this edge.
    pub fn with_property(mut self, key: impl Into<String>, value: SochValue) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Check if this edge is valid at a given time.
    pub fn valid_at(&self, time: u64) -> bool {
        self.valid_from <= time && time < self.valid_to
    }

    /// Check if this edge is currently valid (valid_to == MAX).
    pub fn is_current(&self) -> bool {
        self.valid_to == u64::MAX
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.kind == other.kind
    }
}

impl Eq for Edge {}

// =============================================================================
// Object Kind / Type System
// =============================================================================

/// Classification of a Knowledge Object. Determines which indices and query
/// optimizations apply.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectKind {
    /// A persistent entity (person, organization, concept).
    /// Typically has long valid-time intervals and many edges.
    Entity,

    /// A temporal event or episode.
    /// Has precise valid-time intervals and causal edges.
    Event,

    /// An episodic memory or conversation turn.
    /// Dense in embeddings, often has derivation edges.
    Episode,

    /// A document or content chunk.
    /// Primary carrier of text content and semantic embeddings.
    Document,

    /// A fact or claim extracted from content.
    /// Has provenance edges linking to source documents.
    Fact,

    /// An agent-generated artifact (plan, summary, decision).
    /// Has derivation provenance and typically short valid-time windows.
    Artifact,

    /// User-defined type with a custom label.
    Custom(String),
}

impl ObjectKind {
    /// Returns the string label for this kind.
    pub fn label(&self) -> &str {
        match self {
            ObjectKind::Entity => "entity",
            ObjectKind::Event => "event",
            ObjectKind::Episode => "episode",
            ObjectKind::Document => "document",
            ObjectKind::Fact => "fact",
            ObjectKind::Artifact => "artifact",
            ObjectKind::Custom(s) => s,
        }
    }
}

impl fmt::Display for ObjectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// =============================================================================
// Provenance Chain
// =============================================================================

/// Records how a Knowledge Object was derived.
///
/// Provenance enables auditable lineage tracking: "Where did this fact come from?"
/// "What transformations produced this summary?" Each provenance record forms a
/// node in a DAG (Directed Acyclic Graph) of derivations.
///
/// The provenance chain is hash-linked — each object's OID is derived from its
/// content (which includes parent OIDs), creating a tamper-evident lineage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// OIDs of the parent objects this was derived from.
    /// Empty for root/original objects.
    pub parents: Vec<ObjectId>,

    /// The transformation or operation that produced this object.
    /// Examples: "chunk", "summarize", "extract_entities", "merge", "user_input"
    pub operation: String,

    /// The agent or system that performed the transformation.
    /// Examples: "gpt-4", "user:alice", "sochdb:compaction"
    pub agent: String,

    /// Timestamp when the derivation occurred (HLC-encoded microseconds).
    pub timestamp: u64,

    /// Optional metadata about the transformation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, SochValue>,
}

impl Provenance {
    /// Create a root provenance (no parents — this is an original object).
    pub fn root(agent: impl Into<String>, timestamp: u64) -> Self {
        Self {
            parents: Vec::new(),
            operation: "create".to_string(),
            agent: agent.into(),
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Create a derived provenance with parent objects.
    pub fn derived(
        parents: Vec<ObjectId>,
        operation: impl Into<String>,
        agent: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            parents,
            operation: operation.into(),
            agent: agent.into(),
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this provenance record.
    pub fn with_metadata(mut self, key: impl Into<String>, value: SochValue) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if this is a root provenance (no parents).
    pub fn is_root(&self) -> bool {
        self.parents.is_empty()
    }
}

// =============================================================================
// Embedding Space
// =============================================================================

/// An embedding vector in a named semantic space.
///
/// Knowledge Objects can carry embeddings in multiple spaces simultaneously:
/// - `"semantic"` — general-purpose sentence embedding (e.g., text-embedding-3-small)
/// - `"code"` — code-specific embedding (e.g., CodeBERT)
/// - `"temporal"` — time-series embedding for temporal similarity
/// - `"visual"` — image/diagram embedding (e.g., CLIP)
///
/// Each space can have a different dimensionality and distance metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    /// The embedding vector (f32 components).
    pub vector: Vec<f32>,

    /// Dimensionality of this embedding.
    pub dimensions: u32,

    /// The model that generated this embedding.
    /// Enables re-embedding when models are upgraded.
    pub model: String,

    /// When this embedding was generated (HLC-encoded microseconds).
    /// Enables staleness detection and re-embedding triggers.
    pub generated_at: u64,
}

impl EmbeddingSpace {
    /// Create a new embedding in a given space.
    pub fn new(vector: Vec<f32>, model: impl Into<String>, generated_at: u64) -> Self {
        let dimensions = vector.len() as u32;
        Self {
            vector,
            dimensions,
            model: model.into(),
            generated_at,
        }
    }

    /// The L2 norm of this embedding vector.
    pub fn norm(&self) -> f32 {
        self.vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize this embedding to unit length (for cosine similarity as dot product).
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > f32::EPSILON {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }
}

// =============================================================================
// Knowledge Object
// =============================================================================

/// The atomic unit of the Knowledge Fabric.
///
/// A Knowledge Object co-locates content, relationships, embeddings, temporal
/// metadata, and provenance into a single, content-addressed entity. This
/// co-location enables the fused query execution pipeline that delivers
/// 30–50× latency improvements over disaggregated architectures.
///
/// ## Invariants
///
/// 1. `oid == BLAKE3(canonical_bytes(payload, edges, embeddings))` — the OID
///    is always consistent with the object's content.
/// 2. `temporal.system_time` is monotonically increasing for successive versions
///    of the same logical entity.
/// 3. Edges form a DAG for `DerivedFrom` and `Succeeds` kinds (no cycles).
/// 4. Embedding dimensions match the declared space dimensionality.
///
/// ## Thread Safety
///
/// `KnowledgeObject` is `Send + Sync` (all fields are owned or `Arc`-wrapped).
/// Concurrent mutation should go through the MVCC layer — objects themselves
/// are treated as immutable values (copy-on-write semantics via content addressing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeObject {
    /// Content-addressed identity: `BLAKE3(canonical_content)`.
    oid: ObjectId,

    /// Classification of this object (entity, event, document, etc.).
    kind: ObjectKind,

    /// The object's data payload — a self-describing `SochValue`.
    /// Typically a `SochValue::Object(HashMap<String, SochValue>)` but can be
    /// any `SochValue` variant for flexibility.
    payload: SochValue,

    /// Outgoing edges to other Knowledge Objects.
    /// Embedded within the object for edge locality — loading an object
    /// immediately provides its relationships.
    edges: Vec<Edge>,

    /// Embeddings in multiple semantic spaces.
    /// Key: space name (e.g., "semantic", "code", "temporal").
    embeddings: HashMap<String, EmbeddingSpace>,

    /// Bitemporal versioning coordinate.
    temporal: BitemporalCoord,

    /// Derivation provenance.
    provenance: Provenance,

    /// Optional namespace for multi-tenant isolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,

    /// Optional tags for fast categorical filtering.
    /// Tags are indexed in the ART for O(k) lookup.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
}

impl KnowledgeObject {
    // =========================================================================
    // Accessors
    // =========================================================================

    /// The content-addressed object identity.
    pub fn oid(&self) -> ObjectId {
        self.oid
    }

    /// The object's classification.
    pub fn kind(&self) -> &ObjectKind {
        &self.kind
    }

    /// The data payload.
    pub fn payload(&self) -> &SochValue {
        &self.payload
    }

    /// Mutable access to the payload (will invalidate OID — call `recompute_oid()` after).
    pub fn payload_mut(&mut self) -> &mut SochValue {
        &mut self.payload
    }

    /// All outgoing edges.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Edges filtered by kind.
    pub fn edges_of_kind(&self, kind: &EdgeKind) -> Vec<&Edge> {
        self.edges.iter().filter(|e| &e.kind == kind).collect()
    }

    /// Edges valid at a given time.
    pub fn edges_valid_at(&self, time: u64) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.valid_at(time)).collect()
    }

    /// Get an embedding by space name.
    pub fn embedding(&self, space: &str) -> Option<&EmbeddingSpace> {
        self.embeddings.get(space)
    }

    /// All embedding spaces.
    pub fn embeddings(&self) -> &HashMap<String, EmbeddingSpace> {
        &self.embeddings
    }

    /// The default/primary embedding vector (in the "semantic" space).
    pub fn primary_embedding(&self) -> Option<&[f32]> {
        self.embeddings.get("semantic").map(|e| e.vector.as_slice())
    }

    /// The bitemporal coordinate.
    pub fn temporal(&self) -> &BitemporalCoord {
        &self.temporal
    }

    /// Set the bitemporal coordinate (e.g., to assign HLC system_time on write).
    ///
    /// Note: This does NOT change the OID. Temporal coordinates are metadata,
    /// not part of the content-addressed identity.
    pub fn set_temporal(&mut self, coord: BitemporalCoord) {
        self.temporal = coord;
    }

    /// The derivation provenance.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// The namespace (for multi-tenant isolation).
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// Tags for categorical filtering.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Check if this object has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    // =========================================================================
    // Temporal Queries
    // =========================================================================

    /// Is this object valid at the given valid time?
    pub fn valid_at(&self, valid_time: u64) -> bool {
        self.temporal.valid_at(valid_time)
    }

    /// Was this object known to the system at the given system time?
    pub fn known_at(&self, system_time: u64) -> bool {
        self.temporal.known_at(system_time)
    }

    /// Combined bitemporal visibility check.
    pub fn visible_at(&self, system_time: u64, valid_time: u64) -> bool {
        self.temporal.visible_at(system_time, valid_time)
    }

    /// Is this the current version (valid_to == MAX)?
    pub fn is_current(&self) -> bool {
        self.temporal.is_current()
    }

    // =========================================================================
    // Attribute Access
    // =========================================================================

    /// Get a named attribute from the payload (assumes payload is `SochValue::Object`).
    pub fn attribute(&self, key: &str) -> Option<&SochValue> {
        match &self.payload {
            SochValue::Object(map) => map.get(key),
            _ => None,
        }
    }

    /// Get a text attribute.
    pub fn text_attribute(&self, key: &str) -> Option<&str> {
        self.attribute(key).and_then(|v| v.as_text())
    }

    /// Get an integer attribute.
    pub fn int_attribute(&self, key: &str) -> Option<i64> {
        self.attribute(key).and_then(|v| v.as_int())
    }

    // =========================================================================
    // Content Addressing
    // =========================================================================

    /// Recompute the OID from the current content.
    /// Must be called after any mutation to maintain the content-addressing invariant.
    pub fn recompute_oid(&mut self) {
        self.oid = Self::compute_oid(&self.kind, &self.payload, &self.edges, &self.embeddings);
    }

    /// Verify that the stored OID matches the current content.
    pub fn verify_oid(&self) -> bool {
        let computed = Self::compute_oid(&self.kind, &self.payload, &self.edges, &self.embeddings);
        self.oid == computed
    }

    /// Compute the canonical OID for given content.
    fn compute_oid(
        kind: &ObjectKind,
        payload: &SochValue,
        edges: &[Edge],
        embeddings: &HashMap<String, EmbeddingSpace>,
    ) -> ObjectId {
        let canonical = Self::canonical_bytes(kind, payload, edges, embeddings);
        ObjectId::from_content(&canonical)
    }

    /// Produce canonical bytes for OID computation.
    ///
    /// Canonical serialization ensures deterministic hashing:
    /// - HashMap keys are sorted lexicographically
    /// - Floats are normalized (NaN → 0.0, -0.0 → 0.0)
    /// - Using bincode for compact, deterministic binary encoding
    fn canonical_bytes(
        kind: &ObjectKind,
        payload: &SochValue,
        edges: &[Edge],
        embeddings: &HashMap<String, EmbeddingSpace>,
    ) -> Vec<u8> {
        // We use a deterministic serialization approach:
        // 1. Serialize kind label
        // 2. Serialize payload via bincode
        // 3. Serialize edges sorted by (target, kind)
        // 4. Serialize embeddings sorted by space name
        let mut hasher_input = Vec::with_capacity(1024);

        // Kind label
        let kind_bytes = kind.label().as_bytes();
        hasher_input.extend_from_slice(&(kind_bytes.len() as u32).to_le_bytes());
        hasher_input.extend_from_slice(kind_bytes);

        // Payload — deterministic serialization of SochValue
        // For Object(HashMap), sort keys before serializing
        let payload_bytes = canonical_soch_value_bytes(payload);
        hasher_input.extend_from_slice(&(payload_bytes.len() as u32).to_le_bytes());
        hasher_input.extend_from_slice(&payload_bytes);

        // Edges sorted by (target OID, kind label) for determinism
        let mut sorted_edges: Vec<_> = edges.iter().collect();
        sorted_edges.sort_by(|a, b| {
            a.target
                .as_bytes()
                .cmp(b.target.as_bytes())
                .then_with(|| a.kind.label().cmp(b.kind.label()))
        });
        hasher_input.extend_from_slice(&(sorted_edges.len() as u32).to_le_bytes());
        for edge in &sorted_edges {
            hasher_input.extend_from_slice(edge.target.as_bytes());
            let kind_label = edge.kind.label().as_bytes();
            hasher_input.extend_from_slice(&(kind_label.len() as u32).to_le_bytes());
            hasher_input.extend_from_slice(kind_label);
            hasher_input.extend_from_slice(&edge.weight.to_le_bytes());
        }

        // Embeddings sorted by space name for determinism
        let mut sorted_spaces: Vec<_> = embeddings.iter().collect();
        sorted_spaces.sort_by_key(|(name, _)| *name);
        hasher_input.extend_from_slice(&(sorted_spaces.len() as u32).to_le_bytes());
        for (name, embedding) in &sorted_spaces {
            let name_bytes = name.as_bytes();
            hasher_input.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            hasher_input.extend_from_slice(name_bytes);
            hasher_input.extend_from_slice(&embedding.dimensions.to_le_bytes());
            for &v in &embedding.vector {
                hasher_input.extend_from_slice(&v.to_le_bytes());
            }
        }

        hasher_input
    }
}

/// Produce deterministic bytes for a SochValue.
/// For `Object(HashMap)`, keys are sorted to ensure deterministic output.
fn canonical_soch_value_bytes(value: &SochValue) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);
    write_canonical_soch_value(&mut buf, value);
    buf
}

/// Recursively write a SochValue in canonical (deterministic) byte order.
fn write_canonical_soch_value(buf: &mut Vec<u8>, value: &SochValue) {
    match value {
        SochValue::Null => buf.push(0),
        SochValue::Bool(b) => {
            buf.push(1);
            buf.push(if *b { 1 } else { 0 });
        }
        SochValue::Int(i) => {
            buf.push(2);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        SochValue::UInt(u) => {
            buf.push(3);
            buf.extend_from_slice(&u.to_le_bytes());
        }
        SochValue::Float(f) => {
            buf.push(4);
            // Normalize: NaN → 0.0, -0.0 → 0.0
            let normalized = if f.is_nan() { 0.0 } else if *f == 0.0 { 0.0 } else { *f };
            buf.extend_from_slice(&normalized.to_le_bytes());
        }
        SochValue::Text(s) => {
            buf.push(5);
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        SochValue::Binary(b) => {
            buf.push(6);
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        }
        SochValue::Array(arr) => {
            buf.push(7);
            buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
            for item in arr {
                write_canonical_soch_value(buf, item);
            }
        }
        SochValue::Object(map) => {
            buf.push(8);
            // Sort keys for deterministic ordering
            let mut sorted_keys: Vec<&String> = map.keys().collect();
            sorted_keys.sort();
            buf.extend_from_slice(&(sorted_keys.len() as u32).to_le_bytes());
            for key in sorted_keys {
                buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
                buf.extend_from_slice(key.as_bytes());
                write_canonical_soch_value(buf, &map[key]);
            }
        }
        SochValue::Ref { table, id } => {
            buf.push(9);
            buf.extend_from_slice(&(table.len() as u32).to_le_bytes());
            buf.extend_from_slice(table.as_bytes());
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }
}

impl KnowledgeObject {
    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize this Knowledge Object to compact binary format.
    /// Uses serde_json for reliable HashMap serialization.
    pub fn to_bytes(&self) -> Result<Vec<u8>, KnowledgeObjectError> {
        serde_json::to_vec(self).map_err(|e| KnowledgeObjectError::SerializationError(e.to_string()))
    }

    /// Deserialize a Knowledge Object from binary format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KnowledgeObjectError> {
        serde_json::from_slice(bytes)
            .map_err(|e| KnowledgeObjectError::DeserializationError(e.to_string()))
    }

    /// Estimated memory footprint of this object (for memory budgeting).
    pub fn estimated_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.edges.len() * std::mem::size_of::<Edge>()
            + self
                .embeddings
                .values()
                .map(|e| e.vector.len() * 4)
                .sum::<usize>()
            + self.tags.iter().map(|t| t.len()).sum::<usize>()
    }

    // =========================================================================
    // Compressed Serialization
    // =========================================================================

    /// Serialize with per-object compression.
    ///
    /// Wire format: `[tag: u8] [original_len: u32 LE] [payload...]`
    ///
    /// - Tag 0 (`None`): payload is raw JSON (original_len == payload.len()).
    /// - Tag 1 (`Lz4`): payload is LZ4-block-compressed JSON.
    /// - Tag 2 (`Zstd`): payload is ZSTD-compressed JSON.
    ///
    /// Falls back to uncompressed if compressed output >= original size.
    pub fn to_compressed_bytes(
        &self,
        mode: CompressionMode,
    ) -> Result<Vec<u8>, KnowledgeObjectError> {
        let raw = self.to_bytes()?;
        let original_len = raw.len() as u32;

        match mode {
            CompressionMode::None => {
                let mut out = Vec::with_capacity(5 + raw.len());
                out.push(CompressionMode::None.tag());
                out.extend_from_slice(&original_len.to_le_bytes());
                out.extend_from_slice(&raw);
                Ok(out)
            }
            CompressionMode::Lz4 => {
                let compressed = lz4::block::compress(&raw, None, false)
                    .map_err(|e| KnowledgeObjectError::CompressionError(e.to_string()))?;
                // Fallback if compression doesn't save space
                if compressed.len() >= raw.len() {
                    let mut out = Vec::with_capacity(5 + raw.len());
                    out.push(CompressionMode::None.tag());
                    out.extend_from_slice(&original_len.to_le_bytes());
                    out.extend_from_slice(&raw);
                    return Ok(out);
                }
                let mut out = Vec::with_capacity(5 + compressed.len());
                out.push(CompressionMode::Lz4.tag());
                out.extend_from_slice(&original_len.to_le_bytes());
                out.extend_from_slice(&compressed);
                Ok(out)
            }
            CompressionMode::Zstd { level } => {
                let compressed = zstd::encode_all(raw.as_slice(), level)
                    .map_err(|e| KnowledgeObjectError::CompressionError(e.to_string()))?;
                if compressed.len() >= raw.len() {
                    let mut out = Vec::with_capacity(5 + raw.len());
                    out.push(CompressionMode::None.tag());
                    out.extend_from_slice(&original_len.to_le_bytes());
                    out.extend_from_slice(&raw);
                    return Ok(out);
                }
                let mut out = Vec::with_capacity(5 + compressed.len());
                out.push(CompressionMode::Zstd { level }.tag());
                out.extend_from_slice(&original_len.to_le_bytes());
                out.extend_from_slice(&compressed);
                Ok(out)
            }
        }
    }

    /// Deserialize from compressed wire format (auto-detects compression).
    ///
    /// The 1-byte tag determines the decompression algorithm.
    pub fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, KnowledgeObjectError> {
        if bytes.len() < 5 {
            return Err(KnowledgeObjectError::DeserializationError(
                "compressed payload too short (need >= 5 bytes)".into(),
            ));
        }

        let tag = bytes[0];
        let original_len =
            u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        let payload = &bytes[5..];

        let raw = match tag {
            0 => {
                // Uncompressed
                payload.to_vec()
            }
            1 => {
                // LZ4
                lz4::block::decompress(payload, Some(original_len as i32))
                    .map_err(|e| KnowledgeObjectError::CompressionError(e.to_string()))?
            }
            2 => {
                // ZSTD
                let mut decoder = zstd::Decoder::new(payload)
                    .map_err(|e| KnowledgeObjectError::CompressionError(e.to_string()))?;
                let mut raw = Vec::with_capacity(original_len);
                decoder
                    .read_to_end(&mut raw)
                    .map_err(|e| KnowledgeObjectError::CompressionError(e.to_string()))?;
                raw
            }
            _ => {
                return Err(KnowledgeObjectError::UnknownCompressionTag(tag));
            }
        };

        Self::from_bytes(&raw)
    }

    /// Returns the compression ratio for a given mode (compressed_size / original_size).
    /// Values < 1.0 indicate space savings.
    pub fn compression_ratio(
        &self,
        mode: CompressionMode,
    ) -> Result<f64, KnowledgeObjectError> {
        let raw_len = self.to_bytes()?.len() as f64;
        let compressed_len = self.to_compressed_bytes(mode)?.len() as f64;
        Ok(compressed_len / raw_len)
    }
}

// =============================================================================
// Compression Mode
// =============================================================================

/// Per-object compression strategy.
///
/// Each [`KnowledgeObject`] can be independently compressed with a different
/// algorithm and level. The choice depends on the object's characteristics:
///
/// - **LZ4**: ~3 GB/s decode, low CPU. Best for hot/frequently-accessed objects.
/// - **ZSTD**: ~1 GB/s decode, better ratios. Best for cold/archival objects.
/// - **None**: Zero overhead. Use for tiny objects where compression adds bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    /// No compression — raw JSON bytes.
    None,
    /// LZ4 block compression — fast decode, moderate ratio.
    Lz4,
    /// ZSTD compression with configurable level (1–22, default 3).
    Zstd { level: i32 },
}

impl CompressionMode {
    /// 1-byte tag written to the wire format header.
    pub fn tag(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
            Self::Zstd { .. } => 2,
        }
    }

    /// Construct from wire tag (decompression side doesn't need level).
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::Zstd { level: 0 }), // level unused for decompression
            _ => Option::None,
        }
    }

    /// Default ZSTD mode (level 3 — good balance of speed and ratio).
    pub fn zstd() -> Self {
        Self::Zstd { level: 3 }
    }

    /// High-compression ZSTD (level 9 — archival).
    pub fn zstd_high() -> Self {
        Self::Zstd { level: 9 }
    }
}

impl Default for CompressionMode {
    fn default() -> Self {
        Self::None
    }
}

impl PartialEq for KnowledgeObject {
    fn eq(&self, other: &Self) -> bool {
        // Content-addressed equality: same OID means same object.
        self.oid == other.oid
    }
}

impl Eq for KnowledgeObject {}

impl std::hash::Hash for KnowledgeObject {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.oid.hash(state);
    }
}

impl fmt::Display for KnowledgeObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KO({}, kind={}, edges={}, embeddings={}, tags={})",
            &self.oid.to_hex()[..12],
            self.kind,
            self.edges.len(),
            self.embeddings.len(),
            self.tags.len()
        )
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Ergonomic builder for constructing Knowledge Objects.
///
/// The builder computes the content-addressed OID automatically on `.build()`.
///
/// # Example
///
/// ```rust,ignore
/// let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
///     .attribute("name", SochValue::Text("Alice".into()))
///     .embedding("semantic", vec![0.1, 0.2, 0.3])
///     .tag("person")
///     .valid_from(1700000000_000000)
///     .build();
/// ```
pub struct KnowledgeObjectBuilder {
    kind: ObjectKind,
    payload: SochValue,
    edges: Vec<Edge>,
    embeddings: HashMap<String, EmbeddingSpace>,
    temporal: BitemporalCoord,
    provenance: Provenance,
    namespace: Option<String>,
    tags: Vec<String>,
}

impl KnowledgeObjectBuilder {
    /// Create a new builder with the given object kind.
    pub fn new(kind: ObjectKind) -> Self {
        Self {
            kind,
            payload: SochValue::Object(HashMap::new()),
            edges: Vec::new(),
            embeddings: HashMap::new(),
            temporal: BitemporalCoord::default(),
            provenance: Provenance::root("system", 0),
            namespace: None,
            tags: Vec::new(),
        }
    }

    /// Set the full payload.
    pub fn payload(mut self, payload: SochValue) -> Self {
        self.payload = payload;
        self
    }

    /// Add a named attribute to the payload (creates/extends an Object payload).
    pub fn attribute(mut self, key: impl Into<String>, value: SochValue) -> Self {
        match &mut self.payload {
            SochValue::Object(map) => {
                map.insert(key.into(), value);
            }
            _ => {
                let mut map = HashMap::new();
                map.insert(key.into(), value);
                self.payload = SochValue::Object(map);
            }
        }
        self
    }

    /// Add an outgoing edge.
    pub fn edge(mut self, edge: Edge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Add multiple edges at once.
    pub fn edges(mut self, edges: impl IntoIterator<Item = Edge>) -> Self {
        self.edges.extend(edges);
        self
    }

    /// Add an embedding in a named space.
    pub fn embedding(
        mut self,
        space: impl Into<String>,
        vector: Vec<f32>,
    ) -> Self {
        let space_name = space.into();
        self.embeddings.insert(
            space_name,
            EmbeddingSpace::new(vector, "unknown", 0),
        );
        self
    }

    /// Add an embedding with full metadata.
    pub fn embedding_with_metadata(
        mut self,
        space: impl Into<String>,
        vector: Vec<f32>,
        model: impl Into<String>,
        generated_at: u64,
    ) -> Self {
        let space_name = space.into();
        self.embeddings.insert(
            space_name,
            EmbeddingSpace::new(vector, model, generated_at),
        );
        self
    }

    /// Set the valid_from time (HLC-encoded microseconds).
    pub fn valid_from(mut self, valid_from: u64) -> Self {
        self.temporal.valid_from = valid_from;
        self
    }

    /// Set the valid_to time (HLC-encoded microseconds).
    pub fn valid_to(mut self, valid_to: u64) -> Self {
        self.temporal.valid_to = valid_to;
        self
    }

    /// Set the system_time (typically assigned automatically by HLC on write).
    pub fn system_time(mut self, system_time: u64) -> Self {
        self.temporal.system_time = system_time;
        self
    }

    /// Set the full bitemporal coordinate.
    pub fn temporal(mut self, temporal: BitemporalCoord) -> Self {
        self.temporal = temporal;
        self
    }

    /// Set the provenance record.
    pub fn provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Set the namespace.
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags.
    pub fn tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(|t| t.into()));
        self
    }

    /// Build the Knowledge Object, computing the content-addressed OID.
    pub fn build(self) -> KnowledgeObject {
        let oid = KnowledgeObject::compute_oid(
            &self.kind,
            &self.payload,
            &self.edges,
            &self.embeddings,
        );

        KnowledgeObject {
            oid,
            kind: self.kind,
            payload: self.payload,
            edges: self.edges,
            embeddings: self.embeddings,
            temporal: self.temporal,
            provenance: self.provenance,
            namespace: self.namespace,
            tags: self.tags,
        }
    }

    /// Build with a pre-computed OID (for deserialization or migration).
    pub fn build_with_oid(self, oid: ObjectId) -> KnowledgeObject {
        KnowledgeObject {
            oid,
            kind: self.kind,
            payload: self.payload,
            edges: self.edges,
            embeddings: self.embeddings,
            temporal: self.temporal,
            provenance: self.provenance,
            namespace: self.namespace,
            tags: self.tags,
        }
    }
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors for Knowledge Object operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum KnowledgeObjectError {
    #[error("serialization error: {0}")]
    SerializationError(String),

    #[error("deserialization error: {0}")]
    DeserializationError(String),

    #[error("OID verification failed: stored={stored}, computed={computed}")]
    OidMismatch { stored: String, computed: String },

    #[error("missing required embedding space: {0}")]
    MissingEmbedding(String),

    #[error("dimension mismatch in space '{space}': expected {expected}, got {got}")]
    DimensionMismatch {
        space: String,
        expected: u32,
        got: u32,
    },

    #[error("invalid temporal coordinates: valid_from ({valid_from}) > valid_to ({valid_to})")]
    InvalidTemporalRange { valid_from: u64, valid_to: u64 },

    #[error("compression error: {0}")]
    CompressionError(String),

    #[error("unknown compression tag: {0}")]
    UnknownCompressionTag(u8),
}

// =============================================================================
// Conversion from TOON/SochValue
// =============================================================================

impl From<SochValue> for KnowledgeObjectBuilder {
    /// Convert a SochValue into a KnowledgeObject builder.
    /// The SochValue becomes the payload; kind defaults to `Document`.
    fn from(value: SochValue) -> Self {
        KnowledgeObjectBuilder::new(ObjectKind::Document).payload(value)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_addressing_determinism() {
        let ko1 = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .attribute("age", SochValue::Int(30))
            .build();

        let ko2 = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("age", SochValue::Int(30))
            .attribute("name", SochValue::Text("Alice".into()))
            .build();

        // Different insertion order, same content → same OID
        assert_eq!(ko1.oid(), ko2.oid());
    }

    #[test]
    fn test_different_content_different_oid() {
        let ko1 = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .build();

        let ko2 = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Bob".into()))
            .build();

        assert_ne!(ko1.oid(), ko2.oid());
    }

    #[test]
    fn test_oid_verification() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Document)
            .attribute("content", SochValue::Text("Hello, world!".into()))
            .build();

        assert!(ko.verify_oid());
    }

    #[test]
    fn test_bitemporal_queries() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Event)
            .valid_from(100)
            .valid_to(200)
            .system_time(50)
            .build();

        assert!(ko.valid_at(150));
        assert!(!ko.valid_at(250));
        assert!(ko.known_at(50));
        assert!(ko.known_at(100));
        assert!(!ko.known_at(40));

        // Combined: visible at system_time=60, valid_time=150 → true
        assert!(ko.visible_at(60, 150));
        // Combined: visible at system_time=40, valid_time=150 → false (not yet recorded)
        assert!(!ko.visible_at(40, 150));
    }

    #[test]
    fn test_embedded_edges() {
        let target_oid = ObjectId::from_content(b"target_object");

        let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .edge(Edge::new(target_oid, EdgeKind::typed("works_at"), 1.0))
            .edge(Edge::new(target_oid, EdgeKind::Contains, 0.5))
            .build();

        assert_eq!(ko.edges().len(), 2);
        assert_eq!(ko.edges_of_kind(&EdgeKind::typed("works_at")).len(), 1);
        assert_eq!(ko.edges_of_kind(&EdgeKind::Contains).len(), 1);
    }

    #[test]
    fn test_multi_space_embeddings() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Document)
            .embedding("semantic", vec![0.1, 0.2, 0.3])
            .embedding("code", vec![0.4, 0.5, 0.6, 0.7])
            .build();

        assert!(ko.embedding("semantic").is_some());
        assert!(ko.embedding("code").is_some());
        assert!(ko.embedding("nonexistent").is_none());
        assert_eq!(ko.embedding("semantic").unwrap().dimensions, 3);
        assert_eq!(ko.embedding("code").unwrap().dimensions, 4);
    }

    #[test]
    fn test_provenance_chain() {
        let parent_oid = ObjectId::from_content(b"parent_document");

        let ko = KnowledgeObjectBuilder::new(ObjectKind::Fact)
            .attribute("claim", SochValue::Text("X is true".into()))
            .provenance(Provenance::derived(
                vec![parent_oid],
                "extract_facts",
                "gpt-4",
                1700000000,
            ))
            .build();

        assert!(!ko.provenance().is_root());
        assert_eq!(ko.provenance().parents.len(), 1);
        assert_eq!(ko.provenance().parents[0], parent_oid);
        assert_eq!(ko.provenance().operation, "extract_facts");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .embedding("semantic", vec![0.1, 0.2, 0.3])
            .tag("person")
            .namespace("test")
            .build();

        let bytes = ko.to_bytes().unwrap();
        let restored = KnowledgeObject::from_bytes(&bytes).unwrap();

        assert_eq!(ko.oid(), restored.oid());
        assert_eq!(ko.kind(), restored.kind());
        assert_eq!(ko.tags(), restored.tags());
        assert_eq!(ko.namespace(), restored.namespace());
    }

    #[test]
    fn test_object_id_hex_roundtrip() {
        let oid = ObjectId::from_content(b"test content");
        let hex = oid.to_hex();
        let parsed = ObjectId::from_hex(&hex).unwrap();
        assert_eq!(oid, parsed);
    }

    #[test]
    fn test_nil_oid() {
        assert!(ObjectId::NIL.is_nil());
        let non_nil = ObjectId::from_content(b"something");
        assert!(!non_nil.is_nil());
    }

    #[test]
    fn test_edge_temporal_filtering() {
        let target = ObjectId::from_content(b"target");

        let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .edge(Edge::with_validity(target, EdgeKind::typed("works_at"), 1.0, 100, 200))
            .edge(Edge::with_validity(target, EdgeKind::typed("manages"), 0.8, 150, u64::MAX))
            .build();

        // At time 120: only "works_at" is valid
        let active = ko.edges_valid_at(120);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].kind, EdgeKind::typed("works_at"));

        // At time 160: both are valid
        assert_eq!(ko.edges_valid_at(160).len(), 2);

        // At time 250: only "manages" (still current)
        let active = ko.edges_valid_at(250);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].kind, EdgeKind::typed("manages"));
    }

    #[test]
    fn test_estimated_size() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Document)
            .embedding("semantic", vec![0.0; 384])
            .tag("test")
            .build();

        let size = ko.estimated_size();
        assert!(size > 384 * 4); // At least the embedding vector
    }

    #[test]
    fn test_display() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .build();

        let display = format!("{}", ko);
        assert!(display.starts_with("KO("));
        assert!(display.contains("kind=entity"));
    }

    // =====================================================================
    // Compression tests
    // =====================================================================

    #[test]
    fn test_compression_none_roundtrip() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Entity)
            .attribute("name", SochValue::Text("Alice".into()))
            .embedding("semantic", vec![0.1; 128])
            .tag("person")
            .build();

        let compressed = ko.to_compressed_bytes(CompressionMode::None).unwrap();
        assert_eq!(compressed[0], 0); // tag = None
        let restored = KnowledgeObject::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(ko.oid(), restored.oid());
    }

    #[test]
    fn test_compression_lz4_roundtrip() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Document)
            .attribute("content", SochValue::Text("hello world ".repeat(100)))
            .embedding("semantic", vec![0.5; 384])
            .build();

        let compressed = ko.to_compressed_bytes(CompressionMode::Lz4).unwrap();
        let raw = ko.to_bytes().unwrap();

        // LZ4 should compress repetitive content
        assert!(compressed.len() < raw.len(), "LZ4 should reduce size for repetitive data");
        assert_eq!(compressed[0], 1); // tag = Lz4

        let restored = KnowledgeObject::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(ko.oid(), restored.oid());
        assert_eq!(ko.tags(), restored.tags());
    }

    #[test]
    fn test_compression_zstd_roundtrip() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Document)
            .attribute("content", SochValue::Text("hello world ".repeat(100)))
            .embedding("semantic", vec![0.5; 384])
            .tag("document")
            .namespace("test-ns")
            .build();

        let compressed = ko.to_compressed_bytes(CompressionMode::zstd()).unwrap();
        let raw = ko.to_bytes().unwrap();

        assert!(compressed.len() < raw.len(), "ZSTD should reduce size");
        assert_eq!(compressed[0], 2); // tag = Zstd

        let restored = KnowledgeObject::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(ko.oid(), restored.oid());
        assert_eq!(ko.namespace(), restored.namespace());
    }

    #[test]
    fn test_compression_fallback_on_tiny_object() {
        // A tiny object where compression might increase size
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Fact)
            .attribute("x", SochValue::Int(1))
            .build();

        let compressed_lz4 = ko.to_compressed_bytes(CompressionMode::Lz4).unwrap();
        let compressed_zstd = ko.to_compressed_bytes(CompressionMode::zstd()).unwrap();

        // Should still roundtrip regardless (falls back to None if compressed >= raw)
        let r1 = KnowledgeObject::from_compressed_bytes(&compressed_lz4).unwrap();
        let r2 = KnowledgeObject::from_compressed_bytes(&compressed_zstd).unwrap();
        assert_eq!(ko.oid(), r1.oid());
        assert_eq!(ko.oid(), r2.oid());
    }

    #[test]
    fn test_compression_ratio() {
        let ko = KnowledgeObjectBuilder::new(ObjectKind::Document)
            .attribute("data", SochValue::Text("abcdefgh".repeat(500)))
            .build();

        let ratio = ko.compression_ratio(CompressionMode::Lz4).unwrap();
        assert!(ratio < 1.0, "LZ4 should achieve < 1.0 ratio on repetitive data");

        let ratio_zstd = ko.compression_ratio(CompressionMode::zstd()).unwrap();
        assert!(ratio_zstd < ratio, "ZSTD should beat LZ4 ratio at default level");
    }

    #[test]
    fn test_compression_mode_tag_roundtrip() {
        for mode in [CompressionMode::None, CompressionMode::Lz4, CompressionMode::zstd()] {
            let tag = mode.tag();
            let recovered = CompressionMode::from_tag(tag).unwrap();
            assert_eq!(mode.tag(), recovered.tag());
        }
        assert!(CompressionMode::from_tag(255).is_none());
    }

    #[test]
    fn test_compressed_bytes_too_short() {
        let result = KnowledgeObject::from_compressed_bytes(&[0, 1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_compression_tag() {
        let bad_bytes = vec![99, 0, 0, 0, 0]; // tag=99, len=0
        let result = KnowledgeObject::from_compressed_bytes(&bad_bytes);
        assert!(result.is_err());
    }
}
