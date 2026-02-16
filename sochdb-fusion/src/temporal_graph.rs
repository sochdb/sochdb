// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Temporal CSR Graph — Application-Level Graph Index with Temporal Edges
//!
//! This module generalizes SochDB's existing HNSW-internal CSR graph (`csr_graph.rs`)
//! to a general-purpose application graph over Knowledge Objects. Unlike the
//! KV-backed graph overlay (`sochdb-client/src/graph.rs`), which requires a KV lookup
//! and JSON deserialization per edge, this CSR graph provides:
//!
//! - **O(1) neighbor access**: contiguous memory layout eliminates pointer chasing
//! - **Cache-friendly traversal**: multi-hop BFS/DFS with predictable memory access
//! - **Temporal filtering**: edges carry validity intervals, enabling time-travel queries
//! - **Weight-based ranking**: edge weights enable shortest-path and influence propagation
//!
//! ## Memory Layout
//!
//! ```text
//! Node 0:  offsets[0]..offsets[1]  → edges[0..3]   (3 neighbors)
//! Node 1:  offsets[1]..offsets[2]  → edges[3..5]   (2 neighbors)
//! Node 2:  offsets[2]..offsets[3]  → edges[5..5]   (0 neighbors — isolated)
//! Node 3:  offsets[3]..offsets[4]  → edges[5..10]  (5 neighbors)
//! ...
//!
//! offsets: [0, 3, 5, 5, 10, ...]    ← prefix-sum of degrees
//! edges:   [TemporalEdge; total_edges]  ← contiguous edge array
//! ```
//!
//! ## Memory Comparison (1M nodes, avg degree 16)
//!
//! | Representation         | Memory   | Edge Access    |
//! |------------------------|----------|----------------|
//! | KV-backed (JSON)       | ~2.5 GB  | ~50 μs/edge    |
//! | HashMap<Vec<Edge>>     | ~550 MB  | ~200 ns/edge   |
//! | CSR (this module)      | ~300 MB  | ~10 ns/edge    |
//!
//! The CSR representation also enables SIMD-parallel neighbor scanning and
//! prefetch-friendly traversal patterns.

use serde::{Deserialize, Serialize};
use sochdb_core::knowledge_object::{EdgeKind, ObjectId};
use std::collections::HashMap;
use std::fmt;

/// A temporal edge in the CSR graph.
///
/// Compact representation: 32 bytes per edge (vs ~200+ bytes for JSON-serialized KV edges).
///
/// | Field       | Type  | Size | Purpose                    |
/// |-------------|-------|------|----------------------------|
/// | target      | u32   | 4B   | Internal node ID           |
/// | kind_id     | u16   | 2B   | Index into edge kind table |
/// | weight      | f32   | 4B   | Relationship strength      |
/// | valid_from  | u64   | 8B   | Temporal validity start    |
/// | valid_to    | u64   | 8B   | Temporal validity end      |
/// | _padding    | [u8;6]| 6B   | Alignment padding          |
///
/// Total: 32 bytes (cache-line aligned pair).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct TemporalEdge {
    /// Target node (internal ID, not ObjectId — mapped via `id_map`).
    pub target: u32,
    /// Index into the `GraphBuilder`'s edge kind table.
    pub kind_id: u16,
    /// Relationship strength in [0.0, 1.0].
    pub weight: f32,
    /// Temporal validity start (HLC-encoded microseconds).
    pub valid_from: u64,
    /// Temporal validity end (exclusive). `u64::MAX` = still valid.
    pub valid_to: u64,
}

impl TemporalEdge {
    /// Create a new temporal edge.
    pub fn new(target: u32, kind_id: u16, weight: f32, valid_from: u64, valid_to: u64) -> Self {
        Self {
            target,
            kind_id,
            weight,
            valid_from,
            valid_to,
        }
    }

    /// Check if this edge is valid at a given time.
    #[inline]
    pub fn valid_at(&self, time: u64) -> bool {
        self.valid_from <= time && time < self.valid_to
    }

    /// Check if this edge is currently valid.
    #[inline]
    pub fn is_current(&self) -> bool {
        self.valid_to == u64::MAX
    }
}

/// Immutable Temporal CSR Graph — application-level relationship index.
///
/// This is the generalization of `sochdb-index::csr_graph::CsrGraph` from
/// HNSW-internal to application-level relationships between Knowledge Objects.
///
/// Built via [`GraphBuilder`] and then frozen into an immutable, cache-optimized
/// representation for query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCsrGraph {
    /// Prefix-sum of node degrees. `offsets[i]..offsets[i+1]` gives edges for node `i`.
    offsets: Vec<u64>,

    /// Contiguous edge array.
    edges: Vec<TemporalEdge>,

    /// Number of nodes.
    num_nodes: usize,

    /// Bidirectional mapping: ObjectId ↔ internal u32 ID.
    oid_to_internal: HashMap<ObjectId, u32>,
    internal_to_oid: Vec<ObjectId>,

    /// Edge kind intern table: kind_id → EdgeKind.
    edge_kinds: Vec<EdgeKind>,
    kind_to_id: HashMap<String, u16>,
}

impl TemporalCsrGraph {
    /// Number of nodes in the graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of edges in the graph.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Look up the internal ID for an ObjectId.
    #[inline]
    pub fn internal_id(&self, oid: &ObjectId) -> Option<u32> {
        self.oid_to_internal.get(oid).copied()
    }

    /// Look up the ObjectId for an internal ID.
    #[inline]
    pub fn object_id(&self, internal: u32) -> Option<&ObjectId> {
        self.internal_to_oid.get(internal as usize)
    }

    /// Get all edges for a node (O(1) — contiguous slice).
    ///
    /// This is the **hot path** — it returns a contiguous slice into the
    /// edge array, enabling linear iteration with predictable prefetching.
    #[inline]
    pub fn edges(&self, node: u32) -> &[TemporalEdge] {
        let start = self.offsets[node as usize] as usize;
        let end = self.offsets[node as usize + 1] as usize;
        &self.edges[start..end]
    }

    /// Get edges for a node filtered by temporal validity.
    pub fn edges_valid_at(&self, node: u32, time: u64) -> Vec<&TemporalEdge> {
        self.edges(node)
            .iter()
            .filter(|e| e.valid_at(time))
            .collect()
    }

    /// Get edges for a node filtered by edge kind.
    pub fn edges_of_kind(&self, node: u32, kind_id: u16) -> Vec<&TemporalEdge> {
        self.edges(node)
            .iter()
            .filter(|e| e.kind_id == kind_id)
            .collect()
    }

    /// Degree of a node (total edges, including temporally expired ones).
    #[inline]
    pub fn degree(&self, node: u32) -> usize {
        let start = self.offsets[node as usize];
        let end = self.offsets[node as usize + 1];
        (end - start) as usize
    }

    /// Resolve an edge kind ID to its `EdgeKind`.
    #[inline]
    pub fn edge_kind(&self, kind_id: u16) -> Option<&EdgeKind> {
        self.edge_kinds.get(kind_id as usize)
    }

    /// Get the kind_id for a given edge kind label.
    pub fn kind_id(&self, label: &str) -> Option<u16> {
        self.kind_to_id.get(label).copied()
    }

    /// BFS traversal from a source node, returning all reachable nodes within
    /// `max_hops` hops, optionally filtered by time.
    ///
    /// Returns: `Vec<(node_id, hop_distance, path_weight)>`
    pub fn bfs(
        &self,
        source: u32,
        max_hops: u32,
        valid_time: Option<u64>,
    ) -> Vec<(u32, u32, f32)> {
        let mut visited = vec![false; self.num_nodes];
        let mut results = Vec::new();
        let mut queue = std::collections::VecDeque::new();

        visited[source as usize] = true;
        queue.push_back((source, 0u32, 1.0f32));

        while let Some((node, depth, path_weight)) = queue.pop_front() {
            if depth > 0 {
                results.push((node, depth, path_weight));
            }

            if depth >= max_hops {
                continue;
            }

            for edge in self.edges(node) {
                // Apply temporal filter if requested
                if let Some(time) = valid_time {
                    if !edge.valid_at(time) {
                        continue;
                    }
                }

                let target = edge.target;
                if !visited[target as usize] {
                    visited[target as usize] = true;
                    queue.push_back((target, depth + 1, path_weight * edge.weight));
                }
            }
        }

        results
    }

    /// Convert BFS results to ObjectIds.
    pub fn bfs_objects(
        &self,
        source: &ObjectId,
        max_hops: u32,
        valid_time: Option<u64>,
    ) -> Vec<(ObjectId, u32, f32)> {
        let Some(internal) = self.internal_id(source) else {
            return Vec::new();
        };

        self.bfs(internal, max_hops, valid_time)
            .into_iter()
            .filter_map(|(node, depth, weight)| {
                self.object_id(node).map(|oid| (*oid, depth, weight))
            })
            .collect()
    }

    /// Get the 1-hop neighborhood as a BitSet (for fused query composition).
    ///
    /// This is the key integration point with the fusion pipeline: the graph
    /// traversal produces a CandidateMask that can be ANDed with vector search
    /// results without materialization.
    pub fn neighborhood_bitset(
        &self,
        source: u32,
        max_hops: u32,
        valid_time: Option<u64>,
    ) -> crate::BitSet {
        let mut bitset = crate::BitSet::with_capacity(self.num_nodes);

        let reachable = self.bfs(source, max_hops, valid_time);
        for (node, _, _) in reachable {
            bitset.set(node as usize);
        }

        bitset
    }

    /// Memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let offsets = self.offsets.len() * std::mem::size_of::<u64>();
        let edges = self.edges.len() * std::mem::size_of::<TemporalEdge>();
        let oid_map = self.oid_to_internal.len() * (32 + 4); // OID + u32
        let internal_map = self.internal_to_oid.len() * 32;
        offsets + edges + oid_map + internal_map
    }
}

impl fmt::Display for TemporalCsrGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TemporalCsrGraph(nodes={}, edges={}, kinds={}, mem={} KB)",
            self.num_nodes,
            self.edges.len(),
            self.edge_kinds.len(),
            self.memory_usage() / 1024,
        )
    }
}

// =============================================================================
// Graph Builder
// =============================================================================

/// Builder for constructing a `TemporalCsrGraph`.
///
/// Edges are collected in adjacency-list format, then compacted into CSR
/// on `.build()`. The builder handles:
///
/// - ObjectId → internal ID mapping (automatic assignment)
/// - Edge kind interning (string → u16)
/// - Offset computation (prefix sum)
///
/// # Example
///
/// ```rust,ignore
/// let mut builder = GraphBuilder::new();
///
/// let alice = ObjectId::from_content(b"alice");
/// let acme = ObjectId::from_content(b"acme");
///
/// builder.add_edge(alice, acme, EdgeKind::typed("works_at"), 1.0, 0, u64::MAX);
///
/// let graph = builder.build();
/// assert_eq!(graph.num_nodes(), 2);
/// assert_eq!(graph.num_edges(), 1);
/// ```
pub struct GraphBuilder {
    /// Adjacency lists: node_id → Vec<(target, kind_id, weight, valid_from, valid_to)>
    adjacency: Vec<Vec<TemporalEdge>>,

    /// ObjectId → internal ID
    oid_to_internal: HashMap<ObjectId, u32>,
    internal_to_oid: Vec<ObjectId>,

    /// Edge kind intern table
    edge_kinds: Vec<EdgeKind>,
    kind_to_id: HashMap<String, u16>,
}

impl GraphBuilder {
    /// Create a new graph builder.
    pub fn new() -> Self {
        Self {
            adjacency: Vec::new(),
            oid_to_internal: HashMap::new(),
            internal_to_oid: Vec::new(),
            edge_kinds: Vec::new(),
            kind_to_id: HashMap::new(),
        }
    }

    /// Create a builder with pre-allocated capacity.
    pub fn with_capacity(num_nodes: usize) -> Self {
        Self {
            adjacency: Vec::with_capacity(num_nodes),
            oid_to_internal: HashMap::with_capacity(num_nodes),
            internal_to_oid: Vec::with_capacity(num_nodes),
            edge_kinds: Vec::new(),
            kind_to_id: HashMap::new(),
        }
    }

    /// Get or create an internal ID for an ObjectId.
    fn get_or_create_id(&mut self, oid: ObjectId) -> u32 {
        if let Some(&id) = self.oid_to_internal.get(&oid) {
            return id;
        }
        let id = self.internal_to_oid.len() as u32;
        self.oid_to_internal.insert(oid, id);
        self.internal_to_oid.push(oid);
        self.adjacency.push(Vec::new());
        id
    }

    /// Get or create a kind_id for an EdgeKind.
    fn get_or_create_kind(&mut self, kind: &EdgeKind) -> u16 {
        let label = kind.label().to_string();
        if let Some(&id) = self.kind_to_id.get(&label) {
            return id;
        }
        let id = self.edge_kinds.len() as u16;
        self.kind_to_id.insert(label, id);
        self.edge_kinds.push(kind.clone());
        id
    }

    /// Add a directed edge between two Knowledge Objects.
    pub fn add_edge(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        kind: EdgeKind,
        weight: f32,
        valid_from: u64,
        valid_to: u64,
    ) {
        let source_id = self.get_or_create_id(source);
        let target_id = self.get_or_create_id(target);
        let kind_id = self.get_or_create_kind(&kind);

        self.adjacency[source_id as usize].push(TemporalEdge::new(
            target_id, kind_id, weight, valid_from, valid_to,
        ));
    }

    /// Add a bidirectional edge (two directed edges).
    pub fn add_undirected_edge(
        &mut self,
        a: ObjectId,
        b: ObjectId,
        kind: EdgeKind,
        weight: f32,
        valid_from: u64,
        valid_to: u64,
    ) {
        self.add_edge(a, b, kind.clone(), weight, valid_from, valid_to);
        self.add_edge(b, a, kind, weight, valid_from, valid_to);
    }

    /// Register a node without any edges (ensures isolated nodes appear in the graph).
    pub fn add_node(&mut self, oid: ObjectId) -> u32 {
        self.get_or_create_id(oid)
    }

    /// Number of nodes added so far.
    pub fn num_nodes(&self) -> usize {
        self.internal_to_oid.len()
    }

    /// Number of edges added so far.
    pub fn num_edges(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum()
    }

    /// Build the immutable CSR graph.
    ///
    /// This freezes the graph into the cache-optimized CSR representation.
    /// The adjacency lists are compacted into contiguous arrays with a
    /// prefix-sum offset index.
    pub fn build(self) -> TemporalCsrGraph {
        let num_nodes = self.internal_to_oid.len();
        let total_edges: usize = self.adjacency.iter().map(|adj| adj.len()).sum();

        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut edges = Vec::with_capacity(total_edges);

        let mut offset = 0u64;
        for adj in &self.adjacency {
            offsets.push(offset);
            edges.extend_from_slice(adj);
            offset += adj.len() as u64;
        }
        offsets.push(offset);

        TemporalCsrGraph {
            offsets,
            edges,
            num_nodes,
            oid_to_internal: self.oid_to_internal,
            internal_to_oid: self.internal_to_oid,
            edge_kinds: self.edge_kinds,
            kind_to_id: self.kind_to_id,
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Conversion from Knowledge Objects
// =============================================================================

impl GraphBuilder {
    /// Bulk-load edges from a collection of Knowledge Objects.
    ///
    /// This extracts embedded edges from each object and adds them to the graph,
    /// creating the CSR representation that mirrors the embedded edge data but
    /// optimized for traversal.
    pub fn from_knowledge_objects<'a>(
        objects: impl IntoIterator<Item = &'a sochdb_core::KnowledgeObject>,
    ) -> Self {
        let mut builder = Self::new();

        for obj in objects {
            let source = obj.oid();
            builder.add_node(source);

            for edge in obj.edges() {
                builder.add_edge(
                    source,
                    edge.target,
                    edge.kind.clone(),
                    edge.weight,
                    edge.valid_from,
                    edge.valid_to,
                );
            }
        }

        builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_oid(name: &str) -> ObjectId {
        ObjectId::from_content(name.as_bytes())
    }

    #[test]
    fn test_graph_builder_basic() {
        let mut builder = GraphBuilder::new();

        let alice = make_oid("alice");
        let bob = make_oid("bob");
        let acme = make_oid("acme");

        builder.add_edge(alice, acme, EdgeKind::typed("works_at"), 1.0, 0, u64::MAX);
        builder.add_edge(bob, acme, EdgeKind::typed("works_at"), 0.8, 0, u64::MAX);
        builder.add_edge(alice, bob, EdgeKind::typed("knows"), 0.5, 0, u64::MAX);

        let graph = builder.build();

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);
    }

    #[test]
    fn test_csr_neighbor_access() {
        let mut builder = GraphBuilder::new();

        let a = make_oid("a");
        let b = make_oid("b");
        let c = make_oid("c");

        builder.add_edge(a, b, EdgeKind::typed("rel"), 1.0, 0, u64::MAX);
        builder.add_edge(a, c, EdgeKind::typed("rel"), 0.5, 0, u64::MAX);

        let graph = builder.build();

        let a_id = graph.internal_id(&a).unwrap();
        let edges = graph.edges(a_id);

        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_temporal_edge_filtering() {
        let mut builder = GraphBuilder::new();

        let a = make_oid("a");
        let b = make_oid("b");
        let c = make_oid("c");

        // a→b: valid from 100 to 200
        builder.add_edge(a, b, EdgeKind::typed("rel"), 1.0, 100, 200);
        // a→c: valid from 150 to MAX (still current)
        builder.add_edge(a, c, EdgeKind::typed("rel"), 0.5, 150, u64::MAX);

        let graph = builder.build();
        let a_id = graph.internal_id(&a).unwrap();

        // At time 120: only a→b
        let valid_120 = graph.edges_valid_at(a_id, 120);
        assert_eq!(valid_120.len(), 1);

        // At time 160: both a→b and a→c
        let valid_160 = graph.edges_valid_at(a_id, 160);
        assert_eq!(valid_160.len(), 2);

        // At time 250: only a→c
        let valid_250 = graph.edges_valid_at(a_id, 250);
        assert_eq!(valid_250.len(), 1);
    }

    #[test]
    fn test_bfs_traversal() {
        let mut builder = GraphBuilder::new();

        let a = make_oid("a");
        let b = make_oid("b");
        let c = make_oid("c");
        let d = make_oid("d");

        // a → b → c → d (linear chain)
        builder.add_edge(a, b, EdgeKind::typed("next"), 1.0, 0, u64::MAX);
        builder.add_edge(b, c, EdgeKind::typed("next"), 1.0, 0, u64::MAX);
        builder.add_edge(c, d, EdgeKind::typed("next"), 1.0, 0, u64::MAX);

        let graph = builder.build();
        let a_id = graph.internal_id(&a).unwrap();

        // 1-hop from a: only b
        let hops_1 = graph.bfs(a_id, 1, None);
        assert_eq!(hops_1.len(), 1);
        assert_eq!(hops_1[0].1, 1); // depth 1

        // 2-hops from a: b and c
        let hops_2 = graph.bfs(a_id, 2, None);
        assert_eq!(hops_2.len(), 2);

        // 3-hops from a: b, c, and d
        let hops_3 = graph.bfs(a_id, 3, None);
        assert_eq!(hops_3.len(), 3);
    }

    #[test]
    fn test_neighborhood_bitset() {
        let mut builder = GraphBuilder::new();

        let a = make_oid("a");
        let b = make_oid("b");
        let c = make_oid("c");

        builder.add_edge(a, b, EdgeKind::typed("rel"), 1.0, 0, u64::MAX);
        builder.add_edge(a, c, EdgeKind::typed("rel"), 0.5, 0, u64::MAX);

        let graph = builder.build();
        let a_id = graph.internal_id(&a).unwrap();

        let bitset = graph.neighborhood_bitset(a_id, 1, None);

        let b_id = graph.internal_id(&b).unwrap();
        let c_id = graph.internal_id(&c).unwrap();

        assert!(bitset.contains(b_id as usize));
        assert!(bitset.contains(c_id as usize));
        assert!(!bitset.contains(a_id as usize)); // source not included
    }

    #[test]
    fn test_isolated_nodes() {
        let mut builder = GraphBuilder::new();

        let a = make_oid("a");
        let b = make_oid("b");

        builder.add_node(a);
        builder.add_node(b);

        let graph = builder.build();

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 0);

        let a_id = graph.internal_id(&a).unwrap();
        assert_eq!(graph.degree(a_id), 0);
    }

    #[test]
    fn test_bfs_with_temporal_filter() {
        let mut builder = GraphBuilder::new();

        let a = make_oid("a");
        let b = make_oid("b");
        let c = make_oid("c");

        // a→b valid at time 100-200, a→c valid at time 300+
        builder.add_edge(a, b, EdgeKind::typed("rel"), 1.0, 100, 200);
        builder.add_edge(a, c, EdgeKind::typed("rel"), 1.0, 300, u64::MAX);

        let graph = builder.build();
        let a_id = graph.internal_id(&a).unwrap();

        // At time 150: only b is reachable
        let reachable = graph.bfs(a_id, 1, Some(150));
        assert_eq!(reachable.len(), 1);

        // At time 400: only c is reachable
        let reachable = graph.bfs(a_id, 1, Some(400));
        assert_eq!(reachable.len(), 1);
    }

    #[test]
    fn test_memory_usage() {
        let mut builder = GraphBuilder::new();

        for i in 0u32..1000 {
            let src = ObjectId::from_content(&i.to_le_bytes());
            let tgt = ObjectId::from_content(&(i + 1000).to_le_bytes());
            builder.add_edge(src, tgt, EdgeKind::typed("rel"), 1.0, 0, u64::MAX);
        }

        let graph = builder.build();
        let mem = graph.memory_usage();

        // Should be much less than KV-backed (which would be ~200 bytes per edge)
        assert!(mem < 1_000_000); // < 1 MB for 1000 edges
    }
}
