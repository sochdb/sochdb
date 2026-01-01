// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Compressed Sparse Row (CSR) adjacency format for HNSW graph.
//!
//! This module provides a cache-efficient graph representation that:
//! - Eliminates per-iteration allocations during search
//! - Stores neighbors in contiguous memory for linear iteration
//! - Uses dense u32 internal IDs for 4× bandwidth reduction
//!
//! # Data Layout
//!
//! ```text
//! Layer 0 (base layer):
//!   offsets:   [0, 3, 7, 12, ...]  (cumulative neighbor count per node)
//!   neighbors: [2, 5, 7 | 1, 3, 8, 9 | 0, 4, 6, 11, 15 | ...]
//!                 ^^^^^^^   ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
//!                 node 0    node 1        node 2
//! ```
//!
//! # Iteration Pattern
//!
//! ```rust,ignore
//! // Old (allocates per iteration):
//! let neighbors: SmallVec<[u128; 32]> = layer_data.neighbors
//!     .iter()
//!     .filter(|&&id| visited.insert(id))
//!     .copied()
//!     .collect();
//!
//! // New (zero allocation):
//! let start = csr.offsets[node_id];
//! let end = csr.offsets[node_id + 1];
//! for &neighbor_id in &csr.neighbors[start..end] {
//!     if visited.insert(neighbor_id) {
//!         // process neighbor
//!     }
//! }
//! ```
//!
//! # Memory Comparison (1M nodes, M=16)
//!
//! | Representation | Neighbor Storage | Total Graph |
//! |----------------|------------------|-------------|
//! | Vec<Vec<u128>> | 256 MB + 16M pointers | ~300 MB |
//! | SmallVec<u128> | 512 MB inline | ~550 MB |
//! | CSR<u32>       | 64 MB + 4 MB offsets | ~68 MB |

use crate::internal_id::InternalId;

/// CSR representation for a single layer's adjacency lists.
#[derive(Debug, Clone)]
pub struct CsrLayer {
    /// Offset array: offsets[i] is the start index of node i's neighbors in `neighbors`.
    /// Length: num_nodes + 1
    /// offsets[num_nodes] = neighbors.len() (sentinel)
    pub offsets: Vec<u32>,

    /// Flattened neighbor array containing all edges.
    /// Length: total number of edges in this layer
    pub neighbors: Vec<InternalId>,

    /// Maximum neighbors per node in this layer (M or M0).
    pub max_degree: usize,
}

impl CsrLayer {
    /// Create a new empty CSR layer with capacity for `num_nodes`.
    pub fn new(num_nodes: usize, max_degree: usize) -> Self {
        Self {
            offsets: vec![0; num_nodes + 1],
            neighbors: Vec::with_capacity(num_nodes * max_degree / 2), // Avg degree assumption
            max_degree,
        }
    }

    /// Create from existing adjacency lists (for migration from Vec<Vec>).
    ///
    /// # Arguments
    /// * `adjacency` - Slice of neighbor lists, indexed by internal node ID
    pub fn from_adjacency_lists(adjacency: &[Vec<InternalId>], max_degree: usize) -> Self {
        let num_nodes = adjacency.len();
        let total_edges: usize = adjacency.iter().map(|v| v.len()).sum();

        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut neighbors = Vec::with_capacity(total_edges);

        let mut offset = 0u32;
        for adj_list in adjacency {
            offsets.push(offset);
            neighbors.extend_from_slice(adj_list);
            offset += adj_list.len() as u32;
        }
        offsets.push(offset); // Sentinel

        Self {
            offsets,
            neighbors,
            max_degree,
        }
    }

    /// Get neighbors of a node as a slice.
    ///
    /// This is the hot-path operation - returns a contiguous slice with
    /// no allocation or iteration overhead.
    #[inline]
    pub fn neighbors(&self, node: InternalId) -> &[InternalId] {
        let idx = node.get() as usize;
        if idx >= self.offsets.len() - 1 {
            return &[];
        }
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        &self.neighbors[start..end]
    }

    /// Get number of neighbors for a node.
    #[inline]
    pub fn degree(&self, node: InternalId) -> usize {
        let idx = node.get() as usize;
        if idx >= self.offsets.len() - 1 {
            return 0;
        }
        (self.offsets[idx + 1] - self.offsets[idx]) as usize
    }

    /// Number of nodes in this layer.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total number of edges in this layer.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.neighbors.len()
    }

    /// Memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.offsets.len() * 4 + self.neighbors.len() * 4
    }
}

/// Immutable search view for HNSW graph.
///
/// This is a read-optimized snapshot of the graph that:
/// - Uses CSR format for all layers
/// - Uses dense u32 internal IDs
/// - Is completely lock-free for search operations
/// - Enables predictable prefetching patterns
///
/// Created via `HnswIndex::create_search_view()` when the graph is stable.
#[derive(Debug, Clone)]
pub struct CsrGraph {
    /// CSR adjacency for each layer (layer 0 is the base layer).
    pub layers: Vec<CsrLayer>,

    /// Entry point node (internal ID).
    pub entry_point: Option<InternalId>,

    /// Maximum layer index.
    pub max_layer: usize,

    /// Total number of nodes.
    pub num_nodes: usize,
}

impl CsrGraph {
    /// Create a new empty CSR graph.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            entry_point: None,
            max_layer: 0,
            num_nodes: 0,
        }
    }

    /// Create a CSR graph with specified number of layers and nodes.
    pub fn with_capacity(num_layers: usize, num_nodes: usize, max_degree: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| CsrLayer::new(num_nodes, max_degree))
            .collect();

        Self {
            layers,
            entry_point: None,
            max_layer: num_layers.saturating_sub(1),
            num_nodes,
        }
    }

    /// Get neighbors at a specific layer.
    #[inline]
    pub fn neighbors(&self, node: InternalId, layer: usize) -> &[InternalId] {
        if layer < self.layers.len() {
            self.layers[layer].neighbors(node)
        } else {
            &[]
        }
    }

    /// Check if a node exists at a given layer.
    #[inline]
    pub fn has_node_at_layer(&self, node: InternalId, layer: usize) -> bool {
        if layer >= self.layers.len() {
            return false;
        }
        (node.get() as usize) < self.layers[layer].num_nodes()
    }

    /// Total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.layers.iter().map(|l| l.memory_usage()).sum()
    }

    /// Get statistics about the graph.
    pub fn stats(&self) -> CsrGraphStats {
        let layer_stats: Vec<LayerStats> = self
            .layers
            .iter()
            .map(|layer| LayerStats {
                num_nodes: layer.num_nodes(),
                num_edges: layer.num_edges(),
                avg_degree: if layer.num_nodes() > 0 {
                    layer.num_edges() as f64 / layer.num_nodes() as f64
                } else {
                    0.0
                },
                memory_bytes: layer.memory_usage(),
            })
            .collect();

        CsrGraphStats {
            num_nodes: self.num_nodes,
            num_layers: self.layers.len(),
            total_edges: layer_stats.iter().map(|s| s.num_edges).sum(),
            total_memory_bytes: layer_stats.iter().map(|s| s.memory_bytes).sum(),
            layer_stats,
        }
    }
}

impl Default for CsrGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a single layer.
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f64,
    pub memory_bytes: usize,
}

/// Statistics for the entire CSR graph.
#[derive(Debug, Clone)]
pub struct CsrGraphStats {
    pub num_nodes: usize,
    pub num_layers: usize,
    pub total_edges: usize,
    pub total_memory_bytes: usize,
    pub layer_stats: Vec<LayerStats>,
}

/// Builder for incrementally constructing a CSR graph.
///
/// Used during index construction when edges are added incrementally.
/// After construction, call `build()` to get the immutable `CsrGraph`.
pub struct CsrGraphBuilder {
    /// Temporary adjacency lists (will be converted to CSR).
    layers: Vec<Vec<Vec<InternalId>>>,

    /// Max degree per layer (M0 for layer 0, M for others).
    max_degrees: Vec<usize>,

    /// Entry point node.
    entry_point: Option<InternalId>,
}

impl CsrGraphBuilder {
    /// Create a new builder for a graph with specified layers.
    pub fn new(num_layers: usize, max_degree: usize, max_degree_layer0: usize) -> Self {
        let max_degrees = (0..num_layers)
            .map(|l| if l == 0 { max_degree_layer0 } else { max_degree })
            .collect();

        let layers = (0..num_layers).map(|_| Vec::new()).collect();

        Self {
            layers,
            max_degrees,
            entry_point: None,
        }
    }

    /// Ensure node exists in all layers up to `layer`.
    fn ensure_node(&mut self, node: InternalId, layer: usize) {
        let idx = node.get() as usize;
        
        // Expand layers vector if needed
        while self.layers.len() <= layer {
            self.layers.push(Vec::new());
            self.max_degrees.push(*self.max_degrees.last().unwrap_or(&16));
        }

        // Expand each layer's node list if needed
        for l in 0..=layer {
            while self.layers[l].len() <= idx {
                self.layers[l].push(Vec::new());
            }
        }
    }

    /// Add an edge between two nodes at a specific layer.
    pub fn add_edge(&mut self, from: InternalId, to: InternalId, layer: usize) {
        self.ensure_node(from, layer);
        self.ensure_node(to, layer);

        let max_deg = self.max_degrees.get(layer).copied().unwrap_or(16);
        let neighbors = &mut self.layers[layer][from.get() as usize];
        
        if !neighbors.contains(&to) && neighbors.len() < max_deg {
            neighbors.push(to);
        }
    }

    /// Add bidirectional edge.
    pub fn add_bidirectional_edge(&mut self, a: InternalId, b: InternalId, layer: usize) {
        self.add_edge(a, b, layer);
        self.add_edge(b, a, layer);
    }

    /// Set the entry point.
    pub fn set_entry_point(&mut self, node: InternalId) {
        self.entry_point = Some(node);
    }

    /// Get current number of nodes (maximum across all layers).
    pub fn num_nodes(&self) -> usize {
        self.layers.iter().map(|l| l.len()).max().unwrap_or(0)
    }

    /// Build the final immutable CSR graph.
    pub fn build(self) -> CsrGraph {
        let num_nodes = self.num_nodes();
        let max_layer = self.layers.len().saturating_sub(1);

        let layers: Vec<CsrLayer> = self
            .layers
            .into_iter()
            .zip(self.max_degrees.iter())
            .map(|(adj, &max_deg)| CsrLayer::from_adjacency_lists(&adj, max_deg))
            .collect();

        CsrGraph {
            layers,
            entry_point: self.entry_point,
            max_layer,
            num_nodes,
        }
    }
}

/// Search candidate using internal ID.
#[derive(Debug, Clone)]
pub struct InternalSearchCandidate {
    pub distance: f32,
    pub id: InternalId,
}

impl Eq for InternalSearchCandidate {}

impl PartialEq for InternalSearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl PartialOrd for InternalSearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalSearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.get().cmp(&other.id.get()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_layer_basic() {
        let adjacency = vec![
            vec![InternalId::new(1), InternalId::new(2)],
            vec![InternalId::new(0), InternalId::new(2), InternalId::new(3)],
            vec![InternalId::new(0), InternalId::new(1)],
            vec![InternalId::new(1)],
        ];

        let layer = CsrLayer::from_adjacency_lists(&adjacency, 16);

        assert_eq!(layer.num_nodes(), 4);
        assert_eq!(layer.num_edges(), 8);

        // Check neighbors
        let n0 = layer.neighbors(InternalId::new(0));
        assert_eq!(n0.len(), 2);
        assert_eq!(n0[0], InternalId::new(1));
        assert_eq!(n0[1], InternalId::new(2));

        let n1 = layer.neighbors(InternalId::new(1));
        assert_eq!(n1.len(), 3);

        let n3 = layer.neighbors(InternalId::new(3));
        assert_eq!(n3.len(), 1);
        assert_eq!(n3[0], InternalId::new(1));
    }

    #[test]
    fn test_csr_layer_degree() {
        let adjacency = vec![
            vec![InternalId::new(1)],
            vec![InternalId::new(0), InternalId::new(2)],
            vec![],
        ];

        let layer = CsrLayer::from_adjacency_lists(&adjacency, 16);

        assert_eq!(layer.degree(InternalId::new(0)), 1);
        assert_eq!(layer.degree(InternalId::new(1)), 2);
        assert_eq!(layer.degree(InternalId::new(2)), 0);
    }

    #[test]
    fn test_csr_layer_empty() {
        let layer = CsrLayer::new(0, 16);
        assert_eq!(layer.num_nodes(), 0);
        assert_eq!(layer.num_edges(), 0);
        assert_eq!(layer.neighbors(InternalId::new(0)).len(), 0);
    }

    #[test]
    fn test_csr_graph_builder() {
        let mut builder = CsrGraphBuilder::new(3, 16, 32);

        // Add edges
        builder.add_bidirectional_edge(InternalId::new(0), InternalId::new(1), 0);
        builder.add_bidirectional_edge(InternalId::new(1), InternalId::new(2), 0);
        builder.add_bidirectional_edge(InternalId::new(0), InternalId::new(2), 0);

        // Add to higher layer
        builder.add_bidirectional_edge(InternalId::new(0), InternalId::new(2), 1);

        builder.set_entry_point(InternalId::new(0));

        let graph = builder.build();

        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.entry_point, Some(InternalId::new(0)));

        // Check layer 0 connectivity
        let n0 = graph.neighbors(InternalId::new(0), 0);
        assert_eq!(n0.len(), 2);

        // Check layer 1 connectivity
        let n0_l1 = graph.neighbors(InternalId::new(0), 1);
        assert_eq!(n0_l1.len(), 1);
        assert_eq!(n0_l1[0], InternalId::new(2));
    }

    #[test]
    fn test_csr_graph_stats() {
        let adjacency = vec![
            vec![InternalId::new(1), InternalId::new(2)],
            vec![InternalId::new(0)],
            vec![InternalId::new(0)],
        ];

        let layer = CsrLayer::from_adjacency_lists(&adjacency, 16);
        let graph = CsrGraph {
            layers: vec![layer],
            entry_point: Some(InternalId::new(0)),
            max_layer: 0,
            num_nodes: 3,
        };

        let stats = graph.stats();
        assert_eq!(stats.num_nodes, 3);
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.total_edges, 4);
        assert_eq!(stats.layer_stats[0].avg_degree, 4.0 / 3.0);
    }

    #[test]
    fn test_internal_search_candidate_ordering() {
        use std::collections::BinaryHeap;

        let mut heap = BinaryHeap::new();
        heap.push(InternalSearchCandidate {
            distance: 0.5,
            id: InternalId::new(1),
        });
        heap.push(InternalSearchCandidate {
            distance: 0.1,
            id: InternalId::new(2),
        });
        heap.push(InternalSearchCandidate {
            distance: 0.3,
            id: InternalId::new(3),
        });

        // Min-heap: should pop smallest distance first
        let first = heap.pop().unwrap();
        assert_eq!(first.distance, 0.1);
        assert_eq!(first.id, InternalId::new(2));
    }

    #[test]
    fn test_csr_memory_usage() {
        // 1000 nodes, avg 16 neighbors
        let mut adjacency = Vec::with_capacity(1000);
        for i in 0..1000u32 {
            let neighbors: Vec<InternalId> = (0..16)
                .map(|j| InternalId::new((i + j + 1) % 1000))
                .collect();
            adjacency.push(neighbors);
        }

        let layer = CsrLayer::from_adjacency_lists(&adjacency, 32);

        // Expected: 1001 * 4 (offsets) + 16000 * 4 (neighbors) = 68004 bytes
        let expected = 1001 * 4 + 16000 * 4;
        assert_eq!(layer.memory_usage(), expected);

        // Compare to Vec<Vec<u128>>: 1000 * (24 + 16 * 16) = 280000 bytes
        // CSR is ~4x smaller
    }
}
