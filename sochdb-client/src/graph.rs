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

//! Semi-GraphDB Overlay for Agent Memory.
//!
//! Provides a lightweight graph layer on top of SochDB's KV storage for modeling
//! agent memory relationships:
//!
//! - Entity-to-entity relationships (user <-> conversation <-> message)
//! - Causal chains (action1 -> action2 -> action3)
//! - Reference graphs (document <- citation <- quote)
//!
//! # Storage Model (Phase 1b — binary keys, no JSON)
//!
//! Keys and values use compact binary encoding via `sochdb_core::edge_encoding`:
//!
//! - **Node keys**: `[0x01][ns_hash][record_id_key]` → binary `{node_type, properties}`
//! - **Edge keys**: `[0x02][ns_hash][from_key][edge_type_hash][to_key]` → binary `{edge_type, from, to, properties}`
//! - **Reverse index**: `[0x03][ns_hash][edge_type_hash][to_key][from_key]` → empty
//!
//! Properties use `SochValue` instead of `serde_json::Value`, eliminating the
//! JSON serialization overhead (~3-4× write amplification reduction).
//!
//! # Example
//!
//! ```rust,ignore
//! use sochdb::graph::{GraphOverlay, GraphNode, GraphEdge};
//! use sochdb::Connection;
//! use sochdb_core::{RecordId, SochValue};
//! use std::collections::HashMap;
//!
//! let conn = Connection::open("./agent_memory")?;
//! let graph = GraphOverlay::new(conn, "agent_001");
//!
//! // Create nodes
//! let mut props = HashMap::new();
//! props.insert("name".to_string(), SochValue::Text("Alice".to_string()));
//! let rid = RecordId::new("user", 1);
//! graph.add_node(&rid, "User", Some(props))?;
//!
//! // Create edges
//! let conv_id = RecordId::from_string("conv", "abc");
//! graph.add_edge(&rid, "STARTED", &conv_id, None)?;
//!
//! // Traverse graph
//! let path = graph.shortest_path(&rid, &conv_id, 10, None)?;
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use sochdb_core::edge_encoding;
use sochdb_core::record_id::RecordId;
use sochdb_core::SochValue;

use crate::ConnectionTrait;
use crate::error::{ClientError, Result};

/// Graph traversal order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalOrder {
    /// Breadth-first search
    BFS,
    /// Depth-first search
    DFS,
}

/// Edge direction for neighbor queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}

/// A node in the graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: RecordId,
    pub node_type: String,
    pub properties: HashMap<String, SochValue>,
}

/// An edge in the graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from_id: RecordId,
    pub edge_type: String,
    pub to_id: RecordId,
    pub properties: HashMap<String, SochValue>,
}

/// A neighboring node with its connecting edge.
#[derive(Debug, Clone)]
pub struct Neighbor {
    pub node_id: RecordId,
    pub edge: GraphEdge,
}

/// A subgraph containing nodes and edges.
#[derive(Debug, Clone)]
pub struct Subgraph {
    pub nodes: HashMap<String, GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Lightweight graph overlay on SochDB.
///
/// Provides graph operations for agent memory without a full graph database.
/// Uses the underlying KV store for persistence with O(1) node/edge operations.
///
/// All keys and values use compact binary encoding — no JSON serialization.
pub struct GraphOverlay<C: ConnectionTrait> {
    conn: C,
    namespace: String,
}

impl<C: ConnectionTrait> GraphOverlay<C> {
    /// Create a new graph overlay.
    ///
    /// # Arguments
    ///
    /// * `conn` - SochDB connection
    /// * `namespace` - Namespace for graph isolation (e.g., agent_id)
    pub fn new(conn: C, namespace: impl Into<String>) -> Self {
        let namespace = namespace.into();
        Self { conn, namespace }
    }

    /// Access the namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    // =========================================================================
    // Node Operations
    // =========================================================================

    /// Add a node to the graph.
    ///
    /// # Arguments
    ///
    /// * `record_id` - Unique node identifier (table:id)
    /// * `node_type` - Node type label (e.g., "User", "Message", "Tool")
    /// * `properties` - Optional node properties
    ///
    /// # Returns
    ///
    /// The created GraphNode
    pub fn add_node(
        &self,
        record_id: &RecordId,
        node_type: &str,
        properties: Option<HashMap<String, SochValue>>,
    ) -> Result<GraphNode> {
        let props = properties.unwrap_or_default();
        let node = GraphNode {
            id: record_id.clone(),
            node_type: node_type.to_string(),
            properties: props.clone(),
        };

        let key = edge_encoding::node_key(&self.namespace, record_id);
        let value = edge_encoding::encode_node_value(node_type, &props);
        self.conn.put(&key, &value)?;
        Ok(node)
    }

    /// Get a node by RecordId.
    pub fn get_node(&self, record_id: &RecordId) -> Result<Option<GraphNode>> {
        let key = edge_encoding::node_key(&self.namespace, record_id);
        match self.conn.get(&key)? {
            Some(data) => {
                let (node_type, properties) = edge_encoding::decode_node_value(&data)
                    .ok_or_else(|| ClientError::Serialization("corrupt node value".to_string()))?;
                Ok(Some(GraphNode {
                    id: record_id.clone(),
                    node_type,
                    properties,
                }))
            }
            None => Ok(None),
        }
    }

    /// Update a node's properties or type.
    pub fn update_node(
        &self,
        record_id: &RecordId,
        properties: Option<HashMap<String, SochValue>>,
        node_type: Option<&str>,
    ) -> Result<Option<GraphNode>> {
        let mut node = match self.get_node(record_id)? {
            Some(n) => n,
            None => return Ok(None),
        };

        if let Some(props) = properties {
            for (k, v) in props {
                node.properties.insert(k, v);
            }
        }
        if let Some(nt) = node_type {
            node.node_type = nt.to_string();
        }

        let key = edge_encoding::node_key(&self.namespace, record_id);
        let value = edge_encoding::encode_node_value(&node.node_type, &node.properties);
        self.conn.put(&key, &value)?;
        Ok(Some(node))
    }

    /// Delete a node from the graph.
    pub fn delete_node(&self, record_id: &RecordId, cascade: bool) -> Result<bool> {
        if self.get_node(record_id)?.is_none() {
            return Ok(false);
        }

        if cascade {
            // Delete outgoing edges
            for edge in self.get_edges(record_id, None)? {
                self.delete_edge(&edge.from_id, &edge.edge_type, &edge.to_id)?;
            }

            // Delete incoming edges
            for edge in self.get_incoming_edges(record_id, None)? {
                self.delete_edge(&edge.from_id, &edge.edge_type, &edge.to_id)?;
            }
        }

        let key = edge_encoding::node_key(&self.namespace, record_id);
        self.conn.delete(&key)?;
        Ok(true)
    }

    /// Check if a node exists.
    pub fn node_exists(&self, record_id: &RecordId) -> Result<bool> {
        let key = edge_encoding::node_key(&self.namespace, record_id);
        Ok(self.conn.get(&key)?.is_some())
    }

    // =========================================================================
    // Edge Operations
    // =========================================================================

    /// Add an edge between two nodes.
    pub fn add_edge(
        &self,
        from_id: &RecordId,
        edge_type: &str,
        to_id: &RecordId,
        properties: Option<HashMap<String, SochValue>>,
    ) -> Result<GraphEdge> {
        let props = properties.unwrap_or_default();
        let edge = GraphEdge {
            from_id: from_id.clone(),
            edge_type: edge_type.to_string(),
            to_id: to_id.clone(),
            properties: props.clone(),
        };

        // Store edge
        let key = edge_encoding::edge_key(&self.namespace, from_id, edge_type, to_id);
        let value = edge_encoding::encode_edge_value(from_id, edge_type, to_id, &props);
        self.conn.put(&key, &value)?;

        // Store reverse index (value is empty — all info is in the key)
        let rev_key = edge_encoding::reverse_key(&self.namespace, edge_type, to_id, from_id);
        self.conn.put(&rev_key, &[])?;

        Ok(edge)
    }

    /// Get a specific edge.
    pub fn get_edge(
        &self,
        from_id: &RecordId,
        edge_type: &str,
        to_id: &RecordId,
    ) -> Result<Option<GraphEdge>> {
        let key = edge_encoding::edge_key(&self.namespace, from_id, edge_type, to_id);
        match self.conn.get(&key)? {
            Some(data) => {
                let decoded = edge_encoding::decode_edge_value(&data)
                    .ok_or_else(|| ClientError::Serialization("corrupt edge value".to_string()))?;
                Ok(Some(GraphEdge {
                    from_id: decoded.from_id,
                    edge_type: decoded.edge_type,
                    to_id: decoded.to_id,
                    properties: decoded.properties,
                }))
            }
            None => Ok(None),
        }
    }

    /// Get all outgoing edges from a node, optionally filtered by edge type.
    pub fn get_edges(
        &self,
        from_id: &RecordId,
        edge_type: Option<&str>,
    ) -> Result<Vec<GraphEdge>> {
        let prefix = match edge_type {
            Some(et) => edge_encoding::edge_from_type_prefix(&self.namespace, from_id, et),
            None => edge_encoding::edge_from_prefix(&self.namespace, from_id),
        };
        let results = self.conn.scan(&prefix)?;

        let mut edges = Vec::new();
        for (_, value) in results {
            if let Some(decoded) = edge_encoding::decode_edge_value(&value) {
                edges.push(GraphEdge {
                    from_id: decoded.from_id,
                    edge_type: decoded.edge_type,
                    to_id: decoded.to_id,
                    properties: decoded.properties,
                });
            }
        }

        Ok(edges)
    }

    /// Get all incoming edges to a node, optionally filtered by edge type.
    pub fn get_incoming_edges(
        &self,
        to_id: &RecordId,
        edge_type: Option<&str>,
    ) -> Result<Vec<GraphEdge>> {
        let mut edges = Vec::new();

        if let Some(et) = edge_type {
            // Query specific edge type via reverse index
            let prefix = edge_encoding::reverse_type_to_prefix(&self.namespace, et, to_id);
            let results = self.conn.scan(&prefix)?;

            for (rev_key, _) in results {
                // Decode the reverse key to get from_key
                if let Some(decoded) = edge_encoding::decode_reverse_key(&rev_key) {
                    // Reconstruct from_id from its key bytes
                    if let Some(from_id) = RecordId::from_key(&decoded.from_key) {
                        if let Some(edge) = self.get_edge(&from_id, et, to_id)? {
                            edges.push(edge);
                        }
                    }
                }
            }
        } else {
            // Scan all reverse entries for this namespace
            let prefix = edge_encoding::reverse_prefix(&self.namespace);
            let results = self.conn.scan(&prefix)?;

            for (rev_key, _) in results {
                if let Some(decoded) = edge_encoding::decode_reverse_key(&rev_key) {
                    // Check if the to_key matches
                    let to_key = to_id.to_key();
                    if decoded.to_key == to_key {
                        if let Some(from_id) = RecordId::from_key(&decoded.from_key) {
                            // We need to scan all edge types from this from_id to find edges to to_id
                            let from_prefix = edge_encoding::edge_from_prefix(&self.namespace, &from_id);
                            let from_results = self.conn.scan(&from_prefix)?;
                            for (_, val) in from_results {
                                if let Some(edge_decoded) = edge_encoding::decode_edge_value(&val) {
                                    if edge_decoded.to_id == *to_id {
                                        edges.push(GraphEdge {
                                            from_id: edge_decoded.from_id,
                                            edge_type: edge_decoded.edge_type,
                                            to_id: edge_decoded.to_id,
                                            properties: edge_decoded.properties,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(edges)
    }

    /// Delete an edge.
    pub fn delete_edge(
        &self,
        from_id: &RecordId,
        edge_type: &str,
        to_id: &RecordId,
    ) -> Result<bool> {
        let key = edge_encoding::edge_key(&self.namespace, from_id, edge_type, to_id);
        if self.conn.get(&key)?.is_none() {
            return Ok(false);
        }

        self.conn.delete(&key)?;

        // Delete reverse index
        let rev_key = edge_encoding::reverse_key(&self.namespace, edge_type, to_id, from_id);
        self.conn.delete(&rev_key)?;

        Ok(true)
    }

    // =========================================================================
    // Traversal Operations
    // =========================================================================

    /// Breadth-first search from a starting node.
    pub fn bfs(
        &self,
        start_id: &RecordId,
        max_depth: usize,
        edge_types: Option<&[&str]>,
        node_types: Option<&[&str]>,
    ) -> Result<Vec<RecordId>> {
        self.traverse(start_id, max_depth, edge_types, node_types, TraversalOrder::BFS)
    }

    /// Depth-first search from a starting node.
    pub fn dfs(
        &self,
        start_id: &RecordId,
        max_depth: usize,
        edge_types: Option<&[&str]>,
        node_types: Option<&[&str]>,
    ) -> Result<Vec<RecordId>> {
        self.traverse(start_id, max_depth, edge_types, node_types, TraversalOrder::DFS)
    }

    fn traverse(
        &self,
        start_id: &RecordId,
        max_depth: usize,
        edge_types: Option<&[&str]>,
        node_types: Option<&[&str]>,
        order: TraversalOrder,
    ) -> Result<Vec<RecordId>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();

        let edge_type_set: HashSet<&str> = edge_types
            .map(|e| e.iter().copied().collect())
            .unwrap_or_default();
        let node_type_set: HashSet<&str> = node_types
            .map(|n| n.iter().copied().collect())
            .unwrap_or_default();

        let mut frontier: VecDeque<(RecordId, usize)> = VecDeque::new();
        frontier.push_back((start_id.clone(), 0));

        while let Some((node_id, depth)) = match order {
            TraversalOrder::BFS => frontier.pop_front(),
            TraversalOrder::DFS => frontier.pop_back(),
        } {
            let node_key_bytes = node_id.to_key();
            if visited.contains(&node_key_bytes) {
                continue;
            }
            visited.insert(node_key_bytes);

            // Check node type filter
            if node_types.is_some() && !node_type_set.is_empty() {
                if let Some(node) = self.get_node(&node_id)? {
                    if !node_type_set.contains(node.node_type.as_str()) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            result.push(node_id.clone());

            if depth >= max_depth {
                continue;
            }

            // Get outgoing edges
            for edge in self.get_edges(&node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }
                let to_key = edge.to_id.to_key();
                if !visited.contains(&to_key) {
                    frontier.push_back((edge.to_id, depth + 1));
                }
            }
        }

        Ok(result)
    }

    /// Find shortest path between two nodes using BFS.
    pub fn shortest_path(
        &self,
        from_id: &RecordId,
        to_id: &RecordId,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> Result<Option<Vec<RecordId>>> {
        if from_id == to_id {
            return Ok(Some(vec![from_id.clone()]));
        }

        let mut visited: HashSet<Vec<u8>> = HashSet::new();
        visited.insert(from_id.to_key());
        let mut parent: HashMap<Vec<u8>, (Vec<u8>, RecordId)> = HashMap::new();

        let edge_type_set: HashSet<&str> = edge_types
            .map(|e| e.iter().copied().collect())
            .unwrap_or_default();

        let mut frontier: VecDeque<(RecordId, usize)> = VecDeque::new();
        frontier.push_back((from_id.clone(), 0));

        let to_key = to_id.to_key();

        while let Some((node_id, depth)) = frontier.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let node_key_bytes = node_id.to_key();

            for edge in self.get_edges(&node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }

                let next_key = edge.to_id.to_key();
                if visited.contains(&next_key) {
                    continue;
                }

                visited.insert(next_key.clone());
                parent.insert(next_key.clone(), (node_key_bytes.clone(), node_id.clone()));

                if next_key == to_key {
                    // Reconstruct path
                    let mut path = vec![to_id.clone()];
                    let mut curr_key = to_key.clone();
                    while let Some((_parent_key, parent_rid)) = parent.get(&curr_key) {
                        path.push(parent_rid.clone());
                        curr_key = _parent_key.clone();
                    }
                    path.reverse();
                    return Ok(Some(path));
                }

                frontier.push_back((edge.to_id, depth + 1));
            }
        }

        Ok(None) // No path found
    }

    // =========================================================================
    // Query Operations
    // =========================================================================

    /// Get neighboring nodes with their connecting edges.
    pub fn get_neighbors(
        &self,
        node_id: &RecordId,
        edge_types: Option<&[&str]>,
        direction: EdgeDirection,
    ) -> Result<Vec<Neighbor>> {
        let mut neighbors = Vec::new();
        let edge_type_set: HashSet<&str> = edge_types
            .map(|e| e.iter().copied().collect())
            .unwrap_or_default();

        if matches!(direction, EdgeDirection::Outgoing | EdgeDirection::Both) {
            for edge in self.get_edges(node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }
                neighbors.push(Neighbor {
                    node_id: edge.to_id.clone(),
                    edge,
                });
            }
        }

        if matches!(direction, EdgeDirection::Incoming | EdgeDirection::Both) {
            for edge in self.get_incoming_edges(node_id, None)? {
                if edge_types.is_some() && !edge_type_set.is_empty() {
                    if !edge_type_set.contains(edge.edge_type.as_str()) {
                        continue;
                    }
                }
                neighbors.push(Neighbor {
                    node_id: edge.from_id.clone(),
                    edge,
                });
            }
        }

        Ok(neighbors)
    }

    /// Get all nodes of a specific type.
    ///
    /// Note: This scans all nodes in the namespace, use sparingly for large graphs.
    pub fn get_nodes_by_type(
        &self,
        node_type: &str,
        limit: usize,
    ) -> Result<Vec<GraphNode>> {
        let prefix = edge_encoding::node_prefix(&self.namespace);
        let results = self.conn.scan(&prefix)?;

        let mut nodes = Vec::new();
        for (key, value) in results {
            if let Some((nt, properties)) = edge_encoding::decode_node_value(&value) {
                if nt == node_type {
                    // Reconstruct RecordId from key (lossy — table name is hash)
                    // The node key is [0x01][ns_hash: 4B][rid_key...]
                    let rid_key = &key[5..]; // skip tag + ns_hash
                    let rid = RecordId::from_key(rid_key).unwrap_or_else(|| {
                        RecordId::from_string("_unknown", "?")
                    });
                    nodes.push(GraphNode {
                        id: rid,
                        node_type: nt,
                        properties,
                    });
                    if limit > 0 && nodes.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(nodes)
    }

    /// Get a subgraph starting from a node.
    pub fn get_subgraph(
        &self,
        start_id: &RecordId,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> Result<Subgraph> {
        let node_ids = self.bfs(start_id, max_depth, edge_types, None)?;

        let mut nodes = HashMap::new();
        let mut edges = Vec::new();

        // Collect all nodes (keyed by display string for lookup)
        let mut node_key_set: HashSet<Vec<u8>> = HashSet::new();
        for rid in &node_ids {
            if let Some(node) = self.get_node(rid)? {
                node_key_set.insert(rid.to_key());
                nodes.insert(rid.to_string(), node);
            }
        }

        // Collect edges where both endpoints are in the subgraph
        for rid in &node_ids {
            for edge in self.get_edges(rid, None)? {
                if node_key_set.contains(&edge.to_id.to_key()) {
                    edges.push(edge);
                }
            }
        }

        Ok(Subgraph { nodes, edges })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Simple in-memory KV store for testing the graph overlay.
    #[derive(Clone)]
    struct MemKV {
        data: Arc<Mutex<std::collections::BTreeMap<Vec<u8>, Vec<u8>>>>,
    }

    impl MemKV {
        fn new() -> Self {
            Self {
                data: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
            }
        }
    }

    impl ConnectionTrait for MemKV {
        fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
            self.data.lock().unwrap().insert(key.to_vec(), value.to_vec());
            Ok(())
        }

        fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
            Ok(self.data.lock().unwrap().get(key).cloned())
        }

        fn delete(&self, key: &[u8]) -> Result<()> {
            self.data.lock().unwrap().remove(key);
            Ok(())
        }

        fn scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let data = self.data.lock().unwrap();
            let results: Vec<_> = data
                .range(prefix.to_vec()..)
                .take_while(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Ok(results)
        }
    }

    #[test]
    fn test_add_and_get_node() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let rid = RecordId::new("user", 1);
        let mut props = HashMap::new();
        props.insert("name".to_string(), SochValue::Text("Alice".to_string()));

        let node = graph.add_node(&rid, "User", Some(props)).unwrap();
        assert_eq!(node.node_type, "User");
        assert_eq!(node.id, rid);

        let fetched = graph.get_node(&rid).unwrap().unwrap();
        assert_eq!(fetched.node_type, "User");
        assert_eq!(
            fetched.properties.get("name"),
            Some(&SochValue::Text("Alice".to_string()))
        );
    }

    #[test]
    fn test_update_node() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let rid = RecordId::new("user", 1);
        graph.add_node(&rid, "User", None).unwrap();

        let mut new_props = HashMap::new();
        new_props.insert("email".to_string(), SochValue::Text("a@b.com".to_string()));

        let updated = graph
            .update_node(&rid, Some(new_props), Some("Admin"))
            .unwrap()
            .unwrap();
        assert_eq!(updated.node_type, "Admin");
        assert!(updated.properties.contains_key("email"));
    }

    #[test]
    fn test_delete_node() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let rid = RecordId::new("user", 1);
        graph.add_node(&rid, "User", None).unwrap();

        assert!(graph.node_exists(&rid).unwrap());
        assert!(graph.delete_node(&rid, false).unwrap());
        assert!(!graph.node_exists(&rid).unwrap());
    }

    #[test]
    fn test_add_and_get_edge() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let u1 = RecordId::new("user", 1);
        let c1 = RecordId::from_string("conv", "abc");
        graph.add_node(&u1, "User", None).unwrap();
        graph.add_node(&c1, "Conversation", None).unwrap();

        let edge = graph.add_edge(&u1, "STARTED", &c1, None).unwrap();
        assert_eq!(edge.edge_type, "STARTED");
        assert_eq!(edge.from_id, u1);
        assert_eq!(edge.to_id, c1);

        let fetched = graph.get_edge(&u1, "STARTED", &c1).unwrap().unwrap();
        assert_eq!(fetched.edge_type, "STARTED");
    }

    #[test]
    fn test_get_outgoing_edges() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let u1 = RecordId::new("user", 1);
        let c1 = RecordId::new("conv", 1);
        let c2 = RecordId::new("conv", 2);
        graph.add_node(&u1, "User", None).unwrap();
        graph.add_node(&c1, "Conv", None).unwrap();
        graph.add_node(&c2, "Conv", None).unwrap();

        graph.add_edge(&u1, "STARTED", &c1, None).unwrap();
        graph.add_edge(&u1, "STARTED", &c2, None).unwrap();

        let edges = graph.get_edges(&u1, None).unwrap();
        assert_eq!(edges.len(), 2);

        let filtered = graph.get_edges(&u1, Some("STARTED")).unwrap();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_delete_edge() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let u1 = RecordId::new("user", 1);
        let c1 = RecordId::new("conv", 1);
        graph.add_edge(&u1, "SENT", &c1, None).unwrap();

        assert!(graph.delete_edge(&u1, "SENT", &c1).unwrap());
        assert!(!graph.delete_edge(&u1, "SENT", &c1).unwrap());
        assert!(graph.get_edge(&u1, "SENT", &c1).unwrap().is_none());
    }

    #[test]
    fn test_cascade_delete() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let u1 = RecordId::new("user", 1);
        let c1 = RecordId::new("conv", 1);
        graph.add_node(&u1, "User", None).unwrap();
        graph.add_node(&c1, "Conv", None).unwrap();
        graph.add_edge(&u1, "OWNS", &c1, None).unwrap();

        assert!(graph.delete_node(&u1, true).unwrap());
        assert!(graph.get_edge(&u1, "OWNS", &c1).unwrap().is_none());
    }

    #[test]
    fn test_bfs_traversal() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        // Chain: A -> B -> C
        let a = RecordId::new("node", 1);
        let b = RecordId::new("node", 2);
        let c = RecordId::new("node", 3);
        graph.add_node(&a, "N", None).unwrap();
        graph.add_node(&b, "N", None).unwrap();
        graph.add_node(&c, "N", None).unwrap();
        graph.add_edge(&a, "NEXT", &b, None).unwrap();
        graph.add_edge(&b, "NEXT", &c, None).unwrap();

        let visited = graph.bfs(&a, 5, None, None).unwrap();
        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], a);
    }

    #[test]
    fn test_shortest_path() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let a = RecordId::new("n", 1);
        let b = RecordId::new("n", 2);
        let c = RecordId::new("n", 3);
        graph.add_node(&a, "N", None).unwrap();
        graph.add_node(&b, "N", None).unwrap();
        graph.add_node(&c, "N", None).unwrap();
        graph.add_edge(&a, "E", &b, None).unwrap();
        graph.add_edge(&b, "E", &c, None).unwrap();

        let path = graph.shortest_path(&a, &c, 10, None).unwrap().unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], a);
        assert_eq!(path[2], c);
    }

    #[test]
    fn test_get_neighbors() {
        let kv = MemKV::new();
        let graph = GraphOverlay::new(kv, "test");

        let a = RecordId::new("n", 1);
        let b = RecordId::new("n", 2);
        graph.add_node(&a, "N", None).unwrap();
        graph.add_node(&b, "N", None).unwrap();
        graph.add_edge(&a, "LINK", &b, None).unwrap();

        let neighbors = graph
            .get_neighbors(&a, None, EdgeDirection::Outgoing)
            .unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].node_id, b);
    }

    #[test]
    fn test_namespace_isolation() {
        let kv = MemKV::new();
        let g1 = GraphOverlay::new(kv.clone(), "ns1");
        let g2 = GraphOverlay::new(kv, "ns2");

        let rid = RecordId::new("user", 1);
        g1.add_node(&rid, "User", None).unwrap();

        assert!(g1.node_exists(&rid).unwrap());
        assert!(!g2.node_exists(&rid).unwrap());
    }
}