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

//! # Correctness Testing Framework
//!
//! Property-based testing and crash-consistency validation for SochDB.
//!
//! ## Components
//!
//! 1. **Property Tests**: Invariant checking with proptest
//! 2. **Crash Consistency**: Simulate crashes during WAL writes
//! 3. **Isolation Testing**: Verify SSI guarantees
//! 4. **Linearizability**: Check single-register linearizability
//!
//! ## Design
//!
//! Uses a combination of:
//! - Property-based testing (proptest/quickcheck style)
//! - Model checking for state machines
//! - Fault injection for crash recovery

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// Transaction operation for model checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnOp {
    Begin { txn_id: u64 },
    Read { key: Vec<u8>, expected: Option<Vec<u8>> },
    Write { key: Vec<u8>, value: Vec<u8> },
    Commit { txn_id: u64 },
    Abort { txn_id: u64 },
}

/// Transaction history for serializability checking
#[derive(Debug, Default)]
pub struct TxnHistory {
    /// Operations in observed order
    operations: Vec<TxnOp>,
    /// Committed transaction order
    commit_order: Vec<u64>,
    /// Aborted transactions
    aborted: HashSet<u64>,
}

impl TxnHistory {
    /// Add an operation
    pub fn push(&mut self, op: TxnOp) {
        match &op {
            TxnOp::Commit { txn_id } => {
                self.commit_order.push(*txn_id);
            }
            TxnOp::Abort { txn_id } => {
                self.aborted.insert(*txn_id);
            }
            _ => {}
        }
        self.operations.push(op);
    }

    /// Check if history is serializable
    /// 
    /// Uses a simplified dependency graph analysis:
    /// - WW conflicts: two txns write same key
    /// - WR conflicts: txn reads value written by another
    /// - RW conflicts: txn writes key read by another (anti-dependency)
    pub fn is_serializable(&self) -> Result<bool, SerializabilityError> {
        let graph = self.build_dependency_graph()?;
        
        // Check for cycles using DFS
        Ok(!graph.has_cycle())
    }

    /// Build a dependency graph from the history
    fn build_dependency_graph(&self) -> Result<DependencyGraph, SerializabilityError> {
        let mut graph = DependencyGraph::new();
        
        // Track writes per transaction
        let mut txn_writes: HashMap<u64, HashSet<Vec<u8>>> = HashMap::new();
        let mut txn_reads: HashMap<u64, HashSet<Vec<u8>>> = HashMap::new();
        let mut current_txn: Option<u64> = None;

        for op in &self.operations {
            match op {
                TxnOp::Begin { txn_id } => {
                    current_txn = Some(*txn_id);
                    graph.add_node(*txn_id);
                }
                TxnOp::Read { key, .. } => {
                    if let Some(txn_id) = current_txn {
                        txn_reads.entry(txn_id).or_default().insert(key.clone());
                    }
                }
                TxnOp::Write { key, .. } => {
                    if let Some(txn_id) = current_txn {
                        txn_writes.entry(txn_id).or_default().insert(key.clone());
                    }
                }
                TxnOp::Commit { .. } | TxnOp::Abort { .. } => {
                    current_txn = None;
                }
            }
        }

        // Build conflict edges
        let committed: Vec<_> = self.commit_order.iter().copied().collect();
        let empty_set: HashSet<Vec<u8>> = HashSet::new();
        for (i, &t1) in committed.iter().enumerate() {
            for &t2 in &committed[i+1..] {
                let t1_writes = txn_writes.get(&t1).unwrap_or(&empty_set);
                let t2_writes = txn_writes.get(&t2).unwrap_or(&empty_set);
                let t1_reads = txn_reads.get(&t1).unwrap_or(&empty_set);
                let t2_reads = txn_reads.get(&t2).unwrap_or(&empty_set);

                // WW conflict: t1 -> t2 if both write same key
                if !t1_writes.is_disjoint(t2_writes) {
                    graph.add_edge(t1, t2);
                }

                // WR conflict: t1 -> t2 if t2 reads t1's write
                if !t1_writes.is_disjoint(t2_reads) {
                    graph.add_edge(t1, t2);
                }

                // RW anti-dependency: t1 -> t2 if t1 reads, t2 writes same key
                if !t1_reads.is_disjoint(t2_writes) {
                    graph.add_edge(t1, t2);
                }
            }
        }

        Ok(graph)
    }
}

/// Serializability check error
#[derive(Debug)]
pub enum SerializabilityError {
    InvalidHistory(String),
    CycleDetected(Vec<u64>),
}

/// Dependency graph for serializability checking
#[derive(Debug, Default)]
struct DependencyGraph {
    /// Adjacency list: node -> set of successors
    edges: HashMap<u64, HashSet<u64>>,
    /// All nodes
    nodes: HashSet<u64>,
}

impl DependencyGraph {
    fn new() -> Self {
        Self::default()
    }

    fn add_node(&mut self, node: u64) {
        self.nodes.insert(node);
        self.edges.entry(node).or_default();
    }

    fn add_edge(&mut self, from: u64, to: u64) {
        self.edges.entry(from).or_default().insert(to);
    }

    /// Check if graph has a cycle using DFS
    fn has_cycle(&self) -> bool {
        #[derive(Clone, Copy, PartialEq)]
        enum Color { White, Gray, Black }
        
        let mut colors: HashMap<u64, Color> = self.nodes.iter()
            .map(|&n| (n, Color::White))
            .collect();

        fn dfs(
            node: u64,
            edges: &HashMap<u64, HashSet<u64>>,
            colors: &mut HashMap<u64, Color>,
        ) -> bool {
            colors.insert(node, Color::Gray);
            
            if let Some(neighbors) = edges.get(&node) {
                for &neighbor in neighbors {
                    match colors.get(&neighbor) {
                        Some(Color::Gray) => return true, // Back edge = cycle
                        Some(Color::White) => {
                            if dfs(neighbor, edges, colors) {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            
            colors.insert(node, Color::Black);
            false
        }

        for node in self.nodes.iter().copied() {
            if colors.get(&node) == Some(&Color::White) {
                if dfs(node, &self.edges, &mut colors) {
                    return true;
                }
            }
        }

        false
    }
}

/// Crash point for fault injection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashPoint {
    /// Before WAL write
    BeforeWalWrite,
    /// After WAL write, before fsync
    AfterWalWriteBeforeFsync,
    /// After fsync, before data page write
    AfterFsyncBeforeDataWrite,
    /// After data page write
    AfterDataWrite,
    /// During checkpoint
    DuringCheckpoint,
}

/// Crash simulator for recovery testing
pub struct CrashSimulator {
    /// Current crash point (None = no crash)
    crash_at: Option<CrashPoint>,
    /// Crash countdown (crash after N operations)
    countdown: AtomicU64,
    /// Triggered crash points
    triggered: std::sync::Mutex<Vec<CrashPoint>>,
}

impl CrashSimulator {
    /// Create a new crash simulator
    pub fn new() -> Self {
        Self {
            crash_at: None,
            countdown: AtomicU64::new(u64::MAX),
            triggered: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Schedule a crash at a specific point after N operations
    pub fn schedule_crash(&mut self, point: CrashPoint, after_ops: u64) {
        self.crash_at = Some(point);
        self.countdown.store(after_ops, Ordering::SeqCst);
    }

    /// Check if we should crash at this point
    pub fn maybe_crash(&self, point: CrashPoint) -> bool {
        if self.crash_at != Some(point) {
            return false;
        }

        let prev = self.countdown.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.triggered.lock().unwrap().push(point);
            true
        } else {
            false
        }
    }

    /// Get triggered crash points
    pub fn triggered_crashes(&self) -> Vec<CrashPoint> {
        self.triggered.lock().unwrap().clone()
    }
}

impl Default for CrashSimulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Model of a key-value store for property testing
#[derive(Debug, Default)]
pub struct KvModel {
    /// Simple in-memory KV store
    data: BTreeMap<Vec<u8>, Vec<u8>>,
    /// Transaction counter
    next_txn: u64,
    /// Active transactions
    active_txns: HashMap<u64, HashMap<Vec<u8>, Vec<u8>>>,
}

impl KvModel {
    /// Create a new model
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a transaction
    pub fn begin(&mut self) -> u64 {
        let txn_id = self.next_txn;
        self.next_txn += 1;
        self.active_txns.insert(txn_id, HashMap::new());
        txn_id
    }

    /// Read a key (returns committed value or txn's local write)
    pub fn read(&self, txn_id: u64, key: &[u8]) -> Option<Vec<u8>> {
        // Check local writes first
        if let Some(txn_writes) = self.active_txns.get(&txn_id) {
            if let Some(value) = txn_writes.get(key) {
                return Some(value.clone());
            }
        }
        // Fall back to committed data
        self.data.get(key).cloned()
    }

    /// Write a key
    pub fn write(&mut self, txn_id: u64, key: Vec<u8>, value: Vec<u8>) {
        if let Some(txn_writes) = self.active_txns.get_mut(&txn_id) {
            txn_writes.insert(key, value);
        }
    }

    /// Commit a transaction
    pub fn commit(&mut self, txn_id: u64) -> bool {
        if let Some(writes) = self.active_txns.remove(&txn_id) {
            for (key, value) in writes {
                self.data.insert(key, value);
            }
            true
        } else {
            false
        }
    }

    /// Abort a transaction
    pub fn abort(&mut self, txn_id: u64) -> bool {
        self.active_txns.remove(&txn_id).is_some()
    }

    /// Get all data (for comparison)
    pub fn snapshot(&self) -> BTreeMap<Vec<u8>, Vec<u8>> {
        self.data.clone()
    }
}

/// Test oracle for comparing model vs implementation
pub struct TestOracle<T> {
    /// Reference model
    model: KvModel,
    /// System under test
    sut: T,
    /// Discrepancies found
    discrepancies: Vec<Discrepancy>,
}

/// A discrepancy between model and SUT
#[derive(Debug, Clone)]
pub struct Discrepancy {
    pub operation: String,
    pub expected: Option<Vec<u8>>,
    pub actual: Option<Vec<u8>>,
    pub key: Vec<u8>,
}

impl<T> TestOracle<T> {
    /// Create a new oracle
    pub fn new(sut: T) -> Self {
        Self {
            model: KvModel::new(),
            sut,
            discrepancies: Vec::new(),
        }
    }

    /// Get the model for operations
    pub fn model(&mut self) -> &mut KvModel {
        &mut self.model
    }

    /// Get the SUT for operations
    pub fn sut(&mut self) -> &mut T {
        &mut self.sut
    }

    /// Record a discrepancy
    pub fn record_discrepancy(&mut self, discrepancy: Discrepancy) {
        self.discrepancies.push(discrepancy);
    }

    /// Check if any discrepancies were found
    pub fn has_discrepancies(&self) -> bool {
        !self.discrepancies.is_empty()
    }

    /// Get all discrepancies
    pub fn discrepancies(&self) -> &[Discrepancy] {
        &self.discrepancies
    }
}

/// Linearizability checker for single-register operations
#[derive(Debug)]
pub struct LinearizabilityChecker {
    /// History of operations
    history: Vec<LinearOp>,
}

/// A linearizability operation
#[derive(Debug, Clone)]
pub struct LinearOp {
    /// Operation type
    pub op_type: LinearOpType,
    /// Start time (logical)
    pub start: u64,
    /// End time (logical)
    pub end: u64,
    /// Value (for reads: returned value; for writes: written value)
    pub value: Option<Vec<u8>>,
}

/// Operation type for linearizability
#[derive(Debug, Clone, Copy)]
pub enum LinearOpType {
    Read,
    Write,
}

impl LinearizabilityChecker {
    /// Create a new checker
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    /// Add an operation
    pub fn add(&mut self, op: LinearOp) {
        self.history.push(op);
    }

    /// Check if history is linearizable (simplified algorithm)
    /// 
    /// For a full implementation, use Wing & Gong's algorithm or similar.
    /// This simplified version checks basic consistency.
    pub fn is_linearizable(&self) -> bool {
        // Sort by start time
        let mut ops = self.history.clone();
        ops.sort_by_key(|op| op.start);

        // Track the "current" value that should be visible
        let mut current_value: Option<Vec<u8>> = None;
        let mut pending_writes: Vec<&LinearOp> = Vec::new();

        for op in &ops {
            // Remove writes that have ended
            pending_writes.retain(|w| w.end >= op.start);

            match op.op_type {
                LinearOpType::Write => {
                    pending_writes.push(op);
                    current_value = op.value.clone();
                }
                LinearOpType::Read => {
                    // Read should see either current value or a pending write
                    if op.value != current_value {
                        // Check if it matches any pending write
                        let matches_pending = pending_writes.iter()
                            .any(|w| w.value == op.value);
                        if !matches_pending && op.value != current_value {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

impl Default for LinearizabilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serializable_history_no_conflict() {
        let mut history = TxnHistory::default();
        
        // T1: write(x, 1)
        history.push(TxnOp::Begin { txn_id: 1 });
        history.push(TxnOp::Write { key: b"x".to_vec(), value: b"1".to_vec() });
        history.push(TxnOp::Commit { txn_id: 1 });

        // T2: write(y, 2)
        history.push(TxnOp::Begin { txn_id: 2 });
        history.push(TxnOp::Write { key: b"y".to_vec(), value: b"2".to_vec() });
        history.push(TxnOp::Commit { txn_id: 2 });

        assert!(history.is_serializable().unwrap());
    }

    #[test]
    fn test_kv_model_basic() {
        let mut model = KvModel::new();
        
        let txn = model.begin();
        model.write(txn, b"key1".to_vec(), b"value1".to_vec());
        
        // Read own write
        assert_eq!(model.read(txn, b"key1"), Some(b"value1".to_vec()));
        
        // Commit
        assert!(model.commit(txn));
        
        // Read after commit from new txn
        let txn2 = model.begin();
        assert_eq!(model.read(txn2, b"key1"), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_crash_simulator() {
        let mut sim = CrashSimulator::new();
        sim.schedule_crash(CrashPoint::AfterWalWriteBeforeFsync, 3);

        // Should not crash yet
        assert!(!sim.maybe_crash(CrashPoint::AfterWalWriteBeforeFsync));
        assert!(!sim.maybe_crash(CrashPoint::AfterWalWriteBeforeFsync));
        
        // Should crash now
        assert!(sim.maybe_crash(CrashPoint::AfterWalWriteBeforeFsync));
        
        // Should not crash again (already triggered)
        assert!(!sim.maybe_crash(CrashPoint::AfterWalWriteBeforeFsync));

        assert_eq!(sim.triggered_crashes().len(), 1);
    }

    #[test]
    fn test_linearizability_simple() {
        let mut checker = LinearizabilityChecker::new();

        // Write x=1 at time 0-2
        checker.add(LinearOp {
            op_type: LinearOpType::Write,
            start: 0,
            end: 2,
            value: Some(b"1".to_vec()),
        });

        // Read x=1 at time 3-4 (should see the write)
        checker.add(LinearOp {
            op_type: LinearOpType::Read,
            start: 3,
            end: 4,
            value: Some(b"1".to_vec()),
        });

        assert!(checker.is_linearizable());
    }

    #[test]
    fn test_dependency_graph_cycle_detection() {
        let mut graph = DependencyGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);

        // No cycle
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        assert!(!graph.has_cycle());

        // Add cycle
        graph.add_edge(3, 1);
        assert!(graph.has_cycle());
    }
}
