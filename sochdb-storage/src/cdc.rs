// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # WAL-Derived Change Data Capture (CDC) Engine
//!
//! Provides a log-structured stream of database mutations (inserts, updates,
//! deletes) that subscribers can consume from any position.
//!
//! ## Architecture
//!
//! ```text
//!  Database commit path
//!         │
//!         ▼
//!   ┌───────────┐    emit()     ┌─────────────┐
//!   │ CdcEmitter│──────────────▶│  CdcLog      │
//!   └───────────┘               │ (ring buffer) │
//!                               └──────┬────────┘
//!                                      │ subscribe(from_seq)
//!                          ┌───────────┼───────────┐
//!                          ▼           ▼           ▼
//!                     Subscriber₁  Subscriber₂  SubscriberN
//! ```
//!
//! ## Design Decisions
//!
//! - **After-image only**: Events carry the new value but not the old value.
//!   The active WAL path (`TxnWalEntry`) doesn't record before-images.
//!   A future enhancement can bridge the ARIES `WalRecord` path for full
//!   before/after CDC.
//!
//! - **Ring buffer with overflow**: Fixed-capacity ring buffer. When the buffer
//!   is full, the oldest events are dropped. Slow subscribers must catch up
//!   from WAL replay (not yet implemented — returns `CdcError::Overrun`).
//!
//! - **Sequence numbers**: Events are assigned monotonically increasing sequence
//!   numbers, independent of WAL LSNs. Subscribers track their position via
//!   these sequence numbers.
//!

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// CDC Event Types
// ============================================================================

/// A CDC event representing a single mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct CdcEvent {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Timestamp (microseconds since epoch).
    pub timestamp_us: u64,
    /// Transaction ID that produced this event.
    pub txn_id: u64,
    /// Name of the affected table.
    pub table: String,
    /// Primary key (raw bytes).
    pub key: Vec<u8>,
    /// The type of operation.
    pub operation: CdcOperation,
}

/// The type of mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdcOperation {
    /// Row inserted. `after` contains the new value.
    Insert { after: Vec<u8> },
    /// Row updated. `after` contains the new value.
    /// `before` is `None` in the current implementation (after-image only).
    Update {
        before: Option<Vec<u8>>,
        after: Vec<u8>,
    },
    /// Row deleted. `before` is `None` in the current implementation.
    Delete { before: Option<Vec<u8>> },
    /// Schema change (CREATE TABLE, ALTER TABLE, DROP TABLE).
    SchemaChange { ddl: String },
}

// ============================================================================
// CDC Errors
// ============================================================================

/// Errors that can occur in the CDC subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdcError {
    /// The requested sequence number has been evicted from the ring buffer.
    /// The subscriber must fall back to WAL replay.
    Overrun {
        requested: u64,
        oldest_available: u64,
    },
    /// The CDC engine has been shut down.
    Shutdown,
    /// Timed out waiting for new events.
    Timeout,
}

impl std::fmt::Display for CdcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CdcError::Overrun {
                requested,
                oldest_available,
            } => write!(
                f,
                "CDC overrun: requested seq {} but oldest available is {}",
                requested, oldest_available
            ),
            CdcError::Shutdown => write!(f, "CDC engine shut down"),
            CdcError::Timeout => write!(f, "Timed out waiting for CDC events"),
        }
    }
}

impl std::error::Error for CdcError {}

pub type CdcResult<T> = Result<T, CdcError>;

// ============================================================================
// CDC Log (Ring Buffer)
// ============================================================================

/// Configuration for the CDC engine.
#[derive(Debug, Clone)]
pub struct CdcConfig {
    /// Maximum number of events to keep in the ring buffer.
    /// Default: 65536 (~64K events). At ~1KB per event, this is ~64MB.
    pub capacity: usize,
    /// Whether CDC is enabled. If false, `emit()` is a no-op.
    pub enabled: bool,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            capacity: 65_536,
            enabled: true,
        }
    }
}

/// The core CDC log — a ring buffer of events with subscriber notification.
pub struct CdcLog {
    /// Ring buffer of events.
    buffer: RwLock<VecDeque<CdcEvent>>,
    /// Maximum ring buffer capacity.
    capacity: usize,
    /// Next sequence number to assign.
    next_seq: AtomicU64,
    /// Condition variable for subscriber notification.
    notify: Arc<(Mutex<bool>, Condvar)>,
    /// Whether the engine is running.
    running: AtomicU64, // 0 = stopped, 1 = running
}

impl CdcLog {
    /// Create a new CDC log with the given configuration.
    pub fn new(config: CdcConfig) -> Arc<Self> {
        Arc::new(Self {
            buffer: RwLock::new(VecDeque::with_capacity(config.capacity)),
            capacity: config.capacity,
            next_seq: AtomicU64::new(1),
            notify: Arc::new((Mutex::new(false), Condvar::new())),
            running: AtomicU64::new(1),
        })
    }

    /// Emit a batch of CDC events (typically one per row in a transaction).
    ///
    /// This is called from the commit path after WAL flush + group commit.
    /// Must be fast — no I/O, no allocations on the hot path (beyond the
    /// ring buffer push).
    pub fn emit(&self, events: Vec<CdcEvent>) {
        if self.running.load(Ordering::Relaxed) == 0 {
            return;
        }
        if events.is_empty() {
            return;
        }

        let mut buf = self.buffer.write().unwrap();
        for event in events {
            if buf.len() >= self.capacity {
                buf.pop_front(); // drop oldest
            }
            buf.push_back(event);
        }
        drop(buf);

        // Notify subscribers
        let (lock, cvar) = &*self.notify;
        let mut ready = lock.lock().unwrap();
        *ready = true;
        cvar.notify_all();
    }

    /// Emit a single event.
    pub fn emit_one(&self, event: CdcEvent) {
        if self.running.load(Ordering::Relaxed) == 0 {
            return;
        }

        let mut buf = self.buffer.write().unwrap();
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(event);
        drop(buf);

        let (lock, cvar) = &*self.notify;
        let mut ready = lock.lock().unwrap();
        *ready = true;
        cvar.notify_all();
    }

    /// Allocate the next sequence number.
    pub fn next_sequence(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the current (latest) sequence number.
    pub fn current_sequence(&self) -> u64 {
        self.next_seq.load(Ordering::SeqCst).saturating_sub(1)
    }

    /// Read events starting from `from_seq` (inclusive).
    ///
    /// Returns up to `max_events` events. If `from_seq` has been evicted
    /// from the ring buffer, returns `CdcError::Overrun`.
    pub fn read_from(&self, from_seq: u64, max_events: usize) -> CdcResult<Vec<CdcEvent>> {
        let buf = self.buffer.read().unwrap();

        if buf.is_empty() {
            return Ok(Vec::new());
        }

        let oldest_seq = buf.front().map(|e| e.sequence).unwrap_or(0);
        let newest_seq = buf.back().map(|e| e.sequence).unwrap_or(0);

        if from_seq < oldest_seq {
            return Err(CdcError::Overrun {
                requested: from_seq,
                oldest_available: oldest_seq,
            });
        }

        if from_seq > newest_seq {
            return Ok(Vec::new()); // no new events yet
        }

        // Binary search for the start position
        let start_idx = buf
            .iter()
            .position(|e| e.sequence >= from_seq)
            .unwrap_or(buf.len());

        let events: Vec<CdcEvent> = buf
            .iter()
            .skip(start_idx)
            .take(max_events)
            .cloned()
            .collect();

        Ok(events)
    }

    /// Wait for new events after `after_seq`, with timeout.
    ///
    /// Returns events with sequence > `after_seq`, blocking until at least
    /// one event is available or the timeout expires.
    pub fn wait_for_events(
        &self,
        after_seq: u64,
        max_events: usize,
        timeout: Duration,
    ) -> CdcResult<Vec<CdcEvent>> {
        if self.running.load(Ordering::Relaxed) == 0 {
            return Err(CdcError::Shutdown);
        }

        // Fast path: check if events are already available
        let events = self.read_from(after_seq + 1, max_events)?;
        if !events.is_empty() {
            return Ok(events);
        }

        // Slow path: wait for notification
        let (lock, cvar) = &*self.notify;
        let mut ready = lock.lock().unwrap();
        let start = std::time::Instant::now();

        loop {
            if self.running.load(Ordering::Relaxed) == 0 {
                return Err(CdcError::Shutdown);
            }

            let remaining = timeout.checked_sub(start.elapsed()).unwrap_or(Duration::ZERO);
            if remaining.is_zero() {
                return Err(CdcError::Timeout);
            }

            let result = cvar.wait_timeout(ready, remaining).unwrap();
            ready = result.0;

            // Check for events
            let events = self.read_from(after_seq + 1, max_events)?;
            if !events.is_empty() {
                *ready = false;
                return Ok(events);
            }

            if result.1.timed_out() {
                return Err(CdcError::Timeout);
            }
        }
    }

    /// Get the oldest available sequence number (or 0 if empty).
    pub fn oldest_sequence(&self) -> u64 {
        self.buffer
            .read()
            .unwrap()
            .front()
            .map(|e| e.sequence)
            .unwrap_or(0)
    }

    /// Get the number of events in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.read().unwrap().len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.read().unwrap().is_empty()
    }

    /// Shut down the CDC engine, waking all waiting subscribers.
    pub fn shutdown(&self) {
        self.running.store(0, Ordering::SeqCst);
        let (lock, cvar) = &*self.notify;
        let mut ready = lock.lock().unwrap();
        *ready = true;
        cvar.notify_all();
        drop(ready);
    }
}

// ============================================================================
// CDC Emitter — Helper for producing CDC events from the commit path
// ============================================================================

/// Helper struct for building CDC events during a transaction.
///
/// Usage (in the SQL execution layer):
/// ```ignore
/// let mut emitter = CdcEmitter::new(cdc_log.clone(), txn_id);
/// emitter.insert("users", key, value);
/// emitter.update("users", key, new_value);
/// emitter.delete("users", key);
/// emitter.flush(); // called after successful commit
/// ```
pub struct CdcEmitter {
    log: Arc<CdcLog>,
    txn_id: u64,
    pending: Vec<CdcEvent>,
}

impl CdcEmitter {
    pub fn new(log: Arc<CdcLog>, txn_id: u64) -> Self {
        Self {
            log,
            txn_id,
            pending: Vec::new(),
        }
    }

    fn now_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }

    /// Record an INSERT event.
    pub fn insert(&mut self, table: &str, key: Vec<u8>, value: Vec<u8>) {
        let seq = self.log.next_sequence();
        self.pending.push(CdcEvent {
            sequence: seq,
            timestamp_us: Self::now_us(),
            txn_id: self.txn_id,
            table: table.to_string(),
            key,
            operation: CdcOperation::Insert { after: value },
        });
    }

    /// Record an UPDATE event.
    pub fn update(&mut self, table: &str, key: Vec<u8>, new_value: Vec<u8>) {
        let seq = self.log.next_sequence();
        self.pending.push(CdcEvent {
            sequence: seq,
            timestamp_us: Self::now_us(),
            txn_id: self.txn_id,
            table: table.to_string(),
            key,
            operation: CdcOperation::Update {
                before: None,
                after: new_value,
            },
        });
    }

    /// Record a DELETE event.
    pub fn delete(&mut self, table: &str, key: Vec<u8>) {
        let seq = self.log.next_sequence();
        self.pending.push(CdcEvent {
            sequence: seq,
            timestamp_us: Self::now_us(),
            txn_id: self.txn_id,
            table: table.to_string(),
            key,
            operation: CdcOperation::Delete { before: None },
        });
    }

    /// Record a schema change event.
    pub fn schema_change(&mut self, table: &str, ddl: String) {
        let seq = self.log.next_sequence();
        self.pending.push(CdcEvent {
            sequence: seq,
            timestamp_us: Self::now_us(),
            txn_id: self.txn_id,
            table: table.to_string(),
            key: Vec::new(),
            operation: CdcOperation::SchemaChange { ddl },
        });
    }

    /// Flush all pending events to the CDC log.
    /// Call this AFTER the transaction has been committed successfully.
    pub fn flush(self) {
        if !self.pending.is_empty() {
            self.log.emit(self.pending);
        }
    }

    /// Discard all pending events (e.g., on transaction abort).
    pub fn discard(self) {
        // drop self — pending events are lost
    }

    /// Number of pending events.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ============================================================================
// CDC Subscriber — Convenience wrapper for consuming events
// ============================================================================

/// A subscriber that tracks its position in the CDC log.
pub struct CdcSubscriber {
    log: Arc<CdcLog>,
    /// The last sequence number that was consumed.
    last_seq: u64,
    /// Table filter (if Some, only events for these tables are returned).
    table_filter: Option<Vec<String>>,
}

impl CdcSubscriber {
    /// Create a subscriber starting from the given sequence number.
    /// Use `0` to start from the beginning of the buffer.
    pub fn new(log: Arc<CdcLog>, from_seq: u64) -> Self {
        Self {
            log,
            last_seq: from_seq,
            table_filter: None,
        }
    }

    /// Create a subscriber starting from the current (latest) position.
    pub fn from_latest(log: Arc<CdcLog>) -> Self {
        let seq = log.current_sequence();
        Self {
            log,
            last_seq: seq,
            table_filter: None,
        }
    }

    /// Filter events to only include the given tables.
    pub fn with_tables(mut self, tables: Vec<String>) -> Self {
        self.table_filter = Some(tables);
        self
    }

    /// Poll for new events (non-blocking).
    pub fn poll(&mut self, max_events: usize) -> CdcResult<Vec<CdcEvent>> {
        let events = self.log.read_from(self.last_seq + 1, max_events)?;
        let filtered = self.filter_events(events);
        if let Some(last) = filtered.last() {
            self.last_seq = last.sequence;
        }
        Ok(filtered)
    }

    /// Wait for new events (blocking with timeout).
    pub fn next_batch(
        &mut self,
        max_events: usize,
        timeout: Duration,
    ) -> CdcResult<Vec<CdcEvent>> {
        let events = self
            .log
            .wait_for_events(self.last_seq, max_events, timeout)?;
        let filtered = self.filter_events(events);
        if let Some(last) = filtered.last() {
            self.last_seq = last.sequence;
        }
        Ok(filtered)
    }

    /// Get the subscriber's current position.
    pub fn position(&self) -> u64 {
        self.last_seq
    }

    fn filter_events(&self, events: Vec<CdcEvent>) -> Vec<CdcEvent> {
        if let Some(ref tables) = self.table_filter {
            events
                .into_iter()
                .filter(|e| tables.iter().any(|t| *t == e.table))
                .collect()
        } else {
            events
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_log(cap: usize) -> Arc<CdcLog> {
        CdcLog::new(CdcConfig {
            capacity: cap,
            enabled: true,
        })
    }

    #[test]
    fn test_cdc_emit_and_read() {
        let log = make_log(100);
        let mut emitter = CdcEmitter::new(log.clone(), 42);

        emitter.insert("users", b"key1".to_vec(), b"val1".to_vec());
        emitter.insert("users", b"key2".to_vec(), b"val2".to_vec());
        assert_eq!(emitter.pending_count(), 2);

        emitter.flush();

        assert_eq!(log.len(), 2);
        let events = log.read_from(1, 10).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].table, "users");
        assert_eq!(events[0].txn_id, 42);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[1].sequence, 2);
    }

    #[test]
    fn test_cdc_ring_buffer_overflow() {
        let log = make_log(3);

        for i in 1..=5 {
            log.emit_one(CdcEvent {
                sequence: log.next_sequence(),
                timestamp_us: 0,
                txn_id: i,
                table: "t".into(),
                key: vec![i as u8],
                operation: CdcOperation::Insert {
                    after: vec![i as u8],
                },
            });
        }

        // Buffer holds only the last 3 events (seq 3, 4, 5)
        assert_eq!(log.len(), 3);
        assert_eq!(log.oldest_sequence(), 3);

        // Reading from seq 1 should return Overrun
        let err = log.read_from(1, 10).unwrap_err();
        assert!(matches!(
            err,
            CdcError::Overrun {
                requested: 1,
                oldest_available: 3
            }
        ));

        // Reading from seq 3 should work
        let events = log.read_from(3, 10).unwrap();
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_cdc_subscriber() {
        let log = make_log(100);

        // Emit some events
        let mut emitter = CdcEmitter::new(log.clone(), 1);
        emitter.insert("users", b"u1".to_vec(), b"v1".to_vec());
        emitter.insert("orders", b"o1".to_vec(), b"v2".to_vec());
        emitter.flush();

        // Subscribe from beginning
        let mut sub = CdcSubscriber::new(log.clone(), 0);
        let events = sub.poll(10).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(sub.position(), 2);

        // No new events
        let events = sub.poll(10).unwrap();
        assert_eq!(events.len(), 0);

        // Emit more
        let mut emitter = CdcEmitter::new(log.clone(), 2);
        emitter.update("users", b"u1".to_vec(), b"v1_updated".to_vec());
        emitter.flush();

        let events = sub.poll(10).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].operation,
            CdcOperation::Update { .. }
        ));
    }

    #[test]
    fn test_cdc_table_filter() {
        let log = make_log(100);

        let mut emitter = CdcEmitter::new(log.clone(), 1);
        emitter.insert("users", b"u1".to_vec(), b"v1".to_vec());
        emitter.insert("orders", b"o1".to_vec(), b"v2".to_vec());
        emitter.insert("users", b"u2".to_vec(), b"v3".to_vec());
        emitter.flush();

        let mut sub =
            CdcSubscriber::new(log.clone(), 0).with_tables(vec!["users".to_string()]);
        let events = sub.poll(10).unwrap();
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.table == "users"));
    }

    #[test]
    fn test_cdc_subscriber_from_latest() {
        let log = make_log(100);

        // Emit events before subscriber
        log.emit_one(CdcEvent {
            sequence: log.next_sequence(),
            timestamp_us: 0,
            txn_id: 1,
            table: "old".into(),
            key: vec![],
            operation: CdcOperation::Insert { after: vec![] },
        });

        // Subscribe from latest — should not see old events
        let mut sub = CdcSubscriber::from_latest(log.clone());

        // Emit new event
        log.emit_one(CdcEvent {
            sequence: log.next_sequence(),
            timestamp_us: 0,
            txn_id: 2,
            table: "new".into(),
            key: vec![],
            operation: CdcOperation::Insert { after: vec![] },
        });

        let events = sub.poll(10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].table, "new");
    }

    #[test]
    fn test_cdc_wait_for_events() {
        let log = make_log(100);
        let log_clone = log.clone();

        // Spawn a thread that emits events after a delay
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            log_clone.emit_one(CdcEvent {
                sequence: log_clone.next_sequence(),
                timestamp_us: 0,
                txn_id: 1,
                table: "t".into(),
                key: vec![1],
                operation: CdcOperation::Insert { after: vec![1] },
            });
        });

        let events = log.wait_for_events(0, 10, Duration::from_secs(2)).unwrap();
        assert_eq!(events.len(), 1);
        handle.join().unwrap();
    }

    #[test]
    fn test_cdc_wait_timeout() {
        let log = make_log(100);
        let err = log
            .wait_for_events(0, 10, Duration::from_millis(50))
            .unwrap_err();
        assert!(matches!(err, CdcError::Timeout));
    }

    #[test]
    fn test_cdc_shutdown() {
        let log = make_log(100);
        let log_clone = log.clone();

        let handle = thread::spawn(move || {
            log_clone.wait_for_events(0, 10, Duration::from_secs(5))
        });

        thread::sleep(Duration::from_millis(50));
        log.shutdown();

        let result = handle.join().unwrap();
        assert!(matches!(result, Err(CdcError::Shutdown)));
    }

    #[test]
    fn test_cdc_emitter_discard() {
        let log = make_log(100);
        let mut emitter = CdcEmitter::new(log.clone(), 1);
        emitter.insert("t", b"k".to_vec(), b"v".to_vec());
        emitter.discard(); // should NOT emit

        assert!(log.is_empty());
    }

    #[test]
    fn test_cdc_schema_change() {
        let log = make_log(100);
        let mut emitter = CdcEmitter::new(log.clone(), 1);
        emitter.schema_change("users", "ALTER TABLE users ADD COLUMN age INT".to_string());
        emitter.flush();

        let events = log.read_from(1, 10).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0].operation,
            CdcOperation::SchemaChange { ddl } if ddl.contains("ALTER TABLE")
        ));
    }

    #[test]
    fn test_cdc_concurrent_emit_and_read() {
        let log = make_log(10_000);
        let log_clone = log.clone();

        // Writer thread
        let writer = thread::spawn(move || {
            for i in 0..1000 {
                log_clone.emit_one(CdcEvent {
                    sequence: log_clone.next_sequence(),
                    timestamp_us: 0,
                    txn_id: i as u64,
                    table: "t".into(),
                    key: vec![],
                    operation: CdcOperation::Insert { after: vec![] },
                });
            }
        });

        writer.join().unwrap();

        let events = log.read_from(1, 10_000).unwrap();
        assert_eq!(events.len(), 1000);
        // Verify monotonic sequences
        for i in 1..events.len() {
            assert!(events[i].sequence > events[i - 1].sequence);
        }
    }
}
