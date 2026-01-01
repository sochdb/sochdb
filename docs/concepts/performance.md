# ToonDB Performance Optimizations

## Summary

This document details the performance improvements made to ToonDB's storage layer, achieving **~15× improvement** from ~55k ops/sec to ~800k-1.3M ops/sec for sequential inserts, making it competitive with SQLite (~1.16M ops/sec).

## Benchmark Results

| Benchmark | Before | After | Improvement |
|-----------|--------|-------|-------------|
| ToonDB Embedded (WAL) | ~55k ops/sec | ~761k ops/sec | **13.8×** |
| ToonDB put_raw | N/A | ~1.30M ops/sec | Direct storage |
| ToonDB insert_row_slice | N/A | ~1.29M ops/sec | Zero-alloc API |
| SQLite File (WAL) | ~1.16M ops/sec | — | Baseline |

---

## Root Cause Analysis

### Discovery: 10× Overhead in Database Layer

Initial profiling revealed that raw `DurableStorage::put()` achieved **1.5M ops/sec**, but the `Database` layer only achieved **~150k ops/sec**. This pointed to overhead in the transaction/database abstraction layer rather than the storage engine itself.

### Key Bottleneck: Group Commit

The group commit mechanism, designed to batch multiple transactions for better fsync amortization, was **counterproductive for sequential single-threaded inserts**:

```
Group Commit Flow (BEFORE):
┌─────────────────────────────────────────────────┐
│  1. Acquire mutex                               │
│  2. Add to pending batch                        │
│  3. Wait on condvar for batch fill OR timeout   │  ← BLOCKING!
│  4. Leader does fsync                           │
│  5. Wake all waiters                            │
└─────────────────────────────────────────────────┘
```

For sequential inserts, each operation was waiting for either:
- Other transactions to join the batch (none coming)
- A timeout (adding latency)

**Solution**: Disabled group commit for the embedded connection benchmark, allowing direct WAL writes without the coordination overhead.

### Architecture Context

ToonDB has a properly layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  Client Layer                                               │
│  ├── Connection (alias for DurableConnection) ← DEFAULT     │
│  ├── InMemoryConnection (ToonConnection) ← Testing only     │
│  └── EmbeddedConnection (optional, wraps Database)          │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer                                              │
│  ├── Database (tables, schemas, MVCC coordination)          │
│  ├── DurableStorage (WAL, transactions, sync modes)         │
│  └── TxnWal (transaction-aware WAL with CRC32)              │
└─────────────────────────────────────────────────────────────┘
```

The `DurableConnection` properly routes through the WAL/MVCC kernel, making all optimizations in the storage layer visible to the client API.

---

## Optimizations Applied

### 1. Single-Pass CRC32 Calculation

**File**: `toondb-storage/src/txn_wal.rs`

**Before**: CRC was calculated in a separate pass over the data.

**After**: CRC32 is calculated incrementally as data is written to the buffer.

```rust
pub fn write_no_flush_refs(&self, txn_id: TxnId, key: &[u8], value: &[u8]) -> Result<()> {
    let mut writer = self.writer.lock();
    
    // Single-pass: write and compute CRC simultaneously
    let mut crc = 0u32;
    
    // Write and accumulate CRC for each field
    writer.write_all(&txn_id.to_le_bytes())?;
    crc = crc32c::crc32c_append(crc, &txn_id.to_le_bytes());
    
    // ... continues for all fields
}
```

**Impact**: Eliminated redundant memory traversal, ~5-10% improvement.

---

### 2. Zero-Allocation WAL Writes

**File**: `toondb-storage/src/txn_wal.rs`

**Before**: Each WAL write allocated a `Vec<u8>` buffer.

**After**: New `write_no_flush_refs()` method writes directly to BufWriter using references.

```rust
// OLD: Allocates Vec for each write
fn write(&self, key: Vec<u8>, value: Vec<u8>)

// NEW: Zero allocation, uses references
fn write_no_flush_refs(&self, txn_id: TxnId, key: &[u8], value: &[u8])
```

**Impact**: Eliminated per-write heap allocations.

---

### 3. DashMap for Lock-Free Concurrent Access

**Files**: 
- `toondb-storage/src/durable_storage.rs`
- `toondb-storage/src/database.rs`

**Before**: `RwLock<HashMap>` for MVCC tracking and table metadata.

**After**: `DashMap` for lock-free concurrent reads.

```rust
// OLD: Lock contention on every access
tables: RwLock<HashMap<String, TableMetadata>>
active_txns: RwLock<HashMap<TxnId, TxnState>>

// NEW: Lock-free concurrent access
tables: DashMap<String, TableMetadata>
active_txns: DashMap<TxnId, TxnState>
```

**Impact**: Reduced lock contention, especially beneficial for concurrent workloads.

---

### 4. Zero-Allocation Row Insertion API

**Files**:
- `toondb-storage/src/packed_row.rs`
- `toondb-storage/src/database.rs`

**Before**: `insert_row()` required a `HashMap<String, ToonValue>` allocation per row.

**After**: New `insert_row_slice()` accepts `&[(&str, ToonValue)]` directly.

```rust
// OLD: Allocates HashMap per row
pub fn insert_row(&self, txn: TxnHandle, table: &str, 
                  row: HashMap<String, ToonValue>) -> Result<u64>

// NEW: Zero allocation, uses slice
pub fn insert_row_slice(&self, txn: TxnHandle, table: &str,
                        row: &[(&str, ToonValue)]) -> Result<u64>
```

**PackedRow Enhancement**:
```rust
// NEW: Pack from slice of references
pub fn pack_slice(values: &[Option<&ToonValue>]) -> Vec<u8>

// NEW: Unpack to pre-sized Vec
pub fn unpack_to_vec(&self) -> Vec<Option<ToonValue>>
```

**Impact**: Achieved 1.29M ops/sec, matching raw storage performance.

---

### 5. Cached Schema Lookup

**File**: `toondb-storage/src/database.rs`

**Before**: Schema was fetched and parsed on every insert.

**After**: Schema is cached in `DashMap<String, Vec<String>>` after first access.

```rust
packed_schemas: DashMap<String, Vec<String>>  // table_name -> column_names
```

**Impact**: Eliminated repeated schema parsing overhead.

---

### 6. Lazy Query Iterator

**File**: `toondb-storage/src/database.rs`

**Before**: `execute()` collected all results into a `Vec` before returning.

**After**: `execute_iter()` returns a lazy iterator that fetches rows on demand.

```rust
pub fn execute_iter(self) -> QueryRowIterator<'a> {
    QueryRowIterator {
        inner: self.storage.scan_prefix(&self.path_prefix).into_iter(),
        // ...
    }
}
```

**Impact**: Reduced memory usage for large result sets, enables streaming.

---

### 7. Group Commit Bypass for Sequential Workloads

**File**: `toondb-storage/src/durable_storage.rs`

For benchmarks and sequential workloads, group commit can be disabled:

```rust
let config = DurableStorageConfig {
    group_commit: false,  // Disable for sequential inserts
    sync_mode: 1,         // NORMAL sync every 100 commits
    ..Default::default()
};
```

**Impact**: Eliminated condvar wait overhead, ~5× improvement for sequential inserts.

---

## Architecture Insight

```
Performance Stack (ops/sec):
┌─────────────────────────────────────────┐
│  insert_row (HashMap)     ~150k         │  ← HashMap allocation overhead
├─────────────────────────────────────────┤
│  insert_row_slice         ~1.29M        │  ← Zero-allocation path
├─────────────────────────────────────────┤
│  put_raw                  ~1.30M        │  ← Direct storage bypass
├─────────────────────────────────────────┤
│  DurableStorage::put      ~1.50M        │  ← Raw storage layer
└─────────────────────────────────────────┘

Bottleneck removed: Group commit coordination
```

---

## Key Takeaways

1. **Group commit is workload-dependent**: Great for concurrent multi-tenant, counterproductive for sequential single-threaded.

2. **Allocation matters at scale**: HashMap per row adds up to significant overhead at 1M+ ops/sec.

3. **Lock-free structures help**: DashMap eliminates lock contention for concurrent access patterns.

4. **Layer overhead accumulates**: Raw storage was fast; overhead was in the abstraction layers.

5. **Measure before optimizing**: Profiling revealed the true bottleneck was group commit, not I/O.

---

## Future Optimizations

See [Architecture](/concepts/architecture) for remaining optimization areas:

- **Memtable size limits**: Flush to disk when threshold exceeded
- **WAL compaction**: Checkpoint/compaction to reclaim disk space
- **Adaptive group commit**: Switch modes based on workload pattern
- **Connection pooling**: For multi-tenant scenarios

---

## Remaining Architectural Gaps

Based on the comprehensive architecture analysis in `task.md`, the following optimizations are planned:

### High Priority (P0-P1)

| Gap | Current State | Target | Impact |
|-----|---------------|--------|--------|
| **Epoch-Based GC** | O(n) scan all keys | O(expired_versions) per cycle | 10-100× GC reduction |
| **Client GroupCommit** | Duplicate implementation in client/storage | Single source in storage layer | Eliminates confusion |
| **Adaptive Group Commit** | Fixed thresholds (sync every 100 commits) | Little's Law: W* = √(τ/λ) | Better latency tuning |

### Medium Priority (P2)

| Gap | Current State | Target | Impact |
|-----|---------------|--------|--------|
| **SSI Validation** | read_set/write_set tracked but not validated | Full dangerous structure detection | Serializability |
| **Vector Index WAL** | HNSW uses periodic snapshots | WAL integration for crash recovery | ACID for embeddings |
| **Cardinality Estimation** | Static `column_cardinalities` | HyperLogLog++ streaming updates | 5× plan quality |

### Lower Priority (P3)

| Gap | Current State | Target | Impact |
|-----|---------------|--------|--------|
| **Clock-Pro Buffer** | TinyLFU via moka | Clock-Pro + FIFO ghost cache | 2× hit rate for scans |
| **io_uring Integration** | Sync fallback on non-Linux | SQ polling + batched submission | 3× IOPS on Linux |
| **Tiered Compaction** | Implicit leveled strategy | Hybrid tiered/leveled | 70% less write amp |

### Formulas Reference

**Optimal Group Commit Wait Time** (Little's Law):
```
W* = √(τ/λ)

Where:
- τ = fsync latency (~5ms on NVMe)
- λ = arrival rate (ops/sec)

For λ = 10,000 TPS, τ = 5ms:
  W* = √(0.005/10000) = 0.707ms
  Expected batch size = λ × W* = 7 transactions
```

**MVCC GC with Epoch-Based Reclamation**:
```
Memory bound = 3 × epoch_duration × write_rate

For epoch_duration = 100ms, write_rate = 100K/s:
  Max retained = 30,000 versions = ~1.9MB
```

---

## Reproducing Benchmarks

```bash
cd toon_database
cargo run -p benchmarks --release
```

This runs comparisons against SQLite and tests various ToonDB APIs.
