# sochdb-storage

High-performance storage engine for SochDB with ACID transactions and crash recovery.

## Overview

`sochdb-storage` provides the persistence layer for SochDB, featuring:

- **Write-Ahead Logging (WAL)**: Durable writes with crash recovery
- **MVCC Transactions**: Snapshot isolation for concurrent access
- **Columnar Storage**: 80% I/O reduction via projection pushdown
- **Memory-Mapped I/O**: Zero-copy reads for maximum throughput

## Features

- **ACID Transactions**: Full transactional guarantees with rollback support
- **Crash Recovery**: Automatic recovery from WAL after unexpected shutdown
- **Batch Writes**: Group commit for high write throughput
- **Compaction**: Background compaction to reclaim space
- **FFI Layer**: C-compatible FFI for Python/Node.js bindings

## Installation

```toml
[dependencies]
sochdb-storage = "0.2.5"
```

## Usage

Most users should use the high-level [`sochdb`](https://crates.io/crates/sochdb) crate:

```rust
use sochdb::Database;

let db = Database::open("./my_data")?;

// Transactional writes
let txn = db.transaction()?;
txn.put("users/1", b"Alice")?;
txn.commit()?;
```

## Architecture

```
┌─────────────────────────────────────────┐
│           sochdb-storage                │
├─────────────────────────────────────────┤
│  Transaction Layer (MVCC, Isolation)    │
├─────────────────────────────────────────┤
│  WAL (Write-Ahead Log, Durability)      │
├─────────────────────────────────────────┤
│  Page Cache (Buffer Pool, LRU)          │
├─────────────────────────────────────────┤
│  Storage Engine (LSM-tree, Compaction)  │
└─────────────────────────────────────────┘
```

## Performance

| Operation | Throughput |
|-----------|------------|
| Point reads | ~500K ops/sec |
| Sequential writes | ~200K ops/sec |
| Batch writes (1000) | ~2M ops/sec |
| Recovery time | <100ms for 1GB WAL |

## Crate Structure

| Crate | Purpose |
|-------|---------|
| [`sochdb`](https://crates.io/crates/sochdb) | High-level client API (start here) |
| [`sochdb-core`](https://crates.io/crates/sochdb-core) | Core types and traits |
| `sochdb-storage` | Storage engine (this crate) |
| [`sochdb-index`](https://crates.io/crates/sochdb-index) | HNSW vector indexing |
| [`sochdb-query`](https://crates.io/crates/sochdb-query) | Query planning and execution |

## License

AGPL-3.0-or-later - see [LICENSE](../LICENSE) for details.
