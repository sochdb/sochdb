# ToonDB Python SDK

[![PyPI version](https://badge.fury.io/py/toondb.svg)](https://badge.fury.io/py/toondb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**ToonDB is an AI-native database with token-optimized output, O(|path|) lookups, built-in vector search, and durable transactions.**

Python client SDK for [ToonDB](https://github.com/toondb/toondb) - the database optimized for LLM context retrieval.

## Installation

```bash
pip install toondb-client
```

**Zero compilation required** - pre-built binaries are bundled for all major platforms:
- Linux x86_64 and aarch64 (glibc ≥ 2.17)
- macOS Intel and Apple Silicon (universal2)
- Windows x64

## Features

### Core Database
- 🚀 **Embedded Mode**: Direct FFI access to ToonDB for single-process applications
- 🔗 **IPC Mode**: Multi-process access via Unix domain sockets
- 📁 **Path-Native API**: Hierarchical data organization with O(|path|) lookups
- 💾 **ACID Transactions**: Full transaction support with snapshot isolation
- 🔍 **Range Scans**: Efficient prefix and range queries
- 🎯 **Token-Optimized**: TOON format output designed for LLM context windows

### High-Performance Vector Operations
- ⚡ **Bulk API**: Bypass FFI overhead for high-throughput vector ingestion (~1,600 vec/s vs ~130 vec/s with FFI)
- 🔢 **SIMD Kernels**: Auto-dispatched AVX2/NEON optimizations for distance calculations
- 📊 **Multi-Format Input**: Support for raw float32, NumPy .npy, and in-memory arrays
- 🔎 **HNSW Indexing**: Build and query approximate nearest neighbor indexes

### Distribution
- 📦 **Zero-Compile Install**: Pre-built Rust binaries bundled in wheels
- 🌍 **Cross-Platform**: Linux, macOS, Windows with automatic platform detection
- 🔧 **Fallback Chain**: Bundled binary → PATH → cargo target (for development)

## Detailed Installation

### From PyPI (Recommended)

```bash
pip install toondb-client
```

### From Source (Development)

```bash
git clone https://github.com/toondb/toondb.git
cd toondb/toondb-python-sdk
pip install -e .

# Build the Rust binary (required for bulk operations)
cargo build --release -p toondb-tools
```

## Quick Start

### Embedded Mode (Recommended for single-process apps)

```python
from toondb import Database

# Open a database (creates if doesn't exist)
with Database.open("./my_database") as db:
    # Simple key-value operations
    db.put(b"user:123", b'{"name": "Alice", "email": "alice@example.com"}')
    value = db.get(b"user:123")
    
    # Path-native API
    db.put_path("users/alice/email", b"alice@example.com")
    email = db.get_path("users/alice/email")
    
    # Transactions
    with db.transaction() as txn:
        txn.put(b"key1", b"value1")
        txn.put(b"key2", b"value2")
        # Automatically commits on exit, or aborts on exception
```

### IPC Mode (For multi-process access)

```python
from toondb import IpcClient

# Connect to a running ToonDB IPC server
client = IpcClient.connect("/tmp/toondb.sock")

# Same API as embedded mode
client.put(b"key", b"value")
value = client.get(b"key")

# Query Builder
results = client.query("users/") \
    .limit(10) \
    .select(["name", "email"]) \
    .to_list()
```

### Bulk Vector Ingestion (Bypass FFI for Max Throughput)

For large-scale vector index building, the Bulk API bypasses Python FFI overhead
by shelling out to the native `toondb-bulk` binary:

```python
from toondb.bulk import bulk_build_index, bulk_query_index
import numpy as np

# Generate or load embeddings (10K × 768D)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index at ~1,600 vec/s (vs ~130 vec/s with FFI)
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,
    ef_construction=100,
)

print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")
```

**Query the index:**

```python
# Single query
query = np.random.randn(768).astype(np.float32)
results = bulk_query_index(
    index="my_index.hnsw",
    query=query,
    k=10,
    ef_search=64,
)

for neighbor in results:
    print(f"ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
```

**Performance Comparison (768D vectors):**

| Method | Throughput | Overhead |
|--------|------------|----------|
| Python FFI | ~130 vec/s | 12× slower |
| Bulk API | ~1,600 vec/s | 1.0× baseline |

## Use Cases

### User Session Management

```python
from toondb import Database
import json

with Database.open("./sessions") as db:
    # Store session
    session = {"user_id": "123", "token": "abc", "expires": "2024-12-31"}
    db.put(b"session:abc123", json.dumps(session).encode())
    
    # Retrieve session
    data = db.get(b"session:abc123")
    if data:
        session = json.loads(data.decode())
```

### Configuration Store

```python
from toondb import Database

with Database.open("./config") as db:
    # Hierarchical configuration
    db.put_path("api/auth/timeout", b"30")
    db.put_path("api/auth/retries", b"3")
    db.put_path("api/storage/endpoint", b"https://storage.example.com")
    
    # Read config
    timeout = db.get_path("api/auth/timeout")  # b"30"
```

### Document Storage with Indexing

```python
from toondb import Database
import json

with Database.open("./docs") as db:
    # Store document with category index
    doc = {"title": "Hello World", "category": "tutorials"}
    doc_id = "doc_001"
    
    with db.transaction() as txn:
        txn.put(f"docs:{doc_id}".encode(), json.dumps(doc).encode())
        txn.put(f"idx:category:tutorials:{doc_id}".encode(), b"1")
    
    # Query by category using prefix scan
    for key, _ in db.scan(b"idx:category:tutorials:", b"idx:category:tutorials;"):
        doc_id = key.decode().split(":")[-1]
        print(f"Found: {doc_id}")
```

## Building the Native Library

For embedded mode, you need to build the Rust library:

```bash
# Clone ToonDB
git clone https://github.com/toondb/toondb.git
cd toondb

# Build release
cargo build --release

# Set library path
export TOONDB_LIB_PATH=$(pwd)/target/release
```

## API Reference

### Database (Embedded Mode)

| Method | Description |
|--------|-------------|
| `Database.open(path)` | Open/create database |
| `put(key, value)` | Store key-value pair |
| `get(key)` | Retrieve value (None if missing) |
| `delete(key)` | Delete a key |
| `put_path(path, value)` | Store at hierarchical path |
| `get_path(path)` | Retrieve by path |
| `scan(start, end)` | Iterate key range |
| `transaction()` | Begin ACID transaction |
| `checkpoint()` | Force durability checkpoint |
| `stats()` | Get storage statistics |

### IpcClient

| Method | Description |
|--------|-------------|
| `IpcClient.connect(path)` | Connect to IPC server |
| `ping()` | Check latency |
| `query(prefix)` | Create query builder |
| `scan(prefix)` | Scan keys with prefix |
| `begin_transaction()` | Start transaction |
| `commit(txn_id)` | Commit transaction |
| `abort(txn_id)` | Abort transaction |

### Bulk API

| Function | Description |
|----------|-------------|
| `bulk_build_index(embeddings, output, ...)` | Build HNSW index from numpy array |
| `bulk_query_index(index, query, k, ...)` | Query HNSW index for k nearest neighbors |
| `bulk_info(index)` | Get index metadata (vectors, dimension, etc.) |
| `convert_embeddings_to_raw(embeddings, path)` | Convert numpy array to raw f32 format |
| `get_toondb_bulk_path()` | Get path to bundled toondb-bulk binary |

## Documentation

- [Full SDK Documentation](docs/SDK_DOCUMENTATION.md)
- [Bulk Operations Guide](../docs/BULK_OPERATIONS.md)
- [Python Distribution Architecture](../docs/PYTHON_DISTRIBUTION.md)
- [Examples](examples/)
- [ToonDB Repository](https://github.com/toondb/toondb)

## Platform Support

| Platform | Wheel Tag | Notes |
|----------|-----------|-------|
| Linux x86_64 | `manylinux_2_17_x86_64` | glibc ≥ 2.17 (CentOS 7+) |
| Linux aarch64 | `manylinux_2_17_aarch64` | ARM servers (AWS Graviton) |
| macOS | `macosx_11_0_universal2` | Intel + Apple Silicon |
| Windows | `win_amd64` | Windows 10+ x64 |

Binary resolution order:
1. Bundled in wheel (`_bin/<platform>/toondb-bulk`)
2. System PATH (`toondb-bulk`)
3. Cargo target directory (development)

## Requirements

- Python 3.9+
- NumPy (for vector operations)
- No Rust toolchain required for installation

## License

Apache License 2.0

## Author

**Sushanth** - [GitHub](https://github.com/sushanthpy)
