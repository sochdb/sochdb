# ToonDB Python SDK Documentation

A comprehensive Python client SDK for **ToonDB** - the database optimized for LLM context retrieval.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Embedded Mode (FFI)](#embedded-mode-ffi)
4. [IPC Client Mode](#ipc-client-mode)
5. [API Reference](#api-reference)
6. [Use Cases](#use-cases)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### From PyPI

```bash
pip install toondb-client
```

### From Source

```bash
cd toondb-python-sdk
pip install -e .
```

### Native Library (Required for Embedded Mode)

For embedded mode, build the Rust native library:

```bash
cd /path/to/toon_database
cargo build --release
```

Set the library path:

```bash
export TOONDB_LIB_PATH=/path/to/toon_database/target/release
```

---

## Quick Start

### Embedded Mode (Single Process)

```python
from toondb import Database

# Open database (creates if not exists)
db = Database.open("./my_database")

# Store and retrieve data
db.put(b"user:1", b'{"name": "Alice"}')
value = db.get(b"user:1")
print(value)  # b'{"name": "Alice"}'

# Clean up
db.close()
```

### With Context Manager

```python
from toondb import Database

with Database.open("./my_database") as db:
    db.put(b"key", b"value")
    value = db.get(b"key")
# Database automatically closed
```

---

## Embedded Mode (FFI)

The embedded mode provides direct access to ToonDB via FFI to the Rust library. This is the recommended mode for single-process applications.

### Opening a Database

```python
from toondb import Database

# Basic open
db = Database.open("./data")

# Context manager (recommended)
with Database.open("./data") as db:
    # operations here
    pass
```

### Key-Value Operations

```python
# Put (create/update)
db.put(b"key", b"value")

# Get (returns None if not found)
value = db.get(b"key")

# Delete
db.delete(b"key")
```

### Path-Native API

ToonDB supports hierarchical data organization using paths:

```python
# Store at path
db.put_path("users/alice/email", b"alice@example.com")
db.put_path("config/app/theme", b"dark")

# Retrieve by path
email = db.get_path("users/alice/email")
```

### Transactions

Transactions provide ACID guarantees:

```python
# Auto-commit with context manager
with db.transaction() as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    # Automatically commits on success
    # Automatically aborts on exception

# Manual control
txn = db.transaction()
txn.put(b"key", b"value")
txn.commit()  # or txn.abort()
```

### Range Scans

```python
# Scan all keys
for key, value in db.scan():
    print(key, value)

# Scan range [start, end)
for key, value in db.scan(b"user:", b"user;"):
    print(key, value)

# Prefix scan pattern
prefix = b"log:2024-01-15:"
end = prefix[:-1] + bytes([prefix[-1] + 1])
for key, value in db.scan(prefix, end):
    print(key, value)
```

### Administrative Operations

```python
# Force checkpoint to disk
lsn = db.checkpoint()

# Get storage statistics
stats = db.stats()
print(stats)
# {'memtable_size_bytes': 1024, 'wal_size_bytes': 4096, ...}
```

---

## IPC Client Mode

IPC mode allows multi-process access to ToonDB via Unix domain sockets.

### Connecting

```python
from toondb import IpcClient

client = IpcClient.connect("/tmp/toondb.sock", timeout=30.0)
```

### Basic Operations

```python
# Ping (returns latency in seconds)
latency = client.ping()

# Put/Get/Delete - same as embedded
client.put(b"key", b"value")
value = client.get(b"key")
client.delete(b"key")
```

### Path API

```python
# Note: IPC uses list of path segments
client.put_path(["users", "alice", "email"], b"alice@example.com")
email = client.get_path(["users", "alice", "email"])
```

### Transactions

```python
# Begin transaction
txn_id = client.begin_transaction()

# Perform operations
# (Note: IPC operations don't use txn_id directly yet)

# Commit or abort
commit_ts = client.commit(txn_id)
# or: client.abort(txn_id)
```

### Query Builder

```python
from toondb import Query

# Fluent query interface
results = client.query("users/") \
    .limit(10) \
    .offset(0) \
    .select(["name", "email"]) \
    .execute()  # Returns TOON string

# Parse to list of dicts
results_list = client.query("users/") \
    .limit(10) \
    .to_list()
```

### Scan

```python
# Scan with prefix
results = client.scan("users/")
# Returns: [{"key": b"users/1", "value": b"..."}, ...]
```

---

## Bulk Vector Operations

The Bulk API provides high-throughput vector operations by bypassing Python FFI overhead.
Instead of crossing the Python/Rust boundary for each vector, it:

1. Writes vectors to a memory-mapped file
2. Spawns the native `toondb-bulk` binary as a subprocess
3. Returns results via stdout/file

### Why Bulk Operations?

| Method | 768D Throughput | Overhead |
|--------|-----------------|----------|
| Python FFI | ~130 vec/s | 12× slower |
| Bulk API | ~1,600 vec/s | 1.0× baseline |

### Building an Index

```python
from toondb.bulk import bulk_build_index
import numpy as np

# Your embeddings (N × D)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,                    # HNSW max connections
    ef_construction=100,     # Construction search depth
    threads=0,               # 0 = auto
    quiet=False,             # Show progress
)

print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")
```

### Querying an Index

```python
from toondb.bulk import bulk_query_index
import numpy as np

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

### Binary Resolution

The SDK automatically finds the `toondb-bulk` binary:

```python
from toondb.bulk import get_toondb_bulk_path

# Returns path to bundled or installed binary
path = get_toondb_bulk_path()
print(f"Using binary: {path}")
```

Resolution order:
1. **Bundled in wheel**: `_bin/<platform>/toondb-bulk`
2. **System PATH**: `which toondb-bulk`
3. **Cargo target**: `../target/release/toondb-bulk` (development)

### Bulk API Reference

| Function | Description |
|----------|-------------|
| `bulk_build_index(embeddings, output, ...)` | Build HNSW index from numpy array |
| `bulk_query_index(index, query, k, ...)` | Query HNSW index for k nearest neighbors |
| `bulk_info(index)` | Get index metadata |
| `convert_embeddings_to_raw(embeddings, path)` | Convert to raw f32 format |
| `get_toondb_bulk_path()` | Get path to toondb-bulk binary |

### BulkBuildStats

```python
@dataclass
class BulkBuildStats:
    vectors: int          # Total vectors indexed
    dimension: int        # Vector dimension
    elapsed_seconds: float
    rate: float           # vec/s
    index_size_bytes: int
```

---

## API Reference

### class Database

| Method | Description |
|--------|-------------|
| `open(path: str) -> Database` | Open/create database at path |
| `close()` | Close the database |
| `put(key: bytes, value: bytes)` | Store key-value pair |
| `get(key: bytes) -> Optional[bytes]` | Retrieve value by key |
| `delete(key: bytes)` | Delete a key |
| `put_path(path: str, value: bytes)` | Store at hierarchical path |
| `get_path(path: str) -> Optional[bytes]` | Retrieve by path |
| `scan(start: bytes, end: bytes)` | Iterate key range |
| `transaction() -> Transaction` | Begin new transaction |
| `checkpoint() -> int` | Force checkpoint, returns LSN |
| `stats() -> dict` | Get storage statistics |

### class Transaction

| Method | Description |
|--------|-------------|
| `put(key: bytes, value: bytes)` | Put within transaction |
| `get(key: bytes) -> Optional[bytes]` | Get with snapshot isolation |
| `delete(key: bytes)` | Delete within transaction |
| `put_path(path: str, value: bytes)` | Put at path |
| `get_path(path: str) -> Optional[bytes]` | Get at path |
| `scan(start: bytes, end: bytes)` | Scan within transaction |
| `commit() -> int` | Commit transaction |
| `abort()` | Abort/rollback transaction |

### class IpcClient

| Method | Description |
|--------|-------------|
| `connect(path: str, timeout: float) -> IpcClient` | Connect to IPC server |
| `close()` | Close connection |
| `ping() -> float` | Ping, returns latency |
| `put(key: bytes, value: bytes)` | Store key-value |
| `get(key: bytes) -> Optional[bytes]` | Retrieve value |
| `delete(key: bytes)` | Delete key |
| `put_path(path: List[str], value: bytes)` | Store at path |
| `get_path(path: List[str]) -> Optional[bytes]` | Get at path |
| `query(prefix: str) -> Query` | Create query builder |
| `scan(prefix: str) -> List[dict]` | Scan with prefix |
| `begin_transaction() -> int` | Begin transaction |
| `commit(txn_id: int) -> int` | Commit transaction |
| `abort(txn_id: int)` | Abort transaction |
| `checkpoint()` | Force checkpoint |
| `stats() -> dict` | Get statistics |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `ToonDBError` | Base exception |
| `ConnectionError` | Connection failed |
| `TransactionError` | Transaction operation failed |
| `ProtocolError` | Wire protocol error |
| `DatabaseError` | Database operation failed |

---

## Use Cases

### 1. Session Cache

```python
from toondb import Database
import json
from datetime import datetime, timedelta

class SessionCache:
    def __init__(self, db, ttl_hours=24):
        self.db = db
        self.ttl = timedelta(hours=ttl_hours)
    
    def set(self, session_id: str, user_data: dict):
        expires = (datetime.utcnow() + self.ttl).isoformat()
        value = {"data": user_data, "expires": expires}
        self.db.put(f"session:{session_id}".encode(), 
                    json.dumps(value).encode())
    
    def get(self, session_id: str) -> dict | None:
        raw = self.db.get(f"session:{session_id}".encode())
        if not raw:
            return None
        value = json.loads(raw.decode())
        if datetime.fromisoformat(value["expires"]) < datetime.utcnow():
            self.delete(session_id)
            return None
        return value["data"]
    
    def delete(self, session_id: str):
        self.db.delete(f"session:{session_id}".encode())
```

### 2. User Management with Secondary Index

```python
class UserStore:
    def __init__(self, db):
        self.db = db
    
    def create_user(self, email: str, name: str) -> str:
        # Check uniqueness
        if self.db.get(f"idx:email:{email}".encode()):
            raise ValueError("Email exists")
        
        user_id = f"user_{int(time.time()*1000)}"
        user = {"id": user_id, "email": email, "name": name}
        
        with self.db.transaction() as txn:
            txn.put(f"users:{user_id}".encode(), 
                    json.dumps(user).encode())
            txn.put(f"idx:email:{email}".encode(), 
                    user_id.encode())
        
        return user_id
    
    def get_by_email(self, email: str) -> dict | None:
        user_id = self.db.get(f"idx:email:{email}".encode())
        if not user_id:
            return None
        data = self.db.get(f"users:{user_id.decode()}".encode())
        return json.loads(data.decode()) if data else None
```

### 3. Document Store

```python
class DocumentStore:
    def __init__(self, db, collection: str):
        self.db = db
        self.collection = collection
    
    def insert(self, doc: dict, doc_id: str = None) -> str:
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]
        key = f"doc:{self.collection}:{doc_id}".encode()
        self.db.put(key, json.dumps(doc).encode())
        return doc_id
    
    def find_all(self) -> list[dict]:
        prefix = f"doc:{self.collection}:".encode()
        docs = []
        for key, val in self.db.scan(prefix, prefix[:-1] + b";"):
            docs.append(json.loads(val.decode()))
        return docs
```

### 4. Feature Flags

```python
class FeatureFlags:
    def __init__(self, db, environment: str):
        self.db = db
        self.env = environment
    
    def set(self, feature: str, enabled: bool):
        path = f"features/{self.env}/{feature}"
        self.db.put_path(path, b"true" if enabled else b"false")
    
    def is_enabled(self, feature: str) -> bool:
        path = f"features/{self.env}/{feature}"
        val = self.db.get_path(path)
        return val and val.decode().lower() == "true"
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# ✓ Good
with Database.open("./data") as db:
    db.put(b"key", b"value")

# ✗ Avoid
db = Database.open("./data")
db.put(b"key", b"value")
# Easy to forget db.close()
```

### 2. Batch Operations in Transactions

```python
# ✓ Good - single transaction
with db.transaction() as txn:
    for item in items:
        txn.put(item.key, item.value)

# ✗ Slow - many small transactions
for item in items:
    db.put(item.key, item.value)
```

### 3. Use Appropriate Key Prefixes

```python
# ✓ Good - organized, scannable
db.put(b"user:123:profile", data)
db.put(b"user:123:settings", data)
db.put(b"order:456:items", data)

# ✗ Bad - no structure
db.put(b"user123profile", data)
```

### 4. Handle Missing Keys

```python
# ✓ Good
value = db.get(b"key")
if value is None:
    # Handle missing key
    pass

# ✗ Bad - assumes key exists
value = db.get(b"key").decode()  # AttributeError if None
```

### 5. Error Handling

```python
from toondb.errors import DatabaseError, TransactionError

try:
    with db.transaction() as txn:
        txn.put(b"key", b"value")
except TransactionError as e:
    print(f"Transaction failed: {e}")
except DatabaseError as e:
    print(f"Database error: {e}")
```

---

## Troubleshooting

### Library Not Found

```
DatabaseError: Could not find libtoondb_storage.dylib
```

**Solution**: Set `TOONDB_LIB_PATH` environment variable:
```bash
export TOONDB_LIB_PATH=/path/to/target/release
```

### Connection Refused (IPC)

```
ConnectionError: Failed to connect to /tmp/toondb.sock
```

**Solution**: Ensure IPC server is running:
```bash
cargo run --bin ipc_server -- --socket /tmp/toondb.sock
```

### Transaction Already Completed

```
TransactionError: Transaction already committed
```

**Solution**: Don't reuse transaction objects after commit/abort:
```python
txn = db.transaction()
txn.commit()
# txn.put(...)  # ✗ Error!

# Create new transaction instead
txn2 = db.transaction()
txn2.put(...)
```

---

## Version Compatibility

| SDK Version | ToonDB Version | Python |
|-------------|----------------|--------|
| 0.1.x       | 0.1.x          | 3.9+   |

---

## License

Apache License 2.0 - Same as ToonDB core.
