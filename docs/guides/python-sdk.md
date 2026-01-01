# Tutorial: Python SDK

> **Time:** 15 minutes  
> **Difficulty:** Beginner  
> **Prerequisites:** Python 3.9+, pip

Complete walkthrough of ToonDB's Python SDK covering all access modes.

---

## What You'll Learn

- ✅ Embedded mode (FFI) for single-process apps
- ✅ IPC mode for multi-process scenarios  
- ✅ Bulk API for high-throughput vector operations
- ✅ When to use each mode

---

## Installation

```bash
pip install toondb-client
```

**Zero compilation required** — pre-built binaries for:
- Linux (x86_64, aarch64)
- macOS (Intel, Apple Silicon)
- Windows (x64)

---

## Mode 1: Embedded (FFI)

Best for: Single-process applications, CLI tools, testing

### Basic Usage

```python
from toondb import Database

# Open database (creates if not exists)
db = Database.open("./my_database")

# Simple key-value operations
db.put(b"users/alice/name", b"Alice Smith")
db.put(b"users/alice/email", b"alice@example.com")

# Read data
name = db.get(b"users/alice/name")
print(name.decode())  # "Alice Smith"

# Check if key exists (get returns None if not found)
value = db.get(b"users/alice/name")
if value is not None:
    print("Key exists!")

# Delete
db.delete(b"users/alice/email")

# Close when done
db.close()
```

### Transactions

```python
from toondb import Database

db = Database.open("./my_database")

# Transaction with context manager (auto-commit/rollback)
with db.transaction() as txn:
    txn.put(b"accounts/1/balance", b"1000")
    txn.put(b"accounts/2/balance", b"500")
    # Commits automatically on exit
    # Rolls back on exception

# Manual transaction control
txn = db.transaction()
try:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    txn.commit()
except Exception:
    txn.abort()
    raise

db.close()
```

### Prefix Scans

```python
from toondb import Database

db = Database.open("./my_database")

# Store hierarchical data
db.put(b"users/1/name", b"Alice")
db.put(b"users/1/email", b"alice@ex.com")
db.put(b"users/2/name", b"Bob")
db.put(b"users/2/email", b"bob@ex.com")

# Scan by prefix (use bytes for start/end range)
print("All users:")
for key, value in db.scan(b"users/"):
    print(f"  {key.decode()}: {value.decode()}")

# Scan specific user
print("\nUser 1 data:")
for key, value in db.scan(b"users/1/"):
    print(f"  {key.decode()}: {value.decode()}")

db.close()
```

---

## Mode 2: IPC (Client-Server)

Best for: Multi-process apps, microservices, long-running servers

### Starting the Server

```bash
# Start ToonDB server
toondb-server --socket /tmp/toondb.sock --db ./server_data
```

### Client Usage

```python
from toondb import IpcClient

# Connect to server using class method
client = IpcClient.connect("/tmp/toondb.sock")

# Key-value operations
client.put(b"key", b"value")
value = client.get(b"key")

# Path-based operations
client.put_path(["users", "alice", "name"], b"Alice")
value = client.get_path(["users", "alice", "name"])

# Transactions
txn_id = client.begin_transaction()
# Note: use regular put/get while in transaction
client.commit(txn_id)
# Or: client.abort(txn_id)

# Server stats
stats = client.stats()
print(f"Stats: {stats}")

# Check connection with ping
latency = client.ping()
print(f"Latency: {latency:.3f}s")

client.close()
```

### Using Context Manager

```python
from toondb import IpcClient

# Auto-close on exit
with IpcClient.connect("/tmp/toondb.sock") as client:
    client.put(b"key", b"value")
# Connection automatically closed
```

---

## Mode 3: Bulk API

Best for: Large-scale vector indexing, batch operations

### Building Vector Index

```python
from toondb.bulk import bulk_build_index, bulk_query_index
import numpy as np

# Generate embeddings (replace with your embedding model)
embeddings = np.random.randn(100000, 384).astype(np.float32)

# Build HNSW index (~12x faster than FFI)
stats = bulk_build_index(
    embeddings,
    output="./vectors.hnsw",
    m=16,               # Graph connectivity
    ef_construction=200, # Build quality
    quantization="f32"   # f32, f16, or bf16
)

print(f"Built index: {stats['vectors_indexed']} vectors")
print(f"Build time: {stats['build_time_ms']}ms")
```

### Querying Index

```python
from toondb.bulk import bulk_query_index
import numpy as np

# Query vector
query = np.random.randn(384).astype(np.float32)

# Find nearest neighbors
results = bulk_query_index(
    index_path="./vectors.hnsw",
    query=query,
    k=10,    # Number of results
    ef=50    # Search quality (optional)
)

for id, distance in results:
    print(f"ID: {id}, Distance: {distance:.4f}")
```

### Batch Queries

For multiple query vectors, call `bulk_query_index` in a loop:

```python
from toondb.bulk import bulk_query_index
import numpy as np

# Multiple query vectors
queries = np.random.randn(100, 384).astype(np.float32)

# Query each vector
all_results = []
for query in queries:
    results = bulk_query_index(
        index_path="./vectors.hnsw",
        query=query,
        k=10,
        ef=50
    )
    all_results.append(results)

for i, results in enumerate(all_results):
    print(f"Query {i}: {len(results)} results")
```

---

## Choosing the Right Mode

| Criteria | Embedded | IPC | Bulk |
|----------|----------|-----|------|
| **Latency** | ~15ns overhead | ~100µs per call | Batch amortized |
| **Throughput** | 100K ops/s | 10K ops/s | 1.6M vec/s |
| **Multi-process** | ❌ | ✅ | ✅ |
| **Use case** | CLI, testing | Services | Vector ops |

### Decision Tree

```
Is this a vector-heavy workload?
├─ Yes → Use Bulk API
└─ No
   ├─ Single process?
   │  ├─ Yes → Use Embedded
   │  └─ No → Use IPC
   └─ Need server isolation?
      ├─ Yes → Use IPC
      └─ No → Use Embedded
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""ToonDB Python SDK demo covering all modes."""

from toondb import Database, IpcClient
import numpy as np
import json

def demo_embedded():
    """Embedded mode demo."""
    print("=== Embedded Mode ===")
    
    db = Database.open("./demo_embedded")
    
    # Store user data
    with db.transaction() as txn:
        txn.put(b"users/1/profile", json.dumps({
            "name": "Alice",
            "email": "alice@example.com"
        }).encode())
        txn.put(b"users/1/preferences", json.dumps({
            "theme": "dark",
            "notifications": True
        }).encode())
    
    # Read back
    profile = json.loads(db.get(b"users/1/profile").decode())
    print(f"User: {profile['name']} <{profile['email']}>")
    
    # Scan
    print("All user data:")
    for key, value in db.scan(b"users/1/"):
        print(f"  {key.decode()}")
    
    db.close()


def demo_bulk():
    """Bulk API demo."""
    print("\n=== Bulk API Mode ===")
    
    try:
        from toondb.bulk import bulk_build_index, bulk_query_index
    except ImportError:
        print("Bulk API requires numpy")
        return
    
    # Create embeddings
    print("Generating embeddings...")
    embeddings = np.random.randn(10000, 384).astype(np.float32)
    
    # Build index
    print("Building index...")
    stats = bulk_build_index(
        embeddings,
        output="./demo.hnsw",
        m=16,
        ef_construction=100
    )
    print(f"Built: {stats}")
    
    # Query
    query = np.random.randn(384).astype(np.float32)
    results = bulk_query_index(
        index_path="./demo.hnsw",
        query=query,
        k=5
    )
    
    print("Top 5 results:")
    for id, dist in results:
        print(f"  ID={id}, distance={dist:.4f}")


def main():
    demo_embedded()
    demo_bulk()
    print("\n✅ All demos completed!")


if __name__ == "__main__":
    main()
```

---

## Environment Setup

### Required Environment Variable

For embedded mode (FFI), set the library path:

```bash
# After building from source
export TOONDB_LIB_PATH=/path/to/toondb/target/release

# Or if using pip install (usually not needed)
# The library is bundled in the wheel
```

### Verify Installation

```python
from toondb import Database

# Quick test - create a database file
db = Database.open("./test_db")
db.put(b"test", b"works")
assert db.get(b"test") == b"works"
print("✅ ToonDB Python SDK working!")
db.close()
```

---

## Next Steps

| Goal | Resource |
|------|----------|
| Build agent memory | [First App Tutorial](/getting-started/first-app) |
| Add vector search | [Vector Search Tutorial](/guides/vector-search) |
| Deploy to production | [Deployment Guide](/guides/deployment) |
| API reference | [API Documentation](/api-reference/python-api) |

---

*Tutorial completed! You now know all three Python SDK access modes.* 🎉
