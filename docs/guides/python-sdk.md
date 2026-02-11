# Python SDK Guide

> **Version:** 0.5.3  
> **Time:** 45 minutes  
> **Difficulty:** Beginner to Intermediate  
> **Prerequisites:** Python 3.9+

Complete guide to SochDB's Python SDK covering dual-mode architecture (embedded FFI + server gRPC), namespaces, collections, vector search, priority queues, and advanced features.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture: Dual-Mode](#architecture-dual-mode)
4. [Namespace & Collections](#namespace--collections)
5. [Vector Search](#vector-search)
6. [Priority Queue](#priority-queue)
7. [SQL Database](#sql-database)
8. [Key-Value Operations](#key-value-operations)
9. [Path API](#path-api)
10. [Prefix Scanning](#prefix-scanning)
11. [Transactions](#transactions)
12. [Graph Overlay](#graph-overlay)
13. [Temporal Graph](#temporal-graph)
14. [Server Mode (gRPC/IPC)](#server-mode-grpcipc)
15. [CLI Tools](#cli-tools)
16. [Advanced Features](#advanced-features)
    - [TOON Format](#toon-format)
    - [Batched Scanning](#batched-scanning)
    - [Statistics & Monitoring](#statistics--monitoring)
    - [Manual Checkpoint](#manual-checkpoint)
17. [Error Handling](#error-handling)
18. [Best Practices](#best-practices)
19. [Complete Examples](#complete-examples)

---

## Installation

```bash
pip install sochdb
```

**What's New in 0.4.7:**
- ✅ Improved FFI stability and error messages
- ✅ Better platform detection for native libraries

**What's New in 0.4.6:**
- ✅ Temporal Graph API for time-aware relationships
- ✅ Enhanced Graph Overlay with BFS/DFS traversal

**What's New in 0.4.3:**
- ✅ Priority Queue API with ordered-key task entries
- ✅ Streaming TopK for efficient ORDER BY + LIMIT
- ✅ Backend-agnostic queue (FFI, gRPC, In-Memory)

**What's New in 0.4.1:**
- ✅ Namespace API for multi-tenant isolation
- ✅ Collection API with auto-dimension vectors
- ✅ Lock error types (DatabaseLockedError, LockTimeoutError)
- ✅ Concurrent mode support

**What's New in 0.4.0:**
- ✅ Project rename: ToonDB → SochDB
- ✅ Dual-mode architecture: Embedded (FFI) + Server (gRPC)
- ✅ VectorIndex class with native HNSW

> **Import Note:** Install with `pip install sochdb`, import as `from sochdb import Database`

**Pre-built for:**
- Linux (x86_64, aarch64)
- macOS (Intel, Apple Silicon)
- Windows (x64)

---

## Quick Start

### Embedded Mode

```python
from sochdb import Database

# Open database
with Database.open("./my_database") as db:
    # Put and Get
    db.put(b"user:123", b'{"name":"Alice","age":30}')
    value = db.get(b"user:123")
    print(value.decode())
    # Output: {"name":"Alice","age":30}
```

**Output:**
```
{"name":"Alice","age":30}
```

---

## Architecture: Dual-Mode

SochDB Python SDK supports **two deployment modes**:

### Embedded Mode (FFI) - Recommended for Single Process

Direct FFI bindings to Rust libraries. No server required.

```python
from sochdb import Database

# Direct FFI - no server needed
with Database.open("./mydb") as db:
    db.put(b"key", b"value")
    value = db.get(b"key")
```

**Best for:** Local development, notebooks, simple apps, edge deployments.

### Server Mode (gRPC) - For Distributed Systems

Thin client connecting to sochdb-grpc server.

```python
from sochdb import SochDBClient

# Connect to server
client = SochDBClient("localhost:50051")
client.put_kv("namespace", "key", b"value")
value = client.get_kv("namespace", "key")
client.close()
```

**Best for:** Production, multi-language environments, microservices.

### Concurrent Mode (v0.4.1+)

Multi-process access to the same database with MVCC.

```python
from sochdb import Database

# Multiple processes can access simultaneously
db = Database.open_concurrent("./shared_db")
print(f"Concurrent mode: {db.is_concurrent}")  # True
```

---

## Namespace & Collections

**New in v0.4.1** — Type-safe multi-tenant isolation with vector collections.

### Creating Namespaces

```python
from sochdb import Database, NamespaceConfig

with Database.open("./multi_tenant") as db:
    # Create namespace for tenant
    config = NamespaceConfig(
        name="tenant_123",
        display_name="Acme Corp",
        labels={"tier": "enterprise"},
    )
    ns = db.create_namespace(config)
    
    # Or get existing
    ns = db.namespace("tenant_123")
```

### Creating Collections

```python
from sochdb import CollectionConfig, DistanceMetric

# Create vector collection
config = CollectionConfig(
    name="documents",
    dimension=384,  # None = auto-infer from first vector
    metric=DistanceMetric.COSINE,
    m=16,
    ef_construction=100,
)
collection = ns.create_collection(config)

# Or simpler
collection = ns.create_collection("embeddings", dimension=768)
```

### Vector Operations

```python
# Insert vectors with metadata
collection.insert(
    vector=[0.1, 0.2, 0.3, ...],  # 384-dim
    metadata={"source": "web", "url": "https://..."},
    id="doc_001"  # Optional, auto-generated if omitted
)

# Batch insert
collection.insert_batch(
    vectors=[[0.1, ...], [0.2, ...], [0.3, ...]],
    metadatas=[{"type": "a"}, {"type": "b"}, {"type": "c"}],
    ids=["doc_1", "doc_2", "doc_3"]
)
```

### Unified Search API

```python
from sochdb import SearchRequest

# Vector search
results = collection.search(
    SearchRequest(
        vector=query_embedding,
        k=10,
        filter={"source": "web"},
    )
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score:.4f}")

# Keyword search (if hybrid enabled)
results = collection.search(
    SearchRequest(
        text_query="machine learning",
        k=10,
    )
)

# Hybrid search (RRF fusion)
results = collection.search(
    SearchRequest(
        vector=query_embedding,
        text_query="ML algorithms",
        k=10,
        alpha=0.7,  # 0.7 vector + 0.3 keyword
    )
)
```

### Convenience Methods

```python
# Quick vector search
results = collection.vector_search(query_embedding, k=10)

# Quick keyword search
results = collection.keyword_search("neural networks", k=10)

# Quick hybrid search
results = collection.hybrid_search(query_embedding, "deep learning", k=10)
```

---

## Vector Search

### Native VectorIndex (HNSW)

```python
from sochdb import VectorIndex

# Create index
index = VectorIndex(
    dimension=768,
    metric="cosine",  # cosine, euclidean, dot_product
    m=16,
    ef_construction=100,
)

# Insert vectors
import numpy as np
embeddings = np.random.randn(10000, 768).astype(np.float32)

for i, vec in enumerate(embeddings):
    index.insert(f"doc_{i}", vec)

# Or batch insert (faster)
ids = [f"doc_{i}" for i in range(len(embeddings))]
index.insert_batch(ids, embeddings)  # ~15,000 vec/s with FFI

# Search
query = np.random.randn(768).astype(np.float32)
results = index.search(query, k=10, ef_search=64)

for id, distance in results:
    print(f"{id}: {distance:.4f}")
```

### Bulk Operations (Legacy)

```python
from sochdb.bulk import bulk_build_index, bulk_query_index

# Build HNSW index
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,
    ef_construction=100,
    metric="cosine"
)
print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")

# Query
results = bulk_query_index(
    index="my_index.hnsw",
    query=query,
    k=10,
    ef_search=64
)
```

---

## Priority Queue

**New in v0.4.3** — First-class queue API with ordered-key task entries.

### Creating Queues

```python
from sochdb import Database
from sochdb.queue import PriorityQueue, QueueConfig

db = Database.open("./queue_db")

# Create queue with config
config = QueueConfig(
    visibility_timeout_ms=30000,  # 30s lease
    max_attempts=3,
    dead_letter_queue="failed_tasks",
)
queue = PriorityQueue.from_database(db, "tasks", config)
```

### Enqueue Tasks

```python
# Enqueue with priority (lower = higher priority)
task_id = queue.enqueue(
    priority=1,  # High priority
    payload=b'{"action": "process_order", "order_id": 123}',
    metadata={"source": "api"},
)
print(f"Enqueued task: {task_id}")

# Delayed task
import time
queue.enqueue(
    priority=5,
    payload=b"delayed task",
    delay_ms=60000,  # Visible in 1 minute
)
```

### Dequeue and Process

```python
# Dequeue (claims task with lease)
task = queue.dequeue(worker_id="worker-1")

if task:
    try:
        # Process task
        payload = task.payload
        print(f"Processing: {payload}")
        
        # Acknowledge completion
        queue.ack(task.task_id)
    except Exception as e:
        # Negative ack (retry or dead-letter)
        queue.nack(task.task_id)
```

### Queue Statistics

```python
stats = queue.stats()
print(f"Pending: {stats.pending}")
print(f"In-flight: {stats.claimed}")
print(f"Dead-lettered: {stats.dead_lettered}")
```

### Streaming TopK

For efficient ORDER BY + LIMIT queries:

```python
from sochdb.queue import StreamingTopK

# Get top 100 highest priority tasks
top_tasks = queue.top_k(k=100)
for task in top_tasks:
    print(f"Priority {task.priority}: {task.task_id}")
```

---

## SQL Database

### CREATE TABLE

```python
from sochdb import Database

with Database.open("./sql_db") as db:
    # Create table
    db.execute_sql("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER
        )
    """)
    
    # Insert data
    db.execute_sql("""
        INSERT INTO users (id, name, email, age)
        VALUES (1, 'Alice', 'alice@example.com', 30)
    """)
    
    db.execute_sql("""
        INSERT INTO users (id, name, email, age)
        VALUES (2, 'Bob', 'bob@example.com', 25)
    """)
```

**Output:**
```
Table 'users' created
2 rows inserted
```

### SELECT Queries

```python
# Select all
results = db.execute_sql("SELECT * FROM users")
for row in results:
    print(row)

# Output:
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 30}
# {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'age': 25}

# WHERE clause
results = db.execute_sql("SELECT name, age FROM users WHERE age > 26")
for row in results:
    print(f"{row['name']}: {row['age']} years old")

# Output:
# Alice: 30 years old
```

### JOIN Queries

```python
# Create orders table
db.execute_sql("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product TEXT,
        amount REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")

# Insert orders
db.execute_sql("INSERT INTO orders VALUES (1, 1, 'Laptop', 999.99)")
db.execute_sql("INSERT INTO orders VALUES (2, 1, 'Mouse', 25.00)")
db.execute_sql("INSERT INTO orders VALUES (3, 2, 'Keyboard', 75.00)")

# JOIN query
results = db.execute_sql("""
    SELECT users.name, orders.product, orders.amount
    FROM users
    JOIN orders ON users.id = orders.user_id
    WHERE orders.amount > 50
    ORDER BY orders.amount DESC
""")

for row in results:
    print(f"{row['name']} bought {row['product']} for ${row['amount']}")
```

**Output:**
```
Alice bought Laptop for $999.99
Bob bought Keyboard for $75.0
```

### Aggregations

```python
# GROUP BY with aggregations
results = db.execute_sql("""
    SELECT users.name, COUNT(*) as order_count, SUM(orders.amount) as total
    FROM users
    JOIN orders ON users.id = orders.user_id
    GROUP BY users.name
    ORDER BY total DESC
""")

for row in results:
    print(f"{row['name']}: {row['order_count']} orders, ${row['total']} total")
```

**Output:**
```
Alice: 2 orders, $1024.99 total
Bob: 1 orders, $75.0 total
```

### UPDATE and DELETE

```python
# Update
db.execute_sql("UPDATE users SET age = 31 WHERE name = 'Alice'")

# Delete
db.execute_sql("DELETE FROM users WHERE age < 26")

# Verify
results = db.execute_sql("SELECT name, age FROM users")
for row in results:
    print(row)

# Output:
# {'name': 'Alice', 'age': 31}
```

---

## Key-Value Operations

### Basic Operations

```python
# Put
db.put(b"key", b"value")

# Get
value = db.get(b"key")
if value:
    print(value.decode())
else:
    print("Key not found")

# Delete
db.delete(b"key")

# Output:
# value
# Key not found (after delete)
```

### JSON Data

```python
import json

# Store JSON
user = {"name": "Alice", "email": "alice@example.com", "age": 30}
db.put(b"users/alice", json.dumps(user).encode())

# Retrieve JSON
value = db.get(b"users/alice")
if value:
    user = json.loads(value.decode())
    print(f"Name: {user['name']}, Age: {user['age']}")

# Output:
# Name: Alice, Age: 30
```

---

## Path API

```python
# Store hierarchical data
db.put_path("users/alice/email", b"alice@example.com")
db.put_path("users/alice/age", b"30")
db.put_path("users/alice/settings/theme", b"dark")

# Retrieve by path
email = db.get_path("users/alice/email")
print(f"Alice's email: {email.decode()}")

# Output:
# Alice's email: alice@example.com
```

---

## Prefix Scanning

⭐ **Most efficient way to iterate keys:**

```python
# Insert multi-tenant data
db.put(b"tenants/acme/users/1", b'{"name":"Alice"}')
db.put(b"tenants/acme/users/2", b'{"name":"Bob"}')
db.put(b"tenants/acme/orders/1", b'{"total":100}')
db.put(b"tenants/globex/users/1", b'{"name":"Charlie"}')

# Scan only ACME Corp data (tenant isolation)
results = list(db.scan(b"tenants/acme/", b"tenants/acme;"))
print(f"ACME Corp has {len(results)} items:")
for key, value in results:
    print(f"  {key.decode()}: {value.decode()}")
```

**Output:**
```
ACME Corp has 3 items:
  tenants/acme/orders/1: {"total":100}
  tenants/acme/users/1: {"name":"Alice"}
  tenants/acme/users/2: {"name":"Bob"}
```

**Why use scan():**
- **Fast**: O(|prefix|) performance
- **Isolated**: Perfect for multi-tenant apps
- **Efficient**: Binary-safe iteration

---

## Transactions

### Automatic Transactions

```python
# Context manager handles commit/abort
with db.transaction() as txn:
    txn.put(b"account:1:balance", b"1000")
    txn.put(b"account:2:balance", b"500")
    # Commits on success, aborts on exception
```

**Output:**
```
✅ Transaction committed
```

### Manual Control

```python
txn = db.begin_transaction()
try:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    
    # Scan within transaction
    for key, value in txn.scan(b"key", b"key~"):
        print(f"{key.decode()}: {value.decode()}")
    
    txn.commit()
except Exception as e:
    txn.abort()
    raise
```

**Output:**
```
key1: value1
key2: value2
✅ Transaction committed
```

---

## Graph Overlay

**New in v0.3.3** — Lightweight graph layer for agent memory relationships.

### Creating Nodes and Edges

```python
from sochdb import Database

with Database.open("./agent_memory") as db:
    # Create nodes
    db.graph_add_node(
        namespace="agent_001",
        node_id="user_alice",
        node_type="User",
        properties={"name": "Alice", "role": "admin"}
    )
    
    db.graph_add_node(
        namespace="agent_001",
        node_id="conv_123",
        node_type="Conversation",
        properties={"topic": "Support request"}
    )
    
    # Create edge
    db.graph_add_edge(
        namespace="agent_001",
        from_id="user_alice",
        edge_type="STARTED",
        to_id="conv_123",
        properties={"timestamp": "2026-01-15T10:30:00Z"}
    )
```

### Traversal

```python
# BFS traversal from a node
results = db.graph_traverse(
    namespace="agent_001",
    start_node="user_alice",
    max_depth=3,
    order="bfs"  # or "dfs"
)

for node in results:
    print(f"Node: {node['id']} ({node['type']})")
```

---

## Temporal Graph

**New in v0.4.6** — Time-aware relationships for historical queries.

### Adding Temporal Edges

```python
from sochdb import Database

with Database.open("./temporal_db") as db:
    # Add temporal edge with validity period
    db.add_temporal_edge(
        namespace="org",
        from_id="alice",
        edge_type="WORKS_AT",
        to_id="acme_corp",
        valid_from=1704067200000,  # 2024-01-01 (ms)
        valid_until=1735689600000,  # 2025-01-01 (ms)
        properties={"role": "Engineer"}
    )
    
    # Add another edge (current)
    db.add_temporal_edge(
        namespace="org",
        from_id="alice",
        edge_type="WORKS_AT",
        to_id="globex_inc",
        valid_from=1735689600000,  # 2025-01-01
        valid_until=0,  # 0 = no end (current)
        properties={"role": "Senior Engineer"}
    )
```

### Querying at a Point in Time

```python
# Query: "Where did Alice work on 2024-06-15?"
timestamp = 1718409600000  # 2024-06-15 in ms

results = db.query_temporal_graph(
    namespace="org",
    node="alice",
    mode="point_in_time",
    timestamp=timestamp,
    edge_type="WORKS_AT"
)

for edge in results:
    print(f"Worked at: {edge['to_id']} as {edge['properties']['role']}")
# Output: Worked at: acme_corp as Engineer
```

### Querying a Time Range

```python
# Query: "All jobs Alice had in 2024"
results = db.query_temporal_graph(
    namespace="org",
    node="alice",
    mode="range",
    start_time=1704067200000,  # 2024-01-01
    end_time=1735689599999,    # 2024-12-31
    edge_type="WORKS_AT"
)

for edge in results:
    print(f"{edge['to_id']}: {edge['valid_from']} - {edge['valid_until']}")
```

---

## Query Builder

Returns results in **TOON format** (token-optimized for LLMs):

```python
# Insert structured data
db.put(b"products/laptop", b'{"name":"Laptop","price":999,"stock":5}')
db.put(b"products/mouse", b'{"name":"Mouse","price":25,"stock":20}')

# Query with column selection
results = db.query("products/") \
    .select(["name", "price"]) \
    .limit(10) \
    .to_list()

for key, value in results:
    print(f"{key.decode()}: {value.decode()}")
```

**Output (TOON Format):**
```
products/laptop: result[1]{name,price}:Laptop,999
products/mouse: result[1]{name,price}:Mouse,25
```

---

## Vector Search

### Bulk HNSW Index Building

```python
from sochdb.bulk import bulk_build_index, bulk_query_index
import numpy as np

# Generate embeddings (10K × 768D)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index at ~1,600 vec/s
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,
    ef_construction=100,
    metric="cosine"
)

print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")
```

**Output:**
```
Built 10000 vectors at 1598 vec/s
Index size: 45.2 MB
```

### Query HNSW Index

```python
# Single query vector
query = np.random.randn(768).astype(np.float32)

results = bulk_query_index(
    index="my_index.hnsw",
    query=query,
    k=10,
    ef_search=64
)

print(f"Top {len(results)} nearest neighbors:")
for i, neighbor in enumerate(results):
    print(f"{i+1}. ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
```

**Output:**
```
Top 10 nearest neighbors:
1. ID: 3421, Distance: 0.1234
2. ID: 7892, Distance: 0.1456
3. ID: 1205, Distance: 0.1678
...
```

**Performance:**
- Python FFI: ~130 vec/s
- Bulk API: ~1,600 vec/s (12× faster)

---

## Server Mode (gRPC/IPC)

For distributed systems and multi-process applications.

### gRPC Client

```python
from sochdb import SochDBClient

# Connect to gRPC server
client = SochDBClient("localhost:50051")

# Key-Value operations
client.put_kv("my_namespace", "user:123", b'{"name": "Alice"}')
value = client.get_kv("my_namespace", "user:123")
print(value.decode())

# Vector search
results = client.vector_search(
    namespace="my_namespace",
    collection="documents",
    query=[0.1, 0.2, 0.3, ...],
    k=10
)

# Graph operations
client.add_graph_node(
    namespace="agent",
    node_id="user_1",
    node_type="User",
    properties={"name": "Alice"}
)

client.close()
```

### IPC Client (Unix Socket)

For multi-process applications on the same machine:

```bash
# Start IPC server
sochdb-server --db ./my_database
```

```python
from sochdb import IpcClient

client = IpcClient.connect("./my_database/sochdb.sock")

client.put(b"key", b"value")
value = client.get(b"key")
print(value.decode())

client.close()
```

### IPC Mode (Legacy)

For multi-process applications, SochDB provides a high-performance IPC server with Unix domain socket communication.

> **Deep Dive:** See [IPC Server Capabilities](../servers/IPC_SERVER.md) for wire protocol details, internals, and architecture.

### Quick Start

```bash
# Start the IPC server (globally available after pip install)
sochdb-server --db ./my_database

# Check status
sochdb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)
```

```python
# Connect from Python (or any other process)
from sochdb import IpcClient

client = IpcClient.connect("./my_database/sochdb.sock")

client.put(b"key", b"value")
value = client.get(b"key")
print(value.decode())
# Output: value
```

### sochdb-server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db PATH` | `./sochdb_data` | Database directory |
| `--socket PATH` | `<db>/sochdb.sock` | Unix socket path |
| `--max-clients N` | `100` | Maximum concurrent connections |
| `--timeout-ms MS` | `30000` | Connection timeout (30s) |
| `--log-level LEVEL` | `info` | trace/debug/info/warn/error |

### Server Commands

```bash
# Start server
sochdb-server --db ./my_database

# Check if running
sochdb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)
#         Socket: ./my_database/sochdb.sock
#         Database: /absolute/path/to/my_database

# Stop server gracefully
sochdb-server stop --db ./my_database
```

### Production Configuration

```bash
# High-traffic production setup
sochdb-server \
    --db /var/lib/sochdb/production \
    --socket /var/run/sochdb.sock \
    --max-clients 500 \
    --timeout-ms 60000 \
    --log-level info
```

### Wire Protocol

The IPC server uses a binary protocol for high-performance communication. See the [Deep Dive](../servers/IPC_SERVER.md) for full opcode usage.

### Server Statistics

The IPC server tracks real-time metrics accessible via `client.stats()`:

```python
from sochdb import IpcClient

client = IpcClient.connect("./my_database/sochdb.sock")
stats = client.stats()

print(f"Connections: {stats['connections_active']}/{stats['connections_total']}")
print(f"Requests: {stats['requests_success']} success, {stats['requests_error']} errors")
print(f"Throughput: {stats['bytes_received']} bytes in, {stats['bytes_sent']} bytes out")
print(f"Uptime: {stats['uptime_secs']} seconds")
print(f"Active transactions: {stats['active_transactions']}")
```

---

## CLI Tools

Three CLI tools are available globally after `pip install sochdb-client`:

### sochdb-bulk

High-performance bulk vector operations (~1,600 vec/s).

> **Deep Dive:** See [Bulk Operations Capabilities](../servers/BULK_OPERATIONS.md) for benchmarks, file formats, and internals.

```bash
# Build HNSW index from embeddings
sochdb-bulk build-index \
    --input embeddings.npy \
    --output index.hnsw \
    --dimension 768 \
    --max-connections 16 \
    --ef-construction 100 \
    --metric cosine

# Query k-nearest neighbors
sochdb-bulk query \
    --index index.hnsw \
    --query query_vector.raw \
    --k 10 \
    --ef 64

# Get index metadata
sochdb-bulk info --index index.hnsw
# Output:
# Dimension: 768
# Vectors: 100000
# Max connections: 16

# Convert between formats
sochdb-bulk convert \
    --input vectors.npy \
    --output vectors.raw \
    --to-format raw_f32 \
    --dimension 768
```

### sochdb-grpc-server

gRPC server for remote vector search operations.

> **Deep Dive:** See [gRPC Server Capabilities](../servers/GRPC_SERVER.md) for service methods, HNSW configuration, and proto definitions.

```bash
# Start gRPC server
sochdb-grpc-server --host 0.0.0.0 --port 50051

# Check status
sochdb-grpc-server status --port 50051
```

**gRPC Service Methods:** See [gRPC Deep Dive](../servers/GRPC_SERVER.md) for full method signatures.

**Python gRPC Client Example:**

```python
import grpc
from sochdb_pb2 import (
    CreateIndexRequest, SearchRequest, HnswConfig
)
from sochdb_pb2_grpc import VectorIndexServiceStub

# Connect to gRPC server
channel = grpc.insecure_channel('localhost:50051')
stub = VectorIndexServiceStub(channel)

# Create index
response = stub.CreateIndex(CreateIndexRequest(
    name="my_index",
    dimension=768,
    metric=1,  # COSINE
    config=HnswConfig(
        max_connections=16,
        ef_construction=200,
        ef_search=50
    )
))
print(f"Created: {response.info.name}")

# Search
import numpy as np
query = np.random.randn(768).astype(np.float32)
response = stub.Search(SearchRequest(
    index_name="my_index",
    query=query.tolist(),
    k=10
))
for result in response.results:
    print(f"ID: {result.id}, Distance: {result.distance:.4f}")
```

### Environment Variables

Override bundled binaries with custom paths:

```bash
export SOCHDB_SERVER_PATH=/path/to/sochdb-server
export SOCHDB_BULK_PATH=/path/to/sochdb-bulk
export SOCHDB_GRPC_SERVER_PATH=/path/to/sochdb-grpc-server
```

---

## Advanced Features

### TOON Format

**Token-Optimized Output Notation** - Achieve **40-66% token reduction** for LLM context.

```python
from sochdb import Database

# Sample records
records = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
]

# Convert to TOON format
toon_str = Database.to_toon("users", records, ["name", "email"])
print(toon_str)
# Output: users[2]{name,email}:Alice,alice@example.com;Bob,bob@example.com

# Parse TOON back to records
table_name, fields, records = Database.from_toon(toon_str)
print(records)
# Output: [{"name": "Alice", "email": "alice@example.com"}, ...]
```

**Token Comparison:**
- JSON (compact): ~165 tokens
- TOON format: ~70 tokens (**59% reduction!**)

**Use Case: RAG with LLMs**

```python
from sochdb import Database
import openai

with Database.open("./knowledge_base") as db:
    # Query relevant documents
    results = db.execute_sql("""
        SELECT title, content 
        FROM documents 
        WHERE category = 'technical'
        LIMIT 10
    """)
    
    # Convert to TOON for efficient context
    records = [dict(row) for row in results]
    toon_context = Database.to_toon("documents", records, ["title", "content"])
    
    # Send to LLM (saves tokens!)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Context:\n{toon_context}"},
            {"role": "user", "content": "Summarize the documents"}
        ]
    )
```

---

### Batched Scanning

**1000× fewer FFI calls** for large dataset scans.

```python
from sochdb import Database

with Database.open("./my_db") as db:
    # Insert 10K test records
    with db.transaction() as txn:
        for i in range(10000):
            txn.put(f"item:{i:05d}".encode(), f"value:{i}".encode())
    
    # Regular scan: 10,000 FFI calls
    txn = db.transaction()
    count = sum(1 for _ in txn.scan(b"item:", b"item;"))
    txn.abort()
    print(f"Regular scan: {count} items")
    
    # Batched scan: 10 FFI calls (1000× fewer!)
    txn = db.transaction()
    count = sum(1 for _ in txn.scan_batched(
        start=b"item:",
        end=b"item;",
        batch_size=1000  # Fetch 1000 results per FFI call
    ))
    txn.abort()
    print(f"Batched scan: {count} items (much faster!)")
```

**Performance:**

| Dataset | Regular Scan | Batched Scan | Speedup |
|---------|--------------|--------------|---------|
| 10K items | 15ms | 2ms | 7.5× |
| 100K items | 150ms | 12ms | 12.5× |

---

### Statistics & Monitoring

```python
from sochdb import Database

with Database.open("./my_db") as db:
    # Perform operations
    for i in range(1000):
        db.put(f"key:{i}".encode(), f"value:{i}".encode())
    
    # Get runtime statistics
    stats = db.stats()
    
    print(f"Keys: {stats['keys_count']:,}")
    print(f"Bytes written: {stats['bytes_written']:,}")
    print(f"Bytes read: {stats['bytes_read']:,}")
    print(f"Transactions: {stats['transactions_committed']}")
    
    # Cache metrics
    hits = stats['cache_hits']
    misses = stats['cache_misses']
    hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0
    print(f"Cache hit rate: {hit_rate:.1f}%")
```

**Available Metrics:**
- `keys_count` - Total keys
- `bytes_written` - Cumulative writes
- `bytes_read` - Cumulative reads
- `transactions_committed` - Successful transactions
- `cache_hits` / `cache_misses` - Cache performance

---

### Manual Checkpoint

Force durability checkpoint to flush data to disk.

```python
from sochdb import Database

with Database.open("./my_db") as db:
    # Bulk import
    print("Importing 10K records...")
    with db.transaction() as txn:
        for i in range(10000):
            txn.put(f"bulk:{i}".encode(), f"data:{i}".encode())
    
    # Force checkpoint
    lsn = db.checkpoint()
    print(f"Checkpoint complete at LSN {lsn}")
    print("All data is durable on disk!")
```

**When to Use:**
- ✅ Before backups
- ✅ After bulk imports
- ✅ Before system shutdown
- ✅ Periodic durability (every 5 minutes)

---

### Python Plugins

Run Python code as database triggers.

```python
from sochdb.plugins import PythonPlugin, PluginRegistry, TriggerEvent, TriggerAbort

# Define validation plugin
plugin = PythonPlugin(
    name="user_validator",
    code='''
def on_before_insert(row: dict) -> dict:
    """Validate and transform data."""
    # Normalize email
    if "email" in row:
        row["email"] = row["email"].lower().strip()
    
    # Validate age
    if row.get("age", 0) < 0:
        raise TriggerAbort("Age cannot be negative", code="INVALID_AGE")
    
    # Add timestamp
    import time
    row["created_at"] = time.time()
    
    return row
''',
    triggers={"users": ["BEFORE INSERT"]}
)

# Register and use
registry = PluginRegistry()
registry.register(plugin)

# Fire trigger
row = {"name": "Alice", "email": "  ALICE@EXAMPLE.COM  ", "age": 30}
result = registry.fire("users", TriggerEvent.BEFORE_INSERT, row)
print(result["email"])  # "alice@example.com"
print(result["created_at"])  # 1704182400.0
```

**Available Events:**
- `BEFORE_INSERT`, `AFTER_INSERT`
- `BEFORE_UPDATE`, `AFTER_UPDATE`
- `BEFORE_DELETE`, `AFTER_DELETE`

---

### Transaction Advanced

```python
from sochdb import Database

with Database.open("./my_db") as db:
    # Get transaction ID
    txn = db.transaction()
    print(f"Transaction ID: {txn.id}")
    
    # Perform operations
    txn.put(b"key", b"value")
    
    # Commit returns LSN (Log Sequence Number)
    lsn = txn.commit()
    print(f"Committed at LSN: {lsn}")
    
    # Execute SQL within transaction
    txn2 = db.transaction()
    txn2.execute("INSERT INTO users VALUES (1, 'Alice')")
    txn2.put(b"user:1:metadata", b'{"verified": true}')
    txn2.commit()  # Atomic SQL + KV operation
```

---

## Error Handling

**New in v0.4.1** — Comprehensive error types for production applications.

### Error Hierarchy

```python
from sochdb.errors import (
    SochDBError,           # Base error
    DatabaseError,         # General database errors
    TransactionError,      # Transaction failures
    ConnectionError,       # Connection issues
    ProtocolError,         # Wire protocol errors
    
    # Namespace errors
    NamespaceNotFoundError,
    NamespaceExistsError,
    
    # Collection errors
    CollectionNotFoundError,
    CollectionExistsError,
    CollectionConfigError,
    
    # Validation errors
    ValidationError,
    DimensionMismatchError,
    
    # Lock errors (v0.4.1)
    LockError,
    DatabaseLockedError,
    LockTimeoutError,
    EpochMismatchError,
    SplitBrainError,
)
```

### Handling Lock Errors

```python
from sochdb import Database
from sochdb.errors import DatabaseLockedError, LockTimeoutError

try:
    db = Database.open("./shared_db")
except DatabaseLockedError as e:
    print(f"Database locked by another process: {e}")
    # Retry with concurrent mode
    db = Database.open_concurrent("./shared_db")
except LockTimeoutError as e:
    print(f"Timed out waiting for lock: {e}")
```

### Handling Namespace Errors

```python
from sochdb.errors import NamespaceNotFoundError, NamespaceExistsError

try:
    ns = db.namespace("tenant_999")
except NamespaceNotFoundError:
    # Create if not exists
    ns = db.create_namespace("tenant_999")

try:
    db.create_namespace("tenant_123")
except NamespaceExistsError:
    print("Namespace already exists")
```

### Handling Vector Dimension Errors

```python
from sochdb.errors import DimensionMismatchError

try:
    collection.insert([1.0, 2.0, 3.0])  # 3-dim
except DimensionMismatchError as e:
    print(f"Expected {e.expected} dimensions, got {e.actual}")
```

---

## Best Practices

### 1. Use SQL for Structured Data

```python
# ✅ Good: Use SQL for relational data
db.execute_sql("CREATE TABLE users (...)")
db.execute_sql("INSERT INTO users VALUES (...)")
results = db.execute_sql("SELECT * FROM users WHERE age > 25")
```

### 2. Use K-V for Unstructured Data

```python
# ✅ Good: Use K-V for documents, blobs, cache
db.put(b"cache:user:123", json.dumps(user).encode())
db.put(b"blob:image:456", image_bytes)
```

### 3. Use scan() for Multi-Tenancy

```python
# ✅ Good: Efficient tenant isolation
tenant_id = "acme"
prefix = f"tenants/{tenant_id}/".encode()
end = f"tenants/{tenant_id};".encode()
data = list(db.scan(prefix, end))
```

### 4. Use Transactions

```python
# ✅ Good: Atomic operations
with db.transaction() as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
```

### 5. Use Bulk API for Vectors

```python
# ✅ Good: Fast bulk operations
bulk_build_index(embeddings, "index.hnsw")

# ❌ Bad: Slow FFI loop
for vec in vectors:
    index.insert(vec)  # 12× slower!
```

### 6. Use Batched Scanning for Large Datasets

```python
# ✅ Good: Fast batched scan
txn = db.transaction()
for key, value in txn.scan_batched(b"prefix:", b"prefix;", batch_size=1000):
    process(key, value)
txn.abort()

# ❌ Bad: Slow regular scan for large datasets
for key, value in txn.scan(b"prefix:", b"prefix;"):
    process(key, value)  # 1000× more FFI calls!
```

### 7. Use TOON Format for LLM Context

```python
# ✅ Good: Token-efficient for LLMs
results = db.execute_sql("SELECT * FROM users LIMIT 100")
records = [dict(row) for row in results]
toon_context = Database.to_toon("users", records, ["name", "email"])
# Send to LLM - saves 40-66% tokens!

# ❌ Bad: Wasteful JSON for LLM context
json_context = json.dumps(records)  # Uses 2× more tokens
```

### 8. Always Use Context Managers

```python
# ✅ Good: Automatic cleanup
with Database.open("./db") as db:
    db.put(b"key", b"value")

# ❌ Bad: Manual cleanup required
db = Database.open("./db")
db.put(b"key", b"value")
db.close()
```

---

## Complete Examples

### Example 1: Multi-Tenant SaaS with SQL + K-V

```python
from sochdb import Database
import json

def main():
    with Database.open("./saas_db") as db:
        # SQL for tenant metadata
        db.execute_sql("""
            CREATE TABLE IF NOT EXISTS tenants (
                id INTEGER PRIMARY KEY,
                name TEXT,
                created_at TEXT
            )
        """)
        
        db.execute_sql("INSERT INTO tenants VALUES (1, 'ACME Corp', '2026-01-01')")
        db.execute_sql("INSERT INTO tenants VALUES (2, 'Globex Inc', '2026-01-01')")
        
        # K-V for tenant-specific data
        db.put(b"tenants/1/users/alice", b'{"role":"admin","email":"alice@acme.com"}')
        db.put(b"tenants/1/users/bob", b'{"role":"user","email":"bob@acme.com"}')
        db.put(b"tenants/2/users/charlie", b'{"role":"admin","email":"charlie@globex.com"}')
        
        # Query SQL
        tenants = db.execute_sql("SELECT * FROM tenants ORDER BY name")
        
        for tenant in tenants:
            tenant_id = tenant['id']
            tenant_name = tenant['name']
            
            # Scan tenant-specific K-V data
            prefix = f"tenants/{tenant_id}/".encode()
            end = f"tenants/{tenant_id};".encode()
            users = list(db.scan(prefix, end))
            
            print(f"\n{tenant_name} ({len(users)} users):")
            for key, value in users:
                user_data = json.loads(value.decode())
                print(f"  {key.decode()}: {user_data['email']} ({user_data['role']})")

if __name__ == "__main__":
    main()
```

**Output:**
```
ACME Corp (2 users):
  tenants/1/users/alice: alice@acme.com (admin)
  tenants/1/users/bob: bob@acme.com (user)

Globex Inc (1 users):
  tenants/2/users/charlie: charlie@globex.com (admin)
```

### Example 2: E-commerce with SQL

```python
from sochdb import Database

with Database.open("./ecommerce") as db:
    # Create schema
    db.execute_sql("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL,
            category TEXT
        )
    """)
    
    db.execute_sql("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER,
            total REAL
        )
    """)
    
    # Insert data
    db.execute_sql("INSERT INTO products VALUES (1, 'Laptop', 999.99, 'Electronics')")
    db.execute_sql("INSERT INTO products VALUES (2, 'Mouse', 25.00, 'Electronics')")
    db.execute_sql("INSERT INTO products VALUES (3, 'Desk', 299.99, 'Furniture')")
    
    db.execute_sql("INSERT INTO orders VALUES (1, 1, 2, 1999.98)")
    db.execute_sql("INSERT INTO orders VALUES (2, 2, 5, 125.00)")
    
    # Analytics query
    results = db.execute_sql("""
        SELECT 
            products.category,
            COUNT(orders.id) as order_count,
            SUM(orders.total) as revenue
        FROM products
        JOIN orders ON products.id = orders.product_id
        GROUP BY products.category
        ORDER BY revenue DESC
    """)
    
    print("Category Performance:")
    for row in results:
        print(f"{row['category']}: {row['order_count']} orders, ${row['revenue']:.2f}")
```

**Output:**
```
Category Performance:
Electronics: 2 orders, $2124.98
```

---

## API Reference

### Database (Embedded)

| Method | Description |
|--------|-------------|
| `Database.open(path)` | Open/create database |
| `put(key: bytes, value: bytes)` | Store key-value |
| `get(key: bytes) -> bytes \| None` | Retrieve value |
| `delete(key: bytes)` | Delete key |
| `put_path(path: str, value: bytes)` | Store by path |
| `get_path(path: str) -> bytes \| None` | Get by path |
| `delete_path(path: str)` | Delete by path |
| `scan(start: bytes, end: bytes)` | Iterate range |
| `scan_prefix(prefix: bytes)` | Scan keys matching prefix |
| `transaction()` | Begin transaction |
| `execute_sql(query: str)` | Execute SQL ⭐ |
| `execute(query: str)` | Alias for execute_sql() |
| `checkpoint() -> int` | Force checkpoint, returns LSN |
| `stats() -> dict` | Get runtime statistics |
| `to_toon(table, records, fields) -> str` | Convert to TOON format (static) |
| `from_toon(toon_str) -> tuple` | Parse TOON format (static) |

### Transaction

| Method | Description |
|--------|-------------|
| `id` | Transaction ID (property) |
| `put(key: bytes, value: bytes)` | Put within transaction |
| `get(key: bytes) -> bytes \| None` | Get with snapshot isolation |
| `delete(key: bytes)` | Delete within transaction |
| `scan(start: bytes, end: bytes)` | Scan within transaction |
| `scan_prefix(prefix: bytes)` | Scan keys matching prefix |
| `scan_batched(start, end, batch_size)` | High-performance batched scan |
| `execute(sql: str)` | Execute SQL within transaction |
| `commit() -> int` | Commit, returns LSN |
| `abort()` | Abort/rollback |

### Namespace

| Method | Description |
|--------|-------------|
| `db.create_namespace(config)` | Create new namespace |
| `db.namespace(name)` | Get existing namespace |
| `db.list_namespaces()` | List all namespaces |
| `db.delete_namespace(name)` | Delete namespace |
| `ns.create_collection(config)` | Create collection |
| `ns.collection(name)` | Get collection |
| `ns.list_collections()` | List collections |

### Collection

| Method | Description |
|--------|-------------|
| `insert(vector, metadata, id)` | Insert single vector |
| `insert_batch(vectors, metadatas, ids)` | Batch insert |
| `search(request)` | Unified search |
| `vector_search(vector, k)` | Vector similarity search |
| `keyword_search(text, k)` | BM25 keyword search |
| `hybrid_search(vector, text, k)` | RRF fusion search |
| `delete(id)` | Delete by ID |
| `count()` | Vector count |

### PriorityQueue

| Method | Description |
|--------|-------------|
| `PriorityQueue.from_database(db, name)` | Create from database |
| `PriorityQueue.from_client(client, name)` | Create from gRPC client |
| `enqueue(priority, payload, ...)` | Add task |
| `dequeue(worker_id)` | Claim task |
| `ack(task_id)` | Acknowledge completion |
| `nack(task_id)` | Negative ack (retry) |
| `stats()` | Queue statistics |
| `top_k(k)` | Get top K tasks |

### SochDBClient (gRPC)

| Method | Description |
|--------|-------------|
| `SochDBClient(address)` | Connect to server |
| `put_kv(namespace, key, value)` | Put key-value |
| `get_kv(namespace, key)` | Get value |
| `vector_search(namespace, collection, query, k)` | Vector search |
| `add_graph_node(...)` | Add graph node |
| `add_graph_edge(...)` | Add graph edge |
| `close()` | Close connection |

### IpcClient

| Method | Description |
|--------|-------------|
| `IpcClient.connect(path)` | Connect to server |
| `ping() -> float` | Check latency |
| `query(prefix: str)` | Create query builder |
| `scan(prefix: str)` | Scan prefix |

### VectorIndex

| Method | Description |
|--------|-------------|
| `VectorIndex(dimension, metric, m, ef_construction)` | Create index |
| `insert(id, vector)` | Insert single vector |
| `insert_batch(ids, vectors)` | Batch insert (~15K vec/s) |
| `search(vector, k, ef_search)` | Search k-NN |
| `delete(id)` | Delete by ID |
| `save(path)` | Save to disk |
| `load(path)` | Load from disk |

### Bulk API

| Function | Description |
|----------|-------------|
| `bulk_build_index(...)` | Build HNSW (~1,600 vec/s) |
| `bulk_query_index(...)` | Query k-NN |
| `bulk_info(index)` | Get index metadata |

---

## Configuration

```python
db = Database.open("./my_db", config={
    "create_if_missing": True,
    "wal_enabled": True,
    "sync_mode": "normal",  # "full", "normal", "off"
    "memtable_size_bytes": 64 * 1024 * 1024,
})
```

---

## Testing

```bash
# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_sql.py -v

# With coverage
pytest --cov=sochdb tests/
```

---

## Resources

- [Python SDK GitHub](https://github.com/sochdb/sochdb-python-sdk)
- [PyPI Package](https://pypi.org/project/sochdb/)
- [API Reference](../api-reference/python-api.md)
- [Go SDK](./go-sdk.md)
- [JavaScript SDK](./nodejs-sdk.md)
- [Rust SDK](./rust-sdk.md)

---

*Last updated: February 2026 (v0.5.3)*
