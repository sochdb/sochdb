<h1 align="center">
    <img src="https://github.com/sochdb/sochdb/raw/main/sochdbicon.png" alt="SochDB icon" width="72" height="72" />
    <br>
    SochDB
</h1>

## What is SochDB?

SochDB is a **single database** that replaces your vector DB + relational DB + prompt packer stack. Store structured data, embeddings, and conversation history together—then ask SochDB to assemble token-optimized context for your LLM.

## Comparison

### Database + retrieval layer

| Feature | SochDB | SQLite + vec | Postgres + pgvector | Chroma | LanceDB |
|---------|--------|--------|----------------------|--------|---------|
| Embedded | ✅ | ✅ | ❌ | ✅ | ✅ |
| Vector search | ✅ HNSW | ⚠️ (via extension) | ✅ (HNSW / IVFFlat) | ✅ | ✅ |
| Full SQL (user-facing) | ✅ SQL-92 | ✅ | ✅ | ❌ | ✅ |
| Hybrid search (vector + keyword) | ✅ | ⚠️ (DIY) | ⚠️ (DIY) | ⚠️ (limited) | ✅ |
| Context builder | ✅ | ❌ | ❌ | ❌ | ❌ |
| Token budgeting | ✅ | ❌ | ❌ | ❌ | ❌ |
| Graph overlay | ✅ | ❌ | ❌ | ❌ | ❌ |
| ACID transactions | ✅ | ✅ | ✅ | ⚠️ (limited) | ❌ |
| Columnar storage | ✅ | ❌ | ❌ | ❌ | ✅ |


### Memory / agent-memory layer

| Feature | SochDB | Mem0 | Letta | Graphiti |
|--------|--------|------|-------|----------|
| Primary focus | DB + retrieval + context | Memory layer | Agent framework + memory | Temporal knowledge-graph memory |
| Long-term memory primitives | ✅ | ✅ | ✅ | ✅ |
| Token-aware context budgeting | ✅ | ❌ | ❌ | ❌ |
| Graph-based memory | ✅ | ❌ | ❌ | ✅ |
| Built-in vector store | ✅ | ❌ (BYO) | ❌ (BYO) | ❌ (BYO) |
| Built-in agent runtime | ❌ | ❌ | ✅ | ❌ |
| Drop-in “memory add-on” to existing apps | ✅ | ✅ | ⚠️ | ✅ |


**Quick links:** [📚 Documentation](https://sochdb.dev) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [TOON Format](#-toon-format) • [Benchmarks](#-benchmarks) • [RFD](docs/rfds/RFD-001-ai-native-database.md)

---

## Why SochDB?

### ❌ The Typical AI Agent Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              YOUR APPLICATION                                    │
└───────┬─────────────────┬─────────────────┬─────────────────┬───────────────────┘
        │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────┐
│   Postgres    │ │   Pinecone    │ │    Redis      │ │    Custom Code            │
│   (metadata)  │ │   (vectors)   │ │  (sessions)   │ │    (context assembly)     │
│               │ │               │ │               │ │                           │
│ • User data   │ │ • Embeddings  │ │ • Chat state  │ │ • Token counting          │
│ • Settings    │ │ • Similarity  │ │ • Cache       │ │ • Truncation logic        │
│ • History     │ │   search      │ │ • Temp data   │ │ • Prompt packing          │
│               │ │               │ │               │ │ • Multi-source fusion     │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────────────────┘
        │                 │                 │                 │
        └─────────────────┴─────────────────┴─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  😰 You manage all of this:   │
                    │  • 4 different query languages │
                    │  • 4 sets of credentials       │
                    │  • 4 failure modes             │
                    │  • No cross-system transactions│
                    │  • Weeks of glue code          │
                    └───────────────────────────────┘
```

### ✅ With SochDB

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              YOUR APPLICATION                                    │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │             SochDB                   │
                    │                                      │
                    │   SQL + Vectors + Context Builder    │
                    │                                      │
                    │   • One query language               │
                    │   • One connection                   │
                    │   • ACID transactions                │
                    │   • Token budgeting built-in         │
                    │                                      │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  😎 What you actually ship:       │
                    │  • Single ~700KB embedded DB      │
                    │  • Zero external dependencies     │
                    │  • Works offline                  │
                    │  • Deploys anywhere               │
                    └───────────────────────────────────┘
```

### The Problem → Solution

| Challenge | Traditional Stack | SochDB |
|-----------|------------------|--------|
| **Token waste** | JSON/SQL bloat in prompts | TOON format for dense output |
| **RAG plumbing** | Separate vector DB + glue code | Built-in HNSW with hybrid search |
| **Context assembly** | Custom packer per use case | One query with token budget |
| **I/O overhead** | Multiple DB round-trips | Single columnar read |
| **Consistency** | Distributed transaction headaches | Local ACID guarantees |
| **Deployment** | Manage 4 services | Single binary, embed anywhere |

---

## Key Features

🧠 **Context Query Builder** — Assemble system + user + history + retrieval under a token budget  
🔍 **Hybrid Search** — HNSW vectors + BM25 keywords with reciprocal rank fusion  
🕸️ **Graph Overlay** — Lightweight relationship tracking for agent memory  
⚡ **Embedded-First** — ~700KB binary, no runtime dependencies, SQLite-style simplicity  
🔒 **Full ACID** — MVCC + WAL + Serializable Snapshot Isolation  
📊 **Columnar Storage** — Read only the columns you need  

---

## Why SochDB exists

Most "agent stacks" still glue together:

* a KV store (sessions / state)
* a vector DB (retrieval)
* a prompt packer (context budgeting, truncation)
* a relational DB (metadata)

…and then spend weeks maintaining brittle context assembly and token budgeting.

**SochDB collapses that stack into one LLM‑native substrate**: you store structured data + embeddings + history *and* ask the DB to produce a token‑efficient context payload.

---

## What you can rely on today

### ✅ LLM + agent primitives

- **TOON**: compact, model-friendly output for context windows
- **Graph Overlay**: lightweight agent-memory graph with BFS/DFS traversal and relationship tracking
- **ContextQuery builder**: token budgets, deduplication, and multi-source fusion
- **Policy hooks**: safety controls with pre-built policy templates and audit trails
- **Tool routing**: multi-agent coordination with dynamic discovery and load balancing
- **Hybrid retrieval**: vector + BM25 keyword with Reciprocal Rank Fusion (RRF)
- **Multi-vector documents**: chunk-level aggregation (max / mean / first)
- **Vector search (HNSW)**: integrated into retrieval workflows

### ✅ Database fundamentals

- **SQL (SQL-92)**: SELECT / INSERT / UPDATE / DELETE / JOINs
  - **AST-based query executor**: unified SQL processing with dialect normalization
  - **Multi-dialect compatibility**: MySQL, PostgreSQL, SQLite
  - **Idempotent DDL**: `CREATE TABLE IF NOT EXISTS`, `DROP TABLE IF EXISTS`
- **ACID transactions** with **MVCC**
- **WAL durability** + **group commit**
- **Serializable Snapshot Isolation (SSI)**
- **Columnar storage** with projection pushdown (read only the columns you need)
- **Sync-first architecture**: async runtime (tokio) is optional
  - ~500KB smaller binaries for embedded use cases
  - Follows SQLite-style design for maximum compatibility

### ✅ Developer experience

- **Rust client**: `sochdb`
- **Python & Nodejs & Golang SDK** with:
  - **Embedded mode (FFI)** for lowest latency
  - **IPC mode (Unix sockets)** for multi-process / service deployments
  - **Namespace isolation** for multi-tenant apps
  - **Typed error taxonomy** with remediation hints
- **Bulk vector operations** for high-throughput ingestion
  - **BatchAccumulator**: deferred graph construction — 4–5× faster inserts via zero-FFI numpy accumulation + single bulk Rayon-parallel HNSW build

### Known limits

- **Single-node only** (no replication / clustering yet)
---

## SochDB in one picture

| Problem           | Typical approach               | SochDB approach                     |
| ----------------- | ------------------------------ | ----------------------------------- |
| Token waste       | JSON/SQL payload bloat         | **TOON**: dense, table-like output  |
| RAG plumbing      | External vector DB + glue      | **Built-in HNSW** + quantization    |
| Context assembly  | multiple reads + custom packer | **One context query** with a budget |
| I/O amplification | row store reads all columns    | **columnar** + projection pushdown  |

---

## 📦 Quick Start

### Installation

Choose your preferred SDK:

```bash
# Rust - add to Cargo.toml
sochdb = "0.2"
```

### SDK Repositories

Language SDKs are maintained in separate repositories with their own release cycles:

| Language | Repository | Installation |
|----------|------------|-------------|
| **Python** | [sochdb-python-sdk](https://github.com/sochdb/sochdb-python-sdk) | `pip install sochdb` |

### 🐳 Docker Deployment

SochDB includes a production-ready Docker setup with gRPC server:

```bash
# Pull and run from Docker Hub
docker pull sushanth53/sochdb:latest
docker run -d -p 50051:50051 sushanth53/sochdb:latest

# Or use docker-compose
cd docker
docker compose up -d
```

**Docker Hub:** [`sushanth53/sochdb`](https://hub.docker.com/r/sushanth53/sochdb)

**Features:**
- ✅ Production-ready image (159MB)
- ✅ High availability setup with Traefik
- ✅ Prometheus + Grafana monitoring
- ✅ gRPC-Web support via Envoy
- ✅ Comprehensive test suite included

**Performance (tested on Apple M-series):**
- Single-threaded: ~2K ops/sec
- Concurrent (10 threads): ~10.5K ops/sec  
- Latency p99: <2.2ms

See [docker/README.md](docker/README.md) for full documentation.
| **Node.js/TypeScript** | [sochdb-nodejs-sdk](https://github.com/sochdb/sochdb-nodejs-sdk) | `npm install @sochdb/sochdb` |
| **Go** | [sochdb-go](https://github.com/sochdb/sochdb-go) | `go get github.com/sochdb/sochdb-go@latest` |
| **Rust** | This repository | `cargo add sochdb` |

### Examples

- **Python Examples**: [sochdb-python-examples](https://github.com/sochdb/sochdb-python-examples)
- **Node.js Examples**: [sochdb-nodejs-examples](https://github.com/sochdb/sochdb-nodejs-examples)
- **Go Examples**: [sochdb-golang-examples](https://github.com/sochdb/sochdb-golang-examples)

### Benchmarks

For performance comparisons and benchmarks, see [sochdb-benchmarks](https://github.com/sochdb/sochdb-benchmarks).

#### Vector Search — VectorDBBench (OpenAI/COHERE 50K × 1536D, Apple M1 Pro)

<p align="center">
  <img src="docs/assets/benchmark_comparison.svg" alt="SochDB vs ChromaDB vs LanceDB benchmark comparison" width="800" />
</p>

| Metric | SochDB | ChromaDB | LanceDB |
|--------|--------|----------|---------|
| Recall@100 | 0.9898 | 0.9967 | 0.9671 |
| Avg Latency | **3.2 ms** | 15.2 ms | 9.6 ms |
| P95 Latency | **4.2 ms** | 18.4 ms | 10.5 ms |
| P99 Latency | **4.9 ms** | 26.4 ms | 12.2 ms |
| Insert (50K vecs) | **5.1 s** | 64.7 s | 7.0 s |
| Total Load | **17.4 s** | 64.7 s | 30.2 s |

> SochDB HNSW config: m=16, ef_construction=200, ef_search=500.
> Insert uses the Python SDK's `BatchAccumulator` for deferred graph construction
> (zero FFI during accumulation, single bulk `insert_batch()` with Rayon parallelism).
> See [full benchmark details](#-benchmarks) for methodology and analysis.

#### Memory Agent — MemoryAgentBench (Ruler QA1 197K, Azure OpenAI gpt-4.1-mini)

##### Head-to-Head: SochDB vs RAG Competitors

<p align="center">
  <img src="docs/assets/head_to_head_benchmark.svg" alt="SochDB vs RAG competitors head-to-head benchmark" width="800" />
</p>

| Rank | System | EM% | F1% | Correct | Build | Query | Queries | Type |
|:---:|--------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 🥇 | **SochDB V2** | **60.0** | **61.7** | **12/20** | 1.9s | **2.1s** | 20/20 | Multi-Perspective RRF |
| 🥈 | SochDB + HyDE | 30.0 | 42.6 | 6/20 | 3.3s | 37.0s | 20/20 | Embedded Vector DB |
| 🥉 | GraphRAG | 25.0 | 40.6 | 5/20 | 16.2s | 11.9s | 20/20 | Knowledge Graph |
| 3 | SochDB + Rerank | 25.0 | 40.2 | 5/20 | 3.2s | 27.9s | 20/20 | Embedded Vector DB |
| 5 | SochDB + Advanced | 25.0 | 37.8 | 5/20 | 3.3s | 14.0s | 20/20 | Embedded Vector DB |
| 6 | SochDB Hybrid | 20.0 | 23.4 | 4/20 | **0.01s** | 0.8s | 20/20 | Embedded Vector DB |
| 7 | Self-RAG | 15.0 | 18.6 | 3/20 | 12.9s | 0.9s | 20/20 | Self-Reflection RAG |
| 8 | BM25 | 10.0 | 31.4 | 2/20 | 0.06s | 27.4s | 20/20 | Lexical Search |
| 9 | Embedding RAG | 5.0 | 18.9 | 1/20 | 0.3s | 37.8s | 20/20 | FAISS + Embedding |
| 10 | Mem0 | 5.0 | 18.5 | 1/20 | 51.7s | 1.0s | 20/20 | Memory-as-a-Service |

> **SochDB V2 is #1** — 60% EM, 2× the previous best (30%), 2.4× better than GraphRAG (25%). V2 solved 4 queries that NO other system could answer.
> **V2 innovations**: Multi-Perspective RRF (3 embedding angles fused) + Few-Shot Precision Extraction (7 calibrated examples).
> **GraphRAG** at 25% EM is limited by ContextualCompressionRetriever bottleneck (~848 tokens vs SochDB's ~80K).
> **Self-RAG** results impacted by Azure content filter rejecting self-reflection prompts.
> All systems use the same LLM (gpt-4.1-mini), dataset (Ruler QA1 197K), and evaluation framework ([MemoryAgentBench](https://arxiv.org/abs/2507.05257), UCSD).

##### SochDB Modes Detail

<p align="center">
  <img src="docs/assets/memory_agent_benchmark.svg" alt="SochDB search modes detail benchmark" width="800" />
</p>

| Metric | SochDB + HyDE | SochDB + Rerank | SochDB (baseline) | Mem0 |
|--------|:---:|:---:|:---:|:---:|
| Exact Match | **30.0%** | 25.0% | 20.0% | 5.0% |
| F1 Score | **42.6%** | 40.2% | 30.3% | 18.5% |
| Substring Match | 45.0% | **50.0%** | 20.0% | 30.0% |
| ROUGE-L F1 | **44.0%** | 42.9% | 29.5% | 17.9% |
| Memory Build | 3.3 s | 3.2 s | **0.01 s** | 51.7 s |
| Query Time | 37.0 s | 27.9 s | **6.6 s** | 1.0 s |
| **Best For** | 🎯 Max Accuracy | 🏆 **Recommended** | ⚡ Max Speed | — |

> **🏆 Recommended**: Use **Rerank** for best overall balance (highest substring match 50%, strong F1 40.2%, 27% faster than HyDE).
> Use **HyDE** when exact match matters most. Use **baseline** when latency is critical.
> See [full benchmark details](#-benchmarks) for developer configuration guide, substring match analysis, and learnings.

### Hello World

#### Python

```python
from sochdb import Database

db = Database.open("./my_db")
db.put(b"users/alice", b"Alice Smith")
print(db.get(b"users/alice").decode())  # "Alice Smith"
db.close()
```

#### Node.js / TypeScript

```typescript
import { SochDatabase } from '@sochdb/sochdb';

const db = new SochDatabase('./my_db');
await db.put('users/alice', 'Alice Smith');
console.log(await db.get('users/alice'));  // "Alice Smith"
await db.close();
```

#### Go

```go
package main

import (
    "fmt"
    sochdb "github.com/sochdb/sochdb-go"
)

func main() {
    db, _ := sochdb.Open("./my_db")
    defer db.Close()
    
    db.Put([]byte("users/alice"), []byte("Alice Smith"))
    value, _ := db.Get([]byte("users/alice"))
    fmt.Println(string(value))  // "Alice Smith"
}
```

#### Rust

```rust
use sochdb::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("./my_db")?;
    
    db.put(b"users/alice", b"Alice Smith")?;
    if let Some(value) = db.get(b"users/alice")? {
        println!("{}", String::from_utf8_lossy(&value));  // "Alice Smith"
    }
    Ok(())
}
```

### 🕸️ Graph Overlay for Agent Memory (v0.3.3)

Build lightweight graph structures on top of SochDB's KV storage for agent memory:

#### Python

```python
from sochdb import Database, GraphOverlay

db = Database.open("./my_db")
graph = GraphOverlay(db, namespace="agent_memory")

# Build conversation graph
graph.add_node("msg_1", {"role": "user", "content": "What's the weather?"})
graph.add_node("msg_2", {"role": "assistant", "content": "Let me check..."})
graph.add_node("msg_3", {"role": "tool", "content": "Sunny, 72°F"})
graph.add_node("msg_4", {"role": "assistant", "content": "It's sunny and 72°F"})

# Link causal relationships
graph.add_edge("msg_1", "msg_2", {"type": "triggers"})
graph.add_edge("msg_2", "msg_3", {"type": "invokes_tool"})
graph.add_edge("msg_3", "msg_4", {"type": "provides_context"})

# Traverse conversation history (BFS)
path = graph.bfs("msg_1", "msg_4")
print(f"Conversation flow: {' → '.join(path)}")

# Get all tool invocations (neighbors by edge type)
tools = graph.get_neighbors("msg_2", edge_filter={"type": "invokes_tool"})
print(f"Tools used: {tools}")

db.close()
```

#### Go

```go
package main

import (
    "fmt"
    sochdb "github.com/sochdb/sochdb-go"
)

func main() {
    db, _ := sochdb.Open("./my_db")
    defer db.Close()
    
    graph := sochdb.NewGraphOverlay(db, "agent_memory")
    
    // Build agent action graph
    graph.AddNode("action_1", map[string]interface{}{
        "type": "search", "query": "best restaurants",
    })
    graph.AddNode("action_2", map[string]interface{}{
        "type": "filter", "criteria": "italian",
    })
    
    graph.AddEdge("action_1", "action_2", map[string]interface{}{
        "relationship": "feeds_into",
    })
    
    // Find dependencies (DFS)
    deps := graph.DFS("action_1", 10)
    fmt.Printf("Action dependencies: %v\n", deps)
}
```

#### Node.js/TypeScript

```typescript
import { Database, GraphOverlay } from '@sochdb/sochdb';

const db = await Database.open('./my_db');
const graph = new GraphOverlay(db, 'agent_memory');

// Track entity relationships
await graph.addNode('entity_alice', { type: 'person', name: 'Alice' });
await graph.addNode('entity_acme', { type: 'company', name: 'Acme Corp' });
await graph.addNode('entity_project', { type: 'project', name: 'AI Initiative' });

await graph.addEdge('entity_alice', 'entity_acme', { relationship: 'works_at' });
await graph.addEdge('entity_alice', 'entity_project', { relationship: 'leads' });

// Find all entities Alice is connected to
const connections = await graph.getNeighbors('entity_alice');
console.log(`Alice is connected to: ${connections.length} entities`);

await db.close();
```

**Use Cases:**
- Agent conversation history with causal chains
- Entity relationship tracking across sessions
- Action dependency graphs for planning
- Knowledge graph construction

### Namespace Isolation (v0.3.0)

#### Python

```python
from sochdb import Database, CollectionConfig, DistanceMetric

db = Database.open("./my_db")

# Create namespace for tenant isolation
with db.use_namespace("tenant_acme") as ns:
    # Create vector collection with frozen config
    collection = ns.create_collection(
        CollectionConfig(
            name="documents",
            dimension=384,
            metric=DistanceMetric.COSINE,
            enable_hybrid_search=True,  # Enable keyword search
            content_field="text"
        )
    )
    
    # Insert multi-vector document (e.g., chunked document)
    collection.insert_multi(
        id="doc_123",
        vectors=[chunk_embedding_1, chunk_embedding_2, chunk_embedding_3],
        metadata={"title": "SochDB Guide", "author": "Alice"},
        chunk_texts=["Intro text", "Body text", "Conclusion"],
        aggregate="max"  # Use max score across chunks
    )
    
    # Hybrid search: vector + keyword with RRF fusion
    results = collection.hybrid_search(
        vector=query_embedding,
        text_query="database performance",
        k=10,
        alpha=0.7  # 70% vector, 30% keyword
    )

db.close()
```

### ContextQuery for LLM Retrieval (v0.3.0)

#### Python

```python
from sochdb import Database, ContextQuery, DeduplicationStrategy

db = Database.open("./my_db")
ns = db.namespace("tenant_acme")
collection = ns.collection("documents")

# Build context with token budgeting
context = (
    ContextQuery(collection)
    .add_vector_query(query_embedding, weight=0.7)
    .add_keyword_query("machine learning optimization", weight=0.3)
    .with_token_budget(4000)  # Fit within model context window
    .with_min_relevance(0.5)  # Filter low-quality results
    .with_deduplication(DeduplicationStrategy.EXACT)
    .execute()
)

# Use in LLM prompt
prompt = f"""Context:
{context.as_markdown()}

Question: {user_question}
"""

print(f"Retrieved {len(context)} chunks using {context.total_tokens} tokens")
db.close()
```

### Vector Search Example

#### Python

```python
from sochdb import VectorIndex
import numpy as np

# Create HNSW index
index = VectorIndex(
    path="./vectors",
    dimension=384,
    metric="cosine"
)

# Add vectors
embeddings = np.random.randn(1000, 384).astype(np.float32)
for i, embedding in enumerate(embeddings):
    index.add(str(i), embedding.tolist())

# Build the index
index.build()

# Search
query = np.random.randn(384).astype(np.float32)
results = index.search(query.tolist(), k=10)
print(results)  # [{'id': '1', 'distance': 0.23}, ...]
```

#### Node.js / TypeScript

```typescript
import { VectorIndex } from '@sochdb/sochdb';

// Instantiate VectorIndex with path and config
const index = new VectorIndex('./vectors', {
  dimension: 384,
  metric: 'cosine'
});

// Add vectors and build index
await index.add('doc1', embedding1);
await index.add('doc2', embedding2);
await index.build();

// Search
const results = await index.search(queryEmbedding, 10);
console.log(results);  // [{ id: 'doc1', distance: 0.23 }, ...]
```

### SDK Feature Matrix

| Feature | Python | Node.js | Go | Rust |
|---------|--------|---------|-----|------|
| Basic KV | ✅ | ✅ | ✅ | ✅ |
| Transactions | ✅ | ✅ | ✅ | ✅ |
| SQL Operations | ✅ | ✅ | ✅ | ✅ |
| Vector Search | ✅ | ✅ | ✅ | ✅ |
| Path API | ✅ | ✅ | ✅ | ✅ |
| Prefix Scanning | ✅ | ✅ | ✅ | ✅ |
| Query Builder | ✅ | ✅ | ✅ | ✅ |

> **Note:** While SDKs are maintained in separate repositories, they share the same core functionality and API design. Refer to individual SDK repositories for language-specific documentation and examples.

---

## 🏗 Architecture

```text
App / Agent Runtime
   │
   ├─ sochdb-client (Rust / Python)
   │
   ├─ sochdb-query   (planner + TOON encoder + context builder)
   └─ sochdb-kernel  (MVCC + WAL + catalog)
        ├─ sochdb-storage (columnar LSCS + mmap)
        └─ sochdb-index   (B-Tree + HNSW)
```

### Crate Overview

| Crate | Description | Key Components |
|-------|-------------|----------------|
| `sochdb-core` | Core types and TOON format | `SochValue`, `SochSchema`, `SochTable`, codec |
| `sochdb-kernel` | Database kernel | WAL, MVCC, transactions, catalog |
| `sochdb-storage` | Storage engine | LSCS columnar, mmap, block checksums |
| `sochdb-index` | Index structures | B-Tree, HNSW vector index |
| `sochdb-query` | Query execution | Cost optimizer, context builder, SOCH-QL |
| `sochdb-client` | Client SDK | `SochConnection`, `PathQuery`, `BatchWriter` |
| `sochdb-plugin-logging` | Logging plugin | Structured logging, tracing |

---

## 📄 TOON Format

TOON (Tabular Object-Oriented Notation) is SochDB's compact serialization format designed specifically for LLM context windows—a token-optimized format that dramatically reduces token consumption.

### Format Specification

```ebnf
document     ::= table_header newline row*
table_header ::= name "[" count "]" "{" fields "}" ":"
name         ::= identifier
count        ::= integer
fields       ::= field ("," field)*
field        ::= identifier
row          ::= value ("," value)* newline
value        ::= null | bool | number | string | array | ref
```

### Token Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                      JSON (156 tokens)                          │
├─────────────────────────────────────────────────────────────────┤
│ [                                                               │
│   {"id": 1, "name": "Alice", "email": "alice@example.com"},    │
│   {"id": 2, "name": "Bob", "email": "bob@example.com"},        │
│   {"id": 3, "name": "Charlie", "email": "charlie@example.com"} │
│ ]                                                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      TOON (52 tokens) — 67% reduction!          │
├─────────────────────────────────────────────────────────────────┤
│ users[3]{id,name,email}:                                        │
│ 1,Alice,alice@example.com                                       │
│ 2,Bob,bob@example.com                                           │
│ 3,Charlie,charlie@example.com                                   │
└─────────────────────────────────────────────────────────────────┘
```

### TOON Value Types

| Type | TOON Syntax | Example |
|------|-------------|---------|
| Null | `∅` | `∅` |
| Boolean | `T` / `F` | `T` |
| Integer | number | `42`, `-17` |
| Float | decimal | `3.14159` |
| String | text or `"quoted"` | `Alice`, `"hello, world"` |
| Array | `[items]` | `[1,2,3]` |
| Reference | `ref(table,id)` | `ref(users,42)` |
| Binary | `b64:data` | `b64:SGVsbG8=` |

---

## 🔍 Vector Search

SochDB includes an HNSW (Hierarchical Navigable Small World) index for similarity search.

### Configuration

```rust
use sochdb_index::{HNSWIndex, HNSWConfig, DistanceMetric};

// Create index with custom parameters
let config = HNSWConfig {
    m: 16,                          // Max connections per layer
    m_max: 32,                      // Max connections at layer 0
    ef_construction: 200,           // Build-time search width
    ef_search: 50,                  // Query-time search width
    metric: DistanceMetric::Cosine, // Or Euclidean, DotProduct
    ..Default::default()
};

let index = HNSWIndex::with_config(config);
```

### Vector Operations

```rust
use sochdb::{SochConnection, VectorCollection, SearchResult};

let conn = SochConnection::open("./vectors")?;

// Insert vectors
let embedding: Vec<f32> = get_embedding("Hello world");
conn.vector_insert("documents", 1, &embedding, Some(metadata))?;

// Search similar vectors
let query_embedding = get_embedding("Hi there");
let results: Vec<SearchResult> = conn.vector_search("documents", &query_embedding, 10)?;

for result in results {
    println!("ID: {}, Distance: {:.4}", result.id, result.distance);
}
```

### Distance Metrics

| Metric | Use Case | Formula |
|--------|----------|---------|
| `Cosine` | Text embeddings, normalized vectors | `1 - (a·b)/(‖a‖‖b‖)` |
| `Euclidean` | Spatial data, unnormalized | `√Σ(aᵢ-bᵢ)²` |
| `DotProduct` | When vectors are pre-normalized | `-a·b` |

### High-Throughput Ingestion (Python SDK)

The Python SDK's `BatchAccumulator` provides **4–5× faster inserts** by deferring HNSW graph construction:

```python
from sochdb import VectorIndex
import numpy as np

index = VectorIndex(dimension=1536, max_connections=16, ef_construction=200)

# Option 1: Context manager (auto-flush)
with index.batch_accumulator(estimated_size=50_000) as acc:
    acc.add(ids, vectors)   # Zero FFI, pure numpy memcpy
# HNSW graph built in one shot on exit

# Option 2: Explicit control
acc = index.batch_accumulator(50_000)
acc.add(chunk1_ids, chunk1_vecs)
acc.add(chunk2_ids, chunk2_vecs)
inserted = acc.flush()  # Single bulk FFI call → full Rayon parallelism

# Option 3: Cross-process persistence
acc.save("/tmp/vectors")     # Persist to disk
acc2 = index.batch_accumulator()
acc2.load("/tmp/vectors")    # Load in another process
acc2.flush()                 # Build HNSW
```

### Vector Quantization

SochDB supports optional quantization to reduce memory usage with minimal recall loss:

| Precision | Memory | Search Latency | Use Case |
|-----------|--------|----------------|----------|
| `F32` | 100% (baseline) | Baseline | Maximum precision |
| `F16` | 50% | ~Same | General embeddings |
| `BF16` | 50% | ~Same | ML model compatibility |

> **Tip**: F16 typically provides 50% memory reduction with <1% recall degradation for most embedding models.

---

## 🔐 Transactions

SochDB provides **ACID transactions** with MVCC (Multi-Version Concurrency Control) and WAL durability.

### ACID Guarantees

| Property | Implementation |
|----------|----------------|
| **Atomicity** | Buffered writes with all-or-nothing commit |
| **Consistency** | Schema validation before commit |
| **Isolation** | MVCC snapshots with read/write set tracking |
| **Durability** | WAL with fsync, group commit support |

### Transaction Modes

```rust
use sochdb::{SochConnection, ClientTransaction, IsolationLevel};

// Auto-commit (implicit transaction per operation)
conn.put("users/1/name", b"Alice")?;

// Explicit transaction with isolation level
let txn = conn.begin_with_isolation(IsolationLevel::Serializable)?;
conn.put_in_txn(txn, "users/1/name", b"Alice")?;
conn.put_in_txn(txn, "users/1/email", b"alice@example.com")?;
conn.commit(txn)?;  // SSI validation happens here

// Rollback on error
let txn = conn.begin()?;
if let Err(e) = do_something(&conn, txn) {
    conn.rollback(txn)?;
    return Err(e);
}
conn.commit(txn)?;
```

### Isolation Levels

| Level | Description | Status |
|-------|-------------|--------|
| `ReadCommitted` | Sees committed data at statement start | ✅ Implemented |
| `SnapshotIsolation` | Reads see consistent point-in-time view | ✅ Implemented |
| `Serializable` | SSI with rw-antidependency cycle detection | ✅ Implemented |

### WAL Sync Modes

```rust
use sochdb_kernel::SyncMode;

let config = DatabaseConfig {
    sync_mode: SyncMode::Normal,  // Group commit (recommended)
    // sync_mode: SyncMode::Full, // Fsync every commit (safest)
    // sync_mode: SyncMode::Off,  // Periodic fsync (fastest)
    ..Default::default()
};
```

### Durability Presets

SochDB provides pre-configured durability settings for common use cases:

| Preset | Sync Mode | Group Commit | Best For |
|--------|-----------|--------------|----------|
| `throughput_optimized()` | Normal | Large batches | High-volume ingestion |
| `latency_optimized()` | Full | Small batches | Real-time applications |
| `max_durability()` | Full | Disabled | Financial/critical data |

```rust
use sochdb::ConnectionConfig;

// High-throughput batch processing
let config = ConnectionConfig::throughput_optimized();

// Low-latency real-time access
let config = ConnectionConfig::latency_optimized();

// Maximum durability (fsync every commit, no batching)
let config = ConnectionConfig::max_durability();
```

---

## 🌳 Path API

SochDB's unique path-based API provides **O(|path|)** resolution via the Trie-Columnar Hybrid (TCH) structure.

### Path Format

```
collection/document_id/field
table/row_id/column
```

### Operations

```rust
use sochdb::{SochConnection, PathQuery};

let conn = SochConnection::open("./data")?;

// Put a value at a path
conn.put("users/1/name", b"Alice")?;
conn.put("users/1/profile/avatar", avatar_bytes)?;

// Get a value
let name = conn.get("users/1/name")?;

// Delete at path
conn.delete("users/1/profile/avatar")?;

// Scan by prefix (returns all matching key-value pairs)
let user_data = conn.scan("users/1/")?;
for (key, value) in user_data {
    println!("{}: {:?}", key, value);
}

// Query using PathQuery builder
let results = PathQuery::from_path(&conn, "users")
    .select(&["id", "name", "email"])
    .where_eq("status", "active")
    .order_by("created_at", Order::Desc)
    .limit(10)
    .execute()?;
```

### Path Resolution

```
Path: "users/1/name"
      
      TCH Resolution (O(3) = O(|path|))
      ┌─────────────────────────────────┐
      │  users  →  1  →  name           │
      │    ↓       ↓       ↓            │
      │  Table   Row   Column           │
      │  Lookup  Index  Access          │
      └─────────────────────────────────┘
      
vs    B-Tree (O(log N))
      ┌─────────────────────────────────┐
      │  Binary search through          │
      │  potentially millions of keys   │
      └─────────────────────────────────┘
```

### Optional Ordered Index

SochDB's ordered index can be disabled for write-optimized workloads:

```rust
use sochdb::ConnectionConfig;

// Default: ordered index enabled (O(log N) prefix scans)
let config = ConnectionConfig::default();

// Write-optimized: disable ordered index (~20% faster writes)
let mut config = ConnectionConfig::default();
config.enable_ordered_index = false;
// Note: scan_prefix becomes O(N) instead of O(log N + K)
```

| Mode | Write Speed | Prefix Scan | Use Case |
|------|-------------|-------------|----------|
| Ordered index **on** | Baseline | O(log N + K) | Read-heavy, prefix queries |
| Ordered index **off** | ~20% faster | O(N) | Write-heavy, point lookups |

---

## 📊 Context Query Builder

Build LLM context with automatic token budget management.

```rust
use sochdb_query::{ContextSection, ContextSelectQuery};
use sochdb::ContextQueryBuilder;

let context = ContextQueryBuilder::new()
    .for_session("session_123")
    .with_budget(4096)  // Token budget
    
    // System prompt (highest priority)
    .literal("SYSTEM", -1, "You are a helpful assistant")
    
    // User profile from database
    .section("USER", 0)
        .get("user.profile.{name, email, preferences}")
        .done()
    
    // Recent conversation history
    .section("HISTORY", 1)
        .last(10, "messages")
        .where_eq("session_id", session_id)
        .done()
    
    // Relevant documents via vector search
    .section("DOCS", 2)
        .search("knowledge_base", "query_embedding", 5)
        .min_score(0.7)
        .done()
    
    .truncation(TruncationStrategy::PriorityDrop)
    .format(ContextFormat::Soch)
    .execute()?;

println!("Tokens used: {}/{}", context.token_count, 4096);
println!("Context:\n{}", context.context);
```

---

## 🔌 Plugin System

SochDB uses a plugin architecture for extensibility without dependency bloat.

### Extension Types

| Extension | Purpose | Example |
|-----------|---------|---------|
| `StorageExtension` | Alternative backends | RocksDB, LSCS |
| `IndexExtension` | Custom indexes | Learned index, full-text |
| `ObservabilityExtension` | Metrics/tracing | Prometheus, DataDog |
| `CompressionExtension` | Compression algos | LZ4, Zstd |

### Implementing a Plugin

```rust
use sochdb_kernel::{Extension, ExtensionInfo, ObservabilityExtension};

struct PrometheusMetrics { /* ... */ }

impl Extension for PrometheusMetrics {
    fn info(&self) -> ExtensionInfo {
        ExtensionInfo {
            name: "prometheus-metrics".into(),
            version: "1.0.0".into(),
            description: "Prometheus metrics export".into(),
            author: "Your Name".into(),
            capabilities: vec![ExtensionCapability::Observability],
        }
    }
    
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl ObservabilityExtension for PrometheusMetrics {
    fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        // Push to Prometheus
    }
    
    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        // Set gauge value
    }
    
    fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        // Record histogram
    }
    
    // ... tracing methods
}

// Register the plugin
db.plugins().register_observability(Box::new(PrometheusMetrics::new()))?;
```

---

## 🧮 Batch Operations

High-throughput batch operations with group commit optimization.

```rust
use sochdb::{SochConnection, BatchWriter, GroupCommitConfig};

let conn = SochConnection::open("./data")?;

// Batch insert with auto-commit
let result = conn.batch()
    .max_batch_size(1000)
    .auto_commit(true)
    .insert("events", vec![("id", id1), ("data", data1)])
    .insert("events", vec![("id", id2), ("data", data2)])
    // ... more inserts
    .execute()?;

println!("Executed: {}, Failed: {}, Duration: {}ms", 
    result.ops_executed, result.ops_failed, result.duration_ms);

// Bulk insert for large datasets
let rows: Vec<Vec<(&str, SochValue)>> = generate_rows(10_000);
let result = conn.bulk_insert("events", rows)?;
```

### Group Commit Formula

SochDB calculates optimal batch size using:

```
N* = √(2 × L_fsync × λ / C_wait)

Where:
- L_fsync = fsync latency (~5ms typical)
- λ = arrival rate (ops/sec)
- C_wait = cost per unit wait time
```

---

## 📈 Benchmarks

> **Version**: 0.4.0+ | **Benchmark Date**: February 2026 | **Hardware**: Apple M1 Pro (ARM64)
> **Vector Search**: [VectorDBBench](https://github.com/zilliztech/VectorDBBench) (Zilliz) | **Memory Agent**: [MemoryAgentBench](https://arxiv.org/abs/2507.05257) (UCSD)

### VectorDBBench: 50K-Vector Comparison (SochDB vs ChromaDB vs LanceDB)

We benchmarked SochDB against ChromaDB and LanceDB using **[VectorDBBench](https://github.com/zilliztech/VectorDBBench)** — the industry-standard open-source benchmark from Zilliz. All databases ran on the same hardware with their recommended HNSW configurations.

<p align="center">
  <img src="docs/assets/benchmark_comparison.svg" alt="SochDB vs ChromaDB vs LanceDB benchmark comparison" width="800" />
</p>

#### Test Setup
- **Dataset**: COHERE/OpenAI 50,000 vectors × 768–1536 dimensions
- **Queries**: VectorDBBench standard query set (k=100)
- **Distance Metric**: Cosine similarity
- **Ground Truth**: VectorDBBench precomputed ground truth (brute-force)
- **Processes**: Insert, optimize, and search in separate subprocesses (VectorDBBench default)

#### Configuration

| Parameter | SochDB | ChromaDB | LanceDB |
|-----------|--------|----------|----------|
| Index Type | HNSW | HNSW | IVF_HNSW_SQ |
| M | 16 | 16 | — |
| ef_construction | 200 | 200 | — |
| ef_search | 500 | 500 | — |
| Version | 0.4.9 (SDK) | 1.4.1 | 0.19.0 |

#### Results

| Metric | SochDB | ChromaDB | LanceDB |
|--------|--------|----------|----------|
| **Recall@100** | 0.9898 | 0.9967 | 0.9671 |
| **Avg Latency** | **3.2 ms** ✅ | 15.2 ms | 9.6 ms |
| **P95 Latency** | **4.2 ms** ✅ | 18.4 ms | 10.5 ms |
| **P99 Latency** | **4.9 ms** ✅ | 26.4 ms | 12.2 ms |
| **Insert (50K vecs)** | **5.1 s** ✅ | 64.7 s | 7.0 s |
| **Total Load** | **17.4 s** ✅ | 64.7 s | 30.2 s |

#### Key Findings

- 🏎️ **SochDB search is 4.75× faster than ChromaDB** (3.2 ms vs 15.2 ms average)
- 🏎️ **SochDB search is 3× faster than LanceDB** (3.2 ms vs 9.6 ms average)
- ⚡ **SochDB insert is 12.7× faster than ChromaDB** (5.1 s vs 64.7 s for 50K vectors)
- ⚡ **SochDB insert is 1.4× faster than LanceDB** (5.1 s vs 7.0 s)
- 📐 **SochDB total load is 1.7× faster than LanceDB, 3.7× faster than ChromaDB**
- 🎯 **SochDB recall (98.98%)** is within 1% of ChromaDB while being 4.75× faster

#### How SochDB Achieves Fast Inserts: `BatchAccumulator`

SochDB's Python SDK includes a **`BatchAccumulator`** API that separates data accumulation from HNSW graph construction:

```
┌────────────────────────────────────────────────────────────────────┐
│                    BatchAccumulator Pipeline                       │
├───────────────────────┬──────────────────────────────────────────┤
│  Phase 1: Accumulate  │  Phase 2: Flush                          │
│  ──────────────────── │  ─────────────────                        │
│  • add(ids, vecs)     │  • Single insert_batch() FFI call        │
│  • Pure numpy memcpy  │  • Full Rayon parallel HNSW build        │
│  • Zero FFI calls     │  • Wave-parallel (32-node waves)         │
│  • ~0.1 s for 50K     │  • Adaptive ef (capped at 48)            │
│                       │  • ~13 s for 50K vectors                 │
└───────────────────────┴──────────────────────────────────────────┘
```

```python
from sochdb import VectorIndex

index = VectorIndex(dimension=1536, max_connections=16, ef_construction=200)

# Deferred insert: zero FFI, pure numpy memcpy
with index.batch_accumulator(estimated_size=50_000) as acc:
    for batch_ids, batch_vecs in data_loader:
        acc.add(batch_ids, batch_vecs)       # ~0.1s total
    # flush() called automatically → single bulk HNSW build (~13s)
```

#### End-to-End RAG Bottleneck Analysis

| Component | Time | % of Total |
|-----------|------|------------|
| **Embedding API (Azure OpenAI)** | 59.5s | **99.7%** |
| SochDB Insert (1K vectors) | 0.133s | 0.2% |
| SochDB Search (100 queries) | 0.046s | 0.1% |

> 🎯 **The embedding API is 333× slower than SochDB operations.** In production RAG systems, the database is never the bottleneck — your LLM API calls are.

---

### MemoryAgentBench: Head-to-Head RAG Comparison

> **Version**: 0.4.9 | **Benchmark Date**: February 2026 | **LLM**: Azure OpenAI gpt-4.1-mini | **Framework**: [MemoryAgentBench](https://arxiv.org/abs/2507.05257) (UCSD)

We evaluated SochDB head-to-head against **7 RAG competitors** using **MemoryAgentBench** — an academic benchmark from UCSD that tests how well memory systems help LLMs retrieve facts from multi-turn conversations over long contexts (up to 197K+ tokens).

<p align="center">
  <img src="docs/assets/head_to_head_benchmark.svg" alt="SochDB vs RAG competitors head-to-head benchmark" width="800" />
</p>

#### Head-to-Head Results (gpt-4.1-mini, Ruler QA1 197K, 20 queries)

| Rank | System | EM % | F1 % | Correct | Build (s) | Query (s) | Queries | Type |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 🥇 | **SochDB V2** | **60.0** | **61.7** | **12/20** | 1.9 | **2.1** | ✅ 20/20 | Multi-Perspective RRF |
| 🥈 | SochDB + HyDE | 30.0 | 42.6 | 6/20 | 3.3 | 37.0 | ✅ 20/20 | Embedded HNSW |
| 🥉 | GraphRAG | 25.0 | 40.6 | 5/20 | 16.2 | 11.9 | ✅ 20/20 | Knowledge Graph + NER |
| 3 | SochDB + Rerank | 25.0 | 40.2 | 5/20 | 3.2 | 27.9 | ✅ 20/20 | Embedded HNSW |
| 5 | SochDB + Advanced | 25.0 | 37.8 | 5/20 | 3.3 | 14.0 | ✅ 20/20 | Embedded HNSW |
| 6 | SochDB Hybrid | 20.0 | 23.4 | 4/20 | **0.01** | 0.8 | ✅ 20/20 | Embedded HNSW |
| 7 | Self-RAG | 15.0 | 18.6 | 3/20 | 12.9 | 0.9 | ✅ 20/20 | Adaptive Retrieval |
| 8 | BM25 | 10.0 | 31.4 | 2/20 | 0.06 | 27.4 | ✅ 20/20 | Lexical Search |
| 9 | Embedding RAG | 5.0 | 18.9 | 1/20 | 0.3 | 37.8 | ✅ 20/20 | FAISS + Embedding |
| 10 | Mem0 | 5.0 | 18.5 | 1/20 | 51.7 | 1.0 | ✅ 20/20 | Memory-as-a-Service |
| — | RAPTOR | — | — | — | — | — | 0/20 | Tree Summarization |

> All systems completed 20/20 queries except RAPTOR.
> **SochDB V2** solved 4 queries that NO other system could: Q2 (Denmark, Iceland and Norway), Q7 (Catholic), Q11 (King Charles III), Q12 (Epte).
> **Self-RAG** results impacted by Azure content filter rejecting self-reflection prompts (~50% of queries blocked).

#### Key Findings — Head-to-Head

- 🏆 **SochDB V2 dominates at 60% EM** — 2× the previous best (30%), 2.4× better than GraphRAG (25%)
- 🏆 **V2 solves 4 previously-impossible queries** via Multi-Perspective RRF (3 embedding angles) + Few-Shot Precision Extraction
- 🏆 **SochDB is the only embedded system** — zero external dependencies (no LangChain, spaCy, FAISS, or network services)
- 🏆 **V2 query time is 18× faster** than HyDE v1 (2.1s vs 37.0s) and 6× faster than GraphRAG (11.9s)
- 📊 **GraphRAG is limited by ContextualCompressionRetriever** — reduces context to ~848 tokens (vs SochDB's ~80K)
- ⚡ **SochDB Hybrid is 40× faster than any competitor** (0.8s query) while still competitive at 20% EM
- 🧩 **BM25 has surprisingly high substring match (70%)** but low exact match (10%) — retrieves relevant docs but can't extract precise answers
- 📉 **Embedding RAG and Mem0 tied at 5% EM** — basic vector similarity alone is insufficient for long-context QA

#### Test Setup
- **Dataset**: Ruler QA1 197K (197,000-token context, 100 QA pairs, key-value retrieval)
- **Embeddings**: Azure OpenAI text-embedding-3-small (1536D)
- **LLM**: gpt-4.1-mini (all systems use the same LLM for fair comparison)
- **Competitors tested**: GraphRAG, Self-RAG, BM25, Embedding RAG (FAISS), Mem0, RAPTOR
- **Task**: Accurate Retrieval — memorize long conversations, then answer factual queries
- **Queries**: 20 per system (max_test_queries_ablation=20)
- **Metrics**: Exact match, F1, substring match, ROUGE-L (standard MemoryAgentBench metrics)

#### SochDB Configuration Comparison

| Configuration | EM % | F1 % | Sub-EM % | ROUGE-L | Build (s) | Query (s) | Best For |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **SochDB + HyDE** | **30.0** | **42.6** | 45.0 | **44.0** | 3.29 | 37.0 | 🎯 Max Accuracy |
| **SochDB + Rerank** | 25.0 | 40.2 | **50.0** | 42.9 | 3.24 | 27.9 | 🏆 **Recommended** |
| **SochDB + Advanced** | 25.0 | 37.8 | 35.0 | 37.5 | 3.25 | 14.0 | ⚠️ Don't stack |
| **SochDB (gpt-4.1)** | 20.0 | 30.3 | 20.0 | 29.5 | **0.01** | **6.6** | ⚡ Max Speed |
| **Mem0** | 5.0 | 18.5 | 30.0 | 17.9 | 51.7 | 1.0 | — |

#### 📘 Developer Configuration Guide

> **TL;DR**: Use **Rerank** for most use cases. Use **HyDE** when exact match is critical. Use **baseline** for real-time. **Never stack all features together** — it's slower and less accurate.

| Your Priority | Recommended Config | Why |
|:---|:---|:---|
| **Best overall balance** | SochDB + Rerank | Highest substring match (50%), strong F1 (40.2%), 27% faster than HyDE |
| **Maximum exact accuracy** | SochDB + HyDE | Best EM (30%) and F1 (42.6%) — HyDE bridges question↔document vocabulary gap |
| **Lowest latency / real-time** | SochDB baseline | 0.01s build, 6.6s query — no extra LLM calls during retrieval |
| **Fuzzy/partial matching** | SochDB + Rerank | 50% substring match — cross-encoder reranker surfaces relevant context even on partial matches |

**Key decision factors**:

1. **Retrieval strategy matters more than model size.** Upgrading from gpt-4.1-mini → gpt-4.1 (full) gave identical 20% EM. But adding HyDE to gpt-4.1-mini boosted EM to 30% (+50% improvement). Invest in retrieval, not bigger LLMs.

2. **Don't stack features.** The Advanced config (HyDE + Hybrid + Rerank combined) scored *worse* than HyDE or Rerank alone (25% EM, 35% Sub-EM). Each feature adds its own noise — pick the one that matches your use case.

3. **Rerank is the best all-rounder.** It's 27% faster than HyDE (27.9s vs 37.0s query time), has the highest substring match (50%), near-equivalent F1 (40.2% vs 42.6%), and only 5pp behind HyDE on exact match.

#### Understanding Substring Match (Sub-EM)

**What it measures**: Does the gold answer appear *anywhere* inside the prediction, or vice versa?

Substring match is not a pure accuracy metric — it **correlates with answer verbosity**. Here's why different configs score differently:

| Config | EM % | Sub-EM % | Gap | Explanation |
|:---|:---:|:---:|:---:|:---|
| **SochDB (baseline)** | 20 | 20 | 0 | Short precise answers (~4 tokens). When wrong, no overlap at all. |
| **SochDB + HyDE** | 30 | 45 | +15 | Slightly longer answers. Wrong predictions still contain gold keywords. |
| **SochDB + Rerank** | 25 | 50 | +25 | Reranker surfaces better context → predictions contain gold as substring. |
| **Mem0** | 5 | 30 | +25 | Very verbose answers (~13 tokens). Gold words appear by chance in long text. |

**Example**: Gold answer is "Catholic"
- Baseline predicts "Orthodox" → Sub-EM ❌ (short, no overlap)
- Rerank predicts "Catholic orthodoxy" → Sub-EM ✅ (gold is a substring)
- Mem0 predicts "The predominant religion was Catholic Christianity" → Sub-EM ✅ (verbose, gold appears)

> ⚠️ **For developers**: High Sub-EM with low EM (like Mem0: 5% EM / 30% Sub-EM) means the system is *vaguely right but imprecise*. High EM with proportional Sub-EM (like HyDE: 30% EM / 45% Sub-EM) means the system gives **useful answers**. Rerank's 50% Sub-EM with 25% EM is the sweet spot — it frequently gets the right entity even if not the exact formatting.

#### Key Findings

- 🏆 **SochDB + HyDE achieves 6× higher exact match than Mem0** (30.0% vs 5.0%)
- 🏆 **SochDB + Rerank is the recommended config** — best substring match (50%) with strong F1 (40.2%) and 27% faster than HyDE
- ⚡ **SochDB builds memory 5,170× faster than Mem0** (0.01s vs 51.7s)
- 🧠 **Retrieval strategy > model size**: gpt-4.1 = gpt-4.1-mini at same retrieval (both 20% EM), but HyDE on mini → 30% EM
- ⚠️ **Don't stack all features**: Advanced (HyDE+Hybrid+Rerank) scores *worse* than using HyDE or Rerank independently
- 📊 **Substring match tracks answer verbosity**, not pure accuracy — use EM and F1 for quality decisions

#### Multi-Dataset Results (SochDB, 100 queries)

| Dataset | Context | EM % | F1 % | Sub-EM % |
|---------|---------|:---:|:---:|:---:|
| Ruler QA1 197K | 197K tokens | 13.0 | 27.0 | 38.0 |
| Ruler QA2 421K | 421K tokens | **31.0** | **42.7** | **49.0** |
| LongMemEval | 400K tokens | 3.3 | 9.7 | 4.0 |

> Note: Multi-dataset runs used gpt-4o-mini/gpt-4.1-mini with k=100. QA2 achieved higher accuracy than QA1 due to more distinctive key-value patterns.

#### Why SochDB is Different

1. **No External Dependencies**: SochDB is the only system in this benchmark that requires **zero Python packages** for its core operation. GraphRAG needs LangChain + spaCy + FAISS + NER. Self-RAG needs custom retrieval chains. Even BM25 needs a ranking library. SochDB runs as an embedded Rust library via FFI.

2. **Reliable Under Pressure**: SochDB completed 100% of queries (20/20) on every configuration. GraphRAG failed 50% due to API rate limits during NER extraction. RAPTOR couldn't even start. Self-RAG was blocked by content filters.

3. **Memory Build Speed**: SochDB stores embeddings directly in its HNSW index — no LLM calls during memorization. GraphRAG needs LLM NER calls per document chunk (10× slower). Mem0 processes each memory through an extraction pipeline (5,170× slower).

4. **Retrieval Quality**: SochDB's HyDE generates a synthetic answer before searching, bridging the question-document gap. This single technique (zero external dependencies) achieves 75% of GraphRAG's accuracy with 10× faster build and 100% completion rate.

5. **Honest Assessment**: GraphRAG's knowledge graph approach is genuinely more accurate on the queries it completes. For applications where reliability and deployment simplicity matter more than peak accuracy, SochDB is the better choice. For research workloads with high API quotas, GraphRAG is worth considering.

---

### Recall Benchmarks (Search Quality)

SochDB's HNSW index achieves **>98% recall@10** with sub-millisecond latency using real Azure OpenAI embeddings.

#### Test Methodology
- Ground truth computed via brute-force cosine similarity
- Recall@k = (# correct results in top-k) / k
- Tested across multiple HNSW configurations

#### Results by HNSW Configuration

| Configuration | Search (ms) | R@1 | R@5 | R@10 | R@20 | R@50 |
|---------------|-------------|-----|-----|------|------|------|
| **M=8, ef_c=50** | **0.42** | 0.990 | **0.994** | **0.991** | 0.994 | 0.991 |
| M=16, ef_c=100 | 0.47 | 0.980 | 0.986 | 0.982 | 0.984 | 0.986 |
| M=16, ef_c=200 | 0.44 | 0.970 | 0.984 | 0.988 | 0.990 | 0.986 |
| M=32, ef_c=200 | 0.47 | 0.980 | 0.982 | 0.981 | 0.984 | 0.985 |
| M=32, ef_c=400 | 0.52 | 0.990 | 0.986 | 0.983 | 0.979 | 0.981 |

**Key Insights**:
- All configurations achieve **>98% recall@10** with real embeddings
- **Best recall**: 99.1% @ 0.42ms (M=8, ef_c=50)
- **Recommended for RAG**: M=16, ef_c=100 (balanced speed + quality)
- Smaller `M` values work well for text embeddings due to natural clustering

#### Recommended HNSW Settings

| Use Case | M | ef_construction | Expected Recall@10 | Latency |
|----------|---|-----------------|-------------------|---------|
| **Real-time RAG** | 8 | 50 | ~99% | <0.5ms |
| **Balanced** | 16 | 100 | ~98% | <0.5ms |
| **Maximum Quality** | 16 | 200 | ~99% | <0.5ms |
| **Large-scale (10M+)** | 32 | 200 | ~97% | <1ms |

---

### Token Efficiency (TOON vs JSON)

| Dataset | JSON Tokens | TOON Tokens | Reduction |
|---------|-------------|-------------|-----------|
| Users (100 rows, 5 cols) | 2,340 | 782 | **66.6%** |
| Events (1000 rows, 3 cols) | 18,200 | 7,650 | **58.0%** |
| Products (500 rows, 8 cols) | 15,600 | 5,980 | **61.7%** |

---

### I/O Reduction (Columnar Storage)

| Query | Row Store | SochDB Columnar | Reduction |
|-------|-----------|-----------------|-----------| 
| SELECT 2 of 10 cols | 100% | 20% | **80%** |
| SELECT 1 of 20 cols | 100% | 5% | **95%** |

---

### KV Performance (vs SQLite)

> **Methodology**: SochDB vs SQLite under similar durability settings (`WAL` mode, `synchronous=NORMAL`). Results on Apple M-series hardware, 100k records.

| Database | Mode | Insert Rate | Notes |
|----------|------|-------------|-------|
| **SQLite** | File (WAL) | ~1.16M ops/sec | Industry standard |
| **SochDB** | Embedded (WAL) | ~760k ops/sec | Group commit disabled |
| **SochDB** | put_raw | ~1.30M ops/sec | Direct storage layer |
| **SochDB** | insert_row_slice | ~1.29M ops/sec | Zero-allocation API |

---

### Running Benchmarks Yourself

```bash
# Install Python 3.12 (recommended for ChromaDB compatibility)
brew install python@3.12
python3.12 -m venv .venv312
source .venv312/bin/activate

# Install dependencies
pip install chromadb lancedb python-dotenv requests numpy
pip install -e sochdb-python-sdk/

# Build SochDB release library
cargo build --release

# Run real embedding benchmark (requires Azure OpenAI credentials in .env)
SOCHDB_LIB_PATH=target/release python3 benchmarks/real_embedding_benchmark.py

# Run recall benchmark
SOCHDB_LIB_PATH=target/release python3 benchmarks/recall_benchmark.py

# Run Rust benchmarks (SochDB vs SQLite)
cargo run -p benchmarks --release
```

> **Note**: Performance varies by workload. SochDB excels in LLM context assembly scenarios (token-efficient output, vector search, context budget management). SQLite remains the gold standard for general-purpose relational workloads.

---

## 🛠 Configuration Reference

### DatabaseConfig

```rust
pub struct DatabaseConfig {
    /// Enable group commit for better throughput
    pub group_commit: bool,           // default: true
    
    /// WAL sync mode
    pub sync_mode: SyncMode,          // default: Normal
    
    /// Maximum WAL size before checkpoint
    pub max_wal_size: u64,            // default: 64MB
    
    /// Memtable size before flush
    pub memtable_size: usize,         // default: 4MB
    
    /// Block cache size
    pub block_cache_size: usize,      // default: 64MB
    
    /// Compression algorithm
    pub compression: Compression,      // default: LZ4
}
```

### HNSWConfig

```rust
pub struct HNSWConfig {
    /// Max connections per node per layer
    pub m: usize,                     // default: 16
    
    /// Max connections at layer 0
    pub m_max: usize,                 // default: 32
    
    /// Construction-time search width
    pub ef_construction: usize,       // default: 200
    
    /// Query-time search width (adjustable)
    pub ef_search: usize,             // default: 50
    
    /// Distance metric
    pub metric: DistanceMetric,       // default: Cosine
    
    /// Level multiplier (mL = 1/ln(M))
    pub ml: f32,                      // default: calculated
}
```

---

## 📚 API Reference

### SochConnection

| Method | Description | Returns |
|--------|-------------|---------|
| `open(path)` | Open/create database | `Result<SochConnection>` |
| `create_table(schema)` | Create a new table | `Result<CreateResult>` |
| `drop_table(name)` | Drop a table | `Result<DropResult>` |
| `batch()` | Start a batch writer | `BatchWriter` |
| `put(path, value)` | Put value at path | `Result<()>` |
| `get(path)` | Get value at path | `Result<Option<Vec<u8>>>` |
| `delete(path)` | Delete at path | `Result<()>` |
| `scan(prefix)` | Scan path prefix | `Result<Vec<(String, Vec<u8>)>>` |
| `begin()` | Begin transaction | `Result<TxnHandle>` |
| `commit(txn)` | Commit transaction | `Result<()>` |
| `rollback(txn)` | Rollback transaction | `Result<()>` |
| `vector_insert(...)` | Insert vector | `Result<()>` |
| `vector_search(...)` | Search similar vectors | `Result<Vec<SearchResult>>` |
| `fsync()` | Force sync to disk | `Result<()>` |
| `checkpoint()` | Create checkpoint | `Result<u64>` |
| `stats()` | Get statistics | `ClientStats` |

### PathQuery

| Method | Description | Returns |
|--------|-------------|---------|
| `from_path(conn, path)` | Create query from path | `PathQuery` |
| `select(cols)` | Select columns | `Self` |
| `project(cols)` | Alias for select | `Self` |
| `where_eq(field, val)` | Equality filter | `Self` |
| `where_gt(field, val)` | Greater than filter | `Self` |
| `where_like(field, pat)` | Pattern match | `Self` |
| `order_by(field, dir)` | Sort results | `Self` |
| `limit(n)` | Limit results | `Self` |
| `offset(n)` | Skip results | `Self` |
| `execute()` | Execute query | `Result<QueryResult>` |
| `execute_toon()` | Execute and return TOON | `Result<String>` |

### SochValue

| Variant | Rust Type | Description |
|---------|-----------|-------------|
| `Null` | — | Null value |
| `Bool(bool)` | `bool` | Boolean |
| `Int(i64)` | `i64` | Signed integer |
| `UInt(u64)` | `u64` | Unsigned integer |
| `Float(f64)` | `f64` | 64-bit float |
| `Text(String)` | `String` | UTF-8 string |
| `Binary(Vec<u8>)` | `Vec<u8>` | Binary data |
| `Array(Vec<SochValue>)` | `Vec<SochValue>` | Array of values |
| `Object(HashMap<String, SochValue>)` | `HashMap` | Key-value object |
| `Ref { table, id }` | — | Foreign key reference |

### SochType

| Type | Description |
|------|-------------|
| `Int` | 64-bit signed integer |
| `UInt` | 64-bit unsigned integer |
| `Float` | 64-bit float |
| `Text` | UTF-8 string |
| `Bool` | Boolean |
| `Bytes` | Binary data |
| `Vector(dim)` | Float vector with dimension |
| `Array(inner)` | Array of inner type |
| `Optional(inner)` | Nullable type |
| `Ref(table)` | Foreign key to table |

---

## 🔧 Building from Source

### Prerequisites

- Rust 2024 edition (1.75+)
- Clang/LLVM (for SIMD optimizations)

### Build

```bash
# Clone the repository
git clone https://github.com/sochdb/sochdb.git
cd sochdb

# Build all crates
cargo build --release

# Run tests
cargo test --all

# Run benchmarks
cargo bench
```

### Feature Flags

| Feature | Crate | Description |
|---------|-------|-------------|
| `simd` | sochdb-client | SIMD optimizations for column access |
| `embedded` | sochdb-client | Use kernel directly (no IPC) |
| `full` | sochdb-kernel | All kernel features |

---

## ⚠️ Before heavy production use

* **Single node** (no replication / clustering)
* **WAL growth**: call `checkpoint()` periodically for long-running services
* **Group commit**: tune per workload (disable for strictly sequential writes)

---

## 🚧 Roadmap (high level)

* Cost-based optimizer: experimental
* Agent flow metadata schema: planned
* Agent runtime library: planned
* Adaptive group commit: planned
* WAL compaction / auto-truncation: planned

---

## 🤖 Vision: SochDB as an Agentic Framework Foundation

SochDB is designed to be the **brain, memory, and registry** for AI agents—not by embedding a programming language, but by storing agent metadata that external runtimes interpret.

### The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Agent Runtime│    │    SochDB    │    │     LLM      │   │
│  │  (executor)  │◄──►│  (metadata)  │    │   (worker)   │   │
│  └──────┬───────┘    └──────────────┘    └──────▲───────┘   │
│         │                                        │           │
│         │  1. Load flow from DB                  │           │
│         │  2. Build prompt from node config      │           │
│         │  3. Call LLM ─────────────────────────►│           │
│         │  4. Parse result, update state         │           │
│         │  5. Choose next edge, repeat           │           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### What SochDB Stores

| Table | Purpose |
|-------|---------|
| `agent_flows` | Flow definitions: name, entry node, version |
| `agent_nodes` | Nodes: LLM steps, tool calls, decisions, loops, reflections |
| `agent_edges` | Edges with conditions for routing |
| `agent_sessions` | Runtime state per user/conversation |
| `agent_reflections` | Feedback and learning data |

### Node Types

Flows are graphs where each node has a `kind`:

- **`llm_step`** — Call the LLM with a prompt template
- **`tool_call`** — Execute a tool (API, function, DB query)
- **`decision`** — Branch based on previous output
- **`loop_start` / `loop_end`** — Iteration with exit conditions
- **`reflection`** — Ask LLM to evaluate and improve
- **`subflow`** — Invoke another flow

### Example: Support Agent Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Classify   │────►│  Retrieve   │────►│   Answer    │
│   Intent    │     │   Context   │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐            │
                    │   Reflect   │◄───────────┘
                    │  (optional) │
                    └─────────────┘
```

The LLM only sees **one node at a time**:

```text
flow: support_assistant
node: classify_intent
goal: classify the user's message
input:
  user_message: "I can't access my account"
context:
  last_episodes: [...]
allowed_outputs: ["billing", "bug", "feature", "other"]
```

This keeps prompts small and stable. The runtime handles control flow.

### Why This Approach

| Benefit | Description |
|---------|-------------|
| **Separation of concerns** | SochDB = data, Runtime = execution, LLM = reasoning |
| **Language-agnostic** | Rust, Python, TypeScript runtimes share the same flows |
| **Debuggable** | Every step, state change, and decision is in the DB |
| **Learnable** | Reflection nodes + stored feedback enable continuous improvement |
| **No prompt injection risk** | LLM never sees "execute this code"—just structured tasks |

### Built-in Patterns (Planned)

Templates for common agentic patterns:

- **Reflection loop** — Execute, evaluate, retry if needed
- **Tree-of-thought** — Parallel exploration with best-path selection
- **Self-correction** — Validate output, fix errors automatically
- **Tool-first-then-answer** — Gather data before responding

These ship as rows in `agent_flows` / `agent_nodes` that you can clone and customize.

---

## ☁️ Cloud Roadmap

> **Local-first success unlocks the cloud.**

SochDB is currently a **local-first, embedded database** — and it's working great! Based on the success of this MVP, I'm exploring a cloud offering:

| Phase | Status | Description |
|-------|--------|-------------|
| **Local MVP** | ✅ Live | Embedded + IPC modes, full ACID, vector search |
| **Cloud (SochDB Cloud)** | 🚧 On the way | Hosted, managed SochDB with sync |

**Your feedback shapes the cloud roadmap.** If you're interested in a hosted solution, let us know what you need!

---

## 💬 A Note from the Creator

> **This is an MVP — and your support makes it better.**

SochDB started as an experiment: *what if databases were designed for LLMs from day one?* The result is what you see here — a working, tested, and (I hope) useful database.

But here's the thing: **software gets better with users.** Every bug report, feature request, and "hey, this broke" message helps SochDB become more robust. You might find rough edges. You might encounter surprises. That's expected — and fixable!

**What I need from you:**
- 🐛 **Report bugs** — even small ones
- 💡 **Request features** — what's missing for your use case?
- ⭐ **Star the repo** — it helps others discover SochDB
- 📣 **Share your experience** — blog posts, tweets, anything

Your usage and feedback don't just help me — they help everyone building with SochDB. Let's make this great together.

> **Note:** SochDB is a **single-person project** built over weekends and spare time. I'm the sole developer, architect, and maintainer. This means you might find rough edges, incomplete features, or areas that need polish. The good news? Your contributions can make a real impact. More hands on this project means more advanced features, better stability, and faster progress. Every PR, issue report, and suggestion directly shapes what SochDB becomes.

*— Sushanth*

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
cargo install cargo-watch cargo-criterion

# Run in watch mode
cargo watch -x "test --all"

# Run specific benchmark
cargo criterion --bench vector_search
```

---

## 📄 License

**SochDB (sochdb)**  
Copyright (C) 2026 Sushanth Reddy Vanagala

This project is licensed under the **GNU Affero General Public License v3.0 or later** (AGPL-3.0-or-later).  
See the [LICENSE](LICENSE) file for the full text.

**What this means:**
- ✅ You can use, modify, and distribute SochDB freely
- ✅ You must share your modifications under AGPL-3.0
- ✅ If you run SochDB as a network service, you must share your source code
- 📖 Full license text: https://www.gnu.org/licenses/agpl-3.0.html

For commercial licensing options or questions, contact: sushanth@sochdb.dev

---

## 🙏 Acknowledgments

- HNSW algorithm: [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- MVCC implementation inspired by PostgreSQL and SQLite
- Columnar storage design influenced by Apache Arrow
- Vamana (DiskANN): Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node", NeurIPS 2019
- CoreNN: https://github.com/wilsonzlin/CoreNN
- HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI 2018
- PGM-Index: Ferragina & Vinciguerra, "The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds", VLDB 2020
- ARIES: Mohan et al., "ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging", ACM TODS 1992
- SSI: Cahill et al., "Serializable Isolation for Snapshot Databases", ACM SIGMOD 2008
- LSM-Tree: O'Neil et al., "The Log-Structured Merge-Tree (LSM-Tree)", Acta Informatica 1996
- Soch https://github.com/toon-format/toon

---

**Built with ❤️ for the AI era**

[GitHub](https://github.com/sochdb/sochdb) • [Documentation](https://sochdb.dev)
