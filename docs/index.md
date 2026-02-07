---
sidebar_position: 1
slug: /
title: Introduction
---

# SochDB Documentation

Welcome to the official SochDB documentation. SochDB is **The LLM-Native Database** — a high-performance embedded database designed specifically for AI applications.

**Current Version:** v0.4.5 (Core) | Python SDK v0.4.7 | Node.js SDK v0.5.1 | Go SDK v0.4.3

---

## Key Features

| Feature | Description |
|---------|-------------|
| **40-66% Fewer Tokens** | TOON format optimized for LLM consumption |
| **Dual-Mode Architecture** | Embedded (FFI) + Server (gRPC) for flexible deployment |
| **Namespace & Collections** (v0.4.1) | Type-safe multi-tenant isolation with vector collections |
| **Priority Queue** (v0.4.3) | First-class queue API with ordered-key task entries |
| **Memory System** (v0.4.2) | Extraction, consolidation, and hybrid retrieval for agents |
| **MCP Integration** (v0.4.3) | Model Context Protocol for Claude and LLM agents |
| **Graph Overlay** (v0.3.3) | Lightweight graph layer for agent memory with BFS/DFS traversal |
| **Temporal Graph** (v0.4.6) | Time-aware relationships with point-in-time queries |
| **Hybrid Search** | Vector + BM25 keyword search with RRF fusion |
| **Context Builder** | Token-aware retrieval with configurable limits |
| **Semantic Cache** | Cache LLM responses with similarity lookup |
| **Policy Service** (v0.4.3) | Access control policies for namespaces |
| **Blazing Fast** | Rust-powered with zero-copy and SIMD |
| **Vector Search** | Built-in HNSW indexing (~15,000 vec/s) |
| **Embeddable** | In-process or client-server mode |
| **Multi-Language** | Native SDKs for Rust, Python, Node.js, Go |

---

## Quick Install

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="rust" label="Rust" default>

```bash
cargo add sochdb
```

```rust
use sochdb::Database;

fn main() -> anyhow::Result<()> {
    let db = Database::open("./my_app_db")?;

    db.with_transaction(|txn| {
        txn.put(b"users/alice", br#"{"name": "Alice", "role": "admin"}"#)?;
        Ok(())
    })?;

    if let Some(user) = db.get(b"users/alice")? {
        println!("{}", String::from_utf8_lossy(&user));
    }

    Ok(())
}
```

</TabItem>
<TabItem value="python" label="Python">

```bash
pip install sochdb
```

```python
from sochdb import Database

db = Database.open("./my_app_db")

with db.transaction() as txn:
    txn.put(b"users/alice", b'{"name": "Alice", "role": "admin"}')

user = db.get(b"users/alice")
print(user.decode())  # {"name": "Alice", "role": "admin"}

db.close()
```

</TabItem>
<TabItem value="nodejs" label="Node.js">

```bash
npm install @sochdb/sochdb
```

```typescript
import { Database } from '@sochdb/sochdb';

const db = Database.open('./my_app_db');

await db.withTransaction(async (txn) => {
  await txn.put('users/alice', '{"name": "Alice", "role": "admin"}');
});

const user = await db.get('users/alice');
console.log(user?.toString());

await db.close();
```

</TabItem>
<TabItem value="go" label="Go">

```bash
go get github.com/sochdb/sochdb-go@v0.4.3
```

```go
package main

import (
    "fmt"
    sochdb "github.com/sochdb/sochdb-go"
)

func main() {
    db, _ := sochdb.Open("./my_app_db")
    defer db.Close()

    db.WithTransaction(func(txn *sochdb.Transaction) error {
        return txn.Put("users/alice", []byte(`{"name": "Alice", "role": "admin"}`))
    })

    user, _ := db.Get("users/alice")
    fmt.Println(string(user))
}
```

</TabItem>
</Tabs>

→ [Full Quick Start Guide](/getting-started/quickstart)

---

## Documentation Sections

### 🚀 Getting Started
Step-by-step guides to get you up and running quickly.

- [Quick Start](/getting-started/quickstart) — 5-minute intro
- [Installation](/getting-started/installation) — All platforms
- [First App](/getting-started/first-app) — Build something real

### 📖 Guides
Task-oriented guides for specific use cases.

**Language SDKs:**
- [Rust SDK](/guides/rust-sdk) — Native Rust guide
- [Python SDK](/guides/python-sdk) — Complete Python guide (v0.4.7)
- [Node.js SDK](/guides/nodejs-sdk) — TypeScript/JavaScript guide (v0.5.1)
- [Go SDK](/guides/go-sdk) — Go client guide (v0.4.3)

**Features:**
- [SQL Guide](/guides/sql-guide) — Working with SQL queries
- [Vector Search](/guides/vector-search) — HNSW indexing
- [Bulk Operations](/guides/bulk-operations) — Batch processing
- [Deployment](/guides/deployment) — Production setup

**AI Agent Safety & Memory:**
- [Policy & Safety Hooks](/guides/policy-hooks) — Pre/post operation validation
- [Multi-Agent Tool Routing](/guides/tool-routing) — Route tools across agents
- [Graph Overlay](/guides/graph-overlay) — Lightweight graph for agent memory
- [Context Query](/guides/context-query) — Token-aware retrieval for LLMs

### 💡 Concepts
Deep dives into SochDB's architecture and design.

- [Architecture](/concepts/architecture) — System design
- [TOON Format](/concepts/toon-format) — Token-optimized format
- [Performance](/concepts/performance) — Optimization guide

### 📋 API Reference
Complete technical specifications.

- [SQL API](/api-reference/sql-api) — SQL query reference
- [Rust API](/api-reference/rust-api) — Crate documentation
- [Python API](/api-reference/python-api) — Full Python API docs
- [Node.js API](/api-reference/nodejs-api) — TypeScript/JavaScript API
- [Go API](/api-reference/go-api) — Go package documentation

### 🛠️ Server Reference
Deep technical documentation for SochDB servers and tools.

- [IPC Server](/servers/IPC_SERVER.md) — Wire protocol & architecture
- [gRPC Server](/servers/GRPC_SERVER.md) — Vector search service
- [Bulk Operations](/servers/BULK_OPERATIONS.md) — High-performance tools

### 🍳 Cookbook
Recipes for common tasks.

- [Vector Indexing](/cookbook/vector-indexing) — Embedding workflows
- [MCP Integration](/cookbook/mcp-integration) — Claude integration
- [Logging](/cookbook/logging) — Observability setup

---

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Get started in 5 minutes | [Quick Start](/getting-started/quickstart) |
| Use Namespace & Collections | [Python SDK](/guides/python-sdk#namespace--collections) |
| Use Priority Queues | [Python SDK](/guides/python-sdk#priority-queue) |
| Use Memory System | [Node.js SDK](/guides/nodejs-sdk#memory-system) |
| Use SQL queries | [SQL Guide](/guides/sql-guide) |
| Use the Rust SDK | [Rust Guide](/guides/rust-sdk) |
| Use the Python SDK | [Python Guide](/guides/python-sdk) |
| Use the Node.js SDK | [Node.js Guide](/guides/nodejs-sdk) |
| Use the Go SDK | [Go Guide](/guides/go-sdk) |
| Add vector search | [Vector Search](/guides/vector-search) |
| Integrate with Claude (MCP) | [MCP Integration](/cookbook/mcp-integration) |
| Enforce agent safety policies | [Policy Hooks](/guides/policy-hooks) |
| Route tools across agents | [Tool Routing](/guides/tool-routing) |
| Model agent memory relationships | [Graph Overlay](/guides/graph-overlay) |
| Build token-aware context | [Context Query](/guides/context-query) |
| Understand the architecture | [Architecture](/concepts/architecture) |
| See the SQL API reference | [SQL API](/api-reference/sql-api) |

---

## External Links

- [**sochdb.dev**](https://sochdb.dev) — Main website
- [**GitHub**](https://github.com/sochdb/sochdb) — Source code
- [**Python SDK**](https://github.com/sochdb/sochdb-python-sdk) — Python SDK repo
- [**Node.js SDK**](https://github.com/sochdb/sochdb-nodejs-sdk) — Node.js SDK repo
- [**Go SDK**](https://github.com/sochdb/sochdb-go) — Go SDK repo
- [**Discussions**](https://github.com/sochdb/sochdb/discussions) — Community Q&A
