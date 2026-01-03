# ToonDB v0.3.0 Release Notes

**Release Date:** January 3, 2026

## 🎉 What's New

ToonDB v0.3.0 is a major feature release focused on **multi-tenancy, hybrid search, and LLM-native context retrieval**. This release introduces namespace isolation, BM25+vector fusion, multi-vector documents, and a token-aware context query builder.

---

## 🚀 Major Features

### 1. Namespace Isolation for Multi-Tenancy

**Type-safe tenant isolation at the storage layer**

- ✅ `NamespaceRouter` with O(1) hash-map lookup
- ✅ On-disk layout: `data/namespaces/{tenant}/collections/{collection}/`
- ✅ `NamespaceHandle` and `CollectionHandle` abstractions
- ✅ Safe prefix iteration with automatic tenant scoping
- ✅ Python SDK: `Namespace` class with context manager support

**Python Example:**
```python
from toondb import Database

db = Database.open("./my_db")

# Create isolated namespace
with db.use_namespace("tenant_acme") as ns:
    collection = ns.create_collection("documents", dimension=384)
    collection.insert(id=1, vector=[...], metadata={...})
    
    # All operations automatically scoped to tenant_acme
    results = collection.search(query_vector, k=10)

db.close()
```

**Benefits:**
- Eliminates cross-tenant data leakage by construction
- No manual prefix management required
- CRUD operations on namespaces and collections

**See:** [Namespace API Reference](./api-reference/python-namespace.md)

---

### 2. Hybrid Search (Vector + BM25 Keyword)

**Best-of-both-worlds retrieval with Reciprocal Rank Fusion**

- ✅ BM25 scorer with Robertson-Sparck Jones IDF
- ✅ Inverted index with posting lists and term positions
- ✅ RRF fusion combining vector and keyword results
- ✅ Configurable weights for vector vs. keyword components
- ✅ Python SDK: Unified `search()` API with `hybrid_search()` convenience method

**Python Example:**
```python
from toondb import Database, SearchRequest

db = Database.open("./my_db")
ns = db.namespace("tenant_acme")
collection = ns.collection("documents")

# Hybrid search: 70% vector, 30% keyword
results = collection.hybrid_search(
    vector=query_embedding,
    text_query="machine learning optimization",
    k=10,
    alpha=0.7  # Vector weight
)

for result in results:
    print(f"{result.id}: {result.score:.3f}")
```

**How RRF Works:**
```
RRF_score(d) = Σ weight_i / (k + rank_i(d))
```
- Default k=60 (robust across datasets)
- Handles score normalization automatically
- Deduplicates results across components

**See:** [Hybrid Search Guide](./guides/hybrid-search.md)

---

### 3. Multi-Vector Documents

**Store multiple embeddings per document with aggregation**

- ✅ `MultiVectorMapping` for doc_id ↔ vector_ids tracking
- ✅ Aggregation methods: max, mean, first, last, sum
- ✅ Document-level and chunk-level scoring
- ✅ Python SDK: `insert_multi()` method

**Python Example:**
```python
from toondb import Database

db = Database.open("./my_db")
ns = db.namespace("tenant_acme")
collection = ns.collection("documents")

# Insert document with 3 chunk embeddings
collection.insert_multi(
    id="doc_123",
    vectors=[chunk_emb_1, chunk_emb_2, chunk_emb_3],
    metadata={"title": "ToonDB Guide", "author": "Alice"},
    chunk_texts=["Intro", "Body", "Conclusion"],
    aggregate="max"  # Use max score across chunks
)

# Search returns document-level results
results = collection.search(query_vector, k=10)
```

**Use Cases:**
- Long documents split into chunks
- Multi-modal embeddings (text + image)
- Hierarchical document structure

**See:** [Multi-Vector Guide](./guides/multi-vector-documents.md)

---

### 4. ContextQuery Builder for LLM Retrieval

**Token-aware context assembly with budget management**

- ✅ `ContextQuery` builder with fluent API
- ✅ Token budgeting (4 chars ≈ 1 token heuristic, tiktoken integration)
- ✅ Multi-source fusion (vector + keyword queries)
- ✅ Deduplication (exact, semantic)
- ✅ Relevance filtering
- ✅ Multiple output formats (text, markdown, JSON)

**Python Example:**
```python
from toondb import Database, ContextQuery, DeduplicationStrategy

db = Database.open("./my_db")
ns = db.namespace("tenant_acme")
collection = ns.collection("documents")

# Build context with token budget
context = (
    ContextQuery(collection)
    .add_vector_query(query_embedding, weight=0.7)
    .add_keyword_query("machine learning", weight=0.3)
    .with_token_budget(4000)  # Fit within model limit
    .with_min_relevance(0.5)
    .with_deduplication(DeduplicationStrategy.EXACT)
    .execute()
)

# Use in LLM prompt
prompt = f"""Context ({context.total_tokens} tokens):
{context.as_markdown()}

Question: {user_question}
"""

print(f"Retrieved {len(context)} chunks, dropped {context.dropped_count}")
```

**Features:**
- Automatic token counting and budgeting
- Prioritizes highest-scoring chunks
- Dedups similar content
- Metadata filtering support

**See:** [ContextQuery Guide](./guides/context-query.md)

---

### 5. Tombstone-Based Logical Deletion

**O(1) deletion checks during vector search**

- ✅ `TombstoneManager` for tracking deleted IDs
- ✅ `TombstoneFilter` for filtering during search
- ✅ Persistent storage in `.tomb` files
- ✅ `effective_k` computation to maintain result quality
- ✅ Batch deletion and compaction

**Python Example:**
```python
collection.delete(doc_id=123)  # Marks as deleted

# Search automatically filters tombstones
results = collection.search(query_vector, k=10)  # Never returns deleted docs
```

**Performance:**
- O(1) tombstone lookup via HashSet
- No index rebuild required
- Compaction removes stale tombstones

**See:** [Tombstones Internals](./internals/tombstones.md)

---

### 6. Enhanced Error Taxonomy

**Machine-readable error codes with remediation hints**

- ✅ `ErrorCode` enum with 1xxx-9xxx ranges
- ✅ Hierarchical exception classes
- ✅ Cross-language consistency (Rust ↔ Python)
- ✅ Remediation hints for common errors

**Python Example:**
```python
from toondb import Database, NamespaceNotFoundError, ErrorCode

db = Database.open("./my_db")

try:
    ns = db.namespace("missing_tenant")
except NamespaceNotFoundError as e:
    print(f"Error {e.code}: {e.message}")
    print(f"Fix: {e.remediation}")
    # Error 3001: Namespace not found: missing_tenant
    # Fix: Create the namespace first with db.create_namespace('missing_tenant')
```

**Error Code Ranges:**
- 1xxx: Connection/Transport
- 2xxx: Transaction
- 3xxx: Namespace
- 4xxx: Collection
- 5xxx: Query
- 6xxx: Validation
- 7xxx: Resource
- 8xxx: Authorization
- 9xxx: Internal

**See:** [Error Handling Guide](./guides/error-handling.md)

---

## 🏗️ Architecture Improvements

### Storage Layer
- Namespace routing with O(1) lookup
- On-disk layout: `data/namespaces/{tenant}/collections/{collection}/`
- Metadata storage in `_namespaces.meta`
- Safe prefix iteration with `next_prefix()` algorithm

### Vector Layer
- BM25 scorer with configurable k1, b parameters
- Inverted index with term positions
- RRF fusion engine
- Multi-vector aggregation
- Tombstone filtering during search

### Python SDK
- Frozen `CollectionConfig` dataclass (immutable)
- Unified `search()` API with convenience wrappers
- Context manager support for namespaces
- Type hints throughout

---

## 📊 Test Coverage

**Rust Tests:** 33 passing
- BM25: 6 tests
- Inverted Index: 7 tests
- Hybrid Search: 6 tests
- Tombstones: 7 tests
- Multi-Vector: 7 tests

**Python Tests:** Comprehensive suite
- Error taxonomy validation
- Collection config validation
- Search request validation
- ContextQuery builder
- Namespace CRUD operations

---

## 📦 Installation

### Python SDK

```bash
pip install --upgrade toondb-client
```

### Rust

```toml
[dependencies]
toondb = "0.3"
toondb-vector = "0.3"
toondb-storage = "0.3"
```

---

## 🔄 Migration Guide

### From v0.2.x to v0.3.0

**1. Namespace API (Optional - Backward Compatible)**

Old (still works):
```python
db = Database.open("./my_db")
db.put(b"users/alice", b"data")
```

New (recommended for multi-tenant apps):
```python
db = Database.open("./my_db")
ns = db.namespace("tenant_a")
collection = ns.collection("users")
collection.insert(id="alice", vector=[...], metadata={...})
```

**2. Error Handling (Enhanced)**

Old:
```python
try:
    db.get(b"missing_key")
except DatabaseError:
    pass
```

New:
```python
from toondb import NamespaceNotFoundError, ErrorCode

try:
    ns = db.namespace("missing")
except NamespaceNotFoundError as e:
    print(f"Error {e.code}: {e.remediation}")
```

**3. Search API (Unified)**

The old vector search API is unchanged. New unified API:
```python
# Vector-only (unchanged)
results = collection.vector_search(query_vector, k=10)

# NEW: Keyword search
results = collection.keyword_search("machine learning", k=10)

# NEW: Hybrid search
results = collection.hybrid_search(query_vector, "ML", k=10, alpha=0.7)

# NEW: Unified API
from toondb import SearchRequest
request = SearchRequest(vector=query_vector, text_query="ML", k=10, alpha=0.7)
results = collection.search(request)
```

---

## 🐛 Bug Fixes

- Fixed borrowing issue in BM25 scorer IDF computation
- Removed unused imports in inverted index module
- Fixed error code enum consistency between Rust and Python

---

## ⚠️ Breaking Changes

**None.** v0.3.0 is fully backward compatible with v0.2.x.

---

## 🎯 Roadmap

**Coming in v0.3.1:**
- Reranking models integration
- Cross-encoder support
- Advanced filtering (range queries, arrays)

**Coming in v0.4.0:**
- Distributed namespace sharding
- Replication and clustering
- Time-travel queries

---

## 📚 Documentation

- [Namespace API Reference](./api-reference/python-namespace.md)
- [Hybrid Search Guide](./guides/hybrid-search.md)
- [Multi-Vector Documents](./guides/multi-vector-documents.md)
- [ContextQuery Guide](./guides/context-query.md)
- [Error Handling](./guides/error-handling.md)

---

## 🙏 Contributors

- [@sushanthpy](https://github.com/sushanthpy) - Core architecture and implementation

---

## 📄 License

Apache 2.0

---

**Full Changelog:** https://github.com/toondb/toondb/compare/v0.2.9...v0.3.0
