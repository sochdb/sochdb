# How to Build and Query Vector Indexes

> Create HNSW indexes for semantic search and similarity queries.

---

## Problem

You have vector embeddings (from OpenAI, sentence-transformers, etc.) and need to perform fast similarity search.

---

## Solution

### 1. Create a Vector Index (Python)

```python
from toondb import Database
from toondb.bulk import bulk_build_index
import numpy as np

# Generate or load embeddings
embeddings = np.random.randn(10000, 384).astype(np.float32)
ids = np.arange(10000, dtype=np.uint64)

# Build HNSW index with optimal parameters
stats = bulk_build_index(
    embeddings,
    output="./my_index.hnsw",
    ids=ids,                    # Optional: custom IDs (must be uint64)
    m=16,                       # Graph connectivity (higher = better recall, more memory)
    ef_construction=200,        # Build quality (higher = better index, slower build)
)

print(f"Built index: {stats.vectors} vectors in {stats.elapsed_secs:.2f}s")
```

### 2. Query the Index

```python
from toondb.bulk import bulk_query_index

# Query vector (from your embedding model)
query = np.random.randn(384).astype(np.float32)

# Find k nearest neighbors
results = bulk_query_index(
    index_path="./my_index.hnsw",
    query=query,
    k=10,                       # Number of results
    ef=50                       # Search quality (optional, higher = better recall, slower)
)

for id, distance in results:
    print(f"ID: {id}, Distance: {distance:.4f}")
```

### 3. Integrated with Database (Rust)

```rust
use toondb::{ToonConnection, SchemaBuilder, ToonType, ToonValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conn = ToonConnection::open("./vector_db")?;

    // Create table with vector column
    let schema = SchemaBuilder::table("documents")
        .field("id", ToonType::UInt)
        .field("content", ToonType::Text)
        .field("embedding", ToonType::Vector(384))
        .primary_key("id")
        .index("embedding", IndexType::HNSW {
            m: 16,
            ef_construction: 200,
        })
        .build();

    conn.create_table(schema)?;

    // Insert document with embedding
    let embedding: Vec<f32> = get_embedding("Hello world");
    conn.insert("documents", vec![
        ("id", ToonValue::UInt(1)),
        ("content", ToonValue::Text("Hello world".into())),
        ("embedding", ToonValue::Vector(embedding)),
    ])?;

    // Vector search
    let query_embedding = get_embedding("Hi there");
    let results = conn.query("documents")
        .vector_search("embedding", &query_embedding, 10)
        .select(&["id", "content"])
        .execute()?;

    Ok(())
}
```

---

## HNSW Parameter Tuning

### Build Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `m` | 16 | 4-64 | Higher = better recall, more memory |
| `ef_construction` | 200 | 50-500 | Higher = better index quality, slower build |

**Guidelines:**
- Small datasets (< 10K): `m=8`, `ef_construction=100`
- Medium (10K-1M): `m=16`, `ef_construction=200`
- Large (> 1M): `m=32`, `ef_construction=400`

### Search Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `ef` | 50 | 10-500 | Higher = better recall, slower query |
| `k` | 10 | 1-1000 | Number of results |

**Guidelines:**
- Fast search (< 1ms): `ef=20`
- Balanced: `ef=50`
- High recall (> 99%): `ef=200`

### Quantization Trade-offs

| Type | Memory | Precision | Use Case |
|------|--------|-----------|----------|
| `f32` | 100% | Full | Default, best accuracy |
| `f16` | 50% | ~0.1% loss | Large indexes, memory-constrained |
| `bf16` | 50% | ~0.5% loss | ML models trained with bf16 |

---

## Example: Semantic Search System

```python
#!/usr/bin/env python3
"""Semantic search over documents using ToonDB."""

from toondb import Database
from toondb.bulk import bulk_build_index, bulk_query_index
import numpy as np

# Simulated embedding function (replace with real model)
def get_embedding(text: str, dim: int = 384) -> np.ndarray:
    # In production: use sentence-transformers, OpenAI, etc.
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

class SemanticSearch:
    def __init__(self, db_path: str, index_path: str):
        self.db = Database.open(db_path)
        self.index_path = index_path
        self.documents = []
    
    def add_documents(self, docs: list[dict]):
        """Add documents with their embeddings."""
        embeddings = []
        
        for i, doc in enumerate(docs):
            doc_id = len(self.documents) + i
            
            # Store document
            self.db.put(
                f"docs/{doc_id}/content".encode(),
                doc["content"].encode()
            )
            if "metadata" in doc:
                self.db.put(
                    f"docs/{doc_id}/metadata".encode(),
                    str(doc["metadata"]).encode()
                )
            
            # Get embedding
            embedding = get_embedding(doc["content"])
            embeddings.append(embedding)
            self.documents.append(doc)
        
        # Rebuild index with all embeddings
        all_embeddings = np.array(embeddings)
        bulk_build_index(
            all_embeddings,
            output=self.index_path,
            m=16,
            ef_construction=200
        )
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for similar documents."""
        query_embedding = get_embedding(query)
        
        results = bulk_query_index(
            index_path=self.index_path,
            query=query_embedding,
            k=k,
            ef=50
        )
        
        search_results = []
        for doc_id, distance in results:
            content = self.db.get(f"docs/{doc_id}/content".encode())
            if content:
                search_results.append({
                    "id": doc_id,
                    "content": content.decode(),
                    "score": 1.0 - distance  # Convert distance to similarity
                })
        
        return search_results


# Usage
search = SemanticSearch("./search_db", "./search.hnsw")

# Add documents
search.add_documents([
    {"content": "ToonDB is an LLM-native database"},
    {"content": "Vector search enables semantic queries"},
    {"content": "HNSW provides fast approximate nearest neighbor search"},
    {"content": "Python SDK makes integration easy"},
])

# Search
results = search.search("database for AI applications", k=3)
for r in results:
    print(f"[{r['score']:.3f}] {r['content']}")
```

---

## Discussion

### When to Use Vector Search

✅ **Good for:**
- Semantic similarity (find similar documents)
- Recommendation systems
- RAG (Retrieval Augmented Generation)
- Image/audio similarity

❌ **Not for:**
- Exact matching (use regular indexes)
- Structured queries (use SQL-like queries)
- Small datasets (< 1000 items, just brute force)

### Memory Estimation

```
Memory = vectors × dimensions × bytes_per_element × overhead

Example: 1M vectors × 384 dims × 4 bytes × 1.5 overhead
       = 1,000,000 × 384 × 4 × 1.5
       ≈ 2.3 GB
```

With F16 quantization: ~1.15 GB

---

## See Also

- [Vector Search Tutorial](/guides/vector-search) — Semantic search guide
- [Performance Optimization](/concepts/performance) — Tuning tips
- [MCP Integration](/cookbook/mcp-integration) — Claude integration

