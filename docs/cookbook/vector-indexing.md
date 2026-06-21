# How to Build and Query Vector Indexes

> Create HNSW indexes for semantic search and similarity queries.

---

## Problem

You have vector embeddings (from OpenAI, sentence-transformers, etc.) and need to perform fast similarity search.

---

## Solution

### 1. Create a Vector Index (Python)

```python
from sochdb import Database
from sochdb.bulk import bulk_build_index
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
from sochdb.bulk import bulk_query_index

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

### gRPC Distance Metric Semantics

gRPC search responses include the index metric at response level so clients can
interpret `result.distance` without separately fetching index configuration.

```python
response = client.search(
    index_name="vectors",
    query=query,
    k=10,
    ef=100,
)

print(response.metric)

for result in response.results:
    print(result.id, result.distance)
```

Distances are ordered ascending. For cosine, distance is `1 - cosine_similarity`.
For Euclidean L2 in the current HNSW path, lower values are closer. For dot
product indexes, the raw dot product is negated before return, so lower values
are more similar and the raw dot product is `-result.distance`.

### Vector Metadata

Vector inserts can include optional typed metadata for general multi-vector
retrieval. `parent_id` identifies a logical source shared by several vectors,
and `view_type` describes the representation stored in that point. Both fields
are optional; a missing `parent_id` is not replaced with the vector ID in raw
search results.

```python
request = sochdb_pb2.InsertBatchRequest(
    index_name="memories",
    ids=[712001, 712002],
    vectors=[*turn_vector, *event_vector],
    metadata=[
        sochdb_pb2.VectorMetadata(
            parent_id=712,
            view_type="turn",
        ),
        sochdb_pb2.VectorMetadata(
            parent_id=712,
            view_type="event",
        ),
    ],
)

client.InsertBatch(request)
```

Search results return stored metadata when present:

```python
for result in response.results:
    print(result.id, result.distance, result.parent_id, result.view_type)
```

### Parent-Aware Grouped Search

Grouped search is still ANN search over indexed points. The server first asks
the point-level index for `candidate_k` neighbors, then groups that fixed
candidate pool by `parent_id`. `k` is the final grouped result budget, so a
grouped query can return fewer than `k` results when the candidate pool contains
fewer than `k` distinct parents.

For vectors without `parent_id`, grouped search treats the vector ID as its own
group key. `result.distance` remains the winning point distance and must be
interpreted using `response.metric`.

```python
request = sochdb_pb2.SearchRequest(
    index_name="memories",
    query=query_vector,
    k=20,
    ef=200,
    grouping=sochdb_pb2.GroupingOptions(
        group_by=sochdb_pb2.GROUP_BY_PARENT_ID,
        candidate_k=100,
        max_per_group=1,
    ),
)

response = client.Search(request)

print(response.metric)
print(response.grouping.raw_candidate_count)
print(response.grouping.unique_group_count)

for result in response.results:
    print(result.id, result.parent_id, result.view_type, result.distance)
```

With `max_per_group = 1`, only the best-ranked point for each parent is
returned. With `max_per_group > 1`, the server preserves the top points from
each group without exceeding that per-parent limit. Grouping is a general
multi-vector document feature and does not apply score aggregation, reranking,
or benchmark-specific rules.

### Synthetic Grouping Check

The grouped-search tests include a deterministic fixture where parent `1` has
three highly ranked views and parents `2`, `3`, and `4` have one view each:

```text
raw top-3 parents:
1, 1, 1

grouped top-3 parents:
1, 2, 3
```

The fixture reports the raw candidate count, unique parent count, grouped result
count, and the microsecond cost of the server-side grouping pass over the fixed
candidate pool. It is a functional check only; it does not claim semantic recall
improvement.

### 3. Integrated with Database (Rust)

```rust
use sochdb::{SochConnection, SchemaBuilder, SochType, SochValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conn = SochConnection::open("./vector_db")?;

    // Create table with vector column
    let schema = SchemaBuilder::table("documents")
        .field("id", SochType::UInt)
        .field("content", SochType::Text)
        .field("embedding", SochType::Vector(384))
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
        ("id", SochValue::UInt(1)),
        ("content", SochValue::Text("Hello world".into())),
        ("embedding", SochValue::Vector(embedding)),
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
"""Semantic search over documents using SochDB."""

from sochdb import Database
from sochdb.bulk import bulk_build_index, bulk_query_index
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
    {"content": "SochDB is an LLM-native database"},
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
