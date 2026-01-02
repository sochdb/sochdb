# Tutorial: Vector Search with ToonDB

> **🔧 Skill Level:** Intermediate  
> **⏱️ Time Required:** 20 minutes  
> **📦 Requirements:** Python 3.9+, numpy, sentence-transformers

Learn how to build a semantic search system using ToonDB's HNSW vector index.

---

## 🎯 What You'll Build

A document search system that:
- ✅ Stores documents with vector embeddings
- ✅ Performs semantic similarity search
- ✅ Returns relevant results based on meaning, not keywords

---

## Step 1: Setup

```bash
# Create project
mkdir semantic-search && cd semantic-search
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install toondb-client numpy sentence-transformers
```

> **Note:** `sentence-transformers` provides real embedding models. For production, you might also use OpenAI, Cohere, or other embedding APIs.

---

## Step 2: Understand Vector Search

### What are Embeddings?

Embeddings convert text into numerical vectors that capture semantic meaning:

```
"The cat sat on the mat"  →  [0.12, -0.34, 0.56, ...]  (384 dimensions)
"A feline rested on a rug" →  [0.11, -0.32, 0.55, ...]  (similar vector!)
"Python programming"       →  [-0.45, 0.78, -0.23, ...] (different vector)
```

Similar meanings = similar vectors = small distance

### How HNSW Works

HNSW (Hierarchical Navigable Small World) is a graph-based index:

```
Layer 2:       ●───────────────────────●
               │                       │
Layer 1:       ●───●───────●───────────●───────●
               │   │       │           │       │
Layer 0:   ●───●───●───●───●───●───●───●───●───●───●
```

- **Search:** Start at top layer, greedily descend
- **Complexity:** O(log N) average
- **Recall:** 95%+ at practical settings

---

## Step 3: Create the Embedding Service

Create `embeddings.py`:

```python
"""Embedding service using sentence-transformers."""

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformer model.
        
        Models:
        - all-MiniLM-L6-v2: Fast, good quality, 384 dims
        - all-mpnet-base-v2: Better quality, 768 dims, slower
        - all-distilroberta-v1: Balanced, 768 dims
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Convert texts to embeddings."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings.astype(np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Convert a single text to embedding."""
        return self.embed([text])[0]


# Test it
if __name__ == "__main__":
    service = EmbeddingService()
    
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above a sleepy hound",
        "Python programming is fun",
    ]
    
    embeddings = service.embed(texts)
    
    # Check similarity (dot product of normalized vectors = cosine similarity)
    sim_01 = np.dot(embeddings[0], embeddings[1])
    sim_02 = np.dot(embeddings[0], embeddings[2])
    
    print(f"Similarity (fox/canine): {sim_01:.3f}")  # Should be high (~0.7+)
    print(f"Similarity (fox/python): {sim_02:.3f}")  # Should be low (~0.2)
```

Run it:
```bash
python embeddings.py
# Loading model: all-MiniLM-L6-v2...
# Model loaded. Embedding dimension: 384
# Similarity (fox/canine): 0.763
# Similarity (fox/python): 0.186
```

---

## Step 4: Build the Search System

Create `search.py`:

```python
"""Semantic search system with ToonDB."""

from toondb import Database
from toondb.bulk import bulk_build_index, bulk_query_index
from embeddings import EmbeddingService
import numpy as np
import json
import os

class SemanticSearch:
    def __init__(self, db_path: str = "./search_db"):
        self.db = Database.open(db_path)
        self.embeddings = EmbeddingService()
        self.index_path = os.path.join(db_path, "vectors.hnsw")
        self._documents = []
        self._load_documents()
    
    def _load_documents(self):
        """Load existing documents from database."""
        for key, value in self.db.scan(b"docs/"):
            doc = json.loads(value.decode())
            self._documents.append(doc)
        print(f"Loaded {len(self._documents)} existing documents")
    
    def add_documents(self, documents: list[dict]):
        """Add documents with automatic embedding generation.
        
        Each document should have at least 'content' field.
        Additional fields (title, metadata) are preserved.
        """
        print(f"Adding {len(documents)} documents...")
        
        # Extract content for embedding
        contents = [doc["content"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embeddings.embed(contents)
        
        # Store documents
        start_id = len(self._documents)
        with self.db.transaction() as txn:
            for i, doc in enumerate(documents):
                doc_id = start_id + i
                doc["id"] = doc_id
                txn.put(f"docs/{doc_id}".encode(), json.dumps(doc).encode())
                self._documents.append(doc)
        
        # Rebuild index with all embeddings
        self._rebuild_index()
        
        print(f"✅ Added {len(documents)} documents (total: {len(self._documents)})")
    
    def _rebuild_index(self):
        """Rebuild HNSW index from all documents."""
        if not self._documents:
            return
        
        # Get all embeddings
        contents = [doc["content"] for doc in self._documents]
        embeddings = self.embeddings.embed(contents)
        
        # Build HNSW index
        stats = bulk_build_index(
            embeddings,
            output=self.index_path,
            m=16,               # Graph connectivity
            ef_construction=200 # Build quality
        )
        
        print(f"Index built: {stats.get('vectors_indexed', len(embeddings))} vectors")
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for documents similar to query.
        
        Returns documents with similarity scores.
        """
        if not self._documents:
            return []
        
        # Embed query
        query_embedding = self.embeddings.embed_single(query)
        
        # Search index
        try:
            results = bulk_query_index(
                index_path=self.index_path,
                query=query_embedding,
                k=k,
                ef=50
            )
        except Exception as e:
            print(f"Index search failed: {e}")
            # Fallback to brute force
            return self._brute_force_search(query_embedding, k)
        
        # Map results to documents
        search_results = []
        for doc_id, distance in results:
            if doc_id < len(self._documents):
                doc = self._documents[doc_id].copy()
                doc["score"] = 1.0 - distance  # Convert distance to similarity
                doc["distance"] = distance
                search_results.append(doc)
        
        return search_results
    
    def _brute_force_search(self, query: np.ndarray, k: int) -> list[dict]:
        """Fallback brute-force search."""
        contents = [doc["content"] for doc in self._documents]
        embeddings = self.embeddings.embed(contents)
        
        # Compute similarities
        similarities = np.dot(embeddings, query)
        top_k = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k:
            doc = self._documents[idx].copy()
            doc["score"] = float(similarities[idx])
            results.append(doc)
        
        return results


def main():
    # Initialize
    search = SemanticSearch()
    
    # Add sample documents
    documents = [
        {
            "title": "ToonDB Overview",
            "content": "ToonDB is an LLM-native database designed for AI applications. It provides token-efficient storage and vector search capabilities."
        },
        {
            "title": "Vector Search Basics",
            "content": "Vector search uses embeddings to find semantically similar documents. HNSW is a popular algorithm for approximate nearest neighbor search."
        },
        {
            "title": "Python Development",
            "content": "Python is a versatile programming language popular for data science, web development, and AI applications."
        },
        {
            "title": "Database Transactions",
            "content": "ACID transactions ensure data integrity. ToonDB supports MVCC with serializable snapshot isolation for concurrent access."
        },
        {
            "title": "Machine Learning Models",
            "content": "Embedding models convert text to vectors. Popular models include sentence-transformers, OpenAI embeddings, and Cohere embeddings."
        },
    ]
    
    search.add_documents(documents)
    
    # Search examples
    print("\n" + "="*60)
    
    queries = [
        "How does ToonDB handle AI workloads?",
        "What is HNSW algorithm?",
        "How to ensure data consistency?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        results = search.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['score']:.3f}] {result['title']}")
            print(f"   {result['content'][:80]}...")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python search.py
```

Expected output:
```
Loading model: all-MiniLM-L6-v2...
Model loaded. Embedding dimension: 384
Loaded 0 existing documents
Adding 5 documents...
Index built: 5 vectors
✅ Added 5 documents (total: 5)

============================================================

🔍 Query: How does ToonDB handle AI workloads?
----------------------------------------
1. [0.842] ToonDB Overview
   ToonDB is an LLM-native database designed for AI applications...
2. [0.534] Machine Learning Models
   Embedding models convert text to vectors...
3. [0.423] Vector Search Basics
   Vector search uses embeddings to find semantically similar documents...

🔍 Query: What is HNSW algorithm?
----------------------------------------
1. [0.756] Vector Search Basics
   Vector search uses embeddings to find semantically similar documents...
2. [0.412] ToonDB Overview
   ToonDB is an LLM-native database designed for AI applications...
3. [0.389] Machine Learning Models
   Embedding models convert text to vectors...

🔍 Query: How to ensure data consistency?
----------------------------------------
1. [0.698] Database Transactions
   ACID transactions ensure data integrity...
2. [0.312] ToonDB Overview
   ToonDB is an LLM-native database designed for AI applications...
3. [0.287] Vector Search Basics
   Vector search uses embeddings to find semantically similar documents...
```

---

## Step 5: Tune Performance

### Index Parameters

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `m=8` | Faster build, less memory | Lower recall |
| `m=16` | Balanced (default) | Good recall |
| `m=32` | Better recall | More memory, slower build |
| `ef_construction=100` | Faster build | Slightly lower quality |
| `ef_construction=200` | Good quality | Balanced |
| `ef_construction=400` | Best quality | Slow build |
| `ef=20` | Fast search | Lower recall |
| `ef=50` | Balanced | Good recall |
| `ef=100` | High recall | Slower search |

### Memory Estimation

```
Memory ≈ vectors × dimensions × 4 bytes × 1.5 overhead

Example: 100,000 vectors × 384 dims × 4 × 1.5 = 230 MB
```

---

## Step 6: Production Considerations

### Use Persistent Embeddings

Don't regenerate embeddings on every load:

```python
def add_documents_with_cache(self, documents):
    """Store embeddings alongside documents."""
    contents = [doc["content"] for doc in documents]
    embeddings = self.embeddings.embed(contents)
    
    with self.db.transaction() as txn:
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = len(self._documents) + i
            
            # Store document
            txn.put(f"docs/{doc_id}", json.dumps(doc).encode())
            
            # Store embedding (binary)
            txn.put(f"embeddings/{doc_id}", emb.tobytes())
```

### Batch Processing

For large datasets:

```python
def add_documents_batched(self, documents, batch_size=1000):
    """Add documents in batches to manage memory."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        self.add_documents(batch)
        print(f"Processed {min(i + batch_size, len(documents))}/{len(documents)}")
```

---

## What You Learned

| Concept | What You Did |
|---------|--------------|
| **Embeddings** | Converted text to vectors using sentence-transformers |
| **HNSW indexing** | Built fast approximate nearest neighbor index |
| **Semantic search** | Found similar documents by meaning, not keywords |
| **ToonDB integration** | Stored documents and vectors together |

---

## Next Steps

| Goal | Resource |
|------|----------|
| Build RAG system | [MCP Integration](/cookbook/mcp-integration) |
| Use with MCP | [MCP Integration](/cookbook/mcp-integration) |
| Optimize performance | [Performance Guide](/concepts/performance) |
| Production deployment | [Deployment Guide](/guides/deployment) |

---

*Tutorial completed! You've built a working semantic search system with ToonDB.* 🎉

