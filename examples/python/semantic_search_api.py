#!/usr/bin/env python3
"""
Semantic Search API with ToonDB

A production-ready semantic search service showing:
- REST-style API patterns
- Document CRUD with embeddings
- Batch operations
- Metadata filtering

Usage:
    python3 examples/python/semantic_search_api.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import requests

# Add ToonDB SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Document:
    """Document with text and metadata."""
    id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass 
class SearchResult:
    """Search result with score."""
    document: Document
    score: float


# =============================================================================
# Semantic Search Service
# =============================================================================

class SemanticSearchService:
    """Production-ready semantic search service using ToonDB."""
    
    def __init__(self, collection_name: str = "default"):
        from toondb import VectorIndex
        
        self.collection_name = collection_name
        self.dimension = 1536
        
        # Vector index
        self.index = VectorIndex(
            dimension=self.dimension,
            max_connections=32,  # Higher for better recall
            ef_construction=200  # Higher for better index quality
        )
        
        # Document storage (in production, use ToonDB KV)
        self.documents: Dict[str, Document] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        # Azure embeddings
        self.embed_endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.embed_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.embed_deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.embed_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches."""
        url = f"{self.embed_endpoint.rstrip('/')}/openai/deployments/{self.embed_deployment}/embeddings?api-version={self.embed_version}"
        headers = {"api-key": self.embed_key, "Content-Type": "application/json"}
        
        all_embeddings = []
        batch_size = 16  # Azure limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = requests.post(url, headers=headers, json={"input": batch, "model": self.embed_deployment})
            response.raise_for_status()
            embeddings = [item["embedding"] for item in response.json()["data"]]
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    # ==========================================================================
    # CRUD Operations
    # ==========================================================================
    
    def add(self, text: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Document:
        """Add a single document."""
        doc_id = doc_id or str(uuid.uuid4())
        embedding = self._embed([text])[0]
        
        doc = Document(id=doc_id, text=text, metadata=metadata or {}, embedding=embedding)
        
        # Store and index
        idx = self.next_idx
        self.next_idx += 1
        
        ids = np.array([idx], dtype=np.uint64)
        self.index.insert_batch(ids, embedding.reshape(1, -1))
        
        self.documents[doc_id] = doc
        self.id_to_idx[doc_id] = idx
        self.idx_to_id[idx] = doc_id
        
        return doc
    
    def add_batch(self, items: List[Dict]) -> List[Document]:
        """Add multiple documents."""
        docs = []
        texts = [item["text"] for item in items]
        embeddings = self._embed(texts)
        
        start_idx = self.next_idx
        ids = np.arange(start_idx, start_idx + len(items), dtype=np.uint64)
        self.index.insert_batch(ids, embeddings)
        
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            doc_id = item.get("id") or str(uuid.uuid4())
            doc = Document(
                id=doc_id,
                text=item["text"],
                metadata=item.get("metadata", {}),
                embedding=embedding
            )
            
            idx = start_idx + i
            self.documents[doc_id] = doc
            self.id_to_idx[doc_id] = idx
            self.idx_to_id[idx] = doc_id
            docs.append(doc)
        
        self.next_idx += len(items)
        return docs
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document (marks as deleted, ToonDB doesn't support delete yet)."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            # Note: Vector remains in index but won't be returned
            return True
        return False
    
    # ==========================================================================
    # Search Operations
    # ==========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        query_embedding = self._embed([query])[0]
        
        # Get more results if filtering
        k = top_k * 3 if filter_metadata else top_k
        results = self.index.search(query_embedding, k=k)
        
        search_results = []
        for idx, score in results:
            doc_id = self.idx_to_id.get(int(idx))
            if doc_id and doc_id in self.documents:
                doc = self.documents[doc_id]
                
                # Apply metadata filter
                if filter_metadata:
                    match = all(
                        doc.metadata.get(k) == v
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue
                
                search_results.append(SearchResult(document=doc, score=float(score)))
                
                if len(search_results) >= top_k:
                    break
        
        return search_results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        keyword_boost: float = 0.3
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword matching."""
        # Semantic search
        semantic_results = self.search(query, top_k=top_k * 2)
        
        # Keyword matching boost
        query_words = set(query.lower().split())
        
        for result in semantic_results:
            doc_words = set(result.document.text.lower().split())
            overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
            result.score = result.score * (1 - keyword_boost) + overlap * keyword_boost
        
        # Re-sort and return top_k
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:top_k]
    
    # ==========================================================================
    # Stats
    # ==========================================================================
    
    def stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "collection": self.collection_name,
            "document_count": len(self.documents),
            "dimension": self.dimension,
            "index_size": self.next_idx,
        }


# =============================================================================
# Main
# =============================================================================

def run_search_api_demo():
    """Run the semantic search API demo."""
    
    print("="*70)
    print("  SEMANTIC SEARCH API + TOONDB")
    print("="*70)
    
    # Initialize service
    print("\n1. Initializing search service...")
    service = SemanticSearchService(collection_name="tech_docs")
    print("   ✓ Service ready")
    
    # Add documents
    print("\n2. Adding documents...")
    
    documents = [
        {"text": "ToonDB is a high-performance database for AI agents.", "metadata": {"category": "overview", "priority": "high"}},
        {"text": "ToonDB achieves 117,000 vector inserts per second.", "metadata": {"category": "performance", "priority": "high"}},
        {"text": "ToonDB uses HNSW algorithm with SIMD acceleration.", "metadata": {"category": "architecture", "priority": "medium"}},
        {"text": "ToonDB is 24x faster than ChromaDB on benchmarks.", "metadata": {"category": "performance", "priority": "high"}},
        {"text": "ToonDB provides MCP integration for LLM tools.", "metadata": {"category": "integration", "priority": "medium"}},
        {"text": "ToonDB is open source under Apache 2.0 license.", "metadata": {"category": "licensing", "priority": "low"}},
        {"text": "ToonDB supports session management and context queries.", "metadata": {"category": "features", "priority": "medium"}},
        {"text": "ToonDB's TCH storage engine provides O(path) resolution.", "metadata": {"category": "architecture", "priority": "high"}},
    ]
    
    start = time.perf_counter()
    docs = service.add_batch(documents)
    add_time = (time.perf_counter() - start) * 1000
    
    print(f"   ✓ Added {len(docs)} documents in {add_time:.0f}ms")
    print(f"   Stats: {service.stats()}")
    
    # Search
    print("\n3. Running searches...\n")
    print("-"*70)
    
    queries = [
        {"query": "How fast is ToonDB?", "filter": None},
        {"query": "Tell me about the architecture", "filter": None},
        {"query": "Performance benchmarks", "filter": {"category": "performance"}},
        {"query": "What license?", "filter": None},
    ]
    
    for q in queries:
        print(f"\nQuery: {q['query']}")
        if q['filter']:
            print(f"Filter: {q['filter']}")
        
        start = time.perf_counter()
        results = service.search(q['query'], top_k=3, filter_metadata=q.get('filter'))
        latency = (time.perf_counter() - start) * 1000
        
        print(f"\nResults ({latency:.1f}ms):")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.document.text[:60]}...")
            print(f"      Metadata: {r.document.metadata}")
        
        print("-"*70)
    
    # Hybrid search
    print("\n4. Hybrid search example...")
    results = service.hybrid_search("ToonDB performance speed fast", top_k=3)
    print(f"\nHybrid search results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.document.text[:60]}...")
    
    print("\n" + "="*70)
    print("  ✓ Semantic search API demo completed!")
    print("="*70)


if __name__ == "__main__":
    run_search_api_demo()
