#!/usr/bin/env python3
"""
LlamaIndex + SochDB Integration Example

Demonstrates RAG with:
- SochDB as a custom VectorStore for LlamaIndex
- Document ingestion and chunking
- Query engine with SochDB retrieval

Usage:
    pip install sochdb python-dotenv requests numpy llama-index llama-index-embeddings-azure-openai
    python3 examples/python/llamaindex_rag.py

    # Or from this monorepo:
    cd sochdb-python
    maturin develop --release
    cd ..
    python3 examples/python/llamaindex_rag.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import time
import numpy as np
import requests
from typing import List, Any, Optional
from dataclasses import dataclass

# Add SochDB package path for monorepo source usage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../sochdb-python"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# SochDB Vector Store (LlamaIndex-compatible interface)
# =============================================================================

@dataclass
class TextNode:
    """Simple text node for document storage."""
    id: str
    text: str
    metadata: dict


class SochDBVectorStore:
    """
    SochDB-backed vector store with LlamaIndex-compatible interface.
    
    In production, you would implement the actual LlamaIndex VectorStore interface:
    from llama_index.core.vector_stores import VectorStore
    """
    
    def __init__(self, dimension: int = 1536):
        from sochdb import VectorIndex
        
        self.dimension = dimension
        self.index = VectorIndex(dimension=dimension, max_connections=16, ef_construction=100)
        self.nodes: List[TextNode] = []
        
        # Azure embeddings
        self.embed_endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.embed_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.embed_deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.embed_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via Azure."""
        url = f"{self.embed_endpoint.rstrip('/')}/openai/deployments/{self.embed_deployment}/embeddings?api-version={self.embed_version}"
        headers = {"api-key": self.embed_key, "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={"input": texts, "model": self.embed_deployment})
        response.raise_for_status()
        return np.array([item["embedding"] for item in response.json()["data"]], dtype=np.float32)
    
    def add(self, nodes: List[TextNode]) -> List[str]:
        """Add nodes to the vector store."""
        texts = [node.text for node in nodes]
        embeddings = self._embed(texts)
        
        start_id = len(self.nodes)
        ids = np.arange(start_id, start_id + len(nodes), dtype=np.uint64)
        
        self.index.insert_batch(ids, embeddings)
        self.nodes.extend(nodes)
        
        return [node.id for node in nodes]
    
    def query(self, query_str: str, top_k: int = 5) -> List[TextNode]:
        """Query the vector store."""
        query_embedding = self._embed([query_str])[0]
        results = self.index.search(query_embedding, k=top_k)
        
        retrieved = []
        for idx, score in results:
            if int(idx) < len(self.nodes):
                retrieved.append(self.nodes[int(idx)])
        
        return retrieved


# =============================================================================
# Document Processing
# =============================================================================

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple text chunker."""
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        end = min(i + chunk_size, len(words))
        chunk = ' '.join(words[i:end])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks


# =============================================================================
# RAG Query Engine
# =============================================================================

class SochDBQueryEngine:
    """Simple RAG query engine using SochDB."""
    
    def __init__(self, vector_store: SochDBVectorStore):
        self.vector_store = vector_store
        
        # Azure LLM
        self.llm_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
        self.llm_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.llm_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        self.llm_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    def _call_llm(self, prompt: str) -> str:
        """Call Azure OpenAI."""
        url = f"{self.llm_endpoint.rstrip('/')}/openai/deployments/{self.llm_deployment}/chat/completions?api-version={self.llm_version}"
        headers = {"api-key": self.llm_key, "Content-Type": "application/json"}
        
        response = requests.post(url, headers=headers, json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.0
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def query(self, query_str: str, top_k: int = 3) -> str:
        """Execute a RAG query."""
        # Retrieve relevant nodes
        nodes = self.vector_store.query(query_str, top_k=top_k)
        
        # Build context
        context = "\n\n".join([f"[{i+1}] {node.text}" for i, node in enumerate(nodes)])
        
        # Generate response
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query_str}

Answer:"""
        
        return self._call_llm(prompt)


# =============================================================================
# Main Example
# =============================================================================

def run_llamaindex_example():
    """Run the LlamaIndex-style RAG example."""
    
    print("="*70)
    print("  LLAMAINDEX-STYLE RAG + SOCHDB")
    print("="*70)
    
    # ==========================================================================
    # Step 1: Create vector store
    # ==========================================================================
    print("\n1. Creating SochDB vector store...")
    vector_store = SochDBVectorStore(dimension=1536)
    print("   ✓ Vector store initialized")
    
    # ==========================================================================
    # Step 2: Ingest documents
    # ==========================================================================
    print("\n2. Ingesting documents...")
    
    # Sample documents (in production, load from files)
    documents = [
        """SochDB is a high-performance database specifically designed for AI agents 
        and Large Language Model (LLM) applications. It provides blazing-fast vector 
        search with sub-millisecond latency, making it ideal for real-time AI systems.
        The database achieves 117,000 vector inserts per second and 0.03ms search latency.""",
        
        """SochDB uses a Trie-Columnar Hybrid (TCH) storage engine that combines the 
        benefits of tries for hierarchical key access with columnar storage for analytical 
        queries. This unique architecture provides O(path) resolution time for lookups 
        and efficient range scans.""",
        
        """SochDB's vector search is powered by HNSW (Hierarchical Navigable Small World) 
        algorithm with SIMD acceleration. The implementation uses AVX2/AVX-512 intrinsics 
        for computing distances, achieving up to 8x speedup compared to scalar implementations.""",
        
        """SochDB is 24x faster than ChromaDB on vector search benchmarks. It also 
        outperforms SQLite by 24% on insert operations for key-value workloads. The 
        database is optimized for write-heavy, low-latency requirements of AI agents.""",
        
        """SochDB provides built-in support for session management, context queries, 
        and token budget enforcement. It also includes MCP (Model Context Protocol) 
        integration for seamless LLM tool usage.""",
    ]
    
    # Chunk and create nodes
    all_nodes = []
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc, chunk_size=100, overlap=20)
        for chunk_idx, chunk in enumerate(chunks):
            node = TextNode(
                id=f"doc_{doc_idx}_chunk_{chunk_idx}",
                text=chunk,
                metadata={"doc_id": doc_idx, "chunk_id": chunk_idx}
            )
            all_nodes.append(node)
    
    # Add to vector store
    start = time.perf_counter()
    vector_store.add(all_nodes)
    ingest_time = (time.perf_counter() - start) * 1000
    
    print(f"   ✓ Ingested {len(all_nodes)} chunks in {ingest_time:.0f}ms")
    
    # ==========================================================================
    # Step 3: Create query engine
    # ==========================================================================
    print("\n3. Creating query engine...")
    query_engine = SochDBQueryEngine(vector_store)
    print("   ✓ Query engine ready")
    
    # ==========================================================================
    # Step 4: Run queries
    # ==========================================================================
    print("\n4. Running queries...\n")
    print("-"*70)
    
    queries = [
        "What is SochDB and what is it designed for?",
        "How does SochDB's vector search work?",
        "How does SochDB compare to ChromaDB?",
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        
        start = time.perf_counter()
        response = query_engine.query(query, top_k=3)
        latency = (time.perf_counter() - start) * 1000
        
        print(f"\nA: {response}")
        print(f"\n(Latency: {latency:.0f}ms)")
        print("-"*70)
    
    # ==========================================================================
    # Example LlamaIndex Code
    # ==========================================================================
    
    print("\n5. Example LlamaIndex integration code:\n")
    
    example_code = '''
# pip install llama-index llama-index-vector-stores-custom

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import VectorStore
from sochdb import VectorIndex

class SochDBVectorStore(VectorStore):
    """Custom LlamaIndex vector store using SochDB."""
    
    def __init__(self, dimension: int = 1536):
        self._index = VectorIndex(dimension=dimension)
        self._nodes = []
    
    def add(self, nodes, **kwargs):
        embeddings = np.array([node.embedding for node in nodes])
        ids = np.arange(len(self._nodes), len(self._nodes) + len(nodes))
        self._index.insert_batch(ids, embeddings)
        self._nodes.extend(nodes)
    
    def query(self, query_embedding, top_k=10):
        results = self._index.search(query_embedding, k=top_k)
        return [self._nodes[int(idx)] for idx, _ in results]

# Usage
vector_store = SochDBVectorStore()
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
)
query_engine = index.as_query_engine()
response = query_engine.query("What is SochDB?")
'''
    
    print(example_code)
    
    print("\n" + "="*70)
    print("  ✓ LlamaIndex-style RAG completed!")
    print("="*70)


if __name__ == "__main__":
    run_llamaindex_example()
