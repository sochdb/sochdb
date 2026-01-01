#!/usr/bin/env python3
"""
Real Embedding Benchmark: 1000+ documents with actual Azure OpenAI embeddings

Tests end-to-end performance including:
1. Embedding generation time
2. Vector insertion time
3. Search latency with real embeddings
4. Comparison across ToonDB, ChromaDB, LanceDB

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 benchmarks/real_embedding_benchmark.py
"""

import os
import sys
import time
import json
import tempfile
import shutil
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Document Generator
# =============================================================================

TOPICS = [
    "machine learning", "database optimization", "cloud computing", 
    "API design", "security best practices", "performance tuning",
    "microservices architecture", "data pipelines", "monitoring",
    "error handling", "testing strategies", "deployment automation"
]

TEMPLATES = [
    "How to implement {topic} in production systems with best practices",
    "Troubleshooting {topic} issues and common error patterns",
    "{topic} comparison: approaches and trade-offs for enterprise",
    "Getting started with {topic}: a comprehensive guide",
    "Advanced {topic} techniques for high-scale applications",
    "Best practices for {topic} in distributed systems",
    "{topic} optimization strategies for reducing latency",
    "Migrating legacy systems to modern {topic} patterns",
]

def generate_documents(count: int) -> List[str]:
    """Generate realistic document texts."""
    docs = []
    for i in range(count):
        topic = random.choice(TOPICS)
        template = random.choice(TEMPLATES)
        base = template.format(topic=topic)
        # Add some variation
        suffix = f" - Document {i} covering {random.choice(['beginner', 'intermediate', 'advanced'])} level content."
        docs.append(base + suffix)
    return docs


# =============================================================================
# Embedding Client with Metrics
# =============================================================================

@dataclass
class EmbeddingMetrics:
    total_calls: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0
    errors: int = 0


class EmbeddingClient:
    """Azure OpenAI embedding client with metrics."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        self.dimension = 1536
        self.metrics = EmbeddingMetrics()
    
    def embed_batch(self, texts: List[str], batch_size: int = 16) -> Tuple[np.ndarray, float]:
        """Embed texts in batches, returns embeddings and total time."""
        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/embeddings?api-version={self.version}"
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        
        all_embeddings = []
        total_time = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            start = time.perf_counter()
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json={"input": batch, "model": self.deployment},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                
                embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(embeddings)
                
                # Track metrics
                self.metrics.total_calls += 1
                self.metrics.total_tokens += data.get("usage", {}).get("total_tokens", 0)
                
            except Exception as e:
                print(f"   Error on batch {i//batch_size}: {e}")
                self.metrics.errors += 1
                # Return zeros for failed batch
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
            
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed
            self.metrics.total_time_ms += elapsed
            
            # Progress indicator
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Embedded {i + len(batch)}/{len(texts)} texts...")
        
        return np.array(all_embeddings, dtype=np.float32), total_time


# =============================================================================
# Database Adapters
# =============================================================================

class ToonDBAdapter:
    name = "ToonDB"
    
    def __init__(self, dimension: int):
        from toondb import VectorIndex
        self.index = VectorIndex(dimension=dimension, max_connections=32, ef_construction=200)
        self.count = 0
    
    def insert(self, embeddings: np.ndarray) -> float:
        start = time.perf_counter()
        ids = np.arange(self.count, self.count + len(embeddings), dtype=np.uint64)
        self.index.insert_batch(ids, embeddings)
        self.count += len(embeddings)
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List, float]:
        start = time.perf_counter()
        results = list(self.index.search(query, k=k))
        return results, (time.perf_counter() - start) * 1000
    
    def cleanup(self):
        pass


class ChromaDBAdapter:
    name = "ChromaDB"
    
    def __init__(self, dimension: int):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=f"bench_{random.randint(0, 999999)}",
            metadata={"hnsw:space": "cosine"}
        )
        self.count = 0
    
    def insert(self, embeddings: np.ndarray) -> float:
        start = time.perf_counter()
        ids = [f"id_{self.count + i}" for i in range(len(embeddings))]
        self.collection.add(embeddings=embeddings.tolist(), ids=ids)
        self.count += len(embeddings)
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self.collection.query(query_embeddings=[query.tolist()], n_results=k)
        return results, (time.perf_counter() - start) * 1000
    
    def cleanup(self):
        pass


class LanceDBAdapter:
    name = "LanceDB"
    
    def __init__(self, dimension: int):
        import lancedb
        self.tmp_dir = tempfile.mkdtemp()
        self.db = lancedb.connect(self.tmp_dir)
        self.table = None
        self.count = 0
    
    def insert(self, embeddings: np.ndarray) -> float:
        start = time.perf_counter()
        data = [{"id": self.count + i, "vector": vec.tolist()} for i, vec in enumerate(embeddings)]
        if self.table is None:
            self.table = self.db.create_table("vectors", data, mode="overwrite")
        else:
            self.table.add(data)
        self.count += len(embeddings)
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self.table.search(query.tolist()).limit(k).to_list()
        return results, (time.perf_counter() - start) * 1000
    
    def cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("="*70)
    print("  REAL EMBEDDING BENCHMARK")
    print("  1000+ Documents with Azure OpenAI Embeddings")
    print("="*70)
    
    NUM_DOCS = 1000
    NUM_QUERIES = 100
    BATCH_SIZE = 16
    
    # Initialize embedding client
    embed_client = EmbeddingClient()
    
    # Generate documents
    print(f"\n1. Generating {NUM_DOCS} documents...")
    docs = generate_documents(NUM_DOCS)
    queries = generate_documents(NUM_QUERIES)
    print(f"   ✓ Generated {len(docs)} docs and {len(queries)} queries")
    
    # Embed documents
    print(f"\n2. Embedding {NUM_DOCS} documents (this will make ~{NUM_DOCS // BATCH_SIZE} API calls)...")
    doc_embeddings, doc_embed_time = embed_client.embed_batch(docs, batch_size=BATCH_SIZE)
    print(f"   ✓ Embedded {len(doc_embeddings)} docs in {doc_embed_time/1000:.1f}s")
    print(f"   API calls: {embed_client.metrics.total_calls}")
    print(f"   Tokens: {embed_client.metrics.total_tokens:,}")
    print(f"   Avg latency: {doc_embed_time / (NUM_DOCS // BATCH_SIZE):.0f}ms per batch")
    
    # Embed queries
    print(f"\n3. Embedding {NUM_QUERIES} queries...")
    query_embeddings, query_embed_time = embed_client.embed_batch(queries, batch_size=BATCH_SIZE)
    print(f"   ✓ Embedded {len(query_embeddings)} queries in {query_embed_time/1000:.1f}s")
    
    # Total embedding stats
    print(f"\n" + "-"*70)
    print("  EMBEDDING API STATS")
    print("-"*70)
    print(f"   Total API calls: {embed_client.metrics.total_calls}")
    print(f"   Total tokens: {embed_client.metrics.total_tokens:,}")
    print(f"   Total time: {embed_client.metrics.total_time_ms/1000:.1f}s")
    print(f"   Errors: {embed_client.metrics.errors}")
    print(f"   Throughput: {(NUM_DOCS + NUM_QUERIES) / (embed_client.metrics.total_time_ms/1000):.1f} docs/s")
    
    # Benchmark databases
    print(f"\n4. Benchmarking vector databases...")
    print("-"*70)
    
    adapters = [
        ToonDBAdapter(embed_client.dimension),
        ChromaDBAdapter(embed_client.dimension),
        LanceDBAdapter(embed_client.dimension),
    ]
    
    results = {}
    
    for adapter in adapters:
        print(f"\n   [{adapter.name}]")
        
        # Insert
        insert_time = adapter.insert(doc_embeddings)
        print(f"   Insert {NUM_DOCS} vectors: {insert_time:.1f}ms ({NUM_DOCS / (insert_time/1000):,.0f} vec/s)")
        
        # Search
        search_times = []
        for qe in query_embeddings:
            _, latency = adapter.search(qe, k=10)
            search_times.append(latency)
        
        avg_search = np.mean(search_times)
        p50_search = np.percentile(search_times, 50)
        p99_search = np.percentile(search_times, 99)
        
        print(f"   Search: {avg_search:.2f}ms avg, {p50_search:.2f}ms p50, {p99_search:.2f}ms p99")
        
        results[adapter.name] = {
            "insert_ms": insert_time,
            "search_avg_ms": avg_search,
            "search_p50_ms": p50_search,
            "search_p99_ms": p99_search,
        }
        
        adapter.cleanup()
    
    # Final summary
    print(f"\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    print(f"\n  Embedding API (Azure OpenAI):")
    print(f"    {NUM_DOCS + NUM_QUERIES} texts → {embed_client.metrics.total_time_ms/1000:.1f}s")
    print(f"    Throughput: {(NUM_DOCS + NUM_QUERIES) / (embed_client.metrics.total_time_ms/1000):.1f} docs/s")
    
    print(f"\n  Vector Database Performance:")
    print(f"  {'Database':<12} {'Insert (ms)':<15} {'Search p50':<12} {'Search p99':<12}")
    print(f"  {'-'*50}")
    
    for name, metrics in results.items():
        print(f"  {name:<12} {metrics['insert_ms']:<15.1f} {metrics['search_p50_ms']:<12.2f} {metrics['search_p99_ms']:<12.2f}")
    
    # Bottleneck analysis
    print(f"\n  📊 BOTTLENECK ANALYSIS")
    print(f"  {'-'*50}")
    
    embed_time = embed_client.metrics.total_time_ms
    toondb_insert = results.get("ToonDB", {}).get("insert_ms", 0)
    toondb_search = results.get("ToonDB", {}).get("search_avg_ms", 0) * NUM_QUERIES
    
    total_time = embed_time + toondb_insert + toondb_search
    
    print(f"  Embedding API:  {embed_time/1000:.1f}s ({embed_time/total_time*100:.1f}%)")
    print(f"  ToonDB Insert:  {toondb_insert/1000:.3f}s ({toondb_insert/total_time*100:.1f}%)")
    print(f"  ToonDB Search:  {toondb_search/1000:.3f}s ({toondb_search/total_time*100:.1f}%)")
    
    print(f"\n  🎯 Conclusion: Embedding API is {embed_time / (toondb_insert + toondb_search):.0f}x slower than ToonDB")
    print("="*70)


if __name__ == "__main__":
    main()
