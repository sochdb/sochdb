#!/usr/bin/env python3
"""
Real-World Use Case Benchmark: ToonDB vs ChromaDB vs LanceDB

Compares performance on production-like scenarios:
1. Customer Support RAG (multi-tenant, filtering)
2. E-commerce Search (multi-vector, facets)
3. Semantic Deduplication (threshold matching)
4. Code Search (hybrid ranking)

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 benchmarks/realworld_comparison.py
"""

import os
import sys
import time
import json
import random
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Embedding Helper
# =============================================================================

class EmbeddingClient:
    """Azure OpenAI embedding client."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        self.dimension = 1536
        self._cache = {}
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts with caching."""
        cache_key = tuple(texts)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/embeddings?api-version={self.version}"
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = requests.post(url, headers=headers, json={"input": batch, "model": self.deployment})
            response.raise_for_status()
            embeddings = [item["embedding"] for item in response.json()["data"]]
            all_embeddings.extend(embeddings)
        
        result = np.array(all_embeddings, dtype=np.float32)
        self._cache[cache_key] = result
        return result


# =============================================================================
# Database Adapters
# =============================================================================

class ToonDBAdapter:
    """ToonDB vector search adapter."""
    
    def __init__(self, dimension: int):
        from toondb import VectorIndex
        self.index = VectorIndex(dimension=dimension, max_connections=32, ef_construction=200)
        self.metadata: Dict[int, Dict] = {}
        self.next_id = 0
    
    def insert(self, vectors: np.ndarray, metadata_list: List[Dict]) -> float:
        start = time.perf_counter()
        ids = np.arange(self.next_id, self.next_id + len(vectors), dtype=np.uint64)
        self.index.insert_batch(ids, vectors)
        for i, meta in enumerate(metadata_list):
            self.metadata[self.next_id + i] = meta
        self.next_id += len(vectors)
        return time.perf_counter() - start
    
    def search(self, query: np.ndarray, k: int, filters: Optional[Dict] = None) -> Tuple[List[Tuple[int, float]], float]:
        start = time.perf_counter()
        # Over-fetch for filtering
        fetch_k = k * 5 if filters else k
        results = self.index.search(query, k=fetch_k)
        
        if filters:
            filtered = []
            for idx, score in results:
                meta = self.metadata.get(int(idx), {})
                if self._matches_filter(meta, filters):
                    filtered.append((int(idx), float(score)))
                    if len(filtered) >= k:
                        break
            results = filtered
        else:
            results = [(int(idx), float(score)) for idx, score in results[:k]]
        
        return results, time.perf_counter() - start
    
    def _matches_filter(self, meta: Dict, filters: Dict) -> bool:
        for key, value in filters.items():
            if key not in meta:
                return False
            if meta[key] != value:
                return False
        return True
    
    def cleanup(self):
        pass


class ChromaDBAdapter:
    """ChromaDB vector search adapter."""
    
    def __init__(self, dimension: int):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=f"bench_{random.randint(0, 999999)}",
            metadata={"hnsw:space": "cosine"}
        )
        self.dimension = dimension
    
    def insert(self, vectors: np.ndarray, metadata_list: List[Dict]) -> float:
        start = time.perf_counter()
        ids = [f"id_{i}" for i in range(len(vectors))]
        self.collection.add(
            embeddings=vectors.tolist(),
            metadatas=metadata_list,
            ids=ids
        )
        return time.perf_counter() - start
    
    def search(self, query: np.ndarray, k: int, filters: Optional[Dict] = None) -> Tuple[List[Tuple[int, float]], float]:
        start = time.perf_counter()
        
        where_filter = None
        if filters:
            if len(filters) == 1:
                key, value = list(filters.items())[0]
                where_filter = {key: {"$eq": value}}
            else:
                where_filter = {"$and": [{key: {"$eq": value}} for key, value in filters.items()]}
        
        results = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
            where=where_filter
        )
        
        output = []
        if results['ids'] and results['ids'][0]:
            for i, id_str in enumerate(results['ids'][0]):
                idx = int(id_str.split('_')[1])
                dist = results['distances'][0][i] if results['distances'] else 0.0
                output.append((idx, dist))
        
        return output, time.perf_counter() - start
    
    def cleanup(self):
        pass


class LanceDBAdapter:
    """LanceDB vector search adapter."""
    
    def __init__(self, dimension: int):
        import lancedb
        self.tmp_dir = tempfile.mkdtemp()
        self.db = lancedb.connect(self.tmp_dir)
        self.dimension = dimension
        self.table = None
    
    def insert(self, vectors: np.ndarray, metadata_list: List[Dict]) -> float:
        start = time.perf_counter()
        
        data = []
        for i, (vec, meta) in enumerate(zip(vectors, metadata_list)):
            row = {"id": i, "vector": vec.tolist()}
            row.update(meta)
            data.append(row)
        
        self.table = self.db.create_table(f"bench_{random.randint(0, 999999)}", data, mode="overwrite")
        
        return time.perf_counter() - start
    
    def search(self, query: np.ndarray, k: int, filters: Optional[Dict] = None) -> Tuple[List[Tuple[int, float]], float]:
        start = time.perf_counter()
        
        search = self.table.search(query.tolist())
        
        if filters:
            filter_str = " AND ".join([f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value}" for key, value in filters.items()])
            search = search.where(filter_str)
        
        results = search.limit(k).to_list()
        
        output = [(row['id'], row.get('_distance', 0.0)) for row in results]
        
        return output, time.perf_counter() - start
    
    def cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# =============================================================================
# Benchmark Scenarios
# =============================================================================

def run_customer_support_benchmark(embed_client: EmbeddingClient, adapters: Dict[str, object]):
    """Multi-tenant customer support with filtering."""
    print("\n" + "="*70)
    print("  SCENARIO 1: Customer Support RAG (Multi-tenant + ACL)")
    print("="*70)
    
    # Generate data
    tenants = ["acme", "globex", "initech"]
    access_levels = [0, 1, 2]
    
    articles = []
    for tenant in tenants:
        for i in range(20):
            articles.append({
                "text": f"{tenant} support article {i}: troubleshooting login issues and password recovery",
                "tenant": tenant,
                "access_level": random.choice(access_levels),
            })
    
    texts = [a["text"] for a in articles]
    embeddings = embed_client.embed(texts)
    metadata = [{"tenant": a["tenant"], "access_level": a["access_level"]} for a in articles]
    
    queries = [
        ("How do I reset my password?", {"tenant": "acme", "access_level": 1}),
        ("Billing invoice download", {"tenant": "globex"}),
        ("Dashboard performance", {"tenant": "initech", "access_level": 2}),
    ]
    query_texts = [q[0] for q in queries]
    query_embeddings = embed_client.embed(query_texts)
    
    results = {}
    
    for name, adapter in adapters.items():
        print(f"\n   [{name}]")
        
        # Insert
        insert_time = adapter.insert(embeddings, metadata)
        print(f"   Insert: {len(articles)} docs in {insert_time*1000:.1f}ms ({len(articles)/insert_time:.0f} doc/s)")
        
        # Search
        search_times = []
        for i, (query_text, filters) in enumerate(queries):
            _, latency = adapter.search(query_embeddings[i], k=5, filters=filters)
            search_times.append(latency * 1000)
        
        avg_search = np.mean(search_times)
        print(f"   Search: {avg_search:.2f}ms avg (filtered)")
        
        results[name] = {"insert_ms": insert_time * 1000, "search_ms": avg_search}
    
    return results


def run_ecommerce_benchmark(embed_client: EmbeddingClient, adapters: Dict[str, object]):
    """E-commerce product search with facets."""
    print("\n" + "="*70)
    print("  SCENARIO 2: E-commerce Search (Multi-vector + Facets)")
    print("="*70)
    
    # Generate products
    brands = ["Nike", "Adidas", "Puma", "Reebok"]
    categories = ["shoes", "apparel", "accessories"]
    
    products = []
    for i in range(100):
        products.append({
            "text": f"{random.choice(brands)} {random.choice(['running', 'training', 'casual'])} {random.choice(categories)} product {i}",
            "brand": random.choice(brands),
            "category": random.choice(categories),
            "price": random.randint(50, 200),
        })
    
    texts = [p["text"] for p in products]
    embeddings = embed_client.embed(texts)
    metadata = [{"brand": p["brand"], "category": p["category"]} for p in products]
    
    queries = [
        ("running shoes", {"brand": "Nike"}),
        ("training apparel", {"category": "apparel"}),
        ("casual accessories", {}),
    ]
    query_texts = [q[0] for q in queries]
    query_embeddings = embed_client.embed(query_texts)
    
    results = {}
    
    for name, adapter in adapters.items():
        print(f"\n   [{name}]")
        
        insert_time = adapter.insert(embeddings, metadata)
        print(f"   Insert: {len(products)} products in {insert_time*1000:.1f}ms")
        
        search_times = []
        for i, (query_text, filters) in enumerate(queries):
            _, latency = adapter.search(query_embeddings[i], k=10, filters=filters if filters else None)
            search_times.append(latency * 1000)
        
        avg_search = np.mean(search_times)
        print(f"   Search: {avg_search:.2f}ms avg")
        
        results[name] = {"insert_ms": insert_time * 1000, "search_ms": avg_search}
    
    return results


def run_dedup_benchmark(embed_client: EmbeddingClient, adapters: Dict[str, object]):
    """Semantic deduplication benchmark."""
    print("\n" + "="*70)
    print("  SCENARIO 3: Semantic Deduplication")
    print("="*70)
    
    # Generate tickets with duplicates
    base_tickets = [
        "Login not working, authentication error",
        "Payment failed, credit card declined",
        "App crashes on startup",
        "Cannot upload files, timeout error",
        "Password reset email not received",
    ]
    
    tickets = []
    for base in base_tickets:
        tickets.append(base)
        # Add variations
        tickets.append(f"{base} - urgent")
        tickets.append(f"Issue: {base}")
    
    embeddings = embed_client.embed(tickets)
    metadata = [{"id": i} for i in range(len(tickets))]
    
    # Test queries (should find duplicates)
    test_queries = [
        "Can't log in, auth failing",
        "Payment not processing",
        "Application won't start",
    ]
    query_embeddings = embed_client.embed(test_queries)
    
    results = {}
    
    for name, adapter in adapters.items():
        print(f"\n   [{name}]")
        
        insert_time = adapter.insert(embeddings, metadata)
        print(f"   Insert: {len(tickets)} tickets in {insert_time*1000:.1f}ms")
        
        search_times = []
        for qe in query_embeddings:
            _, latency = adapter.search(qe, k=5)
            search_times.append(latency * 1000)
        
        avg_search = np.mean(search_times)
        print(f"   Search: {avg_search:.2f}ms avg (duplicate detection)")
        
        results[name] = {"insert_ms": insert_time * 1000, "search_ms": avg_search}
    
    return results


def run_code_search_benchmark(embed_client: EmbeddingClient, adapters: Dict[str, object]):
    """Code semantic search benchmark."""
    print("\n" + "="*70)
    print("  SCENARIO 4: Code Search")
    print("="*70)
    
    # Generate code snippets
    snippets = [
        "func RateLimiter(requests int) *Limiter { return &Limiter{rate: requests} }",
        "def rate_limiter(max_requests): return RateLimiter(max_requests)",
        "func RefreshToken(token string) (*TokenPair, error) { return generatePair() }",
        "async def refresh_token(token: str) -> TokenPair: return generate_pair()",
        "func NewConnectionPool(dsn string) (*Pool, error) { return pgxpool.New(dsn) }",
        "def create_pool(dsn: str) -> Pool: return asyncpg.create_pool(dsn)",
        "func VectorSearch(query []float32, k int) []Result { return index.Search(query, k) }",
        "def vector_search(query: np.ndarray, k: int) -> List[Result]: return index.search(query, k)",
    ]
    
    embeddings = embed_client.embed(snippets)
    metadata = [{"lang": "go" if "func" in s else "python"} for s in snippets]
    
    queries = [
        ("rate limiter implementation", None),
        ("token refresh", None),
        ("database connection pool", {"lang": "go"}),
        ("vector similarity search", {"lang": "python"}),
    ]
    query_texts = [q[0] for q in queries]
    query_embeddings = embed_client.embed(query_texts)
    
    results = {}
    
    for name, adapter in adapters.items():
        print(f"\n   [{name}]")
        
        insert_time = adapter.insert(embeddings, metadata)
        print(f"   Insert: {len(snippets)} snippets in {insert_time*1000:.1f}ms")
        
        search_times = []
        for i, (_, filters) in enumerate(queries):
            _, latency = adapter.search(query_embeddings[i], k=3, filters=filters)
            search_times.append(latency * 1000)
        
        avg_search = np.mean(search_times)
        print(f"   Search: {avg_search:.2f}ms avg")
        
        results[name] = {"insert_ms": insert_time * 1000, "search_ms": avg_search}
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("  REAL-WORLD USE CASE BENCHMARK")
    print("  ToonDB vs ChromaDB vs LanceDB")
    print("="*70)
    
    embed_client = EmbeddingClient()
    dimension = embed_client.dimension
    
    all_results = {}
    
    scenarios = [
        ("Customer Support RAG", run_customer_support_benchmark),
        ("E-commerce Search", run_ecommerce_benchmark),
        ("Semantic Dedup", run_dedup_benchmark),
        ("Code Search", run_code_search_benchmark),
    ]
    
    for scenario_name, scenario_fn in scenarios:
        # Create fresh adapters for each scenario
        adapters = {
            "ToonDB": ToonDBAdapter(dimension),
            "ChromaDB": ChromaDBAdapter(dimension),
            "LanceDB": LanceDBAdapter(dimension),
        }
        
        try:
            results = scenario_fn(embed_client, adapters)
            all_results[scenario_name] = results
        finally:
            for adapter in adapters.values():
                adapter.cleanup()
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    print("\n{:<25} {:>12} {:>12} {:>12}".format("Scenario", "ToonDB", "ChromaDB", "LanceDB"))
    print("-"*70)
    
    for scenario_name, results in all_results.items():
        toondb_ms = results.get("ToonDB", {}).get("search_ms", 0)
        chroma_ms = results.get("ChromaDB", {}).get("search_ms", 0)
        lance_ms = results.get("LanceDB", {}).get("search_ms", 0)
        
        print("{:<25} {:>10.2f}ms {:>10.2f}ms {:>10.2f}ms".format(
            scenario_name, toondb_ms, chroma_ms, lance_ms
        ))
    
    # Compute averages
    print("-"*70)
    
    toondb_avg = np.mean([r.get("ToonDB", {}).get("search_ms", 0) for r in all_results.values()])
    chroma_avg = np.mean([r.get("ChromaDB", {}).get("search_ms", 0) for r in all_results.values()])
    lance_avg = np.mean([r.get("LanceDB", {}).get("search_ms", 0) for r in all_results.values()])
    
    print("{:<25} {:>10.2f}ms {:>10.2f}ms {:>10.2f}ms".format(
        "AVERAGE", toondb_avg, chroma_avg, lance_avg
    ))
    
    print("\n" + "="*70)
    print("  🏆 SPEEDUP vs ToonDB")
    print("="*70)
    
    if toondb_avg > 0:
        print(f"\n  ChromaDB: {chroma_avg/toondb_avg:.1f}x slower")
        print(f"  LanceDB:  {lance_avg/toondb_avg:.1f}x slower")
    
    print("\n  ✓ Real-world benchmark completed!")
    print("="*70)


if __name__ == "__main__":
    main()
