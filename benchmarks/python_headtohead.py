#!/usr/bin/env python3
"""
TRUE Head-to-Head: ToonDB Python SDK vs ChromaDB
Both using Python APIs for fair comparison

Run with: python3 benchmarks/python_headtohead.py
"""

import time
import os
import shutil
import numpy as np
import json
from typing import List, Tuple

# Configuration
DIM = 128
NUM_VECTORS = 10_000
NUM_QUERIES = 100
TOP_K = 10

def generate_vectors(n: int, dim: int) -> np.ndarray:
    """Generate random unit vectors"""
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def benchmark_chromadb(vectors: np.ndarray, queries: np.ndarray):
    """Benchmark ChromaDB"""
    try:
        import chromadb
    except ImportError:
        print("ChromaDB not installed. Run: pip install chromadb")
        return None

    print("\n" + "="*60)
    print("ChromaDB (Python Native)")
    print("="*60)
    
    chroma_dir = "/tmp/chroma_h2h"
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.create_collection(
        name="benchmark",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Insert
    ids = [str(i) for i in range(len(vectors))]
    embeddings = vectors.tolist()
    
    start = time.perf_counter()
    batch_size = 5000
    for i in range(0, len(vectors), batch_size):
        end_idx = min(i + batch_size, len(vectors))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx]
        )
    insert_time = time.perf_counter() - start
    insert_rate = len(vectors) / insert_time
    print(f"Insert: {insert_time:.3f}s ({insert_rate:.0f} vec/sec)")
    
    # Query
    latencies = []
    for query in queries:
        start = time.perf_counter()
        results = collection.query(
            query_embeddings=[query.tolist()],
            n_results=TOP_K
        )
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    qps = 1000 / avg_latency
    
    print(f"Search: {avg_latency:.2f}ms avg, {p50:.2f}ms p50, {p99:.2f}ms p99")
    print(f"QPS: {qps:.0f}")
    
    shutil.rmtree(chroma_dir, ignore_errors=True)
    
    return {
        "insert_rate": insert_rate,
        "avg_latency": avg_latency,
        "p50": p50,
        "p99": p99,
        "qps": qps
    }

def benchmark_toondb_kv(vectors: np.ndarray, queries: np.ndarray):
    """Benchmark ToonDB Python SDK (key-value mode)"""
    try:
        from toondb import Database
    except ImportError:
        print("ToonDB Python SDK not installed.")
        print("Install with: cd toondb-python-sdk && pip install -e .")
        return None

    print("\n" + "="*60)
    print("ToonDB Python SDK (KV Mode)")
    print("="*60)
    
    toon_dir = "/tmp/toondb_h2h"
    if os.path.exists(toon_dir):
        shutil.rmtree(toon_dir)
    
    try:
        db = Database.open(toon_dir)
        
        # Insert vectors as JSON
        start = time.perf_counter()
        with db.transaction() as txn:
            for i, vec in enumerate(vectors):
                key = f"vec:{i:06d}".encode()
                value = json.dumps({"id": i, "embedding": vec.tolist()}).encode()
                txn.put(key, value)
        insert_time = time.perf_counter() - start
        insert_rate = len(vectors) / insert_time
        print(f"Insert: {insert_time:.3f}s ({insert_rate:.0f} vec/sec)")
        
        # "Search" - scan + brute force (no native HNSW in KV mode)
        latencies = []
        for query in queries:
            start = time.perf_counter()
            # Brute force search since KV mode doesn't have HNSW
            best_score = -float('inf')
            best_id = -1
            for key, value in db.scan(b"vec:", b"vec;"):
                data = json.loads(value.decode())
                vec = np.array(data["embedding"], dtype=np.float32)
                score = np.dot(query, vec)
                if score > best_score:
                    best_score = score
                    best_id = data["id"]
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        qps = 1000 / avg_latency
        
        print(f"Search (brute-force): {avg_latency:.2f}ms avg, {p50:.2f}ms p50")
        print(f"QPS: {qps:.0f}")
        print("Note: ToonDB KV mode uses brute-force. Use toondb-index for HNSW.")
        
        db.close()
        shutil.rmtree(toon_dir, ignore_errors=True)
        
        return {
            "insert_rate": insert_rate,
            "avg_latency": avg_latency,
            "p50": p50,
            "p99": p99,
            "qps": qps
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def benchmark_toondb_kv_insert_only(vectors: np.ndarray):
    """Benchmark just ToonDB insert performance"""
    try:
        from toondb import Database
    except ImportError:
        return None

    print("\n" + "="*60)
    print("ToonDB Python SDK - Insert Only Benchmark")
    print("="*60)
    
    toon_dir = "/tmp/toondb_insert"
    if os.path.exists(toon_dir):
        shutil.rmtree(toon_dir)
    
    try:
        db = Database.open(toon_dir)
        
        # Test 1: Single transaction bulk insert
        start = time.perf_counter()
        with db.transaction() as txn:
            for i, vec in enumerate(vectors):
                key = f"vec:{i:06d}".encode()
                value = json.dumps({"id": i, "embedding": vec.tolist()}).encode()
                txn.put(key, value)
        bulk_time = time.perf_counter() - start
        bulk_rate = len(vectors) / bulk_time
        print(f"Bulk insert (1 txn): {bulk_time:.3f}s ({bulk_rate:.0f} vec/sec)")
        
        db.close()
        shutil.rmtree(toon_dir, ignore_errors=True)
        
        # Test 2: Many small transactions
        db = Database.open(toon_dir)
        start = time.perf_counter()
        for i, vec in enumerate(vectors[:1000]):  # Only 1K for small txns
            with db.transaction() as txn:
                key = f"vec:{i:06d}".encode()
                value = json.dumps({"id": i, "embedding": vec.tolist()}).encode()
                txn.put(key, value)
        small_txn_time = time.perf_counter() - start
        small_txn_rate = 1000 / small_txn_time
        print(f"Small txns (1K): {small_txn_time:.3f}s ({small_txn_rate:.0f} vec/sec)")
        
        db.close()
        shutil.rmtree(toon_dir, ignore_errors=True)
        
        return {"bulk_rate": bulk_rate, "small_txn_rate": small_txn_rate}
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("="*60)
    print("   TRUE HEAD-TO-HEAD: ToonDB Python SDK vs ChromaDB")
    print("="*60)
    print(f"\nConfig: {NUM_VECTORS} vectors, {DIM}-dim, {NUM_QUERIES} queries, top-{TOP_K}")
    
    # Generate test data
    print("\nGenerating test vectors...")
    vectors = generate_vectors(NUM_VECTORS, DIM)
    queries = generate_vectors(NUM_QUERIES, DIM)
    
    results = {}
    
    # ChromaDB
    results['chromadb'] = benchmark_chromadb(vectors, queries)
    
    # ToonDB
    results['toondb_kv'] = benchmark_toondb_kv(vectors[:1000], queries[:10])  # Smaller for brute-force
    results['toondb_insert'] = benchmark_toondb_kv_insert_only(vectors)
    
    # Summary
    print("\n" + "="*60)
    print("   SUMMARY: Python API Comparison")
    print("="*60)
    
    print(f"\n{'Database':<25} {'Insert (vec/s)':<18} {'Search (ms)':<15}")
    print("-"*60)
    
    if results.get('chromadb'):
        r = results['chromadb']
        print(f"{'ChromaDB':<25} {r['insert_rate']:.0f}{'':<12} {r['avg_latency']:.2f}")
    
    if results.get('toondb_insert'):
        r = results['toondb_insert']
        print(f"{'ToonDB (bulk insert)':<25} {r['bulk_rate']:.0f}{'':<12} {'N/A (KV mode)'}")
        print(f"{'ToonDB (small txns)':<25} {r['small_txn_rate']:.0f}{'':<12} {'N/A (KV mode)'}")
    
    print("\n" + "="*60)
    print("   NOTE")
    print("="*60)
    print("""
• ToonDB Python SDK is for KEY-VALUE storage, not vector search
• For vector search, use toondb-index (Rust HNSW implementation)
• ChromaDB is purpose-built for embeddings with built-in HNSW

Fair comparison metrics:
• INSERT: ToonDB vs ChromaDB (both via Python)
• SEARCH: Only meaningful for ChromaDB (HNSW) vs ToonDB-Index (Rust)
""")

if __name__ == "__main__":
    main()
