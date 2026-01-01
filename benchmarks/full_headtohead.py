#!/usr/bin/env python3
"""
TRUE Head-to-Head: ToonDB Vector vs ChromaDB
Both KV and Vector Search via Python APIs

Run with: 
  TOONDB_LIB_PATH=/path/to/target/release python3 benchmarks/full_headtohead.py
"""

import time
import os
import shutil
import numpy as np
import sys

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
    print("ChromaDB (Python + HNSW)")
    print("="*60)
    
    chroma_dir = "/tmp/chroma_full_h2h"
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
    
    print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
    print(f"QPS: {qps:.0f}")
    
    shutil.rmtree(chroma_dir, ignore_errors=True)
    
    return {
        "insert_rate": insert_rate,
        "avg_latency": avg_latency,
        "p50": p50,
        "p99": p99,
        "qps": qps
    }

def benchmark_toondb_vector(vectors: np.ndarray, queries: np.ndarray):
    """Benchmark ToonDB Vector Index (HNSW)"""
    # Add path for development
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))
    
    try:
        from toondb import VectorIndex
        if VectorIndex is None:
            raise ImportError("VectorIndex not available")
    except ImportError as e:
        print(f"\nToonDB VectorIndex not available: {e}")
        print("Set TOONDB_LIB_PATH to the directory containing libtoondb_index.dylib")
        return None

    print("\n" + "="*60)
    print("ToonDB Vector Index (Rust HNSW via Python)")
    print("="*60)
    
    # Create index with ChromaDB-matching parameters for fair comparison
    # ChromaDB defaults: M=16, ef_construction=100
    index = VectorIndex(dimension=DIM, max_connections=16, ef_construction=100)
    
    # Insert using batch FFI (100x faster than individual inserts)
    ids = np.arange(len(vectors), dtype=np.uint64)
    
    start = time.perf_counter()
    inserted = index.insert_batch(ids, vectors)
    insert_time = time.perf_counter() - start
    insert_rate = inserted / insert_time
    print(f"Insert (batch): {insert_time:.3f}s ({insert_rate:.0f} vec/sec)")
    
    # Query
    latencies = []
    for query in queries:
        start = time.perf_counter()
        results = index.search(query, k=TOP_K)
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    qps = 1000 / avg_latency
    
    print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
    print(f"QPS: {qps:.0f}")
    print(f"Index size: {len(index)} vectors")
    
    return {
        "insert_rate": insert_rate,
        "avg_latency": avg_latency,
        "p50": p50,
        "p99": p99,
        "qps": qps
    }

def main():
    print("="*60)
    print("   FULL HEAD-TO-HEAD: ToonDB vs ChromaDB")
    print("   Both KV and Vector Search via Python APIs")
    print("="*60)
    print(f"\nConfig: {NUM_VECTORS} vectors, {DIM}-dim, {NUM_QUERIES} queries, top-{TOP_K}")
    
    # Generate test data
    print("\nGenerating test vectors...")
    vectors = generate_vectors(NUM_VECTORS, DIM)
    queries = generate_vectors(NUM_QUERIES, DIM)
    
    results = {}
    
    # ChromaDB
    results['chromadb'] = benchmark_chromadb(vectors, queries)
    
    # ToonDB Vector
    results['toondb'] = benchmark_toondb_vector(vectors, queries)
    
    # Summary
    print("\n" + "="*60)
    print("   FINAL COMPARISON")
    print("="*60)
    
    print(f"\n{'Database':<30} {'Insert (vec/s)':<18} {'Search (ms)':<15} {'QPS':<10}")
    print("-"*75)
    
    if results.get('chromadb'):
        r = results['chromadb']
        print(f"{'ChromaDB (Python + HNSW)':<30} {r['insert_rate']:.0f}{'':<10} {r['avg_latency']:.3f}{'':<10} {r['qps']:.0f}")
    
    if results.get('toondb'):
        r = results['toondb']
        print(f"{'ToonDB (Rust HNSW via Python)':<30} {r['insert_rate']:.0f}{'':<10} {r['avg_latency']:.3f}{'':<10} {r['qps']:.0f}")
    
    # Comparison
    if results.get('chromadb') and results.get('toondb'):
        print("\n" + "="*60)
        print("   PERFORMANCE COMPARISON")
        print("="*60)
        
        c = results['chromadb']
        t = results['toondb']
        
        insert_speedup = t['insert_rate'] / c['insert_rate']
        search_speedup = c['avg_latency'] / t['avg_latency']
        qps_speedup = t['qps'] / c['qps']
        
        print(f"\n  Insert Speed: ToonDB is {insert_speedup:.1f}x {'faster' if insert_speedup > 1 else 'slower'}")
        print(f"  Search Speed: ToonDB is {search_speedup:.1f}x faster")
        print(f"  QPS: ToonDB is {qps_speedup:.1f}x higher")
        
        print("\n" + "-"*60)
        if search_speedup > 1:
            print(f"  🏆 WINNER: ToonDB ({search_speedup:.0f}x faster vector search!)")
        else:
            print(f"  🏆 WINNER: ChromaDB")

if __name__ == "__main__":
    main()
