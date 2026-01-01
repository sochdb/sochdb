#!/usr/bin/env python3
"""
Vector Database Benchmark: ChromaDB vs ToonDB-Index vs SQLite-VSS

Run with: python3 vector_db_benchmark.py

Prereqs:
  pip install chromadb numpy
"""

import time
import os
import shutil
import numpy as np
from typing import List, Tuple

# Configuration
DIM = 768  # Common embedding dimension
NUM_VECTORS = 10_000
NUM_QUERIES = 100
TOP_K = 10

def generate_vectors(n: int, dim: int) -> np.ndarray:
    """Generate random unit vectors"""
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def brute_force_search(vectors: np.ndarray, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """Baseline brute-force search for recall calculation"""
    scores = np.dot(vectors, query)
    top_k_indices = np.argsort(scores)[-k:][::-1]
    return [(int(i), float(scores[i])) for i in top_k_indices]

def benchmark_chromadb(vectors: np.ndarray, queries: np.ndarray):
    """Benchmark ChromaDB"""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("ChromaDB not installed. Run: pip install chromadb")
        return None

    print("\n--- ChromaDB ---")
    
    # Create client
    chroma_dir = "/tmp/chroma_bench"
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
    # ChromaDB has batch limits, insert in chunks
    batch_size = 5000
    for i in range(0, len(vectors), batch_size):
        end_idx = min(i + batch_size, len(vectors))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx]
        )
    insert_time = time.perf_counter() - start
    print(f"Insert {len(vectors)} vectors: {insert_time:.3f}s ({len(vectors)/insert_time:.0f} vec/sec)")
    
    # Query
    latencies = []
    for query in queries:
        start = time.perf_counter()
        results = collection.query(
            query_embeddings=[query.tolist()],
            n_results=TOP_K
        )
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"Query latency: {avg_latency:.2f}ms avg, {p99_latency:.2f}ms p99")
    print(f"QPS: {1000 / avg_latency:.0f}")
    
    # Cleanup
    client = None
    shutil.rmtree(chroma_dir, ignore_errors=True)
    
    return {"insert_time": insert_time, "avg_latency": avg_latency, "p99_latency": p99_latency}

def benchmark_numpy_baseline(vectors: np.ndarray, queries: np.ndarray):
    """Benchmark brute-force NumPy baseline"""
    print("\n--- NumPy Brute-Force (Baseline) ---")
    
    # Insert is just storing in memory
    start = time.perf_counter()
    stored = vectors.copy()
    insert_time = time.perf_counter() - start
    print(f"Insert {len(vectors)} vectors: {insert_time:.6f}s (memory copy)")
    
    # Query with dot product
    latencies = []
    for query in queries:
        start = time.perf_counter()
        scores = np.dot(stored, query)
        top_k_indices = np.argsort(scores)[-TOP_K:][::-1]
        _ = [(int(i), float(scores[i])) for i in top_k_indices]
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"Query latency: {avg_latency:.2f}ms avg, {p99_latency:.2f}ms p99")
    print(f"QPS: {1000 / avg_latency:.0f}")
    
    return {"insert_time": insert_time, "avg_latency": avg_latency, "p99_latency": p99_latency}

def benchmark_toondb_index_subprocess():
    """Run ToonDB index benchmarks via subprocess"""
    print("\n--- ToonDB-Index (HNSW) ---")
    print("(Run via: cd toondb-index && cargo run --release --bin hnsw_perf)")
    print("See toondb-index/benches/ for detailed benchmarks")
    return None

def main():
    print("=" * 60)
    print("Vector Database Benchmark")
    print(f"Vectors: {NUM_VECTORS}, Dimension: {DIM}, Queries: {NUM_QUERIES}")
    print("=" * 60)
    
    # Generate test data
    print("\nGenerating random vectors...")
    np.random.seed(42)
    vectors = generate_vectors(NUM_VECTORS, DIM)
    queries = generate_vectors(NUM_QUERIES, DIM)
    print(f"Generated {NUM_VECTORS} vectors of dim {DIM}")
    
    results = {}
    
    # NumPy baseline
    results['numpy'] = benchmark_numpy_baseline(vectors, queries)
    
    # ChromaDB
    results['chromadb'] = benchmark_chromadb(vectors, queries)
    
    # ToonDB (via subprocess - needs Rust compiled)
    benchmark_toondb_index_subprocess()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Database':<20} {'Insert (vec/s)':<18} {'Query (ms)':<15} {'QPS':<10}")
    print("-" * 60)
    
    if results.get('numpy'):
        r = results['numpy']
        print(f"{'NumPy (brute)':<20} {'N/A (memory)':<18} {r['avg_latency']:.2f}{'':<11} {1000/r['avg_latency']:.0f}")
    
    if results.get('chromadb'):
        r = results['chromadb']
        print(f"{'ChromaDB (HNSW)':<20} {NUM_VECTORS/r['insert_time']:.0f}{'':<12} {r['avg_latency']:.2f}{'':<11} {1000/r['avg_latency']:.0f}")
    
    print("\nNote: ToonDB-Vector benchmarks run separately via Rust")
    print("Run: cargo run --release -p engine-rs --bin bench")

if __name__ == "__main__":
    main()
