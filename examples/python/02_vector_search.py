#!/usr/bin/env python3
"""
ToonDB Vector Search Example
============================

This example demonstrates ToonDB's vector search capabilities:
1. Creating a vector index
2. Inserting vectors (single and batch)
3. Searching for nearest neighbors
4. Measuring recall accuracy

Expected Output:
    ✓ Index created (768 dimensions)
    ✓ Batch insert: 1000 vectors
    ✓ Self-retrieval: 100% accuracy
    ✓ Recall@10: >90%
    ✓ Search latency: <10ms

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 examples/python/02_vector_search.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

import numpy as np


def main():
    print("=" * 60)
    print("  ToonDB Vector Search Example")
    print("=" * 60)
    
    try:
        from toondb import VectorIndex
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Set TOONDB_LIB_PATH to target/release directory")
        return 1
    
    # Configuration
    dimension = 768
    n_vectors = 1000
    k = 10
    
    print(f"\n[Config] Dimension: {dimension}, Vectors: {n_vectors}, k: {k}")
    
    # 1. Create index
    print("\n[1] Creating vector index...")
    index = VectorIndex(
        dimension=dimension,
        max_connections=16,
        ef_construction=100
    )
    print(f"    ✓ Index created ({dimension} dimensions)")
    
    # 2. Generate test vectors
    print("\n[2] Generating test vectors...")
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    # Normalize for cosine similarity
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    print(f"    Generated {n_vectors} normalized vectors")
    
    # 3. Insert vectors
    print("\n[3] Inserting vectors...")
    ids = np.arange(n_vectors, dtype=np.uint64)
    
    start = time.perf_counter()
    index.insert_batch(ids, vectors)
    insert_time = time.perf_counter() - start
    
    rate = n_vectors / insert_time
    print(f"    ✓ Batch insert: {n_vectors} vectors in {insert_time:.2f}s ({rate:.0f} vec/s)")
    
    # 4. Test self-retrieval
    print("\n[4] Testing self-retrieval...")
    correct = 0
    for i in range(10):
        query = vectors[i]
        results = index.search(query, k=1)
        if results and results[0][0] == i:
            correct += 1
    
    accuracy = correct / 10 * 100
    status = "✓" if accuracy == 100 else "⚠"
    print(f"    {status} Self-retrieval: {accuracy:.0f}% accuracy")
    
    # 5. Test recall@k
    print("\n[5] Testing recall@10...")
    num_queries = 50
    recalls = []
    
    for i in range(num_queries):
        query = vectors[i]
        
        # ToonDB results
        results = index.search(query, k=k)
        toondb_ids = set([r[0] for r in results])
        
        # Ground truth (brute-force cosine similarity)
        similarities = np.dot(vectors, query)
        ground_truth = set(np.argsort(similarities)[-k:])
        
        recall = len(toondb_ids & ground_truth) / k
        recalls.append(recall)
    
    avg_recall = np.mean(recalls) * 100
    status = "✓" if avg_recall >= 90 else "⚠"
    print(f"    {status} Recall@{k}: {avg_recall:.1f}%")
    
    # 6. Measure search latency
    print("\n[6] Measuring search latency...")
    latencies = []
    
    for _ in range(100):
        query = vectors[np.random.randint(n_vectors)]
        start = time.perf_counter()
        index.search(query, k=k)
        latencies.append((time.perf_counter() - start) * 1000)
    
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    
    status = "✓" if p50 < 10 else "⚠"
    print(f"    {status} Search latency: p50={p50:.2f}ms, p99={p99:.2f}ms")
    
    print("\n" + "=" * 60)
    print("  ✅ Vector search example complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
