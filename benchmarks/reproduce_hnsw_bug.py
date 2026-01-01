#!/usr/bin/env python3
"""
ToonDB HNSW Bug Reproduction Script

This script demonstrates a critical retrieval bug in ToonDB's HNSW implementation.

EXPECTED BEHAVIOR:
- When querying for an exact vector that was just inserted, it should be returned
  as the #1 result with distance ≈ 0.0

ACTUAL BEHAVIOR:
- The exact vector is NOT found in the top-k results
- Completely different vectors are returned with high distances (0.8+)

STEPS TO REPRODUCE:
1. Build ToonDB in release mode:
   cd /Users/sushanth/toondb && cargo build --release

2. Run this script:
   PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 benchmarks/reproduce_hnsw_bug.py

INDUSTRY STANDARD:
- HNSW should achieve >95% Recall@10 on random vectors
- Self-retrieval should always return the exact vector with dist=0
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))

from toondb import VectorIndex


def test_self_retrieval():
    """Test that inserted vectors can be retrieved by querying themselves."""
    print("=" * 70)
    print("  TEST 1: Self-Retrieval (query for exact inserted vectors)")
    print("=" * 70)
    
    dim = 128
    num_vectors = 100
    
    # Create index
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    # Generate and insert random vectors
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    print(f"\n  Inserting {num_vectors} vectors...")
    index.insert_batch(ids, vectors)
    print(f"  Index size: {len(index)} vectors")
    
    # Test: Query for each inserted vector
    print("\n  Testing self-retrieval:")
    failures = 0
    
    for test_id in [0, 25, 50, 75, 99]:
        query = vectors[test_id]
        results = index.search(query, k=5)
        
        result_ids = [int(r[0]) for r in results]
        result_dists = [float(r[1]) for r in results]
        
        found = test_id in result_ids
        position = result_ids.index(test_id) + 1 if found else None
        
        status = "✓ PASS" if found and position == 1 else "❌ FAIL"
        if not found:
            failures += 1
        
        print(f"\n    Query for ID {test_id}:")
        print(f"      Expected: ID={test_id}, distance≈0.0")
        print(f"      Actual top-3: {list(zip(result_ids[:3], [f'{d:.4f}' for d in result_dists[:3]]))}")
        print(f"      Status: {status}")
    
    print(f"\n  Result: {failures}/{5} tests FAILED")
    return failures == 0


def test_recall_at_k():
    """Test Recall@10 using brute-force ground truth."""
    print("\n" + "=" * 70)
    print("  TEST 2: Recall@10 (compare to brute-force ground truth)")
    print("=" * 70)
    
    dim = 128
    num_vectors = 1000
    num_queries = 100
    k = 10
    
    # Create index
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    # Generate vectors
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    print(f"\n  Inserting {num_vectors} vectors...")
    index.insert_batch(ids, vectors)
    
    # Compute ground truth using cosine distance (ToonDB's default)
    print("  Computing ground truth (brute-force cosine)...")
    
    # Normalize for cosine
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Cosine similarity -> distance
    similarities = queries_norm @ vectors_norm.T
    distances = 1.0 - similarities
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Test HNSW recall
    print(f"  Testing HNSW Recall@{k}...")
    
    recalls = []
    for i, query in enumerate(queries):
        results = index.search(query, k=k)
        predicted = set(int(r[0]) for r in results)
        gt = set(ground_truth[i].tolist())
        
        recall = len(predicted & gt) / len(gt) if gt else 1.0
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    min_recall = np.min(recalls)
    max_recall = np.max(recalls)
    
    print(f"\n  Results:")
    print(f"    Average Recall@{k}: {avg_recall:.1%}")
    print(f"    Min Recall@{k}: {min_recall:.1%}")
    print(f"    Max Recall@{k}: {max_recall:.1%}")
    
    # Industry standard: HNSW should achieve >90% recall
    expected_recall = 0.90
    status = "✓ PASS" if avg_recall >= expected_recall else "❌ FAIL"
    print(f"\n  Industry Standard: Recall@{k} >= {expected_recall:.0%}")
    print(f"  Status: {status}")
    
    return avg_recall >= expected_recall


def main():
    print("=" * 70)
    print("  TOONDB HNSW BUG REPRODUCTION")
    print("=" * 70)
    print("\n  This script demonstrates retrieval issues in ToonDB's HNSW.")
    
    test1_pass = test_self_retrieval()
    test2_pass = test_recall_at_k()
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (Self-Retrieval): {'✓ PASS' if test1_pass else '❌ FAIL'}")
    print(f"  Test 2 (Recall@10):      {'✓ PASS' if test2_pass else '❌ FAIL'}")
    
    if not test1_pass or not test2_pass:
        print("\n  ⚠️  CRITICAL: ToonDB HNSW has retrieval correctness issues!")
        print("     The index is fast but returns WRONG results.")
        print("\n  Expected: Both tests should PASS")
        print("  Actual: One or more tests FAILED")
        return 1
    else:
        print("\n  ✓ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
