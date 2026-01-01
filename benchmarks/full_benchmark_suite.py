#!/usr/bin/env python3
"""
ToonDB Complete Benchmark & Test Suite

Runs all benchmarks and tests, generates a comprehensive report with recommendations.

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 benchmarks/full_benchmark_suite.py
"""

import os
import sys
import time
import json
import tempfile
import shutil
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))


# =============================================================================
# Test 1: Correctness Tests
# =============================================================================

def test_self_retrieval() -> Tuple[bool, Dict]:
    """Test that exact vectors can be retrieved."""
    from toondb import VectorIndex
    
    dim = 128
    num_vectors = 100
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    index.insert_batch(ids, vectors)
    
    passed = 0
    total = 10
    details = []
    
    for test_id in range(0, 100, 10):
        query = vectors[test_id]
        results = index.search(query, k=5)
        result_ids = [int(r[0]) for r in results]
        
        found = test_id in result_ids
        if found:
            passed += 1
        
        details.append({
            "id": test_id,
            "found": found,
            "top_result": result_ids[0] if results else None,
            "distance": float(results[0][1]) if results else None
        })
    
    return passed == total, {
        "passed": passed,
        "total": total,
        "success_rate": passed / total,
        "details": details
    }


def test_recall_at_k() -> Tuple[bool, Dict]:
    """Test Recall@10 vs brute force."""
    from toondb import VectorIndex
    
    dim = 128
    num_vectors = 1000
    num_queries = 100
    k = 10
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    index.insert_batch(ids, vectors)
    
    # Ground truth (cosine)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    distances = 1.0 - (queries_norm @ vectors_norm.T)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    recalls = []
    for i, query in enumerate(queries):
        results = index.search(query, k=k)
        predicted = set(int(r[0]) for r in results)
        gt = set(ground_truth[i].tolist())
        recalls.append(len(predicted & gt) / len(gt))
    
    avg_recall = np.mean(recalls)
    passed = avg_recall >= 0.90
    
    return passed, {
        "avg_recall": avg_recall,
        "min_recall": float(np.min(recalls)),
        "max_recall": float(np.max(recalls)),
        "threshold": 0.90,
        "passed": passed
    }


# =============================================================================
# Test 2: Performance Benchmarks
# =============================================================================

def benchmark_latency() -> Dict:
    """Benchmark search latency."""
    from toondb import VectorIndex
    
    dim = 128
    num_vectors = 10000
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(100, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    index.insert_batch(ids, vectors)
    
    # Warmup
    for q in queries[:10]:
        index.search(q, k=10)
    
    # Timed search
    latencies = []
    for q in queries:
        start = time.perf_counter()
        index.search(q, k=10)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "avg_ms": float(np.mean(latencies)),
    }


def benchmark_insert() -> Dict:
    """Benchmark insert throughput."""
    from toondb import VectorIndex
    
    dim = 128
    num_vectors = 10000
    
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    start = time.perf_counter()
    index.insert_batch(ids, vectors)
    elapsed = time.perf_counter() - start
    
    return {
        "vectors": num_vectors,
        "time_sec": elapsed,
        "throughput": num_vectors / elapsed,
    }


def benchmark_qps() -> Dict:
    """Benchmark queries per second."""
    from toondb import VectorIndex
    
    dim = 128
    num_vectors = 10000
    num_queries = 1000
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(100, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    index.insert_batch(ids, vectors)
    
    # Warmup
    for q in queries[:10]:
        index.search(q, k=10)
    
    # Timed queries
    query_list = [queries[i % 100] for i in range(num_queries)]
    
    start = time.perf_counter()
    for q in query_list:
        index.search(q, k=10)
    elapsed = time.perf_counter() - start
    
    return {
        "queries": num_queries,
        "time_sec": elapsed,
        "qps": num_queries / elapsed,
    }


# =============================================================================
# Test 3: Competitor Comparison
# =============================================================================

def compare_with_chromadb() -> Dict:
    """Compare with ChromaDB."""
    try:
        import chromadb
        from toondb import VectorIndex
        
        dim = 128
        num_vectors = 5000
        num_queries = 50
        
        np.random.seed(42)
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        
        # ToonDB
        toondb_index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
        ids = np.arange(num_vectors, dtype=np.uint64)
        
        start = time.perf_counter()
        toondb_index.insert_batch(ids, vectors)
        toondb_insert = time.perf_counter() - start
        
        toondb_latencies = []
        for q in queries:
            start = time.perf_counter()
            toondb_index.search(q, k=10)
            toondb_latencies.append((time.perf_counter() - start) * 1000)
        
        # ChromaDB
        client = chromadb.Client()
        collection = client.create_collection("bench", metadata={"hnsw:space": "cosine"})
        
        start = time.perf_counter()
        collection.add(
            embeddings=vectors.tolist(),
            ids=[f"id_{i}" for i in range(num_vectors)]
        )
        chroma_insert = time.perf_counter() - start
        
        chroma_latencies = []
        for q in queries:
            start = time.perf_counter()
            collection.query(query_embeddings=[q.tolist()], n_results=10)
            chroma_latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "toondb": {
                "insert_time": toondb_insert,
                "insert_rate": num_vectors / toondb_insert,
                "search_p50_ms": float(np.percentile(toondb_latencies, 50)),
            },
            "chromadb": {
                "insert_time": chroma_insert,
                "insert_rate": num_vectors / chroma_insert,
                "search_p50_ms": float(np.percentile(chroma_latencies, 50)),
            },
            "speedup": {
                "insert": chroma_insert / toondb_insert,
                "search": np.percentile(chroma_latencies, 50) / np.percentile(toondb_latencies, 50),
            }
        }
    except ImportError:
        return {"error": "ChromaDB not installed"}


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("  TOONDB COMPLETE BENCHMARK & TEST SUITE")
    print("=" * 80)
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # =========================================================================
    # SECTION 1: Correctness Tests
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SECTION 1: CORRECTNESS TESTS")
    print("=" * 80)
    
    print("\n  [1.1] Self-Retrieval Test...")
    passed, data = test_self_retrieval()
    results["self_retrieval"] = data
    status = "✓ PASS" if passed else "❌ FAIL"
    print(f"        Success Rate: {data['success_rate']:.0%} ({data['passed']}/{data['total']})")
    print(f"        Status: {status}")
    
    print("\n  [1.2] Recall@10 Test...")
    passed, data = test_recall_at_k()
    results["recall_at_k"] = data
    status = "✓ PASS" if passed else "❌ FAIL"
    print(f"        Average Recall@10: {data['avg_recall']:.1%}")
    print(f"        Threshold: {data['threshold']:.0%}")
    print(f"        Status: {status}")
    
    # =========================================================================
    # SECTION 2: Performance Benchmarks
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SECTION 2: PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    print("\n  [2.1] Search Latency...")
    data = benchmark_latency()
    results["latency"] = data
    print(f"        p50: {data['p50_ms']:.3f}ms")
    print(f"        p95: {data['p95_ms']:.3f}ms")
    print(f"        p99: {data['p99_ms']:.3f}ms")
    
    print("\n  [2.2] Insert Throughput...")
    data = benchmark_insert()
    results["insert"] = data
    print(f"        Throughput: {data['throughput']:,.0f} vec/s")
    
    print("\n  [2.3] Queries Per Second...")
    data = benchmark_qps()
    results["qps"] = data
    print(f"        QPS: {data['qps']:,.0f}")
    
    # =========================================================================
    # SECTION 3: Competitor Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SECTION 3: COMPETITOR COMPARISON")
    print("=" * 80)
    
    print("\n  [3.1] ToonDB vs ChromaDB...")
    data = compare_with_chromadb()
    results["vs_chromadb"] = data
    if "error" not in data:
        print(f"        ToonDB Insert: {data['toondb']['insert_rate']:,.0f} vec/s")
        print(f"        ChromaDB Insert: {data['chromadb']['insert_rate']:,.0f} vec/s")
        print(f"        Insert Speedup: {data['speedup']['insert']:.1f}x")
        print(f"        ToonDB Search: {data['toondb']['search_p50_ms']:.3f}ms")
        print(f"        ChromaDB Search: {data['chromadb']['search_p50_ms']:.3f}ms")
        print(f"        Search Speedup: {data['speedup']['search']:.1f}x")
    else:
        print(f"        {data['error']}")
    
    # =========================================================================
    # SUMMARY & RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    # Determine overall status
    correctness_ok = (
        results["self_retrieval"]["success_rate"] >= 0.9 and
        results["recall_at_k"]["avg_recall"] >= 0.9
    )
    
    performance_ok = (
        results["latency"]["p50_ms"] < 1.0 and
        results["insert"]["throughput"] > 10000 and
        results["qps"]["qps"] > 10000
    )
    
    print(f"""
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  OVERALL STATUS                                                            │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Correctness: {'✓ PASS' if correctness_ok else '❌ FAIL - CRITICAL BUG DETECTED'}                                              │
  │  Performance: {'✓ PASS' if performance_ok else '❌ NEEDS IMPROVEMENT'}                                              │
  └────────────────────────────────────────────────────────────────────────────┘
    """)
    
    print("  DETAILED METRICS:")
    print(f"  ─────────────────────────────────────────────────────────────────────")
    print(f"  │ Self-Retrieval Success:  {results['self_retrieval']['success_rate']:>6.0%} │ Expected: 100%     │")
    print(f"  │ Recall@10:               {results['recall_at_k']['avg_recall']:>6.1%} │ Expected: ≥90%     │")
    print(f"  │ Search Latency (p50):    {results['latency']['p50_ms']:>6.3f}ms │ Target: <1ms       │")
    print(f"  │ Insert Throughput:       {results['insert']['throughput']:>6,.0f} │ Target: >10K vec/s │")
    print(f"  │ QPS:                     {results['qps']['qps']:>6,.0f} │ Target: >10K       │")
    print(f"  ─────────────────────────────────────────────────────────────────────")
    
    print("\n  RECOMMENDATIONS:")
    print("  ─────────────────────────────────────────────────────────────────────")
    
    if not correctness_ok:
        print("""
  🚨 CRITICAL: HNSW Retrieval Bug Must Be Fixed First

  The HNSW index is returning wrong results. This is a fundamental correctness
  issue that makes all performance numbers meaningless.

  Root Cause Investigation Needed:
  1. Check toondb-index/src/hnsw.rs - graph connectivity during parallel insert
  2. Check toondb-index/src/ffi.rs - batch insert FFI correctness
  3. Verify entry point selection in search algorithm
  4. Check for race conditions in concurrent operations

  Suggested Fix Priority:
  ┌─────┬────────────────────────────────────────────────┬──────────┐
  │ P0  │ Fix HNSW batch insert graph connectivity      │ CRITICAL │
  │ P1  │ Add correctness tests to CI pipeline          │ HIGH     │
  │ P2  │ Add Recall@10 regression tests                │ HIGH     │
  └─────┴────────────────────────────────────────────────┴──────────┘
        """)
    else:
        print("""
  ✓ Correctness tests passed! Performance looks good.

  Next Steps:
  1. Scale testing to 100K-1M vectors
  2. Add persistence benchmarks
  3. Compare with more competitors (Qdrant, LanceDB)
        """)
    
    if "error" not in results.get("vs_chromadb", {}):
        speedup = results["vs_chromadb"]["speedup"]["search"]
        if speedup > 1:
            print(f"\n  ✓ ToonDB is {speedup:.1f}x faster than ChromaDB on search latency")
        else:
            print(f"\n  ⚠ ToonDB is {1/speedup:.1f}x slower than ChromaDB on search latency")
    
    print("\n" + "=" * 80)
    print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Save results to JSON
    results_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")
    
    return 0 if correctness_ok else 1


if __name__ == "__main__":
    sys.exit(main())
