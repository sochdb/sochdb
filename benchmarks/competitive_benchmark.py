#!/usr/bin/env python3
"""
SochDB Competitive Benchmark — Compare against FAISS, Qdrant, Milvus baselines.

Usage:
    python3 benchmarks/competitive_benchmark.py --baseline faiss --scale 1m --dim 768
    python3 benchmarks/competitive_benchmark.py --baseline all --scale 1m

Baselines available:
    faiss   — FAISS IVF-PQ and HNSW (pip install faiss-cpu)
    hnswlib — hnswlib standalone (pip install hnswlib)
    sochdb  — SochDB native (pip install sochdb / maturin develop)
"""

import argparse
import json
import os
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List

import numpy as np


@dataclass
class CompResult:
    engine: str
    scale: str
    dimension: int
    build_seconds: float
    search_qps: float
    search_p50_ms: float
    search_p95_ms: float
    search_p99_ms: float
    recall_at_10: float
    ram_mb: float


def generate_data(n: int, dim: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    data = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    data /= norms
    return data


def brute_force_knn(data, queries, k=10):
    gt = np.zeros((queries.shape[0], k), dtype=np.int64)
    for i, q in enumerate(queries):
        sims = data @ q
        top = np.argpartition(-sims, k)[:k]
        gt[i] = top[np.argsort(-sims[top])]
    return gt


def compute_recall(predicted_ids, ground_truth, k=10):
    hits = 0
    total = 0
    for pred, gt in zip(predicted_ids, ground_truth):
        hits += len(set(pred[:k].tolist()) & set(gt[:k].tolist()))
        total += k
    return hits / total if total > 0 else 0.0


# =============================================================================
# Baseline: FAISS
# =============================================================================

def bench_faiss(n: int, dim: int, queries: np.ndarray, ground_truth: np.ndarray, data: np.ndarray) -> CompResult:
    import faiss
    import psutil

    print("  Building FAISS HNSW index...")
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 200

    t0 = time.monotonic()
    index.add(data)
    build_time = time.monotonic() - t0

    index.hnsw.efSearch = 128
    latencies = []
    all_ids = []
    for q in queries:
        t0 = time.monotonic()
        D, I = index.search(q.reshape(1, -1), 10)
        latencies.append((time.monotonic() - t0) * 1000)
        all_ids.append(I[0])

    recall = compute_recall(np.array(all_ids), ground_truth)
    ram = psutil.Process().memory_info().rss / (1024 * 1024)

    return CompResult(
        engine="faiss-hnsw",
        scale=f"{n//1_000_000}m",
        dimension=dim,
        build_seconds=build_time,
        search_qps=len(queries) / (sum(latencies) / 1000),
        search_p50_ms=statistics.median(latencies),
        search_p95_ms=np.percentile(latencies, 95),
        search_p99_ms=np.percentile(latencies, 99),
        recall_at_10=recall,
        ram_mb=ram,
    )


# =============================================================================
# Baseline: hnswlib
# =============================================================================

def bench_hnswlib(n: int, dim: int, queries: np.ndarray, ground_truth: np.ndarray, data: np.ndarray) -> CompResult:
    import hnswlib
    import psutil

    print("  Building hnswlib index...")
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)

    t0 = time.monotonic()
    index.add_items(data, np.arange(n))
    build_time = time.monotonic() - t0

    index.set_ef(128)
    latencies = []
    all_ids = []
    for q in queries:
        t0 = time.monotonic()
        ids, dists = index.knn_query(q, k=10)
        latencies.append((time.monotonic() - t0) * 1000)
        all_ids.append(ids[0])

    recall = compute_recall(np.array(all_ids), ground_truth)
    ram = psutil.Process().memory_info().rss / (1024 * 1024)

    return CompResult(
        engine="hnswlib",
        scale=f"{n//1_000_000}m",
        dimension=dim,
        build_seconds=build_time,
        search_qps=len(queries) / (sum(latencies) / 1000),
        search_p50_ms=statistics.median(latencies),
        search_p95_ms=np.percentile(latencies, 95),
        search_p99_ms=np.percentile(latencies, 99),
        recall_at_10=recall,
        ram_mb=ram,
    )


# =============================================================================
# SochDB
# =============================================================================

def bench_sochdb(n: int, dim: int, queries: np.ndarray, ground_truth: np.ndarray, data: np.ndarray) -> CompResult:
    import sochdb
    import psutil

    print("  Building SochDB HNSW index...")
    index = sochdb.HnswIndex(dimension=dim, m=16, ef_construction=200)

    t0 = time.monotonic()
    ids = np.arange(n, dtype=np.uint64)
    batch_size = 50_000
    for offset in range(0, n, batch_size):
        end = min(offset + batch_size, n)
        index.insert_batch_with_ids(ids[offset:end], data[offset:end])
    build_time = time.monotonic() - t0

    latencies = []
    all_ids = []
    for q in queries:
        t0 = time.monotonic()
        r_ids, r_dists = index.search(q, k=10, ef_search=128)
        latencies.append((time.monotonic() - t0) * 1000)
        all_ids.append(r_ids)

    recall = compute_recall(np.array(all_ids), ground_truth)
    ram = psutil.Process().memory_info().rss / (1024 * 1024)

    return CompResult(
        engine="sochdb",
        scale=f"{n//1_000_000}m",
        dimension=dim,
        build_seconds=build_time,
        search_qps=len(queries) / (sum(latencies) / 1000),
        search_p50_ms=statistics.median(latencies),
        search_p95_ms=np.percentile(latencies, 95),
        search_p99_ms=np.percentile(latencies, 99),
        recall_at_10=recall,
        ram_mb=ram,
    )


# =============================================================================
# Main
# =============================================================================

BASELINES = {
    "faiss": bench_faiss,
    "hnswlib": bench_hnswlib,
    "sochdb": bench_sochdb,
}

SCALES = {"1m": 1_000_000, "10m": 10_000_000}


def main():
    parser = argparse.ArgumentParser(description="SochDB Competitive Benchmark")
    parser.add_argument("--baseline", choices=list(BASELINES.keys()) + ["all"], default="all")
    parser.add_argument("--scale", choices=list(SCALES.keys()), default="1m")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()

    n = SCALES[args.scale]
    dim = args.dim
    n_queries = 1000

    print(f"Generating {n:,} vectors (dim={dim})...")
    data = generate_data(n, dim)
    queries = generate_data(n_queries, dim, seed=9999)

    print(f"Computing ground truth...")
    gt = brute_force_knn(data, queries, k=10)

    engines = list(BASELINES.keys()) if args.baseline == "all" else [args.baseline]
    results: List[CompResult] = []

    for engine in engines:
        print(f"\n--- {engine.upper()} ---")
        try:
            result = BASELINES[engine](n, dim, queries, gt, data)
            results.append(result)
            print(f"  Build: {result.build_seconds:.1f}s | QPS: {result.search_qps:,.0f} | "
                  f"P95: {result.search_p95_ms:.2f}ms | Recall@10: {result.recall_at_10:.4f} | "
                  f"RAM: {result.ram_mb:,.0f}MB")
        except ImportError as e:
            print(f"  SKIPPED (missing dependency): {e}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, f"competitive_{args.scale}_{dim}d.json")
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print comparison table
    if results:
        print(f"\n{'Engine':<15} {'Build(s)':<10} {'QPS':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Recall@10':<10} {'RAM(MB)':<10}")
        print("-" * 95)
        for r in results:
            print(f"{r.engine:<15} {r.build_seconds:<10.1f} {r.search_qps:<10,.0f} {r.search_p50_ms:<10.2f} "
                  f"{r.search_p95_ms:<10.2f} {r.search_p99_ms:<10.2f} {r.recall_at_10:<10.4f} {r.ram_mb:<10,.0f}")


if __name__ == "__main__":
    main()
