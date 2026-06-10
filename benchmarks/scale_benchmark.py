#!/usr/bin/env python3
"""
SochDB Scale Benchmark Suite
=============================

Benchmark framework for validating SochDB performance at scale:
  - 10M, 100M, 1B vector targets
  - Dimensions: 384, 768, 1536, 3072
  - Metrics: recall@k, QPS, p50/p95/p99 latency, RAM/vector, disk/vector
  - Workloads: read-heavy, write-heavy, mixed, delete-heavy
  - Operations: compaction under load, restarts, concurrent access

Usage:
    python3 benchmarks/scale_benchmark.py --scale 10m --dim 768 --metric cosine
    python3 benchmarks/scale_benchmark.py --scale 100m --dim 384 --metric l2 --workload mixed
    python3 benchmarks/scale_benchmark.py --all  # Full matrix

Requirements:
    pip install numpy grpcio grpcio-tools psutil tabulate
"""

import argparse
import json
import os
import sys
import time
import statistics
import resource
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

SCALES = {
    "1m":   1_000_000,
    "10m":  10_000_000,
    "100m": 100_000_000,
    "1b":   1_000_000_000,
}

DIMENSIONS = [384, 768, 1536, 3072]

WORKLOADS = ["read_heavy", "write_heavy", "mixed", "delete_heavy"]


@dataclass
class BenchConfig:
    """Benchmark configuration."""
    scale: str = "10m"
    dimension: int = 768
    metric: str = "cosine"          # cosine | l2 | ip
    workload: str = "read_heavy"
    batch_size: int = 10_000
    search_k: int = 10
    ef_search: int = 128
    ef_construction: int = 200
    m: int = 16
    n_search_queries: int = 10_000
    n_concurrent: int = 8
    warmup_queries: int = 1000
    output_dir: str = "benchmarks/results"

    @property
    def n_vectors(self) -> int:
        return SCALES[self.scale]


@dataclass
class BenchResult:
    """Benchmark results."""
    config: dict = field(default_factory=dict)
    # Insertion metrics
    insert_total_seconds: float = 0.0
    insert_vectors_per_second: float = 0.0
    insert_p50_ms: float = 0.0
    insert_p95_ms: float = 0.0
    insert_p99_ms: float = 0.0
    # Search metrics
    search_qps: float = 0.0
    search_p50_ms: float = 0.0
    search_p95_ms: float = 0.0
    search_p99_ms: float = 0.0
    search_recall_at_k: float = 0.0
    # Resource metrics
    ram_total_mb: float = 0.0
    ram_per_vector_bytes: float = 0.0
    disk_total_mb: float = 0.0
    disk_per_vector_bytes: float = 0.0
    # Workload-specific
    mixed_read_qps: float = 0.0
    mixed_write_qps: float = 0.0
    delete_qps: float = 0.0
    compaction_duration_seconds: float = 0.0
    restart_recovery_seconds: float = 0.0


# =============================================================================
# Vector Generation
# =============================================================================

def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random float32 vectors. Uses chunked generation for large N."""
    rng = np.random.RandomState(seed)
    # For very large datasets, generate in chunks to avoid OOM
    chunk_size = min(n, 100_000)
    chunks = []
    remaining = n
    while remaining > 0:
        batch = min(chunk_size, remaining)
        chunk = rng.randn(batch, dim).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        chunk /= norms
        chunks.append(chunk)
        remaining -= batch
    return np.vstack(chunks)


def generate_ground_truth(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Compute exact k-NN ground truth via brute force (for recall measurement)."""
    n_queries = queries.shape[0]
    gt = np.zeros((n_queries, k), dtype=np.int64)

    # Process in batches to limit memory
    batch_size = 100
    for i in range(0, n_queries, batch_size):
        batch_end = min(i + batch_size, n_queries)
        q_batch = queries[i:batch_end]
        # Compute distances: (batch, dim) x (dim, n) -> (batch, n)
        # For cosine on normalized vectors: distance = 1 - dot product
        sims = q_batch @ data.T
        for j in range(q_batch.shape[0]):
            top_k = np.argpartition(-sims[j], k)[:k]
            top_k = top_k[np.argsort(-sims[j][top_k])]
            gt[i + j] = top_k

    return gt


# =============================================================================
# Benchmark Runners
# =============================================================================

def bench_insertion(config: BenchConfig, index, vectors: np.ndarray) -> dict:
    """Benchmark bulk insertion throughput."""
    n = vectors.shape[0]
    batch_latencies = []

    start = time.monotonic()
    for offset in range(0, n, config.batch_size):
        batch_end = min(offset + config.batch_size, n)
        batch = vectors[offset:batch_end]
        ids = np.arange(offset, batch_end, dtype=np.uint64)

        batch_start = time.monotonic()
        index.insert_batch_with_ids(ids, batch)
        batch_latencies.append((time.monotonic() - batch_start) * 1000)

        if (offset // config.batch_size) % 100 == 0:
            elapsed = time.monotonic() - start
            rate = batch_end / elapsed if elapsed > 0 else 0
            print(f"  Inserted {batch_end:,}/{n:,} ({rate:,.0f} vec/s)")

    total = time.monotonic() - start
    return {
        "insert_total_seconds": total,
        "insert_vectors_per_second": n / total,
        "insert_p50_ms": statistics.median(batch_latencies),
        "insert_p95_ms": np.percentile(batch_latencies, 95),
        "insert_p99_ms": np.percentile(batch_latencies, 99),
    }


def bench_search(config: BenchConfig, index, queries: np.ndarray, ground_truth: Optional[np.ndarray]) -> dict:
    """Benchmark search latency and recall."""
    latencies = []

    # Warmup
    for i in range(min(config.warmup_queries, queries.shape[0])):
        index.search(queries[i], k=config.search_k, ef_search=config.ef_search)

    # Measured queries
    for i in range(config.n_search_queries):
        q = queries[i % queries.shape[0]]
        t0 = time.monotonic()
        ids, dists = index.search(q, k=config.search_k, ef_search=config.ef_search)
        latencies.append((time.monotonic() - t0) * 1000)

    # Recall computation
    recall = 0.0
    if ground_truth is not None:
        hits = 0
        total = 0
        for i in range(min(config.n_search_queries, queries.shape[0])):
            q = queries[i]
            ids, _ = index.search(q, k=config.search_k, ef_search=config.ef_search)
            gt_set = set(ground_truth[i].tolist())
            hits += len(set(ids.tolist()) & gt_set)
            total += config.search_k
        recall = hits / total if total > 0 else 0.0

    total_time = sum(latencies) / 1000.0
    return {
        "search_qps": config.n_search_queries / total_time,
        "search_p50_ms": statistics.median(latencies),
        "search_p95_ms": np.percentile(latencies, 95),
        "search_p99_ms": np.percentile(latencies, 99),
        "search_recall_at_k": recall,
    }


def bench_mixed_workload(config: BenchConfig, index, vectors: np.ndarray, queries: np.ndarray) -> dict:
    """Benchmark mixed read/write workload (80% reads, 20% writes)."""
    import threading
    import queue

    read_latencies = []
    write_latencies = []
    n_ops = 10_000
    write_offset = vectors.shape[0]  # Start writing after existing data

    rng = np.random.RandomState(99)
    ops = rng.choice(["read", "write"], size=n_ops, p=[0.8, 0.2])

    write_idx = 0
    for op in ops:
        if op == "read":
            q = queries[rng.randint(queries.shape[0])]
            t0 = time.monotonic()
            index.search(q, k=config.search_k, ef_search=config.ef_search)
            read_latencies.append((time.monotonic() - t0) * 1000)
        else:
            vec = rng.randn(config.dimension).astype(np.float32)
            vec /= np.linalg.norm(vec)
            t0 = time.monotonic()
            index.insert_batch_with_ids(
                np.array([write_offset + write_idx], dtype=np.uint64),
                vec.reshape(1, -1)
            )
            write_latencies.append((time.monotonic() - t0) * 1000)
            write_idx += 1

    return {
        "mixed_read_qps": len(read_latencies) / (sum(read_latencies) / 1000.0) if read_latencies else 0,
        "mixed_write_qps": len(write_latencies) / (sum(write_latencies) / 1000.0) if write_latencies else 0,
    }


def measure_resources(data_dir: str, n_vectors: int) -> dict:
    """Measure RAM and disk usage."""
    import psutil
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)

    disk_mb = 0.0
    if os.path.isdir(data_dir):
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    disk_mb += os.path.getsize(fp) / (1024 * 1024)

    return {
        "ram_total_mb": ram_mb,
        "ram_per_vector_bytes": (ram_mb * 1024 * 1024) / n_vectors if n_vectors > 0 else 0,
        "disk_total_mb": disk_mb,
        "disk_per_vector_bytes": (disk_mb * 1024 * 1024) / n_vectors if n_vectors > 0 else 0,
    }


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmark(config: BenchConfig):
    """Run the full benchmark suite for a given configuration."""
    print(f"\n{'='*60}")
    print(f"SochDB Scale Benchmark")
    print(f"  Scale:     {config.scale} ({config.n_vectors:,} vectors)")
    print(f"  Dimension: {config.dimension}")
    print(f"  Metric:    {config.metric}")
    print(f"  Workload:  {config.workload}")
    print(f"{'='*60}\n")

    try:
        import sochdb
    except ImportError:
        print("ERROR: sochdb Python package not installed.")
        print("Install with: cd sochdb-python && maturin develop --release")
        sys.exit(1)

    # Create index
    data_dir = f"/tmp/sochdb_bench_{config.scale}_{config.dimension}"
    os.makedirs(data_dir, exist_ok=True)

    print(f"[1/5] Creating HNSW index (M={config.m}, ef_c={config.ef_construction})...")
    index = sochdb.HnswIndex(
        dimension=config.dimension,
        m=config.m,
        ef_construction=config.ef_construction,
    )

    # Generate data (chunked for large datasets)
    print(f"[2/5] Generating {config.n_vectors:,} vectors (dim={config.dimension})...")
    # For benchmarks > 10M, we generate and insert in chunks
    if config.n_vectors > 1_000_000:
        chunk_size = 1_000_000
        n_chunks = (config.n_vectors + chunk_size - 1) // chunk_size
        insert_results = {"insert_total_seconds": 0, "insert_vectors_per_second": 0}
        batch_latencies = []

        total_start = time.monotonic()
        for chunk_idx in range(n_chunks):
            offset = chunk_idx * chunk_size
            n_this = min(chunk_size, config.n_vectors - offset)
            vectors = generate_vectors(n_this, config.dimension, seed=42 + chunk_idx)
            ids = np.arange(offset, offset + n_this, dtype=np.uint64)

            t0 = time.monotonic()
            index.insert_batch_with_ids(ids, vectors)
            batch_latencies.append((time.monotonic() - t0) * 1000)

            elapsed = time.monotonic() - total_start
            rate = (offset + n_this) / elapsed
            print(f"  Chunk {chunk_idx+1}/{n_chunks}: {offset+n_this:,} vectors ({rate:,.0f} vec/s)")
            del vectors  # Free memory

        total_time = time.monotonic() - total_start
        insert_results = {
            "insert_total_seconds": total_time,
            "insert_vectors_per_second": config.n_vectors / total_time,
            "insert_p50_ms": statistics.median(batch_latencies),
            "insert_p95_ms": np.percentile(batch_latencies, 95),
            "insert_p99_ms": np.percentile(batch_latencies, 99),
        }
    else:
        vectors = generate_vectors(config.n_vectors, config.dimension)
        print(f"[3/5] Inserting vectors...")
        insert_results = bench_insertion(config, index, vectors)

    print(f"  -> Insert throughput: {insert_results['insert_vectors_per_second']:,.0f} vec/s")

    # Generate queries
    print(f"[3/5] Generating search queries...")
    n_queries = min(config.n_search_queries, 10_000)
    queries = generate_vectors(n_queries, config.dimension, seed=9999)

    # Ground truth (only for small enough datasets)
    ground_truth = None
    if config.n_vectors <= 1_000_000:
        print(f"  Computing ground truth (brute force)...")
        if 'vectors' in dir():
            ground_truth = generate_ground_truth(vectors, queries, config.search_k)

    # Search benchmark
    print(f"[4/5] Running search benchmark ({n_queries:,} queries)...")
    search_results = bench_search(config, index, queries, ground_truth)
    print(f"  -> QPS: {search_results['search_qps']:,.0f}")
    print(f"  -> P50/P95/P99: {search_results['search_p50_ms']:.2f}/{search_results['search_p95_ms']:.2f}/{search_results['search_p99_ms']:.2f} ms")
    if ground_truth is not None:
        print(f"  -> Recall@{config.search_k}: {search_results['search_recall_at_k']:.4f}")

    # Mixed workload
    mixed_results = {}
    if config.workload == "mixed":
        print(f"[4b] Running mixed workload benchmark...")
        mixed_results = bench_mixed_workload(config, index, vectors if 'vectors' in dir() else generate_vectors(10000, config.dimension), queries)
        print(f"  -> Read QPS: {mixed_results['mixed_read_qps']:,.0f}, Write QPS: {mixed_results['mixed_write_qps']:,.0f}")

    # Resource measurement
    print(f"[5/5] Measuring resource usage...")
    resource_results = measure_resources(data_dir, config.n_vectors)
    print(f"  -> RAM: {resource_results['ram_total_mb']:,.0f} MB ({resource_results['ram_per_vector_bytes']:.1f} bytes/vec)")
    print(f"  -> Disk: {resource_results['disk_total_mb']:,.0f} MB ({resource_results['disk_per_vector_bytes']:.1f} bytes/vec)")

    # Assemble results
    result = BenchResult(config=asdict(config))
    for d in [insert_results, search_results, mixed_results, resource_results]:
        for k, v in d.items():
            setattr(result, k, v)

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    result_file = os.path.join(
        config.output_dir,
        f"bench_{config.scale}_{config.dimension}d_{config.metric}_{config.workload}.json"
    )
    with open(result_file, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nResults saved to {result_file}")

    return result


def run_full_matrix(config: BenchConfig):
    """Run the full benchmark matrix."""
    results = []
    for scale in ["1m", "10m"]:  # Start with feasible sizes
        for dim in [384, 768, 1536]:
            for workload in ["read_heavy", "mixed"]:
                cfg = BenchConfig(
                    scale=scale,
                    dimension=dim,
                    metric=config.metric,
                    workload=workload,
                    output_dir=config.output_dir,
                )
                try:
                    result = run_benchmark(cfg)
                    results.append(asdict(result))
                except Exception as e:
                    print(f"FAILED: {scale}/{dim}d/{workload}: {e}")

    # Summary table
    summary_file = os.path.join(config.output_dir, "benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull matrix summary saved to {summary_file}")

    # Print summary table
    try:
        from tabulate import tabulate
        rows = []
        for r in results:
            rows.append([
                r["config"]["scale"],
                r["config"]["dimension"],
                r["config"]["workload"],
                f"{r['insert_vectors_per_second']:,.0f}",
                f"{r['search_qps']:,.0f}",
                f"{r['search_p95_ms']:.1f}",
                f"{r['search_recall_at_k']:.4f}" if r['search_recall_at_k'] > 0 else "N/A",
                f"{r['ram_per_vector_bytes']:.1f}",
            ])
        print("\n" + tabulate(rows, headers=[
            "Scale", "Dim", "Workload", "Insert vec/s", "Search QPS",
            "P95 ms", "Recall@10", "RAM/vec (B)"
        ]))
    except ImportError:
        pass


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SochDB Scale Benchmark Suite")
    parser.add_argument("--scale", choices=list(SCALES.keys()), default="10m",
                       help="Dataset scale")
    parser.add_argument("--dim", type=int, choices=DIMENSIONS, default=768,
                       help="Vector dimension")
    parser.add_argument("--metric", choices=["cosine", "l2", "ip"], default="cosine",
                       help="Distance metric")
    parser.add_argument("--workload", choices=WORKLOADS, default="read_heavy",
                       help="Workload type")
    parser.add_argument("--batch-size", type=int, default=10_000,
                       help="Insertion batch size")
    parser.add_argument("--n-queries", type=int, default=10_000,
                       help="Number of search queries")
    parser.add_argument("--ef-search", type=int, default=128,
                       help="HNSW ef_search parameter")
    parser.add_argument("--output", default="benchmarks/results",
                       help="Output directory")
    parser.add_argument("--all", action="store_true",
                       help="Run full benchmark matrix")
    args = parser.parse_args()

    config = BenchConfig(
        scale=args.scale,
        dimension=args.dim,
        metric=args.metric,
        workload=args.workload,
        batch_size=args.batch_size,
        n_search_queries=args.n_queries,
        ef_search=args.ef_search,
        output_dir=args.output,
    )

    if args.all:
        run_full_matrix(config)
    else:
        run_benchmark(config)


if __name__ == "__main__":
    main()
