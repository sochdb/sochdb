#!/usr/bin/env python3
"""
Focused search-path profiling harness.

Goal: isolate SochDB HNSW *query* latency at equal recall against hnswlib on the
same data / same HNSW params, at a scale that exercises the graph path (well
above the flat-scan threshold) but runs in seconds so we can iterate.

It prints recall@10 and latency percentiles for each engine plus a SochDB/best
ratio so a regression or win is obvious.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def gen(n: int, dim: int, seed: int, centers: np.ndarray | None = None) -> np.ndarray:
    """Clustered unit vectors (realistic embedding geometry).

    Random Gaussian vectors in high-dim are all ~equidistant (recall is noise),
    so we draw from a mixture of cluster centers to give well-defined neighbors.
    """
    rng = np.random.RandomState(seed)
    if centers is None:
        n_clusters = max(8, dim // 16)
        centers = rng.randn(n_clusters, dim).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-9)
    assign = rng.randint(0, centers.shape[0], size=n)
    x = centers[assign] + 0.15 * rng.randn(n, dim).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)
    return x.astype(np.float32)


def brute_force(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    gt = np.zeros((queries.shape[0], k), dtype=np.int64)
    for i, q in enumerate(queries):
        sims = data @ q
        top = np.argpartition(-sims, k)[:k]
        gt[i] = top[np.argsort(-sims[top])]
    return gt


def recall(pred: list[np.ndarray], gt: np.ndarray, k: int) -> float:
    hits = total = 0
    for p, g in zip(pred, gt):
        hits += len(set(p[:k].tolist()) & set(g[:k].tolist()))
        total += k
    return hits / total if total else 0.0


def summarize(name: str, lat_ms: list[float], rec: float) -> dict:
    return {
        "engine": name,
        "recall@10": rec,
        "qps": len(lat_ms) / (sum(lat_ms) / 1000.0),
        "p50_ms": statistics.median(lat_ms),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "mean_ms": statistics.mean(lat_ms),
    }


def bench_sochdb(data, queries, gt, m, efc, efs, k):
    import sochdb

    n, dim = data.shape
    idx = sochdb.HnswIndex(dimension=dim, m=m, ef_construction=efc)
    ids = np.arange(n, dtype=np.uint64)
    t0 = time.monotonic()
    bs = 50_000
    for o in range(0, n, bs):
        idx.insert_batch_with_ids(ids[o : min(o + bs, n)], data[o : min(o + bs, n)])
    build = time.monotonic() - t0

    # warmup
    for q in queries[: min(50, len(queries))]:
        idx.search(q, k=k, ef_search=efs)

    lat, preds = [], []
    for q in queries:
        t0 = time.monotonic()
        rids, _ = idx.search(q, k=k, ef_search=efs)
        lat.append((time.monotonic() - t0) * 1000)
        preds.append(np.asarray(rids, dtype=np.int64))
    s = summarize("sochdb", lat, recall(preds, gt, k))
    s["build_s"] = build
    return s


def bench_hnswlib(data, queries, gt, m, efc, efs, k):
    import hnswlib

    n, dim = data.shape
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n, ef_construction=efc, M=m)
    t0 = time.monotonic()
    idx.add_items(data, np.arange(n))
    build = time.monotonic() - t0
    idx.set_ef(efs)

    for q in queries[: min(50, len(queries))]:
        idx.knn_query(q, k=k)

    lat, preds = [], []
    for q in queries:
        t0 = time.monotonic()
        labels, _ = idx.knn_query(q, k=k)
        lat.append((time.monotonic() - t0) * 1000)
        preds.append(labels[0].astype(np.int64))
    s = summarize("hnswlib", lat, recall(preds, gt, k))
    s["build_s"] = build
    return s


def bench_faiss(data, queries, gt, m, efc, efs, k):
    import faiss

    n, dim = data.shape
    idx = faiss.IndexHNSWFlat(dim, m)
    idx.hnsw.efConstruction = efc
    t0 = time.monotonic()
    idx.add(data)
    build = time.monotonic() - t0
    idx.hnsw.efSearch = efs

    for q in queries[: min(50, len(queries))]:
        idx.search(q.reshape(1, -1), k)

    lat, preds = [], []
    for q in queries:
        t0 = time.monotonic()
        _, I = idx.search(q.reshape(1, -1), k)
        lat.append((time.monotonic() - t0) * 1000)
        preds.append(I[0].astype(np.int64))
    s = summarize("faiss-hnsw", lat, recall(preds, gt, k))
    s["build_s"] = build
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print(f"Generating {args.n:,}x{args.dim} + {args.queries} queries (clustered)...")
    rng = np.random.RandomState(7)
    n_clusters = max(8, args.dim // 16)
    centers = rng.randn(n_clusters, args.dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-9)
    data = gen(args.n, args.dim, 42, centers)
    queries = gen(args.queries, args.dim, 9999, centers)
    print("Computing brute-force ground truth...")
    gt = brute_force(data, queries, args.k)

    rows = []
    for fn in (bench_sochdb, bench_hnswlib, bench_faiss):
        try:
            rows.append(fn(data, queries, gt, args.m, args.ef_construction, args.ef_search, args.k))
        except ImportError as e:
            print(f"skip {fn.__name__}: {e}")

    print(f"\n{'engine':<12} {'recall@10':>10} {'qps':>9} {'p50_ms':>8} {'p95_ms':>8} {'p99_ms':>8} {'build_s':>8}")
    print("-" * 72)
    for r in rows:
        print(f"{r['engine']:<12} {r['recall@10']:>10.4f} {r['qps']:>9.0f} "
              f"{r['p50_ms']:>8.3f} {r['p95_ms']:>8.3f} {r['p99_ms']:>8.3f} {r['build_s']:>8.2f}")

    soch = next((r for r in rows if r["engine"] == "sochdb"), None)
    for name in ("hnswlib", "faiss-hnsw"):
        other = next((r for r in rows if r["engine"] == name), None)
        if soch and other:
            faster = "faster" if soch["p50_ms"] < other["p50_ms"] else "SLOWER"
            print(f"\nvs {name}: p50 {soch['p50_ms'] / other['p50_ms']:.2f}x ({faster}), "
                  f"QPS {soch['qps'] / other['qps']:.2f}x, "
                  f"recall delta {soch['recall@10'] - other['recall@10']:+.4f}")


if __name__ == "__main__":
    main()
