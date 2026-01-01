#!/usr/bin/env python3
"""
BEIR Benchmark Adapter for ToonDB

Downloads BEIR datasets and evaluates ToonDB's retrieval quality
against standard IR benchmarks with metrics like nDCG@10 and Recall@k.

BEIR: https://github.com/beir-cellar/beir
Paper: https://openreview.net/forum?id=wCu6T5xFjeJ
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add ToonDB SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))


@dataclass
class BEIRResult:
    """Results from a BEIR evaluation run."""
    dataset: str
    num_docs: int
    num_queries: int
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    precision_at_10: float
    map_at_10: float
    mrr_at_10: float
    avg_latency_ms: float
    p99_latency_ms: float
    qps: float
    index_time_s: float


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Calculate DCG@k."""
    relevances = relevances[:k]
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances)]
    return sum(gains)


def ndcg_at_k(relevances: List[int], ideal_relevances: List[int], k: int) -> float:
    """Calculate nDCG@k."""
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Calculate Recall@k."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Calculate Precision@k."""
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return sum(1 for doc in retrieved_k if doc in relevant) / len(retrieved_k)


def average_precision(retrieved: List[str], relevant: set) -> float:
    """Calculate Average Precision."""
    if not relevant:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    
    return sum_precisions / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: set) -> float:
    """Calculate Reciprocal Rank."""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def load_beir_dataset(dataset_name: str, split: str = "test") -> Tuple[Dict, Dict, Dict]:
    """
    Load a BEIR dataset.
    
    Returns:
        corpus: dict of doc_id -> {"title": str, "text": str}
        queries: dict of query_id -> str
        qrels: dict of query_id -> {doc_id: relevance_score}
    """
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        print("BEIR not installed. Run: pip install beir")
        sys.exit(1)
    
    # Download dataset
    data_path = Path.home() / ".cache" / "beir" / dataset_name
    
    if not data_path.exists():
        print(f"Downloading {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        util.download_and_unzip(url, str(data_path.parent))
    
    # Load data
    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split=split)
    
    return corpus, queries, qrels


def generate_embeddings_batch(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def run_beir_benchmark(
    dataset_name: str,
    split: str = "test",
    top_k: int = 100,
    use_toondb: bool = True,
    max_docs: Optional[int] = None,
    max_queries: Optional[int] = None,
    verbose: bool = True,
) -> BEIRResult:
    """
    Run BEIR benchmark on ToonDB.
    
    Args:
        dataset_name: BEIR dataset name (e.g., "fiqa", "scifact", "nfcorpus")
        split: Dataset split ("test", "dev", "train")
        top_k: Number of results to retrieve
        use_toondb: Use ToonDB index (vs brute force baseline)
        max_docs: Limit corpus size for testing
        max_queries: Limit query count for testing
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  BEIR Benchmark: {dataset_name}")
        print(f"{'='*70}")
    
    # Load dataset
    if verbose:
        print(f"\nLoading {dataset_name} dataset...")
    
    corpus, queries, qrels = load_beir_dataset(dataset_name, split)
    
    # Apply limits
    doc_ids = list(corpus.keys())
    if max_docs:
        doc_ids = doc_ids[:max_docs]
    
    query_ids = list(queries.keys())
    if max_queries:
        query_ids = query_ids[:max_queries]
    
    if verbose:
        print(f"  Corpus: {len(doc_ids)} documents")
        print(f"  Queries: {len(query_ids)} queries")
    
    # Generate document embeddings
    if verbose:
        print("\nGenerating document embeddings...")
    
    doc_texts = [
        f"{corpus[doc_id].get('title', '')} {corpus[doc_id]['text']}" 
        for doc_id in doc_ids
    ]
    
    start = time.perf_counter()
    doc_embeddings = generate_embeddings_batch(doc_texts)
    embed_time = time.perf_counter() - start
    
    if verbose:
        print(f"  Embeddings generated in {embed_time:.2f}s")
    
    dim = doc_embeddings.shape[1]
    
    # Build index
    if use_toondb:
        try:
            from toondb import VectorIndex
            if VectorIndex is None:
                raise ImportError("VectorIndex not available")
        except ImportError as e:
            print(f"ToonDB not available: {e}")
            print("Falling back to brute force")
            use_toondb = False
    
    index_start = time.perf_counter()
    
    if use_toondb:
        if verbose:
            print("\nBuilding ToonDB HNSW index...")
        
        index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
        ids = np.arange(len(doc_ids), dtype=np.uint64)
        inserted = index.insert_batch(ids, doc_embeddings)
        
        if verbose:
            print(f"  Inserted {inserted} vectors")
    else:
        if verbose:
            print("\nUsing brute-force baseline...")
        index = None
    
    index_time = time.perf_counter() - index_start
    
    # Generate query embeddings
    if verbose:
        print("\nGenerating query embeddings...")
    
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = generate_embeddings_batch(query_texts)
    
    # Create doc_id to index mapping
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    idx_to_doc_id = {idx: doc_id for doc_id, idx in doc_id_to_idx.items()}
    
    # Run queries and evaluate
    if verbose:
        print(f"\nRunning {len(query_ids)} queries...")
    
    all_ndcg = []
    all_recall_10 = []
    all_recall_100 = []
    all_precision = []
    all_ap = []
    all_rr = []
    latencies = []
    
    for i, qid in enumerate(query_ids):
        query_vec = query_embeddings[i]
        
        # Get relevant docs for this query
        relevant_docs = set(qrels.get(qid, {}).keys()) & set(doc_ids)
        if not relevant_docs:
            continue
        
        # Search
        start = time.perf_counter()
        
        if use_toondb:
            results = index.search(query_vec, k=top_k)
            retrieved_ids = [idx_to_doc_id[int(r[0])] for r in results if int(r[0]) in idx_to_doc_id]
        else:
            # Brute force
            similarities = np.dot(doc_embeddings, query_vec)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            retrieved_ids = [idx_to_doc_id[idx] for idx in top_indices]
        
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        # Calculate metrics
        relevances = [1 if doc in relevant_docs else 0 for doc in retrieved_ids]
        ideal_relevances = [1] * len(relevant_docs)
        
        all_ndcg.append(ndcg_at_k(relevances, ideal_relevances, 10))
        all_recall_10.append(recall_at_k(retrieved_ids, relevant_docs, 10))
        all_recall_100.append(recall_at_k(retrieved_ids, relevant_docs, 100))
        all_precision.append(precision_at_k(retrieved_ids, relevant_docs, 10))
        all_ap.append(average_precision(retrieved_ids, relevant_docs))
        all_rr.append(reciprocal_rank(retrieved_ids, relevant_docs))
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(query_ids)} queries")
    
    # Calculate final metrics
    result = BEIRResult(
        dataset=dataset_name,
        num_docs=len(doc_ids),
        num_queries=len([qid for qid in query_ids if qrels.get(qid)]),
        ndcg_at_10=np.mean(all_ndcg),
        recall_at_10=np.mean(all_recall_10),
        recall_at_100=np.mean(all_recall_100),
        precision_at_10=np.mean(all_precision),
        map_at_10=np.mean(all_ap),
        mrr_at_10=np.mean(all_rr),
        avg_latency_ms=np.mean(latencies),
        p99_latency_ms=np.percentile(latencies, 99),
        qps=1000 / np.mean(latencies) if latencies else 0,
        index_time_s=index_time,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Results: {dataset_name}")
        print(f"{'='*70}")
        print(f"  nDCG@10:      {result.ndcg_at_10:.4f}")
        print(f"  Recall@10:    {result.recall_at_10:.4f}")
        print(f"  Recall@100:   {result.recall_at_100:.4f}")
        print(f"  Precision@10: {result.precision_at_10:.4f}")
        print(f"  MAP@10:       {result.map_at_10:.4f}")
        print(f"  MRR@10:       {result.mrr_at_10:.4f}")
        print(f"\n  Latency (avg): {result.avg_latency_ms:.3f}ms")
        print(f"  Latency (p99): {result.p99_latency_ms:.3f}ms")
        print(f"  QPS:           {result.qps:.0f}")
        print(f"  Index time:    {result.index_time_s:.2f}s")
    
    return result


def run_multi_dataset_benchmark(
    datasets: List[str],
    output_path: Optional[str] = None,
    max_docs: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> List[BEIRResult]:
    """Run BEIR benchmark on multiple datasets."""
    
    results = []
    
    for dataset in datasets:
        try:
            result = run_beir_benchmark(
                dataset,
                max_docs=max_docs,
                max_queries=max_queries,
            )
            results.append(result)
        except Exception as e:
            print(f"Error on {dataset}: {e}")
    
    # Summary table
    print("\n" + "="*80)
    print("  BEIR BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<15} {'nDCG@10':<10} {'Recall@10':<12} {'Recall@100':<12} {'QPS':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r.dataset:<15} {r.ndcg_at_10:.4f}     {r.recall_at_10:.4f}       {r.recall_at_100:.4f}       {r.qps:.0f}")
    
    # Save results
    if output_path:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "results": [
                {
                    "dataset": r.dataset,
                    "num_docs": r.num_docs,
                    "num_queries": r.num_queries,
                    "ndcg_at_10": r.ndcg_at_10,
                    "recall_at_10": r.recall_at_10,
                    "recall_at_100": r.recall_at_100,
                    "precision_at_10": r.precision_at_10,
                    "map_at_10": r.map_at_10,
                    "mrr_at_10": r.mrr_at_10,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "qps": r.qps,
                    "index_time_s": r.index_time_s,
                }
                for r in results
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="BEIR Benchmark for ToonDB")
    parser.add_argument(
        "--dataset", "-d",
        default="scifact",
        help="BEIR dataset name (scifact, fiqa, nfcorpus, etc.)"
    )
    parser.add_argument(
        "--multi", "-m",
        nargs="+",
        help="Run multiple datasets"
    )
    parser.add_argument("--top-k", "-k", type=int, default=100, help="Top-k retrieval")
    parser.add_argument("--max-docs", type=int, help="Limit corpus size")
    parser.add_argument("--max-queries", type=int, help="Limit query count")
    parser.add_argument("--brute-force", action="store_true", help="Use brute force baseline")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    if args.multi:
        run_multi_dataset_benchmark(
            args.multi,
            output_path=args.output,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
        )
    else:
        result = run_beir_benchmark(
            args.dataset,
            top_k=args.top_k,
            use_toondb=not args.brute_force,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
        )
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "dataset": result.dataset,
                    "ndcg_at_10": result.ndcg_at_10,
                    "recall_at_10": result.recall_at_10,
                    "recall_at_100": result.recall_at_100,
                    "qps": result.qps,
                }, f, indent=2)


if __name__ == "__main__":
    main()
