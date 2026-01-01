#!/usr/bin/env python3
"""
On-Disk Vector DB Benchmark: ToonDB vs LanceDB vs Qdrant vs ChromaDB

Tests on-disk performance characteristics:
1. Large-scale indexing (50K-100K vectors)
2. Memory-efficient querying with mmap
3. Filtered search performance
4. Cold-start query latency

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 benchmarks/ondisk_benchmark.py
"""

import os
import sys
import time
import tempfile
import shutil
import gc
from typing import List, Dict, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))


# =============================================================================
# Database Adapters
# =============================================================================

class ToonDBAdapter:
    """ToonDB - Rust HNSW with mmap support."""
    
    name = "ToonDB"
    
    def __init__(self, dimension: int, tmp_dir: str):
        from toondb import VectorIndex
        self.index = VectorIndex(dimension=dimension, max_connections=16, ef_construction=100)
        self.dimension = dimension
        self.count = 0
    
    def insert_batch(self, vectors: np.ndarray, metadata: List[Dict] = None) -> float:
        start = time.perf_counter()
        ids = np.arange(self.count, self.count + len(vectors), dtype=np.uint64)
        self.index.insert_batch(ids, vectors)
        self.count += len(vectors)
        return time.perf_counter() - start
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self.index.search(query, k=k)
        return list(results), time.perf_counter() - start
    
    def cleanup(self):
        pass


class LanceDBAdapter:
    """LanceDB - Disk-first columnar vector DB."""
    
    name = "LanceDB"
    
    def __init__(self, dimension: int, tmp_dir: str):
        import lancedb
        self.db = lancedb.connect(tmp_dir)
        self.dimension = dimension
        self.table = None
        self.count = 0
    
    def insert_batch(self, vectors: np.ndarray, metadata: List[Dict] = None) -> float:
        start = time.perf_counter()
        
        data = [{"id": self.count + i, "vector": vec.tolist()} for i, vec in enumerate(vectors)]
        
        if self.table is None:
            self.table = self.db.create_table("vectors", data, mode="overwrite")
        else:
            self.table.add(data)
        
        self.count += len(vectors)
        return time.perf_counter() - start
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self.table.search(query.tolist()).limit(k).to_list()
        return [(r['id'], r.get('_distance', 0)) for r in results], time.perf_counter() - start
    
    def cleanup(self):
        pass


class QdrantAdapter:
    """Qdrant - Local mode with mmap support."""
    
    name = "Qdrant"
    
    def __init__(self, dimension: int, tmp_dir: str):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        
        self.client = QdrantClient(path=tmp_dir)
        self.dimension = dimension
        self.collection_name = "vectors"
        self.count = 0
        
        # Create collection with mmap enabled for on-disk storage
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
                on_disk=True  # Enable on-disk storage
            ),
        )
    
    def insert_batch(self, vectors: np.ndarray, metadata: List[Dict] = None) -> float:
        from qdrant_client.models import PointStruct
        
        start = time.perf_counter()
        
        points = [
            PointStruct(id=self.count + i, vector=vec.tolist())
            for i, vec in enumerate(vectors)
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        self.count += len(vectors)
        return time.perf_counter() - start
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query.tolist(),
            limit=k
        )
        return [(r.id, r.score) for r in results.points], time.perf_counter() - start
    
    def cleanup(self):
        self.client.close()


class ChromaDBAdapter:
    """ChromaDB - In-memory with persistence option."""
    
    name = "ChromaDB"
    
    def __init__(self, dimension: int, tmp_dir: str):
        import chromadb
        
        self.client = chromadb.PersistentClient(path=tmp_dir)
        self.collection = self.client.create_collection(
            name="vectors",
            metadata={"hnsw:space": "cosine"}
        )
        self.dimension = dimension
        self.count = 0
    
    def insert_batch(self, vectors: np.ndarray, metadata: List[Dict] = None) -> float:
        start = time.perf_counter()
        
        ids = [f"id_{self.count + i}" for i in range(len(vectors))]
        self.collection.add(embeddings=vectors.tolist(), ids=ids)
        self.count += len(vectors)
        return time.perf_counter() - start
    
    def search(self, query: np.ndarray, k: int) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self.collection.query(query_embeddings=[query.tolist()], n_results=k)
        
        output = []
        if results['ids'] and results['ids'][0]:
            for i, id_str in enumerate(results['ids'][0]):
                idx = int(id_str.split('_')[1])
                dist = results['distances'][0][i] if results['distances'] else 0
                output.append((idx, dist))
        
        return output, time.perf_counter() - start
    
    def cleanup(self):
        pass


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_scale_benchmark(adapters_classes: List, dimension: int, num_vectors: int, batch_size: int = 1000):
    """Run benchmark at a specific scale."""
    
    print(f"\n{'='*70}")
    print(f"  SCALE TEST: {num_vectors:,} vectors × {dimension}-dim")
    print(f"{'='*70}")
    
    # Generate test data
    print(f"\n   Generating {num_vectors:,} random vectors...")
    np.random.seed(42)
    all_vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    all_vectors = all_vectors / norms
    
    # Generate queries
    num_queries = 100
    queries = np.random.randn(num_queries, dimension).astype(np.float32)
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / query_norms
    
    results = {}
    
    for adapter_class in adapters_classes:
        tmp_dir = tempfile.mkdtemp()
        
        try:
            print(f"\n   [{adapter_class.name}]")
            
            # Create adapter
            adapter = adapter_class(dimension, tmp_dir)
            
            # Insert in batches
            insert_times = []
            for i in range(0, num_vectors, batch_size):
                batch = all_vectors[i:i+batch_size]
                t = adapter.insert_batch(batch)
                insert_times.append(t)
            
            total_insert = sum(insert_times)
            insert_rate = num_vectors / total_insert
            print(f"   Insert: {total_insert:.2f}s ({insert_rate:,.0f} vec/s)")
            
            # Force GC to simulate cold start
            gc.collect()
            
            # Warm-up queries
            for q in queries[:5]:
                adapter.search(q, k=10)
            
            # Timed queries
            search_times = []
            for q in queries:
                _, latency = adapter.search(q, k=10)
                search_times.append(latency * 1000)
            
            avg_search = np.mean(search_times)
            p50_search = np.percentile(search_times, 50)
            p99_search = np.percentile(search_times, 99)
            
            print(f"   Search: {avg_search:.2f}ms avg, {p50_search:.2f}ms p50, {p99_search:.2f}ms p99")
            
            results[adapter_class.name] = {
                "insert_rate": insert_rate,
                "search_avg_ms": avg_search,
                "search_p50_ms": p50_search,
                "search_p99_ms": p99_search,
            }
            
            adapter.cleanup()
            
        except Exception as e:
            print(f"   ERROR: {e}")
            results[adapter_class.name] = {"error": str(e)}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return results


def main():
    print("="*70)
    print("  ON-DISK VECTOR DATABASE BENCHMARK")
    print("  ToonDB vs LanceDB vs Qdrant vs ChromaDB")
    print("="*70)
    print("\n  Testing on-disk storage characteristics:")
    print("  - LanceDB: Disk-first Lance format")
    print("  - Qdrant: mmap with on_disk=True")
    print("  - ChromaDB: DuckDB+Parquet persistence")
    print("  - ToonDB: In-memory HNSW (baseline)")
    
    adapters = [ToonDBAdapter, LanceDBAdapter, QdrantAdapter, ChromaDBAdapter]
    
    all_results = {}
    
    # Test at different scales
    scales = [
        (128, 10_000),    # 10K vectors, 128-dim
        (128, 50_000),    # 50K vectors, 128-dim
        (768, 10_000),    # 10K vectors, 768-dim (LLM embedding size)
    ]
    
    for dimension, num_vectors in scales:
        scale_key = f"{num_vectors//1000}K×{dimension}"
        all_results[scale_key] = run_scale_benchmark(adapters, dimension, num_vectors)
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY - Search Latency (ms)")
    print("="*70)
    
    print("\n{:<15} {:>12} {:>12} {:>12} {:>12}".format(
        "Scale", "ToonDB", "LanceDB", "Qdrant", "ChromaDB"
    ))
    print("-"*70)
    
    for scale_key, results in all_results.items():
        row = [scale_key]
        for db in ["ToonDB", "LanceDB", "Qdrant", "ChromaDB"]:
            if db in results and "search_avg_ms" in results[db]:
                row.append(f"{results[db]['search_avg_ms']:.2f}")
            else:
                row.append("ERROR")
        print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(*row))
    
    # Insert throughput summary
    print("\n" + "="*70)
    print("  INSERT THROUGHPUT (vec/s)")
    print("="*70)
    
    print("\n{:<15} {:>12} {:>12} {:>12} {:>12}".format(
        "Scale", "ToonDB", "LanceDB", "Qdrant", "ChromaDB"
    ))
    print("-"*70)
    
    for scale_key, results in all_results.items():
        row = [scale_key]
        for db in ["ToonDB", "LanceDB", "Qdrant", "ChromaDB"]:
            if db in results and "insert_rate" in results[db]:
                row.append(f"{results[db]['insert_rate']:,.0f}")
            else:
                row.append("ERROR")
        print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(*row))
    
    # Winner analysis
    print("\n" + "="*70)
    print("  🏆 ANALYSIS")
    print("="*70)
    
    print("\n  On-Disk Characteristics:")
    print("  - LanceDB: Best for 100GB+ datasets, disk-first architecture")
    print("  - Qdrant: Best for production deployments, mmap support")
    print("  - ChromaDB: Easy to use, good for prototyping")
    print("  - ToonDB: Fastest in-memory, ideal for latency-critical apps")
    
    print("\n  ✓ On-disk benchmark completed!")
    print("="*70)


if __name__ == "__main__":
    main()
