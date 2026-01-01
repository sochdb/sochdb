#!/usr/bin/env python3
"""
Comprehensive Vector Search Benchmark
ToonDB vs ChromaDB vs NumPy (brute-force)

Tests multiple configurations and provides Python SDK examples.
"""

import time
import os
import shutil
import numpy as np
import sys

# Add ToonDB SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))

def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random unit vectors"""
    np.random.seed(seed)
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# =============================================================================
# 1. PYTHON SDK USAGE EXAMPLES
# =============================================================================

def demo_toondb_usage():
    """Show ToonDB Python SDK usage examples"""
    print("\n" + "="*70)
    print("   TOONDB PYTHON SDK USAGE EXAMPLES")
    print("="*70)
    
    from toondb import VectorIndex
    
    print("""
# Installation (from source)
pip install ./toondb-python-sdk

# Or set paths for development:
# TOONDB_LIB_PATH=/path/to/target/release
# PYTHONPATH=/path/to/toondb-python-sdk/src

# 1. Create an index
from toondb import VectorIndex
import numpy as np

index = VectorIndex(
    dimension=128,           # Vector dimension
    max_connections=16,      # HNSW M parameter
    ef_construction=100,     # Construction-time ef
)

# 2. Insert single vector
vector = np.random.randn(128).astype(np.float32)
index.insert(id=0, vector=vector)

# 3. Batch insert (10x faster!)
ids = np.arange(1000, dtype=np.uint64)
vectors = np.random.randn(1000, 128).astype(np.float32)
inserted = index.insert_batch(ids, vectors)
print(f"Inserted {inserted} vectors")

# 4. Search for nearest neighbors
query = np.random.randn(128).astype(np.float32)
results = index.search(query, k=10)

for id, distance in results:
    print(f"ID: {id}, Distance: {distance:.4f}")

# 5. Check index size
print(f"Index contains {len(index)} vectors")
""")
    
    # Actually run the example
    print("\n--- Running Example ---\n")
    
    index = VectorIndex(dimension=128, max_connections=16, ef_construction=100)
    
    # Insert 1000 vectors
    ids = np.arange(1000, dtype=np.uint64)
    vectors = generate_vectors(1000, 128)
    
    start = time.perf_counter()
    inserted = index.insert_batch(ids, vectors)
    insert_time = time.perf_counter() - start
    
    print(f"Inserted {inserted} vectors in {insert_time:.3f}s ({inserted/insert_time:.0f} vec/sec)")
    
    # Search
    query = vectors[42]  # Use vector 42 as query
    
    start = time.perf_counter()
    results = index.search(query, k=5)
    search_time = (time.perf_counter() - start) * 1000
    
    print(f"\nSearch completed in {search_time:.3f}ms")
    print("Top 5 results:")
    for id, distance in results:
        print(f"  ID: {id}, Distance: {distance:.6f}")
    
    print(f"\nExpected ID 42 as first result: {'✓ PASS' if results[0][0] == 42 else '✗ FAIL'}")

# =============================================================================
# 2. NUMPY BRUTE-FORCE BASELINE
# =============================================================================

def benchmark_numpy_bruteforce(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Benchmark pure NumPy brute-force search"""
    print("\n" + "="*70)
    print("   NUMPY BRUTE-FORCE BASELINE")
    print("="*70)
    
    # "Insert" is just storing vectors (instant)
    start = time.perf_counter()
    index_vectors = vectors.copy()
    insert_time = time.perf_counter() - start
    print(f"Insert: {insert_time:.6f}s (numpy array copy)")
    
    # Brute-force search using cosine similarity
    latencies = []
    for query in queries:
        start = time.perf_counter()
        
        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(index_vectors, query)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
    print(f"QPS: {1000/avg_latency:.0f}")
    
    return {
        "insert_rate": len(vectors) / max(insert_time, 0.0001),
        "avg_latency": avg_latency,
        "p50": p50,
        "p99": p99,
    }

# =============================================================================
# 2.5 SQLITE-VSS BENCHMARK
# =============================================================================

def benchmark_sqlite_vss(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Benchmark SQLite with VSS extension for vector search"""
    try:
        import sqlite3
        import sqlite_vss
    except ImportError as e:
        print(f"\nSQLite-VSS not available: {e}")
        print("Install with: pip install sqlite-vss")
        return None

    print("\n" + "="*70)
    print("   SQLITE-VSS (SQLite + Vector Search)")
    print("="*70)
    
    import tempfile
    db_path = tempfile.mktemp(suffix=".db")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        
        dim = vectors.shape[1]
        
        # Create virtual table for vector search
        conn.execute(f"""
            CREATE VIRTUAL TABLE vss_vectors USING vss0(
                embedding({dim})
            )
        """)
        
        # Insert vectors
        start = time.perf_counter()
        for i, vec in enumerate(vectors):
            # VSS expects JSON array format
            vec_json = "[" + ",".join(str(x) for x in vec) + "]"
            conn.execute("INSERT INTO vss_vectors(rowid, embedding) VALUES (?, ?)", 
                        (i, vec_json))
        conn.commit()
        insert_time = time.perf_counter() - start
        insert_rate = len(vectors) / insert_time
        print(f"Insert: {insert_time:.3f}s ({insert_rate:.0f} vec/sec)")
        
        # Search
        latencies = []
        for query in queries:
            query_json = "[" + ",".join(str(x) for x in query) + "]"
            
            start = time.perf_counter()
            results = conn.execute(f"""
                SELECT rowid, distance 
                FROM vss_vectors 
                WHERE vss_search(embedding, ?)
                LIMIT {k}
            """, (query_json,)).fetchall()
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
        print(f"QPS: {1000/avg_latency:.0f}")
        
        conn.close()
        
        return {
            "insert_rate": insert_rate,
            "avg_latency": avg_latency,
            "p50": p50,
            "p99": p99,
        }
        
    except Exception as e:
        print(f"SQLite-VSS error: {e}")
        return None
    finally:
        import os
        if os.path.exists(db_path):
            os.remove(db_path)

# =============================================================================
# 3. CHROMADB BENCHMARK
# =============================================================================

def benchmark_chromadb(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Benchmark ChromaDB"""
    try:
        import chromadb
    except ImportError:
        print("\nChromaDB not installed. Run: pip install chromadb")
        return None

    print("\n" + "="*70)
    print("   CHROMADB (Python + HNSW)")
    print("="*70)
    
    chroma_dir = "/tmp/chroma_comprehensive"
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
            n_results=k
        )
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
    print(f"QPS: {1000/avg_latency:.0f}")
    
    shutil.rmtree(chroma_dir, ignore_errors=True)
    
    return {
        "insert_rate": insert_rate,
        "avg_latency": avg_latency,
        "p50": p50,
        "p99": p99,
    }

# =============================================================================
# 4. TOONDB BENCHMARK
# =============================================================================

def benchmark_toondb(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Benchmark ToonDB Vector Index"""
    try:
        from toondb import VectorIndex
        if VectorIndex is None:
            raise ImportError("VectorIndex not available")
    except ImportError as e:
        print(f"\nToonDB VectorIndex not available: {e}")
        return None

    print("\n" + "="*70)
    print("   TOONDB (Rust HNSW via Python FFI)")
    print("="*70)
    
    dim = vectors.shape[1]
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    # Insert
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
        results = index.search(query, k=k)
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
    print(f"QPS: {1000/avg_latency:.0f}")
    print(f"Index size: {len(index)} vectors")
    
    return {
        "insert_rate": insert_rate,
        "avg_latency": avg_latency,
        "p50": p50,
        "p99": p99,
    }

# =============================================================================
# 5. DUCKDB BENCHMARK  
# =============================================================================

def benchmark_duckdb(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Benchmark DuckDB with VSS extension for vector search"""
    try:
        import duckdb
    except ImportError:
        print("\nDuckDB not installed. Run: pip install duckdb")
        return None

    print("\n" + "="*70)
    print("   DUCKDB (Analytical DB + Vector Search)")
    print("="*70)
    
    dim = vectors.shape[1]
    
    try:
        conn = duckdb.connect(":memory:")
        
        # Install and load VSS extension
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        
        # Create table with vector column
        conn.execute(f"""
            CREATE TABLE vectors (
                id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}]
            )
        """)
        
        # Insert vectors
        start = time.perf_counter()
        for i, vec in enumerate(vectors):
            vec_list = vec.tolist()
            conn.execute("INSERT INTO vectors VALUES (?, ?)", [i, vec_list])
        insert_time = time.perf_counter() - start
        insert_rate = len(vectors) / insert_time
        print(f"Insert: {insert_time:.3f}s ({insert_rate:.0f} vec/sec)")
        
        # Create HNSW index
        conn.execute("""
            CREATE INDEX vectors_idx ON vectors 
            USING HNSW (embedding) WITH (metric = 'cosine')
        """)
        print("HNSW index created")
        
        # Search
        latencies = []
        for query in queries:
            query_list = query.tolist()
            
            start = time.perf_counter()
            results = conn.execute(f"""
                SELECT id, array_cosine_distance(embedding, ?::FLOAT[{dim}]) as distance
                FROM vectors
                ORDER BY distance
                LIMIT {k}
            """, [query_list]).fetchall()
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
        print(f"QPS: {1000/avg_latency:.0f}")
        
        conn.close()
        
        return {
            "insert_rate": insert_rate,
            "avg_latency": avg_latency,
            "p50": p50,
            "p99": p99,
        }
        
    except Exception as e:
        print(f"DuckDB error: {e}")
        return None

# =============================================================================
# 6. LANCEDB BENCHMARK
# =============================================================================

def benchmark_lancedb(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """Benchmark LanceDB for vector search"""
    try:
        import lancedb
        import pyarrow as pa
    except ImportError as e:
        print(f"\nLanceDB not installed: {e}")
        print("Run: pip install lancedb")
        return None

    print("\n" + "="*70)
    print("   LANCEDB (Columnar Vector DB)")
    print("="*70)
    
    import tempfile
    import shutil
    
    db_path = tempfile.mkdtemp(prefix="lancedb_bench_")
    
    try:
        db = lancedb.connect(db_path)
        
        # Create table with vectors
        start = time.perf_counter()
        
        # Prepare data as list of dicts
        data = [
            {"id": i, "vector": vec.tolist()}
            for i, vec in enumerate(vectors)
        ]
        
        table = db.create_table("vectors", data)
        insert_time = time.perf_counter() - start
        insert_rate = len(vectors) / insert_time
        print(f"Insert: {insert_time:.3f}s ({insert_rate:.0f} vec/sec)")
        
        # Create IVF-PQ index for faster search
        try:
            table.create_index(metric="cosine", num_partitions=16, num_sub_vectors=16)
            print("IVF-PQ index created")
        except Exception as e:
            print(f"Index creation skipped: {e}")
        
        # Search
        latencies = []
        for query in queries:
            start = time.perf_counter()
            results = table.search(query.tolist()).limit(k).to_list()
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"Search: {avg_latency:.3f}ms avg, {p50:.3f}ms p50, {p99:.3f}ms p99")
        print(f"QPS: {1000/avg_latency:.0f}")
        
        return {
            "insert_rate": insert_rate,
            "avg_latency": avg_latency,
            "p50": p50,
            "p99": p99,
        }
        
    except Exception as e:
        print(f"LanceDB error: {e}")
        return None
    finally:
        shutil.rmtree(db_path, ignore_errors=True)

# =============================================================================
# 7. MULTI-CONFIGURATION BENCHMARK
# =============================================================================

def run_multi_config_benchmark():
    """Run benchmarks with different configurations"""
    print("\n" + "="*70)
    print("   MULTI-CONFIGURATION BENCHMARK")
    print("="*70)
    
    configs = [
        {"num_vectors": 1000, "dim": 128, "queries": 50},
        {"num_vectors": 10000, "dim": 128, "queries": 100},
        {"num_vectors": 10000, "dim": 384, "queries": 100},  # sentence-transformers dim
        {"num_vectors": 10000, "dim": 768, "queries": 100},  # BERT dim
    ]
    
    from toondb import VectorIndex
    
    results = []
    
    for config in configs:
        n = config["num_vectors"]
        d = config["dim"]
        q = config["queries"]
        
        print(f"\n--- Config: {n} vectors, {d}-dim, {q} queries ---")
        
        vectors = generate_vectors(n, d)
        queries = generate_vectors(q, d, seed=123)
        
        index = VectorIndex(dimension=d, max_connections=16, ef_construction=100)
        ids = np.arange(n, dtype=np.uint64)
        
        # Insert
        start = time.perf_counter()
        inserted = index.insert_batch(ids, vectors)
        insert_time = time.perf_counter() - start
        insert_rate = inserted / insert_time
        
        # Search
        latencies = []
        for query in queries:
            start = time.perf_counter()
            index.search(query, k=10)
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"  Insert: {insert_rate:.0f} vec/sec")
        print(f"  Search: {avg_latency:.3f}ms avg")
        
        results.append({
            "config": f"{n}×{d}",
            "insert_rate": insert_rate,
            "search_ms": avg_latency,
        })
    
    # Summary table
    print("\n" + "-"*50)
    print(f"{'Config':<15} {'Insert (vec/s)':<18} {'Search (ms)':<12}")
    print("-"*50)
    for r in results:
        print(f"{r['config']:<15} {r['insert_rate']:.0f}{'':<10} {r['search_ms']:.3f}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("   COMPREHENSIVE VECTOR SEARCH BENCHMARK")
    print("   ToonDB vs ChromaDB vs DuckDB vs LanceDB vs NumPy")
    print("="*70)
    
    # 1. SDK Usage Examples
    demo_toondb_usage()
    
    # 2. Standard benchmark (10K vectors, 128-dim)
    print("\n\n" + "#"*70)
    print("#  STANDARD BENCHMARK: 10K vectors, 128-dim")
    print("#"*70)
    
    NUM_VECTORS = 10000
    DIM = 128
    NUM_QUERIES = 100
    
    vectors = generate_vectors(NUM_VECTORS, DIM)
    queries = generate_vectors(NUM_QUERIES, DIM, seed=123)
    
    results = {}
    
    # NumPy brute-force
    results['numpy'] = benchmark_numpy_bruteforce(vectors, queries)
    
    # SQLite-VSS
    results['sqlite_vss'] = benchmark_sqlite_vss(vectors, queries)
    
    # ChromaDB
    results['chromadb'] = benchmark_chromadb(vectors, queries)
    
    # DuckDB
    results['duckdb'] = benchmark_duckdb(vectors, queries)
    
    # LanceDB
    results['lancedb'] = benchmark_lancedb(vectors, queries)
    
    # ToonDB
    results['toondb'] = benchmark_toondb(vectors, queries)
    
    # 3. Multi-config benchmark
    run_multi_config_benchmark()
    
    # Final Summary
    print("\n\n" + "="*70)
    print("   FINAL SUMMARY")
    print("="*70)
    
    print(f"\n{'System':<25} {'Insert (vec/s)':<18} {'Search (ms)':<15} {'Speedup vs NumPy':<15}")
    print("-"*75)
    
    numpy_latency = results['numpy']['avg_latency'] if results.get('numpy') else 1
    
    if results.get('numpy'):
        r = results['numpy']
        print(f"{'NumPy (brute-force)':<25} {'N/A':<18} {r['avg_latency']:.3f}{'':<10} {'1.0x (baseline)'}")
    
    if results.get('sqlite_vss'):
        r = results['sqlite_vss']
        speedup = numpy_latency / r['avg_latency']
        print(f"{'SQLite-VSS':<25} {r['insert_rate']:.0f}{'':<10} {r['avg_latency']:.3f}{'':<10} {speedup:.1f}x")
    
    if results.get('chromadb'):
        r = results['chromadb']
        speedup = numpy_latency / r['avg_latency']
        print(f"{'ChromaDB':<25} {r['insert_rate']:.0f}{'':<10} {r['avg_latency']:.3f}{'':<10} {speedup:.1f}x")
    
    if results.get('duckdb'):
        r = results['duckdb']
        speedup = numpy_latency / r['avg_latency']
        print(f"{'DuckDB':<25} {r['insert_rate']:.0f}{'':<10} {r['avg_latency']:.3f}{'':<10} {speedup:.1f}x")
    
    if results.get('lancedb'):
        r = results['lancedb']
        speedup = numpy_latency / r['avg_latency']
        print(f"{'LanceDB':<25} {r['insert_rate']:.0f}{'':<10} {r['avg_latency']:.3f}{'':<10} {speedup:.1f}x")
    
    if results.get('toondb'):
        r = results['toondb']
        speedup = numpy_latency / r['avg_latency']
        print(f"{'ToonDB':<25} {r['insert_rate']:.0f}{'':<10} {r['avg_latency']:.3f}{'':<10} {speedup:.1f}x")
    
    # Winner announcement
    print("\n" + "="*70)
    print("   🏆 WINNER ANALYSIS")
    print("="*70)
    
    # Find best performer
    all_systems = []
    for name in ['chromadb', 'duckdb', 'lancedb', 'toondb']:
        if results.get(name):
            all_systems.append((name, results[name]))
    
    if all_systems:
        # Best insert rate
        best_insert = max(all_systems, key=lambda x: x[1]['insert_rate'])
        # Best search latency (lowest)
        best_search = min(all_systems, key=lambda x: x[1]['avg_latency'])
        
        print(f"\n  Best Insert: {best_insert[0].upper()} ({best_insert[1]['insert_rate']:.0f} vec/s)")
        print(f"  Best Search: {best_search[0].upper()} ({best_search[1]['avg_latency']:.3f}ms)")
        
        if results.get('toondb'):
            t = results['toondb']
            print(f"\n  ToonDB vs competition:")
            for name, r in all_systems:
                if name != 'toondb':
                    insert_ratio = t['insert_rate'] / max(r['insert_rate'], 0.001)
                    search_ratio = r['avg_latency'] / max(t['avg_latency'], 0.001)
                    print(f"    vs {name}: {insert_ratio:.1f}x insert, {search_ratio:.1f}x search")
            
            if best_insert[0] == 'toondb' and best_search[0] == 'toondb':
                print(f"\n  🏆 OVERALL WINNER: ToonDB (faster on BOTH metrics!)")
            elif best_search[0] == 'toondb':
                print(f"\n  🏆 OVERALL: ToonDB for search, {best_insert[0]} for insert")
            else:
                print(f"\n  🏆 OVERALL: {best_search[0]} for search, {best_insert[0]} for insert")


# =============================================================================
# PERF-RUN INTEGRATION
# =============================================================================

def run_benchmark(params: dict, runs: int = 3) -> dict:
    """
    Run benchmark and return standardized metrics for perf-run harness.
    
    Args:
        params: Benchmark parameters from workload TOML
        runs: Number of timed runs
        
    Returns:
        Dictionary of metric_name -> value for JSON output
    """
    # Extract params with defaults
    num_vectors = params.get('n_vectors', 10000)
    dim = params.get('dimension', 128)
    num_queries = params.get('n_queries', 100)
    k = params.get('k', 10)
    
    # Generate data
    vectors = generate_vectors(num_vectors, dim)
    queries = generate_vectors(num_queries, dim, seed=123)
    
    # Collect results across runs
    all_results = {
        'toondb': [],
        'chromadb': [],
        'numpy': [],
    }
    
    for run_idx in range(runs):
        # ToonDB
        try:
            result = benchmark_toondb(vectors, queries, k)
            if result:
                all_results['toondb'].append(result)
        except Exception as e:
            print(f"ToonDB run {run_idx+1} failed: {e}")
        
        # ChromaDB
        try:
            result = benchmark_chromadb(vectors, queries, k)
            if result:
                all_results['chromadb'].append(result)
        except Exception as e:
            print(f"ChromaDB run {run_idx+1} failed: {e}")
        
        # NumPy baseline
        try:
            result = benchmark_numpy_bruteforce(vectors, queries, k)
            if result:
                all_results['numpy'].append(result)
        except Exception as e:
            print(f"NumPy run {run_idx+1} failed: {e}")
    
    # Calculate medians
    def median(values):
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return sorted_vals[n // 2]
    
    metrics = {}
    
    # ToonDB metrics
    if all_results['toondb']:
        toondb_runs = all_results['toondb']
        metrics['toondb_insert_vec_per_s'] = median([r['insert_rate'] for r in toondb_runs])
        metrics['toondb_search_ms_avg'] = median([r['avg_latency'] for r in toondb_runs])
        metrics['toondb_search_ms_p50'] = median([r['p50'] for r in toondb_runs])
        metrics['toondb_search_ms_p99'] = median([r['p99'] for r in toondb_runs])
        metrics['toondb_qps'] = 1000.0 / metrics['toondb_search_ms_avg'] if metrics['toondb_search_ms_avg'] > 0 else 0
    
    # ChromaDB metrics
    if all_results['chromadb']:
        chroma_runs = all_results['chromadb']
        metrics['chromadb_insert_vec_per_s'] = median([r['insert_rate'] for r in chroma_runs])
        metrics['chromadb_search_ms_avg'] = median([r['avg_latency'] for r in chroma_runs])
        metrics['chromadb_search_ms_p50'] = median([r['p50'] for r in chroma_runs])
        metrics['chromadb_search_ms_p99'] = median([r['p99'] for r in chroma_runs])
        metrics['chromadb_qps'] = 1000.0 / metrics['chromadb_search_ms_avg'] if metrics['chromadb_search_ms_avg'] > 0 else 0
    
    # NumPy metrics  
    if all_results['numpy']:
        numpy_runs = all_results['numpy']
        metrics['numpy_insert_vec_per_s'] = median([r['insert_rate'] for r in numpy_runs])
        metrics['numpy_search_ms_avg'] = median([r['avg_latency'] for r in numpy_runs])
    
    return metrics


if __name__ == "__main__":
    main()

