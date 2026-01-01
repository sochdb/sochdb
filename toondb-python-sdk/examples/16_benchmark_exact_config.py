#!/usr/bin/env python3
"""Test with EXACT benchmark configuration."""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from toondb.vector import VectorIndex

def test_exact_benchmark_config():
    print("=" * 80)
    print("TESTING WITH EXACT BENCHMARK CONFIGURATION")
    print("=" * 80)
    
    # EXACT benchmark settings
    dimension = 768
    n_vectors = 10000
    max_connections = 16
    ef_construction = 48
    
    print(f"Configuration:")
    print(f"  Dimension: {dimension}")
    print(f"  Vectors: {n_vectors}")
    print(f"  Max connections: {max_connections}")
    print(f"  EF construction: {ef_construction}")
    print()
    
    # Generate vectors exactly like benchmark
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = np.arange(n_vectors, dtype=np.uint64)
    
    print(f"Data layout:")
    print(f"  IDs dtype: {ids.dtype}, contiguous: {ids.flags['C_CONTIGUOUS']}")
    print(f"  Vectors dtype: {vectors.dtype}, contiguous: {vectors.flags['C_CONTIGUOUS']}")
    print()
    
    # Create index with exact benchmark settings
    index = VectorIndex(
        dimension=dimension,
        max_connections=max_connections,
        ef_construction=ef_construction
    )
    
    # WARMUP (critical for accurate timing)
    print("Warming up...")
    warmup_ids = np.arange(100, dtype=np.uint64)
    warmup_vecs = vectors[:100].copy()
    index.insert_batch_fast(warmup_ids + 1000000, warmup_vecs)
    
    # TIMED RUN (exactly like benchmark)
    print("Running timed insert...")
    start = time.perf_counter()
    inserted = index.insert_batch_fast(ids, vectors)
    elapsed = time.perf_counter() - start
    
    throughput = inserted / elapsed
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Inserted: {inserted}/{n_vectors}")
    print(f"  Throughput: {throughput:.0f} vec/s")
    print(f"  Benchmark target: 800+ vec/s (absolute min)")
    print(f"  Rust baseline claim: ~1600 vec/s")
    print()
    
    # Performance assessment
    if throughput >= 1500:
        print(f"  🎯 EXCELLENT! {throughput:.0f} vec/s - matches Rust baseline")
    elif throughput >= 800:
        print(f"  ✅ PASSES minimum threshold ({throughput:.0f} vec/s >= 800)")
        ratio = throughput / 1600  # vs claimed Rust baseline
        print(f"  📊 FFI efficiency: {ratio:.1%} of claimed Rust performance")
    else:
        print(f"  ❌ BELOW minimum threshold ({throughput:.0f} < 800 vec/s)")
    
    # Test correctness
    print("\nVerifying correctness...")
    test_vector = vectors[0]
    results = index.search(test_vector, k=1)
    
    if results and len(results) > 0:
        found_id, distance = results[0]
        print(f"  Self-search: ID {found_id}, distance {distance:.6f}")
        if distance < 0.1:
            print(f"  ✅ Correctness verified")
        else:
            print(f"  ⚠️ High distance - possible correctness issue")
    else:
        print(f"  ❌ Search failed")
    
    return throughput

def compare_with_chromadb():
    """Compare against ChromaDB baseline."""
    print("\n" + "=" * 80)
    print("CHROMADB COMPARISON")
    print("=" * 80)
    
    toondb_perf = test_exact_benchmark_config()
    chromadb_perf = 14303  # From user's benchmark
    
    gap = chromadb_perf / toondb_perf if toondb_perf > 0 else float('inf')
    
    print(f"\n📊 PERFORMANCE GAP ANALYSIS:")
    print(f"   ChromaDB:  {chromadb_perf:,} vec/s")
    print(f"   ToonDB:    {toondb_perf:,.0f} vec/s")
    print(f"   Gap:       {gap:.1f}x slower")
    print()
    
    if gap < 2:
        print(f"   🎯 Competitive! Less than 2x gap")
    elif gap < 5:
        print(f"   📈 Reasonable gap, room for optimization")
    else:
        print(f"   🔍 Significant gap - needs investigation")
    
    return toondb_perf, chromadb_perf

if __name__ == '__main__':
    try:
        toondb_perf, chromadb_perf = compare_with_chromadb()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()