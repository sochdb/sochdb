#!/usr/bin/env python3
"""
FFI Overhead Analysis

This script investigates the performance gap between:
1. Pure Rust HNSW core performance (~1500 vec/s claimed)  
2. Python FFI performance (~851 vec/s in benchmarks)
3. Our profiling results (~123 vec/s)

Let's identify if the issue is:
- Single vs batch insert API usage
- Memory allocation/copying in FFI
- Different test conditions (vector count, dimension, etc.)
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from toondb.vector import VectorIndex

def test_batch_vs_single_insert():
    """Test if benchmark is using single inserts instead of efficient batch."""
    
    print("=" * 80)
    print("BATCH vs SINGLE INSERT PERFORMANCE TEST")
    print("=" * 80)
    
    np.random.seed(42)
    vectors = np.random.randn(1000, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Test 1: Batch insert (what we expect to be fast)
    print("\n1. BATCH INSERT (optimized path)")
    index_batch = VectorIndex(dimension=768, ef_construction=100)
    ids = list(range(1000))
    
    start = time.perf_counter()
    result_batch = index_batch.insert_batch(ids, vectors)
    batch_time = time.perf_counter() - start
    batch_throughput = 1000 / batch_time
    
    print(f"   Time: {batch_time:.3f}s")
    print(f"   Throughput: {batch_throughput:.0f} vec/s")
    print(f"   Result: {result_batch}")
    
    # Test 2: Single insert loop (what might be slow)
    print("\n2. SINGLE INSERT LOOP (potentially slow path)")
    index_single = VectorIndex(dimension=768, ef_construction=100)
    
    start = time.perf_counter()
    inserted = 0
    for i, vector in enumerate(vectors[:100]):  # Only test 100 for speed
        try:
            result = index_single.insert(i, vector)
            if result:
                inserted += 1
        except Exception as e:
            print(f"   Error inserting vector {i}: {e}")
            break
    single_time = time.perf_counter() - start
    
    # Extrapolate to 1000 vectors
    single_throughput = 100 / single_time
    estimated_1k_time = single_time * 10
    
    print(f"   Time (100 vectors): {single_time:.3f}s")
    print(f"   Throughput: {single_throughput:.0f} vec/s")
    print(f"   Inserted: {inserted}/100")
    print(f"   Estimated 1K time: {estimated_1k_time:.3f}s")
    
    slowdown = batch_throughput / single_throughput if single_throughput > 0 else 0
    print(f"\n📊 BATCH vs SINGLE: {slowdown:.1f}x faster with batch insert")
    
    return batch_throughput, single_throughput


def test_memory_overhead():
    """Test if memory allocation/copying is causing overhead."""
    
    print("\n" + "=" * 80)
    print("MEMORY ALLOCATION OVERHEAD TEST")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Test different vector counts to see scaling
    vector_counts = [100, 500, 1000, 2000]
    
    for count in vector_counts:
        print(f"\nTesting {count} vectors...")
        
        vectors = np.random.randn(count, 768).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = list(range(count))
        
        # Test memory preparation time
        prep_start = time.perf_counter()
        # Simulate what happens inside insert_batch
        ids_arr = np.ascontiguousarray(ids, dtype=np.uint64)
        vectors_contiguous = np.ascontiguousarray(vectors, dtype=np.float32)
        prep_time = time.perf_counter() - prep_start
        
        # Test actual insert
        index = VectorIndex(dimension=768, ef_construction=100)
        
        start = time.perf_counter()
        result = index.insert_batch(ids, vectors)
        insert_time = time.perf_counter() - start
        
        throughput = count / insert_time
        prep_pct = (prep_time / insert_time) * 100
        
        print(f"   Prep time: {prep_time*1000:.3f}ms ({prep_pct:.1f}% of total)")
        print(f"   Insert time: {insert_time:.3f}s")
        print(f"   Throughput: {throughput:.0f} vec/s")
        print(f"   Result: {result}")


def test_different_configurations():
    """Test if benchmark is using different HNSW parameters."""
    
    print("\n" + "=" * 80)  
    print("CONFIGURATION SENSITIVITY TEST")
    print("=" * 80)
    
    np.random.seed(42)
    vectors = np.random.randn(1000, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = list(range(1000))
    
    configs = [
        ("Very Fast (ef=25, M=8)", 25, 8),
        ("Fast (ef=50, M=12)", 50, 12), 
        ("Balanced (ef=100, M=16)", 100, 16),
        ("Quality (ef=150, M=20)", 150, 20),
    ]
    
    for name, ef, m in configs:
        print(f"\nTesting {name}...")
        
        index = VectorIndex(
            dimension=768, 
            ef_construction=ef,
            max_connections=m
        )
        
        start = time.perf_counter()
        result = index.insert_batch(ids, vectors)
        insert_time = time.perf_counter() - start
        
        throughput = 1000 / insert_time
        print(f"   Throughput: {throughput:.0f} vec/s")
        print(f"   Time: {insert_time:.3f}s")
        print(f"   Result: {result}")


def test_rust_core_claim():
    """Try to verify the claim that Rust core can do 1.5K+ vec/s."""
    
    print("\n" + "=" * 80)
    print("RUST CORE PERFORMANCE VERIFICATION")  
    print("=" * 80)
    
    # Test with most aggressive settings to see if we can reach 1500 vec/s
    print("\nTesting most aggressive settings...")
    
    np.random.seed(42)
    vectors = np.random.randn(1000, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = list(range(1000))
    
    # Ultra-fast settings
    index = VectorIndex(
        dimension=768,
        ef_construction=25,  # Very low
        max_connections=8,   # Very low
    )
    
    start = time.perf_counter()
    result = index.insert_batch(ids, vectors)
    insert_time = time.perf_counter() - start
    
    throughput = 1000 / insert_time
    print(f"   Ultra-fast settings: {throughput:.0f} vec/s")
    print(f"   Target: 1500+ vec/s")
    print(f"   Gap: {1500 / throughput if throughput > 0 else 'inf'}x slower than claim")
    
    if throughput < 1500:
        print(f"\n⚠️  Unable to reach claimed 1500 vec/s performance")
        print(f"   Either the claim is inaccurate or there are additional optimizations")
    else:
        print(f"\n✅ Reached target performance!")


def main():
    print("ToonDB FFI Overhead Investigation")
    print("Goal: Understand why benchmark shows 851 vec/s vs claimed 1.5K+ Rust core")
    
    try:
        # Test 1: Batch vs Single insert
        batch_perf, single_perf = test_batch_vs_single_insert()
        
        # Test 2: Memory overhead  
        test_memory_overhead()
        
        # Test 3: Different configurations
        test_different_configurations()
        
        # Test 4: Verify Rust core claim
        test_rust_core_claim()
        
        print("\n" + "=" * 80)
        print("CONCLUSIONS & RECOMMENDATIONS")
        print("=" * 80)
        
        print(f"""
1. BATCH INSERT PERFORMANCE: {batch_perf:.0f} vec/s
   - This is our best Python FFI performance
   - Still far from ChromaDB's 14,303 vec/s
   
2. FFI OVERHEAD ANALYSIS:
   - If Rust core can truly do 1500+ vec/s, then FFI has {1500/batch_perf if batch_perf > 0 else 'inf'}x overhead
   - This suggests significant inefficiency in the FFI layer
   
3. POSSIBLE ISSUES:
   - Memory copying between Python and Rust
   - FFI call overhead for large batches
   - Different algorithm implementation vs ChromaDB
   - Suboptimal HNSW parameter defaults
   
4. NEXT STEPS:
   - Profile the Rust-only performance (no Python FFI)
   - Compare HNSW algorithm with ChromaDB's implementation
   - Investigate zero-copy FFI optimizations
   - Consider using a different vector database API design
""")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()