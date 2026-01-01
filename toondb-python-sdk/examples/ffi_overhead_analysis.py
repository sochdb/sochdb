#!/usr/bin/env python3
"""
FFI Overhead Analysis

Direct comparison of:
1. Pure Rust insert (via profiler binary)
2. Python FFI insert (via toondb module)

Goal: Identify specific sources of the 14x performance gap.
"""

import time
import numpy as np
import sys
import os

# Add toondb-python-sdk to path
sdk_path = os.path.join(os.path.dirname(__file__), 'toondb-python-sdk')
if os.path.exists(sdk_path):
    sys.path.insert(0, sdk_path)

try:
    from toondb import HnswIndex
    TOONDB_AVAILABLE = True
except ImportError as e:
    print(f"ToonDB not available: {e}")
    TOONDB_AVAILABLE = False

def benchmark_ffi_overhead():
    """Test various batch sizes to identify FFI bottlenecks."""
    
    if not TOONDB_AVAILABLE:
        print("❌ ToonDB not available - skipping FFI benchmark")
        return
    
    print("🔬 FFI Overhead Analysis")
    print("=" * 60)
    
    # Test configurations
    configs = [
        (128, 1000),   # Small dimension, medium batch
        (128, 5000),   # Small dimension, large batch
        (768, 1000),   # Large dimension, medium batch
        (768, 5000),   # Large dimension, large batch
    ]
    
    for dim, n_vectors in configs:
        print(f"\n--- {dim}D × {n_vectors} vectors ---")
        
        # Create index
        index = HnswIndex(dimension=dim, m=16, ef_construction=100)
        
        # Generate test data - ensure C-contiguous for zero-copy
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors = np.ascontiguousarray(vectors)  # Ensure contiguous
        
        # Check contiguity
        print(f"Vectors contiguous: {vectors.flags['C_CONTIGUOUS']}")
        
        # Warmup
        index.insert_batch(vectors[:100])
        
        # Benchmark main insert
        start_time = time.perf_counter()
        count = index.insert_batch(vectors[100:])
        elapsed = time.perf_counter() - start_time
        
        actual_inserted = count  # Should be n_vectors - 100
        throughput = actual_inserted / elapsed
        
        print(f"FFI insert: {actual_inserted} vectors in {elapsed*1000:.1f}ms ({throughput:.0f} vec/s)")
        
        # Compare to profiler results (hardcoded for reference)
        if dim == 128:
            rust_throughput = 12000  # From profiler results
        elif dim == 768:
            rust_throughput = 2400   # From profiler results
        else:
            rust_throughput = "unknown"
        
        if rust_throughput != "unknown":
            overhead = rust_throughput / throughput if throughput > 0 else float('inf')
            print(f"Rust core: ~{rust_throughput} vec/s")
            print(f"FFI overhead: {overhead:.1f}x slower")
        
        # Test search performance (for comparison)
        query = np.random.randn(dim).astype(np.float32)
        search_start = time.perf_counter()
        ids, dists = index.search(query, k=10)
        search_elapsed = time.perf_counter() - search_start
        print(f"Search: {search_elapsed*1000:.2f}ms for 10-NN")

def analyze_memory_patterns():
    """Analyze potential memory allocation bottlenecks."""
    
    print("\n🧠 Memory Pattern Analysis")
    print("=" * 60)
    
    # Test different array configurations
    test_cases = [
        ("C-contiguous", lambda x, d: np.ascontiguousarray(x)),
        ("Fortran-contiguous", lambda x, d: np.asfortranarray(x)),
        ("Non-contiguous", lambda x, d: x[::2, :]),  # Skip every other row
    ]
    
    dim, n = 128, 1000
    
    for name, transform in test_cases:
        print(f"\n{name}:")
        
        base_vectors = np.random.randn(n*2, dim).astype(np.float32) 
        vectors = transform(base_vectors, dim)
        
        print(f"  Shape: {vectors.shape}")
        print(f"  C-contiguous: {vectors.flags['C_CONTIGUOUS']}")
        print(f"  F-contiguous: {vectors.flags['F_CONTIGUOUS']}")
        print(f"  Owns data: {vectors.flags['OWNDATA']}")
        
        if TOONDB_AVAILABLE and vectors.flags['C_CONTIGUOUS']:
            try:
                index = HnswIndex(dimension=dim, m=16, ef_construction=100)
                start_time = time.perf_counter()
                count = index.insert_batch(vectors[:n])  # Take first n rows
                elapsed = time.perf_counter() - start_time
                throughput = count / elapsed
                print(f"  Throughput: {throughput:.0f} vec/s")
            except Exception as e:
                print(f"  Error: {e}")
        elif not vectors.flags['C_CONTIGUOUS']:
            print(f"  ⚠️  Non-contiguous - would trigger copy")

if __name__ == "__main__":
    benchmark_ffi_overhead()
    analyze_memory_patterns()
    
    print("\n📊 Summary")
    print("=" * 60)
    print("Pure Rust (profiler):     128D: ~12,000 vec/s,  768D: ~2,400 vec/s") 
    print("Expected FFI performance: Need to reduce 14x overhead")
    print("\nNext: Profile specific FFI bottlenecks:")
    print("- PyO3 array slice creation")
    print("- GIL release/acquire cycles") 
    print("- Memory allocation patterns")
    print("- Rust method call overhead")