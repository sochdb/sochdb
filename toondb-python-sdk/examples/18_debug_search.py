#!/usr/bin/env python3
"""Debug search correctness issue."""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from toondb.vector import VectorIndex

def debug_search_issue():
    print("=" * 80)
    print("DEBUGGING SEARCH CORRECTNESS ISSUE")
    print("=" * 80)
    
    dimension = 768
    n_vectors = 100  # Small for debugging
    
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = np.arange(n_vectors, dtype=np.uint64)
    
    print(f"Creating index with {n_vectors} vectors...")
    
    # Use fast settings
    index = VectorIndex(dimension=dimension, max_connections=8, ef_construction=25)
    
    print("Inserting vectors...")
    inserted = index.insert_batch_fast(ids, vectors)
    print(f"Inserted: {inserted}/{n_vectors}")
    print(f"Index size: {len(index)}")
    
    print("\nTesting search...")
    
    # Test with first vector
    query_vector = vectors[0]
    print(f"Query vector ID: 0")
    print(f"Query vector shape: {query_vector.shape}")
    print(f"Query vector norm: {np.linalg.norm(query_vector):.3f}")
    
    # Search with different k values
    for k in [1, 5, 10]:
        print(f"\nSearching for k={k}:")
        try:
            results = index.search(query_vector, k=k)
            print(f"  Results: {len(results) if results else 0}")
            
            if results:
                for i, (found_id, distance) in enumerate(results[:3]):
                    print(f"    {i}: ID {found_id}, distance {distance:.6f}")
                    
                # Check if we found ourselves
                found_self = any(found_id == 0 for found_id, _ in results)
                print(f"  Found self (ID 0): {found_self}")
                
                if found_self:
                    self_distance = next(distance for found_id, distance in results if found_id == 0)
                    print(f"  Self-distance: {self_distance:.6f} (should be ≈0)")
                    
            else:
                print("  No results returned!")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nTesting manual distance calculation...")
    # Manual verification
    stored_vector = vectors[0]  # Should match query_vector
    manual_distance = np.linalg.norm(query_vector - stored_vector)
    print(f"Manual distance to self: {manual_distance:.6f}")
    
    # Test with different vectors
    print(f"\nTesting other vectors...")
    for test_id in [1, 5, 10]:
        if test_id < n_vectors:
            test_vector = vectors[test_id]
            results = index.search(test_vector, k=3)
            if results:
                found_self = any(found_id == test_id for found_id, _ in results)
                best_distance = results[0][1] if results else float('inf')
                print(f"  Vector {test_id}: found_self={found_self}, best_dist={best_distance:.6f}")
    
    return index

def test_pure_rust_claim():
    print("\n" + "=" * 80)
    print("TESTING PURE RUST PERFORMANCE CLAIM")
    print("=" * 80)
    
    # Let's see if there's a pure Rust benchmark we can run
    print("Looking for pure Rust benchmarks...")
    
    # Check if we can call the Rust benchmark directly
    try:
        import subprocess
        
        # Try to run the Rust benchmark
        result = subprocess.run([
            "/Users/sushanth/toondb/target/release/benchmarks"
        ], capture_output=True, text=True, timeout=30)
        
        print("Rust benchmark output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Could not run Rust benchmark: {e}")
    
    print("\nNote: The 1600 vec/s claim might be from a different test scenario")
    print("or configuration than what we're using in the FFI tests.")

if __name__ == '__main__':
    try:
        index = debug_search_issue()
        test_pure_rust_claim()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()