#!/usr/bin/env python3
"""
ToonDB Bulk Operations Example
==============================

This example demonstrates high-performance bulk operations that bypass
Python FFI overhead by using the native toondb-bulk CLI.

Key Benefits:
- ~10x faster than FFI for large datasets
- Zero Python↔Rust marshalling during build
- Memory-efficient streaming

Expected Output:
    ✓ Binary found at target/release/toondb-bulk
    ✓ Built 10000 vectors at ~1000+ vec/s
    ✓ Index saved to temp file

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 examples/python/03_bulk_operations.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

import numpy as np


def main():
    print("=" * 60)
    print("  ToonDB Bulk Operations Example")
    print("=" * 60)
    
    try:
        from toondb.bulk import bulk_build_index, get_toondb_bulk_path
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return 1
    
    # 1. Check binary
    print("\n[1] Checking toondb-bulk binary...")
    try:
        binary_path = get_toondb_bulk_path()
        print(f"    ✓ Binary found at: {binary_path}")
    except RuntimeError as e:
        print(f"    ❌ Binary not found: {e}")
        print("    Build with: cargo build --release -p toondb-tools")
        return 1
    
    # 2. Generate test vectors
    print("\n[2] Generating test vectors...")
    n_vectors = 10000
    dimension = 384  # Smaller dimension for faster test
    
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    print(f"    Generated {n_vectors} × {dimension}D vectors ({vectors.nbytes / 1024 / 1024:.1f} MB)")
    
    # 3. Build index using bulk API
    print("\n[3] Building index with bulk API...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "bulk_index.hnsw")
        
        try:
            stats = bulk_build_index(
                vectors,
                output=output_path,
                m=16,
                ef_construction=48,  # Lower for speed
                quiet=False,
            )
            
            print(f"\n    Results:")
            print(f"    - Vectors: {stats.vectors}")
            print(f"    - Dimension: {stats.dimension}")
            print(f"    - Time: {stats.elapsed_secs:.2f}s")
            print(f"    - Rate: {stats.rate:.0f} vec/s")
            print(f"    - Output size: {stats.output_size_mb:.1f} MB")
            
            # Check if index was created
            if os.path.exists(output_path):
                print(f"\n    ✓ Index saved to: {output_path}")
            else:
                print(f"\n    ❌ Index file not created")
                return 1
            
        except Exception as e:
            print(f"\n    ❌ Bulk build failed: {e}")
            return 1
    
    print("\n" + "=" * 60)
    print("  ✅ Bulk operations example complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
