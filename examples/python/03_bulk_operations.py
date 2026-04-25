#!/usr/bin/env python3

import os
import sys
import tempfile

import numpy as np
from sochdb import HnswIndex, build_index_from_numpy
from sochdb._bulk import bulk_build_from_file


def main():
    print("=" * 60)
    print("  SochDB Bulk Operations Example")
    print("=" * 60)

    n_vectors = 10000
    dimension = 384
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    print(f"\n    Generated {n_vectors} × {dimension}D vectors ({vectors.nbytes / 1024 / 1024:.1f} MB)")

    print("\n[1] Building index with build_index_from_numpy...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "bulk_index.hnsw")
        index = build_index_from_numpy(vectors, m=16, ef_construction=100)
        index.save(output_path)
        print(f"    ✓ Index saved to: {output_path}")
        ids, dists = index.search(vectors[0], k=5)
        print(f"    ✓ Query returned {len(ids)} results, distances: {dists[:5]}")

    print("\n[2] Building index with incremental insert_batch...")
    index2 = HnswIndex(dimension, m=16, ef_construction=100)
    batch_size = 1000
    for i in range(0, n_vectors, batch_size):
        batch = vectors[i:i + batch_size]
        index2.insert_batch(batch)
    ids2, dists2 = index2.search(vectors[0], k=5)
    print(f"    ✓ Inserted {n_vectors} vectors in {n_vectors // batch_size} batches")
    print(f"    ✓ Query returned {len(ids2)} results, distances: {dists2[:5]}")

    print("\n[3] Bulk build from file using bulk_build_from_file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_path = os.path.join(tmpdir, "vectors.bin")
        output_path = os.path.join(tmpdir, "file_bulk_index.hnsw")
        vectors.tofile(vec_path)
        bulk_build_from_file(vec_path, output_path, dimension=dimension, m=16, ef_construction=100)
        print(f"    ✓ Bulk build from file complete")
        loaded = HnswIndex(dimension)
        loaded.load(output_path)
        ids3, dists3 = loaded.search(vectors[0], k=5)
        print(f"    ✓ Query returned {len(ids3)} results, distances: {dists3[:5]}")

    print("\n" + "=" * 60)
    print("  Bulk operations example complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())