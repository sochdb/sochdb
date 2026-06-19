"""
File-based bulk operations (subprocess path).

This module provides the subprocess-based bulk operations for cases where:
- Dataset is too large to fit in memory
- You need to process files directly (avoiding Python memory)
- You're running in an environment without the native extension

For most use cases, use sochdb.build_index_from_numpy() instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _find_bulk_binary() -> Path | None:
    """Find the sochdb-bulk binary."""
    # Check PATH
    binary = shutil.which("sochdb-bulk")
    if binary:
        return Path(binary)
    
    # Check cargo target
    for root in [Path(__file__).parent.parent.parent.parent, Path.cwd()]:
        for profile in ["release", "debug"]:
            path = root / "target" / profile / "sochdb-bulk"
            if path.exists():
                return path
    
    return None


def bulk_build_from_file(
    input_path: str,
    output_path: str,
    *,
    dimension: int | None = None,
    m: int | None = None,
    ef_construction: int | None = None,
    batch_size: int = 1000,
    threads: int = 0,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Build HNSW index from a file using the sochdb-bulk CLI.
    
    This is the offline/batch processing path. For in-memory NumPy arrays,
    use sochdb.build_index_from_numpy() which is 10x faster.
    
    Args:
        input_path: Path to vector file (.npy or raw .f32).
        output_path: Path to save the HNSW index.
        dimension: Vector dimension (auto-detected for .npy).
        m: HNSW max connections.
        ef_construction: Construction search depth.
        batch_size: Vectors per batch.
        threads: Number of threads (0 = auto).
        quiet: Suppress progress output.
    
    Returns:
        Dict with build statistics.
    
    Raises:
        RuntimeError: If sochdb-bulk binary is not found.
        subprocess.CalledProcessError: If build fails.
    """
    import time

    # The CLI requires concrete ints. build_index_from_file resolves
    # dimension-aware values when the dimension is known up front; otherwise
    # fall back to safe constants (m=16 stays correct for low-dim data; the
    # ef_construction floor is bumped 100 -> 200 to match recommended params).
    if m is None:
        m = 16
    if ef_construction is None:
        ef_construction = 200

    bulk_path = _find_bulk_binary()
    if bulk_path is None:
        raise RuntimeError(
            "Could not find sochdb-bulk binary. Options:\n"
            "  1. Use build_index_from_numpy() (recommended, no binary needed)\n"
            "  2. Build: cargo build --release -p sochdb-tools\n"
            "  3. Install: cargo install --path sochdb-tools"
        )
    
    cmd = [
        str(bulk_path),
        "build-index",
        "--input", str(input_path),
        "--output", str(output_path),
        "--max-connections", str(m),
        "--ef-construction", str(ef_construction),
        "--batch-size", str(batch_size),
        "--threads", str(threads),
    ]
    
    if dimension is not None:
        cmd.extend(["--dimension", str(dimension)])
    
    if quiet:
        cmd.append("--quiet")
    
    start = time.perf_counter()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    
    elapsed = time.perf_counter() - start
    output_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    # Parse output for vector count if available
    vectors = 0
    for line in result.stderr.split('\n'):
        if 'Inserted' in line:
            try:
                vectors = int(line.split()[1])
            except (IndexError, ValueError):
                pass
    
    return {
        "vectors": vectors,
        "elapsed_secs": elapsed,
        "rate": vectors / elapsed if elapsed > 0 and vectors > 0 else 0,
        "output_size_mb": output_size,
        "command": cmd,
    }


def bulk_query_from_file(
    index_path: str,
    query_path: str,
    k: int = 10,
    ef_search: int | None = None,
) -> list[dict]:
    """
    Query HNSW index using sochdb-bulk CLI.
    
    For in-memory queries, use index.search() directly which is faster.
    
    Args:
        index_path: Path to HNSW index.
        query_path: Path to query vector file (.f32).
        k: Number of neighbors.
        ef_search: Search depth.
    
    Returns:
        List of dicts with 'id' and 'distance'.
    """
    bulk_path = _find_bulk_binary()
    if bulk_path is None:
        raise RuntimeError("sochdb-bulk binary not found")
    
    cmd = [
        str(bulk_path),
        "query",
        "--index", str(index_path),
        "--query", str(query_path),
        "--k", str(k),
    ]
    
    if ef_search is not None:
        cmd.extend(["--ef", str(ef_search)])
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    
    # Parse results from stdout
    results = []
    for line in result.stdout.strip().split('\n'):
        if line and ':' in line:
            try:
                parts = line.split()
                id_part = parts[0].rstrip(':')
                dist_part = parts[1] if len(parts) > 1 else "0"
                results.append({
                    "id": int(id_part),
                    "distance": float(dist_part),
                })
            except (IndexError, ValueError):
                pass
    
    return results
