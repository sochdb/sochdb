"""
File-based bulk operations (subprocess path).

This module provides the subprocess-based bulk operations for cases where:
- Dataset is too large to fit in memory
- You need to process files directly (avoiding Python memory)
- You're running in an environment without the native extension

For most use cases, use toondb.build_index_from_numpy() instead.
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
    """Find the toondb-bulk binary."""
    # Check PATH
    binary = shutil.which("toondb-bulk")
    if binary:
        return Path(binary)
    
    # Check cargo target
    for root in [Path(__file__).parent.parent.parent.parent, Path.cwd()]:
        for profile in ["release", "debug"]:
            path = root / "target" / profile / "toondb-bulk"
            if path.exists():
                return path
    
    return None


def bulk_build_from_file(
    input_path: str,
    output_path: str,
    *,
    dimension: int | None = None,
    m: int = 16,
    ef_construction: int = 100,
    batch_size: int = 1000,
    threads: int = 0,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Build HNSW index from a file using the toondb-bulk CLI.
    
    This is the offline/batch processing path. For in-memory NumPy arrays,
    use toondb.build_index_from_numpy() which is 10x faster.
    
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
        RuntimeError: If toondb-bulk binary is not found.
        subprocess.CalledProcessError: If build fails.
    """
    import time
    
    bulk_path = _find_bulk_binary()
    if bulk_path is None:
        raise RuntimeError(
            "Could not find toondb-bulk binary. Options:\n"
            "  1. Use build_index_from_numpy() (recommended, no binary needed)\n"
            "  2. Build: cargo build --release -p toondb-tools\n"
            "  3. Install: cargo install --path toondb-tools"
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
    Query HNSW index using toondb-bulk CLI.
    
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
        raise RuntimeError("toondb-bulk binary not found")
    
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
