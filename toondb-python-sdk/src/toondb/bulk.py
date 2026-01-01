"""
ToonDB Bulk Operations API

This module provides high-performance bulk operations that bypass Python FFI overhead
by shelling out to the native `toondb-bulk` CLI binary.

Why Use Bulk API?
-----------------
Standard Python FFI has ~12× overhead for vector operations due to:
- O(N·d) memcpy per batch
- Python allocation tax
- GIL contention

The Bulk API achieves ~100% native throughput by:
- Writing vectors to mmap-friendly files (raw f32 or npy)
- Spawning `toondb-bulk` as a subprocess
- Zero Python ↔ Rust FFI marshalling during build

Performance Comparison (768D vectors)
-------------------------------------
| Method          | Throughput | Overhead |
|-----------------|------------|----------|
| Python FFI      | ~130 vec/s | 12× slower |
| Bulk API        | ~1,600 vec/s | 1.0× baseline |

Usage
-----
```python
import numpy as np
from toondb.bulk import bulk_build_index

# Generate or load embeddings
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index (bypasses FFI)
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,
    ef_construction=100,
)

print(f"Built {stats['vectors']} vectors at {stats['rate']:.0f} vec/s")
```
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import importlib.resources with fallback for older Python
if sys.version_info >= (3, 11):
    from importlib.resources import files, as_file
else:
    from importlib.resources import files
    from importlib.resources import as_file


@dataclass
class BulkBuildStats:
    """Statistics from a bulk index build."""
    vectors: int
    dimension: int
    elapsed_secs: float
    rate: float  # vectors per second
    output_size_mb: float
    command: list[str]
    
    def __repr__(self) -> str:
        return (
            f"BulkBuildStats(vectors={self.vectors}, dim={self.dimension}, "
            f"elapsed={self.elapsed_secs:.2f}s, rate={self.rate:.0f} vec/s, "
            f"output={self.output_size_mb:.1f} MB)"
        )


# =============================================================================
# Platform Detection (uv-style)
# =============================================================================
# 
# Wheel matrix:
#   - linux-x86_64    -> manylinux_2_17_x86_64
#   - linux-aarch64   -> manylinux_2_17_aarch64
#   - darwin-x86_64   -> macosx_11_0_x86_64
#   - darwin-aarch64  -> macosx_11_0_arm64
#   - darwin-universal2 -> macosx_11_0_universal2 (single binary for both)
#   - windows-x86_64  -> win_amd64
# =============================================================================

def _get_rust_target_triple() -> str:
    """
    Get the Rust target triple for the current platform.
    
    Returns a string like 'aarch64-apple-darwin', 'x86_64-unknown-linux-gnu'.
    This matches the directory structure used in wheel builds.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Normalize architecture
    if machine in ("x86_64", "amd64", "x64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    elif machine in ("i686", "i386", "x86"):
        arch = "i686"
    else:
        arch = machine
    
    # Map to Rust target triple
    if system == "darwin":
        return f"{arch}-apple-darwin"
    elif system.startswith("linux"):
        return f"{arch}-unknown-linux-gnu"
    elif system in ("windows", "win32", "cygwin", "msys"):
        return f"{arch}-pc-windows-msvc"
    else:
        return f"{arch}-unknown-{system}"


def _get_platform_tag() -> str:
    """
    Get the platform tag for binary lookup (legacy format).
    
    Returns a string like 'linux-x86_64', 'darwin-aarch64', 'windows-x86_64'.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Normalize OS
    if system == "darwin":
        system = "darwin"
    elif system.startswith("linux"):
        system = "linux"
    elif system in ("windows", "win32", "cygwin", "msys"):
        system = "windows"
    
    # Normalize architecture
    if machine in ("x86_64", "amd64", "x64"):
        machine = "x86_64"
    elif machine in ("aarch64", "arm64"):
        machine = "aarch64"
    elif machine in ("i686", "i386", "x86"):
        machine = "i686"
    
    return f"{system}-{machine}"


def _get_binary_name() -> str:
    """Get the binary name for the current platform."""
    if platform.system().lower() == "windows":
        return "toondb-bulk.exe"
    return "toondb-bulk"


def _find_bundled_binary() -> Path | None:
    """
    Find the bundled toondb-bulk binary using importlib.resources.
    
    This is the preferred method for installed packages.
    """
    try:
        # Get the _bin package
        pkg = files("toondb")
        rust_triple = _get_rust_target_triple()
        platform_tag = _get_platform_tag()
        binary_name = _get_binary_name()
        
        # Try platform-specific directory first (Rust triple, then legacy)
        candidates = [
            f"_bin/{rust_triple}/{binary_name}",  # Rust target triple (matches wheel)
            f"_bin/{platform_tag}/{binary_name}",  # Legacy platform tag
            # Fallback for universal2 on macOS
            f"_bin/darwin-universal2/{binary_name}" if platform_tag.startswith("darwin") else None,
        ]
        
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                binary_resource = pkg.joinpath(candidate)
                if binary_resource.is_file():
                    # Extract to a real path (needed for subprocess)
                    # For installed packages, files() returns a Traversable
                    # that needs to be converted to a real path
                    return Path(binary_resource)
            except (AttributeError, TypeError, FileNotFoundError):
                continue
                
    except (ImportError, ModuleNotFoundError, TypeError):
        pass
    
    return None


def _find_path_binary() -> Path | None:
    """Find toondb-bulk on PATH."""
    binary = shutil.which("toondb-bulk")
    if binary:
        return Path(binary)
    return None


def _find_dev_binary() -> Path | None:
    """Find binary in Cargo target directory (development mode)."""
    # Walk up from this file to find workspace root
    pkg_dir = Path(__file__).parent
    
    # toondb-python-sdk/src/toondb/bulk.py -> toondb/
    candidates = [
        pkg_dir.parent.parent.parent,  # src -> toondb-python-sdk -> toondb
        pkg_dir.parent.parent.parent.parent,  # In case of different structure
        Path.cwd(),
    ]
    
    binary_name = _get_binary_name()
    
    for workspace_root in candidates:
        for profile in ["release", "debug"]:
            cargo_binary = workspace_root / "target" / profile / binary_name
            if cargo_binary.exists() and cargo_binary.is_file():
                return cargo_binary
    
    return None


def _find_toondb_bulk() -> Path | None:
    """
    Find the toondb-bulk binary.
    
    Search order:
    1. Bundled in package (_bin/<platform>/toondb-bulk) via importlib.resources
    2. Legacy: Direct file path in package directory
    3. PATH (for system-wide installation)
    4. Cargo target directory (for development)
    """
    # 1. Try importlib.resources (preferred for installed packages)
    binary = _find_bundled_binary()
    if binary and binary.exists():
        return binary
    
    # 2. Legacy: Direct file path check
    pkg_dir = Path(__file__).parent
    bin_dir = pkg_dir / "_bin"
    if bin_dir.exists():
        rust_triple = _get_rust_target_triple()
        platform_tag = _get_platform_tag()
        binary_name = _get_binary_name()
        
        # Try Rust target triple first (matches wheel structure)
        rust_binary = bin_dir / rust_triple / binary_name
        if rust_binary.exists():
            return rust_binary
        
        # Try legacy platform tag
        platform_binary = bin_dir / platform_tag / binary_name
        if platform_binary.exists():
            return platform_binary
        
        # Try universal2 on macOS
        if platform_tag.startswith("darwin") or rust_triple.endswith("-apple-darwin"):
            universal_binary = bin_dir / "darwin-universal2" / binary_name
            if universal_binary.exists():
                return universal_binary
    
    # 3. Check PATH
    binary = _find_path_binary()
    if binary:
        return binary
    
    # 4. Development mode
    binary = _find_dev_binary()
    if binary:
        return binary
    
    return None


def get_toondb_bulk_path() -> Path:
    """
    Get the path to the toondb-bulk binary.
    
    Raises:
        RuntimeError: If binary is not found.
    """
    path = _find_toondb_bulk()
    if path is None:
        raise RuntimeError(
            "Could not find toondb-bulk binary. Either:\n"
            "  1. Install the toondb package with native binaries: pip install toondb-client\n"
            "  2. Build from source: cargo build --release -p toondb-tools\n"
            "  3. Add toondb-bulk to your PATH"
        )
    return path


def bulk_build_index(
    embeddings: NDArray[np.float32],
    output: str | Path,
    *,
    ids: NDArray[np.uint64] | None = None,
    m: int = 16,
    ef_construction: int = 100,
    batch_size: int = 1000,
    threads: int = 0,
    quiet: bool = False,
    cleanup_temp: bool = True,
) -> BulkBuildStats:
    """
    Build an HNSW index from embeddings, bypassing Python FFI.
    
    This function writes embeddings to a memory-mapped file and invokes
    the `toondb-bulk` CLI for maximum throughput.
    
    Args:
        embeddings: 2D float32 array of shape (N, D).
        output: Path to save the HNSW index.
        ids: Optional 1D uint64 array of IDs. If None, sequential IDs are used.
        m: HNSW max connections per node.
        ef_construction: HNSW construction search depth.
        batch_size: Vectors per insertion batch.
        threads: Number of threads (0 = auto).
        quiet: Suppress progress output.
        cleanup_temp: Remove temporary files after build.
    
    Returns:
        BulkBuildStats with performance metrics.
    
    Raises:
        ValueError: If embeddings shape is invalid.
        RuntimeError: If toondb-bulk binary is not found.
        subprocess.CalledProcessError: If the build process fails.
    
    Example:
        >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
        >>> stats = bulk_build_index(embeddings, "index.hnsw")
        >>> print(f"Built at {stats.rate:.0f} vec/s")
    """
    # Validate input
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")
    if embeddings.dtype != np.float32:
        raise ValueError(f"Expected float32, got {embeddings.dtype}")
    
    n, d = embeddings.shape
    
    if ids is not None:
        if ids.shape[0] != n:
            raise ValueError(f"IDs length ({ids.shape[0]}) != vectors ({n})")
        if ids.dtype != np.uint64:
            ids = ids.astype(np.uint64)
    
    # Find binary
    bulk_path = get_toondb_bulk_path()
    
    # Create temp directory for data files
    temp_dir = tempfile.mkdtemp(prefix="toondb_bulk_")
    temp_path = Path(temp_dir)
    
    try:
        # Write vectors to raw f32 format (mmap-friendly)
        vectors_file = temp_path / "vectors.f32"
        embeddings.tofile(vectors_file)
        
        # Write metadata
        meta_file = temp_path / "vectors.json"
        with open(meta_file, "w") as f:
            json.dump({"n": n, "dim": d}, f)
        
        # Write optional IDs
        ids_file = None
        if ids is not None:
            ids_file = temp_path / "ids.u64"
            ids.tofile(ids_file)
        
        # Build command
        output_path = Path(output).absolute()
        cmd = [
            str(bulk_path),
            "build-index",
            "--input", str(vectors_file),
            "--output", str(output_path),
            "--dimension", str(d),
            "--format", "raw_f32",
            "--max-connections", str(m),
            "--ef-construction", str(ef_construction),
            "--batch-size", str(batch_size),
            "--threads", str(threads),
        ]
        
        if ids_file:
            cmd.extend(["--ids", str(ids_file)])
        
        if quiet:
            cmd.append("--quiet")
        
        # Run build
        import time
        start = time.perf_counter()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        elapsed = time.perf_counter() - start
        
        # Get output file size
        output_size = output_path.stat().st_size / (1024 * 1024)
        
        return BulkBuildStats(
            vectors=n,
            dimension=d,
            elapsed_secs=elapsed,
            rate=n / elapsed if elapsed > 0 else 0,
            output_size_mb=output_size,
            command=cmd,
        )
        
    except subprocess.CalledProcessError as e:
        # Include stderr in error message
        raise RuntimeError(
            f"toondb-bulk failed:\n"
            f"  Command: {' '.join(e.cmd)}\n"
            f"  Exit code: {e.returncode}\n"
            f"  Stderr: {e.stderr}"
        ) from e
        
    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def bulk_query_index(
    index_path: str | Path,
    query: NDArray[np.float32],
    k: int = 10,
    ef: int | None = None,
) -> list[tuple[int, float]]:
    """
    Query an HNSW index using the bulk CLI.
    
    For single queries, the Python FFI is typically faster.
    This function is useful for testing or when FFI is unavailable.
    
    Args:
        index_path: Path to the HNSW index file.
        query: 1D float32 query vector.
        k: Number of neighbors to return.
        ef: Search expansion factor (default: k).
    
    Returns:
        List of (id, distance) tuples.
    """
    if query.ndim != 1:
        raise ValueError(f"Expected 1D query vector, got {query.ndim}D")
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    
    bulk_path = get_toondb_bulk_path()
    
    # Write query to temp file
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        query.tofile(f)
        query_file = f.name
    
    try:
        cmd = [
            str(bulk_path),
            "query",
            "--index", str(index_path),
            "--query", query_file,
            "--k", str(k),
        ]
        if ef is not None:
            cmd.extend(["--ef", str(ef)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse output
        results = []
        for line in result.stdout.strip().split("\n"):
            if "ID:" in line and "Distance:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "ID:":
                        id_val = int(parts[i + 1])
                    elif part == "Distance:":
                        dist_val = float(parts[i + 1])
                results.append((id_val, dist_val))
        
        return results
        
    finally:
        os.unlink(query_file)


def convert_embeddings_to_raw(
    embeddings: NDArray[np.float32],
    output: str | Path,
    *,
    metric: str | None = None,
) -> Path:
    """
    Convert embeddings to ToonDB's raw f32 format.
    
    This format is optimal for bulk loading:
    - Memory-mappable (no parsing)
    - No per-vector allocations
    - Directly compatible with SIMD operations
    
    Args:
        embeddings: 2D float32 array.
        output: Output file path.
        metric: Optional distance metric ("cosine", "l2", "ip").
    
    Returns:
        Path to the created file.
    
    File format:
        - Main file (N × D × 4 bytes): Row-major float32 vectors
        - Meta file (.json): {"n": N, "dim": D, "metric": "..."}
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    n, d = embeddings.shape
    output_path = Path(output)
    
    # Write vectors
    embeddings.tofile(output_path)
    
    # Write metadata
    meta = {"n": n, "dim": d}
    if metric:
        meta["metric"] = metric
    
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return output_path


def read_raw_embeddings(
    path: str | Path,
    dimension: int | None = None,
) -> NDArray[np.float32]:
    """
    Read embeddings from ToonDB's raw f32 format.
    
    Args:
        path: Path to the raw f32 file.
        dimension: Vector dimension. If None, reads from meta.json.
    
    Returns:
        2D float32 array of shape (N, D).
    """
    path = Path(path)
    
    # Try to get dimension from meta.json
    if dimension is None:
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            dimension = meta["dim"]
        else:
            raise ValueError(
                "Dimension required (no meta.json found). "
                "Use dimension= parameter."
            )
    
    # Memory-map the file
    data = np.memmap(path, dtype=np.float32, mode='r')
    n = len(data) // dimension
    return data.reshape(n, dimension)


__all__ = [
    "bulk_build_index",
    "bulk_query_index",
    "convert_embeddings_to_raw",
    "read_raw_embeddings",
    "BulkBuildStats",
    "get_toondb_bulk_path",
]
