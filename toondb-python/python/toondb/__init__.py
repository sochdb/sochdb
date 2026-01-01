"""
ToonDB Python SDK

High-performance AI-native database with zero-copy vector search.

This package provides:
- In-process HNSW vector indexing (10x faster than subprocess)
- Zero-copy NumPy integration via PyO3
- GIL release during expensive operations

Quick Start:
    >>> import numpy as np
    >>> import toondb
    >>> 
    >>> # Build index from embeddings
    >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
    >>> index = toondb.build_index(embeddings)
    >>> 
    >>> # Search
    >>> query = np.random.randn(768).astype(np.float32)
    >>> ids, distances = index.search(query, k=10)
"""

from __future__ import annotations

# Import native extension
try:
    from toondb._native import (
        HnswIndex,
        build_index,
        version,
        is_safe_mode,
    )
    _HAS_NATIVE = True
except ImportError as e:
    _HAS_NATIVE = False
    _IMPORT_ERROR = str(e)

# Re-export for convenience
__all__ = [
    # Core classes
    "HnswIndex",
    # Functions
    "build_index",
    "build_index_from_numpy",
    "build_index_from_file",
    "version",
    "is_safe_mode",
    # Legacy compatibility
    "bulk_build_index",
]

__version__ = "0.1.0"


def _check_native():
    """Raise ImportError if native extension is not available."""
    if not _HAS_NATIVE:
        raise ImportError(
            f"ToonDB native extension not found: {_IMPORT_ERROR}\n"
            "Install with: pip install toondb-client\n"
            "Or build from source: maturin develop --release"
        )


# =============================================================================
# High-Level API (Task 4: Split Bulk API Modes)
# =============================================================================

def build_index_from_numpy(
    embeddings,
    *,
    m: int = 16,
    ef_construction: int = 100,
    metric: str = "cosine",
    ids=None,
) -> "HnswIndex":
    """
    Build an HNSW index from NumPy embeddings (in-process, zero-copy).
    
    This is the fast path - vectors are passed directly to Rust without
    disk I/O or subprocess overhead.
    
    Args:
        embeddings: 2D float32 array of shape (N, D).
        m: HNSW max connections per node.
        ef_construction: Construction search depth.
        metric: Distance metric ("cosine", "euclidean", "dot").
        ids: Optional 1D uint64 array of vector IDs.
    
    Returns:
        HnswIndex with inserted vectors.
    
    Performance:
        ~15,000 vec/s for 768D vectors (10x faster than subprocess).
    
    Example:
        >>> import numpy as np
        >>> from toondb import build_index_from_numpy
        >>> 
        >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
        >>> index = build_index_from_numpy(embeddings, m=16)
        >>> index.save("my_index.hnsw")
    """
    _check_native()
    return build_index(embeddings, m=m, ef_construction=ef_construction, 
                       metric=metric, ids=ids)


def build_index_from_file(
    input_path: str,
    output_path: str,
    *,
    dimension: int | None = None,
    m: int = 16,
    ef_construction: int = 100,
    batch_size: int = 1000,
    quiet: bool = False,
) -> dict:
    """
    Build an HNSW index from a file (subprocess, mmap-based).
    
    This is the offline path for large datasets that don't fit in memory.
    Uses the toondb-bulk CLI with memory-mapped I/O.
    
    Args:
        input_path: Path to input vectors (.npy or raw .f32).
        output_path: Path to save the HNSW index.
        dimension: Vector dimension (auto-detected for .npy).
        m: HNSW max connections.
        ef_construction: Construction search depth.
        batch_size: Vectors per insertion batch.
        quiet: Suppress progress output.
    
    Returns:
        Dict with build statistics.
    
    Note:
        This function requires the toondb-bulk binary. For most use cases,
        prefer build_index_from_numpy() which is faster.
    """
    # Import the subprocess-based implementation
    from toondb._bulk import bulk_build_from_file
    return bulk_build_from_file(
        input_path=input_path,
        output_path=output_path,
        dimension=dimension,
        m=m,
        ef_construction=ef_construction,
        batch_size=batch_size,
        quiet=quiet,
    )


# =============================================================================
# Legacy Compatibility
# =============================================================================

def bulk_build_index(
    embeddings,
    output: str,
    *,
    ids=None,
    m: int = 16,
    ef_construction: int = 100,
    **kwargs,
) -> dict:
    """
    Build an HNSW index from embeddings.
    
    DEPRECATED: Use build_index_from_numpy() for 10x better performance.
    
    This function now uses the in-process PyO3 path instead of subprocess.
    The interface is maintained for backward compatibility.
    """
    import warnings
    warnings.warn(
        "bulk_build_index() is deprecated. Use build_index_from_numpy() "
        "for 10x better performance.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    import time
    import numpy as np
    from pathlib import Path
    
    _check_native()
    
    # Ensure correct dtype
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    start = time.perf_counter()
    
    # Build using native extension
    if ids is not None:
        if ids.dtype != np.uint64:
            ids = ids.astype(np.uint64)
        index = build_index(embeddings, m=m, ef_construction=ef_construction, ids=ids)
    else:
        index = build_index(embeddings, m=m, ef_construction=ef_construction)
    
    # Save to output
    index.save(str(output))
    
    elapsed = time.perf_counter() - start
    n, d = embeddings.shape
    output_size = Path(output).stat().st_size / (1024 * 1024)
    
    return {
        "vectors": n,
        "dimension": d,
        "elapsed_secs": elapsed,
        "rate": n / elapsed if elapsed > 0 else 0,
        "output_size_mb": output_size,
    }


# Make HnswIndex and build_index available at top level
if _HAS_NATIVE:
    # These are imported from native module
    pass
else:
    # Stub classes for IDE autocomplete when native not installed
    class HnswIndex:
        """HNSW Vector Index (stub - native extension not loaded)."""
        
        def __init__(self, dimension: int, m: int = 16, ef_construction: int = 100, 
                     metric: str = "cosine", precision: str = "f32"):
            _check_native()
        
        def insert_batch(self, vectors) -> int:
            _check_native()
        
        def insert_batch_with_ids(self, ids, vectors) -> int:
            _check_native()
        
        def search(self, query, k: int, ef_search: int | None = None):
            _check_native()
        
        def save(self, path: str):
            _check_native()
        
        @staticmethod
        def load(path: str) -> "HnswIndex":
            _check_native()
