"""
SochDB Python SDK

High-performance AI-native database with zero-copy vector search.

This package provides:
- In-process HNSW vector indexing (10x faster than subprocess)
- Zero-copy NumPy integration via PyO3
- GIL release during expensive operations

Quick Start:
    >>> import numpy as np
    >>> import sochdb
    >>> 
    >>> # Build index from embeddings
    >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
    >>> index = sochdb.build_index(embeddings)
    >>> 
    >>> # Search
    >>> query = np.random.randn(768).astype(np.float32)
    >>> ids, distances = index.search(query, k=10)
"""

from __future__ import annotations

# Import native extension
try:
    from sochdb._native import (
        Database,
        HnswIndex,
        Transaction,
        build_index,
        version,
        is_safe_mode,
        TableDatabase,
    )
    _HAS_NATIVE = True
except ImportError as e:
    _HAS_NATIVE = False
    _IMPORT_ERROR = str(e)

# Re-export for convenience
__all__ = [
    # Core classes
    "Database",
    "HnswIndex",
    "TableDatabase",
    "Transaction",
    # Functions
    "build_index",
    "build_index_from_numpy",
    "build_index_from_file",
    "recommended_hnsw_params",
    "version",
    "is_safe_mode",
    # Scale-out
    "MultiShardHnswIndex",
    # Legacy compatibility
    "bulk_build_index",
]

__version__ = "2.0.2"


def _check_native():
    """Raise ImportError if native extension is not available."""
    if not _HAS_NATIVE:
        raise ImportError(
            f"SochDB native extension not found: {_IMPORT_ERROR}\n"
            "Install with: pip install sochdb\n"
            "Or build from source: maturin develop --release"
        )


# =============================================================================
# High-Level API (Task 4: Split Bulk API Modes)
# =============================================================================

def build_index_from_numpy(
    embeddings,
    *,
    m: int | None = None,
    ef_construction: int | None = None,
    metric: str = "cosine",
    ids=None,
) -> "HnswIndex":
    """
    Build an HNSW index from NumPy embeddings (in-process, zero-copy).
    
    This is the fast path - vectors are passed directly to Rust without
    disk I/O or subprocess overhead.
    
    Args:
        embeddings:      2D float32 array of shape (N, D).
        m:               HNSW max connections per node. Defaults to
                         ``recommended_hnsw_params(D)['m']`` (dimension-aware).
        ef_construction: Construction search depth. Defaults to
                         ``recommended_hnsw_params(D)['ef_construction']``.
        metric:          Distance metric ("cosine", "euclidean", "dot").
        ids:             Optional 1D uint64 array of vector IDs.
    
    Returns:
        HnswIndex with inserted vectors.
    
    Performance:
        ~15,000 vec/s for 768D vectors (10x faster than subprocess).
    
    Example:
        >>> import numpy as np
        >>> from sochdb import build_index_from_numpy
        >>> 
        >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
        >>> index = build_index_from_numpy(embeddings)  # auto-tunes M for 768D
        >>> index.save("my_index.hnsw")
    """
    _check_native()
    dim = embeddings.shape[1] if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1 else None
    if m is None or ef_construction is None:
        params = recommended_hnsw_params(dim) if dim else {"m": 16, "ef_construction": 200}
        m = m if m is not None else params["m"]
        ef_construction = ef_construction if ef_construction is not None else params["ef_construction"]
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
    Uses the sochdb-bulk CLI with memory-mapped I/O.
    
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
        This function requires the sochdb-bulk binary. For most use cases,
        prefer build_index_from_numpy() which is faster.
    """
    # Import the subprocess-based implementation
    from sochdb._bulk import bulk_build_from_file
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
# Multi-Shard Index — 100M to 1B vector scale
# =============================================================================

class MultiShardHnswIndex:
    """
    Horizontally sharded HNSW index for 100M–1B vector scale.

    Distributes vectors across N independent HnswIndex shards by ID hash.
    Search scatters to all shards in parallel threads, then merges top-k.

    Scale estimates (768D, M=32, ef_search=640):
        8 shards  ×  50M vecs/shard  =  400M total  (~192 GB RAM)
        16 shards × 125M vecs/shard  =   2B total  (needs mmap per shard)

    For cross-machine distribution: run one shard per sochdb-server gRPC pod
    and replace the local scatter with parallel gRPC calls — the merge is
    identical.

    Args:
        dimension:       Vector dimensionality.
        n_shards:        Number of shards (default 8; use 16+ for 1B scale).
        m:               HNSW M parameter (auto-tuned if None).
        ef_construction: Build-time search depth (auto-tuned if None).
        ef_search:       Search-time depth (auto-tuned from target_recall).
        metric:          "cosine", "euclidean", or "dot".
        target_recall:   Used to auto-tune ef_search if ef_search is None.

    Example::

        >>> import numpy as np
        >>> from sochdb import MultiShardHnswIndex

        >>> index = MultiShardHnswIndex(dimension=768, n_shards=8)

        >>> # Insert 800K vectors (100K per shard)
        >>> vecs = np.random.randn(800_000, 768).astype(np.float32)
        >>> ids  = np.arange(800_000, dtype=np.uint64)
        >>> index.insert_batch_with_ids(ids, vecs)

        >>> # Search — scatters to 8 shards, merges top-10
        >>> q = np.random.randn(768).astype(np.float32)
        >>> result_ids, distances = index.search(q, k=10)

        >>> # Persist each shard to disk
        >>> index.save("/data/my_index")   # creates /data/my_index_shard_{0..7}.hnsw

        >>> # Reload
        >>> index2 = MultiShardHnswIndex.load("/data/my_index", n_shards=8, dimension=768)
    """

    def __init__(
        self,
        dimension: int,
        n_shards: int = 8,
        m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
        metric: str = "cosine",
        target_recall: float = 0.95,
    ):
        import threading
        _check_native()
        params = recommended_hnsw_params(dimension, target_recall=target_recall)
        self.dimension = dimension
        self.n_shards = n_shards
        self.metric = metric
        self._ef_search = ef_search if ef_search is not None else params["ef_search"]
        _m = m if m is not None else params["m"]
        _efc = ef_construction if ef_construction is not None else params["ef_construction"]
        self.shards: list[HnswIndex] = [
            HnswIndex(dimension=dimension, m=_m, ef_construction=_efc, metric=metric)
            for _ in range(n_shards)
        ]
        self._lock = threading.Lock()
        self._shard_locks = [threading.Lock() for _ in range(n_shards)]
        self._shard_counts = [0] * n_shards
        self._total_inserted = 0

    def insert_batch_with_ids(self, ids, vectors) -> int:
        """
        Insert vectors into shards by ``id % n_shards`` routing.

        Inserts within a shard are batched for efficiency — vectors routed
        to the same shard are collected and inserted together.
        """
        import numpy as np
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if ids.dtype != np.uint64:
            ids = ids.astype(np.uint64)

        # Bucket by shard
        shard_ids: list[list] = [[] for _ in range(self.n_shards)]
        shard_vecs: list[list] = [[] for _ in range(self.n_shards)]
        for i in range(len(ids)):
            s = int(ids[i]) % self.n_shards
            shard_ids[s].append(ids[i])
            shard_vecs[s].append(vectors[i])

        total = 0
        for s in range(self.n_shards):
            if not shard_ids[s]:
                continue
            s_ids = np.array(shard_ids[s], dtype=np.uint64)
            s_vecs = np.array(shard_vecs[s], dtype=np.float32)
            with self._shard_locks[s]:
                self.shards[s].insert_batch_with_ids(s_ids, s_vecs)
            cnt = len(s_ids)
            total += cnt
            with self._lock:
                self._shard_counts[s] += cnt

        with self._lock:
            self._total_inserted += total
        return total

    def search(self, query, k: int = 10, ef_search: int | None = None) -> tuple:
        """
        Scatter-gather search across all shards.

        Each shard returns up to k results; the global top-k are selected
        by distance (ascending — works for both cosine 1-sim and L2).
        Shards are queried in parallel via a thread pool.
        """
        import numpy as np
        import threading

        ef = ef_search if ef_search is not None else self._ef_search
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        all_ids: list = []
        all_dists: list = []
        results_lock = threading.Lock()
        errors: list = []

        def _search_shard(shard: "HnswIndex"):
            try:
                r_ids, r_dists = shard.search(query, k=k, ef_search=ef)
                with results_lock:
                    all_ids.extend(r_ids.tolist() if hasattr(r_ids, 'tolist') else list(r_ids))
                    all_dists.extend(r_dists.tolist() if hasattr(r_dists, 'tolist') else list(r_dists))
            except Exception as e:
                with results_lock:
                    errors.append(e)

        threads = [threading.Thread(target=_search_shard, args=(s,)) for s in self.shards]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if not all_ids:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.float32)

        # Sort by distance ascending (lower = better for any metric in HNSW)
        pairs = sorted(zip(all_dists, all_ids))[:k]
        dists, ids = zip(*pairs)
        return np.array(ids, dtype=np.uint64), np.array(dists, dtype=np.float32)

    @property
    def total_vectors(self) -> int:
        """Total vectors inserted across all shards."""
        return self._total_inserted

    def shard_sizes(self) -> list[int]:
        """Number of vectors per shard (for load-balance inspection)."""
        return list(self._shard_counts)

    def save(self, prefix: str) -> list[str]:
        """
        Save all shards to ``{prefix}_shard_{i}.hnsw``.

        Returns list of saved file paths.
        """
        paths = []
        for i, shard in enumerate(self.shards):
            path = f"{prefix}_shard_{i}.hnsw"
            shard.save(path)
            paths.append(path)
        return paths

    @classmethod
    def load(
        cls,
        prefix: str,
        n_shards: int,
        dimension: int,
        *,
        ef_search: int | None = None,
        metric: str = "cosine",
        target_recall: float = 0.95,
    ) -> "MultiShardHnswIndex":
        """
        Reload a previously saved multi-shard index.

        Args:
            prefix:    Same prefix used in ``save()``.
            n_shards:  Must match the value used when building.
            dimension: Vector dimension.
        """
        _check_native()
        obj = cls.__new__(cls)
        import threading
        obj.dimension = dimension
        obj.n_shards = n_shards
        obj.metric = metric
        params = recommended_hnsw_params(dimension, target_recall=target_recall)
        obj._ef_search = ef_search if ef_search is not None else params["ef_search"]
        obj._lock = threading.Lock()
        obj._shard_locks = [threading.Lock() for _ in range(n_shards)]
        obj.shards = []
        for i in range(n_shards):
            path = f"{prefix}_shard_{i}.hnsw"
            obj.shards.append(HnswIndex.load(path))
        # Recover counts from loaded shards via __len__
        obj._shard_counts = [len(s) for s in obj.shards]
        obj._total_inserted = sum(obj._shard_counts)
        return obj


# =============================================================================
# Parameter Tuning Helpers
# =============================================================================

def recommended_hnsw_params(
    dimension: int,
    n_vectors: int | None = None,
    target_recall: float = 0.95,
) -> dict:
    """Return recommended HNSW build and search parameters for a given dimension.

    Empirically derived from profiling 768D synthetic + real embedding datasets:

    - Dimensions ≤ 128  : M=16  is sufficient (distances separate well).
    - Dimensions 129-512: M=24  balances recall and build cost.
    - Dimensions 513+   : M=32  required to reach >0.95 recall; M=48 for 0.99+.

    ef_search is scaled by target_recall:
    - recall ≥ 0.99 : ef = 40 × M
    - recall ≥ 0.95 : ef = 20 × M
    - recall ≥ 0.90 : ef = 10 × M
    - recall ≥ 0.85 : ef =  6 × M

    Args:
        dimension:     Vector dimension.
        n_vectors:     Approximate dataset size (unused, reserved for future).
        target_recall: Desired recall@10 (0.0–1.0), default 0.95.

    Returns:
        Dict with keys: m, ef_construction, ef_search, note.

    Example::

        >>> params = recommended_hnsw_params(768, target_recall=0.95)
        >>> index = HnswIndex(dimension=768, **{k: v for k, v in params.items()
        ...                                     if k in ('m', 'ef_construction')})
        >>> ids, dists = index.search(query, k=10, ef_search=params['ef_search'])
    """
    if dimension <= 128:
        m = 16
    elif dimension <= 512:
        m = 24
    else:
        m = 32  # 768D+ — 0.982 recall @ ef=3000 vs 0.864 with M=16

    ef_construction = max(200, m * 8)

    if target_recall >= 0.99:
        ef_search = m * 40
    elif target_recall >= 0.95:
        ef_search = m * 20
    elif target_recall >= 0.90:
        ef_search = m * 10
    else:
        ef_search = m * 6

    note = (
        f"dim={dimension}: M={m}, efc={ef_construction}, ef_search={ef_search} "
        f"targets recall≥{target_recall:.0%} on real embedding data. "
        f"Synthetic uniform data needs 2-3× higher ef_search."
    )

    return {
        "m": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "note": note,
    }


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
    class Database:
        """SochDB Database (stub - native extension not loaded)."""

        def __init__(self, *args, **kwargs):
            _check_native()

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

    class Transaction:
        """SochDB Transaction (stub - native extension not loaded)."""

        def __init__(self, *args, **kwargs):
            _check_native()
