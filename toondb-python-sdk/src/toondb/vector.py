#!/usr/bin/env python3
"""
ToonDB Vector Index (HNSW)

Python bindings for ToonDB's high-performance HNSW vector search.
This is 15x faster than ChromaDB for vector search.
"""

import os
import ctypes
import warnings
from typing import List, Tuple, Optional
import numpy as np


# =============================================================================
# TASK 5: SAFE-MODE HYGIENE (Python Side)
# =============================================================================

class PerformanceWarning(UserWarning):
    """Warning for performance-degrading conditions."""
    pass


_SAFE_MODE_WARNED = False


def _check_safe_mode() -> bool:
    """Check if safe mode is enabled and emit warning."""
    global _SAFE_MODE_WARNED
    
    if os.environ.get("TOONDB_BATCH_SAFE_MODE") in ("1", "true", "True"):
        if not _SAFE_MODE_WARNED:
            warnings.warn(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  WARNING: TOONDB_BATCH_SAFE_MODE is enabled.                 ║\n"
                "║  Batch inserts will be 10-100× SLOWER.                       ║\n"
                "║  Unset this environment variable for production use.         ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n",
                PerformanceWarning,
                stacklevel=3
            )
            _SAFE_MODE_WARNED = True
        return True
    return False


def _get_platform_dir() -> str:
    """Get the platform directory name for the current system."""
    import platform as plat
    system = plat.system().lower()
    machine = plat.machine().lower()
    
    # Normalize machine names
    if machine in ("x86_64", "amd64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "aarch64"
    
    return f"{system}-{machine}"


def _find_library():
    """Find the ToonDB index library.
    
    Search order:
    1. TOONDB_LIB_PATH environment variable
    2. Bundled library in wheel (lib/{platform}/)
    3. Package directory
    4. Development build (target/release)
    5. System paths
    """
    # Platform-specific library name
    if os.uname().sysname == "Darwin":
        lib_name = "libtoondb_index.dylib"
    elif os.name == "nt":
        lib_name = "toondb_index.dll"
    else:
        lib_name = "libtoondb_index.so"
    
    pkg_dir = os.path.dirname(__file__)
    platform_dir = _get_platform_dir()
    
    # 1. Environment variable override
    env_path = os.environ.get("TOONDB_LIB_PATH")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        # Maybe it's a directory
        full_path = os.path.join(env_path, lib_name)
        if os.path.exists(full_path):
            return full_path
    
    # Search paths in priority order
    search_paths = [
        # 2. Bundled library in wheel (platform-specific)
        os.path.join(pkg_dir, "lib", platform_dir),
        # 3. Bundled library in wheel (generic)
        os.path.join(pkg_dir, "lib"),
        # 4. Package directory
        pkg_dir,
        # 5. Development builds
        os.path.join(pkg_dir, "..", "..", "..", "target", "release"),
        os.path.join(pkg_dir, "..", "..", "..", "target", "debug"),
        # 6. System paths
        "/usr/local/lib",
        "/usr/lib",
    ]
    
    for path in search_paths:
        full_path = os.path.join(path, lib_name)
        if os.path.exists(full_path):
            return full_path
    
    return None


# Search result structure with FFI-safe ID representation
class CSearchResult(ctypes.Structure):
    _fields_ = [
        ("id_lo", ctypes.c_uint64),  # Lower 64 bits of ID
        ("id_hi", ctypes.c_uint64),  # Upper 64 bits of ID
        ("distance", ctypes.c_float),
    ]


class _FFI:
    """FFI bindings to the vector index library."""
    _lib = None
    
    @classmethod
    def get_lib(cls):
        if cls._lib is None:
            path = _find_library()
            if path is None:
                raise ImportError(
                    "Could not find libtoondb_index. "
                    "Set TOONDB_LIB_PATH environment variable."
                )
            cls._lib = ctypes.CDLL(path)
            cls._setup_bindings()
        return cls._lib
    
    @classmethod
    def _setup_bindings(cls):
        lib = cls._lib
        
        # hnsw_new
        lib.hnsw_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.hnsw_new.restype = ctypes.c_void_p
        
        # hnsw_free
        lib.hnsw_free.argtypes = [ctypes.c_void_p]
        lib.hnsw_free.restype = None
        
        # hnsw_insert
        lib.hnsw_insert.argtypes = [
            ctypes.c_void_p,  # ptr
            ctypes.c_uint64,  # id_lo (lower 64 bits)
            ctypes.c_uint64,  # id_hi (upper 64 bits)
            ctypes.POINTER(ctypes.c_float),  # vector
            ctypes.c_size_t,  # vector_len
        ]
        lib.hnsw_insert.restype = ctypes.c_int
        
        # hnsw_insert_batch (parallel, high-performance)
        lib.hnsw_insert_batch.argtypes = [
            ctypes.c_void_p,  # ptr
            ctypes.POINTER(ctypes.c_uint64),  # ids (N u64 values)
            ctypes.POINTER(ctypes.c_float),   # vectors (N×D f32 values)
            ctypes.c_size_t,  # num_vectors
            ctypes.c_size_t,  # dimension
        ]
        lib.hnsw_insert_batch.restype = ctypes.c_int
        
        # hnsw_insert_batch_flat (zero-allocation, Task 2)
        lib.hnsw_insert_batch_flat.argtypes = [
            ctypes.c_void_p,  # ptr
            ctypes.POINTER(ctypes.c_uint64),  # ids (N u64 values)
            ctypes.POINTER(ctypes.c_float),   # vectors (N×D f32 values)
            ctypes.c_size_t,  # num_vectors
            ctypes.c_size_t,  # dimension
        ]
        lib.hnsw_insert_batch_flat.restype = ctypes.c_int
        
        # hnsw_insert_flat (single-vector, zero-allocation, Task 2)
        lib.hnsw_insert_flat.argtypes = [
            ctypes.c_void_p,  # ptr
            ctypes.c_uint64,  # id_lo
            ctypes.c_uint64,  # id_hi
            ctypes.POINTER(ctypes.c_float),  # vector
            ctypes.c_size_t,  # vector_len
        ]
        lib.hnsw_insert_flat.restype = ctypes.c_int
        
        # hnsw_search
        lib.hnsw_search.argtypes = [
            ctypes.c_void_p,  # ptr
            ctypes.POINTER(ctypes.c_float),  # query
            ctypes.c_size_t,  # query_len
            ctypes.c_size_t,  # k
            ctypes.POINTER(CSearchResult),  # results_out
            ctypes.POINTER(ctypes.c_size_t),  # num_results_out
        ]
        lib.hnsw_search.restype = ctypes.c_int
        
        # hnsw_len
        lib.hnsw_len.argtypes = [ctypes.c_void_p]
        lib.hnsw_len.restype = ctypes.c_size_t
        
        # hnsw_dimension
        lib.hnsw_dimension.argtypes = [ctypes.c_void_p]
        lib.hnsw_dimension.restype = ctypes.c_size_t
        
        # Profiling functions
        lib.toondb_profiling_enable.argtypes = []
        lib.toondb_profiling_enable.restype = None
        
        lib.toondb_profiling_disable.argtypes = []
        lib.toondb_profiling_disable.restype = None
        
        lib.toondb_profiling_dump.argtypes = []
        lib.toondb_profiling_dump.restype = None


def enable_profiling():
    """Enable Rust-side profiling."""
    lib = _FFI.get_lib()
    lib.toondb_profiling_enable()


def disable_profiling():
    """Disable Rust-side profiling."""
    lib = _FFI.get_lib()
    lib.toondb_profiling_disable()


def dump_profiling():
    """Dump Rust-side profiling to file and print summary."""
    lib = _FFI.get_lib()
    lib.toondb_profiling_dump()


class VectorIndex:
    """
    ToonDB HNSW Vector Index.
    
    High-performance approximate nearest neighbor search using HNSW algorithm.
    15x faster than ChromaDB with ~47µs search latency.
    
    Example:
        >>> index = VectorIndex(dimension=128)
        >>> index.insert(0, np.random.randn(128).astype(np.float32))
        >>> results = index.search(query_vector, k=10)
        >>> for id, distance in results:
        ...     print(f"ID: {id}, Distance: {distance}")
    """
    
    def __init__(
        self,
        dimension: int,
        max_connections: int = 16,
        ef_construction: int = 100,  # Reduced from 200 for better performance
    ):
        """
        Create a new vector index.
        
        Args:
            dimension: Vector dimension (e.g., 128, 768, 1536)
            max_connections: Max neighbors per node (default: 16)
            ef_construction: Construction-time ef (default: 200)
        """
        lib = _FFI.get_lib()
        self._ptr = lib.hnsw_new(dimension, max_connections, ef_construction)
        if self._ptr is None:
            raise RuntimeError("Failed to create HNSW index")
        self._dimension = dimension
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr is not None:
            lib = _FFI.get_lib()
            lib.hnsw_free(self._ptr)
            self._ptr = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ptr is not None:
            lib = _FFI.get_lib()
            lib.hnsw_free(self._ptr)
            self._ptr = None
    
    def insert(self, id: int, vector: np.ndarray) -> None:
        """
        Insert a vector into the index.
        
        Args:
            id: Unique vector ID (0 to 2^64-1)
            vector: Float32 numpy array of length `dimension`
        """
        if len(vector) != self._dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}")
        
        lib = _FFI.get_lib()
        
        # Convert vector to contiguous float32
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        vec_ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Split ID into low and high u64 parts
        id_lo = id & 0xFFFFFFFFFFFFFFFF
        id_hi = (id >> 64) & 0xFFFFFFFFFFFFFFFF
        
        result = lib.hnsw_insert(self._ptr, id_lo, id_hi, vec_ptr, len(vec))
        if result != 0:
            raise RuntimeError("Failed to insert vector")
    
    def insert_batch(self, ids: np.ndarray, vectors: np.ndarray) -> int:
        """
        Insert multiple vectors in a single FFI call with parallel processing.
        
        This is the high-performance path - 100x faster than individual inserts.
        Uses zero-copy numpy array passing and parallel HNSW construction.
        
        Args:
            ids: 1D array of uint64 IDs, shape (N,)
            vectors: 2D array of float32 vectors, shape (N, dimension)
        
        Returns:
            Number of successfully inserted vectors
        
        Performance:
            - Individual insert: ~500 vec/sec
            - Batch insert: ~50,000 vec/sec (100x faster)
        
        Example:
            >>> ids = np.arange(10000, dtype=np.uint64)
            >>> vectors = np.random.randn(10000, 128).astype(np.float32)
            >>> inserted = index.insert_batch(ids, vectors)
        """
        if len(vectors.shape) != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        
        num_vectors, dim = vectors.shape
        if dim != self._dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self._dimension}, got {dim}")
        
        if len(ids) != num_vectors:
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({num_vectors})")
        
        lib = _FFI.get_lib()
        
        # Ensure contiguous memory layout for zero-copy FFI
        ids_arr = np.ascontiguousarray(ids, dtype=np.uint64)
        vectors_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        
        # Get raw pointers to numpy data
        ids_ptr = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        vectors_ptr = vectors_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Single FFI call with parallel processing on Rust side
        result = lib.hnsw_insert_batch(
            self._ptr,
            ids_ptr,
            vectors_ptr,
            num_vectors,
            self._dimension,
        )
        
        if result < 0:
            raise RuntimeError("Batch insert failed")
        
        return result
    
    # =========================================================================
    # TASK 3: STRICT LAYOUT ENFORCEMENT (High-Performance Path)
    # =========================================================================
    
    def insert_batch_fast(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        *,
        strict: bool = True
    ) -> int:
        """
        High-performance batch insert with layout enforcement.
        
        This is the **fastest FFI path** for production use. Unlike `insert_batch()`,
        this method:
        1. Validates array layouts upfront (no hidden copies)
        2. Uses the zero-allocation FFI binding
        3. Fails fast on layout violations instead of silently copying
        
        Args:
            ids: 1D uint64 array, must be C-contiguous
            vectors: 2D float32 array, shape (N, D), must be C-contiguous
            strict: If True (default), raise on layout violations instead of copying
        
        Returns:
            Number of successfully inserted vectors
        
        Raises:
            ValueError: If strict=True and arrays violate layout requirements
        
        Performance:
            With proper layout: ~1,500 vec/s @ 768D (near Rust speed)
            With layout violation + strict=False: ~150 vec/s (10x slower copy)
        
        Example:
            >>> # Correct way - preallocate with correct dtype
            >>> ids = np.arange(10000, dtype=np.uint64)
            >>> vectors = np.random.randn(10000, 768).astype(np.float32)
            >>> inserted = index.insert_batch_fast(ids, vectors)
            
            >>> # Wrong way - will raise ValueError with strict=True
            >>> vectors_f64 = np.random.randn(10000, 768)  # float64!
            >>> index.insert_batch_fast(ids, vectors_f64)  # Raises!
        """
        # Check safe mode first
        if _check_safe_mode():
            warnings.warn(
                "insert_batch_fast() called with SAFE_MODE enabled. "
                "Performance will be severely degraded (~100x slower).",
                PerformanceWarning,
                stacklevel=2
            )
        
        # Validate shape
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got {vectors.ndim}D")
        
        n_vectors, dim = vectors.shape
        if dim != self._dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self._dimension}, got {dim}"
            )
        
        if len(ids) != n_vectors:
            raise ValueError(
                f"Number of IDs ({len(ids)}) must match number of vectors ({n_vectors})"
            )
        
        # Strict layout checks
        if strict:
            if vectors.dtype != np.float32:
                raise ValueError(
                    f"vectors.dtype must be float32, got {vectors.dtype}. "
                    f"Use vectors.astype(np.float32) explicitly."
                )
            if not vectors.flags['C_CONTIGUOUS']:
                raise ValueError(
                    "vectors must be C-contiguous (row-major). "
                    "Use np.ascontiguousarray(vectors) explicitly, or check "
                    "if your array is transposed/sliced."
                )
            if ids.dtype != np.uint64:
                raise ValueError(
                    f"ids.dtype must be uint64, got {ids.dtype}. "
                    f"Use ids.astype(np.uint64) explicitly."
                )
            if not ids.flags['C_CONTIGUOUS']:
                raise ValueError(
                    "ids must be C-contiguous. "
                    "Use np.ascontiguousarray(ids) explicitly."
                )
        else:
            # Fallback: silent conversion (existing behavior)
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            ids = np.ascontiguousarray(ids, dtype=np.uint64)
        
        lib = _FFI.get_lib()
        
        # Get raw pointers (no copy needed - layout is validated)
        ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        vectors_ptr = vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Use the zero-allocation FFI binding
        result = lib.hnsw_insert_batch_flat(
            self._ptr,
            ids_ptr,
            vectors_ptr,
            n_vectors,
            self._dimension,
        )
        
        if result < 0:
            raise RuntimeError("Batch insert failed")
        
        return result
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector (float32 numpy array)
            k: Number of neighbors to return
        
        Returns:
            List of (id, distance) tuples, sorted by distance
        """
        if len(query) != self._dimension:
            raise ValueError(f"Query dimension mismatch: expected {self._dimension}, got {len(query)}")
        
        lib = _FFI.get_lib()
        
        # Convert query to contiguous float32
        q = np.ascontiguousarray(query, dtype=np.float32)
        q_ptr = q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Allocate result array
        results = (CSearchResult * k)()
        num_results = ctypes.c_size_t()
        
        result = lib.hnsw_search(
            self._ptr,
            q_ptr,
            len(q),
            k,
            results,
            ctypes.byref(num_results),
        )
        
        if result != 0:
            raise RuntimeError("Search failed")
        
        # Convert results
        output = []
        for i in range(num_results.value):
            r = results[i]
            id = r.id_lo | (r.id_hi << 64)
            output.append((id, r.distance))
        
        return output
    
    def __len__(self) -> int:
        """Get the number of vectors in the index."""
        lib = _FFI.get_lib()
        return lib.hnsw_len(self._ptr)
    
    @property
    def dimension(self) -> int:
        """Get the dimension of vectors in this index."""
        return self._dimension
