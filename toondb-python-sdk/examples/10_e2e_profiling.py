#!/usr/bin/env python3
"""
ToonDB End-to-End Profiling: 1K Vector Insertion into HNSW

This script provides detailed profiling of the complete data path:
  Python SDK → FFI → Rust → HNSW Index

Usage:
    # Standard profiling
    python 10_e2e_profiling.py

    # With memory profiling (requires tracemalloc)
    python 10_e2e_profiling.py --memory

    # With Rust-side tracing (requires TOONDB_PROFILING=1)
    TOONDB_PROFILING=1 python 10_e2e_profiling.py --detailed

Outputs:
    - Console summary with timing breakdown
    - JSON report: profiling_results.json
    - Flame graph data (if --flamegraph)
"""

import os
import sys
import time
import json
import argparse
import tracemalloc
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from toondb.vector import VectorIndex, _FFI, dump_profiling, enable_profiling
except ImportError as e:
    print(f"Error importing toondb: {e}")
    print("Make sure to build the Rust library first: cargo build --release -p toondb-index")
    sys.exit(1)


# =============================================================================
# PROFILING DATA STRUCTURES
# =============================================================================

@dataclass
class TimingStats:
    """Statistics for a timing measurement."""
    name: str
    count: int
    total_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    
    @classmethod
    def from_samples(cls, name: str, samples_ns: List[float]) -> 'TimingStats':
        """Create stats from nanosecond samples."""
        if not samples_ns:
            return cls(name, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        samples_ms = [s / 1_000_000 for s in samples_ns]
        sorted_samples = sorted(samples_ms)
        n = len(sorted_samples)
        
        return cls(
            name=name,
            count=n,
            total_ms=sum(samples_ms),
            mean_ms=statistics.mean(samples_ms),
            std_ms=statistics.stdev(samples_ms) if n > 1 else 0,
            min_ms=sorted_samples[0],
            max_ms=sorted_samples[-1],
            p50_ms=sorted_samples[n // 2],
            p95_ms=sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
            p99_ms=sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1],
        )


@dataclass
class MemoryStats:
    """Memory allocation statistics."""
    peak_mb: float = 0.0
    current_mb: float = 0.0
    allocations: int = 0
    vector_data_mb: float = 0.0  # Expected memory for vector data
    overhead_mb: float = 0.0     # Memory overhead (index structures, etc.)


@dataclass
class LayerProfile:
    """Profile data for a single layer."""
    name: str
    timings: Dict[str, TimingStats] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    
    def add_timing(self, name: str, samples_ns: List[float]):
        self.timings[name] = TimingStats.from_samples(name, samples_ns)
    
    def add_count(self, name: str, value: int):
        self.counts[name] = value


@dataclass
class E2EProfile:
    """Complete end-to-end profile."""
    timestamp: str
    config: Dict[str, Any]
    python_layer: LayerProfile = field(default_factory=lambda: LayerProfile("python"))
    ffi_layer: LayerProfile = field(default_factory=lambda: LayerProfile("ffi"))
    rust_layer: LayerProfile = field(default_factory=lambda: LayerProfile("rust"))
    hnsw_layer: LayerProfile = field(default_factory=lambda: LayerProfile("hnsw"))
    memory: MemoryStats = field(default_factory=MemoryStats)
    summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HIGH-RESOLUTION TIMER
# =============================================================================

class PrecisionTimer:
    """High-resolution timer using time.perf_counter_ns."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.samples: List[int] = []
        self._start: Optional[int] = None
    
    def start(self):
        self._start = time.perf_counter_ns()
    
    def stop(self) -> int:
        if self._start is None:
            raise RuntimeError("Timer not started")
        elapsed = time.perf_counter_ns() - self._start
        self.samples.append(elapsed)
        self._start = None
        return elapsed
    
    @contextmanager
    def measure(self):
        """Context manager for timing a block."""
        self.start()
        try:
            yield
        finally:
            self.stop()
    
    def total_ns(self) -> int:
        return sum(self.samples)
    
    def total_ms(self) -> float:
        return self.total_ns() / 1_000_000
    
    def stats(self) -> TimingStats:
        return TimingStats.from_samples(self.name, self.samples)


# =============================================================================
# PYTHON-SIDE PROFILER
# =============================================================================

class PythonProfiler:
    """Profiles Python SDK operations."""
    
    def __init__(self):
        self.timers: Dict[str, PrecisionTimer] = {}
        self._init_timers()
    
    def _init_timers(self):
        """Initialize all timing categories."""
        timer_names = [
            # Data preparation
            "numpy_allocation",
            "numpy_ascontiguous",
            "dtype_conversion",
            "data_validation",
            
            # FFI overhead
            "ffi_get_lib",
            "ffi_ptr_creation",
            "ffi_call_overhead",
            
            # Batch operations
            "batch_total",
            "per_vector_insert",
        ]
        for name in timer_names:
            self.timers[name] = PrecisionTimer(name)
    
    def timer(self, name: str) -> PrecisionTimer:
        if name not in self.timers:
            self.timers[name] = PrecisionTimer(name)
        return self.timers[name]
    
    def get_stats(self) -> Dict[str, TimingStats]:
        return {name: timer.stats() for name, timer in self.timers.items() if timer.samples}


# =============================================================================
# PROFILED VECTOR INDEX
# =============================================================================

class ProfiledVectorIndex:
    """
    VectorIndex wrapper with comprehensive profiling.
    
    Profiles:
    - Python-side operations (numpy, validation, FFI setup)
    - FFI boundary crossing
    - Rust-side operations (via environment variable)
    """
    
    def __init__(
        self,
        dimension: int,
        max_connections: int = 16,
        ef_construction: int = 200,
    ):
        self.profiler = PythonProfiler()
        
        # Profile index creation
        with self.profiler.timer("index_creation").measure():
            self._index = VectorIndex(
                dimension=dimension,
                max_connections=max_connections,
                ef_construction=ef_construction,
            )
        
        self._dimension = dimension
        self._insert_count = 0
    
    def insert_batch_profiled(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> Dict[str, float]:
        """
        Insert batch with detailed per-phase profiling.
        
        Returns timing breakdown in milliseconds.
        """
        timings = {}
        
        # Phase 1: Input validation
        with self.profiler.timer("data_validation").measure():
            if vectors.ndim != 2:
                raise ValueError(f"vectors must be 2D, got {vectors.ndim}D")
            n_vectors, dim = vectors.shape
            if dim != self._dimension:
                raise ValueError(f"Dimension mismatch: expected {self._dimension}, got {dim}")
            if len(ids) != n_vectors:
                raise ValueError(f"ID count mismatch: {len(ids)} vs {n_vectors}")
        
        # Phase 2: Memory layout check/conversion
        with self.profiler.timer("dtype_conversion").measure():
            needs_id_convert = ids.dtype != np.uint64
            needs_vec_convert = vectors.dtype != np.float32
        
        with self.profiler.timer("numpy_ascontiguous").measure():
            needs_id_contiguous = not ids.flags['C_CONTIGUOUS']
            needs_vec_contiguous = not vectors.flags['C_CONTIGUOUS']
            
            if needs_id_convert or needs_id_contiguous:
                ids = np.ascontiguousarray(ids, dtype=np.uint64)
            if needs_vec_convert or needs_vec_contiguous:
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        
        # Phase 3: FFI pointer creation
        import ctypes
        with self.profiler.timer("ffi_ptr_creation").measure():
            ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            vectors_ptr = vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Phase 4: FFI call (includes Rust processing time)
        with self.profiler.timer("ffi_call_overhead").measure():
            lib = _FFI.get_lib()
        
        with self.profiler.timer("batch_total").measure():
            result = lib.hnsw_insert_batch(
                self._index._ptr,
                ids_ptr,
                vectors_ptr,
                n_vectors,
                self._dimension,
            )
        
        if result < 0:
            raise RuntimeError("Batch insert failed")
        
        self._insert_count += result
        
        # Collect timing breakdown
        for name, timer in self.profiler.timers.items():
            if timer.samples:
                timings[name] = timer.samples[-1] / 1_000_000  # Convert to ms
        
        return timings
    
    def insert_individual_profiled(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> List[Dict[str, float]]:
        """
        Insert vectors one-by-one with per-vector profiling.
        
        Useful for understanding per-vector overhead.
        """
        import ctypes
        
        per_vector_timings = []
        lib = _FFI.get_lib()
        
        for i in range(len(ids)):
            timing = {}
            
            # Extract single vector
            with self.profiler.timer("per_vector_extract").measure():
                vec = vectors[i]
                id_val = int(ids[i])
            
            # Ensure contiguous
            with self.profiler.timer("per_vector_contiguous").measure():
                vec = np.ascontiguousarray(vec, dtype=np.float32)
            
            # Create pointer
            with self.profiler.timer("per_vector_ptr").measure():
                vec_ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # FFI call
            with self.profiler.timer("per_vector_insert").measure():
                result = lib.hnsw_insert(
                    self._index._ptr,
                    id_val & 0xFFFFFFFFFFFFFFFF,  # id_lo
                    0,  # id_hi
                    vec_ptr,
                    self._dimension,
                )
            
            timing['extract_us'] = self.profiler.timer("per_vector_extract").samples[-1] / 1000
            timing['contiguous_us'] = self.profiler.timer("per_vector_contiguous").samples[-1] / 1000
            timing['ptr_us'] = self.profiler.timer("per_vector_ptr").samples[-1] / 1000
            timing['insert_us'] = self.profiler.timer("per_vector_insert").samples[-1] / 1000
            timing['total_us'] = sum(timing.values())
            
            per_vector_timings.append(timing)
            self._insert_count += 1
        
        return per_vector_timings
    
    def search_profiled(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> tuple:
        """Search with profiling."""
        with self.profiler.timer("search_total").measure():
            results = self._index.search(query, k)
        return results
    
    def get_profile(self) -> LayerProfile:
        """Get collected profile data."""
        profile = LayerProfile("python")
        for name, timer in self.profiler.timers.items():
            if timer.samples:
                profile.add_timing(name, timer.samples)
        profile.add_count("vectors_inserted", self._insert_count)
        return profile
    
    def __len__(self):
        return len(self._index)


# =============================================================================
# MAIN PROFILING SCRIPT
# =============================================================================

def generate_test_data(
    num_vectors: int,
    dimension: int,
    seed: int = 42,
) -> tuple:
    """Generate test vectors with profiling."""
    np.random.seed(seed)
    
    t0 = time.perf_counter_ns()
    
    # Generate IDs
    ids = np.arange(num_vectors, dtype=np.uint64)
    
    # Generate random vectors (normalized for cosine similarity)
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    t1 = time.perf_counter_ns()
    
    generation_ms = (t1 - t0) / 1_000_000
    memory_mb = (ids.nbytes + vectors.nbytes) / (1024 * 1024)
    
    return ids, vectors, {
        "generation_ms": generation_ms,
        "memory_mb": memory_mb,
        "num_vectors": num_vectors,
        "dimension": dimension,
    }


def run_batch_profiling(
    num_vectors: int = 1000,
    dimension: int = 768,
    batch_size: Optional[int] = None,
    ef_construction: int = 200,
    max_connections: int = 16,
) -> E2EProfile:
    """
    Run end-to-end profiling of batch insertion.
    
    Args:
        num_vectors: Total vectors to insert
        dimension: Vector dimension
        batch_size: If set, insert in batches of this size
        ef_construction: HNSW construction parameter
        max_connections: HNSW max connections per node
    
    Returns:
        Complete E2E profile
    """
    from datetime import datetime
    
    profile = E2EProfile(
        timestamp=datetime.now().isoformat(),
        config={
            "num_vectors": num_vectors,
            "dimension": dimension,
            "batch_size": batch_size or num_vectors,
            "ef_construction": ef_construction,
            "max_connections": max_connections,
            "safe_mode": os.environ.get("TOONDB_BATCH_SAFE_MODE", "0"),
            "profiling_enabled": os.environ.get("TOONDB_PROFILING", "0"),
        }
    )
    
    # Start memory tracking
    tracemalloc.start()
    
    print(f"\n{'='*70}")
    print(f"ToonDB HNSW End-to-End Profiling")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Vectors:        {num_vectors:,}")
    print(f"  Dimension:      {dimension}")
    print(f"  Batch Size:     {batch_size or 'all at once'}")
    print(f"  EF Construction: {ef_construction}")
    print(f"  Max Connections: {max_connections}")
    print(f"{'='*70}\n")
    
    # Phase 1: Data generation
    print("Phase 1: Generating test data...")
    ids, vectors, gen_stats = generate_test_data(num_vectors, dimension)
    print(f"  Generated {num_vectors:,} vectors in {gen_stats['generation_ms']:.2f} ms")
    print(f"  Data size: {gen_stats['memory_mb']:.2f} MB")
    
    profile.python_layer.timings["data_generation"] = TimingStats(
        name="data_generation",
        count=1,
        total_ms=gen_stats['generation_ms'],
        mean_ms=gen_stats['generation_ms'],
        std_ms=0,
        min_ms=gen_stats['generation_ms'],
        max_ms=gen_stats['generation_ms'],
        p50_ms=gen_stats['generation_ms'],
        p95_ms=gen_stats['generation_ms'],
        p99_ms=gen_stats['generation_ms'],
    )
    
    # Phase 2: Index creation
    print("\nPhase 2: Creating HNSW index...")
    index = ProfiledVectorIndex(
        dimension=dimension,
        max_connections=max_connections,
        ef_construction=ef_construction,
    )
    creation_time = index.profiler.timer("index_creation").samples[0] / 1_000_000
    print(f"  Index created in {creation_time:.2f} ms")
    
    # Phase 3: Batch insertion
    print("\nPhase 3: Inserting vectors...")
    
    batch_size = batch_size or num_vectors
    num_batches = (num_vectors + batch_size - 1) // batch_size
    
    batch_timings = []
    total_insert_start = time.perf_counter_ns()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_vectors)
        
        batch_ids = ids[start_idx:end_idx]
        batch_vectors = vectors[start_idx:end_idx]
        
        timing = index.insert_batch_profiled(batch_ids, batch_vectors)
        batch_timings.append(timing)
        
        if num_batches > 1:
            progress = (batch_idx + 1) / num_batches * 100
            print(f"  Batch {batch_idx + 1}/{num_batches} ({end_idx - start_idx} vectors) - {timing['batch_total']:.2f} ms")
    
    total_insert_time_ns = time.perf_counter_ns() - total_insert_start
    total_insert_time_ms = total_insert_time_ns / 1_000_000
    
    # Aggregate batch timings
    for key in batch_timings[0].keys():
        values_ns = [t[key] * 1_000_000 for t in batch_timings]  # Convert back to ns
        profile.python_layer.add_timing(key, values_ns)
    
    # Calculate throughput
    vectors_per_sec = num_vectors / (total_insert_time_ms / 1000)
    us_per_vector = total_insert_time_ms * 1000 / num_vectors
    
    print(f"\n  Total insert time: {total_insert_time_ms:.2f} ms")
    print(f"  Throughput: {vectors_per_sec:,.0f} vectors/sec")
    print(f"  Latency: {us_per_vector:.2f} µs/vector")
    
    # Phase 4: Verification search
    print("\nPhase 4: Verification search...")
    query = vectors[0]  # Use first vector as query
    results = index.search_profiled(query, k=10)
    search_time = index.profiler.timer("search_total").samples[-1] / 1_000_000
    print(f"  Search completed in {search_time:.2f} ms")
    print(f"  Found {len(results)} results")
    print(f"  Top result ID: {results[0][0]}, distance: {results[0][1]:.4f}")
    
    # Memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    profile.memory = MemoryStats(
        peak_mb=peak / (1024 * 1024),
        current_mb=current / (1024 * 1024),
        vector_data_mb=gen_stats['memory_mb'],
        overhead_mb=(peak / (1024 * 1024)) - gen_stats['memory_mb'],
    )
    
    print(f"\n  Memory - Peak: {profile.memory.peak_mb:.2f} MB, Current: {profile.memory.current_mb:.2f} MB")
    print(f"  Vector data: {profile.memory.vector_data_mb:.2f} MB, Overhead: {profile.memory.overhead_mb:.2f} MB")
    
    # Summary
    profile.summary = {
        "total_vectors": num_vectors,
        "total_insert_time_ms": total_insert_time_ms,
        "vectors_per_second": vectors_per_sec,
        "us_per_vector": us_per_vector,
        "search_time_ms": search_time,
        "peak_memory_mb": profile.memory.peak_mb,
        "index_size": len(index),
    }
    
    # Get layer profile
    profile.python_layer = index.get_profile()
    
    return profile


def run_individual_profiling(
    num_vectors: int = 100,
    dimension: int = 768,
) -> Dict[str, Any]:
    """
    Profile individual vector insertions for latency analysis.
    
    Uses fewer vectors since individual inserts are slow.
    """
    print(f"\n{'='*70}")
    print(f"Individual Insertion Profiling ({num_vectors} vectors)")
    print(f"{'='*70}\n")
    
    ids, vectors, gen_stats = generate_test_data(num_vectors, dimension)
    
    index = ProfiledVectorIndex(dimension=dimension)
    
    per_vector_timings = index.insert_individual_profiled(ids, vectors)
    
    # Aggregate statistics
    insert_times = [t['insert_us'] for t in per_vector_timings]
    total_times = [t['total_us'] for t in per_vector_timings]
    
    stats = {
        "insert_mean_us": statistics.mean(insert_times),
        "insert_std_us": statistics.stdev(insert_times) if len(insert_times) > 1 else 0,
        "insert_p50_us": sorted(insert_times)[len(insert_times) // 2],
        "insert_p99_us": sorted(insert_times)[int(len(insert_times) * 0.99)],
        "total_mean_us": statistics.mean(total_times),
        "overhead_mean_us": statistics.mean([t['extract_us'] + t['contiguous_us'] + t['ptr_us'] 
                                             for t in per_vector_timings]),
    }
    
    print(f"Per-Vector Statistics:")
    print(f"  Insert (Rust):  mean={stats['insert_mean_us']:.1f}µs, p50={stats['insert_p50_us']:.1f}µs, p99={stats['insert_p99_us']:.1f}µs")
    print(f"  Total:          mean={stats['total_mean_us']:.1f}µs")
    print(f"  Python overhead: {stats['overhead_mean_us']:.1f}µs ({stats['overhead_mean_us']/stats['total_mean_us']*100:.1f}%)")
    
    return {
        "per_vector": per_vector_timings,
        "stats": stats,
    }


def print_detailed_breakdown(profile: E2EProfile):
    """Print detailed timing breakdown."""
    print(f"\n{'='*70}")
    print("DETAILED TIMING BREAKDOWN")
    print(f"{'='*70}\n")
    
    print("Python Layer Timings:")
    for name, stats in profile.python_layer.timings.items():
        print(f"  {name:30s}: {stats.mean_ms:8.2f} ms (total: {stats.total_ms:8.2f} ms, n={stats.count})")
    
    print(f"\nThroughput Analysis:")
    summary = profile.summary
    print(f"  Overall: {summary['vectors_per_second']:,.0f} vectors/second")
    print(f"  Latency: {summary['us_per_vector']:.2f} µs/vector")
    
    # Calculate time breakdown
    total_ms = summary['total_insert_time_ms']
    data_validation = profile.python_layer.timings.get('data_validation', TimingStats("", 0, 0, 0, 0, 0, 0, 0, 0, 0)).total_ms
    contiguous = profile.python_layer.timings.get('numpy_ascontiguous', TimingStats("", 0, 0, 0, 0, 0, 0, 0, 0, 0)).total_ms
    ptr_creation = profile.python_layer.timings.get('ffi_ptr_creation', TimingStats("", 0, 0, 0, 0, 0, 0, 0, 0, 0)).total_ms
    ffi_overhead = profile.python_layer.timings.get('ffi_call_overhead', TimingStats("", 0, 0, 0, 0, 0, 0, 0, 0, 0)).total_ms
    batch_total = profile.python_layer.timings.get('batch_total', TimingStats("", 0, 0, 0, 0, 0, 0, 0, 0, 0)).total_ms
    
    python_overhead = data_validation + contiguous + ptr_creation + ffi_overhead
    rust_time = batch_total
    
    print(f"\nTime Breakdown:")
    print(f"  Python overhead: {python_overhead:.2f} ms ({python_overhead/total_ms*100:.1f}%)")
    print(f"    - Validation:  {data_validation:.2f} ms")
    print(f"    - Contiguous:  {contiguous:.2f} ms")
    print(f"    - Ptr creation: {ptr_creation:.2f} ms")
    print(f"    - FFI setup:   {ffi_overhead:.2f} ms")
    print(f"  Rust (FFI call): {rust_time:.2f} ms ({rust_time/total_ms*100:.1f}%)")
    
    print(f"\nMemory Analysis:")
    print(f"  Peak memory:     {profile.memory.peak_mb:.2f} MB")
    print(f"  Vector data:     {profile.memory.vector_data_mb:.2f} MB")
    print(f"  Index overhead:  {profile.memory.overhead_mb:.2f} MB ({profile.memory.overhead_mb/profile.memory.vector_data_mb*100:.0f}% of vector data)")


def save_profile(profile: E2EProfile, filename: str = "profiling_results.json"):
    """Save profile to JSON file."""
    # Convert to dict
    data = {
        "timestamp": profile.timestamp,
        "config": profile.config,
        "summary": profile.summary,
        "memory": asdict(profile.memory),
        "python_layer": {
            "name": profile.python_layer.name,
            "timings": {k: asdict(v) for k, v in profile.python_layer.timings.items()},
            "counts": profile.python_layer.counts,
        },
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nProfile saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="ToonDB HNSW End-to-End Profiling")
    parser.add_argument("--vectors", type=int, default=1000, help="Number of vectors to insert")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: all at once)")
    parser.add_argument("--ef", type=int, default=200, help="ef_construction parameter")
    parser.add_argument("--max-m", type=int, default=16, help="max_connections parameter")
    parser.add_argument("--individual", action="store_true", help="Profile individual inserts (slower)")
    parser.add_argument("--detailed", action="store_true", help="Print detailed breakdown")
    parser.add_argument("--memory", action="store_true", help="Enable memory profiling")
    parser.add_argument("--output", type=str, default="profiling_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Run batch profiling
    profile = run_batch_profiling(
        num_vectors=args.vectors,
        dimension=args.dimension,
        batch_size=args.batch_size,
        ef_construction=args.ef,
        max_connections=args.max_m,
    )
    
    # Run individual profiling if requested
    if args.individual:
        individual_results = run_individual_profiling(
            num_vectors=min(100, args.vectors),
            dimension=args.dimension,
        )
    
    # Print detailed breakdown
    if args.detailed:
        print_detailed_breakdown(profile)
    
    # Save results
    save_profile(profile, args.output)
    
    # Dump Rust-side profiling if enabled
    if os.environ.get("TOONDB_PROFILING") == "1":
        print("\nDumping Rust-side profiling data...")
        dump_profiling()
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total vectors:     {profile.summary['total_vectors']:,}")
    print(f"  Insert time:       {profile.summary['total_insert_time_ms']:.2f} ms")
    print(f"  Throughput:        {profile.summary['vectors_per_second']:,.0f} vec/sec")
    print(f"  Latency:           {profile.summary['us_per_vector']:.2f} µs/vec")
    print(f"  Search time:       {profile.summary['search_time_ms']:.2f} ms")
    print(f"  Peak memory:       {profile.summary['peak_memory_mb']:.2f} MB")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
