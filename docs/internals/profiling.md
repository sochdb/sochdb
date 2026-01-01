# ToonDB End-to-End Profiling Documentation

## Overview

This document describes the comprehensive end-to-end profiling infrastructure for ToonDB's HNSW vector index, covering the entire pipeline from Python SDK through FFI to Rust core.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Python SDK Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 10_e2e_profiling.py                                                  │   │
│  │  - PrecisionTimer (nanosecond resolution)                            │   │
│  │  - ProfiledVectorIndex wrapper                                       │   │
│  │  - Memory tracking via tracemalloc                                   │   │
│  │  - Per-operation timing: numpy, dtype, validation, FFI ptr creation  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ vector.py (FFI bindings)                                             │   │
│  │  - ctypes bindings to libtoondb_index.dylib                          │   │
│  │  - insert_batch → hnsw_insert_batch                                  │   │
│  │  - Profiling control: enable_profiling(), dump_profiling()           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ FFI Boundary
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Rust FFI Layer (ffi.rs)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ hnsw_insert_batch()                                                  │   │
│  │  - Profiled: ffi.slice_from_raw, ffi.id_conversion                   │   │
│  │  - Calls: index.insert_batch_contiguous()                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HNSW Core (hnsw.rs)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ insert_batch_contiguous_bulk()                                       │   │
│  │                                                                       │   │
│  │ Phase 1: Parallel Quantization (hnsw.phase1.quantize_parallel)       │   │
│  │   - rayon parallel iterator                                          │   │
│  │   - Creates HnswNode with QuantizedVector                            │   │
│  │                                                                       │   │
│  │ Phase 2: Map Insert (hnsw.phase2.map_insert)                         │   │
│  │   - Insert nodes into DashMap                                        │   │
│  │   - Update entry point                                               │   │
│  │                                                                       │   │
│  │ Phase 3: Connection Building (hnsw.phase3.*)                         │   │
│  │   - connect_total: Overall connection time                           │   │
│  │   - search_layer: Graph navigation                                   │   │
│  │   - neighbor_select: RNG heuristic ← MAIN BOTTLENECK                 │   │
│  │   - add_connections: Bidirectional edge creation                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Profiling Infrastructure

### Python-Side Profiling

**File**: `toondb-python-sdk/examples/10_e2e_profiling.py`

```python
# Key components
class PrecisionTimer:
    """Nanosecond-precision timer using time.perf_counter_ns()"""
    
class TimingStats:
    """Statistical aggregation: count, total, mean, std, min, max, p50, p95, p99"""

class ProfiledVectorIndex:
    """Wrapper that instruments all VectorIndex operations"""
    
    def insert_batch_profiled(vectors, ids):
        # Times: numpy_ascontiguous, dtype_conversion, data_validation, 
        #        ffi_ptr_creation, ffi_call_overhead, batch_total
```

**Output**: `profiling_results.json`

### Rust-Side Profiling

**File**: `toondb-index/src/profiling.rs`

```rust
// Enabling profiling
TOONDB_PROFILING=1  // Environment variable

// Key components
pub struct Timer { /* start, stop, record */ }
pub struct OpStats { count, total_ns, min_ns, max_ns, item_count }
pub struct ProfileCollector { /* global singleton with thread-safe stats */ }

// FFI exports
pub extern "C" fn toondb_profiling_enable();
pub extern "C" fn toondb_profiling_disable();
pub extern "C" fn toondb_profiling_dump();
```

**Output**: `/tmp/toondb_profile.json`

## Running Profiling

### Basic Usage

```bash
cd toondb-python-sdk

# Enable profiling and run
export TOONDB_PROFILING=1
export TOONDB_LIB_PATH=/path/to/target/release/libtoondb_index.dylib
python3 examples/10_e2e_profiling.py

# Analyze results
python3 examples/11_profiling_analysis.py
```

### Customizable Parameters

Edit `10_e2e_profiling.py`:
```python
# Configuration
NUM_VECTORS = 1000        # Number of vectors to insert
DIMENSION = 768           # Vector dimension (e.g., OpenAI embeddings)
EF_CONSTRUCTION = 200     # HNSW ef parameter (quality vs speed)
MAX_CONNECTIONS = 16      # HNSW M parameter
```

## Profiling Results (1000 vectors, 768d)

### Summary Metrics

| Metric | Value |
|--------|-------|
| Total Insert Time | 13,533 ms (13.5s) |
| Throughput | 74 vectors/sec |
| Latency | 13.5 ms/vector |
| Peak Memory | 9.45 MB |
| Search Time | 1.7 ms |

### Time Breakdown

| Phase | Time (ms) | Percentage | Per-Vector |
|-------|-----------|------------|------------|
| **Neighbor Selection** | 10,365 | **76.6%** | 11.07 ms |
| Add Connections | 1,908 | 14.1% | 2.04 ms |
| Search Layer | 1,076 | 7.9% | 1.15 ms |
| Quantization | 0.99 | `<0.01%` | 1.05 µs |
| Map Insert | 0.11 | `<0.01%` | 0.12 µs |
| FFI Overhead | 0.003 | `<0.01%` | 0.003 µs |

### Bottleneck Analysis

```
CRITICAL BOTTLENECK: Neighbor Selection (RNG Heuristic) - 77%

The select_neighbors_heuristic function implements the Relative Neighborhood 
Graph (RNG) pruning which is essential for HNSW quality but computationally 
expensive.

For each of N vectors inserted:
- Search returns ~ef candidates (200)
- RNG checks each candidate against all previously selected neighbors
- Each check requires a full 768-dimensional distance calculation
- Complexity: O(N × ef × m × D) where D is dimension

With N=1000, ef=200, m=16, D=768:
- Approximately 1000 × 200 × 16 = 3.2M distance calculations
- Each 768d cosine distance = 3 SIMD operations (dot, norm_a, norm_b)
```

## Optimization Recommendations

### 1. Reduce ef_construction

```python
# Current: ef_construction = 200
# Recommended: ef_construction = 100-128
# Trade-off: ~5% recall reduction, 2x faster insert

index = VectorIndex(
    dimension=768,
    ef_construction=100,  # Reduced from 200
    max_connections=16,
)
```

### 2. Use Batch Distance Computation

The RNG heuristic currently does scalar distance calculations. Batching 
candidates and using SIMD would provide 4-8x speedup on distance computation.

### 3. Pre-compute Distance Cache

Cache pairwise distances between candidates during search to avoid 
recomputation in neighbor selection.

### 4. Approximate RNG for Early Inserts

Use simpler heuristics (top-k by distance) when graph is sparse, 
switch to full RNG when graph is denser.

### 5. Parallel Neighbor Selection

The RNG check for each candidate can be parallelized since it's independent.

## Files Created

| File | Purpose |
|------|---------|
| `toondb-python-sdk/examples/10_e2e_profiling.py` | Main profiling script |
| `toondb-python-sdk/examples/11_profiling_analysis.py` | Analysis and visualization |
| `toondb-index/src/profiling.rs` | Rust profiling infrastructure |
| `profiling_results.json` | Python-side profiling output |
| `/tmp/toondb_profile.json` | Rust-side profiling output |

## JSON Output Format

### Python Profile (profiling_results.json)

```json
{
  "timestamp": "2025-12-29T00:33:55",
  "config": {
    "num_vectors": 1000,
    "dimension": 768,
    "ef_construction": 200,
    "max_connections": 16
  },
  "summary": {
    "total_insert_time_ms": 13533.17,
    "vectors_per_second": 73.89,
    "peak_memory_mb": 9.45
  },
  "python_layer": {
    "timings": {
      "numpy_ascontiguous": { "total_ms": 0.008 },
      "dtype_conversion": { "total_ms": 0.013 },
      "ffi_ptr_creation": { "total_ms": 0.032 },
      "batch_total": { "total_ms": 13533.05 }
    }
  }
}
```

### Rust Profile (/tmp/toondb_profile.json)

```json
{
  "total_elapsed_ms": 13537.17,
  "operations": {
    "hnsw.phase3.neighbor_select": {
      "count": 1,
      "total_ms": 10365.00,
      "mean_us": 10365003.96,
      "item_count": 936,
      "per_item_us": 11073.72
    },
    "hnsw.phase3.add_connections": { ... },
    "hnsw.phase3.search_layer": { ... },
    "hnsw.phase1.quantize_parallel": { ... },
    "hnsw.phase2.map_insert": { ... }
  }
}
```

## Conclusion

The profiling reveals that **77% of insertion time is spent in the RNG neighbor 
selection heuristic**, which is the core HNSW algorithm for maintaining graph 
quality. This is expected behavior for high-dimensional vectors.

Key insights:
- Python/FFI overhead is negligible (`<0.001%`)
- Quantization and map operations are very fast (`<0.01%`)
- The bottleneck is algorithmic, not I/O or memory
- Optimization should focus on the neighbor selection loop

For faster inserts at the cost of some recall:
1. Reduce ef_construction from 200 to 100
2. Use lower precision (F16) for distance calculations
3. Implement batched SIMD distance in the RNG loop
