# 🔥 ToonDB Comprehensive Benchmark Report

**Date:** December 27, 2024  
**System:** Apple M1 Max, 10 cores, 32GB RAM  
**Rust:** 1.91.1  
**Git:** 2514ab9  
**Build:** Release (optimized)

---

## 📊 Executive Summary

| Category | ToonDB | Competitor | Winner | Margin |
|----------|--------|------------|--------|--------|
| **KV Insert (Rust)** | **1.24M ops/s** | SQLite: 908K | **ToonDB** | **+37%** ✅ |
| **KV Scan (Rust)** | 2.5M rows/s | SQLite: 18.5M | SQLite | 7.3x |
| **Vector Insert** | **124K vec/s** | ChromaDB: 10.6K | **ToonDB** | **11.7x** ✅ |
| **Vector Search** | **35.4K QPS** | ChromaDB: 1.3K QPS | **ToonDB** | **27x** ✅ |

---

## 🚀 Rust KV Benchmarks (100K records, 5 runs median)

### kv_put_scan Workload

| Metric | Value | Direction |
|--------|-------|-----------|
| **Insert Throughput** | **1,133,700 ops/sec** | ↑ Higher is better |
| **Scan Throughput** | **2,633,210 rows/sec** | ↑ Higher is better |
| **Insert Latency p50** | 0.000 ms | ↓ Lower is better |
| **Insert Latency p99** | 0.000 ms | ↓ Lower is better |
| **Total Duration** | 0.13 sec | ↓ Lower is better |

### sqlite_vs_toondb_360 Workload

| Metric | ToonDB | SQLite | Winner |
|--------|--------|--------|--------|
| **Single Insert** | **1,242,956 ops/s** | 907,548 ops/s | **ToonDB +37%** ✅ |
| **Full Scan** | 2,542,330 rows/s | 18,452,312 rows/s | SQLite 7.3x |

```
INSERT PERFORMANCE (100K records):
ToonDB:  ██████████████████████████████████████████████████  1,242,956 ops/s
SQLite:  ████████████████████████████████████░░░░░░░░░░░░░░    907,548 ops/s
         ────────────────────────────────────────────────────
                         ToonDB is 37% FASTER ✅

SCAN PERFORMANCE (100K records):  
SQLite:  ██████████████████████████████████████████████████  18,452,312 rows/s
ToonDB:  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2,542,330 rows/s
         ────────────────────────────────────────────────────
                         SQLite is 7.3x faster
```

---

## 🔍 Vector Search Benchmarks (10K vectors, 128-dim)

### ToonDB vs ChromaDB vs NumPy

| System | Insert (vec/s) | Search (ms) | QPS | vs NumPy |
|--------|---------------|-------------|-----|----------|
| **ToonDB (Rust HNSW)** | **123,940** | **0.028** | **35,402** | **21.4x faster** ✅ |
| ChromaDB | 10,565 | 0.762 | 1,312 | 0.8x |
| NumPy (brute-force) | N/A | 0.603 | 1,657 | baseline |

```
VECTOR INSERT (10K vectors):
ToonDB:    ██████████████████████████████████████████████████  123,940 vec/s
ChromaDB:  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   10,565 vec/s
           ────────────────────────────────────────────────────
                          ToonDB is 11.7x FASTER ✅

VECTOR SEARCH (100 queries, k=10):
ToonDB:    ████████████████████████████████████████████████  35,402 QPS (0.028ms)
NumPy:     ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1,657 QPS (0.603ms)
ChromaDB:  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1,312 QPS (0.762ms)
           ────────────────────────────────────────────────────
                          ToonDB is 27x FASTER ✅
```

### Multi-Dimension Scaling

| Config | Insert (vec/s) | Search (ms) |
|--------|---------------|-------------|
| 1K × 128d | 153,627 | 0.026 |
| 10K × 128d | 118,571 | 0.029 |
| 10K × 384d | 32,346 | 0.055 |
| 10K × 768d | 13,625 | 0.116 |

---

## 🧪 Perf-Harness Results (agent_memory workload)

### Latency Distribution (HDRHistogram)

| Percentile | Latency |
|------------|---------|
| **p50** | **0.548 ms** |
| **p95** | **0.954 ms** |
| **p99** | **0.991 ms** |
| **p99.9** | **0.999 ms** |

### Performance

| Metric | Value |
|--------|-------|
| **QPS** | 7,543 |
| **Duration** | 10.0 sec |
| **Queries** | 75,433 |

---

## 🐍 Python FFI Benchmarks (100K records)

### ToonDB vs SQLite (via Python)

| Operation | ToonDB | SQLite | Ratio |
|-----------|--------|--------|-------|
| Insert | 401,287 ops/s | 451,431 ops/s | 0.89x |
| Scan | 299,251 rows/s | 3,582,218 rows/s | 0.08x |

> **Note:** Python FFI overhead (~400-900ns per call) reduces ToonDB's advantage. Native Rust shows 37% faster insert.

---

## 📈 Key Findings

### 🎉 ToonDB Victories

1. **KV Insert (Rust): +37% faster than SQLite**
   - ToonDB: 1.24M ops/s
   - SQLite: 908K ops/s
   - **Improvement from earlier benchmarks!**

2. **Vector Insert: 11.7x faster than ChromaDB**
   - ToonDB: 124K vec/s
   - ChromaDB: 10.6K vec/s

3. **Vector Search: 27x faster than ChromaDB**
   - ToonDB: 35.4K QPS (0.028ms latency)
   - ChromaDB: 1.3K QPS (0.762ms latency)

4. **vs NumPy Brute-Force: 21.4x faster**
   - ToonDB HNSW beats even in-memory NumPy search

### ⚠️ Areas for Optimization

1. **KV Scan:** SQLite's contiguous B-tree pages give 7.3x better sequential scan
2. **Python FFI:** Per-call overhead impacts small operations

---

## 🏆 Overall Winner Analysis

### Vector Workloads
```
🏆 OVERALL WINNER: ToonDB (faster on BOTH insert AND search!)
   • Insert: ToonDB wins (11.7x faster)
   • Search: ToonDB wins (27.0x faster)
```

### KV Workloads
```
🏆 MIXED RESULTS:
   • Insert: ToonDB wins (+37% faster) ✅
   • Scan: SQLite wins (7.3x faster due to B-tree locality)
```

---

## 📁 Test Artifacts

| File | Description |
|------|-------------|
| `benchmarks/reports/runs/20251227T234504Z_kv_put_scan_default_2514ab9.json` | KV benchmark results |
| `benchmarks/reports/runs/20251227T234506Z_sqlite_vs_toondb_360_default_2514ab9.json` | SQLite comparison |
| `benchmarks/reports/runs/20251227T234453Z_agent_memory_small_2514ab9.json` | Agent memory workload |

---

## 🔧 Test Configuration

### Rust Benchmarks
- **Runs:** 5 (median selected)
- **Records:** 100,000
- **Build:** Release mode (--release)

### Vector Benchmarks
- **Vectors:** 10,000
- **Dimension:** 128
- **Queries:** 100
- **k:** 10 (top-10 neighbors)
- **HNSW Params:** M=16, ef_construction=100

### SQLite Settings
- `PRAGMA journal_mode = WAL`
- `PRAGMA synchronous = NORMAL`

---

## 🚀 Reproduce These Results

```bash
# Build release
cargo build -p benchmarks --release

# Run Rust KV benchmarks
./target/release/perf-run \
  --workload benchmarks/workloads/rust/kv_put_scan.toml --runs 5 --verbose

./target/release/perf-run \
  --workload benchmarks/workloads/rust/sqlite_vs_toondb_360.toml --runs 5 --verbose

# Run perf-harness with HDRHistogram
./target/release/perf-harness \
  --workload benchmarks/workloads/agent_memory.yaml \
  --tier small --duration 30 -v

# Run vector benchmarks
PYTHONPATH=$(pwd)/toondb-python-sdk/src \
TOONDB_LIB_PATH=$(pwd)/target/release \
python3 benchmarks/comprehensive_benchmark.py
```

---

## 📊 Comparison with Previous Benchmarks

| Metric | Dec 25, 2024 | Dec 27, 2024 | Change |
|--------|-------------|--------------|--------|
| ToonDB Insert vs SQLite | 58% of SQLite | **137% of SQLite** | **+79%** 🎉 |
| Vector Insert | 655 vec/s | **123,940 vec/s** | **189x** 🎉 |
| Vector Search QPS | 1,144 QPS | **35,402 QPS** | **31x** 🎉 |

---

*Report generated: December 27, 2024 23:45 UTC*  
*Benchmark System: perf-run v1.0 + perf-harness v2.0*
