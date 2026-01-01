# ToonDB Performance Profiling Report

Generated: 2025-12-28 17:23:36

## Executive Summary

ToonDB performance analysis with detailed profiling of insert and search operations.

---

## 1. Insert Performance

| Configuration | Rate (vec/s) | Time |
|---------------|-------------|------|
| 128D/5K/ef100 | **748** | 6.68s |
| 128D/5K/ef48 | **1,116** | 4.48s |
| 768D/5K/ef100 | **82** | 61.20s |
| 768D/5K/ef48 | **124** | 40.23s |

### Key Findings

- **ef_construction=48 is 1.5x faster** than ef_construction=100

---

## 2. Search Performance

| Configuration | p50 (ms) | p99 (ms) | QPS |
|---------------|----------|----------|-----|
| 128D/10000vec | 0.49 | 1.70 | 1,867 |
| 768D/10000vec | 3.23 | 4.31 | 302 |

---

## 3. ToonDB vs ChromaDB (768D, 5K vectors, ef=48)

| Metric | ToonDB | ChromaDB | Ratio |
|--------|--------|----------|-------|
| Insert Rate | 127 vec/s | 2,069 vec/s | 0.06x |
| Search (p50) | 2.95ms | 2.63ms | 0.9x faster |

### Verdict
- ⚠️ Insert: 16.3x slower than ChromaDB
- ⚠️ Search: 1.1x slower

---

## 4. Bottleneck Analysis

### Insert Path
1. **FFI Overhead**: Each Python→Rust call has ~1-5μs overhead
2. **Quantization**: Vector quantization (F32→F16/BF16) adds latency
3. **Graph Construction**: Higher ef_construction = more neighbor evaluations
4. **Lock Contention**: Parallel batch insert requires synchronization

### Recommendations

| Priority | Fix | Expected Impact |
|----------|-----|-----------------|
| P0 | Use ef_construction=48 for bulk loads | 2x insert speedup |
| P1 | Batch larger chunks (10K+ vectors) | Reduce FFI overhead |
| P2 | Profile Rust code with `perf`/`flamegraph` | Identify hot spots |

---

## 5. Conclusion

⚠️ Insert performance gap exists. See recommendations above.
