# ToonDB Backend Profiling Report

**Generated:** 2025-12-28 17:30

## Executive Summary

**Key Finding:** The bottleneck is NOT in the Rust HNSW code - it's in the Python FFI path.

| Dimension | Pure Rust | Python FFI | Gap |
|-----------|-----------|------------|-----|
| 128D | **9,590 vec/s** | 748 vec/s | 13x |
| 768D | **1,567 vec/s** | 127 vec/s | 12x |

---

## Rust Insert Performance (ef_construction=48)

| Config | Batch Insert | Rate |
|--------|-------------|------|
| 128D × 1K | 103ms | 9,725 vec/s |
| 128D × 5K | 488ms | 10,246 vec/s |
| 128D × 10K | 1.04s | 9,590 vec/s |
| 768D × 1K | 521ms | 1,920 vec/s |
| 768D × 5K | 3.22s | 1,554 vec/s |
| 768D × 10K | 6.38s | 1,567 vec/s |

---

## Bottleneck Analysis

### ✅ Rust Core is Fast
- 128D: ~10K vec/s (competitive)
- 768D: ~1.5K vec/s (expected for high-dim)

### ⚠️ Python FFI is the Bottleneck
| Path | 768D Rate | Overhead |
|------|-----------|----------|
| Pure Rust | 1,567 vec/s | Baseline |
| Python FFI | 127 vec/s | **12x slower** |

### Root Cause: FFI Data Copying
1. NumPy array → Python bytes → C buffer → Rust Vec
2. Each vector requires memory allocation
3. No SIMD optimization in copy path

---

## Recommendations

| Priority | Fix | Expected Impact |
|----------|-----|-----------------|
| **P0** | Use Rust CLI/binary directly for bulk loads | Eliminate FFI overhead |
| **P1** | Zero-copy FFI with memory-mapped arrays | 5-10x speedup |
| **P2** | Batch FFI: send 1000 vectors per call vs 1 | 10x reduction in call overhead |

---

## Conclusion

**ToonDB's Rust core performs well.** The 12x gap between Rust and Python is due to FFI overhead, not algorithmic issues. For production bulk inserts, use the Rust API directly or implement zero-copy FFI bindings.
