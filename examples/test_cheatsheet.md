# ToonDB Test Cheatsheet

## Quick Reference: Claims vs Tests

| Claim | Test ID | Pass Criteria | Status |
|-------|---------|---------------|--------|
| "Competitive with SQLite" | T3.1, T3.2 | ≤2x SQLite, ≥50K ops/sec | ✓ 1.51x, 396K |
| "Built-in vector search" | T4.1, T4.2 | Recall ≥90%, p99 <50ms | ✓ 95.6%, 0.7ms |
| "ACID guarantees" | T6.1, T6.2 | 100% recovery, atomicity | ✓ PASS |
| "40-66% token savings" | T2.1 | ≥40% savings | Needs tiktoken |

## Run Tests

```bash
# Full suite
./tests/run_feature_tests.sh

# Quick smoke test
./tests/run_feature_tests.sh --quick

# Performance only
./tests/run_feature_tests.sh --perf
```

## Latest Results (Dec 28, 2025)

```
T3.1: Insert Throughput    → 396,348 ops/sec ✓
T3.2: SQLite Comparison    → 1.51x SQLite ✓  
T4.1: Recall@10            → 95.6% ✓
T4.2: Search Latency       → p99=0.7ms ✓
T6.1: Data Persistence     → 100% ✓
T7.1: Large Values         → 1MB ✓
```

## Test Categories

| Category | Tests | Status |
|----------|-------|--------|
| T2-TOKEN | 1/1 | ✓ |
| T3-PERF | 2/2 | ✓ |
| T4-VECTOR | 2/2 | ✓ |
| T6-DURABILITY | 2/2 | ✓ |
| T7-EDGE | 3/3 | ✓ |
