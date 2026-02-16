# sochdb-bench — Comparative Benchmark Suite

Benchmarks **SochDB** against **SQLite**, **DuckDB**, and (optionally) **LanceDB**
across four workload families:

| Workload | What it measures |
|----------|-----------------|
| **OLTP** | Sequential writes, sequential reads, random reads, batch writes, deletes |
| **Analytics** | Bulk insert, filter / aggregate / group-by / range-scan queries |
| **Vector** | Insert + brute-force kNN search |
| **Mixed** | 80 % reads / 20 % writes |
| **Storage** | On-disk size and write amplification |

## Quick Start

```bash
# Run all workloads at default scale (10 K ops)
cargo run --release -- --all

# Higher scale, export results
cargo run --release -- --all --scale 10 --export both

# OLTP only, skip DuckDB
cargo run --release -- --oltp --skip duckdb

# Include LanceDB (adds ~60 s extra compile)
cargo run --release --features lancedb-bench -- --all
```

## CLI Reference

```
sochdb-bench [OPTIONS]

Options:
  --all              Run every workload (default if none specified)
  --oltp             OLTP workloads only
  --analytics        Analytics workloads only
  --vector           Vector workloads only
  --mixed            Mixed read/write workload
  --scale <N>        Multiplier — 1 = 10 K ops, 10 = 100 K  [default: 1]
  --dim <D>          Vector dimension                        [default: 128]
  --k <K>            Top-k for vector search                 [default: 10]
  --value-size <B>   KV value size in bytes                  [default: 256]
  --export <FMT>     "csv", "json", or "both"
  --skip <DB,...>    Skip databases (sochdb, sqlite, duckdb, lancedb)
```

## Criterion Microbenchmarks

```bash
cargo bench --bench micro
```

Measures `point_write`, `point_read`, and `batch_write_1000` for SochDB vs SQLite
with statistically rigorous iteration counts.

## Architecture

```
sochdb-bench/
├── Cargo.toml
├── README.md
├── benches/
│   └── micro.rs            # Criterion benchmarks
└── src/
    ├── main.rs             # CLI entry point (clap)
    ├── lib.rs              # BenchDb trait, DataGen, LatencyRecorder
    ├── workloads.rs        # Workload definitions
    ├── report.rs           # Terminal tables, CSV/JSON export
    └── adapters/
        ├── mod.rs
        ├── sochdb_adapter.rs   # SochDB embedded (sochdb-storage)
        ├── sqlite_adapter.rs   # rusqlite + WAL
        ├── duckdb_adapter.rs   # duckdb-rs, 4 threads
        └── lancedb_adapter.rs  # LanceDB (feature-gated)
```

### Adding a New Database

1. Create `src/adapters/mydb_adapter.rs` implementing `BenchDb`.
2. Add `pub mod mydb_adapter;` to `src/adapters/mod.rs`.
3. Instantiate in `main.rs` alongside the other adapters.

## Output Example

```
━━━ oltp_seq_write ━━━
╭──────────┬────────┬──────────┬──────────────────┬─────────┬─────────┬──────────┬─────────╮
│ Database │ Ops    │ Time (s) │ Throughput       │ p50 (μs)│ p99 (μs)│ p99.9    │ Mean    │
├──────────┼────────┼──────────┼──────────────────┼─────────┼─────────┼──────────┼─────────┤
│ ★ SochDB │ 10.0K  │ 0.042    │ 238.1K           │ 3.2     │ 15.4    │ 42.1     │ 4.2     │
│ SQLite   │ 10.0K  │ 0.198    │ 50.5K            │ 12.8    │ 89.3    │ 245.0    │ 19.8    │
│ DuckDB   │ 10.0K  │ 0.312    │ 32.1K            │ 21.4    │ 120.1   │ 380.0    │ 31.2    │
╰──────────┴────────┴──────────┴──────────────────┴─────────┴─────────┴──────────┴─────────╯
```

★ marks the fastest database for each workload.

## Notes

- All databases write to a shared `tempfile::TempDir` that is cleaned up automatically.
- SochDB uses `DatabaseConfig::throughput_optimized()`.
- SQLite uses WAL mode + NORMAL synchronous + memory-mapped I/O.
- DuckDB is configured with 4 threads and a 2 GB memory limit.
- Vector search is brute-force L2 distance for all databases (no ANN index).
  For SochDB and SQLite/DuckDB this means scanning all stored vectors.
  LanceDB (when enabled) uses its native vector search.
- `DataGen` uses a seeded `ChaCha8Rng` for deterministic, reproducible data.
