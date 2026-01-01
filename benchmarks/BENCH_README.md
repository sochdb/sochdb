# ToonDB Benchmark System

A unified, reproducible benchmark framework for ToonDB performance testing.

## Latest Results (December 27, 2024)

| Category | ToonDB | Competitor | Winner |
|----------|--------|------------|--------|
| **KV Insert** | 1.24M ops/s | SQLite: 908K | **ToonDB +37%** ✅ |
| **Vector Insert** | 124K vec/s | ChromaDB: 10.6K | **ToonDB 11.7x** ✅ |
| **Vector Search** | 35.4K QPS | ChromaDB: 1.3K | **ToonDB 27x** ✅ |

See [BENCHMARK_RESULTS_2024-12-27.md](./BENCHMARK_RESULTS_2024-12-27.md) for full details.

## Directory Structure

```
benchmarks/
├── workloads/           # Benchmark workload definitions
│   ├── rust/            # Rust-native workloads (TOML)
│   │   ├── kv_put_scan.toml
│   │   ├── sqlite_vs_toondb_360.toml
│   │   └── vector_hnsw.toml
│   ├── python/          # Python-based workloads (TOML)
│   │   ├── ffi_kv_vs_sqlite.toml
│   │   └── vector_vs_chroma.toml
│   ├── agent_memory.yaml      # Agent memory workload (YAML)
│   ├── rag_retrieval.yaml     # RAG retrieval workload (YAML)
│   └── timeline_queries.yaml  # Timeline queries workload (YAML)
├── datasets/            # Benchmark datasets
│   ├── manifest.json    # Dataset registry
│   ├── scripts/         # Dataset generation scripts
│   ├── users_100k/      # KV benchmark dataset
│   └── vectors_10k_128/ # Vector benchmark dataset
├── baselines/           # Baseline results by machine
│   └── <machine>/       # e.g., mac-studio, ci-linux
│       └── <workload>/
│           └── default.json
├── reports/             # Benchmark run outputs
│   ├── runs/            # Individual run results (JSON)
│   └── comparisons/     # Baseline comparison reports
└── BENCHMARK_RESULTS_2024-12-27.md  # Latest comprehensive report
```

## Two Benchmark Binaries

### 1. `perf-run` - Original benchmark runner
- Schema v1.0
- TOML workload definitions
- Basic metrics + regression detection

### 2. `perf-harness` - Enhanced harness (NEW)
- Schema v2.0
- YAML workload definitions
- HDRHistogram for proper p50/p95/p99/p99.9
- Recall@k with ground truth
- Environment reproducibility checks
- Regression gates for CI

## Quick Start

### 1. Build the perf-run binary

```bash
cargo build -p benchmarks --bin perf-run --release
```

With jemalloc for better allocation performance:
```bash
cargo build -p benchmarks --bin perf-run --release --features jemalloc
```

### 2. Run a benchmark

```bash
# Run KV put/scan benchmark
cargo run -p benchmarks --bin perf-run --release -- \
  --workload benchmarks/workloads/rust/kv_put_scan.toml \
  --out benchmarks/reports/runs \
  --json

# Run with dataset
cargo run -p benchmarks --bin perf-run --release -- \
  --workload benchmarks/workloads/rust/kv_put_scan.toml \
  --dataset benchmarks/datasets/users_100k \
  --runs 5 \
  --verbose
```

### 3. Compare against baseline

```bash
cargo run -p benchmarks --bin perf-run --release -- \
  --workload benchmarks/workloads/rust/kv_put_scan.toml \
  --baseline benchmarks/baselines/m1max/kv_put_scan/users_100k.json
```

### 4. Update baseline (requires clean git)

```bash
cargo run -p benchmarks --bin perf-run --release -- \
  --workload benchmarks/workloads/rust/kv_put_scan.toml \
  --update-baseline
```

## Workload Definition Format

Workloads are defined in TOML files:

```toml
[workload]
name = "kv_put_scan"
description = "Measures ToonDB KV insert and scan performance"
runner = "rust"        # "rust" or "python"
binary = "perf-run"    # For rust runners
script = "bench.py"    # For python runners

[params]
n = 100000
batch_size = 1000

[metrics]
collect = [
    "insert_throughput_ops_per_s",
    "scan_throughput_rows_per_s",
    "latency_ms_p50",
    "latency_ms_p99",
]

[thresholds]
insert_throughput_ops_per_s = { threshold = 5.0, better = "higher" }
latency_ms_p99 = { threshold = 10.0, better = "lower" }
```

## Output JSON Schema (v1.0)

Each benchmark run produces a JSON file with this structure:

```json
{
  "schema_version": "1.0",
  "run": {
    "id": "20251227T210501Z_kv_put_scan_users_100k_abc1234",
    "timestamp_utc": "2025-12-27T21:05:01Z",
    "git": { "sha": "abc1234", "dirty": false },
    "runner": { "kind": "rust", "command": "...", "duration_s": 12.34 }
  },
  "env": {
    "os": "macos",
    "cpu": "Apple M1 Max",
    "cores": 10,
    "ram_gb": 64,
    "rustc": "rustc 1.75.0",
    "features": ["jemalloc"]
  },
  "workload": {
    "name": "kv_put_scan",
    "params": { "n": 100000, "batch_size": 1000 }
  },
  "dataset": {
    "name": "users_100k",
    "hash": "sha256:..."
  },
  "metrics": {
    "insert_throughput_ops_per_s": { 
      "value": 81234.5, 
      "unit": "ops/s", 
      "better": "higher" 
    },
    "latency_ms_p50": { 
      "value": 1.23, 
      "unit": "ms", 
      "better": "lower" 
    }
  },
  "artifacts": {
    "logs": "benchmarks/reports/runs/.../stdout.log",
    "raw": "benchmarks/reports/runs/.../raw_samples.json"
  }
}
```

## Regression Detection

Default thresholds:
- **Latency p50**: +5% regression threshold
- **Latency p99**: +10% regression threshold  
- **Throughput**: -5% regression threshold
- **Peak RSS**: +8% regression threshold

Regression rules:
- **Lower-is-better** (latency, RSS): regress if `new > baseline * (1 + threshold)`
- **Higher-is-better** (throughput): regress if `new < baseline * (1 - threshold)`

Comparison output:
```json
{
  "schema_version": "1.0",
  "baseline_ref": "benchmarks/baselines/m1max/kv_put_scan/users_100k.json",
  "run_ref": "20251227T210501Z_...",
  "status": "fail",
  "diffs": [
    {
      "metric": "latency_ms_p99",
      "baseline": 4.10,
      "new": 4.72,
      "delta_pct": 15.12,
      "threshold_pct": 10.0,
      "result": "regression"
    }
  ]
}
```

## CI Integration

Add to your CI workflow:

```yaml
- name: Run benchmarks
  run: |
    cargo build -p benchmarks --bin perf-run --release
    
    # Run benchmark and compare with baseline
    ./target/release/perf-run \
      --workload benchmarks/workloads/rust/kv_put_scan.toml \
      --dataset benchmarks/datasets/users_100k \
      --baseline benchmarks/baselines/ci-linux-x86/kv_put_scan/users_100k.json \
      --out benchmarks/reports/runs \
      --runs 5

- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: benchmarks/reports/
```

## Generating Datasets

```bash
# Generate user records
python benchmarks/datasets/scripts/generate_users.py \
  -n 100000 \
  -o benchmarks/datasets/users_100k

# Generate vectors
python benchmarks/datasets/scripts/generate_vectors.py \
  -n 10000 \
  -d 128 \
  -o benchmarks/datasets/vectors_10k_128
```

## Available Workloads

### Rust Workloads

| Workload | Description |
|----------|-------------|
| `kv_put_scan` | Basic KV insert throughput and full-table scan |
| `sqlite_vs_toondb_360` | Comprehensive ToonDB vs SQLite comparison |
| `vector_hnsw` | HNSW vector index insert and search |

### Python Workloads

| Workload | Description |
|----------|-------------|
| `ffi_kv_vs_sqlite` | Python FFI KV operations vs SQLite |
| `vector_vs_chroma` | ToonDB vector vs ChromaDB comparison |
