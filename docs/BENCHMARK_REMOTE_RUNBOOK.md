# SochDB Benchmark Runbook (Remote Server)

Step-by-step guide to build the optimized SochDB engine and run the benchmark
suites **on the remote server** (`root@65.108.78.80`) without touching the
laptop's CPU and **without clobbering the WIP** already on the server.

> **Why not just run on the laptop?** The full suites build 1M–10M vector HNSW
> graphs across all cores (~1600% CPU, ~6 GB RAM). That belongs on a server.

---

## 0. Facts about the environment

| | Laptop (dev) | Remote server |
|---|---|---|
| Host | macOS, Apple Silicon (aarch64) | `65.108.78.80`, Ubuntu 24.04 |
| Arch | **aarch64** | **x86_64** |
| Cores / RAM | 18 / 64 GB | 12 / 62 GB |
| Toolchain | rust + maturin | rust 1.96 + maturin + `sochdb 2.0.2` |
| Repo | `~/sochdb` on `main` @ `f1f8d1b` (dirty) | `/root/sochdb` on `release/0.5.1` @ `f93f850` (**dirty WIP**) |

> ⚠️ **DO NOT overwrite `/root/sochdb`.** It has uncommitted changes to
> `sochdb-index/src/hnsw.rs` and gRPC files. We build in a **separate**
> directory `/root/sochdb-bench` instead.

> ℹ️ **Arch note:** The laptop-only optimizations (aarch64 `PRFM` prefetch,
> high‑ILP NEON dot product) are `#[cfg(target_arch = "aarch64")]` and compile
> *out* on the x86_64 server — that's fine. The arch-independent wins still
> apply on x86_64: **recall backfill (keepPrunedConnections)**, **PQ lock-hoist**,
> and the **normalized cosine fast path**. The x86 `_mm_prefetch`/AVX path is
> native on the server.

---

## 1. (On the laptop) Sync the optimized working tree to the server

This copies the current local working tree (which contains all the HNSW
optimizations) into a fresh directory on the server. It **excludes** heavy/junk
dirs and never touches `/root/sochdb`.

```bash
# Run from the laptop, inside ~/sochdb
cd ~/sochdb

rsync -az --delete \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.venv*/' \
  --exclude '**/__pycache__/' \
  --exclude 'benchmarks/**/datasets/' \
  --exclude 'benchmarks/results/' \
  -e "ssh -i ~/.ssh/poc_server_new" \
  ./  root@65.108.78.80:/root/sochdb-bench/
```

> The first sync uploads the whole source tree (a few hundred MB without
> `target/`). Re-running later is incremental and fast.

---

## 2. (On the server) Open a session

```bash
ssh -i ~/.ssh/poc_server_new root@65.108.78.80
cd /root/sochdb-bench
```

> Tip: run long jobs inside `tmux` so they survive disconnects:
> `tmux new -s bench` (reattach later with `tmux attach -t bench`).

---

## 3. (On the server) Create the benchmark Python env

```bash
cd /root/sochdb-bench

python3 -m venv .venv-bench
source .venv-bench/bin/activate
pip install --upgrade pip

# Core build + bench deps
pip install maturin numpy scikit-learn psutil pyarrow

# Competitor engines
pip install faiss-cpu hnswlib lancedb

# Real-dataset (BEIR/SciFact) prep
pip install ir_datasets
```

---

## 4. (On the server) Build the optimized engine (release)

```bash
cd /root/sochdb-bench/sochdb-python
maturin develop --release        # installs the `sochdb` module into .venv-bench
```

Verify it imported the freshly built module:

```bash
python3 -c "import sochdb; print('OK', sochdb.__file__)"
```

> This is the **only** CPU-heavy compile step (~a few minutes). It uses all
> cores briefly; that's expected on a server.

---

## 5. Benchmark suite A — Competitive (FAISS vs hnswlib vs SochDB)

Random Gaussian data — recall is meaningless here for all engines; this measures
**build time, QPS, p50/p95/p99 latency, RAM**. SochDB should beat **hnswlib**;
FAISS remains the raw-speed reference.

```bash
cd /root/sochdb-bench
source .venv-bench/bin/activate

# Start with 1M @ 768d, all three engines:
python benchmarks/competitive_benchmark.py \
  --baseline all --scale 1m --dim 768 \
  --output benchmarks/results

# Heavier (only if the box is idle): 10M
python benchmarks/competitive_benchmark.py \
  --baseline all --scale 10m --dim 768 \
  --output benchmarks/results
```

Results land in `benchmarks/results/competitive_1m_768d.json` (and `_10m_`).

> To run a single engine: `--baseline sochdb` (or `faiss` / `hnswlib`).

---

## 6. Benchmark suite B — Retrieval on REAL data (SciFact / BEIR)

This is the **legitimate "real data" win target**: SochDB vs `sqlite_faiss`
(exact flat) vs `lancedb`, measuring Recall@5 / MRR / nDCG / latency.

### 6.1 Prepare the dataset (downloads SciFact via ir_datasets)

```bash
cd /root/sochdb-bench
source .venv-bench/bin/activate

python benchmarks/retrieval/prepare_beir_dataset.py \
  --dataset scifact \
  --output-dir benchmarks/retrieval/datasets/scifact
# → corpus.jsonl (5183 docs), queries.jsonl (300), metadata.json
```

### 6.2 Embed docs + queries

> No GPU / sentence-transformers needed — TF-IDF+SVD is deterministic and fast.
> (If you later `pip install sentence-transformers`, use `--backend auto` for
> stronger embeddings and higher absolute recall for **all** engines.)

```bash
python benchmarks/retrieval/embed.py \
  --backend tfidf-svd \
  --dataset-dir benchmarks/retrieval/datasets/scifact \
  --output-dir benchmarks/retrieval/datasets/scifact/emb \
  --svd-dim 256 --max-features 8192
```

### 6.3 Run all three engines against the same embeddings

```bash
DS=benchmarks/retrieval/datasets/scifact
EMB=$DS/emb

# SochDB — fast preset (m=16, efc=100)
python benchmarks/retrieval/run_sochdb.py \
  --dataset-dir $DS --embedding-dir $EMB \
  --m 16 --ef-construction 100 --precision f32 --k 5 \
  --output benchmarks/results/scifact_sochdb_m16.json

# SochDB — quality preset (m=48, efc=200)
python benchmarks/retrieval/run_sochdb.py \
  --dataset-dir $DS --embedding-dir $EMB \
  --m 48 --ef-construction 200 --precision f32 --k 5 \
  --output benchmarks/results/scifact_sochdb_m48.json

# Baseline: SQLite + FAISS flat (exact, 100% recall ceiling)
python benchmarks/retrieval/run_sqlite_faiss.py \
  --dataset-dir $DS --embedding-dir $EMB --k 5 \
  --output benchmarks/results/scifact_sqlite_faiss.json

# Baseline: LanceDB
python benchmarks/retrieval/run_lancedb.py \
  --dataset-dir $DS --embedding-dir $EMB --k 5 \
  --output benchmarks/results/scifact_lancedb.json
```

### 6.4 Evaluate (Recall@5 / MRR / nDCG / latency table)

```bash
python benchmarks/retrieval/evaluate.py --k 5 \
  benchmarks/results/scifact_sochdb_m16.json \
  benchmarks/results/scifact_sochdb_m48.json \
  benchmarks/results/scifact_sqlite_faiss.json \
  benchmarks/results/scifact_lancedb.json \
  --output-json benchmarks/results/scifact_summary.json
```

**What "winning" looks like here:**
- SochDB Recall@5 **matches** flat FAISS (the exact ceiling) at far lower
  latency than LanceDB, and the quality preset (`m=48`) closes the recall gap.
- SochDB beats **LanceDB** decisively on latency and recall.
- (`sqlite_faiss` is exact brute force → 100% recall but only viable at small
  scale; it is the correctness reference, not a scaling competitor.)

---

## 7. Pull results back to the laptop (optional)

```bash
# From the laptop
rsync -az -e "ssh -i ~/.ssh/poc_server_new" \
  root@65.108.78.80:/root/sochdb-bench/benchmarks/results/ \
  ~/sochdb/benchmarks/results-remote/
```

---

## 8. Discipline / guardrails (do not skip)

- **Never** alter or "tune" the benchmark scripts to make SochDB look better —
  every win must come from real engine improvements.
- **Never** `rsync` into `/root/sochdb` — it holds unrelated uncommitted WIP.
  Always use `/root/sochdb-bench`.
- Keep heavy runs (`10m`, `100m`) on the server, **not** the laptop.
- If a run wedges the box, find and stop it:
  `pgrep -fa benchmark` then `kill <PID>`.

---

## 9. One-shot copy/paste (server side, after step 1 sync)

```bash
ssh -i ~/.ssh/poc_server_new root@65.108.78.80
tmux new -s bench
cd /root/sochdb-bench
python3 -m venv .venv-bench && source .venv-bench/bin/activate
pip install -U pip maturin numpy scikit-learn psutil pyarrow faiss-cpu hnswlib lancedb ir_datasets
( cd sochdb-python && maturin develop --release )
python -c "import sochdb; print('engine OK', sochdb.__file__)"

# competitive
python benchmarks/competitive_benchmark.py --baseline all --scale 1m --dim 768 --output benchmarks/results

# scifact end-to-end
python benchmarks/retrieval/prepare_beir_dataset.py --dataset scifact --output-dir benchmarks/retrieval/datasets/scifact
python benchmarks/retrieval/embed.py --backend tfidf-svd --dataset-dir benchmarks/retrieval/datasets/scifact --output-dir benchmarks/retrieval/datasets/scifact/emb --svd-dim 256 --max-features 8192
DS=benchmarks/retrieval/datasets/scifact; EMB=$DS/emb
python benchmarks/retrieval/run_sochdb.py --dataset-dir $DS --embedding-dir $EMB --m 16 --ef-construction 100 --precision f32 --k 5 --output benchmarks/results/scifact_sochdb_m16.json
python benchmarks/retrieval/run_sochdb.py --dataset-dir $DS --embedding-dir $EMB --m 48 --ef-construction 200 --precision f32 --k 5 --output benchmarks/results/scifact_sochdb_m48.json
python benchmarks/retrieval/run_sqlite_faiss.py --dataset-dir $DS --embedding-dir $EMB --k 5 --output benchmarks/results/scifact_sqlite_faiss.json
python benchmarks/retrieval/run_lancedb.py --dataset-dir $DS --embedding-dir $EMB --k 5 --output benchmarks/results/scifact_lancedb.json
python benchmarks/retrieval/evaluate.py --k 5 benchmarks/results/scifact_sochdb_m16.json benchmarks/results/scifact_sochdb_m48.json benchmarks/results/scifact_sqlite_faiss.json benchmarks/results/scifact_lancedb.json --output-json benchmarks/results/scifact_summary.json
```
