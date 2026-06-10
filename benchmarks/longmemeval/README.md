# LongMemEval-S Retrieval Benchmark

This benchmark evaluates SochDB on the same retrieval-only LongMemEval-S shape
used by agentmemory:

- exclude abstention question types
- build one index per question over that question's haystack sessions
- report `recall_any@5/10/20`, `nDCG@10`, and `MRR`

The runner supports:

- `vector`: SochDB native HNSW only
- `hybrid`: SochDB native HNSW + SDK `HybridSearchIndex` BM25/RRF fusion

## Dataset

Download the dataset into the ignored `data/` directory:

```bash
uv run --project sochdb-python --with huggingface_hub python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="xiaowu0162/longmemeval-cleaned",
    filename="longmemeval_s_cleaned.json",
    repo_type="dataset",
    local_dir="benchmarks/longmemeval/data",
)
PY
```

## Run

```bash
uv run --project sochdb-python --with "numpy<2" --with "sentence-transformers<4" \
  python benchmarks/longmemeval/run_sochdb_longmemeval.py \
  --mode vector \
  --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --output benchmarks/longmemeval/results/sochdb_longmemeval_vector.json

uv run --project sochdb-python --with "numpy<2" --with "sentence-transformers<4" \
  python benchmarks/longmemeval/run_sochdb_longmemeval.py \
  --mode hybrid \
  --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --output benchmarks/longmemeval/results/sochdb_longmemeval_hybrid.json
```

Embeddings are cached under `results/embedding_cache/`.

## Local Results

Using `sentence-transformers/all-MiniLM-L6-v2`, `m=16`, `ef_construction=100`,
and `k=20`:

| Mode | recall_any@5 | recall_any@10 | recall_any@20 | nDCG@10 | MRR | query p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vector` | 92.8% | 96.8% | 99.0% | 83.8% | 84.1% | 0.0054 ms |
| `hybrid` | 94.6% | 98.2% | 99.6% | 84.6% | 83.8% | 0.7459 ms |
