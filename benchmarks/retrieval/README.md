# Retrieval Benchmark Harness

This folder contains the first benchmark harness for SochDB's initial product wedge:

- local knowledge retrieval
- lightweight RAG over internal docs
- Python-first local evaluation

The benchmark is meant to answer a small set of practical questions:

1. Is retrieval quality good enough for the first wedge?
2. What latency/footprint tradeoffs do we see?
3. How much workflow complexity does SochDB remove versus local alternatives?

## Current Status

Implemented so far:

- benchmark folder structure
- starter internal-doc corpus
- starter query set with relevance labels
- embedding generation script
- SochDB runner
- SQLite + FAISS runner
- LanceDB runner
- evaluator

Not implemented yet:

- workflow-complexity comparison table
- a larger benchmark summary page

## Dataset Shape

### Corpus

`corpus.jsonl` contains internal-doc-style records with:

- `id`
- `title`
- `body`
- `tags`

### Queries

`queries.jsonl` contains:

- `query_id`
- `query`
- `relevant_ids`

These labels are intentionally simple and human-readable so we can iterate on them quickly.

## Planned Scripts

- `embed.py`
  - generate reproducible embeddings for docs and queries
  - prefers `sentence-transformers`
  - falls back to TF-IDF + SVD if that stack is unavailable
- `run_sochdb.py`
  - benchmark SochDB retrieval flow
- `run_sqlite_faiss.py`
  - benchmark SQLite + FAISS baseline
- `run_lancedb.py`
  - benchmark LanceDB baseline
- `evaluate.py`
  - compute Recall@k, MRR, nDCG, and latency summaries

## Wedge Alignment

This benchmark aligns with the current first evaluator path:

- `examples/python/07_local_knowledge_search.py`
- `docs/getting-started/use-sochdb-when.md`
- `docs/getting-started/local-knowledge-retrieval-comparison.md`

## Public Dataset Path

Alongside the starter internal-doc corpus, we can also prepare a public retrieval dataset from BEIR.

Recommended first pick:

- `SciFact`

Suggested sources:

- Hugging Face dataset card for SciFact: https://huggingface.co/datasets/bigbio/scifact
- BEIR benchmark family: https://huggingface.co/datasets/BeIR

Preparation script:

```bash
conda run -n sochdb-py310 pip install ir_datasets
conda run -n sochdb-py310 python benchmarks/retrieval/prepare_beir_dataset.py --dataset scifact
```

That writes a benchmark-ready dataset under:

- `benchmarks/retrieval/datasets/scifact/corpus.jsonl`
- `benchmarks/retrieval/datasets/scifact/queries.jsonl`
- `benchmarks/retrieval/datasets/scifact/metadata.json`

To run the harness against SciFact instead of the starter internal-doc corpus:

```bash
conda run -n sochdb-py310 python benchmarks/retrieval/embed.py \
  --dataset-dir benchmarks/retrieval/datasets/scifact \
  --output-dir benchmarks/retrieval/results_scifact \
  --backend sentence-transformers

conda run -n sochdb-py310 python benchmarks/retrieval/run_sochdb.py \
  --dataset-dir benchmarks/retrieval/datasets/scifact \
  --embedding-dir benchmarks/retrieval/results_scifact \
  --db-path benchmarks/retrieval/results/sochdb_scifact_db \
  --output benchmarks/retrieval/results/sochdb_scifact.json

conda run -n sochdb-py310 python benchmarks/retrieval/run_sqlite_faiss.py \
  --dataset-dir benchmarks/retrieval/datasets/scifact \
  --embedding-dir benchmarks/retrieval/results_scifact \
  --db-path benchmarks/retrieval/results/sqlite_faiss_scifact.db \
  --output benchmarks/retrieval/results/sqlite_faiss_scifact.json

conda run -n sochdb-py310 python benchmarks/retrieval/run_lancedb.py \
  --dataset-dir benchmarks/retrieval/datasets/scifact \
  --embedding-dir benchmarks/retrieval/results_scifact \
  --db-path benchmarks/retrieval/results/lancedb_scifact \
  --output benchmarks/retrieval/results/lancedb_scifact.json
```

## How To Run

Known working dependency stack in `sochdb-py310` for the `sentence-transformers` backend:

```bash
conda run -n sochdb-py310 pip install "torch==2.2.2" "transformers<5" "sentence-transformers<4" scikit-learn
conda run -n sochdb-py310 pip install "numpy<2"
```

Generate embeddings:

```bash
conda run -n sochdb-py310 python benchmarks/retrieval/embed.py --backend sentence-transformers
```

Run the SochDB benchmark:

```bash
conda run -n sochdb-py310 python benchmarks/retrieval/run_sochdb.py
```

Evaluate the output:

```bash
conda run -n sochdb-py310 python benchmarks/retrieval/evaluate.py benchmarks/retrieval/results/sochdb.json
```

To keep multiple embedding backends side by side:

```bash
conda run -n sochdb-py310 python benchmarks/retrieval/embed.py --backend tfidf-svd --output-dir benchmarks/retrieval/results_tfidf
conda run -n sochdb-py310 python benchmarks/retrieval/run_sochdb.py --embedding-dir benchmarks/retrieval/results_tfidf --output benchmarks/retrieval/results/sochdb_tfidf.json

conda run -n sochdb-py310 python benchmarks/retrieval/embed.py --backend sentence-transformers --output-dir benchmarks/retrieval/results_st
conda run -n sochdb-py310 python benchmarks/retrieval/run_sochdb.py --embedding-dir benchmarks/retrieval/results_st --output benchmarks/retrieval/results/sochdb_st.json

conda run -n sochdb-py310 python benchmarks/retrieval/evaluate.py benchmarks/retrieval/results/sochdb_tfidf.json benchmarks/retrieval/results/sochdb_st.json
```

## Initial Results

Observed on the starter corpus and query set:

| System / Backend | Recall@5 | MRR | nDCG@5 | P50 (ms) | P95 (ms) | Mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sochdb` + `tfidf-svd` | 0.8000 | 0.9750 | 0.8154 | 0.011 | 0.024 | 0.013 |
| `sochdb` + `sentence-transformers/all-MiniLM-L6-v2` | 0.8750 | 1.0000 | 0.8901 | 0.016 | 0.027 | 0.018 |
| `sqlite_faiss` + `sentence-transformers/all-MiniLM-L6-v2` | 0.8750 | 1.0000 | 0.8901 | 0.005 | 0.494 | 0.460 |
| `lancedb` + `sentence-transformers/all-MiniLM-L6-v2` | 0.8750 | 1.0000 | 0.8901 | 2.331 | 4.547 | 3.262 |

Notes:

- the sentence-transformer backend performed better on this small starter dataset
- SochDB and SQLite + FAISS matched on retrieval quality for the sentence-transformer run
- SQLite + FAISS showed a very low median query latency but had one visible outlier query in this initial run, which pushed up its p95 and mean
- LanceDB also matched on retrieval quality, but on this 30-document starter corpus it could not train its PQ-based index and fell back to the non-indexed search path
- the TF-IDF + SVD path is still useful as a local fallback when the model stack is unavailable

## SciFact Results

Observed on the public `SciFact` dataset prepared via `ir_datasets`:

| System | Recall@5 | MRR | nDCG@5 | P50 (ms) | P95 (ms) | Mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sochdb` | 0.7026 | 0.5807 | 0.6056 | 0.149 | 0.221 | 0.155 |
| `sqlite_faiss` | 0.7109 | 0.5883 | 0.6135 | 0.136 | 0.158 | 0.158 |
| `lancedb` | 0.6183 | 0.4843 | 0.5130 | 2.531 | 3.879 | 2.987 |

Notes:

- SochDB is close to SQLite + FAISS on retrieval quality on this larger public dataset
- SochDB and SQLite + FAISS remain in the same general latency range
- LanceDB was materially slower and lower-quality on this run
- this is a much more meaningful comparison point than the tiny starter corpus

## Next Tasks

1. compare workflow complexity and retrieval metrics
2. tune SochDB search/index settings to close the small SciFact quality gap
3. decide whether to track summary outputs separately from generated benchmark artifacts
4. add another public dataset if needed
