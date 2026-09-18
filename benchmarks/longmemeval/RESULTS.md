# SochDB on LongMemEval-S — measured results

`results/` is gitignored, so raw outputs stay local and this file is the tracked
record. Numbers below were produced by the commands in [Reproducing](#reproducing)
on the dataset and revision named there.

## What these numbers are, and what they are not

These are **retrieval** metrics. The harness indexes each question's haystack and
scores whether the gold-evidence turn appears in the top-k. It never asks a model
to answer.

Published LongMemEval headline scores — including the frequently quoted 95.60% —
are **end-to-end question-answering accuracy under an LLM judge**. The two are not
comparable in either direction, and a retrieval number must not be reported
alongside them as though it were the same measurement. Retrieval recall is an
upper bound on what a reader could achieve, not an answer-quality claim.

**Abstention is excluded.** The dataset's abstention questions are filtered out,
so nothing here measures whether the system declines to answer when the corpus
does not support one. Abstention is a first-class memory competency and is
currently unmeasured.

## Results — 500 questions, 246,750 haystack turns

Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim.
Index: HNSW, `m=16`, `ef_construction=100`, cosine, `top_k=20`.

| metric | `vector` | `hybrid` | delta |
|---|---|---|---|
| recall_any@5 | 92.8% | **97.0%** | +4.2 |
| recall_any@10 | 96.8% | **99.0%** | +2.2 |
| recall_any@20 | 99.0% | **99.8%** | +0.8 |
| NDCG@10 | 83.8% | **90.0%** | +6.2 |
| MRR | 84.1% | **90.2%** | +6.1 |
| query p50 | **6.8 us** | 125.5 us | 18x slower |
| query p95 | **8.6 us** | 236.0 us | 27x slower |
| build p50 | **0.81 ms** | 14.94 ms | 18x slower |

`vector` is HNSW alone; `hybrid` adds BM25 and fuses with RRF.

### By question type, recall_any@5

| type | n | `vector` | `hybrid` | delta |
|---|---|---|---|---|
| single-session-user | 70 | **77.1%** | 92.9% | **+15.8** |
| single-session-preference | 30 | 86.7% | 96.7% | +10.0 |
| knowledge-update | 78 | 91.0% | 100.0% | +9.0 |
| temporal-reasoning | 133 | 96.2% | 95.5% | -0.7 |
| multi-session | 133 | 97.7% | 97.7% | 0.0 |
| single-session-assistant | 56 | 98.2% | 100.0% | +1.8 |

The aggregate hides where the fusion actually earns its 18x latency. Five of six
types are at or near saturation for the vector lane alone; essentially all of the
gain is concentrated in `single-session-user`, the one category where the question
paraphrases a single user statement and shares few content words with it. That is
the vocabulary-gap case BM25 is for.

`temporal-reasoning` is the one category the fusion makes slightly worse, which is
consistent with lexical matching rewarding shared date and number tokens that do
not indicate the right session.

So the honest reading is not "hybrid is better." It is that hybrid buys one weak
category at a uniform 18x latency cost across all six, and a router that used BM25
only where the vector lane is uncertain would likely keep the gain without the
cost. That is untested.

## Reproducing

Dataset — pin the **cleaned** release. Scores from before and after the September
2025 cleanup are on different haystacks and are not comparable:

```bash
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="xiaowu0162/longmemeval-cleaned",
    filename="longmemeval_s_cleaned.json",
    repo_type="dataset",
    local_dir="benchmarks/longmemeval/data",
)
PY
```

Run both modes:

```bash
for mode in vector hybrid; do
  uv run --project sochdb-python --with "numpy<2" --with "sentence-transformers<4" \
    python benchmarks/longmemeval/run_sochdb_longmemeval.py \
    --mode "$mode" \
    --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
    --output "benchmarks/longmemeval/results/sochdb_longmemeval_${mode}.json"
done
```

Embeddings are cached under `results/embedding_cache`, so the second run reuses
the first run's vectors.

### Run provenance

| | |
|---|---|
| dataset | `xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned.json` (265 MB) |
| questions | 500 |
| haystack turns | 246,750 |
| host | i9-13900K, 8P+16E, 123 GB RAM |
| single run | no repeats; latency is per-query p50/p95 within the run |

Single-run, single-host, and unreplicated. Enough to characterise the retrieval
lane and to compare the two modes against each other on identical inputs; not
enough to publish as a cross-system comparison.
