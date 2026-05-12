# Local Retrieval Benchmark Summary

## Goal

Evaluate SochDB against realistic local alternatives for the current first wedge:

- local knowledge retrieval
- lightweight RAG
- Python-first local evaluation

This summary is the short version of the fuller benchmark notes in `README.md`.

## Systems Compared

- `sochdb`
- `sqlite_faiss`
- `lancedb`

## Datasets

### Starter corpus

A small internal-doc-style corpus used to validate that the harness, baselines, and evaluator all work end to end.

### SciFact

A larger public retrieval dataset prepared via `ir_datasets`.

Current SciFact shape:

- `5183` docs
- `300` labeled queries

## Key Results

### Starter corpus

All three systems matched closely on retrieval quality.

Main takeaway:

- the harness itself works
- the starter corpus is useful for local validation, but too small for strong comparative claims

### SciFact

| System | Recall@5 | MRR | nDCG@5 | P50 (ms) | P95 (ms) | Mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sochdb` (`m=16`, `ef_construction=100`) | 0.7026 | 0.5807 | 0.6056 | 0.149 | 0.221 | 0.155 |
| `sochdb` (`m=48`, `ef_construction=200`) | 0.7109 | 0.5883 | 0.6135 | 0.304 | 0.696 | 0.376 |
| `sqlite_faiss` | 0.7109 | 0.5883 | 0.6135 | 0.136 | 0.158 | 0.158 |
| `lancedb` | 0.6183 | 0.4843 | 0.5130 | 2.531 | 3.879 | 2.987 |

## What We Learned

### 1. SochDB is credible on a real public dataset

SochDB is not just “working on a toy example.”

On SciFact:

- SochDB default is already close to SQLite + FAISS
- SochDB quality preset matched SQLite + FAISS on retrieval quality

### 2. Tuning matters

The best measured quality preset so far is:

- `m=48`
- `ef_construction=200`
- `precision=f32`

This closes the quality gap, but increases latency relative to the default preset.

### 3. LanceDB was weaker in this setup

On SciFact, LanceDB was:

- slower
- lower quality

So for this specific wedge and current setup, it did not look as strong as SochDB or SQLite + FAISS.

### 4. Workflow simplicity is still an important differentiator

Even when quality is close, the local workflow shape still matters:

- SochDB keeps payload storage and retrieval in one local system
- SQLite + FAISS requires explicit coordination between storage and vector index
- LanceDB has a local story, but did not perform as well here

## Recommended SochDB Presets

### Fast

- `m=16`
- `ef_construction=100`
- `precision=f32`

Use when:

- iterating quickly
- default local evaluation
- latency matters more than squeezing out the last bit of recall

### Quality

- `m=48`
- `ef_construction=200`
- `precision=f32`

Use when:

- retrieval quality matters more
- you can spend more latency budget
- you want the fairest quality comparison against SQLite + FAISS

## Product Interpretation

The benchmark now supports a stronger product statement:

- SochDB is already credible for the Python-first local retrieval wedge
- it can match a strong local baseline on quality with the right preset
- the product case still depends heavily on workflow simplicity and fewer moving parts

## Recommended Next Steps

1. Keep the benchmark summary linked from evaluator-facing docs.
2. Use the `fast` and `quality` presets explicitly in examples and benchmark notes.
3. Gather evaluator feedback on the local retrieval wedge.
4. Add another public dataset only if it materially improves confidence.
