---
sidebar_position: 5
---

# Local Retrieval Start Here

This is the clearest current evaluator path for SochDB.

If you want to understand the product quickly and practically, start here instead of trying every feature in the repo.

## Who This Path Is For

This path is best for:

- Python-first AI engineers
- local or embedded knowledge retrieval
- lightweight RAG over internal docs
- evaluators comparing SochDB against local multi-tool stacks

## What You Will Evaluate

One concrete workflow:

1. store documents locally
2. build or query a local vector index
3. run a semantic query
4. fetch matching records from the same overall system

That is the current strongest SochDB wedge.

## Recommended Evaluation Order

### 1. Confirm Python setup

Start with:

- [Installation](/getting-started/installation)
- [Python Install Matrix](/getting-started/python-install-matrix)

If you only do one install path first, use:

```bash
pip install sochdb
```

### 2. Run the local retrieval demo

Run:

```bash
python3 examples/python/07_local_knowledge_search.py
```

This gives you:

- local document storage
- local vector retrieval
- one narrow embedded workflow

### 3. Read the positioning page

Use:

- [Use SochDB When](/getting-started/use-sochdb-when)

That page answers:

- who SochDB is for first
- when it is a good fit
- when not to start with it

### 4. Compare it to common local alternatives

Use:

- [Local Knowledge Retrieval Comparison](/getting-started/local-knowledge-retrieval-comparison)

That page helps answer:

- what you would actually manage locally
- how SochDB compares to SQLite + FAISS, pgvector, and vector-first tools

### 5. Look at the benchmark summary

Use:

- [Local Retrieval Benchmark Summary](/Users/saisandeepkantareddy/Downloads/sochdb/benchmarks/retrieval/summary.md)

That gives you:

- starter benchmark results
- SciFact public-dataset results
- SochDB fast and quality presets
- workflow-complexity interpretation

## Recommended SochDB Presets

If you benchmark or compare SochDB locally, start with these:

### Fast

- `m=16`
- `ef_construction=100`
- `precision=f32`

Use when:

- you want quick local iteration
- you want the default evaluation path

### Quality

- `m=48`
- `ef_construction=200`
- `precision=f32`

Use when:

- you want the strongest measured quality on the current SciFact benchmark
- you are willing to spend more latency budget

## What Good Evaluation Looks Like

A good first evaluation does **not** try to answer whether SochDB should replace your whole stack.

A good first evaluation asks:

- is the local retrieval workflow clear?
- does it reduce glue code enough to matter?
- are the benchmark results credible enough for this wedge?
- do the fast/quality presets match my priorities?

## Related Docs

- [Installation](/getting-started/installation)
- [Python Install Matrix](/getting-started/python-install-matrix)
- [Use SochDB When](/getting-started/use-sochdb-when)
- [Local Knowledge Retrieval Comparison](/getting-started/local-knowledge-retrieval-comparison)
- [What Works Today](/getting-started/what-works-today)
