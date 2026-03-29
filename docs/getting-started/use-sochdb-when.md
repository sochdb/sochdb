---
sidebar_position: 4
---

# Use SochDB When...

This page is for technical evaluators deciding whether SochDB fits their current workflow.

The short version:

- Use SochDB when you want a local or embedded AI retrieval workflow with fewer moving parts
- Start with one narrow use case, not a full-stack migration
- Prefer SochDB when your pain is glue code between storage, vectors, and retrieval logic

---

## Best First Fit

SochDB is currently easiest to understand and evaluate for:

- Python-first AI engineers
- local or embedded knowledge retrieval
- lightweight RAG over internal docs
- prototypes or product features where you want data + retrieval together

That is the clearest current wedge.

If you're evaluating SochDB for the first time, the best first path is:

1. [Quick Start](/getting-started/quickstart)
2. [Python Install Matrix](/getting-started/python-install-matrix)
3. `examples/python/07_local_knowledge_search.py`

---

## Use SochDB When

### 1. You want fewer moving parts in a local AI workflow

A common stack today looks like:

- app database for records and metadata
- vector database or vector library for embeddings
- custom retrieval and context assembly code

If your pain is managing multiple components for one retrieval feature, SochDB is a good candidate to evaluate.

### 2. You are building local knowledge retrieval or lightweight RAG

This is the strongest current first-use case:

- store document payloads locally
- build or query vector indexes
- retrieve matching records from the same overall system

### 3. You want an embedded-first developer experience

SochDB is a good fit when:

- you prefer local development
- you want to avoid standing up multiple services for your first version
- you want a Python-first workflow for evaluation and iteration

### 4. You want to evaluate one narrow AI feature first

The best adoption path is usually:

- try SochDB for one retrieval-heavy feature
- keep the rest of your stack unchanged
- expand only if that first workflow proves valuable

This is much lower risk than trying to replace an entire platform.

---

## Do Not Start With SochDB When

### 1. You are trying to replace your whole data stack immediately

If your goal is:

- replace existing production Postgres usage
- consolidate all analytics, transactions, search, and AI infrastructure at once

then SochDB is not the right first move.

Use it first for a narrow retrieval or embedded AI workflow.

### 2. You already have a mature stack that solves the problem cleanly

If you already have:

- Postgres + pgvector
- or SQLite + FAISS
- or a vector DB that is working well for your current product

then SochDB only makes sense if it simplifies your workflow enough to justify trying it.

### 3. You only need a pure vector index

If all you need is:

- ANN search
- one index
- no broader database workflow

then a dedicated vector tool or library may be enough.

### 4. You need a broad, already-proven enterprise platform story today

SochDB is strongest today as a focused embedded/local AI workflow layer, not as a “replace everything enterprise platform” pitch.

---

## How SochDB Differs From Common Alternatives

### SQLite + FAISS

Good when:

- you are comfortable stitching pieces together yourself

Tradeoff:

- one system for data
- another for vectors
- app code handles the glue

SochDB is interesting when you want a more unified local workflow.

### Postgres + pgvector

Good when:

- you already live in Postgres
- you want a familiar general-purpose DB base

Tradeoff:

- often heavier than needed for local or embedded first-touch AI workflows

SochDB is interesting when you want a lighter local path with fewer moving parts.

### Qdrant / LanceDB / Chroma

Good when:

- vector retrieval is your main need

Tradeoff:

- you may still need to coordinate broader app data and retrieval behavior outside the vector layer

SochDB is interesting when you want a single retrieval-oriented local system rather than a vector-only component.

---

## Best Current Evaluation Path

If you want to evaluate SochDB seriously, do this:

1. Use the published Python package: `pip install sochdb`
2. Run the local knowledge retrieval demo
3. Decide whether the reduced workflow complexity is meaningful for your use case

That is a better first evaluation path than trying to use every feature in the docs at once.

---

## Honest Current Position

SochDB is broad, but the maturity is not uniform across every surface.

Today, the most trustworthy way to evaluate it is:

- local embedded database workflow
- Python-first usage
- retrieval-oriented examples
- one narrow AI feature at a time

---

## Related Docs

- [Quick Start](/getting-started/quickstart)
- [Installation](/getting-started/installation)
- [Python Install Matrix](/getting-started/python-install-matrix)
- [Local Knowledge Retrieval Comparison](/getting-started/local-knowledge-retrieval-comparison)
- [Python SDK Guide](/guides/python-sdk)
