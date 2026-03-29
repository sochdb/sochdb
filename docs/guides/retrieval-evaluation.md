# Retrieval Evaluation Framework

This guide defines a practical framework for evaluating SochDB retrieval workflows.

It is intended for:

- product decisions
- benchmark planning
- default-setting for embeddings and index parameters
- empirical support for comparisons or paper work

The goal is not to measure everything at once. The goal is to answer a small set of meaningful questions with reproducible experiments.

---

## What We Are Evaluating

This framework evaluates SochDB as a retrieval system for AI workflows, not just as an ANN index.

That means we care about three layers:

1. retrieval quality
2. retrieval efficiency
3. workflow usefulness

Those three layers map more closely to actual user value than a nearest-neighbor benchmark alone.

---

## Primary Evaluation Questions

Start with these questions:

1. How good is SochDB's dense retrieval quality for local knowledge retrieval?
2. When does hybrid retrieval beat dense-only retrieval?
3. What embedding sizes and models give the best quality/speed tradeoff?
4. What HNSW defaults are good enough without over-tuning?
5. Does SochDB reduce workflow complexity enough to justify use over a multi-tool local stack?

If an experiment does not help answer one of these, it is probably not first-priority.

---

## Recommended First Use Case

Use this framework first for:

- local knowledge retrieval
- lightweight RAG over internal docs
- Python-first local evaluation

This aligns with the current strongest wedge:

- `examples/python/07_local_knowledge_search.py`

---

## Metrics

### Retrieval Quality

Use:

- `Recall@k`
- `MRR`
- `nDCG@k`
- `Hit@k`

These are standard, understandable, and paper-friendly.

### Efficiency

Use:

- `P50 latency`
- `P95 latency`
- `P99 latency` when useful
- index build time
- memory footprint
- disk footprint

### Workflow Metrics

These are important for SochDB's product story.

Track:

- number of moving parts
- setup complexity
- amount of retrieval glue code
- local-first evaluation friction

These are not traditional IR metrics, but they matter when comparing SochDB to assembled local stacks.

---

## Evaluation Modes

Use four experiment modes.

### 1. Dense Retrieval

Question:

- how good is basic semantic retrieval quality?

Use:

- one embedding model
- one HNSW configuration
- one realistic corpus

### 2. Hybrid Retrieval

Question:

- when does dense + keyword improve results over dense-only?

Compare:

- dense-only
- keyword-only
- hybrid

### 3. Metadata-Aware Retrieval

Question:

- what happens when retrieval must also respect filters?

Examples:

- tenant
- product area
- access level
- recency window

### 4. Workflow-Level Evaluation

Question:

- is the end-to-end local retrieval workflow simpler or more useful than a multi-tool local stack?

This is where SochDB's differentiated story matters most.

---

## Datasets / Workloads

Do not start with too many datasets.

Use a small but meaningful set.

### Recommended first set

1. internal knowledge base / enterprise docs
2. support or FAQ corpus
3. one domain-specific corpus if needed

Good properties:

- realistic document lengths
- metadata fields
- clear relevant results
- possible filter dimensions

---

## Baselines

Keep the baseline set small and relevant.

### Recommended first baselines

1. SQLite + FAISS
2. Postgres + pgvector
3. one vector-first local tool such as LanceDB or Chroma

Why these:

- they are realistic choices for local Python-first evaluators
- they cover assembled local stack, relational vector extension, and vector-first alternatives

This matches the comparison surface already documented in:

- [Local Knowledge Retrieval Comparison](/getting-started/local-knowledge-retrieval-comparison)

---

## Variables to Sweep

### Embeddings

Questions:

- does a larger embedding actually improve retrieval enough to justify cost?
- what dimension is the best local default?

Suggested comparisons:

- smaller local-friendly embedding model
- stronger but heavier embedding model
- dimension tradeoff where relevant

### HNSW Parameters

Suggested parameters:

- `m`
- `ef_construction`
- search depth / query-time search parameter if exposed

Goal:

- identify good default settings, not just best-case tuned settings

### Retrieval Strategy

Compare:

- dense-only
- keyword-only
- hybrid

This is likely the most product-relevant comparison.

---

## Recommended First Experiment Matrix

### Experiment Set A: Embedding Tradeoff

Measure:

- Recall@k
- MRR
- P50/P95 latency
- memory

Output:

- one table showing quality vs speed vs footprint

### Experiment Set B: Dense vs Hybrid

Measure:

- quality lift
- latency cost

Output:

- one comparison table for dense-only, keyword-only, and hybrid

### Experiment Set C: HNSW Default Selection

Measure:

- retrieval quality
- build time
- query latency

Output:

- recommended starting defaults

### Experiment Set D: Workflow Comparison

Compare:

- SochDB local retrieval workflow
- SQLite + FAISS
- Postgres + pgvector

Measure:

- setup steps
- components managed
- amount of retrieval glue
- whether the result feels meaningfully simpler

Output:

- evaluator-facing comparison asset

---

## Reproducibility Guidance

To make results useful:

- pin embedding models and versions
- record corpus size and document stats
- record hardware and OS
- record Python version and architecture
- record SochDB version / commit
- run multiple trials where latency is reported

This matters especially on Apple Silicon where architecture mismatches can distort results.

---

## What This Framework Should Produce

Good outputs from this work:

- recommended default embeddings for the first wedge
- recommended HNSW settings
- one honest comparison asset
- one benchmark summary table
- one methods/results section for paper contribution

That is enough to support product, docs, and research at the same time.

---

## What Not To Do First

Do not start with:

- huge benchmark grids
- many synthetic datasets
- every possible retrieval mode
- platform-wide claims disconnected from the first wedge

The best first evaluation is:

- one wedge
- one small set of baselines
- one small set of metrics
- one reproducible workflow

---

## Suggested Next Implementation Tasks

1. choose the first corpus
2. define query/relevance labels
3. choose the first embedding comparison
4. choose the first baseline stack
5. create a reproducible benchmark harness

---

## Related Docs

- [Use SochDB When](/getting-started/use-sochdb-when)
- [Local Knowledge Retrieval Comparison](/getting-started/local-knowledge-retrieval-comparison)
- [What Works Today](/getting-started/what-works-today)
- [Vector Search Guide](/guides/vector-search)
