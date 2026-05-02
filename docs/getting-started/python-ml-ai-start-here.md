---
sidebar_position: 5
---

# Python / ML / AI Start Here

If you are a Python-first ML or AI engineer and want to play with SochDB quickly, this is the best current starting point.

The simple recommendation today is:

1. install the published Python package
2. run one local retrieval workflow
3. use that to understand the product before exploring broader features

On Apple Silicon Macs, do this in a native `arm64` Python environment.

## The Short Version

Start with:

```bash
pip install sochdb
python3 examples/python/07_local_knowledge_search.py
```

Then read:

- [Local Retrieval Start Here](/getting-started/local-retrieval-start-here)
- [Use SochDB When](/getting-started/use-sochdb-when)
- [What Works Today](/getting-started/what-works-today)

## What You Should Expect

For a first evaluation, think of SochDB as:

- a Python-friendly local database path
- local document storage plus retrieval
- a good first wedge for lightweight RAG and internal knowledge search

Do **not** start by trying to evaluate every feature or every notebook in the ecosystem at once.

That is where the current product surface gets confusing.

## Best Current User Journey

### 1. Install the published package

Use:

```bash
pip install sochdb
```

If you are on Apple Silicon, check that your Python is native first:

```bash
python - <<'PY'
import platform
print(platform.machine())
PY
```

Expected:

```text
arm64
```

If you hit environment issues, use:

- [Installation](/getting-started/installation)
- [Python Install Matrix](/getting-started/python-install-matrix)

### 2. Run one narrow local workflow

Run:

```bash
python3 examples/python/07_local_knowledge_search.py
```

This is the clearest current path because it shows:

- local records
- local vector indexing
- semantic retrieval
- fetching records back from the same overall system

### 3. Evaluate the local retrieval wedge

Next read:

- [Local Retrieval Start Here](/getting-started/local-retrieval-start-here)
- [Local Knowledge Retrieval Comparison](/getting-started/local-knowledge-retrieval-comparison)
- [Retrieval Evaluation](/guides/retrieval-evaluation)

### 4. Use the maturity docs before branching out

Before you try more advanced surfaces, read:

- [Use SochDB When](/getting-started/use-sochdb-when)
- [What Works Today](/getting-started/what-works-today)

These are important because not every path in the repo ecosystem is equally mature or equally ideal as a first evaluation path.

## Why This Guidance Exists

Right now, a new user can reasonably see multiple Python-related surfaces:

- the published `sochdb` package
- repo-local Python source/build flows
- notebooks and examples
- broader platform and SDK directions

Those are all real parts of the product story, but they are not the same thing.

If your goal is to understand SochDB quickly, the best move is to start with the published Python package and the local retrieval workflow first.

## Recommended First Goal

Your first question should be:

- "Is SochDB a clear and credible way to do local knowledge retrieval in Python?"

Not:

- "Can I validate every platform feature in one sitting?"

If the answer to the first question is yes, then it becomes worth exploring more of the broader product surface.

## Related Docs

- [Installation](/getting-started/installation)
- [Python Install Matrix](/getting-started/python-install-matrix)
- [Local Retrieval Start Here](/getting-started/local-retrieval-start-here)
- [Use SochDB When](/getting-started/use-sochdb-when)
- [Local Knowledge Retrieval Comparison](/getting-started/local-knowledge-retrieval-comparison)
- [What Works Today](/getting-started/what-works-today)
