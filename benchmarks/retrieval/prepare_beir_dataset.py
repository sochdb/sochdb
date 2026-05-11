#!/usr/bin/env python3
"""
Prepare a BEIR dataset into the local benchmark JSONL format.

Current supported source:
    SciFact via ir_datasets (`beir/scifact`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "scifact"

IR_DATASETS_MAP = {
    "scifact": {
        "base": "beir/scifact",
        "qrels": "beir/scifact/test",
    },
}


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_from_ir_datasets(dataset_name: str) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    try:
        import ir_datasets
    except ImportError as exc:
        raise SystemExit(
            "The `ir_datasets` package is required for public benchmark prep. Install it with:\n"
            "  conda run -n sochdb-py310 pip install ir_datasets"
        ) from exc

    dataset_spec = IR_DATASETS_MAP.get(dataset_name)
    if dataset_spec is None:
        supported = ", ".join(sorted(IR_DATASETS_MAP))
        raise SystemExit(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")

    base_id = dataset_spec["base"]
    qrels_id = dataset_spec["qrels"]
    ds = ir_datasets.load(base_id)
    qrels_ds = ir_datasets.load(qrels_id)

    corpus = []
    for doc in ds.docs_iter():
        corpus.append(
            {
                "id": str(doc.doc_id),
                "title": "",
                "body": str(getattr(doc, "text", "")),
                "tags": [],
            }
        )

    qrels_by_query: dict[str, list[str]] = {}
    for qrel in qrels_ds.qrels_iter():
        if int(qrel.relevance) <= 0:
            continue
        qrels_by_query.setdefault(str(qrel.query_id), []).append(str(qrel.doc_id))

    queries = []
    for query in ds.queries_iter():
        query_id = str(query.query_id)
        relevant_ids = sorted(set(qrels_by_query.get(query_id, [])))
        if not relevant_ids:
            continue
        queries.append(
            {
                "query_id": query_id,
                "query": str(query.text),
                "relevant_ids": relevant_ids,
            }
        )

    metadata = {
        "source": "ir_datasets",
        "dataset": dataset_name,
        "dataset_id": base_id,
        "qrels_dataset_id": qrels_id,
        "corpus_count": len(corpus),
        "query_count": len(queries),
        "qrels_count": sum(len(v) for v in qrels_by_query.values()),
        "notes": "Prepared for the SochDB retrieval benchmark harness",
    }

    return corpus, queries, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Public benchmark dataset name, currently supports: scifact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write benchmark-ready JSONL files",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap on the number of queries to emit (0 means no cap)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (ROOT / "datasets" / args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading public retrieval dataset: {args.dataset}")
    corpus, queries, metadata = load_from_ir_datasets(args.dataset)

    if args.max_queries > 0:
        queries = queries[: args.max_queries]
        metadata["query_count"] = len(queries)

    corpus_path = output_dir / "corpus.jsonl"
    queries_path = output_dir / "queries.jsonl"
    metadata_path = output_dir / "metadata.json"

    write_jsonl(corpus_path, corpus)
    write_jsonl(queries_path, queries)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Prepared benchmark dataset:")
    print(f"  - corpus:  {corpus_path} ({len(corpus)} docs)")
    print(f"  - queries: {queries_path} ({len(queries)} labeled queries)")
    print(f"  - meta:    {metadata_path}")


if __name__ == "__main__":
    main()
