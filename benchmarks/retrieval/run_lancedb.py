#!/usr/bin/env python3
"""
Run the retrieval benchmark with a LanceDB baseline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus.jsonl"
QUERIES_PATH = ROOT / "queries.jsonl"
RESULTS_DIR = ROOT / "results"
DEFAULT_DB_PATH = ROOT / "results" / "lancedb"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "lancedb.json"
EMBEDDING_DIR = ROOT / "results"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the local LanceDB directory",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=EMBEDDING_DIR,
        help="Directory containing doc/query embedding outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Result JSON path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k results to retrieve per query",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep existing LanceDB directory instead of recreating it",
    )
    args = parser.parse_args()

    try:
        import lancedb
        import pyarrow as pa
    except ImportError as exc:
        raise SystemExit(
            "LanceDB is required for this benchmark runner. "
            "Install it in the benchmark environment, for example:\n"
            "  conda run -n sochdb-py310 pip install lancedb pyarrow"
        ) from exc

    corpus = load_jsonl(CORPUS_PATH)
    queries = load_jsonl(QUERIES_PATH)

    doc_embeddings = np.load(args.embedding_dir / "doc_embeddings.npy").astype(np.float32)
    query_embeddings = np.load(args.embedding_dir / "query_embeddings.npy").astype(np.float32)
    doc_ids = json.loads((args.embedding_dir / "doc_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((args.embedding_dir / "query_ids.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.embedding_dir / "embedding_metadata.json").read_text(encoding="utf-8"))

    if len(corpus) != len(doc_ids) or len(corpus) != len(doc_embeddings):
        raise SystemExit("Corpus length does not match embedding outputs")
    if len(queries) != len(query_ids) or len(queries) != len(query_embeddings):
        raise SystemExit("Query length does not match embedding outputs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.db_path.exists() and not args.keep_db:
        shutil.rmtree(args.db_path)

    print(f"Opening LanceDB at {args.db_path}...")
    db = lancedb.connect(str(args.db_path))

    rows = []
    for record, vector in zip(corpus, doc_embeddings):
        rows.append(
            {
                "doc_id": record["id"],
                "title": record["title"],
                "body": record["body"],
                "tags_json": json.dumps(record.get("tags", [])),
                "payload_json": json.dumps(record),
                "vector": vector.tolist(),
            }
        )

    schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("body", pa.string()),
            pa.field("tags_json", pa.string()),
            pa.field("payload_json", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), int(doc_embeddings.shape[1]))),
        ]
    )

    print(f"Storing {len(rows)} documents in LanceDB...")
    store_start = time.perf_counter()
    table = db.create_table(
        "documents",
        data=rows,
        schema=schema,
        mode="overwrite",
    )
    store_elapsed = time.perf_counter() - store_start

    print(f"Building LanceDB vector index for {len(doc_ids)} embeddings...")
    build_start = time.perf_counter()
    index_built = False
    index_error: str | None = None
    try:
        try:
            table.create_index(metric="cosine", vector_column_name="vector")
        except TypeError:
            # Older LanceDB builds use a different signature.
            table.create_index("vector")
        index_built = True
    except RuntimeError as exc:
        # Small corpora can fail PQ-based index training. Keep the run usable.
        index_error = str(exc)
        print("LanceDB index build skipped:", index_error)
    build_elapsed = time.perf_counter() - build_start

    id_to_record = {record["id"]: record for record in corpus}
    query_timings: list[float] = []
    query_results: list[dict[str, Any]] = []

    print(f"Running {len(queries)} queries...")
    for query_record, query_id, query_vec in zip(queries, query_ids, query_embeddings):
        query_start = time.perf_counter()
        result = (
            table.search(query_vec.tolist())
            .limit(args.k)
            .to_list()
        )
        query_elapsed = time.perf_counter() - query_start
        query_timings.append(query_elapsed)

        ranked = []
        for row in result:
            doc_id = row["doc_id"]
            ranked.append(
                {
                    "doc_id": doc_id,
                    "distance": float(row.get("_distance", 0.0)),
                    "title": id_to_record.get(doc_id, {}).get("title"),
                }
            )

        query_results.append(
            {
                "query_id": query_id,
                "query": query_record["query"],
                "relevant_ids": query_record["relevant_ids"],
                "results": ranked,
                "latency_ms": round(query_elapsed * 1000, 3),
            }
        )

    output = {
        "system": "lancedb",
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "embedding_model": metadata["model_name"],
        "embedding_dimension": metadata["dimension"],
        "top_k": args.k,
        "storage": {
            "db_path": str(args.db_path),
        },
        "build": {
            "documents_stored": len(corpus),
            "store_time_ms": round(store_elapsed * 1000, 3),
            "index_build_time_ms": round(build_elapsed * 1000, 3),
            "indexed_vectors": len(rows),
            "index_built": index_built,
            "index_error": index_error,
        },
        "query_latency": {
            "p50_ms": round(percentile(query_timings, 50) * 1000, 3),
            "p95_ms": round(percentile(query_timings, 95) * 1000, 3),
            "mean_ms": round(float(np.mean(query_timings)) * 1000, 3),
        },
        "queries": query_results,
    }

    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
