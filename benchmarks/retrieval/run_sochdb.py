#!/usr/bin/env python3
"""
Run the retrieval benchmark with SochDB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from sochdb import Database, HnswIndex


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus.jsonl"
QUERIES_PATH = ROOT / "queries.jsonl"
RESULTS_DIR = ROOT / "results"
DEFAULT_DB_PATH = ROOT / "results" / "sochdb_benchmark_db"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "sochdb.json"
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
        help="Path to the local benchmark DB directory",
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
        help="Keep existing DB directory instead of recreating it",
    )
    args = parser.parse_args()

    corpus = load_jsonl(CORPUS_PATH)
    queries = load_jsonl(QUERIES_PATH)

    doc_embeddings = np.load(args.embedding_dir / "doc_embeddings.npy")
    query_embeddings = np.load(args.embedding_dir / "query_embeddings.npy")
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

    db = Database.open(str(args.db_path))
    try:
        print(f"Storing {len(corpus)} documents in SochDB...")
        store_start = time.perf_counter()
        with db.transaction() as txn:
            for record in corpus:
                key = f"docs/{record['id']}".encode("utf-8")
                db.put(key, json.dumps(record).encode("utf-8"), txn.id)
        store_elapsed = time.perf_counter() - store_start

        dimension = int(doc_embeddings.shape[1])
        index = HnswIndex(
            dimension=dimension,
            m=16,
            ef_construction=100,
            metric="cosine",
        )

        print(f"Building HNSW index for {len(doc_ids)} embeddings...")
        build_start = time.perf_counter()
        inserted = index.insert_batch_with_ids(
            np.array([int(doc_id.split("-")[1]) for doc_id in doc_ids], dtype=np.uint64),
            doc_embeddings.astype(np.float32),
        )
        build_elapsed = time.perf_counter() - build_start

        query_timings: list[float] = []
        query_results: list[dict[str, Any]] = []

        id_to_record = {record["id"]: record for record in corpus}

        print(f"Running {len(queries)} queries...")
        for query_record, query_id, query_vec in zip(queries, query_ids, query_embeddings):
            query_start = time.perf_counter()
            result_ids, distances = index.search(query_vec.astype(np.float32), k=args.k)
            query_elapsed = time.perf_counter() - query_start
            query_timings.append(query_elapsed)

            ranked = []
            for doc_numeric_id, distance in zip(result_ids.tolist(), distances.tolist()):
                doc_id = f"doc-{int(doc_numeric_id):03d}"
                ranked.append(
                    {
                        "doc_id": doc_id,
                        "distance": float(distance),
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
            "system": "sochdb",
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
                "indexed_vectors": int(inserted),
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
    finally:
        db.close()


if __name__ == "__main__":
    main()
