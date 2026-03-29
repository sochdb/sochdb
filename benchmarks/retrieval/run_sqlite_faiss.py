#!/usr/bin/env python3
"""
Run the retrieval benchmark with a SQLite + FAISS baseline.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus.jsonl"
QUERIES_PATH = ROOT / "queries.jsonl"
RESULTS_DIR = ROOT / "results"
DEFAULT_DB_PATH = ROOT / "results" / "sqlite_faiss.db"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "sqlite_faiss.json"
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


def connect_sqlite(db_path: Path, recreate: bool) -> sqlite3.Connection:
    if recreate and db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the local SQLite database file",
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
        help="Keep existing SQLite DB instead of recreating it",
    )
    args = parser.parse_args()

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "FAISS is required for this benchmark runner. "
            "Install it in the benchmark environment, for example:\n"
            "  conda run -n sochdb-py310 pip install faiss-cpu"
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

    connection = connect_sqlite(args.db_path, recreate=not args.keep_db)
    try:
        print(f"Storing {len(corpus)} documents in SQLite...")
        store_start = time.perf_counter()
        rows = [
            (
                record["id"],
                record["title"],
                record["body"],
                json.dumps(record.get("tags", [])),
                json.dumps(record),
            )
            for record in corpus
        ]
        connection.executemany(
            """
            INSERT OR REPLACE INTO documents (
                doc_id,
                title,
                body,
                tags_json,
                payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
        store_elapsed = time.perf_counter() - store_start

        print(f"Building FAISS index for {len(doc_ids)} embeddings...")
        build_start = time.perf_counter()
        dimension = int(doc_embeddings.shape[1])
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings)
        build_elapsed = time.perf_counter() - build_start

        id_to_record = {record["id"]: record for record in corpus}
        query_timings: list[float] = []
        query_results: list[dict[str, Any]] = []

        print(f"Running {len(queries)} queries...")
        for query_record, query_id, query_vec in zip(queries, query_ids, query_embeddings):
            query_start = time.perf_counter()
            scores, result_indices = index.search(query_vec.reshape(1, -1), args.k)
            query_elapsed = time.perf_counter() - query_start
            query_timings.append(query_elapsed)

            ranked = []
            for result_index, score in zip(result_indices[0].tolist(), scores[0].tolist()):
                if result_index < 0:
                    continue
                doc_id = doc_ids[result_index]
                ranked.append(
                    {
                        "doc_id": doc_id,
                        "distance": float(score),
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
            "system": "sqlite_faiss",
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
                "indexed_vectors": int(index.ntotal),
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
        connection.close()


if __name__ == "__main__":
    main()
