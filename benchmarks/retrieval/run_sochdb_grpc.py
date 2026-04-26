#!/usr/bin/env python3
"""
Run the retrieval benchmark against a local or remote SochDB 2.0 gRPC server.

This script intentionally writes the same high-level result shape as the
embedded benchmark harness so the existing evaluator can be reused.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import grpc
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "sochdb_grpc.json"
DEFAULT_K = 5


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
    parser.add_argument("--host", default="127.0.0.1", help="SochDB gRPC host")
    parser.add_argument("--port", type=int, default=50051, help="SochDB gRPC port")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing corpus.jsonl and queries.jsonl",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help="Directory containing doc/query embedding outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Result JSON path",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k results")
    parser.add_argument(
        "--index-name",
        default=f"bench_{int(time.time())}",
        help="Remote index name to create for this run",
    )
    args = parser.parse_args()

    generated_dir = Path(__file__).resolve().parents[2] / "sochdb-sdk" / "python" / "sochdb_sdk" / "generated"
    if not generated_dir.exists():
        raise SystemExit(
            "Generated Python stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
        )

    sys.path.insert(0, str(generated_dir))
    import sochdb_pb2  # type: ignore
    import sochdb_pb2_grpc  # type: ignore

    corpus = load_jsonl(args.dataset_dir / "corpus.jsonl")
    queries = load_jsonl(args.dataset_dir / "queries.jsonl")

    doc_embeddings = np.load(args.embedding_dir / "doc_embeddings.npy")
    query_embeddings = np.load(args.embedding_dir / "query_embeddings.npy")
    doc_ids = json.loads((args.embedding_dir / "doc_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((args.embedding_dir / "query_ids.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.embedding_dir / "embedding_metadata.json").read_text(encoding="utf-8"))

    if len(corpus) != len(doc_ids) or len(corpus) != len(doc_embeddings):
        raise SystemExit("Corpus length does not match embedding outputs")
    if len(queries) != len(query_ids) or len(queries) != len(query_embeddings):
        raise SystemExit("Query length does not match embedding outputs")

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = sochdb_pb2_grpc.VectorIndexServiceStub(channel)

    # Best-effort cleanup from previous failed runs.
    try:
        stub.DropIndex(sochdb_pb2.DropIndexRequest(name=args.index_name))
    except grpc.RpcError:
        pass

    create_start = time.perf_counter()
    create_resp = stub.CreateIndex(
        sochdb_pb2.CreateIndexRequest(
            name=args.index_name,
            dimension=int(doc_embeddings.shape[1]),
            metric=sochdb_pb2.DISTANCE_METRIC_COSINE,
            config=sochdb_pb2.HnswConfig(
                max_connections=16,
                ef_construction=100,
                ef_search=64,
            ),
        )
    )
    create_elapsed = time.perf_counter() - create_start
    if not create_resp.success:
        raise SystemExit(f"CreateIndex failed: {create_resp.error}")

    numeric_ids = np.arange(1, len(doc_ids) + 1, dtype=np.uint64)
    numeric_to_doc_id = {
        int(numeric_id): doc_id for numeric_id, doc_id in zip(numeric_ids.tolist(), doc_ids)
    }
    id_to_record = {record["id"]: record for record in corpus}

    insert_start = time.perf_counter()
    insert_resp = stub.InsertBatch(
        sochdb_pb2.InsertBatchRequest(
            index_name=args.index_name,
            ids=numeric_ids.tolist(),
            vectors=doc_embeddings.astype(np.float32).reshape(-1).tolist(),
        )
    )
    insert_elapsed = time.perf_counter() - insert_start
    if insert_resp.error:
        raise SystemExit(f"InsertBatch failed: {insert_resp.error}")

    query_timings: list[float] = []
    query_results: list[dict[str, Any]] = []

    for query_record, query_id, query_vec in zip(queries, query_ids, query_embeddings):
        query_start = time.perf_counter()
        search_resp = stub.Search(
            sochdb_pb2.SearchRequest(
                index_name=args.index_name,
                query=query_vec.astype(np.float32).tolist(),
                k=args.k,
                ef=64,
            )
        )
        query_elapsed = time.perf_counter() - query_start
        query_timings.append(query_elapsed)

        if search_resp.error:
            raise SystemExit(f"Search failed for query {query_id}: {search_resp.error}")

        ranked = []
        for result in search_resp.results:
            doc_id = numeric_to_doc_id.get(int(result.id))
            if doc_id is None:
                continue
            ranked.append(
                {
                    "doc_id": doc_id,
                    "distance": float(result.distance),
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "system": "sochdb-2.0-grpc",
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "embedding_model": metadata["model_name"],
        "embedding_dimension": metadata["dimension"],
        "top_k": args.k,
        "storage": {
            "endpoint": f"{args.host}:{args.port}",
            "index_name": args.index_name,
            "dataset_dir": str(args.dataset_dir),
        },
        "build": {
            "index_create_time_ms": round(create_elapsed * 1000, 3),
            "insert_time_ms": round(insert_elapsed * 1000, 3),
            "indexed_vectors": int(insert_resp.inserted_count),
            "m": 16,
            "ef_construction": 100,
            "ef_search": 64,
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
