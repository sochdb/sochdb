#!/usr/bin/env python3
"""
Minimal local gRPC vector quickstart for SochDB 2.0.

This example assumes:
1. The local gRPC server is already running:
   cargo run -p sochdb-grpc --bin sochdb-grpc-server -- \
     --host 127.0.0.1 --port 50051 --metrics-port 0 --ws-port 0 --pg-port 0
2. Python stubs have been generated:
   cd sochdb-sdk && ./generate.sh python

Usage:
    python3 examples/python/07_grpc_vector_quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import grpc


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = REPO_ROOT / "sochdb-sdk" / "python" / "sochdb_sdk" / "generated"

if not GENERATED_DIR.exists():
    raise SystemExit(
        "Generated Python stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
    )

sys.path.insert(0, str(GENERATED_DIR))

import sochdb_pb2  # type: ignore  # noqa: E402
import sochdb_pb2_grpc  # type: ignore  # noqa: E402


def main() -> None:
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = sochdb_pb2_grpc.VectorIndexServiceStub(channel)

    index_name = "quickstart_local_index"

    create_resp = stub.CreateIndex(
        sochdb_pb2.CreateIndexRequest(
            name=index_name,
            dimension=4,
            metric=sochdb_pb2.DISTANCE_METRIC_COSINE,
            config=sochdb_pb2.HnswConfig(
                max_connections=16,
                ef_construction=100,
                ef_search=64,
            ),
        )
    )
    print("create_index:", create_resp.success, create_resp.error or "ok")

    insert_resp = stub.InsertBatch(
        sochdb_pb2.InsertBatchRequest(
            index_name=index_name,
            ids=[1, 2, 3],
            vectors=[
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        )
    )
    print("insert_batch:", insert_resp.inserted_count, insert_resp.error or "ok")

    search_resp = stub.Search(
        sochdb_pb2.SearchRequest(
            index_name=index_name,
            query=[1.0, 0.0, 0.0, 0.0],
            k=2,
            ef=64,
        )
    )
    print("search:", search_resp.error or "ok")
    for result in search_resp.results:
        print(f"  id={result.id} distance={result.distance:.4f}")


if __name__ == "__main__":
    main()
