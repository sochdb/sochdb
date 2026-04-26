#!/usr/bin/env python3
"""
Hosted SochDB + Studio end-to-end demo.

This script is meant to be run locally from the SochDB workspace. It writes a
few documents to a hosted SochDB remote instance and then sends a matching
event to the hosted Studio backend so the browser UI can show both the data
plane and the control-plane activity.

Example:
    export STUDIO_API_KEY=soch_sk_xxx
    python examples/python/hosted_studio_remote_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_GRPC_ADDRESS = "studio.agentslab.host:50053"
DEFAULT_STUDIO_BASE_URL = "http://studio.agentslab.host:3000"
DEFAULT_COLLECTION = "demo_docs"
DEFAULT_NAMESPACE = "default"
DEFAULT_DIMENSION = 4


def load_sdk() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    sdk_src = repo_root.parent / "sochdb-python-sdk" / "src"
    if sdk_src.exists():
        sys.path.insert(0, str(sdk_src))

    try:
        from sochdb.grpc_client import SochDBClient  # type: ignore
    except ImportError as exc:  # pragma: no cover - user environment issue
        raise SystemExit(
            "Failed to import SochDB Python SDK gRPC client.\n"
            f"Expected sibling SDK at: {sdk_src}\n"
            "Make sure this repo sits next to `sochdb-python-sdk` and that "
            "`grpcio` is installed in your Python environment."
        ) from exc

    return SochDBClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write demo documents to a hosted SochDB instance and send a matching Studio event."
    )
    parser.add_argument("--grpc-address", default=DEFAULT_GRPC_ADDRESS, help="Hosted SochDB gRPC address.")
    parser.add_argument(
        "--studio-base-url",
        default=DEFAULT_STUDIO_BASE_URL,
        help="Hosted Studio base URL, for example http://studio.agentslab.host:3000",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection to create/use.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Collection namespace.")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION, help="Embedding dimension.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Studio ingestion API key. If omitted, STUDIO_API_KEY is used.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip sending the matching event to Studio. Useful if you only want the remote DB write.",
    )
    return parser.parse_args()


def ensure_collection(client: Any, name: str, namespace: str, dimension: int) -> None:
    try:
        created = client.create_collection(name=name, dimension=dimension, namespace=namespace, metric="cosine")
        status = "created" if created else "already existed"
        print(f"Collection `{namespace}/{name}`: {status}")
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" in message or "exists" in message:
            print(f"Collection `{namespace}/{name}` already exists; continuing.")
            return
        raise


def build_documents(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{run_id}-doc-1",
            "content": "Hosted Studio demo document about SochDB remote ingestion.",
            "embedding": [0.10, 0.20, 0.30, 0.40],
            "metadata": {"source": "local-demo", "topic": "intro", "run_id": run_id},
        },
        {
            "id": f"{run_id}-doc-2",
            "content": "Hosted Studio demo document showing browser-visible remote writes.",
            "embedding": [0.20, 0.10, 0.40, 0.30],
            "metadata": {"source": "local-demo", "topic": "ui", "run_id": run_id},
        },
        {
            "id": f"{run_id}-doc-3",
            "content": "Hosted Studio demo document for search and observability testing.",
            "embedding": [0.90, 0.80, 0.10, 0.00],
            "metadata": {"source": "local-demo", "topic": "ops", "run_id": run_id},
        },
    ]


def ingest_event(studio_base_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    target = f"{studio_base_url.rstrip('/')}/api/studio/ingest/events"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        target,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:  # pragma: no cover - runtime env issue
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Studio ingest failed with HTTP {exc.code}: {details}") from exc


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.environ.get("STUDIO_API_KEY")

    SochDBClient = load_sdk()
    run_id = f"demo-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    documents = build_documents(run_id)

    print(f"Connecting to hosted SochDB at {args.grpc_address}")
    client = SochDBClient(args.grpc_address)
    try:
        ensure_collection(client, args.collection, args.namespace, args.dimension)
        inserted_ids = client.add_documents(args.collection, documents, namespace=args.namespace)
        print(f"Inserted {len(inserted_ids)} documents into `{args.namespace}/{args.collection}`")
        for doc_id in inserted_ids:
            print(f"  - {doc_id}")
    finally:
        client.close()

    if args.skip_ingest:
        print("Skipped Studio event ingestion.")
        print("Next: refresh the hosted Studio Workbench to see the collection counts update.")
        return 0

    if not api_key:
        raise SystemExit(
            "No Studio API key provided.\n"
            "Create one in Studio -> Settings -> Platform and set STUDIO_API_KEY, "
            "or pass --api-key explicitly."
        )

    event_payload = {
        "source": "local-hosted-demo",
        "events": [
            {
                "type": "remote-write",
                "name": "hosted-studio-e2e-demo",
                "status": "ok",
                "metadata": {
                    "run_id": run_id,
                    "collection": args.collection,
                    "namespace": args.namespace,
                    "document_count": len(documents),
                    "document_ids": inserted_ids,
                    "grpc_address": args.grpc_address,
                },
            }
        ],
    }
    response = ingest_event(args.studio_base_url, api_key, event_payload)
    print("Ingested matching Studio event:")
    print(json.dumps(response, indent=2))

    print("\nDemo complete.")
    print(f"1. Open {args.studio_base_url}")
    print("2. Connect to the hosted project/instance")
    print("3. Click `Refresh live state` in Workbench")
    print("4. Open Settings -> Platform to see the matching ingested event")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
