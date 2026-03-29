#!/usr/bin/env python3
"""
SochDB Local Knowledge Search Demo
=================================

This example shows a local-only retrieval workflow for the first ICP:
Python-first AI engineers building internal knowledge search or lightweight
RAG systems without external APIs.

What it demonstrates:
1. Store documents in SochDB
2. Build a local vector index with deterministic embeddings
3. Query the index and fetch full document payloads from the database

Usage:
    python3 examples/python/07_local_knowledge_search.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Iterable

import numpy as np

from sochdb import Database, HnswIndex


DB_PATH = "./knowledge_demo_db"
DIMENSION = 64


DOCUMENTS = [
    {
        "id": 101,
        "title": "Laptop VPN Setup",
        "body": "To access internal dashboards, install the company VPN client and connect before opening private services.",
        "tags": ["it", "security", "access"],
    },
    {
        "id": 102,
        "title": "Expense Reimbursement Policy",
        "body": "Employees should submit travel and meal receipts within 30 days using the finance portal reimbursement form.",
        "tags": ["finance", "policy", "travel"],
    },
    {
        "id": 103,
        "title": "On-Call Incident Process",
        "body": "If production is degraded, page the on-call engineer, open an incident channel, and post updates every 15 minutes.",
        "tags": ["sre", "incident", "operations"],
    },
    {
        "id": 104,
        "title": "Customer Support Escalation",
        "body": "Urgent customer issues should be escalated to tier two support with logs, screenshots, and account identifiers attached.",
        "tags": ["support", "customers", "triage"],
    },
    {
        "id": 105,
        "title": "Access Review Checklist",
        "body": "Managers must review employee access to internal tools quarterly and remove permissions that are no longer required.",
        "tags": ["security", "access", "compliance"],
    },
]


def tokenize(text: str) -> Iterable[str]:
    return (
        token.strip(".,:;!?()[]{}").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}")
    )


def embed_text(text: str, dimension: int = DIMENSION) -> np.ndarray:
    """Deterministic local embedding using hashed token buckets."""
    vec = np.zeros(dimension, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def main() -> None:
    print("=" * 60)
    print("  SochDB Local Knowledge Search")
    print("=" * 60)

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    db = Database.open(DB_PATH)
    index = HnswIndex(dimension=DIMENSION, m=16, ef_construction=100, metric="cosine")

    ids = np.array([doc["id"] for doc in DOCUMENTS], dtype=np.uint64)
    vectors = np.vstack([
        embed_text(f"{doc['title']} {doc['body']} {' '.join(doc['tags'])}")
        for doc in DOCUMENTS
    ]).astype(np.float32)

    print("\n[1] Storing documents in SochDB...")
    with db.transaction() as txn:
        for doc in DOCUMENTS:
            key = f"docs/{doc['id']}".encode("utf-8")
            db.put(key, json.dumps(doc).encode("utf-8"), txn.id)
    print(f"    Stored {len(DOCUMENTS)} documents")

    print("\n[2] Building local HNSW index...")
    inserted = index.insert_batch_with_ids(ids, vectors)
    print(f"    Indexed {inserted} document embeddings")

    query = "How do I access internal tools securely from my laptop?"
    print(f"\n[3] Query: {query}")
    query_vec = embed_text(query)
    result_ids, distances = index.search(query_vec, k=3)

    print("\n[4] Top matches")
    for rank, (doc_id, score) in enumerate(zip(result_ids.tolist(), distances.tolist()), start=1):
        payload = db.get(f"docs/{doc_id}".encode("utf-8"))
        if payload is None:
            continue
        doc = json.loads(payload.decode("utf-8"))
        print(f"    {rank}. {doc['title']} (id={doc_id}, distance={score:.4f})")
        print(f"       {doc['body']}")

    db.close()

    print("\n" + "=" * 60)
    print("  ✅ Demo complete: local data + local retrieval in one workflow")
    print("=" * 60)


if __name__ == "__main__":
    main()
