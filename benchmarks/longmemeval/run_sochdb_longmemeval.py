#!/usr/bin/env python3
"""
Run LongMemEval-S retrieval evaluation with SochDB HNSW.

This mirrors agentmemory's retrieval-only LongMemEval protocol:
- load longmemeval_s_cleaned.json
- exclude abstention question types
- build one haystack index per question
- score whether any gold answer session appears in top-k

The benchmark reports recall_any@5/10/20, nDCG@10, MRR, and latency.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from sochdb import HnswIndex, HybridSearchIndex


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "longmemeval_s_cleaned.json"
DEFAULT_CACHE_DIR = ROOT / "results" / "embedding_cache"
DEFAULT_OUTPUT = ROOT / "results" / "sochdb_longmemeval_vector.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ABSTENTION_TYPES = {
    "single-session-user_abs",
    "multi-session_abs",
    "knowledge-update_abs",
    "temporal-reasoning_abs",
}


@dataclass
class SessionChunk:
    key: str
    session_id: str
    vector_text: str
    bm25_text: str


def chunk_session_to_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


def recall_any(retrieved_session_ids: list[str], gold_session_ids: list[str], k: int) -> float:
    top_k = set(retrieved_session_ids[:k])
    return 1.0 if any(gold_id in top_k for gold_id in gold_session_ids) else 0.0


def dcg(relevances: list[bool], k: int) -> float:
    score = 0.0
    for idx, relevant in enumerate(relevances[:k]):
        if relevant:
            score += 1.0 / math.log2(idx + 2)
    return score


def ndcg(retrieved_session_ids: list[str], gold_session_ids: set[str], k: int) -> float:
    relevances = [session_id in gold_session_ids for session_id in retrieved_session_ids[:k]]
    ideal_relevances = [True] * min(k, len(gold_session_ids))
    ideal = dcg(ideal_relevances, k)
    if ideal == 0.0:
        return 0.0
    return dcg(relevances, k) / ideal


def reciprocal_rank(retrieved_session_ids: list[str], gold_session_ids: set[str]) -> float:
    for idx, session_id in enumerate(retrieved_session_ids, start=1):
        if session_id in gold_session_ids:
            return 1.0 / idx
    return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def load_entries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [entry for entry in raw if entry.get("question_type") not in ABSTENTION_TYPES]


def collect_texts(entries: list[dict[str, Any]]) -> tuple[list[SessionChunk], list[str], list[str]]:
    chunks: list[SessionChunk] = []
    query_ids: list[str] = []
    query_texts: list[str] = []

    for entry in entries:
        query_ids.append(entry["question_id"])
        query_texts.append(entry["question"])
        for session_id, turns in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            chunks.append(
                SessionChunk(
                    key=f"{entry['question_id']}::{session_id}",
                    session_id=session_id,
                    vector_text=chunk_session_to_text(turns)[:512],
                    bm25_text=chunk_session_to_text(turns),
                )
            )

    return chunks, query_ids, query_texts


def encode_or_load(
    *,
    cache_dir: Path,
    chunks: list[SessionChunk],
    query_ids: list[str],
    query_texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_vectors_path = cache_dir / "doc_embeddings.npy"
    query_vectors_path = cache_dir / "query_embeddings.npy"
    doc_keys_path = cache_dir / "doc_keys.json"
    query_keys_path = cache_dir / "query_ids.json"

    expected_doc_keys = [chunk.key for chunk in chunks]
    if (
        doc_vectors_path.exists()
        and query_vectors_path.exists()
        and doc_keys_path.exists()
        and query_keys_path.exists()
        and json.loads(doc_keys_path.read_text(encoding="utf-8")) == expected_doc_keys
        and json.loads(query_keys_path.read_text(encoding="utf-8")) == query_ids
    ):
        return np.load(doc_vectors_path), np.load(query_vectors_path)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(chunks)} haystack sessions...")
    doc_embeddings = model.encode(
        [chunk.vector_text for chunk in chunks],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    print(f"Embedding {len(query_texts)} questions...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(doc_vectors_path, doc_embeddings)
    np.save(query_vectors_path, query_embeddings)
    doc_keys_path.write_text(json.dumps(expected_doc_keys), encoding="utf-8")
    query_keys_path.write_text(json.dumps(query_ids), encoding="utf-8")
    return doc_embeddings, query_embeddings


def run_vector_benchmark(
    *,
    entries: list[dict[str, Any]],
    chunks: list[SessionChunk],
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    m: int,
    ef_construction: int,
    mode: str,
) -> dict[str, Any]:
    doc_offset = 0
    query_latencies_ms: list[float] = []
    build_latencies_ms: list[float] = []
    per_question: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        session_count = len(entry["haystack_session_ids"])
        question_chunks = chunks[doc_offset : doc_offset + session_count]
        vectors = np.ascontiguousarray(
            doc_embeddings[doc_offset : doc_offset + session_count],
            dtype=np.float32,
        )
        doc_offset += session_count

        ids = np.arange(1, session_count + 1, dtype=np.uint64)
        id_to_session = {
            int(numeric_id): chunk.session_id
            for numeric_id, chunk in zip(ids.tolist(), question_chunks)
        }

        build_start = time.perf_counter()
        if mode == "hybrid":
            index = HybridSearchIndex(
                dimension=int(vectors.shape[1]),
                m=m,
                ef_construction=ef_construction,
                metric="cosine",
                bm25_weight=0.4,
                vector_weight=0.6,
            ).build(
                [chunk.session_id for chunk in question_chunks],
                [chunk.bm25_text for chunk in question_chunks],
                vectors,
            )
        else:
            index = HnswIndex(
                dimension=int(vectors.shape[1]),
                m=m,
                ef_construction=ef_construction,
                metric="cosine",
            )
            index.insert_batch_with_ids(ids, vectors)
        build_elapsed_ms = (time.perf_counter() - build_start) * 1000.0
        build_latencies_ms.append(build_elapsed_ms)

        query_start = time.perf_counter()
        if mode == "hybrid":
            hits = index.search(
                entry["question"],
                np.ascontiguousarray(query_embeddings[idx], dtype=np.float32),
                k=k,
                candidate_k=k * 2,
            )
            retrieved_session_ids = [hit.doc_id for hit in hits]
            distances: list[float] = []
        else:
            result_ids, distances_arr = index.search(
                np.ascontiguousarray(query_embeddings[idx], dtype=np.float32),
                k=k,
            )
            retrieved_session_ids = [
                id_to_session[int(result_id)]
                for result_id in result_ids.tolist()
                if int(result_id) in id_to_session
            ]
            distances = [float(distance) for distance in distances_arr.tolist()[:10]]

        query_elapsed_ms = (time.perf_counter() - query_start) * 1000.0
        query_latencies_ms.append(query_elapsed_ms)
        gold = set(entry["answer_session_ids"])
        per_question.append(
            {
                "question_id": entry["question_id"],
                "question_type": entry["question_type"],
                "recall_any_at_5": recall_any(retrieved_session_ids, entry["answer_session_ids"], 5),
                "recall_any_at_10": recall_any(retrieved_session_ids, entry["answer_session_ids"], 10),
                "recall_any_at_20": recall_any(retrieved_session_ids, entry["answer_session_ids"], 20),
                "ndcg_at_10": ndcg(retrieved_session_ids, gold, 10),
                "mrr": reciprocal_rank(retrieved_session_ids, gold),
                "retrieved_session_ids": retrieved_session_ids[:10],
                "gold_session_ids": entry["answer_session_ids"],
                "query_latency_ms": round(query_elapsed_ms, 4),
                "build_latency_ms": round(build_elapsed_ms, 4),
                "distances": distances,
            }
        )

        if (idx + 1) % 50 == 0:
            running = sum(row["recall_any_at_5"] for row in per_question) / len(per_question)
            print(f"  [{idx + 1}/{len(entries)}] running recall_any@5: {running * 100:.1f}%")

    def avg(metric: str) -> float:
        return sum(float(row[metric]) for row in per_question) / len(per_question)

    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in per_question:
        by_type.setdefault(row["question_type"], []).append(row)

    return {
        "system": "sochdb_hnsw_vector" if mode == "vector" else "sochdb_hnsw_bm25_hybrid",
        "mode": mode,
        "questions": len(per_question),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(doc_embeddings.shape[1]),
        "hnsw": {
            "m": m,
            "ef_construction": ef_construction,
            "metric": "cosine",
            "top_k": k,
        },
        "recall_any_at_5": avg("recall_any_at_5"),
        "recall_any_at_10": avg("recall_any_at_10"),
        "recall_any_at_20": avg("recall_any_at_20"),
        "ndcg_at_10": avg("ndcg_at_10"),
        "mrr": avg("mrr"),
        "latency": {
            "query_p50_ms": round(percentile(query_latencies_ms, 50), 4),
            "query_p95_ms": round(percentile(query_latencies_ms, 95), 4),
            "query_mean_ms": round(float(np.mean(query_latencies_ms)), 4),
            "build_p50_ms": round(percentile(build_latencies_ms, 50), 4),
            "build_p95_ms": round(percentile(build_latencies_ms, 95), 4),
            "build_mean_ms": round(float(np.mean(build_latencies_ms)), 4),
        },
        "per_type": {
            question_type: {
                "count": len(rows),
                "recall_any_at_5": sum(float(r["recall_any_at_5"]) for r in rows) / len(rows),
                "recall_any_at_10": sum(float(r["recall_any_at_10"]) for r in rows) / len(rows),
            }
            for question_type, rows in by_type.items()
        },
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    args = parser.parse_args()

    entries = load_entries(args.dataset)
    chunks, query_ids, query_texts = collect_texts(entries)
    print(f"Loaded {len(entries)} questions and {len(chunks)} haystack sessions")

    doc_embeddings, query_embeddings = encode_or_load(
        cache_dir=args.cache_dir,
        chunks=chunks,
        query_ids=query_ids,
        query_texts=query_texts,
        batch_size=args.batch_size,
    )

    result = run_vector_benchmark(
        entries=entries,
        chunks=chunks,
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        k=args.k,
        m=args.m,
        ef_construction=args.ef_construction,
        mode=args.mode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== SochDB LongMemEval-S Results ({args.mode}) ===")
    print(f"Questions:       {result['questions']}")
    print(f"recall_any@5:    {result['recall_any_at_5'] * 100:.1f}%")
    print(f"recall_any@10:   {result['recall_any_at_10'] * 100:.1f}%")
    print(f"recall_any@20:   {result['recall_any_at_20'] * 100:.1f}%")
    print(f"NDCG@10:         {result['ndcg_at_10'] * 100:.1f}%")
    print(f"MRR:             {result['mrr'] * 100:.1f}%")
    print(
        "Query latency:   "
        f"p50={result['latency']['query_p50_ms']:.4f}ms, "
        f"p95={result['latency']['query_p95_ms']:.4f}ms, "
        f"mean={result['latency']['query_mean_ms']:.4f}ms"
    )
    print(f"Saved:           {args.output}")


if __name__ == "__main__":
    main()
