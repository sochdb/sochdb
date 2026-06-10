#!/usr/bin/env python3
"""
Run LoComo / LoComo-Plus retrieval evaluation with SochDB HNSW.

Protocols:
- locomo: session-level retrieval on LoComo10 (excludes adversarial category 5)
- cognitive: cue-dialogue retrieval given trigger queries (LoComo-Plus sixth category)
- all: both subsets

Reports recall_any@5/10/20, nDCG@10, MRR, and latency per subset and category.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sochdb import HnswIndex, HybridSearchIndex


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    os.environ.get("LOCOMO_PLUS_DATA", "/Users/sushanth/git-clone/Locomo-Plus/data")
)
DEFAULT_CACHE_DIR = ROOT / "results" / "embedding_cache"
DEFAULT_OUTPUT = ROOT / "results" / "sochdb_locomo_plus.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "common-sense",
    4: "single-hop",
    5: "adversarial",
}
SKIP_CATEGORIES = {5}


@dataclass
class SessionChunk:
    key: str
    doc_id: str
    vector_text: str
    bm25_text: str


@dataclass
class QueryCase:
    query_id: str
    query_text: str
    gold_doc_ids: list[str]
    category: str
    group_id: str


def recall_any(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    top_k = set(retrieved_doc_ids[:k])
    return 1.0 if any(gold_id in top_k for gold_id in gold_doc_ids) else 0.0


def dcg(relevances: list[bool], k: int) -> float:
    score = 0.0
    for idx, relevant in enumerate(relevances[:k]):
        if relevant:
            score += 1.0 / math.log2(idx + 2)
    return score


def ndcg(retrieved_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    relevances = [doc_id in gold_doc_ids for doc_id in retrieved_doc_ids[:k]]
    ideal_relevances = [True] * min(k, len(gold_doc_ids))
    ideal = dcg(ideal_relevances, k)
    if ideal == 0.0:
        return 0.0
    return dcg(relevances, k) / ideal


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: set[str]) -> float:
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / idx
    return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def session_keys(conv: dict[str, Any]) -> list[str]:
    return sorted(
        k
        for k in conv
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(conv[k], list)
    )


def session_to_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"{turn.get('speaker', '?')}: {turn.get('text', '')}" for turn in turns)


def evidence_to_gold_sessions(evidence: list[str]) -> set[str]:
    gold: set[str] = set()
    for evid in evidence:
        for part in str(evid).split(";"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            session_token = part.split(":")[0].replace("D", "").strip()
            if not session_token.isdigit():
                continue
            gold.add(f"session_{int(session_token)}")
    return gold


def load_locomo_cases(locomo_path: Path) -> tuple[list[SessionChunk], list[QueryCase]]:
    raw = json.loads(locomo_path.read_text(encoding="utf-8"))
    chunks: list[SessionChunk] = []
    queries: list[QueryCase] = []

    for item in raw:
        sample_id = item["sample_id"]
        conv = item["conversation"]
        for sk in session_keys(conv):
            turns = conv.get(sk, [])
            if not turns:
                continue
            doc_id = f"{sample_id}_{sk}"
            text = session_to_text(turns)
            chunks.append(
                SessionChunk(
                    key=f"{sample_id}::{doc_id}",
                    doc_id=doc_id,
                    vector_text=text[:512],
                    bm25_text=text,
                )
            )

        for qi, qa in enumerate(item.get("qa", [])):
            cat = qa.get("category")
            if cat in SKIP_CATEGORIES:
                continue
            evidence = qa.get("evidence") or []
            gold_sessions = evidence_to_gold_sessions(evidence)
            gold_doc_ids = [f"{sample_id}_{sk}" for sk in sorted(gold_sessions)]
            queries.append(
                QueryCase(
                    query_id=f"{sample_id}_q{qi}",
                    query_text=qa.get("question", ""),
                    gold_doc_ids=gold_doc_ids,
                    category=LOCOMO_CATEGORY_NAMES.get(cat, f"category_{cat}"),
                    group_id=sample_id,
                )
            )

    return chunks, queries


def _import_build_conv(data_dir: Path):
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    from build_conv import map_speaker, parse_ab_dialogue

    return map_speaker, parse_ab_dialogue


def cue_to_text(cue_dialogue: str, conv: dict[str, Any], map_speaker, parse_ab_dialogue) -> str:
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")
    turns = map_speaker(parse_ab_dialogue(cue_dialogue or ""), speaker_a, speaker_b)
    return "\n".join(f"{t['speaker']}: {t['text'].strip()}" for t in turns if t.get("text"))


def trigger_to_text(trigger_query: str, conv: dict[str, Any], map_speaker, parse_ab_dialogue) -> str:
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")
    turns = map_speaker(parse_ab_dialogue(trigger_query or ""), speaker_a, speaker_b)
    return "\n".join(f"{t['speaker']}: {t['text'].strip()}" for t in turns if t.get("text"))


def load_cognitive_cases(
    locomo_plus_path: Path,
    locomo_path: Path,
) -> tuple[list[SessionChunk], list[QueryCase]]:
    plus_list = json.loads(locomo_plus_path.read_text(encoding="utf-8"))
    locomo_list = json.loads(locomo_path.read_text(encoding="utf-8"))
    map_speaker, parse_ab_dialogue = _import_build_conv(locomo_path.parent)

    chunks: list[SessionChunk] = []
    queries: list[QueryCase] = []

    for i, plus in enumerate(plus_list):
        locomo_item = locomo_list[i % len(locomo_list)]
        sample_id = locomo_item["sample_id"]
        conv = locomo_item["conversation"]
        group_id = f"cognitive_{i}"

        for sk in session_keys(conv):
            turns = conv.get(sk, [])
            if not turns:
                continue
            doc_id = f"{group_id}_{sk}"
            text = session_to_text(turns)
            chunks.append(
                SessionChunk(
                    key=f"{group_id}::{doc_id}",
                    doc_id=doc_id,
                    vector_text=text[:512],
                    bm25_text=text,
                )
            )

        cue_doc_id = f"{group_id}_cue"
        cue_text = cue_to_text(plus.get("cue_dialogue", ""), conv, map_speaker, parse_ab_dialogue)
        chunks.append(
            SessionChunk(
                key=f"{group_id}::{cue_doc_id}",
                doc_id=cue_doc_id,
                vector_text=cue_text[:512],
                bm25_text=cue_text,
            )
        )

        queries.append(
            QueryCase(
                query_id=f"cognitive_{i}",
                query_text=trigger_to_text(
                    plus.get("trigger_query", ""),
                    conv,
                    map_speaker,
                    parse_ab_dialogue,
                ),
                gold_doc_ids=[cue_doc_id],
                category="Cognitive",
                group_id=group_id,
            )
        )

    return chunks, queries


def encode_or_load(
    *,
    cache_dir: Path,
    cache_tag: str,
    chunks: list[SessionChunk],
    queries: list[QueryCase],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_vectors_path = cache_dir / f"{cache_tag}_doc_embeddings.npy"
    query_vectors_path = cache_dir / f"{cache_tag}_query_embeddings.npy"
    doc_keys_path = cache_dir / f"{cache_tag}_doc_keys.json"
    query_keys_path = cache_dir / f"{cache_tag}_query_ids.json"

    expected_doc_keys = [chunk.key for chunk in chunks]
    expected_query_ids = [query.query_id for query in queries]
    if (
        doc_vectors_path.exists()
        and query_vectors_path.exists()
        and doc_keys_path.exists()
        and query_keys_path.exists()
        and json.loads(doc_keys_path.read_text(encoding="utf-8")) == expected_doc_keys
        and json.loads(query_keys_path.read_text(encoding="utf-8")) == expected_query_ids
    ):
        return np.load(doc_vectors_path), np.load(query_vectors_path)

    print(f"Loading embedding model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(chunks)} documents...")
    doc_embeddings = model.encode(
        [chunk.vector_text for chunk in chunks],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    print(f"Embedding {len(queries)} queries...")
    query_embeddings = model.encode(
        [query.query_text for query in queries],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(doc_vectors_path, doc_embeddings)
    np.save(query_vectors_path, query_embeddings)
    doc_keys_path.write_text(json.dumps(expected_doc_keys), encoding="utf-8")
    query_keys_path.write_text(json.dumps(expected_query_ids), encoding="utf-8")
    return doc_embeddings, query_embeddings


def run_grouped_benchmark(
    *,
    chunks: list[SessionChunk],
    queries: list[QueryCase],
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    m: int,
    ef_construction: int,
    mode: str,
    subset_name: str,
) -> dict[str, Any]:
    chunk_by_key = {chunk.key: (idx, chunk) for idx, chunk in enumerate(chunks)}
    queries_by_group: dict[str, list[tuple[int, QueryCase]]] = {}
    for qidx, query in enumerate(queries):
        queries_by_group.setdefault(query.group_id, []).append((qidx, query))

    query_latencies_ms: list[float] = []
    build_latencies_ms: list[float] = []
    per_question: list[dict[str, Any]] = []

    group_items = list(queries_by_group.items())
    for gidx, (group_id, group_queries) in enumerate(group_items):
        group_chunk_keys = sorted(
            {chunk.key for chunk in chunks if chunk.key.startswith(f"{group_id}::")}
        )
        group_chunks = [chunk_by_key[key][1] for key in group_chunk_keys]
        group_indices = [chunk_by_key[key][0] for key in group_chunk_keys]
        vectors = np.ascontiguousarray(doc_embeddings[group_indices], dtype=np.float32)

        ids = np.arange(1, len(group_chunks) + 1, dtype=np.uint64)
        id_to_doc = {int(numeric_id): chunk.doc_id for numeric_id, chunk in zip(ids.tolist(), group_chunks)}

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
                [chunk.doc_id for chunk in group_chunks],
                [chunk.bm25_text for chunk in group_chunks],
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

        for qidx, query in group_queries:
            query_start = time.perf_counter()
            if mode == "hybrid":
                hits = index.search(
                    query.query_text,
                    np.ascontiguousarray(query_embeddings[qidx], dtype=np.float32),
                    k=k,
                    candidate_k=k * 2,
                )
                retrieved_doc_ids = [hit.doc_id for hit in hits]
                distances: list[float] = []
            else:
                result_ids, distances_arr = index.search(
                    np.ascontiguousarray(query_embeddings[qidx], dtype=np.float32),
                    k=k,
                )
                retrieved_doc_ids = [
                    id_to_doc[int(result_id)]
                    for result_id in result_ids.tolist()
                    if int(result_id) in id_to_doc
                ]
                distances = [float(distance) for distance in distances_arr.tolist()[:10]]

            query_elapsed_ms = (time.perf_counter() - query_start) * 1000.0
            query_latencies_ms.append(query_elapsed_ms)
            gold = set(query.gold_doc_ids)
            per_question.append(
                {
                    "query_id": query.query_id,
                    "category": query.category,
                    "group_id": query.group_id,
                    "recall_any_at_5": recall_any(retrieved_doc_ids, query.gold_doc_ids, 5),
                    "recall_any_at_10": recall_any(retrieved_doc_ids, query.gold_doc_ids, 10),
                    "recall_any_at_20": recall_any(retrieved_doc_ids, query.gold_doc_ids, 20),
                    "ndcg_at_10": ndcg(retrieved_doc_ids, gold, 10),
                    "mrr": reciprocal_rank(retrieved_doc_ids, gold),
                    "retrieved_doc_ids": retrieved_doc_ids[:10],
                    "gold_doc_ids": query.gold_doc_ids,
                    "query_latency_ms": round(query_elapsed_ms, 4),
                    "build_latency_ms": round(build_elapsed_ms, 4),
                    "distances": distances,
                }
            )

        if (gidx + 1) % 10 == 0 or (gidx + 1) == len(group_items):
            running = sum(row["recall_any_at_5"] for row in per_question) / len(per_question)
            print(
                f"  [{subset_name}] groups {gidx + 1}/{len(group_items)} "
                f"running recall_any@5: {running * 100:.1f}%"
            )

    def avg(metric: str) -> float:
        return sum(float(row[metric]) for row in per_question) / len(per_question)

    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in per_question:
        by_type.setdefault(row["category"], []).append(row)

    return {
        "subset": subset_name,
        "system": "sochdb_hnsw_vector" if mode == "vector" else "sochdb_hnsw_bm25_hybrid",
        "mode": mode,
        "questions": len(per_question),
        "groups": len(group_items),
        "documents": len(chunks),
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
                "ndcg_at_10": sum(float(r["ndcg_at_10"]) for r in rows) / len(rows),
                "mrr": sum(float(r["mrr"]) for r in rows) / len(rows),
            }
            for question_type, rows in by_type.items()
        },
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing locomo10.json and locomo_plus.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    parser.add_argument(
        "--subset",
        choices=["locomo", "cognitive", "all"],
        default="all",
        help="Which benchmark subset to run",
    )
    args = parser.parse_args()

    locomo_path = args.data_dir / "locomo10.json"
    locomo_plus_path = args.data_dir / "locomo_plus.json"
    if not locomo_path.exists():
        raise FileNotFoundError(f"Missing dataset: {locomo_path}")

    results: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "mode": args.mode,
        "subsets": {},
    }

    if args.subset in ("locomo", "all"):
        locomo_chunks, locomo_queries = load_locomo_cases(locomo_path)
        print(f"[locomo] {len(locomo_queries)} queries, {len(locomo_chunks)} session documents")
        doc_embeddings, query_embeddings = encode_or_load(
            cache_dir=args.cache_dir,
            cache_tag="locomo",
            chunks=locomo_chunks,
            queries=locomo_queries,
            batch_size=args.batch_size,
        )
        results["subsets"]["locomo"] = run_grouped_benchmark(
            chunks=locomo_chunks,
            queries=locomo_queries,
            doc_embeddings=doc_embeddings,
            query_embeddings=query_embeddings,
            k=args.k,
            m=args.m,
            ef_construction=args.ef_construction,
            mode=args.mode,
            subset_name="locomo",
        )

    if args.subset in ("cognitive", "all"):
        if not locomo_plus_path.exists():
            raise FileNotFoundError(f"Missing dataset: {locomo_plus_path}")
        cognitive_chunks, cognitive_queries = load_cognitive_cases(locomo_plus_path, locomo_path)
        print(
            f"[cognitive] {len(cognitive_queries)} queries, "
            f"{len(cognitive_chunks)} documents (sessions + cue)"
        )
        doc_embeddings, query_embeddings = encode_or_load(
            cache_dir=args.cache_dir,
            cache_tag="cognitive",
            chunks=cognitive_chunks,
            queries=cognitive_queries,
            batch_size=args.batch_size,
        )
        results["subsets"]["cognitive"] = run_grouped_benchmark(
            chunks=cognitive_chunks,
            queries=cognitive_queries,
            doc_embeddings=doc_embeddings,
            query_embeddings=query_embeddings,
            k=args.k,
            m=args.m,
            ef_construction=args.ef_construction,
            mode=args.mode,
            subset_name="cognitive",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n=== SochDB LoComo-Plus Results ({args.mode}) ===")
    for subset_name, subset in results["subsets"].items():
        print(f"\n[{subset_name}]")
        print(f"  Questions:       {subset['questions']}")
        print(f"  recall_any@5:    {subset['recall_any_at_5'] * 100:.1f}%")
        print(f"  recall_any@10:   {subset['recall_any_at_10'] * 100:.1f}%")
        print(f"  recall_any@20:   {subset['recall_any_at_20'] * 100:.1f}%")
        print(f"  NDCG@10:         {subset['ndcg_at_10'] * 100:.1f}%")
        print(f"  MRR:             {subset['mrr'] * 100:.1f}%")
        print(
            "  Query latency:   "
            f"p50={subset['latency']['query_p50_ms']:.4f}ms, "
            f"p95={subset['latency']['query_p95_ms']:.4f}ms"
        )
        print("  Per category:")
        for cat, stats in sorted(subset["per_type"].items()):
            print(
                f"    {cat:12s} n={stats['count']:4d}  "
                f"recall@5={stats['recall_any_at_5'] * 100:5.1f}%  "
                f"recall@10={stats['recall_any_at_10'] * 100:5.1f}%  "
                f"mrr={stats['mrr'] * 100:5.1f}%"
            )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()