#!/usr/bin/env python3
"""
Run the retrieval benchmark with SochDB's native 3-lane hybrid engine.

This harness exercises the real `ThreeLaneHybridIndex` (grep + BM25 + HNSW
fused with RRF inside Rust), as opposed to the dense-only `run_sochdb.py`.

The vector lane is fed the same embeddings as the dense harness; the BM25 and
optional grep lanes are fed the corpus text (title + body). All three lanes
share one AllowedSet and are fused with reciprocal-rank fusion in native code.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from sochdb import ThreeLaneHybridIndex


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "sochdb_fusion.json"
DEFAULT_DATASET_DIR = ROOT
DEFAULT_EMBEDDING_DIR = RESULTS_DIR

# Tokens shorter than this (after normalization) are dropped when deriving a
# grep pattern from the query, to avoid noisy stop-word style matches.
_GREP_MIN_TOKEN_LEN = 4
_GREP_MAX_TERMS = 6


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


def doc_text(record: dict[str, Any]) -> str:
    title = (record.get("title") or "").strip()
    body = (record.get("body") or record.get("text") or "").strip()
    if title and body:
        return f"{title} {body}"
    return title or body


def grep_pattern_for(query: str) -> str | None:
    """Derive a safe alternation grep pattern from salient query terms."""
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", query.lower()) if len(t) >= _GREP_MIN_TOKEN_LEN]
    if not tokens:
        return None
    # Keep the longest few terms (more discriminative), preserve order-free uniqueness.
    seen: set[str] = set()
    unique: list[str] = []
    for tok in sorted(tokens, key=len, reverse=True):
        if tok not in seen:
            seen.add(tok)
            unique.append(tok)
    selected = unique[:_GREP_MAX_TERMS]
    if not selected:
        return None
    # Natural-language prose is mixed-case ("Biomaterials" at sentence starts,
    # title case, ALL-CAPS headings); a case-sensitive lowercase pattern would
    # miss those occurrences entirely. Use an inline `(?i-u)` flag so the lane
    # matches case-insensitively with ASCII-only folding (a far smaller DFA than
    # Unicode case folding — much faster on English prose). The engine strips
    # this leading flag group for literal extraction, so the alternation still
    # drives the trigram index + IDF rather than degrading to a full scan.
    return "(?i-u)" + "|".join(re.escape(tok) for tok in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR,
                        help="Directory containing doc/query embedding outputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="Result JSON path")
    parser.add_argument("--k", type=int, default=5, help="Top-k results per query")
    parser.add_argument("--m", type=int, default=16, help="HNSW max connections per node")
    parser.add_argument("--ef-construction", type=int, default=100, help="HNSW construction depth")
    parser.add_argument("--ef-search", type=int, default=128, help="HNSW query-time search depth")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                        help="Directory containing corpus.jsonl and queries.jsonl")
    parser.add_argument("--method", default="rrf", choices=["rrf", "linear", "max"],
                        help="Fusion method")
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--grep-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--grep", action="store_true",
                        help="Enable the grep lane (pattern derived from query terms)")
    parser.add_argument("--grep-mode", default="rank", choices=["rank", "gate"],
                        help="rank: grep contributes a ranked lane; gate: grep narrows the AllowedSet")
    args = parser.parse_args()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Align corpus text to embedding (doc_ids) order.
    id_to_record = {record["id"]: record for record in corpus}
    texts = [doc_text(id_to_record[doc_id]) for doc_id in doc_ids]

    dimension = int(doc_embeddings.shape[1])
    index = ThreeLaneHybridIndex(
        dimension=dimension,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        metric="cosine",
    )

    print(f"Building 3-lane hybrid index for {len(doc_ids)} documents...")
    build_start = time.perf_counter()
    indexed = index.build(doc_ids, texts, np.ascontiguousarray(doc_embeddings.astype(np.float32)))
    build_elapsed = time.perf_counter() - build_start

    query_timings: list[float] = []
    query_results: list[dict[str, Any]] = []

    print(f"Running {len(queries)} queries (method={args.method}, grep={'on' if args.grep else 'off'})...")
    for query_record, query_id, query_vec in zip(queries, query_ids, query_embeddings):
        query_text = query_record["query"]
        pattern = grep_pattern_for(query_text) if args.grep else None

        query_start = time.perf_counter()
        ranked = index.search(
            query_vec.astype(np.float32),
            query_text,
            k=args.k,
            grep_pattern=pattern,
            grep_mode=args.grep_mode,
            method=args.method,
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            grep_weight=args.grep_weight,
            rrf_k=args.rrf_k,
        )
        query_elapsed = time.perf_counter() - query_start
        query_timings.append(query_elapsed)

        results = [
            {
                "doc_id": doc_id,
                "score": float(score),
                "title": id_to_record.get(doc_id, {}).get("title"),
            }
            for doc_id, score in ranked
        ]

        query_results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "relevant_ids": query_record["relevant_ids"],
                "results": results,
                "latency_ms": round(query_elapsed * 1000, 3),
            }
        )

    output = {
        "system": "sochdb_fusion",
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "embedding_model": metadata["model_name"],
        "embedding_dimension": metadata["dimension"],
        "top_k": args.k,
        "fusion": {
            "method": args.method,
            "grep": args.grep,
            "grep_mode": args.grep_mode,
            "vector_weight": args.vector_weight,
            "bm25_weight": args.bm25_weight,
            "grep_weight": args.grep_weight,
            "rrf_k": args.rrf_k,
        },
        "build": {
            "indexed_documents": int(indexed),
            "index_build_time_ms": round(build_elapsed * 1000, 3),
            "m": args.m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
        },
        "query_latency": {
            "p50_ms": round(percentile(query_timings, 50) * 1000, 3),
            "p95_ms": round(percentile(query_timings, 95) * 1000, 3),
            "mean_ms": round(float(np.mean(query_timings)) * 1000, 3),
        },
        "queries": query_results,
    }

    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved fusion benchmark results to {args.output}")


if __name__ == "__main__":
    main()
