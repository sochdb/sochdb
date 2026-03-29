#!/usr/bin/env python3
"""
Evaluate retrieval benchmark outputs.

Metrics:
    - Recall@k
    - MRR
    - nDCG@k
    - latency summaries
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = [ROOT / "results" / "sochdb.json"]


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    score = 0.0
    for idx, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        if rel:
            score += rel / math.log2(idx + 1)
    return score


def ndcg_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    ideal_hits = min(len(relevant_ids), k)
    ideal_ids = list(relevant_ids)[:ideal_hits]
    ideal_dcg = dcg_at_k(relevant_ids, ideal_ids, ideal_hits)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevant_ids, ranked_ids, k) / ideal_dcg


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_run(payload: dict[str, Any], k: int) -> dict[str, Any]:
    queries = payload.get("queries", [])
    if not queries:
        raise ValueError("No queries found in result payload")

    recalls = []
    mrrs = []
    ndcgs = []

    for query in queries:
        relevant_ids = set(query.get("relevant_ids", []))
        ranked_ids = [result["doc_id"] for result in query.get("results", [])]

        recalls.append(recall_at_k(relevant_ids, ranked_ids, k))
        mrrs.append(reciprocal_rank(relevant_ids, ranked_ids, k))
        ndcgs.append(ndcg_at_k(relevant_ids, ranked_ids, k))

    latency = payload.get("query_latency", {})

    return {
        "system": payload.get("system", "unknown"),
        "query_count": len(queries),
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        "p50_ms": latency.get("p50_ms"),
        "p95_ms": latency.get("p95_ms"),
        "mean_ms": latency.get("mean_ms"),
    }


def print_table(rows: list[dict[str, Any]], k: int) -> None:
    headers = [
        "system",
        f"recall@{k}",
        "mrr",
        f"ndcg@{k}",
        "p50_ms",
        "p95_ms",
        "mean_ms",
    ]

    widths: dict[str, int] = {header: len(header) for header in headers}
    formatted_rows: list[dict[str, str]] = []

    for row in rows:
        formatted: dict[str, str] = {}
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                formatted_value = f"{value:.4f}" if "ms" not in header else f"{value:.3f}"
            elif value is None:
                formatted_value = "-"
            else:
                formatted_value = str(value)
            formatted[header] = formatted_value
            widths[header] = max(widths[header], len(formatted_value))
        formatted_rows.append(formatted)

    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    divider = "-+-".join("-" * widths[header] for header in headers)

    print(header_line)
    print(divider)
    for row in formatted_rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Result JSON files to evaluate",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k cutoff for Recall@k and nDCG@k",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save summary JSON",
    )
    args = parser.parse_args()

    summaries = []
    for result_path in args.results:
        payload = load_json(result_path)
        summary = evaluate_run(payload, args.k)
        summary["result_file"] = str(result_path)
        summaries.append(summary)

    print_table(summaries, args.k)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nSaved summary JSON to {args.output_json}")


if __name__ == "__main__":
    main()
