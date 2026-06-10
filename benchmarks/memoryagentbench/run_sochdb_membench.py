#!/usr/bin/env python3
"""Benchmark sochdb-memory lexical retrieval on MemoryAgentBench (Accurate Retrieval).

Reuses MemoryAgentBench's ConversationCreator + scoring verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MAB_ROOT = os.environ.get(
    "MEMORY_AGENT_BENCH_ROOT", "/Users/sushanth/git-clone/MemoryAgentBench"
)
sys.path.insert(0, MAB_ROOT)

from conversation_creator import ConversationCreator  # noqa: E402
from utils.eval_other_utils import (  # noqa: E402
    drqa_metric_max_over_ground_truths,
    substring_exact_match_score,
)

DEFAULT_TASKS = [
    {
        "dataset": "Accurate_Retrieval",
        "sub_dataset": "ruler_qa1_197K",
        "chunk_size": 4096,
        "context_max_length": 220000,
        "generation_max_length": 50,
    },
    {
        "dataset": "Accurate_Retrieval",
        "sub_dataset": "ruler_qa2_421K",
        "chunk_size": 4096,
        "context_max_length": 524288,
        "generation_max_length": 50,
    },
    {
        "dataset": "Accurate_Retrieval",
        "sub_dataset": "longmemeval_s*",
        "chunk_size": 4096,
        "context_max_length": 400000,
        "generation_max_length": 50,
    },
]


def build_configs(task, max_test_samples):
    agent_config = {"agent_name": "Simple_rag_bm25", "model": "gpt-4o-mini"}
    dataset_config = {
        "dataset": task["dataset"],
        "sub_dataset": task["sub_dataset"],
        "chunk_size": task["chunk_size"],
        "context_max_length": task["context_max_length"],
        "generation_max_length": task["generation_max_length"],
        "max_test_samples": max_test_samples,
        "seed": 42,
    }
    return agent_config, dataset_config


def run_retriever(retriever_bin: str, payload: dict, workdir: str) -> dict:
    os.makedirs(workdir, exist_ok=True)
    in_path = os.path.join(workdir, "retriever_input.json")
    with open(in_path, "w") as f:
        json.dump(payload, f)
    proc = subprocess.run([retriever_bin, in_path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"retriever failed (rc={proc.returncode}):\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def evaluate_task(task: dict, args) -> tuple[dict, list]:
    print(
        f"\n=== Task: {task['sub_dataset']} "
        f"(k={args.k_values}, max_samples={args.max_test_samples}) ==="
    )
    agent_config, dataset_config = build_configs(task, args.max_test_samples)
    creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = creator.get_chunks()
    all_qas = creator.get_query_and_answers()

    contexts_payload = []
    gold = {}
    global_qid = 0
    for ctx_id, (chunks, qas) in enumerate(zip(all_chunks, all_qas)):
        queries = []
        for query, answer, _qa_id in qas:
            queries.append({"query_id": global_qid, "query": query})
            gold[(ctx_id, global_qid)] = answer
            global_qid += 1
        contexts_payload.append(
            {"context_id": ctx_id, "chunks": chunks, "queries": queries}
        )

    payload = {
        "top_k": max(args.k_values),
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "contexts": contexts_payload,
    }

    workdir = os.path.join(args.output_dir, "_work", task["sub_dataset"])
    t0 = time.time()
    retr = run_retriever(args.retriever_bin, payload, workdir)
    wall = time.time() - t0

    per_query = []
    recall_hits = {k: 0 for k in args.k_values}
    build_times = []
    query_times = []

    for r in retr["results"]:
        key = (r["context_id"], r["query_id"])
        answers = gold.get(key)
        texts = r["retrieved_texts"]
        hits_at_k = {}
        for k in args.k_values:
            prediction = "\n\n".join(texts[:k])
            hit = drqa_metric_max_over_ground_truths(
                substring_exact_match_score, prediction, answers
            )
            hits_at_k[k] = int(bool(hit))
            recall_hits[k] += hits_at_k[k]
        build_times.append(r.get("build_ms", 0.0))
        query_times.append(r.get("query_ms", 0.0))
        per_query.append(
            {
                "context_id": r["context_id"],
                "query_id": r["query_id"],
                "answers": answers,
                "retrieved_ids": r["retrieved_ids"],
                "evidence_substring_match_at_k": hits_at_k,
                "build_ms": r.get("build_ms", 0.0),
                "query_ms": r.get("query_ms", 0.0),
            }
        )

    n = len(per_query)
    recall_at_k = {
        str(k): round((recall_hits[k] / n) * 100, 2) if n else 0.0
        for k in args.k_values
    }
    summary = {
        "sub_dataset": task["sub_dataset"],
        "competency": "Accurate Retrieval",
        "retriever": retr["retriever"],
        "num_contexts": len(contexts_payload),
        "num_queries": n,
        "evidence_recall_at_k": recall_at_k,
        "avg_index_build_ms": round(sum(build_times) / len(build_times), 4)
        if build_times
        else 0.0,
        "avg_query_ms": round(sum(query_times) / len(query_times), 4)
        if query_times
        else 0.0,
        "wall_time_s": round(wall, 2),
    }
    topk_str = " ".join(f"@{k}={recall_at_k[str(k)]}%" for k in args.k_values)
    print(
        f"  contexts={summary['num_contexts']} queries={n} "
        f"evidence_recall {topk_str} avg_query={summary['avg_query_ms']}ms"
    )
    return summary, per_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever-bin",
        default=os.environ.get("SOCHDB_MEMBENCH_RETRIEVER", ""),
        help="Path to sochdb-membench-retriever binary",
    )
    parser.add_argument("--max-test-samples", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    if not args.retriever_bin:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        args.retriever_bin = os.path.join(
            repo_root,
            "target",
            "release",
            "sochdb-membench-retriever",
        )

    tasks = DEFAULT_TASKS
    if args.tasks:
        tasks = [t for t in DEFAULT_TASKS if t["sub_dataset"] in args.tasks]

    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = []
    all_details = {}

    for task in tasks:
        summary, per_query = evaluate_task(task, args)
        all_summaries.append(summary)
        all_details[task["sub_dataset"]] = per_query

    report = {
        "benchmark": "MemoryAgentBench / Accurate Retrieval",
        "retriever": "sochdb-memory-lexical",
        "max_test_samples": args.max_test_samples,
        "k_values": args.k_values,
        "metric": "evidence_recall_at_k (substring_exact_match)",
        "tasks": all_summaries,
    }

    out_path = os.path.join(args.output_dir, "sochdb_membench_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    details_path = os.path.join(args.output_dir, "sochdb_membench_details.json")
    with open(details_path, "w") as f:
        json.dump(all_details, f, indent=2)

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()