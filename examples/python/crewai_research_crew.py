#!/usr/bin/env python3
"""
CrewAI + SochDB integration example.

This example mirrors the supported CrewAI integration that now lives in the
published Python SDK. It keeps the demo self-contained by using a tiny local
embedder instead of requiring Azure or another embedding service.

Install:
    pip install "sochdb[crewai]"

Optional environment:
    OPENAI_API_KEY=<your key>
    CREWAI_MODEL=gpt-4o-mini
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from typing import Sequence

from sochdb import Database, Namespace, SochDBKnowledgeStore, create_crewai_tools


def deterministic_embed(texts: Sequence[str], dim: int = 32) -> list[list[float]]:
    """
    Tiny local embedder for examples and tests.

    This keeps the example runnable without a second model service. It is not a
    semantically strong embedding model, but it is enough to demonstrate the
    SochDB + CrewAI tool flow.
    """

    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dim)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        vectors.append([v / norm for v in values])
    return vectors


def build_knowledge_store() -> SochDBKnowledgeStore:
    tempdir = tempfile.mkdtemp(prefix="sochdb-crewai-")
    db = Database.open(tempdir)
    namespace = Namespace(db, "crewai_demo")
    collection = namespace.create_collection("knowledge", dimension=32)

    store = SochDBKnowledgeStore.from_collection(
        collection,
        embedder=deterministic_embed,
    )
    store.add_texts(
        [
            "SochDB supports both embedded and gRPC deployment modes.",
            "The hosted SochDB demo endpoint listens on studio.agentslab.host:50053.",
            "The corrected 10GB benchmark showed about 506 QPS after one-time index load.",
            "BAAI/bge-base-en-v1.5 is the best published SciFact quality result so far.",
        ],
        metadatas=[
            {"topic": "architecture"},
            {"topic": "deployment"},
            {"topic": "benchmark"},
            {"topic": "quality"},
        ],
        ids=["arch-1", "deploy-1", "bench-1", "quality-1"],
    )
    return store


def main() -> None:
    from crewai import Agent, Crew, Task

    store = build_knowledge_store()
    search_tool, remember_tool = create_crewai_tools(store, top_k=3)
    model = os.environ.get("CREWAI_MODEL", "gpt-4o-mini")

    researcher = Agent(
        role="SochDB Researcher",
        goal="Answer questions using the SochDB knowledge base before responding.",
        backstory="You are careful about grounding claims in retrieved project context.",
        llm=model,
        tools=[search_tool, remember_tool],
        verbose=True,
    )

    task = Task(
        description=(
            "Find the current 10GB benchmark takeaway and summarize it in 2-3 sentences. "
            "Use the SochDB tools and avoid guessing."
        ),
        expected_output="A short grounded summary of the latest 10GB benchmark result.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=True)
    result = crew.kickoff()

    print("\n=== Crew Result ===\n")
    print(result)


if __name__ == "__main__":
    main()
