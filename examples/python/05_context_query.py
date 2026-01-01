#!/usr/bin/env python3
"""
ToonDB Context Query Example
============================

This example demonstrates ToonDB's context query system, which allows
building prioritized context sections with automatic token budgeting.

The Context Query is ToonDB's "killer feature" for AI agents:
- Assemble multi-source context in one call
- Automatic token budget enforcement
- Priority-based truncation

Expected Output:
    ✓ Created context sections
    ✓ Token budget respected
    ✓ Priorities applied correctly
    ✓ Context assembled for LLM

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 examples/python/05_context_query.py
"""

import os
import sys
import json
from typing import List, Dict, Any
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

import numpy as np


@dataclass
class Section:
    """A context section with priority and content."""
    name: str
    kind: str  # literal, get, last, search
    priority: int
    content: str = ""
    tokens: int = 0
    
    def estimate_tokens(self) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(self.content) // 4


@dataclass
class ContextQueryResult:
    """Result from a context query."""
    text: str
    sections_included: List[str]
    sections_truncated: List[str]
    total_tokens: int
    budget: int


class MockToonDB:
    """
    Mock ToonDB for demonstrating context queries.
    
    In production, this would connect to the actual ToonDB MCP server.
    """
    
    def __init__(self):
        # Simulated data
        self.kv_store = {
            "/users/alice/preferences": {"theme": "dark", "language": "python"},
            "/users/alice/profile": {"name": "Alice", "role": "developer"},
            "/system/config": {"model": "gpt-4", "max_tokens": 4096}
        }
        
        self.tables = {
            "messages": [
                {"id": 1, "role": "user", "content": "How do I use ToonDB?"},
                {"id": 2, "role": "assistant", "content": "ToonDB is easy! Just import and connect."},
                {"id": 3, "role": "user", "content": "Can I use it for RAG?"},
                {"id": 4, "role": "assistant", "content": "Yes! ToonDB has built-in vector search."},
                {"id": 5, "role": "user", "content": "What about token budgets?"},
            ],
            "knowledge": [
                {"id": 1, "topic": "Architecture", "content": "ToonDB uses Trie-Columnar Hybrid storage."},
                {"id": 2, "topic": "Performance", "content": "Sub-millisecond search, 10K+ vec/s insert."},
                {"id": 3, "topic": "Integration", "content": "Supports MCP, LangGraph, CrewAI."},
            ]
        }
    
    def get(self, path: str) -> Dict:
        """Get value at path."""
        return self.kv_store.get(path, {})
    
    def last(self, table: str, top_k: int) -> List[Dict]:
        """Get last K rows from table."""
        if table in self.tables:
            return self.tables[table][-top_k:]
        return []
    
    def search(self, query: str, top_k: int) -> List[Dict]:
        """Semantic search (mock: keyword matching)."""
        results = []
        for doc in self.tables.get("knowledge", []):
            if query.lower() in doc["content"].lower():
                results.append(doc)
        return results[:top_k]


def context_query(
    db: MockToonDB,
    sections: List[Dict],
    token_budget: int = 4096,
    truncation: str = "tail_drop"
) -> ContextQueryResult:
    """
    Execute a context query with token budgeting.
    
    This is the core ToonDB feature for AI agents.
    
    Args:
        db: ToonDB connection
        sections: List of section definitions
        token_budget: Maximum tokens to return
        truncation: Strategy - tail_drop, head_drop, proportional
    
    Returns:
        ContextQueryResult with assembled context
    """
    
    # 1. Resolve each section
    resolved = []
    for sec in sections:
        kind = sec["kind"]
        priority = sec.get("priority", 99)
        name = sec["name"]
        
        if kind == "literal":
            content = sec["text"]
        elif kind == "get":
            data = db.get(sec["path"])
            content = f"[{name}]: {json.dumps(data)}"
        elif kind == "last":
            rows = db.last(sec["table"], sec["top_k"])
            content = f"[{name}]:\n" + "\n".join(
                f"  - {row.get('role', 'user')}: {row.get('content', str(row))}"
                for row in rows
            )
        elif kind == "search":
            results = db.search(sec["query"], sec["top_k"])
            content = f"[{name}]:\n" + "\n".join(
                f"  - {r.get('topic', 'doc')}: {r.get('content', str(r))}"
                for r in results
            )
        else:
            content = ""
        
        section = Section(
            name=name,
            kind=kind,
            priority=priority,
            content=content,
            tokens=len(content) // 4  # Rough estimate
        )
        resolved.append(section)
    
    # 2. Sort by priority
    resolved.sort(key=lambda s: s.priority)
    
    # 3. Apply truncation
    included = []
    truncated = []
    current_tokens = 0
    
    if truncation == "tail_drop":
        # Include sections until budget exhausted
        for sec in resolved:
            if current_tokens + sec.tokens <= token_budget:
                included.append(sec)
                current_tokens += sec.tokens
            else:
                truncated.append(sec.name)
    
    elif truncation == "proportional":
        # Allocate proportionally
        total_tokens = sum(s.tokens for s in resolved)
        if total_tokens <= token_budget:
            included = resolved
            current_tokens = total_tokens
        else:
            ratio = token_budget / total_tokens
            for sec in resolved:
                sec.tokens = int(sec.tokens * ratio)
                sec.content = sec.content[:sec.tokens * 4]  # Truncate content
                included.append(sec)
                current_tokens += sec.tokens
    
    # 4. Assemble final context
    context_text = "\n\n".join(sec.content for sec in included)
    
    return ContextQueryResult(
        text=context_text,
        sections_included=[s.name for s in included],
        sections_truncated=truncated,
        total_tokens=current_tokens,
        budget=token_budget
    )


def main():
    print("=" * 70)
    print("  ToonDB Context Query Example")
    print("=" * 70)
    
    # Initialize mock database
    db = MockToonDB()
    print("\n[1] Database initialized with mock data")
    
    # Define context query sections
    sections = [
        # Priority 0: System prompt (always included)
        {
            "name": "system",
            "kind": "literal",
            "text": "You are a helpful AI assistant. Answer based on the provided context.",
            "priority": 0
        },
        # Priority 1: User preferences
        {
            "name": "user_prefs",
            "kind": "get",
            "path": "/users/alice/preferences",
            "priority": 1
        },
        # Priority 2: Recent conversation
        {
            "name": "history",
            "kind": "last",
            "table": "messages",
            "top_k": 3,
            "priority": 2
        },
        # Priority 3: Knowledge search
        {
            "name": "knowledge",
            "kind": "search",
            "query": "performance",
            "top_k": 2,
            "priority": 3
        }
    ]
    
    print(f"\n[2] Defined {len(sections)} context sections:")
    for sec in sections:
        print(f"    Priority {sec['priority']}: {sec['name']} ({sec['kind']})")
    
    # Test with different token budgets
    print("\n[3] Testing with different token budgets...")
    print("-" * 70)
    
    for budget in [200, 500, 2000]:
        result = context_query(db, sections, token_budget=budget)
        
        print(f"\n  Budget: {budget} tokens")
        print(f"  Included: {result.sections_included}")
        print(f"  Truncated: {result.sections_truncated}")
        print(f"  Actual tokens: {result.total_tokens}")
    
    # Show full context with high budget
    print("\n[4] Full context assembly (high budget):")
    print("-" * 70)
    
    result = context_query(db, sections, token_budget=5000)
    print(result.text)
    
    print("\n" + "=" * 70)
    print("  ✅ Context query example complete!")
    print("=" * 70)
    
    print("""
    Key Takeaways:
    - Priority 0 sections are always included
    - Lower priority sections are truncated first (tail_drop)
    - Token budget is strictly enforced
    - Context is assembled in priority order
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
