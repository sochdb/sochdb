#!/usr/bin/env python3
"""
ToonDB + LangGraph Integration Example
======================================

This example demonstrates how to integrate ToonDB with LangGraph for building
AI agents with persistent memory and semantic retrieval.

Features:
- ToonDB as unified memory store (KV + Vector)
- Context queries with token budgeting
- LangGraph state management

Expected Output:
    ✓ ToonDB connected
    ✓ Knowledge indexed (5 documents)
    ✓ Agent graph built
    ✓ Query executed with context retrieval
    ✓ Response generated

Usage:
    # Set environment variables
    export PYTHONPATH=toondb-python-sdk/src
    export TOONDB_LIB_PATH=target/release
    
    # Run example
    python3 examples/python/04_langgraph_integration.py
"""

import os
import sys
import json
import time
from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

import numpy as np


@dataclass
class ContextSection:
    """A section of context for the LLM."""
    name: str
    content: str
    priority: int
    tokens: int


class AgentState(TypedDict):
    """LangGraph agent state."""
    messages: List[Dict[str, str]]
    context: str
    user_id: str
    retrieved_docs: List[Dict]


class ToonDBMemory:
    """
    ToonDB-backed memory for LangGraph agents.
    
    Provides:
    - Vector storage for semantic search
    - KV storage for user preferences and state
    - Context queries with token budgeting
    """
    
    def __init__(self, dimension: int = 768):
        from toondb import VectorIndex, Database
        
        self.dimension = dimension
        self.index = VectorIndex(
            dimension=dimension,
            max_connections=16,
            ef_construction=100
        )
        self.documents = {}  # id -> document
        self.next_id = 0
    
    def add_document(self, content: str, embedding: np.ndarray, metadata: Dict = None):
        """Add a document with its embedding."""
        doc_id = self.next_id
        self.next_id += 1
        
        self.documents[doc_id] = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Insert into vector index
        ids = np.array([doc_id], dtype=np.uint64)
        vectors = embedding.reshape(1, -1).astype(np.float32)
        self.index.insert_batch(ids, vectors)
        
        return doc_id
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        results = self.index.search(query_embedding.astype(np.float32), k=k)
        
        docs = []
        for doc_id, distance in results:
            if doc_id in self.documents:
                doc = self.documents[doc_id].copy()
                doc["distance"] = distance
                docs.append(doc)
        
        return docs
    
    def build_context(self, query_embedding: np.ndarray, 
                      user_prefs: Dict = None,
                      token_budget: int = 4000) -> str:
        """
        Build prioritized context for LLM.
        
        Priority 0: System prompt
        Priority 1: User preferences
        Priority 2: Retrieved documents
        """
        sections = []
        
        # Priority 0: System prompt (always included)
        system_prompt = "You are a helpful AI assistant with access to a knowledge base."
        sections.append(ContextSection(
            name="system",
            content=system_prompt,
            priority=0,
            tokens=len(system_prompt.split())  # Rough estimate
        ))
        
        # Priority 1: User preferences
        if user_prefs:
            prefs_text = f"User preferences: {json.dumps(user_prefs)}"
            sections.append(ContextSection(
                name="user_prefs",
                content=prefs_text,
                priority=1,
                tokens=len(prefs_text.split())
            ))
        
        # Priority 2: Retrieved documents
        docs = self.search(query_embedding, k=5)
        for i, doc in enumerate(docs):
            doc_text = f"[Doc {i+1}]: {doc['content']}"
            sections.append(ContextSection(
                name=f"doc_{i}",
                content=doc_text,
                priority=2,
                tokens=len(doc_text.split())
            ))
        
        # Assemble context (respecting token budget)
        # Sort by priority (lower = higher priority)
        sections.sort(key=lambda s: s.priority)
        
        context_parts = []
        current_tokens = 0
        
        for section in sections:
            if current_tokens + section.tokens <= token_budget:
                context_parts.append(section.content)
                current_tokens += section.tokens
        
        return "\n\n".join(context_parts)


def mock_embedding(text: str, dimension: int = 768) -> np.ndarray:
    """Generate a mock embedding for testing without API calls."""
    # Use hash of text for reproducibility
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.randn(dimension).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    return embedding


def main():
    print("=" * 70)
    print("  ToonDB + LangGraph Integration Example")
    print("=" * 70)
    
    # 1. Initialize ToonDB memory
    print("\n[1] Initializing ToonDB memory...")
    try:
        memory = ToonDBMemory(dimension=768)
        print("    ✓ ToonDB connected")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return 1
    
    # 2. Add knowledge documents
    print("\n[2] Indexing knowledge base...")
    knowledge = [
        {
            "topic": "ToonDB Architecture",
            "content": "ToonDB uses a Trie-Columnar Hybrid storage engine for O(path) lookups."
        },
        {
            "topic": "Vector Search",
            "content": "ToonDB implements HNSW with SIMD acceleration for fast vector similarity search."
        },
        {
            "topic": "Context Queries",
            "content": "Context queries allow building prioritized context sections with token budgets."
        },
        {
            "topic": "MCP Integration",
            "content": "ToonDB supports Model Context Protocol for seamless LLM tool integration."
        },
        {
            "topic": "Performance",
            "content": "ToonDB achieves sub-millisecond search latency and 10K+ vectors/sec insert rate."
        }
    ]
    
    for doc in knowledge:
        embedding = mock_embedding(doc["content"])
        memory.add_document(
            content=doc["content"],
            embedding=embedding,
            metadata={"topic": doc["topic"]}
        )
    
    print(f"    ✓ Indexed {len(knowledge)} documents")
    
    # 3. Simulate agent graph
    print("\n[3] Building agent graph...")
    
    def retrieve_context(state: AgentState) -> AgentState:
        """Retrieve relevant context from ToonDB."""
        user_query = state["messages"][-1]["content"]
        query_embedding = mock_embedding(user_query)
        
        # Build context with token budget
        context = memory.build_context(
            query_embedding=query_embedding,
            user_prefs={"language": "en", "expertise": "developer"},
            token_budget=2000
        )
        
        # Get retrieved docs for state
        docs = memory.search(query_embedding, k=3)
        
        return {
            **state,
            "context": context,
            "retrieved_docs": docs
        }
    
    def generate_response(state: AgentState) -> AgentState:
        """Generate response (mock LLM for this example)."""
        # In real usage, this would call an LLM with state["context"]
        docs = state["retrieved_docs"]
        if docs:
            answer = f"Based on the knowledge base: {docs[0]['content']}"
        else:
            answer = "I don't have information about that."
        
        state["messages"].append({"role": "assistant", "content": answer})
        return state
    
    print("    ✓ Agent graph built")
    
    # 4. Run test queries
    print("\n[4] Running test queries...")
    print("-" * 70)
    
    test_queries = [
        "What storage engine does ToonDB use?",
        "How fast is the vector search?",
        "What is the MCP integration?"
    ]
    
    for query in test_queries:
        # Initialize state
        state: AgentState = {
            "messages": [{"role": "user", "content": query}],
            "context": "",
            "user_id": "test_user",
            "retrieved_docs": []
        }
        
        # Run graph nodes
        start = time.perf_counter()
        state = retrieve_context(state)
        state = generate_response(state)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\n  Q: {query}")
        print(f"  A: {state['messages'][-1]['content'][:80]}...")
        print(f"  Retrieved: {len(state['retrieved_docs'])} docs | Time: {elapsed:.1f}ms")
    
    print("\n" + "=" * 70)
    print("  ✅ LangGraph integration example complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
