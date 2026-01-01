#!/usr/bin/env python3
"""
LangGraph + ToonDB Integration Example

Demonstrates a complete agentic workflow with:
1. ToonDB VectorIndex for knowledge retrieval
2. ToonDB Database for memory storage
3. Azure OpenAI for embeddings and LLM
4. LangGraph for workflow orchestration

Usage:
    python3 examples/langgraph_agent.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
from datetime import datetime
from typing import TypedDict, Sequence, Dict, List
import numpy as np
import requests

# Add ToonDB SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Azure OpenAI Clients
# =============================================================================

class AzureEmbeddings:
    """Azure embeddings client."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.api_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.api_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        self.deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
    
    def embed(self, texts: List[str]) -> np.ndarray:
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        response = requests.post(self.url, headers=headers, json={"input": texts, "model": self.deployment})
        response.raise_for_status()
        return np.array([item["embedding"] for item in response.json()["data"]], dtype=np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class AzureLLM:
    """Azure OpenAI LLM client."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_API_BASE")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        self.url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
    
    def complete(self, messages: List[Dict], max_tokens: int = 500) -> str:
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        response = requests.post(self.url, headers=headers, json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.0})
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# =============================================================================
# LangGraph Agent
# =============================================================================

def run_langgraph_agent():
    """Run a complete LangGraph agent with ToonDB retrieval and memory."""
    
    print("="*70)
    print("  LANGGRAPH + TOONDB AGENT EXAMPLE")
    print("="*70)
    
    # Import LangGraph
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage, AIMessage
    
    # Import ToonDB
    from toondb import VectorIndex, Database
    
    # Initialize clients
    embeddings = AzureEmbeddings()
    llm = AzureLLM()
    
    # ==========================================================================
    # Step 1: Setup Knowledge Base
    # ==========================================================================
    print("\n1. Setting up knowledge base...")
    
    knowledge = [
        "ToonDB is a high-performance database designed for AI agents and LLM applications.",
        "ToonDB achieves 117,000 vector inserts per second and 0.03ms search latency.",
        "ToonDB uses a Trie-Columnar Hybrid (TCH) storage engine for hierarchical data.",
        "ToonDB supports session management, context queries, and token budget enforcement.",
        "ToonDB is 24x faster than ChromaDB on vector search benchmarks.",
    ]
    
    kb_embeddings = embeddings.embed(knowledge)
    dim = kb_embeddings.shape[1]
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    ids = np.arange(len(knowledge), dtype=np.uint64)
    index.insert_batch(ids, kb_embeddings)
    
    print(f"   ✓ Indexed {len(knowledge)} knowledge chunks ({dim}-dim)")
    
    # ==========================================================================
    # Step 2: Setup Memory Store (KV)
    # ==========================================================================
    import tempfile
    db_path = tempfile.mktemp(prefix="toondb_memory_")
    db = Database.open(db_path)
    print(f"   ✓ Memory store initialized")
    
    # ==========================================================================
    # Step 3: Define Agent State
    # ==========================================================================
    from langchain_core.messages import BaseMessage
    
    class AgentState(TypedDict):
        messages: Sequence[BaseMessage]
        context: str
        user_id: str
        turn_count: int
    
    # ==========================================================================
    # Step 4: Define Agent Nodes
    # ==========================================================================
    
    def retrieve_context(state: AgentState) -> dict:
        """Retrieve relevant context from ToonDB vector index."""
        query = state["messages"][-1].content
        query_embed = embeddings.embed_single(query)
        results = index.search(query_embed, k=2)
        
        context_parts = [knowledge[int(r[0])] for r in results]
        context = "\n".join(f"- {c}" for c in context_parts)
        
        print(f"\n   [Retrieve] Found {len(results)} relevant chunks")
        return {"context": context}
    
    def load_memory(state: AgentState) -> dict:
        """Load recent conversation memory from ToonDB KV store."""
        user_id = state.get("user_id", "default")
        prefix = f"memory/{user_id}/".encode()
        
        memories = []
        for k, v in db.scan(prefix):
            try:
                memories.append(json.loads(v.decode()))
            except:
                pass
        
        # Add to context if we have memory
        if memories:
            memory_text = "\n".join(f"- Previous: {m.get('summary', '')}" for m in memories[-3:])
            current_context = state.get("context", "")
            combined = f"Previous conversations:\n{memory_text}\n\nRelevant knowledge:\n{current_context}"
            print(f"   [Memory] Loaded {len(memories)} memories")
            return {"context": combined}
        
        return {}
    
    def generate_response(state: AgentState) -> dict:
        """Generate response using LLM with retrieved context."""
        context = state.get("context", "")
        query = state["messages"][-1].content
        
        messages = [
            {"role": "system", "content": f"""You are a helpful AI assistant with access to a knowledge base.

Use the following context to answer the user's question:

{context}

Be concise and accurate."""},
            {"role": "user", "content": query}
        ]
        
        response = llm.complete(messages, max_tokens=200)
        print(f"   [Generate] Response generated")
        
        return {
            "messages": list(state["messages"]) + [AIMessage(content=response)],
            "turn_count": state.get("turn_count", 0) + 1
        }
    
    def save_memory(state: AgentState) -> dict:
        """Save conversation turn to ToonDB memory store."""
        user_id = state.get("user_id", "default")
        turn = state.get("turn_count", 0)
        
        # Get last user message and assistant response
        user_msg = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
        assistant_msg = state["messages"][-1].content
        
        memory = {
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "assistant": assistant_msg[:100],
            "summary": f"User asked about: {user_msg[:50]}..."
        }
        
        key = f"memory/{user_id}/{turn}".encode()
        with db.transaction() as txn:
            txn.put(key, json.dumps(memory).encode())
        
        print(f"   [Save] Memory saved for turn {turn}")
        return {}
    
    # ==========================================================================
    # Step 5: Build LangGraph Workflow
    # ==========================================================================
    print("\n2. Building LangGraph workflow...")
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("load_memory", load_memory)
    graph.add_node("generate", generate_response)
    graph.add_node("save_memory", save_memory)
    
    # Add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "load_memory")
    graph.add_edge("load_memory", "generate")
    graph.add_edge("generate", "save_memory")
    graph.add_edge("save_memory", END)
    
    agent = graph.compile()
    print("   ✓ Workflow compiled")
    
    # ==========================================================================
    # Step 6: Run Conversation
    # ==========================================================================
    print("\n3. Running conversation...\n")
    print("-"*70)
    
    questions = [
        "What is ToonDB designed for?",
        "How fast is ToonDB compared to ChromaDB?",
        "What storage engine does ToonDB use?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Turn {i}] User: {question}")
        
        start = time.perf_counter()
        result = agent.invoke({
            "messages": [HumanMessage(content=question)],
            "context": "",
            "user_id": "demo_user",
            "turn_count": i - 1
        })
        latency = (time.perf_counter() - start) * 1000
        
        answer = result["messages"][-1].content
        print(f"\n[Turn {i}] Assistant: {answer}")
        print(f"         (Latency: {latency:.0f}ms)")
    
    # ==========================================================================
    # Step 7: Show Stored Memories
    # ==========================================================================
    print("\n" + "-"*70)
    print("\n4. Stored memories:")
    
    for k, v in db.scan(b"memory/demo_user/"):
        memory = json.loads(v.decode())
        print(f"   Turn {memory['turn']}: {memory['summary']}")
    
    print("\n" + "="*70)
    print("  ✓ Agent conversation completed!")
    print("="*70)


if __name__ == "__main__":
    run_langgraph_agent()
