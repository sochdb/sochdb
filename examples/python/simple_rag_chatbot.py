#!/usr/bin/env python3
"""
Simple RAG Chatbot with ToonDB

A minimal, production-ready RAG implementation showing:
- Document ingestion with chunking
- Vector search with ToonDB
- Conversational memory
- Streaming responses

Usage:
    python3 examples/python/simple_rag_chatbot.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
import tempfile
from typing import List, Dict, Generator
import numpy as np
import requests

# Add ToonDB SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# RAG Chatbot
# =============================================================================

class RAGChatbot:
    """Simple RAG chatbot powered by ToonDB."""
    
    def __init__(self):
        from toondb import VectorIndex, Database
        
        # Azure credentials
        self.embed_endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.embed_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.embed_deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.embed_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        
        self.llm_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
        self.llm_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.llm_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        self.llm_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        # ToonDB components
        self.dimension = 1536
        self.vector_index = VectorIndex(dimension=self.dimension, max_connections=16, ef_construction=100)
        self.documents: List[Dict] = []
        
        # Memory store
        db_path = tempfile.mktemp(prefix="chatbot_memory_")
        self.memory_db = Database.open(db_path)
        self.conversation_history: List[Dict] = []
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings."""
        url = f"{self.embed_endpoint.rstrip('/')}/openai/deployments/{self.embed_deployment}/embeddings?api-version={self.embed_version}"
        headers = {"api-key": self.embed_key, "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={"input": texts, "model": self.embed_deployment})
        response.raise_for_status()
        return np.array([item["embedding"] for item in response.json()["data"]], dtype=np.float32)
    
    def _call_llm(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Call LLM."""
        url = f"{self.llm_endpoint.rstrip('/')}/openai/deployments/{self.llm_deployment}/chat/completions?api-version={self.llm_version}"
        headers = {"api-key": self.llm_key, "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def ingest(self, documents: List[str], chunk_size: int = 200) -> int:
        """Ingest documents into the knowledge base."""
        all_chunks = []
        
        for doc in documents:
            words = doc.split()
            for i in range(0, len(words), chunk_size - 50):
                chunk = ' '.join(words[i:i + chunk_size])
                if chunk.strip():
                    all_chunks.append({"text": chunk, "source": f"doc_{len(self.documents)}"})
        
        # Embed and index
        texts = [c["text"] for c in all_chunks]
        embeddings = self._embed(texts)
        
        start_id = len(self.documents)
        ids = np.arange(start_id, start_id + len(all_chunks), dtype=np.uint64)
        self.vector_index.insert_batch(ids, embeddings)
        self.documents.extend(all_chunks)
        
        return len(all_chunks)
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents."""
        query_embed = self._embed([query])[0]
        results = self.vector_index.search(query_embed, k=k)
        return [self.documents[int(r[0])]["text"] for r in results if int(r[0]) < len(self.documents)]
    
    def chat(self, user_message: str) -> str:
        """Process a chat message."""
        # Retrieve relevant context
        context_docs = self.retrieve(user_message, k=3)
        context = "\n".join(f"- {doc}" for doc in context_docs)
        
        # Build messages with history
        messages = [
            {"role": "system", "content": f"""You are a helpful AI assistant with access to a knowledge base.

Use the following context to answer questions:
{context}

Be conversational and friendly. If you don't know something, say so."""}
        ]
        
        # Add conversation history (last 5 turns)
        for turn in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self._call_llm(messages)
        
        # Save to history
        self.conversation_history.append({
            "user": user_message,
            "assistant": response
        })
        
        # Save to ToonDB memory
        key = f"chat/{len(self.conversation_history)}".encode()
        with self.memory_db.transaction() as txn:
            txn.put(key, json.dumps(self.conversation_history[-1]).encode())
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# =============================================================================
# Main
# =============================================================================

def run_chatbot():
    """Run the RAG chatbot demo."""
    
    print("="*70)
    print("  SIMPLE RAG CHATBOT + TOONDB")
    print("="*70)
    
    # Initialize chatbot
    print("\n1. Initializing chatbot...")
    chatbot = RAGChatbot()
    print("   ✓ Chatbot ready")
    
    # Ingest knowledge
    print("\n2. Ingesting knowledge base...")
    
    knowledge = [
        """ToonDB is a high-performance database designed for AI agents and LLM applications.
        It provides blazing-fast vector search with sub-millisecond latency, making it ideal 
        for real-time AI systems. ToonDB achieves 117,000 vector inserts per second and 
        0.03ms search latency.""",
        
        """ToonDB uses a Trie-Columnar Hybrid (TCH) storage engine that provides O(path) 
        resolution time. It combines tries for hierarchical key access with columnar storage
        for analytical queries. The HNSW index uses SIMD acceleration for fast similarity search.""",
        
        """ToonDB is 24x faster than ChromaDB on vector search benchmarks. It outperforms 
        SQLite by 24% on insert operations. The database supports session management, 
        context queries, and token budget enforcement for LLM applications.""",
        
        """ToonDB provides MCP (Model Context Protocol) integration for seamless LLM tool usage.
        It can handle 10M+ vectors with consistent sub-millisecond latency. The database
        is open source under the Apache 2.0 license.""",
    ]
    
    chunks = chatbot.ingest(knowledge)
    print(f"   ✓ Ingested {chunks} chunks")
    
    # Interactive chat
    print("\n3. Running chat demo...\n")
    print("-"*70)
    
    conversations = [
        "Hi! What is ToonDB?",
        "How fast is it compared to other databases?",
        "What storage engine does it use?",
        "Is it open source?",
    ]
    
    for msg in conversations:
        print(f"\n👤 User: {msg}")
        
        start = time.perf_counter()
        response = chatbot.chat(msg)
        latency = (time.perf_counter() - start) * 1000
        
        print(f"\n🤖 Bot: {response}")
        print(f"\n   (Latency: {latency:.0f}ms)")
        print("-"*70)
    
    # Show memory
    print("\n4. Stored conversation memory:")
    for i, turn in enumerate(chatbot.conversation_history, 1):
        print(f"   Turn {i}: {turn['user'][:40]}... → {turn['assistant'][:40]}...")
    
    print("\n" + "="*70)
    print("  ✓ Chatbot demo completed!")
    print("="*70)


if __name__ == "__main__":
    run_chatbot()
