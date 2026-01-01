#!/usr/bin/env python3
"""
Real End-to-End LLM Test for ToonDB

Uses actual Azure OpenAI APIs to test:
1. Embedding generation via Azure Cognitive Services
2. Vector storage and retrieval with ToonDB
3. LLM-powered question answering with retrieved context

Requires .env file with Azure credentials.
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np

# Add ToonDB SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../toondb-python-sdk/src"))

# Load environment variables
load_dotenv()


@dataclass
class TestResult:
    """Result from a single test case."""
    question: str
    expected_topic: str
    retrieved_context: str
    llm_answer: str
    retrieval_latency_ms: float
    llm_latency_ms: float
    correct: bool


class AzureEmbeddings:
    """Azure Cognitive Services embeddings client."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.api_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.api_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        self.deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Missing Azure embedding credentials in .env")
        
        self.url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "input": texts,
            "model": self.deployment
        }
        
        response = requests.post(self.url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings, dtype=np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.embed([text])[0]


class AzureLLM:
    """Azure OpenAI LLM client."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_API_BASE")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Missing Azure OpenAI credentials in .env")
        
        self.url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
    
    def complete(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Generate completion."""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0  # Deterministic for testing
        }
        
        response = requests.post(self.url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


def run_real_e2e_test():
    """Run real end-to-end test with Azure OpenAI."""
    
    print("="*70)
    print("  REAL END-TO-END LLM TEST")
    print("  Using Azure OpenAI + ToonDB")
    print("="*70)
    
    # Initialize clients
    print("\n1. Initializing Azure clients...")
    embeddings = AzureEmbeddings()
    llm = AzureLLM()
    print("   ✓ Azure Embeddings connected")
    print("   ✓ Azure OpenAI connected")
    
    # Initialize ToonDB
    print("\n2. Initializing ToonDB...")
    try:
        from toondb import VectorIndex
        if VectorIndex is None:
            raise ImportError("VectorIndex not available")
    except ImportError as e:
        print(f"   ✗ ToonDB not available: {e}")
        return
    
    # Test knowledge base
    knowledge_base = [
        {
            "id": 0,
            "topic": "ToonDB Architecture",
            "content": "ToonDB uses a Trie-Columnar Hybrid (TCH) storage engine that provides O(path) resolution time. It combines the benefits of tries for hierarchical key access with columnar storage for analytical queries."
        },
        {
            "id": 1,
            "topic": "Vector Search",
            "content": "ToonDB's vector search uses HNSW (Hierarchical Navigable Small World) algorithm implemented in Rust with SIMD acceleration. It achieves 117,000 vectors per second insert rate and sub-millisecond search latency."
        },
        {
            "id": 2,
            "topic": "AI Agent Integration",
            "content": "ToonDB is designed for AI agents with built-in session management, context queries, and token budget enforcement. It supports MCP (Model Context Protocol) for seamless LLM integration."
        },
        {
            "id": 3,
            "topic": "Performance",
            "content": "ToonDB outperforms ChromaDB by 24x on search latency and 11x on insert throughput. It also beats SQLite by 24% on insert operations for key-value workloads."
        },
        {
            "id": 4,
            "topic": "Context Queries",
            "content": "ToonDB's ContextQuery system allows building prioritized context sections with token budgets. It supports SEARCH, LAST, GET, and SELECT operators for flexible context assembly."
        }
    ]
    
    # Generate embeddings
    print("\n3. Generating embeddings with Azure...")
    texts = [doc["content"] for doc in knowledge_base]
    
    start = time.perf_counter()
    doc_embeddings = embeddings.embed(texts)
    embed_time = (time.perf_counter() - start) * 1000
    
    dim = doc_embeddings.shape[1]
    print(f"   ✓ Generated {len(texts)} embeddings ({dim}-dim) in {embed_time:.0f}ms")
    
    # Build ToonDB index
    print("\n4. Building ToonDB index...")
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    
    ids = np.arange(len(knowledge_base), dtype=np.uint64)
    start = time.perf_counter()
    inserted = index.insert_batch(ids, doc_embeddings)
    index_time = (time.perf_counter() - start) * 1000
    
    print(f"   ✓ Indexed {inserted} documents in {index_time:.0f}ms")
    
    # Test questions
    test_cases = [
        {
            "question": "What storage engine does ToonDB use?",
            "expected_topic": "ToonDB Architecture"
        },
        {
            "question": "How fast is ToonDB's vector search?",
            "expected_topic": "Vector Search"
        },
        {
            "question": "How does ToonDB compare to ChromaDB?",
            "expected_topic": "Performance"
        },
        {
            "question": "What is ContextQuery?",
            "expected_topic": "Context Queries"
        },
        {
            "question": "Does ToonDB support AI agents?",
            "expected_topic": "AI Agent Integration"
        }
    ]
    
    # Run tests
    print("\n5. Running RAG tests...")
    print("-"*70)
    
    results = []
    total_retrieval_ms = 0
    total_llm_ms = 0
    
    for i, test in enumerate(test_cases):
        question = test["question"]
        expected = test["expected_topic"]
        
        # Generate query embedding
        query_embed = embeddings.embed_single(question)
        
        # Retrieve from ToonDB
        start = time.perf_counter()
        search_results = index.search(query_embed, k=3)
        retrieval_ms = (time.perf_counter() - start) * 1000
        total_retrieval_ms += retrieval_ms
        
        # Get retrieved context
        retrieved_docs = [knowledge_base[int(r[0])] for r in search_results]
        context = "\n\n".join([f"[{doc['topic']}]: {doc['content']}" for doc in retrieved_docs])
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions based ONLY on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        start = time.perf_counter()
        answer = llm.complete(messages, max_tokens=150)
        llm_ms = (time.perf_counter() - start) * 1000
        total_llm_ms += llm_ms
        
        # Check if correct document was retrieved
        top_topic = retrieved_docs[0]["topic"]
        correct = top_topic == expected
        
        result = TestResult(
            question=question,
            expected_topic=expected,
            retrieved_context=top_topic,
            llm_answer=answer,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=llm_ms,
            correct=correct
        )
        results.append(result)
        
        status = "✓" if correct else "✗"
        print(f"\n[Test {i+1}] {status}")
        print(f"  Question: {question}")
        print(f"  Expected: {expected}")
        print(f"  Retrieved: {top_topic}")
        print(f"  Retrieval: {retrieval_ms:.2f}ms | LLM: {llm_ms:.0f}ms")
        print(f"  Answer: {answer[:100]}...")
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    correct_count = sum(1 for r in results if r.correct)
    accuracy = correct_count / len(results) * 100
    
    print(f"\n  Accuracy: {correct_count}/{len(results)} ({accuracy:.0f}%)")
    print(f"  Avg Retrieval: {total_retrieval_ms/len(results):.2f}ms")
    print(f"  Avg LLM: {total_llm_ms/len(results):.0f}ms")
    print(f"  Total Time: {total_retrieval_ms + total_llm_ms:.0f}ms")
    
    if accuracy == 100:
        print("\n  🏆 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {len(results) - correct_count} test(s) had incorrect retrieval")
    
    return results


def run_memory_test():
    """Test agent memory with real LLM."""
    
    print("\n" + "="*70)
    print("  AGENT MEMORY TEST")
    print("="*70)
    
    embeddings = AzureEmbeddings()
    llm = AzureLLM()
    
    from toondb import VectorIndex
    
    # Simulate agent memory (session history)
    memories = [
        {"id": 0, "turn": 1, "role": "user", "content": "My name is Alice and I'm a software engineer."},
        {"id": 1, "turn": 2, "role": "assistant", "content": "Nice to meet you, Alice! What kind of software do you work on?"},
        {"id": 2, "turn": 3, "role": "user", "content": "I work on database systems, specifically ToonDB."},
        {"id": 3, "turn": 4, "role": "assistant", "content": "ToonDB sounds interesting! What makes it special?"},
        {"id": 4, "turn": 5, "role": "user", "content": "It's optimized for AI agents with fast vector search."},
        {"id": 5, "turn": 6, "role": "user", "content": "My favorite programming language is Rust."},
        {"id": 6, "turn": 7, "role": "user", "content": "I live in San Francisco."},
    ]
    
    # Embed memories
    print("\n1. Embedding memories...")
    texts = [m["content"] for m in memories]
    mem_embeddings = embeddings.embed(texts)
    
    # Build index
    dim = mem_embeddings.shape[1]
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    ids = np.arange(len(memories), dtype=np.uint64)
    index.insert_batch(ids, mem_embeddings)
    print(f"   ✓ Indexed {len(memories)} memories")
    
    # Test memory retrieval
    test_queries = [
        "What is the user's name?",
        "What does the user work on?",
        "What programming language does the user prefer?",
        "Where does the user live?",
    ]
    
    print("\n2. Testing memory retrieval + LLM...")
    print("-"*70)
    
    for query in test_queries:
        # Retrieve relevant memories
        query_embed = embeddings.embed_single(query)
        results = index.search(query_embed, k=3)
        
        # Build context from memories
        relevant_mems = [memories[int(r[0])] for r in results]
        context = "\n".join([f"[Turn {m['turn']}] {m['role']}: {m['content']}" for m in relevant_mems])
        
        # Ask LLM
        messages = [
            {"role": "system", "content": "You are an assistant with access to conversation history. Answer based on the memory."},
            {"role": "user", "content": f"Memory:\n{context}\n\nQuestion: {query}"}
        ]
        
        answer = llm.complete(messages, max_tokens=100)
        
        print(f"\n  Q: {query}")
        print(f"  A: {answer}")
    
    print("\n   ✓ Memory test complete!")


if __name__ == "__main__":
    # Check for dotenv
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv...")
        os.system("pip install python-dotenv --user")
        from dotenv import load_dotenv
    
    run_real_e2e_test()
    run_memory_test()
