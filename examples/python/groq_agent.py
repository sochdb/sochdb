#!/usr/bin/env python3
"""
Groq + SochDB Agent Example

A complete agentic workflow with:
1. SochDB HnswIndex for knowledge retrieval
2. Groq LLM (llama-3.1-70b-versatile) for response generation
3. Simple hash-based embeddings (replace with real embeddings for production)

Usage:
    # Set your Groq API key
    export GROQ_API_KEY="your-groq-api-key"

    # For real embeddings, also install: pip install sentence-transformers
    # And uncomment the SentenceTransformerEmbeddings class

    python3 examples/python/groq_agent.py
"""

import os
import time
import hashlib
from typing import Dict, List
import numpy as np

from dotenv import load_dotenv
from sochdb import HnswIndex

load_dotenv()


# =============================================================================
# Embeddings Client - Simple hash-based (for demo)
# For production, install sentence-transformers and use real embeddings
# =============================================================================


class SimpleHashEmbeddings:
    """Simple hash-based embeddings for demonstration purposes.
    Replace with real embeddings (sentence-transformers) for production."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> np.ndarray:
        """Convert text to a deterministic vector using hashing."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Extend hash to desired dimension
        seed = int.from_bytes(hash_bytes[:4], "big")
        np.random.seed(seed)
        return np.random.randn(self.dim).astype(np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array([self._hash_to_vector(t) for t in texts])

    def embed_single(self, text: str) -> np.ndarray:
        return self._hash_to_vector(text)


# =============================================================================
# Optional: Real embeddings using sentence-transformers
# Uncomment if you have sentence-transformers installed
# =============================================================================

# from sentence_transformers import SentenceTransformer
#
# class SentenceTransformerEmbeddings:
#     """Embeddings using sentence-transformers."""
#
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.dim = self.model.get_sentence_embedding_dimension()
#
#     def embed(self, texts: List[str]) -> np.ndarray:
#         embeddings = self.model.encode(texts, convert_to_numpy=True)
#         return embeddings.astype(np.float32)
#
#     def embed_single(self, text: str) -> np.ndarray:
#         return self.embed([text])[0]


# =============================================================================
# Groq LLM Client
# =============================================================================


class GroqLLM:
    """Groq LLM client for chat completion."""

    def __init__(self, model: str = "llama-3.1-70b-versatile"):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(
        self, messages: List[Dict], max_tokens: int = 500, temperature: float = 0.0
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


# =============================================================================
# Main Agent
# =============================================================================


def run_groq_agent():
    """Run a complete Groq agent with SochDB retrieval."""

    print("=" * 70)
    print("  GROQ + SOCHDB AGENT EXAMPLE")
    print("=" * 70)

    # Initialize clients
    print("\n1. Initializing clients...")

    embeddings = SimpleHashEmbeddings(dim=384)
    print(f"   ✓ Embeddings: simple-hash (dim={embeddings.dim})")

    llm = GroqLLM()
    print(f"   ✓ LLM: {llm.model}")

    # ==========================================================================
    # Step 2: Setup Knowledge Base
    # ==========================================================================
    print("\n2. Setting up knowledge base...")

    knowledge = [
        "SochDB is a high-performance database designed for AI agents and LLM applications.",
        "SochDB achieves 117,000 vector inserts per second and 0.03ms search latency.",
        "SochDB uses a Trie-Columnar Hybrid (TCH) storage engine for hierarchical data.",
        "SochDB supports session management, context queries, and token budget enforcement.",
        "SochDB is 24x faster than ChromaDB on vector search benchmarks.",
        "SochDB supports zero-copy vector search with PyO3 integration.",
        "SochDB can handle millions of vectors with HNSW indexing.",
        "SochDB is written in Rust for maximum performance and safety.",
    ]

    kb_embeddings = embeddings.embed(knowledge)
    dim = kb_embeddings.shape[1]

    index = HnswIndex(dimension=dim, m=16, ef_construction=100)
    ids = np.arange(len(knowledge), dtype=np.uint64)
    index.insert_batch_with_ids(ids, kb_embeddings)

    print(f"   ✓ Indexed {len(knowledge)} knowledge chunks ({dim}-dim)")

    # ==========================================================================
    # Step 3: Define Agent Functions
    # ==========================================================================

    def retrieve_context(query: str, k: int = 3) -> str:
        """Retrieve relevant context from SochDB vector index."""
        query_embed = embeddings.embed_single(query)
        ids, dists = index.search(query_embed, k=k)

        context_parts = [knowledge[int(i)] for i in ids if int(i) < len(knowledge)]
        context = "\n".join(f"- {c}" for c in context_parts)

        print(f"   [Retrieve] Found {len(ids)} relevant chunks")
        return context

    def generate_response(query: str, context: str) -> str:
        """Generate response using Groq LLM with retrieved context."""

        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful AI assistant with access to a knowledge base.

Use the following context to answer the user's question. Be concise and accurate.

Knowledge Base:
{context}""",
            },
            {"role": "user", "content": query},
        ]

        response = llm.complete(messages, max_tokens=200)
        return response

    # ==========================================================================
    # Step 4: Run Conversation
    # ==========================================================================
    print("\n3. Running conversation...\n")
    print("-" * 70)

    questions = [
        "What is SochDB designed for?",
        "How fast is SochDB compared to ChromaDB?",
        "What storage engine does SochDB use?",
        "What language is SochDB written in?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[Turn {i}] User: {question}")

        start = time.perf_counter()

        context = retrieve_context(question)
        answer = generate_response(question, context)

        latency = (time.perf_counter() - start) * 1000

        print(f"[Turn {i}] Assistant: {answer}")
        print(f"         (Latency: {latency:.0f}ms)")

    print("\n" + "=" * 70)
    print("  ✓ Agent conversation completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_groq_agent()
