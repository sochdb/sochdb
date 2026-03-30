#!/usr/bin/env python3
"""
CrewAI + SochDB Integration Example

Demonstrates a multi-agent research crew with:
- SochDB VectorIndex for knowledge retrieval
- CrewAI for multi-agent orchestration
- Real embeddings via Azure OpenAI

Usage:
    pip install sochdb python-dotenv requests numpy crewai crewai-tools
    python3 examples/python/crewai_research_crew.py

    # Or from this monorepo:
    cd sochdb-python
    maturin develop --release
    cd ..
    python3 examples/python/crewai_research_crew.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import time
import numpy as np
import requests
from typing import List, Dict

# Add SochDB package path for monorepo source usage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../sochdb-python"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# SochDB + Azure OpenAI Setup
# =============================================================================

class SochDBRetriever:
    """SochDB-powered retrieval tool for CrewAI agents."""
    
    def __init__(self):
        from sochdb import VectorIndex
        
        # Azure embeddings
        self.embed_endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.embed_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.embed_deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.embed_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        
        # Initialize index
        self.dim = 1536  # Azure embedding dimension
        self.index = VectorIndex(dimension=self.dim, max_connections=16, ef_construction=100)
        self.documents = []
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via Azure."""
        url = f"{self.embed_endpoint.rstrip('/')}/openai/deployments/{self.embed_deployment}/embeddings?api-version={self.embed_version}"
        headers = {"api-key": self.embed_key, "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json={"input": texts, "model": self.embed_deployment})
        response.raise_for_status()
        return np.array([item["embedding"] for item in response.json()["data"]], dtype=np.float32)
    
    def add_documents(self, docs: List[str]):
        """Add documents to the knowledge base."""
        embeddings = self.embed(docs)
        start_id = len(self.documents)
        ids = np.arange(start_id, start_id + len(docs), dtype=np.uint64)
        self.index.insert_batch(ids, embeddings)
        self.documents.extend(docs)
        return len(docs)
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant documents."""
        query_embed = self.embed([query])[0]
        results = self.index.search(query_embed, k=k)
        return [self.documents[int(r[0])] for r in results if int(r[0]) < len(self.documents)]


# =============================================================================
# CrewAI Example (Without actual CrewAI to avoid dependency issues)
# =============================================================================

def run_crewai_style_research():
    """
    Simulate a CrewAI-style multi-agent research workflow.
    
    In a real CrewAI setup, you would:
    1. Create Agent instances with roles
    2. Create Task instances
    3. Create a Crew and execute
    
    This example shows the SochDB integration pattern.
    """
    
    print("="*70)
    print("  CREWAI-STYLE RESEARCH CREW + SOCHDB")
    print("="*70)
    
    # Setup retriever
    print("\n1. Setting up SochDB knowledge base...")
    retriever = SochDBRetriever()
    
    # Add knowledge documents
    knowledge = [
        "SochDB is a high-performance database designed for AI agents and LLM applications.",
        "SochDB achieves 117,000 vector inserts per second and 0.03ms search latency.",
        "SochDB uses a Trie-Columnar Hybrid (TCH) storage engine for hierarchical data.",
        "SochDB is 24x faster than ChromaDB on vector search benchmarks.",
        "SochDB supports session management, context queries, and token budget enforcement.",
        "SochDB provides an MCP (Model Context Protocol) integration for LLM tools.",
        "SochDB's HNSW index uses SIMD acceleration for fast similarity search.",
        "SochDB can handle 10M+ vectors with consistent sub-millisecond latency.",
    ]
    
    count = retriever.add_documents(knowledge)
    print(f"   ✓ Added {count} documents to knowledge base")
    
    # ==========================================================================
    # Simulate CrewAI Agents
    # ==========================================================================
    
    print("\n2. Simulating CrewAI agents...\n")
    
    # Azure LLM for agents
    llm_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
    llm_key = os.getenv("AZURE_OPENAI_API_KEY")
    llm_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    llm_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    def call_llm(role: str, task: str, context: str = "") -> str:
        url = f"{llm_endpoint.rstrip('/')}/openai/deployments/{llm_deployment}/chat/completions?api-version={llm_version}"
        headers = {"api-key": llm_key, "Content-Type": "application/json"}
        
        system_prompt = f"You are a {role}. {context}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        
        response = requests.post(url, headers=headers, json={
            "messages": messages, "max_tokens": 300, "temperature": 0.0
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    # Agent 1: Researcher
    print("   [Researcher Agent]")
    research_query = "What makes SochDB fast for AI applications?"
    research_context = retriever.search(research_query, k=3)
    print(f"   Retrieved {len(research_context)} relevant docs")
    
    research_result = call_llm(
        role="Research Analyst specializing in database technologies",
        task=f"Based on this information, summarize SochDB's key performance characteristics:\n\n{chr(10).join(research_context)}",
    )
    print(f"   Research: {research_result[:100]}...")
    
    # Agent 2: Writer
    print("\n   [Writer Agent]")
    write_result = call_llm(
        role="Technical Writer creating documentation",
        task=f"Write a brief product description for SochDB based on this research:\n\n{research_result}",
    )
    print(f"   Article: {write_result[:100]}...")
    
    # Agent 3: Reviewer
    print("\n   [Reviewer Agent]")
    review_result = call_llm(
        role="Technical Editor ensuring accuracy and clarity",
        task=f"Review this product description and provide a final, polished version:\n\n{write_result}",
    )
    print(f"   Final: {review_result[:100]}...")
    
    # ==========================================================================
    # Example CrewAI Code (commented out to avoid dependency)
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("\n3. Example CrewAI integration code:\n")
    
    example_code = '''
# pip install crewai crewai-tools

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from sochdb import VectorIndex

# Create a SochDB retrieval tool
class SochDBSearchTool(BaseTool):
    name: str = "sochdb_search"
    description: str = "Search SochDB knowledge base for relevant information"
    
    def __init__(self, retriever: SochDBRetriever):
        self.retriever = retriever
    
    def _run(self, query: str) -> str:
        results = self.retriever.search(query, k=3)
        return "\\n".join(results)

# Create agents
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information about the topic",
    backstory="Expert at finding and synthesizing information",
    tools=[SochDBSearchTool(retriever)],
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear, accurate documentation",
    backstory="Skilled at translating technical concepts",
)

# Create tasks
research_task = Task(
    description="Research SochDB's performance characteristics",
    agent=researcher,
)

write_task = Task(
    description="Write a product description based on research",
    agent=writer,
)

# Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
)

result = crew.kickoff()
'''
    
    print(example_code)
    
    print("\n" + "="*70)
    print("  ✓ CrewAI-style research completed!")
    print("="*70)


if __name__ == "__main__":
    run_crewai_style_research()
