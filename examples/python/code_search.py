#!/usr/bin/env python3
"""
Code Search with ToonDB

Demonstrates:
- Code embedding search
- Path/language filtering
- Hybrid keyword + semantic search
- MRR@10 evaluation

Usage:
    python3 examples/python/code_search.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class CodeSnippet:
    id: str
    content: str
    path: str
    language: str
    repo: str
    owner_team: str
    function_name: Optional[str] = None


@dataclass
class SearchResult:
    snippet: CodeSnippet
    score: float
    keyword_boost: float = 0.0


# =============================================================================
# Code Search Engine
# =============================================================================

class CodeSearch:
    """Semantic code search with hybrid keyword matching."""
    
    def __init__(self):
        from toondb import VectorIndex
        
        self.dimension = 1536
        self.index = VectorIndex(dimension=self.dimension, max_connections=32, ef_construction=200)
        
        self.snippets: Dict[str, CodeSnippet] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        # Embedding client
        self.embed_endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.embed_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.embed_deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        self.embed_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        url = f"{self.embed_endpoint.rstrip('/')}/openai/deployments/{self.embed_deployment}/embeddings?api-version={self.embed_version}"
        headers = {"api-key": self.embed_key, "Content-Type": "application/json"}
        
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = requests.post(url, headers=headers, json={"input": batch, "model": self.embed_deployment})
            response.raise_for_status()
            embeddings = [item["embedding"] for item in response.json()["data"]]
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def index_code(self, snippets: List[CodeSnippet]) -> int:
        """Index code snippets."""
        texts = [s.content for s in snippets]
        embeddings = self._embed(texts)
        
        start_idx = self.next_idx
        ids = np.arange(start_idx, start_idx + len(snippets), dtype=np.uint64)
        self.index.insert_batch(ids, embeddings)
        
        for i, snippet in enumerate(snippets):
            idx = start_idx + i
            self.snippets[snippet.id] = snippet
            self.id_to_idx[snippet.id] = idx
            self.idx_to_id[idx] = snippet.id
        
        self.next_idx += len(snippets)
        return len(snippets)
    
    def search(
        self,
        query: str,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        repo: Optional[str] = None,
        owner_team: Optional[str] = None,
        top_k: int = 10,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """Search code with filters and hybrid scoring."""
        
        query_embedding = self._embed([query])[0]
        query_keywords = set(query.lower().split())
        
        # Over-fetch for filtering
        k_fetch = top_k * 5
        results = self.index.search(query_embedding, k=k_fetch)
        
        search_results = []
        
        for idx, score in results:
            snippet_id = self.idx_to_id.get(int(idx))
            if not snippet_id or snippet_id not in self.snippets:
                continue
            
            snippet = self.snippets[snippet_id]
            
            # Apply filters
            if language and snippet.language != language:
                continue
            if path_prefix and not snippet.path.startswith(path_prefix):
                continue
            if repo and snippet.repo != repo:
                continue
            if owner_team and snippet.owner_team != owner_team:
                continue
            
            # Keyword boost
            content_words = set(snippet.content.lower().split())
            keyword_overlap = len(query_keywords & content_words) / len(query_keywords) if query_keywords else 0
            
            # Hybrid score (lower is better for distance, so we invert boost)
            hybrid_score = float(score) * (1 - keyword_weight * keyword_overlap)
            
            search_results.append(SearchResult(
                snippet=snippet,
                score=hybrid_score,
                keyword_boost=keyword_overlap
            ))
        
        # Sort by hybrid score
        search_results.sort(key=lambda x: x.score)
        return search_results[:top_k]


# =============================================================================
# Test Runner
# =============================================================================

def run_code_search_test():
    """Run code search test."""
    
    print("="*70)
    print("  CODE SEARCH + TOONDB")
    print("  Semantic + Keyword Hybrid Search")
    print("="*70)
    
    search = CodeSearch()
    
    # Generate synthetic code snippets
    print("\n1. Generating code snippets...")
    
    snippets = [
        # Rate limiters
        CodeSnippet(
            id="rate_limiter_go",
            content="func RateLimiter(requests int, window time.Duration) *Limiter { return &Limiter{rate: requests, window: window, tokens: make(chan struct{}, requests)} }",
            path="services/auth/middleware/rate_limiter.go",
            language="go",
            repo="backend",
            owner_team="platform",
            function_name="RateLimiter"
        ),
        CodeSnippet(
            id="rate_limiter_py",
            content="class RateLimiter: def __init__(self, max_requests, time_window): self.max_requests = max_requests self.time_window = time_window",
            path="services/api/middleware/rate_limiter.py",
            language="python",
            repo="backend",
            owner_team="platform",
            function_name="RateLimiter"
        ),
        # JWT handling
        CodeSnippet(
            id="jwt_refresh_go",
            content="func RefreshToken(ctx context.Context, refreshToken string) (*TokenPair, error) { claims := validateRefreshToken(refreshToken); return generateTokenPair(claims.UserID) }",
            path="services/auth/jwt/refresh.go",
            language="go",
            repo="backend",
            owner_team="security",
            function_name="RefreshToken"
        ),
        CodeSnippet(
            id="jwt_validate_go",
            content="func ValidateJWT(token string, secret []byte) (*Claims, error) { return jwt.ParseWithClaims(token, &Claims{}, func(t *jwt.Token) (interface{}, error) { return secret, nil }) }",
            path="services/auth/jwt/validate.go",
            language="go",
            repo="backend",
            owner_team="security",
            function_name="ValidateJWT"
        ),
        # Database operations
        CodeSnippet(
            id="db_connection_pool",
            content="func NewConnectionPool(dsn string, maxConns int) (*Pool, error) { config := pgxpool.Config{MaxConns: maxConns}; return pgxpool.NewWithConfig(ctx, config) }",
            path="pkg/database/pool.go",
            language="go",
            repo="backend",
            owner_team="platform",
            function_name="NewConnectionPool"
        ),
        CodeSnippet(
            id="db_transaction",
            content="async def execute_transaction(queries: List[str]) -> None: async with pool.acquire() as conn: async with conn.transaction(): for q in queries: await conn.execute(q)",
            path="pkg/database/transaction.py",
            language="python",
            repo="backend",
            owner_team="platform",
            function_name="execute_transaction"
        ),
        # Vector search
        CodeSnippet(
            id="vector_search",
            content="func SearchVectors(query []float32, k int) []Result { return index.Search(query, k, WithEfSearch(128)) }",
            path="pkg/search/vector.go",
            language="go",
            repo="backend",
            owner_team="ml",
            function_name="SearchVectors"
        ),
        CodeSnippet(
            id="embedding_service",
            content="async def generate_embeddings(texts: List[str]) -> np.ndarray: response = await client.embed(texts); return np.array(response.embeddings)",
            path="services/ml/embeddings.py",
            language="python",
            repo="backend",
            owner_team="ml",
            function_name="generate_embeddings"
        ),
    ]
    
    print(f"   Generated {len(snippets)} code snippets")
    
    # Index
    print("\n2. Indexing code...")
    start = time.perf_counter()
    count = search.index_code(snippets)
    index_time = (time.perf_counter() - start) * 1000
    print(f"   ✓ Indexed {count} snippets in {index_time:.0f}ms")
    
    # Test searches
    print("\n3. Running search tests...")
    print("-"*70)
    
    test_cases = [
        {
            "query": "rate limiter implementation",
            "expected_ids": ["rate_limiter_go", "rate_limiter_py"],
            "filters": {}
        },
        {
            "query": "JWT refresh token rotation",
            "expected_ids": ["jwt_refresh_go"],
            "filters": {"language": "go"}
        },
        {
            "query": "database connection pool",
            "expected_ids": ["db_connection_pool"],
            "filters": {"path_prefix": "pkg/database"}
        },
        {
            "query": "embedding generation",
            "expected_ids": ["embedding_service"],
            "filters": {"owner_team": "ml"}
        },
    ]
    
    mrr_scores = []
    
    for test in test_cases:
        print(f"\n   Query: {test['query']}")
        if test['filters']:
            print(f"   Filters: {test['filters']}")
        
        start = time.perf_counter()
        results = search.search(test['query'], top_k=10, **test['filters'])
        latency = (time.perf_counter() - start) * 1000
        
        print(f"   Results: {len(results)} (latency: {latency:.1f}ms)")
        
        # Compute MRR
        expected = set(test['expected_ids'])
        reciprocal_rank = 0.0
        
        for i, r in enumerate(results, 1):
            if r.snippet.id in expected:
                reciprocal_rank = 1.0 / i
                break
        
        mrr_scores.append(reciprocal_rank)
        
        # Show results
        for i, r in enumerate(results[:3], 1):
            match = "✓" if r.snippet.id in expected else " "
            print(f"      {i}. [{match}] {r.snippet.function_name} ({r.snippet.language}) - score={r.score:.3f}")
        
        if reciprocal_rank > 0:
            print(f"   ✓ MRR: {reciprocal_rank:.2f}")
        else:
            print(f"   ✗ Expected result not found in top-k")
    
    # Summary
    mean_mrr = np.mean(mrr_scores)
    
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"\n  MRR@10: {mean_mrr:.3f}")
    print(f"  Queries with hit in top-10: {sum(1 for m in mrr_scores if m > 0)}/{len(mrr_scores)}")
    print("\n  ✓ Code search test completed!")
    print("="*70)


if __name__ == "__main__":
    run_code_search_test()
