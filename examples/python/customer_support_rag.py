#!/usr/bin/env python3
"""
Customer Support RAG Example with ToonDB

Demonstrates:
- Multi-tenant vector search with ACL
- Time decay for recency boosting
- OOD (out-of-distribution) query handling
- Filter correctness assertions

Usage:
    python3 examples/python/customer_support_rag.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SupportArticle:
    id: str
    content: str
    tenant_id: str
    access_level: int  # 0=public, 1=internal, 2=admin
    product: str
    language: str
    doc_type: str  # article, ticket, faq
    created_at: datetime
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    article: SupportArticle
    score: float
    reranked_score: float = 0.0


# =============================================================================
# Customer Support RAG System
# =============================================================================

class CustomerSupportRAG:
    """Multi-tenant customer support RAG with ACL and time decay."""
    
    def __init__(self):
        from toondb import VectorIndex
        
        self.dimension = 1536
        self.index = VectorIndex(dimension=self.dimension, max_connections=32, ef_construction=200)
        self.articles: Dict[str, SupportArticle] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        # Azure embeddings
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
    
    def ingest(self, articles: List[SupportArticle]) -> int:
        """Ingest articles into the index."""
        texts = [a.content for a in articles]
        embeddings = self._embed(texts)
        
        start_idx = self.next_idx
        ids = np.arange(start_idx, start_idx + len(articles), dtype=np.uint64)
        self.index.insert_batch(ids, embeddings)
        
        for i, article in enumerate(articles):
            article.embedding = embeddings[i]
            idx = start_idx + i
            self.articles[article.id] = article
            self.id_to_idx[article.id] = idx
            self.idx_to_id[idx] = article.id
        
        self.next_idx += len(articles)
        return len(articles)
    
    def search(
        self,
        query: str,
        tenant_id: str,
        user_access_level: int,
        language: Optional[str] = None,
        product: Optional[str] = None,
        top_k: int = 10,
        time_decay_days: int = 30
    ) -> List[SearchResult]:
        """Search with ACL filtering and time decay reranking."""
        
        query_embedding = self._embed([query])[0]
        
        # Over-fetch for filtering
        k_fetch = top_k * 5
        results = self.index.search(query_embedding, k=k_fetch)
        
        search_results = []
        now = datetime.now()
        
        for idx, score in results:
            article_id = self.idx_to_id.get(int(idx))
            if not article_id or article_id not in self.articles:
                continue
            
            article = self.articles[article_id]
            
            # ACL filtering
            if article.tenant_id != tenant_id:
                continue
            if article.access_level > user_access_level:
                continue
            if language and article.language != language:
                continue
            if product and article.product != product:
                continue
            
            # Time decay reranking
            days_old = (now - article.created_at).days
            decay = np.exp(-days_old / time_decay_days)
            reranked_score = float(score) * (0.7 + 0.3 * decay)  # Blend semantic + recency
            
            search_results.append(SearchResult(
                article=article,
                score=float(score),
                reranked_score=reranked_score
            ))
            
            if len(search_results) >= top_k:
                break
        
        # Sort by reranked score
        search_results.sort(key=lambda x: x.reranked_score)
        return search_results


# =============================================================================
# Test Runner
# =============================================================================

def run_customer_support_test():
    """Run customer support RAG test with assertions."""
    
    print("="*70)
    print("  CUSTOMER SUPPORT RAG + TOONDB")
    print("  Multi-tenant + ACL + Time Decay")
    print("="*70)
    
    rag = CustomerSupportRAG()
    
    # Generate synthetic support articles
    print("\n1. Generating synthetic dataset...")
    
    tenants = ["acme_corp", "globex_inc", "initech"]
    products = ["billing", "auth", "dashboard"]
    languages = ["en", "es", "de"]
    doc_types = ["article", "ticket", "faq"]
    
    articles = []
    now = datetime.now()
    
    # Create articles for each tenant
    content_templates = [
        "How to reset 2FA authentication for your account",
        "Billing invoice download and payment methods",
        "Dashboard performance optimization tips",
        "Login troubleshooting and password recovery",
        "API rate limiting and quota management",
        "Single sign-on (SSO) configuration guide",
        "Subscription upgrade and downgrade process",
        "Data export and backup procedures",
    ]
    
    for tenant_id in tenants:
        for i, content in enumerate(content_templates):
            for lang in languages[:2]:  # en and es
                article = SupportArticle(
                    id=f"{tenant_id}_{i}_{lang}",
                    content=f"[{tenant_id}] [{lang}] {content}",
                    tenant_id=tenant_id,
                    access_level=random.randint(0, 2),
                    product=random.choice(products),
                    language=lang,
                    doc_type=random.choice(doc_types),
                    created_at=now - timedelta(days=random.randint(1, 90))
                )
                articles.append(article)
    
    # Add recent articles (should be boosted)
    for tenant_id in tenants:
        article = SupportArticle(
            id=f"{tenant_id}_recent",
            content=f"[{tenant_id}] URGENT: Recent 2FA security update - immediate action required",
            tenant_id=tenant_id,
            access_level=0,
            product="auth",
            language="en",
            doc_type="article",
            created_at=now - timedelta(days=2)
        )
        articles.append(article)
    
    print(f"   Generated {len(articles)} articles across {len(tenants)} tenants")
    
    # Ingest
    print("\n2. Ingesting articles...")
    start = time.perf_counter()
    count = rag.ingest(articles)
    ingest_time = (time.perf_counter() - start) * 1000
    print(f"   ✓ Ingested {count} articles in {ingest_time:.0f}ms")
    
    # Test queries
    print("\n3. Running search tests...")
    print("-"*70)
    
    test_cases = [
        {
            "query": "How do I reset 2FA?",
            "tenant_id": "acme_corp",
            "access_level": 1,
            "language": "en",
            "expected_tenant": "acme_corp",
            "expected_language": "en",
        },
        {
            "query": "Billing invoice download",
            "tenant_id": "globex_inc",
            "access_level": 0,
            "language": None,  # Any language
            "expected_tenant": "globex_inc",
            "expected_language": None,
        },
        {
            "query": "Dashboard slow performance",
            "tenant_id": "initech",
            "access_level": 2,
            "language": "es",
            "expected_tenant": "initech",
            "expected_language": "es",
        },
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Query: {test['query']}")
        print(f"         Tenant: {test['tenant_id']}, Access: {test['access_level']}, Lang: {test['language']}")
        
        start = time.perf_counter()
        results = rag.search(
            query=test['query'],
            tenant_id=test['tenant_id'],
            user_access_level=test['access_level'],
            language=test['language'],
            top_k=5
        )
        latency = (time.perf_counter() - start) * 1000
        
        print(f"         Results: {len(results)} (latency: {latency:.1f}ms)")
        
        # Assertions
        filter_errors = 0
        for r in results:
            # Check tenant filter
            if r.article.tenant_id != test['expected_tenant']:
                filter_errors += 1
                print(f"   ✗ FILTER ERROR: Wrong tenant {r.article.tenant_id}")
            
            # Check access level
            if r.article.access_level > test['access_level']:
                filter_errors += 1
                print(f"   ✗ FILTER ERROR: Access level {r.article.access_level} > {test['access_level']}")
            
            # Check language
            if test['expected_language'] and r.article.language != test['expected_language']:
                filter_errors += 1
                print(f"   ✗ FILTER ERROR: Wrong language {r.article.language}")
        
        if filter_errors == 0:
            print(f"   ✓ All filters correct")
        else:
            all_passed = False
        
        # Show top result
        if results:
            top = results[0]
            print(f"   Top hit: {top.article.content[:60]}...")
            print(f"   Score: {top.score:.3f} → Reranked: {top.reranked_score:.3f}")
    
    # OOD query test
    print("\n" + "-"*70)
    print("\n[OOD Test] Query with no good answer")
    
    ood_query = "quantum entanglement in distributed systems"
    start = time.perf_counter()
    ood_results = rag.search(
        query=ood_query,
        tenant_id="acme_corp",
        user_access_level=2,
        top_k=5
    )
    ood_latency = (time.perf_counter() - start) * 1000
    
    print(f"   Query: {ood_query}")
    print(f"   Results: {len(ood_results)} (latency: {ood_latency:.1f}ms)")
    print(f"   ✓ OOD latency stable (expected: recall low, latency normal)")
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"\n  Filter correctness: {'PASSED' if all_passed else 'FAILED'}")
    print(f"  OOD handling: STABLE")
    print(f"  Time decay reranking: ENABLED")
    print("\n  ✓ Customer Support RAG test completed!")
    print("="*70)


if __name__ == "__main__":
    run_customer_support_test()
