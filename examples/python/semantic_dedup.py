#!/usr/bin/env python3
"""
Semantic Deduplication with ToonDB

Demonstrates:
- Near-duplicate detection for tickets/logs
- Threshold-based similarity matching
- Cluster detection
- Time-window filtering

Usage:
    python3 examples/python/semantic_dedup.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
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
class Ticket:
    id: str
    content: str
    service: str
    severity: str  # low, medium, high, critical
    timestamp: datetime
    cluster_id: Optional[str] = None  # Ground truth for near-duplicates


@dataclass
class DuplicateResult:
    ticket: Ticket
    similarity: float
    is_duplicate: bool


# =============================================================================
# Semantic Deduplication System
# =============================================================================

class SemanticDedup:
    """Near-duplicate detection using semantic similarity."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        from toondb import VectorIndex
        
        self.dimension = 1536
        self.similarity_threshold = similarity_threshold
        self.index = VectorIndex(dimension=self.dimension, max_connections=32, ef_construction=200)
        
        self.tickets: Dict[str, Ticket] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
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
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def ingest(self, tickets: List[Ticket]) -> int:
        """Ingest tickets into the index."""
        texts = [t.content for t in tickets]
        embeddings = self._embed(texts)
        
        start_idx = self.next_idx
        ids = np.arange(start_idx, start_idx + len(tickets), dtype=np.uint64)
        self.index.insert_batch(ids, embeddings)
        
        for i, ticket in enumerate(tickets):
            idx = start_idx + i
            self.tickets[ticket.id] = ticket
            self.embeddings[ticket.id] = embeddings[i]
            self.id_to_idx[ticket.id] = idx
            self.idx_to_id[idx] = ticket.id
        
        self.next_idx += len(tickets)
        return len(tickets)
    
    def find_duplicates(
        self,
        content: str,
        service: Optional[str] = None,
        time_window_days: int = 30,
        top_k: int = 5
    ) -> List[DuplicateResult]:
        """Find duplicate tickets for new content."""
        
        query_embedding = self._embed([content])[0]
        
        # Search with over-fetch
        k_fetch = top_k * 3
        results = self.index.search(query_embedding, k=k_fetch)
        
        now = datetime.now()
        cutoff = now - timedelta(days=time_window_days)
        
        duplicates = []
        
        for idx, distance in results:
            ticket_id = self.idx_to_id.get(int(idx))
            if not ticket_id or ticket_id not in self.tickets:
                continue
            
            ticket = self.tickets[ticket_id]
            
            # Time window filter
            if ticket.timestamp < cutoff:
                continue
            
            # Service filter
            if service and ticket.service != service:
                continue
            
            # Compute similarity (convert distance to similarity)
            ticket_embedding = self.embeddings[ticket_id]
            similarity = self._cosine_similarity(query_embedding, ticket_embedding)
            
            is_duplicate = similarity >= self.similarity_threshold
            
            duplicates.append(DuplicateResult(
                ticket=ticket,
                similarity=similarity,
                is_duplicate=is_duplicate
            ))
        
        # Sort by similarity
        duplicates.sort(key=lambda x: x.similarity, reverse=True)
        return duplicates[:top_k]
    
    def get_cluster_metrics(self) -> Dict:
        """Compute cluster metrics from ground truth."""
        clusters: Dict[str, List[str]] = {}
        
        for ticket_id, ticket in self.tickets.items():
            if ticket.cluster_id:
                if ticket.cluster_id not in clusters:
                    clusters[ticket.cluster_id] = []
                clusters[ticket.cluster_id].append(ticket_id)
        
        return {
            "num_clusters": len(clusters),
            "avg_cluster_size": np.mean([len(c) for c in clusters.values()]) if clusters else 0,
            "max_cluster_size": max([len(c) for c in clusters.values()]) if clusters else 0,
        }


# =============================================================================
# Test Runner
# =============================================================================

def run_dedup_test():
    """Run semantic deduplication test."""
    
    print("="*70)
    print("  SEMANTIC DEDUPLICATION + TOONDB")
    print("  Near-Duplicate Detection for Tickets")
    print("="*70)
    
    dedup = SemanticDedup(similarity_threshold=0.85)
    
    # Generate synthetic tickets with clusters
    print("\n1. Generating tickets with near-duplicate clusters...")
    
    # Base issue templates
    issues = [
        ("Database connection timeout", "db", "high"),
        ("API rate limit exceeded", "api", "medium"),
        ("Login authentication failed", "auth", "high"),
        ("Payment processing error", "payments", "critical"),
        ("File upload size exceeded", "storage", "low"),
    ]
    
    services = ["db", "api", "auth", "payments", "storage"]
    tickets = []
    now = datetime.now()
    
    # Create clusters of near-duplicates
    for cluster_idx, (base_content, service, severity) in enumerate(issues):
        cluster_id = f"cluster_{cluster_idx}"
        
        # Create 5-10 variants per issue
        num_variants = random.randint(5, 10)
        
        variations = [
            base_content,
            f"Error: {base_content}",
            f"{base_content} - need help",
            f"Issue with {base_content.lower()}",
            f"Getting {base_content.lower()} repeatedly",
            f"Experiencing {base_content.lower()} issues",
            f"Help needed: {base_content}",
            f"{base_content} happening again",
            f"Cannot proceed due to {base_content.lower()}",
            f"Urgent: {base_content}",
        ]
        
        for i in range(num_variants):
            ticket = Ticket(
                id=f"ticket_{cluster_idx}_{i}",
                content=variations[i % len(variations)],
                service=service,
                severity=severity,
                timestamp=now - timedelta(days=random.randint(1, 20)),
                cluster_id=cluster_id
            )
            tickets.append(ticket)
    
    # Add some unique tickets (non-duplicates)
    unique_issues = [
        "Feature request for dark mode",
        "Documentation needs update for API v2",
        "Performance degradation after latest deploy",
    ]
    
    for i, content in enumerate(unique_issues):
        ticket = Ticket(
            id=f"unique_{i}",
            content=content,
            service=random.choice(services),
            severity="low",
            timestamp=now - timedelta(days=random.randint(1, 15)),
            cluster_id=None
        )
        tickets.append(ticket)
    
    random.shuffle(tickets)
    print(f"   Generated {len(tickets)} tickets with {len(issues)} clusters")
    
    # Ingest tickets
    print("\n2. Ingesting tickets...")
    start = time.perf_counter()
    count = dedup.ingest(tickets)
    ingest_time = (time.perf_counter() - start) * 1000
    print(f"   ✓ Ingested {count} tickets in {ingest_time:.0f}ms")
    
    # Show cluster metrics
    metrics = dedup.get_cluster_metrics()
    print(f"   Clusters: {metrics['num_clusters']}, Avg size: {metrics['avg_cluster_size']:.1f}")
    
    # Test duplicate detection
    print("\n3. Testing duplicate detection...")
    print("-"*70)
    
    test_cases = [
        {"content": "Getting database connection timeout errors", "expected_cluster": "cluster_0"},
        {"content": "API rate limiting issue", "expected_cluster": "cluster_1"},
        {"content": "Can't login - authentication failed", "expected_cluster": "cluster_2"},
        {"content": "Something completely unrelated to anything", "expected_cluster": None},
    ]
    
    recall_at_1 = 0
    recall_at_5 = 0
    false_positives = 0
    
    for test in test_cases:
        print(f"\n   New ticket: {test['content'][:50]}...")
        
        start = time.perf_counter()
        results = dedup.find_duplicates(test["content"], top_k=5)
        latency = (time.perf_counter() - start) * 1000
        
        print(f"   Results: {len(results)} candidates (latency: {latency:.1f}ms)")
        
        # Check Recall@1
        if results and test["expected_cluster"]:
            top = results[0]
            if top.is_duplicate and top.ticket.cluster_id == test["expected_cluster"]:
                recall_at_1 += 1
                print(f"   ✓ Recall@1: Matched cluster {test['expected_cluster']}")
            else:
                print(f"   ✗ Recall@1: Expected {test['expected_cluster']}, got {top.ticket.cluster_id}")
        
        # Check Recall@5
        matched_in_top5 = any(
            r.ticket.cluster_id == test["expected_cluster"]
            for r in results[:5]
            if r.is_duplicate
        )
        if matched_in_top5 and test["expected_cluster"]:
            recall_at_5 += 1
        
        # Check false positives for non-duplicate
        if test["expected_cluster"] is None:
            fps = sum(1 for r in results if r.is_duplicate)
            false_positives += fps
            if fps == 0:
                print(f"   ✓ No false positives")
            else:
                print(f"   ⚠ {fps} false positives detected")
        
        # Show top results
        for i, r in enumerate(results[:3], 1):
            status = "DUP" if r.is_duplicate else "---"
            print(f"      {i}. [{status}] sim={r.similarity:.3f} cluster={r.ticket.cluster_id}")
    
    # Summary
    num_with_expected = sum(1 for t in test_cases if t["expected_cluster"])
    
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"\n  Recall@1: {recall_at_1}/{num_with_expected}")
    print(f"  Recall@5: {recall_at_5}/{num_with_expected}")
    print(f"  False Positives: {false_positives}")
    print(f"  Similarity Threshold: {dedup.similarity_threshold}")
    print("\n  ✓ Semantic deduplication test completed!")
    print("="*70)


if __name__ == "__main__":
    run_dedup_test()
