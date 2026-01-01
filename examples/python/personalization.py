#!/usr/bin/env python3
"""
Personalization / Similar Users with ToonDB

Demonstrates:
- User vector updates (session-based)
- Item catalog retrieval
- Two-stage retrieval (user → items)
- Cold-start handling

Usage:
    python3 examples/python/personalization.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
import random
from typing import List, Dict, Optional, Tuple
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
class User:
    id: str
    region: str
    device: str
    embedding: Optional[np.ndarray] = None
    interaction_count: int = 0  # For cold-start detection


@dataclass
class Item:
    id: str
    title: str
    category: str
    region: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Recommendation:
    item: Item
    score: float
    reason: str


# =============================================================================
# Personalization Engine
# =============================================================================

class PersonalizationEngine:
    """Two-stage personalization with user and item vectors."""
    
    def __init__(self, dimension: int = 128):
        from toondb import VectorIndex
        
        self.dimension = dimension
        
        # Separate indexes for users and items
        self.user_index = VectorIndex(dimension=dimension, max_connections=16, ef_construction=100)
        self.item_index = VectorIndex(dimension=dimension, max_connections=32, ef_construction=200)
        
        self.users: Dict[str, User] = {}
        self.items: Dict[str, Item] = {}
        
        self.user_id_to_idx: Dict[str, int] = {}
        self.user_idx_to_id: Dict[int, str] = {}
        self.item_id_to_idx: Dict[str, int] = {}
        self.item_idx_to_id: Dict[int, str] = {}
        
        self.next_user_idx = 0
        self.next_item_idx = 0
    
    def _generate_random_embedding(self, seed: int = None) -> np.ndarray:
        """Generate a deterministic random embedding."""
        if seed is not None:
            np.random.seed(seed)
        v = np.random.randn(self.dimension).astype(np.float32)
        return v / np.linalg.norm(v)
    
    def add_items(self, items: List[Item]) -> int:
        """Add items to the catalog."""
        embeddings = []
        
        for item in items:
            # Generate embedding based on item properties
            seed = hash(f"{item.id}_{item.category}") % (2**31)
            item.embedding = self._generate_random_embedding(seed)
            embeddings.append(item.embedding)
        
        start_idx = self.next_item_idx
        ids = np.arange(start_idx, start_idx + len(items), dtype=np.uint64)
        embeddings_arr = np.stack(embeddings)
        self.item_index.insert_batch(ids, embeddings_arr)
        
        for i, item in enumerate(items):
            idx = start_idx + i
            self.items[item.id] = item
            self.item_id_to_idx[item.id] = idx
            self.item_idx_to_id[idx] = item.id
        
        self.next_item_idx += len(items)
        return len(items)
    
    def add_users(self, users: List[User]) -> int:
        """Add users with initial embeddings."""
        embeddings = []
        
        for user in users:
            # Generate initial embedding
            seed = hash(f"user_{user.id}") % (2**31)
            user.embedding = self._generate_random_embedding(seed)
            embeddings.append(user.embedding)
        
        start_idx = self.next_user_idx
        ids = np.arange(start_idx, start_idx + len(users), dtype=np.uint64)
        embeddings_arr = np.stack(embeddings)
        self.user_index.insert_batch(ids, embeddings_arr)
        
        for i, user in enumerate(users):
            idx = start_idx + i
            self.users[user.id] = user
            self.user_id_to_idx[user.id] = idx
            self.user_idx_to_id[idx] = user.id
        
        self.next_user_idx += len(users)
        return len(users)
    
    def update_user(self, user_id: str, interaction_items: List[str], learning_rate: float = 0.3):
        """Update user embedding based on interactions."""
        if user_id not in self.users:
            return
        
        user = self.users[user_id]
        
        # Compute interaction signal
        item_embeddings = []
        for item_id in interaction_items:
            if item_id in self.items:
                item_embeddings.append(self.items[item_id].embedding)
        
        if not item_embeddings:
            return
        
        # Average item embedding
        interaction_signal = np.mean(item_embeddings, axis=0)
        
        # Blend with existing embedding
        new_embedding = (1 - learning_rate) * user.embedding + learning_rate * interaction_signal
        new_embedding = new_embedding / np.linalg.norm(new_embedding)
        
        user.embedding = new_embedding
        user.interaction_count += len(interaction_items)
        
        # Note: In production, you'd rebuild index or use an updatable index
    
    def get_recommendations(
        self,
        user_id: str,
        region: Optional[str] = None,
        top_k: int = 10
    ) -> List[Recommendation]:
        """Get personalized recommendations for a user."""
        
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        
        # Check cold-start
        is_cold_start = user.interaction_count < 5
        
        # Search item index with user embedding
        results = self.item_index.search(user.embedding, k=top_k * 3)
        
        recommendations = []
        
        for idx, score in results:
            item_id = self.item_idx_to_id.get(int(idx))
            if not item_id or item_id not in self.items:
                continue
            
            item = self.items[item_id]
            
            # Region filter
            if region and item.region != region:
                continue
            
            reason = "personalized" if not is_cold_start else "popular"
            
            recommendations.append(Recommendation(
                item=item,
                score=float(score),
                reason=reason
            ))
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def find_similar_users(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find users with similar preferences."""
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        results = self.user_index.search(user.embedding, k=top_k + 1)
        
        similar = []
        for idx, score in results:
            other_id = self.user_idx_to_id.get(int(idx))
            if other_id and other_id != user_id:
                similar.append((other_id, float(score)))
        
        return similar[:top_k]


# =============================================================================
# Test Runner
# =============================================================================

def run_personalization_test():
    """Run personalization test."""
    
    print("="*70)
    print("  PERSONALIZATION ENGINE + TOONDB")
    print("  User Vectors + Item Retrieval")
    print("="*70)
    
    engine = PersonalizationEngine(dimension=128)
    
    # Generate items
    print("\n1. Generating item catalog...")
    
    categories = ["electronics", "fashion", "home", "sports", "books"]
    regions = ["US", "EU", "APAC"]
    
    items = []
    for cat in categories:
        for i in range(20):
            item = Item(
                id=f"{cat}_{i}",
                title=f"{cat.title()} Item {i}",
                category=cat,
                region=random.choice(regions)
            )
            items.append(item)
    
    count = engine.add_items(items)
    print(f"   ✓ Added {count} items")
    
    # Generate users
    print("\n2. Generating users...")
    
    users = []
    for i in range(50):
        user = User(
            id=f"user_{i}",
            region=random.choice(regions),
            device=random.choice(["mobile", "desktop", "tablet"]),
            interaction_count=random.randint(0, 20)
        )
        users.append(user)
    
    count = engine.add_users(users)
    print(f"   ✓ Added {count} users")
    
    # Simulate interactions
    print("\n3. Simulating user interactions...")
    
    for user in users[:20]:
        # Each user interacts with items from 1-2 categories
        preferred_cats = random.sample(categories, random.randint(1, 2))
        interaction_items = [
            f"{cat}_{random.randint(0, 19)}"
            for cat in preferred_cats
            for _ in range(random.randint(3, 8))
        ]
        engine.update_user(user.id, interaction_items)
    
    print("   ✓ Updated user embeddings based on interactions")
    
    # Test recommendations
    print("\n4. Testing recommendations...")
    print("-"*70)
    
    test_users = ["user_0", "user_5", "user_49"]
    
    for user_id in test_users:
        user = engine.users[user_id]
        print(f"\n   User: {user_id} (interactions: {user.interaction_count})")
        
        start = time.perf_counter()
        recs = engine.get_recommendations(user_id, top_k=5)
        latency = (time.perf_counter() - start) * 1000
        
        print(f"   Recommendations ({latency:.1f}ms):")
        for i, rec in enumerate(recs[:3], 1):
            print(f"      {i}. {rec.item.title} ({rec.reason}) - score={rec.score:.3f}")
    
    # Test similar users
    print("\n" + "-"*70)
    print("\n5. Testing similar users...")
    
    for user_id in test_users[:1]:
        similar = engine.find_similar_users(user_id, top_k=3)
        print(f"\n   Similar to {user_id}:")
        for other_id, score in similar:
            other = engine.users[other_id]
            print(f"      - {other_id} (interactions: {other.interaction_count}) - distance={score:.3f}")
    
    # Cold-start test
    print("\n" + "-"*70)
    print("\n6. Cold-start handling...")
    
    cold_user = User(id="new_user", region="US", device="mobile", interaction_count=0)
    engine.add_users([cold_user])
    
    recs = engine.get_recommendations("new_user", top_k=5)
    print(f"   New user with 0 interactions: got {len(recs)} recommendations")
    if recs:
        print(f"   Reason: {recs[0].reason}")
    
    print("\n" + "="*70)
    print("  ✓ Personalization test completed!")
    print("="*70)


if __name__ == "__main__":
    run_personalization_test()
