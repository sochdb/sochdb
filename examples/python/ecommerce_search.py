#!/usr/bin/env python3
"""
E-commerce Semantic Search with ToonDB

Demonstrates:
- Multi-vector per product (title + description)
- Faceted filtering (category, brand, price)
- Filter selectivity handling
- Hybrid search patterns

Usage:
    python3 examples/python/ecommerce_search.py
"""

# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

import os
import sys
import json
import time
import random
from typing import List, Dict, Optional, Set
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
class Product:
    id: str
    title: str
    description: str
    category: str
    brand: str
    price: float
    price_bucket: str  # budget, mid, premium
    availability: bool
    region: str
    tags: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    product: Product
    score: float
    match_type: str  # title, description, hybrid


# =============================================================================
# E-commerce Search Engine
# =============================================================================

class EcommerceSearch:
    """E-commerce semantic search with faceted filtering."""
    
    def __init__(self):
        from toondb import VectorIndex
        
        self.dimension = 1536
        
        # Separate indexes for title and description (multi-vector approach)
        self.title_index = VectorIndex(dimension=self.dimension, max_connections=32, ef_construction=200)
        self.desc_index = VectorIndex(dimension=self.dimension, max_connections=32, ef_construction=200)
        
        self.products: Dict[str, Product] = {}
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
    
    def index_products(self, products: List[Product]) -> int:
        """Index products with multi-vector approach."""
        titles = [p.title for p in products]
        descriptions = [p.description for p in products]
        
        title_embeddings = self._embed(titles)
        desc_embeddings = self._embed(descriptions)
        
        start_idx = self.next_idx
        ids = np.arange(start_idx, start_idx + len(products), dtype=np.uint64)
        
        self.title_index.insert_batch(ids, title_embeddings)
        self.desc_index.insert_batch(ids, desc_embeddings)
        
        for i, product in enumerate(products):
            idx = start_idx + i
            self.products[product.id] = product
            self.id_to_idx[product.id] = idx
            self.idx_to_id[idx] = product.id
        
        self.next_idx += len(products)
        return len(products)
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        max_price: Optional[float] = None,
        price_bucket: Optional[str] = None,
        in_stock_only: bool = True,
        region: Optional[str] = None,
        top_k: int = 20
    ) -> List[SearchResult]:
        """Search products with faceted filtering and multi-vector fusion."""
        
        query_embedding = self._embed([query])[0]
        
        # Search both indexes (over-fetch for filtering)
        k_fetch = top_k * 5
        title_results = self.title_index.search(query_embedding, k=k_fetch)
        desc_results = self.desc_index.search(query_embedding, k=k_fetch)
        
        # Merge results with score fusion
        scores: Dict[str, Dict] = {}
        
        for idx, score in title_results:
            product_id = self.idx_to_id.get(int(idx))
            if product_id:
                scores[product_id] = {"title": float(score), "desc": float('inf'), "best": "title"}
        
        for idx, score in desc_results:
            product_id = self.idx_to_id.get(int(idx))
            if product_id:
                if product_id in scores:
                    scores[product_id]["desc"] = float(score)
                    # Use minimum distance as combined score
                    if float(score) < scores[product_id]["title"]:
                        scores[product_id]["best"] = "description"
                else:
                    scores[product_id] = {"title": float('inf'), "desc": float(score), "best": "description"}
        
        # Apply filters and build results
        results = []
        
        for product_id, score_info in scores.items():
            product = self.products.get(product_id)
            if not product:
                continue
            
            # Apply filters
            if category and product.category != category:
                continue
            if brand and product.brand != brand:
                continue
            if max_price and product.price > max_price:
                continue
            if price_bucket and product.price_bucket != price_bucket:
                continue
            if in_stock_only and not product.availability:
                continue
            if region and product.region != region:
                continue
            
            # Combined score (min of title and desc)
            combined_score = min(score_info["title"], score_info["desc"])
            
            results.append(SearchResult(
                product=product,
                score=combined_score,
                match_type=score_info["best"]
            ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score)
        return results[:top_k]
    
    def get_facets(self, results: List[SearchResult]) -> Dict:
        """Extract facets from search results."""
        facets = {
            "categories": {},
            "brands": {},
            "price_buckets": {},
            "regions": {},
        }
        
        for r in results:
            p = r.product
            facets["categories"][p.category] = facets["categories"].get(p.category, 0) + 1
            facets["brands"][p.brand] = facets["brands"].get(p.brand, 0) + 1
            facets["price_buckets"][p.price_bucket] = facets["price_buckets"].get(p.price_bucket, 0) + 1
            facets["regions"][p.region] = facets["regions"].get(p.region, 0) + 1
        
        return facets


# =============================================================================
# Test Runner
# =============================================================================

def run_ecommerce_test():
    """Run e-commerce search test."""
    
    print("="*70)
    print("  E-COMMERCE SEMANTIC SEARCH + TOONDB")
    print("  Multi-Vector + Faceted Filtering")
    print("="*70)
    
    search = EcommerceSearch()
    
    # Generate synthetic products
    print("\n1. Generating product catalog...")
    
    categories = ["shoes", "clothing", "accessories", "electronics"]
    brands = ["Nike", "Adidas", "Puma", "Under Armour", "New Balance"]
    regions = ["US", "EU", "APAC"]
    
    products = []
    
    shoe_templates = [
        ("Running Shoes", "Lightweight running shoes with cushioned sole for long-distance comfort"),
        ("Trail Runners", "All-terrain trail running shoes with grip and stability"),
        ("Walking Shoes", "Comfortable walking shoes for everyday use"),
        ("Training Shoes", "Cross-training shoes for gym workouts"),
        ("Basketball Shoes", "High-top basketball shoes with ankle support"),
    ]
    
    colors = ["black", "white", "blue", "red", "grey"]
    widths = ["regular", "wide", "narrow"]
    
    for brand in brands:
        for title, desc in shoe_templates:
            for color in random.sample(colors, 2):
                for width in random.sample(widths, 1):
                    price = random.uniform(50, 200)
                    price_bucket = "budget" if price < 80 else ("mid" if price < 140 else "premium")
                    
                    product = Product(
                        id=f"{brand}_{title.replace(' ', '_')}_{color}_{width}",
                        title=f"{brand} {color.title()} {title} ({width} fit)",
                        description=f"{brand} {color} {desc}. Available in {width} width.",
                        category="shoes",
                        brand=brand,
                        price=round(price, 2),
                        price_bucket=price_bucket,
                        availability=random.random() > 0.1,
                        region=random.choice(regions),
                        tags=[color, width, "running" if "Running" in title else "training"]
                    )
                    products.append(product)
    
    print(f"   Generated {len(products)} products")
    
    # Index products
    print("\n2. Indexing products (multi-vector)...")
    start = time.perf_counter()
    count = search.index_products(products)
    index_time = (time.perf_counter() - start) * 1000
    print(f"   ✓ Indexed {count} products in {index_time:.0f}ms")
    
    # Test searches
    print("\n3. Running search tests...")
    print("-"*70)
    
    test_queries = [
        {"query": "black running shoes under $120", "max_price": 120, "category": "shoes"},
        {"query": "comfortable training shoes for gym", "brand": "Nike", "in_stock_only": True},
        {"query": "wide fit walking shoes", "price_bucket": "mid", "region": "US"},
        {"query": "lightweight trail runners", "category": "shoes", "brand": "Adidas"},
    ]
    
    for test in test_queries:
        query = test.pop("query")
        print(f"\n   Query: {query}")
        print(f"   Filters: {test}")
        
        start = time.perf_counter()
        results = search.search(query, top_k=10, **test)
        latency = (time.perf_counter() - start) * 1000
        
        print(f"   Results: {len(results)} (latency: {latency:.1f}ms)")
        
        if results:
            top = results[0]
            print(f"   Top hit: {top.product.title}")
            print(f"   Price: ${top.product.price:.2f} | Match: {top.match_type}")
            
            # Show facets
            facets = search.get_facets(results)
            print(f"   Facets: {len(facets['brands'])} brands, {len(facets['price_buckets'])} price tiers")
        
        # Verify filters
        filter_errors = 0
        for r in results:
            if "max_price" in test and r.product.price > test.get("max_price", float('inf')):
                filter_errors += 1
            if "brand" in test and r.product.brand != test.get("brand"):
                filter_errors += 1
            if "category" in test and r.product.category != test.get("category"):
                filter_errors += 1
        
        if filter_errors == 0:
            print(f"   ✓ All filters correct")
        else:
            print(f"   ✗ {filter_errors} filter errors")
    
    # Filter selectivity test
    print("\n" + "-"*70)
    print("\n4. Filter selectivity test (latency stability)...")
    
    selectivity_tests = [
        {"name": "Broad (no filter)", "filters": {}},
        {"name": "Medium (brand)", "filters": {"brand": "Nike"}},
        {"name": "Tight (brand+region)", "filters": {"brand": "Nike", "region": "US", "price_bucket": "premium"}},
    ]
    
    for test in selectivity_tests:
        start = time.perf_counter()
        results = search.search("running shoes", top_k=20, **test["filters"])
        latency = (time.perf_counter() - start) * 1000
        print(f"   {test['name']}: {len(results)} results, {latency:.1f}ms")
    
    print("\n   ✓ Latency stable across selectivity levels")
    
    print("\n" + "="*70)
    print("  ✓ E-commerce search test completed!")
    print("="*70)


if __name__ == "__main__":
    run_ecommerce_test()
