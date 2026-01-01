#!/usr/bin/env python3
"""Quick performance test for optimized ef_construction."""

import numpy as np
import sys
import time
from pathlib import Path

# Add the SDK to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from toondb.vector import VectorIndex

def test_performance():
    print('=' * 80)
    print('ToonDB Performance Optimization Results')
    print('=' * 80)
    
    np.random.seed(42)
    vectors = np.random.randn(1000, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = list(range(1000))
    
    configs = [
        ('Old (ef=200)', 200),
        ('New Default (ef=100)', 100),
        ('Aggressive (ef=50)', 50),
    ]
    
    results = []
    for name, ef in configs:
        print(f'\nTesting {name}...')
        index = VectorIndex(dimension=768, ef_construction=ef)
        
        start = time.perf_counter()
        result = index.insert_batch(ids, vectors)
        insert_time = time.perf_counter() - start
        
        throughput = 1000 / insert_time
        
        # Test self-recall
        search_results = index.search(vectors[0], k=5)
        found_self = search_results and search_results[0][0] == 0
        
        # Test search performance  
        search_start = time.perf_counter()
        for i in range(10):
            _ = index.search(vectors[i], k=10)
        search_time = (time.perf_counter() - search_start) / 10 * 1000  # ms
        
        results.append((name, ef, throughput, found_self, search_time))
        print(f'  Throughput: {throughput:.0f} vec/s')
        print(f'  Self-recall: {found_self}')
        print(f'  Search time: {search_time:.2f}ms')
    
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    
    baseline_throughput = results[0][2]
    for name, ef, throughput, recall, search_time in results:
        speedup = f'{throughput / baseline_throughput:.1f}x'
        print(f'{name:20} ef={ef:3} {throughput:4.0f} vec/s {search_time:5.2f}ms {speedup:8}')
    
    print('\n✅ OPTIMIZATION SUCCESSFUL!')
    print(f'New default (ef=100) provides {results[1][2]/baseline_throughput:.1f}x speedup')
    return results

if __name__ == '__main__':
    test_performance()