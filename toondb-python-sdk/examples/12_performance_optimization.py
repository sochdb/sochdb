#!/usr/bin/env python3
"""
ToonDB Performance Optimization Test

This script tests different ef_construction values to find the optimal
balance between insert speed and recall quality.

The profiling shows neighbor selection consumes 80% of insert time,
which is directly proportional to ef_construction.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add the SDK to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from toondb.vector import VectorIndex
except ImportError:
    print("Error: Could not import VectorIndex")
    print("Make sure TOONDB_LIB_PATH is set to the compiled library")
    sys.exit(1)


def test_performance_vs_quality(
    vectors,
    ef_values=[50, 100, 150, 200],
    max_connections=16,
    num_test_vectors=1000
):
    """Test different ef_construction values for performance vs quality trade-off."""
    
    results = []
    
    print("=" * 80)
    print("ToonDB Performance vs Quality Optimization")
    print("=" * 80)
    print(f"Testing ef_construction values: {ef_values}")
    print(f"Vectors: {num_test_vectors}, Dimension: {vectors.shape[1]}")
    print()
    
    # Generate test query
    query_vector = np.random.randn(vectors.shape[1]).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Test each ef_construction value
    for ef in ef_values:
        print(f"Testing ef_construction = {ef}")
        print("-" * 50)
        
        # Create index
        index = VectorIndex(
            dimension=vectors.shape[1],
            ef_construction=ef,
            max_connections=max_connections,
        )
        
        # Prepare test data
        test_vectors = vectors[:num_test_vectors]
        ids = list(range(num_test_vectors))
        
        # Convert to numpy array if needed
        if not isinstance(test_vectors, np.ndarray):
            test_vectors = np.array(test_vectors, dtype=np.float32)
        
        # Measure insert time
        start_time = time.perf_counter()
        result = index.insert_batch(ids, test_vectors)  # Fixed: ids first, then vectors
        insert_time = time.perf_counter() - start_time
        
        throughput = num_test_vectors / insert_time
        
        print(f"  Insert time: {insert_time:.3f}s")
        print(f"  Throughput: {throughput:.0f} vec/sec")
        print(f"  Result: {result}")
        
        # Measure search performance
        search_times = []
        for _ in range(10):  # Average over 10 searches
            start = time.perf_counter()
            search_results = index.search(query_vector, k=10)
            search_times.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_search_time = sum(search_times) / len(search_times)
        
        print(f"  Search time: {avg_search_time:.3f}ms")
        print(f"  Search results: {len(search_results) if search_results else 0}")
        
        # Test recall by searching for inserted vectors
        recall_scores = []
        for i in range(min(100, num_test_vectors)):  # Test 100 vectors for recall
            test_vec = test_vectors[i]
            search_results = index.search(test_vec, k=10)
            if search_results and search_results[0][0] == i:  # Should find itself first
                recall_scores.append(1.0)
            else:
                recall_scores.append(0.0)
        
        self_recall = sum(recall_scores) / len(recall_scores) * 100
        
        print(f"  Self-recall: {self_recall:.1f}%")
        
        results.append({
            'ef_construction': ef,
            'insert_time': insert_time,
            'throughput': throughput,
            'search_time_ms': avg_search_time,
            'self_recall_pct': self_recall,
            'vectors_inserted': result if isinstance(result, int) else num_test_vectors
        })
        
        print()
    
    return results


def print_optimization_report(results):
    """Print detailed optimization recommendations."""
    
    print("=" * 80)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    # Performance vs Quality Table
    print(f"{'EF':<6} {'Throughput':<12} {'Search ms':<10} {'Self-Recall':<12} {'Speedup':<10}")
    print("-" * 60)
    
    baseline_throughput = None
    for r in results:
        if baseline_throughput is None:
            baseline_throughput = r['throughput']
            speedup = "1.0x"
        else:
            speedup = f"{r['throughput'] / baseline_throughput:.1f}x"
        
        print(f"{r['ef_construction']:<6} "
              f"{r['throughput']:.0f} vec/s{'':<4} "
              f"{r['search_time_ms']:.3f}{'':<6} "
              f"{r['self_recall_pct']:.1f}%{'':<8} "
              f"{speedup}")
    
    print()
    
    # Find optimal configurations
    best_throughput = max(results, key=lambda x: x['throughput'])
    best_recall = max(results, key=lambda x: x['self_recall_pct'])
    
    # Find balanced option (good recall with significant speedup)
    balanced = None
    for r in results:
        if r['self_recall_pct'] >= 95.0:  # Good recall
            if balanced is None or r['throughput'] > balanced['throughput']:
                balanced = r
    
    print("RECOMMENDATIONS:")
    print("-" * 40)
    print(f"📈 Best Throughput: ef={best_throughput['ef_construction']} "
          f"({best_throughput['throughput']:.0f} vec/s, "
          f"{best_throughput['self_recall_pct']:.1f}% recall)")
    
    print(f"🎯 Best Recall: ef={best_recall['ef_construction']} "
          f"({best_recall['throughput']:.0f} vec/s, "
          f"{best_recall['self_recall_pct']:.1f}% recall)")
    
    if balanced:
        print(f"⚖️  Balanced: ef={balanced['ef_construction']} "
              f"({balanced['throughput']:.0f} vec/s, "
              f"{balanced['self_recall_pct']:.1f}% recall)")
    
    # Calculate potential speedup vs current (ef=200)
    current_config = next((r for r in results if r['ef_construction'] == 200), None)
    if current_config and best_throughput['ef_construction'] != 200:
        speedup = best_throughput['throughput'] / current_config['throughput']
        print(f"\n💡 Switching from ef=200 to ef={best_throughput['ef_construction']} "
              f"would give {speedup:.1f}x speedup")
    
    print()


def save_results(results, filename="performance_optimization_results.json"):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {filename}")


def main():
    # Generate test data
    print("Generating test vectors...")
    np.random.seed(42)
    dimension = 768  # OpenAI embedding size
    num_vectors = 2000  # Enough for testing different ef values
    
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    # Normalize to unit vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    print(f"Generated {num_vectors} vectors of dimension {dimension}")
    print()
    
    # Test different ef_construction values
    ef_values = [50, 100, 150, 200]  # Test range from fast to current
    results = test_performance_vs_quality(vectors, ef_values, num_test_vectors=1000)
    
    # Print analysis
    print_optimization_report(results)
    
    # Save results
    save_results(results)
    
    # Additional insight
    print("=" * 80)
    print("PROFILING INSIGHT")
    print("=" * 80)
    print("""
Our previous profiling showed that neighbor selection consumes 80% of insert time.
This is directly proportional to ef_construction because:

1. Search returns ~ef candidates per layer
2. RNG heuristic checks each candidate against all selected neighbors  
3. Each check requires a full 768-dimensional cosine distance calculation

Reducing ef_construction from 200 to 100 should approximately halve the
neighbor selection time, giving ~1.6x overall speedup while maintaining
good recall quality.

For production use, consider:
- ef_construction = 100-128 for balanced performance/quality
- ef_construction = 50-75 for maximum speed with acceptable recall
- Keep max_connections = 16 (already optimal)
""")


if __name__ == '__main__':
    main()