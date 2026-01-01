#!/usr/bin/env python3
"""Parameter sensitivity analysis."""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from toondb.vector import VectorIndex

def test_parameter_sweep():
    print("=" * 80)
    print("HNSW PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    # Test configurations
    configs = [
        # (ef_construction, max_connections, description)
        (25, 8, "Ultra-fast (our optimization)"),
        (50, 16, "Balanced"),
        (100, 16, "High quality"),
        (48, 16, "Benchmark config"),
        (200, 16, "Original ToonDB default"),
    ]
    
    dimension = 768
    n_vectors = 1000  # Smaller for faster testing
    
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = np.arange(n_vectors, dtype=np.uint64)
    
    print(f"Testing {n_vectors} vectors of dimension {dimension}")
    print()
    
    results = []
    
    for ef_construction, max_connections, description in configs:
        print(f"Testing: {description}")
        print(f"  ef_construction={ef_construction}, max_connections={max_connections}")
        
        try:
            index = VectorIndex(
                dimension=dimension,
                max_connections=max_connections,
                ef_construction=ef_construction
            )
            
            # Warmup
            warmup_ids = np.arange(10, dtype=np.uint64)
            warmup_vecs = vectors[:10].copy()
            index.insert_batch_fast(warmup_ids + 100000, warmup_vecs)
            
            # Timed run
            start = time.perf_counter()
            inserted = index.insert_batch_fast(ids, vectors)
            elapsed = time.perf_counter() - start
            
            throughput = inserted / elapsed
            
            # Test search quality
            query_vector = vectors[0]
            search_results = index.search(query_vector, k=5)
            recall = len([r for r in search_results if r[1] < 0.1]) > 0 if search_results else False
            
            results.append({
                'config': description,
                'ef_construction': ef_construction,
                'max_connections': max_connections,
                'throughput': throughput,
                'time': elapsed,
                'recall': recall
            })
            
            print(f"  ⏱️  Time: {elapsed:.2f}s")
            print(f"  🚀 Throughput: {throughput:.0f} vec/s")
            print(f"  🎯 Self-recall: {'✅' if recall else '❌'}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Analysis
    print("=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)
    
    print("Performance Summary:")
    print(f"{'Configuration':<25} {'EF':<4} {'M':<3} {'Throughput':<12} {'Quality':<8}")
    print("-" * 55)
    
    best_throughput = max(results, key=lambda x: x['throughput'])
    
    for r in sorted(results, key=lambda x: x['throughput'], reverse=True):
        quality_mark = "✅" if r['recall'] else "❌"
        throughput_str = f"{r['throughput']:.0f} vec/s"
        print(f"{r['config']:<25} {r['ef_construction']:<4} {r['max_connections']:<3} {throughput_str:<12} {quality_mark:<8}")
    
    print()
    print("Key Insights:")
    
    # EF impact
    ultra_fast = next((r for r in results if r['ef_construction'] == 25), None)
    benchmark = next((r for r in results if r['config'] == "Benchmark config"), None)
    
    if ultra_fast and benchmark:
        speedup = ultra_fast['throughput'] / benchmark['throughput']
        print(f"  🔥 Ultra-fast config is {speedup:.1f}x faster than benchmark config")
    
    # Quality vs speed tradeoff
    fastest = max(results, key=lambda x: x['throughput'])
    print(f"  ⚡ Best performance: {fastest['config']} at {fastest['throughput']:.0f} vec/s")
    
    # Scaling estimate
    scale_factor = 10000 / n_vectors  # Benchmark uses 10k vectors
    estimated_10k = fastest['throughput'] / scale_factor
    
    print(f"  📊 Estimated 10K vector performance: ~{estimated_10k:.0f} vec/s")
    print(f"  🎯 ChromaDB comparison: {14303/estimated_10k:.1f}x gap at best settings")
    
    return results

if __name__ == '__main__':
    try:
        results = test_parameter_sweep()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()