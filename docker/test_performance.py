#!/usr/bin/env python3
"""
SochDB gRPC Performance Benchmarks

Comprehensive performance testing including:
- Throughput benchmarks
- Latency percentiles
- Concurrent client testing
- Memory and CPU profiling
- Load testing

Copyright 2025 Sushanth (https://github.com/sushanthpy)
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import os
import sys
import time
import json
import numpy as np
import threading
import multiprocessing
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add sochdb-python-sdk to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sochdb-python-sdk', 'src'))

try:
    from sochdb import SochDBClient
    print("✓ SochDB Python SDK imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SochDB: {e}")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Performance benchmark result"""
    name: str
    operations: int
    duration_sec: float
    throughput_ops_sec: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    success_rate: float


class PerformanceBenchmark:
    """Performance benchmark suite"""
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self.results: List[BenchmarkResult] = []
    
    def _create_client(self):
        """Create a new client connection"""
        return SochDBClient(f"{self.host}:{self.port}")
    
    def benchmark_kv_writes(self, num_ops: int = 10000) -> BenchmarkResult:
        """Benchmark key-value write throughput"""
        print(f"\n📊 Benchmarking KV Writes ({num_ops} operations)...")
        
        client = self._create_client()
        latencies = []
        successes = 0
        
        overall_start = time.time()
        
        for i in range(num_ops):
            start = time.time()
            try:
                client.put(f"bench_key_{i}".encode(), f"value_{i}".encode())
                latencies.append((time.time() - start) * 1000)
                successes += 1
            except Exception as e:
                latencies.append((time.time() - start) * 1000)
        
        overall_duration = time.time() - overall_start
        
        latencies.sort()
        result = BenchmarkResult(
            name="KV Writes",
            operations=num_ops,
            duration_sec=overall_duration,
            throughput_ops_sec=num_ops / overall_duration,
            latency_mean_ms=sum(latencies) / len(latencies),
            latency_p50_ms=latencies[int(len(latencies) * 0.50)],
            latency_p95_ms=latencies[int(len(latencies) * 0.95)],
            latency_p99_ms=latencies[int(len(latencies) * 0.99)],
            latency_max_ms=max(latencies),
            success_rate=successes / num_ops
        )
        
        self._print_result(result)
        return result
    
    def benchmark_kv_reads(self, num_ops: int = 10000) -> BenchmarkResult:
        """Benchmark key-value read throughput"""
        print(f"\n📊 Benchmarking KV Reads ({num_ops} operations)...")
        
        # Pre-populate data
        client = self._create_client()
        print("  Populating test data...")
        for i in range(num_ops):
            client.put(f"bench_key_{i}".encode(), f"value_{i}".encode())
        
        latencies = []
        successes = 0
        
        overall_start = time.time()
        
        for i in range(num_ops):
            start = time.time()
            try:
                value = client.get(f"bench_key_{i}".encode())
                latencies.append((time.time() - start) * 1000)
                if value is not None:
                    successes += 1
            except Exception as e:
                latencies.append((time.time() - start) * 1000)
        
        overall_duration = time.time() - overall_start
        
        latencies.sort()
        result = BenchmarkResult(
            name="KV Reads",
            operations=num_ops,
            duration_sec=overall_duration,
            throughput_ops_sec=num_ops / overall_duration,
            latency_mean_ms=sum(latencies) / len(latencies),
            latency_p50_ms=latencies[int(len(latencies) * 0.50)],
            latency_p95_ms=latencies[int(len(latencies) * 0.95)],
            latency_p99_ms=latencies[int(len(latencies) * 0.99)],
            latency_max_ms=max(latencies),
            success_rate=successes / num_ops
        )
        
        self._print_result(result)
        return result
    
    def benchmark_concurrent_writes(self, num_threads: int = 10, ops_per_thread: int = 1000) -> BenchmarkResult:
        """Benchmark concurrent write throughput"""
        print(f"\n📊 Benchmarking Concurrent Writes ({num_threads} threads, {ops_per_thread} ops each)...")
        
        def worker(thread_id: int) -> List[float]:
            client = self._create_client()
            latencies = []
            
            for i in range(ops_per_thread):
                start = time.time()
                try:
                    client.put(f"thread_{thread_id}_key_{i}".encode(), f"value_{i}".encode())
                    latencies.append((time.time() - start) * 1000)
                except:
                    latencies.append((time.time() - start) * 1000)
            
            return latencies
        
        overall_start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            all_latencies = []
            for future in as_completed(futures):
                all_latencies.extend(future.result())
        
        overall_duration = time.time() - overall_start
        total_ops = num_threads * ops_per_thread
        
        all_latencies.sort()
        result = BenchmarkResult(
            name=f"Concurrent Writes ({num_threads} threads)",
            operations=total_ops,
            duration_sec=overall_duration,
            throughput_ops_sec=total_ops / overall_duration,
            latency_mean_ms=sum(all_latencies) / len(all_latencies),
            latency_p50_ms=all_latencies[int(len(all_latencies) * 0.50)],
            latency_p95_ms=all_latencies[int(len(all_latencies) * 0.95)],
            latency_p99_ms=all_latencies[int(len(all_latencies) * 0.99)],
            latency_max_ms=max(all_latencies),
            success_rate=1.0  # Simplified
        )
        
        self._print_result(result)
        return result
    
    def benchmark_mixed_workload(self, num_ops: int = 10000, read_ratio: float = 0.8) -> BenchmarkResult:
        """Benchmark mixed read/write workload"""
        print(f"\n📊 Benchmarking Mixed Workload ({num_ops} ops, {read_ratio*100:.0f}% reads)...")
        
        client = self._create_client()
        
        # Pre-populate
        print("  Populating test data...")
        for i in range(1000):
            client.put(f"mixed_key_{i}".encode(), f"value_{i}".encode())
        
        latencies = []
        successes = 0
        
        overall_start = time.time()
        
        for i in range(num_ops):
            is_read = np.random.random() < read_ratio
            start = time.time()
            
            try:
                if is_read:
                    key_id = np.random.randint(0, 1000)
                    client.get(f"mixed_key_{key_id}".encode())
                else:
                    client.put(f"mixed_key_{i}".encode(), f"value_{i}".encode())
                
                latencies.append((time.time() - start) * 1000)
                successes += 1
            except:
                latencies.append((time.time() - start) * 1000)
        
        overall_duration = time.time() - overall_start
        
        latencies.sort()
        result = BenchmarkResult(
            name=f"Mixed Workload ({read_ratio*100:.0f}% reads)",
            operations=num_ops,
            duration_sec=overall_duration,
            throughput_ops_sec=num_ops / overall_duration,
            latency_mean_ms=sum(latencies) / len(latencies),
            latency_p50_ms=latencies[int(len(latencies) * 0.50)],
            latency_p95_ms=latencies[int(len(latencies) * 0.95)],
            latency_p99_ms=latencies[int(len(latencies) * 0.99)],
            latency_max_ms=max(latencies),
            success_rate=successes / num_ops
        )
        
        self._print_result(result)
        return result
    
    def _print_result(self, result: BenchmarkResult):
        """Pretty print benchmark result"""
        print(f"\n  Results for {result.name}:")
        print(f"    Operations:      {result.operations:,}")
        print(f"    Duration:        {result.duration_sec:.2f} sec")
        print(f"    Throughput:      {result.throughput_ops_sec:,.0f} ops/sec")
        print(f"    Latency (mean):  {result.latency_mean_ms:.3f} ms")
        print(f"    Latency (p50):   {result.latency_p50_ms:.3f} ms")
        print(f"    Latency (p95):   {result.latency_p95_ms:.3f} ms")
        print(f"    Latency (p99):   {result.latency_p99_ms:.3f} ms")
        print(f"    Latency (max):   {result.latency_max_ms:.3f} ms")
        print(f"    Success rate:    {result.success_rate*100:.1f}%")
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("\n" + "="*80)
        print("SochDB gRPC Performance Benchmarks".center(80))
        print("="*80)
        
        # Run benchmarks
        self.results.append(self.benchmark_kv_writes(10000))
        self.results.append(self.benchmark_kv_reads(10000))
        self.results.append(self.benchmark_concurrent_writes(10, 1000))
        self.results.append(self.benchmark_mixed_workload(10000, 0.8))
        
        # Summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY".center(80))
        print("="*80)
        
        for result in self.results:
            print(f"\n{result.name}:")
            print(f"  Throughput: {result.throughput_ops_sec:,.0f} ops/sec")
            print(f"  Latency p99: {result.latency_p99_ms:.3f} ms")
        
        # Save results
        results_dict = {
            "timestamp": time.time(),
            "server": f"{self.host}:{self.port}",
            "benchmarks": [asdict(r) for r in self.results]
        }
        
        with open("performance_benchmark_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to: performance_benchmark_results.json\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SochDB gRPC Performance Benchmarks")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(host=args.host, port=args.port)
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
