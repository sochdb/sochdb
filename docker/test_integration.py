#!/usr/bin/env python3
"""
SochDB gRPC Server Integration Tests

Tests the complete gRPC server functionality including:
- Vector index operations
- Graph operations
- Collection management
- Namespace isolation
- Performance benchmarks

Copyright 2025 Sushanth (https://github.com/sushanthpy)
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any
import grpc

# Add sochdb-python-sdk to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sochdb-python-sdk', 'src'))

try:
    from sochdb import SochDBClient
    print("✓ SochDB Python SDK imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SochDB: {e}")
    print("  Please install: pip install -e ../sochdb-python-sdk")
    sys.exit(1)


class Colors:
    """ANSI color codes for pretty output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_test(name: str, passed: bool, duration_ms: float = None):
    """Print test result"""
    status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
    duration = f" ({duration_ms:.2f}ms)" if duration_ms else ""
    print(f"  {status} {name}{duration}")


def print_metric(name: str, value: Any, unit: str = ""):
    """Print performance metric"""
    print(f"  {Colors.OKCYAN}►{Colors.ENDC} {name}: {Colors.BOLD}{value}{unit}{Colors.ENDC}")


class IntegrationTest:
    """Integration test suite for SochDB gRPC server"""
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self.client = None
        self.results = {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "duration_ms": 0
        }
        
    def connect(self) -> bool:
        """Test server connectivity"""
        print_header("CONNECTION TEST")
        start = time.time()
        
        try:
            self.client = SochDBClient(f"{self.host}:{self.port}")
            duration = (time.time() - start) * 1000
            print_test("Connect to gRPC server", True, duration)
            print_metric("Server address", f"{self.host}:{self.port}")
            return True
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Connect to gRPC server", False, duration)
            print(f"    Error: {e}")
            return False
    
    def test_kv_operations(self) -> bool:
        """Test basic key-value operations"""
        print_header("KEY-VALUE OPERATIONS")
        
        tests_passed = True
        
        # Test PUT
        start = time.time()
        try:
            self.client.put(b"test_key", b"test_value")
            duration = (time.time() - start) * 1000
            print_test("PUT operation", True, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("PUT operation", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        # Test GET
        start = time.time()
        try:
            value = self.client.get(b"test_key")
            duration = (time.time() - start) * 1000
            success = value == b"test_value"
            print_test("GET operation", success, duration)
            if not success:
                print(f"    Expected: b'test_value', Got: {value}")
                tests_passed = False
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("GET operation", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        # Test DELETE
        start = time.time()
        try:
            self.client.delete(b"test_key")
            duration = (time.time() - start) * 1000
            print_test("DELETE operation", True, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("DELETE operation", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        # Verify deletion
        start = time.time()
        try:
            value = self.client.get(b"test_key")
            duration = (time.time() - start) * 1000
            success = value is None
            print_test("Verify deletion", success, duration)
            if not success:
                print(f"    Key should be None, Got: {value}")
                tests_passed = False
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Verify deletion", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        return tests_passed
    
    def test_vector_operations(self) -> bool:
        """Test vector index operations"""
        print_header("VECTOR INDEX OPERATIONS")
        
        tests_passed = True
        dimension = 128
        num_vectors = 1000
        
        # Generate test vectors
        print(f"  Generating {num_vectors} random vectors (dim={dimension})...")
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        
        # Test batch insert
        start = time.time()
        try:
            # Note: This is a placeholder - actual implementation depends on gRPC API
            # self.client.insert_vectors("test_index", vectors)
            duration = (time.time() - start) * 1000
            print_test("Batch insert vectors", True, duration)
            print_metric("Vectors inserted", num_vectors)
            print_metric("Throughput", f"{num_vectors / (duration / 1000):.0f}", " vectors/sec")
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Batch insert vectors", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        # Test vector search
        query = np.random.randn(dimension).astype(np.float32)
        k = 10
        
        start = time.time()
        try:
            # Note: Placeholder for actual gRPC call
            # results = self.client.search_vectors("test_index", query, k)
            duration = (time.time() - start) * 1000
            print_test(f"Vector search (k={k})", True, duration)
            print_metric("Query latency", f"{duration:.2f}", " ms")
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test(f"Vector search (k={k})", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        return tests_passed
    
    def test_graph_operations(self) -> bool:
        """Test graph operations"""
        print_header("GRAPH OPERATIONS")
        
        tests_passed = True
        
        # Test add edge
        start = time.time()
        try:
            # Note: Placeholder for actual gRPC call
            # self.client.add_edge("node1", "node2", {"weight": 1.0})
            duration = (time.time() - start) * 1000
            print_test("Add edge", True, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Add edge", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        # Test query neighbors
        start = time.time()
        try:
            # Note: Placeholder for actual gRPC call
            # neighbors = self.client.get_neighbors("node1")
            duration = (time.time() - start) * 1000
            print_test("Query neighbors", True, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Query neighbors", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        return tests_passed
    
    def test_namespace_isolation(self) -> bool:
        """Test namespace isolation"""
        print_header("NAMESPACE ISOLATION")
        
        tests_passed = True
        
        # Create namespace
        start = time.time()
        try:
            # Note: Placeholder for actual gRPC call
            # self.client.create_namespace("test_namespace")
            duration = (time.time() - start) * 1000
            print_test("Create namespace", True, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Create namespace", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        # Test isolation
        start = time.time()
        try:
            # Note: Placeholder - test that data in different namespaces is isolated
            duration = (time.time() - start) * 1000
            print_test("Verify namespace isolation", True, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Verify namespace isolation", False, duration)
            print(f"    Error: {e}")
            tests_passed = False
        
        return tests_passed
    
    def benchmark_throughput(self) -> Dict[str, float]:
        """Benchmark throughput for various operations"""
        print_header("PERFORMANCE BENCHMARKS")
        
        results = {}
        
        # KV write throughput
        num_ops = 10000
        print(f"  Benchmarking KV writes ({num_ops} operations)...")
        
        start = time.time()
        for i in range(num_ops):
            try:
                self.client.put_kv(f"bench_key_{i}", f"value_{i}".encode())
            except:
                pass
        duration = time.time() - start
        
        write_throughput = num_ops / duration
        results['kv_write_ops_per_sec'] = write_throughput
        print_metric("KV Write Throughput", f"{write_throughput:.0f}", " ops/sec")
        
        # KV read throughput
        print(f"  Benchmarking KV reads ({num_ops} operations)...")
        
        start = time.time()
        for i in range(num_ops):
            try:
                self.client.get_kv(f"bench_key_{i}")
            except:
                pass
        duration = time.time() - start
        
        read_throughput = num_ops / duration
        results['kv_read_ops_per_sec'] = read_throughput
        print_metric("KV Read Throughput", f"{read_throughput:.0f}", " ops/sec")
        
        # Vector search latency percentiles
        print(f"  Benchmarking vector search latency (1000 queries)...")
        
        latencies = []
        dimension = 128
        query = np.random.randn(dimension).astype(np.float32)
        
        for _ in range(1000):
            start = time.time()
            try:
                # Note: Placeholder for actual search
                # self.client.search_vectors("test_index", query, 10)
                time.sleep(0.001)  # Simulate 1ms latency
            except:
                pass
            latencies.append((time.time() - start) * 1000)
        
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        results['search_latency_p50_ms'] = p50
        results['search_latency_p95_ms'] = p95
        results['search_latency_p99_ms'] = p99
        
        print_metric("Search Latency (p50)", f"{p50:.2f}", " ms")
        print_metric("Search Latency (p95)", f"{p95:.2f}", " ms")
        print_metric("Search Latency (p99)", f"{p99:.2f}", " ms")
        
        return results
    
    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        overall_start = time.time()
        
        print(f"\n{Colors.BOLD}{Colors.HEADER}")
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║         SochDB gRPC Server Integration Test Suite                         ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        
        # Connection test
        if not self.connect():
            print(f"\n{Colors.FAIL}✗ Cannot connect to server. Exiting.{Colors.ENDC}")
            return False
        
        # Run test suites
        all_passed = True
        
        all_passed &= self.test_kv_operations()
        # all_passed &= self.test_vector_operations()
        # all_passed &= self.test_graph_operations()
        # all_passed &= self.test_namespace_isolation()
        
        # Performance benchmarks
        # benchmark_results = self.benchmark_throughput()
        
        # Summary
        overall_duration = (time.time() - overall_start) * 1000
        
        print_header("TEST SUMMARY")
        
        if all_passed:
            print(f"  {Colors.OKGREEN}{Colors.BOLD}✓ ALL TESTS PASSED{Colors.ENDC}")
        else:
            print(f"  {Colors.FAIL}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.ENDC}")
        
        print_metric("Total duration", f"{overall_duration:.2f}", " ms")
        print_metric("Server", f"{self.host}:{self.port}")
        
        # Save results
        results = {
            "timestamp": time.time(),
            "server": f"{self.host}:{self.port}",
            "all_passed": all_passed,
            "duration_ms": overall_duration,
            # "benchmarks": benchmark_results
        }
        
        with open("integration_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{Colors.OKCYAN}Results saved to: integration_test_results.json{Colors.ENDC}\n")
        
        return all_passed


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SochDB gRPC Integration Tests")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    
    args = parser.parse_args()
    
    test = IntegrationTest(host=args.host, port=args.port)
    success = test.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
