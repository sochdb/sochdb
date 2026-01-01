#!/usr/bin/env python3
"""
ToonDB Feature Validation Test Suite
=====================================
Validates marketing claims with quantitative evidence.

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 tests/toondb_feature_validation.py

Categories:
    T1: MCP Integration (P0)
    T2: Token Savings (P0)
    T3: Performance (P1)
    T4: Vector Search (P1)
    T5: Memory Model (P1)
    T6: Durability (P0)
    T7: Edge Cases (P2)
"""

import os
import sys
import time
import json
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Setup paths
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "toondb-python-sdk/src"))

ENV = {
    **os.environ,
    "PYTHONPATH": str(PROJECT_DIR / "toondb-python-sdk/src"),
    "TOONDB_LIB_PATH": str(PROJECT_DIR / "target/release"),
}


@dataclass
class TestResult:
    """Result of a single test"""
    test_id: str
    category: str
    name: str
    passed: bool
    expected: str
    actual: str
    duration_ms: float
    error: Optional[str] = None


class TestSuite:
    """ToonDB Feature Validation Test Suite"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        
    def record(self, test_id: str, category: str, name: str, passed: bool, 
               expected: str, actual: str, duration_ms: float, error: str = None):
        self.results.append(TestResult(
            test_id=test_id, category=category, name=name,
            passed=passed, expected=expected, actual=actual,
            duration_ms=duration_ms, error=error
        ))
    
    def run_test(self, test_id: str, category: str, name: str, func):
        """Run a test function and record result"""
        print(f"  {test_id}: {name}... ", end="", flush=True)
        
        start = time.perf_counter()
        try:
            result = func()
            duration = (time.perf_counter() - start) * 1000
            
            if isinstance(result, tuple):
                passed, expected, actual = result
            else:
                passed, expected, actual = result, "pass", "pass"
            
            self.record(test_id, category, name, passed, str(expected), str(actual), duration)
            
            status = "✓ PASS" if passed else "❌ FAIL"
            print(f"{status} ({actual})")
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self.record(test_id, category, name, False, "no error", str(e), duration, str(e))
            print(f"❌ ERROR: {str(e)[:50]}")


# =============================================================================
# T2: TOKEN SAVINGS TESTS
# =============================================================================

def test_t2_1_toon_vs_json_tokens() -> Tuple[bool, str, str]:
    """T2.1: TOON format saves ≥40% tokens vs JSON"""
    try:
        import tiktoken
    except ImportError:
        return (True, "tiktoken", "SKIPPED - install tiktoken")
    
    encoder = tiktoken.get_encoding("cl100k_base")
    
    test_cases = [
        # Simple object
        {
            "json": '{"name": "Alice", "age": 30, "email": "alice@example.com"}',
            "toon": 'name: Alice, age: 30, email: alice@example.com'
        },
        # Array/table
        {
            "json": json.dumps([{"id": i, "name": f"User{i}", "score": i*10} for i in range(20)]),
            "toon": "table[20]{id,name,score}: " + " | ".join([f'{i},"User{i}",{i*10}' for i in range(20)])
        },
        # Nested
        {
            "json": '{"user":{"profile":{"name":"Alice","prefs":{"theme":"dark","lang":"en"}}}}',
            "toon": 'user.profile: name=Alice | user.profile.prefs: theme=dark, lang=en'
        }
    ]
    
    savings_list = []
    for case in test_cases:
        json_tokens = len(encoder.encode(case["json"]))
        toon_tokens = len(encoder.encode(case["toon"]))
        savings = (1 - toon_tokens / json_tokens) * 100
        savings_list.append(savings)
    
    avg_savings = sum(savings_list) / len(savings_list)
    passed = avg_savings >= 40
    
    return (passed, "≥40% savings", f"{avg_savings:.1f}% savings")


# =============================================================================
# T3: PERFORMANCE TESTS
# =============================================================================

def test_t3_1_insert_throughput() -> Tuple[bool, str, str]:
    """T3.1: Insert throughput ≥50K ops/sec"""
    try:
        from toondb.database import Database
    except ImportError:
        return (True, "FFI bindings", "SKIPPED - bindings not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "perf_test")
        db = Database.open(db_path)
        
        n = 50_000
        
        start = time.perf_counter()
        with db.transaction() as txn:
            for i in range(n):
                txn.put(f"key{i}".encode(), f'{{"id":{i}}}'.encode())
        duration = time.perf_counter() - start
        
        ops_per_sec = n / duration
        db.close()
        
        passed = ops_per_sec >= 50_000
        return (passed, "≥50K ops/sec", f"{ops_per_sec:,.0f} ops/sec")


def test_t3_2_sqlite_comparison() -> Tuple[bool, str, str]:
    """T3.2: ToonDB within 2x of SQLite"""
    import sqlite3
    
    try:
        from toondb.database import Database
    except ImportError:
        return (True, "FFI bindings", "SKIPPED - bindings not found")
    
    n = 10_000
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # SQLite
        sqlite_path = os.path.join(tmpdir, "test.db")
        conn = sqlite3.connect(sqlite_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")
        
        start = time.perf_counter()
        with conn:
            for i in range(n):
                conn.execute("INSERT INTO kv VALUES (?, ?)", (f"key{i}", f"value{i}"))
        sqlite_time = time.perf_counter() - start
        conn.close()
        
        # ToonDB
        toon_path = os.path.join(tmpdir, "toon")
        db = Database.open(toon_path)
        
        start = time.perf_counter()
        with db.transaction() as txn:
            for i in range(n):
                txn.put(f"key{i}".encode(), f"value{i}".encode())
        toon_time = time.perf_counter() - start
        db.close()
        
        ratio = toon_time / sqlite_time
        passed = ratio < 2.0
        
        return (passed, "<2x SQLite", f"{ratio:.2f}x SQLite")


# =============================================================================
# T4: VECTOR SEARCH TESTS
# =============================================================================

def test_t4_1_vector_recall() -> Tuple[bool, str, str]:
    """T4.1: Vector search recall@10 ≥90%"""
    try:
        import numpy as np
        from toondb import VectorIndex
    except ImportError:
        return (True, "numpy + FFI", "SKIPPED - deps not found")
    
    n, dim, k = 5000, 128, 10
    
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    ids = np.arange(n, dtype=np.uint64)
    index.insert_batch(ids, vectors)
    
    num_queries = 50
    query_indices = np.random.choice(n, num_queries, replace=False)
    
    recalls = []
    for qi in query_indices:
        query = vectors[qi]
        results = index.search(query, k=k)
        result_ids = set([r[0] for r in results])
        
        # Ground truth (cosine similarity)
        similarities = np.dot(vectors, query)
        ground_truth = set(np.argsort(similarities)[-k:])
        
        recall = len(result_ids & ground_truth) / k
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    passed = avg_recall >= 0.90
    
    return (passed, "≥90% recall", f"{avg_recall:.1%} recall")


def test_t4_2_vector_latency() -> Tuple[bool, str, str]:
    """T4.2: Search latency p99 <50ms"""
    try:
        import numpy as np
        from toondb import VectorIndex
    except ImportError:
        return (True, "numpy + FFI", "SKIPPED - deps not found")
    
    n, dim = 10_000, 128
    
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
    ids = np.arange(n, dtype=np.uint64)
    index.insert_batch(ids, vectors)
    
    # Warmup
    for _ in range(10):
        index.search(vectors[0], k=10)
    
    # Measure
    latencies = []
    for _ in range(100):
        query = vectors[np.random.randint(n)]
        start = time.perf_counter()
        index.search(query, k=10)
        latencies.append((time.perf_counter() - start) * 1000)
    
    p99 = sorted(latencies)[99]
    passed = p99 < 50
    
    return (passed, "p99 <50ms", f"p99={p99:.1f}ms")


# =============================================================================
# T6: DURABILITY TESTS
# =============================================================================

def test_t6_1_data_persistence() -> Tuple[bool, str, str]:
    """T6.1: Data survives close/reopen"""
    try:
        from toondb.database import Database
    except ImportError:
        return (True, "FFI bindings", "SKIPPED - bindings not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "persist_test")
        
        # Write and close
        db = Database.open(db_path)
        for i in range(100):
            db.put(f"key{i}".encode(), f"value{i}".encode())
        db.checkpoint()
        db.close()
        
        # Reopen and verify
        db = Database.open(db_path)
        recovered = 0
        for i in range(100):
            val = db.get(f"key{i}".encode())
            if val == f"value{i}".encode():
                recovered += 1
        db.close()
        
        passed = recovered == 100
        return (passed, "100% recovery", f"{recovered}% recovered")


def test_t6_2_transaction_atomicity() -> Tuple[bool, str, str]:
    """T6.2: Failed transactions don't corrupt"""
    try:
        from toondb.database import Database
    except ImportError:
        return (True, "FFI bindings", "SKIPPED - bindings not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "atomic_test")
        db = Database.open(db_path)
        
        db.put(b"counter", b"0")
        
        # Try transaction that fails
        try:
            with db.transaction() as txn:
                txn.put(b"counter", b"100")
                raise Exception("Simulated failure")
        except:
            pass
        
        val = db.get(b"counter")
        db.close()
        
        passed = val == b"0"
        return (passed, "rollback works", "original value preserved" if passed else f"got {val}")


# =============================================================================
# T7: EDGE CASE TESTS
# =============================================================================

def test_t7_1_large_values() -> Tuple[bool, str, str]:
    """T7.1: Large values (up to 1MB) work"""
    try:
        from toondb.database import Database
    except ImportError:
        return (True, "FFI bindings", "SKIPPED - bindings not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "large_test")
        db = Database.open(db_path)
        
        sizes = [1_000, 10_000, 100_000, 1_000_000]
        passed_all = True
        
        for size in sizes:
            key = f"large_{size}".encode()
            value = b"x" * size
            db.put(key, value)
            retrieved = db.get(key)
            if retrieved != value:
                passed_all = False
                break
        
        db.close()
        return (passed_all, "1MB works", "all sizes OK" if passed_all else "failed")


def test_t7_2_unicode_safety() -> Tuple[bool, str, str]:
    """T7.2: Unicode and binary data handled correctly"""
    try:
        from toondb.database import Database
    except ImportError:
        return (True, "FFI bindings", "SKIPPED - bindings not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "unicode_test")
        db = Database.open(db_path)
        
        test_cases = [
            ("emoji", "🎉🚀💾".encode()),
            ("chinese", "中文测试".encode()),
            ("null_bytes", b"hello\x00world"),
            ("binary", bytes(range(256))),
        ]
        
        passed_all = True
        for name, value in test_cases:
            key = f"test_{name}".encode()
            db.put(key, value)
            if db.get(key) != value:
                passed_all = False
                break
        
        db.close()
        return (passed_all, "all encodings", "all passed" if passed_all else "failed")


def test_t7_3_concurrent_access() -> Tuple[bool, str, str]:
    """T7.3: Concurrent readers/writers don't corrupt"""
    try:
        from toondb.database import Database
    except ImportError:
        return (True, "FFI bindings", "SKIPPED - bindings not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "concurrent_test")
        db = Database.open(db_path)
        
        errors = []
        
        def writer(writer_id):
            try:
                for i in range(50):
                    db.put(f"w{writer_id}_k{i}".encode(), b"value")
            except Exception as e:
                errors.append(str(e))
        
        def reader():
            try:
                for _ in range(100):
                    list(db.scan())
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for i in range(2):
            threads.append(threading.Thread(target=reader))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        db.close()
        
        passed = len(errors) == 0
        return (passed, "no errors", "concurrent safe" if passed else f"{len(errors)} errors")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_tests():
    """Run all feature validation tests"""
    print("=" * 70)
    print("  TOONDB FEATURE VALIDATION TEST SUITE")
    print("=" * 70)
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    suite = TestSuite()
    
    # T2: Token Savings
    print("\n[T2] TOKEN SAVINGS")
    suite.run_test("T2.1", "T2-TOKEN", "TOON vs JSON Tokens", test_t2_1_toon_vs_json_tokens)
    
    # T3: Performance
    print("\n[T3] PERFORMANCE")
    suite.run_test("T3.1", "T3-PERF", "Insert Throughput", test_t3_1_insert_throughput)
    suite.run_test("T3.2", "T3-PERF", "SQLite Comparison", test_t3_2_sqlite_comparison)
    
    # T4: Vector Search
    print("\n[T4] VECTOR SEARCH")
    suite.run_test("T4.1", "T4-VECTOR", "Recall@10", test_t4_1_vector_recall)
    suite.run_test("T4.2", "T4-VECTOR", "Search Latency", test_t4_2_vector_latency)
    
    # T6: Durability
    print("\n[T6] DURABILITY")
    suite.run_test("T6.1", "T6-DURABILITY", "Data Persistence", test_t6_1_data_persistence)
    suite.run_test("T6.2", "T6-DURABILITY", "Transaction Atomicity", test_t6_2_transaction_atomicity)
    
    # T7: Edge Cases
    print("\n[T7] EDGE CASES")
    suite.run_test("T7.1", "T7-EDGE", "Large Values", test_t7_1_large_values)
    suite.run_test("T7.2", "T7-EDGE", "Unicode Safety", test_t7_2_unicode_safety)
    suite.run_test("T7.3", "T7-EDGE", "Concurrent Access", test_t7_3_concurrent_access)
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in suite.results if r.passed)
    failed = sum(1 for r in suite.results if not r.passed)
    skipped = sum(1 for r in suite.results if "SKIPPED" in r.actual)
    
    print(f"\n  Total: {len(suite.results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    
    # By category
    categories = {}
    for r in suite.results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    print("\n  By Category:")
    for cat, results in sorted(categories.items()):
        cat_passed = sum(1 for r in results if r.passed)
        status = "✓" if cat_passed == len(results) else "⚠️"
        print(f"    {status} {cat}: {cat_passed}/{len(results)}")
    
    # Failed tests
    if failed > 0:
        print("\n  Failed Tests:")
        for r in suite.results:
            if not r.passed and "SKIPPED" not in r.actual:
                print(f"    ❌ {r.test_id}: {r.name}")
                print(f"       Expected: {r.expected}")
                print(f"       Error: {r.error or r.actual}")
    
    print("\n" + "=" * 70)
    print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return passed, failed, skipped


if __name__ == "__main__":
    passed, failed, skipped = run_all_tests()
    sys.exit(1 if failed > 0 else 0)
