#!/usr/bin/env python3
"""
ToonDB Bulk Ingest Performance Benchmark

This benchmark measures throughput for the Bulk API and compares it against
the Python FFI path to detect performance regressions.

Exit codes:
  0 - All benchmarks pass threshold
  1 - Performance regression detected

Usage:
    python bulk_benchmark.py              # Run benchmark
    python bulk_benchmark.py --ci         # CI mode with exit codes
    python bulk_benchmark.py --baseline   # Generate baseline file
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

# Minimum throughput thresholds (vectors/second)
# Regression detected if below these values
THRESHOLDS = {
    # Bulk CLI should achieve near-native Rust performance
    "bulk_build_768d": 1000.0,    # ~1,600 expected, 1,000 min
    "bulk_build_384d": 2000.0,    # Higher dim = lower throughput
    "bulk_build_1536d": 500.0,    # OpenAI ada-002 dimension
    
    # For comparison: Python FFI is ~10-12× slower
    # These are informational, not regression gates
    "ffi_insert_768d": 100.0,
}

# Test sizes
TEST_SIZES = {
    "small": 1000,      # Quick smoke test
    "medium": 10000,    # Standard benchmark
    "large": 100000,    # Stress test (skip in CI by default)
}

# Default test configuration
DEFAULT_SIZE = "medium"
DEFAULT_DIMENSION = 768


# =============================================================================
# Benchmark Implementation
# =============================================================================

@dataclass
class BenchResult:
    """Result from a single benchmark run."""
    name: str
    vectors: int
    dimension: int
    elapsed_secs: float
    throughput: float  # vectors/second
    method: str
    passed: bool | None = None
    threshold: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_test_vectors(n: int, d: int) -> np.ndarray:
    """Generate random test vectors."""
    rng = np.random.default_rng(42)  # Reproducible
    return rng.standard_normal((n, d)).astype(np.float32)


def benchmark_bulk_cli(
    vectors: np.ndarray,
    output_dir: Path,
    name: str = "bulk_build",
) -> BenchResult:
    """Benchmark the bulk CLI (subprocess path)."""
    n, d = vectors.shape
    
    # Import here to avoid import errors if bulk not available
    try:
        from toondb.bulk import bulk_build_index, get_toondb_bulk_path
        # Verify binary exists
        get_toondb_bulk_path()
    except (ImportError, RuntimeError) as e:
        return BenchResult(
            name=name,
            vectors=n,
            dimension=d,
            elapsed_secs=0.0,
            throughput=0.0,
            method="bulk_cli (unavailable)",
            passed=None,
        )
    
    output_file = output_dir / f"{name}.hnsw"
    
    start = time.perf_counter()
    stats = bulk_build_index(
        vectors,
        output=output_file,
        m=16,
        ef_construction=100,
        quiet=True,
    )
    elapsed = time.perf_counter() - start
    
    throughput = n / elapsed
    
    return BenchResult(
        name=name,
        vectors=n,
        dimension=d,
        elapsed_secs=elapsed,
        throughput=throughput,
        method="bulk_cli",
    )


def benchmark_ffi_insert(
    vectors: np.ndarray,
    output_dir: Path,
    name: str = "ffi_insert",
) -> BenchResult:
    """Benchmark the Python FFI path (for comparison)."""
    n, d = vectors.shape
    
    # Try to import VectorIndex
    try:
        from toondb.vector import VectorIndex
    except ImportError:
        return BenchResult(
            name=name,
            vectors=n,
            dimension=d,
            elapsed_secs=0.0,
            throughput=0.0,
            method="ffi (unavailable)",
            passed=None,
        )
    
    # Create index
    try:
        index = VectorIndex.create(d, m=16, ef_construction=100)
    except Exception as e:
        return BenchResult(
            name=name,
            vectors=n,
            dimension=d,
            elapsed_secs=0.0,
            throughput=0.0,
            method="ffi (error)",
            passed=None,
        )
    
    # Insert in batches
    batch_size = 1000
    start = time.perf_counter()
    
    for i in range(0, n, batch_size):
        batch = vectors[i:i+batch_size]
        ids = list(range(i, i + len(batch)))
        index.add_batch(ids, batch)
    
    elapsed = time.perf_counter() - start
    throughput = n / elapsed
    
    return BenchResult(
        name=name,
        vectors=n,
        dimension=d,
        elapsed_secs=elapsed,
        throughput=throughput,
        method="ffi",
    )


def run_benchmark_suite(
    size: str = DEFAULT_SIZE,
    dimensions: list[int] | None = None,
    include_ffi: bool = True,
) -> list[BenchResult]:
    """Run the full benchmark suite."""
    if dimensions is None:
        dimensions = [384, 768, 1536]
    
    n = TEST_SIZES[size]
    results = []
    
    with tempfile.TemporaryDirectory(prefix="toondb_bench_") as tmpdir:
        output_dir = Path(tmpdir)
        
        for d in dimensions:
            print(f"\n{'='*60}")
            print(f"Benchmark: {n:,} vectors × {d}D")
            print('='*60)
            
            # Generate test data
            print("Generating vectors...", end=" ", flush=True)
            vectors = generate_test_vectors(n, d)
            print(f"done ({vectors.nbytes / 1024 / 1024:.1f} MB)")
            
            # Bulk CLI benchmark
            print("Running bulk CLI benchmark...", end=" ", flush=True)
            bulk_result = benchmark_bulk_cli(
                vectors, output_dir,
                name=f"bulk_build_{d}d"
            )
            print(f"{bulk_result.throughput:.0f} vec/s")
            results.append(bulk_result)
            
            # FFI benchmark (optional)
            if include_ffi:
                print("Running FFI benchmark...", end=" ", flush=True)
                ffi_result = benchmark_ffi_insert(
                    vectors, output_dir,
                    name=f"ffi_insert_{d}d"
                )
                print(f"{ffi_result.throughput:.0f} vec/s")
                results.append(ffi_result)
    
    return results


def check_thresholds(results: list[BenchResult]) -> bool:
    """Check results against thresholds. Returns True if all pass."""
    all_passed = True
    
    print("\n" + "="*60)
    print("REGRESSION CHECK")
    print("="*60)
    
    for result in results:
        if result.name in THRESHOLDS:
            threshold = THRESHOLDS[result.name]
            result.threshold = threshold
            result.passed = result.throughput >= threshold
            
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{result.name}: {result.throughput:.0f} vec/s "
                  f"(threshold: {threshold:.0f}) [{status}]")
            
            if not result.passed:
                all_passed = False
    
    return all_passed


def print_summary(results: list[BenchResult]) -> None:
    """Print benchmark summary table."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Name':<25} {'Vectors':>10} {'Dim':>6} {'Time':>10} {'Rate':>12}")
    print("-"*60)
    
    for r in results:
        if r.throughput > 0:
            print(f"{r.name:<25} {r.vectors:>10,} {r.dimension:>6} "
                  f"{r.elapsed_secs:>9.2f}s {r.throughput:>11,.0f}/s")
    
    # Compute speedup
    bulk_768 = next((r for r in results if r.name == "bulk_build_768d"), None)
    ffi_768 = next((r for r in results if r.name == "ffi_insert_768d"), None)
    
    if bulk_768 and ffi_768 and ffi_768.throughput > 0:
        speedup = bulk_768.throughput / ffi_768.throughput
        print("-"*60)
        print(f"Bulk CLI speedup over FFI: {speedup:.1f}×")


def save_results(results: list[BenchResult], path: Path) -> None:
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "results": [r.to_dict() for r in results],
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ToonDB Bulk Ingest Benchmark")
    parser.add_argument("--ci", action="store_true", 
                       help="CI mode: exit 1 on regression")
    parser.add_argument("--size", choices=TEST_SIZES.keys(), default=DEFAULT_SIZE,
                       help=f"Test size (default: {DEFAULT_SIZE})")
    parser.add_argument("--dimension", type=int, nargs="+", default=None,
                       help="Vector dimensions to test")
    parser.add_argument("--no-ffi", action="store_true",
                       help="Skip FFI benchmark")
    parser.add_argument("--output", type=Path, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--baseline", action="store_true",
                       help="Save as baseline file")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ToonDB Bulk Ingest Performance Benchmark")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Test size: {args.size} ({TEST_SIZES[args.size]:,} vectors)")
    
    # Run benchmarks
    results = run_benchmark_suite(
        size=args.size,
        dimensions=args.dimension,
        include_ffi=not args.no_ffi,
    )
    
    # Print summary
    print_summary(results)
    
    # Check thresholds
    all_passed = check_thresholds(results)
    
    # Save results
    if args.output:
        save_results(results, args.output)
    elif args.baseline:
        save_results(results, Path("bulk_benchmark_baseline.json"))
    
    # Exit code
    if args.ci and not all_passed:
        print("\n❌ REGRESSION DETECTED")
        return 1
    
    print("\n✓ Benchmark complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
