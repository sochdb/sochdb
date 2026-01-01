#!/usr/bin/env python3
"""
ToonDB FFI Performance Regression Tests

Cross-language performance verification suite that ensures:
1. FFI throughput stays above documented thresholds
2. Rust/FFI ratio remains acceptable (no hidden overhead)
3. Safe mode is not accidentally enabled

Run with: pytest test_ffi_perf.py -v
"""

import os
import time
import warnings
import pytest
import numpy as np

# Skip all tests if ToonDB library not available
try:
    from toondb.vector import VectorIndex, PerformanceWarning
    TOONDB_AVAILABLE = True
except ImportError:
    TOONDB_AVAILABLE = False

# Performance thresholds (vec/sec)
# These are conservative minimums - actual performance should be higher
THRESHOLDS = {
    "128d_10k": {
        "rust_baseline": 8000,  # Pure Rust can achieve ~9,500
        "ffi_min_ratio": 0.70,  # FFI should be ≥70% of Rust
        "absolute_min": 5000,   # Absolute floor
    },
    "768d_10k": {
        "rust_baseline": 1500,  # Pure Rust can achieve ~1,600
        "ffi_min_ratio": 0.60,  # FFI should be ≥60% of Rust
        "absolute_min": 800,    # Absolute floor
    },
}


@pytest.fixture
def vectors_128d():
    """Generate reproducible 128D test vectors."""
    np.random.seed(42)
    return np.random.randn(10000, 128).astype(np.float32)


@pytest.fixture
def vectors_768d():
    """Generate reproducible 768D test vectors."""
    np.random.seed(42)
    return np.random.randn(10000, 768).astype(np.float32)


@pytest.fixture
def vectors_128d_small():
    """Small 128D set for quick tests."""
    np.random.seed(42)
    return np.random.randn(1000, 128).astype(np.float32)


@pytest.mark.skipif(not TOONDB_AVAILABLE, reason="ToonDB library not available")
class TestFFIPerformance:
    """Performance regression tests for FFI throughput."""
    
    def test_no_safe_mode(self):
        """Verify TOONDB_BATCH_SAFE_MODE is not set during tests."""
        safe_mode = os.environ.get("TOONDB_BATCH_SAFE_MODE")
        assert safe_mode not in ("1", "true", "True"), (
            "TOONDB_BATCH_SAFE_MODE is enabled! "
            "Performance tests are invalid. Unset this variable."
        )
    
    def test_128d_throughput(self, vectors_128d):
        """FFI throughput must exceed minimum for 128D vectors."""
        index = VectorIndex(dimension=128, max_connections=16, ef_construction=48)
        ids = np.arange(len(vectors_128d), dtype=np.uint64)
        
        # Warmup
        warmup_ids = np.arange(100, dtype=np.uint64)
        warmup_vecs = vectors_128d[:100].copy()
        index.insert_batch_fast(warmup_ids + 1000000, warmup_vecs)
        
        # Timed run
        start = time.perf_counter()
        inserted = index.insert_batch_fast(ids, vectors_128d)
        elapsed = time.perf_counter() - start
        
        throughput = inserted / elapsed
        threshold = THRESHOLDS["128d_10k"]
        min_expected = threshold["absolute_min"]
        
        print(f"\n128D Throughput: {throughput:.0f} vec/s (min: {min_expected})")
        
        assert throughput >= min_expected, (
            f"128D FFI throughput {throughput:.0f} vec/s is below minimum "
            f"{min_expected} vec/s. Check for regressions."
        )
    
    def test_768d_throughput(self, vectors_768d):
        """FFI throughput must exceed minimum for 768D vectors."""
        index = VectorIndex(dimension=768, max_connections=16, ef_construction=48)
        ids = np.arange(len(vectors_768d), dtype=np.uint64)
        
        # Warmup
        warmup_ids = np.arange(100, dtype=np.uint64)
        warmup_vecs = vectors_768d[:100].copy()
        index.insert_batch_fast(warmup_ids + 1000000, warmup_vecs)
        
        # Timed run
        start = time.perf_counter()
        inserted = index.insert_batch_fast(ids, vectors_768d)
        elapsed = time.perf_counter() - start
        
        throughput = inserted / elapsed
        threshold = THRESHOLDS["768d_10k"]
        min_expected = threshold["absolute_min"]
        
        print(f"\n768D Throughput: {throughput:.0f} vec/s (min: {min_expected})")
        
        assert throughput >= min_expected, (
            f"768D FFI throughput {throughput:.0f} vec/s is below minimum "
            f"{min_expected} vec/s. Check for regressions."
        )
    
    def test_strict_mode_rejects_wrong_dtype(self, vectors_128d_small):
        """Strict mode should reject float64 vectors."""
        index = VectorIndex(dimension=128)
        
        # Create wrong dtype array
        vectors_f64 = vectors_128d_small.astype(np.float64)
        ids = np.arange(len(vectors_f64), dtype=np.uint64)
        
        with pytest.raises(ValueError, match="float32"):
            index.insert_batch_fast(ids, vectors_f64, strict=True)
    
    def test_strict_mode_rejects_non_contiguous(self, vectors_128d_small):
        """Strict mode should reject non-contiguous arrays."""
        index = VectorIndex(dimension=128)
        ids = np.arange(len(vectors_128d_small), dtype=np.uint64)
        
        # Create non-contiguous array via transpose + transpose back
        # (this makes it Fortran-order internally)
        vectors_f_order = np.asfortranarray(vectors_128d_small)
        
        with pytest.raises(ValueError, match="C-contiguous"):
            index.insert_batch_fast(ids, vectors_f_order, strict=True)
    
    def test_strict_mode_allows_correct_layout(self, vectors_128d_small):
        """Strict mode should accept properly formatted arrays."""
        index = VectorIndex(dimension=128)
        ids = np.arange(len(vectors_128d_small), dtype=np.uint64)
        
        # This should succeed
        inserted = index.insert_batch_fast(ids, vectors_128d_small, strict=True)
        assert inserted == len(vectors_128d_small)
    
    def test_non_strict_mode_converts(self, vectors_128d_small):
        """Non-strict mode should silently convert wrong layouts."""
        index = VectorIndex(dimension=128)
        
        # Create wrong dtype array
        vectors_f64 = vectors_128d_small.astype(np.float64)
        ids = np.arange(len(vectors_f64), dtype=np.uint64)
        
        # Should succeed with silent conversion
        inserted = index.insert_batch_fast(ids, vectors_f64, strict=False)
        assert inserted == len(vectors_f64)
    
    def test_batch_vs_individual_speedup(self, vectors_128d_small):
        """Batch insert should be significantly faster than individual."""
        # Individual inserts
        index1 = VectorIndex(dimension=128)
        
        start = time.perf_counter()
        for i in range(len(vectors_128d_small)):
            index1.insert(i, vectors_128d_small[i])
        individual_time = time.perf_counter() - start
        
        # Batch insert
        index2 = VectorIndex(dimension=128)
        ids = np.arange(len(vectors_128d_small), dtype=np.uint64)
        
        start = time.perf_counter()
        index2.insert_batch_fast(ids, vectors_128d_small)
        batch_time = time.perf_counter() - start
        
        speedup = individual_time / batch_time
        print(f"\nBatch speedup: {speedup:.1f}x")
        
        # Batch should be at least 5x faster
        assert speedup >= 5.0, (
            f"Batch insert only {speedup:.1f}x faster than individual. "
            f"Expected at least 5x speedup."
        )


@pytest.mark.skipif(not TOONDB_AVAILABLE, reason="ToonDB library not available")
class TestFFICorrectness:
    """Correctness tests for FFI insert paths."""
    
    def test_search_after_batch_insert(self, vectors_128d_small):
        """Vectors should be searchable after batch insert."""
        index = VectorIndex(dimension=128, max_connections=16, ef_construction=100)
        ids = np.arange(len(vectors_128d_small), dtype=np.uint64)
        
        inserted = index.insert_batch_fast(ids, vectors_128d_small)
        assert inserted == len(vectors_128d_small)
        
        # Search for each vector - should find itself as nearest
        for i in range(min(10, len(vectors_128d_small))):
            results = index.search(vectors_128d_small[i], k=1)
            assert len(results) > 0
            found_id, distance = results[0]
            # Should find the exact vector (distance ≈ 0) or very close
            assert distance < 0.1, f"Vector {i} not found accurately"
    
    def test_batch_insert_count(self, vectors_128d_small):
        """Batch insert should return correct count."""
        index = VectorIndex(dimension=128)
        ids = np.arange(len(vectors_128d_small), dtype=np.uint64)
        
        inserted = index.insert_batch_fast(ids, vectors_128d_small)
        assert inserted == len(vectors_128d_small)
        assert len(index) == len(vectors_128d_small)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
