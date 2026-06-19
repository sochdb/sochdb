"""Tests for two retrieval-pipeline fixes:

1. ``HybridSearchIndex`` no longer hardcodes ``m=16`` — it auto-tunes the HNSW
   build params to the embedding dimension via ``recommended_hnsw_params``
   (m=32 / ef_construction=256 for 768D+), matching the native engine default.
2. ``HnswIndex(seed=..., deterministic_build=True)`` makes builds reproducible:
   per-node levels are pinned by ``(id, seed)`` and the connect phase runs in a
   fixed order, so two builds over identical inputs return identical results.

The determinism tests require a native extension rebuilt with the new ``seed``
kwarg (``maturin develop --release``); they skip gracefully otherwise.
"""

import numpy as np
import pytest

import sochdb
from sochdb import HnswIndex, HybridSearchIndex, recommended_hnsw_params


# --------------------------------------------------------------------------- #
# Issue 1 — dimension-aware M default
# --------------------------------------------------------------------------- #

def test_recommended_params_tiers_m_by_dimension():
    assert recommended_hnsw_params(128)["m"] == 16
    assert recommended_hnsw_params(512)["m"] == 24
    assert recommended_hnsw_params(768)["m"] == 32
    assert recommended_hnsw_params(1536)["m"] == 32
    # ef_construction floor is max(200, m*8).
    assert recommended_hnsw_params(768)["ef_construction"] == 256
    assert recommended_hnsw_params(128)["ef_construction"] == 200


def test_hybrid_default_m_is_dimension_aware(monkeypatch):
    # HybridSearchIndex must default m/ef_construction to recommended_hnsw_params
    # rather than the old hardcoded m=16 / ef_construction=100.
    captured: dict = {}

    class _Spy:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr(sochdb, "HnswIndex", _Spy)

    HybridSearchIndex(dimension=1536)
    assert captured["m"] == 32
    assert captured["ef_construction"] == 256

    HybridSearchIndex(dimension=768)
    assert captured["m"] == 32
    assert captured["ef_construction"] == 256

    HybridSearchIndex(dimension=128)
    assert captured["m"] == 16  # low-dim tier intentionally unchanged
    assert captured["ef_construction"] == 200


def test_hybrid_explicit_params_are_preserved(monkeypatch):
    # Backward compatibility: explicit m / ef_construction are passed verbatim.
    captured: dict = {}

    class _Spy:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr(sochdb, "HnswIndex", _Spy)

    HybridSearchIndex(dimension=1536, m=8, ef_construction=50)
    assert captured["m"] == 8
    assert captured["ef_construction"] == 50

    # Partial override: explicit m kept, ef_construction auto-tuned.
    HybridSearchIndex(dimension=1536, m=48)
    assert captured["m"] == 48
    assert captured["ef_construction"] == 256


# --------------------------------------------------------------------------- #
# Issue 2 — reproducible builds
# --------------------------------------------------------------------------- #

def _supports_seed() -> bool:
    try:
        HnswIndex(4, seed=1, deterministic_build=True)
        return True
    except TypeError:
        return False


_NEEDS_SEED = pytest.mark.skipif(
    not _supports_seed(),
    reason="native extension predates the seed= kwarg; rebuild with maturin",
)


@_NEEDS_SEED
def test_deterministic_build_reproduces_topk():
    rng = np.random.RandomState(0)
    X = np.ascontiguousarray(rng.randn(2000, 64).astype(np.float32))

    def build():
        ix = HnswIndex(64, m=32, ef_construction=200, seed=7, deterministic_build=True)
        ix.insert_batch(X)
        return ix

    a, b = build(), build()
    for i in range(0, 200, 8):
        ids_a, _ = a.search(X[i], 10)
        ids_b, _ = b.search(X[i], 10)
        assert list(ids_a) == list(ids_b), f"query {i} top-k differs across builds"


@_NEEDS_SEED
def test_seed_without_deterministic_build_pins_levels_not_graph():
    # Documents the Tier-1 boundary: a plain seed makes levels reproducible but
    # does NOT guarantee identical neighbor graphs under the parallel builder,
    # so top-k MAY differ. We only assert the build runs and returns k results.
    rng = np.random.RandomState(1)
    X = np.ascontiguousarray(rng.randn(1000, 64).astype(np.float32))
    ix = HnswIndex(64, m=32, ef_construction=200, seed=11)  # deterministic_build=False
    ix.insert_batch(X)
    ids, _ = ix.search(X[0], 10)
    assert len(list(ids)) == 10


@_NEEDS_SEED
def test_hybrid_search_index_accepts_seed():
    # The wrapper forwards seed / deterministic_build to the native HnswIndex.
    rng = np.random.RandomState(2)
    X = np.ascontiguousarray(rng.randn(300, 32).astype(np.float32))
    docs = [f"doc-{i}" for i in range(len(X))]
    texts = [f"document number {i}" for i in range(len(X))]

    def build():
        ix = HybridSearchIndex(dimension=32, seed=5, deterministic_build=True)
        ix.build(docs, texts, X)
        return ix

    a, b = build(), build()
    ra = a.search("document number 10", X[10], k=10)
    rb = b.search("document number 10", X[10], k=10)
    assert [r.doc_id for r in ra] == [r.doc_id for r in rb]
