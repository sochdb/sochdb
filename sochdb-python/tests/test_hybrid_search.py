import numpy as np

from sochdb import BM25Index, HybridSearchIndex


def test_bm25_ranks_keyword_match_first():
    index = BM25Index()
    index.add_document(1, "OAuth token refresh and session middleware")
    index.add_document(2, "Invoices, card payments, and receipts")

    results = index.search("refresh token", k=2)

    assert [doc_id for doc_id, _score in results] == [1]
    assert results[0][1] > 0


def test_hybrid_search_fuses_keyword_and_vector_ranks():
    index = HybridSearchIndex(dimension=2, m=4, ef_construction=20)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    index.build(
        ["vector_match", "keyword_match"],
        [
            "semantic-only nearest vector document",
            "OAuth token refresh keyword document",
        ],
        embeddings,
    )

    results = index.search(
        "refresh token",
        np.array([1.0, 0.0], dtype=np.float32),
        k=2,
    )

    assert results[0].doc_id == "keyword_match"
    assert results[0].bm25_rank == 1
    assert results[0].vector_rank is not None
    assert {result.doc_id for result in results} == {"keyword_match", "vector_match"}


def test_effective_rrf_k_scales_with_pool_depth():
    index = HybridSearchIndex(dimension=2, m=4, ef_construction=20, rrf_k=60.0)
    # Shallow per-conversation pools shrink k so rank-1 keeps a real advantage.
    assert index._effective_k(10) == 5.0
    assert index._effective_k(28) == 14.0
    # A single-doc pool floors at 1 (never 0).
    assert index._effective_k(1) == 1.0
    # Deep pools (pool >= 2*k) keep the canonical configured k untouched.
    assert index._effective_k(120) == 60.0
    assert index._effective_k(5000) == 60.0


def test_effective_rrf_k_respects_opt_out():
    index = HybridSearchIndex(
        dimension=2, m=4, ef_construction=20, rrf_k=60.0, adaptive_rrf_k=False
    )
    # With adaptation disabled the configured k is used at every pool depth.
    assert index._effective_k(10) == 60.0
    assert index._effective_k(5000) == 60.0
