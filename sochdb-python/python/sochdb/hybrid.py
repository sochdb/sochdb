"""Hybrid retrieval wrapper for native SochDB BM25/RRF primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HybridSearchResult:
    doc_id: str
    score: float
    vector_score: float
    bm25_score: float
    vector_rank: int | None
    bm25_rank: int | None


class HybridSearchIndex:
    """Compose native HNSW, BM25, and RRF for local hybrid retrieval."""

    def __init__(
        self,
        dimension: int,
        *,
        m: int | None = None,
        ef_construction: int | None = None,
        metric: str = "cosine",
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: float = 60.0,
        adaptive_rrf_k: bool = True,
        seed: int | None = None,
        deterministic_build: bool = False,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        from sochdb import (
            BM25Index,
            HnswIndex,
            RRFFusion,
            _check_native,
            recommended_hnsw_params,
        )

        _check_native()

        # Auto-tune HNSW build params to the embedding dimension unless the
        # caller pinned them explicitly. A hardcoded m=16 caps recall ~0.86 on
        # 768D+ embeddings; the native HnswIndex already defaults to m=32, so a
        # None default here keeps high-dim hybrid indexes at parity instead of
        # silently halving graph degree. (m=16 stays correct for dim<=128.)
        params = recommended_hnsw_params(dimension)
        m = params["m"] if m is None else m
        ef_construction = (
            params["ef_construction"] if ef_construction is None else ef_construction
        )

        self.dimension = dimension
        self.bm25 = BM25Index()
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._adaptive_rrf_k = adaptive_rrf_k
        self._RRFFusion = RRFFusion
        self.rrf = RRFFusion(k=rrf_k, vector_weight=vector_weight, lexical_weight=bm25_weight)
        # Only forward reproducibility kwargs when actually requested, so the
        # M-default fix above remains rebuild-free against an older native .so
        # (seed/deterministic_build require the rebuilt extension).
        hnsw_kwargs = dict(
            dimension=dimension,
            m=m,
            ef_construction=ef_construction,
            metric=metric,
        )
        if seed is not None or deterministic_build:
            hnsw_kwargs["seed"] = seed
            hnsw_kwargs["deterministic_build"] = deterministic_build
        self._index = HnswIndex(**hnsw_kwargs)
        self._numeric_to_doc_id: dict[int, str] = {}
        self._doc_id_to_numeric: dict[str, int] = {}

    def build(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        embeddings: np.ndarray,
    ) -> "HybridSearchIndex":
        """Build both native BM25 and native HNSW indexes."""

        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if len(doc_ids) != len(embeddings):
            raise ValueError("doc_ids and embeddings must have the same length")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        ids = [str(doc_id) for doc_id in doc_ids]
        numeric_ids = np.arange(1, len(ids) + 1, dtype=np.uint64)
        self._numeric_to_doc_id = {
            int(numeric_id): doc_id
            for numeric_id, doc_id in zip(numeric_ids.tolist(), ids)
        }
        self._doc_id_to_numeric = {
            doc_id: int(numeric_id)
            for numeric_id, doc_id in zip(numeric_ids.tolist(), ids)
        }

        for numeric_id, text in zip(numeric_ids.tolist(), texts):
            self.bm25.add_document(int(numeric_id), text)
        self._index.insert_batch_with_ids(
            numeric_ids,
            np.ascontiguousarray(embeddings, dtype=np.float32),
        )
        return self

    def vector_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> list[HybridSearchResult]:
        ids, distances = self._index.search(
            np.ascontiguousarray(query_embedding, dtype=np.float32),
            k=k,
        )
        results: list[HybridSearchResult] = []
        for rank, (numeric_id, distance) in enumerate(
            zip(ids.tolist(), distances.tolist()),
            start=1,
        ):
            doc_id = self._numeric_to_doc_id.get(int(numeric_id))
            if doc_id is None:
                continue
            results.append(
                HybridSearchResult(
                    doc_id=doc_id,
                    score=0.0,
                    vector_score=float(distance),
                    bm25_score=0.0,
                    vector_rank=rank,
                    bm25_rank=None,
                )
            )
        return results

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        *,
        k: int = 10,
        candidate_k: int | None = None,
    ) -> list[HybridSearchResult]:
        """Search with native BM25 + native HNSW + native RRF fusion."""

        if k <= 0:
            return []

        candidate_k = candidate_k or max(k * 2, 20)
        vector_hits = self.vector_search(query_embedding, k=candidate_k)
        vector_pairs = [
            (self._doc_id_to_numeric[hit.doc_id], hit.vector_score)
            for hit in vector_hits
        ]
        bm25_pairs = self.bm25.search(query_text, candidate_k)
        rrf = self._fusion_for(vector_pairs, bm25_pairs)
        fused_pairs = rrf.fuse(vector_pairs, bm25_pairs, k)
        vector_by_numeric = {
            self._doc_id_to_numeric[hit.doc_id]: hit
            for hit in vector_hits
        }
        bm25_by_numeric = {
            int(doc_id): (rank, float(score))
            for rank, (doc_id, score) in enumerate(bm25_pairs, start=1)
        }

        results: list[HybridSearchResult] = []
        for numeric_id, score in fused_pairs:
            doc_id = self._numeric_to_doc_id.get(int(numeric_id))
            if doc_id is None:
                continue
            vector_hit = vector_by_numeric.get(int(numeric_id))
            bm25_hit = bm25_by_numeric.get(int(numeric_id))
            results.append(
                HybridSearchResult(
                    doc_id=doc_id,
                    score=float(score),
                    vector_score=vector_hit.vector_score if vector_hit else 0.0,
                    bm25_score=bm25_hit[1] if bm25_hit else 0.0,
                    vector_rank=vector_hit.vector_rank if vector_hit else None,
                    bm25_rank=bm25_hit[0] if bm25_hit else None,
                )
            )
        return results

    @property
    def size(self) -> int:
        return len(self._numeric_to_doc_id)

    def _fusion_for(self, vector_pairs, bm25_pairs):
        """Return an RRF combiner whose ``k`` is matched to the fused pool depth.

        RRF scores a doc as ``Σ w / (k + rank)``; ``k`` controls how gently a
        top rank is preferred over deeper ranks. The canonical ``k=60`` assumes
        deep result lists (hundreds+). On a shallow fused pool (e.g. retrieval
        within a single ~28-document conversation) that constant collapses the
        rank signal: ``(k+1)`` and ``(k+N)`` are nearly equal, so the weaker
        lane can override the stronger one. Capping ``k`` at half the pool depth
        keeps rank-1 at least ~2x ahead of the median rank, while large corpora
        (pool ≥ 2·k) are left at the configured ``k`` unchanged.
        """

        if not self._adaptive_rrf_k:
            return self.rrf
        pool = len({nid for nid, _ in vector_pairs} | {int(d) for d, _ in bm25_pairs})
        eff_k = self._effective_k(pool)
        if eff_k == self._rrf_k:
            return self.rrf
        return self._RRFFusion(
            k=eff_k,
            vector_weight=self._vector_weight,
            lexical_weight=self._bm25_weight,
        )

    def _effective_k(self, pool_size: int) -> float:
        """Pool-adaptive RRF ``k`` capped at half the fused-candidate depth.

        Returns the configured ``k`` when adaptation is disabled or the pool is
        deep enough (``pool_size >= 2 * k``); otherwise ``max(1, pool/2)`` so a
        rank-1 document keeps a ~2x advantage over the median rank.
        """

        if not self._adaptive_rrf_k:
            return self._rrf_k
        return min(self._rrf_k, max(1.0, pool_size / 2.0))
