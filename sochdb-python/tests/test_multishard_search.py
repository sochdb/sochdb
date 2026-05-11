import threading

import numpy as np
import pytest

from sochdb import MultiShardHnswIndex


class _FakeShard:
    def __init__(self, ids=None, dists=None, error=None):
        self._ids = np.array(ids or [], dtype=np.uint64)
        self._dists = np.array(dists or [], dtype=np.float32)
        self._error = error

    def search(self, query, k: int, ef_search: int | None = None):
        if self._error is not None:
            raise self._error
        return self._ids[:k], self._dists[:k]


def _build_index(shards):
    index = MultiShardHnswIndex.__new__(MultiShardHnswIndex)
    index.dimension = 3
    index.n_shards = len(shards)
    index.metric = "cosine"
    index._ef_search = 64
    index.shards = shards
    index._lock = threading.Lock()
    index._shard_locks = [threading.Lock() for _ in shards]
    index._shard_counts = [0] * len(shards)
    index._total_inserted = 0
    return index


def test_search_raises_when_any_shard_fails_by_default():
    index = _build_index(
        [
            _FakeShard(ids=[11], dists=[0.11]),
            _FakeShard(error=ValueError("shard 1 failed")),
        ]
    )

    with pytest.raises(RuntimeError, match="1 shard searches failed: shard 1 failed"):
        index.search(np.array([0.1, 0.2, 0.3], dtype=np.float32), k=5)


def test_search_partial_returns_results_and_warns():
    index = _build_index(
        [
            _FakeShard(ids=[22, 33], dists=[0.22, 0.33]),
            _FakeShard(error=RuntimeError("shard 2 timed out")),
            _FakeShard(ids=[44], dists=[0.44]),
        ]
    )

    with pytest.warns(RuntimeWarning, match="1 shard searches failed: shard 2 timed out"):
        ids, dists = index.search(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            k=2,
            failure_policy="partial",
        )

    assert ids.tolist() == [22, 33]
    assert dists.tolist() == pytest.approx([0.22, 0.33])


def test_search_rejects_unknown_failure_policy():
    index = _build_index([_FakeShard(ids=[1], dists=[0.1])])

    with pytest.raises(
        ValueError,
        match="failure_policy must be one of: 'raise', 'partial', 'ignore'",
    ):
        index.search(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            failure_policy="warn",
        )
