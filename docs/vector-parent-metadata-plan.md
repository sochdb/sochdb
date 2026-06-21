# Vector Parent/View Metadata Plan

## Summary

Add durable optional `parent_id` and `view_type` metadata to each vector record
so that multi-vector logical records can be represented, persisted, and returned
in search results.

## 1. Current Insert/Search API Shape

### gRPC Insert (Batch)

```proto
message InsertBatchRequest {
  string index_name = 1;
  repeated uint64 ids = 2;
  repeated float vectors = 3;
  repeated VectorMetadata metadata = 4;
}
```

### gRPC Insert (Stream)

```proto
message InsertStreamRequest {
  string index_name = 1;
  uint64 id = 2;
  repeated float vector = 3;
  VectorMetadata metadata = 4;
}
```

### gRPC Search Result

```proto
message SearchResult {
  uint64 id = 1;
  float distance = 2;
  string metric = 3;
  optional uint64 parent_id = 4;
  optional string view_type = 5;
}
```

### Python Native (PyO3)

- `HnswIndex.insert_batch(vectors)` — auto IDs, no metadata
- `HnswIndex.insert_batch_with_ids(ids, vectors)` — explicit IDs, no metadata
- `HnswIndex.set_metadata(node_id, metadata)` — single node
- `HnswIndex.set_metadata_batch(node_ids, metadata_list)` — batch
- `HnswIndex.search(query, k)` → `(ids, distances)`
- `HnswIndex.search_with_metadata(query, k)` → `[SearchResult, ...]` **(new)**

## 2. Exact Protobuf Fields Available

| Message | Field | Number | Type | Note |
|---------|-------|--------|------|------|
| `VectorMetadata` | `parent_id` | 1 | `optional uint64` | 0 is valid when explicitly present |
| `VectorMetadata` | `view_type` | 2 | `optional string` | e.g. "turn", "event" |
| `InsertBatchRequest` | `metadata` | 4 | `repeated VectorMetadata` | length must match `ids` if present |
| `InsertStreamRequest` | `metadata` | 4 | `VectorMetadata` | optional per-vector |
| `SearchResult` | `parent_id` | 4 | `optional uint64` | absent when not stored |
| `SearchResult` | `view_type` | 5 | `optional string` | absent when not stored |

Both `proto/sochdb.proto` and `sochdb-grpc/proto/sochdb.proto` are kept in sync.
Generated Python bindings (`sochdb_pb2.py`) already include these fields.

## 3. Where Metadata Is Stored

Metadata is stored inside `HnswIndex` as an index-owned sidecar:

```rust
pub(crate) metadata_store: Arc<RwLock<Vec<Option<Vec<(String, String)>>>>>,
```

- Indexed by `dense_index` (same as vector_store and internal_nodes).
- Scoped per index — two indexes with the same external vector ID do not share metadata.
- Missing metadata is represented as `None`.
- Explicit `parent_id = 0` is stored as the string `"0"` under key `"parent_id"`.

Helper methods on `HnswIndex`:
- `set_metadata(node_id: u128, metadata: Vec<(String, String)>)`
- `set_metadata_batch(entries: &[(u128, Vec<(String, String)>)])`
- `get_metadata(node_id: u128) -> Option<Vec<(String, String)>>`

## 4. How Metadata Is Persisted

A trailer is appended after the main bincode snapshot:

```text
[bincode IndexSnapshot][metadata trailer]
```

Trailer format:

```text
8-byte magic: "SCMETA01"
bincode(MetadataTrailer {
    version: u32,
    entries: Vec<(u128, Vec<(String, String)>)>,
})
```

Write path (`save_to_disk` / `save_to_disk_compressed`):
- Serialize the index snapshot.
- Call `write_metadata_trailer` to append metadata for vectors that have it.
- If no vectors have metadata, the trailer is omitted entirely.

Read path (`load_from_disk` / `load_from_disk_compressed`):
- Deserialize the index snapshot.
- Call `read_metadata_trailer`.
- If EOF is reached immediately after the snapshot, return `None` (old snapshot without metadata).
- If magic mismatches, return an error.
- If deserialization fails, return an error.
- If a trailer is present, call `set_metadata_batch` to restore entries.

## 5. Old Snapshot Compatibility Plan

| Scenario | Behaviour |
|----------|-----------|
| Old snapshot → new reader | Trailer is absent. `read_metadata_trailer` returns `Ok(None)`. Index loads successfully with empty metadata. |
| New snapshot → old reader | Old reader stops after the bincode snapshot and ignores the trailing bytes. Metadata is lost but the index vectors and graph load correctly. |
| Malformed trailer magic | `load_from_disk` returns `Err("Invalid metadata trailer magic")`. |
| Truncated trailer | `load_from_disk` returns an error describing trailer/EOF failure. |

Crash safety: the current implementation writes directly to the target path.
Full atomicity (temp file + fsync + rename) is not claimed in this PR.

## 6. Python Exposure Plan

### Protobuf / gRPC Client

Already supported via generated `sochdb_pb2.py`:

```python
request = sochdb_pb2.InsertBatchRequest(
    index_name="memories",
    ids=[712001, 712002],
    vectors=[...],
    metadata=[
        sochdb_pb2.VectorMetadata(parent_id=712, view_type="turn"),
        sochdb_pb2.VectorMetadata(parent_id=712, view_type="event"),
    ],
)
```

Search results expose:
```python
for result in response.results:
    print(result.id, result.distance, result.parent_id, result.view_type)
```

### Native PyO3 Extension

- `HnswIndex.set_metadata` and `set_metadata_batch` already exist for insertion-side metadata.
- **New:** `HnswIndex.search_with_metadata(query, k, ef_search=None)` returns a list of
  `SearchResult` objects with `id`, `distance`, `parent_id`, `view_type` attributes.
- The existing `search(query, k)` method continues to return `(ids, distances)` for backward compatibility.

## 7. Tests to Add / Verify

### Already present and passing
- `legacy_insert_without_metadata_returns_absent_metadata`
- `batch_insert_with_mixed_metadata_is_returned_by_search`
- `batch_insert_rejects_metadata_length_mismatch`
- `search_result_reports_*_metric`
- `search_batch_results_report_metric`
- `test_save_and_load_preserves_metadata_trailer`
- `test_save_and_load_without_metadata_trailer_leaves_metadata_empty`
- `test_search_result_metadata_presence_round_trips` (Python protobuf)
- `test_insert_batch_metadata_model_supports_mixed_presence` (Python protobuf)

### Added in this PR
- `search_batch_returns_metadata_for_mixed_presence` — batch search propagates metadata.
- `test_malformed_trailer_rejects_invalid_magic` — corrupt magic after snapshot is rejected.
- `test_truncated_trailer_produces_error` — truncated trailer returns controlled error.
- `test_index_isolation_metadata_not_leaked` — same IDs in two indexes do not share metadata after save/load.

### Verification script
- `scripts/verify-vector-metadata-persistence.sh` — deterministic bash fixture that runs
  focused cargo tests and prints machine-readable success lines.
