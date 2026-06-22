# Vector Grouped Search — Implementation Plan

## Motivation

In multi-vector memory/document retrieval, one logical source can have
multiple vector views (e.g. turn, event, entity, neighbor-window). Without
grouped search, multiple views from the same parent can consume top‑K slots,
hurting small‑K retrieval quality.

## Proto Design

### New types

```proto
enum GroupBy {
  GROUP_BY_UNSPECIFIED = 0;
  GROUP_BY_PARENT_ID = 1;
}

message GroupingOptions {
  GroupBy group_by = 1;
  uint32 max_per_group = 2;
  uint32 candidate_k = 3;
}

message GroupingInfo {
  GroupBy group_by = 1;
  uint32 requested_k = 2;
  uint32 candidate_k = 3;
  uint32 raw_candidate_count = 4;
  uint32 returned_group_count = 5;
}
```

### Field placement

| Message              | Field | Type             | Notes                            |
|---------------------|-------|------------------|----------------------------------|
| `SearchRequest`     | 5     | `GroupingOptions`| Optional, absent = no grouping   |
| `SearchResponse`    | 5     | `GroupingInfo`   | Present iff grouping was active  |
| `SearchBatchRequest`| 6     | `GroupingOptions`| Optional, applies to all queries |
| `QueryResults`      | 2     | `GroupingInfo`   | Per-query grouping diagnostics   |

No existing field numbers are changed.

## Grouping Semantics

### When grouping is active (`group_by = GROUP_BY_PARENT_ID`)

1. Server runs ANN search with `candidate_k` vectors (over‑fetch).
2. For each candidate, read the stored `parent_id` via `index.get_metadata()`.
3. Group candidates by parent ID.
4. If a vector has no `parent_id`, use its own vector ID as the fallback group key.
5. Within each group, keep only the best candidate (lowest distance).
6. Truncate to `k` groups, preserving distance order.
7. Responses carry `GroupingInfo` with diagnostics.

### Defaults and validation

| Parameter      | Default                | Validation                       |
|---------------|------------------------|----------------------------------|
| `candidate_k` | `max(k * 4, k)`        | Must be ≥ k if grouping is set   |
| `max_per_group`| 1                     | 0 → treated as 1; >1 supported   |

### When grouping is absent

- The existing unfiltered‑search path is used unchanged.
- No `GroupingInfo` appears in the response.

## Fallback behaviour

When a vector has no `parent_id` metadata, its own *vector ID* becomes the
group key.  This ensures every vector still appears at most once in a grouped
result set (its own ID is unique), preserving backward‑compatible behaviour
for un‑annotated inserts.

## Batch‑search behaviour

If `SearchBatchRequest.grouping` is set, the same grouping options apply to
every query.  Each per‑query `QueryResults` carries its own `GroupingInfo`
so callers can inspect diagnostics per query.

## Tests (sochdb-grpc)

| Test                                      | What it proves                                      |
|-------------------------------------------|-----------------------------------------------------|
| grouped_search_returns_unique_parents     | Two views from same parent collapse to one          |
| grouped_search_preserves_best_distance    | The closer view wins within a group                 |
| grouped_search_missing_parent_fallback    | Vectors without parent_id group by vector ID        |
| grouped_search_parent_zero_is_grouped     | parent_id=0 is treated like any other parent        |
| grouped_search_candidate_overfetch        | candidate_k > k recovers parents hidden by dupes     |
| grouped_search_batch_grouping             | Batch search honors the same grouping options        |
| grouped_search_rejects_invalid_options    | candidate_k < k or other invalid config → error      |
| grouped_search_response_contains_info     | GroupingInfo is populated with correct diagnostics  |
| ungrouped_search_is_unchanged             | Search without grouping returns identical results   |
| grouped_search_without_metadata_is_safe   | Empty index with grouping returns empty results     |

## Compatibility risks

- Proto additions are backward‑compatible (new optional fields).
- Old clients that never set `grouping` see identical behaviour.
- Old servers that receive the new fields ignore them (proto3 default).
- Binary wire format is fully backward‑compatible.

## Forbidden scope

- No LoCoMo / benchmark changes
- No metadata filtering
- No BM25 / hybrid search
- No package version bumps
- No CI / workflow changes
- No HNSW tuning
- No distance metric rewrites
