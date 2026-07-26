# Search Provenance

Vector search requests can opt into summary diagnostics with
`include_diagnostics`. The default response is unchanged and does not include
diagnostics or per-result ranks.

Diagnostics report the requested and effective candidate limits, candidates
considered, result counts, distance metric, grouping summary, and whether the
candidate set was truncated. When grouping is enabled, `groups_considered`,
`groups_returned`, and `excluded_by_group_limit` describe the parent grouping
stage. `max_groups` is zero because the current grouping API does not impose a
maximum group count.

Returned results are ordered by distance ascending, then by stable vector ID
ascending when distances are equal or numerically indistinguishable. This
ordering is shared by single and batch search and applies before grouping.
Grouped search preserves that order after selecting the best allowed result
per group. Diagnostics add only summary data; they do not expose vectors,
internal graph state, or complete candidate exclusion traces.

The first implementation intentionally does not report every excluded
candidate. Such a trace would scale with the candidate set and is outside the
default-size observability contract. Persistence is unaffected because all
diagnostics are computed at request time.
