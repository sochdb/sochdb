# SochDB Remediation Checklist (v2.0.2)

Dependency-ordered, actionable plan derived from
[ARCHITECTURE_REVIEW_2026-06.md](ARCHITECTURE_REVIEW_2026-06.md). Severity: **S1**
silent-correctness/durability · **S2** trust-surface/latent defect · **S3** scalability ·
**S4** hygiene. Each task lists prerequisites, concrete steps, and acceptance criteria.

## Ordering rationale

1. **Task 2a** (PQ fail-fast) — trivial, stops a live silent-wrong-answer path immediately.
2. **Task 1** (wire UPDATE/DELETE) — live durability/coherence failure; depends on first settling
   the TCH-vs-`DurableStorage` source-of-truth question.
3. **Task 3** + **Task 7** — trust-surface and API hygiene; batch together (both touch
   quarantined/deprecated public surface).
4. **Task 4** — protects long-running availability.
5. **Tasks 5, 6** — scale/throughput with self-documented payoffs.
6. **Task 2b** (full PQ) — schedule after the fail-fast guard is in place.

---

## [x] Task 2a — PQ fail-fast (S1, XS) — DONE

- **Prereq:** none.
- **Files:** [sochdb-index/src/unified_quant.rs](../sochdb-index/src/unified_quant.rs#L200),
  [unified_quant.rs](../sochdb-index/src/unified_quant.rs#L234).
- **Steps:**
  1. Make `from_f32(QuantLevel::PQ)` / `decode` PQ-branch return a typed
     `Err(Unsupported("PQ requires trained codebook"))` instead of zero codes/zeros.
  2. Ensure the error propagates to the public API boundary; no path returns origin vectors.
- **Acceptance:** selecting PQ without a trained codebook fails with a typed error; no test or
  code path produces all-zero PQ vectors. Existing F16/BF16/I8 paths unchanged.

## [x] Task 1 — Wire UPDATE/DELETE into the durable path (S1, L) — DONE

- **Prereq:** decide source-of-truth (TCH must become a cache of `DurableStorage`, not a fork).
- **Files:** [sochdb-client/src/crud.rs](../sochdb-client/src/crud.rs#L180) (update),
  [crud.rs](../sochdb-client/src/crud.rs#L232) (delete),
  [crud.rs](../sochdb-client/src/crud.rs#L499) /
  [crud.rs](../sochdb-client/src/crud.rs#L517) (txn paths).
- **Steps:**
  1. Capture **before-image** + after-image (not just `affected_count`) so ARIES UNDO is defined.
  2. Route `affected_row_ids` to `storage.apply_mutation(txn, before, after)` →
     `TxnWal.append(Update/Delete)` → secondary/vector index maintenance → `cdc.emit`.
  3. Commit through the wired flush + `sync_all` path.
  4. Remove the four `// TODO: Wire mutation_result.affected_row_ids …` markers.
- **Acceptance:** UPDATE/DELETE via the builder API is WAL-logged, survives a crash/replay, is
  reflected in secondary/vector indexes, and emits CDC; TCH and `DurableStorage` agree after
  mutation. Add a crash-recovery integration test.
- **Resolve open question first:** confirm no indirect TCH→storage sync already exists.

## [~] Task 3 — Quarantined WAL modules: liveness hang FIXED; sealing already in place (S2, M)

- **Update (2026-06):** the quarantined modules (`production_wal`, `pitr`, `aries_recovery`,
  `checkpoint`, `wal_fencing`, `columnar_wal`, `io_uring_wal`) are gated behind
  `#[cfg(feature = "experimental")]` in [lib.rs](../sochdb-storage/src/lib.rs#L108), so they do
  **not** ship in the default build — the "pollutes the public API / ships a buggy module"
  concern is already mitigated by feature-gating.
- **Done:** fixed the latent `commit_txn` group-commit hang in
  [production_wal.rs](../sochdb-storage/src/production_wal.rs) using a leader/timeout-driven group
  commit: a non-flushing committer now parks on `flush_cv.wait_timeout(buffer, remaining)` and
  flushes the batch itself if no peer did; every `flush_buffer_locked` now `notify_all`s parked
  committers. Added a non-ignored regression test `test_lone_commit_does_not_hang`; the
  previously-`#[ignore]`d `test_wal_basic_operations` (which calls `commit_txn`) now passes.
- **Remaining (maintainer decision):** whether to fully delete the experimental modules vs keep
  them feature-gated; confirm on-disk-format relationship between `txn_wal` (wired) and
  `aries_recovery`/`production_wal` before any future wiring.

## [ ] Task 3-orig — Seal/remove quarantined WAL modules (superseded by above)

- **Prereq:** confirm on-disk-format relationship between `txn_wal` (wired) and
  `aries_recovery`/`production_wal` (would wiring require migration?).
- **Files:** [sochdb-storage/src/lib.rs](../sochdb-storage/src/lib.rs#L79) (exports) +
  `production_wal.rs`, `aries_recovery.rs`, `checkpoint.rs`, `pitr.rs`, `wal_fencing.rs`,
  `columnar_wal.rs`, `io_uring_wal.rs`.
- **Choose one path:**
  - **A — Delete (preferred):** remove the seven `[quarantined: unwired]` modules; retain
    `TxnWal`/`durable_storage`. Reduces public API, binary size, compile time, and trust surface.
  - **B — Fix-then-seal:** make them `#[cfg(test)]` or a non-`pub experimental` module, AND fix the
    [`commit_txn`](../sochdb-storage/src/production_wal.rs#L672) hang: spawn a flusher thread that
    does `flush_cv.wait_timeout(buf, flush_interval)` → coalesced `sync_all` → notify waiters (or a
    leader/timeout-driven group commit). The declared
    [`flush_cv`](../sochdb-storage/src/production_wal.rs#L500) must actually be waited/notified.
- **Acceptance:** one authoritative WAL/recovery implementation reachable from the public API; no
  public module that can block a commit indefinitely; module docs match the wired path.

## [x] Task 7 — Reconcile connection docs; remove overdue stubs (S4, S) — DONE

- **Files:** [sochdb-client/src/connection.rs](../sochdb-client/src/connection.rs#L2304) (type doc
  vs [field doc](../sochdb-client/src/connection.rs#L2339)),
  [WalStorageManager](../sochdb-client/src/connection.rs#L160),
  [txn stub](../sochdb-client/src/connection.rs#L223).
- **Steps:**
  1. Rewrite the `SochConnection` type doc to state actual behavior (durable-backed vs
     ephemeral-temp-dir); remove the contradictory "NOT FOR PRODUCTION / no MVCC / no ACID" header
     if the type is in fact durable-backed.
  2. Delete the `REMOVAL SCHEDULED: v0.3.0` stubs, or quarantine behind `#[cfg(test)]` /
     non-`pub experimental`.
- **Acceptance:** no contradictory docs on `SochConnection`; no public constructor that silently
  provides zero durability.

## [x] Task 4 — Supervise background workers; harden hot-path panics — DONE (S2, M)

- **Files:** [sochdb-storage/src/supervisor.rs](../sochdb-storage/src/supervisor.rs) (new),
  [sochdb-storage/src/dirty_tracking.rs](../sochdb-storage/src/dirty_tracking.rs) (migrated).
- **Done:**
  1. Added `Supervisor::spawn(running, body)` returning a `SupervisedWorker`: each iteration of the
     worker body runs inside `catch_unwind`, so a panic is **contained** — it is counted, the worker
     is flagged unhealthy, a geometric bounded backoff (`base_backoff`→`max_backoff`, reset on the
     next success) is applied, and the loop **restarts** instead of dying. Liveness is observable via
     `WorkerHealth` (`panics()`, `restarts()`, `iterations()`, `is_healthy()`, `is_finished()`).
  2. Migrated the detached dirty-tracking aggregator (`BatchedDirtyTracker`) onto it as the first
     adopter; `aggregator_health()` exposes the signal. Cooperative shutdown via the shared
     `Arc<AtomicBool>` is preserved.
- **Tests:** fault-injection `test_panic_is_contained_and_loop_survives` (worker panics then recovers
  and makes progress), `test_health_unhealthy_immediately_after_panic`, plus clean-shutdown and
  stop-step tests; dirty_tracking suite still green.
- **Note:** the remaining detached workers (LSM compaction, GC, event-driven flusher, HNSW repair)
  can now be migrated incrementally onto the same `Supervisor` primitive.

## [x] Task 5 — Shard the HNSW `vector_store` RwLock — VALIDATED; wiring benchmark-gated (S3, M)

- **Files:** [ShardedVectorStore](../sochdb-index/src/hnsw.rs#L2104),
  [vector_store bottleneck](../sochdb-index/src/hnsw.rs#L2226).
- **Done:** the drafted 64-shard `ShardedVectorStore` is now **unit-tested and validated**
  (`sharded_vector_store_tests`): `push` hands out unique sequential indices under 8-thread
  contention, and `get`/`set`/`with`/`len`/`clear` all round-trip. It is a tested, ready component
  rather than untested dead code.
- **Wiring deferred (benchmark-gated, by design):** flipping `HnswIndex.vector_store` to the sharded
  store is **not** a mechanical swap. The distance hot path holds borrows from a single
  `vector_store.read()` across loops (`g.get(i).unwrap_or(&node.vector)`); the sharded API returns
  owned values (`get`) or a closure borrow (`with`), so ~20 callsites must be restructured, and the
  contiguous-slab sequential-scan **cache locality** that distance computation depends on is traded
  for insert concurrency. The in-code estimate (~3–5× on mixed workloads) is unvalidated, so this
  must be measured before flipping the core index — doing it blind risks a recall/latency
  regression. Documented in the in-code header at `ShardedVectorStore`.

## [~] Task 6 — MVCC single-writer ceiling: DOCUMENTED; coalescing remains (S3, M)

- **Files:** [sochdb-storage/src/mvcc_concurrent.rs](../sochdb-storage/src/mvcc_concurrent.rs).
- **Done:** added a prominent "Concurrency Contract" section to the module docs: Multi-Reader
  **Single-Writer**, snapshot isolation for readers against a serial writer timeline, and the
  explicit throughput ceiling `write_throughput ≤ 1/(t_crit + t_fsync)` with the batching payoff
  `B/(t_crit + t_fsync)`.
- **Coalescing already exists — DONE:** the write-coalescing layer is **already implemented and
  wired** as [`EventDrivenGroupCommit`](../sochdb-storage/src/group_commit.rs#L82) (single fsync per
  batch, adaptive sizing via Little's Law `N* = sqrt(2·L_fsync·λ/C_wait)`), constructed and routed
  through `TxnWal` by [`durable_storage`](../sochdb-storage/src/durable_storage.rs#L2511). The MVCC
  contract doc now cross-references it. A second in-MVCC coalescing layer was intentionally **not**
  added (it would duplicate/contend with the existing one and risk the isolation contract).

## [x] Task 2b — Full Product Quantization — DONE (S1→feature, L)

- **Prereq:** Task 2a landed (fail-fast guard in place).
- **Files:** [sochdb-index/src/unified_quant.rs](../sochdb-index/src/unified_quant.rs#L296),
  [sochdb-index/src/product_quantization.rs](../sochdb-index/src/product_quantization.rs).
- **Done:** the real PQ engine (`product_quantization`) already implements per-subspace k-means
  (Lloyd + k-means++ init, 256 centroids → 1 byte/subspace), `encode`, `decode`, and ADC search via
  a precomputed query→centroid `DistanceTable`. Wired it into the unified format with two new
  codebook-aware methods on `UnifiedQuantizedVector`:
  `from_f32_pq(data, &PQCodebooks)` (real nearest-centroid encode) and
  `to_f32_pq(&PQCodebooks)` (centroid-concatenation reconstruction). The codebook-less
  `from_f32(_, PQ)` keeps the Task-2a fail-fast guard; the codebook precondition is now satisfied by
  the codebook-aware path.
- **Tests:** `test_pq_roundtrip_with_codebook` (train → encode → decode, non-zero + bounded MSE),
  `test_to_f32_pq_rejects_non_pq`.

---

## Progress tracker

| Task | Severity | Effort | Depends on | Status |
|------|----------|--------|------------|--------|
| 2a — PQ fail-fast | S1 | XS | — | [x] done |
| 1 — Wire UPDATE/DELETE | S1 | L | source-of-truth decision | [x] done |
| 3 — Fix group-commit hang | S2 | M | (modules already feature-gated) | [x] hang fixed |
| 7 — Reconcile docs/stubs | S4 | S | (batch w/ 3) | [x] done |
| 4 — Supervise workers | S2 | M | spawn-site audit | [x] done (Supervisor + dirty-aggregator) |
| 5 — Shard vector_store | S3 | M | benchmark | [x] validated; wiring benchmark-gated |
| 6 — Document MVCC ceiling | S3 | M | — | [x] done (coalescing = EventDrivenGroupCommit) |
| 2b — Full PQ | feature | L | 2a | [x] done |
