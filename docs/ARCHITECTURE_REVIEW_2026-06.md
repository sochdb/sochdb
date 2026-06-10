# SochDB Architecture Review — June 2026 (v2.0.2)

**Scope:** Storage durability, MVCC concurrency, HNSW/vector indexing, FFI safety,
vector quantization, and the high-level client mutation API.

**Method:** Static reading of the architecturally load-bearing modules. No code was
executed; all concurrency/liveness claims are from static analysis and are flagged as
such. Every finding cites concrete code that was verified against the tree at v2.0.2.

**How to read this document:** Section 1 records what is *genuinely solid* so it is not
"fixed" by mistake. Section 2 records the seven findings where the implementation diverges
from the architecture the code advertises. The companion file
[REMEDIATION_CHECKLIST.md](REMEDIATION_CHECKLIST.md) turns these into a dependency-ordered,
actionable plan.

---

## 1. Verified solid (do not "fix")

- **Wired durability path is real.** `DurableStorage` → `TxnWal` flushes and `sync_all()`s on
  commit and has a real recovery replay (`replay_for_recovery`). This is the shipping path.
- **FFI panic firewall is correctly applied.** Every `extern "C"` body opens with the
  `ffi_guard!{…}` macro (`catch_unwind` + stable error sentinel). This is the precondition the
  release profile's `panic="unwind"` relies on. The "unguarded" appearances are multi-line
  signature false positives.
- **HNSW concurrency is honest and sophisticated.** Explicitly *not* lock-free; uses DashMap
  node storage, per-layer `RwLock`, and packed atomic navigation state. The one true ceiling is
  documented in-code (see Finding 5).
- **Vector deletes are a sound design, not a missing feature.** Handled at the `sochdb-vector`
  segment layer via tombstones + compaction (Lucene/Milvus-style immutable segments), not in the
  HNSW graph.
- **Block compression is live.** `lz4`/`zstd` codecs do real work; `select_compression` picks by
  content type. The "stub returns None" text is a stale test comment, not the live code.
- **Cost optimizer uses real sketch-based cardinality estimation**, not constants.

---

## 2. Findings

Each finding lists the verified evidence location, the consequence, and the corrective
direction. Severity legend: **S1** silent-correctness/durability on a user-facing path ·
**S2** trust-surface / latent defect · **S3** scalability ceiling · **S4** hygiene/docs.

### Finding 1 — UPDATE/DELETE mutations are not wired into the durable path — **S1**

- **Evidence:** [sochdb-client/src/crud.rs](../sochdb-client/src/crud.rs#L180),
  [crud.rs](../sochdb-client/src/crud.rs#L232),
  [crud.rs](../sochdb-client/src/crud.rs#L499),
  [crud.rs](../sochdb-client/src/crud.rs#L517) — four
  `// TODO: Wire mutation_result.affected_row_ids to storage backend / transaction WAL`.
- **Behavior:** `UpdateBuilder::execute` / `DeleteBuilder::execute` mutate only the in-memory
  Trie-Columnar Hybrid (`tch.write().update_rows(...)` / `delete_rows(...)`). The computed
  `affected_row_ids` is then discarded. The connection already holds `storage: Arc<…>` (the
  canonical WAL/MVCC/SSI engine), but the mutation never reaches it.
- **Consequence:** A builder-API UPDATE/DELETE is visible to same-process TCH reads but is **not**
  WAL-logged (lost on crash), **not** reflected in secondary/vector indexes (stale recall, phantom
  rows), and **not** emitted to CDC. The TCH and `DurableStorage` diverge after the first mutation.
- **Required direction:** write-through with redo/undo logging. Capture the **before-image** (today
  only `affected_count` is retained, so ARIES UNDO of an UPDATE is undefined), then
  `storage.apply_mutation(txn, before, after)` → `TxnWal.append` → index maintenance →
  `cdc.emit` → commit (flush + sync).
- **Open question:** No TCH→storage checkpoint/sync entry point was found
  (`checkpoint.rs` in the client is unrelated workflow-state storage). An indirect sync path may
  exist but was not traced; the authors' own TODOs indicate the wiring is absent.

### Finding 2 — Product Quantization silently returns zero vectors — **S1**

- **Evidence:** [sochdb-index/src/unified_quant.rs](../sochdb-index/src/unified_quant.rs#L200)
  ("PQ decode requires codebooks - return zeros as placeholder") and the `from_f32` /
  encode path at [unified_quant.rs](../sochdb-index/src/unified_quant.rs#L234).
- **Behavior:** `from_f32(QuantLevel::PQ)` trains no codebook (allocates all-zero codes); `decode()`
  for PQ returns `vec![0.0; dimension]`. The zero vector is then fed into distance computation.
- **Consequence:** Every PQ-quantized vector collapses to the origin, so distances are meaningless
  and recall is effectively random — **with no error surfaced**. F16/BF16/I8 quantization is real;
  only PQ is a non-functional placeholder masquerading as a supported `QuantLevel`. A
  silent-wrong-answer path in an ANN engine is worse than an unsupported one.
- **Required direction:** *Interim (trivial, land immediately):* make `QuantLevel::PQ` return a typed
  `Err(Unsupported("PQ requires trained codebook"))` at the API boundary. *Full:* per-subspace
  k-means codebook training, nearest-centroid encode, ADC search with a precomputed m×256
  query→centroid distance table.

### Finding 3 — "Quarantined: unwired" WAL/recovery modules + a latent group-commit liveness bug — **S2** — *RESOLVED (2026-06)*

> **Status update (2026-06):** Two corrections to the original finding, plus a fix.
> 1. The quarantined modules are gated behind `#[cfg(feature = "experimental")]` in
>    [lib.rs](../sochdb-storage/src/lib.rs#L108), so they do **not** ship in the default build or
>    default public API — the "ships in v2.0.2 / pollutes the public API" concern is mitigated by
>    feature-gating (the original claim that they were unconditional `pub mod` was based on a
>    flattened dump without cfg context).
> 2. The latent `commit_txn` hang has been **fixed** (see below): a non-flushing committer now
>    parks on `flush_cv.wait_timeout(buffer, remaining)` and self-flushes if no peer did, and every
>    `flush_buffer_locked` now `notify_all`s parked committers. Regression test
>    `test_lone_commit_does_not_hang` added; the previously-`#[ignore]`d `test_wal_basic_operations`
>    (which calls `commit_txn`) now passes.

- **Evidence (original):** [sochdb-storage/src/lib.rs](../sochdb-storage/src/lib.rs#L79) and
  following — seven modules tagged `[quarantined: unwired]` (`aries_recovery`, `checkpoint`,
  `pitr`, `production_wal`, `wal_fencing`, `columnar_wal`, `io_uring_wal`), now confirmed gated
  behind the `experimental` feature.
- **Documented ≠ real hazard:** `production_wal` ("ARIES recovery", "group commit") is **not** the
  shipping path (the engine wires `TxnWal`); it is only built under the `experimental` feature.
- **Latent liveness defect (now fixed):**
  `WriteAheadLog::commit_txn` enqueued a waiter then blocked on `rx.recv()`, but only flushed
  `if buffer.should_flush(...)`. With no background flusher and the `flush_cv` Condvar never
  waited/notified, a solo or trailing commit blocked indefinitely. Fixed via leader/timeout-driven
  group commit.
- **Uncertainty:** Static analysis. Because the module was feature-gated and unwired, the bug was
  **latent**, not a live outage — which is why it was fixed pre-emptively before any promotion.
- **Required direction:** delete the quarantined modules (retain `TxnWal`/`durable_storage`), or
  seal them behind `#[cfg(test)]` / a non-`pub` `experimental` module; if `production_wal` is to be
  wired later, first add a flusher thread (`flush_cv.wait_timeout(buf, flush_interval)` → coalesced
  `sync_all` → notify waiters) or a leader/timeout-driven group-commit.

### Finding 4 — Detached background workers are unsupervised; hot-path `unwrap()`/`expect()` panics — **S2**

- **Evidence:** ~75 `thread::spawn` sites; ~3,728 `.unwrap()` and ~213 `.expect()` across the tree.
- **Behavior:** With release `panic="unwind"`, an FFI-boundary panic is contained by the firewall,
  but a panic in a **detached** background worker (LSM compaction, GC, the event-driven flusher,
  HNSW connectivity repair) terminates only that thread. No visible supervisor retains/monitors the
  `JoinHandle` or restarts it.
- **Consequence:** Silent degradation. The DB keeps serving while compaction stalls (space + read
  amplification grow), GC stops (tombstones/WAL accumulate), or `flat_neighbors` is never rebuilt
  (`flat_neighbors_valid` stuck false → slow-path search). The system looks healthy while
  monotonically degrading.
- **Uncertainty:** Spawn sites were sampled, not exhaustively audited; some workers may have local
  catch/restart. Recommendation is to make supervision a uniform, enforced policy.
- **Required direction:** a `Supervisor::spawn(name, worker)` wrapper with per-iteration
  `catch_unwind`, restart with bounded backoff, a panic metric, and a `worker_health` signal.

### Finding 5 — Global `vector_store` RwLock is an HNSW scalability ceiling — **S3**

- **Evidence:** [sochdb-index/src/hnsw.rs](../sochdb-index/src/hnsw.rs#L2226)
  ("⚠️ KNOWN BOTTLENECK: global RwLock serialises concurrent searches vs inserts") and
  [TODO(T7)](../sochdb-index/src/hnsw.rs#L2227). A `ShardedVectorStore` struct already exists
  ([hnsw.rs](../sochdb-index/src/hnsw.rs#L2104)) but is not wired in.
- **Behavior:** Node storage (DashMap) and nav state (atomics) are concurrent, but the contiguous
  vector slab is a single `RwLock<Vec<…>>`. Any insert takes the write lock and excludes all
  concurrent search reads of the slab.
- **Consequence:** Reader parallelism collapses under mixed read/write workloads. This is a
  scalability ceiling, not a correctness bug.
- **Required direction:** lock striping — `ShardedVectorStore { shards: [RwLock<Vec<…>>; 64] }`,
  `shard = dense_index % 64`. In-code estimate: ~3–5× throughput under concurrent search+insert.
  Trade-off: a cross-shard scan touches S locks (O(S)).

### Finding 6 — Single-writer MVCC ceiling is correct-as-designed but under-documented — **S3**

- **Evidence:** `sochdb-storage/src/mvcc_concurrent.rs` — mmap'd metadata file with one
  `writer_lock: AtomicU32` (0=free, else pid). Readers are lock-free via snapshot timestamps (HLC);
  writers are serialized by the single lock. No concurrent-writer protocol exists by construction.
- **Behavior:** A legitimate SQLite-class embedded design, but a hard write-throughput ceiling for
  multi-process or write-heavy deployments, and easy for an integrator to miss.
- **Required direction:** keep the model; add an **in-process** batching layer that coalesces many
  logical writes under one held `writer_lock` (one critical section, one fsync via the wired
  group-commit path), and document the ceiling prominently. Throughput goes from
  `≤ 1/(t_crit + t_fsync)` to `B/(t_crit + t_fsync)` for batch size `B`.

### Finding 7 — Stale connection docs + overdue no-durability/no-MVCC stubs — **S4**

- **Evidence:** [sochdb-client/src/connection.rs](../sochdb-client/src/connection.rs#L2304)
  ("⚠️ NOT FOR PRODUCTION USE") contradicts the same type's field doc at
  [connection.rs](../sochdb-client/src/connection.rs#L2339) ("canonical engine … DurableStorage with
  WAL, MVCC, and SSI"). Deprecated stubs `WalStorageManager`
  ([connection.rs](../sochdb-client/src/connection.rs#L160)) and the in-memory txn stub
  ([connection.rs](../sochdb-client/src/connection.rs#L223)) carry
  `REMOVAL SCHEDULED: v0.3.0` yet remain present and constructible at **v2.0.2**.
- **Consequence:** A reviewer cannot trust either doc statement; a "no durability" type that still
  compiles is a footgun for anyone instantiating it expecting persistence.
- **Required direction:** rewrite the type doc to state actual behavior; delete the v0.3.0-scheduled
  stubs or quarantine them behind `#[cfg(test)]` / a non-`pub` `experimental` module.

---

## 3. Open questions for a maintainer

1. Does any indirect sync flush TCH mutations to `DurableStorage`? (Finding 1)
2. Do the ~75 spawn sites include local restart logic? (Finding 4)
3. What is the on-disk-format relationship between `txn_wal` (wired) and the quarantined
   `aries_recovery`/`production_wal` formats — would a future switch require migration? (Finding 3)
