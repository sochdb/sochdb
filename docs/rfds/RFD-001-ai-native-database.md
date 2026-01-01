# RFD: ToonDB — AI‑Native Database for Agent Memory & Context

**Author:** sushanth reddy vangala (https://github.com/sushanthpy)
**Status:** Draft (Request for Discussion)
**Last updated:** 2025‑12‑31
**Audience:** Agent/LLM application engineers, platform engineers, database/infra engineers

---

## 1. Summary

**ToonDB** is an **LLM‑native database** designed to offload **contextual engineering** (context assembly, token budgeting, truncation, and compact serialization) so teams can focus on **agent orchestration** (routing, tool policy, planning, graph execution).

ToonDB combines:

* **Token‑optimized output** (TOON / tabular LLM‑friendly encoding)
* **Context Query Builder** (assemble multi‑section context under a token budget)
* **O(|path|) lookups** for agent state / memory
* **Built‑in vector search (HNSW)**
* **ACID durability** via MVCC + WAL

**Proposed designation:** Adopt ToonDB as the default **agent memory + retrieval substrate** for projects that need reliable local/edge memory, deterministic context assembly, and fast retrieval without stitching multiple stores.

---

## 2. Problem

Production agent stacks repeatedly re‑implement the same "boring middle":

1. **Memory retrieval**

* Fetch user profile
* Fetch last N turns
* Fetch tool results
* Fetch relevant docs (vector search)

2. **Context construction**

* Merge sources into a prompt template
* Enforce a strict token budget
* Truncate safely without breaking semantics
* Serialize efficiently (avoid JSON bloat)

3. **Operational risks**

* Token blowups causing latency and cost spikes
* Drift between chat history store and vector store
* Inconsistent truncation rules across workflows
* Debuggability gaps ("why did the agent do that?")

**Outcome:** teams spend significant engineering time building glue, and the context pipeline becomes fragile and hard to reason about.

---

## 3. Goals / Non‑goals

### Goals

* Provide a **single substrate** for:

  * agent state (KV/path)
  * retrieval (vector search)
  * deterministic context assembly (budgeting, truncation, formatting)
* Improve **developer velocity** by eliminating prompt‑packing glue code
* Reduce **LLM costs** by making compact context the default
* Support **embedded‑first deployments** (local apps, edge agents) while allowing service/sidecar patterns
* Offer clear integration points with modern agent orchestrators (LangGraph, CrewAI, custom)

### Non‑goals (current)

* Global distributed database (clustering/replication)
* Full relational SQL feature parity
* Serving as a general analytics warehouse

---

## 4. Proposed Solution

### 4.1 Core concept: Offload contextual engineering

Instead of each orchestrator implementing prompt assembly, ToonDB introduces **Context Queries**:

* The application declares **sections** (SYSTEM/USER/HISTORY/DOCS/etc.)
* Each section declares how to fetch data (path lookups, projections, vector search)
* The query executes with a **token budget** and a **truncation policy**
* Output is returned in a compact, model‑friendly format (e.g., TOON)

This turns context construction into a **queryable primitive**—testable, reproducible, and consistent across workflows.

### 4.2 Data model primitives

* **Path‑first KV**: hierarchical keys for agent state

  * Example: `users/123/preferences`, `sessions/abc/history/…`
  * **O(|path|)** resolution target

* **Columnar storage** for structured records with projection pushdown

* **Vector indexes (HNSW)** for retrieval

* **ACID transactions** using MVCC + WAL

### 4.3 Access modes

* **Embedded mode (FFI)** for lowest latency and simplest deployment
* **IPC / sidecar** for multi‑process access (unix sockets)
* **gRPC vector service** for polyglot or service boundary deployment
* **MCP server** for tool‑based access in agent runtimes/editors

---

## 5. How ToonDB fits into an agentic stack

### 5.1 LangGraph (graph orchestrators)

Recommended node pattern:

1. **Context Builder node**

* Run one Context Query to produce the LLM input context under budget

2. **LLM node**

* Execute the model call using ToonDB context output

3. **Tool nodes**

* Use ToonDB as a tool (via MCP) or as an internal library for writes

4. **Commit node**

* Persist new memories, tool results, and embeddings in a single transaction

### 5.2 CrewAI / multi-agent role systems

* Expose ToonDB as a shared "Memory & Retrieval Tool" with namespaced scopes
* Agents write structured observations, retrieved evidence, and summaries back to ToonDB
* Context queries enforce consistent packing and truncation across roles

### 5.3 Custom orchestrators

* Treat ToonDB as the **memory plane**
* Keep orchestration logic (routing, policies, plans) in the orchestrator
* Move context selection, formatting, and budgeting into ToonDB

---

## 6. Benefits

### 6.1 Developer velocity

* Removes repeated implementation of:

  * token counting / budget enforcement
  * truncation heuristics
  * prompt packing / templating glue
  * multi-store synchronization logic

### 6.2 Cost and performance

* Token‑optimized context reduces LLM spend and latency
* Projection pushdown reduces IO and unnecessary decoding
* Bulk vector operations avoid language boundary overhead in ingestion

### 6.3 Reliability and consistency

* Deterministic context assembly: same query, same rules
* Durable state: transactions + WAL
* Debuggable: context queries become an artifact you can log and test

---

## 7. Design Details (Proposed API shapes)

### 7.1 Context Query DSL

Pseudo‑interface:

* `ContextQueryBuilder::new()`
* `.with_budget(tokens)`
* `.section(name, priority)`
* `.get(path / projection)`
* `.last(n, history_path)`
* `.search(collection, embedding, k)`
* `.format(TOON | JSON | …)`

Output:

* `ContextPayload { sections: […], tokens_used, truncation_events, … }`

### 7.2 Observability hooks

Recommended standard fields:

* `context_tokens_budget`
* `context_tokens_used`
* `sections_included`
* `sections_truncated`
* `retrieval_hits`
* `vector_search_latency_ms`
* `txn_commit_latency_ms`

---

## 8. Alternatives considered

### A) "Framework-only" approach (LangGraph/CrewAI + standard stores)

Pros:

* Familiar, modular

Cons:

* Context assembly remains bespoke and duplicated
* Token and truncation behavior inconsistent across flows
* Multiple persistence layers increase operational and logical complexity

### B) Dedicated vector DB + Redis/Postgres

Pros:

* Strong ecosystem

Cons:

* Still requires a prompt packer and coordination logic
* Token bloat persists (JSON-heavy retrieval payloads)
* Harder to make deterministic and testable context rules

---

## 9. Risks and Mitigations

### 9.1 Security boundary for remote access

Risk:

* Service deployment requires authentication/authorization and transport security

Mitigations:

* Add mTLS and/or JWT auth for gRPC
* Derive/validate auth scopes server-side
* Provide structured audit logs (and later OpenTelemetry)

### 9.2 Single-node constraints

Risk:

* Not suitable for multi-region HA requirements yet

Mitigations:

* Position as embedded/sidecar first
* Support export/import and backup primitives
* Define a replication roadmap if demand proves strong

### 9.3 Compatibility with agent tooling

Risk:

* Users want drop-in integrations

Mitigations:

* Provide first-party adapters:

  * LangGraph "context node"
  * CrewAI tool wrapper
  * MCP tools with good ergonomics

---

## 10. Open Questions

1. What is the default truncation policy per section type?
2. What are the minimum observability requirements for production adoption?
3. How should schema evolution be handled for agent memory tables?
4. What's the compatibility story for embeddings models and dimensions?

---

## 11. Proposed milestones

**M0 (now):**

* Stable embedded mode + Python SDK
* Context Query Builder MVP + TOON output
* Built-in HNSW retrieval

**M1:**

* Authenticated gRPC service mode (mTLS/JWT)
* Context query explain plan / debug trace
* OTEL metrics + tracing

**M2:**

* Replication design doc (if demand)
* Multi-tenant namespaces + quotas
* Managed lifecycle for checkpoints / WAL compaction

---

## 12. Discussion prompts

* Would you adopt ToonDB as a default memory substrate in your agent stack? Why/why not?
* Which integration path matters most: embedded, MCP, or gRPC?
* What production blocker is most urgent: security, observability, replication, or tooling adapters?
