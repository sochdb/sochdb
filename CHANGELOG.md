# Changelog

All notable changes to ToonDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.3.1] - 2026-01-04

### Added
- **Anonymous Usage Analytics** — Optional, privacy-respecting usage information collection
  - `database_opened` event — Track database initialization across SDKs
  - `error` event — Static error tracking (error type + code location only, no sensitive data)
  - Stable anonymous machine ID (SHA-256 hash, no PII)
  - Environment variable opt-out: `TOONDB_DISABLE_ANALYTICS=true`
  - Graceful degradation when analytics dependencies unavailable
  - Python: Optional `posthog` dependency in `[analytics]` extra
  - JavaScript: Optional `posthog-node` in `optionalDependencies`
  - Rust: Feature flag `analytics` for `toondb-core`

### Changed
- **Analytics Privacy-First Design** — No sensitive data collection
  - Error tracking sends only static `error_type` and `location` (e.g., "query_error" @ "sql.execute")
  - No dynamic error messages, user data, file paths, or identifiable information
  - SDK context: version, OS, architecture only
  - All data sent to PostHog with user-controlled opt-out
- **Documentation Updates** — Version references updated to 0.3.1 across all guides
  - Updated benchmark versions in README
  - Updated installation instructions in all SDK guides
  - Consistent versioning across Python, JavaScript, Rust, and Go documentation

### Fixed
- **JavaScript Analytics Import** — Fixed ESM import path in `database.ts` (`.js` extension required)
- **Rust Analytics Feature** — Enabled `analytics` feature by default in client crates
  - Added to `toondb-client`, `toondb-python`, `toondb-grpc`
  - Added `json` feature to `ureq` dependency for PostHog integration
- **Test Version Mismatch** — Updated JavaScript test expectations to match 0.3.1 version

---

## [0.3.0] - 2026-01-03

### Added
- **Namespace Isolation** — Type-safe multi-tenancy with logical database namespaces
  - `Namespace` and `Collection` APIs for tenant-scoped data isolation
  - No key prefixing required — true isolation at the storage layer
  - Per-namespace configuration and collection management
  - Available in Python, Go, Node.js, and Rust SDKs
- **Hybrid Search** — Dense vector (HNSW) + sparse BM25 text search
  - `HybridSearchEngine` with Reciprocal Rank Fusion (RRF)
  - Configurable alpha weighting (vector vs. keyword balance)
  - `InvertedIndex` for full-text search with BM25 scoring
  - Automatic tokenization with stop-word filtering
- **Multi-Vector Documents** — Multiple embeddings per document
  - Store title, abstract, and content vectors separately
  - Aggregation strategies: max-pooling, mean-pooling, weighted sum
  - `MultiVectorMapping` for chunk-level semantic search
- **Context-Aware Queries** — LLM-optimized retrieval with token budgeting
  - `ContextQuery` builder with fluent API
  - Token estimation for GPT-4, Claude, Gemini models
  - Semantic deduplication to avoid redundant results
  - Automatic content extraction with configurable fields
- **Tombstone-Based Deletion** — Logical deletion for vector indices
  - `TombstoneManager` for efficient soft deletes
  - Filtering during search without rebuilding index
  - Lazy compaction for performance
- **Enhanced Error Taxonomy** — Structured error handling across all SDKs
  - `ErrorCode` enum with 40+ specific error types
  - Hierarchical error categories (Storage, Index, Query, IPC)
  - Remediation hints for common errors
  - Python SDK: Enhanced exception classes with error codes

### Changed
- **SDK Documentation** — Comprehensive updates across all languages
  - Removed version numbers from installation instructions (easier maintenance)
  - Added "What's New in Latest Release" section to all READMEs
  - Developer-friendly examples for namespace, hybrid search, and multi-vector
  - Consistent documentation structure across Python, Go, Node.js, and Rust SDKs
- **Python SDK** — Namespace API integration in Database class
  - `create_namespace()`, `namespace()`, `list_namespaces()` methods
  - Namespace examples in Quick Start section
- **Go SDK** — Updated with 0.3.0 feature examples
- **Node.js SDK** — TypeScript examples for all new features
- **Rust SDK** — Updated with NamespaceHandle and new vector APIs

### Fixed
- **Node.js VERSION Bug** — VERSION constant was stuck at 0.2.8 instead of matching package.json
  - Fixed `toondb-js/src/index.ts` VERSION export
  - Updated `database.test.ts` to test correct version

### Breaking Changes
- Namespace API requires explicit namespace creation before use
- Collection configuration is immutable after creation
- Python SDK: Import path remains `toondb` (package name: `toondb-client`)

---

## [0.2.9] - 2026-01-02

### Added
- **Comprehensive benchmark suite** with real-world LLM embeddings (Azure OpenAI)
  - ToonDB vs ChromaDB: **3× faster** vector search
  - ToonDB vs LanceDB: **22× faster** vector search
  - Recall@k benchmarks showing **>98% recall** with sub-millisecond latency
  - End-to-end RAG bottleneck analysis (API is 333× slower than database)
- **Full SQL engine support in Go SDK** with DDL/DML operations matching Python/JS SDKs
- **Community health files** for open source project
  - CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
  - SECURITY.md with vulnerability reporting policy and response timelines
  - Issue templates (bug report, feature request, support) with YAML validation
- **Unified release workflow** with automated SDK publishing to crates.io, PyPI, and npm
- **360° performance report** with retrieval quality, latency, throughput, and resource efficiency metrics

### Changed
- Simplified release workflow with improved error handling and protected branch support
- Updated documentation with consistent SDK guides across all languages
- Enhanced benchmark reports with real-world embedding comparisons

### Fixed
- Rust compilation errors in storage.rs
- Go SDK test output formatting (removed redundant newlines)
- Path dependencies for crates.io publishing
- Wire protocol documentation for all SDKs

---

## [0.2.7] - 2026-01-01

### Added
- **Full SQL engine support** in Python SDK with DDL/DML operations (CREATE/DROP TABLE, INSERT, SELECT, UPDATE, DELETE)
- **Full SQL engine support** in JavaScript SDK with complete SQL parser and executor
- **Go embedded server mode** - automatically starts/stops toondb-server without external setup
- **Transaction SQL support** - execute() method added to Transaction class in Python SDK
- SQL storage using KV backend with `_sql/tables/` prefix (tables and rows stored as JSON)
- WHERE clause support with operators: =, !=, <, >, >=, <=, LIKE, NOT LIKE
- ORDER BY with ASC/DESC, LIMIT, and OFFSET support
- Data types: INT, TEXT, FLOAT, BOOL, BLOB

### Changed
- JavaScript stats() response format changed from key=value to valid JSON
- Go SDK now defaults to embedded mode (Config.Embedded = true)
- Python SDK execute_sql() added as alias for execute() for documentation consistency

### Fixed
- **Critical**: Python SQL API now returns actual query results (was returning empty rows)
- **Critical**: IPC server stats command now returns valid JSON format
- **Critical**: Go SDK no longer requires external server process
- JavaScript SDK ESM imports now use explicit .js extensions

---

## [0.2.6] - 2026-01-01

### Added
- **Enhanced scan() method** across all SDKs for efficient prefix-based iteration and multi-tenant isolation
- **Full SQL support** in Python SDK (CREATE TABLE, INSERT, SELECT, JOIN, WHERE, GROUP BY, aggregations)
- **SQL integration** in Rust SDK via toondb-query crate with async IPC
- **Bulk vector API** in Python SDK (~1,600 vec/s, 12× faster than FFI loop)
- **Zero-copy reads** in Rust SDK for large value optimization
- **Async IPC client** in Rust SDK with Tokio runtime
- **SQL examples** in all SDK READMEs and examples/ directory
- **Comprehensive SDK guides** in docs/guides/ for Go, Python, JavaScript/Node.js, and Rust
- **SQL API documentation** in docs/api-reference/sql-api.md

### Changed
- Updated all SDK READMEs with v0.2.6 features and complete examples
- Improved documentation structure with consistent formatting across all guides
- Enhanced TypeScript definitions in JavaScript SDK
- Updated wire protocol documentation for Little Endian format
- Improved error messages across all SDKs

### Fixed
- **Path operations** (putPath/getPath) in JavaScript SDK now use correct wire format
- Binary encoding issues in Go SDK path operations
- Python SDK _bin/ directory now properly excluded from git (built during CI/CD)
- Scan range calculation simplified using semicolon trick (';' after '/' in ASCII)

---

## [0.2.3] - 2025-12-31

### Added
- Go SDK in-repo (`toondb-go/`) with examples
- Rust SDK guide and multi-language docs index updates

### Fixed
- TBP writer/reader null bitmap sizing and fixed-row offset correctness
- Crates.io publishability for workspace path dependencies

---

## [0.1.0] - 2024-12-27

### Added

#### Core Database
- **ACID Transactions** with MVCC and Serializable Snapshot Isolation (SSI)
- **Write-Ahead Log (WAL)** with group commit optimization
- **Path-based data model** with O(|path|) resolution via Trie-Columnar Hybrid (TCH)
- **Columnar storage** with automatic projection pushdown

#### LLM Features
- **TOON Format** — 40-66% token reduction compared to JSON
- **Context Query Builder** — token budgeting and priority-based truncation
- **Vector Search** — HNSW index with F32/F16/BF16 quantization

#### Python SDK
- **Embedded mode** via FFI for single-process applications
- **IPC mode** via Unix sockets for multi-process scenarios
- **Bulk API** for high-throughput vector operations (~1,600 vec/s vs ~130 vec/s FFI)
- Pre-built wheels for Linux (x86_64, aarch64), macOS (universal2), Windows (x64)

#### MCP Integration
- **toondb-mcp** server for Claude Desktop, Cursor, and Goose
- Tools: `toondb_put`, `toondb_get`, `toondb_scan`, `toondb_delete`, `toondb_context_query`

#### Indexing
- **HNSW** vector index with configurable M, ef_construction, ef_search
- **B-Tree** index for ordered key access
- **Bloom filters** for existence checks

### Performance
- Ordered index can be disabled for ~20% faster writes (point lookups only)
- Group commit reduces fsync overhead for high-throughput writes
- Columnar storage minimizes I/O for selective queries

### Known Limitations
- Single-node only (no distributed mode, replication, or clustering)
- Python SDK requires `TOONDB_LIB_PATH` environment variable for FFI mode

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.2.3 | 2025-12-31 | Multi-SDK + publish readiness |
| 0.1.0 | 2024-12-27 | Initial release |

---

## Upgrade Guide

### Upgrading to 0.1.0

This is the initial release. No upgrade path required.

---

## Contributors

Thanks to all contributors who helped make ToonDB possible!

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## Links

- [Documentation](docs/index.md)
- [Quick Start](docs/QUICKSTART.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
