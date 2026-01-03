# Changelog

All notable changes to ToonDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
