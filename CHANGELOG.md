# Changelog

All notable changes to ToonDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Diátaxis-based documentation structure
- Comprehensive contributor guide with one-command setup
- Cookbook with deployment, logging, vector search guides
- First-app tutorial for Python SDK

### Changed
- Improved CONTRIBUTING.md with directory structure and PR process

### Fixed
- (No fixes yet)

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
