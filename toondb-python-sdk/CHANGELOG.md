# Changelog

All notable changes to the ToonDB Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2025-01-xx

### Fixed
- **Platform detection bug**: Fixed binary resolution using Rust target triple format (`aarch64-apple-darwin`) instead of Python platform tag format (`darwin-aarch64`)
- Improved documentation accuracy across all doc files

### Changed
- Updated to match latest Rust SDK API patterns

## [Unreleased]

### Added

#### Cross-Platform Binary Distribution
- **Zero-compile installation**: Pre-built `toondb-bulk` binaries bundled in wheels
- **Platform support matrix**:
  - `manylinux_2_17_x86_64` - Linux x86_64 (glibc ≥ 2.17)
  - `manylinux_2_17_aarch64` - Linux ARM64 (AWS Graviton, etc.)
  - `macosx_11_0_universal2` - macOS Intel + Apple Silicon
  - `win_amd64` - Windows x64
- **Automatic binary resolution** with fallback chain:
  1. Bundled in wheel (`_bin/<platform>/toondb-bulk`)
  2. System PATH (`which toondb-bulk`)
  3. Cargo target directory (development mode)

#### Bulk API Enhancements
- `bulk_query_index()` - Query HNSW indexes for k nearest neighbors
- `bulk_info()` - Get index metadata (vector count, dimension, etc.)
- `get_toondb_bulk_path()` - Get resolved path to toondb-bulk binary
- `_get_platform_tag()` - Platform detection (linux-x86_64, darwin-aarch64, etc.)
- `_find_bundled_binary()` - Uses `importlib.resources` for installed packages

#### CI/CD Infrastructure
- GitHub Actions workflow for building platform-specific wheels
- cibuildwheel configuration for cross-platform builds
- QEMU emulation for ARM64 Linux builds
- PyPI publishing with trusted publishing

#### Documentation
- [PYTHON_DISTRIBUTION.md](../docs/PYTHON_DISTRIBUTION.md) - Full distribution architecture
- Updated [BULK_OPERATIONS.md](../docs/BULK_OPERATIONS.md) with troubleshooting
- Updated [SDK_DOCUMENTATION.md](docs/SDK_DOCUMENTATION.md) with Bulk API reference
- Updated [ARCHITECTURE.md](../docs/ARCHITECTURE.md) with Python SDK section

### Changed

- Package renamed from `toondb-client` to `toondb`
- Wheel tags changed from `any` to platform-specific (`py3-none-<platform>`)
- Binary resolution now uses `importlib.resources` instead of `__file__` paths

### Technical Details

#### Distribution Model
Follows the "uv-style" approach where:
- Wheels are tagged `py3-none-<platform>` (not CPython-ABI-tied)
- One wheel per platform (not per Python version)
- Artifact count: O(P·A) where P=platforms, A=architectures

#### Linux Compatibility
- **manylinux_2_17** baseline (glibc ≥ 2.17)
- Covers: CentOS 7+, RHEL 7+, Ubuntu 14.04+, Debian 8+
- Same baseline used by `uv` for production deployments

#### macOS Strategy
- **universal2** fat binaries containing both x86_64 and arm64
- Created with `lipo -create` during build
- Minimum macOS 11.0 (Big Sur)

## [0.1.0] - 2024-12-XX

### Added

- Initial release
- Embedded mode with FFI access to ToonDB
- IPC client mode for multi-process access
- Path-native API with O(|path|) lookups
- ACID transactions with snapshot isolation
- Range scans and prefix queries
- TOON format output for LLM context optimization
- Bulk API for high-throughput vector ingestion
  - `bulk_build_index()` - Build HNSW indexes at ~1,600 vec/s
  - `convert_embeddings_to_raw()` - Convert numpy to raw f32
- Support for raw f32 and NumPy .npy input formats

### Performance

| Method | 768D Throughput | Notes |
|--------|-----------------|-------|
| Python FFI | ~130 vec/s | Direct FFI calls |
| Bulk API | ~1,600 vec/s | Subprocess to toondb-bulk |

FFI overhead eliminated by subprocess approach for bulk operations.
