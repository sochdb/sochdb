# Contributing to ToonDB

Thank you for your interest in contributing to ToonDB! This document provides guidelines and information for contributors.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Quick Setup](#quick-setup) ⭐ **Start Here**
- [Development Environment](#development-environment)
- [Directory Structure](#directory-structure)
- [Architecture Overview](#architecture-overview)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Getting Help](#getting-help)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

**TL;DR:** Be respectful, be constructive, assume good intent.

---

## Quick Setup

### One-Command Setup (Recommended)

```bash
# Clone, build, and test in one go
git clone https://github.com/toondb/toondb.git && \
cd toondb && \
make setup
```

The `make setup` command:
1. Installs required Rust toolchain
2. Installs development tools (clippy, rustfmt, cargo-watch)
3. Builds all crates
4. Runs the test suite
5. Sets up git hooks

### Manual Setup

If you prefer manual setup or `make setup` fails:

#### Prerequisites

| Requirement | Version | Check | Install |
|-------------|---------|-------|---------|
| **Rust** | ≥1.75.0 (2024 edition) | `rustc --version` | [rustup.rs](https://rustup.rs/) |
| **Git** | Any recent | `git --version` | OS package manager |
| **Clang** | ≥14 (optional, for SIMD) | `clang --version` | OS package manager |
| **Python** | ≥3.9 (for SDK) | `python --version` | OS package manager |

#### Steps

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/toondb.git
cd toondb
git remote add upstream https://github.com/toondb/toondb.git

# 2. Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 3. Install development tools
cargo install cargo-watch cargo-criterion cargo-deny cargo-udeps

# 4. Build and test
cargo build --all
cargo test --all

# 5. Verify clippy passes
cargo clippy --all -- -D warnings
```

---

## Development Environment

### Recommended Tools

```bash
# Install useful development tools
cargo install cargo-watch     # Auto-rebuild on changes
cargo install cargo-criterion # Benchmarking
cargo install cargo-deny      # Dependency auditing
cargo install cargo-udeps     # Find unused dependencies
cargo install cargo-expand    # Macro expansion
```

### IDE Setup

**VS Code** (recommended):

Create `.vscode/settings.json`:
```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.check.command": "clippy",
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    },
    "rust-analyzer.procMacro.enable": true,
    "rust-analyzer.cargo.buildScripts.enable": true
}
```

Create `.vscode/launch.json` for debugging:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug unit tests",
            "cargo": {
                "args": ["test", "--no-run", "--lib"],
                "filter": {
                    "kind": "lib"
                }
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        },
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug toondb-server",
            "cargo": {
                "args": ["build", "--bin", "toondb-server"]
            },
            "args": ["--config", "toondb-server-config.toml"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

Recommended extensions:
- `rust-lang.rust-analyzer` — Rust language support
- `vadimcn.vscode-lldb` — Debugger
- `tamasfe.even-better-toml` — TOML support
- `serayuzgur.crates` — Cargo.toml dependency management

**IntelliJ IDEA/CLion**:
- Install Rust plugin
- Enable "Run clippy on save"
- Set cargo features to "all"
- Import as Cargo project

### Environment Variables

```bash
# Enable debug logging
export RUST_LOG=toondb=debug

# Enable backtraces
export RUST_BACKTRACE=1

# Run with sanitizers (requires nightly)
RUSTFLAGS="-Z sanitizer=address" cargo +nightly test
```

---

## Directory Structure

Understanding the codebase layout:

```
toondb/
├── Cargo.toml              # Workspace manifest
├── README.md               # Project overview
├── CONTRIBUTING.md         # This file
├── CHANGELOG.md            # Version history
├── LICENSE                 # Apache 2.0
│
├── docs/                   # 📚 Documentation
│   ├── index.md            # Documentation home
│   ├── QUICKSTART.md       # Getting started
│   ├── API.md              # API reference
│   ├── ARCHITECTURE.md     # Deep technical docs
│   ├── tutorials/          # Learning-oriented guides
│   └── cookbook/           # Problem-oriented recipes
│
├── toondb-core/            # 🧱 Core types & utilities
│   └── src/
│       ├── toon.rs         # ToonValue enum
│       ├── toon_codec.rs   # TOON format encoder/decoder
│       ├── schema_*.rs     # Schema definitions
│       └── path_trie.rs    # Trie-based path resolution
│
├── toondb-kernel/          # ⚙️ Database engine
│   └── src/
│       ├── wal.rs          # Write-ahead log
│       ├── transaction.rs  # MVCC transactions
│       ├── plugin.rs       # Plugin system
│       └── wasm_*.rs       # WASM runtime
│
├── toondb-index/           # 🔍 Index implementations
│   └── src/
│       ├── hnsw.rs         # HNSW vector index
│       ├── btree.rs        # B-tree index
│       └── bloom.rs        # Bloom filters
│
├── toondb-query/           # 🔎 Query engine
│   └── src/
│       ├── parser.rs       # Query parsing
│       ├── planner.rs      # Query planning
│       └── executor.rs     # Query execution
│
├── toondb-client/          # 📦 Client SDK
│   └── src/
│       ├── connection.rs   # Database connection
│       └── batch.rs        # Batch operations
│
├── toondb-python/          # 🐍 Python bindings (PyO3)
│   ├── src/lib.rs          # Rust FFI code
│   └── python/             # Python package
│
├── toondb-mcp/             # 🤖 MCP server for LLMs
│   └── src/
│       └── main.rs         # MCP protocol implementation
│
├── toondb-grpc/            # 🌐 gRPC server
│   ├── proto/              # Protocol buffers
│   └── src/
│
├── benchmarks/             # 📊 Performance benchmarks
│   ├── src/                # Criterion benchmarks
│   └── python/             # Python comparison benchmarks
│
└── examples/               # 📖 Example code
    ├── rust/               # Rust examples
    └── python/             # Python examples
```

### Key Files for New Contributors

| File | Why It Matters |
|------|---------------|
| [toondb-core/src/toon.rs](toondb-core/src/toon.rs) | Core value type, start here |
| [toondb-kernel/src/wal.rs](toondb-kernel/src/wal.rs) | Durability implementation |
| [toondb-index/src/hnsw.rs](toondb-index/src/hnsw.rs) | Vector search algorithm |
| [toondb-kernel/src/transaction.rs](toondb-kernel/src/transaction.rs) | MVCC implementation |

---

## Architecture Overview

### Crate Dependency Graph

```
                    ┌─────────────────┐
                    │  toondb-client  │
                    │   (SDK/API)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │ toondb-  │   │ toondb-  │   │ toondb-  │
       │  query   │   │  index   │   │  kernel  │
       └────┬─────┘   └────┬─────┘   └────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
             ┌──────────┐  ┌──────────┐
             │ toondb-  │  │ toondb-  │
             │ storage  │  │  core    │
             └──────────┘  └──────────┘
```

### Data Flow

```
                     Request Flow
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    toondb-client                         │
│  • Parse query                                          │
│  • Validate schema                                       │
│  • Format TOON output                                    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    toondb-kernel                         │
│  • Begin transaction (MVCC snapshot)                    │
│  • Acquire locks if needed                              │
│  • Execute operation                                     │
│  • Write to WAL (on commit)                             │
└─────────────────────────┬───────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│    toondb-index       │   │    toondb-storage     │
│  • HNSW search        │   │  • Read/write blocks  │
│  • B-tree lookup      │   │  • Compression        │
│  • Bloom filter       │   │  • Memtable/SST       │
└───────────────────────┘   └───────────────────────┘
```

### Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **ToonValue** | `toondb-core/src/toon.rs` | Core value type |
| **TOON Codec** | `toondb-core/src/toon_codec.rs` | Serialization/parsing |
| **WAL** | `toondb-kernel/src/wal.rs` | Write-ahead logging |
| **MVCC** | `toondb-kernel/src/transaction.rs` | Multi-version concurrency |
| **HNSW** | `toondb-index/src/hnsw.rs` | Vector index |
| **LSCS** | `toondb-storage/src/lscs.rs` | Columnar storage |
| **TCH** | `toondb-client/src/connection.rs` | Trie-Columnar Hybrid |

### Design Principles

1. **Zero-copy where possible**: Use `&[u8]` slices, `memmap2`
2. **Lock-free reads**: MVCC snapshots, atomic operations
3. **Minimal allocations**: Arena allocators, object pools
4. **Fail-safe**: CRC32 checksums, WAL recovery

---

## Making Changes

### Branch Naming

```
feature/description    # New features
fix/issue-number       # Bug fixes
docs/description       # Documentation
perf/description       # Performance improvements
refactor/description   # Code refactoring
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(hnsw): add parallel index construction

Implement parallel insertion using rayon. This improves
build performance by ~4x on 8-core machines.

Closes #123
```

```
fix(wal): handle truncated records during recovery

Previously, truncated WAL records at the end of the file
would cause a panic. Now they are silently skipped with
a warning log.

Fixes #456
```

---

## Testing

### Test Organization

```
crate/
├── src/
│   └── module.rs           # Unit tests at bottom of file
├── tests/
│   └── integration_test.rs # Integration tests
└── benches/
    └── benchmark.rs        # Performance benchmarks
```

### Running Tests

```bash
# All tests
cargo test --all

# Specific crate
cargo test -p toondb-kernel

# Specific test
cargo test -p toondb-index test_hnsw_insert

# With coverage (requires cargo-tarpaulin)
cargo tarpaulin --all --out Html
```

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // Use descriptive names
    #[test]
    fn insert_and_retrieve_returns_same_value() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path()).unwrap();
        
        db.put(b"key", b"value").unwrap();
        let result = db.get(b"key").unwrap();
        
        assert_eq!(result, Some(b"value".to_vec()));
    }

    // Test edge cases
    #[test]
    fn get_nonexistent_key_returns_none() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path()).unwrap();
        
        let result = db.get(b"missing").unwrap();
        
        assert_eq!(result, None);
    }

    // Test error conditions
    #[test]
    fn open_invalid_path_returns_error() {
        let result = Database::open("/nonexistent/path/db");
        
        assert!(result.is_err());
    }
}
```

### Property-Based Testing

Use `proptest` for property-based tests:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_decode_roundtrip(value: ToonValue) {
        let encoded = value.encode();
        let decoded = ToonValue::decode(&encoded).unwrap();
        prop_assert_eq!(value, decoded);
    }
}
```

### Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_hnsw_search(c: &mut Criterion) {
    let index = setup_index_with_vectors(100_000);
    let query = random_vector(384);
    
    c.bench_function("hnsw_search_100k", |b| {
        b.iter(|| {
            index.search(black_box(&query), black_box(10))
        })
    });
}

criterion_group!(benches, benchmark_hnsw_search);
criterion_main!(benches);
```

Run benchmarks:
```bash
cargo bench -p toondb-index
```

---

## Pull Request Process

### Before Submitting

Run the full check suite:

```bash
# One-command validation
make check

# Or manually:
cargo fmt --all --check           # Formatting
cargo clippy --all -- -D warnings # Lints  
cargo test --all                  # Tests
cargo doc --no-deps               # Documentation builds
```

### Creating Your PR

1. **Create a branch** from `main`:
   ```bash
   git checkout main && git pull upstream main
   git checkout -b feature/my-feature
   ```

2. **Make your changes** with tests

3. **Push and create PR**:
   ```bash
   git push origin feature/my-feature
   # Then open PR on GitHub
   ```

### PR Template

```markdown
## Summary
Brief description of changes.

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation
- [ ] Performance improvement

## Testing
- [ ] Added new tests
- [ ] All tests pass locally
- [ ] Tested manually (describe how)

## Checklist
- [ ] Code compiles without warnings
- [ ] Follows code style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if notable)

## Related Issues
Closes #123
```

### Review Timeline

| PR Type | Initial Review | Merge Time |
|---------|----------------|------------|
| Bug fix | 1-2 days | 2-3 days |
| Feature | 3-5 days | 1-2 weeks |
| Breaking change | 1 week | 2-3 weeks |
| Documentation | 1-2 days | 2-3 days |

### Review Process

1. **Maintainers review** within 3 business days
2. **Address feedback** by pushing new commits (don't force-push)
3. **CI must pass** — all checks green
4. **Two approvals** required for core crates
5. **Maintainer merges** via squash merge
6. **Delete your branch** after merge

### After Merge

- Your commit appears in `main`
- Will be included in next release
- You'll be credited in CHANGELOG.md

---

## Code Style

### Rust Style

Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/):

```rust
// Use descriptive names
fn calculate_optimal_batch_size(arrival_rate: f64) -> usize { ... }

// Document public APIs
/// Inserts a vector into the index.
///
/// # Arguments
///
/// * `id` - Unique identifier for the vector
/// * `vector` - The embedding vector
///
/// # Returns
///
/// The node index in the graph, or an error if insertion fails.
///
/// # Example
///
/// ```
/// let index = HNSWIndex::new();
/// index.insert(1, &[0.1, 0.2, 0.3])?;
/// ```
pub fn insert(&self, id: u64, vector: &[f32]) -> Result<usize, Error> { ... }

// Use type aliases for clarity
pub type EdgeId = u64;
pub type VectorId = u32;

// Prefer enums over boolean flags
pub enum SyncMode {
    Full,
    Normal,
    Off,
}

// Group related methods with comments
impl Database {
    // --- Transaction Management ---
    
    pub fn begin(&self) -> Result<TxnHandle> { ... }
    pub fn commit(&self, txn: TxnHandle) -> Result<()> { ... }
    pub fn rollback(&self, txn: TxnHandle) -> Result<()> { ... }
    
    // --- Data Operations ---
    
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> { ... }
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> { ... }
}
```

### Error Handling

```rust
// Use thiserror for library errors
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("transaction {txn_id} has been aborted")]
    TransactionAborted { txn_id: u64 },
    
    #[error("table '{name}' not found")]
    TableNotFound { name: String },
}

// Provide context with anyhow in applications
use anyhow::{Context, Result};

fn load_config(path: &Path) -> Result<Config> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config from {:?}", path))?;
    
    toml::from_str(&contents)
        .context("failed to parse config")
}
```

### Performance Considerations

```rust
// Avoid allocations in hot paths
fn process_records(&self, records: &[Record]) {
    // Pre-allocate
    let mut buffer = Vec::with_capacity(records.len() * 64);
    
    for record in records {
        // Reuse buffer
        buffer.clear();
        self.serialize_into(&mut buffer, record);
        self.write_buffer(&buffer);
    }
}

// Use iterators instead of collecting
fn find_matches<'a>(&'a self, predicate: impl Fn(&Item) -> bool + 'a) 
    -> impl Iterator<Item = &'a Item> 
{
    self.items.iter().filter(predicate)
}

// Mark inline for small hot functions
#[inline]
fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}
```

---

## Documentation

### Doc Comments

```rust
//! Module-level documentation goes here.
//!
//! # Overview
//!
//! This module provides...
//!
//! # Examples
//!
//! ```
//! use toondb::SomeType;
//! let x = SomeType::new();
//! ```

/// Function/struct documentation.
///
/// More detailed explanation here.
///
/// # Arguments
///
/// * `param1` - Description of param1
///
/// # Returns
///
/// Description of return value.
///
/// # Errors
///
/// Returns [`Error::Kind`] when...
///
/// # Panics
///
/// Panics if...
///
/// # Safety
///
/// (for unsafe functions) Caller must ensure...
///
/// # Examples
///
/// ```
/// // Example code
/// ```
pub fn example_function(param1: u32) -> Result<String, Error> { ... }
```

### Architecture Decision Records (ADRs)

For significant architectural decisions, create ADRs in `docs/adr/`:

```markdown
# ADR-001: Use HNSW for Vector Index

## Status
Accepted

## Context
We need a vector similarity search algorithm that provides sub-linear
query time for approximate nearest neighbor search.

## Decision
We will use HNSW (Hierarchical Navigable Small World) because:
- O(log N) average query time
- Good recall (>95% at practical settings)
- Memory-efficient graph structure
- Well-understood algorithm with production usage

## Alternatives Considered
- **IVF-PQ**: Lower memory, but more complex and lower recall
- **Annoy**: Simpler, but slower builds and lower recall
- **FAISS**: External dependency, harder to embed

## Consequences
- Good: Sub-linear search time, high recall
- Good: Single-threaded insert works well for our use case
- Bad: Higher memory usage than quantization-based methods
- Bad: Graph updates can be expensive
```

---

## Testing Guidelines

See [docs/testing.md](docs/testing.md) for comprehensive testing guidelines.

### Quick Reference

```bash
# Run all tests
cargo test --all

# Run specific crate tests
cargo test -p toondb-kernel

# Run with output
cargo test -- --nocapture

# Run ignored (slow) tests
cargo test -- --ignored

# Run benchmarks
cargo bench -p toondb-index

# Coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html --output-dir coverage/
```

### Coverage Requirements

| Crate | Minimum Coverage |
|-------|------------------|
| toondb-core | 80% |
| toondb-kernel | 75% |
| toondb-index | 70% |
| Others | 60% |

PRs that drop coverage below these thresholds may be rejected.

---

## Getting Help

### Resources

| Resource | Use For |
|----------|---------|
| [Documentation](docs/index.md) | User guides, tutorials |
| [Architecture](docs/ARCHITECTURE.md) | Technical deep-dives |
| [API Reference](docs/API.md) | API documentation |

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Discord**: Real-time chat (link in README)

### First-Time Contributors

Look for issues labeled:
- `good first issue` — Beginner-friendly
- `help wanted` — We need your help
- `documentation` — Docs improvements

---

## Governance

### Maintainers

PRs require approval from at least one maintainer. Core crates (`toondb-kernel`, `toondb-core`) require two approvals.

### Release Process

1. Update `CHANGELOG.md` with release notes
2. Bump version in `Cargo.toml` files
3. Create release PR
4. After merge, tag release: `git tag v0.1.x`
5. CI publishes to crates.io

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

---

Thank you for contributing to ToonDB! 🎉

*Your contribution helps make LLM-native data storage better for everyone.*
