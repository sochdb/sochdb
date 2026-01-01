# Testing Guide

Comprehensive testing guidelines for ToonDB contributors.

---

## Table of Contents

- [Running Tests](#running-tests)
- [Test Organization](#test-organization)
- [Writing Tests](#writing-tests)
- [Coverage Requirements](#coverage-requirements)
- [Benchmarking](#benchmarking)
- [CI Pipeline](#ci-pipeline)

---

## Running Tests

### Quick Commands

```bash
# Run all tests
cargo test --all

# Run tests for a specific crate
cargo test -p toondb-kernel
cargo test -p toondb-index
cargo test -p toondb-core

# Run a specific test
cargo test -p toondb-kernel test_transaction_commit

# Run with output (see println! statements)
cargo test --all -- --nocapture

# Run ignored (slow) tests
cargo test --all -- --ignored

# Run only doc tests
cargo test --doc

# Run tests in release mode (faster, but less debug info)
cargo test --release --all
```

### Watch Mode

For development, use cargo-watch to auto-run tests on file changes:

```bash
# Install cargo-watch
cargo install cargo-watch

# Watch and run all tests
cargo watch -x 'test --all'

# Watch specific crate
cargo watch -x 'test -p toondb-kernel'

# Watch and run specific test
cargo watch -x 'test -p toondb-index test_hnsw'
```

---

## Test Organization

### Directory Structure

```
toondb-*/
├── src/
│   ├── lib.rs
│   └── module.rs          # Unit tests at bottom of file
├── tests/
│   └── integration_test.rs # Integration tests
└── benches/
    └── benchmark.rs        # Performance benchmarks
```

### Test Locations

| Test Type | Location | When to Use |
|-----------|----------|-------------|
| **Unit tests** | Bottom of `src/*.rs` | Test individual functions/structs |
| **Integration tests** | `tests/*.rs` | Test public API, cross-module |
| **Doc tests** | Doc comments | Example code in documentation |
| **Benchmarks** | `benches/*.rs` | Performance measurements |

---

## Writing Tests

### Unit Tests

Place at the bottom of the source file:

```rust
// src/toon.rs

pub fn encode_varint(value: u64) -> Vec<u8> {
    // ... implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_varint_zero() {
        assert_eq!(encode_varint(0), vec![0x00]);
    }

    #[test]
    fn encode_varint_small_value() {
        assert_eq!(encode_varint(127), vec![0x7F]);
    }

    #[test]
    fn encode_varint_requires_continuation() {
        assert_eq!(encode_varint(128), vec![0x80, 0x01]);
    }

    #[test]
    fn encode_varint_max_u64() {
        let result = encode_varint(u64::MAX);
        assert_eq!(result.len(), 10); // Max varint size for u64
    }
}
```

### Naming Conventions

Use descriptive names that explain what's being tested:

```rust
// ✅ Good: Describes behavior
#[test]
fn insert_and_retrieve_returns_same_value() { ... }

#[test]
fn get_nonexistent_key_returns_none() { ... }

#[test]
fn transaction_rollback_discards_writes() { ... }

// ❌ Bad: Vague names
#[test]
fn test_insert() { ... }

#[test]
fn test1() { ... }
```

### Using Temporary Directories

Always use `tempfile` for tests that need filesystem access:

```rust
use tempfile::TempDir;

#[test]
fn database_persists_across_reopens() {
    let dir = TempDir::new().unwrap();
    
    // First open: write data
    {
        let db = Database::open(dir.path()).unwrap();
        db.put(b"key", b"value").unwrap();
    }
    
    // Second open: read data back
    {
        let db = Database::open(dir.path()).unwrap();
        let result = db.get(b"key").unwrap();
        assert_eq!(result, Some(b"value".to_vec()));
    }
    
    // TempDir automatically cleaned up when dropped
}
```

### Testing Error Conditions

```rust
#[test]
fn open_invalid_path_returns_io_error() {
    let result = Database::open("/nonexistent/path/that/cannot/exist");
    
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, KernelError::Io(_)));
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn access_beyond_array_panics() {
    let arr = [1, 2, 3];
    let _ = arr[10]; // Should panic
}
```

### Property-Based Testing

Use `proptest` for testing invariants across many inputs:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_decode_roundtrip(value: i64) {
        let encoded = ToonValue::Int(value).encode();
        let decoded = ToonValue::decode(&encoded).unwrap();
        
        prop_assert_eq!(ToonValue::Int(value), decoded);
    }

    #[test]
    fn varint_roundtrip(value: u64) {
        let encoded = encode_varint(value);
        let (decoded, _) = decode_varint(&encoded);
        
        prop_assert_eq!(value, decoded);
    }

    #[test]
    fn insert_get_consistent(
        key in "[a-z]{1,100}",
        value in prop::collection::vec(any::<u8>(), 0..1000)
    ) {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path()).unwrap();
        
        db.put(key.as_bytes(), &value).unwrap();
        let result = db.get(key.as_bytes()).unwrap();
        
        prop_assert_eq!(Some(value), result);
    }
}
```

### Integration Tests

Place in `tests/` directory:

```rust
// tests/transaction_integration.rs

use toondb_kernel::Database;
use tempfile::TempDir;

#[test]
fn concurrent_transactions_serialize_correctly() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();

    // Start two transactions
    let txn1 = db.begin().unwrap();
    let txn2 = db.begin().unwrap();

    // Both read same key
    let v1 = db.get_in_txn(&txn1, b"counter").unwrap();
    let v2 = db.get_in_txn(&txn2, b"counter").unwrap();

    // Both try to increment
    db.put_in_txn(&txn1, b"counter", b"1").unwrap();
    db.put_in_txn(&txn2, b"counter", b"1").unwrap();

    // First commit succeeds
    assert!(db.commit(txn1).is_ok());

    // Second commit fails (SSI conflict)
    assert!(db.commit(txn2).is_err());
}
```

### Doc Tests

Include examples in documentation that are automatically tested:

```rust
/// Encodes a value to TOON format.
///
/// # Examples
///
/// ```
/// use toondb_core::ToonValue;
///
/// let value = ToonValue::Int(42);
/// let encoded = value.to_toon();
/// assert_eq!(encoded, "42");
/// ```
///
/// Arrays are encoded with brackets:
///
/// ```
/// use toondb_core::ToonValue;
///
/// let arr = ToonValue::Array(vec![
///     ToonValue::Int(1),
///     ToonValue::Int(2),
/// ]);
/// assert_eq!(arr.to_toon(), "[1,2]");
/// ```
pub fn to_toon(&self) -> String {
    // ...
}
```

---

## Coverage Requirements

### Minimum Coverage by Crate

| Crate | Minimum Coverage | Notes |
|-------|------------------|-------|
| `toondb-core` | **80%** | Core types, must be well-tested |
| `toondb-kernel` | **75%** | Engine internals |
| `toondb-index` | **70%** | Index implementations |
| `toondb-query` | **70%** | Query processing |
| `toondb-client` | **65%** | SDK surface |
| Others | **60%** | Utilities, plugins |

### Checking Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage for all crates
cargo tarpaulin --all --out Html --output-dir coverage/

# Run coverage for specific crate
cargo tarpaulin -p toondb-kernel --out Html

# Run with ignored tests
cargo tarpaulin --all --run-types Tests,Doctests --out Html

# Quick coverage summary (no HTML)
cargo tarpaulin --all --out Stdout
```

### Coverage in CI

PRs that drop coverage below the thresholds will receive a warning comment. Significant drops may block merge.

---

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -p toondb-index hnsw_search

# Run with baseline comparison
cargo bench -- --baseline main

# Save baseline
cargo bench -- --save-baseline main
```

### Writing Benchmarks

```rust
// benches/hnsw_benchmark.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use toondb_index::HNSWIndex;

fn benchmark_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    
    for size in [1_000, 10_000, 100_000].iter() {
        let index = create_index_with_vectors(*size);
        let query = random_vector(384);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    index.search(black_box(&query), black_box(10))
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_hnsw_insert(c: &mut Criterion) {
    c.bench_function("hnsw_insert_single", |b| {
        let mut index = HNSWIndex::new(HNSWConfig::default());
        let mut id = 0u64;
        
        b.iter(|| {
            let vector = random_vector(384);
            index.insert(black_box(id), black_box(&vector)).unwrap();
            id += 1;
        })
    });
}

criterion_group!(benches, benchmark_hnsw_search, benchmark_hnsw_insert);
criterion_main!(benches);
```

### Performance Regression Detection

CI runs benchmarks on main branch. Significant regressions (>10%) will be flagged.

---

## CI Pipeline

### What CI Checks

```yaml
# .github/workflows/ci.yml (simplified)

jobs:
  test:
    - cargo fmt --all --check         # Formatting
    - cargo clippy --all -- -D warnings   # Lints
    - cargo test --all                # Unit + integration tests
    - cargo test --doc                # Doc tests
    - cargo tarpaulin --all           # Coverage
  
  bench:
    - cargo bench --no-run            # Benchmarks compile
    
  docs:
    - cargo doc --no-deps             # Documentation builds
```

### Fixing CI Failures

#### Formatting

```bash
cargo fmt --all
```

#### Clippy Warnings

```bash
# See all warnings
cargo clippy --all

# Auto-fix where possible
cargo clippy --all --fix
```

#### Failing Tests

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo test --all

# Run specific failing test
cargo test test_name -- --nocapture
```

---

## Best Practices

### Do

- ✅ Write tests before or alongside code
- ✅ Test edge cases (empty, zero, max values)
- ✅ Test error conditions
- ✅ Use property-based testing for invariants
- ✅ Keep tests fast (< 1 second each)
- ✅ Use descriptive test names

### Don't

- ❌ Test private implementation details
- ❌ Write flaky tests (random failures)
- ❌ Skip tests in CI
- ❌ Commit with failing tests
- ❌ Mock when you can use the real thing

---

## Troubleshooting

### Tests Pass Locally but Fail in CI

1. Check for platform-specific code
2. Verify no hardcoded paths
3. Check for timing-dependent tests
4. Ensure tests don't depend on execution order

### Flaky Tests

If a test sometimes fails:

1. Check for race conditions
2. Add retries for network operations
3. Use deterministic seeds for randomness
4. Increase timeouts for slow operations

### Slow Tests

```bash
# Find slow tests
cargo test --all -- -Z unstable-options --report-time

# Mark slow tests as ignored
#[test]
#[ignore = "slow: runs full compaction"]
fn test_full_compaction() { ... }
```

---

## See Also

- [Style Guide](/contributing/style-guide) — Documentation standards
- [Architecture](/concepts/architecture) — System design
- [Quick Start](/getting-started/quickstart) — Getting started

