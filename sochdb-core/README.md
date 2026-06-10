# sochdb-core

Core types and traits for SochDB - the LLM-native database.

## Overview

`sochdb-core` provides the foundational types, error handling, and trait definitions used throughout the SochDB ecosystem. This crate is typically not used directly - instead, use the [`sochdb`](https://crates.io/crates/sochdb) client crate.

## Features

- **Core Types**: Value types, keys, and serialization primitives
- **Error Handling**: Unified error types across all SochDB crates
- **Trait Definitions**: Common traits for storage backends and query execution
- **TOON Format**: Token-optimized notation for LLM efficiency (40-66% fewer tokens than JSON)

## Installation

```toml
[dependencies]
sochdb-core = "0.2.5"
```

## Usage

Most users should use the high-level [`sochdb`](https://crates.io/crates/sochdb) crate instead:

```rust
use sochdb::Database;

let db = Database::open("./my_data")?;
```

## Crate Structure

SochDB is organized into several crates:

| Crate | Purpose |
|-------|---------|
| [`sochdb`](https://crates.io/crates/sochdb) | High-level client API (start here) |
| `sochdb-core` | Core types and traits (this crate) |
| [`sochdb-storage`](https://crates.io/crates/sochdb-storage) | Storage engine with WAL |
| [`sochdb-index`](https://crates.io/crates/sochdb-index) | HNSW vector indexing |
| [`sochdb-query`](https://crates.io/crates/sochdb-query) | Query planning and execution |

## License

AGPL-3.0-or-later - see [LICENSE](../LICENSE) for details.
