# sochdb-query

Query planning and execution engine for SochDB.

## Overview

`sochdb-query` provides the query layer for SochDB, featuring:

- **Query Builder**: Fluent API for constructing queries
- **Query Optimizer**: Cost-based optimization for efficient execution
- **Projection Pushdown**: Only read columns you need
- **Filter Pushdown**: Apply filters at storage layer
- **TOON Output**: Token-optimized format (40-66% fewer tokens than JSON)

## Features

- **SQL-like Queries**: Familiar query patterns without SQL parsing overhead
- **Path-based Access**: O(|path|) resolution independent of data size
- **Aggregations**: COUNT, SUM, AVG, MIN, MAX with efficient execution
- **Sorting & Pagination**: ORDER BY, LIMIT, OFFSET support
- **Join Support**: Nested loop and hash joins

## Installation

```toml
[dependencies]
sochdb-query = "0.2.5"
```

## Usage

Most users should use the high-level [`sochdb`](https://crates.io/crates/sochdb) crate:

```rust
use sochdb::{Database, Query};

let db = Database::open("./my_data")?;

// Query with builder pattern
let results = db.query("users")
    .filter("age", ">", 21)
    .select(&["name", "email"])
    .order_by("name")
    .limit(10)
    .execute()?;

// Results in token-efficient TOON format
println!("{}", results.to_toon()); 
// users[3]{name,email}: Alice,alice@...|Bob,bob@...|Carol,carol@...
```

## Query Execution Pipeline

```
┌──────────────┐
│    Query     │  User query (builder or path)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Parse     │  Validate and normalize
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Optimize   │  Pushdown filters, projections
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Execute    │  Scan, filter, project, sort
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Format     │  TOON or JSON output
└──────────────┘
```

## Crate Structure

| Crate | Purpose |
|-------|---------|
| [`sochdb`](https://crates.io/crates/sochdb) | High-level client API (start here) |
| [`sochdb-core`](https://crates.io/crates/sochdb-core) | Core types and traits |
| [`sochdb-storage`](https://crates.io/crates/sochdb-storage) | Storage engine with WAL |
| [`sochdb-index`](https://crates.io/crates/sochdb-index) | HNSW vector indexing |
| `sochdb-query` | Query execution (this crate) |

## License

AGPL-3.0-or-later - see [LICENSE](../LICENSE) for details.
