# toondb

The official Rust SDK for [ToonDB](https://toondb.dev) - an LLM-optimized database with native vector search.

## Features

- 🚀 **Zero-copy reads** - Direct access to memory-mapped data
- 🔍 **Native vector search** - Built-in HNSW index for embeddings
- 📊 **Columnar storage** - Efficient for analytical queries
- 🔒 **Thread-safe** - Safe concurrent access with MVCC

## Quick Start

```rust
use toondb::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open or create a database
    let db = Database::open("./my_data")?;
    
    // Create a collection for storing documents with embeddings
    let collection = db.create_collection("documents")
        .with_vector_dimension(384)
        .with_hnsw_index()
        .build()?;
    
    // Insert data
    collection.insert(b"doc1")
        .set("title", "Hello World")
        .set_vector(&embedding)?;
    
    // Query with vector similarity
    let results = collection.vector_search(&query_embedding)
        .with_limit(10)
        .execute()?;
    
    Ok(())
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
toondb = "0.1"
```

## SDKs

| Platform | Package | Install |
|----------|---------|---------|
| Rust | [`toondb`](https://crates.io/crates/toondb) | `cargo add toondb` |
| Python | [`toondb-client`](https://pypi.org/project/toondb-client/) | `pip install toondb-client` |
| JavaScript | `toondb-client` | Coming soon |

## Documentation

- [API Reference](https://docs.toondb.dev/api-reference/)
- [Getting Started Guide](https://docs.toondb.dev/getting-started/)
- [Examples](https://github.com/toondb/toondb/tree/main/examples/rust)

## License

Apache-2.0
