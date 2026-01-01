# Tutorial: Rust SDK

> **Time:** 15 minutes  
> **Difficulty:** Beginner  
> **Prerequisites:** Rust 1.70+

Complete walkthrough of ToonDB's Rust SDK covering all access modes.

---

## What You'll Learn

- ✅ Basic key-value operations
- ✅ Path-native API for hierarchical data
- ✅ Transactions for atomic operations
- ✅ Query builder for prefix scans
- ✅ Vector search with HNSW
- ✅ Error handling patterns

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
toondb = "0.2"
```

---

## Quick Start

```rust
use toondb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    // Open database (creates if not exists)
    let db = Database::open("./my_database")?;

    // Simple key-value operations
    db.put(b"users/alice/name", b"Alice Smith")?;
    db.put(b"users/alice/email", b"alice@example.com")?;

    // Read data
    if let Some(value) = db.get(b"users/alice/name")? {
        println!("{}", String::from_utf8_lossy(&value)); // "Alice Smith"
    }

    // Check if key exists
    match db.get(b"users/bob/name")? {
        Some(_) => println!("Key exists"),
        None => println!("Key doesn't exist"),
    }

    // Delete
    db.delete(b"users/alice/email")?;

    Ok(())
}
```

---

## Path-Native API

ToonDB treats paths as first-class citizens:

```rust
use toondb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    let db = Database::open("./my_database")?;

    // Store hierarchical data using path API
    db.put_path(&["users", "alice", "profile", "name"], b"Alice")?;
    db.put_path(&["users", "alice", "profile", "age"], b"30")?;
    db.put_path(&["users", "alice", "settings", "theme"], b"dark")?;

    // Read by path
    if let Some(theme) = db.get_path(&["users", "alice", "settings", "theme"])? {
        println!("{}", String::from_utf8_lossy(&theme)); // "dark"
    }

    Ok(())
}
```

---

## Transactions

ACID-compliant transactions:

```rust
use toondb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    let db = Database::open("./my_database")?;

    // Transaction with closure (auto commit/rollback)
    db.with_transaction(|txn| {
        txn.put(b"accounts/1/balance", b"1000")?;
        txn.put(b"accounts/2/balance", b"500")?;
        Ok(())
        // Commits automatically on Ok
        // Rolls back on Err
    })?;

    // Manual transaction control
    let txn = db.begin_transaction()?;
    
    if let Err(e) = txn.put(b"key", b"value") {
        txn.abort()?;
        return Err(e.into());
    }
    
    txn.commit()?;

    Ok(())
}
```

---

## Query Builder

Fluent API for prefix scans:

```rust
use toondb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    let db = Database::open("./my_database")?;

    // Store some data
    db.put(b"products/001", br#"{"name":"Widget","price":9.99}"#)?;
    db.put(b"products/002", br#"{"name":"Gadget","price":19.99}"#)?;
    db.put(b"products/003", br#"{"name":"Gizmo","price":14.99}"#)?;

    // Query with fluent API
    let results = db.query("products/")
        .limit(10)
        .offset(0)
        .execute()?;

    for (key, value) in results {
        println!("Key: {}, Value: {}", 
            String::from_utf8_lossy(&key),
            String::from_utf8_lossy(&value));
    }

    // Get first result
    if let Some((key, value)) = db.query("products/").first()? {
        println!("First: {}", String::from_utf8_lossy(&value));
    }

    // Count results
    let count = db.query("products/").count()?;
    println!("Found {} products", count);

    Ok(())
}
```

---

## Vector Search

HNSW approximate nearest neighbor search:

```rust
use toondb::{VectorIndex, VectorIndexConfig, DistanceMetric};
use anyhow::Result;

fn main() -> Result<()> {
    // Create vector index configuration
    let config = VectorIndexConfig {
        dimension: 384,
        metric: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 100,
        ef_search: 50,
    };
    
    let index = VectorIndex::new("./vectors", config)?;

    // Build index from embeddings
    let vectors: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3], // ... 384 dims
        vec![0.4, 0.5, 0.6],
    ];
    let labels = vec!["doc1", "doc2"];

    index.bulk_build(&vectors, Some(&labels))?;

    // Query nearest neighbors
    let query_vec = vec![0.15, 0.25, 0.35]; // ... 384 dims
    let results = index.query(&query_vec, 10)?; // k=10

    for result in results {
        println!("ID: {}, Distance: {:.4}, Label: {:?}",
            result.id, result.distance, result.label);
    }

    Ok(())
}
```

### Distance Utilities

```rust
use toondb::vector::{cosine_distance, euclidean_distance, normalize};

fn main() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.707, 0.707, 0.0];

    // Cosine distance (0 = identical, 2 = opposite)
    let cos_dist = cosine_distance(&a, &b);
    println!("Cosine distance: {:.4}", cos_dist);

    // Euclidean distance
    let euc_dist = euclidean_distance(&a, &b);
    println!("Euclidean distance: {:.4}", euc_dist);

    // Normalize to unit length
    let v = vec![3.0, 4.0];
    let normalized = normalize(&v);
    println!("Normalized: {:?}", normalized); // [0.6, 0.8]
}
```

---

## Error Handling

```rust
use toondb::{Database, ToonDBError};
use anyhow::Result;

fn main() -> Result<()> {
    let db = match Database::open("./my_database") {
        Ok(db) => db,
        Err(ToonDBError::ConnectionFailed(msg)) => {
            eprintln!("Connection failed: {}", msg);
            return Err(anyhow::anyhow!("Connection failed"));
        }
        Err(e) => return Err(e.into()),
    };

    // Pattern matching on errors
    match db.get(b"key") {
        Ok(Some(value)) => println!("Found: {:?}", value),
        Ok(None) => println!("Key not found"),
        Err(ToonDBError::DatabaseClosed) => {
            eprintln!("Database was closed");
        }
        Err(e) => return Err(e.into()),
    }

    // Transaction error handling
    let result = db.with_transaction(|txn| {
        txn.put(b"key", b"value")?;
        Err(ToonDBError::Custom("simulated failure".into()))
    });
    
    match result {
        Ok(_) => println!("Transaction committed"),
        Err(e) => println!("Transaction failed: {}", e),
    }

    Ok(())
}
```

### Error Types

```rust
pub enum ToonDBError {
    /// Key not found (informational, not typically an error)
    NotFound,
    
    /// Database is closed
    DatabaseClosed,
    
    /// Transaction already committed
    TransactionCommitted,
    
    /// Transaction already aborted
    TransactionAborted,
    
    /// Connection to server failed
    ConnectionFailed(String),
    
    /// Wire protocol error
    ProtocolError(String),
    
    /// Vector dimension mismatch
    VectorDimensionMismatch { expected: usize, got: usize },
    
    /// I/O error
    Io(std::io::Error),
    
    /// Custom error
    Custom(String),
}
```

---

## Configuration Options

```rust
use toondb::{Database, DatabaseConfig, SyncMode};
use anyhow::Result;

fn main() -> Result<()> {
    let config = DatabaseConfig {
        // Path to database directory (required)
        path: "./my_database".into(),

        // Create directory if missing (default: true)
        create_if_missing: true,

        // Enable Write-Ahead Logging (default: true)
        wal_enabled: true,

        // Sync mode (default: Normal)
        sync_mode: SyncMode::Normal, // Full, Normal, or Off

        // Maximum memtable size before flush (default: 64MB)
        memtable_size_bytes: 64 * 1024 * 1024,
    };

    let db = Database::open_with_config(config)?;

    Ok(())
}
```

---

## Best Practices

### 1. Use RAII for Database Lifecycle

```rust
fn process_data() -> Result<()> {
    let db = Database::open("./my_database")?;
    // Database automatically closes when `db` goes out of scope
    
    db.put(b"key", b"value")?;
    
    Ok(())
} // db.close() called automatically via Drop
```

### 2. Use with_transaction for Atomic Operations

```rust
// Good: automatic commit/rollback
db.with_transaction(|txn| {
    txn.put(b"key1", b"value1")?;
    txn.put(b"key2", b"value2")?;
    Ok(())
})?;

// Avoid: manual transaction handling (error-prone)
let txn = db.begin_transaction()?;
// ...
txn.commit()?;
```

### 3. Handle `Option<Vec<u8>>` Returns

```rust
match db.get(b"key")? {
    Some(value) => {
        // Process value
    }
    None => {
        // Key doesn't exist - this is NOT an error
    }
}
```

### 4. Use Byte Literals for Keys

```rust
// Good: byte literals
db.put(b"users/alice", b"data")?;

// Also fine: String conversion
let key = format!("users/{}", user_id);
db.put(key.as_bytes(), data)?;
```

---

## Async Support

ToonDB provides async variants for use with Tokio:

```rust
use toondb::AsyncDatabase;
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db = AsyncDatabase::open("./my_database").await?;

    db.put(b"key", b"value").await?;
    
    let value = db.get(b"key").await?;
    
    db.with_transaction(|txn| async move {
        txn.put(b"k1", b"v1").await?;
        txn.put(b"k2", b"v2").await?;
        Ok(())
    }).await?;

    Ok(())
}
```

---

## Next Steps

- [Vector Search Guide](./vector-search.md) - Deep dive into HNSW
- [Bulk Operations](./bulk-operations.md) - High-throughput indexing
- [API Reference](../api-reference/rust.md) - Complete API docs
