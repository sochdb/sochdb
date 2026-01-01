# Tutorial: Go SDK

> **Time:** 15 minutes  
> **Difficulty:** Beginner  
> **Prerequisites:** Go 1.21+

Complete walkthrough of ToonDB's Go SDK covering all access modes.

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

```bash
go get github.com/toondb/toondb/toondb-go
```

**Pure Go** — No CGO dependencies required.

---

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    // Open database (creates if not exists)
    db, err := toondb.Open("./my_database")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Simple key-value operations
    err = db.PutString("users/alice/name", "Alice Smith")
    if err != nil {
        log.Fatal(err)
    }

    // Read data
    value, err := db.GetString("users/alice/name")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(value)) // "Alice Smith"

    // Check if key exists (nil = not found)
    value, _ = db.GetString("users/bob/name")
    if value == nil {
        fmt.Println("Key doesn't exist")
    }

    // Delete
    err = db.DeleteString("users/alice/email")
}
```

---

## Path-Native API

ToonDB treats paths as first-class citizens:

```go
package main

import (
    "fmt"
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    db, err := toondb.Open("./my_database")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Store hierarchical data
    db.PutPath("users/alice/profile/name", []byte("Alice"))
    db.PutPath("users/alice/profile/age", []byte("30"))
    db.PutPath("users/alice/settings/theme", []byte("dark"))

    // Read by path
    theme, err := db.GetPath("users/alice/settings/theme")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(theme)) // "dark"
}
```

---

## Transactions

Atomic operations with automatic commit/abort:

```go
package main

import (
    "fmt"
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    db, err := toondb.Open("./my_database")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Transaction with automatic commit/abort
    err = db.WithTransaction(func(txn *toondb.Transaction) error {
        if err := txn.PutString("accounts/1/balance", "1000"); err != nil {
            return err // Transaction aborts
        }
        if err := txn.PutString("accounts/2/balance", "500"); err != nil {
            return err
        }
        return nil // Transaction commits
    })
    if err != nil {
        log.Fatal(err)
    }

    // Manual transaction (prefer WithTransaction)
    txn, err := db.BeginTransaction()
    if err != nil {
        log.Fatal(err)
    }
    
    if err := txn.PutString("key", "value"); err != nil {
        txn.Abort()
        log.Fatal(err)
    }
    
    if err := txn.Commit(); err != nil {
        log.Fatal(err)
    }
}
```

---

## Query Builder

Fluent API for prefix scans:

```go
package main

import (
    "fmt"
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    db, err := toondb.Open("./my_database")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Store some data
    db.PutPathString("products/001", `{"name":"Widget","price":9.99}`)
    db.PutPathString("products/002", `{"name":"Gadget","price":19.99}`)
    db.PutPathString("products/003", `{"name":"Gizmo","price":14.99}`)

    // Query with fluent API
    results, err := db.Query("products/").
        Limit(10).
        Offset(0).
        Select("name", "price").
        Execute()
    if err != nil {
        log.Fatal(err)
    }

    for _, kv := range results {
        fmt.Printf("Key: %s, Value: %s\n", kv.Key, kv.Value)
    }

    // Get first result
    first, err := db.Query("products/").First()
    if first != nil {
        fmt.Printf("First: %s\n", first.Value)
    }

    // Count results
    count, err := db.Query("products/").Count()
    fmt.Printf("Found %d products\n", count)

    // Check existence
    exists, err := db.Query("products/").Exists()
    fmt.Printf("Products exist: %v\n", exists)

    // Iterate with callback
    err = db.Query("products/").ForEach(func(kv toondb.KeyValue) error {
        fmt.Println(string(kv.Key))
        return nil
    })
}
```

---

## Vector Search

HNSW approximate nearest neighbor search:

```go
package main

import (
    "fmt"
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    // Create vector index configuration
    config := &toondb.VectorIndexConfig{
        Dimension:      384,
        Metric:         toondb.Cosine,
        M:              16,
        EfConstruction: 100,
        EfSearch:       50,
    }
    index := toondb.NewVectorIndex("./vectors", config)

    // Build index from embeddings
    vectors := [][]float32{
        {0.1, 0.2, 0.3}, // ... 384 dims
        {0.4, 0.5, 0.6},
    }
    labels := []string{"doc1", "doc2"}

    if err := index.BulkBuild(vectors, labels); err != nil {
        log.Fatal(err)
    }

    // Query nearest neighbors
    queryVec := []float32{0.15, 0.25, 0.35} // ... 384 dims
    results, err := index.Query(queryVec, 10, 50) // k=10, ef_search=50
    if err != nil {
        log.Fatal(err)
    }

    for _, r := range results {
        fmt.Printf("ID: %d, Distance: %.4f, Label: %s\n",
            r.ID, r.Distance, r.Label)
    }
}
```

### Distance Utilities

```go
package main

import (
    "fmt"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    a := []float32{1, 0, 0}
    b := []float32{0.707, 0.707, 0}

    // Cosine distance (0 = identical, 2 = opposite)
    cosDist := toondb.ComputeCosineDistance(a, b)
    fmt.Printf("Cosine distance: %.4f\n", cosDist)

    // Euclidean distance
    eucDist := toondb.ComputeEuclideanDistance(a, b)
    fmt.Printf("Euclidean distance: %.4f\n", eucDist)

    // Normalize to unit length
    v := []float32{3, 4}
    normalized := toondb.NormalizeVector(v)
    fmt.Printf("Normalized: %v\n", normalized) // [0.6, 0.8]
}
```

---

## Error Handling

```go
package main

import (
    "errors"
    "fmt"
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    db, err := toondb.Open("./my_database")
    if err != nil {
        var connErr *toondb.ConnectionError
        if errors.As(err, &connErr) {
            log.Fatalf("Connection failed to %s: %v", connErr.Address, connErr.Err)
        }
        log.Fatal(err)
    }
    defer db.Close()

    // Check for specific errors
    value, err := db.Get([]byte("key"))
    if err != nil {
        if errors.Is(err, toondb.ErrClosed) {
            fmt.Println("Database is closed")
        }
        log.Fatal(err)
    }

    // nil value means key not found (not an error)
    if value == nil {
        fmt.Println("Key not found")
    }

    // Transaction errors
    err = db.WithTransaction(func(txn *toondb.Transaction) error {
        return fmt.Errorf("simulated failure")
    })
    if err != nil {
        fmt.Println("Transaction failed:", err)
    }
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `ErrNotFound` | Key was not found |
| `ErrClosed` | Database is closed |
| `ErrTxnCommitted` | Transaction already committed |
| `ErrTxnAborted` | Transaction already aborted |
| `ErrConnectionFailed` | Failed to connect to server |
| `ErrProtocol` | Wire protocol error |
| `ErrVectorDimension` | Vector dimension mismatch |

### Error Wrapper Types

| Type | Description |
|------|-------------|
| `ToonDBError` | Wraps errors with operation context |
| `ConnectionError` | Connection-specific failures |
| `TransactionError` | Transaction-specific failures |
| `ProtocolError` | Wire protocol errors |

---

## Configuration Options

```go
package main

import (
    "log"

    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    config := &toondb.Config{
        // Path to database directory (required)
        Path: "./my_database",

        // Create directory if missing (default: true)
        CreateIfMissing: true,

        // Enable Write-Ahead Logging (default: true)
        WALEnabled: true,

        // Sync mode: "full", "normal", or "off" (default: "normal")
        SyncMode: "normal",

        // Maximum memtable size before flush (default: 64MB)
        MemtableSizeBytes: 64 * 1024 * 1024,
    }

    db, err := toondb.OpenWithConfig(config)
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()
}
```

---

## Best Practices

### 1. Always Close the Database

```go
db, err := toondb.Open("./my_database")
if err != nil {
    log.Fatal(err)
}
defer db.Close() // Always defer close
```

### 2. Use WithTransaction for Atomic Operations

```go
// Good: automatic commit/abort
err := db.WithTransaction(func(txn *toondb.Transaction) error {
    // Operations...
    return nil
})

// Avoid: manual transaction handling (error-prone)
txn, _ := db.BeginTransaction()
// ...
txn.Commit()
```

### 3. Handle nil Values

```go
value, err := db.Get(key)
if err != nil {
    return err
}
if value == nil {
    // Key doesn't exist - this is NOT an error!
    return fmt.Errorf("key not found: %s", key)
}
```

### 4. Batch Operations in Transactions

```go
// Efficient: batch multiple writes
err := db.WithTransaction(func(txn *toondb.Transaction) error {
    for _, item := range items {
        if err := txn.Put(item.Key, item.Value); err != nil {
            return err
        }
    }
    return nil
})
```

---

## Next Steps

- [Vector Search Guide](./vector-search.md) - Deep dive into HNSW
- [Bulk Operations](./bulk-operations.md) - High-throughput indexing
- [API Reference](../api-reference/go.md) - Complete API docs
