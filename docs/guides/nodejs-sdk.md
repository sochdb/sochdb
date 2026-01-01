# Tutorial: Node.js/TypeScript SDK

> **Time:** 15 minutes  
> **Difficulty:** Beginner  
> **Prerequisites:** Node.js 18+, npm or yarn

Complete walkthrough of ToonDB's JavaScript/TypeScript SDK covering all access modes.

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
npm install @sushanth/toondb
# or
yarn add @sushanth/toondb
```

**Zero compilation required** — pre-built for all platforms:
- Linux (x86_64, aarch64)
- macOS (Intel, Apple Silicon)
- Windows (x64)

Supports both **CommonJS** and **ES Modules**.

---

## Quick Start

```typescript
import { Database } from '@sushanth/toondb';

// Open database (creates if not exists)
const db = await Database.open('./my_database');

// Simple key-value operations
await db.put('users/alice/name', 'Alice Smith');
await db.put('users/alice/email', 'alice@example.com');

// Read data
const name = await db.get('users/alice/name');
console.log(name?.toString()); // "Alice Smith"

// Check if key exists
const value = await db.get('users/bob/name');
if (value === null) {
  console.log("Key doesn't exist");
}

// Delete
await db.delete('users/alice/email');

// Close when done
await db.close();
```

---

## Path-Native API

ToonDB treats paths as first-class citizens:

```typescript
import { Database } from '@sushanth/toondb';

const db = await Database.open('./my_database');

// Store hierarchical data
await db.putPath('users/alice/profile/name', 'Alice');
await db.putPath('users/alice/profile/age', '30');
await db.putPath('users/alice/settings/theme', 'dark');

// Read by path
const theme = await db.getPath('users/alice/settings/theme');
console.log(theme?.toString()); // "dark"

db.close();
```

---

## Transactions

Atomic operations with automatic commit/abort:

```typescript
import { Database } from '@sushanth/toondb';

const db = await Database.open('./my_database');

// Transaction with automatic commit/abort
await db.withTransaction(async (txn) => {
  await txn.put('accounts/1/balance', '1000');
  await txn.put('accounts/2/balance', '500');
  
  // Commits automatically on success
  // Aborts automatically on error
});

// Return values from transactions
const total = await db.withTransaction(async (txn) => {
  const bal1 = await txn.get('accounts/1/balance');
  const bal2 = await txn.get('accounts/2/balance');
  
  return parseInt(bal1?.toString() || '0') + 
         parseInt(bal2?.toString() || '0');
});

console.log(`Total: ${total}`);

db.close();
```

---

## Query Builder

Fluent API for prefix scans:

```typescript
import { Database } from '@sushanth/toondb';

const db = await Database.open('./my_database');

// Store some data
await db.putPath('products/001', JSON.stringify({ name: 'Widget', price: 9.99 }));
await db.putPath('products/002', JSON.stringify({ name: 'Gadget', price: 19.99 }));
await db.putPath('products/003', JSON.stringify({ name: 'Gizmo', price: 14.99 }));

// Query with fluent API
const results = await db.query('products/')
  .limit(10)
  .offset(0)
  .select(['name', 'price'])
  .toList();

for (const item of results) {
  console.log(item);
}

// Get first result
const first = await db.query('products/').first();

// Count results
const count = await db.query('products/').count();
console.log(`Found ${count} products`);

db.close();
```

---

## Vector Search

HNSW approximate nearest neighbor search:

```typescript
import { VectorIndex } from '@sushanth/toondb';

// Create vector index
const index = new VectorIndex('./vectors', {
  dimension: 384,
  metric: 'cosine',
  m: 16,
  efConstruction: 100,
});

// Build index from embeddings
const vectors = [
  [0.1, 0.2, 0.3, /* ... 384 dims */],
  [0.4, 0.5, 0.6, /* ... 384 dims */],
];
const labels = ['doc1', 'doc2'];

await index.bulkBuild(vectors, labels);

// Query nearest neighbors
const queryVec = [0.15, 0.25, 0.35, /* ... 384 dims */];
const results = await VectorIndex.query('./vectors', new Float32Array(queryVec), {
  k: 10,
  efSearch: 64,
});

for (const r of results) {
  console.log(`ID: ${r.id}, Distance: ${r.distance.toFixed(4)}`);
}
```

### Distance Utilities

```typescript
import { VectorIndex } from '@sushanth/toondb';

const a = [1, 0, 0];
const b = [0.707, 0.707, 0];

// Cosine distance (0 = identical, 2 = opposite)
const cosDist = VectorIndex.computeCosineDistance(a, b);

// Euclidean distance
const eucDist = VectorIndex.computeEuclideanDistance(a, b);

// Dot product
const dot = VectorIndex.computeDotProduct(a, b);

// Normalize to unit length
const normalized = VectorIndex.normalizeVector([3, 4]); // [0.6, 0.8]
```

---

## Error Handling

```typescript
import { 
  Database, 
  ToonDBError, 
  ConnectionError, 
  TransactionError,
  DatabaseError 
} from '@sushanth/toondb';

try {
  const db = await Database.open('./my_database');
  
  await db.withTransaction(async (txn) => {
    await txn.put('key', 'value');
    throw new Error('Simulated failure');
  });
  
} catch (err) {
  if (err instanceof ConnectionError) {
    console.error('Connection failed:', err.message);
  } else if (err instanceof TransactionError) {
    console.error('Transaction failed:', err.message);
  } else if (err instanceof DatabaseError) {
    console.error('Database error:', err.message);
  } else if (err instanceof ToonDBError) {
    console.error('ToonDB error:', err.message);
  } else {
    throw err;
  }
}
```

### Error Hierarchy

| Error Type | Description |
|------------|-------------|
| `ToonDBError` | Base class for all ToonDB errors |
| `ConnectionError` | Connection to server failed |
| `TransactionError` | Transaction commit/abort failed |
| `ProtocolError` | Wire protocol error |
| `DatabaseError` | General database operation error |

---

## Configuration Options

```typescript
import { Database } from '@sushanth/toondb';

const db = await Database.open({
  // Path to database directory (required)
  path: './my_database',
  
  // Create directory if missing (default: true)
  createIfMissing: true,
  
  // Enable Write-Ahead Logging (default: true)
  walEnabled: true,
  
  // Sync mode: 'full' | 'normal' | 'off' (default: 'normal')
  syncMode: 'normal',
  
  // Maximum memtable size before flush (default: 64MB)
  memtableSizeBytes: 64 * 1024 * 1024,
});
```

---

## TypeScript Support

Full type definitions included:

```typescript
import { 
  Database, 
  DatabaseConfig,
  Transaction,
  Query,
  VectorIndex,
  VectorIndexConfig,
  VectorSearchResult,
  IpcClient,
  IpcClientConfig,
} from '@sushanth/toondb';

// All types are exported
const config: DatabaseConfig = {
  path: './my_db',
  walEnabled: true,
};

// Async/await fully typed
const db: Database = await Database.open(config);
const value: Buffer | null = await db.get('key');
```

---

## Best Practices

### 1. Always Close the Database

```typescript
const db = await Database.open('./my_database');
try {
  // Operations...
} finally {
  await db.close();
}
```

### 2. Use withTransaction for Atomic Operations

```typescript
// Good: automatic commit/abort
await db.withTransaction(async (txn) => {
  await txn.put('key1', 'value1');
  await txn.put('key2', 'value2');
});

// Avoid: manual transaction handling is error-prone
```

### 3. Handle null Returns

```typescript
const value = await db.get('key');
if (value === null) {
  // Key doesn't exist - NOT an error
}
```

### 4. Batch Operations in Transactions

```typescript
// Efficient: batch writes in single transaction
await db.withTransaction(async (txn) => {
  for (const item of items) {
    await txn.put(item.key, item.value);
  }
});
```

---

## Next Steps

- [Vector Search Guide](./vector-search.md) - Deep dive into HNSW
- [Bulk Operations](./bulk-operations.md) - High-throughput indexing
- [API Reference](../api-reference/nodejs.md) - Complete API docs
