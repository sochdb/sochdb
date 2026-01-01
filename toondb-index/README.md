# ToonDB Index

High-performance vector indexing and embedding integration for agent observability.

## Features

### Vector Indices

- **HNSW** - Hierarchical Navigable Small World graphs for approximate nearest neighbor search
- **Vamana** - DiskANN-style single-layer graph with Product Quantization (32x compression)

### Embedding Integration

Complete embedding pipeline for semantic search:

- **Providers**: Local ONNX (offline), OpenAI API (cloud), Mock (testing)
- **Pipeline**: Batched processing with background workers
- **Storage**: Persistent embedding storage with PQ compression
- **Normalization**: SIMD-optimized L2 normalization
- **Integration**: Seamless connection to HNSW/Vamana indices

## Quick Start

### Embedding Provider

```rust
use toondb_index::embedding::{LocalEmbeddingProvider, EmbeddingProvider};

// Create local provider (offline, no API cost)
let provider = LocalEmbeddingProvider::default_provider()?;

// Embed text
let vector = provider.embed("Find traces with errors")?;
println!("Embedded to {} dimensions", vector.len());

// Batch embedding
let texts = vec!["query 1", "query 2", "query 3"];
let vectors = provider.embed_batch(&texts)?;
```

### Embedding Integration with Index

```rust
use toondb_index::embedding::{
    EmbeddingIntegration, IntegrationConfig, LocalEmbeddingProvider,
};
use std::sync::Arc;

// Create provider and integration
let provider = Arc::new(LocalEmbeddingProvider::default_provider()?);
let mut integration = EmbeddingIntegration::new(provider, IntegrationConfig::default())?;

// Start background worker
integration.start_background_worker();

// Submit traces for embedding (non-blocking)
integration.submit_for_embedding(edge_id, "Error in authentication module")?;

// Semantic search
let results = integration.semantic_search("find auth errors", 10)?;
for result in results {
    println!("Edge {}: similarity={:.3}", result.edge_id, result.similarity);
}
```

### Vamana with Product Quantization

```rust
use toondb_index::{VamanaConfig, VamanaIndex, PQCodebooks};

// Train PQ codebooks on sample vectors
let codebooks = PQCodebooks::train(&sample_vectors, 20, 8);

// Create Vamana index with PQ
let config = VamanaConfig::default();
let mut index = VamanaIndex::new(384, config);

// Insert vectors (automatically PQ-encoded)
for (id, vector) in vectors {
    index.insert(id, vector);
}

// Search
let results = index.search(&query_vector, 10);
```

## Memory Efficiency

With Product Quantization (PQ):
- 384-dim vector (1536 bytes as f32) → 48 bytes as PQ codes
- **32x compression ratio**

| Vectors | Full F32 | PQ Compressed | Savings |
|---------|----------|---------------|---------|
| 1M      | 1.5 GB   | 48 MB         | 97%     |
| 10M     | 15 GB    | 480 MB        | 97%     |
| 100M    | 150 GB   | 4.8 GB        | 97%     |

## Architecture

```
Trace → LSM Write (sync) → CSR Update (sync) → Embedding Queue (async)
                                                       ↓
                                                Background Worker
                                                       ↓
                                          Embed → Normalize → PQ Encode
                                                       ↓
                                                 HNSW/Vamana Insert
```

## Modules

- `hnsw` - HNSW index with concurrent insert/search
- `vamana` - DiskANN-style single-layer graph
- `product_quantization` - 32x vector compression
- `embedding/provider` - Embedding abstraction trait
- `embedding/pipeline` - Batched processing pipeline
- `embedding/storage` - Persistent embedding storage
- `embedding/normalize` - SIMD L2 normalization
- `embedding/index_integration` - Connect pipeline to index

## Testing

```bash
# Run all tests
cargo test -p toondb-index

# Run embedding tests only
cargo test -p toondb-index embedding::

# Run integration tests
cargo test -p toondb-index --test vamana_integration_test
```

## Performance

### Embedding Latency

| Provider | Single Text | Batch 32 |
|----------|-------------|----------|
| Local    | ~5 ms       | ~50 ms   |
| OpenAI   | ~100 ms     | ~150 ms  |

### Search Latency (10M vectors)

| Index  | Top-10 | Top-100 |
|--------|--------|---------|
| HNSW   | <5 ms  | <10 ms  |
| Vamana | <3 ms  | <8 ms   |

## License

Same as the parent ToonDB project.
