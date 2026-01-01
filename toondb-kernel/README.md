# ToonDB Kernel

The minimal ACID core of ToonDB with a plugin architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Extension Layer                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ LSCS Plugin │ │Vector Plugin│ │ Observability Plugin│   │
│  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘   │
│         │               │                    │              │
│         ▼               ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Plugin Manager (Registry)               │   │
│  └─────────────────────────┬───────────────────────────┘   │
└────────────────────────────┼────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Kernel API (Traits)                  │   │
│  │   KernelStorage, KernelTransaction, KernelCatalog    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │   WAL    │ │   MVCC   │ │  Pager   │ │   Catalog    │   │
│  │ Recovery │ │   Txn    │ │  Buffer  │ │   Schema     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
│                                                             │
│                     KERNEL (~5K LOC)                        │
│              Auditable • Stable API • ACID                  │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Minimal Core**: Only ACID-critical code in kernel (<5K LOC)
2. **Plugin Everything**: Storage backends, indices, observability are plugins
3. **No Dependency Bloat**: Core has minimal deps, plugins bring their own
4. **Stable API**: Kernel API is versioned, plugins can evolve independently
5. **Auditable**: Small enough for formal verification

## Plugin Categories

| Category | Trait | Purpose |
|----------|-------|---------|
| Storage | `StorageExtension` | Alternative storage backends (LSCS, RocksDB, etc.) |
| Index | `IndexExtension` | Custom index types (vector, learned, full-text) |
| Observability | `ObservabilityExtension` | Metrics, tracing, logging backends |
| Compression | `CompressionExtension` | Compression algorithms |

## Why Plugin Architecture for Observability?

### Problems with Baked-In Monitoring:

1. **Dependency Bloat**: Forces every user to pull in Prometheus, Grafana, etc. even if they use DataDog, CloudWatch, or nothing
2. **Version Conflicts**: User's app might need different Prometheus client version
3. **Deployment Flexibility**: Embedded use cases (mobile, WASM) don't need monitoring
4. **Vendor Lock-in**: Forces specific monitoring stack on users
5. **Binary Size**: Adds MBs to binary for features most users don't need

### Plugin Architecture Benefits:

```
Core ToonDB: 5 MB binary ← Everyone downloads this
  + prometheus-plugin: +500 KB ← Only if you want Prometheus
  + datadog-plugin: +300 KB ← Or DataDog
  + opentelemetry-plugin: +400 KB ← Or OpenTelemetry
  + logging-plugin-json: +50 KB ← Structured logging
  + logging-plugin-logfmt: +50 KB ← Or logfmt style
```

## Usage

### Basic Kernel Usage

```rust
use toondb_kernel::{
    PluginManager, TxnManager, WalManager,
    IsolationLevel,
};

// Create kernel components
let wal = WalManager::open("data/wal")?;
let txn_mgr = TxnManager::new();
let plugins = PluginManager::new();

// Begin a transaction
let txn_id = txn_mgr.begin();

// Log operations
wal.log_begin(txn_id)?;
wal.log_update(txn_id, page_id, redo_data, undo_data)?;

// Commit
txn_mgr.commit(txn_id)?;
wal.log_commit(txn_id)?;
```

### Adding Observability Plugin

```rust
use toondb_kernel::{PluginManager, ObservabilityExtension, Extension, ExtensionInfo};
use std::sync::Arc;

// Define a custom observability plugin
struct PrometheusPlugin { /* ... */ }

impl Extension for PrometheusPlugin {
    fn info(&self) -> ExtensionInfo {
        ExtensionInfo {
            name: "prometheus".into(),
            version: "1.0.0".into(),
            description: "Prometheus metrics exporter".into(),
            author: "Your Team".into(),
            capabilities: vec![ExtensionCapability::Observability],
        }
    }
    // ... other required methods
}

impl ObservabilityExtension for PrometheusPlugin {
    fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        // Push to Prometheus
    }
    // ... other metrics methods
}

// Register the plugin
let plugins = PluginManager::new();
plugins.register_observability(Arc::new(PrometheusPlugin::new()))?;

// Kernel will now emit metrics to all registered observability plugins
plugins.counter_inc("toondb_txns_committed", 1, &[("isolation", "serializable")]);
```

### Null Observability (Default)

When no observability plugin is registered, the kernel uses `NullObservability` which has zero overhead:

```rust
let plugins = PluginManager::new();
assert!(!plugins.has_observability()); // No observability configured

// These calls are no-ops with zero overhead
plugins.counter_inc("metric", 1, &[]);
plugins.gauge_set("gauge", 42.0, &[]);
```

## Kernel API Stability

The kernel API follows semantic versioning:

- **1.x.x**: No breaking changes to trait signatures
- **2.0.0**: May introduce breaking changes with migration guide

Core traits that are stable:
- `KernelStorage`
- `KernelTransaction`  
- `KernelCatalog`
- `KernelRecovery`
- `Extension` and sub-traits

## Code Budget

To ensure auditability, the kernel maintains a strict code budget:

| Component | Target LOC |
|-----------|-----------|
| WAL | ~1,000 |
| Transaction | ~800 |
| Page Manager | ~1,000 |
| Catalog | ~500 |
| Plugin System | ~700 |
| Error Handling | ~500 |
| **Total** | **~5,000** |

## License

Apache-2.0 OR MIT
