# How to Configure Logging

> Enable debug logging for troubleshooting and audit logs for production.

---

## Problem

You need to see what ToonDB is doing internally for debugging, or capture audit logs for compliance.

---

## Solution

### 1. Environment Variable (Quick)

```bash
# Debug logging (development)
export RUST_LOG=toondb=debug

# Trace logging (verbose)
export RUST_LOG=toondb=trace

# Specific module only
export RUST_LOG=toondb_kernel::wal=debug

# Multiple modules
export RUST_LOG=toondb_kernel=debug,toondb_index::hnsw=trace
```

### 2. Configuration File (Production)

Create `toondb.toml`:

```toml
[logging]
# Log level: error, warn, info, debug, trace
level = "info"

# Log format: "json" or "pretty"
format = "json"

# Output destination
output = "file"  # "stdout", "stderr", or "file"
file_path = "/var/log/toondb/toondb.log"

# Log rotation
max_size_mb = 100
max_files = 10
compress = true

# Audit logging (separate from debug logs)
[logging.audit]
enabled = true
file_path = "/var/log/toondb/audit.log"
events = ["transaction_commit", "schema_change", "auth_failure"]
```

### 3. Programmatic Configuration (Rust)

```rust
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn setup_logging() {
    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(EnvFilter::from_default_env()
            .add_directive("toondb=debug".parse().unwrap()))
        .init();
}
```

### 4. Python SDK Logging

```python
import logging

# Enable ToonDB debug logs
logging.basicConfig(level=logging.DEBUG)

# Or configure specific logger
logger = logging.getLogger("toondb")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("toondb.log")
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)
```

---

## Example Output

### Debug Format
```
2024-12-30T10:15:32.123Z DEBUG toondb_kernel::wal > Appending record: txn_id=42, size=1024
2024-12-30T10:15:32.124Z DEBUG toondb_kernel::mvcc > Snapshot created: ts=1000042
2024-12-30T10:15:32.125Z DEBUG toondb_index::hnsw > Search: k=10, ef=50, candidates=127
```

### JSON Format (Production)
```json
{"timestamp":"2024-12-30T10:15:32.123Z","level":"INFO","target":"toondb_kernel::txn","message":"Transaction committed","txn_id":42,"duration_ms":3.2}
```

### Audit Log
```json
{"event":"transaction_commit","txn_id":42,"user":"app_service","timestamp":"2024-12-30T10:15:32Z","tables_modified":["users","orders"]}
```

---

## Discussion

### Log Levels

| Level | Use Case | Performance Impact |
|-------|----------|-------------------|
| `error` | Production (critical only) | Minimal |
| `warn` | Production (with warnings) | Minimal |
| `info` | Production (recommended) | Low |
| `debug` | Development/troubleshooting | Moderate |
| `trace` | Deep debugging only | High |

### Performance Considerations

- JSON format is ~10% slower than plain text
- File output with rotation is recommended for production
- Audit logs should be on a separate file for compliance
- Use `RUST_LOG` filtering to minimize overhead

### Security Notes

- Audit logs may contain sensitive paths/keys
- Rotate and encrypt log files in production
- Consider log shipping to SIEM for compliance

---

## See Also

- [Profiling Guide](/internals/profiling) — Performance analysis
- [Deployment Guide](/guides/deployment) — Production setup
- [Performance Guide](/concepts/performance) — Optimization tips

