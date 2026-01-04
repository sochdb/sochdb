# ToonDB Analytics

ToonDB includes optional, privacy-respecting analytics to help improve the database. This page explains what data is collected, how to disable analytics, and our privacy practices.

## What Is Collected

ToonDB collects **anonymous usage metrics** to help us understand:

- Which SDK features are most used
- Performance characteristics (latency, throughput)
- Error patterns for debugging
- Platform distribution (OS, architecture)

### Example Event Data

```json
{
  "event": "vector_search",
  "properties": {
    "sdk": "python",
    "sdk_version": "0.3.1",
    "os": "Darwin",
    "arch": "arm64",
    "dimension": 1536,
    "k": 10,
    "latency_ms": 45.2
  },
  "distinct_id": "a1b2c3d4e5f6g7h8"  // Anonymous machine hash
}
```

## What Is NOT Collected

We **never** collect:

- ❌ Database contents or query data
- ❌ API keys or credentials
- ❌ Personal information (names, emails, IPs)
- ❌ File paths or directory structures
- ❌ Hostnames (only a hash is used for distinct_id)

## Disabling Analytics

To disable all analytics, set the environment variable:

```bash
# Bash/Zsh
export TOONDB_DISABLE_ANALYTICS=true

# Windows PowerShell
$env:TOONDB_DISABLE_ANALYTICS = "true"

# Windows CMD
set TOONDB_DISABLE_ANALYTICS=true

# In Python
import os
os.environ["TOONDB_DISABLE_ANALYTICS"] = "true"

# In Node.js
process.env.TOONDB_DISABLE_ANALYTICS = "true";
```

### Verifying Analytics Status

#### Python
```python
from toondb import is_analytics_disabled
print(f"Analytics disabled: {is_analytics_disabled()}")
```

#### JavaScript/TypeScript
```typescript
import { isAnalyticsDisabled } from '@sushanth/toondb';
console.log(`Analytics disabled: ${isAnalyticsDisabled()}`);
```

#### Rust
```rust
use toondb_core::analytics::is_analytics_disabled;
println!("Analytics disabled: {}", is_analytics_disabled());
```

## Analytics Provider

ToonDB uses [PostHog](https://posthog.com) for analytics. PostHog is an open-source product analytics platform that respects user privacy and is GDPR compliant.

- **Data is sent to**: `https://us.i.posthog.com`
- **Data retention**: Aggregated metrics only
- **No third-party sharing**: Data is only used by ToonDB developers

## Optional Dependency

The analytics package is **optional**:

- **Python**: Install with `pip install toondb-client[analytics]`
- **Node.js**: posthog-node is in `optionalDependencies`
- **Rust**: Enable the `analytics` feature flag

If the analytics package is not installed, all tracking functions become no-ops.

## Events Tracked

| Event | Description | Properties |
|-------|-------------|------------|
| `database_opened` | Database connection established | mode, has_custom_path |
| `vector_search` | Vector similarity search performed | dimension, k, latency_ms |
| `batch_insert` | Batch vector insertion | count, dimension, latency_ms |
| `error` | Error occurred (sanitized) | error_type, error_message |

## Source Code

Analytics implementation is fully open source:

- Python: [toondb-python-sdk/src/toondb/analytics.py](https://github.com/toondb/toondb/blob/main/toondb-python-sdk/src/toondb/analytics.py)
- JavaScript: [toondb-js/src/analytics.ts](https://github.com/toondb/toondb/blob/main/toondb-js/src/analytics.ts)
- Rust: [toondb-core/src/analytics.rs](https://github.com/toondb/toondb/blob/main/toondb-core/src/analytics.rs)

## Questions?

If you have any questions or concerns about analytics, please [open an issue](https://github.com/toondb/toondb/issues) or email sushanth@toondb.dev.
