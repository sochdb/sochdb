# ToonDB Python SDK Examples

This directory contains practical examples demonstrating ToonDB Python SDK usage across various scenarios.

## Prerequisites

```bash
# Install the SDK
cd toondb-python-sdk
pip install -e .

# Build the native library (for embedded mode)
cd ..
cargo build --release
```

## Examples Overview

| Example | Mode | Description |
|---------|------|-------------|
| `01_basic_operations.py` | Embedded | Basic CRUD operations |
| `02_transactions.py` | Embedded | Transaction handling and rollback |
| `03_path_navigation.py` | Embedded | Hierarchical path-based data access |
| `04_scan_and_range.py` | Embedded | Range scans and batch operations |
| `05_user_store.py` | Embedded | Real-world user management example |
| `06_json_documents.py` | Embedded | Storing and querying JSON documents |
| `07_session_cache.py` | Embedded | Session caching use case |
| `08_ipc_client.py` | IPC | Multi-process access via IPC |

## Running Examples

### Embedded Mode Examples (01-07)

```bash
# Set the library path
export TOONDB_LIB_PATH=/path/to/toon_database/target/release

# Run any example
python examples/01_basic_operations.py
```

### IPC Mode Example (08)

Requires a running ToonDB IPC server:

```bash
# Start the server first (from toondb-storage)
cargo run --bin ipc_server -- --socket /tmp/toondb.sock

# Then run the example
python examples/08_ipc_client.py
```

## Directory Structure

```
examples/
├── README.md           # This file
├── 01_basic_operations.py      # Simple key-value CRUD
├── 02_transactions.py          # ACID transaction examples
├── 03_path_navigation.py       # Path-native API examples
├── 04_scan_and_range.py        # Range queries
├── 05_user_store.py            # User management use case
├── 06_json_documents.py        # JSON document storage
├── 07_session_cache.py         # Session caching pattern
├── 08_ipc_client.py            # IPC client examples
└── shared/
    └── mock_server.py          # Mock server for testing
```

## Quick Start

The simplest example to get started:

```python
from toondb import Database

# Open/create a database
db = Database.open("./my_data")

# Store data
db.put(b"greeting", b"Hello, ToonDB!")

# Retrieve data
value = db.get(b"greeting")
print(value)  # b"Hello, ToonDB!"

# Clean up
db.close()
```

## Use Case Reference

- **Key-Value Cache**: Examples 1, 7
- **User/Entity Management**: Example 5
- **Document Storage**: Example 6
- **Multi-process Access**: Example 8
- **Batch Operations**: Example 4
- **Hierarchical Data**: Example 3
