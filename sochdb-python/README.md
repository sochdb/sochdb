# SochDB Python SDK

Python bindings for SochDB's embedded database and native vector capabilities.

## 10-Minute Quickstart

### Option 1: Use the published package

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install sochdb
```

### Option 2: Work from this monorepo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install maturin
cd sochdb-python
maturin develop --release
```

### Verify the SDK

```python
from sochdb import Database

db = Database.open("./quickstart_db")

with db.transaction() as txn:
    db.put(b"users/alice", b'{"name":"Alice"}', txn.id)

print("alice:", db.get(b"users/alice"))
db.close()
```

Expected output:

```text
alice: b'{"name":"Alice"}'
```

## Notes

- Python 3.9+ is required.
- On macOS, Python architecture must match the native library architecture:
  - Apple Silicon Python should use `arm64`
  - Intel / Rosetta Python should use `x86_64`
- Mixed-architecture setups can fail at native library load time.
