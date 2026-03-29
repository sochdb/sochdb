---
sidebar_position: 3
---

# Python Install Matrix

Use this page to choose the right SochDB Python installation path for your environment.

The short version:

- Use `pip install sochdb` for the published package
- Use `sochdb-python/` + `maturin develop --release` when working from this monorepo
- On macOS, your Python architecture must match the native extension architecture

---

## Recommended Paths

| Use case | Recommended path | Status |
|----------|------------------|--------|
| Try SochDB quickly as a Python user | `pip install sochdb` | Recommended |
| Work on the Python SDK from source | `cd sochdb-python && maturin develop --release` | Recommended |
| Apple Silicon Mac with native Python | `pip install sochdb` in an `arm64` Python env | Validated |
| Apple Silicon Mac with Rosetta / Intel Python | Prefer switching to native `arm64` Python first | Use carefully |

---

## Validated Python Paths

### 1. Published package on native macOS Apple Silicon

This path was validated successfully with:

- Python 3.9
- `arm64`
- clean virtual environment
- `pip install sochdb`

Example:

```bash
/usr/bin/python3 -m venv /tmp/sochdb-arm64-test
source /tmp/sochdb-arm64-test/bin/activate
python -m pip install --upgrade pip
pip install sochdb
```

Verify architecture:

```bash
python - <<'PY'
import platform, sys
print("executable:", sys.executable)
print("machine:", platform.machine())
PY
```

Expected on Apple Silicon:

```text
machine: arm64
```

### 2. Monorepo source build

Use this when you are developing inside this repo:

```bash
cd sochdb-python
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

This installs the editable SochDB Python package into that environment.

---

## macOS Architecture Guide

On Apple Silicon Macs, the machine is typically `arm64`, but you can still end up
running an `x86_64` Python under Rosetta.

That matters because SochDB's Python package uses native Rust extensions.

### Good match

- `arm64` Python
- `arm64` native extension

### Bad match

- `x86_64` Python
- `arm64` native extension

This can fail at import/load time with errors like:

- incompatible architecture
- `dlopen(...) mach-o file, but is an incompatible architecture`

Check your Python architecture:

```bash
python - <<'PY'
import platform, sys
print("executable:", sys.executable)
print("machine:", platform.machine())
PY
```

If you are on an Apple Silicon Mac, prefer:

- native `arm64` Python
- native `arm64` virtual environments

Avoid mixing:

- Intel/Rosetta Python
- native Apple Silicon builds

---

## Quick Troubleshooting

### `ModuleNotFoundError: No module named 'sochdb'`

Install or upgrade the published package:

```bash
pip install --upgrade sochdb
```

### Native library import / `dlopen` architecture error on macOS

Check Python architecture first:

```bash
python - <<'PY'
import platform
print(platform.machine())
PY
```

If this prints `x86_64` on an Apple Silicon Mac, switch to a native `arm64` Python environment.

### Working from source but imports still look wrong

Make sure you built from the monorepo Python package:

```bash
cd sochdb-python
maturin develop --release
```

Then run your demo or script from that same Python environment.

---

## Recommended Verification Script

```python
from sochdb import Database

db = Database.open("./verify_db")

with db.transaction() as txn:
    db.put(b"test/key", b"value", txn.id)

value = db.get(b"test/key")
assert value == b"value"
print("SochDB Python path works")

db.close()
```

---

## Related Docs

- [Installation](/getting-started/installation)
- [Quick Start](/getting-started/quickstart)
- [Python SDK Guide](/guides/python-sdk)
