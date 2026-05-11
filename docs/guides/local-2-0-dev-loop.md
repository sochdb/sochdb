# Local 2.0 Dev Loop

This is the smallest reproducible local workflow for working on the SochDB 2.0
client/server path.

It is intended for contributors who want to:

- start the local gRPC server
- generate Python client stubs
- run a minimal vector insert/search flow

## What This Covers

This workflow exercises the `2.0` server/client path:

- `sochdb-grpc-server`
- generated Python gRPC stubs
- vector index creation
- batch insert
- k-NN search

It is different from the embedded/local `Database.open(...)` path.

## Prerequisites

- Rust toolchain
- `protoc`
- Python 3
- `grpcio` and `grpcio-tools`

On macOS:

```bash
brew install protobuf
```

Create a temporary Python venv for SDK codegen/testing:

```bash
python3 -m venv /tmp/sochdb2-sdk-venv
source /tmp/sochdb2-sdk-venv/bin/activate
pip install grpcio grpcio-tools
```

## 1. Start the Local gRPC Server

From the repo root:

```bash
cargo run -p sochdb-grpc --bin sochdb-grpc-server -- \
  --host 127.0.0.1 \
  --port 50051 \
  --metrics-port 0 \
  --ws-port 0 \
  --pg-port 0
```

This keeps the first local loop focused on the gRPC vector path only.

## 2. Generate Python Stubs

In another shell:

```bash
source /tmp/sochdb2-sdk-venv/bin/activate
cd sochdb-sdk
./generate.sh python
```

Generated files land in:

- `sochdb-sdk/python/sochdb_sdk/generated/sochdb_pb2.py`
- `sochdb-sdk/python/sochdb_sdk/generated/sochdb_pb2_grpc.py`

## 3. Run the Minimal Quickstart

From the repo root:

```bash
source /tmp/sochdb2-sdk-venv/bin/activate
python examples/python/07_grpc_vector_quickstart.py
```

Expected output:

```text
create_index: True ok
insert_batch: 3 ok
search: ok
  id=1 distance=0.0000
  id=3 distance=1.0000
```

## Files Involved

- [`examples/python/07_grpc_vector_quickstart.py`](/private/tmp/sochdb-2.0/examples/python/07_grpc_vector_quickstart.py)
- [`sochdb-sdk/generate.sh`](/private/tmp/sochdb-2.0/sochdb-sdk/generate.sh)
- [`sochdb-sdk/python/sochdb_sdk/client.py`](/private/tmp/sochdb-2.0/sochdb-sdk/python/sochdb_sdk/client.py)
- [`sochdb-grpc/proto/sochdb.proto`](/private/tmp/sochdb-2.0/sochdb-grpc/proto/sochdb.proto)

## Current Rough Edges

- The generated Python stubs are the most reliable path right now.
- The thin `sochdb_sdk` wrapper exists, but it should be checked against the
  current proto/service surface before treating it as the default example path.
- This loop assumes a contributor workflow, not a polished packaged install
  story yet.

## Best Next Steps

After this loop works locally:

1. Align the thin Python SDK wrapper with the current proto surface.
2. Add a small server-side benchmark runner using this local setup.
3. Write a short note clarifying `embedded/local` vs `2.0 client/server`.
