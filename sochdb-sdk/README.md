# SochDB SDK Codegen

Auto-generates thin client SDKs from `sochdb.proto` for Python, Go, and Node.js.

## Architecture

SochDB uses a **"Thick Server / Thin Client"** architecture. All business logic
runs in the Rust server — SDKs are thin RPC wrappers (~200 LOC each) that provide
idiomatic access to the 12 gRPC services.

## Quick Start

```bash
# Generate all SDKs
./generate.sh all

# Generate for a specific language
./generate.sh python
./generate.sh go
./generate.sh node
```

## Prerequisites

| Language | Tool | Install |
|----------|------|---------|
| Python | grpcio-tools | `pip install grpcio-tools` |
| Go | protoc-gen-go | `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest` |
| Node.js | ts-proto | `npm install -g ts-proto` |
| All | protoc | [protobuf releases](https://github.com/protocolbuffers/protobuf/releases) |

## SDK Structure

```
sochdb-sdk/
├── generate.sh          # Codegen script
├── python/
│   ├── sochdb_sdk/
│   │   ├── __init__.py
│   │   ├── client.py     # Thin ergonomic wrapper
│   │   └── generated/    # Auto-generated protobuf stubs
│   └── setup.py
├── go/
│   ├── client.go          # Thin ergonomic wrapper
│   └── sochdbv1/          # Auto-generated protobuf stubs
└── node/
    ├── src/
    │   ├── client.ts      # Thin ergonomic wrapper
    │   └── generated/     # Auto-generated protobuf stubs
    └── package.json
```

## Services

The SDK provides access to all 12 SochDB gRPC services:

| Service | Description |
|---------|-------------|
| VectorIndexService | HNSW vector similarity search |
| GraphService | Graph overlay for agent memory |
| PolicyService | Policy evaluation |
| ContextService | LLM context assembly |
| CollectionService | Collection management |
| NamespaceService | Multi-tenant namespaces |
| SemanticCacheService | Semantic caching |
| TraceService | Distributed tracing |
| CheckpointService | State snapshots |
| McpService | MCP tool routing |
| KvService | Key-value operations |
| SubscriptionService | Real-time change notifications |

## Alternative Transports

In addition to gRPC, SochDB also supports:

- **WebSocket** (`ws://host:8080/`) — JSON protocol for browsers
- **PostgreSQL wire protocol** (`postgresql://host:5433/sochdb`) — psql/ORM compatible
