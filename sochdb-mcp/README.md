# SochDB MCP Server

Minimal Model Context Protocol (MCP) server for SochDB - an AI-native database.

## Overview

This is a thin adapter layer that exposes SochDB's AI-native features via the MCP protocol, allowing LLM clients like Claude Desktop, Cursor, and ChatGPT to interact with your database.

```
MCP Client (Claude, Cursor, etc.)
     │
     │ JSON-RPC over stdio
     ▼
┌─────────────────────────────────┐
│  sochdb-mcp                     │
│  - Stdio framing (~50 lines)    │
│  - JSON-RPC dispatch            │
│  - MCP methods                  │
└─────────────────────────────────┘
     │
     │ Direct Rust calls
     ▼
┌─────────────────────────────────┐
│  SochDB                         │
│  - Context queries → AI context │
│  - TOON format → token savings  │
│  - Path-based access            │
└─────────────────────────────────┘
```

## Installation

```bash
# Build from source
cargo build --release --package sochdb-mcp

# Binary is at target/release/sochdb-mcp
```

## Usage

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sochdb": {
      "command": "/path/to/sochdb-mcp",
      "args": ["--db", "/path/to/your/database"]
    }
  }
}
```

### Cursor

Add to your `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "sochdb": {
      "command": "/path/to/sochdb-mcp",
      "args": ["--db", "./data"]
    }
  }
}
```

### Command Line Testing

```bash
# Test the server manually
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | sochdb-mcp --db ./test_data
```

## Available Tools

### `sochdb.context_query`

Fetch AI-optimized context with token budgeting. This is SochDB's killer feature for LLMs.

```json
{
  "name": "sochdb.context_query",
  "arguments": {
    "sections": [
      {
        "name": "recent_messages",
        "priority": 10,
        "kind": "last",
        "table": "messages",
        "top_k": 5
      },
      {
        "name": "user_profile",
        "priority": 20,
        "kind": "get",
        "path": "users/current"
      }
    ],
    "token_budget": 2000
  }
}
```

### `sochdb.query`

Execute SochQL queries directly.

```json
{
  "name": "sochdb.query",
  "arguments": {
    "query": "SELECT * FROM users WHERE active = true"
  }
}
```

### `sochdb.get` / `sochdb.put` / `sochdb.delete`

Path-based CRUD operations with O(|path|) complexity.

```json
{
  "name": "sochdb.get",
  "arguments": {
    "path": "users/123/profile/name"
  }
}
```

### `sochdb.list_tables` / `sochdb.describe`

Schema introspection tools.

## Architecture

The server is intentionally minimal:

- **~100 lines** for stdio framing
- **~100 lines** for JSON-RPC types
- **~200 lines** for MCP protocol handling
- **~200 lines** for tool execution

No external MCP framework. Just `serde_json` + SochDB.

## Protocol Support

- Protocol version: `2024-11-05`
- Transport: stdio (stdin/stdout with Content-Length framing)
- Capabilities:
  - `tools`: list and call
  - `resources`: list and read (exposes tables)

## Why This Approach?

1. **SochDB is already AI-native**: Operations, SochQL, context engine, and token budgets are built-in. The MCP layer just exposes them.

2. **Thin is good**: The MCP server doesn't need to understand schemas or indexing. It translates between JSON-RPC and SochDB calls.

3. **No framework bloat**: A few hundred lines of code vs. pulling in a heavy MCP SDK.

## Logging

All logs go to stderr (required for stdio transport - stdout is for protocol messages only).

Set `RUST_LOG=sochdb_mcp=debug` for verbose output.

## License

AGPL-3.0-or-later
