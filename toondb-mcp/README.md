# ToonDB MCP Server

Minimal Model Context Protocol (MCP) server for ToonDB - an AI-native database.

## Overview

This is a thin adapter layer that exposes ToonDB's AI-native features via the MCP protocol, allowing LLM clients like Claude Desktop, Cursor, and ChatGPT to interact with your database.

```
MCP Client (Claude, Cursor, etc.)
     │
     │ JSON-RPC over stdio
     ▼
┌─────────────────────────────────┐
│  toondb-mcp                     │
│  - Stdio framing (~50 lines)    │
│  - JSON-RPC dispatch            │
│  - MCP methods                  │
└─────────────────────────────────┘
     │
     │ Direct Rust calls
     ▼
┌─────────────────────────────────┐
│  ToonDB                         │
│  - Context queries → AI context │
│  - TOON format → token savings  │
│  - Path-based access            │
└─────────────────────────────────┘
```

## Installation

```bash
# Build from source
cargo build --release --package toondb-mcp

# Binary is at target/release/toondb-mcp
```

## Usage

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "toondb": {
      "command": "/path/to/toondb-mcp",
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
    "toondb": {
      "command": "/path/to/toondb-mcp",
      "args": ["--db", "./data"]
    }
  }
}
```

### Command Line Testing

```bash
# Test the server manually
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | toondb-mcp --db ./test_data
```

## Available Tools

### `toondb.context_query`

Fetch AI-optimized context with token budgeting. This is ToonDB's killer feature for LLMs.

```json
{
  "name": "toondb.context_query",
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

### `toondb.query`

Execute ToonQL queries directly.

```json
{
  "name": "toondb.query",
  "arguments": {
    "query": "SELECT * FROM users WHERE active = true"
  }
}
```

### `toondb.get` / `toondb.put` / `toondb.delete`

Path-based CRUD operations with O(|path|) complexity.

```json
{
  "name": "toondb.get",
  "arguments": {
    "path": "users/123/profile/name"
  }
}
```

### `toondb.list_tables` / `toondb.describe`

Schema introspection tools.

## Architecture

The server is intentionally minimal:

- **~100 lines** for stdio framing
- **~100 lines** for JSON-RPC types
- **~200 lines** for MCP protocol handling
- **~200 lines** for tool execution

No external MCP framework. Just `serde_json` + ToonDB.

## Protocol Support

- Protocol version: `2024-11-05`
- Transport: stdio (stdin/stdout with Content-Length framing)
- Capabilities:
  - `tools`: list and call
  - `resources`: list and read (exposes tables)

## Why This Approach?

1. **ToonDB is already AI-native**: Operations, ToonQL, context engine, and token budgets are built-in. The MCP layer just exposes them.

2. **Thin is good**: The MCP server doesn't need to understand schemas or indexing. It translates between JSON-RPC and ToonDB calls.

3. **No framework bloat**: A few hundred lines of code vs. pulling in a heavy MCP SDK.

## Logging

All logs go to stderr (required for stdio transport - stdout is for protocol messages only).

Set `RUST_LOG=toondb_mcp=debug` for verbose output.

## License

Apache-2.0
