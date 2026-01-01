# ToonDB MCP Integration Guide

Connect ToonDB to your favorite AI assistant using the **Model Context Protocol (MCP)**.

## Quick Start

```bash
# Build the MCP server
cargo build --release -p toondb-mcp

# Test it works
./target/release/toondb-mcp --db ./my_database --help
```

---

## Claude Desktop

**Config location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Steps:**
1. Open Claude Desktop → Settings → Developer → Edit Config
2. Add this configuration:

```json
{
  "mcpServers": {
    "toondb": {
      "command": "/path/to/toondb-mcp",
      "args": ["--db", "/path/to/your/database"],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

3. Restart Claude Desktop
4. Look for the 🔧 tools icon to confirm ToonDB is connected

---

## Cursor IDE

**Config location:** `~/.cursor/mcp.json`

**Steps:**
1. Open Cursor → Settings (`Cmd+,`) → Cursor Settings → Tools & MCP
2. Click "Add new global MCP server"
3. Add this configuration:

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

4. Restart Cursor

---

## VS Code + Continue.dev

**Config location:** `~/.continue/config.json`

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "/path/to/toondb-mcp",
          "args": ["--db", "/path/to/your/database"]
        }
      }
    ]
  }
}
```

---

## Zed Editor

**Config location:** `~/.config/zed/settings.json`

```json
{
  "context_servers": {
    "toondb": {
      "command": {
        "path": "/path/to/toondb-mcp",
        "args": ["--db", "/path/to/your/database"]
      }
    }
  }
}
```

---

## Generic MCP Client

For any MCP-compatible client, use:

| Setting | Value |
|---------|-------|
| Transport | stdio |
| Command | `/path/to/toondb-mcp` |
| Arguments | `--db /path/to/database` |
| Env (optional) | `RUST_LOG=info` |

---

## Available Tools

Once connected, these ToonDB tools become available:

| Tool | Description |
|------|-------------|
| `put` | Store a key-value pair |
| `get` | Retrieve a value by key |
| `delete` | Remove a key |
| `scan` | List keys with optional prefix |
| `query` | Execute ToonQL queries |
| `create_table` | Create a new table |
| `insert` | Insert rows into a table |
| `select` | Query table data |
| `create_episode` | Create conversation episode |
| `add_turn` | Add message to episode |
| `search_episodes` | Search conversation history |

---

## Troubleshooting

**"Server not found"**
- Use absolute path to `toondb-mcp` binary
- Ensure `cargo build --release` completed successfully

**"Database connection failed"**
- Check the database path exists
- Ensure you have read/write permissions

**"No tools showing"**
- Restart the AI application after config change
- Check logs: set `RUST_LOG=debug` in env
