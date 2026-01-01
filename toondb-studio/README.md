# ToonDB Studio

**ToonDB Studio** is a professional desktop database administration tool for ToonDB, built with Tauri 2.0. It provides a modern, intuitive interface for managing ToonDB databases.

Similar to:
- **DBeaver** (SQL databases)
- **MongoDB Compass** (MongoDB)
- **TablePlus** (Multi-database)
- **Redis Insight** (Redis)
- **pgAdmin** (PostgreSQL)

## Features

### 🗂️ Multi-Connection Support
- Manage multiple ToonDB instances
- Local file, Unix socket, TCP/IP, and embedded connections
- Connection persistence and auto-reconnect

### ⌨️ SQL + ToonQL Query Editor
- Full SQL support with syntax highlighting
- ToonQL for vector search and key-value operations
- Query history and saved queries
- Multiple result views (Table, JSON, TOON)

### 🧠 Vector Similarity Search
- Visual HNSW similarity search interface
- Text-to-embedding conversion
- Cosine, Euclidean, and Dot Product distance metrics
- Similarity score display

### 📋 Table Browser
- Schema view with columns, types, and constraints
- Data grid with pagination
- Inline cell editing
- Export (CSV, JSON, TOON)

### 📁 Path Browser
- Hierarchical path navigation for KV store
- Key-value viewer
- Path prefix scanning

### 📊 Real-Time Monitoring
- MVCC status
- WAL and memtable metrics
- Active transactions and snapshots
- Garbage version count

### ⚡ Admin Tools
- Force checkpoint
- Run garbage collection
- Analyze tables
- Compact SST files

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `⌘ + Enter` | Execute query |
| `⌘ + S` | Save query |
| `⌘ + N` | New query tab |
| `⌘ + W` | Close tab |
| `⌘ + /` | Toggle comment |
| `⌘ + ⇧ + F` | Format query |
| `⌘ + ⇧ + E` | Focus explorer |
| `⌘ + ⇧ + R` | Refresh schema |
| `F5` | Execute query |
| `Esc` | Cancel query |

## Development

### Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [Rust](https://rustup.rs/) 1.70+
- [Tauri CLI](https://tauri.app/v2/guides/getting-started/setup) v2

### Setup

```bash
# Install dependencies
npm install

# Run in development mode
npm run tauri dev

# Build for production
npm run tauri build
```

### Project Structure

```
toondb-studio/
├── src-tauri/              # Rust backend
│   ├── src/
│   │   ├── commands/       # Tauri IPC commands
│   │   │   ├── admin.rs    # Checkpoint, GC, etc.
│   │   │   ├── query.rs    # Query execution
│   │   │   ├── schema.rs   # Schema introspection
│   │   │   └── vector.rs   # Vector search
│   │   ├── connection/     # Connection management
│   │   └── parser/         # SQL/ToonQL parsers
│   └── Cargo.toml
├── src/                    # React frontend
│   ├── components/
│   ├── hooks/
│   │   ├── useToonDB.ts    # ToonDB API hook
│   │   ├── useQuery.ts     # Query state
│   │   └── useSchema.ts    # Schema state
│   ├── stores/
│   └── App.tsx
└── package.json
```

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Schema load | <100ms | Cache after first load |
| Simple query | <50ms | Point lookups |
| Full scan | <500ms | Up to 100K rows |
| Vector search | <100ms | HNSW with ef_search=50 |
| Cell update | <20ms | Inline editing |
| Export 10K rows | <2s | CSV/JSON |

## License

Apache-2.0

## Contributing

Contributions are welcome! Please read our [Contributing Guide](../CONTRIBUTING.md) for details.
