# ToonDB Developer Usage Guide

> **AI-Native Database for Agentic Applications**
> 
> A comprehensive guide for integrating ToonDB into modern LLM-powered systems.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Integration Patterns](#integration-patterns)
4. [MCP Tools Reference](#mcp-tools-reference)
5. [Python SDK Usage](#python-sdk-usage)
6. [Performance Optimization](#performance-optimization)
7. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
# Build from source
git clone https://github.com/toondb/toondb
cd toondb
cargo build --release

# Install Python SDK
pip install toondb-client

# Set library path (required for FFI)
export TOONDB_LIB_PATH=/path/to/toondb/target/release
```

### Your First Database

```python
from toondb import Database

# Open database (creates if not exists)
db = Database.open("./my_agent_data")

# Store data using path-based access
db.put_path("users/alice/preferences", b'{"theme": "dark", "language": "en"}')
db.put_path("users/alice/history/001", b'{"query": "What is RAG?", "ts": 1234567890}')

# Retrieve with O(|path|) complexity
prefs = db.get_path("users/alice/preferences")

# Transaction support
with db.transaction() as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    # Auto-commits on exit, rolls back on exception

db.close()
```

### MCP Server Setup (Claude Desktop / Cursor)

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "toondb": {
      "command": "toondb-mcp",
      "args": ["--db", "./agent_memory"]
    }
  }
}
```

---

## Core Concepts

### 1. Path-Based Data Model

ToonDB uses hierarchical paths instead of tables, enabling O(|path|) lookups:

```
/users/{user_id}/profile
/users/{user_id}/preferences
/users/{user_id}/history/{event_id}
/episodes/{episode_id}/events/{event_id}
/entities/{entity_type}/{entity_id}/facts
```

**Why This Matters for Agents:**
- Direct access without query parsing
- Natural hierarchical organization (user → session → messages)
- Prefix scans for related data (`/users/alice/*`)

```python
# Path-based access patterns
db.get_path("users/alice/profile")           # Single lookup
db.scan(b"users/alice/history/", b"users/alice/history/~")  # Range scan
```

### 2. TOON Format (40-66% Token Savings)

ToonDB's wire format is optimized for LLM consumption:

```
# JSON (7,500 tokens for 100 rows × 5 fields)
{"field1": "value1", "field2": "value2", ...}

# TOON (2,550 tokens - 66% savings)
table[100]{f1,f2,f3,f4,f5}:
  v1,v2,v3,v4,v5
  v1,v2,v3,v4,v5
  ...
```

**Request TOON format in queries:**

```python
result = toondb.context_query(
    sections=[...],
    format="toon"  # Default, 40-66% fewer tokens
)
```

### 3. Context Query: The Killer Feature

Assemble LLM context in a single call with automatic token budgeting:

```python
context = toondb.context_query(
    sections=[
        # Priority 0 (highest): Always include
        {
            "name": "system",
            "kind": "literal",
            "text": "You are a helpful assistant.",
            "priority": 0
        },
        # Priority 1: User context
        {
            "name": "user_prefs",
            "kind": "get",
            "path": "/users/alice/preferences",
            "priority": 1
        },
        # Priority 2: Recent history
        {
            "name": "history",
            "kind": "last",
            "table": "messages",
            "top_k": 10,
            "priority": 2
        },
        # Priority 3: Semantic search (lowest priority, truncated first)
        {
            "name": "knowledge",
            "kind": "search",
            "query": "How to configure authentication?",
            "top_k": 5,
            "priority": 3
        }
    ],
    token_budget=4096,
    truncation="tail_drop"  # Drop lowest priority first
)
```

**Section Types:**

| Kind | Description | Required Fields |
|------|-------------|-----------------|
| `literal` | Static text | `text` |
| `get` | Fetch single path | `path` |
| `last` | Recent rows from table | `table`, `top_k` |
| `search` | Vector similarity search | `query`, `top_k` |

**Truncation Strategies:**

| Strategy | Behavior |
|----------|----------|
| `tail_drop` | Remove lowest-priority sections first |
| `head_drop` | Remove highest-priority sections first |
| `proportional` | Trim all sections proportionally |

---

## Integration Patterns

### LangGraph Integration

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
import subprocess
import json

class ToonDBMCPClient:
    """MCP client for ToonDB integration with LangGraph"""
    
    def __init__(self, db_path: str):
        self.proc = subprocess.Popen(
            ["toondb-mcp", "--db", db_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        self._initialize()
    
    def _send_request(self, method: str, params: dict) -> dict:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        # Send with Content-Length header (LSP framing)
        body = json.dumps(request)
        message = f"Content-Length: {len(body)}\r\n\r\n{body}"
        self.proc.stdin.write(message)
        self.proc.stdin.flush()
        
        # Read response
        headers = {}
        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                break
            key, value = line.split(": ", 1)
            headers[key] = value
        
        content_length = int(headers.get("Content-Length", 0))
        response = self.proc.stdout.read(content_length)
        return json.loads(response)
    
    def _initialize(self):
        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "langgraph-agent"}
        })
    
    def context_query(self, sections: list, token_budget: int = 4000) -> str:
        """Build context with automatic token budgeting"""
        result = self._send_request("tools/call", {
            "name": "toondb_context_query",
            "arguments": {
                "sections": sections,
                "token_budget": token_budget,
                "format": "toon"
            }
        })
        return result.get("result", {}).get("content", [{}])[0].get("text", "")
    
    def memory_search(self, query: str, k: int = 5) -> list:
        """Semantic search over episodes"""
        result = self._send_request("tools/call", {
            "name": "memory_search_episodes",
            "arguments": {"query": query, "k": k}
        })
        return json.loads(result.get("result", {}).get("content", [{}])[0].get("text", "[]"))


# LangGraph State
class AgentState(TypedDict):
    messages: list
    context: str
    user_id: str


# Build the graph
def build_agent_graph(toondb: ToonDBMCPClient):
    
    def retrieve_context(state: AgentState) -> AgentState:
        """Retrieve relevant context from ToonDB"""
        user_query = state["messages"][-1]["content"]
        
        context = toondb.context_query(
            sections=[
                {"name": "user", "kind": "get", "path": f"/users/{state['user_id']}/preferences"},
                {"name": "history", "kind": "last", "table": "messages", "top_k": 5},
                {"name": "knowledge", "kind": "search", "query": user_query, "top_k": 3}
            ],
            token_budget=3000
        )
        
        return {**state, "context": context}
    
    def generate_response(state: AgentState) -> AgentState:
        """Generate response using retrieved context"""
        # Your LLM call here, using state["context"]
        pass
    
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_response)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.set_entry_point("retrieve")
    
    return graph.compile()


# Usage
toondb = ToonDBMCPClient("./agent_memory")
agent = build_agent_graph(toondb)
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I reset my password?"}],
    "user_id": "alice"
})
```

### CrewAI Integration

```python
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
import subprocess
import json

class ToonDBContextTool(BaseTool):
    name: str = "toondb_context"
    description: str = "Retrieve relevant context from the knowledge base with automatic token budgeting"
    
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self._start_server()
    
    def _start_server(self):
        self.proc = subprocess.Popen(
            ["toondb-mcp", "--db", self.db_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        # Initialize MCP...
    
    def _run(self, query: str, token_budget: int = 2000) -> str:
        """Execute context retrieval"""
        sections = [
            {"name": "knowledge", "kind": "search", "query": query, "top_k": 5},
            {"name": "recent", "kind": "last", "table": "interactions", "top_k": 3}
        ]
        # MCP call to toondb_context_query
        return self._call_mcp("toondb_context_query", {
            "sections": sections,
            "token_budget": token_budget
        })


class ToonDBMemoryTool(BaseTool):
    name: str = "toondb_memory"
    description: str = "Store and retrieve agent memories"
    
    def _run(self, action: str, **kwargs) -> str:
        if action == "store":
            return self._store_memory(kwargs["key"], kwargs["value"])
        elif action == "recall":
            return self._recall_memory(kwargs["key"])


# Create agents with ToonDB tools
researcher = Agent(
    role="Research Analyst",
    goal="Find relevant information from the knowledge base",
    tools=[ToonDBContextTool("./research_db")],
    verbose=True
)

writer = Agent(
    role="Content Writer", 
    goal="Write reports based on research",
    tools=[ToonDBMemoryTool("./research_db")],
    verbose=True
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[
        Task(description="Research the topic: {topic}", agent=researcher),
        Task(description="Write a summary report", agent=writer)
    ]
)
```

### OpenAI Function Calling Bridge

```python
import openai
import json

class ToonDBOpenAIBridge:
    """Bridge ToonDB MCP tools to OpenAI function calling format"""
    
    def __init__(self, db_path: str):
        self.toondb = ToonDBMCPClient(db_path)
        self.client = openai.OpenAI()
    
    def get_tools_schema(self) -> list:
        """Convert MCP tools to OpenAI function format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search the knowledge base for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_user_context",
                    "description": "Get user preferences and history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "include_history": {"type": "boolean", "default": True}
                        },
                        "required": ["user_id"]
                    }
                }
            }
        ]
    
    def execute_function(self, name: str, arguments: dict) -> str:
        """Execute a function call via ToonDB"""
        if name == "search_knowledge":
            return self.toondb.context_query(
                sections=[{
                    "name": "results",
                    "kind": "search",
                    "query": arguments["query"],
                    "top_k": arguments.get("max_results", 5)
                }],
                token_budget=2000
            )
        elif name == "get_user_context":
            sections = [
                {"name": "prefs", "kind": "get", "path": f"/users/{arguments['user_id']}/preferences"}
            ]
            if arguments.get("include_history", True):
                sections.append({
                    "name": "history",
                    "kind": "last",
                    "table": f"users/{arguments['user_id']}/history",
                    "top_k": 10
                })
            return self.toondb.context_query(sections=sections, token_budget=1500)
    
    def chat(self, messages: list) -> str:
        """Chat with function calling support"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.get_tools_schema()
        )
        
        # Handle tool calls
        if response.choices[0].message.tool_calls:
            tool_results = []
            for tool_call in response.choices[0].message.tool_calls:
                result = self.execute_function(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # Continue conversation with tool results
            messages.append(response.choices[0].message)
            messages.extend(tool_results)
            return self.chat(messages)
        
        return response.choices[0].message.content
```

---

## MCP Tools Reference

### Core Database Tools

#### `toondb_context_query`

Fetch AI-optimized context with token budgeting.

```json
{
  "name": "toondb_context_query",
  "arguments": {
    "sections": [
      {"name": "system", "kind": "literal", "text": "..."},
      {"name": "user", "kind": "get", "path": "/users/123"},
      {"name": "history", "kind": "last", "table": "messages", "top_k": 10},
      {"name": "knowledge", "kind": "search", "query": "...", "top_k": 5}
    ],
    "token_budget": 4096,
    "format": "toon",
    "truncation": "tail_drop"
  }
}
```

#### `toondb_query`

Execute ToonQL queries with TOON output.

```json
{
  "name": "toondb_query",
  "arguments": {
    "query": "SELECT id, name, score FROM users WHERE score > 80 ORDER BY score DESC",
    "format": "toon",
    "limit": 100
  }
}
```

#### `toondb_get` / `toondb_put` / `toondb_delete`

Path-based CRUD operations.

```json
// GET
{"name": "toondb_get", "arguments": {"path": "/users/alice/profile"}}

// PUT
{"name": "toondb_put", "arguments": {"path": "/users/alice/profile", "value": {"name": "Alice"}}}

// DELETE
{"name": "toondb_delete", "arguments": {"path": "/users/alice/temp_data"}}
```

### Memory Tools

#### `memory_search_episodes`

Vector search over conversation/task episodes.

```json
{
  "name": "memory_search_episodes",
  "arguments": {
    "query": "debugging authentication issues",
    "k": 5,
    "episode_type": "conversation",
    "entity_id": "user_alice"
  }
}
```

#### `memory_get_episode_timeline`

Retrieve event timeline for an episode.

```json
{
  "name": "memory_get_episode_timeline",
  "arguments": {
    "episode_id": "ep_12345",
    "max_events": 50,
    "role": "assistant",
    "include_metrics": true
  }
}
```

#### `memory_build_context`

Automatic context assembly based on goal.

```json
{
  "name": "memory_build_context",
  "arguments": {
    "goal": "Help user debug their Python code",
    "token_budget": 4000,
    "session_id": "sess_abc",
    "entity_ids": ["user_alice", "project_myapp"],
    "include_schema": false
  }
}
```

### Log Tools

#### `logs_tail`

Get recent rows from a log table.

```json
{
  "name": "logs_tail",
  "arguments": {
    "table": "agent_actions",
    "limit": 20,
    "where": {"level": "error"},
    "columns": ["timestamp", "action", "error_message"]
  }
}
```

---

## Python SDK Usage

### Basic Operations

```python
from toondb import Database

# Open database
db = Database.open("./data")

# Simple key-value operations
db.put(b"key", b"value")
value = db.get(b"key")  # Returns bytes or None
db.delete(b"key")

# Path-based operations (recommended)
db.put_path("users/alice/email", b"alice@example.com")
email = db.get_path("users/alice/email")

# Transactions
with db.transaction() as txn:
    txn.put(b"account/balance", b"1000")
    txn.put(b"account/updated_at", b"2024-01-01")
    # Auto-commits on successful exit

# Scanning
for key, value in db.scan(b"users/", b"users/~"):
    print(f"{key}: {value}")

# Checkpoint (force WAL flush)
lsn = db.checkpoint()

# Statistics
stats = db.stats()
print(f"Memtable size: {stats['memtable_size_bytes']}")
print(f"Active transactions: {stats['active_transactions']}")

db.close()
```

### Vector Operations

```python
from toondb import VectorIndex
import numpy as np

# Create index
index = VectorIndex(
    dimension=768,
    max_connections=16,      # HNSW M parameter
    ef_construction=100      # Build-time search width
)

# Insert vectors
ids = [1, 2, 3, 4, 5]
vectors = np.random.randn(5, 768).astype(np.float32)
index.add_batch(ids, vectors)

# Search
query = np.random.randn(768).astype(np.float32)
results = index.search(query, k=10, ef_search=50)

for id, distance in results:
    print(f"ID: {id}, Distance: {distance:.4f}")

# Persistence
index.save("./vectors.hnsw")
loaded_index = VectorIndex.load("./vectors.hnsw")
```

### Bulk Operations

```python
from toondb.bulk import bulk_build_index
import numpy as np

# Generate embeddings
vectors = np.random.randn(100000, 768).astype(np.float32)

# Build index efficiently (bypasses Python FFI overhead)
stats = bulk_build_index(
    vectors,
    output="./large_index.hnsw",
    m=16,
    ef_construction=200,
    quiet=False  # Show progress
)

print(f"Built index with {stats.vectors} vectors in {stats.elapsed_secs:.1f}s")
print(f"Throughput: {stats.rate:.0f} vectors/sec")
```

---

## Performance Optimization

### Token Budget Tuning

```python
# Conservative: Prioritize completeness
context = toondb.context_query(
    sections=[...],
    token_budget=6000,        # Higher budget
    truncation="proportional" # Preserve all sections
)

# Aggressive: Prioritize speed/cost
context = toondb.context_query(
    sections=[...],
    token_budget=2000,        # Tight budget
    truncation="tail_drop"    # Drop low-priority first
)
```

### Batch Operations

```python
# SLOW: Individual puts
for i in range(10000):
    db.put(f"key_{i}".encode(), f"value_{i}".encode())

# FAST: Transactional batch
with db.transaction() as txn:
    for i in range(10000):
        txn.put(f"key_{i}".encode(), f"value_{i}".encode())
# Single commit, ~10x faster
```

### Connection Pooling (Server Mode)

```python
from concurrent.futures import ThreadPoolExecutor
from toondb import Database

# Shared database instance (thread-safe)
db = Database.open("./data")

def process_request(user_id: str):
    with db.transaction() as txn:
        data = txn.get(f"users/{user_id}".encode())
        # Process...
        return data

# Concurrent processing
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_request, user_ids))
```

### Vector Search Tuning

```python
# Build-time: Higher ef_construction = better recall, slower build
index = VectorIndex(dimension=768, max_connections=16, ef_construction=200)

# Query-time: Higher ef_search = better recall, slower search
results = index.search(query, k=10, ef_search=100)

# Recommended settings by use case:
# - Real-time chat: ef_construction=100, ef_search=50
# - Batch processing: ef_construction=200, ef_search=200
# - Maximum recall: ef_construction=400, ef_search=400
```

---

## Best Practices

### 1. Path Design

```python
# GOOD: Hierarchical, scannable
"/users/{user_id}/profile"
"/users/{user_id}/sessions/{session_id}/messages/{msg_id}"
"/projects/{project_id}/documents/{doc_id}"

# BAD: Flat, requires full scans
"/user_alice_profile"
"/user_alice_session_123_message_456"
```

### 2. Token Budget Strategy

```python
# Allocate budget based on model context window
MODEL_CONTEXT = 8192
SYSTEM_PROMPT_TOKENS = 500
OUTPUT_RESERVE = 1000
AVAILABLE_CONTEXT = MODEL_CONTEXT - SYSTEM_PROMPT_TOKENS - OUTPUT_RESERVE  # 6692

context = toondb.context_query(
    sections=[
        {"name": "critical", "priority": 0, ...},   # Always included
        {"name": "important", "priority": 1, ...},  # Included if room
        {"name": "nice_to_have", "priority": 2, ...} # Truncated first
    ],
    token_budget=AVAILABLE_CONTEXT
)
```

### 3. Episode/Event Modeling

```python
from uuid import uuid4
from datetime import datetime
import json

# Store conversation as episode with events
episode_id = f"ep_{uuid4()}"

# Episode metadata
db.put_path(f"episodes/{episode_id}/meta", json.dumps({
    "type": "conversation",
    "user_id": user_id,
    "started_at": datetime.now().isoformat(),
    "embedding": embedding.tolist()  # For semantic search
}).encode())

# Events within episode
for i, message in enumerate(messages):
    db.put_path(f"episodes/{episode_id}/events/{i:06d}", json.dumps({
        "role": message["role"],
        "content": message["content"],
        "timestamp": message["timestamp"]
    }).encode())
```

### 4. Error Handling

```python
from toondb import Database, ToonDBError

try:
    db = Database.open("./data")
    
    with db.transaction() as txn:
        txn.put(b"key", b"value")
        # If exception here, auto-rollback
        
except ToonDBError as e:
    if "transaction conflict" in str(e):
        # Retry with backoff
        pass
    elif "database locked" in str(e):
        # Wait and retry
        pass
    else:
        raise
finally:
    db.close()
```

### 5. Monitoring

```python
import time

# Periodic stats collection
def collect_metrics():
    stats = db.stats()
    
    metrics = {
        "memtable_size_mb": stats["memtable_size_bytes"] / 1024 / 1024,
        "active_txns": stats["active_transactions"],
        "checkpoint_lsn": stats.get("last_checkpoint_lsn", 0),
    }
    
    # Send to your monitoring system
    send_to_prometheus(metrics)

# Schedule every 30 seconds
while True:
    collect_metrics()
    time.sleep(30)
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `Library not found` | `TOONDB_LIB_PATH` not set | Export path to `libtoondb_storage.so/dylib` |
| `Transaction conflict` | Concurrent write to same key | Implement retry with exponential backoff |
| `Token budget exceeded` | Sections too large | Increase budget or reduce `top_k` |
| `Vector search returns wrong results` | Known HNSW bug | Check GitHub issues, apply patches |

### Debug Logging

```bash
# Enable verbose logging
RUST_LOG=toondb=debug toondb-mcp --db ./data

# MCP protocol debugging
RUST_LOG=toondb_mcp=trace toondb-mcp --db ./data
```

---

## Version Compatibility

| ToonDB Version | Python | Rust | MCP Protocol |
|----------------|--------|------|--------------|
| 0.1.x | 3.9+ | 1.75+ | 2024-11-05 |

---

## Resources

- **GitHub**: https://github.com/toondb/toondb
- **MCP Specification**: https://modelcontextprotocol.io
- **Issue Tracker**: https://github.com/toondb/toondb/issues
