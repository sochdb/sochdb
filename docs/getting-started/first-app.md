# Tutorial: Build Your First ToonDB Application

> **Time:** 15 minutes  
> **Difficulty:** Beginner  
> **Prerequisites:** Python 3.9+ or Rust 2024 edition

In this tutorial, you'll build a simple agent memory system that stores conversation history, user preferences, and semantic search using vector embeddings.

---

## What You'll Build

A personal assistant memory store with:
- ✅ User preferences storage
- ✅ Conversation history tracking  
- ✅ Semantic search with vector embeddings
- ✅ Token-efficient LLM context assembly

---

## Step 1: Project Setup

### Python

```bash
# Create project directory
mkdir agent-memory && cd agent-memory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install toondb-client numpy
```

### Rust

```bash
# Create new project
cargo new agent-memory
cd agent-memory

# Add dependencies to Cargo.toml
cat >> Cargo.toml << 'EOF'
[dependencies]
toondb = "0.1"
toondb-kernel = "0.1"
EOF
```

---

## Step 2: Initialize the Database

### Python

Create `main.py`:

```python
from toondb import Database
import json

# Open database (creates if not exists)
db = Database.open("./agent_memory")

print("✅ Database initialized at ./agent_memory")

db.close()
```

Run it:
```bash
python main.py
# Output: ✅ Database initialized at ./agent_memory
```

### Rust

Create `src/main.rs`:

```rust
use toondb::ToonConnection;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conn = ToonConnection::open("./agent_memory")?;
    println!("✅ Database initialized at ./agent_memory");
    Ok(())
}
```

Run it:
```bash
cargo run
# Output: ✅ Database initialized at ./agent_memory
```

---

## Step 3: Store User Preferences

### Python

Add to `main.py`:

```python
def store_preferences(db, user_id: str, prefs: dict):
    """Store user preferences using path-based access."""
    with db.transaction() as txn:
        # Store each preference as a separate key for granular access
        for key, value in prefs.items():
            path = f"users/{user_id}/preferences/{key}"
            txn.put(path.encode(), json.dumps(value).encode())
    print(f"✅ Stored preferences for user {user_id}")

def get_preferences(db, user_id: str) -> dict:
    """Retrieve all preferences for a user."""
    prefs = {}
    prefix = f"users/{user_id}/preferences/"
    for key, value in db.scan(prefix.encode()):
        pref_name = key.decode().split("/")[-1]
        prefs[pref_name] = json.loads(value.decode())
    return prefs

# Test it
db = Database.open("./agent_memory")

store_preferences(db, "alice", {
    "theme": "dark",
    "language": "en",
    "notifications": True,
    "model": "claude-sonnet"
})

prefs = get_preferences(db, "alice")
print(f"Alice's preferences: {prefs}")
```

Run it:
```bash
python main.py
# Output:
# ✅ Stored preferences for user alice
# Alice's preferences: {'theme': 'dark', 'language': 'en', ...}
```

---

## Step 4: Track Conversation History

### Python

Add to `main.py`:

```python
from datetime import datetime

def add_message(db, user_id: str, role: str, content: str):
    """Add a message to conversation history."""
    timestamp = datetime.now().isoformat()
    message_id = timestamp.replace(":", "-").replace(".", "-")
    
    with db.transaction() as txn:
        path = f"users/{user_id}/history/{message_id}"
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        txn.put(path.encode(), json.dumps(message).encode())
    
    return message_id

def get_recent_messages(db, user_id: str, limit: int = 10) -> list:
    """Get recent messages (most recent first)."""
    messages = []
    prefix = f"users/{user_id}/history/"
    
    for key, value in db.scan(prefix.encode()):
        messages.append(json.loads(value.decode()))
    
    # Sort by timestamp descending and limit
    messages.sort(key=lambda m: m["timestamp"], reverse=True)
    return messages[:limit]

# Test it
db = Database.open("./agent_memory")

add_message(db, "alice", "user", "What is the capital of France?")
add_message(db, "alice", "assistant", "The capital of France is Paris.")
add_message(db, "alice", "user", "What's the population?")
add_message(db, "alice", "assistant", "Paris has about 2.1 million people in the city proper.")

history = get_recent_messages(db, "alice", limit=2)
print("Recent messages:")
for msg in history:
    print(f"  [{msg['role']}]: {msg['content'][:50]}...")

db.close()
```

---

## Step 5: Add Vector Search (Semantic Memory)

### Python

Add to `main.py`:

```python
import numpy as np
from toondb.bulk import bulk_build_index, bulk_query_index
import os

def create_embeddings(texts: list[str]) -> np.ndarray:
    """Create mock embeddings (replace with real embedding model)."""
    # In production, use sentence-transformers, OpenAI, etc.
    np.random.seed(42)
    return np.random.randn(len(texts), 384).astype(np.float32)

def build_memory_index(db, user_id: str):
    """Build vector index from conversation history."""
    messages = []
    prefix = f"users/{user_id}/history/"
    
    for key, value in db.scan(prefix.encode()):
        msg = json.loads(value.decode())
        messages.append(msg["content"])
    
    if not messages:
        print("No messages to index")
        return None
    
    # Create embeddings
    embeddings = create_embeddings(messages)
    
    # Ensure index directory exists
    os.makedirs("./agent_memory/indexes", exist_ok=True)
    
    # Build HNSW index
    index_path = f"./agent_memory/indexes/{user_id}_memory.hnsw"
    stats = bulk_build_index(
        embeddings, 
        output=index_path,
        m=16,  # Graph connectivity
        ef_construction=200  # Build quality
    )
    
    print(f"✅ Built index with {len(messages)} vectors")
    return index_path

def search_memory(index_path: str, query: str, k: int = 3):
    """Search conversation memory semantically."""
    query_embedding = create_embeddings([query])[0]
    
    results = bulk_query_index(
        index_path=index_path,
        query=query_embedding,
        k=k
    )
    
    return results

# Test it (only if bulk operations are available)
try:
    db = Database.open("./agent_memory")
    index_path = build_memory_index(db, "alice")
    
    if index_path:
        results = search_memory(index_path, "France capital city")
        print(f"Search results: {results}")
    db.close()
except ImportError:
    print("Note: bulk operations require numpy")
```

---

## Step 6: Assemble LLM Context

### Python

Add to `main.py`:

```python
def build_llm_context(db, user_id: str, current_query: str, token_budget: int = 4000):
    """
    Build token-efficient context for LLM.
    Uses TOON format for 40-66% token savings.
    """
    context_parts = []
    
    # 1. System prompt (priority 0 - always include)
    context_parts.append({
        "priority": 0,
        "content": "You are a helpful assistant with access to user preferences and history."
    })
    
    # 2. User preferences (priority 1)
    prefs = get_preferences(db, user_id)
    if prefs:
        # TOON format is more token-efficient
        pref_toon = f"prefs[1]{{" + ",".join(prefs.keys()) + "}}:" + ",".join(str(v) for v in prefs.values())
        context_parts.append({
            "priority": 1,
            "content": f"User preferences: {pref_toon}"
        })
    
    # 3. Recent history (priority 2)
    history = get_recent_messages(db, user_id, limit=5)
    if history:
        history_lines = []
        for msg in reversed(history):  # Chronological order
            history_lines.append(f"{msg['role']}: {msg['content']}")
        context_parts.append({
            "priority": 2,
            "content": "Recent conversation:\n" + "\n".join(history_lines)
        })
    
    # 4. Current query (priority 0 - always include)
    context_parts.append({
        "priority": 0,
        "content": f"Current query: {current_query}"
    })
    
    # Sort by priority and assemble
    context_parts.sort(key=lambda x: x["priority"])
    
    full_context = "\n\n".join(part["content"] for part in context_parts)
    
    # In production, truncate by token budget with priority-based trimming
    return full_context

# Test it
db = Database.open("./agent_memory")

context = build_llm_context(db, "alice", "Tell me more about Paris")
print("=== LLM Context ===")
print(context)
print(f"\nContext length: {len(context)} characters")

db.close()
```

---

## Step 7: Complete Application

Here's the complete `main.py`:

```python
#!/usr/bin/env python3
"""
Agent Memory System with ToonDB
A simple personal assistant memory store.
"""

from toondb import Database
from datetime import datetime
import json

class AgentMemory:
    def __init__(self, path: str = "./agent_memory"):
        self.db = Database.open(path)
    
    def close(self):
        self.db.close()
    
    def store_preference(self, user_id: str, key: str, value):
        path = f"users/{user_id}/preferences/{key}"
        with self.db.transaction() as txn:
            txn.put(path.encode(), json.dumps(value).encode())
    
    def get_preferences(self, user_id: str) -> dict:
        prefs = {}
        prefix = f"users/{user_id}/preferences/"
        for key, value in self.db.scan(prefix.encode()):
            pref_name = key.decode().split("/")[-1]
            prefs[pref_name] = json.loads(value.decode())
        return prefs
    
    def add_message(self, user_id: str, role: str, content: str) -> str:
        timestamp = datetime.now().isoformat()
        message_id = timestamp.replace(":", "-").replace(".", "-")
        
        with self.db.transaction() as txn:
            path = f"users/{user_id}/history/{message_id}"
            txn.put(
                path.encode(),
                json.dumps({"role": role, "content": content, "ts": timestamp}).encode()
            )
        return message_id
    
    def get_history(self, user_id: str, limit: int = 10) -> list:
        messages = []
        prefix = f"users/{user_id}/history/"
        for _, value in self.db.scan(prefix.encode()):
            messages.append(json.loads(value.decode()))
        messages.sort(key=lambda m: m["ts"], reverse=True)
        return messages[:limit]
    
    def build_context(self, user_id: str, query: str) -> str:
        parts = [
            "System: You are a helpful assistant.",
            f"Preferences: {json.dumps(self.get_preferences(user_id))}",
        ]
        
        history = self.get_history(user_id, limit=5)
        if history:
            parts.append("Recent history:")
            for msg in reversed(history):
                parts.append(f"  {msg['role']}: {msg['content']}")
        
        parts.append(f"Query: {query}")
        return "\n".join(parts)


def main():
    print("🎬 ToonDB Agent Memory Demo\n")
    
    # Initialize
    memory = AgentMemory()
    user = "demo_user"
    
    # Store preferences
    memory.store_preference(user, "name", "Demo User")
    memory.store_preference(user, "theme", "dark")
    print(f"✅ Stored preferences: {memory.get_preferences(user)}")
    
    # Add conversation
    memory.add_message(user, "user", "Hello, who are you?")
    memory.add_message(user, "assistant", "I'm your AI assistant with persistent memory!")
    memory.add_message(user, "user", "What can you remember?")
    print(f"✅ Added {len(memory.get_history(user))} messages")
    
    # Build context
    context = memory.build_context(user, "Summarize our conversation")
    print(f"\n📝 LLM Context ({len(context)} chars):\n")
    print(context)
    
    # Clean up
    memory.close()
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
```

Run the complete application:

```bash
python main.py
```

Expected output:
```
🎬 ToonDB Agent Memory Demo

✅ Stored preferences: {'name': 'Demo User', 'theme': 'dark'}
✅ Added 3 messages

📝 LLM Context (298 chars):

System: You are a helpful assistant.
Preferences: {"name": "Demo User", "theme": "dark"}
Recent history:
  user: Hello, who are you?
  assistant: I'm your AI assistant with persistent memory!
  user: What can you remember?
Query: Summarize our conversation

✅ Demo complete!
```

---

## What You Learned

| Concept | What You Did |
|---------|--------------|
| **Path-based access** | Used `users/{id}/preferences/{key}` for O(path) lookups |
| **Transactions** | Wrapped writes in `with db.transaction()` for atomicity |
| **Prefix scans** | Used `db.scan(prefix)` for range queries |
| **Context assembly** | Built token-efficient LLM prompts |

---

## Next Steps

| Goal | Resource |
|------|----------|
| Add real embeddings | [Vector Search Tutorial](/guides/vector-search) |
| Use with Claude/MCP | [MCP Integration](/cookbook/mcp-integration) |
| Production deployment | [Deployment Guide](/guides/deployment) |
| Performance tuning | [Performance Guide](/concepts/performance) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'toondb'`

```bash
pip install --upgrade toondb-client
```

### `Permission denied` on database path

```bash
mkdir -p ./agent_memory
chmod 755 ./agent_memory
```

### Transaction failed

Transactions auto-rollback on exceptions. Check your data types match the schema.

---

*Tutorial completed! You've built a working agent memory system with ToonDB.* 🎉
