# ToonDB Examples

Real-world examples showing how to use ToonDB for AI/LLM applications.

## Examples by Language

### Python (`examples/python/`)

#### AI Framework Integrations

| Example | Description | Frameworks |
|---------|-------------|------------|
| [langgraph_agent.py](python/langgraph_agent.py) | Multi-turn agent with memory | LangGraph, LangChain |
| [crewai_research_crew.py](python/crewai_research_crew.py) | Multi-agent research workflow | CrewAI |
| [llamaindex_rag.py](python/llamaindex_rag.py) | RAG with custom vector store | LlamaIndex |
| [simple_rag_chatbot.py](python/simple_rag_chatbot.py) | Minimal chatbot with memory | Pure Python |
| [semantic_search_api.py](python/semantic_search_api.py) | Search API with filtering | REST-style |

#### Practical VectorDB Use Cases

| Example | Description | Key Features |
|---------|-------------|--------------|
| [customer_support_rag.py](python/customer_support_rag.py) | Multi-tenant support system | ACL, time decay, OOD handling |
| [ecommerce_search.py](python/ecommerce_search.py) | Product semantic search | Multi-vector, faceted filtering |
| [semantic_dedup.py](python/semantic_dedup.py) | Near-duplicate detection | Threshold matching, clustering |
| [code_search.py](python/code_search.py) | Codebase semantic search | Hybrid keyword+semantic, MRR |
| [personalization.py](python/personalization.py) | User recommendation engine | User vectors, cold-start |
| [security_qa_triage.py](python/security_qa_triage.py) | Security-safe QA | Injection detection, PII redaction |

#### Test Suites

| Example | Description | Frameworks |
|---------|-------------|------------|
| [comprehensive_e2e_test.py](python/comprehensive_e2e_test.py) | Full test suite | All features |
| [real_llm_test.py](python/real_llm_test.py) | Real LLM integration test | Azure OpenAI |

### Rust (`examples/rust/`)

| Example | Description | Key Features |
|---------|-------------|--------------|
| [01_basic_database.rs](rust/01_basic_database.rs) | Basic KV operations | Put, Get, Delete, Path API |
| [02_transactions.rs](rust/02_transactions.rs) | ACID transactions | with_transaction, rollback |
| [03_vector_search.rs](rust/03_vector_search.rs) | Vector similarity search | HNSW, bulk indexing |

### Node.js/TypeScript (`examples/nodejs/`)

| Example | Description | Key Features |
|---------|-------------|--------------|
| [01_basic_database.ts](nodejs/01_basic_database.ts) | Basic KV operations | Put, Get, Delete, Path API |
| [02_transactions.ts](nodejs/02_transactions.ts) | ACID transactions | withTransaction, rollback |
| [03_vector_search.ts](nodejs/03_vector_search.ts) | Vector similarity search | HNSW, bulk indexing |

### Go (`examples/go/`)

| Example | Description | Key Features |
|---------|-------------|--------------|
| [01_basic_database.go](go/01_basic_database.go) | Basic KV operations | Put, Get, Delete, Path API |
| [02_transactions.go](go/02_transactions.go) | ACID transactions | WithTransaction, rollback |
| [03_vector_search.go](go/03_vector_search.go) | Vector similarity search | HNSW, bulk indexing |

## Quick Start

### 1. Setup Environment

```bash
# Install Python dependencies
pip install python-dotenv requests numpy

# For LangGraph
pip install langgraph langchain-core

# For CrewAI (optional)
pip install crewai crewai-tools
```

### 2. Configure Azure OpenAI

Create a `.env` file in the repository root:

```env
AZURE_OPENAI_API_KEY="your-key"
AZURE_OPENAI_API_BASE="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4.1"

AZURE_EMEBEDDING_DEPLOYMENT_NAME="embedding"
AZURE_EMEBEDDING_ENDPOINT="https://your-cognitive-services.cognitiveservices.azure.com/"
AZURE_EMEBEDDING_API_KEY="your-key"
AZURE_EMEBEDDING_API_VERSION="2024-12-01-preview"
```

### 3. Run Examples

```bash
# Set environment
export PYTHONPATH=$(pwd)/toondb-python-sdk/src
export TOONDB_LIB_PATH=$(pwd)/target/release

# Run any example
python3 examples/python/langgraph_agent.py
python3 examples/python/simple_rag_chatbot.py
python3 examples/python/semantic_search_api.py
```

## Example Highlights

### LangGraph Agent

```python
from toondb import VectorIndex, Database

# Vector retrieval
index = VectorIndex(dimension=1536)
index.insert_batch(ids, embeddings)
results = index.search(query_embedding, k=5)

# Memory storage
db = Database.open("/tmp/memory")
with db.transaction() as txn:
    txn.put(b"memory/1", json.dumps(memory).encode())
```

### Semantic Search API

```python
from toondb import VectorIndex

# High-quality index settings
index = VectorIndex(
    dimension=1536,
    max_connections=32,      # Higher = better recall
    ef_construction=200      # Higher = better index quality
)

# Batch insert
ids = np.arange(len(docs), dtype=np.uint64)
index.insert_batch(ids, embeddings)

# Search with post-filter
results = index.search(query_embed, k=20)
filtered = [r for r in results if metadata_matches(r)]
```

### RAG Chatbot

```python
class RAGChatbot:
    def chat(self, user_message: str) -> str:
        # 1. Retrieve context
        context = self.retrieve(user_message, k=3)
        
        # 2. Build prompt with history
        messages = [
            {"role": "system", "content": f"Context: {context}"},
            *self.history[-5:],
            {"role": "user", "content": user_message}
        ]
        
        # 3. Generate response
        response = self.llm(messages)
        
        # 4. Save to memory
        self.save_memory(user_message, response)
        
        return response
```

## Performance Tips

1. **Index Settings**
   - `max_connections=16` is good for most use cases
   - Increase to `32` for higher recall requirements
   - `ef_construction=100-200` balances build time and quality

2. **Batch Operations**
   - Use `insert_batch()` for bulk inserts (10-100x faster)
   - Batch embedding API calls (16 texts per call for Azure)

3. **Filtering**
   - ToonDB uses post-filtering; over-fetch by 3x for filtered queries
   - Store metadata externally for complex filter logic

4. **Memory Management**
   - Use KV store for conversation history
   - Prefix keys with user/session IDs for efficient scans

## License

Apache 2.0 - See [LICENSE](../LICENSE)
