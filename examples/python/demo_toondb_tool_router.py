#!/usr/bin/env python3
"""
Demo: ToonDB tool routing + context retrieval + trace persistence.

Implements the standard agent pattern:
retrieve context + shortlist tools -> bind tools dynamically -> call model -> persist traces

- Uses ToonDB Database.open() for KV/path storage.  (recommended single-process mode)
- Uses ToonDB VectorIndex for semantic search (docs + tools).
- Prints token estimates for:
  (A) baseline: ALL tools + full context
  (B) toondb: top-k tools + token-budgeted context

Usage:
    # From repo root:
    cargo build --release
    
    pip install -U numpy tiktoken python-dotenv
    pip install -U langgraph langchain-openai langchain-core openai  # optional but recommended
    
    export PYTHONPATH=toondb-python-sdk/src
    export TOONDB_LIB_PATH=target/release
    
    python3 demo_toondb_tool_router.py
"""

import os
import sys
import json
import time
import uuid
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

# ToonDB imports (SDK)
from toondb import Database, VectorIndex

# Optional: more accurate token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Optional: Real LLM integration
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


# -----------------------------
# Azure OpenAI Clients (optional)
# -----------------------------

class AzureEmbeddings:
    """Azure embeddings client for real embeddings."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_EMEBEDDING_ENDPOINT")
        self.api_key = os.getenv("AZURE_EMEBEDDING_API_KEY")
        self.api_version = os.getenv("AZURE_EMEBEDDING_API_VERSION", "2024-12-01-preview")
        self.deployment = os.getenv("AZURE_EMEBEDDING_DEPLOYMENT_NAME", "embedding")
        
        if self.endpoint and self.api_key:
            import requests
            self._requests = requests
            self.url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
            self.available = True
        else:
            self.available = False
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("Azure embeddings not configured")
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        response = self._requests.post(self.url, headers=headers, json={"input": texts, "model": self.deployment})
        response.raise_for_status()
        return np.array([item["embedding"] for item in response.json()["data"]], dtype=np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class AzureLLM:
    """Azure OpenAI LLM client."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_API_BASE")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        
        if self.endpoint and self.api_key:
            import requests
            self._requests = requests
            self.url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
            self.available = True
        else:
            self.available = False
    
    def complete(self, messages: List[Dict], max_tokens: int = 500, tools: Optional[List[Dict]] = None) -> Dict:
        if not self.available:
            raise RuntimeError("Azure LLM not configured")
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = self._requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


# -----------------------------
# Utilities
# -----------------------------

def mock_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Deterministic local embedding (no API calls).
    Uses SHA256 for stable seeding across runs.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def approx_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Approximate token count - uses tiktoken if available, else 4 chars/token."""
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def jdump(obj: Any) -> str:
    """JSON serialize compactly."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# -----------------------------
# ToonDB-backed registry
# -----------------------------

@dataclass
class ToolSpec:
    """Tool specification with OpenAI-compatible schema."""
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI-style tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToonDBToolRouter:
    """
    Stores tool specs in ToonDB KV (path-native API) and indexes descriptions in VectorIndex.
    Supports shortlist(query, k) -> List[tool_schema]
    """
    def __init__(self, db: Database, dim: int = 384, use_real_embeddings: bool = False):
        self.db = db
        self.dim = dim
        self.tool_index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
        self._next_tool_id = 0
        
        # Optionally use real embeddings
        self.embeddings = None
        if use_real_embeddings:
            try:
                e = AzureEmbeddings()
                if e.available:
                    self.embeddings = e
                    # Update dimension based on real embedding
                    test_embed = e.embed_single("test")
                    self.dim = len(test_embed)
                    self.tool_index = VectorIndex(dimension=self.dim, max_connections=16, ef_construction=100)
            except Exception:
                pass

    def _tool_path(self, name: str) -> str:
        return f"catalog/tools/{name}"

    def _tool_id_path(self, tool_id: int) -> str:
        return f"catalog/tool_ids/{tool_id}"
    
    def _embed(self, text: str) -> np.ndarray:
        if self.embeddings:
            return self.embeddings.embed_single(text)
        return mock_embedding(text, self.dim)

    def upsert_tools(self, tools: List[ToolSpec]) -> None:
        """Insert or update tools in the registry."""
        ids = []
        vecs = []

        for t in tools:
            # Persist the canonical tool schema in ToonDB
            self.db.put_path(self._tool_path(t.name), jdump(t.to_openai_tool()).encode("utf-8"))

            tool_id = self._next_tool_id
            self._next_tool_id += 1

            # Persist reverse mapping: tool_id -> tool_name
            self.db.put_path(self._tool_id_path(tool_id), t.name.encode("utf-8"))

            ids.append(tool_id)
            vecs.append(self._embed(f"{t.name}\n{t.description}"))

        self.tool_index.insert_batch(np.array(ids, dtype=np.uint64), np.stack(vecs).astype(np.float32))

    def shortlist(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """Get top-k most relevant tools for a query."""
        qv = self._embed(query).astype(np.float32)
        hits = self.tool_index.search(qv, k=k)

        tools = []
        for tool_id, dist in hits:
            name_bytes = self.db.get_path(self._tool_id_path(int(tool_id)))
            if not name_bytes:
                continue
            name = name_bytes.decode("utf-8")
            spec_bytes = self.db.get_path(self._tool_path(name))
            if not spec_bytes:
                continue
            tools.append(json.loads(spec_bytes.decode("utf-8")))
        return tools


class ToonDBDocStore:
    """
    Stores documents in ToonDB KV and indexes content in VectorIndex.
    """
    def __init__(self, db: Database, dim: int = 384, use_real_embeddings: bool = False):
        self.db = db
        self.dim = dim
        self.doc_index = VectorIndex(dimension=dim, max_connections=16, ef_construction=100)
        self._next_doc_id = 0
        self._doc_texts = {}  # Keep texts in-memory for demo
        
        # Optionally use real embeddings
        self.embeddings = None
        if use_real_embeddings:
            try:
                e = AzureEmbeddings()
                if e.available:
                    self.embeddings = e
                    test_embed = e.embed_single("test")
                    self.dim = len(test_embed)
                    self.doc_index = VectorIndex(dimension=self.dim, max_connections=16, ef_construction=100)
            except Exception:
                pass

    def _doc_path(self, doc_id: int) -> str:
        return f"docs/{doc_id}/text"
    
    def _embed(self, text: str) -> np.ndarray:
        if self.embeddings:
            return self.embeddings.embed_single(text)
        return mock_embedding(text, self.dim)

    def add_docs(self, texts: List[str]) -> None:
        """Add documents to the store."""
        ids, vecs = [], []
        for text in texts:
            doc_id = self._next_doc_id
            self._next_doc_id += 1
            self.db.put_path(self._doc_path(doc_id), text.encode("utf-8"))
            self._doc_texts[doc_id] = text
            ids.append(doc_id)
            vecs.append(self._embed(text))
        self.doc_index.insert_batch(np.array(ids, dtype=np.uint64), np.stack(vecs).astype(np.float32))

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """Retrieve top-k relevant documents."""
        qv = self._embed(query).astype(np.float32)
        hits = self.doc_index.search(qv, k=k)
        out = []
        for doc_id, dist in hits:
            b = self.db.get_path(self._doc_path(int(doc_id)))
            if b:
                out.append(b.decode("utf-8"))
        return out

    def build_context(self, query: str, token_budget: int = 900, k: int = 6) -> str:
        """
        Simple token-budgeted context packer.
        """
        chunks = self.retrieve(query, k=k)
        ctx = []
        used = 0
        for c in chunks:
            t = approx_tokens(c)
            if used + t > token_budget:
                break
            ctx.append(c)
            used += t
        return "\n\n---\n\n".join(ctx)


# -----------------------------
# Demo "business" tools
# -----------------------------

def build_demo_tools(n_total: int = 60) -> List[ToolSpec]:
    """
    Create a realistic-ish tool inventory: a few relevant customer-support tools + many distractors.
    """
    core = [
        ToolSpec(
            name="get_order",
            description="Lookup an order by order_id and return status, items, and payment method.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        ),
        ToolSpec(
            name="refund_order",
            description="Issue a refund for an order. Use after validating eligibility and amount.",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount_usd": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "amount_usd"],
            },
        ),
        ToolSpec(
            name="update_crm",
            description="Update the CRM case notes and customer status flags.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["customer_id", "notes"],
            },
        ),
        ToolSpec(
            name="create_support_ticket",
            description="Create a support ticket and assign it to a queue (billing, tech, fraud).",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}, "queue": {"type": "string"}},
                "required": ["title"],
            },
        ),
        ToolSpec(
            name="get_customer",
            description="Lookup customer profile by customer_id including contact info, tier, and history.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        ),
        ToolSpec(
            name="send_email",
            description="Send an email notification to a customer about order updates or promotions.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        ),
    ]

    # Distractors - unrelated tools
    extras = []
    distractor_prefixes = [
        ("analytics", "Generate {} analytics report for internal dashboards."),
        ("export", "Export {} data to external systems via API."),
        ("ops", "Internal operations tool for {} management."),
        ("sync", "Synchronize {} data with external services."),
        ("archive", "Archive old {} records to cold storage."),
        ("validate", "Validate {} data integrity and consistency."),
    ]
    
    for i in range(max(0, n_total - len(core))):
        prefix, desc_template = distractor_prefixes[i % len(distractor_prefixes)]
        extras.append(
            ToolSpec(
                name=f"{prefix}_tool_{i:03d}",
                description=desc_template.format(f"internal-{i}"),
                parameters={"type": "object", "properties": {}, "required": []},
            )
        )
    return core + extras


def baseline_prompt_payload(query: str, all_tools: List[Dict[str, Any]], full_context: str) -> str:
    """
    What the model "sees" (roughly) if you dump everything in-context.
    """
    return (
        "SYSTEM: You are a support agent.\n\n"
        f"CONTEXT:\n{full_context}\n\n"
        f"TOOLS:\n{jdump(all_tools)}\n\n"
        f"USER:\n{query}\n"
    )


def routed_prompt_payload(query: str, routed_tools: List[Dict[str, Any]], ctx: str) -> str:
    return (
        "SYSTEM: You are a support agent.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"TOOLS:\n{jdump(routed_tools)}\n\n"
        f"USER:\n{query}\n"
    )


# -----------------------------
# LangGraph Integration (optional)
# -----------------------------

def run_langgraph_demo(query: str, routed_tools: List[Dict], ctx: str, llm: AzureLLM) -> Optional[Dict]:
    """
    Optional: Run a LangGraph-style workflow with dynamic tool binding.
    
    This demonstrates:
    1. retrieve context + shortlist tools (already done by ToonDB)
    2. bind tools dynamically (subset) 
    3. call model with the bound tools
    """
    try:
        from langgraph.graph import StateGraph, START, END
        from langchain_core.messages import HumanMessage, AIMessage
    except ImportError:
        print("\n⚠️  LangGraph not installed. Skipping LangGraph demo.")
        print("   Install with: pip install langgraph langchain-core")
        return None
    
    from typing import TypedDict, Sequence
    from langchain_core.messages import BaseMessage
    
    class AgentState(TypedDict):
        messages: Sequence[BaseMessage]
        context: str
        tools: List[Dict]
        response: Optional[str]
    
    def call_model(state: AgentState) -> dict:
        """Call the model with dynamically bound tools."""
        messages = [
            {"role": "system", "content": f"You are a helpful support agent.\n\nContext:\n{state['context']}"},
            {"role": "user", "content": state["messages"][-1].content}
        ]
        
        # Call with only the routed tools (not all 60!)
        response = llm.complete(messages, max_tokens=300, tools=state["tools"])
        
        choice = response["choices"][0]
        if choice.get("finish_reason") == "tool_calls":
            tool_calls = choice["message"].get("tool_calls", [])
            return {"response": f"[Tool Calls: {[tc['function']['name'] for tc in tool_calls]}]"}
        else:
            content = choice["message"].get("content", "")
            return {"response": content}
    
    # Build simple graph
    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    
    agent = graph.compile()
    
    result = agent.invoke({
        "messages": [HumanMessage(content=query)],
        "context": ctx,
        "tools": routed_tools,
        "response": None,
    })
    
    return result


# -----------------------------
# Main
# -----------------------------

def main():
    query = os.environ.get("DEMO_QUERY", "Please refund order 1234 for $19.99 and note it in CRM for customer C-77.")
    model_name = os.environ.get("DEMO_MODEL", "gpt-4o-mini")

    db_path = os.environ.get("TOONDB_DEMO_PATH", "./toondb_demo_db")
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Configuration
    use_real_embeddings = os.environ.get("DEMO_REAL_EMBEDDINGS", "").lower() in ("1", "true")
    use_real_llm = os.environ.get("DEMO_REAL_LLM", "").lower() in ("1", "true")
    
    print("\n" + "=" * 70)
    print("  TOONDB TOOL ROUTING DEMO")
    print("  Context Retrieval + Dynamic Tool Binding + Trace Persistence")
    print("=" * 70)

    with Database.open(db_path) as db:
        # 1) Setup stores
        print("\n📦 Setting up ToonDB stores...")
        dim = 384 if not use_real_embeddings else 1536  # Azure embeddings are 1536-dim
        router = ToonDBToolRouter(db, dim=dim, use_real_embeddings=use_real_embeddings)
        docs = ToonDBDocStore(db, dim=dim, use_real_embeddings=use_real_embeddings)

        # 2) Seed docs + tools
        print("📝 Seeding documents and tools...")
        docs.add_docs([
            "Refund policy: Refunds allowed within 30 days for unused items. Partial refunds require manager approval. Full refunds are auto-approved for orders under $50.",
            "CRM policy: Always record refund reason and include payment method, timestamp, and ticket link. Update customer tier if refund count exceeds 3 in 30 days.",
            "Fraud note: If 3+ refunds in 7 days, route to fraud queue and do not auto-refund. Flag customer account for review.",
            "Order lookup guide: Validate order_id format (ORD-XXXX), confirm status is DELIVERED or CANCELLED before refund. Check payment method for refund routing.",
            "Customer support SLA: Response within 4 hours for billing issues, 24 hours for general inquiries. Escalate unresolved issues after 3 contacts.",
            "Payment processing: Refunds take 3-5 business days for credit cards, 1-2 days for digital wallets. Gift card refunds are instant store credit.",
        ])
        
        tool_specs = build_demo_tools(n_total=int(os.environ.get("DEMO_TOOL_COUNT", "60")))
        router.upsert_tools(tool_specs)
        
        print(f"   ✓ Indexed {docs._next_doc_id} documents")
        print(f"   ✓ Indexed {len(tool_specs)} tools")

        # 3) Retrieve context + shortlist tools
        print("\n🔍 Routing query...")
        ctx_budget = int(os.environ.get("DEMO_CTX_BUDGET", "300"))
        ctx = docs.build_context(query, token_budget=ctx_budget, k=6)
        routed_tools = router.shortlist(query, k=int(os.environ.get("DEMO_TOPK_TOOLS", "6")))

        all_tools = [t.to_openai_tool() for t in tool_specs]

        # 4) Compare token footprints
        full_ctx = "\n\n".join(docs.retrieve(query, k=20))
        base_payload = baseline_prompt_payload(query, all_tools, full_ctx)
        routed_payload = routed_prompt_payload(query, routed_tools, ctx)

        base_tokens = approx_tokens(base_payload, model=model_name)
        routed_tokens = approx_tokens(routed_payload, model=model_name)

        print(f"\n📊 Token Comparison:")
        print(f"   Query: {query[:60]}...")
        print(f"\n   ┌─────────────────────────────────────────────────────────────┐")
        print(f"   │  Approach     │  Tools  │  Context Budget  │  Total Tokens  │")
        print(f"   ├─────────────────────────────────────────────────────────────┤")
        print(f"   │  Baseline     │  {len(all_tools):5}  │     all docs     │    {base_tokens:6,}     │")
        print(f"   │  ToonDB       │  {len(routed_tools):5}  │     {ctx_budget:4} tokens  │    {routed_tokens:6,}     │")
        print(f"   └─────────────────────────────────────────────────────────────┘")
        print(f"\n   💰 Token savings: {base_tokens - routed_tokens:,} tokens ({100*(base_tokens-routed_tokens)/base_tokens:.1f}% reduction)")

        # Show which tools were selected
        print(f"\n🛠️  Shortlisted tools ({len(routed_tools)}):")
        for t in routed_tools:
            print(f"   • {t['function']['name']}: {t['function']['description'][:50]}...")

        # 5) Persist trace/outcomes back into ToonDB
        print(f"\n💾 Persisting trace to ToonDB...")
        db.put_path(f"runs/{run_id}/timestamp", timestamp.encode("utf-8"))
        db.put_path(f"runs/{run_id}/query", query.encode("utf-8"))
        db.put_path(f"runs/{run_id}/baseline_tokens", str(base_tokens).encode("utf-8"))
        db.put_path(f"runs/{run_id}/routed_tokens", str(routed_tokens).encode("utf-8"))
        db.put_path(f"runs/{run_id}/tool_shortlist", jdump([t["function"]["name"] for t in routed_tools]).encode("utf-8"))
        db.put_path(f"runs/{run_id}/context_used", ctx.encode("utf-8"))
        db.put_path(f"runs/{run_id}/token_savings", str(base_tokens - routed_tokens).encode("utf-8"))

        print(f"   Stored trace paths under: runs/{run_id}/")
        print(f"   • query")
        print(f"   • tool_shortlist")
        print(f"   • context_used")
        print(f"   • baseline_tokens / routed_tokens / token_savings")

        # 6) Optional: Real LLM call with LangGraph
        if use_real_llm:
            print("\n🤖 Calling real LLM with LangGraph workflow...")
            llm = AzureLLM()
            if llm.available:
                result = run_langgraph_demo(query, routed_tools, ctx, llm)
                if result:
                    print(f"\n   LLM Response: {result.get('response', 'N/A')[:200]}...")
                    # Persist LLM response
                    db.put_path(f"runs/{run_id}/llm_response", str(result.get('response', '')).encode("utf-8"))
            else:
                print("   ⚠️  Azure LLM not configured. Set AZURE_OPENAI_* env vars.")

        # 7) Show how to integrate with LangGraph "bind_tools" pattern
        print("\n" + "─" * 70)
        print("📘 LangGraph Integration Pattern:")
        print("─" * 70)
        print("""
    # Standard LangGraph pattern with ToonDB routing:
    
    from toondb import Database, VectorIndex
    from langgraph.graph import StateGraph
    from langchain_openai import ChatOpenAI
    
    # 1. Get routed tools (what we just did)
    routed_tools = router.shortlist(user_msg, k=6)
    
    # 2. Bind tools dynamically (LangGraph style)
    model = ChatOpenAI(model="gpt-4o")
    bound_model = model.bind_tools(routed_tools)  # Only passes subset!
    
    # 3. Build graph with bound model
    def call_model(state):
        return {"messages": [bound_model.invoke(state["messages"])]}
    
    # 4. Persist traces (what we just did)
    db.put_path(f"runs/{run_id}/tool_shortlist", ...)
        """)
        
        print("\n" + "=" * 70)
        print(f"  ✅ Demo complete!")
        print(f"  DB: {db_path}")
        print(f"  Run ID: {run_id}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
