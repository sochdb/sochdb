# SPDX-License-Identifier: AGPL-3.0-or-later
# SochDB Python SDK — Thin Client Wrapper
"""
SochDB Python SDK — Ergonomic gRPC client for SochDB.

Usage:
    from sochdb_sdk import SochDB

    db = SochDB("localhost:50051")
    results = db.vectors.search(collection="docs", vector=[0.1, 0.2], top_k=5)
    db.kv.put("session:abc", '{"user": "alice"}')
    value = db.kv.get("session:abc")
"""

from __future__ import annotations

import grpc
from typing import Any, Dict, List, Optional, Sequence

__version__ = "0.5.0"
__all__ = ["SochDB", "VectorClient", "KvClient", "GraphClient", "SqlClient"]


class SochDB:
    """Main SochDB client — provides access to all services."""

    def __init__(
        self,
        address: str = "localhost:50051",
        *,
        api_key: Optional[str] = None,
        secure: bool = False,
        options: Optional[List[tuple]] = None,
    ):
        """
        Connect to a SochDB server.

        Args:
            address: Server address (host:port)
            api_key: API key for authentication
            secure: Use TLS
            options: gRPC channel options
        """
        self._address = address
        self._api_key = api_key

        if secure:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(address, credentials, options=options)
        else:
            self._channel = grpc.insecure_channel(address, options=options)

        self._metadata = []
        if api_key:
            self._metadata.append(("x-api-key", api_key))

        # Lazy-init service clients
        self._vectors: Optional[VectorClient] = None
        self._kv: Optional[KvClient] = None
        self._graph: Optional[GraphClient] = None
        self._sql: Optional[SqlClient] = None
        self._collections: Optional[CollectionClient] = None
        self._namespaces: Optional[NamespaceClient] = None

    @property
    def vectors(self) -> "VectorClient":
        if self._vectors is None:
            self._vectors = VectorClient(self._channel, self._metadata)
        return self._vectors

    @property
    def kv(self) -> "KvClient":
        if self._kv is None:
            self._kv = KvClient(self._channel, self._metadata)
        return self._kv

    @property
    def graph(self) -> "GraphClient":
        if self._graph is None:
            self._graph = GraphClient(self._channel, self._metadata)
        return self._graph

    @property
    def sql(self) -> "SqlClient":
        if self._sql is None:
            self._sql = SqlClient(self._channel, self._metadata)
        return self._sql

    @property
    def collections(self) -> "CollectionClient":
        if self._collections is None:
            self._collections = CollectionClient(self._channel, self._metadata)
        return self._collections

    @property
    def namespaces(self) -> "NamespaceClient":
        if self._namespaces is None:
            self._namespaces = NamespaceClient(self._channel, self._metadata)
        return self._namespaces

    def close(self):
        """Close the gRPC channel."""
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _BaseClient:
    """Base class for service clients."""

    def __init__(self, channel: grpc.Channel, metadata: list):
        self._channel = channel
        self._metadata = metadata

    def _call(self, method, request, **kwargs):
        """Make a unary-unary gRPC call with auth metadata."""
        return method(request, metadata=self._metadata, **kwargs)


class VectorClient(_BaseClient):
    """Vector similarity search operations."""

    def search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "default",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            collection: Collection name
            vector: Query vector
            top_k: Number of results
            namespace: Namespace

        Returns:
            List of {id, score, metadata} dicts
        """
        # Import generated stubs (available after running generate.sh)
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.VectorIndexServiceStub(self._channel)
            request = sochdb_pb2.SearchRequest(
                namespace=namespace,
                collection=collection,
                query_vector=vector,
                top_k=top_k,
            )
            response = self._call(stub.Search, request)
            return [
                {"id": r.id, "score": r.score, "metadata": dict(r.metadata)}
                for r in response.results
            ]
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, str]] = None,
        namespace: str = "default",
    ):
        """Insert or update a vector."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.VectorIndexServiceStub(self._channel)
            request = sochdb_pb2.UpsertRequest(
                namespace=namespace,
                collection=collection,
                id=id,
                vector=vector,
                metadata=metadata or {},
            )
            return self._call(stub.Upsert, request)
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )


class KvClient(_BaseClient):
    """Key-value operations."""

    def get(self, key: str, namespace: str = "default") -> Optional[bytes]:
        """Get a value by key. Returns None if not found."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.KvServiceStub(self._channel)
            request = sochdb_pb2.KvGetRequest(namespace=namespace, key=key)
            response = self._call(stub.Get, request)
            return response.value if response.found else None
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )

    def put(
        self,
        key: str,
        value: str | bytes,
        namespace: str = "default",
        ttl_seconds: int = 0,
    ):
        """Put a key-value pair."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            if isinstance(value, str):
                value = value.encode("utf-8")
            stub = sochdb_pb2_grpc.KvServiceStub(self._channel)
            request = sochdb_pb2.KvPutRequest(
                namespace=namespace,
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
            )
            return self._call(stub.Put, request)
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a key. Returns True if the key existed."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.KvServiceStub(self._channel)
            request = sochdb_pb2.KvDeleteRequest(namespace=namespace, key=key)
            response = self._call(stub.Delete, request)
            return response.deleted
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )


class GraphClient(_BaseClient):
    """Graph operations for agent memory."""

    def add_node(
        self,
        graph_id: str,
        node_id: str,
        label: str = "",
        properties: Optional[Dict[str, str]] = None,
        namespace: str = "default",
    ):
        """Add a node to the graph."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.GraphServiceStub(self._channel)
            request = sochdb_pb2.AddNodeRequest(
                namespace=namespace,
                graph_id=graph_id,
                node_id=node_id,
                label=label,
                properties=properties or {},
            )
            return self._call(stub.AddNode, request)
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )

    def add_edge(
        self,
        graph_id: str,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        weight: float = 1.0,
        namespace: str = "default",
    ):
        """Add an edge to the graph."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.GraphServiceStub(self._channel)
            request = sochdb_pb2.AddEdgeRequest(
                namespace=namespace,
                graph_id=graph_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
            )
            return self._call(stub.AddEdge, request)
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )


class SqlClient(_BaseClient):
    """SQL query operations (via MCP or direct)."""

    def execute(self, query: str, params: Optional[list] = None) -> Dict[str, Any]:
        """
        Execute a SQL query.

        Returns a dict with 'columns' and 'rows' for SELECT,
        or 'rows_affected' for DML.
        """
        # SQL is currently routed through MCP service as a tool call
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.McpServiceStub(self._channel)
            request = sochdb_pb2.ToolCallRequest(
                tool_name="sql_execute",
                arguments={"query": query},
            )
            response = self._call(stub.CallTool, request)
            return {"result": response.result}
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )


class CollectionClient(_BaseClient):
    """Collection management."""

    def create(
        self,
        name: str,
        dimension: int,
        namespace: str = "default",
    ):
        """Create a new vector collection."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.CollectionServiceStub(self._channel)
            request = sochdb_pb2.CreateCollectionRequest(
                namespace=namespace,
                name=name,
                dimension=dimension,
            )
            return self._call(stub.CreateCollection, request)
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )

    def list(self, namespace: str = "default"):
        """List all collections."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.CollectionServiceStub(self._channel)
            request = sochdb_pb2.ListCollectionsRequest(namespace=namespace)
            response = self._call(stub.ListCollections, request)
            return [c.name for c in response.collections]
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )


class NamespaceClient(_BaseClient):
    """Namespace (multi-tenant) management."""

    def create(self, name: str):
        """Create a namespace."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.NamespaceServiceStub(self._channel)
            request = sochdb_pb2.CreateNamespaceRequest(name=name)
            return self._call(stub.CreateNamespace, request)
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )

    def list(self):
        """List all namespaces."""
        try:
            from .generated import sochdb_pb2, sochdb_pb2_grpc

            stub = sochdb_pb2_grpc.NamespaceServiceStub(self._channel)
            request = sochdb_pb2.ListNamespacesRequest()
            response = self._call(stub.ListNamespaces, request)
            return [ns.name for ns in response.namespaces]
        except ImportError:
            raise RuntimeError(
                "Generated stubs not found. Run: cd sochdb-sdk && ./generate.sh python"
            )
