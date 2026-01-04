# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ToonDB Python SDK

A Python client for ToonDB - the database optimized for LLM context retrieval.

Provides two modes of access:
- Embedded: Direct database access via FFI (single process)
- IPC: Client-server access via Unix sockets (multi-process)
- Vector: HNSW vector search (15x faster than ChromaDB)
"""

from .ipc_client import IpcClient
from .database import Database, Transaction
from .query import Query, SQLQueryResult
from .errors import (
    ToonDBError, 
    ConnectionError, 
    TransactionError, 
    ProtocolError,
    # Error taxonomy (Task 11)
    ErrorCode,
    NamespaceError,
    NamespaceNotFoundError,
    NamespaceExistsError,
    CollectionError,
    CollectionNotFoundError,
    CollectionExistsError,
    CollectionConfigError,
    ValidationError,
    DimensionMismatchError,
    QueryError,
)
from .namespace import (
    # Namespace handle (Task 8)
    Namespace,
    NamespaceConfig,
    # Collection (Task 9)
    Collection,
    CollectionConfig,
    DistanceMetric,
    QuantizationType,
    # Search (Task 10)
    SearchRequest,
    SearchResult,
    SearchResults,
)
from .context import (
    # ContextQuery (Task 12)
    ContextQuery,
    ContextResult,
    ContextChunk,
    TokenEstimator,
    DeduplicationStrategy,
    estimate_tokens,
    split_by_tokens,
)

# Vector search (optional - requires libtoondb_index)
try:
    from .vector import VectorIndex
except ImportError:
    VectorIndex = None

# Bulk operations (optional - requires toondb-bulk binary)
try:
    from .bulk import bulk_build_index, bulk_query_index, BulkBuildStats, QueryResult
except ImportError:
    bulk_build_index = None
    bulk_query_index = None
    BulkBuildStats = None
    QueryResult = None

# Analytics (optional - requires posthog)
try:
    from .analytics import (
        capture as capture_analytics,
        capture_error,
        shutdown as shutdown_analytics,
        track_database_open,
        track_vector_search,
        track_batch_insert,
        is_analytics_disabled,
    )
except ImportError:
    capture_analytics = None
    capture_error = None
    shutdown_analytics = None
    track_database_open = None
    track_vector_search = None
    track_batch_insert = None
    is_analytics_disabled = lambda: True

__version__ = "0.3.1"
__all__ = [
    # Core
    "Database",
    "Transaction", 
    "Query",
    "SQLQueryResult",
    "IpcClient",
    "VectorIndex",
    
    # Namespace (Task 8)
    "Namespace",
    "NamespaceConfig",
    
    # Collection (Task 9)
    "Collection",
    "CollectionConfig",
    "DistanceMetric",
    "QuantizationType",
    
    # Search (Task 10)
    "SearchRequest",
    "SearchResult",
    "SearchResults",
    
    # ContextQuery (Task 12)
    "ContextQuery",
    "ContextResult",
    "ContextChunk",
    "TokenEstimator",
    "DeduplicationStrategy",
    "estimate_tokens",
    "split_by_tokens",
    
    # Bulk operations
    "bulk_build_index",
    "bulk_query_index",
    "BulkBuildStats",
    "QueryResult",
    
    # Analytics (disabled with TOONDB_DISABLE_ANALYTICS=true)
    "capture_analytics",
    "capture_error",
    "shutdown_analytics",
    "track_database_open",
    "track_vector_search",
    "track_batch_insert",
    "is_analytics_disabled",
    
    # Errors
    "ToonDBError",
    "ConnectionError",
    "TransactionError",
    "ProtocolError",
    "ErrorCode",
    "NamespaceError",
    "NamespaceNotFoundError",
    "NamespaceExistsError",
    "CollectionError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "CollectionConfigError",
    "ValidationError",
    "DimensionMismatchError",
    "QueryError",
]
