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
ToonDB Error Types

SDK Error Taxonomy (Task 16): Cross-language consistent error types
with machine-readable codes and actionable remediation messages.

Error Code Ranges:
- 1xxx: Connection/Transport errors
- 2xxx: Transaction errors  
- 3xxx: Namespace/Scope errors
- 4xxx: Collection errors
- 5xxx: Query errors
- 6xxx: Validation errors
- 7xxx: Resource errors
- 8xxx: Authorization errors
- 9xxx: Internal errors
"""

from enum import IntEnum
from typing import Optional, Dict, Any


class ErrorCode(IntEnum):
    """Machine-readable error codes matching Rust error codes."""
    
    # Connection errors (1xxx)
    CONNECTION_FAILED = 1001
    CONNECTION_TIMEOUT = 1002
    CONNECTION_CLOSED = 1003
    PROTOCOL_ERROR = 1004
    
    # Transaction errors (2xxx)
    TRANSACTION_ABORTED = 2001
    TRANSACTION_CONFLICT = 2002
    TRANSACTION_EXPIRED = 2003
    TRANSACTION_READ_ONLY = 2004
    TRANSACTION_NOT_FOUND = 2005
    
    # Namespace errors (3xxx)
    NAMESPACE_NOT_FOUND = 3001
    NAMESPACE_ALREADY_EXISTS = 3002
    NAMESPACE_INVALID_NAME = 3003
    NAMESPACE_ACCESS_DENIED = 3004
    NAMESPACE_READ_ONLY = 3005
    
    # Collection errors (4xxx)
    COLLECTION_NOT_FOUND = 4001
    COLLECTION_ALREADY_EXISTS = 4002
    COLLECTION_INVALID_CONFIG = 4003
    COLLECTION_FROZEN = 4004
    
    # Query errors (5xxx)
    QUERY_INVALID = 5001
    QUERY_TIMEOUT = 5002
    QUERY_CANCELLED = 5003
    QUERY_TOO_LARGE = 5004
    
    # Validation errors (6xxx)
    INVALID_VECTOR_DIMENSION = 6001
    INVALID_METADATA = 6002
    INVALID_ID = 6003
    INVALID_FILTER = 6004
    MISSING_REQUIRED_FIELD = 6005
    
    # Resource errors (7xxx)
    NOT_FOUND = 7001
    ALREADY_EXISTS = 7002
    QUOTA_EXCEEDED = 7003
    RESOURCE_EXHAUSTED = 7004
    
    # Authorization errors (8xxx)
    UNAUTHORIZED = 8001
    FORBIDDEN = 8002
    TOKEN_EXPIRED = 8003
    SCOPE_VIOLATION = 8004
    
    # Internal errors (9xxx)
    INTERNAL_ERROR = 9001
    NOT_IMPLEMENTED = 9002
    STORAGE_ERROR = 9003
    FFI_ERROR = 9004


class ToonDBError(Exception):
    """
    Base exception for ToonDB errors.
    
    All ToonDB exceptions inherit from this class, providing:
    - Machine-readable error codes
    - Human-readable messages
    - Optional remediation hints
    - Optional context data
    
    Example:
        try:
            db.get_namespace("missing")
        except NamespaceError as e:
            print(f"Error {e.code}: {e.message}")
            print(f"Remediation: {e.remediation}")
    """
    
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    
    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        remediation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.remediation = remediation
        self.context = context or {}
    
    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "remediation": self.remediation,
            "context": self.context,
        }


class ConnectionError(ToonDBError):
    """Failed to connect to ToonDB server or open database."""
    code = ErrorCode.CONNECTION_FAILED


class TransactionError(ToonDBError):
    """Transaction-related error."""
    code = ErrorCode.TRANSACTION_ABORTED


class TransactionConflictError(TransactionError):
    """Transaction aborted due to write-write conflict."""
    code = ErrorCode.TRANSACTION_CONFLICT
    
    def __init__(self, message: str = "Transaction aborted due to conflict"):
        super().__init__(
            message,
            remediation="Retry the transaction. Consider using optimistic locking or reducing contention."
        )


class ProtocolError(ToonDBError):
    """Wire protocol error."""
    code = ErrorCode.PROTOCOL_ERROR


class DatabaseError(ToonDBError):
    """Database operation error."""
    code = ErrorCode.INTERNAL_ERROR


# ============================================================================
# Namespace Errors (Task 3: Multi-tenancy)
# ============================================================================

class NamespaceError(ToonDBError):
    """Base class for namespace-related errors."""
    
    @property
    def namespace(self) -> Optional[str]:
        """Get the namespace from context."""
        return self.context.get("namespace")


class NamespaceNotFoundError(NamespaceError):
    """Namespace does not exist."""
    code = ErrorCode.NAMESPACE_NOT_FOUND
    
    def __init__(self, namespace: str):
        super().__init__(
            f"Namespace not found: {namespace}",
            remediation=f"Create the namespace first with db.create_namespace('{namespace}')",
            context={"namespace": namespace},
        )


class NamespaceExistsError(NamespaceError):
    """Namespace already exists."""
    code = ErrorCode.NAMESPACE_ALREADY_EXISTS
    
    def __init__(self, namespace: str):
        super().__init__(
            f"Namespace already exists: {namespace}",
            remediation=f"Use db.get_namespace('{namespace}') to access existing namespace",
            context={"namespace": namespace},
        )


class NamespaceAccessError(NamespaceError):
    """Access denied to namespace."""
    code = ErrorCode.NAMESPACE_ACCESS_DENIED
    
    def __init__(self, namespace: str, reason: str = "insufficient permissions"):
        super().__init__(
            f"Access denied to namespace '{namespace}': {reason}",
            remediation="Check your capability token includes access to this namespace",
            context={"namespace": namespace, "reason": reason},
        )


# ============================================================================
# Collection Errors
# ============================================================================

class CollectionError(ToonDBError):
    """Base class for collection-related errors."""
    
    @property
    def collection(self) -> Optional[str]:
        """Get the collection from context."""
        return self.context.get("collection")
    
    @property
    def namespace(self) -> Optional[str]:
        """Get the namespace from context."""
        return self.context.get("namespace")


class CollectionNotFoundError(CollectionError):
    """Collection does not exist."""
    code = ErrorCode.COLLECTION_NOT_FOUND
    
    def __init__(self, collection: str, namespace: Optional[str] = None):
        ns_prefix = f"in namespace '{namespace}' " if namespace else ""
        super().__init__(
            f"Collection not found {ns_prefix}: {collection}",
            remediation=f"Create the collection first with collection = ns.create_collection('{collection}', ...)",
            context={"collection": collection, "namespace": namespace},
        )


class CollectionExistsError(CollectionError):
    """Collection already exists."""
    code = ErrorCode.COLLECTION_ALREADY_EXISTS
    
    def __init__(self, collection: str, namespace: Optional[str] = None):
        ns_prefix = f"in namespace '{namespace}' " if namespace else ""
        super().__init__(
            f"Collection already exists {ns_prefix}: {collection}",
            remediation=f"Use ns.get_collection('{collection}') to access existing collection",
            context={"collection": collection, "namespace": namespace},
        )


class CollectionConfigError(CollectionError):
    """Invalid collection configuration."""
    code = ErrorCode.COLLECTION_INVALID_CONFIG


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(ToonDBError):
    """Base class for validation errors."""
    pass


class DimensionMismatchError(ValidationError):
    """Vector dimension does not match collection configuration."""
    code = ErrorCode.INVALID_VECTOR_DIMENSION
    
    def __init__(self, expected: int, actual: int):
        super().__init__(
            f"Vector dimension mismatch: expected {expected}, got {actual}",
            remediation=f"Ensure all vectors have exactly {expected} dimensions",
            context={"expected": expected, "actual": actual},
        )


class InvalidMetadataError(ValidationError):
    """Invalid metadata format."""
    code = ErrorCode.INVALID_METADATA


class ScopeViolationError(ValidationError):
    """Operation violates namespace scope."""
    code = ErrorCode.SCOPE_VIOLATION
    
    def __init__(self, message: str = "Cross-namespace operation not allowed"):
        super().__init__(
            message,
            remediation="Use scan_prefix() instead of scan() for namespace-safe iteration. "
                       "Ensure all operations are within a single namespace context.",
        )


# ============================================================================
# Query Errors
# ============================================================================

class QueryError(ToonDBError):
    """Base class for query errors."""
    pass


class QueryTimeoutError(QueryError):
    """Query execution timed out."""
    code = ErrorCode.QUERY_TIMEOUT
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            f"Query timed out after {timeout_seconds}s",
            remediation="Increase timeout, add filters to reduce result set, or use pagination",
            context={"timeout_seconds": timeout_seconds},
        )


class EmbeddingError(ToonDBError):
    """Error related to embedding generation."""
    code = ErrorCode.NOT_IMPLEMENTED
    
    def __init__(self, message: str = "Embedding feature not available"):
        super().__init__(
            message,
            remediation="Install embeddings extra: pip install toondb-client[embeddings]",
        )


# ============================================================================
# Error Mapping from Rust
# ============================================================================

# Map Rust error codes to Python exceptions
_ERROR_MAP: Dict[int, type] = {
    ErrorCode.CONNECTION_FAILED: ConnectionError,
    ErrorCode.TRANSACTION_ABORTED: TransactionError,
    ErrorCode.TRANSACTION_CONFLICT: TransactionConflictError,
    ErrorCode.NAMESPACE_NOT_FOUND: NamespaceNotFoundError,
    ErrorCode.NAMESPACE_ALREADY_EXISTS: NamespaceExistsError,
    ErrorCode.NAMESPACE_ACCESS_DENIED: NamespaceAccessError,
    ErrorCode.COLLECTION_NOT_FOUND: CollectionNotFoundError,
    ErrorCode.COLLECTION_ALREADY_EXISTS: CollectionExistsError,
    ErrorCode.INVALID_VECTOR_DIMENSION: DimensionMismatchError,
    ErrorCode.SCOPE_VIOLATION: ScopeViolationError,
    ErrorCode.QUERY_TIMEOUT: QueryTimeoutError,
}


def from_rust_error(code: int, message: str, context: Optional[Dict[str, Any]] = None) -> ToonDBError:
    """
    Convert a Rust error code to the appropriate Python exception.
    
    This is used by FFI bindings to map Rust errors to typed Python exceptions.
    """
    error_class = _ERROR_MAP.get(code, ToonDBError)
    
    # Handle special cases that need constructor arguments
    if error_class == NamespaceNotFoundError and context and "namespace" in context:
        return NamespaceNotFoundError(context["namespace"])
    if error_class == CollectionNotFoundError and context and "collection" in context:
        return CollectionNotFoundError(context["collection"], context.get("namespace"))
    if error_class == DimensionMismatchError and context:
        return DimensionMismatchError(context.get("expected", 0), context.get("actual", 0))
    
    # Generic case
    try:
        error_code = ErrorCode(code)
    except ValueError:
        error_code = ErrorCode.INTERNAL_ERROR
    
    return error_class(message, code=error_code, context=context)

