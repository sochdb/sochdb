# SPDX-License-Identifier: AGPL-3.0-or-later
"""SochDB Python SDK — Thin gRPC client for SochDB."""

from .client import (
    SochDB,
    VectorClient,
    KvClient,
    GraphClient,
    SqlClient,
    CollectionClient,
    NamespaceClient,
)

__version__ = "2.0.0"
__all__ = [
    "SochDB",
    "VectorClient",
    "KvClient",
    "GraphClient",
    "SqlClient",
    "CollectionClient",
    "NamespaceClient",
]
