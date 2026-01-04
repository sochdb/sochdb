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
ToonDB Analytics - Anonymous usage tracking with PostHog

This module provides anonymous, privacy-respecting analytics to help
improve ToonDB. All tracking can be disabled by setting:

    TOONDB_DISABLE_ANALYTICS=true

No personally identifiable information (PII) is collected. Only aggregate
usage patterns are tracked to understand:
- Which features are most used
- Performance characteristics
- Error patterns for debugging
"""

import os
import sys
import hashlib
import platform
from typing import Optional, Dict, Any
from functools import lru_cache

# PostHog configuration
POSTHOG_API_KEY = "phc_zf0hm6ZmPUJj1pM07Kigqvphh1ClhKX1NahRU4G0bfu"
POSTHOG_HOST = "https://us.i.posthog.com"


def _is_analytics_disabled() -> bool:
    """Check if analytics is disabled via environment variable."""
    disable_var = os.environ.get("TOONDB_DISABLE_ANALYTICS", "").lower()
    return disable_var in ("true", "1", "yes", "on")


# Public alias for checking analytics status
def is_analytics_disabled() -> bool:
    """
    Check if analytics tracking is disabled.
    
    Analytics is disabled when:
    - TOONDB_DISABLE_ANALYTICS environment variable is set to 'true', '1', 'yes', or 'on'
    - posthog package is not installed
    
    Returns:
        True if analytics is disabled, False otherwise.
    """
    return _is_analytics_disabled()


@lru_cache(maxsize=1)
def _get_anonymous_id() -> str:
    """
    Generate a stable anonymous ID for this machine.
    
    Uses a hash of machine-specific but non-identifying information.
    The same machine will always get the same ID, but the ID cannot
    be reversed to identify the machine.
    """
    try:
        # Combine various machine identifiers
        machine_info = [
            platform.node(),  # hostname (hashed, not sent raw)
            platform.machine(),
            platform.system(),
            str(os.getuid()) if hasattr(os, 'getuid') else "windows",
        ]
        combined = "|".join(machine_info)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    except Exception:
        return "anonymous"


@lru_cache(maxsize=1)
def _get_posthog_client():
    """Lazily initialize PostHog client."""
    if _is_analytics_disabled():
        return None
    
    try:
        from posthog import Posthog
        # Use sync_mode=True for immediate sending during development
        # In production, sync_mode=False is fine as we flush on shutdown
        return Posthog(
            POSTHOG_API_KEY,
            host=POSTHOG_HOST,
            sync_mode=True,  # Send immediately to ensure events aren't lost
        )
    except ImportError:
        # posthog package not installed - analytics disabled
        return None
    except Exception:
        # Any other error - fail silently
        return None


def capture(
    event: str,
    properties: Optional[Dict[str, Any]] = None,
    distinct_id: Optional[str] = None,
) -> None:
    """
    Capture an analytics event.
    
    Args:
        event: Event name (e.g., "database_opened", "vector_search")
        properties: Optional event properties
        distinct_id: Optional distinct ID (defaults to anonymous machine ID)
    
    This function is a no-op if:
    - TOONDB_DISABLE_ANALYTICS=true
    - posthog package is not installed
    - Any error occurs (fails silently)
    """
    if _is_analytics_disabled():
        return
    
    client = _get_posthog_client()
    if client is None:
        return
    
    try:
        # Build properties with SDK context
        event_properties = {
            "sdk": "python",
            "sdk_version": _get_sdk_version(),
            "python_version": platform.python_version(),
            "os": platform.system(),
            "arch": platform.machine(),
        }
        
        if properties:
            event_properties.update(properties)
        
        # PostHog Python API: capture(event, distinct_id=..., properties=...)
        client.capture(
            event,
            distinct_id=distinct_id or _get_anonymous_id(),
            properties=event_properties,
        )
        
        # Flush to ensure event is sent (especially important in sync_mode)
        client.flush()
    except Exception:
        # Never let analytics break user code
        pass
        # Never let analytics break user code
        pass


def capture_error(
    error_type: str,
    location: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Capture an error event for debugging.
    
    Only sends static information - no dynamic error messages.
    
    Args:
        error_type: Static error category (e.g., "connection_error", "query_error", "timeout_error")
        location: Static code location (e.g., "database.open", "query.execute", "transaction.commit")
        properties: Optional additional static properties only
    
    Example:
        capture_error("connection_error", "database.open")
        capture_error("query_error", "sql.execute", {"query_type": "SELECT"})
    """
    event_properties = {
        "error_type": error_type,
        "location": location,
    }
    if properties:
        event_properties.update(properties)
    
    capture("error", event_properties)


@lru_cache(maxsize=1)
def _get_sdk_version() -> str:
    """Get the ToonDB SDK version."""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def shutdown() -> None:
    """Flush any pending events and shutdown the client."""
    if _is_analytics_disabled():
        return
    
    client = _get_posthog_client()
    if client is not None:
        try:
            client.shutdown()
        except Exception:
            pass


# Convenience functions for common events
def track_database_open(db_path: str, mode: str = "embedded") -> None:
    """Track database open event."""
    capture("database_opened", {
        "mode": mode,
        "has_custom_path": db_path != ":memory:",
    })


def track_vector_search(dimension: int, k: int, latency_ms: float) -> None:
    """Track vector search event."""
    capture("vector_search", {
        "dimension": dimension,
        "k": k,
        "latency_ms": round(latency_ms, 2),
    })


def track_batch_insert(count: int, dimension: int, latency_ms: float) -> None:
    """Track batch insert event."""
    capture("batch_insert", {
        "count": count,
        "dimension": dimension,
        "latency_ms": round(latency_ms, 2),
    })
