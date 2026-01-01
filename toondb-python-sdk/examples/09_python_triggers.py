#!/usr/bin/env python3
"""
Example: Python Plugin Triggers

Demonstrates how to use Python plugins as database triggers for:
- Validation (email format, required fields)
- Transformation (normalize data)
- Fraud detection (ML-based)
- Audit logging

Run with:
    python examples/09_python_triggers.py
"""

from toondb.plugins import (
    PythonPlugin,
    PluginRegistry,
    TriggerEvent,
    TriggerAbort,
    RuntimeConfig,
    generate_trigger_code,
)

print("=" * 60)
print("  ToonDB Python Plugin System - Demo")
print("=" * 60)
print()

# ============================================================================
# Example 1: Simple Email Validation
# ============================================================================

print("Example 1: Email Validation Trigger")
print("-" * 40)

email_validator = PythonPlugin(
    name="email_validator",
    code='''
def on_before_insert(row: dict) -> dict:
    """Validate and normalize email addresses."""
    email = row.get("email", "")
    
    # Validation
    if "@" not in email:
        raise TriggerAbort("Invalid email format", code="INVALID_EMAIL")
    
    # Check for disposable domains
    disposable = ["tempmail.com", "throwaway.com", "10minutemail.com"]
    domain = email.split("@")[1].lower()
    if domain in disposable:
        raise TriggerAbort("Disposable email not allowed", code="DISPOSABLE_EMAIL")
    
    # Normalization
    row["email"] = email.lower().strip()
    
    return row
''',
    triggers={"users": ["BEFORE INSERT", "BEFORE UPDATE"]}
)

# Create registry and register plugin
registry = PluginRegistry()
registry.register(email_validator)

print(f"Registered plugins: {registry.list_plugins()}")

# Test with valid email
valid_row = {"name": "Alice", "email": "ALICE@Example.COM"}
result = registry.fire("users", TriggerEvent.BEFORE_INSERT, valid_row)
print(f"Valid email result: {result}")
# Output: {'name': 'Alice', 'email': 'alice@example.com'}

# Test with invalid email
try:
    invalid_row = {"name": "Bob", "email": "not-an-email"}
    registry.fire("users", TriggerEvent.BEFORE_INSERT, invalid_row)
except TriggerAbort as e:
    print(f"Invalid email caught: {e} (code: {e.code})")

print()

# ============================================================================
# Example 2: Amount Limit Validation
# ============================================================================

print("Example 2: Transaction Amount Validation")
print("-" * 40)

amount_validator = PythonPlugin(
    name="amount_validator",
    code='''
def on_before_insert(row: dict) -> dict:
    """Validate transaction amounts."""
    amount = row.get("amount", 0)
    
    # Basic limit
    if amount > 10000:
        raise TriggerAbort(
            f"Amount ${amount} exceeds limit of $10,000",
            code="LIMIT_EXCEEDED"
        )
    
    # Suspicious amount patterns
    if amount > 9000 and amount < 9100:
        row["flagged"] = True
        row["flag_reason"] = "Suspicious amount pattern"
    
    return row
''',
    triggers={"transactions": ["BEFORE INSERT"]}
)

registry.register(amount_validator)

# Test normal transaction
normal_tx = {"amount": 500, "merchant": "Coffee Shop"}
result = registry.fire("transactions", TriggerEvent.BEFORE_INSERT, normal_tx)
print(f"Normal transaction: {result}")

# Test suspicious transaction
suspicious_tx = {"amount": 9050, "merchant": "Unknown"}
result = registry.fire("transactions", TriggerEvent.BEFORE_INSERT, suspicious_tx)
print(f"Suspicious transaction: {result}")

# Test over-limit transaction
try:
    over_limit_tx = {"amount": 50000, "merchant": "Luxury Store"}
    registry.fire("transactions", TriggerEvent.BEFORE_INSERT, over_limit_tx)
except TriggerAbort as e:
    print(f"Over-limit caught: {e}")

print()

# ============================================================================
# Example 3: AI-Generated Trigger (Placeholder)
# ============================================================================

print("Example 3: AI-Generated Trigger")
print("-" * 40)

# Generate trigger from natural language
ai_code = generate_trigger_code(
    instruction="Validate that order total equals sum of line items, "
                "apply 10% discount for premium customers"
)
print("Generated code:")
print(ai_code)

print()

# ============================================================================
# Example 4: Audit Logging (AFTER INSERT)
# ============================================================================

print("Example 4: Audit Logging")
print("-" * 40)

# Track what gets logged
audit_log = []

audit_logger = PythonPlugin(
    name="audit_logger",
    code='''
def on_after_insert(row: dict) -> dict:
    """Log all inserts for audit trail."""
    import json
    from datetime import datetime
    
    # In production, this would write to an audit table
    log_entry = {
        "event": "INSERT",
        "table": "orders",
        "data": row,
        "timestamp": datetime.now().isoformat()
    }
    print(f"[AUDIT] {json.dumps(log_entry, indent=2)}")
    return row
''',
    triggers={"orders": ["AFTER INSERT"]}
)

registry.register(audit_logger)

# Test audit logging
order = {"order_id": 12345, "customer": "Alice", "total": 99.99}
registry.fire("orders", TriggerEvent.AFTER_INSERT, order)

print()

# ============================================================================
# Example 5: ML-Ready Plugin Config
# ============================================================================

print("Example 5: ML Plugin Configuration")
print("-" * 40)

# This shows how you'd configure a plugin for ML workloads
# In production, the actual numpy/pandas code runs in WASM

ml_plugin = PythonPlugin(
    name="fraud_detector",
    version="2.0.0",
    packages=["numpy", "pandas", "scikit-learn"],
    code='''
# In production, this runs with full numpy/pandas support
import numpy as np
import pandas as pd

# Simulated model (in reality, load from file)
THRESHOLD = 0.7

def on_before_insert(row: dict) -> dict:
    """ML-based fraud detection."""
    features = [
        row.get("amount", 0) / 10000,  # Normalized amount
        1 if row.get("is_international", False) else 0,
        row.get("hour", 12) / 24,  # Normalized hour
    ]
    
    # Simulated model prediction
    risk_score = sum(features) / len(features)
    
    if risk_score > THRESHOLD:
        raise TriggerAbort(f"Fraud risk score {risk_score:.2f} exceeds threshold")
    
    row["risk_score"] = risk_score
    return row
''',
    config=RuntimeConfig.with_ml_packages(),
    triggers={"payments": ["BEFORE INSERT"]}
)

print(f"ML Plugin config:")
print(f"  Memory limit: {ml_plugin.config.memory_limit_mb} MB")
print(f"  Timeout: {ml_plugin.config.timeout_ms} ms")
print(f"  Packages: {ml_plugin.config.packages}")
print(f"  Plugin packages: {ml_plugin.packages}")

print()
print("=" * 60)
print("  ✅ All examples completed successfully!")
print("=" * 60)
