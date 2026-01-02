#!/usr/bin/env python3
"""Test IPC client get() behavior"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test Python truthiness
empty_bytes = b''
print(f"Empty bytes b'': {empty_bytes!r}")
print(f"Truthy? {bool(empty_bytes)}")
print(f"Expression 'b\"\" if b\"\" else None': {empty_bytes if empty_bytes else None}")
print(f"\nSo the expression would return None for empty bytes!")
