#!/usr/bin/env python3
"""
Test script to verify Python SDK get() behavior
Tests that get() returns None for non-existent keys
"""

import sys
import os
import tempfile
import shutil

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from toondb import Database

def test_get_returns_none():
    """Test that get() returns None for missing keys"""
    test_dir = tempfile.mkdtemp(prefix='toondb_test_')
    
    try:
        print("🧪 Testing Python SDK get() behavior...\n")
        
        # Test 1: get() for non-existent key
        print("Test 1: get() returns None for missing keys")
        db = Database.open(test_dir)
        
        result = db.get(b'non_existent_key')
        print(f"  Result for missing key: {result!r}")
        
        if result is None:
            print("  ✓ PASS: get() returns None for missing keys\n")
        else:
            print(f"  ❌ FAIL: Expected None, got: {result!r}\n")
            db.close()
            return False
        
        # Test 2: get() for key with empty value
        print("Test 2: get() for key with empty bytes value")
        db.put(b'empty_key', b'')
        result = db.get(b'empty_key')
        print(f"  Result for empty value: {result!r}")
        
        if result == b'':
            print("  ✓ PASS: get() returns empty bytes for empty value\n")
        else:
            print(f"  ❌ FAIL: Expected b'', got: {result!r}\n")
            db.close()
            return False
        
        # Test 3: get() for key with actual value
        print("Test 3: get() for key with value")
        db.put(b'test_key', b'test_value')
        result = db.get(b'test_key')
        print(f"  Result for existing key: {result!r}")
        
        if result == b'test_value':
            print("  ✓ PASS: get() returns correct value\n")
        else:
            print(f"  ❌ FAIL: Expected b'test_value', got: {result!r}\n")
            db.close()
            return False
        
        db.close()
        print("✅ All tests passed!")
        return True
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == '__main__':
    success = test_get_returns_none()
    sys.exit(0 if success else 1)
