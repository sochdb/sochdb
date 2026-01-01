#!/usr/bin/env python3
"""
Example 01: Basic Operations
============================

This example demonstrates fundamental CRUD operations with ToonDB:
- Opening/closing a database
- Put, Get, Delete operations
- Using context managers for safe cleanup

Difficulty: Beginner
Mode: Embedded (FFI)
"""

import os
import shutil
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from toondb import Database
from toondb.errors import DatabaseError

# Database directory for this example
DB_PATH = "./example_01_db"


def cleanup():
    """Clean up any existing database from previous runs."""
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    print(f"✓ Cleaned up {DB_PATH}")


def example_basic_crud():
    """Basic Create, Read, Update, Delete operations."""
    print("\n" + "=" * 60)
    print("Example 1.1: Basic CRUD Operations")
    print("=" * 60)
    
    # Open a database (creates if doesn't exist)
    db = Database.open(DB_PATH)
    print(f"✓ Opened database at {DB_PATH}")
    
    # -------------------------------------------------------------------------
    # CREATE (Put)
    # -------------------------------------------------------------------------
    print("\n--- PUT Operations ---")
    
    # Simple key-value storage
    db.put(b"user:1", b"Alice")
    db.put(b"user:2", b"Bob")
    db.put(b"user:3", b"Charlie")
    print("✓ Stored 3 users")
    
    # Store binary data
    db.put(b"config:theme", b"dark")
    db.put(b"config:lang", b"en-US")
    print("✓ Stored configuration")
    
    # -------------------------------------------------------------------------
    # READ (Get)
    # -------------------------------------------------------------------------
    print("\n--- GET Operations ---")
    
    # Get existing keys
    user1 = db.get(b"user:1")
    print(f"  user:1 = {user1.decode('utf-8')}")  # Alice
    
    user2 = db.get(b"user:2")
    print(f"  user:2 = {user2.decode('utf-8')}")  # Bob
    
    # Get non-existent key returns None
    missing = db.get(b"user:999")
    print(f"  user:999 = {missing}")  # None
    
    # -------------------------------------------------------------------------
    # UPDATE (Put with same key)
    # -------------------------------------------------------------------------
    print("\n--- UPDATE Operations ---")
    
    # Update overwrites existing value
    db.put(b"user:1", b"Alice Smith")
    updated_user = db.get(b"user:1")
    print(f"  Updated user:1 = {updated_user.decode('utf-8')}")  # Alice Smith
    
    # -------------------------------------------------------------------------
    # DELETE
    # -------------------------------------------------------------------------
    print("\n--- DELETE Operations ---")
    
    # Delete a key
    db.delete(b"user:3")
    deleted = db.get(b"user:3")
    print(f"  After delete, user:3 = {deleted}")  # None
    
    # Deleting non-existent key is safe (no error)
    db.delete(b"user:nonexistent")
    print("✓ Deleting non-existent key: no error")
    
    # Clean up
    db.close()
    print("\n✓ Database closed")


def example_context_manager():
    """Using context manager for automatic cleanup."""
    print("\n" + "=" * 60)
    print("Example 1.2: Context Manager Pattern")
    print("=" * 60)
    
    # Using 'with' ensures database is closed even if an error occurs
    with Database.open(DB_PATH) as db:
        # Store some data
        db.put(b"temp:key1", b"temporary value 1")
        db.put(b"temp:key2", b"temporary value 2")
        
        # Read back
        val = db.get(b"temp:key1")
        print(f"  Read: {val.decode('utf-8')}")
    
    # Database is automatically closed here
    print("✓ Database automatically closed by context manager")


def example_error_handling():
    """Proper error handling patterns."""
    print("\n" + "=" * 60)
    print("Example 1.3: Error Handling")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        try:
            # Normal operation
            db.put(b"test:key", b"test value")
            value = db.get(b"test:key")
            print(f"  Read value: {value.decode('utf-8')}")
            
        except DatabaseError as e:
            print(f"  Database error: {e}")
        except Exception as e:
            print(f"  Unexpected error: {e}")
    
    print("✓ Error handling complete")


def example_binary_data():
    """Storing various binary data types."""
    print("\n" + "=" * 60)
    print("Example 1.4: Binary Data")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Store raw bytes
        raw_bytes = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD])
        db.put(b"binary:raw", raw_bytes)
        
        # Store serialized data (e.g., from pickle, msgpack, protobuf)
        import json
        data = {"name": "Alice", "age": 30, "active": True}
        json_bytes = json.dumps(data).encode('utf-8')
        db.put(b"binary:json", json_bytes)
        
        # Read back
        retrieved_raw = db.get(b"binary:raw")
        retrieved_json = db.get(b"binary:json")
        
        print(f"  Raw bytes: {list(retrieved_raw)}")
        print(f"  JSON data: {json.loads(retrieved_json.decode('utf-8'))}")
    
    print("✓ Binary data handling complete")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ToonDB Python SDK - Example 01: Basic Operations")
    print("=" * 60)
    
    # Clean up from previous runs
    cleanup()
    
    # Run examples
    example_basic_crud()
    example_context_manager()
    example_error_handling()
    example_binary_data()
    
    # Final cleanup
    cleanup()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
