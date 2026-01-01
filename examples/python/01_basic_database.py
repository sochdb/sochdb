#!/usr/bin/env python3
"""
ToonDB Basic Usage Example
==========================

This example demonstrates the core functionality of ToonDB:
1. Database operations (put, get, delete)
2. Path-based access
3. Transactions
4. Range scanning

Expected Output:
    ✓ Basic put/get works
    ✓ Path-based access works
    ✓ Transaction commits correctly
    ✓ Range scan works
    ✓ Database stats available

Usage:
    PYTHONPATH=toondb-python-sdk/src TOONDB_LIB_PATH=target/release python3 examples/python/01_basic_database.py
"""

import os
import sys
import tempfile

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../toondb-python-sdk/src"))


def main():
    print("=" * 60)
    print("  ToonDB Basic Database Example")
    print("=" * 60)
    
    try:
        from toondb import Database
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Set TOONDB_LIB_PATH to target/release directory")
        return 1
    
    # Create temp directory for test database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        
        # 1. Open database
        print("\n[1] Opening database...")
        db = Database.open(db_path)
        print(f"    Database opened at: {db_path}")
        
        # 2. Basic put/get
        print("\n[2] Testing basic put/get...")
        db.put(b"greeting", b"Hello, ToonDB!")
        value = db.get(b"greeting")
        assert value == b"Hello, ToonDB!", f"Expected 'Hello, ToonDB!', got {value}"
        print("    ✓ Basic put/get works")
        
        # 3. Path-based access
        print("\n[3] Testing path-based access...")
        db.put_path("users/alice/name", b"Alice Smith")
        db.put_path("users/alice/email", b"alice@example.com")
        db.put_path("users/bob/name", b"Bob Jones")
        
        name = db.get_path("users/alice/name")
        assert name == b"Alice Smith", f"Expected 'Alice Smith', got {name}"
        print("    ✓ Path-based access works")
        
        # 4. Transactions
        print("\n[4] Testing transactions...")
        with db.transaction() as txn:
            txn.put(b"counter", b"100")
            txn.put(b"updated", b"true")
        
        counter = db.get(b"counter")
        assert counter == b"100", f"Expected '100', got {counter}"
        print("    ✓ Transaction commits correctly")
        
        # 5. Range scanning
        print("\n[5] Testing range scan...")
        users = list(db.scan(b"users/", b"users/~"))
        print(f"    Found {len(users)} user entries:")
        for key, value in users:
            print(f"      {key.decode()}: {value.decode()}")
        assert len(users) >= 3, f"Expected at least 3 users, got {len(users)}"
        print("    ✓ Range scan works")
        
        # 6. Database stats
        print("\n[6] Getting database stats...")
        try:
            stats = db.stats()
            print(f"    Memtable size: {stats.get('memtable_size_bytes', 'N/A')} bytes")
            print("    ✓ Database stats available")
        except Exception as e:
            print(f"    ⚠ Stats not available: {e}")
        
        # 7. Checkpoint
        print("\n[7] Creating checkpoint...")
        db.checkpoint()
        print("    ✓ Checkpoint created")
        
        # Cleanup
        db.close()
        print("\n[8] Database closed successfully")
    
    print("\n" + "=" * 60)
    print("  ✅ All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
