#!/usr/bin/env python3
"""
SochDB Basic Usage Example
==========================

This example demonstrates the core functionality of SochDB:
1. Database operations (put, get, delete)
2. Prefix-based access
3. Transactions
4. Prefix scanning

Usage:
    python3 examples/python/01_basic_database.py
"""

import os
import sys
import tempfile


def main():
    print("=" * 60)
    print("  SochDB Basic Database Example")
    print("=" * 60)

    try:
        import sochdb
        from sochdb import Database, Transaction
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Install with: pip install sochdb")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")

        # 1. Open database
        print("\n[1] Opening database...")
        db = Database.open(db_path)
        print(f"    Database opened at: {db_path}")

        # 2. Basic put/get
        print("\n[2] Testing basic put/get...")
        db.put(b"greeting", b"Hello, SochDB!")
        value = db.get(b"greeting")
        assert value == b"Hello, SochDB!", f"Expected 'Hello, SochDB!', got {value}"
        print("    ✓ Basic put/get works")

        # 3. Prefix-based access
        print("\n[3] Testing prefix-based access...")
        db.put(b"users/alice/name", b"Alice Smith")
        db.put(b"users/alice/email", b"alice@example.com")
        db.put(b"users/bob/name", b"Bob Jones")

        name = db.get(b"users/alice/name")
        assert name == b"Alice Smith", f"Expected 'Alice Smith', got {name}"
        print("    ✓ Prefix-based access works")

        # 4. Transactions
        print("\n[4] Testing transactions...")
        txn = Transaction(db)
        with txn as t:
            db.put(b"counter", b"100", txn=t.id)
            db.put(b"updated", b"true", txn=t.id)

        counter = db.get(b"counter")
        assert counter == b"100", f"Expected '100', got {counter}"
        print("    ✓ Transaction commits correctly")

        # 5. Prefix scanning
        print("\n[5] Testing prefix scan...")
        users = list(db.scan(b"users/"))
        print(f"    Found {len(users)} user entries:")
        for key, value in users:
            print(f"      {key.decode()}: {value.decode()}")
        assert len(users) >= 3, f"Expected at least 3 users, got {len(users)}"
        print("    ✓ Prefix scan works")

        # 6. Checkpoint & GC
        print("\n[6] Checkpoint and GC...")
        cp = db.checkpoint()
        print(f"    Checkpoint seq: {cp}")
        gc = db.gc()
        print(f"    GC collected: {gc} versions")
        db.fsync()
        print("    ✓ Checkpoint, GC, and fsync work")

    print("\n" + "=" * 60)
    print("  ✅ All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())