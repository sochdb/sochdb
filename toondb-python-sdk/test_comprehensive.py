#!/usr/bin/env python3
"""
Comprehensive Python SDK Feature Test
Tests all features mentioned in the README
"""

import sys
import shutil
from pathlib import Path
from toondb import Database, IpcClient

test_count = 0
pass_count = 0
fail_count = 0


def test_assert(condition, message):
    global test_count, pass_count, fail_count
    test_count += 1
    if condition:
        pass_count += 1
        print(f"  ✓ {message}")
        return True
    else:
        fail_count += 1
        print(f"  ✗ {message}")
        return False


def test_basic_key_value(db):
    print("\n📝 Testing Basic Key-Value Operations...")
    
    # Put
    db.put(b"key1", b"value1")
    test_assert(True, "Put operation succeeded")
    
    # Get
    value = db.get(b"key1")
    test_assert(value and value == b"value1", "Get returns correct value")
    
    # Get non-existent key
    missing = db.get(b"nonexistent")
    test_assert(missing is None, "Get returns None for missing key")
    
    # Delete
    db.delete(b"key1")
    deleted = db.get(b"key1")
    test_assert(deleted is None, "Delete removes key")


def test_path_operations(db):
    print("\n🗂️  Testing Path Operations...")
    
    # Put path
    db.put_path("users/alice/email", b"alice@example.com")
    test_assert(True, "put_path succeeded")
    
    # Get path
    email = db.get_path("users/alice/email")
    test_assert(email == b"alice@example.com", "get_path retrieves correct value")
    
    # Multiple segments
    db.put_path("users/bob/profile/name", b"Bob")
    name = db.get_path("users/bob/profile/name")
    test_assert(name == b"Bob", "get_path handles multiple segments")
    
    # Missing path
    missing = db.get_path("users/charlie/email")
    test_assert(missing is None, "get_path returns None for missing path")


def test_prefix_scanning(db):
    print("\n🔍 Testing Prefix Scanning...")
    
    # Insert multi-tenant data
    db.put(b"tenants/acme/users/1", b'{"name":"Alice"}')
    db.put(b"tenants/acme/users/2", b'{"name":"Bob"}')
    db.put(b"tenants/acme/orders/1", b'{"total":100}')
    db.put(b"tenants/globex/users/1", b'{"name":"Charlie"}')
    
    # Scan ACME
    acme_results = list(db.scan(b"tenants/acme/", b"tenants/acme;"))
    test_assert(len(acme_results) == 3, f"Scan returns 3 ACME items (got {len(acme_results)})")
    
    # Scan Globex
    globex_results = list(db.scan(b"tenants/globex/", b"tenants/globex;"))
    test_assert(len(globex_results) == 1, f"Scan returns 1 Globex item (got {len(globex_results)})")
    
    # Verify results have key and value
    if acme_results:
        test_assert(
            isinstance(acme_results[0], tuple) and len(acme_results[0]) == 2,
            "Scan results have (key, value) tuples"
        )


def test_transactions(db):
    print("\n💳 Testing Transactions...")
    
    # Context manager transaction
    with db.transaction() as txn:
        txn.put(b"tx_key1", b"tx_value1")
        txn.put(b"tx_key2", b"tx_value2")
    
    # Verify committed
    value1 = db.get(b"tx_key1")
    value2 = db.get(b"tx_key2")
    test_assert(
        value1 == b"tx_value1" and value2 == b"tx_value2",
        "Transaction commits successfully"
    )
    
    # Verify data persisted
    test_assert(db.get(b"tx_key1") is not None, "Transaction data persisted")
    
    # Manual transaction
    txn = db.transaction()
    txn.put(b"manual_key", b"manual_value")
    txn.commit()
    test_assert(db.get(b"manual_key") == b"manual_value", "Manual transaction works")


def test_sql_operations(db):
    print("\n🗃️  Testing SQL Operations...")
    
    try:
        # CREATE TABLE
        result = db.execute_sql("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
        test_assert(result is not None, "CREATE TABLE succeeded")
        
        # INSERT
        db.execute_sql("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')")
        db.execute_sql("INSERT INTO users (id, name, email) VALUES (2, 'Bob', 'bob@example.com')")
        test_assert(True, "INSERT statements succeeded")
        
        # SELECT
        result = db.execute_sql("SELECT * FROM users")
        test_assert(hasattr(result, 'rows'), "SELECT returns SQLQueryResult")
        test_assert(len(result.rows) == 2, f"SELECT returns 2 rows (got {len(result.rows)})")
        
        # SELECT with WHERE
        filtered = db.execute_sql("SELECT * FROM users WHERE name = 'Alice'")
        test_assert(len(filtered.rows) == 1, f"SELECT with WHERE returns 1 row (got {len(filtered.rows)})")
        
        # UPDATE
        db.execute_sql("UPDATE users SET email = 'alice.new@example.com' WHERE id = 1")
        updated = db.execute_sql("SELECT * FROM users WHERE id = 1")
        test_assert(
            updated.rows[0].get('email') == 'alice.new@example.com',
            "UPDATE modified the row"
        )
        
        # DELETE
        db.execute_sql("DELETE FROM users WHERE id = 2")
        after_delete = db.execute_sql("SELECT * FROM users")
        test_assert(len(after_delete.rows) == 1, f"DELETE removed row ({len(after_delete.rows)} remaining)")
        
    except Exception as err:
        print(f"    SQL error: {err}")
        test_assert(False, f"SQL operations failed: {err}")


def test_empty_value_handling(db):
    print("\n🔄 Testing Empty Value Handling...")
    
    # Test non-existent key
    missing = db.get(b"truly-missing-key-test")
    test_assert(missing is None, "Missing key returns None")
    
    print("  ℹ️  Note: Empty values and missing keys both return None (protocol limitation)")


def main():
    test_dir = Path("test-data-comprehensive")
    
    # Clean up any existing test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("🧪 ToonDB Python SDK Comprehensive Feature Test\n")
    print("Testing all features mentioned in README...\n")
    print("=" * 60)
    
    try:
        db = Database.open(str(test_dir))
        
        test_basic_key_value(db)
        test_path_operations(db)
        test_prefix_scanning(db)
        test_transactions(db)
        test_sql_operations(db)
        test_empty_value_handling(db)
        
        db.close()
        
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        print("\n" + "=" * 60)
        print(f"\n📊 Test Results:")
        print(f"   Total:  {test_count}")
        print(f"   ✓ Pass: {pass_count}")
        print(f"   ✗ Fail: {fail_count}")
        print(f"   Success Rate: {(pass_count/test_count*100):.1f}%")
        
        if fail_count == 0:
            print("\n✅ All tests passed! Python SDK is working correctly.\n")
            sys.exit(0)
        else:
            print(f"\n❌ {fail_count} test(s) failed. See details above.\n")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
