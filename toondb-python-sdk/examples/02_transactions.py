#!/usr/bin/env python3
"""
Example 02: Transactions
========================

This example demonstrates transaction handling in ToonDB:
- Beginning and committing transactions
- Automatic commit with context managers
- Manual abort/rollback
- MVCC snapshot isolation
- Batch operations within transactions

Difficulty: Intermediate
Mode: Embedded (FFI)
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from toondb import Database, Transaction
from toondb.errors import TransactionError, DatabaseError

DB_PATH = "./example_02_db"


def cleanup():
    """Clean up database."""
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    print(f"✓ Cleaned up {DB_PATH}")


def example_basic_transaction():
    """Basic transaction with context manager (auto-commit)."""
    print("\n" + "=" * 60)
    print("Example 2.1: Basic Transaction (Auto-Commit)")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Transaction with context manager - auto-commits on success
        with db.transaction() as txn:
            txn.put(b"account:1:balance", b"1000")
            txn.put(b"account:2:balance", b"500")
            print("  Wrote account balances in transaction")
        # Transaction automatically commits here
        
        # Verify data is committed
        balance1 = db.get(b"account:1:balance")
        balance2 = db.get(b"account:2:balance")
        print(f"  Account 1: {balance1.decode('utf-8')}")
        print(f"  Account 2: {balance2.decode('utf-8')}")
    
    print("✓ Transaction auto-committed")


def example_manual_commit():
    """Manual transaction control."""
    print("\n" + "=" * 60)
    print("Example 2.2: Manual Transaction Control")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Begin transaction manually
        txn = db.transaction()
        print(f"  Transaction ID: {txn.id}")
        
        # Perform operations
        txn.put(b"order:1001", b"pending")
        txn.put(b"order:1002", b"processing")
        
        # Read within transaction sees uncommitted changes
        order1 = txn.get(b"order:1001")
        print(f"  Within txn, order:1001 = {order1.decode('utf-8')}")
        
        # Explicit commit
        commit_ts = txn.commit()
        print(f"  Committed at timestamp: {commit_ts}")
        
        # Verify outside transaction
        order1_after = db.get(b"order:1001")
        print(f"  After commit, order:1001 = {order1_after.decode('utf-8')}")
    
    print("✓ Manual transaction complete")


def example_transaction_abort():
    """Transaction abort/rollback."""
    print("\n" + "=" * 60)
    print("Example 2.3: Transaction Abort (Rollback)")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # First, set initial value
        db.put(b"counter", b"0")
        initial = db.get(b"counter")
        print(f"  Initial counter: {initial.decode('utf-8')}")
        
        # Start transaction that we will abort
        txn = db.transaction()
        
        # Make changes within transaction
        txn.put(b"counter", b"100")
        
        # Read within transaction sees the change
        in_txn = txn.get(b"counter")
        print(f"  Within txn, counter = {in_txn.decode('utf-8')}")
        
        # Abort the transaction
        txn.abort()
        print("  Transaction aborted")
        
        # Verify original value is preserved
        after_abort = db.get(b"counter")
        print(f"  After abort, counter = {after_abort.decode('utf-8')}")
    
    print("✓ Transaction abort complete - changes rolled back")


def example_exception_rollback():
    """Automatic rollback on exception."""
    print("\n" + "=" * 60)
    print("Example 2.4: Automatic Rollback on Exception")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Set initial value
        db.put(b"important_data", b"original")
        print(f"  Initial: {db.get(b'important_data').decode('utf-8')}")
        
        try:
            with db.transaction() as txn:
                # Make some changes
                txn.put(b"important_data", b"modified")
                print("  Modified within transaction")
                
                # Simulate an error
                raise ValueError("Simulated application error!")
                
                # This line never executes
                txn.put(b"important_data", b"final")
                
        except ValueError as e:
            print(f"  Caught exception: {e}")
        
        # Transaction was automatically aborted
        final_value = db.get(b"important_data")
        print(f"  After exception, value = {final_value.decode('utf-8')}")
    
    print("✓ Automatic rollback preserved data integrity")


def example_money_transfer():
    """Real-world example: Atomic money transfer."""
    print("\n" + "=" * 60)
    print("Example 2.5: Atomic Money Transfer")
    print("=" * 60)
    
    def transfer_money(db, from_account: bytes, to_account: bytes, amount: int):
        """Transfer money atomically between accounts."""
        with db.transaction() as txn:
            # Read current balances
            from_balance_bytes = txn.get(from_account)
            to_balance_bytes = txn.get(to_account)
            
            if from_balance_bytes is None or to_balance_bytes is None:
                raise ValueError("Account not found")
            
            from_balance = int(from_balance_bytes.decode('utf-8'))
            to_balance = int(to_balance_bytes.decode('utf-8'))
            
            # Check sufficient funds
            if from_balance < amount:
                raise ValueError(f"Insufficient funds: have {from_balance}, need {amount}")
            
            # Perform transfer
            new_from_balance = from_balance - amount
            new_to_balance = to_balance + amount
            
            txn.put(from_account, str(new_from_balance).encode('utf-8'))
            txn.put(to_account, str(new_to_balance).encode('utf-8'))
            
            print(f"  Transferred ${amount}")
            print(f"    {from_account.decode()}: {from_balance} → {new_from_balance}")
            print(f"    {to_account.decode()}: {to_balance} → {new_to_balance}")
        
        # Transaction commits automatically if no exception
    
    with Database.open(DB_PATH) as db:
        # Initialize accounts
        db.put(b"wallet:alice", b"1000")
        db.put(b"wallet:bob", b"500")
        print("  Initial balances:")
        print(f"    Alice: ${db.get(b'wallet:alice').decode()}")
        print(f"    Bob: ${db.get(b'wallet:bob').decode()}")
        
        # Successful transfer
        print("\n  Transfer $200 from Alice to Bob:")
        transfer_money(db, b"wallet:alice", b"wallet:bob", 200)
        
        # Try invalid transfer (insufficient funds)
        print("\n  Try transfer $1000 from Alice to Bob:")
        try:
            transfer_money(db, b"wallet:alice", b"wallet:bob", 1000)
        except ValueError as e:
            print(f"  Transfer failed: {e}")
        
        # Final balances (second transfer was rolled back)
        print("\n  Final balances:")
        print(f"    Alice: ${db.get(b'wallet:alice').decode()}")
        print(f"    Bob: ${db.get(b'wallet:bob').decode()}")
    
    print("\n✓ Money transfer example complete")


def example_batch_operations():
    """Batch multiple operations in single transaction."""
    print("\n" + "=" * 60)
    print("Example 2.6: Batch Operations")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Batch insert many items in one transaction (efficient)
        items = [(f"item:{i}".encode(), f"value_{i}".encode()) for i in range(100)]
        
        with db.transaction() as txn:
            for key, value in items:
                txn.put(key, value)
            print(f"  Inserted {len(items)} items in single transaction")
        
        # Verify
        sample = db.get(b"item:50")
        print(f"  Sample read - item:50 = {sample.decode('utf-8')}")
        
        # Batch delete
        with db.transaction() as txn:
            for i in range(0, 100, 2):  # Delete even-numbered items
                txn.delete(f"item:{i}".encode())
            print("  Deleted 50 items (even numbers)")
        
        # Verify deletion
        deleted = db.get(b"item:50")
        remaining = db.get(b"item:51")
        print(f"  item:50 (deleted) = {deleted}")
        print(f"  item:51 (kept) = {remaining.decode('utf-8') if remaining else 'None'}")
    
    print("✓ Batch operations complete")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ToonDB Python SDK - Example 02: Transactions")
    print("=" * 60)
    
    cleanup()
    
    example_basic_transaction()
    example_manual_commit()
    example_transaction_abort()
    example_exception_rollback()
    example_money_transfer()
    example_batch_operations()
    
    cleanup()
    
    print("\n" + "=" * 60)
    print("All transaction examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
