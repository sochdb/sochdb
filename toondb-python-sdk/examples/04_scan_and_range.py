#!/usr/bin/env python3
"""
Example 04: Scan and Range Queries
==================================

This example demonstrates scanning and range operations:
- Basic range scans
- Prefix scans
- Iterating over results
- Batch processing with scans

Difficulty: Intermediate
Mode: Embedded (FFI)
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from toondb import Database
from toondb.errors import DatabaseError

DB_PATH = "./example_04_db"


def cleanup():
    """Clean up database."""
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    print(f"✓ Cleaned up {DB_PATH}")


def example_basic_scan():
    """Basic scan over all keys."""
    print("\n" + "=" * 60)
    print("Example 4.1: Basic Scan")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Insert some data
        db.put(b"alpha", b"first")
        db.put(b"beta", b"second")
        db.put(b"gamma", b"third")
        db.put(b"delta", b"fourth")
        print("✓ Inserted 4 keys")
        
        # Scan all keys (empty start and end)
        print("\nAll keys:")
        count = 0
        for key, value in db.scan():
            print(f"  {key.decode()}: {value.decode()}")
            count += 1
        
        print(f"\n✓ Scanned {count} keys")


def example_range_scan():
    """Scan a specific key range."""
    print("\n" + "=" * 60)
    print("Example 4.2: Range Scan")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Insert ordered keys
        for i in range(100):
            key = f"key:{i:03d}".encode()
            value = f"value_{i}".encode()
            db.put(key, value)
        print("✓ Inserted 100 keys (key:000 to key:099)")
        
        # Scan a range [start, end)
        start = b"key:025"
        end = b"key:030"
        
        print(f"\nKeys in range [{start.decode()}, {end.decode()}):")
        for key, value in db.scan(start, end):
            print(f"  {key.decode()}: {value.decode()}")


def example_prefix_scan():
    """Scan keys with a common prefix."""
    print("\n" + "=" * 60)
    print("Example 4.3: Prefix Scan")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Insert data with different prefixes
        categories = {
            "user:": ["alice", "bob", "charlie"],
            "order:": ["1001", "1002", "1003", "1004"],
            "product:": ["laptop", "mouse", "keyboard"],
        }
        
        for prefix, items in categories.items():
            for item in items:
                key = f"{prefix}{item}".encode()
                db.put(key, f"data_{item}".encode())
        
        print("✓ Inserted data with multiple prefixes")
        
        # Scan only user keys
        # Range: ["user:", "user;" ) - semicolon comes after colon in ASCII
        print("\nUsers (prefix scan):")
        for key, value in db.scan(b"user:", b"user;"):
            print(f"  {key.decode()}: {value.decode()}")
        
        # Scan only order keys
        print("\nOrders (prefix scan):")
        for key, value in db.scan(b"order:", b"order;"):
            print(f"  {key.decode()}: {value.decode()}")
        
        # Scan only product keys
        print("\nProducts (prefix scan):")
        for key, value in db.scan(b"product:", b"product;"):
            print(f"  {key.decode()}: {value.decode()}")


def example_scan_with_transaction():
    """Scan within a transaction context."""
    print("\n" + "=" * 60)
    print("Example 4.4: Scan with Transaction")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Setup initial data
        with db.transaction() as txn:
            for i in range(10):
                txn.put(f"txn_item:{i}".encode(), f"value_{i}".encode())
        print("✓ Initial data committed")
        
        # Scan within a transaction
        with db.transaction() as txn:
            print("\nScanning within transaction:")
            items = list(txn.scan(b"txn_item:", b"txn_item;"))
            print(f"  Found {len(items)} items")
            
            # Modify during scan (add new item)
            txn.put(b"txn_item:new", b"new_value")
            
            # Scan sees the new item (within same transaction)
            items_after = list(txn.scan(b"txn_item:", b"txn_item;"))
            print(f"  After modification: {len(items_after)} items")
        
        # Verify outside transaction
        final_items = list(db.scan(b"txn_item:", b"txn_item;"))
        print(f"  Committed items: {len(final_items)}")


def example_batch_processing():
    """Process large datasets with scans."""
    print("\n" + "=" * 60)
    print("Example 4.5: Batch Processing with Scans")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Insert a larger dataset
        num_records = 1000
        print(f"Inserting {num_records} records...")
        
        with db.transaction() as txn:
            for i in range(num_records):
                txn.put(f"record:{i:05d}".encode(), f"data_{i}".encode())
        print(f"✓ Inserted {num_records} records")
        
        # Process in batches using range scans
        batch_size = 100
        processed = 0
        
        print(f"\nProcessing in batches of {batch_size}:")
        
        # Process first few batches as example
        for batch_num in range(3):  # Just show 3 batches
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            
            start_key = f"record:{start_idx:05d}".encode()
            end_key = f"record:{end_idx:05d}".encode()
            
            batch_count = 0
            for key, value in db.scan(start_key, end_key):
                batch_count += 1
                # Simulate processing
                _ = value.decode().upper()
            
            processed += batch_count
            print(f"  Batch {batch_num + 1}: Processed {batch_count} records")
        
        print(f"\n✓ Demonstrated batch processing ({processed} records shown)")


def example_analytics_scan():
    """Real-world example: Analytics data scanning."""
    print("\n" + "=" * 60)
    print("Example 4.6: Analytics Data Scanning")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        import random
        
        # Insert time-series analytics events
        events = ["page_view", "click", "purchase", "signup"]
        
        with db.transaction() as txn:
            for day in range(1, 8):  # 7 days
                for hour in range(24):
                    for _ in range(random.randint(5, 20)):  # Random events per hour
                        event = random.choice(events)
                        ts = f"2024-01-{day:02d}T{hour:02d}:{random.randint(0,59):02d}"
                        key = f"analytics:{ts}:{random.randint(1000, 9999)}".encode()
                        txn.put(key, event.encode())
        
        print("✓ Generated 7 days of analytics events")
        
        # Query events for a specific day
        target_day = "2024-01-03"
        start = f"analytics:{target_day}T00:00".encode()
        end = f"analytics:{target_day}T23:60".encode()  # Note: "T24:00" wouldn't work correctly
        
        event_counts = {}
        total_events = 0
        
        for key, value in db.scan(start, end):
            event = value.decode()
            event_counts[event] = event_counts.get(event, 0) + 1
            total_events += 1
        
        print(f"\nEvents on {target_day}:")
        print(f"  Total events: {total_events}")
        for event, count in sorted(event_counts.items()):
            pct = (count / total_events * 100) if total_events > 0 else 0
            print(f"  {event}: {count} ({pct:.1f}%)")


def example_cleanup_with_scan():
    """Delete old data using scan."""
    print("\n" + "=" * 60)
    print("Example 4.7: Data Cleanup with Scan")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Insert data with timestamps
        with db.transaction() as txn:
            for i in range(50):
                # Old entries (to delete)
                txn.put(f"log:2024-01-01:{i:03d}".encode(), b"old_entry")
            for i in range(50):
                # New entries (to keep)
                txn.put(f"log:2024-01-15:{i:03d}".encode(), b"new_entry")
        
        print("✓ Inserted 100 log entries (50 old, 50 new)")
        
        # Count total
        total_before = len(list(db.scan(b"log:", b"log;")))
        print(f"  Total entries before cleanup: {total_before}")
        
        # Delete old entries
        with db.transaction() as txn:
            old_keys = []
            for key, _ in txn.scan(b"log:2024-01-01", b"log:2024-01-02"):
                old_keys.append(key)
            
            for key in old_keys:
                txn.delete(key)
            
            print(f"  Deleted {len(old_keys)} old entries")
        
        # Count after cleanup
        total_after = len(list(db.scan(b"log:", b"log;")))
        print(f"  Total entries after cleanup: {total_after}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ToonDB Python SDK - Example 04: Scan and Range Queries")
    print("=" * 60)
    
    cleanup()
    
    example_basic_scan()
    example_range_scan()
    example_prefix_scan()
    example_scan_with_transaction()
    example_batch_processing()
    example_analytics_scan()
    example_cleanup_with_scan()
    
    cleanup()
    
    print("\n" + "=" * 60)
    print("All scan examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
