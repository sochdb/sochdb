#!/usr/bin/env python3
"""
SochDB Key-Value Query Examples

Demonstrates KV data patterns in SochDB:
- Creating structured data with key prefixes
- Inserting, reading, updating, and deleting KV pairs
- Scanning and filtering by prefix
- Transactions with begin/commit/abort
"""

from sochdb import Database


def create_data(db: Database) -> None:
    """Create structured data using key prefixes (replaces CREATE TABLE + INSERT)."""
    print("\nCreating Structured Data with Key Prefixes")
    print("=" * 60)

    # Instead of CREATE TABLE, we organize data using key prefixes like
    # "users:<id>:<field>" and "posts:<id>:<field>"

    # Insert users (replaces INSERT INTO users ...)
    users = [
        (1, "Alice", "alice@example.com", "30", "2024-01-01"),
        (2, "Bob", "bob@example.com", "25", "2024-01-02"),
        (3, "Charlie", "charlie@example.com", "35", "2024-01-03"),
        (4, "Diana", "diana@example.com", "28", "2024-01-04"),
    ]

    for user_id, name, email, age, created_at in users:
        db.put(f"users:{user_id}:name".encode(), name.encode())
        db.put(f"users:{user_id}:email".encode(), email.encode())
        db.put(f"users:{user_id}:age".encode(), age.encode())
        db.put(f"users:{user_id}:created_at".encode(), created_at.encode())
        print(f"  Inserted user: {name}")

    # Insert posts (replaces INSERT INTO posts ...)
    posts = [
        (1, 1, "First Post", "Hello World!", "10", "2024-01-05"),
        (2, 1, "Second Post", "SochDB is awesome", "25", "2024-01-06"),
        (3, 2, "Bob's Thoughts", "KV queries are easy", "15", "2024-01-07"),
        (4, 3, "Charlie's Guide", "Database tips", "30", "2024-01-08"),
        (5, 3, "Advanced Topics", "Performance tuning", "50", "2024-01-09"),
    ]

    for post_id, user_id, title, content, likes, published_at in posts:
        db.put(f"posts:{post_id}:user_id".encode(), str(user_id).encode())
        db.put(f"posts:{post_id}:title".encode(), title.encode())
        db.put(f"posts:{post_id}:content".encode(), content.encode())
        db.put(f"posts:{post_id}:likes".encode(), likes.encode())
        db.put(f"posts:{post_id}:published_at".encode(), published_at.encode())
        print(f"  Inserted post: {title}")


def scan_queries(db: Database) -> None:
    """Scan and filter data (replaces SELECT queries)."""
    print("\nScanning and Filtering Data")
    print("=" * 60)

    # Simple scan: all users (replaces SELECT * FROM users)
    print("\n1. All users:")
    rows = db.scan(b"users:")
    print(f"   Found {len(rows)} user fields")

    # Collect users into a dict for easier display
    users = {}
    for key, value in rows:
        parts = key.decode().split(":")
        uid = int(parts[1])
        field = parts[2]
        users.setdefault(uid, {})[field] = value.decode()

    for uid in sorted(users):
        u = users[uid]
        print(f"   - {u.get('name', '?')} ({u.get('email', '?')})")

    # SELECT with WHERE: filter users older than 28
    print("\n2. Users older than 28:")
    for uid in sorted(users):
        age = int(users[uid].get("age", "0"))
        if age > 28:
            print(f"   - {users[uid]['name']}: {age} years old")

    # ORDER BY likes descending: scan posts and sort in Python
    print("\n3. Posts ordered by likes (descending):")
    posts = {}
    for key, value in db.scan(b"posts:"):
        parts = key.decode().split(":")
        pid = int(parts[1])
        field = parts[2]
        posts.setdefault(pid, {})[field] = value.decode()

    sorted_posts = sorted(
        posts.items(), key=lambda x: int(x[1].get("likes", "0")), reverse=True
    )
    for pid, p in sorted_posts:
        print(f"   - {p['title']}: {p['likes']} likes")

    # LIMIT: top 3 most liked posts
    print("\n4. Top 3 most liked posts:")
    for pid, p in sorted_posts[:3]:
        print(f"   - {p['title']}: {p['likes']} likes")

    # COUNT: total posts
    print("\n5. Count total posts:")
    print(f"   Total posts: {len(posts)}")


def update_operations(db: Database) -> None:
    """Update data by overwriting keys (replaces UPDATE)."""
    print("\nUpdate Operations")
    print("=" * 60)

    # Update single field: Alice's age (replaces UPDATE ... SET age = 31)
    print("\n1. Update Alice's age:")
    db.put(b"users:1:age", b"31")
    val = db.get(b"users:1:age")
    print(f"   Alice's new age: {val.decode() if val else 'not found'}")

    # Update multiple: increment likes on posts by user_id=1
    print("\n2. Increment likes on all posts by user 1:")
    posts = {}
    for key, value in db.scan(b"posts:"):
        parts = key.decode().split(":")
        pid = int(parts[1])
        field = parts[2]
        posts.setdefault(pid, {})[field] = value.decode()

    for pid, p in posts.items():
        if p.get("user_id") == "1":
            new_likes = int(p["likes"]) + 5
            db.put(f"posts:{pid}:likes".encode(), str(new_likes).encode())
            print(f"   - {p['title']}: {new_likes} likes")


def delete_operations(db: Database) -> None:
    """Delete keys (replaces DELETE FROM)."""
    print("\nDelete Operations")
    print("=" * 60)

    # Count before delete
    before = len(db.scan(b"posts:"))
    print(f"Post fields before delete: {before}")

    # Delete post with id=5: remove all fields for that post
    print("\nDeleting post id=5...")
    for key, _ in db.scan(b"posts:5:"):
        db.delete(key)
    print("  Deleted post with id = 5")

    # Count after delete
    after = len(db.scan(b"posts:"))
    print(f"Post fields after delete: {after}")


def transactions(db: Database) -> None:
    """Use transactions with begin/commit/abort."""
    print("\nTransactions (begin / commit / abort)")
    print("=" * 60)

    from sochdb import Transaction
    print("\n1. Successful transaction:")
    txn = Transaction(db)
    try:
        with txn as t:
            db.put(b"users:5:name", b"Eve", txn=t.id)
            db.put(b"users:5:email", b"eve@example.com", txn=t.id)
            db.put(b"users:5:age", b"26", txn=t.id)
        print("  Transaction committed successfully")
    except Exception as e:
        print(f"  Transaction failed: {e}")

    # Verify the data was persisted
    name = db.get(b"users:5:name")
    if name:
        print(f"  New user: {name.decode()}")

    # Aborted transaction: changes are rolled back
    print("\n2. Aborted transaction (changes rolled back):")
    txn2 = Transaction(db)
    try:
        with txn2 as t:
            db.put(b"users:6:name", b"Frank", txn=t.id)
            db.put(b"users:6:age", b"40", txn=t.id)
            raise ValueError("Simulated error")
    except ValueError:
        try:
            txn2.abort()
        except Exception:
            pass
        print("  Transaction aborted (Frank NOT stored)")

    # Verify Frank was NOT stored
    frank = db.get(b"users:6:name")
    if frank:
        print(f"  Unexpected: {frank.decode()}")
    else:
        print("  Confirmed: Frank not in database")


def complex_queries(db: Database) -> None:
    """Combine scans with Python filtering (replaces complex SQL)."""
    print("\nComplex Filtering")
    print("=" * 60)

    # Collect all users
    users = {}
    for key, value in db.scan(b"users:"):
        parts = key.decode().split(":")
        uid = int(parts[1])
        field = parts[2]
        users.setdefault(uid, {})[field] = value.decode()

    # Users aged 25-30 (replaces WHERE age >= 25 AND age <= 30)
    print("\n1. Users aged 25-30:")
    for uid in sorted(users):
        age = int(users[uid].get("age", "0"))
        if 25 <= age <= 30:
            u = users[uid]
            print(f"   - {u['name']}: {age} years ({u['email']})")

    # Collect all posts
    posts = {}
    for key, value in db.scan(b"posts:"):
        parts = key.decode().split(":")
        pid = int(parts[1])
        field = parts[2]
        posts.setdefault(pid, {})[field] = value.decode()

    # Posts with "Post" in title (replaces WHERE title LIKE '%Post%')
    print("\n2. Posts with 'Post' in title:")
    for pid in sorted(posts):
        title = posts[pid].get("title", "")
        if "Post" in title:
            print(f"   - {title}: {posts[pid].get('likes', '0')} likes")


def main():
    """Main demonstration."""
    print("=" * 60)
    print("SochDB Key-Value Query Examples")
    print("=" * 60)
    print("\nSochDB uses key-value pairs with prefix-based organization")
    print("instead of SQL. Scans return all keys matching a prefix.\n")

    # Open database
    db_path = "./demo_kv_db"
    print(f"Opening database: {db_path}")

    # Clean up existing database
    import os
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = Database.open(db_path)
    print("Database opened successfully")

    try:
        # Run demonstrations
        create_data(db)
        scan_queries(db)
        update_operations(db)
        delete_operations(db)
        transactions(db)
        complex_queries(db)

        print("\n" + "=" * 60)
        print("All KV examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()