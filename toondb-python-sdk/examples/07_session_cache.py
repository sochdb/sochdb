#!/usr/bin/env python3
"""
Example 07: Session Cache - In-memory session caching pattern
"""

import os, shutil, sys, json, time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from toondb import Database

DB_PATH = "./example_07_db"

def cleanup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

class SessionCache:
    def __init__(self, db: Database, ttl_seconds: int = 3600):
        self.db = db
        self.ttl = ttl_seconds
    
    def set(self, session_id: str, data: dict) -> None:
        expires = datetime.utcnow() + timedelta(seconds=self.ttl)
        value = {"data": data, "expires": expires.isoformat()}
        self.db.put(f"session:{session_id}".encode(), json.dumps(value).encode())
    
    def get(self, session_id: str) -> dict:
        raw = self.db.get(f"session:{session_id}".encode())
        if not raw:
            return None
        value = json.loads(raw.decode())
        if datetime.fromisoformat(value["expires"]) < datetime.utcnow():
            self.delete(session_id)
            return None
        return value["data"]
    
    def delete(self, session_id: str) -> None:
        self.db.delete(f"session:{session_id}".encode())
    
    def cleanup_expired(self) -> int:
        count = 0
        for key, val in self.db.scan(b"session:", b"session;"):
            value = json.loads(val.decode())
            if datetime.fromisoformat(value["expires"]) < datetime.utcnow():
                self.db.delete(key)
                count += 1
        return count

def main():
    print("=" * 60)
    print("ToonDB - Example 07: Session Cache")
    print("=" * 60)
    
    cleanup()
    
    with Database.open(DB_PATH) as db:
        cache = SessionCache(db, ttl_seconds=5)  # 5 second TTL for demo
        
        # Set sessions
        cache.set("sess_abc123", {"user_id": "user_1", "role": "admin"})
        cache.set("sess_def456", {"user_id": "user_2", "role": "viewer"})
        print("✓ Created 2 sessions")
        
        # Get session
        session = cache.get("sess_abc123")
        print(f"✓ Session data: user_id={session['user_id']}, role={session['role']}")
        
        # Delete
        cache.delete("sess_def456")
        print("✓ Deleted session sess_def456")
        
        # Verify
        deleted = cache.get("sess_def456")
        print(f"✓ Deleted session returns: {deleted}")
    
    cleanup()
    print("✓ Example completed!")

if __name__ == "__main__":
    main()
