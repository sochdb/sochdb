#!/usr/bin/env python3
"""
Example 05: User Store
======================

A complete real-world example: User management system with ToonDB.
Demonstrates:
- User CRUD operations
- Email uniqueness via secondary index
- Session management
- Password hashing (simulated)
- Activity logging

Difficulty: Intermediate
Mode: Embedded (FFI)
"""

import os
import shutil
import sys
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from toondb import Database
from toondb.errors import DatabaseError

DB_PATH = "./example_05_db"


def cleanup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class User:
    id: str
    email: str
    name: str
    password_hash: str
    created_at: str
    updated_at: str
    is_active: bool = True
    
    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode('utf-8')
    
    @classmethod
    def from_json(cls, data: bytes) -> "User":
        return cls(**json.loads(data.decode('utf-8')))


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: str
    expires_at: str
    ip_address: str
    
    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode('utf-8')
    
    @classmethod
    def from_json(cls, data: bytes) -> "Session":
        return cls(**json.loads(data.decode('utf-8')))


# =============================================================================
# User Store
# =============================================================================

class UserStore:
    """ToonDB-backed user management system."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def _hash_password(self, password: str) -> str:
        """Simulate password hashing (use bcrypt in production!)."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a simple unique ID."""
        import random
        return f"{prefix}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    # -------------------------------------------------------------------------
    # User CRUD
    # -------------------------------------------------------------------------
    
    def create_user(self, email: str, name: str, password: str) -> User:
        """Create a new user with email uniqueness check."""
        # Check if email already exists (secondary index)
        email_index_key = f"idx:email:{email.lower()}".encode()
        existing = self.db.get(email_index_key)
        if existing:
            raise ValueError(f"Email {email} already registered")
        
        # Create user
        now = datetime.utcnow().isoformat()
        user = User(
            id=self._generate_id("user"),
            email=email.lower(),
            name=name,
            password_hash=self._hash_password(password),
            created_at=now,
            updated_at=now,
        )
        
        # Store atomically with secondary index
        with self.db.transaction() as txn:
            # Primary storage
            txn.put(f"users:{user.id}".encode(), user.to_json())
            # Email index (for uniqueness and lookup)
            txn.put(email_index_key, user.id.encode())
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        data = self.db.get(f"users:{user_id}".encode())
        if data:
            return User.from_json(data)
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email (using index)."""
        email_lower = email.lower()
        user_id = self.db.get(f"idx:email:{email_lower}".encode())
        if user_id:
            return self.get_user(user_id.decode())
        return None
    
    def update_user(self, user_id: str, name: Optional[str] = None) -> Optional[User]:
        """Update user fields."""
        user = self.get_user(user_id)
        if not user:
            return None
        
        if name:
            user.name = name
        user.updated_at = datetime.utcnow().isoformat()
        
        self.db.put(f"users:{user_id}".encode(), user.to_json())
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and all related data."""
        user = self.get_user(user_id)
        if not user:
            return False
        
        with self.db.transaction() as txn:
            # Delete user
            txn.delete(f"users:{user_id}".encode())
            # Delete email index
            txn.delete(f"idx:email:{user.email}".encode())
            # Note: Sessions would also be deleted in production
        
        return True
    
    def list_users(self, limit: int = 100) -> List[User]:
        """List all users."""
        users = []
        count = 0
        for key, value in self.db.scan(b"users:", b"users;"):
            if count >= limit:
                break
            users.append(User.from_json(value))
            count += 1
        return users
    
    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    
    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        password_hash = self._hash_password(password)
        if user.password_hash != password_hash:
            return None
        
        # Log activity
        self._log_activity(user.id, "login")
        return user
    
    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------
    
    def create_session(self, user_id: str, ip_address: str = "127.0.0.1") -> Session:
        """Create a new session for a user."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=24)
        
        session = Session(
            session_id=self._generate_id("sess"),
            user_id=user_id,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            ip_address=ip_address,
        )
        
        with self.db.transaction() as txn:
            # Store session
            txn.put(f"sessions:{session.session_id}".encode(), session.to_json())
            # Index by user (for listing user's sessions)
            txn.put(f"idx:user_sessions:{user_id}:{session.session_id}".encode(), b"1")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        data = self.db.get(f"sessions:{session_id}".encode())
        if data:
            session = Session.from_json(data)
            # Check expiration
            if datetime.fromisoformat(session.expires_at) < datetime.utcnow():
                self.invalidate_session(session_id)
                return None
            return session
        return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate (delete) a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        with self.db.transaction() as txn:
            txn.delete(f"sessions:{session_id}".encode())
            txn.delete(f"idx:user_sessions:{session.user_id}:{session_id}".encode())
        
        return True
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        sessions = []
        prefix = f"idx:user_sessions:{user_id}:".encode()
        end = f"idx:user_sessions:{user_id};".encode()
        
        for key, _ in self.db.scan(prefix, end):
            session_id = key.decode().split(":")[-1]
            session = self.get_session(session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    # -------------------------------------------------------------------------
    # Activity Logging
    # -------------------------------------------------------------------------
    
    def _log_activity(self, user_id: str, action: str):
        """Log user activity."""
        timestamp = datetime.utcnow().isoformat()
        key = f"activity:{user_id}:{timestamp}".encode()
        self.db.put(key, action.encode())
    
    def get_user_activity(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent activity for a user."""
        prefix = f"activity:{user_id}:".encode()
        end = f"activity:{user_id};".encode()
        
        activities = []
        for key, value in self.db.scan(prefix, end):
            parts = key.decode().split(":")
            activities.append({
                "timestamp": parts[2],
                "action": value.decode()
            })
            if len(activities) >= limit:
                break
        
        return activities


# =============================================================================
# Demo
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("ToonDB Python SDK - Example 05: User Store")
    print("=" * 60)
    
    cleanup()
    
    with Database.open(DB_PATH) as db:
        store = UserStore(db)
        
        # Create users
        print("\n--- Creating Users ---")
        alice = store.create_user("alice@example.com", "Alice Johnson", "password123")
        bob = store.create_user("bob@example.com", "Bob Smith", "secret456")
        charlie = store.create_user("charlie@example.com", "Charlie Brown", "mypass789")
        
        print(f"✓ Created user: {alice.name} ({alice.id})")
        print(f"✓ Created user: {bob.name} ({bob.id})")
        print(f"✓ Created user: {charlie.name} ({charlie.id})")
        
        # Try duplicate email
        print("\n--- Email Uniqueness ---")
        try:
            store.create_user("alice@example.com", "Fake Alice", "test")
        except ValueError as e:
            print(f"✓ Prevented duplicate: {e}")
        
        # Lookup by email
        print("\n--- Email Lookup ---")
        found = store.get_user_by_email("bob@example.com")
        print(f"✓ Found by email: {found.name}")
        
        # Authentication
        print("\n--- Authentication ---")
        auth_user = store.authenticate("alice@example.com", "password123")
        if auth_user:
            print(f"✓ Authenticated: {auth_user.name}")
        
        bad_auth = store.authenticate("alice@example.com", "wrongpassword")
        if not bad_auth:
            print("✓ Rejected wrong password")
        
        # Sessions
        print("\n--- Session Management ---")
        session = store.create_session(alice.id, "192.168.1.100")
        print(f"✓ Created session: {session.session_id}")
        print(f"  Expires: {session.expires_at}")
        
        # Get session
        retrieved = store.get_session(session.session_id)
        print(f"✓ Retrieved session for user: {retrieved.user_id}")
        
        # List sessions
        alice_sessions = store.get_user_sessions(alice.id)
        print(f"✓ Alice has {len(alice_sessions)} active session(s)")
        
        # Update user
        print("\n--- Update User ---")
        updated = store.update_user(bob.id, name="Robert Smith")
        print(f"✓ Updated name: {updated.name}")
        
        # List all users
        print("\n--- List Users ---")
        users = store.list_users()
        for user in users:
            print(f"  - {user.name} <{user.email}>")
        
        # Activity log
        print("\n--- Activity Log ---")
        activities = store.get_user_activity(alice.id)
        for act in activities:
            print(f"  [{act['timestamp']}] {act['action']}")
        
        # Delete user
        print("\n--- Delete User ---")
        deleted = store.delete_user(charlie.id)
        print(f"✓ Deleted Charlie: {deleted}")
        
        # Verify deletion
        remaining = store.list_users()
        print(f"  Remaining users: {len(remaining)}")
    
    cleanup()
    
    print("\n" + "=" * 60)
    print("User Store example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
