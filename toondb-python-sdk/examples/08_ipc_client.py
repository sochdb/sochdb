#!/usr/bin/env python3
"""
Example 08: IPC Client - Multi-process access via Unix socket

NOTE: This example requires a running ToonDB IPC server.
Start the server first: cargo run --bin ipc_server -- --socket /tmp/toondb.sock
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from toondb import IpcClient, Query
from toondb.errors import ConnectionError

SOCKET_PATH = "/tmp/toondb.sock"

def main():
    print("=" * 60)
    print("ToonDB - Example 08: IPC Client")
    print("=" * 60)
    
    try:
        client = IpcClient.connect(SOCKET_PATH, timeout=5.0)
        print(f"✓ Connected to {SOCKET_PATH}")
    except ConnectionError as e:
        print(f"✗ Could not connect: {e}")
        print("\nTo run this example, start the IPC server first:")
        print("  cargo run --bin ipc_server -- --socket /tmp/toondb.sock")
        return
    
    try:
        # Ping
        latency = client.ping()
        print(f"✓ Ping: {latency*1000:.2f}ms")
        
        # Put/Get
        client.put(b"hello", b"world")
        value = client.get(b"hello")
        print(f"✓ Put/Get: hello={value.decode()}")
        
        # Path API
        client.put_path(["users", "alice", "email"], b"alice@example.com")
        email = client.get_path(["users", "alice", "email"])
        print(f"✓ Path: users/alice/email={email.decode()}")
        
        # Transaction
        txn_id = client.begin_transaction()
        # Note: IPC transactions require passing txn_id to operations
        commit_ts = client.commit(txn_id)
        print(f"✓ Transaction committed: {commit_ts}")
        
        # Stats
        stats = client.stats()
        print(f"✓ Stats: {stats}")
        
    finally:
        client.close()
        print("✓ Connection closed")

if __name__ == "__main__":
    main()
