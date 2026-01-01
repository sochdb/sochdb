# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket
import struct
import threading
import time
import pytest
from typing import List, Dict, Any
from faker import Faker
from toondb import IpcClient, Query, ToonDBError
from toondb.ipc_client import OpCode, Message

# Mock Server that supports new opcodes
class AdvancedMockServer:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.running = False
        self.thread = None
        self.requests = []
        self.data = {} # Simple in-memory store for the mock
        self.fake = Faker()
        self.active_conns = []

    def start(self):
        self.running = True
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(1)
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        for conn in self.active_conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except OSError:
                pass
        if self.thread:
            self.thread.join()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

    def _run(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
                self.active_conns.append(conn)
                self._handle_client(conn)
            except OSError:
                break

    def _handle_client(self, conn: socket.socket):
        print("Server: Client connected")
        try:
            while self.running:
                print("Server: Waiting for message")
                msg = Message.decode(conn)
                print(f"Server: Received opcode {msg.opcode}")
                self.requests.append((msg.opcode, msg.payload))
                response = self._handle_request(msg.opcode, msg.payload)
                print(f"Server: Sending response opcode {response.opcode}")
                conn.sendall(response.encode())
        except (ConnectionResetError, BrokenPipeError, struct.error) as e:
            print(f"Server: Connection error: {e}")
        except Exception as e:
            print(f"Server: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Server: Closing connection")
            conn.close()

    def _handle_request(self, opcode: int, payload: bytes) -> Message:
        if opcode == OpCode.PING:
            return Message(OpCode.PONG)
        
        elif opcode == OpCode.CREATE_TABLE:
            # Parse schema (simplified)
            # Just return OK
            return Message(OpCode.OK)
            
        elif opcode == OpCode.QUERY:
            # Parse query payload
            # Return a fake TOON string
            # Format: result[N]{cols}: row1; row2
            
            # For testing, we'll just return a fixed response based on the "path"
            # Extract path to decide what to return
            offset = 0
            path_len = struct.unpack("<H", payload[offset:offset+2])[0]
            offset += 2
            path = payload[offset:offset+path_len].decode("utf-8")
            
            if "users" in path:
                # Generate fake users
                users = []
                for _ in range(5):
                    users.append(f"{self.fake.name()},{self.fake.email()}")
                
                toon_resp = f"result[5]{{name,email}}: {'; '.join(users)}"
                return Message(OpCode.VALUE, toon_resp.encode("utf-8"))
            
            return Message(OpCode.VALUE, b"result[0]{}:")

        elif opcode == OpCode.SCAN:
            # Return fake scan results
            # Format: count(4) + [key_len(2) + key + val_len(4) + val]...
            count = 3
            resp = struct.pack("<I", count)
            for i in range(count):
                key = f"key{i}".encode("utf-8")
                val = f"val{i}".encode("utf-8")
                resp += struct.pack("<H", len(key)) + key
                resp += struct.pack("<I", len(val)) + val
            return Message(OpCode.VALUE, resp)

        return Message(OpCode.ERROR, b"Unknown opcode")

@pytest.fixture
def mock_server():
    socket_path = "/tmp/toondb_test.sock"
    server = AdvancedMockServer(socket_path)
    server.start()
    yield server
    server.stop()

@pytest.fixture
def client(mock_server):
    return IpcClient.connect(mock_server.socket_path)

def test_create_table(client, mock_server):
    columns = [
        {"name": "name", "type": 3, "nullable": False},
        {"name": "age", "type": 0, "nullable": True}
    ]
    client.create_table("users", columns)
    
    assert len(mock_server.requests) == 1
    assert mock_server.requests[0][0] == OpCode.CREATE_TABLE

def test_query_users(client, mock_server):
    # Test Query builder
    query = Query(client, "users/")
    results = query.select(["name", "email"]).limit(5).to_list()
    
    assert len(results) == 5
    assert "name" in results[0]
    assert "email" in results[0]
    
    assert len(mock_server.requests) == 1
    assert mock_server.requests[0][0] == OpCode.QUERY

def test_scan(client, mock_server):
    results = client.scan("prefix")
    
    assert len(results) == 3
    assert results[0]["key"] == b"key0"
    assert results[0]["value"] == b"val0"
    
    assert len(mock_server.requests) == 1
    assert mock_server.requests[0][0] == OpCode.SCAN
