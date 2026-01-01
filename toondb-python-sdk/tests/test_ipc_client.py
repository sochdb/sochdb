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

"""Tests for ToonDB IPC Client."""

import pytest
import socket
import struct
import threading
import time
from typing import List, Tuple

from toondb.ipc_client import IpcClient, Message, OpCode
from toondb.errors import ConnectionError, DatabaseError


class MockServer:
    """Simple mock IPC server for testing."""
    
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.sock = None
        self.running = False
        self.thread = None
        self.requests: List[Tuple[int, bytes]] = []
    
    def start(self):
        import os
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(1)
        self.sock.settimeout(1.0)
        self.running = True
        
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)  # Wait for server to start
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()
        import os
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
    
    def _run(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
                conn.settimeout(1.0)
                self._handle_client(conn)
            except socket.timeout:
                continue
            except Exception:
                break
    
    def _handle_client(self, conn: socket.socket):
        try:
            while self.running:
                # Read request
                opcode_data = self._recv_exact(conn, 1)
                if not opcode_data:
                    break
                opcode = opcode_data[0]
                
                len_data = self._recv_exact(conn, 4)
                if not len_data:
                    break
                length = struct.unpack("<I", len_data)[0]
                
                payload = b""
                if length > 0:
                    payload = self._recv_exact(conn, length)
                
                self.requests.append((opcode, payload))
                
                # Send response
                response = self._handle_request(opcode, payload)
                conn.sendall(response.encode())
                
        except Exception:
            pass
        finally:
            conn.close()
    
    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                break
            data += chunk
        return data
    
    def _handle_request(self, opcode: int, payload: bytes) -> Message:
        if opcode == OpCode.PING:
            return Message(OpCode.PONG)
        elif opcode == OpCode.PUT:
            return Message(OpCode.OK)
        elif opcode == OpCode.GET:
            return Message(OpCode.VALUE, b"test_value")
        elif opcode == OpCode.DELETE:
            return Message(OpCode.OK)
        elif opcode == OpCode.BEGIN_TXN:
            return Message(OpCode.TXN_ID, struct.pack("<Q", 1))
        elif opcode == OpCode.COMMIT_TXN:
            return Message(OpCode.TXN_ID, struct.pack("<Q", 2))
        elif opcode == OpCode.ABORT_TXN:
            return Message(OpCode.OK)
        elif opcode == OpCode.PUT_PATH:
            return Message(OpCode.OK)
        elif opcode == OpCode.GET_PATH:
            return Message(OpCode.VALUE, b"path_value")
        elif opcode == OpCode.CHECKPOINT:
            return Message(OpCode.OK)
        elif opcode == OpCode.STATS:
            return Message(OpCode.STATS_RESP, b"requests_total=10\nuptime_secs=60")
        else:
            return Message(OpCode.ERROR, b"Unknown opcode")


@pytest.fixture
def mock_server(tmp_path):
    """Create a mock server for testing."""
    # Use /tmp to avoid path length issues with AF_UNIX
    import tempfile
    import uuid
    socket_path = f"/tmp/toondb_test_{uuid.uuid4().hex[:8]}.sock"
    server = MockServer(socket_path)
    server.start()
    yield server, socket_path
    server.stop()


class TestMessage:
    """Tests for wire protocol message encoding/decoding."""
    
    def test_encode_simple(self):
        msg = Message(OpCode.PING)
        encoded = msg.encode()
        assert encoded == bytes([0x0E, 0, 0, 0, 0])
    
    def test_encode_with_payload(self):
        msg = Message(OpCode.PUT, b"hello")
        encoded = msg.encode()
        assert encoded == bytes([0x01, 5, 0, 0, 0]) + b"hello"
    
    def test_roundtrip(self):
        import io
        
        original = Message(OpCode.PUT, b"test payload")
        encoded = original.encode()
        
        # Create a mock socket-like object
        class MockSocket:
            def __init__(self, data):
                self.buffer = io.BytesIO(data)
            
            def recv(self, n):
                return self.buffer.read(n)
        
        mock = MockSocket(encoded)
        decoded = Message.decode(mock)
        
        assert decoded.opcode == original.opcode
        assert decoded.payload == original.payload


class TestIpcClient:
    """Tests for IPC client."""
    
    def test_connect(self, mock_server):
        server, socket_path = mock_server
        client = IpcClient.connect(socket_path)
        assert client is not None
        client.close()
    
    def test_connect_nonexistent(self, tmp_path):
        with pytest.raises(ConnectionError):
            IpcClient.connect(str(tmp_path / "nonexistent.sock"))
    
    def test_ping(self, mock_server):
        server, socket_path = mock_server
        client = IpcClient.connect(socket_path)
        latency = client.ping()
        assert latency >= 0
        client.close()
    
    def test_put_get(self, mock_server):
        server, socket_path = mock_server
        with IpcClient.connect(socket_path) as client:
            client.put(b"key", b"value")
            value = client.get(b"key")
            assert value == b"test_value"  # Mock always returns this
    
    def test_delete(self, mock_server):
        server, socket_path = mock_server
        with IpcClient.connect(socket_path) as client:
            client.delete(b"key")
            # Check that request was received
            assert any(op == OpCode.DELETE for op, _ in server.requests)
    
    def test_transaction(self, mock_server):
        server, socket_path = mock_server
        with IpcClient.connect(socket_path) as client:
            txn_id = client.begin_transaction()
            assert txn_id == 1
            
            commit_ts = client.commit(txn_id)
            assert commit_ts == 2
    
    def test_abort_transaction(self, mock_server):
        server, socket_path = mock_server
        with IpcClient.connect(socket_path) as client:
            txn_id = client.begin_transaction()
            client.abort(txn_id)
            assert any(op == OpCode.ABORT_TXN for op, _ in server.requests)
    
    def test_path_api(self, mock_server):
        server, socket_path = mock_server
        with IpcClient.connect(socket_path) as client:
            client.put_path(["users", "alice", "email"], b"alice@example.com")
            assert any(op == OpCode.PUT_PATH for op, _ in server.requests)
    
    def test_stats(self, mock_server):
        server, socket_path = mock_server
        with IpcClient.connect(socket_path) as client:
            stats = client.stats()
            assert stats["requests_total"] == 10
            assert stats["uptime_secs"] == 60
    
    def test_checkpoint(self, mock_server):
        server, socket_path = mock_server
        with IpcClient.connect(socket_path) as client:
            client.checkpoint()
            assert any(op == OpCode.CHECKPOINT for op, _ in server.requests)
