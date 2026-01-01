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

"""
ToonDB IPC Client

Connects to a ToonDB IPC server via Unix domain socket.
"""

import socket
import struct
from typing import Optional, Dict, List, Any
from .errors import ConnectionError, TransactionError, ProtocolError, DatabaseError


class OpCode:
    """Wire protocol opcodes."""
    # Client → Server
    PUT = 0x01
    GET = 0x02
    DELETE = 0x03
    BEGIN_TXN = 0x04
    COMMIT_TXN = 0x05
    ABORT_TXN = 0x06
    QUERY = 0x07
    CREATE_TABLE = 0x08
    PUT_PATH = 0x09
    GET_PATH = 0x0A
    SCAN = 0x0B
    CHECKPOINT = 0x0C
    STATS = 0x0D
    PING = 0x0E
    
    # Server → Client
    OK = 0x80
    ERROR = 0x81
    VALUE = 0x82
    TXN_ID = 0x83
    ROW = 0x84
    END_STREAM = 0x85
    STATS_RESP = 0x86
    PONG = 0x87


class Message:
    """Wire protocol message."""
    
    MAX_SIZE = 16 * 1024 * 1024  # 16 MB
    
    def __init__(self, opcode: int, payload: bytes = b""):
        self.opcode = opcode
        self.payload = payload
    
    def encode(self) -> bytes:
        """Encode message to wire format: opcode (1) + length (4 LE) + payload."""
        return (
            bytes([self.opcode]) +
            struct.pack("<I", len(self.payload)) +
            self.payload
        )
    
    @classmethod
    def decode(cls, sock: socket.socket) -> "Message":
        """Decode a message from the socket."""
        # Read opcode (1 byte)
        opcode_data = cls._recv_exact(sock, 1)
        if not opcode_data:
            raise ConnectionError("Connection closed")
        opcode = opcode_data[0]
        
        # Read length (4 bytes LE)
        len_data = cls._recv_exact(sock, 4)
        if not len_data:
            raise ConnectionError("Connection closed while reading length")
        length = struct.unpack("<I", len_data)[0]
        
        # Validate length
        if length > cls.MAX_SIZE:
            raise ProtocolError(f"Message too large: {length} bytes")
        
        # Read payload
        payload = b""
        if length > 0:
            payload = cls._recv_exact(sock, length)
            if len(payload) != length:
                raise ConnectionError("Connection closed while reading payload")
        
        return cls(opcode, payload)
    
    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                break
            data += chunk
        return data


class IpcClient:
    """
    ToonDB IPC Client.
    
    Connects to a ToonDB server via Unix domain socket.
    
    Example:
        client = IpcClient.connect("/tmp/toondb.sock")
        client.put(b"key", b"value")
        value = client.get(b"key")
    """
    
    def __init__(self, sock: socket.socket):
        self._sock = sock
    
    @classmethod
    def connect(cls, socket_path: str, timeout: float = 30.0) -> "IpcClient":
        """
        Connect to a ToonDB IPC server.
        
        Args:
            socket_path: Path to the Unix domain socket.
            timeout: Connection and I/O timeout in seconds.
            
        Returns:
            Connected IpcClient instance.
            
        Raises:
            ConnectionError: If connection fails.
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(socket_path)
            return cls(sock)
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to {socket_path}: {e}")
    
    def close(self):
        """Close the connection."""
        try:
            self._sock.close()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _request(self, msg: Message) -> Message:
        """Send a request and receive the response."""
        try:
            self._sock.sendall(msg.encode())
            return Message.decode(self._sock)
        except socket.error as e:
            raise ConnectionError(f"Socket error: {e}")
    
    def _check_ok(self, resp: Message) -> None:
        """Check if response is OK, raise if error."""
        if resp.opcode == OpCode.OK:
            return
        if resp.opcode == OpCode.ERROR:
            raise DatabaseError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")
    
    # =========================================================================
    # Basic Operations
    # =========================================================================
    
    def ping(self) -> float:
        """
        Ping the server.
        
        Returns:
            Round-trip latency in seconds.
        """
        import time
        start = time.monotonic()
        resp = self._request(Message(OpCode.PING))
        if resp.opcode != OpCode.PONG:
            raise ProtocolError(f"Expected PONG, got {resp.opcode:#x}")
        return time.monotonic() - start
    
    def put(self, key: bytes, value: bytes) -> None:
        """
        Put a key-value pair.
        
        Args:
            key: The key bytes.
            value: The value bytes.
        """
        payload = struct.pack("<I", len(key)) + key + value
        resp = self._request(Message(OpCode.PUT, payload))
        self._check_ok(resp)
    
    def get(self, key: bytes) -> Optional[bytes]:
        """
        Get a value by key.
        
        Args:
            key: The key bytes.
            
        Returns:
            The value bytes, or None if not found.
        """
        resp = self._request(Message(OpCode.GET, key))
        if resp.opcode == OpCode.VALUE:
            return resp.payload if resp.payload else None
        if resp.opcode == OpCode.ERROR:
            raise DatabaseError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")
    
    def delete(self, key: bytes) -> None:
        """
        Delete a key.
        
        Args:
            key: The key bytes.
        """
        resp = self._request(Message(OpCode.DELETE, key))
        self._check_ok(resp)
    
    # =========================================================================
    # Path-Native API
    # =========================================================================
    
    def put_path(self, path: List[str], value: bytes) -> None:
        """
        Put a value at a hierarchical path.
        
        Args:
            path: Path segments (e.g., ["users", "alice", "email"])
            value: The value bytes.
        """
        payload = self._encode_path(path) + value
        resp = self._request(Message(OpCode.PUT_PATH, payload))
        self._check_ok(resp)
    
    def get_path(self, path: List[str]) -> Optional[bytes]:
        """
        Get a value at a hierarchical path.
        
        Args:
            path: Path segments (e.g., ["users", "alice", "email"])
            
        Returns:
            The value bytes, or None if not found.
        """
        payload = self._encode_path(path)
        resp = self._request(Message(OpCode.GET_PATH, payload))
        if resp.opcode == OpCode.VALUE:
            return resp.payload if resp.payload else None
        if resp.opcode == OpCode.ERROR:
            raise DatabaseError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")
    
    def _encode_path(self, path: List[str]) -> bytes:
        """Encode a path for the wire protocol."""
        result = struct.pack("<H", len(path))
        for segment in path:
            seg_bytes = segment.encode("utf-8")
            result += struct.pack("<H", len(seg_bytes)) + seg_bytes
        return result
    
    def query(self, path_prefix: str, limit: Optional[int] = None, 
              offset: Optional[int] = None, columns: Optional[List[str]] = None) -> str:
        """
        Execute a query.
        
        Args:
            path_prefix: Path prefix to query.
            limit: Max results.
            offset: Results to skip.
            columns: Columns to select.
            
        Returns:
            TOON formatted string.
        """
        # Payload: path_len(2) + path + limit(4) + offset(4) + cols_count(2) + [col_len(2) + col]...
        path_bytes = path_prefix.encode("utf-8")
        payload = struct.pack("<H", len(path_bytes)) + path_bytes
        
        payload += struct.pack("<I", limit or 0)
        payload += struct.pack("<I", offset or 0)
        
        cols = columns or []
        payload += struct.pack("<H", len(cols))
        for col in cols:
            col_bytes = col.encode("utf-8")
            payload += struct.pack("<H", len(col_bytes)) + col_bytes
            
        resp = self._request(Message(OpCode.QUERY, payload))
        if resp.opcode == OpCode.VALUE:
            return resp.payload.decode("utf-8", errors="replace")
        if resp.opcode == OpCode.ERROR:
            raise DatabaseError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")

    def scan(self, prefix: str) -> List[Dict[str, bytes]]:
        """
        Scan keys with prefix.
        
        Args:
            prefix: Key prefix.
            
        Returns:
            List of {"key": bytes, "value": bytes}
        """
        payload = prefix.encode("utf-8")
        resp = self._request(Message(OpCode.SCAN, payload))
        
        if resp.opcode == OpCode.VALUE:
            # Parse: count(4) + [key_len(2) + key + val_len(4) + val]...
            data = resp.payload
            if len(data) < 4:
                return []
                
            count = struct.unpack("<I", data[:4])[0]
            offset = 4
            results = []
            
            for _ in range(count):
                if offset + 2 > len(data): break
                k_len = struct.unpack("<H", data[offset:offset+2])[0]
                offset += 2
                if offset + k_len > len(data): break
                key = data[offset:offset+k_len]
                offset += k_len
                
                if offset + 4 > len(data): break
                v_len = struct.unpack("<I", data[offset:offset+4])[0]
                offset += 4
                if offset + v_len > len(data): break
                val = data[offset:offset+v_len]
                offset += v_len
                
                results.append({"key": key, "value": val})
                
            return results
            
        if resp.opcode == OpCode.ERROR:
            raise DatabaseError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")

    def create_table(self, name: str, columns: List[Dict[str, Any]]) -> None:
        """
        Create a table.
        
        Args:
            name: Table name.
            columns: List of dicts with 'name', 'type', 'nullable'.
                     Type: 0=Int64, 1=UInt64, 2=Float64, 3=Text, 4=Binary, 5=Bool
        """
        # Payload: name_len(2) + name + col_count(2) + [col_name_len(2) + col_name + type(1) + nullable(1)]...
        name_bytes = name.encode("utf-8")
        payload = struct.pack("<H", len(name_bytes)) + name_bytes
        
        payload += struct.pack("<H", len(columns))
        for col in columns:
            c_name = col["name"].encode("utf-8")
            payload += struct.pack("<H", len(c_name)) + c_name
            payload += bytes([col["type"]])
            payload += bytes([1 if col.get("nullable", False) else 0])
            
        resp = self._request(Message(OpCode.CREATE_TABLE, payload))
        self._check_ok(resp)
    
    # =========================================================================
    # Transaction API
    # =========================================================================
    
    def begin_transaction(self) -> int:
        """
        Begin a new transaction.
        
        Returns:
            Transaction ID.
        """
        resp = self._request(Message(OpCode.BEGIN_TXN))
        if resp.opcode == OpCode.TXN_ID and len(resp.payload) >= 8:
            return struct.unpack("<Q", resp.payload[:8])[0]
        if resp.opcode == OpCode.ERROR:
            raise TransactionError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")
    
    def commit(self, txn_id: int) -> int:
        """
        Commit a transaction.
        
        Args:
            txn_id: Transaction ID from begin_transaction().
            
        Returns:
            Commit timestamp.
        """
        payload = struct.pack("<Q", txn_id)
        resp = self._request(Message(OpCode.COMMIT_TXN, payload))
        if resp.opcode == OpCode.TXN_ID and len(resp.payload) >= 8:
            return struct.unpack("<Q", resp.payload[:8])[0]
        if resp.opcode == OpCode.ERROR:
            raise TransactionError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")
    
    def abort(self, txn_id: int) -> None:
        """
        Abort a transaction.
        
        Args:
            txn_id: Transaction ID from begin_transaction().
        """
        payload = struct.pack("<Q", txn_id)
        resp = self._request(Message(OpCode.ABORT_TXN, payload))
        self._check_ok(resp)
    
    # =========================================================================
    # Administrative Operations
    # =========================================================================
    
    def checkpoint(self) -> None:
        """Force a checkpoint."""
        resp = self._request(Message(OpCode.CHECKPOINT))
        self._check_ok(resp)
    
    def stats(self) -> Dict[str, int]:
        """
        Get server statistics.
        
        Returns:
            Dictionary of stat name to value.
        """
        resp = self._request(Message(OpCode.STATS))
        if resp.opcode == OpCode.STATS_RESP:
            stats = {}
            for line in resp.payload.decode("utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    try:
                        stats[key] = int(value)
                    except ValueError:
                        pass
            return stats
        if resp.opcode == OpCode.ERROR:
            raise DatabaseError(resp.payload.decode("utf-8", errors="replace"))
        raise ProtocolError(f"Unexpected opcode: {resp.opcode:#x}")
