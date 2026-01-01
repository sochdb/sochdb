#!/usr/bin/env python3
"""
Mock Server for testing IPC client examples.
Run this if you don't have the Rust IPC server available.
"""

import os, socket, struct, threading, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from toondb.ipc_client import OpCode, Message

class MockServer:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.running = False
        self.data = {}
    
    def start(self):
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(5)
        self.running = True
        print(f"Mock server listening on {self.socket_path}")
        
        while self.running:
            try:
                conn, _ = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn,)).start()
            except:
                break
    
    def _handle(self, conn):
        try:
            while self.running:
                msg = Message.decode(conn)
                resp = self._process(msg)
                conn.sendall(resp.encode())
        except:
            pass
        finally:
            conn.close()
    
    def _process(self, msg):
        if msg.opcode == OpCode.PING:
            return Message(OpCode.PONG)
        elif msg.opcode == OpCode.PUT:
            key_len = struct.unpack("<I", msg.payload[:4])[0]
            key = msg.payload[4:4+key_len]
            val = msg.payload[4+key_len:]
            self.data[key] = val
            return Message(OpCode.OK)
        elif msg.opcode == OpCode.GET:
            val = self.data.get(msg.payload, b"")
            return Message(OpCode.VALUE, val)
        elif msg.opcode == OpCode.BEGIN_TXN:
            return Message(OpCode.TXN_ID, struct.pack("<Q", 1))
        elif msg.opcode == OpCode.COMMIT_TXN:
            return Message(OpCode.TXN_ID, struct.pack("<Q", 1))
        elif msg.opcode == OpCode.STATS:
            return Message(OpCode.STATS_RESP, b"keys=0\n")
        return Message(OpCode.OK)
    
    def stop(self):
        self.running = False
        self.sock.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

if __name__ == "__main__":
    server = MockServer("/tmp/toondb.sock")
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
