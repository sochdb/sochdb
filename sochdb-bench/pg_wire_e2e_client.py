#!/usr/bin/env python3
"""Self-contained PostgreSQL wire-protocol client for end-to-end testing the
SochDB pg_wire SQL executor. Zero external dependencies (raw sockets only).

Speaks the v3.0 simple query protocol: StartupMessage -> AuthenticationOk ->
ParameterStatus* -> BackendKeyData -> ReadyForQuery, then Query ('Q') /
RowDescription ('T') / DataRow ('D') / CommandComplete ('C') / ErrorResponse
('E') / ReadyForQuery ('Z').

Usage: pg_wire_e2e_client.py <host> <port>
Exit code 0 = all assertions passed, non-zero = failure.
"""

import socket
import struct
import sys

PROTOCOL_V3 = 196608  # 3 << 16


class PgClient:
    def __init__(self, host, port, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""

    # ---- low-level framing ----
    def _recv_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("server closed connection")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _read_message(self):
        """Return (type_byte:str, payload:bytes)."""
        type_byte = self._recv_exact(1)
        (length,) = struct.unpack("!I", self._recv_exact(4))
        payload = self._recv_exact(length - 4) if length > 4 else b""
        return chr(type_byte[0]), payload

    # ---- handshake ----
    def startup(self, user="postgres", database="postgres"):
        params = b""
        for k, v in (("user", user), ("database", database)):
            params += k.encode() + b"\x00" + v.encode() + b"\x00"
        params += b"\x00"
        body = struct.pack("!I", PROTOCOL_V3) + params
        self.sock.sendall(struct.pack("!I", len(body) + 4) + body)

        saw_auth_ok = False
        saw_ready = False
        while not saw_ready:
            t, payload = self._read_message()
            if t == "R":
                (code,) = struct.unpack("!I", payload[:4])
                if code == 0:
                    saw_auth_ok = True
            elif t == "Z":
                saw_ready = True
            elif t == "E":
                raise RuntimeError("startup error: " + _decode_error(payload))
            # ignore S (ParameterStatus), K (BackendKeyData), N (Notice)
        if not saw_auth_ok:
            raise RuntimeError("did not receive AuthenticationOk")
        return True

    # ---- simple query ----
    def query(self, sql):
        body = sql.encode() + b"\x00"
        self.sock.sendall(b"Q" + struct.pack("!I", len(body) + 4) + body)

        columns = []
        rows = []
        command_tag = None
        error = None
        while True:
            t, payload = self._read_message()
            if t == "T":  # RowDescription
                columns = _parse_row_description(payload)
            elif t == "D":  # DataRow
                rows.append(_parse_data_row(payload))
            elif t == "C":  # CommandComplete
                command_tag = payload.rstrip(b"\x00").decode()
            elif t == "E":  # ErrorResponse
                error = _decode_error(payload)
            elif t == "I":  # EmptyQueryResponse
                command_tag = ""
            elif t == "Z":  # ReadyForQuery -> turn complete
                break
            # ignore N (Notice), S (ParameterStatus)
        return {"columns": columns, "rows": rows, "tag": command_tag, "error": error}

    def close(self):
        try:
            self.sock.sendall(b"X" + struct.pack("!I", 4))  # Terminate
        except OSError:
            pass
        self.sock.close()


def _parse_row_description(payload):
    (field_count,) = struct.unpack("!H", payload[:2])
    pos = 2
    names = []
    for _ in range(field_count):
        end = payload.index(b"\x00", pos)
        names.append(payload[pos:end].decode())
        pos = end + 1
        pos += 18  # table_oid(4) col_attr(2) type_oid(4) type_size(2) type_mod(4) format(2)
    return names


def _parse_data_row(payload):
    (col_count,) = struct.unpack("!H", payload[:2])
    pos = 2
    values = []
    for _ in range(col_count):
        (length,) = struct.unpack("!i", payload[pos:pos + 4])
        pos += 4
        if length == -1:
            values.append(None)
        else:
            values.append(payload[pos:pos + length].decode())
            pos += length
    return values


def _decode_error(payload):
    fields = {}
    pos = 0
    while pos < len(payload):
        code = payload[pos:pos + 1]
        if code == b"\x00":
            break
        end = payload.index(b"\x00", pos + 1)
        fields[code.decode()] = payload[pos + 1:end].decode()
        pos = end + 1
    return fields.get("M", str(fields))


# ---------------------------------------------------------------------------
# Test scenario
# ---------------------------------------------------------------------------
def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5433

    failures = []

    def check(cond, msg):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    c = PgClient(host, port)
    print(f"Connecting to SochDB pg_wire at {host}:{port} ...")
    c.startup()
    print("Handshake OK (AuthenticationOk + ReadyForQuery received)\n")

    # 1. CREATE TABLE
    r = c.query("CREATE TABLE accounts (id INT, name TEXT, balance INT)")
    check(r["error"] is None, f"CREATE TABLE (tag={r['tag']!r}, err={r['error']})")

    # 2. INSERT rows
    for i, (name, bal) in enumerate([("alice", 100), ("bob", 250), ("carol", 50)], start=1):
        r = c.query(f"INSERT INTO accounts (id, name, balance) VALUES ({i}, '{name}', {bal})")
        check(r["error"] is None, f"INSERT {name} (tag={r['tag']!r}, err={r['error']})")

    # 3. SELECT all -> expect 3 real rows
    r = c.query("SELECT id, name, balance FROM accounts")
    check(r["error"] is None, f"SELECT no error (err={r['error']})")
    check(len(r["rows"]) == 3, f"SELECT returned 3 rows (got {len(r['rows'])})")
    print(f"    columns={r['columns']}")
    for row in r["rows"]:
        print(f"    row={row}")
    names = {row[1] for row in r["rows"] if len(row) > 1}
    check(names == {"alice", "bob", "carol"},
          f"SELECT returned the inserted names (got {sorted(names)})")

    # 4. Prove it's NOT the echo executor: echo would not return structured rows
    #    for a SELECT and the tag would echo the SQL. Here we require a real
    #    SELECT command tag and real column metadata.
    check(r["tag"] is not None and r["tag"].upper().startswith("SELECT"),
          f"CommandComplete tag is a real SELECT tag (got {r['tag']!r})")
    check(r["columns"] == ["id", "name", "balance"],
          f"RowDescription matches projected columns (got {r['columns']})")

    c.close()

    print()
    if failures:
        print(f"END-TO-END RESULT: FAILED ({len(failures)} assertion(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("END-TO-END RESULT: ALL PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
