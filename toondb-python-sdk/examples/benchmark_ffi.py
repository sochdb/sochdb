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

import time
import os
import shutil
import ctypes
from toondb.database import Database

def benchmark_ffi():
    DB_PATH = "./bench_ffi_db"
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    print(f"Opening database at {DB_PATH}...")
    db = Database.open(DB_PATH)
    
    N = 100_000
    print(f"Benchmarking with {N} records...")
    
    # Pre-generate data to avoid measuring Python string overhead
    print("Pre-generating data...")
    keys = [f"users/{i}".encode('utf-8') for i in range(N)]
    # JSON-like value similar to Rust benchmark
    values = [f'{{"id":{i},"name":"User {i}","email":"user{i}@example.com","score":{i % 100}}}'.encode('utf-8') for i in range(N)]
    
    print("\n--- ToonDB FFI Benchmark ---")
    
    # Insert Benchmark
    start_time = time.perf_counter()
    
    # Single transaction for fair comparison with Rust "put_raw" benchmark
    with db.transaction() as txn:
        for i in range(N):
            txn.put(keys[i], values[i])
            
    end_time = time.perf_counter()
    insert_duration = end_time - start_time
    print(f"Insert: {insert_duration:.2f}s")
    print(f"Insert Rate: {N / insert_duration:.0f} ops/sec")
    
    # Scan Benchmark
    start_time = time.perf_counter()
    
    # Scan all
    count = 0
    for _ in db.scan():
        count += 1
        
    end_time = time.perf_counter()
    scan_duration = end_time - start_time
    print(f"Read (Scan): {scan_duration:.2f}s ({count} rows)")
    print(f"Read Rate: {count / scan_duration:.0f} ops/sec")
    
    db.close()
    # Cleanup
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # --- SQLite Benchmark ---
    import sqlite3
    print("\n--- SQLite Benchmark ---")
    SQLITE_DB_PATH = "./bench_sqlite.db"
    if os.path.exists(SQLITE_DB_PATH):
        os.remove(SQLITE_DB_PATH)
        
    conn = sqlite3.connect(SQLITE_DB_PATH)
    # Optimize SQLite for fair comparison (WAL mode, normal sync)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("CREATE TABLE users (key TEXT PRIMARY KEY, value TEXT)")
    
    # Insert Benchmark
    start_time = time.perf_counter()
    
    # Use executemany for fair comparison (batch insert)
    # But ToonDB benchmark used a loop in a transaction, so let's match that exactly
    # to measure Python overhead + DB overhead per op
    with conn:
        for i in range(N):
            conn.execute("INSERT INTO users (key, value) VALUES (?, ?)", (keys[i], values[i]))
            
    end_time = time.perf_counter()
    sqlite_insert_duration = end_time - start_time
    print(f"Insert: {sqlite_insert_duration:.2f}s")
    print(f"Insert Rate: {N / sqlite_insert_duration:.0f} ops/sec")
    
    # Scan Benchmark
    start_time = time.perf_counter()
    
    cursor = conn.execute("SELECT key, value FROM users")
    count = 0
    for _ in cursor:
        count += 1
        
    end_time = time.perf_counter()
    sqlite_scan_duration = end_time - start_time
    print(f"Read (Scan): {sqlite_scan_duration:.2f}s ({count} rows)")
    print(f"Read Rate: {count / sqlite_scan_duration:.0f} ops/sec")
    
    conn.close()
    if os.path.exists(SQLITE_DB_PATH):
        os.remove(SQLITE_DB_PATH)
        
    # Comparison
    print("\n--- Comparison (ToonDB vs SQLite) ---")
    print(f"Insert Speedup: {sqlite_insert_duration / insert_duration:.2f}x")
    print(f"Scan Speedup:   {sqlite_scan_duration / scan_duration:.2f}x")

if __name__ == "__main__":
    benchmark_ffi()
