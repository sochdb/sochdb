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
import shutil
from toondb.database import Database

DB_PATH = "./test_ffi_db"

def test_ffi():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
    print(f"Opening database at {DB_PATH}...")
    db = Database.open(DB_PATH)
    
    print("Writing data...")
    db.put(b"key1", b"value1")
    db.put(b"key2", b"value2")
    
    print("Reading data...")
    val1 = db.get(b"key1")
    val2 = db.get(b"key2")
    
    print(f"key1: {val1}")
    print(f"key2: {val2}")
    
    assert val1 == b"value1"
    assert val2 == b"value2"
    
    print("Testing paths...")
    db.put_path("users/alice/email", b"alice@example.com")
    email = db.get_path("users/alice/email")
    print(f"email: {email}")
    assert email == b"alice@example.com"
    
    print("Testing transactions...")
    with db.transaction() as txn:
        txn.put(b"txn_key", b"txn_value")
        # Should be visible inside txn
        assert txn.get(b"txn_key") == b"txn_value"
        
    # Should be visible after commit
    assert db.get(b"txn_key") == b"txn_value"
    
    print("Testing delete...")
    db.delete(b"key1")
    val = db.get(b"key1")
    assert val is None
    
    print("Testing scan...")
    # Insert some data for scanning
    db.put(b"scan:1", b"val1")
    db.put(b"scan:2", b"val2")
    db.put(b"scan:3", b"val3")
    
    # Scan all
    results = list(db.scan(b"scan:", b"scan;"))
    print(f"Scan results: {results}")
    assert len(results) == 3
    assert results[0] == (b"scan:1", b"val1")
    assert results[2] == (b"scan:3", b"val3")
    
    # Scan range
    results = list(db.scan(b"scan:2", b"scan:4"))
    assert len(results) == 2
    assert results[0] == (b"scan:2", b"val2")
    
    print("Testing checkpoint...")
    lsn = db.checkpoint()
    print(f"Checkpoint LSN: {lsn}")
    assert lsn >= 0
    
    print("Testing stats...")
    stats = db.stats()
    print(f"Stats: {stats}")
    assert stats["memtable_size_bytes"] >= 0
    assert stats["active_transactions"] >= 0
    
    db.close()
    print("Success!")

if __name__ == "__main__":
    try:
        test_ffi()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
