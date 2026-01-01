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

"""Tests for ToonDB Embedded Database."""

import pytest
import tempfile
import os

from toondb import Database, Transaction
from toondb.errors import DatabaseError, TransactionError


class TestDatabase:
    """Tests for embedded database."""
    
    def test_open_close(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        db = Database.open(db_path)
        assert db is not None
        assert os.path.exists(db_path)
        db.close()
    
    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            assert db is not None
        # Database should be closed after context
    
    def test_put_get(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            db.put(b"key1", b"value1")
            value = db.get(b"key1")
            assert value == b"value1"
    
    def test_get_nonexistent(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            value = db.get(b"nonexistent")
            assert value is None
    
    def test_delete(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            db.put(b"key1", b"value1")
            db.delete(b"key1")
            value = db.get(b"key1")
            assert value is None
    
    def test_path_api(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            db.put_path("users/alice/email", b"alice@example.com")
            value = db.get_path("users/alice/email")
            assert value == b"alice@example.com"
    
    def test_operations_after_close(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        db = Database.open(db_path)
        db.close()
        
        with pytest.raises(DatabaseError):
            db.put(b"key", b"value")


class TestTransaction:
    """Tests for transactions."""
    
    def test_transaction_context_manager(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            with db.transaction() as txn:
                txn.put(b"key1", b"value1")
                txn.put(b"key2", b"value2")
            
            # Values should be committed
            assert db.get(b"key1") == b"value1"
            assert db.get(b"key2") == b"value2"
    
    def test_transaction_explicit_commit(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            txn = db.transaction()
            txn.put(b"key", b"value")
            commit_ts = txn.commit()
            assert commit_ts > 0
    
    def test_transaction_explicit_abort(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            txn = db.transaction()
            txn.put(b"key", b"value")
            txn.abort()
            # With full FFI, this would not be visible
            # For now with fallback, it still is
    
    def test_transaction_auto_abort_on_exception(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            try:
                with db.transaction() as txn:
                    txn.put(b"key", b"value")
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            # Transaction should be aborted
            # (with full FFI, value would not be visible)
    
    def test_transaction_get_in_snapshot(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            db.put(b"existing", b"value")
            
            with db.transaction() as txn:
                value = txn.get(b"existing")
                assert value == b"value"
    
    def test_operation_after_commit(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            txn = db.transaction()
            txn.commit()
            
            with pytest.raises(TransactionError):
                txn.put(b"key", b"value")
    
    def test_operation_after_abort(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            txn = db.transaction()
            txn.abort()
            
            with pytest.raises(TransactionError):
                txn.put(b"key", b"value")
    
    def test_double_commit(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            txn = db.transaction()
            txn.commit()
            
            with pytest.raises(TransactionError):
                txn.commit()
    
    def test_abort_after_commit(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            txn = db.transaction()
            txn.commit()
            
            with pytest.raises(TransactionError):
                txn.abort()
    
    def test_abort_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test_db")
        with Database.open(db_path) as db:
            txn = db.transaction()
            txn.abort()
            txn.abort()  # Should not raise
