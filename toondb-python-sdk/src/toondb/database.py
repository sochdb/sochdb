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
ToonDB Embedded Database

Direct database access via FFI to the Rust library.
This is the recommended mode for single-process applications.
"""

import os
import sys
import ctypes
from typing import Optional
from contextlib import contextmanager
from .errors import DatabaseError, TransactionError


def _get_target_triple() -> str:
    """Get the Rust target triple for the current platform."""
    import platform
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "aarch64-apple-darwin"
        return "x86_64-apple-darwin"
    elif system == "windows":
        return "x86_64-pc-windows-msvc"
    else:  # Linux
        if machine in ("arm64", "aarch64"):
            return "aarch64-unknown-linux-gnu"
        return "x86_64-unknown-linux-gnu"


def _find_library() -> str:
    """Find the ToonDB native library.
    
    Search order:
    1. TOONDB_LIB_PATH environment variable
    2. Bundled library in wheel (lib/{target}/)
    3. Package directory
    4. Development build (target/release, target/debug)
    5. System paths (/usr/local/lib, /usr/lib)
    """
    # Platform-specific library name
    if sys.platform == "darwin":
        lib_name = "libtoondb_storage.dylib"
    elif sys.platform == "win32":
        lib_name = "toondb_storage.dll"
    else:
        lib_name = "libtoondb_storage.so"
    
    pkg_dir = os.path.dirname(__file__)
    target = _get_target_triple()
    
    # Search paths in priority order
    search_paths = []
    
    # 1. Environment variable override
    env_path = os.environ.get("TOONDB_LIB_PATH")
    if env_path:
        search_paths.append(env_path)
    
    # 2. Bundled library in wheel (platform-specific)
    search_paths.append(os.path.join(pkg_dir, "lib", target))
    
    # 3. Bundled library in wheel (generic)
    search_paths.append(os.path.join(pkg_dir, "lib"))
    
    # 4. Same directory as this file
    search_paths.append(pkg_dir)
    
    # 5. Package root
    search_paths.append(os.path.dirname(os.path.dirname(pkg_dir)))
    
    # 6. Development builds (relative to package)
    search_paths.extend([
        os.path.join(pkg_dir, "..", "..", "..", "target", "release"),
        os.path.join(pkg_dir, "..", "..", "..", "target", "debug"),
    ])
    
    # 7. System paths
    search_paths.extend(["/usr/local/lib", "/usr/lib"])
    
    for path in search_paths:
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            return lib_path
    
    raise DatabaseError(
        f"Could not find {lib_name}. "
        f"Searched in: {', '.join(search_paths[:5])}... "
        "Set TOONDB_LIB_PATH environment variable or install toondb-client with pip."
    )


class C_TxnHandle(ctypes.Structure):
    _fields_ = [
        ("txn_id", ctypes.c_uint64),
        ("snapshot_ts", ctypes.c_uint64),
    ]


class C_StorageStats(ctypes.Structure):
    _fields_ = [
        ("memtable_size_bytes", ctypes.c_uint64),
        ("wal_size_bytes", ctypes.c_uint64),
        ("active_transactions", ctypes.c_size_t),
        ("min_active_snapshot", ctypes.c_uint64),
        ("last_checkpoint_lsn", ctypes.c_uint64),
    ]


class _FFI:
    """FFI bindings to the native library."""
    
    _lib = None
    
    @classmethod
    def get_lib(cls):
        if cls._lib is None:
            lib_path = _find_library()
            cls._lib = ctypes.CDLL(lib_path)
            cls._setup_bindings()
        return cls._lib
    
    @classmethod
    def _setup_bindings(cls):
        """Set up function signatures for the native library."""
        lib = cls._lib
        
        # Database lifecycle
        # toondb_open(path: *const c_char) -> *mut DatabasePtr
        lib.toondb_open.argtypes = [ctypes.c_char_p]
        lib.toondb_open.restype = ctypes.c_void_p
        
        # toondb_close(ptr: *mut DatabasePtr)
        lib.toondb_close.argtypes = [ctypes.c_void_p]
        lib.toondb_close.restype = None
        
        # Transaction API
        # toondb_begin_txn(ptr: *mut DatabasePtr) -> C_TxnHandle
        lib.toondb_begin_txn.argtypes = [ctypes.c_void_p]
        lib.toondb_begin_txn.restype = C_TxnHandle
        
        # toondb_commit(ptr: *mut DatabasePtr, handle: C_TxnHandle) -> c_int
        lib.toondb_commit.argtypes = [ctypes.c_void_p, C_TxnHandle]
        lib.toondb_commit.restype = ctypes.c_int
        
        # toondb_abort(ptr: *mut DatabasePtr, handle: C_TxnHandle) -> c_int
        lib.toondb_abort.argtypes = [ctypes.c_void_p, C_TxnHandle]
        lib.toondb_abort.restype = ctypes.c_int
        
        # Key-Value API
        # toondb_put(ptr, handle, key_ptr, key_len, val_ptr, val_len) -> c_int
        lib.toondb_put.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.toondb_put.restype = ctypes.c_int
        
        # toondb_get(ptr, handle, key_ptr, key_len, val_out, len_out) -> c_int
        lib.toondb_get.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.toondb_get.restype = ctypes.c_int
        
        # toondb_delete(ptr, handle, key_ptr, key_len) -> c_int
        lib.toondb_delete.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.toondb_delete.restype = ctypes.c_int
        
        # toondb_free_bytes(ptr, len)
        lib.toondb_free_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        lib.toondb_free_bytes.restype = None
        
        # Path API
        # toondb_put_path(ptr, handle, path_ptr, val_ptr, val_len) -> c_int
        lib.toondb_put_path.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.toondb_put_path.restype = ctypes.c_int
        
        # toondb_get_path(ptr, handle, path_ptr, val_out, len_out) -> c_int
        lib.toondb_get_path.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.toondb_get_path.restype = ctypes.c_int

        # Scan API
        # toondb_scan(ptr, handle, start_ptr, start_len, end_ptr, end_len) -> *mut ScanIteratorPtr
        lib.toondb_scan.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.toondb_scan.restype = ctypes.c_void_p
        
        # toondb_scan_next(iter_ptr, key_out, key_len_out, val_out, val_len_out) -> c_int
        lib.toondb_scan_next.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.toondb_scan_next.restype = ctypes.c_int
        
        # toondb_scan_free(iter_ptr)
        lib.toondb_scan_free.argtypes = [ctypes.c_void_p]
        lib.toondb_scan_free.restype = None
        
        # toondb_scan_batch(iter_ptr, batch_size, result_out, result_len_out) -> c_int
        # Batched scan for reduced FFI overhead
        lib.toondb_scan_batch.argtypes = [
            ctypes.c_void_p,  # iter_ptr
            ctypes.c_size_t,  # batch_size
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # result_out
            ctypes.POINTER(ctypes.c_size_t)  # result_len_out
        ]
        lib.toondb_scan_batch.restype = ctypes.c_int
        
        # Checkpoint API
        # toondb_checkpoint(ptr) -> u64
        lib.toondb_checkpoint.argtypes = [ctypes.c_void_p]
        lib.toondb_checkpoint.restype = ctypes.c_uint64
        
        # Stats API
        # toondb_stats(ptr) -> C_StorageStats
        lib.toondb_stats.argtypes = [ctypes.c_void_p]
        lib.toondb_stats.restype = C_StorageStats


class Transaction:
    """
    A database transaction.
    
    Use with a context manager for automatic commit/abort:
    
        with db.transaction() as txn:
            txn.put(b"key", b"value")
            # Auto-commits on success, auto-aborts on exception
    """
    
    def __init__(self, db: "Database", handle: C_TxnHandle):
        self._db = db
        self._handle = handle
        self._committed = False
        self._aborted = False
        self._lib = _FFI.get_lib()
    
    @property
    def id(self) -> int:
        """Get the transaction ID."""
        return self._handle.txn_id
    
    def put(self, key: bytes, value: bytes) -> None:
        """Put a key-value pair in this transaction."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        key_ptr = (ctypes.c_uint8 * len(key)).from_buffer_copy(key)
        val_ptr = (ctypes.c_uint8 * len(value)).from_buffer_copy(value)
        
        res = self._lib.toondb_put(
            self._db._handle, self._handle,
            key_ptr, len(key),
            val_ptr, len(value)
        )
        if res != 0:
            raise DatabaseError("Failed to put value")
    
    def get(self, key: bytes) -> Optional[bytes]:
        """Get a value in this transaction's snapshot."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        key_ptr = (ctypes.c_uint8 * len(key)).from_buffer_copy(key)
        val_out = ctypes.POINTER(ctypes.c_uint8)()
        len_out = ctypes.c_size_t()
        
        res = self._lib.toondb_get(
            self._db._handle, self._handle,
            key_ptr, len(key),
            ctypes.byref(val_out), ctypes.byref(len_out)
        )
        
        if res == 1: # Not found
            return None
        elif res != 0:
            raise DatabaseError("Failed to get value")
        
        # Copy data to Python bytes
        data = bytes(val_out[:len_out.value])
        
        # Free Rust memory
        self._lib.toondb_free_bytes(val_out, len_out)
        
        return data
    
    def delete(self, key: bytes) -> None:
        """Delete a key in this transaction."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        key_ptr = (ctypes.c_uint8 * len(key)).from_buffer_copy(key)
        
        res = self._lib.toondb_delete(
            self._db._handle, self._handle,
            key_ptr, len(key)
        )
        if res != 0:
            raise DatabaseError("Failed to delete key")
    
    def put_path(self, path: str, value: bytes) -> None:
        """Put a value at a path."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
            
        path_bytes = path.encode("utf-8")
        val_ptr = (ctypes.c_uint8 * len(value)).from_buffer_copy(value)
        
        res = self._lib.toondb_put_path(
            self._db._handle, self._handle,
            path_bytes,
            val_ptr, len(value)
        )
        if res != 0:
            raise DatabaseError("Failed to put path")

    def get_path(self, path: str) -> Optional[bytes]:
        """Get a value at a path."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
            
        path_bytes = path.encode("utf-8")
        val_out = ctypes.POINTER(ctypes.c_uint8)()
        len_out = ctypes.c_size_t()
        
        res = self._lib.toondb_get_path(
            self._db._handle, self._handle,
            path_bytes,
            ctypes.byref(val_out), ctypes.byref(len_out)
        )
        
        if res == 1: # Not found
            return None
        elif res != 0:
            raise DatabaseError("Failed to get path")
            
        data = bytes(val_out[:len_out.value])
        self._lib.toondb_free_bytes(val_out, len_out)
        return data

    def scan(self, start: bytes = b"", end: bytes = b""):
        """
        Scan keys in range [start, end).
        
        Args:
            start: Start key (inclusive). Empty means from beginning.
            end: End key (exclusive). Empty means to end.
            
        Yields:
            (key, value) tuples.
        """
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
            
        start_ptr = (ctypes.c_uint8 * len(start)).from_buffer_copy(start)
        end_ptr = (ctypes.c_uint8 * len(end)).from_buffer_copy(end)
        
        iter_ptr = self._lib.toondb_scan(
            self._db._handle, self._handle,
            start_ptr, len(start),
            end_ptr, len(end)
        )
        
        if not iter_ptr:
            return
            
        try:
            key_out = ctypes.POINTER(ctypes.c_uint8)()
            key_len = ctypes.c_size_t()
            val_out = ctypes.POINTER(ctypes.c_uint8)()
            val_len = ctypes.c_size_t()
            
            while True:
                res = self._lib.toondb_scan_next(
                    iter_ptr,
                    ctypes.byref(key_out), ctypes.byref(key_len),
                    ctypes.byref(val_out), ctypes.byref(val_len)
                )
                
                if res == 1: # End of scan
                    break
                elif res != 0: # Error
                    raise DatabaseError("Scan failed")
                    
                # Copy data
                key = bytes(key_out[:key_len.value])
                val = bytes(val_out[:val_len.value])
                
                # Free Rust memory
                self._lib.toondb_free_bytes(key_out, key_len)
                self._lib.toondb_free_bytes(val_out, val_len)
                
                yield key, val
        finally:
            self._lib.toondb_scan_free(iter_ptr)

    def scan_batched(self, start: bytes = b"", end: bytes = b"", batch_size: int = 1000):
        """
        Scan keys in range [start, end) with batched FFI calls.
        
        This is a high-performance scan that fetches multiple results per FFI call,
        dramatically reducing overhead for large scans.
        
        Performance comparison (10,000 results, 500ns FFI overhead):
        - scan():        10,000 FFI calls = 5ms overhead
        - scan_batched(): 10 FFI calls = 5µs overhead (1000x faster)
        
        Args:
            start: Start key (inclusive). Empty means from beginning.
            end: End key (exclusive). Empty means to end.
            batch_size: Number of results to fetch per FFI call. Default 1000.
            
        Yields:
            (key, value) tuples.
        """
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        if batch_size <= 0:
            batch_size = 1000
            
        start_ptr = (ctypes.c_uint8 * len(start)).from_buffer_copy(start)
        end_ptr = (ctypes.c_uint8 * len(end)).from_buffer_copy(end)
        
        iter_ptr = self._lib.toondb_scan(
            self._db._handle, self._handle,
            start_ptr, len(start),
            end_ptr, len(end)
        )
        
        if not iter_ptr:
            return
            
        try:
            result_out = ctypes.POINTER(ctypes.c_uint8)()
            result_len = ctypes.c_size_t()
            
            while True:
                res = self._lib.toondb_scan_batch(
                    iter_ptr,
                    batch_size,
                    ctypes.byref(result_out),
                    ctypes.byref(result_len)
                )
                
                if res == 1:  # Scan complete
                    # Free the minimal buffer allocated
                    if result_out and result_len.value > 0:
                        self._lib.toondb_free_bytes(result_out, result_len)
                    break
                elif res != 0:  # Error
                    if result_out and result_len.value > 0:
                        self._lib.toondb_free_bytes(result_out, result_len)
                    raise DatabaseError("Batched scan failed")
                
                # Parse batch result
                # Format: [num_results: u32][is_done: u8][entries...]
                data = bytes(result_out[:result_len.value])
                
                if len(data) < 5:
                    self._lib.toondb_free_bytes(result_out, result_len)
                    break
                    
                num_results = int.from_bytes(data[0:4], 'little')
                is_done = data[4] != 0
                
                offset = 5
                for _ in range(num_results):
                    if offset + 8 > len(data):
                        break
                    key_len = int.from_bytes(data[offset:offset+4], 'little')
                    val_len = int.from_bytes(data[offset+4:offset+8], 'little')
                    offset += 8
                    
                    if offset + key_len + val_len > len(data):
                        break
                    
                    key = data[offset:offset+key_len]
                    offset += key_len
                    val = data[offset:offset+val_len]
                    offset += val_len
                    
                    yield key, val
                
                # Free batch buffer
                self._lib.toondb_free_bytes(result_out, result_len)
                
                if is_done:
                    break
        finally:
            self._lib.toondb_scan_free(iter_ptr)

    def commit(self) -> int:
        """
        Commit the transaction.
        
        Returns:
            Commit timestamp.
        """
        if self._committed:
            raise TransactionError("Transaction already committed")
        if self._aborted:
            raise TransactionError("Transaction already aborted")
        
        res = self._lib.toondb_commit(self._db._handle, self._handle)
        if res != 0:
            raise TransactionError("Failed to commit transaction")
            
        self._committed = True
        return 0 # TODO: Return actual commit timestamp if exposed
    
    def abort(self) -> None:
        """Abort the transaction."""
        if self._committed:
            raise TransactionError("Transaction already committed")
        if self._aborted:
            return  # Abort is idempotent
        
        self._lib.toondb_abort(self._db._handle, self._handle)
        self._aborted = True
    
    def __enter__(self) -> "Transaction":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Exception occurred, abort
            self.abort()
        elif not self._committed and not self._aborted:
            # No exception and not yet completed, commit
            self.commit()


class Database:
    """
    ToonDB Embedded Database.
    
    Provides direct access to a ToonDB database file.
    This is the recommended mode for single-process applications.
    
    Example:
        db = Database.open("./my_database")
        db.put(b"key", b"value")
        value = db.get(b"key")
        db.close()
    
    Or with context manager:
        with Database.open("./my_database") as db:
            db.put(b"key", b"value")
    """
    
    def __init__(self, path: str, _handle):
        """
        Initialize a database connection.
        
        Use Database.open() to create instances.
        """
        self._path = path
        self._handle = _handle
        self._closed = False
        self._lib = _FFI.get_lib()
    
    @classmethod
    def open(cls, path: str) -> "Database":
        """
        Open a database at the given path.
        
        Creates the database if it doesn't exist.
        
        Args:
            path: Path to the database directory.
            
        Returns:
            Database instance.
        """
        lib = _FFI.get_lib()
        path_bytes = path.encode("utf-8")
        handle = lib.toondb_open(path_bytes)
        
        if not handle:
            raise DatabaseError(f"Failed to open database at {path}")
            
        return cls(path, handle)
    
    def close(self) -> None:
        """Close the database."""
        if self._closed:
            return
        
        if self._handle:
            self._lib.toondb_close(self._handle)
            self._handle = None
            
        self._closed = True
    
    def __enter__(self) -> "Database":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # =========================================================================
    # Key-Value API (auto-commit)
    # =========================================================================
    
    def put(self, key: bytes, value: bytes) -> None:
        """
        Put a key-value pair (auto-commit).
        
        Args:
            key: The key bytes.
            value: The value bytes.
        """
        with self.transaction() as txn:
            txn.put(key, value)
    
    def get(self, key: bytes) -> Optional[bytes]:
        """
        Get a value by key.
        
        Args:
            key: The key bytes.
            
        Returns:
            The value bytes, or None if not found.
        """
        # For single reads, we still need a transaction for MVCC consistency
        with self.transaction() as txn:
            return txn.get(key)
    
    def delete(self, key: bytes) -> None:
        """
        Delete a key (auto-commit).
        
        Args:
            key: The key bytes.
        """
        with self.transaction() as txn:
            txn.delete(key)
    
    # =========================================================================
    # Path-Native API
    # =========================================================================
    
    def put_path(self, path: str, value: bytes) -> None:
        """
        Put a value at a path (auto-commit).
        
        Args:
            path: Path string (e.g., "users/alice/email")
            value: The value bytes.
        """
        with self.transaction() as txn:
            txn.put_path(path, value)
    
    def get_path(self, path: str) -> Optional[bytes]:
        """
        Get a value at a path.
        
        Args:
            path: Path string (e.g., "users/alice/email")
            
        Returns:
            The value bytes, or None if not found.
        """
        with self.transaction() as txn:
            return txn.get_path(path)

    def scan(self, start: bytes = b"", end: bytes = b""):
        """
        Scan keys in range (auto-commit transaction).
        
        Args:
            start: Start key (inclusive).
            end: End key (exclusive).
            
        Yields:
            (key, value) tuples.
        """
        with self.transaction() as txn:
            yield from txn.scan(start, end)
    
    def delete_path(self, path: str) -> None:
        """
        Delete at a path (auto-commit).
        
        Args:
            path: Path string (e.g., "users/alice/email")
        """
        # Currently no direct delete_path FFI, use key-based delete if possible
        # or implement delete_path in FFI. For now, assume path is key.
        self.delete(path.encode("utf-8"))
    
    # =========================================================================
    # Transaction API
    # =========================================================================
    
    def transaction(self) -> Transaction:
        """
        Begin a new transaction.
        
        Returns:
            Transaction object that can be used as a context manager.
            
        Example:
            with db.transaction() as txn:
                txn.put(b"key1", b"value1")
                txn.put(b"key2", b"value2")
                # Auto-commits on success
        """
        self._check_open()
        handle = self._lib.toondb_begin_txn(self._handle)
        if handle.txn_id == 0:
            raise DatabaseError("Failed to begin transaction")
            
        return Transaction(self, handle)
    
    # =========================================================================
    # Administrative Operations
    # =========================================================================
    
    def checkpoint(self) -> int:
        """
        Force a checkpoint to disk.
        
        Returns:
            LSN of the checkpoint.
        """
        self._check_open()
        return self._lib.toondb_checkpoint(self._handle)
        
    def stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with statistics.
        """
        self._check_open()
        stats = self._lib.toondb_stats(self._handle)
        return {
            "memtable_size_bytes": stats.memtable_size_bytes,
            "wal_size_bytes": stats.wal_size_bytes,
            "active_transactions": stats.active_transactions,
            "min_active_snapshot": stats.min_active_snapshot,
            "last_checkpoint_lsn": stats.last_checkpoint_lsn,
        }
    
    def _check_open(self) -> None:
        """Check that database is open."""
        if self._closed:
            raise DatabaseError("Database is closed")
