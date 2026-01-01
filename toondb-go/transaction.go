// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

import "sync"

// Transaction represents an atomic database transaction.
//
// Use Database.WithTransaction for automatic commit/abort handling.
type Transaction struct {
	db        *Database
	txnID     uint64
	mu        sync.Mutex
	committed bool
	aborted   bool
}

// Get retrieves a value by key within the transaction.
func (txn *Transaction) Get(key []byte) ([]byte, error) {
	txn.mu.Lock()
	defer txn.mu.Unlock()

	if err := txn.ensureActive(); err != nil {
		return nil, err
	}

	return txn.db.client.Get(key)
}

// GetString retrieves a value by string key within the transaction.
func (txn *Transaction) GetString(key string) ([]byte, error) {
	return txn.Get([]byte(key))
}

// Put stores a key-value pair within the transaction.
func (txn *Transaction) Put(key, value []byte) error {
	txn.mu.Lock()
	defer txn.mu.Unlock()

	if err := txn.ensureActive(); err != nil {
		return err
	}

	return txn.db.client.Put(key, value)
}

// PutString stores a key-value pair with string key and value.
func (txn *Transaction) PutString(key, value string) error {
	return txn.Put([]byte(key), []byte(value))
}

// Delete removes a key within the transaction.
func (txn *Transaction) Delete(key []byte) error {
	txn.mu.Lock()
	defer txn.mu.Unlock()

	if err := txn.ensureActive(); err != nil {
		return err
	}

	return txn.db.client.Delete(key)
}

// DeleteString removes a key by string within the transaction.
func (txn *Transaction) DeleteString(key string) error {
	return txn.Delete([]byte(key))
}

// GetPath retrieves a value by path within the transaction.
func (txn *Transaction) GetPath(path string) ([]byte, error) {
	txn.mu.Lock()
	defer txn.mu.Unlock()

	if err := txn.ensureActive(); err != nil {
		return nil, err
	}

	return txn.db.client.GetPath(path)
}

// PutPath stores a value at a path within the transaction.
func (txn *Transaction) PutPath(path string, value []byte) error {
	txn.mu.Lock()
	defer txn.mu.Unlock()

	if err := txn.ensureActive(); err != nil {
		return err
	}

	return txn.db.client.PutPath(path, value)
}

// PutPathString stores a string value at a path within the transaction.
func (txn *Transaction) PutPathString(path, value string) error {
	return txn.PutPath(path, []byte(value))
}

// Commit commits the transaction.
func (txn *Transaction) Commit() error {
	txn.mu.Lock()
	defer txn.mu.Unlock()

	if err := txn.ensureActive(); err != nil {
		return err
	}

	if err := txn.db.client.CommitTransaction(txn.txnID); err != nil {
		return &TransactionError{
			TxnID:   txn.txnID,
			Op:      "commit",
			Message: err.Error(),
			Err:     err,
		}
	}

	txn.committed = true
	return nil
}

// Abort aborts/rolls back the transaction.
func (txn *Transaction) Abort() error {
	txn.mu.Lock()
	defer txn.mu.Unlock()

	if txn.committed || txn.aborted {
		return nil
	}

	if err := txn.db.client.AbortTransaction(txn.txnID); err != nil {
		return &TransactionError{
			TxnID:   txn.txnID,
			Op:      "abort",
			Message: err.Error(),
			Err:     err,
		}
	}

	txn.aborted = true
	return nil
}

// ID returns the transaction ID.
func (txn *Transaction) ID() uint64 {
	return txn.txnID
}

// IsActive returns true if the transaction is still active.
func (txn *Transaction) IsActive() bool {
	txn.mu.Lock()
	defer txn.mu.Unlock()
	return !txn.committed && !txn.aborted
}

func (txn *Transaction) ensureActive() error {
	if txn.committed {
		return ErrTxnCommitted
	}
	if txn.aborted {
		return ErrTxnAborted
	}
	return nil
}
