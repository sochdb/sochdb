// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Package toondb provides a Go client for ToonDB, a high-performance
// embedded document database with HNSW vector search.
package toondb

import (
	"errors"
	"fmt"
)

// Standard errors returned by ToonDB operations.
var (
	// ErrNotFound is returned when a key is not found.
	ErrNotFound = errors.New("toondb: key not found")

	// ErrClosed is returned when operating on a closed database.
	ErrClosed = errors.New("toondb: database is closed")

	// ErrTxnClosed is returned when operating on a closed transaction.
	ErrTxnClosed = errors.New("toondb: transaction is closed")

	// ErrTxnCommitted is returned when operating on a committed transaction.
	ErrTxnCommitted = errors.New("toondb: transaction already committed")

	// ErrTxnAborted is returned when operating on an aborted transaction.
	ErrTxnAborted = errors.New("toondb: transaction already aborted")

	// ErrConnectionFailed is returned when connection to the server fails.
	ErrConnectionFailed = errors.New("toondb: connection failed")

	// ErrProtocol is returned on wire protocol errors.
	ErrProtocol = errors.New("toondb: protocol error")

	// ErrInvalidPath is returned when a path is malformed.
	ErrInvalidPath = errors.New("toondb: invalid path")

	// ErrVectorDimension is returned when vector dimensions don't match.
	ErrVectorDimension = errors.New("toondb: vector dimension mismatch")
)

// ToonDBError wraps errors with additional context.
type ToonDBError struct {
	Op      string // Operation that failed (e.g., "get", "put", "connect")
	Path    string // Path or key involved
	Message string // Human-readable message
	Err     error  // Underlying error
}

// Error implements the error interface.
func (e *ToonDBError) Error() string {
	if e.Path != "" {
		return fmt.Sprintf("toondb: %s %s: %s", e.Op, e.Path, e.Message)
	}
	return fmt.Sprintf("toondb: %s: %s", e.Op, e.Message)
}

// Unwrap returns the underlying error.
func (e *ToonDBError) Unwrap() error {
	return e.Err
}

// Is implements error matching.
func (e *ToonDBError) Is(target error) bool {
	if e.Err != nil {
		return errors.Is(e.Err, target)
	}
	return false
}

// ConnectionError represents a connection failure.
type ConnectionError struct {
	Address string
	Err     error
}

// Error implements the error interface.
func (e *ConnectionError) Error() string {
	return fmt.Sprintf("toondb: failed to connect to %s: %v", e.Address, e.Err)
}

// Unwrap returns the underlying error.
func (e *ConnectionError) Unwrap() error {
	return e.Err
}

// TransactionError represents a transaction-related failure.
type TransactionError struct {
	TxnID   uint64
	Op      string
	Message string
	Err     error
}

// Error implements the error interface.
func (e *TransactionError) Error() string {
	return fmt.Sprintf("toondb: transaction %d %s: %s", e.TxnID, e.Op, e.Message)
}

// Unwrap returns the underlying error.
func (e *TransactionError) Unwrap() error {
	return e.Err
}

// ProtocolError represents a wire protocol error.
type ProtocolError struct {
	Expected string
	Got      string
	Message  string
}

// Error implements the error interface.
func (e *ProtocolError) Error() string {
	if e.Expected != "" && e.Got != "" {
		return fmt.Sprintf("toondb: protocol error: expected %s, got %s", e.Expected, e.Got)
	}
	return fmt.Sprintf("toondb: protocol error: %s", e.Message)
}
