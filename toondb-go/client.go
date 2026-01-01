// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"path/filepath"
	"sync"
)

// OpCode represents the wire protocol operation codes.
type OpCode uint8

const (
	OpGet             OpCode = 0x01
	OpPut             OpCode = 0x02
	OpDelete          OpCode = 0x03
	OpGetPath         OpCode = 0x04
	OpPutPath         OpCode = 0x05
	OpQuery           OpCode = 0x06
	OpBeginTxn        OpCode = 0x10
	OpCommitTxn       OpCode = 0x11
	OpAbortTxn        OpCode = 0x12
	OpCheckpoint      OpCode = 0x20
	OpStats           OpCode = 0x21
	OpVectorQuery     OpCode = 0x30
	OpVectorBulkBuild OpCode = 0x31
	OpVectorInfo      OpCode = 0x32
	OpOK              OpCode = 0x80
	OpError           OpCode = 0x81
	OpNotFound        OpCode = 0x82
)

// IPCClient handles low-level IPC communication with the ToonDB server.
type IPCClient struct {
	conn   net.Conn
	mu     sync.Mutex
	closed bool
}

// Connect establishes a connection to the ToonDB server.
func Connect(socketPath string) (*IPCClient, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, &ConnectionError{Address: socketPath, Err: err}
	}
	return &IPCClient{conn: conn}, nil
}

// ConnectToDatabase connects to a database at the given path.
func ConnectToDatabase(dbPath string) (*IPCClient, error) {
	socketPath := filepath.Join(dbPath, "toondb.sock")
	return Connect(socketPath)
}

// Close closes the connection.
func (c *IPCClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true
	return c.conn.Close()
}

// Get retrieves a value by key.
func (c *IPCClient) Get(key []byte) ([]byte, error) {
	return c.sendKeyOp(OpGet, key)
}

// Put stores a key-value pair.
func (c *IPCClient) Put(key, value []byte) error {
	_, err := c.sendKeyValueOp(OpPut, key, value)
	return err
}

// Delete removes a key.
func (c *IPCClient) Delete(key []byte) error {
	_, err := c.sendKeyOp(OpDelete, key)
	return err
}

// GetPath retrieves a value by path.
func (c *IPCClient) GetPath(path string) ([]byte, error) {
	return c.sendKeyOp(OpGetPath, []byte(path))
}

// PutPath stores a value at a path.
func (c *IPCClient) PutPath(path string, value []byte) error {
	_, err := c.sendKeyValueOp(OpPutPath, []byte(path), value)
	return err
}

// Query executes a prefix query.
func (c *IPCClient) Query(prefix string, limit, offset int) ([]KeyValue, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	// Build query message
	prefixBytes := []byte(prefix)
	msgLen := 1 + 4 + len(prefixBytes) + 4 + 4 // op + prefix_len + prefix + limit + offset
	msg := make([]byte, 4+msgLen)
	binary.BigEndian.PutUint32(msg[0:4], uint32(msgLen))
	msg[4] = byte(OpQuery)
	binary.BigEndian.PutUint32(msg[5:9], uint32(len(prefixBytes)))
	copy(msg[9:9+len(prefixBytes)], prefixBytes)
	binary.BigEndian.PutUint32(msg[9+len(prefixBytes):13+len(prefixBytes)], uint32(limit))
	binary.BigEndian.PutUint32(msg[13+len(prefixBytes):17+len(prefixBytes)], uint32(offset))

	if _, err := c.conn.Write(msg); err != nil {
		return nil, err
	}

	// Read response
	return c.readQueryResponse()
}

// BeginTransaction starts a new transaction.
func (c *IPCClient) BeginTransaction() (uint64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return 0, ErrClosed
	}

	msg := []byte{0, 0, 0, 1, byte(OpBeginTxn)}
	if _, err := c.conn.Write(msg); err != nil {
		return 0, err
	}

	// Read response with transaction ID
	header := make([]byte, 4)
	if _, err := io.ReadFull(c.conn, header); err != nil {
		return 0, err
	}

	respLen := binary.BigEndian.Uint32(header)
	resp := make([]byte, respLen)
	if _, err := io.ReadFull(c.conn, resp); err != nil {
		return 0, err
	}

	if resp[0] != byte(OpOK) {
		return 0, c.parseError(resp)
	}

	if len(resp) < 9 {
		return 0, &ProtocolError{Message: "invalid transaction response"}
	}

	txnID := binary.BigEndian.Uint64(resp[1:9])
	return txnID, nil
}

// CommitTransaction commits a transaction.
func (c *IPCClient) CommitTransaction(txnID uint64) error {
	return c.sendTxnOp(OpCommitTxn, txnID)
}

// AbortTransaction aborts a transaction.
func (c *IPCClient) AbortTransaction(txnID uint64) error {
	return c.sendTxnOp(OpAbortTxn, txnID)
}

// Checkpoint forces a checkpoint.
func (c *IPCClient) Checkpoint() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrClosed
	}

	msg := []byte{0, 0, 0, 1, byte(OpCheckpoint)}
	if _, err := c.conn.Write(msg); err != nil {
		return err
	}

	return c.readSimpleResponse()
}

// Stats retrieves storage statistics.
func (c *IPCClient) Stats() (*StorageStats, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	msg := []byte{0, 0, 0, 1, byte(OpStats)}
	if _, err := c.conn.Write(msg); err != nil {
		return nil, err
	}

	// Read response
	header := make([]byte, 4)
	if _, err := io.ReadFull(c.conn, header); err != nil {
		return nil, err
	}

	respLen := binary.BigEndian.Uint32(header)
	resp := make([]byte, respLen)
	if _, err := io.ReadFull(c.conn, resp); err != nil {
		return nil, err
	}

	if resp[0] != byte(OpOK) {
		return nil, c.parseError(resp)
	}

	if len(resp) < 21 {
		return nil, &ProtocolError{Message: "invalid stats response"}
	}

	return &StorageStats{
		MemtableSizeBytes:  binary.BigEndian.Uint64(resp[1:9]),
		WALSizeBytes:       binary.BigEndian.Uint64(resp[9:17]),
		ActiveTransactions: int(binary.BigEndian.Uint32(resp[17:21])),
	}, nil
}

// Helper methods

func (c *IPCClient) sendKeyOp(op OpCode, key []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	// Message format: [len:4][op:1][key_len:4][key:...]
	msgLen := 1 + 4 + len(key)
	msg := make([]byte, 4+msgLen)
	binary.BigEndian.PutUint32(msg[0:4], uint32(msgLen))
	msg[4] = byte(op)
	binary.BigEndian.PutUint32(msg[5:9], uint32(len(key)))
	copy(msg[9:], key)

	if _, err := c.conn.Write(msg); err != nil {
		return nil, err
	}

	return c.readValueResponse()
}

func (c *IPCClient) sendKeyValueOp(op OpCode, key, value []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	// Message format: [len:4][op:1][key_len:4][key:...][value_len:4][value:...]
	msgLen := 1 + 4 + len(key) + 4 + len(value)
	msg := make([]byte, 4+msgLen)
	binary.BigEndian.PutUint32(msg[0:4], uint32(msgLen))
	msg[4] = byte(op)
	binary.BigEndian.PutUint32(msg[5:9], uint32(len(key)))
	copy(msg[9:9+len(key)], key)
	binary.BigEndian.PutUint32(msg[9+len(key):13+len(key)], uint32(len(value)))
	copy(msg[13+len(key):], value)

	if _, err := c.conn.Write(msg); err != nil {
		return nil, err
	}

	return c.readValueResponse()
}

func (c *IPCClient) sendTxnOp(op OpCode, txnID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrClosed
	}

	// Message format: [len:4][op:1][txn_id:8]
	msg := make([]byte, 13)
	binary.BigEndian.PutUint32(msg[0:4], 9) // 1 + 8
	msg[4] = byte(op)
	binary.BigEndian.PutUint64(msg[5:13], txnID)

	if _, err := c.conn.Write(msg); err != nil {
		return err
	}

	return c.readSimpleResponse()
}

func (c *IPCClient) readValueResponse() ([]byte, error) {
	header := make([]byte, 4)
	if _, err := io.ReadFull(c.conn, header); err != nil {
		return nil, err
	}

	respLen := binary.BigEndian.Uint32(header)
	resp := make([]byte, respLen)
	if _, err := io.ReadFull(c.conn, resp); err != nil {
		return nil, err
	}

	switch OpCode(resp[0]) {
	case OpOK:
		if len(resp) < 5 {
			return nil, nil
		}
		valueLen := binary.BigEndian.Uint32(resp[1:5])
		if len(resp) < int(5+valueLen) {
			return nil, &ProtocolError{Message: "truncated value response"}
		}
		return resp[5 : 5+valueLen], nil
	case OpNotFound:
		return nil, nil
	case OpError:
		return nil, c.parseError(resp)
	default:
		return nil, &ProtocolError{Message: fmt.Sprintf("unexpected opcode: %d", resp[0])}
	}
}

func (c *IPCClient) readSimpleResponse() error {
	header := make([]byte, 4)
	if _, err := io.ReadFull(c.conn, header); err != nil {
		return err
	}

	respLen := binary.BigEndian.Uint32(header)
	resp := make([]byte, respLen)
	if _, err := io.ReadFull(c.conn, resp); err != nil {
		return err
	}

	if resp[0] != byte(OpOK) {
		return c.parseError(resp)
	}

	return nil
}

func (c *IPCClient) readQueryResponse() ([]KeyValue, error) {
	header := make([]byte, 4)
	if _, err := io.ReadFull(c.conn, header); err != nil {
		return nil, err
	}

	respLen := binary.BigEndian.Uint32(header)
	resp := make([]byte, respLen)
	if _, err := io.ReadFull(c.conn, resp); err != nil {
		return nil, err
	}

	if resp[0] != byte(OpOK) {
		return nil, c.parseError(resp)
	}

	if len(resp) < 5 {
		return nil, &ProtocolError{Message: "invalid query response"}
	}

	count := binary.BigEndian.Uint32(resp[1:5])
	results := make([]KeyValue, 0, count)
	offset := 5

	for i := uint32(0); i < count; i++ {
		if offset+4 > len(resp) {
			return nil, &ProtocolError{Message: "truncated query response"}
		}
		keyLen := binary.BigEndian.Uint32(resp[offset : offset+4])
		offset += 4

		if offset+int(keyLen)+4 > len(resp) {
			return nil, &ProtocolError{Message: "truncated query response"}
		}
		key := make([]byte, keyLen)
		copy(key, resp[offset:offset+int(keyLen)])
		offset += int(keyLen)

		valueLen := binary.BigEndian.Uint32(resp[offset : offset+4])
		offset += 4

		if offset+int(valueLen) > len(resp) {
			return nil, &ProtocolError{Message: "truncated query response"}
		}
		value := make([]byte, valueLen)
		copy(value, resp[offset:offset+int(valueLen)])
		offset += int(valueLen)

		results = append(results, KeyValue{Key: key, Value: value})
	}

	return results, nil
}

func (c *IPCClient) parseError(resp []byte) error {
	if len(resp) < 5 {
		return &ProtocolError{Message: "malformed error response"}
	}
	msgLen := binary.BigEndian.Uint32(resp[1:5])
	if len(resp) < int(5+msgLen) {
		return &ProtocolError{Message: "truncated error message"}
	}
	msg := string(resp[5 : 5+msgLen])
	return &ToonDBError{Op: "remote", Message: msg}
}

// KeyValue represents a key-value pair returned from queries.
type KeyValue struct {
	Key   []byte
	Value []byte
}

// StorageStats contains database storage statistics.
type StorageStats struct {
	MemtableSizeBytes  uint64
	WALSizeBytes       uint64
	ActiveTransactions int
}
