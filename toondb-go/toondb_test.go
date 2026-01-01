// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestErrors(t *testing.T) {
	t.Run("ToonDBError", func(t *testing.T) {
		err := &ToonDBError{
			Op:      "get",
			Path:    "users/alice",
			Message: "not found",
		}
		assert.Contains(t, err.Error(), "get")
		assert.Contains(t, err.Error(), "users/alice")
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("ToonDBError without path", func(t *testing.T) {
		err := &ToonDBError{
			Op:      "connect",
			Message: "connection refused",
		}
		assert.Contains(t, err.Error(), "connect")
		assert.Contains(t, err.Error(), "connection refused")
		assert.NotContains(t, err.Error(), "  ") // no double space
	})

	t.Run("ConnectionError", func(t *testing.T) {
		err := &ConnectionError{
			Address: "/tmp/toondb.sock",
			Err:     errors.New("connection refused"),
		}
		assert.Contains(t, err.Error(), "/tmp/toondb.sock")
		assert.Contains(t, err.Error(), "connection refused")
	})

	t.Run("TransactionError", func(t *testing.T) {
		err := &TransactionError{
			TxnID:   12345,
			Op:      "commit",
			Message: "conflict detected",
		}
		assert.Contains(t, err.Error(), "12345")
		assert.Contains(t, err.Error(), "commit")
		assert.Contains(t, err.Error(), "conflict")
	})

	t.Run("ProtocolError", func(t *testing.T) {
		err := &ProtocolError{
			Expected: "OK",
			Got:      "ERROR",
		}
		assert.Contains(t, err.Error(), "OK")
		assert.Contains(t, err.Error(), "ERROR")

		err2 := &ProtocolError{
			Message: "malformed response",
		}
		assert.Contains(t, err2.Error(), "malformed")
	})

	t.Run("Unwrap", func(t *testing.T) {
		inner := errors.New("inner error")
		err := &ToonDBError{
			Op:      "put",
			Message: "failed",
			Err:     inner,
		}
		assert.Equal(t, inner, err.Unwrap())
		assert.True(t, errors.Is(err, inner))
	})
}

func TestConfig(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		cfg := DefaultConfig("./testdb")
		assert.Equal(t, "./testdb", cfg.Path)
		assert.True(t, cfg.CreateIfMissing)
		assert.True(t, cfg.WALEnabled)
		assert.Equal(t, "normal", cfg.SyncMode)
		assert.Equal(t, int64(64*1024*1024), cfg.MemtableSizeBytes)
	})
}

func TestQuery(t *testing.T) {
	t.Run("QueryBuilder", func(t *testing.T) {
		q := NewQuery(nil, "users/")
		assert.NotNil(t, q)

		q.Limit(50).Offset(10).Select("name", "email")
		assert.Equal(t, 50, q.limitVal)
		assert.Equal(t, 10, q.offsetVal)
		assert.Equal(t, []string{"name", "email"}, q.selectKeys)
	})

	t.Run("QueryChaining", func(t *testing.T) {
		q := NewQuery(nil, "products/")
		result := q.Limit(100).Offset(20)
		assert.Equal(t, q, result) // Ensure fluent API returns same query
	})
}

func TestVector(t *testing.T) {
	t.Run("DefaultVectorIndexConfig", func(t *testing.T) {
		cfg := DefaultVectorIndexConfig(384)
		assert.Equal(t, 384, cfg.Dimension)
		assert.Equal(t, Cosine, cfg.Metric)
		assert.Equal(t, 16, cfg.M)
		assert.Equal(t, 100, cfg.EfConstruction)
		assert.Equal(t, 50, cfg.EfSearch)
	})

	t.Run("NewVectorIndex", func(t *testing.T) {
		idx := NewVectorIndex("./vectors", nil)
		assert.NotNil(t, idx)
		assert.Equal(t, "./vectors", idx.path)
	})

	t.Run("ComputeCosineDistance", func(t *testing.T) {
		a := []float32{1, 0, 0}
		b := []float32{1, 0, 0}
		dist := ComputeCosineDistance(a, b)
		assert.InDelta(t, 0.0, dist, 0.0001) // identical vectors = 0 distance

		c := []float32{-1, 0, 0}
		dist = ComputeCosineDistance(a, c)
		assert.InDelta(t, 2.0, dist, 0.0001) // opposite vectors = 2 distance
	})

	t.Run("ComputeEuclideanDistance", func(t *testing.T) {
		a := []float32{0, 0, 0}
		b := []float32{3, 4, 0}
		dist := ComputeEuclideanDistance(a, b)
		assert.InDelta(t, 5.0, dist, 0.0001) // 3-4-5 triangle
	})

	t.Run("NormalizeVector", func(t *testing.T) {
		v := []float32{3, 4}
		normalized := NormalizeVector(v)
		assert.InDelta(t, 0.6, normalized[0], 0.0001)
		assert.InDelta(t, 0.8, normalized[1], 0.0001)

		// Check unit length
		var normSq float32
		for _, x := range normalized {
			normSq += x * x
		}
		assert.InDelta(t, 1.0, normSq, 0.0001)
	})

	t.Run("DimensionMismatch", func(t *testing.T) {
		a := []float32{1, 2, 3}
		b := []float32{1, 2}
		dist := ComputeCosineDistance(a, b)
		require.Greater(t, dist, float32(1e30))
	})
}

func TestKeyValue(t *testing.T) {
	t.Run("KeyValueStruct", func(t *testing.T) {
		kv := KeyValue{
			Key:   []byte("test/key"),
			Value: []byte(`{"data": "value"}`),
		}
		assert.Equal(t, "test/key", string(kv.Key))
		assert.Contains(t, string(kv.Value), "data")
	})
}

func TestStorageStats(t *testing.T) {
	t.Run("StorageStatsStruct", func(t *testing.T) {
		stats := StorageStats{
			MemtableSizeBytes:  1024 * 1024,
			WALSizeBytes:       512 * 1024,
			ActiveTransactions: 3,
		}
		assert.Equal(t, uint64(1024*1024), stats.MemtableSizeBytes)
		assert.Equal(t, uint64(512*1024), stats.WALSizeBytes)
		assert.Equal(t, 3, stats.ActiveTransactions)
	})
}
