// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

// Query is a fluent query builder for prefix scans.
//
// Example:
//
//	results, err := db.Query("users/").
//	    Limit(10).
//	    Offset(0).
//	    Select("name", "email").
//	    Execute()
type Query struct {
	client     *IPCClient
	prefix     string
	limitVal   int
	offsetVal  int
	selectKeys []string
}

// NewQuery creates a new Query builder.
func NewQuery(client *IPCClient, prefix string) *Query {
	return &Query{
		client:    client,
		prefix:    prefix,
		limitVal:  100, // default limit
		offsetVal: 0,
	}
}

// Limit sets the maximum number of results to return.
func (q *Query) Limit(n int) *Query {
	q.limitVal = n
	return q
}

// Offset sets the number of results to skip.
func (q *Query) Offset(n int) *Query {
	q.offsetVal = n
	return q
}

// Select specifies which fields to return (for document queries).
func (q *Query) Select(keys ...string) *Query {
	q.selectKeys = keys
	return q
}

// Execute runs the query and returns results.
func (q *Query) Execute() ([]KeyValue, error) {
	return q.client.Query(q.prefix, q.limitVal, q.offsetVal)
}

// ToList is an alias for Execute.
func (q *Query) ToList() ([]KeyValue, error) {
	return q.Execute()
}

// First returns the first result or nil.
func (q *Query) First() (*KeyValue, error) {
	q.limitVal = 1
	results, err := q.Execute()
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return &results[0], nil
}

// Count returns the count of matching items.
//
// Note: This fetches all results to count them.
// For large datasets, use a limit or implement server-side counting.
func (q *Query) Count() (int, error) {
	oldLimit := q.limitVal
	q.limitVal = 10000 // reasonable max for counting
	results, err := q.Execute()
	q.limitVal = oldLimit
	if err != nil {
		return 0, err
	}
	return len(results), nil
}

// Exists checks if any results match.
func (q *Query) Exists() (bool, error) {
	result, err := q.First()
	if err != nil {
		return false, err
	}
	return result != nil, nil
}

// ForEach iterates over all results.
func (q *Query) ForEach(fn func(kv KeyValue) error) error {
	results, err := q.Execute()
	if err != nil {
		return err
	}
	for _, kv := range results {
		if err := fn(kv); err != nil {
			return err
		}
	}
	return nil
}

// Keys returns just the keys from the query.
func (q *Query) Keys() ([][]byte, error) {
	results, err := q.Execute()
	if err != nil {
		return nil, err
	}
	keys := make([][]byte, len(results))
	for i, kv := range results {
		keys[i] = kv.Key
	}
	return keys, nil
}

// Values returns just the values from the query.
func (q *Query) Values() ([][]byte, error) {
	results, err := q.Execute()
	if err != nil {
		return nil, err
	}
	values := make([][]byte, len(results))
	for i, kv := range results {
		values[i] = kv.Value
	}
	return values, nil
}
