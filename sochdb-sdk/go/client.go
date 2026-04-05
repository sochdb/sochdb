// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB Go SDK — Thin gRPC client wrapper

package sochdb

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// Client is the main SochDB client.
type Client struct {
	conn     *grpc.ClientConn
	address  string
	apiKey   string
	metadata metadata.MD
}

// Option configures the client.
type Option func(*Client)

// WithAPIKey sets the API key for authentication.
func WithAPIKey(key string) Option {
	return func(c *Client) {
		c.apiKey = key
	}
}

// New creates a new SochDB client.
func New(address string, opts ...Option) (*Client, error) {
	c := &Client{address: address}
	for _, opt := range opts {
		opt(c)
	}

	conn, err := grpc.NewClient(
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("sochdb: connect to %s: %w", address, err)
	}
	c.conn = conn

	if c.apiKey != "" {
		c.metadata = metadata.Pairs("x-api-key", c.apiKey)
	}

	return c, nil
}

// Close closes the client connection.
func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// ctx adds auth metadata to a context.
func (c *Client) ctx(parent context.Context) context.Context {
	if c.metadata != nil {
		return metadata.NewOutgoingContext(parent, c.metadata)
	}
	return parent
}

// ctxTimeout creates a context with timeout and auth metadata.
func (c *Client) ctxTimeout(timeout time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	return c.ctx(ctx), cancel
}

// Conn returns the underlying gRPC connection for direct stub access.
func (c *Client) Conn() *grpc.ClientConn {
	return c.conn
}
