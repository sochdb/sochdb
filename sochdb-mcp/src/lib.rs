//! SochDB MCP Library
//!
//! This module exposes the MCP server for embedding in other applications
//! like SochDB Studio.

pub mod agentic_search;
pub mod framing;
pub mod jsonrpc;
pub mod mcp;
pub mod tools;

// Re-export key types for embedding
pub use jsonrpc::{RpcRequest, RpcResponse};
pub use mcp::McpServer;
