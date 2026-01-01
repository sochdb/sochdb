//! ToonDB MCP Library
//! 
//! This module exposes the MCP server for embedding in other applications
//! like ToonDB Studio.

pub mod framing;
pub mod jsonrpc;
pub mod mcp;
pub mod tools;

// Re-export key types for embedding
pub use mcp::McpServer;
pub use jsonrpc::{RpcRequest, RpcResponse};
