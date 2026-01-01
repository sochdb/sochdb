// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! JSON-RPC 2.0 types for MCP
//!
//! Minimal implementation - just enough to handle MCP protocol.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 Request
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct RpcRequest {
    /// Must be "2.0"
    pub jsonrpc: String,

    /// Request ID (can be number, string, or null/omitted for notifications)
    #[serde(default)]
    pub id: Value,

    /// Method name
    pub method: String,

    /// Optional parameters
    #[serde(default)]
    pub params: Value,
}

#[allow(dead_code)]
impl RpcRequest {
    /// Check if this is a notification (no response expected)
    /// A notification either has no id field or id is null
    pub fn is_notification(&self) -> bool {
        self.id.is_null()
    }
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Serialize)]
pub struct RpcResponse {
    /// Always "2.0"
    pub jsonrpc: &'static str,

    /// Request ID (echo from request)
    pub id: Value,

    /// Result (if success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,

    /// Error (if failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl RpcResponse {
    /// Create a success response
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: Value, code: i32, message: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data,
            }),
        }
    }

    /// Parse error (invalid JSON)
    pub fn parse_error() -> Self {
        Self::error(Value::Null, -32700, "Parse error", None)
    }

    /// Invalid request error
    #[allow(dead_code)]
    pub fn invalid_request(id: Value) -> Self {
        Self::error(id, -32600, "Invalid request", None)
    }

    /// Method not found error
    pub fn method_not_found(id: Value, method: &str) -> Self {
        Self::error(id, -32601, format!("Method not found: {}", method), None)
    }

    /// Invalid params error
    pub fn invalid_params(id: Value, message: impl Into<String>) -> Self {
        Self::error(id, -32602, message, None)
    }

    /// Internal error
    #[allow(dead_code)]
    pub fn internal_error(id: Value, message: impl Into<String>) -> Self {
        Self::error(id, -32603, message, None)
    }

    /// ToonDB-specific error
    pub fn toondb_error(id: Value, message: impl Into<String>) -> Self {
        Self::error(id, -32001, message, None)
    }
}

/// JSON-RPC 2.0 Error
#[derive(Debug, Serialize)]
pub struct RpcError {
    /// Error code
    pub code: i32,

    /// Error message
    pub message: String,

    /// Optional additional data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}
