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

//! ToonDB MCP Server - Minimal JSON-RPC/MCP adapter
//!
//! A thin layer that exposes ToonDB's AI-native features via MCP protocol.
//!
//! ## Architecture
//!
//! ```text
//! MCP Client (Claude, Cursor, etc.)
//!      │
//!      │ JSON-RPC over stdio
//!      ▼
//! ┌─────────────────────────────────┐
//! │  toondb-mcp (this crate)        │
//! │  - Stdio framing                │
//! │  - JSON-RPC dispatch            │
//! │  - MCP methods                  │
//! └─────────────────────────────────┘
//!      │
//!      │ Direct Rust calls
//!      ▼
//! ┌─────────────────────────────────┐
//! │  ToonDB                         │
//! │  - Catalog operations → tools   │
//! │  - Context queries → AI context │
//! │  - TOON format → token savings  │
//! └─────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Run as MCP server (stdio transport)
//! toondb-mcp --db ./data
//!
//! # In Claude Desktop config:
//! # "mcpServers": { "toondb": { "command": "toondb-mcp", "args": ["--db", "./data"] } }
//! ```

mod framing;
mod jsonrpc;
mod mcp;
mod tools;

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use tracing::{Level, error, info};
use tracing_subscriber::FmtSubscriber;

use toondb::connection::EmbeddedConnection;

use crate::framing::{read_message, write_message_format, WireFormat};
use crate::jsonrpc::{RpcRequest, RpcResponse};
use crate::mcp::McpServer;

fn main() {
    // Initialize tracing to stderr (stdout is for protocol only!)
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_writer(io::stderr)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber");

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let db_path = args
        .iter()
        .position(|a| a == "--db")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./toondb_data"));

    info!("ToonDB MCP server starting, db_path={:?}", db_path);

    // Open EmbeddedConnection for real on-disk persistence
    let conn = match EmbeddedConnection::open(&db_path) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            error!("Failed to open ToonDB: {}", e);
            std::process::exit(1);
        }
    };

    // Create MCP server
    let server = McpServer::new(conn);

    // Main loop: read JSON-RPC messages from stdin, dispatch, write responses to stdout
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut reader = stdin.lock();

    // Track wire format (detect from first message)
    #[allow(unused_assignments)]
    let mut wire_format = WireFormat::ContentLength;

    loop {
        // Read framed message
        let (msg, _format) = match read_message(&mut reader) {
            Ok(Some((m, f))) => {
                // Update wire format based on client's format
                wire_format = f;
                (m, f)
            }
            Ok(None) => {
                info!("EOF on stdin, shutting down");
                break;
            }
            Err(e) => {
                error!("Failed to read message: {}", e);
                continue;
            }
        };

        // Parse as JSON-RPC request
        let req: RpcRequest = match serde_json::from_slice(&msg) {
            Ok(r) => r,
            Err(e) => {
                error!("Invalid JSON-RPC: {}", e);
                let resp = RpcResponse::parse_error();
                if let Err(e) = write_message_format(&mut stdout, &resp, wire_format) {
                    error!("Failed to write error response: {}", e);
                }
                continue;
            }
        };

        // Check if this is a notification (no response expected)
        let is_notification = req.is_notification();

        // Dispatch and get response
        let resp = server.dispatch(&req);

        // Only send response if this is NOT a notification
        // JSON-RPC 2.0: Notifications MUST NOT have responses
        if !is_notification {
            // Write response in same format as request
            if let Err(e) = write_message_format(&mut stdout, &resp, wire_format) {
                error!("Failed to write response: {}", e);
            }

            // Flush immediately for responsive communication
            if let Err(e) = stdout.flush() {
                error!("Failed to flush stdout: {}", e);
            }
        }
    }

    info!("ToonDB MCP server shutting down");
}
