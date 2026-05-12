// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # PostgreSQL Wire Protocol — Simple Query (Task 5)
//!
//! Implements the PostgreSQL simple query protocol (v3), allowing standard
//! PostgreSQL clients (`psql`, ORMs, drivers) to connect and execute SQL.
//!
//! ## Scope (v1)
//!
//! - Simple Query Protocol only (no Extended Query / prepared statements)
//! - No SSL/TLS (cleartext only for v1)
//! - Trust authentication (no password required)
//! - Type mapping: Int→INT8, Float→FLOAT8, Text→TEXT, Bool→BOOL, Binary→BYTEA
//!
//! ## Usage
//!
//! ```bash
//! # Connect with psql
//! psql -h 127.0.0.1 -p 5433 -d sochdb
//! ```
//!
//! ## Protocol Reference
//!
//! See <https://www.postgresql.org/docs/current/protocol-message-formats.html>

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// ============================================================================
// PG Wire Protocol Constants
// ============================================================================

/// Protocol version 3.0 (major=3, minor=0 → 196608)
const PROTOCOL_VERSION_3: i32 = 196608;

/// SSL request magic number
const SSL_REQUEST: i32 = 80877103;

/// Cancel request magic number
const CANCEL_REQUEST: i32 = 80877102;

// Message type identifiers (frontend → backend)
const MSG_QUERY: u8 = b'Q';
const MSG_TERMINATE: u8 = b'X';

// Message type identifiers (backend → frontend)
const MSG_AUTH: u8 = b'R';
const MSG_PARAMETER_STATUS: u8 = b'S';
const MSG_BACKEND_KEY_DATA: u8 = b'K';
const MSG_READY_FOR_QUERY: u8 = b'Z';
const MSG_ROW_DESCRIPTION: u8 = b'T';
const MSG_DATA_ROW: u8 = b'D';
const MSG_COMMAND_COMPLETE: u8 = b'C';
const MSG_ERROR_RESPONSE: u8 = b'E';
const MSG_NOTICE_RESPONSE: u8 = b'N';
const MSG_EMPTY_QUERY_RESPONSE: u8 = b'I';

// PostgreSQL type OIDs
const OID_BOOL: i32 = 16;
const OID_INT8: i32 = 20;
const OID_FLOAT8: i32 = 701;
const OID_TEXT: i32 = 25;
const OID_BYTEA: i32 = 17;
const OID_JSONB: i32 = 3802;
const OID_UNKNOWN: i32 = 705;

// Transaction status indicators
const TX_IDLE: u8 = b'I';
const TX_IN_BLOCK: u8 = b'T';
const TX_FAILED: u8 = b'E';

// ============================================================================
// PG Type Mapping
// ============================================================================

/// Map a column type string to a PostgreSQL OID.
pub fn type_to_oid(type_name: &str) -> i32 {
    match type_name.to_uppercase().as_str() {
        "INT" | "INTEGER" | "BIGINT" | "INT8" | "SERIAL" | "BIGSERIAL" => OID_INT8,
        "FLOAT" | "DOUBLE" | "FLOAT8" | "REAL" | "FLOAT4" | "NUMERIC" | "DECIMAL" => OID_FLOAT8,
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" | "CHARACTER VARYING" => OID_TEXT,
        "BOOL" | "BOOLEAN" => OID_BOOL,
        "BYTEA" | "BINARY" | "BLOB" => OID_BYTEA,
        "JSON" | "JSONB" | "MAP" | "OBJECT" | "ARRAY" => OID_JSONB,
        _ => OID_TEXT, // Default to TEXT
    }
}

/// Format size for a PG type (negative = variable length).
pub fn type_size(oid: i32) -> i16 {
    match oid {
        OID_BOOL => 1,
        OID_INT8 => 8,
        OID_FLOAT8 => 8,
        _ => -1, // variable-length
    }
}

// ============================================================================
// SQL Executor Trait (abstracts over SqlBridge)
// ============================================================================

/// Result from executing a SQL statement via the PG wire layer.
#[derive(Debug, Clone)]
pub enum PgResult {
    /// Rows returned (SELECT-like)
    Rows {
        columns: Vec<PgColumn>,
        rows: Vec<Vec<Option<String>>>,
    },
    /// DML result (INSERT/UPDATE/DELETE)
    RowsAffected { tag: String, count: u64 },
    /// DDL result (CREATE/DROP/ALTER)
    Command { tag: String },
}

/// Column metadata for PG wire RowDescription.
#[derive(Debug, Clone)]
pub struct PgColumn {
    pub name: String,
    pub type_oid: i32,
    pub type_size: i16,
    pub type_modifier: i32,
}

impl PgColumn {
    pub fn new(name: &str, type_name: &str) -> Self {
        let oid = type_to_oid(type_name);
        Self {
            name: name.to_string(),
            type_oid: oid,
            type_size: type_size(oid),
            type_modifier: -1,
        }
    }

    pub fn text(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_oid: OID_TEXT,
            type_size: -1,
            type_modifier: -1,
        }
    }
}

/// Trait for SQL execution backends.
///
/// Implement this to wire the PG protocol to a real SQL engine.
pub trait PgSqlExecutor: Send + Sync + 'static {
    fn execute(&self, sql: &str) -> Result<PgResult, String>;
}

/// A simple executor that echoes queries (for testing / placeholder).
#[derive(Clone)]
pub struct EchoPgExecutor;

impl PgSqlExecutor for EchoPgExecutor {
    fn execute(&self, sql: &str) -> Result<PgResult, String> {
        let trimmed = sql.trim();
        if trimmed.is_empty() {
            return Ok(PgResult::Command {
                tag: "EMPTY".to_string(),
            });
        }

        // Handle SET/SHOW/RESET commands that psql sends
        let upper = trimmed.to_uppercase();
        if upper.starts_with("SET ") || upper.starts_with("RESET ") {
            return Ok(PgResult::Command {
                tag: "SET".to_string(),
            });
        }
        if upper.starts_with("SHOW ") {
            let param = trimmed[5..].trim().trim_end_matches(';');
            return Ok(PgResult::Rows {
                columns: vec![PgColumn::text(param)],
                rows: vec![vec![Some("on".to_string())]],
            });
        }

        // Default: return query as a single-column result
        Ok(PgResult::Rows {
            columns: vec![PgColumn::text("result")],
            rows: vec![vec![Some(format!("Executed: {}", trimmed))]],
        })
    }
}

// ============================================================================
// PG Wire Server
// ============================================================================

/// PostgreSQL wire protocol server configuration.
pub struct PgWireConfig {
    pub addr: SocketAddr,
    pub server_version: String,
}

/// Start the PG wire protocol server.
pub fn start<E: PgSqlExecutor + Clone>(
    config: PgWireConfig,
    executor: E,
) -> tokio::task::JoinHandle<()> {
    let executor = Arc::new(executor);
    tokio::spawn(async move {
        let listener = match TcpListener::bind(&config.addr).await {
            Ok(l) => {
                tracing::info!("PG wire server listening on {}", config.addr);
                l
            }
            Err(e) => {
                tracing::error!("Failed to bind PG wire server to {}: {}", config.addr, e);
                return;
            }
        };

        let server_version = Arc::new(config.server_version);
        let mut conn_id: u32 = 0;

        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    conn_id = conn_id.wrapping_add(1);
                    tracing::debug!("PG wire connection from {} (id={})", peer, conn_id);
                    let exec = executor.clone();
                    let ver = server_version.clone();
                    let cid = conn_id;
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, exec, &ver, cid).await {
                            tracing::debug!("PG wire connection {} error: {}", cid, e);
                        }
                    });
                }
                Err(e) => {
                    tracing::error!("PG wire accept error: {}", e);
                }
            }
        }
    })
}

/// Handle a single PostgreSQL wire protocol connection.
async fn handle_connection<E: PgSqlExecutor>(
    mut stream: TcpStream,
    executor: Arc<E>,
    server_version: &str,
    conn_id: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Phase 1: Startup
    let _startup = read_startup(&mut stream).await?;

    // Send AuthenticationOk
    send_auth_ok(&mut stream).await?;

    // Send parameter status messages (psql expects these)
    send_parameter_status(&mut stream, "server_version", server_version).await?;
    send_parameter_status(&mut stream, "server_encoding", "UTF8").await?;
    send_parameter_status(&mut stream, "client_encoding", "UTF8").await?;
    send_parameter_status(&mut stream, "DateStyle", "ISO, MDY").await?;
    send_parameter_status(&mut stream, "integer_datetimes", "on").await?;
    send_parameter_status(&mut stream, "standard_conforming_strings", "on").await?;
    send_parameter_status(&mut stream, "application_name", "").await?;

    // Send BackendKeyData
    send_backend_key_data(&mut stream, conn_id as i32, 0).await?;

    // Send ReadyForQuery (idle)
    send_ready_for_query(&mut stream, TX_IDLE).await?;

    // Phase 2: Query loop
    let mut tx_status = TX_IDLE;
    loop {
        // Read message type (1 byte) + length (4 bytes)
        let msg_type = match stream.read_u8().await {
            Ok(t) => t,
            Err(_) => break, // Client disconnected
        };
        let msg_len = stream.read_i32().await? as usize;
        if msg_len < 4 {
            break;
        }
        let payload_len = msg_len - 4;

        match msg_type {
            MSG_QUERY => {
                let mut payload = vec![0u8; payload_len];
                stream.read_exact(&mut payload).await?;
                // Remove trailing null byte
                if payload.last() == Some(&0) {
                    payload.pop();
                }
                let sql = String::from_utf8_lossy(&payload).to_string();
                tracing::debug!("PG query (conn={}): {}", conn_id, sql);

                if sql.trim().is_empty() {
                    stream.write_u8(MSG_EMPTY_QUERY_RESPONSE).await?;
                    write_i32(&mut stream, 4).await?;
                    send_ready_for_query(&mut stream, tx_status).await?;
                    continue;
                }

                // Track BEGIN/COMMIT/ROLLBACK for transaction status
                let upper = sql.trim().to_uppercase();
                if upper.starts_with("BEGIN") {
                    tx_status = TX_IN_BLOCK;
                } else if upper.starts_with("COMMIT") || upper.starts_with("END") {
                    tx_status = TX_IDLE;
                } else if upper.starts_with("ROLLBACK") {
                    tx_status = TX_IDLE;
                }

                // Execute query
                match executor.execute(&sql) {
                    Ok(PgResult::Rows { columns, rows }) => {
                        send_row_description(&mut stream, &columns).await?;
                        for row in &rows {
                            send_data_row(&mut stream, row).await?;
                        }
                        let tag = format!("SELECT {}", rows.len());
                        send_command_complete(&mut stream, &tag).await?;
                    }
                    Ok(PgResult::RowsAffected { tag, count }) => {
                        let full_tag = format!("{} {}", tag, count);
                        send_command_complete(&mut stream, &full_tag).await?;
                    }
                    Ok(PgResult::Command { tag }) => {
                        send_command_complete(&mut stream, &tag).await?;
                    }
                    Err(e) => {
                        send_error(&mut stream, "ERROR", "42000", &e).await?;
                        if tx_status == TX_IN_BLOCK {
                            tx_status = TX_FAILED;
                        }
                    }
                }

                send_ready_for_query(&mut stream, tx_status).await?;
            }

            MSG_TERMINATE => {
                tracing::debug!("PG wire client {} sent Terminate", conn_id);
                break;
            }

            other => {
                // Read and discard unknown message payload
                let mut discard = vec![0u8; payload_len];
                stream.read_exact(&mut discard).await?;
                tracing::debug!(
                    "PG wire: ignoring unknown message type {} from conn {}",
                    other as char,
                    conn_id
                );
                send_error(
                    &mut stream,
                    "ERROR",
                    "0A000",
                    &format!("Unsupported message type: {}", other as char),
                )
                .await?;
                send_ready_for_query(&mut stream, tx_status).await?;
            }
        }
    }

    Ok(())
}

// ============================================================================
// PG Wire Message Reading
// ============================================================================

/// Startup parameters from the client.
#[derive(Debug)]
pub struct StartupMessage {
    pub params: HashMap<String, String>,
}

/// Read the startup message from the client.
async fn read_startup(
    stream: &mut TcpStream,
) -> Result<StartupMessage, Box<dyn std::error::Error + Send + Sync>> {
    let msg_len = stream.read_i32().await? as usize;
    if msg_len < 8 || msg_len > 10240 {
        return Err("Invalid startup message length".into());
    }

    let version = stream.read_i32().await?;

    // Handle SSL request
    if version == SSL_REQUEST {
        // Send 'N' (no SSL) and read the real startup
        stream.write_u8(b'N').await?;
        stream.flush().await?;
        // Recursively read the actual startup message
        return Box::pin(read_startup(stream)).await;
    }

    // Handle cancel request
    if version == CANCEL_REQUEST {
        // Read and ignore cancel data (pid + secret)
        let _pid = stream.read_i32().await?;
        let _secret = stream.read_i32().await?;
        return Err("Cancel request not supported".into());
    }

    if version != PROTOCOL_VERSION_3 {
        return Err(format!("Unsupported protocol version: {}", version).into());
    }

    // Read key-value parameters
    let payload_len = msg_len - 8; // Already read 4 (len) + 4 (version)
    let mut payload = vec![0u8; payload_len];
    stream.read_exact(&mut payload).await?;

    let mut params = HashMap::new();
    let parts: Vec<&[u8]> = payload.split(|b| *b == 0).collect();
    let mut i = 0;
    while i + 1 < parts.len() {
        let key = String::from_utf8_lossy(parts[i]).to_string();
        let value = String::from_utf8_lossy(parts[i + 1]).to_string();
        if key.is_empty() {
            break;
        }
        params.insert(key, value);
        i += 2;
    }

    Ok(StartupMessage { params })
}

// ============================================================================
// PG Wire Message Writing
// ============================================================================

async fn write_i32(
    stream: &mut TcpStream,
    val: i32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    stream.write_all(&val.to_be_bytes()).await?;
    Ok(())
}

async fn write_i16(
    stream: &mut TcpStream,
    val: i16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    stream.write_all(&val.to_be_bytes()).await?;
    Ok(())
}

/// Write a null-terminated string.
async fn write_cstring(
    stream: &mut TcpStream,
    s: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    stream.write_all(s.as_bytes()).await?;
    stream.write_u8(0).await?;
    Ok(())
}

/// Send AuthenticationOk (R message, auth type 0).
async fn send_auth_ok(
    stream: &mut TcpStream,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    stream.write_u8(MSG_AUTH).await?;
    write_i32(stream, 8).await?; // length: 4 (len) + 4 (auth type)
    write_i32(stream, 0).await?; // auth type: 0 = ok
    stream.flush().await?;
    Ok(())
}

/// Send a ParameterStatus message.
async fn send_parameter_status(
    stream: &mut TcpStream,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let len = 4 + key.len() + 1 + value.len() + 1; // len + key\0 + value\0
    stream.write_u8(MSG_PARAMETER_STATUS).await?;
    write_i32(stream, len as i32).await?;
    write_cstring(stream, key).await?;
    write_cstring(stream, value).await?;
    Ok(())
}

/// Send BackendKeyData.
async fn send_backend_key_data(
    stream: &mut TcpStream,
    pid: i32,
    secret_key: i32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    stream.write_u8(MSG_BACKEND_KEY_DATA).await?;
    write_i32(stream, 12).await?; // length: 4 + 4 (pid) + 4 (key)
    write_i32(stream, pid).await?;
    write_i32(stream, secret_key).await?;
    Ok(())
}

/// Send ReadyForQuery with transaction status.
async fn send_ready_for_query(
    stream: &mut TcpStream,
    status: u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    stream.write_u8(MSG_READY_FOR_QUERY).await?;
    write_i32(stream, 5).await?; // length: 4 + 1 (status)
    stream.write_u8(status).await?;
    stream.flush().await?;
    Ok(())
}

/// Send RowDescription message.
async fn send_row_description(
    stream: &mut TcpStream,
    columns: &[PgColumn],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Calculate total length:
    // 4 (len) + 2 (num fields)
    // For each field: name\0 + 4 (table_oid) + 2 (col_num)
    //   + 4 (type_oid) + 2 (type_size) + 4 (type_modifier) + 2 (format_code)
    let mut body_len: usize = 2; // num fields
    for col in columns {
        body_len += col.name.len() + 1 + 4 + 2 + 4 + 2 + 4 + 2; // 18 bytes per field + name\0
    }

    stream.write_u8(MSG_ROW_DESCRIPTION).await?;
    write_i32(stream, (4 + body_len) as i32).await?;
    write_i16(stream, columns.len() as i16).await?;

    for col in columns {
        write_cstring(stream, &col.name).await?;
        write_i32(stream, 0).await?; // table OID (0 = calculated)
        write_i16(stream, 0).await?; // column number
        write_i32(stream, col.type_oid).await?; // type OID
        write_i16(stream, col.type_size).await?; // type size
        write_i32(stream, col.type_modifier).await?; // type modifier
        write_i16(stream, 0).await?; // format code (0 = text)
    }

    Ok(())
}

/// Send a DataRow message.
async fn send_data_row(
    stream: &mut TcpStream,
    row: &[Option<String>],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Calculate length: 4 (len) + 2 (num cols) + per-field (4 len + data)
    let mut body_len: usize = 2;
    for val in row {
        body_len += 4; // column length (or -1 for null)
        if let Some(s) = val {
            body_len += s.len();
        }
    }

    stream.write_u8(MSG_DATA_ROW).await?;
    write_i32(stream, (4 + body_len) as i32).await?;
    write_i16(stream, row.len() as i16).await?;

    for val in row {
        match val {
            Some(s) => {
                write_i32(stream, s.len() as i32).await?;
                stream.write_all(s.as_bytes()).await?;
            }
            None => {
                write_i32(stream, -1).await?; // NULL
            }
        }
    }

    Ok(())
}

/// Send CommandComplete message.
async fn send_command_complete(
    stream: &mut TcpStream,
    tag: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let len = 4 + tag.len() + 1; // len + tag\0
    stream.write_u8(MSG_COMMAND_COMPLETE).await?;
    write_i32(stream, len as i32).await?;
    write_cstring(stream, tag).await?;
    Ok(())
}

/// Send ErrorResponse message.
async fn send_error(
    stream: &mut TcpStream,
    severity: &str,
    code: &str,
    message: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Fields: S (severity), V (severity non-localized), C (code), M (message), \0
    let mut body = Vec::new();
    body.push(b'S');
    body.extend_from_slice(severity.as_bytes());
    body.push(0);
    body.push(b'V');
    body.extend_from_slice(severity.as_bytes());
    body.push(0);
    body.push(b'C');
    body.extend_from_slice(code.as_bytes());
    body.push(0);
    body.push(b'M');
    body.extend_from_slice(message.as_bytes());
    body.push(0);
    body.push(0); // terminator

    stream.write_u8(MSG_ERROR_RESPONSE).await?;
    write_i32(stream, (4 + body.len()) as i32).await?;
    stream.write_all(&body).await?;
    stream.flush().await?;
    Ok(())
}

// ============================================================================
// Helper: Build startup message bytes for testing
// ============================================================================

/// Build a PostgreSQL startup message (for test clients).
pub fn build_startup_message(params: &[(&str, &str)]) -> Vec<u8> {
    let mut body = Vec::new();
    // Protocol version 3.0
    body.extend_from_slice(&PROTOCOL_VERSION_3.to_be_bytes());
    for (key, value) in params {
        body.extend_from_slice(key.as_bytes());
        body.push(0);
        body.extend_from_slice(value.as_bytes());
        body.push(0);
    }
    body.push(0); // terminator

    let len = (4 + body.len()) as i32;
    let mut msg = Vec::new();
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&body);
    msg
}

/// Build a PG Query message ('Q').
pub fn build_query_message(sql: &str) -> Vec<u8> {
    let len = (4 + sql.len() + 1) as i32;
    let mut msg = Vec::new();
    msg.push(MSG_QUERY);
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(sql.as_bytes());
    msg.push(0);
    msg
}

/// Build a PG Terminate message ('X').
pub fn build_terminate_message() -> Vec<u8> {
    let mut msg = Vec::new();
    msg.push(MSG_TERMINATE);
    msg.extend_from_slice(&4_i32.to_be_bytes());
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    #[test]
    fn test_type_to_oid() {
        assert_eq!(type_to_oid("INT"), OID_INT8);
        assert_eq!(type_to_oid("integer"), OID_INT8);
        assert_eq!(type_to_oid("TEXT"), OID_TEXT);
        assert_eq!(type_to_oid("varchar"), OID_TEXT);
        assert_eq!(type_to_oid("BOOL"), OID_BOOL);
        assert_eq!(type_to_oid("FLOAT"), OID_FLOAT8);
        assert_eq!(type_to_oid("BYTEA"), OID_BYTEA);
        assert_eq!(type_to_oid("JSONB"), OID_JSONB);
        assert_eq!(type_to_oid("unknown_type"), OID_TEXT); // default
    }

    #[test]
    fn test_type_size() {
        assert_eq!(type_size(OID_BOOL), 1);
        assert_eq!(type_size(OID_INT8), 8);
        assert_eq!(type_size(OID_FLOAT8), 8);
        assert_eq!(type_size(OID_TEXT), -1);
        assert_eq!(type_size(OID_JSONB), -1);
    }

    #[test]
    fn test_pg_column_new() {
        let col = PgColumn::new("age", "INT");
        assert_eq!(col.name, "age");
        assert_eq!(col.type_oid, OID_INT8);
        assert_eq!(col.type_size, 8);
    }

    #[test]
    fn test_pg_column_text() {
        let col = PgColumn::text("name");
        assert_eq!(col.name, "name");
        assert_eq!(col.type_oid, OID_TEXT);
        assert_eq!(col.type_size, -1);
    }

    #[test]
    fn test_echo_executor_sql() {
        let exec = EchoPgExecutor;
        match exec.execute("SELECT 1") {
            Ok(PgResult::Rows { columns, rows }) => {
                assert_eq!(columns.len(), 1);
                assert_eq!(columns[0].name, "result");
                assert_eq!(rows.len(), 1);
                assert!(rows[0][0].as_ref().unwrap().contains("SELECT 1"));
            }
            other => panic!("Expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn test_echo_executor_set() {
        let exec = EchoPgExecutor;
        match exec.execute("SET client_encoding TO 'UTF8'") {
            Ok(PgResult::Command { tag }) => assert_eq!(tag, "SET"),
            other => panic!("Expected Command, got {:?}", other),
        }
    }

    #[test]
    fn test_echo_executor_show() {
        let exec = EchoPgExecutor;
        match exec.execute("SHOW server_version") {
            Ok(PgResult::Rows { columns, rows }) => {
                assert_eq!(columns[0].name, "server_version");
                assert_eq!(rows.len(), 1);
            }
            other => panic!("Expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn test_echo_executor_empty() {
        let exec = EchoPgExecutor;
        match exec.execute("") {
            Ok(PgResult::Command { tag }) => assert_eq!(tag, "EMPTY"),
            other => panic!("Expected Command(EMPTY), got {:?}", other),
        }
    }

    #[test]
    fn test_build_startup_message() {
        let msg = build_startup_message(&[("user", "test"), ("database", "mydb")]);
        assert!(msg.len() > 8);
        // First 4 bytes = length
        let len = i32::from_be_bytes([msg[0], msg[1], msg[2], msg[3]]);
        assert_eq!(len as usize, msg.len());
        // Next 4 bytes = protocol version
        let version = i32::from_be_bytes([msg[4], msg[5], msg[6], msg[7]]);
        assert_eq!(version, PROTOCOL_VERSION_3);
    }

    #[test]
    fn test_build_query_message() {
        let msg = build_query_message("SELECT 1");
        assert_eq!(msg[0], MSG_QUERY);
        let len = i32::from_be_bytes([msg[1], msg[2], msg[3], msg[4]]);
        assert_eq!(len as usize, msg.len() - 1); // length doesn't include type byte
    }

    #[test]
    fn test_build_terminate_message() {
        let msg = build_terminate_message();
        assert_eq!(msg[0], MSG_TERMINATE);
        assert_eq!(msg.len(), 5);
    }

    /// Read all available bytes with a small delay to let the server flush.
    async fn read_all_available(client: &mut TcpStream) -> Vec<u8> {
        let mut all = Vec::new();
        let mut buf = vec![0u8; 8192];
        // Give server time to process and flush
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        loop {
            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                client.read(&mut buf),
            )
            .await
            {
                Ok(Ok(n)) if n > 0 => {
                    all.extend_from_slice(&buf[..n]);
                }
                _ => break,
            }
        }
        all
    }

    #[tokio::test]
    async fn test_pg_wire_full_session() {
        // Start the PG wire server on a random port
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener); // Free the port

        let config = PgWireConfig {
            addr,
            server_version: "SochDB 0.5.0-test".into(),
        };

        let _handle = start(config, EchoPgExecutor);

        // Give the server a moment to bind
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Connect as a raw TCP client
        let mut client = TcpStream::connect(addr).await.unwrap();

        // 1. Send startup message
        let startup = build_startup_message(&[("user", "test"), ("database", "sochdb")]);
        client.write_all(&startup).await.unwrap();
        client.flush().await.unwrap();

        // Read all startup response
        let startup_resp = read_all_available(&mut client).await;
        assert!(!startup_resp.is_empty(), "No startup response received");
        // First byte should be 'R' (AuthenticationOk)
        assert_eq!(startup_resp[0], MSG_AUTH);
        // Last message should be ReadyForQuery: Z + len(5) + 'I'
        let last3 = &startup_resp[startup_resp.len() - 6..];
        assert_eq!(last3[0], MSG_READY_FOR_QUERY);

        // 2. Send a query
        let query = build_query_message("SELECT 1");
        client.write_all(&query).await.unwrap();
        client.flush().await.unwrap();

        // Read query response
        let query_resp = read_all_available(&mut client).await;
        assert!(!query_resp.is_empty(), "No query response received");
        // Should start with 'T' (RowDescription)
        assert_eq!(query_resp[0], MSG_ROW_DESCRIPTION);

        // 3. Send terminate
        let term = build_terminate_message();
        client.write_all(&term).await.unwrap();
        client.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_pg_wire_ssl_negotiation() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let config = PgWireConfig {
            addr,
            server_version: "SochDB-test".into(),
        };

        let _handle = start(config, EchoPgExecutor);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let mut client = TcpStream::connect(addr).await.unwrap();

        // Send SSL request
        let ssl_len: i32 = 8;
        client.write_all(&ssl_len.to_be_bytes()).await.unwrap();
        client
            .write_all(&SSL_REQUEST.to_be_bytes())
            .await
            .unwrap();
        client.flush().await.unwrap();

        // Server should respond with 'N' (no SSL)
        let response = client.read_u8().await.unwrap();
        assert_eq!(response, b'N');

        // Now send real startup
        let startup = build_startup_message(&[("user", "test")]);
        client.write_all(&startup).await.unwrap();
        client.flush().await.unwrap();

        let mut buf = vec![0u8; 4096];
        let n = client.read(&mut buf).await.unwrap();
        assert!(n > 0);
        assert_eq!(buf[0], MSG_AUTH); // AuthenticationOk
    }
}
