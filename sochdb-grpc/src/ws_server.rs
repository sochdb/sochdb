// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # WebSocket Gateway (Task 4 — Transport layer)
//!
//! Provides a JSON-over-WebSocket transport for SochDB, enabling browser
//! and thin clients to interact with the database without gRPC tooling.
//!
//! ## Message Protocol
//!
//! Clients send JSON messages with this shape:
//!
//! ```json
//! { "id": "req-1", "type": "sql", "payload": { "query": "SELECT * FROM users" } }
//! ```
//!
//! Server responds with:
//!
//! ```json
//! { "id": "req-1", "type": "result", "payload": { "columns": [...], "rows": [...] } }
//! ```
//!
//! ## Supported Message Types
//!
//! | Client → Server | Description |
//! |-----------------|-------------|
//! | `sql`           | Execute a SQL query |
//! | `kv_get`        | Get a key-value entry |
//! | `kv_put`        | Put a key-value entry |
//! | `kv_delete`     | Delete a key-value entry |
//! | `subscribe`     | Subscribe to CDC events (streaming) |
//! | `ping`          | Health check |

use std::net::SocketAddr;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

/// WebSocket message from client → server.
#[derive(Debug, Deserialize)]
pub struct WsRequest {
    /// Unique request ID for correlating responses
    pub id: String,
    /// Message type: sql, kv_get, kv_put, kv_delete, subscribe, ping
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Type-specific payload
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// WebSocket message from server → client.
#[derive(Debug, Serialize)]
pub struct WsResponse {
    /// Correlates with request ID
    pub id: String,
    /// Response type: result, error, event, pong
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Type-specific payload
    pub payload: serde_json::Value,
}

/// SQL query payload.
#[derive(Debug, Deserialize)]
pub struct SqlPayload {
    pub query: String,
    #[serde(default)]
    pub params: Vec<serde_json::Value>,
}

/// KV get payload.
#[derive(Debug, Deserialize)]
pub struct KvGetPayload {
    pub key: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
}

/// KV put payload.
#[derive(Debug, Deserialize)]
pub struct KvPutPayload {
    pub key: String,
    pub value: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default)]
    pub ttl_seconds: u64,
}

/// KV delete payload.
#[derive(Debug, Deserialize)]
pub struct KvDeletePayload {
    pub key: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
}

/// Subscribe payload.
#[derive(Debug, Deserialize)]
pub struct SubscribePayload {
    #[serde(default)]
    pub tables: Vec<String>,
    #[serde(default)]
    pub operations: Vec<String>,
    #[serde(default)]
    pub start_sequence: u64,
}

fn default_namespace() -> String {
    "default".to_string()
}

/// KV store shared between WebSocket connections (mirrors gRPC KvServer).
pub type KvStore = Arc<dashmap::DashMap<String, KvEntry>>;

/// A KV entry with value and optional TTL.
#[derive(Debug, Clone)]
pub struct KvEntry {
    pub value: Vec<u8>,
    pub expires_at: Option<u64>,
}

/// WebSocket server configuration.
pub struct WsConfig {
    pub addr: SocketAddr,
    pub kv_store: KvStore,
    pub cdc_log: Option<Arc<sochdb_storage::cdc::CdcLog>>,
}

/// Start the WebSocket server on a background tokio task.
///
/// Returns a `JoinHandle` for the listener task.
pub fn start(config: WsConfig) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let listener = match TcpListener::bind(&config.addr).await {
            Ok(l) => {
                tracing::info!("WebSocket server listening on ws://{}", config.addr);
                l
            }
            Err(e) => {
                tracing::error!("Failed to bind WebSocket server to {}: {}", config.addr, e);
                return;
            }
        };

        let kv_store = config.kv_store;
        let cdc_log = config.cdc_log;

        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    tracing::debug!("WebSocket connection from {}", peer);
                    let kv = kv_store.clone();
                    let cdc = cdc_log.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, kv, cdc).await {
                            tracing::debug!("WebSocket connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    tracing::error!("WebSocket accept error: {}", e);
                }
            }
        }
    })
}

/// Handle a single WebSocket connection.
async fn handle_connection(
    stream: TcpStream,
    kv_store: KvStore,
    cdc_log: Option<Arc<sochdb_storage::cdc::CdcLog>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ws_stream = tokio_tungstenite::accept_async(stream).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Channel for subscription events (pushed asynchronously)
    let (event_tx, mut event_rx) = mpsc::channel::<WsResponse>(256);

    loop {
        tokio::select! {
            // Forward subscription events to client
            Some(event) = event_rx.recv() => {
                let json = serde_json::to_string(&event)?;
                ws_sender.send(Message::Text(json.into())).await?;
            }
            // Handle incoming messages
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        let response = handle_message(
                            &text,
                            &kv_store,
                            &cdc_log,
                            &event_tx,
                        ).await;
                        let json = serde_json::to_string(&response)?;
                        ws_sender.send(Message::Text(json.into())).await?;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        ws_sender.send(Message::Pong(data)).await?;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => {
                        tracing::debug!("WebSocket read error: {}", e);
                        break;
                    }
                    _ => {} // Binary, Pong, Frame — ignore
                }
            }
        }
    }

    Ok(())
}

/// Handle a single JSON message and return a response.
async fn handle_message(
    text: &str,
    kv_store: &KvStore,
    cdc_log: &Option<Arc<sochdb_storage::cdc::CdcLog>>,
    event_tx: &mpsc::Sender<WsResponse>,
) -> WsResponse {
    let req: WsRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => {
            return WsResponse {
                id: String::new(),
                msg_type: "error".into(),
                payload: serde_json::json!({ "message": format!("Invalid JSON: {}", e) }),
            };
        }
    };

    let id = req.id.clone();

    match req.msg_type.as_str() {
        "ping" => WsResponse {
            id,
            msg_type: "pong".into(),
            payload: serde_json::json!({ "ts": now_ms() }),
        },

        "sql" => handle_sql(&req).await,

        "kv_get" => handle_kv_get(&req, kv_store).await,
        "kv_put" => handle_kv_put(&req, kv_store).await,
        "kv_delete" => handle_kv_delete(&req, kv_store).await,

        "subscribe" => handle_subscribe(&req, cdc_log, event_tx).await,

        other => WsResponse {
            id,
            msg_type: "error".into(),
            payload: serde_json::json!({ "message": format!("Unknown message type: {}", other) }),
        },
    }
}

/// Handle SQL query execution.
async fn handle_sql(req: &WsRequest) -> WsResponse {
    let payload: SqlPayload = match serde_json::from_value(req.payload.clone()) {
        Ok(p) => p,
        Err(e) => {
            return WsResponse {
                id: req.id.clone(),
                msg_type: "error".into(),
                payload: serde_json::json!({ "message": format!("Invalid SQL payload: {}", e) }),
            };
        }
    };

    // SQL execution is not wired to a persistent database in the WS layer.
    // This is a placeholder that demonstrates the protocol. In production,
    // this would be wired to a shared Database/SqlBridge instance.
    WsResponse {
        id: req.id.clone(),
        msg_type: "result".into(),
        payload: serde_json::json!({
            "message": "SQL execution via WebSocket",
            "query": payload.query,
            "note": "Wire to SqlBridge for full execution"
        }),
    }
}

/// Handle KV get.
async fn handle_kv_get(req: &WsRequest, kv_store: &KvStore) -> WsResponse {
    let payload: KvGetPayload = match serde_json::from_value(req.payload.clone()) {
        Ok(p) => p,
        Err(e) => {
            return WsResponse {
                id: req.id.clone(),
                msg_type: "error".into(),
                payload: serde_json::json!({ "message": format!("Invalid payload: {}", e) }),
            };
        }
    };

    let full_key = format!("{}:{}", payload.namespace, payload.key);
    match kv_store.get(&full_key) {
        Some(entry) => {
            // Check TTL
            if let Some(exp) = entry.expires_at {
                if now_ms() / 1000 > exp {
                    kv_store.remove(&full_key);
                    return WsResponse {
                        id: req.id.clone(),
                        msg_type: "result".into(),
                        payload: serde_json::json!({ "found": false }),
                    };
                }
            }
            let value_str = String::from_utf8_lossy(&entry.value).to_string();
            WsResponse {
                id: req.id.clone(),
                msg_type: "result".into(),
                payload: serde_json::json!({ "found": true, "value": value_str }),
            }
        }
        None => WsResponse {
            id: req.id.clone(),
            msg_type: "result".into(),
            payload: serde_json::json!({ "found": false }),
        },
    }
}

/// Handle KV put.
async fn handle_kv_put(req: &WsRequest, kv_store: &KvStore) -> WsResponse {
    let payload: KvPutPayload = match serde_json::from_value(req.payload.clone()) {
        Ok(p) => p,
        Err(e) => {
            return WsResponse {
                id: req.id.clone(),
                msg_type: "error".into(),
                payload: serde_json::json!({ "message": format!("Invalid payload: {}", e) }),
            };
        }
    };

    let full_key = format!("{}:{}", payload.namespace, payload.key);
    let expires_at = if payload.ttl_seconds > 0 {
        Some(now_ms() / 1000 + payload.ttl_seconds)
    } else {
        None
    };

    kv_store.insert(
        full_key,
        KvEntry {
            value: payload.value.into_bytes(),
            expires_at,
        },
    );

    WsResponse {
        id: req.id.clone(),
        msg_type: "result".into(),
        payload: serde_json::json!({ "ok": true }),
    }
}

/// Handle KV delete.
async fn handle_kv_delete(req: &WsRequest, kv_store: &KvStore) -> WsResponse {
    let payload: KvDeletePayload = match serde_json::from_value(req.payload.clone()) {
        Ok(p) => p,
        Err(e) => {
            return WsResponse {
                id: req.id.clone(),
                msg_type: "error".into(),
                payload: serde_json::json!({ "message": format!("Invalid payload: {}", e) }),
            };
        }
    };

    let full_key = format!("{}:{}", payload.namespace, payload.key);
    let existed = kv_store.remove(&full_key).is_some();

    WsResponse {
        id: req.id.clone(),
        msg_type: "result".into(),
        payload: serde_json::json!({ "deleted": existed }),
    }
}

/// Handle subscribe request — starts pushing CDC events to the client via event_tx.
async fn handle_subscribe(
    req: &WsRequest,
    cdc_log: &Option<Arc<sochdb_storage::cdc::CdcLog>>,
    event_tx: &mpsc::Sender<WsResponse>,
) -> WsResponse {
    let cdc = match cdc_log {
        Some(log) => log.clone(),
        None => {
            return WsResponse {
                id: req.id.clone(),
                msg_type: "error".into(),
                payload: serde_json::json!({ "message": "CDC not enabled" }),
            };
        }
    };

    let payload: SubscribePayload = match serde_json::from_value(req.payload.clone()) {
        Ok(p) => p,
        Err(e) => {
            return WsResponse {
                id: req.id.clone(),
                msg_type: "error".into(),
                payload: serde_json::json!({ "message": format!("Invalid payload: {}", e) }),
            };
        }
    };

    let sub_id = req.id.clone();
    let event_tx = event_tx.clone();

    // Start subscription in background
    tokio::spawn(async move {
        use sochdb_storage::cdc::CdcSubscriber;
        use std::time::Duration;

        let mut subscriber = if payload.start_sequence > 0 {
            CdcSubscriber::new(cdc, payload.start_sequence)
        } else {
            CdcSubscriber::from_latest(cdc)
        };

        if !payload.tables.is_empty() {
            subscriber = subscriber.with_tables(payload.tables);
        }

        loop {
            let events = match subscriber.next_batch(32, Duration::from_millis(200)) {
                Ok(events) => events,
                Err(_) => continue,
            };

            for event in events {
                let ws_event = WsResponse {
                    id: sub_id.clone(),
                    msg_type: "event".into(),
                    payload: serde_json::json!({
                        "sequence": event.sequence,
                        "timestamp_us": event.timestamp_us,
                        "txn_id": event.txn_id,
                        "table": event.table,
                        "operation": format!("{:?}", event.operation),
                    }),
                };

                if event_tx.send(ws_event).await.is_err() {
                    // Client disconnected
                    return;
                }
            }
        }
    });

    WsResponse {
        id: req.id.clone(),
        msg_type: "result".into(),
        payload: serde_json::json!({ "subscribed": true }),
    }
}

/// Current time in milliseconds.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ws_request_sql() {
        let json = r#"{"id":"r1","type":"sql","payload":{"query":"SELECT 1"}}"#;
        let req: WsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.id, "r1");
        assert_eq!(req.msg_type, "sql");
    }

    #[test]
    fn test_parse_ws_request_ping() {
        let json = r#"{"id":"p1","type":"ping"}"#;
        let req: WsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.msg_type, "ping");
    }

    #[test]
    fn test_parse_ws_request_kv_get() {
        let json = r#"{"id":"k1","type":"kv_get","payload":{"key":"mykey","namespace":"ns1"}}"#;
        let req: WsRequest = serde_json::from_str(json).unwrap();
        let payload: KvGetPayload = serde_json::from_value(req.payload).unwrap();
        assert_eq!(payload.key, "mykey");
        assert_eq!(payload.namespace, "ns1");
    }

    #[test]
    fn test_parse_ws_request_kv_put() {
        let json = r#"{"id":"k2","type":"kv_put","payload":{"key":"k","value":"v","ttl_seconds":60}}"#;
        let req: WsRequest = serde_json::from_str(json).unwrap();
        let payload: KvPutPayload = serde_json::from_value(req.payload).unwrap();
        assert_eq!(payload.key, "k");
        assert_eq!(payload.value, "v");
        assert_eq!(payload.ttl_seconds, 60);
    }

    #[test]
    fn test_serialize_ws_response() {
        let resp = WsResponse {
            id: "r1".into(),
            msg_type: "result".into(),
            payload: serde_json::json!({"count": 42}),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"id\":\"r1\""));
        assert!(json.contains("\"type\":\"result\""));
        assert!(json.contains("\"count\":42"));
    }

    #[tokio::test]
    async fn test_handle_ping() {
        let kv_store: KvStore = Arc::new(dashmap::DashMap::new());
        let cdc_log = None;
        let (event_tx, _rx) = mpsc::channel(16);

        let resp = handle_message(
            r#"{"id":"p1","type":"ping","payload":{}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;

        assert_eq!(resp.id, "p1");
        assert_eq!(resp.msg_type, "pong");
    }

    #[tokio::test]
    async fn test_handle_kv_put_get_delete() {
        let kv_store: KvStore = Arc::new(dashmap::DashMap::new());
        let cdc_log = None;
        let (event_tx, _rx) = mpsc::channel(16);

        // Put
        let resp = handle_message(
            r#"{"id":"1","type":"kv_put","payload":{"key":"hello","value":"world"}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;
        assert_eq!(resp.payload["ok"], true);

        // Get
        let resp = handle_message(
            r#"{"id":"2","type":"kv_get","payload":{"key":"hello"}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;
        assert_eq!(resp.payload["found"], true);
        assert_eq!(resp.payload["value"], "world");

        // Delete
        let resp = handle_message(
            r#"{"id":"3","type":"kv_delete","payload":{"key":"hello"}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;
        assert_eq!(resp.payload["deleted"], true);

        // Get after delete
        let resp = handle_message(
            r#"{"id":"4","type":"kv_get","payload":{"key":"hello"}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;
        assert_eq!(resp.payload["found"], false);
    }

    #[tokio::test]
    async fn test_handle_unknown_type() {
        let kv_store: KvStore = Arc::new(dashmap::DashMap::new());
        let cdc_log = None;
        let (event_tx, _rx) = mpsc::channel(16);

        let resp = handle_message(
            r#"{"id":"x","type":"foobar","payload":{}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;

        assert_eq!(resp.msg_type, "error");
        assert!(resp.payload["message"]
            .as_str()
            .unwrap()
            .contains("Unknown"));
    }

    #[tokio::test]
    async fn test_handle_invalid_json() {
        let kv_store: KvStore = Arc::new(dashmap::DashMap::new());
        let cdc_log = None;
        let (event_tx, _rx) = mpsc::channel(16);

        let resp = handle_message("not json at all", &kv_store, &cdc_log, &event_tx).await;
        assert_eq!(resp.msg_type, "error");
    }

    #[tokio::test]
    async fn test_handle_subscribe_no_cdc() {
        let kv_store: KvStore = Arc::new(dashmap::DashMap::new());
        let cdc_log = None;
        let (event_tx, _rx) = mpsc::channel(16);

        let resp = handle_message(
            r#"{"id":"s1","type":"subscribe","payload":{"tables":["users"]}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;

        assert_eq!(resp.msg_type, "error");
        assert!(resp.payload["message"]
            .as_str()
            .unwrap()
            .contains("CDC"));
    }

    #[tokio::test]
    async fn test_handle_sql() {
        let kv_store: KvStore = Arc::new(dashmap::DashMap::new());
        let cdc_log = None;
        let (event_tx, _rx) = mpsc::channel(16);

        let resp = handle_message(
            r#"{"id":"q1","type":"sql","payload":{"query":"SELECT 1"}}"#,
            &kv_store,
            &cdc_log,
            &event_tx,
        )
        .await;

        assert_eq!(resp.id, "q1");
        assert_eq!(resp.msg_type, "result");
        assert_eq!(resp.payload["query"], "SELECT 1");
    }
}
