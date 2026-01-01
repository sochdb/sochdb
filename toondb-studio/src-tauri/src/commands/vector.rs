//! Vector search commands

use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;
use crate::state::AppState;

/// Vector search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorResult {
    pub id: String,
    pub score: f32,
    pub content: String,
    pub metadata: serde_json::Value,
}

/// Perform a vector similarity search
#[tauri::command]
pub async fn vector_search(
    state: State<'_, Arc<AppState>>,
    table: String,
    column: String,
    query_text: String,
    limit: usize,
) -> Result<Vec<VectorResult>, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // Use MCP for vector search
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref().ok_or("MCP server not initialized")?;
    
    let req = toondb_mcp::RpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::Value::Number(1.into()),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": "vector_search",
            "arguments": {
                "table": table,
                "column": column,
                "query": query_text,
                "limit": limit
            }
        }),
    };
    
    let _resp = mcp.dispatch(&req);
    
    // Return empty for now - real implementation would parse MCP response
    Ok(vec![])
}

/// Get vector columns for a table
#[tauri::command]
pub async fn get_vector_columns(
    state: State<'_, Arc<AppState>>,
    _table: String,
) -> Result<Vec<String>, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // TODO: Query vector columns via MCP
    Ok(vec![])
}
