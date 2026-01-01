//! Query execution commands

use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;
use crate::state::AppState;

/// Result of a query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub stats: QueryStats,
}

/// Query execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    pub row_count: usize,
    pub execution_time_ms: f64,
    pub scanned_rows: usize,
}

/// Execute a SQL/ToonQL query
#[tauri::command]
pub async fn execute_query(
    state: State<'_, Arc<AppState>>,
    query: String,
) -> Result<QueryResult, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }

    let start = std::time::Instant::now();
    
    // Use MCP tools/call for query execution
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref().ok_or("MCP server not initialized")?;
    
    // Create query request
    let req = toondb_mcp::RpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::Value::Number(1.into()),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": "query",
            "arguments": { "query": query }
        }),
    };
    
    let resp = mcp.dispatch(&req);
    let execution_time = start.elapsed().as_secs_f64() * 1000.0;
    
    match resp.result {
        Some(result) => {
            // Parse MCP result into QueryResult
            Ok(QueryResult {
                columns: vec!["result".to_string()],
                rows: vec![vec![result]],
                stats: QueryStats {
                    row_count: 1,
                    execution_time_ms: execution_time,
                    scanned_rows: 0,
                },
            })
        },
        None => {
            let err_msg = resp.error.map(|e| e.message).unwrap_or("Query failed".to_string());
            Err(err_msg)
        }
    }
}

/// Get query explanation/plan
#[tauri::command]
pub async fn explain_query(
    state: State<'_, Arc<AppState>>,
    _query: String,
) -> Result<String, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // TODO: Implement explain via MCP
    Ok("Query plan: Full scan (explain not yet implemented)".to_string())
}
