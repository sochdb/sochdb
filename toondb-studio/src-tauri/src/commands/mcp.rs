//! MCP Commands - Expose MCP tools via Tauri IPC

use serde_json::Value;
use tauri::State;
use std::sync::Arc;

use crate::state::AppState;
use toondb_mcp::RpcRequest;

/// List available MCP tools
#[tauri::command]
pub async fn mcp_list_tools(
    state: State<'_, Arc<AppState>>,
) -> Result<Value, String> {
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref().ok_or("MCP server not initialized. Connect to a database first.")?;
    
    // Create a tools/list request
    let req = RpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Value::Number(1.into()),
        method: "tools/list".to_string(),
        params: Value::Null,
    };
    
    let resp = mcp.dispatch(&req);
    
    match resp.result {
        Some(result) => Ok(result),
        None => Err(resp.error.map(|e| e.message).unwrap_or("Unknown error".to_string())),
    }
}

/// Call an MCP tool
#[tauri::command]
pub async fn mcp_call_tool(
    state: State<'_, Arc<AppState>>,
    tool_name: String,
    arguments: Value,
) -> Result<Value, String> {
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref().ok_or("MCP server not initialized. Connect to a database first.")?;
    
    // Create a tools/call request
    let req = RpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Value::Number(1.into()),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": tool_name,
            "arguments": arguments,
        }),
    };
    
    let resp = mcp.dispatch(&req);
    
    match resp.result {
        Some(result) => Ok(result),
        None => Err(resp.error.map(|e| e.message).unwrap_or("Unknown error".to_string())),
    }
}

/// Get service status (DB connection + MCP)
#[tauri::command]
pub async fn get_service_status(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::state::ServiceStatus, String> {
    Ok(state.get_status().await)
}

