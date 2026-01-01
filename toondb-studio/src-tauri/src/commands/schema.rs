//! Schema introspection commands

use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;
use crate::state::AppState;

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub is_primary_key: bool,
}

/// Table definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    pub name: String,
    pub row_count: usize,
    pub columns: Vec<ColumnInfo>,
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub name: String,
    pub table_name: String,
    pub column_name: String,
    pub index_type: String,
    pub config: Option<String>,
}

/// Path (KV store) info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathInfo {
    pub path: String,
    pub count: usize,
}

/// Complete database schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    pub tables: Vec<TableInfo>,
    pub indexes: Vec<IndexInfo>,
    pub paths: Vec<PathInfo>,
}

/// Get the complete database schema
#[tauri::command]
pub async fn get_schema(
    state: State<'_, Arc<AppState>>,
) -> Result<DatabaseSchema, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }

    // Use MCP to list tables via scan
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref().ok_or("MCP server not initialized")?;
    
    // List tables via MCP
    let req = toondb_mcp::RpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::Value::Number(1.into()),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": "list_tables",
            "arguments": {}
        }),
    };
    
    let _resp = mcp.dispatch(&req);
    
    // Return basic schema - real implementation would parse MCP response
    Ok(DatabaseSchema {
        tables: vec![],
        indexes: vec![],
        paths: vec![],
    })
}

/// Get information about a specific table
#[tauri::command]
pub async fn get_table_info(
    state: State<'_, Arc<AppState>>,
    table_name: String,
) -> Result<TableInfo, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // TODO: Query table schema via MCP
    Ok(TableInfo {
        name: table_name,
        row_count: 0,
        columns: vec![],
    })
}
