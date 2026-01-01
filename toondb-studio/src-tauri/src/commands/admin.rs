//! Admin commands for database operations

use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;
use std::path::PathBuf;

use crate::state::AppState;

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub memtable_size_bytes: usize,
    pub wal_size_bytes: usize,
    pub total_tables: usize,
    pub total_rows: usize,
    pub active_transactions: usize,
    pub last_checkpoint_lsn: u64,
    pub uptime_seconds: u64,
    pub version: String,
    pub active_snapshots: usize,
    pub min_active_timestamp: u64,
    pub garbage_versions: usize,
}

/// Connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub path: String,
    pub connected: bool,
    pub version: String,
    pub uptime_seconds: u64,
}

/// Connect to a ToonDB database
#[tauri::command]
pub async fn connect(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<ConnectionInfo, String> {
    let path_buf = PathBuf::from(&path);
    
    // Use unified AppState connect
    state.connect(path_buf).await?;

    Ok(ConnectionInfo {
        path,
        connected: true,
        version: "0.1.0".to_string(),
        uptime_seconds: 0,
    })
}

/// Disconnect from the current database
#[tauri::command]
pub async fn disconnect(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.disconnect().await;
    Ok(())
}

/// Get current database statistics
#[tauri::command]
pub async fn get_stats(state: State<'_, Arc<AppState>>) -> Result<DatabaseStats, String> {
    let status = state.get_status().await;
    
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // Get real stats from MCP server's database connection
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref()
        .ok_or_else(|| "MCP server not initialized".to_string())?;
    
    let db_stats = mcp.db_stats();
    let conn = mcp.connection();
    
    // Count tables from scan (tables are top-level paths)
    conn.begin().ok();
    let scan_result = conn.scan("/").unwrap_or_default();
    let mut table_set = std::collections::HashSet::new();
    let mut total_rows = 0usize;
    for (key, _) in &scan_result {
        let parts: Vec<&str> = key.trim_start_matches('/').split('/').collect();
        if let Some(first) = parts.first() {
            if !first.is_empty() {
                table_set.insert(first.to_string());
            }
        }
        total_rows += 1;
    }
    conn.abort().ok();
    
    Ok(DatabaseStats {
        memtable_size_bytes: db_stats.bytes_written as usize,
        wal_size_bytes: db_stats.bytes_read as usize,
        total_tables: table_set.len(),
        total_rows,
        active_transactions: db_stats.transactions_started.saturating_sub(
            db_stats.transactions_committed + db_stats.transactions_aborted
        ) as usize,
        last_checkpoint_lsn: 0, // Not exposed in Stats yet
        uptime_seconds: 0, // TODO: Track startup time
        version: env!("CARGO_PKG_VERSION").to_string(),
        active_snapshots: 0, // TODO: Expose from MVCC
        min_active_timestamp: 0, // TODO: Expose from MVCC
        garbage_versions: 0, // TODO: Expose from GC
    })
}

/// Force a WAL checkpoint
#[tauri::command]
pub async fn checkpoint(state: State<'_, Arc<AppState>>) -> Result<u64, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // Get connection from MCP server and force checkpoint
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref()
        .ok_or_else(|| "MCP server not initialized".to_string())?;
    
    // Force sync - this ensures all WAL entries are flushed
    mcp.connection().kernel().fsync()
        .map_err(|e| format!("Checkpoint failed: {}", e))?;
    
    // Return the current LSN (bytes written as proxy)
    let stats = mcp.db_stats();
    Ok(stats.bytes_written)
}

/// Run garbage collection
#[tauri::command]
pub async fn gc(state: State<'_, Arc<AppState>>) -> Result<usize, String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // Get connection from MCP server and run GC
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref()
        .ok_or_else(|| "MCP server not initialized".to_string())?;
    
    let reclaimed = mcp.connection().gc();
    Ok(reclaimed)
}

/// Analyze table statistics
#[allow(dead_code)]
#[tauri::command]
pub async fn analyze() -> Result<(), String> {
    Ok(())
}

/// Compact SST files
#[tauri::command]
pub async fn compact(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let status = state.get_status().await;
    if !status.db_connected {
        return Err("No active connection".to_string());
    }
    
    // Get connection from MCP server
    let mcp_lock = state.mcp_server.read().await;
    let mcp = mcp_lock.as_ref()
        .ok_or_else(|| "MCP server not initialized".to_string())?;
    
    // Run GC which cleans up old versions (compaction is automatic via LSM)
    let _ = mcp.connection().gc();
    
    // Force sync to persist any changes
    mcp.connection().kernel().fsync()
        .map_err(|e| format!("Compact sync failed: {}", e))?;
    
    Ok(())
}
/// Get the current command policy
#[tauri::command]
pub async fn get_policy(state: State<'_, Arc<AppState>>) -> Result<crate::policy::CommandPolicy, String> {
    Ok(state.get_policy().await)
}

/// Set the command policy
#[tauri::command]
pub async fn set_policy(
    state: State<'_, Arc<AppState>>,
    policy: crate::policy::CommandPolicy,
) -> Result<(), String> {
    state.set_policy(policy).await;
    Ok(())
}

/// Check if a command is allowed under current policy
#[tauri::command]
pub async fn check_command(
    state: State<'_, Arc<AppState>>,
    command: String,
) -> Result<crate::policy::PolicyCheck, String> {
    let policy = state.get_policy().await;
    Ok(crate::policy::check_command(&policy, &command))
}

/// Get list of commands allowed under current policy
#[tauri::command]
pub async fn get_allowed_commands(state: State<'_, Arc<AppState>>) -> Result<Vec<String>, String> {
    let policy = state.get_policy().await;
    Ok(policy.get_allowed_commands().iter().map(|s| s.to_string()).collect())
}