//! Connection manager for handling multiple ToonDB connections

use super::pool::ConnectionPool;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Saved connection configuration
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedConnection {
    pub id: String,
    pub name: String,
    pub path: String,
    pub connection_type: ConnectionType,
}

/// Type of connection
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionType {
    LocalFile,
    UnixSocket,
    TcpIp,
    Embedded,
}

/// Connection manager state
#[allow(dead_code)]
pub struct ConnectionManager {
    pool: Arc<ConnectionPool>,
    saved_connections: Vec<SavedConnection>,
    active_connection_id: Option<String>,
}

#[allow(dead_code)]
impl ConnectionManager {
    pub fn new() -> Self {
        Self {
            pool: Arc::new(ConnectionPool::new()),
            saved_connections: vec![
                SavedConnection {
                    id: "local-dev".to_string(),
                    name: "Local Development".to_string(),
                    path: "./dev.toondb".to_string(),
                    connection_type: ConnectionType::LocalFile,
                },
                SavedConnection {
                    id: "production".to_string(),
                    name: "Production".to_string(),
                    path: "/var/toondb/prod".to_string(),
                    connection_type: ConnectionType::LocalFile,
                },
                SavedConnection {
                    id: "remote".to_string(),
                    name: "Remote Server".to_string(),
                    path: "192.168.1.10:7654".to_string(),
                    connection_type: ConnectionType::TcpIp,
                },
            ],
            active_connection_id: Some("local-dev".to_string()),
        }
    }

    /// Get the connection pool
    pub fn pool(&self) -> Arc<ConnectionPool> {
        self.pool.clone()
    }

    /// Get saved connections
    pub fn saved_connections(&self) -> &[SavedConnection] {
        &self.saved_connections
    }

    /// Add a new saved connection
    pub fn add_saved_connection(&mut self, conn: SavedConnection) {
        self.saved_connections.push(conn);
    }

    /// Remove a saved connection
    pub fn remove_saved_connection(&mut self, id: &str) {
        self.saved_connections.retain(|c| c.id != id);
    }

    /// Get the active connection ID
    pub fn active_connection_id(&self) -> Option<&str> {
        self.active_connection_id.as_deref()
    }

    /// Set the active connection
    pub fn set_active_connection(&mut self, id: String) {
        self.active_connection_id = Some(id);
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}
