//! Connection pool for ToonDB connections

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use toondb::DurableToonClient;

/// Connection pool entry
#[allow(dead_code)]
#[derive(Clone)]
pub struct PoolEntry {
    pub path: String,
    pub connected: bool,
    pub client: Option<Arc<DurableToonClient>>,
}

/// Connection pool for managing multiple database connections
#[allow(dead_code)]
pub struct ConnectionPool {
    connections: Arc<RwLock<HashMap<String, PoolEntry>>>,
}

#[allow(dead_code)]
impl ConnectionPool {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a connection to the pool
    pub async fn add(&self, id: String, path: String, client: Arc<DurableToonClient>) -> Result<(), String> {
        let mut conns = self.connections.write().await;
        conns.insert(
            id,
            PoolEntry {
                path,
                connected: true,
                client: Some(client),
            },
        );
        Ok(())
    }

    /// Remove a connection from the pool
    pub async fn remove(&self, id: &str) -> Result<(), String> {
        let mut conns = self.connections.write().await;
        conns.remove(id);
        Ok(())
    }

    /// Get a connection from the pool
    pub async fn get(&self, id: &str) -> Option<PoolEntry> {
        let conns = self.connections.read().await;
        conns.get(id).cloned()
    }

    /// List all connections
    pub async fn list(&self) -> Vec<(String, PoolEntry)> {
        let conns = self.connections.read().await;
        conns
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}
