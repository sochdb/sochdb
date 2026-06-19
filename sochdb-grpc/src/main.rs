// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! SochDB gRPC Server
//!
//! Starts a comprehensive gRPC server with all SochDB services.
//! This implements the "Thick Server / Thin Client" architecture where
//! all business logic lives in Rust, enabling thin SDK wrappers.
//!
//! ## Services
//!
//! - VectorIndexService: HNSW vector operations
//! - GraphService: Graph overlay for agent memory
//! - PolicyService: Policy evaluation
//! - ContextService: LLM context assembly
//! - CollectionService: Collection management
//! - NamespaceService: Multi-tenant namespaces
//! - SemanticCacheService: Semantic caching
//! - TraceService: Distributed tracing
//! - CheckpointService: State snapshots
//! - McpService: MCP tool routing
//! - KvService: Key-value operations
//!
//! ## Usage
//!
//! ```bash
//! # Start on default port 50051
//! sochdb-grpc-server
//!
//! # Start on custom port
//! sochdb-grpc-server --port 8080
//!
//! # Bind to specific address
//! sochdb-grpc-server --host 0.0.0.0 --port 50051
//! ```

use std::sync::Arc;

use clap::Parser;
use tonic::transport::Server;
use tonic_health::server::health_reporter;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

use sochdb_grpc::{
    SecurityConfig, SecurityService, VectorIndexServer,
    auth_interceptor::AuthInterceptor,
    checkpoint_server::CheckpointServer,
    collection_server::CollectionServer,
    context_server::ContextServer,
    graph_server::GraphServer,
    kv_server::KvServer,
    mcp_server::McpServer,
    namespace_server::NamespaceServer,
    policy_server::PolicyServer,
    proto::{
        checkpoint_service_server::CheckpointServiceServer,
        collection_service_server::CollectionServiceServer,
        context_service_server::ContextServiceServer, graph_service_server::GraphServiceServer,
        kv_service_server::KvServiceServer, mcp_service_server::McpServiceServer,
        namespace_service_server::NamespaceServiceServer,
        policy_service_server::PolicyServiceServer,
        semantic_cache_service_server::SemanticCacheServiceServer,
        subscription_service_server::SubscriptionServiceServer,
        trace_service_server::TraceServiceServer,
        vector_index_service_server::VectorIndexServiceServer,
    },
    security::{AuthMethod, Principal, Role},
    semantic_cache_server::SemanticCacheServer,
    subscription_server::SubscriptionServer,
    trace_server::TraceServer,
};
use sochdb_memory::MemoryStore;

/// SochDB gRPC Server
#[derive(Parser, Debug)]
#[command(name = "sochdb-grpc-server")]
#[command(about = "SochDB gRPC server - Thick Server / Thin Client architecture")]
#[command(version)]
struct Args {
    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "50051")]
    port: u16,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,

    /// Prometheus metrics HTTP port (0 to disable)
    #[arg(long, default_value = "9090")]
    metrics_port: u16,

    /// WebSocket gateway port (0 to disable)
    #[arg(long, default_value = "8080")]
    ws_port: u16,

    /// PostgreSQL wire protocol port (0 to disable)
    #[arg(long, default_value = "5433")]
    pg_port: u16,

    /// Enable gRPC authentication (Task 7)
    #[arg(long)]
    auth: bool,

    /// Register an API key for authentication (requires --auth)
    #[arg(long = "api-key", env = "SOCHDB_API_KEY")]
    api_key: Option<String>,

    /// TLS certificate PEM path (enables TLS)
    #[arg(long = "tls-cert", env = "SOCHDB_TLS_CERT")]
    tls_cert: Option<String>,

    /// TLS private key PEM path
    #[arg(long = "tls-key", env = "SOCHDB_TLS_KEY")]
    tls_key: Option<String>,

    /// CA certificate for mTLS client verification
    #[arg(long = "tls-ca", env = "SOCHDB_TLS_CA")]
    tls_ca: Option<String>,

    /// Secrets mount path (Kubernetes Secrets volume)
    #[arg(long = "secrets-path", env = "SOCHDB_SECRETS_PATH")]
    secrets_path: Option<String>,

    /// Persistent data directory for the PostgreSQL wire SQL engine. When set,
    /// `--pg-port` executes real SQL (SELECT/INSERT/UPDATE/DELETE/DDL, incl.
    /// JOINs) against a database at this path instead of echoing queries back.
    /// This store is independent of the in-memory gRPC services.
    #[arg(long = "pg-data-dir", env = "SOCHDB_PG_DATA_DIR")]
    pg_data_dir: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize tracing
    let filter = if args.debug {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    let addr = format!("{}:{}", args.host, args.port).parse()?;

    // Start Prometheus metrics HTTP server (Task 8)
    let _metrics_handle = if args.metrics_port > 0 {
        Some(sochdb_grpc::metrics_server::start(
            args.host.clone(),
            args.metrics_port,
        ))
    } else {
        tracing::info!("Prometheus metrics endpoint disabled");
        None
    };

    // Start WebSocket gateway (Task 4)
    let _ws_handle = if args.ws_port > 0 {
        let ws_addr = format!("{}:{}", args.host, args.ws_port).parse()?;
        let kv_store: sochdb_grpc::ws_server::KvStore =
            std::sync::Arc::new(dashmap::DashMap::new());
        Some(sochdb_grpc::ws_server::start(
            sochdb_grpc::ws_server::WsConfig {
                addr: ws_addr,
                kv_store,
                cdc_log: None,
            },
        ))
    } else {
        tracing::info!("WebSocket gateway disabled");
        None
    };

    // Start PG wire protocol server (Task 5)
    let _pg_handle = if args.pg_port > 0 {
        let pg_addr = format!("{}:{}", args.host, args.pg_port).parse()?;
        let config = sochdb_grpc::pg_wire::PgWireConfig {
            addr: pg_addr,
            server_version: format!("SochDB {}", env!("CARGO_PKG_VERSION")),
        };
        match args.pg_data_dir.as_deref() {
            // Real SQL engine over a persistent database.
            Some(dir) => {
                // Trust authentication is acceptable on loopback only. Binding
                // a writable SQL database to a non-loopback address exposes it
                // unauthenticated on the network (pg_wire has no auth layer,
                // unlike the gRPC interceptor), so warn loudly.
                let is_loopback = matches!(args.host.as_str(), "127.0.0.1" | "::1" | "localhost");
                if !is_loopback {
                    tracing::warn!(
                        "PG wire protocol is serving a WRITABLE SQL database on non-loopback \
                         address '{}' with NO authentication (trust auth). Anyone who can reach \
                         this port can read and modify all data. Bind to 127.0.0.1 or place it \
                         behind an authenticating proxy.",
                        args.host
                    );
                }
                // Fail fast: if the operator explicitly requested real SQL but
                // the database cannot be opened, refuse to start rather than
                // silently degrading to the echo executor (which would return
                // fabricated rows that look like real results).
                let db = sochdb_storage::Database::open(dir)
                    .map_err(|e| format!("failed to open PG SQL database at '{}': {}", dir, e))?;
                tracing::info!(
                    "PG wire protocol executing real SQL against database at '{}'",
                    dir
                );
                Some(sochdb_grpc::pg_wire::start(
                    config,
                    sochdb_grpc::pg_wire::DatabasePgExecutor::new(db),
                ))
            }
            // No data dir configured: keep the echo placeholder (unchanged default).
            None => {
                tracing::info!(
                    "PG wire protocol using echo executor (set --pg-data-dir to enable real SQL)"
                );
                Some(sochdb_grpc::pg_wire::start(
                    config,
                    sochdb_grpc::pg_wire::EchoPgExecutor,
                ))
            }
        }
    } else {
        tracing::info!("PG wire protocol disabled");
        None
    };

    // Create all service instances
    // NamespaceServer is created first and cloned to all services that need
    // quota enforcement — the inner DashMap is Arc-wrapped so all clones share state.
    let namespace_server = NamespaceServer::new();
    let vector_server = VectorIndexServer::with_namespace_server(namespace_server.clone());
    let graph_server = GraphServer::with_namespace_server(namespace_server.clone());
    let collection_server = CollectionServer::with_namespace_server(namespace_server.clone());
    let policy_server = Arc::new(PolicyServer::new());
    let kv_server = KvServer::with_namespace_server(namespace_server.clone())
        .with_policy_server(policy_server.clone());
    let memory_store = Arc::new(MemoryStore::with_defaults());
    let context_server = ContextServer::with_memory_store(Arc::clone(&memory_store));
    let semantic_cache_server = SemanticCacheServer::new();
    let trace_server = TraceServer::new();
    let checkpoint_server = CheckpointServer::new();
    let mcp_server = McpServer::new();

    // Load secrets from Kubernetes Secrets mount or environment variables
    let secrets = if let Some(ref path) = args.secrets_path {
        let provider = sochdb_grpc::security::SecretsProvider::from_mount(path);
        if let Err(e) = provider.refresh() {
            tracing::warn!("Failed to load secrets from {}: {}", path, e);
        }
        Some(provider)
    } else {
        let provider = sochdb_grpc::security::SecretsProvider::from_env();
        let _ = provider.refresh();
        Some(provider)
    };

    // Create CDC log for subscriptions
    let cdc_log = sochdb_storage::cdc::CdcLog::new(sochdb_storage::cdc::CdcConfig {
        enabled: true,
        capacity: 65536,
    });
    let subscription_server = SubscriptionServer::new(cdc_log);

    // Create authentication interceptor (Task 7)
    let auth = if args.auth {
        let mut sec_config = SecurityConfig::default();
        sec_config.api_key_enabled = true;
        sec_config.jwt_enabled = true;
        // Optional server-side pepper for API-key hashing (HMAC-SHA256). Sourced
        // from a secret manager / KMS via env; when absent, falls back to bare
        // SHA-256 for backward compatibility.
        sec_config.api_key_pepper = std::env::var("SOCHDB_API_KEY_PEPPER")
            .ok()
            .filter(|p| !p.is_empty());

        let security = Arc::new(SecurityService::new(sec_config));

        // Apply secrets (JWT key, API keys) from secrets provider
        if let Some(ref secrets_provider) = secrets {
            secrets_provider.apply_to_security(&security);
        }

        let interceptor = AuthInterceptor::new(security, true);
        // Register API key if provided via CLI
        if let Some(ref key) = args.api_key {
            interceptor.register_api_key(
                key,
                Principal {
                    id: "api-key-user".to_string(),
                    tenant_id: "default".to_string(),
                    capabilities: Role::Owner.capabilities(),
                    expires_at: None,
                    auth_method: AuthMethod::ApiKey,
                },
            );
            tracing::info!("Registered API key (keyed-hash, Owner role)");
        }
        tracing::info!("Authentication enabled");
        interceptor
    } else {
        tracing::info!("Authentication disabled (use --auth to enable)");
        AuthInterceptor::disabled()
    };

    // Create gRPC health service for Kubernetes probes
    let (health_reporter, health_service) = health_reporter();

    // Mark the overall service as serving (empty service name = overall health)
    // The empty string "" represents overall server health
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;

    tracing::info!("Starting SochDB gRPC server on {}", addr);
    tracing::info!("Server version: {}", env!("CARGO_PKG_VERSION"));

    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║            SochDB gRPC Server (Thick Server)                 ║
╠══════════════════════════════════════════════════════════════╣
║  Server:     {}                                   
║  Version:    {}                                            
║  Auth:       {}                                            
║                                                              ║
║  Services:                                                   ║
║    - VectorIndexService    Vector index operations           ║
║    - GraphService          Graph overlay                     ║
║    - PolicyService         Policy evaluation                 ║
║    - ContextService        LLM context assembly              ║
║    - CollectionService     Collection management             ║
║    - NamespaceService      Multi-tenant namespaces           ║
║    - SemanticCacheService  Semantic caching                  ║
║    - TraceService          Distributed tracing               ║
║    - CheckpointService     State snapshots                   ║
║    - McpService            MCP tool routing                  ║
║    - KvService             Key-value operations              ║
║    - SubscriptionService   Real-time change notifications    ║
║                                                              ║
║  Metrics:  http://0.0.0.0:{}/metrics                        ║
║  WebSocket: ws://{}:{}/                                     ║
║  PG Wire:   postgresql://{}:{}/sochdb                        ║
╚══════════════════════════════════════════════════════════════╝
"#,
        addr,
        env!("CARGO_PKG_VERSION"),
        if args.auth { "ENABLED" } else { "disabled" },
        args.metrics_port,
        args.host,
        args.ws_port,
        args.host,
        args.pg_port
    );

    // ── TLS Configuration ─────────────────────────────────────────────
    let tls_mode = if let (Some(cert), Some(key)) = (&args.tls_cert, &args.tls_key) {
        match sochdb_grpc::security::TlsProvider::new(
            cert.as_str(),
            key.as_str(),
            args.tls_ca.as_deref(),
        ) {
            Ok(provider) => {
                let tls_config = provider
                    .configure_server()
                    .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
                if provider.is_mtls_enabled() {
                    tracing::info!("TLS + mTLS enabled");
                } else {
                    tracing::info!("TLS enabled (no mTLS)");
                }
                Some(tls_config)
            }
            Err(e) => {
                tracing::error!("Failed to load TLS certificates: {}", e);
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        }
    } else {
        tracing::info!("TLS disabled (use --tls-cert and --tls-key to enable)");
        None
    };

    // Health service is NOT behind auth (Kubernetes probes must be unauthenticated)
    // All other services go through the auth interceptor
    //
    // PolicyServer is Arc-wrapped so KvServer can call evaluate_internal() directly.
    // We clone the inner PolicyServer for the gRPC service registration.
    let policy_grpc = PolicyServer::new(); // Separate instance for gRPC (same trait)
    let mut builder = Server::builder();
    if let Some(tls_config) = tls_mode {
        builder = builder.tls_config(tls_config)?;
    }

    builder
        .add_service(health_service)
        // SECURITY: the vector service MUST be wrapped with the auth interceptor
        // like every other service — otherwise the interceptor never runs for
        // vector RPCs, extract_principal falls back to an (over-privileged)
        // anonymous principal, and the entire vector data plane is readable /
        // writable / destroyable with no credentials EVEN WHEN --auth is on.
        // Configure the message-size limits on the inner server first, then wrap
        // it in the interceptor (max_*_message_size is not available post-wrap).
        .add_service(tonic::codegen::InterceptedService::new(
            VectorIndexServiceServer::new(vector_server)
                .max_decoding_message_size(64 * 1024 * 1024)
                .max_encoding_message_size(64 * 1024 * 1024),
            auth.clone(),
        ))
        .add_service(GraphServiceServer::with_interceptor(
            graph_server,
            auth.clone(),
        ))
        .add_service(PolicyServiceServer::with_interceptor(
            policy_grpc,
            auth.clone(),
        ))
        .add_service(ContextServiceServer::with_interceptor(
            context_server,
            auth.clone(),
        ))
        .add_service(CollectionServiceServer::with_interceptor(
            collection_server,
            auth.clone(),
        ))
        .add_service(NamespaceServiceServer::with_interceptor(
            namespace_server,
            auth.clone(),
        ))
        .add_service(SemanticCacheServiceServer::with_interceptor(
            semantic_cache_server,
            auth.clone(),
        ))
        .add_service(TraceServiceServer::with_interceptor(
            trace_server,
            auth.clone(),
        ))
        .add_service(CheckpointServiceServer::with_interceptor(
            checkpoint_server,
            auth.clone(),
        ))
        .add_service(McpServiceServer::with_interceptor(mcp_server, auth.clone()))
        .add_service(KvServiceServer::with_interceptor(kv_server, auth.clone()))
        .add_service(SubscriptionServiceServer::with_interceptor(
            subscription_server,
            auth.clone(),
        ))
        .serve(addr)
        .await?;

    Ok(())
}
