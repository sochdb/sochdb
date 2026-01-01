// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! ToonDB gRPC Vector Index Server
//!
//! Starts a gRPC server for vector index operations.
//!
//! ## Usage
//!
//! ```bash
//! # Start on default port 50051
//! toondb-grpc-server
//!
//! # Start on custom port
//! toondb-grpc-server --port 8080
//!
//! # Bind to specific address
//! toondb-grpc-server --host 0.0.0.0 --port 50051
//! ```

use clap::Parser;
use tonic::transport::Server;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use toondb_grpc::VectorIndexServer;

/// ToonDB gRPC Vector Index Server
#[derive(Parser, Debug)]
#[command(name = "toondb-grpc-server")]
#[command(about = "ToonDB gRPC server for vector index operations")]
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
    let server = VectorIndexServer::new();
    
    tracing::info!("Starting ToonDB gRPC server on {}", addr);
    tracing::info!("Server version: {}", env!("CARGO_PKG_VERSION"));
    
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║                 ToonDB gRPC Vector Index                     ║
╠══════════════════════════════════════════════════════════════╣
║  Server:     {}                                   
║  Version:    {}                                            
║                                                              ║
║  Endpoints:                                                  ║
║    - CreateIndex    - Create new HNSW index                  ║
║    - InsertBatch    - Batch vector insertion                 ║
║    - InsertStream   - Streaming insertion                    ║
║    - Search         - K-nearest neighbor search              ║
║    - SearchBatch    - Batch search                           ║
║    - GetStats       - Index statistics                       ║
║    - HealthCheck    - Health check                           ║
╚══════════════════════════════════════════════════════════════╝
"#,
        addr,
        env!("CARGO_PKG_VERSION")
    );
    
    Server::builder()
        .add_service(server.into_service())
        .serve(addr)
        .await?;
    
    Ok(())
}
