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

//! ToonDB gRPC Vector Index Service
//!
//! This crate provides a gRPC interface for ToonDB's HNSW vector index.
//! It enables cross-language clients to perform vector operations over
//! the network using Protocol Buffers.
//!
//! ## Features
//!
//! - **CreateIndex**: Create new HNSW indices with custom configuration
//! - **InsertBatch**: Efficient batch vector insertion
//! - **InsertStream**: Streaming insertion for large datasets
//! - **Search**: K-nearest neighbor search
//! - **SearchBatch**: Batch search for multiple queries
//!
//! ## Usage
//!
//! ```bash
//! # Start the gRPC server
//! toondb-grpc-server --port 50051
//!
//! # From Python client
//! import grpc
//! from toondb.proto import toondb_pb2, toondb_pb2_grpc
//!
//! channel = grpc.insecure_channel('localhost:50051')
//! stub = toondb_pb2_grpc.VectorIndexServiceStub(channel)
//!
//! # Create index
//! response = stub.CreateIndex(toondb_pb2.CreateIndexRequest(
//!     name="embeddings",
//!     dimension=768,
//! ))
//! ```

pub mod proto {
    // Include generated protobuf code
    tonic::include_proto!("toondb.vector.v1");
}

pub mod server;
pub mod error;

pub use server::VectorIndexServer;
pub use error::GrpcError;
