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

//! Error types for gRPC service

use thiserror::Error;
use tonic::Status;

/// Errors that can occur in the gRPC service
#[derive(Error, Debug)]
pub enum GrpcError {
    #[error("Index not found: {0}")]
    IndexNotFound(String),
    
    #[error("Index already exists: {0}")]
    IndexAlreadyExists(String),
    
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("HNSW error: {0}")]
    HnswError(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<GrpcError> for Status {
    fn from(err: GrpcError) -> Self {
        match err {
            GrpcError::IndexNotFound(name) => {
                Status::not_found(format!("Index not found: {}", name))
            }
            GrpcError::IndexAlreadyExists(name) => {
                Status::already_exists(format!("Index already exists: {}", name))
            }
            GrpcError::DimensionMismatch { expected, actual } => {
                Status::invalid_argument(format!(
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                ))
            }
            GrpcError::InvalidRequest(msg) => Status::invalid_argument(msg),
            GrpcError::HnswError(msg) => Status::internal(format!("HNSW error: {}", msg)),
            GrpcError::Internal(msg) => Status::internal(msg),
        }
    }
}
