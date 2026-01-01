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

//! ToonDB WASM - Vector Index for Browser JavaScript
//!
//! This crate provides a WebAssembly build of ToonDB's HNSW vector index
//! for use in browser JavaScript applications.
//!
//! ## Features
//!
//! - Pure Rust implementation compiled to WASM
//! - Zero network dependencies - runs entirely in browser
//! - Efficient typed array interop with JavaScript
//! - Suitable for client-side embedding search, recommendations, etc.
//!
//! ## Usage (JavaScript/TypeScript)
//!
//! ```javascript
//! import init, { WasmVectorIndex } from 'toondb-wasm';
//!
//! async function main() {
//!   // Initialize WASM module
//!   await init();
//!
//!   // Create index with dimension=768, M=16, ef_construction=100
//!   const index = new WasmVectorIndex(768, 16, 100);
//!
//!   // Insert vectors (Float32Array)
//!   const ids = BigUint64Array.from([1n, 2n, 3n]);
//!   const vectors = new Float32Array(768 * 3);
//!   // ... fill vectors ...
//!   
//!   const inserted = index.insertBatch(ids, vectors);
//!   console.log(`Inserted ${inserted} vectors`);
//!
//!   // Search
//!   const query = new Float32Array(768);
//!   const results = index.search(query, 10);
//!   console.log('Results:', results);
//! }
//! ```
//!
//! ## Build Instructions
//!
//! ```bash
//! # Install wasm-pack
//! cargo install wasm-pack
//!
//! # Build for web
//! cd toondb-wasm
//! wasm-pack build --target web --release
//!
//! # Build for bundler (webpack, rollup, etc.)
//! wasm-pack build --target bundler --release
//! ```

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

mod hnsw_core;

pub use hnsw_core::WasmVectorIndex;

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Search result returned from vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct SearchResult {
    /// Vector ID
    pub id: u64,
    /// Distance to query vector
    pub distance: f32,
}

#[wasm_bindgen]
impl SearchResult {
    #[wasm_bindgen(constructor)]
    pub fn new(id: u64, distance: f32) -> Self {
        Self { id, distance }
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct IndexStats {
    /// Number of vectors in index
    pub num_vectors: u32,
    /// Vector dimension
    pub dimension: u32,
    /// Maximum layer in HNSW graph
    pub max_layer: u32,
    /// Average connections per node
    pub avg_connections: f32,
}

#[wasm_bindgen]
impl IndexStats {
    #[wasm_bindgen(getter)]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn console_log(s: &str) {
    web_sys::console::log_1(&JsValue::from_str(s));
}

/// Get current performance timestamp (high resolution)
#[wasm_bindgen]
pub fn performance_now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}
