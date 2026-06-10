//! Query execution pipeline.
//!
//! Implements the full query flow: rotate → RDF → BPS → union → filter → rerank → verify

pub mod controller;
pub mod engine;

pub use controller::AdaptiveController;
pub use engine::QueryEngine;
