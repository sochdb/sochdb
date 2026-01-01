//! Query execution pipeline.
//!
//! Implements the full query flow: rotate → RDF → BPS → union → filter → rerank → verify

pub mod engine;
pub mod controller;

pub use engine::QueryEngine;
pub use controller::AdaptiveController;
