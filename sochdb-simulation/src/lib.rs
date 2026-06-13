//! # SochDB Simulation Environment
//!
//! Models the full SochDB stack in **standalone** (embedded) and **distributed**
//! (gRPC cluster) topologies. Compares simulated performance against expected
//! scores from `sochdb-bench`, SLOs, and retrieval benchmarks.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    sochdb-simulation                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Scenarios ──► Engine ──► Component Models ──► OpResults    │
//! │       │                              ▲                       │
//! │       └── Topology (standalone / distributed)                │
//! │                                                              │
//! │  Expected Targets (bench + SLO + retrieval) ──► Scorer       │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod calibration;
pub mod component;
pub mod engine;
pub mod expected;
pub mod release;
pub mod report;
pub mod scenario;
pub mod score;
pub mod topology;

pub use component::{Component, SimEnvironment};
pub use engine::{OpResult, ScenarioResult, SimulationEngine};
pub use expected::ExpectedStore;
pub use release::validator::{GateResult, GateStatus, ReleaseScorecard, ReleaseValidator};
pub use scenario::Scenario;
pub use score::{Grade, Scorecard, Scorer};
pub use topology::{Operation, Topology};
