// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! Prometheus Metrics HTTP Server (Task 8)
//!
//! Serves a `/metrics` endpoint for Prometheus scraping. Runs as a
//! lightweight HTTP server alongside the main gRPC server.
//!
//! ## Architecture
//!
//! ```text
//!   Prometheus Scraper
//!         │
//!    GET /metrics
//!         │
//!    ┌────▼────────────────────┐
//!    │ tiny_http server (:9090)│
//!    │                         │
//!    │  ┌───────────────────┐  │
//!    │  │ prometheus crate  │  │
//!    │  │ default registry  │  │
//!    │  │                   │  │
//!    │  │ HNSW metrics ◄───┼──┼── sochdb-index/metrics.rs
//!    │  │ gRPC metrics ◄───┼──┼── this module
//!    │  │ DB   metrics ◄───┼──┼── this module
//!    │  └───────────────────┘  │
//!    └─────────────────────────┘
//! ```
//!
//! ## Endpoints
//!
//! - `GET /metrics` — Prometheus text format
//! - `GET /health` — Simple health check (200 OK)
//! - All other paths — 404
//!
//! ## Usage
//!
//! ```rust,no_run
//! use sochdb_grpc::metrics_server;
//!
//! // Spawns the HTTP server in a background thread
//! let handle = metrics_server::start(9090);
//! ```

use lazy_static::lazy_static;
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramVec, TextEncoder,
    register_counter, register_counter_vec, register_gauge, register_gauge_vec,
    register_histogram, register_histogram_vec,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

// =============================================================================
// gRPC Service Metrics
// =============================================================================

lazy_static! {
    // --- Request counters ---

    /// Total gRPC requests by service and method
    pub static ref GRPC_REQUESTS_TOTAL: CounterVec = register_counter_vec!(
        "sochdb_grpc_requests_total",
        "Total gRPC requests by service and method",
        &["service", "method"]
    )
    .unwrap();

    /// Total gRPC errors by service, method, and error code
    pub static ref GRPC_ERRORS_TOTAL: CounterVec = register_counter_vec!(
        "sochdb_grpc_errors_total",
        "Total gRPC errors by service, method, and code",
        &["service", "method", "code"]
    )
    .unwrap();

    /// gRPC request latency by service and method
    pub static ref GRPC_LATENCY: HistogramVec = register_histogram_vec!(
        "sochdb_grpc_request_duration_seconds",
        "gRPC request latency in seconds",
        &["service", "method"],
        vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    )
    .unwrap();

    // --- Connection metrics ---

    /// Current active gRPC connections
    pub static ref GRPC_ACTIVE_CONNECTIONS: Gauge = register_gauge!(
        "sochdb_grpc_active_connections",
        "Current number of active gRPC connections"
    )
    .unwrap();

    // --- Database metrics ---

    /// Total SQL queries executed
    pub static ref SQL_QUERIES_TOTAL: CounterVec = register_counter_vec!(
        "sochdb_sql_queries_total",
        "Total SQL queries by statement type",
        &["statement_type"]
    )
    .unwrap();

    /// SQL query latency
    pub static ref SQL_QUERY_LATENCY: HistogramVec = register_histogram_vec!(
        "sochdb_sql_query_duration_seconds",
        "SQL query latency in seconds",
        &["statement_type"],
        vec![0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 30.0]
    )
    .unwrap();

    /// Total transactions by outcome
    pub static ref TXN_TOTAL: CounterVec = register_counter_vec!(
        "sochdb_transactions_total",
        "Total transactions by outcome",
        &["outcome"]
    )
    .unwrap();

    // --- Storage metrics ---

    /// Current number of tables
    pub static ref TABLES_COUNT: Gauge = register_gauge!(
        "sochdb_tables_count",
        "Current number of tables in the database"
    )
    .unwrap();

    /// Storage bytes on disk
    pub static ref STORAGE_BYTES: Gauge = register_gauge!(
        "sochdb_storage_bytes",
        "Approximate storage size in bytes"
    )
    .unwrap();

    /// WAL size in bytes
    pub static ref WAL_BYTES: Gauge = register_gauge!(
        "sochdb_wal_bytes",
        "WAL segment size in bytes"
    )
    .unwrap();

    /// WAL writes per second
    pub static ref WAL_WRITES_TOTAL: Counter = register_counter!(
        "sochdb_wal_writes_total",
        "Total WAL write operations"
    )
    .unwrap();

    /// WAL fsync operations
    pub static ref WAL_FSYNC_TOTAL: Counter = register_counter!(
        "sochdb_wal_fsync_total",
        "Total WAL fsync operations"
    )
    .unwrap();

    // --- Cache metrics ---

    /// Semantic cache hit/miss
    pub static ref CACHE_OPS: CounterVec = register_counter_vec!(
        "sochdb_cache_operations_total",
        "Cache operations by result",
        &["result"]
    )
    .unwrap();

    // --- Process metrics ---

    /// Uptime in seconds
    pub static ref UPTIME_SECONDS: Gauge = register_gauge!(
        "sochdb_uptime_seconds",
        "Server uptime in seconds"
    )
    .unwrap();

    /// Version info (constant gauge with labels)
    pub static ref BUILD_INFO: GaugeVec = register_gauge_vec!(
        "sochdb_build_info",
        "Build information",
        &["version", "rustc"]
    )
    .unwrap();
}

// =============================================================================
// Metrics Recording Helpers
// =============================================================================

/// Record a gRPC request (call at start of each RPC handler)
pub fn record_grpc_request(service: &str, method: &str) {
    GRPC_REQUESTS_TOTAL
        .with_label_values(&[service, method])
        .inc();
}

/// Record a gRPC error
pub fn record_grpc_error(service: &str, method: &str, code: &str) {
    GRPC_ERRORS_TOTAL
        .with_label_values(&[service, method, code])
        .inc();
}

/// Start a gRPC latency timer. The returned guard records on drop.
pub fn start_grpc_timer(service: &str, method: &str) -> prometheus::HistogramTimer {
    GRPC_LATENCY
        .with_label_values(&[service, method])
        .start_timer()
}

/// Record a SQL query execution
pub fn record_sql_query(statement_type: &str, duration_secs: f64) {
    SQL_QUERIES_TOTAL
        .with_label_values(&[statement_type])
        .inc();
    SQL_QUERY_LATENCY
        .with_label_values(&[statement_type])
        .observe(duration_secs);
}

/// Record a transaction outcome (committed / rolled_back / conflict)
pub fn record_transaction(outcome: &str) {
    TXN_TOTAL.with_label_values(&[outcome]).inc();
}

/// Record a cache operation (hit / miss)
pub fn record_cache_op(result: &str) {
    CACHE_OPS.with_label_values(&[result]).inc();
}

// =============================================================================
// HTTP Server
// =============================================================================

/// Handle for the metrics HTTP server thread
pub struct MetricsServerHandle {
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MetricsServerHandle {
    /// Signal the server to shut down
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Drop for MetricsServerHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Don't join — the blocking accept() loop will see shutdown on next
        // request or timeout.
    }
}

/// Start the Prometheus metrics HTTP server on the given port.
///
/// Returns a handle that can be used to shut down the server.
/// The server runs in a background OS thread (not a tokio task)
/// to avoid blocking the async runtime.
pub fn start(port: u16) -> MetricsServerHandle {
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    // Set build info gauge
    BUILD_INFO
        .with_label_values(&[env!("CARGO_PKG_VERSION"), "stable"])
        .set(1.0);

    let thread = thread::Builder::new()
        .name("sochdb-metrics-http".to_string())
        .spawn(move || {
            run_server(port, shutdown_clone);
        })
        .expect("failed to spawn metrics HTTP thread");

    MetricsServerHandle {
        shutdown,
        thread: Some(thread),
    }
}

fn run_server(port: u16, shutdown: Arc<AtomicBool>) {
    let addr = format!("0.0.0.0:{}", port);
    let server = match tiny_http::Server::http(&addr) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to start metrics HTTP server on {}: {}", addr, e);
            return;
        }
    };

    tracing::info!("Prometheus metrics server listening on http://{}/metrics", addr);

    // Process requests until shutdown
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Use recv_timeout to check shutdown periodically
        let request = match server.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(Some(req)) => req,
            Ok(None) => continue, // timeout — loop to check shutdown
            Err(e) => {
                if !shutdown.load(Ordering::Relaxed) {
                    tracing::warn!("Metrics server recv error: {}", e);
                }
                continue;
            }
        };

        let path = request.url().to_string();
        let method = request.method().to_string();

        match (method.as_str(), path.as_str()) {
            ("GET", "/metrics") => handle_metrics(request),
            ("GET", "/health") => handle_health(request),
            _ => handle_not_found(request),
        }
    }

    tracing::info!("Metrics HTTP server shut down");
}

fn handle_metrics(request: tiny_http::Request) {
    // Update uptime
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    UPTIME_SECONDS.set(start.elapsed().as_secs_f64());

    // Gather all metrics from the default prometheus registry.
    // This automatically includes:
    //   - HNSW metrics (registered in sochdb-index/src/metrics.rs)
    //   - gRPC metrics (registered above)
    //   - DB/storage/WAL metrics (registered above)
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::with_capacity(8192);

    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        tracing::error!("Failed to encode metrics: {}", e);
        let response = tiny_http::Response::from_string("Internal Server Error\n")
            .with_status_code(500);
        let _ = request.respond(response);
        return;
    }

    let response = tiny_http::Response::from_data(buffer)
        .with_header(
            tiny_http::Header::from_bytes(
                &b"Content-Type"[..],
                &b"text/plain; version=0.0.4; charset=utf-8"[..],
            )
            .unwrap(),
        )
        .with_status_code(200);

    let _ = request.respond(response);
}

fn handle_health(request: tiny_http::Request) {
    let response = tiny_http::Response::from_string("OK\n").with_status_code(200);
    let _ = request.respond(response);
}

fn handle_not_found(request: tiny_http::Request) {
    let body = "404 Not Found\n\nAvailable endpoints:\n  GET /metrics\n  GET /health\n";
    let response = tiny_http::Response::from_string(body).with_status_code(404);
    let _ = request.respond(response);
}

// =============================================================================
// ObservabilityExtension Implementation (bridges kernel plugin API → Prometheus)
// =============================================================================

/// Prometheus-backed implementation of the kernel's `ObservabilityExtension` trait.
///
/// This bridges the generic kernel plugin metrics API to concrete Prometheus
/// counters/gauges/histograms. Install via:
/// ```ignore
/// kernel.register_extension(Box::new(PrometheusObservability));
/// ```
pub struct PrometheusObservability;

impl PrometheusObservability {
    /// Record counter increment using the Prometheus default registry.
    pub fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        // For well-known metrics, route to the static lazy_static counters
        match name {
            "grpc_requests" => {
                if let (Some(svc), Some(method)) = (
                    labels.iter().find(|(k, _)| *k == "service").map(|(_, v)| *v),
                    labels.iter().find(|(k, _)| *k == "method").map(|(_, v)| *v),
                ) {
                    GRPC_REQUESTS_TOTAL
                        .with_label_values(&[svc, method])
                        .inc_by(value as f64);
                }
            }
            "wal_writes" => WAL_WRITES_TOTAL.inc_by(value as f64),
            "wal_fsync" => WAL_FSYNC_TOTAL.inc_by(value as f64),
            _ => {
                // Dynamic counter — log a trace for unknown metrics
                tracing::trace!("Unknown counter: {} += {}", name, value);
            }
        }
    }

    /// Record gauge value using the Prometheus default registry.
    pub fn gauge_set(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
        match name {
            "tables_count" => TABLES_COUNT.set(value),
            "storage_bytes" => STORAGE_BYTES.set(value),
            "wal_bytes" => WAL_BYTES.set(value),
            "active_connections" => GRPC_ACTIVE_CONNECTIONS.set(value),
            _ => {
                tracing::trace!("Unknown gauge: {} = {}", name, value);
            }
        }
    }

    /// Record histogram observation using the Prometheus default registry.
    pub fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        match name {
            "grpc_latency" => {
                if let (Some(svc), Some(method)) = (
                    labels.iter().find(|(k, _)| *k == "service").map(|(_, v)| *v),
                    labels.iter().find(|(k, _)| *k == "method").map(|(_, v)| *v),
                ) {
                    GRPC_LATENCY
                        .with_label_values(&[svc, method])
                        .observe(value);
                }
            }
            "sql_query_latency" => {
                if let Some(stmt) = labels
                    .iter()
                    .find(|(k, _)| *k == "statement_type")
                    .map(|(_, v)| *v)
                {
                    SQL_QUERY_LATENCY
                        .with_label_values(&[stmt])
                        .observe(value);
                }
            }
            _ => {
                tracing::trace!("Unknown histogram: {} = {}", name, value);
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_grpc_request() {
        record_grpc_request("VectorIndex", "Search");
        record_grpc_request("VectorIndex", "Search");
        record_grpc_request("VectorIndex", "Insert");

        let val = GRPC_REQUESTS_TOTAL
            .with_label_values(&["VectorIndex", "Search"])
            .get();
        assert!(val >= 2.0);
    }

    #[test]
    fn test_record_grpc_error() {
        record_grpc_error("VectorIndex", "Search", "NOT_FOUND");
        let val = GRPC_ERRORS_TOTAL
            .with_label_values(&["VectorIndex", "Search", "NOT_FOUND"])
            .get();
        assert!(val >= 1.0);
    }

    #[test]
    fn test_record_sql_query() {
        record_sql_query("SELECT", 0.005);
        record_sql_query("INSERT", 0.001);

        let val = SQL_QUERIES_TOTAL
            .with_label_values(&["SELECT"])
            .get();
        assert!(val >= 1.0);
    }

    #[test]
    fn test_record_transaction() {
        record_transaction("committed");
        record_transaction("rolled_back");

        let committed = TXN_TOTAL.with_label_values(&["committed"]).get();
        assert!(committed >= 1.0);
    }

    #[test]
    fn test_record_cache_op() {
        record_cache_op("hit");
        record_cache_op("miss");

        let hits = CACHE_OPS.with_label_values(&["hit"]).get();
        assert!(hits >= 1.0);
    }

    #[test]
    fn test_prometheus_observability_counter() {
        let obs = PrometheusObservability;
        obs.counter_inc("wal_writes", 5, &[]);
        let val = WAL_WRITES_TOTAL.get();
        assert!(val >= 5.0);
    }

    #[test]
    fn test_prometheus_observability_gauge() {
        let obs = PrometheusObservability;
        obs.gauge_set("tables_count", 42.0, &[]);
        assert!((TABLES_COUNT.get() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_info() {
        // Trigger lazy_static initialization
        BUILD_INFO
            .with_label_values(&[env!("CARGO_PKG_VERSION"), "stable"])
            .set(1.0);

        let families = prometheus::gather();
        let build = families
            .iter()
            .find(|f| f.get_name() == "sochdb_build_info");
        assert!(build.is_some());
    }

    #[test]
    fn test_metrics_encoding() {
        // Force some metrics to be recorded
        record_grpc_request("TestSvc", "TestMethod");
        UPTIME_SECONDS.set(123.0);

        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("sochdb_grpc_requests_total"));
        assert!(output.contains("sochdb_uptime_seconds"));
    }
}
