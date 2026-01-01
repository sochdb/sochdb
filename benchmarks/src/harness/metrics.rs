// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! HDRHistogram-based metrics collection for accurate percentile measurement.

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// High-precision latency histogram using HDRHistogram.
///
/// Provides accurate percentile measurements (p50, p95, p99, p99.9)
/// without the sorting overhead of naive implementations.
pub struct LatencyHistogram {
    histogram: Histogram<u64>,
    start_time: Option<Instant>,
}

impl LatencyHistogram {
    /// Create a new histogram with nanosecond precision.
    ///
    /// # Arguments
    /// * `max_latency_ns` - Maximum expected latency in nanoseconds (default: 60s)
    /// * `sigfigs` - Significant figures of precision (1-5, default: 3)
    pub fn new() -> Self {
        // 60 second max, 3 significant figures
        Self::with_config(60_000_000_000, 3)
    }

    /// Create with custom configuration.
    pub fn with_config(max_latency_ns: u64, sigfigs: u8) -> Self {
        let histogram = Histogram::new_with_max(max_latency_ns, sigfigs)
            .expect("Failed to create histogram");
        Self {
            histogram,
            start_time: None,
        }
    }

    /// Start timing an operation.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop timing and record the elapsed duration.
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            let elapsed = start.elapsed();
            self.record(elapsed);
        }
    }

    /// Record a latency value directly.
    pub fn record(&mut self, latency: Duration) {
        let nanos = latency.as_nanos() as u64;
        // Clamp to max if needed
        let nanos = nanos.min(self.histogram.high());
        let _ = self.histogram.record(nanos);
    }

    /// Record a latency in nanoseconds.
    pub fn record_ns(&mut self, latency_ns: u64) {
        let latency_ns = latency_ns.min(self.histogram.high());
        let _ = self.histogram.record(latency_ns);
    }

    /// Get a percentile value as Duration.
    pub fn percentile(&self, p: f64) -> Duration {
        let nanos = self.histogram.value_at_percentile(p);
        Duration::from_nanos(nanos)
    }

    /// Get p50 latency.
    pub fn p50(&self) -> Duration {
        self.percentile(50.0)
    }

    /// Get p95 latency.
    pub fn p95(&self) -> Duration {
        self.percentile(95.0)
    }

    /// Get p99 latency.
    pub fn p99(&self) -> Duration {
        self.percentile(99.0)
    }

    /// Get p99.9 latency.
    pub fn p999(&self) -> Duration {
        self.percentile(99.9)
    }

    /// Get mean latency.
    pub fn mean(&self) -> Duration {
        Duration::from_nanos(self.histogram.mean() as u64)
    }

    /// Get min latency.
    pub fn min(&self) -> Duration {
        Duration::from_nanos(self.histogram.min())
    }

    /// Get max latency.
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.histogram.max())
    }

    /// Get total count of recorded values.
    pub fn count(&self) -> u64 {
        self.histogram.len()
    }

    /// Export summary as a serializable struct.
    pub fn summary(&self) -> LatencySummary {
        LatencySummary {
            count: self.count(),
            min_ns: self.min().as_nanos() as u64,
            max_ns: self.max().as_nanos() as u64,
            mean_ns: self.mean().as_nanos() as u64,
            p50_ns: self.p50().as_nanos() as u64,
            p95_ns: self.p95().as_nanos() as u64,
            p99_ns: self.p99().as_nanos() as u64,
            p999_ns: self.p999().as_nanos() as u64,
        }
    }

    /// Reset the histogram.
    pub fn reset(&mut self) {
        self.histogram.reset();
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable latency summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySummary {
    pub count: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
}

impl LatencySummary {
    /// Get p50 in milliseconds.
    pub fn p50_ms(&self) -> f64 {
        self.p50_ns as f64 / 1_000_000.0
    }

    /// Get p95 in milliseconds.
    pub fn p95_ms(&self) -> f64 {
        self.p95_ns as f64 / 1_000_000.0
    }

    /// Get p99 in milliseconds.
    pub fn p99_ms(&self) -> f64 {
        self.p99_ns as f64 / 1_000_000.0
    }

    /// Get p99.9 in milliseconds.
    pub fn p999_ms(&self) -> f64 {
        self.p999_ns as f64 / 1_000_000.0
    }

    /// Get mean in milliseconds.
    pub fn mean_ms(&self) -> f64 {
        self.mean_ns as f64 / 1_000_000.0
    }
}

/// Comprehensive metrics collector for benchmark runs.
pub struct MetricsCollector {
    /// Latency histogram for query operations
    pub query_latency: LatencyHistogram,
    /// Latency histogram for insert operations
    pub insert_latency: LatencyHistogram,
    /// Total queries executed
    pub queries_executed: u64,
    /// Total inserts executed
    pub inserts_executed: u64,
    /// Errors encountered
    pub errors: u64,
    /// Timeouts encountered
    pub timeouts: u64,
    /// Start time of the benchmark
    start_time: Option<Instant>,
    /// End time of the benchmark
    end_time: Option<Instant>,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            query_latency: LatencyHistogram::new(),
            insert_latency: LatencyHistogram::new(),
            queries_executed: 0,
            inserts_executed: 0,
            errors: 0,
            timeouts: 0,
            start_time: None,
            end_time: None,
        }
    }

    /// Start the benchmark timer.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop the benchmark timer.
    pub fn stop(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Record a query with its latency.
    pub fn record_query(&mut self, latency: Duration) {
        self.query_latency.record(latency);
        self.queries_executed += 1;
    }

    /// Record an insert with its latency.
    pub fn record_insert(&mut self, latency: Duration) {
        self.insert_latency.record(latency);
        self.inserts_executed += 1;
    }

    /// Record an error.
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Record a timeout.
    pub fn record_timeout(&mut self) {
        self.timeouts += 1;
    }

    /// Get total elapsed time.
    pub fn elapsed(&self) -> Duration {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => end.duration_since(start),
            (Some(start), None) => start.elapsed(),
            _ => Duration::ZERO,
        }
    }

    /// Calculate queries per second.
    pub fn qps(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 {
            self.queries_executed as f64 / secs
        } else {
            0.0
        }
    }

    /// Calculate inserts per second.
    pub fn insert_rate(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 {
            self.inserts_executed as f64 / secs
        } else {
            0.0
        }
    }

    /// Get error rate as percentage.
    pub fn error_rate(&self) -> f64 {
        let total = self.queries_executed + self.inserts_executed;
        if total > 0 {
            (self.errors as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Export as a serializable summary.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            duration_s: self.elapsed().as_secs_f64(),
            queries_executed: self.queries_executed,
            inserts_executed: self.inserts_executed,
            qps: self.qps(),
            insert_rate: self.insert_rate(),
            errors: self.errors,
            timeouts: self.timeouts,
            error_rate_pct: self.error_rate(),
            query_latency: self.query_latency.summary(),
            insert_latency: self.insert_latency.summary(),
        }
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.query_latency.reset();
        self.insert_latency.reset();
        self.queries_executed = 0;
        self.inserts_executed = 0;
        self.errors = 0;
        self.timeouts = 0;
        self.start_time = None;
        self.end_time = None;
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable metrics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub duration_s: f64,
    pub queries_executed: u64,
    pub inserts_executed: u64,
    pub qps: f64,
    pub insert_rate: f64,
    pub errors: u64,
    pub timeouts: u64,
    pub error_rate_pct: f64,
    pub query_latency: LatencySummary,
    pub insert_latency: LatencySummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_histogram() {
        let mut hist = LatencyHistogram::new();
        
        // Record some values
        for i in 1..=100 {
            hist.record(Duration::from_micros(i * 10));
        }

        assert_eq!(hist.count(), 100);
        assert!(hist.p50().as_micros() > 0);
        assert!(hist.p99().as_micros() > hist.p50().as_micros());
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        collector.start();

        for i in 0..100 {
            collector.record_query(Duration::from_micros(i * 10 + 100));
        }

        collector.stop();

        let summary = collector.summary();
        assert_eq!(summary.queries_executed, 100);
        assert!(summary.qps > 0.0);
    }
}
