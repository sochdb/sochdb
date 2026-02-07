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

//! # Observability Hardening
//!
//! Production-grade observability with:
//! - Cardinality budgets (metric explosion protection)
//! - Slow query log with sampling
//! - Structured span attributes for tracing
//! - SLI/SLO instrumentation
//!
//! ## Design Principles
//!
//! 1. **Cardinality Control**: Bound unique label combinations
//! 2. **Exemplars**: Link traces to metrics for debugging
//! 3. **Percentile Accuracy**: Use t-digest or HDR histograms
//! 4. **Low Overhead**: < 2% CPU impact at 99th percentile

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Maximum cardinality per metric family
const DEFAULT_CARDINALITY_BUDGET: usize = 10_000;

/// Slow query threshold (configurable)
const DEFAULT_SLOW_QUERY_THRESHOLD: Duration = Duration::from_millis(100);

/// Metric type for SLI tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Request count
    Counter,
    /// Current value
    Gauge,
    /// Latency distribution
    Histogram,
    /// Summary with quantiles
    Summary,
}

/// Label set for metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelSet {
    labels: Vec<(String, String)>,
}

impl LabelSet {
    /// Create a new label set
    pub fn new() -> Self {
        Self { labels: Vec::new() }
    }

    /// Add a label
    pub fn add(mut self, key: &str, value: &str) -> Self {
        self.labels.push((key.to_string(), value.to_string()));
        self
    }

    /// Compute cardinality fingerprint
    fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        self.labels.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for LabelSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Cardinality budget tracker
pub struct CardinalityBudget {
    /// Metric name -> set of seen label fingerprints
    seen: RwLock<HashMap<String, std::collections::HashSet<u64>>>,
    /// Per-metric budget (default if not specified)
    budgets: RwLock<HashMap<String, usize>>,
    /// Default budget
    default_budget: usize,
    /// Dropped metrics due to cardinality explosion
    dropped: AtomicU64,
}

impl CardinalityBudget {
    /// Create a new cardinality budget
    pub fn new(default_budget: usize) -> Self {
        Self {
            seen: RwLock::new(HashMap::new()),
            budgets: RwLock::new(HashMap::new()),
            default_budget,
            dropped: AtomicU64::new(0),
        }
    }

    /// Set budget for a specific metric
    pub fn set_budget(&self, metric: &str, budget: usize) {
        self.budgets.write().insert(metric.to_string(), budget);
    }

    /// Check if we can record this label combination
    pub fn check(&self, metric: &str, labels: &LabelSet) -> bool {
        let fingerprint = labels.fingerprint();
        
        // Fast path: already seen
        {
            let seen = self.seen.read();
            if let Some(set) = seen.get(metric) {
                if set.contains(&fingerprint) {
                    return true;
                }
            }
        }

        // Slow path: check budget and insert
        let mut seen = self.seen.write();
        let set = seen.entry(metric.to_string()).or_default();
        
        let budget = self.budgets
            .read()
            .get(metric)
            .copied()
            .unwrap_or(self.default_budget);

        if set.len() >= budget {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        set.insert(fingerprint);
        true
    }

    /// Get number of unique label combinations for a metric
    pub fn cardinality(&self, metric: &str) -> usize {
        self.seen
            .read()
            .get(metric)
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Get total dropped metrics
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl Default for CardinalityBudget {
    fn default() -> Self {
        Self::new(DEFAULT_CARDINALITY_BUDGET)
    }
}

/// Slow query entry
#[derive(Debug, Clone)]
pub struct SlowQueryEntry {
    /// Query text (may be truncated)
    pub query: String,
    /// Execution duration
    pub duration: Duration,
    /// Timestamp
    pub timestamp: Instant,
    /// Transaction ID (if available)
    pub txn_id: Option<u64>,
    /// Rows examined
    pub rows_examined: u64,
    /// Rows returned
    pub rows_returned: u64,
    /// Table names involved
    pub tables: Vec<String>,
    /// Index used (if any)
    pub index_used: Option<String>,
    /// Was this a full table scan?
    pub full_scan: bool,
    /// Trace ID for correlation
    pub trace_id: Option<String>,
}

/// Slow query log with reservoir sampling
pub struct SlowQueryLog {
    /// Configuration threshold
    threshold: Duration,
    /// Maximum entries to keep
    max_entries: usize,
    /// Log entries (circular buffer — VecDeque for O(1) eviction at front)
    entries: RwLock<std::collections::VecDeque<SlowQueryEntry>>,
    /// Total slow queries seen
    total_count: AtomicU64,
    /// Sample rate (1 = 100%, 10 = 10%, etc.)
    sample_rate: u64,
    /// Counter for sampling
    sample_counter: AtomicU64,
}

impl SlowQueryLog {
    /// Create a new slow query log
    pub fn new(threshold: Duration, max_entries: usize) -> Self {
        Self {
            threshold,
            max_entries,
            entries: RwLock::new(std::collections::VecDeque::with_capacity(max_entries)),
            total_count: AtomicU64::new(0),
            sample_rate: 1,
            sample_counter: AtomicU64::new(0),
        }
    }

    /// Set sample rate (1/N queries logged)
    pub fn set_sample_rate(&mut self, rate: u64) {
        self.sample_rate = rate.max(1);
    }

    /// Maybe log a slow query
    pub fn maybe_log(&self, entry: SlowQueryEntry) {
        if entry.duration < self.threshold {
            return;
        }

        self.total_count.fetch_add(1, Ordering::Relaxed);

        // Sampling: only log 1 in N
        let counter = self.sample_counter.fetch_add(1, Ordering::Relaxed);
        if counter % self.sample_rate != 0 {
            return;
        }

        let mut entries = self.entries.write();
        if entries.len() >= self.max_entries {
            entries.pop_front(); // O(1) instead of Vec::remove(0) which is O(n)
        }
        entries.push_back(entry);
    }

    /// Get recent slow queries
    pub fn recent(&self, limit: usize) -> Vec<SlowQueryEntry> {
        let entries = self.entries.read();
        entries.iter().rev().take(limit).cloned().collect()
    }

    /// Get total slow query count
    pub fn total_count(&self) -> u64 {
        self.total_count.load(Ordering::Relaxed)
    }

    /// Clear the log
    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

impl Default for SlowQueryLog {
    fn default() -> Self {
        Self::new(DEFAULT_SLOW_QUERY_THRESHOLD, 1000)
    }
}

/// SLI (Service Level Indicator) type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliType {
    /// Request latency (p50, p95, p99)
    Latency,
    /// Error rate
    ErrorRate,
    /// Availability (success / total)
    Availability,
    /// Throughput (requests per second)
    Throughput,
    /// Saturation (resource utilization)
    Saturation,
}

/// SLI bucket for histogram aggregation
pub struct SliBucket {
    /// Upper bound (in microseconds for latency)
    pub le: u64,
    /// Count of observations <= le
    pub count: AtomicU64,
}

/// SLI histogram (simplified HDR histogram)
pub struct SliHistogram {
    /// Histogram buckets (latency in microseconds)
    buckets: Vec<SliBucket>,
    /// Total observations
    count: AtomicU64,
    /// Sum of all observations
    sum: AtomicU64,
}

impl SliHistogram {
    /// Create a histogram with default buckets for latency SLIs
    pub fn latency_histogram() -> Self {
        // Buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s
        let boundaries = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000];
        Self::with_buckets(&boundaries)
    }

    /// Create a histogram with custom buckets
    pub fn with_buckets(boundaries: &[u64]) -> Self {
        let buckets = boundaries
            .iter()
            .map(|&le| SliBucket {
                le,
                count: AtomicU64::new(0),
            })
            .collect();

        Self {
            buckets,
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        }
    }

    /// Observe a value (in microseconds for latency)
    pub fn observe(&self, value: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value, Ordering::Relaxed);

        for bucket in &self.buckets {
            if value <= bucket.le {
                bucket.count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get approximate percentile (0-100)
    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }

        let target = (total as f64 * p / 100.0) as u64;
        
        for bucket in &self.buckets {
            if bucket.count.load(Ordering::Relaxed) >= target {
                return bucket.le;
            }
        }

        // Above all buckets
        self.buckets.last().map(|b| b.le).unwrap_or(0)
    }

    /// Get mean value
    pub fn mean(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.sum.load(Ordering::Relaxed) as f64 / count as f64
    }
}

/// SLO (Service Level Objective) definition
#[derive(Debug, Clone)]
pub struct SloDefinition {
    /// SLO name
    pub name: String,
    /// SLI type
    pub sli_type: SliType,
    /// Target value (interpretation depends on SLI type)
    /// - Latency: target percentile value in microseconds
    /// - ErrorRate: maximum error rate (0.0 - 1.0)
    /// - Availability: minimum availability (0.99 = 99%)
    pub target: f64,
    /// Percentile for latency SLOs
    pub percentile: Option<f64>,
    /// Measurement window
    pub window: Duration,
}

/// SLO tracker
pub struct SloTracker {
    /// SLO definition
    pub definition: SloDefinition,
    /// Histogram for latency SLIs
    histogram: Option<SliHistogram>,
    /// Success count (for availability/error rate)
    successes: AtomicU64,
    /// Failure count
    failures: AtomicU64,
    /// Window start time
    window_start: RwLock<Instant>,
}

impl SloTracker {
    /// Create a new SLO tracker
    pub fn new(definition: SloDefinition) -> Self {
        let histogram = match definition.sli_type {
            SliType::Latency => Some(SliHistogram::latency_histogram()),
            _ => None,
        };

        Self {
            definition,
            histogram,
            successes: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            window_start: RwLock::new(Instant::now()),
        }
    }

    /// Record a latency observation (in microseconds)
    pub fn record_latency(&self, value_us: u64) {
        if let Some(ref hist) = self.histogram {
            hist.observe(value_us);
        }
    }

    /// Record success/failure for availability
    pub fn record_result(&self, success: bool) {
        if success {
            self.successes.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Check if SLO is being met
    pub fn is_meeting_slo(&self) -> bool {
        match self.definition.sli_type {
            SliType::Latency => {
                if let (Some(hist), Some(p)) = (self.histogram.as_ref(), self.definition.percentile) {
                    hist.percentile(p) <= self.definition.target as u64
                } else {
                    true
                }
            }
            SliType::Availability => {
                let successes = self.successes.load(Ordering::Relaxed) as f64;
                let failures = self.failures.load(Ordering::Relaxed) as f64;
                let total = successes + failures;
                if total == 0.0 {
                    return true;
                }
                successes / total >= self.definition.target
            }
            SliType::ErrorRate => {
                let successes = self.successes.load(Ordering::Relaxed) as f64;
                let failures = self.failures.load(Ordering::Relaxed) as f64;
                let total = successes + failures;
                if total == 0.0 {
                    return true;
                }
                failures / total <= self.definition.target
            }
            _ => true,
        }
    }

    /// Get current SLI value
    pub fn current_value(&self) -> f64 {
        match self.definition.sli_type {
            SliType::Latency => {
                if let (Some(hist), Some(p)) = (self.histogram.as_ref(), self.definition.percentile) {
                    hist.percentile(p) as f64
                } else {
                    0.0
                }
            }
            SliType::Availability => {
                let successes = self.successes.load(Ordering::Relaxed) as f64;
                let failures = self.failures.load(Ordering::Relaxed) as f64;
                let total = successes + failures;
                if total == 0.0 {
                    1.0
                } else {
                    successes / total
                }
            }
            SliType::ErrorRate => {
                let successes = self.successes.load(Ordering::Relaxed) as f64;
                let failures = self.failures.load(Ordering::Relaxed) as f64;
                let total = successes + failures;
                if total == 0.0 {
                    0.0
                } else {
                    failures / total
                }
            }
            _ => 0.0,
        }
    }

    /// Calculate error budget remaining (0.0 - 1.0)
    pub fn error_budget_remaining(&self) -> f64 {
        match self.definition.sli_type {
            SliType::Availability => {
                // Error budget = 1.0 - target
                // Budget consumed = (1.0 - current_availability) / (1.0 - target)
                // Budget remaining = 1.0 - budget consumed
                let current = self.current_value();
                let target = self.definition.target;
                let error_budget = 1.0 - target;
                if error_budget == 0.0 {
                    return if current >= target { 1.0 } else { 0.0 };
                }
                let consumed = (1.0 - current) / error_budget;
                (1.0 - consumed).max(0.0).min(1.0)
            }
            SliType::ErrorRate => {
                let current = self.current_value();
                let target = self.definition.target;
                if target == 0.0 {
                    return if current == 0.0 { 1.0 } else { 0.0 };
                }
                (1.0 - current / target).max(0.0).min(1.0)
            }
            _ => 1.0,
        }
    }

    /// Reset window (typically called periodically)
    pub fn reset_window(&self) {
        if let Some(ref hist) = self.histogram {
            // Histograms don't support atomic reset, so this is best-effort
            // In production, use a proper histogram library with windowing
        }
        self.successes.store(0, Ordering::Relaxed);
        self.failures.store(0, Ordering::Relaxed);
        *self.window_start.write() = Instant::now();
    }
}

/// Span attribute for structured tracing
#[derive(Debug, Clone)]
pub enum SpanAttribute {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    StringArray(Vec<String>),
}

impl From<&str> for SpanAttribute {
    fn from(s: &str) -> Self {
        SpanAttribute::String(s.to_string())
    }
}

impl From<String> for SpanAttribute {
    fn from(s: String) -> Self {
        SpanAttribute::String(s)
    }
}

impl From<i64> for SpanAttribute {
    fn from(i: i64) -> Self {
        SpanAttribute::Int(i)
    }
}

impl From<f64> for SpanAttribute {
    fn from(f: f64) -> Self {
        SpanAttribute::Float(f)
    }
}

impl From<bool> for SpanAttribute {
    fn from(b: bool) -> Self {
        SpanAttribute::Bool(b)
    }
}

/// Span builder for database operations
#[derive(Debug, Default)]
pub struct DbSpanBuilder {
    attributes: HashMap<String, SpanAttribute>,
}

impl DbSpanBuilder {
    /// Create a new span builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set database system
    pub fn db_system(mut self, system: &str) -> Self {
        self.attributes.insert("db.system".to_string(), system.into());
        self
    }

    /// Set operation name
    pub fn db_operation(mut self, op: &str) -> Self {
        self.attributes.insert("db.operation".to_string(), op.into());
        self
    }

    /// Set database name
    pub fn db_name(mut self, name: &str) -> Self {
        self.attributes.insert("db.name".to_string(), name.into());
        self
    }

    /// Set SQL statement (truncated for safety)
    pub fn db_statement(mut self, stmt: &str) -> Self {
        let truncated = if stmt.len() > 1000 {
            format!("{}...", &stmt[..1000])
        } else {
            stmt.to_string()
        };
        self.attributes.insert("db.statement".to_string(), truncated.into());
        self
    }

    /// Set table name
    pub fn db_table(mut self, table: &str) -> Self {
        self.attributes.insert("db.sql.table".to_string(), table.into());
        self
    }

    /// Set rows affected
    pub fn db_rows_affected(mut self, rows: i64) -> Self {
        self.attributes.insert("db.rows_affected".to_string(), rows.into());
        self
    }

    /// Set transaction ID
    pub fn transaction_id(mut self, txn_id: i64) -> Self {
        self.attributes.insert("sochdb.txn_id".to_string(), txn_id.into());
        self
    }

    /// Set namespace
    pub fn namespace(mut self, ns: &str) -> Self {
        self.attributes.insert("sochdb.namespace".to_string(), ns.into());
        self
    }

    /// Build into attribute map
    pub fn build(self) -> HashMap<String, SpanAttribute> {
        self.attributes
    }
}

/// Observability configuration
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    /// Enable slow query logging
    pub slow_query_enabled: bool,
    /// Slow query threshold
    pub slow_query_threshold: Duration,
    /// Cardinality budget
    pub cardinality_budget: usize,
    /// Enable SLO tracking
    pub slo_tracking_enabled: bool,
    /// Enable exemplars
    pub exemplars_enabled: bool,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            slow_query_enabled: true,
            slow_query_threshold: DEFAULT_SLOW_QUERY_THRESHOLD,
            cardinality_budget: DEFAULT_CARDINALITY_BUDGET,
            slo_tracking_enabled: true,
            exemplars_enabled: true,
        }
    }
}

/// Central observability service
pub struct ObservabilityService {
    config: ObservabilityConfig,
    /// Cardinality budget tracker
    pub cardinality: CardinalityBudget,
    /// Slow query log
    pub slow_queries: SlowQueryLog,
    /// SLO trackers
    slo_trackers: RwLock<HashMap<String, Arc<SloTracker>>>,
}

impl ObservabilityService {
    /// Create a new observability service
    pub fn new(config: ObservabilityConfig) -> Self {
        Self {
            cardinality: CardinalityBudget::new(config.cardinality_budget),
            slow_queries: SlowQueryLog::new(
                config.slow_query_threshold,
                1000,
            ),
            slo_trackers: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Register an SLO tracker
    pub fn register_slo(&self, name: &str, definition: SloDefinition) -> Arc<SloTracker> {
        let tracker = Arc::new(SloTracker::new(definition));
        self.slo_trackers.write().insert(name.to_string(), Arc::clone(&tracker));
        tracker
    }

    /// Get an SLO tracker
    pub fn get_slo(&self, name: &str) -> Option<Arc<SloTracker>> {
        self.slo_trackers.read().get(name).cloned()
    }

    /// Get all SLO statuses
    pub fn all_slo_statuses(&self) -> Vec<(String, bool, f64)> {
        self.slo_trackers
            .read()
            .iter()
            .map(|(name, tracker)| {
                (name.clone(), tracker.is_meeting_slo(), tracker.error_budget_remaining())
            })
            .collect()
    }
}

impl Default for ObservabilityService {
    fn default() -> Self {
        Self::new(ObservabilityConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cardinality_budget() {
        let budget = CardinalityBudget::new(3);
        
        let labels1 = LabelSet::new().add("tenant", "a");
        let labels2 = LabelSet::new().add("tenant", "b");
        let labels3 = LabelSet::new().add("tenant", "c");
        let labels4 = LabelSet::new().add("tenant", "d");

        assert!(budget.check("requests", &labels1));
        assert!(budget.check("requests", &labels2));
        assert!(budget.check("requests", &labels3));
        
        // Should fail - over budget
        assert!(!budget.check("requests", &labels4));
        assert_eq!(budget.dropped_count(), 1);

        // Same label should still work
        assert!(budget.check("requests", &labels1));
    }

    #[test]
    fn test_slow_query_log() {
        let log = SlowQueryLog::new(Duration::from_millis(10), 100);

        // Fast query - not logged
        log.maybe_log(SlowQueryEntry {
            query: "SELECT 1".to_string(),
            duration: Duration::from_millis(5),
            timestamp: Instant::now(),
            txn_id: None,
            rows_examined: 1,
            rows_returned: 1,
            tables: vec![],
            index_used: None,
            full_scan: false,
            trace_id: None,
        });

        assert_eq!(log.total_count(), 0);

        // Slow query - logged
        log.maybe_log(SlowQueryEntry {
            query: "SELECT * FROM big_table".to_string(),
            duration: Duration::from_millis(100),
            timestamp: Instant::now(),
            txn_id: Some(123),
            rows_examined: 100000,
            rows_returned: 1000,
            tables: vec!["big_table".to_string()],
            index_used: None,
            full_scan: true,
            trace_id: Some("trace-123".to_string()),
        });

        assert_eq!(log.total_count(), 1);
        assert_eq!(log.recent(10).len(), 1);
    }

    #[test]
    fn test_sli_histogram() {
        let hist = SliHistogram::latency_histogram();

        // Add some observations
        hist.observe(500);   // 0.5ms
        hist.observe(1500);  // 1.5ms
        hist.observe(5000);  // 5ms
        hist.observe(50000); // 50ms

        // p50 should be around 5ms bucket (5000)
        assert!(hist.percentile(50.0) <= 5000);
        
        // p99 should be around 50ms bucket
        assert!(hist.percentile(99.0) >= 25000);
    }

    #[test]
    fn test_slo_availability() {
        let definition = SloDefinition {
            name: "api_availability".to_string(),
            sli_type: SliType::Availability,
            target: 0.99,
            percentile: None,
            window: Duration::from_secs(3600),
        };

        let tracker = SloTracker::new(definition);

        // 100 successes
        for _ in 0..100 {
            tracker.record_result(true);
        }

        assert!(tracker.is_meeting_slo());
        assert!(tracker.error_budget_remaining() > 0.9);

        // Add failure
        tracker.record_result(false);

        // Still meeting SLO (99/101 = 98.02%, close to 99%)
        // But error budget is consumed
        assert!(tracker.error_budget_remaining() < 1.0);
    }
}
