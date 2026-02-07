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

//! # Admission Control with Explicit Cost Model
//!
//! Implements admission control to prevent unbounded queueing and ensure
//! stable p99 latency under load spikes. Uses multi-dimensional resource budgets
//! and fair queuing per tenant.
//!
//! ## Design Principles
//!
//! 1. **Cost Estimation Before Execution**: Estimate CPU, I/O, and memory costs
//!    before accepting a query.
//!
//! 2. **Multi-Dimensional Tokens**: Separate budgets for:
//!    - CPU tokens (compute-bound work)
//!    - Random IOPS tokens (point reads)
//!    - Sequential bandwidth tokens (scans)
//!    - Memory tokens (buffer pool pressure)
//!
//! 3. **Fair Queuing**: WFQ (Weighted Fair Queuing) or DRR (Deficit Round Robin)
//!    to prevent tenant starvation.
//!
//! 4. **Partial Results**: Opt-in only, with explicit recall degradation warning.
//!
//! ## Queueing Theory
//!
//! Stability requires: offered_load < service_capacity in each dimension.
//! Token buckets provide: rate limiting with burst handling.
//! WFQ provides: O(1) amortized scheduling with fairness guarantees.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

/// Resource dimension for admission control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceDimension {
    /// CPU compute tokens (microseconds of CPU time)
    Cpu,
    /// Random I/O operations (point reads)
    RandomIops,
    /// Sequential bandwidth (MB/s)
    SequentialBandwidth,
    /// Memory allocation (bytes)
    Memory,
    /// Concurrent connections
    Connections,
}

impl ResourceDimension {
    pub fn name(&self) -> &'static str {
        match self {
            ResourceDimension::Cpu => "cpu",
            ResourceDimension::RandomIops => "random_iops",
            ResourceDimension::SequentialBandwidth => "seq_bandwidth",
            ResourceDimension::Memory => "memory",
            ResourceDimension::Connections => "connections",
        }
    }
}

/// Estimated cost of an operation
#[derive(Debug, Clone, Default)]
pub struct OperationCost {
    /// Estimated CPU time (microseconds)
    pub cpu_us: u64,
    /// Estimated random I/O operations
    pub random_iops: u64,
    /// Estimated sequential read bytes
    pub sequential_bytes: u64,
    /// Estimated memory usage (bytes)
    pub memory_bytes: u64,
    /// Priority (higher = more important)
    pub priority: u32,
}

impl OperationCost {
    /// Create a zero-cost estimate (for very cheap operations)
    pub fn zero() -> Self {
        Self::default()
    }

    /// Create a point-read cost estimate
    pub fn point_read() -> Self {
        Self {
            cpu_us: 10,
            random_iops: 1,
            sequential_bytes: 0,
            memory_bytes: 4096,
            priority: 100,
        }
    }

    /// Create a scan cost estimate
    pub fn scan(rows: usize, row_bytes: usize) -> Self {
        Self {
            cpu_us: (rows * 5) as u64, // ~5us per row
            random_iops: 0,
            sequential_bytes: (rows * row_bytes) as u64,
            memory_bytes: (rows * row_bytes).min(64 * 1024 * 1024) as u64, // Cap at 64MB
            priority: 50,
        }
    }

    /// Create a vector search cost estimate
    pub fn vector_search(dimension: usize, ef_search: usize, candidates: usize) -> Self {
        // HNSW complexity: O(dimension * ef_search * log(n))
        let distance_calcs = ef_search * candidates;
        Self {
            cpu_us: (distance_calcs * dimension / 100) as u64, // ~100 dims per microsecond
            random_iops: (ef_search / 10).max(1) as u64, // Random node access
            sequential_bytes: 0,
            memory_bytes: (ef_search * dimension * 4) as u64, // Candidate vectors
            priority: 75,
        }
    }

    /// Create a write cost estimate
    pub fn write(bytes: usize) -> Self {
        Self {
            cpu_us: (bytes / 100).max(10) as u64,
            random_iops: 0,
            sequential_bytes: bytes as u64, // WAL write
            memory_bytes: bytes as u64,
            priority: 100,
        }
    }

    /// Total weighted cost for simple comparisons
    pub fn total_weighted_cost(&self) -> u64 {
        self.cpu_us + self.random_iops * 100 + self.sequential_bytes / 1024 + self.memory_bytes / 4096
    }
}

/// Token bucket for rate limiting
pub struct TokenBucket {
    /// Current tokens available
    tokens: AtomicI64,
    /// Maximum tokens (bucket capacity)
    capacity: i64,
    /// Token refill rate per second
    refill_rate: i64,
    /// Last refill timestamp (ms since epoch)
    last_refill: AtomicU64,
}

impl TokenBucket {
    /// Create a new token bucket
    pub fn new(capacity: i64, refill_rate: i64) -> Self {
        Self {
            tokens: AtomicI64::new(capacity),
            capacity,
            refill_rate,
            last_refill: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            ),
        }
    }

    /// Try to acquire tokens, returns true if successful
    pub fn try_acquire(&self, tokens: i64) -> bool {
        self.refill();

        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < tokens {
                return false;
            }
            if self
                .tokens
                .compare_exchange_weak(
                    current,
                    current - tokens,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let last = self.last_refill.load(Ordering::Relaxed);
        let elapsed_ms = now.saturating_sub(last);

        if elapsed_ms > 0 {
            let tokens_to_add = (self.refill_rate * elapsed_ms as i64) / 1000;
            if tokens_to_add > 0 {
                if self
                    .last_refill
                    .compare_exchange(last, now, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    let current = self.tokens.load(Ordering::Relaxed);
                    let new_tokens = (current + tokens_to_add).min(self.capacity);
                    self.tokens.store(new_tokens, Ordering::Release);
                }
            }
        }
    }

    /// Return tokens (for cancelled operations)
    pub fn release(&self, tokens: i64) {
        let current = self.tokens.load(Ordering::Relaxed);
        let new_tokens = (current + tokens).min(self.capacity);
        self.tokens.store(new_tokens, Ordering::Release);
    }

    /// Current available tokens
    pub fn available(&self) -> i64 {
        self.refill();
        self.tokens.load(Ordering::Relaxed)
    }

    /// Utilization ratio (1.0 = empty, 0.0 = full)
    pub fn utilization(&self) -> f64 {
        1.0 - (self.available() as f64 / self.capacity as f64)
    }
}

/// Per-tenant quota and state
pub struct TenantQuota {
    /// Tenant identifier
    pub tenant_id: String,
    /// Weight for fair queuing (higher = more share)
    pub weight: u32,
    /// Token buckets per resource dimension
    buckets: HashMap<ResourceDimension, TokenBucket>,
    /// Deficit counter for DRR
    deficit: AtomicI64,
    /// Queue of pending requests
    pending_count: AtomicU64,
    /// Total requests processed
    total_requests: AtomicU64,
    /// Total requests rejected
    rejected_requests: AtomicU64,
}

impl TenantQuota {
    /// Create a new tenant quota with default limits
    pub fn new(tenant_id: String, weight: u32) -> Self {
        let mut buckets = HashMap::new();

        // Default limits per tenant
        buckets.insert(
            ResourceDimension::Cpu,
            TokenBucket::new(10_000_000, 1_000_000), // 10s burst, 1s/s refill
        );
        buckets.insert(
            ResourceDimension::RandomIops,
            TokenBucket::new(10_000, 1_000), // 10K burst, 1K/s
        );
        buckets.insert(
            ResourceDimension::SequentialBandwidth,
            TokenBucket::new(1_000_000_000, 100_000_000), // 1GB burst, 100MB/s
        );
        buckets.insert(
            ResourceDimension::Memory,
            TokenBucket::new(1_000_000_000, 500_000_000), // 1GB burst, 500MB/s
        );
        buckets.insert(
            ResourceDimension::Connections,
            TokenBucket::new(100, 10), // 100 burst, 10/s
        );

        Self {
            tenant_id,
            weight,
            buckets,
            deficit: AtomicI64::new(0),
            pending_count: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
        }
    }

    /// Try to acquire resources for an operation
    pub fn try_acquire(&self, cost: &OperationCost) -> bool {
        // Check all dimensions
        let cpu_ok = self
            .buckets
            .get(&ResourceDimension::Cpu)
            .map(|b| b.available() >= cost.cpu_us as i64)
            .unwrap_or(true);

        let iops_ok = self
            .buckets
            .get(&ResourceDimension::RandomIops)
            .map(|b| b.available() >= cost.random_iops as i64)
            .unwrap_or(true);

        let bandwidth_ok = self
            .buckets
            .get(&ResourceDimension::SequentialBandwidth)
            .map(|b| b.available() >= cost.sequential_bytes as i64)
            .unwrap_or(true);

        let memory_ok = self
            .buckets
            .get(&ResourceDimension::Memory)
            .map(|b| b.available() >= cost.memory_bytes as i64)
            .unwrap_or(true);

        if cpu_ok && iops_ok && bandwidth_ok && memory_ok {
            // Acquire all resources atomically (best effort)
            if let Some(b) = self.buckets.get(&ResourceDimension::Cpu) {
                b.try_acquire(cost.cpu_us as i64);
            }
            if let Some(b) = self.buckets.get(&ResourceDimension::RandomIops) {
                b.try_acquire(cost.random_iops as i64);
            }
            if let Some(b) = self.buckets.get(&ResourceDimension::SequentialBandwidth) {
                b.try_acquire(cost.sequential_bytes as i64);
            }
            if let Some(b) = self.buckets.get(&ResourceDimension::Memory) {
                b.try_acquire(cost.memory_bytes as i64);
            }
            self.total_requests.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            self.rejected_requests.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Release resources after operation completes
    pub fn release(&self, cost: &OperationCost) {
        if let Some(b) = self.buckets.get(&ResourceDimension::Memory) {
            b.release(cost.memory_bytes as i64);
        }
    }

    /// Get utilization across all dimensions
    pub fn utilization(&self) -> HashMap<ResourceDimension, f64> {
        self.buckets
            .iter()
            .map(|(dim, bucket)| (*dim, bucket.utilization()))
            .collect()
    }
}

/// Admission decision
#[derive(Debug, Clone)]
pub enum AdmissionDecision {
    /// Request admitted with cost
    Admit { cost: OperationCost },
    /// Request rejected due to overload
    Reject {
        reason: String,
        retry_after_ms: Option<u64>,
    },
    /// Request can proceed with partial results
    PartialAdmit {
        cost: OperationCost,
        max_results: usize,
        recall_warning: String,
    },
}

/// Admission control configuration
#[derive(Debug, Clone)]
pub struct AdmissionConfig {
    /// Global token bucket capacities
    pub global_limits: HashMap<ResourceDimension, (i64, i64)>, // (capacity, refill_rate)
    /// Default tenant weight
    pub default_tenant_weight: u32,
    /// Enable partial results
    pub allow_partial_results: bool,
    /// Queue depth warning threshold
    pub queue_depth_warning: usize,
    /// Queue depth rejection threshold
    pub queue_depth_rejection: usize,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        let mut global_limits = HashMap::new();
        global_limits.insert(ResourceDimension::Cpu, (100_000_000, 10_000_000));
        global_limits.insert(ResourceDimension::RandomIops, (100_000, 10_000));
        global_limits.insert(ResourceDimension::SequentialBandwidth, (10_000_000_000, 1_000_000_000));
        global_limits.insert(ResourceDimension::Memory, (10_000_000_000, 2_000_000_000));
        global_limits.insert(ResourceDimension::Connections, (1000, 100));

        Self {
            global_limits,
            default_tenant_weight: 100,
            allow_partial_results: false,
            queue_depth_warning: 100,
            queue_depth_rejection: 1000,
        }
    }
}

/// Admission controller
pub struct AdmissionController {
    config: AdmissionConfig,
    /// Global token buckets
    global_buckets: HashMap<ResourceDimension, TokenBucket>,
    /// Per-tenant quotas
    tenants: RwLock<HashMap<String, Arc<TenantQuota>>>,
    /// Current queue depth
    queue_depth: AtomicU64,
    /// Metrics
    metrics: AdmissionMetrics,
}

/// Admission control metrics
#[derive(Default)]
pub struct AdmissionMetrics {
    pub total_requests: AtomicU64,
    pub admitted_requests: AtomicU64,
    pub rejected_requests: AtomicU64,
    pub partial_requests: AtomicU64,
    pub avg_queue_depth: AtomicU64,
}

impl AdmissionController {
    /// Create a new admission controller
    pub fn new(config: AdmissionConfig) -> Self {
        let mut global_buckets = HashMap::new();
        for (dim, (capacity, rate)) in &config.global_limits {
            global_buckets.insert(*dim, TokenBucket::new(*capacity, *rate));
        }

        Self {
            config,
            global_buckets,
            tenants: RwLock::new(HashMap::new()),
            queue_depth: AtomicU64::new(0),
            metrics: AdmissionMetrics::default(),
        }
    }

    /// Register a tenant
    pub fn register_tenant(&self, tenant_id: &str, weight: u32) {
        let mut tenants = self.tenants.write();
        if !tenants.contains_key(tenant_id) {
            tenants.insert(
                tenant_id.to_string(),
                Arc::new(TenantQuota::new(tenant_id.to_string(), weight)),
            );
        }
    }

    /// Get or create tenant quota
    fn get_tenant(&self, tenant_id: &str) -> Arc<TenantQuota> {
        {
            let tenants = self.tenants.read();
            if let Some(tenant) = tenants.get(tenant_id) {
                return tenant.clone();
            }
        }

        // Create new tenant
        let mut tenants = self.tenants.write();
        tenants
            .entry(tenant_id.to_string())
            .or_insert_with(|| {
                Arc::new(TenantQuota::new(
                    tenant_id.to_string(),
                    self.config.default_tenant_weight,
                ))
            })
            .clone()
    }

    /// Evaluate admission for a request
    pub fn evaluate(
        &self,
        tenant_id: &str,
        cost: OperationCost,
        allow_partial: bool,
    ) -> AdmissionDecision {
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

        // Check queue depth
        let depth = self.queue_depth.load(Ordering::Relaxed);
        if depth >= self.config.queue_depth_rejection as u64 {
            self.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return AdmissionDecision::Reject {
                reason: format!("Queue depth {} exceeds limit", depth),
                retry_after_ms: Some(100),
            };
        }

        // Check global limits
        let global_ok = self.check_global_limits(&cost);
        if !global_ok {
            self.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return AdmissionDecision::Reject {
                reason: "Global resource limits exceeded".to_string(),
                retry_after_ms: Some(50),
            };
        }

        // Check tenant limits
        let tenant = self.get_tenant(tenant_id);
        if tenant.try_acquire(&cost) {
            self.queue_depth.fetch_add(1, Ordering::Relaxed);
            self.metrics.admitted_requests.fetch_add(1, Ordering::Relaxed);
            AdmissionDecision::Admit { cost }
        } else if allow_partial && self.config.allow_partial_results {
            // Try with reduced cost
            let reduced_cost = OperationCost {
                cpu_us: cost.cpu_us / 4,
                random_iops: cost.random_iops / 4,
                sequential_bytes: cost.sequential_bytes / 4,
                memory_bytes: cost.memory_bytes / 4,
                priority: cost.priority,
            };
            if tenant.try_acquire(&reduced_cost) {
                self.queue_depth.fetch_add(1, Ordering::Relaxed);
                self.metrics.partial_requests.fetch_add(1, Ordering::Relaxed);
                AdmissionDecision::PartialAdmit {
                    cost: reduced_cost,
                    max_results: 25, // 25% of normal
                    recall_warning: "Results limited due to load - recall may be degraded".to_string(),
                }
            } else {
                self.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
                AdmissionDecision::Reject {
                    reason: format!("Tenant {} quota exceeded", tenant_id),
                    retry_after_ms: Some(100),
                }
            }
        } else {
            self.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
            AdmissionDecision::Reject {
                reason: format!("Tenant {} quota exceeded", tenant_id),
                retry_after_ms: Some(100),
            }
        }
    }

    /// Check global resource limits
    fn check_global_limits(&self, cost: &OperationCost) -> bool {
        let cpu_ok = self
            .global_buckets
            .get(&ResourceDimension::Cpu)
            .map(|b| b.available() >= cost.cpu_us as i64)
            .unwrap_or(true);

        let iops_ok = self
            .global_buckets
            .get(&ResourceDimension::RandomIops)
            .map(|b| b.available() >= cost.random_iops as i64)
            .unwrap_or(true);

        cpu_ok && iops_ok
    }

    /// Complete a request (release resources)
    pub fn complete(&self, tenant_id: &str, cost: &OperationCost) {
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
        let tenant = self.get_tenant(tenant_id);
        tenant.release(cost);
    }

    /// Get current system load
    pub fn system_load(&self) -> SystemLoad {
        let mut utilization = HashMap::new();
        for (dim, bucket) in &self.global_buckets {
            utilization.insert(*dim, bucket.utilization());
        }

        SystemLoad {
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            utilization,
            total_requests: self.metrics.total_requests.load(Ordering::Relaxed),
            admitted_requests: self.metrics.admitted_requests.load(Ordering::Relaxed),
            rejected_requests: self.metrics.rejected_requests.load(Ordering::Relaxed),
        }
    }
}

/// Current system load
#[derive(Debug)]
pub struct SystemLoad {
    pub queue_depth: u64,
    pub utilization: HashMap<ResourceDimension, f64>,
    pub total_requests: u64,
    pub admitted_requests: u64,
    pub rejected_requests: u64,
}

impl SystemLoad {
    /// Is the system overloaded?
    pub fn is_overloaded(&self) -> bool {
        self.utilization.values().any(|&u| u > 0.9)
    }

    /// Admission rate
    pub fn admission_rate(&self) -> f64 {
        if self.total_requests == 0 {
            1.0
        } else {
            self.admitted_requests as f64 / self.total_requests as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(100, 10);
        assert_eq!(bucket.available(), 100);

        assert!(bucket.try_acquire(50));
        assert_eq!(bucket.available(), 50);

        assert!(!bucket.try_acquire(60));
        assert_eq!(bucket.available(), 50);

        bucket.release(25);
        assert_eq!(bucket.available(), 75);
    }

    #[test]
    fn test_operation_cost_estimation() {
        let point_read = OperationCost::point_read();
        assert!(point_read.random_iops > 0);
        assert_eq!(point_read.sequential_bytes, 0);

        let scan = OperationCost::scan(1000, 100);
        assert_eq!(scan.sequential_bytes, 100_000);
        assert_eq!(scan.random_iops, 0);

        let vector = OperationCost::vector_search(128, 64, 100);
        assert!(vector.cpu_us > 0);
    }

    #[test]
    fn test_admission_controller_basic() {
        let controller = AdmissionController::new(AdmissionConfig::default());
        controller.register_tenant("test", 100);

        let cost = OperationCost::point_read();
        let decision = controller.evaluate("test", cost.clone(), false);

        assert!(matches!(decision, AdmissionDecision::Admit { .. }));

        controller.complete("test", &cost);
    }

    #[test]
    fn test_tenant_quota_exhaustion() {
        let quota = TenantQuota::new("test".to_string(), 100);

        // Exhaust CPU tokens
        let huge_cost = OperationCost {
            cpu_us: 100_000_000, // 100 seconds
            random_iops: 0,
            sequential_bytes: 0,
            memory_bytes: 0,
            priority: 100,
        };

        assert!(!quota.try_acquire(&huge_cost));
    }
}
