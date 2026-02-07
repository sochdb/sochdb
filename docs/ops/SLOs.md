# SochDB Service Level Objectives (SLOs)

This document defines the Service Level Objectives for SochDB deployments. SLOs provide measurable targets that guide engineering priorities and operator expectations.

## Availability

### Target: 99.9% Availability (Single-Node)

**Definition**: The percentage of time the service responds to health checks within the timeout period.

**Measurement**:
- `readiness_probe_success_rate = successful_probes / total_probes`
- Measured over a 30-day rolling window

**Budget**:
- 99.9% = 43.2 minutes of downtime per month
- 99.5% = 216 minutes of downtime per month (degraded target)

**Exclusions**:
- Planned maintenance windows (with 24h notice)
- Force majeure events

**Notes**:
- Single-node deployments cannot provide zero-downtime upgrades
- Use maintenance windows for version upgrades

---

## Latency

### Target: p99 < 10ms for Point Reads

**Definition**: 99th percentile latency for key-value GET operations on cached data.

**Measurement**:
- `histogram_quantile(0.99, rate(sochdb_request_duration_seconds_bucket{operation="get"}[5m]))`

**Assumptions**:
- Data fits in cache (working set < allocated memory)
- Network latency excluded (measured at gRPC handler)
- SSD/NVMe storage with <1ms access latency

### Target: p99 < 100ms for Vector Search (k=10, dim=128)

**Definition**: 99th percentile latency for HNSW approximate nearest neighbor search.

**Measurement**:
- `histogram_quantile(0.99, rate(sochdb_vector_search_duration_seconds_bucket[5m]))`

**Assumptions**:
- ef_search = 50 (default)
- Index fits in memory
- dimension = 128, k = 10

### Target: p50 < 5ms for Writes

**Definition**: Median latency for write operations (group commit enabled).

**Measurement**:
- `histogram_quantile(0.5, rate(sochdb_request_duration_seconds_bucket{operation="put"}[5m]))`

**Notes**:
- With group commit (10ms flush interval), p50 should be < 5ms
- p99 may be up to 15ms due to fsync batching

---

## Durability

### Target: Zero Data Loss on Committed Transactions

**Definition**: Any transaction that received a successful commit response is durably stored.

**Measurement**:
- Verified by crash testing and recovery validation
- `sochdb_wal_fsync_errors_total` should be 0

**Assumptions**:
- `durability.level = group_commit` or `durable`
- Storage hardware with power-loss protection (PLP)
- No operator intervention that bypasses durability

### RPO (Recovery Point Objective): 10ms

**Definition**: Maximum data loss window in case of crash.

**Measurement**:
- Equal to group commit flush interval

**Notes**:
- Can be reduced to 0 with `durability.level = durable` (at cost of throughput)

### RTO (Recovery Time Objective): 5 minutes

**Definition**: Time from crash detection to service availability.

**Measurement**:
- `sochdb_boot_duration_seconds{phase="total"}`

**Assumptions**:
- WAL size < 1GB
- Checkpoint frequency = default (128MB)
- SSD storage

**Notes**:
- RTO scales with O(|WAL| + |checkpoint|)
- Configure `boot.budgets` to match expected recovery time

---

## Throughput

### Target: 10,000 ops/sec for Mixed Workload

**Definition**: Combined read/write operations per second at p99 < SLO.

**Measurement**:
- `sum(rate(sochdb_requests_total[5m]))`

**Assumptions**:
- 80% reads, 20% writes
- 4 CPU cores, 8GB RAM
- Working set fits in memory

### Target: 50MB/s Write Throughput

**Definition**: Sustained write bandwidth to WAL + data files.

**Measurement**:
- `rate(sochdb_bytes_written_total[5m])`

**Assumptions**:
- NVMe storage
- Group commit enabled
- Compression enabled

---

## Resource Efficiency

### Target: Memory Usage < 90% of Limit

**Definition**: Process memory should not approach cgroup limit.

**Measurement**:
- `sochdb_memory_usage_bytes / sochdb_memory_limit_bytes < 0.9`

**Alert Threshold**: > 85%

**Notes**:
- OOM killer is not autoscaling
- Set `resources.requests.memory == resources.limits.memory`

### Target: CPU Usage < 80% Sustained

**Definition**: Average CPU utilization should leave headroom for spikes.

**Measurement**:
- `avg(rate(container_cpu_usage_seconds_total[5m])) / container_spec_cpu_quota`

---

## Error Budget

### Definition

Error budget = 1 - SLO target = acceptable unreliability

For 99.9% availability:
- Monthly budget: 43.2 minutes
- Weekly budget: 10.1 minutes

### Usage Guidelines

1. **>50% budget remaining**: Normal development velocity
2. **25-50% budget remaining**: Prioritize reliability work
3. **<25% budget remaining**: Feature freeze, focus on stability
4. **Budget exhausted**: All hands on reliability

### Burn Rate Alerts

| Alert Level | Burn Rate | Time to Exhaustion |
|-------------|-----------|-------------------|
| Warning     | 2x        | 15 days           |
| Critical    | 10x       | 3 days            |
| Emergency   | 100x      | 7.2 hours         |

---

## Metrics Reference

### Key Metrics for SLO Monitoring

```promql
# Availability
sochdb_health_status{probe="readiness"} == 1

# Latency
histogram_quantile(0.99, rate(sochdb_request_duration_seconds_bucket[5m]))

# Throughput
sum(rate(sochdb_requests_total[5m]))

# Error rate
sum(rate(sochdb_requests_total{status="error"}[5m])) / sum(rate(sochdb_requests_total[5m]))

# Durability
sochdb_wal_fsync_errors_total
sochdb_durability_level_info{level="group_commit"}

# Resource efficiency
sochdb_memory_usage_bytes / sochdb_memory_limit_bytes
rate(container_cpu_usage_seconds_total[5m])
```

---

## SLO Review Process

1. **Weekly**: Review SLO dashboards and burn rate
2. **Monthly**: Error budget report and incident review
3. **Quarterly**: SLO target revision based on capability and customer needs
4. **Annually**: Full SLO framework review
