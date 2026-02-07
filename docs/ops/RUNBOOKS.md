# SochDB Operational Runbooks

This document provides step-by-step procedures for common operational scenarios.

---

## Table of Contents

1. [Disk 90% Full](#disk-90-full)
2. [Compaction Debt Runaway](#compaction-debt-runaway)
3. [WAL Corruption Recovery](#wal-corruption-recovery)
4. [Memory Pressure](#memory-pressure)
5. [Snapshot/Backup Failure](#snapshot-failure)
6. [Boot Failure / Crash Loop](#boot-failure)
7. [High Latency Investigation](#high-latency)
8. [Graceful Shutdown](#graceful-shutdown)

---

## Disk 90% Full

### Symptoms
- Alert: `sochdb_disk_usage_ratio > 0.9`
- Health status: `degraded` with condition `disk_pressure`
- Write errors in logs

### Impact
- Writes may fail with ENOSPC
- WAL growth can cause crash
- Compaction blocked

### Immediate Actions

1. **Check current usage**:
   ```bash
   kubectl exec -it $POD -- df -h /data
   kubectl exec -it $POD -- du -sh /data/*
   ```

2. **Identify largest consumers**:
   ```bash
   kubectl exec -it $POD -- ls -lhS /data/wal/
   kubectl exec -it $POD -- ls -lhS /data/sstables/
   ```

3. **Trigger manual checkpoint** (truncates WAL):
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/checkpoint
   ```

4. **Force compaction** (reclaims SSTable space):
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/compact?level=major
   ```

### Long-term Resolution

1. **Expand PVC** (if storage class supports):
   ```bash
   kubectl patch pvc sochdb-data-0 -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'
   ```

2. **Enable WAL archiving** to offload old segments:
   ```yaml
   durability:
     archiving:
       enabled: true
       destination: s3://bucket/wal-archive
   ```

3. **Review data retention** and add TTL policies

---

## Compaction Debt Runaway

### Symptoms
- Alert: `sochdb_compaction_debt_ratio > 0.8`
- Health status: `degraded` with condition `compaction_debt`
- Read latency increasing (more levels to search)

### Impact
- Read amplification increases
- Space amplification increases
- Write stalls if debt is critical

### Immediate Actions

1. **Check compaction status**:
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/metrics | grep compaction
   ```

2. **Increase compaction parallelism temporarily**:
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/config \
     -d '{"compaction":{"parallelism":8}}'
   ```

3. **Trigger major compaction** (during low-traffic period):
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/compact?level=major
   ```

### Long-term Resolution

1. **Tune compaction settings**:
   ```yaml
   compaction:
     parallelism: 4
     level_size_multiplier: 10
     target_file_size_mb: 64
   ```

2. **Consider workload shaping**:
   - Reduce write rate during peak hours
   - Batch writes to reduce WAL entries

---

## WAL Corruption Recovery

### Symptoms
- Boot failure with WAL CRC errors
- Alert: `sochdb_wal_corruption_detected`
- Logs show: `WAL record checksum mismatch at LSN X`

### Impact
- Service cannot start
- Potential data loss for uncommitted transactions

### Immediate Actions

1. **DO NOT delete WAL files manually**

2. **Attempt normal recovery first**:
   ```bash
   kubectl exec -it $POD -- sochdb-server --recovery-mode=normal
   ```

3. **If normal fails, try force recovery** (skips corrupt records):
   ```bash
   kubectl set env statefulset/sochdb SOCHDB_BOOT_RECOVERY_MODE=force
   kubectl rollout restart statefulset/sochdb
   ```

4. **Check for data loss**:
   ```bash
   kubectl logs $POD | grep "skipped\|rolled back"
   ```

### Post-Recovery

1. **Create immediate backup**:
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/backup
   ```

2. **Investigate root cause**:
   - Check for hardware issues (smartctl)
   - Review power loss events
   - Check storage latency spikes

3. **Restore from known-good backup if needed**

---

## Memory Pressure

### Symptoms
- Alert: `sochdb_memory_usage_ratio > 0.85`
- Health status: `degraded` with condition `memory_pressure`
- OOMKilled events in pod history

### Impact
- Cache eviction degrades performance
- Risk of OOM kill
- Query timeouts

### Immediate Actions

1. **Check memory breakdown**:
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/debug/memory
   ```

2. **Reduce cache size temporarily**:
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/config \
     -d '{"cache":{"max_size_mb":512}}'
   ```

3. **Trigger cache eviction**:
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/cache/evict?target_mb=256
   ```

4. **Kill expensive queries** (if any):
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/admin/queries/active
   kubectl exec -it $POD -- curl -X DELETE http://localhost:8080/admin/queries/QUERY_ID
   ```

### Long-term Resolution

1. **Increase memory allocation**:
   ```yaml
   resources:
     requests:
       memory: "4Gi"
     limits:
       memory: "4Gi"  # MUST match requests
   ```

2. **Enable admission control** to reject expensive queries:
   ```yaml
   admissionControl:
     enabled: true
     queueDepthRejection: 500
   ```

---

## Snapshot Failure

### Symptoms
- Backup job failed
- Alert: `sochdb_backup_failed`
- Logs show snapshot errors

### Impact
- RPO at risk if backups consistently fail
- No PITR capability

### Immediate Actions

1. **Check backup status**:
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/admin/backup/status
   ```

2. **Check quiesce state** (might be stuck):
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/admin/quiesce/status
   ```

3. **Force resume if stuck**:
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/quiesce/resume
   ```

4. **Retry backup**:
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/backup
   ```

### Long-term Resolution

1. **Schedule backups during low-traffic periods**

2. **Use VolumeSnapshot** instead of application-level backup:
   ```yaml
   sidecars:
     backupCoordinator:
       enabled: true
   ```

---

## Boot Failure

### Symptoms
- Pod in CrashLoopBackOff
- Startup probe failing
- Logs show boot FSM stuck in non-Ready state

### Immediate Actions

1. **Check boot phase**:
   ```bash
   kubectl logs $POD --previous | grep -E "phase|boot|recovery"
   ```

2. **Identify stuck phase**:
   - `init`: Configuration or permission issue
   - `migrate`: Schema migration failed
   - `recover`: WAL replay taking too long or corrupted
   - `warmup`: Cache preload issue

3. **Increase startup probe timeout** (for long recovery):
   ```yaml
   probes:
     startup:
       failureThreshold: 120  # 20 minutes
   ```

4. **Try force recovery mode**:
   ```bash
   kubectl set env statefulset/sochdb SOCHDB_BOOT_RECOVERY_MODE=force
   ```

5. **Try read-only mode** (for forensics):
   ```bash
   kubectl set env statefulset/sochdb SOCHDB_BOOT_RECOVERY_MODE=readonly
   ```

---

## High Latency

### Symptoms
- Alert: `sochdb_request_latency_p99 > 100ms`
- SLO violation
- User complaints

### Investigation Steps

1. **Check overall health**:
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/health
   ```

2. **Check queue depth** (admission control):
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/metrics | grep queue_depth
   ```

3. **Check slow query log**:
   ```bash
   kubectl exec -it $POD -- tail -100 /data/logs/slow_query.log
   ```

4. **Check compaction status** (affects read latency):
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/metrics | grep compaction_debt
   ```

5. **Check cache hit rate**:
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/metrics | grep cache_hit
   ```

### Common Causes

| Cause | Indicator | Fix |
|-------|-----------|-----|
| Cache miss | hit_rate < 0.8 | Increase cache size |
| Compaction debt | debt_ratio > 0.5 | Trigger compaction |
| Queue depth | depth > 100 | Enable admission control |
| Disk latency | fsync_p99 > 10ms | Check storage health |
| Large scans | slow_query_log | Add indexes |

---

## Graceful Shutdown

### Procedure

1. **Mark node as not ready** (stop accepting new requests):
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/shutdown/prepare
   ```

2. **Wait for drain** (in-flight requests complete):
   ```bash
   kubectl exec -it $POD -- curl http://localhost:8080/admin/shutdown/status
   # Wait until active_requests = 0
   ```

3. **Trigger checkpoint** (reduce recovery time):
   ```bash
   kubectl exec -it $POD -- curl -X POST http://localhost:8080/admin/checkpoint
   ```

4. **Initiate shutdown**:
   ```bash
   kubectl delete pod $POD --grace-period=120
   ```

### For Rolling Updates

The StatefulSet is configured with `terminationGracePeriodSeconds: 120`. Kubernetes will:
1. Send SIGTERM
2. Wait up to 120 seconds for graceful shutdown
3. Send SIGKILL if process doesn't exit

---

## Contact

For issues not covered by these runbooks:
- Slack: #sochdb-ops
- Email: ops@sochdb.dev
- On-call: PagerDuty escalation policy
