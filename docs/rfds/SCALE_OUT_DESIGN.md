# SochDB Scale-Out Design Spike

## Executive Summary

This document outlines the design considerations for evolving SochDB from a single-node embedded database to a distributed system. The design prioritizes **Raft-based replication first** before considering sharding, following the principle of "get replication right before distribution."

## Design Principles

1. **Replication Before Sharding**: Master replication solves availability; sharding solves scale
2. **Single-Writer Semantics**: Maintain strong consistency guarantees
3. **Async Replication Default**: Synchronous optional for critical workloads
4. **Read Replicas**: Scale reads horizontally
5. **Automatic Failover**: Sub-second leader election

## Architecture Options

### Option 1: Raft-Based Replication (Recommended for v1)

```
┌─────────────────────────────────────────────────────────┐
│                     Raft Cluster                         │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐            │
│  │ Leader  │────▶│ Follower│     │ Follower│            │
│  │ (Write) │     │ (Read)  │     │ (Read)  │            │
│  └─────────┘     └─────────┘     └─────────┘            │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │              WAL + State Machine                 │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Strong consistency (linearizable reads from leader)
- Automatic failover
- Well-understood protocol
- Works well for SochDB's write patterns

**Cons:**
- 3+ nodes minimum
- Cross-region latency impacts writes
- Single-writer bottleneck

### Option 2: Primary-Standby with Streaming Replication

```
┌──────────────┐          ┌──────────────┐
│   Primary    │──WAL──▶ │   Standby    │
│   (R/W)      │  Stream  │   (Read)     │
└──────────────┘          └──────────────┘
       │
       ▼
┌──────────────┐
│   Standby 2  │
│   (Read)     │
└──────────────┘
```

**Pros:**
- Simpler than Raft
- Lower overhead (async by default)
- Proven PostgreSQL model

**Cons:**
- Manual failover or external coordinator (etcd/Consul)
- Potential data loss on failover (async)

### Option 3: CockroachDB-style Range Sharding (Future)

```
┌─────────────────────────────────────────────────────────┐
│                    Meta Range (Raft)                     │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Range 1: [A-M] │ Range 2: [N-Z] │ Range 3: [...]  │ │
│  │   (3 replicas) │   (3 replicas) │   (3 replicas)  │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Horizontal write scaling
- Automatic rebalancing
- Geo-distribution

**Cons:**
- Significant complexity
- Distributed transactions (2PC/2PL)
- Cross-shard queries

## Recommended Phased Approach

### Phase 1: Leader-Follower with Raft (v1.0)

**Scope:**
- 3-node Raft cluster
- Single leader for writes
- Read replicas for horizontal read scaling
- Automatic leader election
- WAL-based state machine replication

**Components:**

```rust
/// Raft node role
pub enum NodeRole {
    Leader,
    Follower,
    Candidate,
}

/// Raft configuration
pub struct RaftConfig {
    /// Node ID
    pub node_id: u64,
    /// Cluster members
    pub peers: Vec<(u64, String)>,
    /// Election timeout range
    pub election_timeout: (Duration, Duration),
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Snapshot threshold (log entries)
    pub snapshot_threshold: u64,
}

/// Raft log entry
pub struct LogEntry {
    /// Term when entry was received
    pub term: u64,
    /// Log index
    pub index: u64,
    /// Command to apply to state machine
    pub command: Vec<u8>,
}
```

**Integration Points:**
- WAL entries become Raft log entries
- Commit only after Raft consensus
- State machine = SochDB storage engine

### Phase 2: Read Replicas (v1.1)

**Scope:**
- Async read replicas outside Raft quorum
- Eventual consistency for reads
- Follower reads with stale tolerance

```rust
/// Read consistency level
pub enum ReadConsistency {
    /// Read from leader (linearizable)
    Leader,
    /// Read from any follower (eventual)
    Follower,
    /// Read from follower with max staleness
    Bounded { max_staleness: Duration },
    /// Read from local node (fastest, least consistent)
    Local,
}
```

### Phase 3: Multi-Region (v1.2)

**Scope:**
- Geo-distributed Raft groups
- Witness nodes for odd quorum
- Regional read routing

```
Region A (Primary)          Region B (DR)
┌─────────────────┐        ┌─────────────────┐
│ Node 1 (Leader) │◀──────▶│ Node 3 (Follower)│
│ Node 2 (Follow) │        │ Node 4 (Witness) │
└─────────────────┘        └─────────────────┘
```

### Phase 4: Range Sharding (v2.0, if needed)

**Deferred until:**
- Single-node write throughput is bottleneck
- Data size exceeds single-node capacity
- Strong customer demand

## Replication Protocol Details

### Write Path (Raft)

```
Client                Leader              Followers
  │                     │                     │
  │──── Write ─────────▶│                     │
  │                     │── AppendEntries ───▶│
  │                     │◀── Success ─────────│
  │                     │                     │
  │                     │ (Quorum reached)    │
  │                     │── Apply to SM ─────▶│
  │◀─── Ack ───────────│                     │
```

### Read Path (Linearizable)

```
Client                Leader              Followers
  │                     │                     │
  │──── Read ──────────▶│                     │
  │                     │── Heartbeat ───────▶│
  │                     │◀── Ack ────────────│
  │                     │                     │
  │                     │ (Quorum confirmed)  │
  │◀─── Response ──────│                     │
```

### Read Path (Follower Read)

```
Client                Follower
  │                     │
  │──── Read ──────────▶│
  │                     │ (Check lease valid)
  │◀─── Response ──────│
```

## Kubernetes Operator

### Custom Resource Definition

```yaml
apiVersion: sochdb.io/v1
kind: SochDBCluster
metadata:
  name: production
spec:
  replicas: 3
  version: "1.0.0"
  resources:
    requests:
      cpu: "2"
      memory: "8Gi"
    limits:
      cpu: "4"
      memory: "16Gi"
  storage:
    size: 100Gi
    storageClass: premium-ssd
  topology:
    zones:
      - us-east-1a
      - us-east-1b
      - us-east-1c
  backup:
    schedule: "0 * * * *"
    retention: 7d
```

### Operator Responsibilities

1. **Cluster Lifecycle**
   - Bootstrap new cluster
   - Scale up/down
   - Rolling upgrades

2. **Failure Handling**
   - Detect node failures
   - Trigger Raft reconfiguration
   - Replace failed nodes

3. **Backup Management**
   - Schedule base snapshots
   - Manage WAL archiving
   - PITR coordination

## Storage Engine Considerations

### Shared-Nothing Architecture

Each node maintains:
- Complete WAL
- Full data copy
- Local indexes

### Log Shipping

```
Leader                    Follower
┌──────────┐             ┌──────────┐
│   WAL    │──Entries───▶│   WAL    │
│ (append) │             │ (append) │
└──────────┘             └──────────┘
     │                        │
     ▼                        ▼
┌──────────┐             ┌──────────┐
│  Engine  │             │  Engine  │
│ (apply)  │             │ (apply)  │
└──────────┘             └──────────┘
```

### Snapshot Transfer

For new nodes or far-behind followers:

```
Leader                    New Node
┌──────────┐             ┌──────────┐
│ Snapshot │──Transfer──▶│ Restore  │
│ (frozen) │   (SST)     │  Files   │
└──────────┘             └──────────┘
                              │
                              ▼
                         ┌──────────┐
                         │ Catch up │
                         │   WAL    │
                         └──────────┘
```

## Consistency Model

### Default: Sequential Consistency

- All operations appear in same order on all replicas
- Client sees own writes (read-your-writes)
- Sufficient for most LLM workloads

### Optional: Linearizability

- Real-time ordering
- Requires quorum read or leader lease
- Higher latency

### Eventual (for reads)

- Follower reads may be stale
- Configurable staleness bound
- Best for read-heavy analytics

## Transaction Handling

### Single-Node Transactions (v1)

No change to existing SSI implementation.

### Distributed Transactions (v2, deferred)

If range sharding is implemented:
- 2PC for cross-shard transactions
- Parallel commit optimization
- Spanner-style commit wait (TrueTime-like)

## Monitoring & Observability

### Replication Metrics

```
# Replication lag (seconds behind leader)
sochdb_replication_lag_seconds{node="follower1"} 0.05

# Raft term
sochdb_raft_term 42

# Raft role
sochdb_raft_role{role="leader"} 1

# Log entries applied
sochdb_raft_applied_index 1234567

# Snapshot size
sochdb_raft_snapshot_bytes 1073741824
```

### Alerts

```yaml
groups:
  - name: sochdb-replication
    rules:
      - alert: HighReplicationLag
        expr: sochdb_replication_lag_seconds > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Replication lag exceeds 5 seconds"

      - alert: NoLeader
        expr: sum(sochdb_raft_role{role="leader"}) == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "No Raft leader elected"
```

## Open Questions

1. **Raft Library**: Use existing (raft-rs, openraft) or custom?
2. **Network Layer**: gRPC or custom protocol?
3. **Membership Changes**: Joint consensus or single-step?
4. **Learner Nodes**: Support non-voting replicas?
5. **Witness Nodes**: Metadata-only nodes for quorum?

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 3-4 months | 3-node Raft cluster |
| Phase 2 | 1-2 months | Read replicas |
| Phase 3 | 2-3 months | Multi-region |
| Phase 4 | 6+ months | Range sharding (if needed) |

## Conclusion

The recommended approach is to implement Raft-based replication in Phase 1, which provides:
- High availability with automatic failover
- Strong consistency guarantees
- Foundation for future scaling

Range sharding (Phase 4) should be deferred until there is clear evidence that single-node write throughput is insufficient for customer workloads.
