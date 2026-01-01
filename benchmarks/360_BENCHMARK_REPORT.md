# ToonDB 360° Performance Report

Generated: 2025-12-27 23:57:46


## 1. Retrieval Quality

| Metric | Value |
|--------|-------|
| Recall@1 | 0.000 |
| Recall@5 | 0.002 |
| Recall@10 | 0.003 |
| Recall@20 | 0.003 |
| Recall@100 | 0.004 |
| MRR | 0.030 |
| NDCG@10 | 0.007 |

## 2. Query Latency

| Condition | p50 (ms) | p95 (ms) | p99 (ms) |
|-----------|----------|----------|----------|
| cold | 0.026 | 0.040 | 0.063 |
| warm | 0.026 | 0.031 | 0.038 |
| k=1 | 0.022 | 0.030 | 0.034 |
| k=10 | 0.024 | 0.030 | 0.032 |
| k=100 | 0.029 | 0.036 | 0.058 |

## 3. Throughput

- Single-thread QPS: 37,976
- 4-thread QPS: 29,164

## 4. Ingestion

- Insert rate: 116,479 vec/s
- Time-to-searchable: 0.49ms

## 5. Resource Efficiency

- RAM per vector: 99.9 bytes
- Index size: 1.0 MB

## 6. Agent Memory Performance

- Read recall: 7.00%
- Staleness error rate: 76.00%
- Memory latency overhead: 0.028ms