//! Benchmark workload definitions.
//!
//! Each function takes a `&mut dyn BenchDb` and a config, runs the workload,
//! and returns a `WorkloadResult`.

use crate::{AnalyticsRow, BenchDb, BenchResult, DataGen, LatencyRecorder, WorkloadResult};

// ────────────────────────────────────────────────────────────────────────────────
// Config
// ────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    pub scale: usize,       // multiplier — 1 = 10 000 ops, 10 = 100 000, etc.
    pub value_size: usize,  // bytes per value in KV workloads
    pub batch_size: usize,  // batch write chunk size
    pub dim: usize,         // vector dimension
    pub k: usize,           // top-k for ANN search
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            scale: 10_000,
            value_size: 256,
            batch_size: 1000,
            dim: 128,
            k: 10,
        }
    }
}

impl WorkloadConfig {
    pub fn n(&self) -> usize {
        self.scale
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// OLTP Workloads
// ────────────────────────────────────────────────────────────────────────────────

/// Sequential point writes.
pub fn oltp_sequential_writes(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = cfg.n();
    let mut gen = DataGen::new(42);
    let mut rec = LatencyRecorder::new();

    db.setup_kv_table()?;

    for i in 0..n {
        let key = gen.kv_key(i as u64);
        let val = gen.random_value(cfg.value_size);
        let t = rec.start();
        db.put(&key, &val)?;
        rec.record(t);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "oltp_seq_write", &rec))
}

/// Sequential point reads (read-back after write).
pub fn oltp_sequential_reads(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = cfg.n();
    let gen = DataGen::new(42);
    let mut rec = LatencyRecorder::new();

    for i in 0..n {
        let key = gen.kv_key(i as u64);
        let t = rec.start();
        let _ = db.get(&key)?;
        rec.record(t);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "oltp_seq_read", &rec))
}

/// Random point reads.
pub fn oltp_random_reads(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = cfg.n();
    let mut gen = DataGen::new(99);
    let data_gen = DataGen::new(42);
    let mut rec = LatencyRecorder::new();

    let indices = gen.shuffled_indices(n);

    for &i in &indices {
        let key = data_gen.kv_key(i as u64);
        let t = rec.start();
        let _ = db.get(&key)?;
        rec.record(t);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "oltp_rand_read", &rec))
}

/// Batch writes in chunks.
pub fn oltp_batch_write(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = cfg.n();
    let mut gen = DataGen::new(77);
    let mut rec = LatencyRecorder::new();

    // Use a separate KV namespace to avoid collisions with sequential writes.
    let mut offset = n as u64;
    for chunk_start in (0..n).step_by(cfg.batch_size) {
        let chunk_end = (chunk_start + cfg.batch_size).min(n);
        let count = chunk_end - chunk_start;
        let keys: Vec<Vec<u8>> = (0..count)
            .map(|_| {
                offset += 1;
                format!("bk:{:08x}", offset).into_bytes()
            })
            .collect();
        let vals: Vec<Vec<u8>> = (0..count).map(|_| gen.random_value(cfg.value_size)).collect();
        let pairs: Vec<(&[u8], &[u8])> = keys
            .iter()
            .zip(vals.iter())
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let t = rec.start();
        db.batch_put(&pairs)?;
        rec.record_batch(t.elapsed(), count as u64);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "oltp_batch_write", &rec))
}

/// Point deletes.
pub fn oltp_deletes(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = (cfg.n() / 5).min(50_000).max(100); // delete ~20%
    let gen = DataGen::new(42);
    let mut rec = LatencyRecorder::new();

    for i in 0..n {
        let key = gen.kv_key(i as u64);
        let t = rec.start();
        db.delete(&key)?;
        rec.record(t);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "oltp_delete", &rec))
}

// ────────────────────────────────────────────────────────────────────────────────
// Analytics Workloads
// ────────────────────────────────────────────────────────────────────────────────

/// Bulk insert analytics rows.
pub fn analytics_bulk_insert(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = cfg.n();
    let mut gen = DataGen::new(42);
    let mut rec = LatencyRecorder::new();

    db.setup_analytics_table()?;

    // Insert in batches of 1000.
    for chunk_start in (0..n).step_by(1000) {
        let chunk_end = (chunk_start + 1000).min(n);
        let rows: Vec<AnalyticsRow> = (chunk_start..chunk_end)
            .map(|i| gen.analytics_row(i as u64))
            .collect();
        let t = rec.start();
        db.insert_analytics_batch(&rows)?;
        let count = rows.len() as u64;
        rec.record_batch(t.elapsed(), count);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "analytics_bulk_insert", &rec))
}

/// Run analytics queries: filter, aggregate, group-by, range scan.
pub fn analytics_queries(
    db: &mut dyn BenchDb,
    _cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let mut rec = LatencyRecorder::new();
    let iterations = 20;

    for _ in 0..iterations {
        // filter: amount > 5000
        let t = rec.start();
        let count = db.scan_filter_amount_gt(5000.0)?;
        rec.record(t);
        let _ = count;

        // aggregate: SUM(amount)
        let t = rec.start();
        let _sum = db.aggregate_sum_amount()?;
        rec.record(t);

        // group by category
        let t = rec.start();
        let _groups = db.group_by_category_count()?;
        rec.record(t);

        // range scan on timestamp
        let t = rec.start();
        let _range = db.range_scan_ts(1_700_000_000, 1_700_100_000)?;
        rec.record(t);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "analytics_queries", &rec))
}

// ────────────────────────────────────────────────────────────────────────────────
// Vector Workloads
// ────────────────────────────────────────────────────────────────────────────────

/// Insert vectors.
pub fn vector_insert(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = cfg.n().min(50_000); // cap vector count
    let mut gen = DataGen::new(42);
    let mut rec = LatencyRecorder::new();

    db.setup_vector_table(cfg.dim)?;

    // Insert in batches of 500.
    for chunk_start in (0..n).step_by(500) {
        let chunk_end = (chunk_start + 500).min(n);
        let batch: Vec<(u64, Vec<f32>, Option<String>)> = (chunk_start..chunk_end)
            .map(|i| {
                let v = gen.random_vector(cfg.dim);
                let meta = if i % 10 == 0 {
                    Some(format!("meta-{}", i))
                } else {
                    None
                };
                (i as u64, v, meta)
            })
            .collect();
        let t = rec.start();
        db.insert_vector_batch(&batch)?;
        let count = batch.len() as u64;
        rec.record_batch(t.elapsed(), count);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "vector_insert", &rec))
}

/// Vector search queries.
pub fn vector_search(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let queries = 200;
    let mut gen = DataGen::new(999);
    let mut rec = LatencyRecorder::new();

    for _ in 0..queries {
        let q = gen.random_vector(cfg.dim);
        let t = rec.start();
        let _results = db.vector_search(&q, cfg.k)?;
        rec.record(t);
    }

    Ok(WorkloadResult::from_recorder(db.name(), "vector_search", &rec))
}

// ────────────────────────────────────────────────────────────────────────────────
// Storage Efficiency
// ────────────────────────────────────────────────────────────────────────────────

/// Measure DB size on disk after the benchmark data is written.
pub fn storage_efficiency(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let size = db.db_size_bytes()?;
    let raw_data = (cfg.n() * (10 + cfg.value_size)) as u64; // approx raw data size

    let mut result = WorkloadResult {
        db_name: db.name().to_string(),
        workload: "storage_efficiency".to_string(),
        ops: 0,
        total_secs: 0.0,
        throughput: 0.0,
        p50_us: 0.0,
        p99_us: 0.0,
        p999_us: 0.0,
        mean_us: 0.0,
        extra: std::collections::HashMap::new(),
    };

    result.extra.insert("db_size_bytes".into(), size.to_string());
    result.extra.insert("raw_data_bytes".into(), raw_data.to_string());
    if raw_data > 0 {
        let ratio = size as f64 / raw_data as f64;
        result.extra.insert("amplification".into(), format!("{:.2}x", ratio));
    }

    Ok(result)
}

// ────────────────────────────────────────────────────────────────────────────────
// Mixed Workload
// ────────────────────────────────────────────────────────────────────────────────

/// 80/20 read-heavy mixed workload.
pub fn mixed_read_heavy(
    db: &mut dyn BenchDb,
    cfg: &WorkloadConfig,
) -> BenchResult<WorkloadResult> {
    let n = cfg.n();
    let mut gen = DataGen::new(55);
    let data_gen = DataGen::new(42);
    let mut rec = LatencyRecorder::new();

    for _ in 0..n {
        let r: f64 = gen.random_u64() as f64 / u64::MAX as f64;
        if r < 0.8 {
            // read
            let idx = gen.random_u64() % (n as u64).max(1);
            let key = data_gen.kv_key(idx);
            let t = rec.start();
            let _ = db.get(&key)?;
            rec.record(t);
        } else {
            // write
            let key = format!("mx:{:08x}", gen.random_u64()).into_bytes();
            let val = gen.random_value(cfg.value_size);
            let t = rec.start();
            db.put(&key, &val)?;
            rec.record(t);
        }
    }

    Ok(WorkloadResult::from_recorder(db.name(), "mixed_80r_20w", &rec))
}
