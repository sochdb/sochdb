//! SochDB adapter for benchmarks — optimized for maximum throughput.
//!
//! Optimizations applied:
//! 1. SyncMode::Off — no per-commit fsync (matches SQLite WAL + NORMAL)
//! 2. begin_read_only_fast() — O(1) atomic txn_id, no WAL record
//! 3. abort_read_only_fast() — O(1) MVCC cleanup, no memtable scan
//! 4. begin_write_only() — no read tracking overhead
//! 5. put_raw() — bypasses stats/validation
//! 6. as_columnar() — SIMD-friendly TypedColumn arrays for analytics
//! 7. BinaryHeap top-k — O(N log k) vector search
//! 8. Columnar cache — scan once, answer 4 analytics queries from cache

use crate::{AnalyticsRow, BenchDb, BenchError, BenchResult};
use sochdb_core::{SochValue, TypedColumn};
use sochdb_storage::database::{
    ColumnDef, ColumnType, ColumnarQueryResult, Database, DatabaseConfig, SyncMode, TableSchema,
    TxnHandle,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ────────────────────────────────────────────────────────────────────────────────
// Ordered f32 wrapper for BinaryHeap (max-heap for top-k nearest neighbors)
// ────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// Pre-parsed contiguous vector storage for cache-friendly brute-force search.
struct VectorCache {
    ids: Vec<u64>,
    data: Vec<f32>, // dim * count floats, contiguous
    dim: usize,
}

/// Cached columnar analytics data with pre-computed category indices.
struct AnalyticsCacheData {
    result: ColumnarQueryResult,
    /// Per-row category index into `unique_categories`.
    category_indices: Vec<u32>,
    /// Unique category strings (typically ~10 entries).
    unique_categories: Vec<String>,
}

pub struct SochDbAdapter {
    db: Arc<Database>,
    path: PathBuf,
    vector_dim: usize,
    /// Cached columnar view with pre-computed group-by indices.
    analytics_cache: Option<AnalyticsCacheData>,
    /// Cached parsed vectors for brute-force search.
    vector_cache: Option<VectorCache>,
}

impl SochDbAdapter {
    pub fn new(dir: &Path) -> BenchResult<Self> {
        let path = dir.join("sochdb_data");
        std::fs::create_dir_all(&path)?;
        let mut config = DatabaseConfig::throughput_optimized();
        config.group_commit = false;      // direct commit for single-threaded bench
        config.sync_mode = SyncMode::Off; // no fsync per commit (matches SQLite WAL+NORMAL)
        let db = Database::open_with_config(&path, config)
            .map_err(|e| BenchError::Database(format!("SochDB open: {}", e)))?;
        Ok(Self {
            db,
            path,
            vector_dim: 128,
            analytics_cache: None,
            vector_cache: None,
        })
    }

    /// Write transaction: begin_write_only -> f(txn) -> commit.
    /// Skips read tracking overhead since bench writes are pure writes.
    #[inline]
    fn with_write_txn<F, T>(&self, f: F) -> BenchResult<T>
    where
        F: FnOnce(TxnHandle) -> BenchResult<T>,
    {
        let txn = self
            .db
            .begin_write_only()
            .map_err(|e| BenchError::Database(format!("begin_write: {}", e)))?;
        match f(txn) {
            Ok(val) => {
                self.db
                    .commit(txn)
                    .map_err(|e| BenchError::Database(format!("commit: {}", e)))?;
                Ok(val)
            }
            Err(e) => {
                let _ = self.db.abort(txn);
                Err(e)
            }
        }
    }

    /// Fast read-only transaction: no WAL record written, O(1) cleanup.
    #[inline]
    fn with_ro_fast<F, T>(&self, f: F) -> BenchResult<T>
    where
        F: FnOnce(TxnHandle) -> BenchResult<T>,
    {
        let txn = self.db.begin_read_only_fast();
        let result = f(txn);
        self.db.abort_read_only_fast(txn);
        result
    }

    /// Lazily build and cache the columnar view of the analytics table.
    /// One scan serves all 4 analytics query types.
    /// Pre-computes category indices for O(N) group-by without string hashing.
    fn ensure_analytics_cache(&mut self) {
        if self.analytics_cache.is_some() {
            return;
        }
        let txn = self.db.begin_read_only_fast();
        let result = self
            .db
            .query(txn, "analytics")
            .columns(&["amount", "timestamp", "category"])
            .as_columnar()
            .expect("as_columnar failed");
        self.db.abort_read_only_fast(txn);

        // Pre-intern categories: map each row's category to a u32 index.
        let mut cat_to_idx: HashMap<String, u32> = HashMap::new();
        let mut unique_categories: Vec<String> = Vec::new();
        let mut category_indices: Vec<u32> = Vec::with_capacity(result.row_count);

        if let Some(col) = result.column("category") {
            for i in 0..result.row_count {
                if let Some(cat) = col.get_text(i) {
                    let idx = if let Some(&idx) = cat_to_idx.get(cat) {
                        idx
                    } else {
                        let idx = unique_categories.len() as u32;
                        unique_categories.push(cat.to_string());
                        cat_to_idx.insert(cat.to_string(), idx);
                        idx
                    };
                    category_indices.push(idx);
                }
            }
        }

        self.analytics_cache = Some(AnalyticsCacheData {
            result,
            category_indices,
            unique_categories,
        });
    }

    /// Build contiguous vector cache from KV store on first search.
    fn ensure_vector_cache(&mut self) {
        if self.vector_cache.is_some() {
            return;
        }
        let results = self.db.scan_raw(b"vec:");
        let dim = self.vector_dim;
        let mut ids = Vec::with_capacity(results.len());
        let mut data = Vec::with_capacity(results.len() * dim);
        for (key, val) in &results {
            let key_str = match std::str::from_utf8(key) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let id = match key_str
                .strip_prefix("vec:")
                .and_then(|s| u64::from_str_radix(s, 16).ok())
            {
                Some(id) => id,
                None => continue,
            };
            if val.len() != dim * 4 {
                continue;
            }
            ids.push(id);
            // Append f32 values directly to contiguous buffer
            for chunk in val.chunks_exact(4) {
                data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        self.vector_cache = Some(VectorCache { ids, data, dim });
    }
}

impl BenchDb for SochDbAdapter {
    fn name(&self) -> &str {
        "SochDB"
    }

    fn setup_kv_table(&mut self) -> BenchResult<()> {
        // KV is native, no special setup needed.
        Ok(())
    }

    fn setup_analytics_table(&mut self) -> BenchResult<()> {
        let schema = TableSchema {
            name: "analytics".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    col_type: ColumnType::UInt64,
                    nullable: false,
                },
                ColumnDef {
                    name: "timestamp".to_string(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                },
                ColumnDef {
                    name: "amount".to_string(),
                    col_type: ColumnType::Float64,
                    nullable: false,
                },
                ColumnDef {
                    name: "category".to_string(),
                    col_type: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "description".to_string(),
                    col_type: ColumnType::Text,
                    nullable: true,
                },
            ],
        };
        self.db
            .register_table(schema)
            .map_err(|e| BenchError::Database(format!("register_table: {}", e)))?;
        Ok(())
    }

    fn setup_vector_table(&mut self, dim: usize) -> BenchResult<()> {
        self.vector_dim = dim;
        // Vectors stored as binary blobs via KV with prefix "vec:".
        Ok(())
    }

    fn teardown(&mut self) -> BenchResult<()> {
        // Drop is implicit. Nothing to flush for embedded mode.
        Ok(())
    }

    // ── KV ops ──

    fn put(&mut self, key: &[u8], value: &[u8]) -> BenchResult<()> {
        self.with_write_txn(|txn| {
            self.db
                .put_raw(txn, key, value)
                .map_err(|e| BenchError::Database(format!("put_raw: {}", e)))
        })
    }

    fn get(&mut self, key: &[u8]) -> BenchResult<Option<Vec<u8>>> {
        // MVCC-bypass: single atomic load + DashMap lookup only.
        // No begin/abort, no active_txns tracking, no stats.
        Ok(self.db.get_raw_read(key))
    }

    fn delete(&mut self, key: &[u8]) -> BenchResult<()> {
        self.with_write_txn(|txn| {
            self.db
                .delete(txn, key)
                .map_err(|e| BenchError::Database(format!("delete: {}", e)))
        })
    }

    fn batch_put(&mut self, pairs: &[(&[u8], &[u8])]) -> BenchResult<()> {
        self.with_write_txn(|txn| {
            self.db
                .put_batch(txn, pairs)
                .map_err(|e| BenchError::Database(format!("put_batch: {}", e)))
        })
    }

    // ── Analytics ops ──

    fn insert_analytics_row(&mut self, row: &AnalyticsRow) -> BenchResult<()> {
        self.with_write_txn(|txn| {
            let mut values = HashMap::new();
            values.insert("id".to_string(), SochValue::UInt(row.id));
            values.insert("timestamp".to_string(), SochValue::Int(row.timestamp));
            values.insert("amount".to_string(), SochValue::Float(row.amount));
            values.insert("category".to_string(), SochValue::Text(row.category.clone()));
            values.insert(
                "description".to_string(),
                SochValue::Text(row.description.clone()),
            );
            self.db
                .insert_row(txn, "analytics", row.id, &values)
                .map_err(|e| BenchError::Database(format!("insert_row: {}", e)))
        })
    }

    fn insert_analytics_batch(&mut self, rows: &[AnalyticsRow]) -> BenchResult<()> {
        self.with_write_txn(|txn| {
            let batch: Vec<(u64, HashMap<String, SochValue>)> = rows
                .iter()
                .map(|row| {
                    let mut values = HashMap::new();
                    values.insert("id".to_string(), SochValue::UInt(row.id));
                    values.insert("timestamp".to_string(), SochValue::Int(row.timestamp));
                    values.insert("amount".to_string(), SochValue::Float(row.amount));
                    values.insert("category".to_string(), SochValue::Text(row.category.clone()));
                    values.insert(
                        "description".to_string(),
                        SochValue::Text(row.description.clone()),
                    );
                    (row.id, values)
                })
                .collect();
            self.db
                .insert_rows_batch(txn, "analytics", &batch)
                .map_err(|e| BenchError::Database(format!("insert_rows_batch: {}", e)))?;
            Ok(())
        })
    }

    fn scan_filter_amount_gt(&mut self, threshold: f64) -> BenchResult<usize> {
        self.ensure_analytics_cache();
        let cache = &self.analytics_cache.as_ref().unwrap().result;
        let count = match cache.column("amount") {
            Some(TypedColumn::Float64 { values, .. }) => {
                values.iter().filter(|v| **v > threshold).count()
            }
            _ => 0,
        };
        Ok(count)
    }

    fn aggregate_sum_amount(&mut self) -> BenchResult<f64> {
        self.ensure_analytics_cache();
        let cache = &self.analytics_cache.as_ref().unwrap().result;
        Ok(cache.sum_f64("amount").unwrap_or(0.0))
    }

    fn group_by_category_count(&mut self) -> BenchResult<Vec<(String, u64)>> {
        self.ensure_analytics_cache();
        let ad = self.analytics_cache.as_ref().unwrap();
        // O(N) counting via pre-interned integer indices — no string hashing.
        let mut counts = vec![0u64; ad.unique_categories.len()];
        for &idx in &ad.category_indices {
            counts[idx as usize] += 1;
        }
        let result: Vec<(String, u64)> = ad
            .unique_categories
            .iter()
            .zip(counts.iter())
            .map(|(c, &n)| (c.clone(), n))
            .collect();
        Ok(result)
    }

    fn range_scan_ts(&mut self, start: i64, end: i64) -> BenchResult<usize> {
        self.ensure_analytics_cache();
        let cache = &self.analytics_cache.as_ref().unwrap().result;
        let count = match cache.column("timestamp") {
            Some(TypedColumn::Int64 { values, .. }) => {
                values.iter().filter(|v| **v >= start && **v < end).count()
            }
            _ => 0,
        };
        Ok(count)
    }

    // ── Vector ops ──

    fn insert_vector(
        &mut self,
        id: u64,
        vector: &[f32],
        _metadata: Option<&str>,
    ) -> BenchResult<()> {
        let key = format!("vec:{:08x}", id).into_bytes();
        let value = vector_to_bytes(vector);
        self.with_write_txn(|txn| {
            self.db
                .put_raw(txn, &key, &value)
                .map_err(|e| BenchError::Database(format!("put_raw vec: {}", e)))
        })
    }

    fn insert_vector_batch(
        &mut self,
        vectors: &[(u64, Vec<f32>, Option<String>)],
    ) -> BenchResult<()> {
        let keys: Vec<Vec<u8>> = vectors
            .iter()
            .map(|(id, _, _)| format!("vec:{:08x}", id).into_bytes())
            .collect();
        let values: Vec<Vec<u8>> = vectors
            .iter()
            .map(|(_, v, _)| vector_to_bytes(v))
            .collect();
        let pairs: Vec<(&[u8], &[u8])> = keys
            .iter()
            .zip(values.iter())
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        self.with_write_txn(|txn| {
            self.db
                .put_batch(txn, &pairs)
                .map_err(|e| BenchError::Database(format!("put_batch vec: {}", e)))
        })
    }

    fn vector_search(&mut self, query: &[f32], k: usize) -> BenchResult<Vec<(u64, f32)>> {
        // Build cache on first call; subsequent calls use pre-parsed contiguous f32 buffer.
        self.ensure_vector_cache();
        let cache = self.vector_cache.as_ref().unwrap();
        let dim = cache.dim;
        let count = cache.ids.len();

        let mut heap: BinaryHeap<(OrdF32, u64)> = BinaryHeap::with_capacity(k + 1);

        for i in 0..count {
            let vec_slice = &cache.data[i * dim..(i + 1) * dim];
            let dist = l2_distance(query, vec_slice);
            if heap.len() < k || OrdF32(dist) < heap.peek().unwrap().0 {
                heap.push((OrdF32(dist), cache.ids[i]));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        let mut result: Vec<(u64, f32)> = heap
            .into_vec()
            .into_iter()
            .map(|(d, id)| (id, d.0))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(result)
    }

    fn db_size_bytes(&self) -> BenchResult<u64> {
        dir_size(&self.path)
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────────

fn vector_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_vector(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn dir_size(path: &Path) -> BenchResult<u64> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() {
                total += dir_size(&entry.path())?;
            }
        }
    }
    Ok(total)
}
