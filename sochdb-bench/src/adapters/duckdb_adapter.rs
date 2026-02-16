//! DuckDB adapter.
//!
//! Configuration: 4 threads, 2 GB memory limit.

use crate::{AnalyticsRow, BenchDb, BenchError, BenchResult};
use duckdb::{params, Connection};
use std::path::{Path, PathBuf};

pub struct DuckDbAdapter {
    conn: Connection,
    path: PathBuf,
}

impl DuckDbAdapter {
    pub fn new(dir: &Path) -> BenchResult<Self> {
        let path = dir.join("bench.duckdb");
        let conn = Connection::open(&path)
            .map_err(|e| BenchError::Database(format!("DuckDB open: {}", e)))?;

        conn.execute_batch(
            "SET threads = 4;
             SET memory_limit = '2GB';",
        )
        .map_err(|e| BenchError::Database(format!("DuckDB config: {}", e)))?;

        Ok(Self { conn, path })
    }
}

impl BenchDb for DuckDbAdapter {
    fn name(&self) -> &str {
        "DuckDB"
    }

    fn setup_kv_table(&mut self) -> BenchResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS kv (
                    key   BLOB PRIMARY KEY,
                    value BLOB NOT NULL
                );",
            )
            .map_err(|e| BenchError::Database(format!("create kv: {}", e)))?;
        Ok(())
    }

    fn setup_analytics_table(&mut self) -> BenchResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS analytics (
                    id          UBIGINT PRIMARY KEY,
                    timestamp   BIGINT  NOT NULL,
                    amount      DOUBLE  NOT NULL,
                    category    VARCHAR NOT NULL,
                    description VARCHAR
                );",
            )
            .map_err(|e| BenchError::Database(format!("create analytics: {}", e)))?;
        Ok(())
    }

    fn setup_vector_table(&mut self, _dim: usize) -> BenchResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS vectors (
                    id       UBIGINT PRIMARY KEY,
                    vector   BLOB    NOT NULL,
                    metadata VARCHAR
                );",
            )
            .map_err(|e| BenchError::Database(format!("create vectors: {}", e)))?;
        Ok(())
    }

    fn teardown(&mut self) -> BenchResult<()> {
        let _ = self.conn.execute_batch("CHECKPOINT;");
        Ok(())
    }

    // ── KV ops ──

    fn put(&mut self, key: &[u8], value: &[u8]) -> BenchResult<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES ($1, $2)",
                params![key, value],
            )
            .map_err(|e| BenchError::Database(format!("put: {}", e)))?;
        Ok(())
    }

    fn get(&mut self, key: &[u8]) -> BenchResult<Option<Vec<u8>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM kv WHERE key = $1")
            .map_err(|e| BenchError::Database(format!("prepare get: {}", e)))?;
        let mut rows = stmt
            .query(params![key])
            .map_err(|e| BenchError::Database(format!("get: {}", e)))?;
        match rows.next().map_err(|e| BenchError::Database(format!("next: {}", e)))? {
            Some(row) => {
                let val: Vec<u8> = row
                    .get(0)
                    .map_err(|e| BenchError::Database(format!("get col: {}", e)))?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    }

    fn delete(&mut self, key: &[u8]) -> BenchResult<()> {
        self.conn
            .execute("DELETE FROM kv WHERE key = $1", params![key])
            .map_err(|e| BenchError::Database(format!("delete: {}", e)))?;
        Ok(())
    }

    fn batch_put(&mut self, pairs: &[(&[u8], &[u8])]) -> BenchResult<()> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| BenchError::Database(format!("begin: {}", e)))?;
        {
            let mut stmt = tx
                .prepare("INSERT OR REPLACE INTO kv (key, value) VALUES ($1, $2)")
                .map_err(|e| BenchError::Database(format!("prepare batch: {}", e)))?;
            for (k, v) in pairs {
                stmt.execute(params![*k, *v])
                    .map_err(|e| BenchError::Database(format!("batch: {}", e)))?;
            }
        }
        tx.commit()
            .map_err(|e| BenchError::Database(format!("commit: {}", e)))?;
        Ok(())
    }

    // ── Analytics ops ──

    fn insert_analytics_row(&mut self, row: &AnalyticsRow) -> BenchResult<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO analytics (id, timestamp, amount, category, description)
                 VALUES ($1, $2, $3, $4, $5)",
                params![
                    row.id,
                    row.timestamp,
                    row.amount,
                    &row.category,
                    &row.description
                ],
            )
            .map_err(|e| BenchError::Database(format!("insert: {}", e)))?;
        Ok(())
    }

    fn insert_analytics_batch(&mut self, rows: &[AnalyticsRow]) -> BenchResult<()> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| BenchError::Database(format!("begin: {}", e)))?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT OR REPLACE INTO analytics (id, timestamp, amount, category, description)
                     VALUES ($1, $2, $3, $4, $5)",
                )
                .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
            for row in rows {
                stmt.execute(params![
                    row.id,
                    row.timestamp,
                    row.amount,
                    &row.category,
                    &row.description
                ])
                .map_err(|e| BenchError::Database(format!("batch: {}", e)))?;
            }
        }
        tx.commit()
            .map_err(|e| BenchError::Database(format!("commit: {}", e)))?;
        Ok(())
    }

    fn scan_filter_amount_gt(&mut self, threshold: f64) -> BenchResult<usize> {
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM analytics WHERE amount > $1")
            .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
        let mut rows = stmt
            .query(params![threshold])
            .map_err(|e| BenchError::Database(format!("filter: {}", e)))?;
        let row = rows
            .next()
            .map_err(|e| BenchError::Database(format!("next: {}", e)))?
            .ok_or_else(|| BenchError::Database("no result".into()))?;
        let count: u64 = row
            .get(0)
            .map_err(|e| BenchError::Database(format!("get: {}", e)))?;
        Ok(count as usize)
    }

    fn aggregate_sum_amount(&mut self) -> BenchResult<f64> {
        let mut stmt = self
            .conn
            .prepare("SELECT COALESCE(SUM(amount), 0.0) FROM analytics")
            .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| BenchError::Database(format!("sum: {}", e)))?;
        let row = rows
            .next()
            .map_err(|e| BenchError::Database(format!("next: {}", e)))?
            .ok_or_else(|| BenchError::Database("no result".into()))?;
        let sum: f64 = row
            .get(0)
            .map_err(|e| BenchError::Database(format!("get: {}", e)))?;
        Ok(sum)
    }

    fn group_by_category_count(&mut self) -> BenchResult<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT category, COUNT(*) FROM analytics GROUP BY category")
            .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| BenchError::Database(format!("group: {}", e)))?;
        let mut result = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|e| BenchError::Database(format!("next: {}", e)))?
        {
            let cat: String = row
                .get(0)
                .map_err(|e| BenchError::Database(format!("cat: {}", e)))?;
            let count: u64 = row
                .get(1)
                .map_err(|e| BenchError::Database(format!("count: {}", e)))?;
            result.push((cat, count));
        }
        Ok(result)
    }

    fn range_scan_ts(&mut self, start: i64, end: i64) -> BenchResult<usize> {
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM analytics WHERE timestamp >= $1 AND timestamp < $2")
            .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
        let mut rows = stmt
            .query(params![start, end])
            .map_err(|e| BenchError::Database(format!("range: {}", e)))?;
        let row = rows
            .next()
            .map_err(|e| BenchError::Database(format!("next: {}", e)))?
            .ok_or_else(|| BenchError::Database("no result".into()))?;
        let count: u64 = row
            .get(0)
            .map_err(|e| BenchError::Database(format!("get: {}", e)))?;
        Ok(count as usize)
    }

    // ── Vector ops ──

    fn insert_vector(
        &mut self,
        id: u64,
        vector: &[f32],
        metadata: Option<&str>,
    ) -> BenchResult<()> {
        let blob = vector_to_bytes(vector);
        self.conn
            .execute(
                "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES ($1, $2, $3)",
                params![id, blob, metadata],
            )
            .map_err(|e| BenchError::Database(format!("insert vector: {}", e)))?;
        Ok(())
    }

    fn insert_vector_batch(
        &mut self,
        vectors: &[(u64, Vec<f32>, Option<String>)],
    ) -> BenchResult<()> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| BenchError::Database(format!("begin: {}", e)))?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES ($1, $2, $3)",
                )
                .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
            for (id, vec, meta) in vectors {
                let blob = vector_to_bytes(vec);
                let meta_ref = meta.as_deref();
                stmt.execute(params![*id, blob, meta_ref])
                    .map_err(|e| BenchError::Database(format!("batch vec: {}", e)))?;
            }
        }
        tx.commit()
            .map_err(|e| BenchError::Database(format!("commit: {}", e)))?;
        Ok(())
    }

    fn vector_search(&mut self, query: &[f32], k: usize) -> BenchResult<Vec<(u64, f32)>> {
        // Brute-force: load all vectors, compute L2 distance.
        let mut stmt = self
            .conn
            .prepare("SELECT id, vector FROM vectors")
            .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| BenchError::Database(format!("search: {}", e)))?;

        let mut scored = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|e| BenchError::Database(format!("next: {}", e)))?
        {
            let id: u64 = row
                .get(0)
                .map_err(|e| BenchError::Database(format!("id: {}", e)))?;
            let blob: Vec<u8> = row
                .get(1)
                .map_err(|e| BenchError::Database(format!("blob: {}", e)))?;
            let vec = bytes_to_vector(&blob);
            if vec.len() == query.len() {
                let dist = l2_distance(query, &vec);
                scored.push((id, dist));
            }
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        Ok(scored)
    }

    fn db_size_bytes(&self) -> BenchResult<u64> {
        let meta = std::fs::metadata(&self.path).map_err(BenchError::Io)?;
        let mut total = meta.len();
        // DuckDB WAL file.
        let wal = self.path.with_extension("duckdb.wal");
        if wal.exists() {
            total += std::fs::metadata(&wal)?.len();
        }
        Ok(total)
    }
}

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
