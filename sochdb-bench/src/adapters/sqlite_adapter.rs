//! SQLite adapter (via rusqlite).
//!
//! Configuration: WAL mode, NORMAL synchronous, WITHOUT ROWID for KV table.

use crate::{AnalyticsRow, BenchDb, BenchError, BenchResult};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};

pub struct SqliteAdapter {
    conn: Connection,
    path: PathBuf,
}

impl SqliteAdapter {
    pub fn new(dir: &Path) -> BenchResult<Self> {
        let path = dir.join("bench.sqlite3");
        let conn = Connection::open(&path)
            .map_err(|e| BenchError::Database(format!("SQLite open: {}", e)))?;

        // Tune for throughput.
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -64000;
             PRAGMA mmap_size = 268435456;
             PRAGMA temp_store = MEMORY;",
        )
        .map_err(|e| BenchError::Database(format!("SQLite pragma: {}", e)))?;

        Ok(Self { conn, path })
    }
}

impl BenchDb for SqliteAdapter {
    fn name(&self) -> &str {
        "SQLite"
    }

    fn setup_kv_table(&mut self) -> BenchResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS kv (
                    key   BLOB PRIMARY KEY,
                    value BLOB NOT NULL
                ) WITHOUT ROWID;",
            )
            .map_err(|e| BenchError::Database(format!("create kv: {}", e)))?;
        Ok(())
    }

    fn setup_analytics_table(&mut self) -> BenchResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS analytics (
                    id          INTEGER PRIMARY KEY,
                    timestamp   INTEGER NOT NULL,
                    amount      REAL    NOT NULL,
                    category    TEXT    NOT NULL,
                    description TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_analytics_ts  ON analytics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_analytics_cat ON analytics(category);",
            )
            .map_err(|e| BenchError::Database(format!("create analytics: {}", e)))?;
        Ok(())
    }

    fn setup_vector_table(&mut self, _dim: usize) -> BenchResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS vectors (
                    id       INTEGER PRIMARY KEY,
                    vector   BLOB    NOT NULL,
                    metadata TEXT
                );",
            )
            .map_err(|e| BenchError::Database(format!("create vectors: {}", e)))?;
        Ok(())
    }

    fn teardown(&mut self) -> BenchResult<()> {
        // WAL checkpoint to consolidate.
        let _ = self.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
        Ok(())
    }

    // ── KV ops ──

    fn put(&mut self, key: &[u8], value: &[u8]) -> BenchResult<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
                params![key, value],
            )
            .map_err(|e| BenchError::Database(format!("put: {}", e)))?;
        Ok(())
    }

    fn get(&mut self, key: &[u8]) -> BenchResult<Option<Vec<u8>>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT value FROM kv WHERE key = ?1")
            .map_err(|e| BenchError::Database(format!("prepare get: {}", e)))?;
        let result: Result<Vec<u8>, _> = stmt.query_row(params![key], |row| row.get(0));
        match result {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(BenchError::Database(format!("get: {}", e))),
        }
    }

    fn delete(&mut self, key: &[u8]) -> BenchResult<()> {
        self.conn
            .execute("DELETE FROM kv WHERE key = ?1", params![key])
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
                .prepare_cached("INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)")
                .map_err(|e| BenchError::Database(format!("prepare batch: {}", e)))?;
            for (k, v) in pairs {
                stmt.execute(params![*k, *v])
                    .map_err(|e| BenchError::Database(format!("batch put: {}", e)))?;
            }
        }
        tx.commit()
            .map_err(|e| BenchError::Database(format!("commit batch: {}", e)))?;
        Ok(())
    }

    // ── Analytics ops ──

    fn insert_analytics_row(&mut self, row: &AnalyticsRow) -> BenchResult<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO analytics (id, timestamp, amount, category, description)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![row.id as i64, row.timestamp, row.amount, row.category, row.description],
            )
            .map_err(|e| BenchError::Database(format!("insert analytics: {}", e)))?;
        Ok(())
    }

    fn insert_analytics_batch(&mut self, rows: &[AnalyticsRow]) -> BenchResult<()> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| BenchError::Database(format!("begin: {}", e)))?;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO analytics (id, timestamp, amount, category, description)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(|e| BenchError::Database(format!("prepare batch: {}", e)))?;
            for row in rows {
                stmt.execute(params![
                    row.id as i64,
                    row.timestamp,
                    row.amount,
                    row.category,
                    row.description
                ])
                .map_err(|e| BenchError::Database(format!("batch insert: {}", e)))?;
            }
        }
        tx.commit()
            .map_err(|e| BenchError::Database(format!("commit: {}", e)))?;
        Ok(())
    }

    fn scan_filter_amount_gt(&mut self, threshold: f64) -> BenchResult<usize> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT COUNT(*) FROM analytics WHERE amount > ?1")
            .map_err(|e| BenchError::Database(format!("prepare filter: {}", e)))?;
        let count: usize = stmt
            .query_row(params![threshold], |row| row.get(0))
            .map_err(|e| BenchError::Database(format!("filter: {}", e)))?;
        Ok(count)
    }

    fn aggregate_sum_amount(&mut self) -> BenchResult<f64> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT COALESCE(SUM(amount), 0.0) FROM analytics")
            .map_err(|e| BenchError::Database(format!("prepare sum: {}", e)))?;
        let sum: f64 = stmt
            .query_row([], |row| row.get(0))
            .map_err(|e| BenchError::Database(format!("sum: {}", e)))?;
        Ok(sum)
    }

    fn group_by_category_count(&mut self) -> BenchResult<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT category, COUNT(*) FROM analytics GROUP BY category")
            .map_err(|e| BenchError::Database(format!("prepare group: {}", e)))?;
        let rows = stmt
            .query_map([], |row| {
                let cat: String = row.get(0)?;
                let count: u64 = row.get(1)?;
                Ok((cat, count))
            })
            .map_err(|e| BenchError::Database(format!("group: {}", e)))?;
        let mut result = Vec::new();
        for r in rows {
            result.push(r.map_err(|e| BenchError::Database(format!("row: {}", e)))?);
        }
        Ok(result)
    }

    fn range_scan_ts(&mut self, start: i64, end: i64) -> BenchResult<usize> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT COUNT(*) FROM analytics WHERE timestamp >= ?1 AND timestamp < ?2",
            )
            .map_err(|e| BenchError::Database(format!("prepare range: {}", e)))?;
        let count: usize = stmt
            .query_row(params![start, end], |row| row.get(0))
            .map_err(|e| BenchError::Database(format!("range: {}", e)))?;
        Ok(count)
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
                "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES (?1, ?2, ?3)",
                params![id as i64, blob, metadata],
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
                .prepare_cached(
                    "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES (?1, ?2, ?3)",
                )
                .map_err(|e| BenchError::Database(format!("prepare: {}", e)))?;
            for (id, vec, meta) in vectors {
                let blob = vector_to_bytes(vec);
                let meta_ref = meta.as_deref();
                stmt.execute(params![*id as i64, blob, meta_ref])
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
            .prepare_cached("SELECT id, vector FROM vectors")
            .map_err(|e| BenchError::Database(format!("prepare search: {}", e)))?;
        let rows = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id as u64, blob))
            })
            .map_err(|e| BenchError::Database(format!("search: {}", e)))?;

        let mut scored = Vec::new();
        for r in rows {
            let (id, blob) = r.map_err(|e| BenchError::Database(format!("row: {}", e)))?;
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
        let meta = std::fs::metadata(&self.path)
            .map_err(|e| BenchError::Io(e))?;
        let mut total = meta.len();
        // Add WAL and SHM files if they exist.
        let wal = self.path.with_extension("sqlite3-wal");
        if wal.exists() {
            total += std::fs::metadata(&wal)?.len();
        }
        let shm = self.path.with_extension("sqlite3-shm");
        if shm.exists() {
            total += std::fs::metadata(&shm)?.len();
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
