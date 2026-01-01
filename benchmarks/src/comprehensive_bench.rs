// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// Comprehensive 360-degree ToonDB vs SQLite Benchmark

use rusqlite::{params, Connection as SqliteConnection};
use std::collections::HashMap;
use std::time::Instant;
use tempfile::TempDir;
use toondb::prelude::*;

/// Results structure for easy comparison
#[derive(Default, Clone)]
pub struct BenchmarkResult {
    pub ops_per_sec: f64,
    pub time_ms: f64,
    pub count: usize,
}

impl BenchmarkResult {
    pub fn new(count: usize, duration: std::time::Duration) -> Self {
        Self {
            ops_per_sec: count as f64 / duration.as_secs_f64(),
            time_ms: duration.as_secs_f64() * 1000.0,
            count,
        }
    }
}

#[derive(Default)]
pub struct ComprehensiveResults {
    pub sqlite_file: HashMap<String, BenchmarkResult>,
    pub sqlite_memory: HashMap<String, BenchmarkResult>,
    pub toondb_wal: HashMap<String, BenchmarkResult>,
    pub toondb_memory: HashMap<String, BenchmarkResult>,
    pub toondb_fast: HashMap<String, BenchmarkResult>,
}

pub fn run_comprehensive_benchmark() -> Result<ComprehensiveResults, Box<dyn std::error::Error>> {
    let mut results = ComprehensiveResults::default();
    let temp_dir = TempDir::new()?;

    println!("\n{'='*60}");
    println!("   360° COMPREHENSIVE BENCHMARK: ToonDB vs SQLite");
    println!("{'='*60}\n");

    // Test configurations
    let record_counts = [1_000, 10_000, 100_000];
    
    for n in record_counts {
        println!("\n### Testing with {} records ###\n", n);
        
        // Pre-generate test data
        let names: Vec<String> = (0..n).map(|i| format!("User {}", i)).collect();
        let emails: Vec<String> = (0..n).map(|i| format!("user{}@example.com", i)).collect();
        let scores: Vec<i64> = (0..n).map(|i| (i % 100) as i64).collect();

        // ===== SQLite File =====
        println!("--- SQLite (File) ---");
        {
            let sqlite_path = temp_dir.path().join(format!("sqlite_file_{}.db", n));
            let mut conn = SqliteConnection::open(&sqlite_path)?;
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "synchronous", "NORMAL")?;
            conn.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, score INTEGER)",
                [],
            )?;

            // 1. Bulk Insert
            let start = Instant::now();
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare("INSERT INTO users VALUES (?, ?, ?, ?)")?;
                for i in 0..n {
                    stmt.execute(params![i, &names[i], &emails[i], scores[i]])?;
                }
            }
            tx.commit()?;
            let bulk_insert = BenchmarkResult::new(n, start.elapsed());
            println!("  Bulk Insert:    {:>10.0} ops/sec ({:.2}ms)", bulk_insert.ops_per_sec, bulk_insert.time_ms);
            results.sqlite_file.insert(format!("bulk_insert_{}", n), bulk_insert);

            // 2. Point Lookup (by ID)
            let start = Instant::now();
            let mut stmt = conn.prepare("SELECT * FROM users WHERE id = ?")?;
            for i in 0..n.min(10000) {
                let _: Option<(i64, String, String, i64)> = stmt.query_row(params![i], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                }).ok();
            }
            let point_lookup = BenchmarkResult::new(n.min(10000), start.elapsed());
            println!("  Point Lookup:   {:>10.0} ops/sec ({:.2}ms)", point_lookup.ops_per_sec, point_lookup.time_ms);
            results.sqlite_file.insert(format!("point_lookup_{}", n), point_lookup);

            // 3. Full Scan
            let start = Instant::now();
            let mut stmt = conn.prepare("SELECT * FROM users")?;
            let count = stmt.query_map([], |_| Ok(()))?.count();
            let full_scan = BenchmarkResult::new(count, start.elapsed());
            println!("  Full Scan:      {:>10.0} ops/sec ({:.2}ms)", full_scan.ops_per_sec, full_scan.time_ms);
            results.sqlite_file.insert(format!("full_scan_{}", n), full_scan);

            // 4. Range Query
            let start = Instant::now();
            let mut stmt = conn.prepare("SELECT * FROM users WHERE score BETWEEN 20 AND 40")?;
            let count = stmt.query_map([], |_| Ok(()))?.count();
            let range_query = BenchmarkResult::new(count, start.elapsed());
            println!("  Range Query:    {:>10.0} ops/sec ({:.2}ms, {} rows)", range_query.ops_per_sec, range_query.time_ms, count);
            results.sqlite_file.insert(format!("range_query_{}", n), range_query);

            // 5. Update
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("UPDATE users SET score = score + 1 WHERE id < ?", params![n / 2])?;
            tx.commit()?;
            let update = BenchmarkResult::new(n / 2, start.elapsed());
            println!("  Update (50%):   {:>10.0} ops/sec ({:.2}ms)", update.ops_per_sec, update.time_ms);
            results.sqlite_file.insert(format!("update_{}", n), update);

            // 6. Delete
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM users WHERE id >= ?", params![n / 2])?;
            tx.commit()?;
            let delete = BenchmarkResult::new(n / 2, start.elapsed());
            println!("  Delete (50%):   {:>10.0} ops/sec ({:.2}ms)", delete.ops_per_sec, delete.time_ms);
            results.sqlite_file.insert(format!("delete_{}", n), delete);
        }

        // ===== SQLite In-Memory =====
        println!("\n--- SQLite (In-Memory) ---");
        {
            let mut conn = SqliteConnection::open_in_memory()?;
            conn.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, score INTEGER)",
                [],
            )?;

            // 1. Bulk Insert
            let start = Instant::now();
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare("INSERT INTO users VALUES (?, ?, ?, ?)")?;
                for i in 0..n {
                    stmt.execute(params![i, &names[i], &emails[i], scores[i]])?;
                }
            }
            tx.commit()?;
            let bulk_insert = BenchmarkResult::new(n, start.elapsed());
            println!("  Bulk Insert:    {:>10.0} ops/sec ({:.2}ms)", bulk_insert.ops_per_sec, bulk_insert.time_ms);
            results.sqlite_memory.insert(format!("bulk_insert_{}", n), bulk_insert);

            // 2. Point Lookup
            let start = Instant::now();
            let mut stmt = conn.prepare("SELECT * FROM users WHERE id = ?")?;
            for i in 0..n.min(10000) {
                let _: Option<(i64, String, String, i64)> = stmt.query_row(params![i], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                }).ok();
            }
            let point_lookup = BenchmarkResult::new(n.min(10000), start.elapsed());
            println!("  Point Lookup:   {:>10.0} ops/sec ({:.2}ms)", point_lookup.ops_per_sec, point_lookup.time_ms);
            results.sqlite_memory.insert(format!("point_lookup_{}", n), point_lookup);

            // 3. Full Scan
            let start = Instant::now();
            let mut stmt = conn.prepare("SELECT * FROM users")?;
            let count = stmt.query_map([], |_| Ok(()))?.count();
            let full_scan = BenchmarkResult::new(count, start.elapsed());
            println!("  Full Scan:      {:>10.0} ops/sec ({:.2}ms)", full_scan.ops_per_sec, full_scan.time_ms);
            results.sqlite_memory.insert(format!("full_scan_{}", n), full_scan);

            // 4. Range Query
            let start = Instant::now();
            let mut stmt = conn.prepare("SELECT * FROM users WHERE score BETWEEN 20 AND 40")?;
            let count = stmt.query_map([], |_| Ok(()))?.count();
            let range_query = BenchmarkResult::new(count, start.elapsed());
            println!("  Range Query:    {:>10.0} ops/sec ({:.2}ms, {} rows)", range_query.ops_per_sec, range_query.time_ms, count);
            results.sqlite_memory.insert(format!("range_query_{}", n), range_query);

            // 5. Update
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("UPDATE users SET score = score + 1 WHERE id < ?", params![n / 2])?;
            tx.commit()?;
            let update = BenchmarkResult::new(n / 2, start.elapsed());
            println!("  Update (50%):   {:>10.0} ops/sec ({:.2}ms)", update.ops_per_sec, update.time_ms);
            results.sqlite_memory.insert(format!("update_{}", n), update);

            // 6. Delete
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM users WHERE id >= ?", params![n / 2])?;
            tx.commit()?;
            let delete = BenchmarkResult::new(n / 2, start.elapsed());
            println!("  Delete (50%):   {:>10.0} ops/sec ({:.2}ms)", delete.ops_per_sec, delete.time_ms);
            results.sqlite_memory.insert(format!("delete_{}", n), delete);
        }

        // ===== ToonDB (Embedded WAL) =====
        println!("\n--- ToonDB (Embedded WAL) ---");
        {
            let toon_path = temp_dir.path().join(format!("toondb_wal_{}", n));
            let config = toondb_storage::database::DatabaseConfig {
                group_commit: false,
                ..Default::default()
            };
            let conn = toondb::EmbeddedConnection::open_with_config(&toon_path, config)?;

            let schema = toondb::connection::KernelTableSchema {
                name: "users".to_string(),
                columns: vec![
                    toondb::connection::KernelColumnDef {
                        name: "id".to_string(),
                        col_type: toondb::connection::KernelColumnType::Int64,
                        nullable: false,
                    },
                    toondb::connection::KernelColumnDef {
                        name: "name".to_string(),
                        col_type: toondb::connection::KernelColumnType::Text,
                        nullable: false,
                    },
                    toondb::connection::KernelColumnDef {
                        name: "email".to_string(),
                        col_type: toondb::connection::KernelColumnType::Text,
                        nullable: false,
                    },
                    toondb::connection::KernelColumnDef {
                        name: "score".to_string(),
                        col_type: toondb::connection::KernelColumnType::Int64,
                        nullable: false,
                    },
                ],
            };
            conn.register_table(schema)?;

            // Pre-create ToonValues
            let toon_values: Vec<[ToonValue; 4]> = (0..n)
                .map(|i| [
                    ToonValue::Int(i as i64),
                    ToonValue::Text(names[i].clone()),
                    ToonValue::Text(emails[i].clone()),
                    ToonValue::Int(scores[i]),
                ])
                .collect();

            // 1. Bulk Insert
            let start = Instant::now();
            conn.begin()?;
            for i in 0..n {
                let row = &toon_values[i];
                let values: [Option<&ToonValue>; 4] = [Some(&row[0]), Some(&row[1]), Some(&row[2]), Some(&row[3])];
                conn.insert_row_slice("users", i as u64, &values)?;
            }
            conn.commit()?;
            let bulk_insert = BenchmarkResult::new(n, start.elapsed());
            println!("  Bulk Insert:    {:>10.0} ops/sec ({:.2}ms)", bulk_insert.ops_per_sec, bulk_insert.time_ms);
            results.toondb_wal.insert(format!("bulk_insert_{}", n), bulk_insert);

            // 2. Full Scan
            let start = Instant::now();
            let result = conn.query("users").execute()?;
            let full_scan = BenchmarkResult::new(result.rows_scanned, start.elapsed());
            println!("  Full Scan:      {:>10.0} ops/sec ({:.2}ms)", full_scan.ops_per_sec, full_scan.time_ms);
            results.toondb_wal.insert(format!("full_scan_{}", n), full_scan);
        }

        // ===== ToonDB (In-Memory) =====
        println!("\n--- ToonDB (In-Memory) ---");
        {
            let toon_path = temp_dir.path().join(format!("toondb_mem_{}", n));
            let conn = toondb::InMemoryConnection::open(&toon_path)?;

            conn.register_table(
                "users",
                &[
                    ("id".to_string(), toondb::connection::FieldType::Int64),
                    ("name".to_string(), toondb::connection::FieldType::Text),
                    ("email".to_string(), toondb::connection::FieldType::Text),
                    ("score".to_string(), toondb::connection::FieldType::Int64),
                ],
            )?;

            // Pre-create ToonValues
            let toon_values: Vec<[ToonValue; 4]> = (0..n)
                .map(|i| [
                    ToonValue::Int(i as i64),
                    ToonValue::Text(names[i].clone()),
                    ToonValue::Text(emails[i].clone()),
                    ToonValue::Int(scores[i]),
                ])
                .collect();

            // 1. Bulk Insert
            let start = Instant::now();
            let batch_size = 1000;
            for chunk_start in (0..n).step_by(batch_size) {
                let end = (chunk_start + batch_size).min(n);
                let mut builder = conn.insert_into("users");
                for i in chunk_start..end {
                    let row = &toon_values[i];
                    builder = builder
                        .set("id", row[0].clone())
                        .set("name", row[1].clone())
                        .set("email", row[2].clone())
                        .set("score", row[3].clone());
                    if i < end - 1 {
                        builder = builder.row();
                    }
                }
                builder.execute()?;
            }
            let bulk_insert = BenchmarkResult::new(n, start.elapsed());
            println!("  Bulk Insert:    {:>10.0} ops/sec ({:.2}ms)", bulk_insert.ops_per_sec, bulk_insert.time_ms);
            results.toondb_memory.insert(format!("bulk_insert_{}", n), bulk_insert);

            // 2. Full Scan
            let start = Instant::now();
            let rows = conn.find("users").all()?;
            let full_scan = BenchmarkResult::new(rows.len(), start.elapsed());
            println!("  Full Scan:      {:>10.0} ops/sec ({:.2}ms)", full_scan.ops_per_sec, full_scan.time_ms);
            results.toondb_memory.insert(format!("full_scan_{}", n), full_scan);
        }

        // ===== ToonDB Fast Mode =====
        println!("\n--- ToonDB (Fast Mode - WriteOptimized Policy) ---");
        {
            use toondb_storage::database::{ColumnDef, ColumnType, Database, DatabaseConfig, TableSchema};
            use toondb_storage::index_policy::IndexPolicy;

            let fast_path = temp_dir.path().join(format!("toondb_fast_{}", n));
            let config = DatabaseConfig {
                group_commit: false,
                default_index_policy: IndexPolicy::WriteOptimized, // Replaces enable_ordered_index: false
                ..Default::default()
            };
            let db = Database::open_with_config(&fast_path, config)?;

            let schema = TableSchema {
                name: "users".to_string(),
                columns: vec![
                    ColumnDef { name: "id".to_string(), col_type: ColumnType::Int64, nullable: false },
                    ColumnDef { name: "name".to_string(), col_type: ColumnType::Text, nullable: false },
                    ColumnDef { name: "email".to_string(), col_type: ColumnType::Text, nullable: false },
                    ColumnDef { name: "score".to_string(), col_type: ColumnType::Int64, nullable: false },
                ],
            };
            db.register_table(schema)?;

            // Pre-create ToonValues
            let toon_values: Vec<[ToonValue; 4]> = (0..n)
                .map(|i| [
                    ToonValue::Int(i as i64),
                    ToonValue::Text(names[i].clone()),
                    ToonValue::Text(emails[i].clone()),
                    ToonValue::Int(scores[i]),
                ])
                .collect();

            // 1. Bulk Insert
            let start = Instant::now();
            let txn = db.begin_transaction()?;
            for i in 0..n {
                let row = &toon_values[i];
                let values: &[Option<&ToonValue>] = &[Some(&row[0]), Some(&row[1]), Some(&row[2]), Some(&row[3])];
                db.insert_row_slice(txn, "users", i as u64, values)?;
            }
            db.commit(txn)?;
            let bulk_insert = BenchmarkResult::new(n, start.elapsed());
            println!("  Bulk Insert:    {:>10.0} ops/sec ({:.2}ms)", bulk_insert.ops_per_sec, bulk_insert.time_ms);
            results.toondb_fast.insert(format!("bulk_insert_{}", n), bulk_insert);
        }
    }

    Ok(results)
}

pub fn print_summary(results: &ComprehensiveResults) {
    println!("\n\n{'='*80}");
    println!("                    360° BENCHMARK SUMMARY REPORT");
    println!("{'='*80}\n");

    println!("## BULK INSERT PERFORMANCE (ops/sec)\n");
    println!("| Records | SQLite File | SQLite Memory | ToonDB WAL | ToonDB Memory | ToonDB Fast |");
    println!("|---------|-------------|---------------|------------|---------------|-------------|");

    for n in [1000, 10000, 100000] {
        let sf = results.sqlite_file.get(&format!("bulk_insert_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let sm = results.sqlite_memory.get(&format!("bulk_insert_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let tw = results.toondb_wal.get(&format!("bulk_insert_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let tm = results.toondb_memory.get(&format!("bulk_insert_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let tf = results.toondb_fast.get(&format!("bulk_insert_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        
        println!("| {:>7} | {:>11.0} | {:>13.0} | {:>10.0} | {:>13.0} | {:>11.0} |",
            format!("{}K", n / 1000), sf, sm, tw, tm, tf);
    }

    println!("\n## FULL SCAN PERFORMANCE (ops/sec)\n");
    println!("| Records | SQLite File | SQLite Memory | ToonDB WAL | ToonDB Memory |");
    println!("|---------|-------------|---------------|------------|---------------|");

    for n in [1000, 10000, 100000] {
        let sf = results.sqlite_file.get(&format!("full_scan_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let sm = results.sqlite_memory.get(&format!("full_scan_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let tw = results.toondb_wal.get(&format!("full_scan_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let tm = results.toondb_memory.get(&format!("full_scan_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        
        println!("| {:>7} | {:>11.0} | {:>13.0} | {:>10.0} | {:>13.0} |",
            format!("{}K", n / 1000), sf, sm, tw, tm);
    }

    println!("\n## POINT LOOKUP PERFORMANCE (ops/sec)\n");
    println!("| Records | SQLite File | SQLite Memory |");
    println!("|---------|-------------|---------------|");

    for n in [1000, 10000, 100000] {
        let sf = results.sqlite_file.get(&format!("point_lookup_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let sm = results.sqlite_memory.get(&format!("point_lookup_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        
        println!("| {:>7} | {:>11.0} | {:>13.0} |",
            format!("{}K", n / 1000), sf, sm);
    }

    println!("\n## UPDATE PERFORMANCE (50% of records, ops/sec)\n");
    println!("| Records | SQLite File | SQLite Memory |");
    println!("|---------|-------------|---------------|");

    for n in [1000, 10000, 100000] {
        let sf = results.sqlite_file.get(&format!("update_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let sm = results.sqlite_memory.get(&format!("update_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        
        println!("| {:>7} | {:>11.0} | {:>13.0} |",
            format!("{}K", n / 1000), sf, sm);
    }

    println!("\n## DELETE PERFORMANCE (50% of records, ops/sec)\n");
    println!("| Records | SQLite File | SQLite Memory |");
    println!("|---------|-------------|---------------|");

    for n in [1000, 10000, 100000] {
        let sf = results.sqlite_file.get(&format!("delete_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        let sm = results.sqlite_memory.get(&format!("delete_{}", n)).map(|r| r.ops_per_sec).unwrap_or(0.0);
        
        println!("| {:>7} | {:>11.0} | {:>13.0} |",
            format!("{}K", n / 1000), sf, sm);
    }

    // Calculate averages for 100K
    if let (Some(sf), Some(sm), Some(tw), Some(tm), Some(tf)) = (
        results.sqlite_file.get("bulk_insert_100000"),
        results.sqlite_memory.get("bulk_insert_100000"),
        results.toondb_wal.get("bulk_insert_100000"),
        results.toondb_memory.get("bulk_insert_100000"),
        results.toondb_fast.get("bulk_insert_100000"),
    ) {
        println!("\n## COMPARISON RATIOS (vs SQLite File @ 100K records)\n");
        println!("| Database | Insert Ratio | Notes |");
        println!("|----------|--------------|-------|");
        println!("| SQLite File | 100% | Baseline |");
        println!("| SQLite Memory | {:.0}% | +{:.0}% faster |", sm.ops_per_sec / sf.ops_per_sec * 100.0, (sm.ops_per_sec / sf.ops_per_sec - 1.0) * 100.0);
        println!("| ToonDB WAL | {:.0}% | -{:.0}% slower |", tw.ops_per_sec / sf.ops_per_sec * 100.0, (1.0 - tw.ops_per_sec / sf.ops_per_sec) * 100.0);
        println!("| ToonDB Memory | {:.0}% | {} |", tm.ops_per_sec / sf.ops_per_sec * 100.0, 
            if tm.ops_per_sec > sf.ops_per_sec { format!("+{:.0}% faster", (tm.ops_per_sec / sf.ops_per_sec - 1.0) * 100.0) }
            else { format!("-{:.0}% slower", (1.0 - tm.ops_per_sec / sf.ops_per_sec) * 100.0) });
        println!("| ToonDB Fast | {:.0}% | {} |", tf.ops_per_sec / sf.ops_per_sec * 100.0,
            if tf.ops_per_sec > sf.ops_per_sec { format!("+{:.0}% faster", (tf.ops_per_sec / sf.ops_per_sec - 1.0) * 100.0) }
            else { format!("-{:.0}% slower", (1.0 - tf.ops_per_sec / sf.ops_per_sec) * 100.0) });
    }
}
