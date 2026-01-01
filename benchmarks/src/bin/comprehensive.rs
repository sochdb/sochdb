// Copyright 2025 Sushanth (https://github.com/sushanthpy)
// 360-degree Comprehensive ToonDB vs SQLite Benchmark

use rusqlite::{params, Connection as SqliteConnection};
use std::time::Instant;
use tempfile::TempDir;
use toondb::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n======================================================================");
    println!("   🔥 360° COMPREHENSIVE BENCHMARK: ToonDB vs SQLite 🔥");
    println!("======================================================================\n");
    println!("Test sizes: 1K, 10K, 100K records\n");

    let temp_dir = TempDir::new()?;
    let record_counts = [1_000, 10_000, 100_000];

    // Store results for final summary
    let mut all_results: Vec<BenchResult> = Vec::new();

    for n in record_counts {
        println!("\n==================================================");
        println!("  Testing with {} records", format_num(n));
        println!("==================================================");

        // Pre-generate test data
        let names: Vec<String> = (0..n).map(|i| format!("User {}", i)).collect();
        let emails: Vec<String> = (0..n).map(|i| format!("user{}@example.com", i)).collect();
        let scores: Vec<i64> = (0..n).map(|i| (i % 100) as i64).collect();

        // ========================
        // SQLite (File-based WAL)
        // ========================
        println!("\n📁 SQLite (File-based WAL)");
        println!("----------------------------------------");
        {
            let sqlite_path = temp_dir.path().join(format!("sqlite_file_{}.db", n));
            let mut conn = SqliteConnection::open(&sqlite_path)?;
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "synchronous", "NORMAL")?;
            conn.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, score INTEGER)",
                [],
            )?;

            // Bulk Insert
            let start = Instant::now();
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare("INSERT INTO users VALUES (?, ?, ?, ?)")?;
                for i in 0..n {
                    stmt.execute(params![i, &names[i], &emails[i], scores[i]])?;
                }
            }
            tx.commit()?;
            let insert_time = start.elapsed();
            let insert_rate = n as f64 / insert_time.as_secs_f64();
            println!("  Insert:      {:>12.0} ops/sec  ({:>8.2}ms)", insert_rate, insert_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite File", n, op: "Insert", ops_per_sec: insert_rate });

            // Point Lookup
            let lookup_count = n.min(10000);
            let (lookup_rate, lookup_time) = {
                let start = Instant::now();
                let mut stmt = conn.prepare("SELECT * FROM users WHERE id = ?")?;
                for i in 0..lookup_count {
                    let _: Option<(i64, String, String, i64)> = stmt.query_row(params![i], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                    }).ok();
                }
                let elapsed = start.elapsed();
                (lookup_count as f64 / elapsed.as_secs_f64(), elapsed)
            };
            println!("  Point Lookup:{:>12.0} ops/sec  ({:>8.2}ms)", lookup_rate, lookup_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite File", n, op: "Point Lookup", ops_per_sec: lookup_rate });

            // Full Scan
            let (scan_rate, scan_time, _scan_count) = {
                let start = Instant::now();
                let mut stmt = conn.prepare("SELECT * FROM users")?;
                let count = stmt.query_map([], |_| Ok(()))?.count();
                let elapsed = start.elapsed();
                (count as f64 / elapsed.as_secs_f64(), elapsed, count)
            };
            println!("  Full Scan:   {:>12.0} ops/sec  ({:>8.2}ms)", scan_rate, scan_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite File", n, op: "Full Scan", ops_per_sec: scan_rate });

            // Range Query (score between 20-40 = ~21% of data)
            let (range_rate, range_time, range_count) = {
                let start = Instant::now();
                let mut stmt = conn.prepare("SELECT * FROM users WHERE score BETWEEN 20 AND 40")?;
                let count = stmt.query_map([], |_| Ok(()))?.count();
                let elapsed = start.elapsed();
                (count as f64 / elapsed.as_secs_f64(), elapsed, count)
            };
            println!("  Range Query: {:>12.0} ops/sec  ({:>8.2}ms, {} rows)", range_rate, range_time.as_secs_f64() * 1000.0, range_count);
            all_results.push(BenchResult { db: "SQLite File", n, op: "Range Query", ops_per_sec: range_rate });

            // Update 50%
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("UPDATE users SET score = score + 1 WHERE id < ?", params![n / 2])?;
            tx.commit()?;
            let update_time = start.elapsed();
            let update_rate = (n / 2) as f64 / update_time.as_secs_f64();
            println!("  Update 50%:  {:>12.0} ops/sec  ({:>8.2}ms)", update_rate, update_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite File", n, op: "Update", ops_per_sec: update_rate });

            // Delete 50%
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM users WHERE id >= ?", params![n / 2])?;
            tx.commit()?;
            let delete_time = start.elapsed();
            let delete_rate = (n / 2) as f64 / delete_time.as_secs_f64();
            println!("  Delete 50%:  {:>12.0} ops/sec  ({:>8.2}ms)", delete_rate, delete_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite File", n, op: "Delete", ops_per_sec: delete_rate });
        }

        // ========================
        // SQLite (In-Memory)
        // ========================
        println!("\n💾 SQLite (In-Memory)");
        println!("----------------------------------------");
        {
            let mut conn = SqliteConnection::open_in_memory()?;
            conn.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, score INTEGER)",
                [],
            )?;

            // Bulk Insert
            let start = Instant::now();
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare("INSERT INTO users VALUES (?, ?, ?, ?)")?;
                for i in 0..n {
                    stmt.execute(params![i, &names[i], &emails[i], scores[i]])?;
                }
            }
            tx.commit()?;
            let insert_time = start.elapsed();
            let insert_rate = n as f64 / insert_time.as_secs_f64();
            println!("  Insert:      {:>12.0} ops/sec  ({:>8.2}ms)", insert_rate, insert_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite Memory", n, op: "Insert", ops_per_sec: insert_rate });

            // Point Lookup
            let lookup_count = n.min(10000);
            let (lookup_rate, lookup_time) = {
                let start = Instant::now();
                let mut stmt = conn.prepare("SELECT * FROM users WHERE id = ?")?;
                for i in 0..lookup_count {
                    let _: Option<(i64, String, String, i64)> = stmt.query_row(params![i], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                    }).ok();
                }
                let elapsed = start.elapsed();
                (lookup_count as f64 / elapsed.as_secs_f64(), elapsed)
            };
            println!("  Point Lookup:{:>12.0} ops/sec  ({:>8.2}ms)", lookup_rate, lookup_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite Memory", n, op: "Point Lookup", ops_per_sec: lookup_rate });

            // Full Scan
            let (scan_rate, scan_time) = {
                let start = Instant::now();
                let mut stmt = conn.prepare("SELECT * FROM users")?;
                let count = stmt.query_map([], |_| Ok(()))?.count();
                let elapsed = start.elapsed();
                (count as f64 / elapsed.as_secs_f64(), elapsed)
            };
            println!("  Full Scan:   {:>12.0} ops/sec  ({:>8.2}ms)", scan_rate, scan_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite Memory", n, op: "Full Scan", ops_per_sec: scan_rate });

            // Range Query
            let (range_rate, range_time, range_count) = {
                let start = Instant::now();
                let mut stmt = conn.prepare("SELECT * FROM users WHERE score BETWEEN 20 AND 40")?;
                let count = stmt.query_map([], |_| Ok(()))?.count();
                let elapsed = start.elapsed();
                (count as f64 / elapsed.as_secs_f64(), elapsed, count)
            };
            println!("  Range Query: {:>12.0} ops/sec  ({:>8.2}ms, {} rows)", range_rate, range_time.as_secs_f64() * 1000.0, range_count);
            all_results.push(BenchResult { db: "SQLite Memory", n, op: "Range Query", ops_per_sec: range_rate });

            // Update 50%
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("UPDATE users SET score = score + 1 WHERE id < ?", params![n / 2])?;
            tx.commit()?;
            let update_time = start.elapsed();
            let update_rate = (n / 2) as f64 / update_time.as_secs_f64();
            println!("  Update 50%:  {:>12.0} ops/sec  ({:>8.2}ms)", update_rate, update_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite Memory", n, op: "Update", ops_per_sec: update_rate });

            // Delete 50%
            let start = Instant::now();
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM users WHERE id >= ?", params![n / 2])?;
            tx.commit()?;
            let delete_time = start.elapsed();
            let delete_rate = (n / 2) as f64 / delete_time.as_secs_f64();
            println!("  Delete 50%:  {:>12.0} ops/sec  ({:>8.2}ms)", delete_rate, delete_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "SQLite Memory", n, op: "Delete", ops_per_sec: delete_rate });
        }

        // ========================
        // ToonDB (Embedded WAL)
        // ========================
        println!("\n🎯 ToonDB (Embedded WAL)");
        println!("----------------------------------------");
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

            // Bulk Insert
            let start = Instant::now();
            conn.begin()?;
            for i in 0..n {
                let row = &toon_values[i];
                let values: [Option<&ToonValue>; 4] = [Some(&row[0]), Some(&row[1]), Some(&row[2]), Some(&row[3])];
                conn.insert_row_slice("users", i as u64, &values)?;
            }
            conn.commit()?;
            let insert_time = start.elapsed();
            let insert_rate = n as f64 / insert_time.as_secs_f64();
            println!("  Insert:      {:>12.0} ops/sec  ({:>8.2}ms)", insert_rate, insert_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "ToonDB WAL", n, op: "Insert", ops_per_sec: insert_rate });

            // Full Scan
            let start = Instant::now();
            let result = conn.query("users").execute()?;
            let scan_time = start.elapsed();
            let scan_rate = result.rows_scanned as f64 / scan_time.as_secs_f64();
            println!("  Full Scan:   {:>12.0} ops/sec  ({:>8.2}ms)", scan_rate, scan_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "ToonDB WAL", n, op: "Full Scan", ops_per_sec: scan_rate });
        }

        // ========================
        // ToonDB (In-Memory)
        // ========================
        println!("\n⚡ ToonDB (In-Memory)");
        println!("----------------------------------------");
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

            // Bulk Insert
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
            let insert_time = start.elapsed();
            let insert_rate = n as f64 / insert_time.as_secs_f64();
            println!("  Insert:      {:>12.0} ops/sec  ({:>8.2}ms)", insert_rate, insert_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "ToonDB Memory", n, op: "Insert", ops_per_sec: insert_rate });

            // Full Scan
            let start = Instant::now();
            let rows = conn.find("users").all()?;
            let scan_time = start.elapsed();
            let scan_rate = rows.len() as f64 / scan_time.as_secs_f64();
            println!("  Full Scan:   {:>12.0} ops/sec  ({:>8.2}ms)", scan_rate, scan_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "ToonDB Memory", n, op: "Full Scan", ops_per_sec: scan_rate });
        }

        // ========================
        // ToonDB (Fast Mode)
        // ========================
        println!("\n🚀 ToonDB (Fast Mode - No Ordered Index)");
        println!("----------------------------------------");
        {
            use toondb_storage::database::{ColumnDef, ColumnType, Database, DatabaseConfig, TableSchema};

            let fast_path = temp_dir.path().join(format!("toondb_fast_{}", n));
            let config = DatabaseConfig {
                group_commit: false,
                default_index_policy: toondb_storage::index_policy::IndexPolicy::WriteOptimized,
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

            // Bulk Insert
            let start = Instant::now();
            let txn = db.begin_transaction()?;
            for i in 0..n {
                let row = &toon_values[i];
                let values: &[Option<&ToonValue>] = &[Some(&row[0]), Some(&row[1]), Some(&row[2]), Some(&row[3])];
                db.insert_row_slice(txn, "users", i as u64, values)?;
            }
            db.commit(txn)?;
            let insert_time = start.elapsed();
            let insert_rate = n as f64 / insert_time.as_secs_f64();
            println!("  Insert:      {:>12.0} ops/sec  ({:>8.2}ms)", insert_rate, insert_time.as_secs_f64() * 1000.0);
            all_results.push(BenchResult { db: "ToonDB Fast", n, op: "Insert", ops_per_sec: insert_rate });
        }
    }

    // Print final summary
    print_summary(&all_results);

    Ok(())
}

struct BenchResult {
    db: &'static str,
    n: usize,
    op: &'static str,
    ops_per_sec: f64,
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

fn print_summary(results: &[BenchResult]) {
    println!("\n\n================================================================================");
    println!("                    📊 360° BENCHMARK SUMMARY REPORT 📊");
    println!("================================================================================\n");

    // Insert comparison at 100K
    println!("## 📥 INSERT PERFORMANCE @ 100K records\n");
    println!("| Database       | ops/sec      | vs SQLite File |");
    println!("|----------------|--------------|----------------|");
    
    let sqlite_file_insert = results.iter()
        .find(|r| r.db == "SQLite File" && r.n == 100000 && r.op == "Insert")
        .map(|r| r.ops_per_sec)
        .unwrap_or(1.0);
    
    for db in ["SQLite File", "SQLite Memory", "ToonDB WAL", "ToonDB Memory", "ToonDB Fast"] {
        if let Some(r) = results.iter().find(|r| r.db == db && r.n == 100000 && r.op == "Insert") {
            let ratio = r.ops_per_sec / sqlite_file_insert * 100.0;
            println!("| {:14} | {:>12.0} | {:>13.0}% |", db, r.ops_per_sec, ratio);
        }
    }

    // Full Scan comparison at 100K
    println!("\n## 📖 FULL SCAN PERFORMANCE @ 100K records\n");
    println!("| Database       | ops/sec      | vs SQLite File |");
    println!("|----------------|--------------|----------------|");
    
    let sqlite_file_scan = results.iter()
        .find(|r| r.db == "SQLite File" && r.n == 100000 && r.op == "Full Scan")
        .map(|r| r.ops_per_sec)
        .unwrap_or(1.0);
    
    for db in ["SQLite File", "SQLite Memory", "ToonDB WAL", "ToonDB Memory"] {
        if let Some(r) = results.iter().find(|r| r.db == db && r.n == 100000 && r.op == "Full Scan") {
            let ratio = r.ops_per_sec / sqlite_file_scan * 100.0;
            println!("| {:14} | {:>12.0} | {:>13.0}% |", db, r.ops_per_sec, ratio);
        }
    }

    // Point Lookup at 100K
    println!("\n## 🔍 POINT LOOKUP PERFORMANCE @ 100K records\n");
    println!("| Database       | ops/sec      |");
    println!("|----------------|--------------|");
    
    for db in ["SQLite File", "SQLite Memory"] {
        if let Some(r) = results.iter().find(|r| r.db == db && r.n == 100000 && r.op == "Point Lookup") {
            println!("| {:14} | {:>12.0} |", db, r.ops_per_sec);
        }
    }

    // Update/Delete at 100K
    println!("\n## ✏️ UPDATE/DELETE PERFORMANCE @ 100K records (50% of rows)\n");
    println!("| Database       | Update ops/s | Delete ops/s |");
    println!("|----------------|--------------|--------------|");
    
    for db in ["SQLite File", "SQLite Memory"] {
        let update = results.iter().find(|r| r.db == db && r.n == 100000 && r.op == "Update").map(|r| r.ops_per_sec).unwrap_or(0.0);
        let delete = results.iter().find(|r| r.db == db && r.n == 100000 && r.op == "Delete").map(|r| r.ops_per_sec).unwrap_or(0.0);
        println!("| {:14} | {:>12.0} | {:>12.0} |", db, update, delete);
    }

    // Scalability
    println!("\n## 📈 SCALABILITY (Insert ops/sec at different sizes)\n");
    println!("| Database       |      1K      |     10K      |    100K      |");
    println!("|----------------|--------------|--------------|--------------|");
    
    for db in ["SQLite File", "SQLite Memory", "ToonDB WAL", "ToonDB Memory", "ToonDB Fast"] {
        let v1k = results.iter().find(|r| r.db == db && r.n == 1000 && r.op == "Insert").map(|r| r.ops_per_sec).unwrap_or(0.0);
        let v10k = results.iter().find(|r| r.db == db && r.n == 10000 && r.op == "Insert").map(|r| r.ops_per_sec).unwrap_or(0.0);
        let v100k = results.iter().find(|r| r.db == db && r.n == 100000 && r.op == "Insert").map(|r| r.ops_per_sec).unwrap_or(0.0);
        println!("| {:14} | {:>12.0} | {:>12.0} | {:>12.0} |", db, v1k, v10k, v100k);
    }

    // Key insights
    println!("\n## 🔑 KEY INSIGHTS\n");
    
    let toondb_mem = results.iter().find(|r| r.db == "ToonDB Memory" && r.n == 100000 && r.op == "Insert").map(|r| r.ops_per_sec).unwrap_or(0.0);
    let toondb_fast = results.iter().find(|r| r.db == "ToonDB Fast" && r.n == 100000 && r.op == "Insert").map(|r| r.ops_per_sec).unwrap_or(0.0);
    let sqlite_mem = results.iter().find(|r| r.db == "SQLite Memory" && r.n == 100000 && r.op == "Insert").map(|r| r.ops_per_sec).unwrap_or(0.0);
    
    println!("• ToonDB Memory Insert: {:.0}% of SQLite Memory ({:.0}K vs {:.0}K ops/sec)", 
        toondb_mem / sqlite_mem * 100.0, toondb_mem / 1000.0, sqlite_mem / 1000.0);
    println!("• ToonDB Fast Mode Insert: {:.0}% of SQLite File ({:.0}K vs {:.0}K ops/sec)",
        toondb_fast / sqlite_file_insert * 100.0, toondb_fast / 1000.0, sqlite_file_insert / 1000.0);
    
    let toondb_scan = results.iter().find(|r| r.db == "ToonDB Memory" && r.n == 100000 && r.op == "Full Scan").map(|r| r.ops_per_sec).unwrap_or(1.0);
    println!("• ToonDB Read Gap: {:.1}x slower than SQLite (needs optimization)",
        sqlite_file_scan / toondb_scan);

    println!("\n## ✅ TOONDB ADVANTAGES\n");
    println!("• Lock-free concurrent writers (vs SQLite single-writer)");
    println!("• MVCC with SSI isolation (vs SQLite table locks)");
    println!("• Native vector search (HNSW index)");
    println!("• LLM-native MCP protocol support");
    println!("• Streaming results for large datasets");

    println!("\n================================================================================");
    println!("                         END OF BENCHMARK REPORT");
    println!("================================================================================\n");
}
