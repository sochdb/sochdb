// Copyright 2025 Sushanth
// Head-to-head benchmark: ToonDB vs SQLite

use rusqlite::{params, Connection as SqliteConnection};
use std::time::Instant;
use tempfile::TempDir;
use toondb::EmbeddedConnection;
use toondb::connection::{KernelTableSchema, KernelColumnDef, KernelColumnType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                  HEAD-TO-HEAD: ToonDB vs SQLite");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let temp_dir = TempDir::new()?;
    
    // Test 1: Single-threaded insert
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: Single-Threaded Insert (100K records)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    benchmark_single_insert(&temp_dir, 100_000)?;
    
    // Test 2: Batch transaction sizes
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: Batch Transaction Sizes (50K total records)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    benchmark_batch_sizes(&temp_dir)?;
    
    // Test 3: Point lookups
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: Random Point Lookups (10K lookups from 50K rows)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    benchmark_point_lookups(&temp_dir, 50_000, 10_000)?;
    
    // Test 4: Full scan
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: Full Table Scan (100K rows)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    benchmark_full_scan(&temp_dir, 100_000)?;
    
    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                         BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");
    
    Ok(())
}

fn create_toondb(path: &std::path::Path) -> Result<EmbeddedConnection, Box<dyn std::error::Error>> {
    let config = toondb_storage::database::DatabaseConfig {
        group_commit: false,
        ..Default::default()
    };
    let conn = EmbeddedConnection::open_with_config(path, config)?;
    
    let schema = KernelTableSchema {
        name: "t".to_string(),
        columns: vec![
            KernelColumnDef { name: "id".to_string(), col_type: KernelColumnType::Int64, nullable: false },
            KernelColumnDef { name: "data".to_string(), col_type: KernelColumnType::Text, nullable: false },
        ],
    };
    conn.register_table(schema)?;
    Ok(conn)
}

fn benchmark_single_insert(temp_dir: &TempDir, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    // SQLite File
    let sqlite_path = temp_dir.path().join("sqlite_single.db");
    let mut sqlite = SqliteConnection::open(&sqlite_path)?;
    sqlite.pragma_update(None, "journal_mode", "WAL")?;
    sqlite.pragma_update(None, "synchronous", "NORMAL")?;
    sqlite.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, data TEXT)", [])?;
    
    let start = Instant::now();
    let tx = sqlite.transaction()?;
    {
        let mut stmt = tx.prepare("INSERT INTO t VALUES (?, ?)")?;
        for i in 0..n {
            stmt.execute(params![i as i64, format!("data{}", i)])?;
        }
    }
    tx.commit()?;
    let sqlite_time = start.elapsed();
    let sqlite_ops = n as f64 / sqlite_time.as_secs_f64();
    
    // SQLite In-Memory
    let mut sqlite_mem = SqliteConnection::open_in_memory()?;
    sqlite_mem.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, data TEXT)", [])?;
    
    let start = Instant::now();
    let tx = sqlite_mem.transaction()?;
    {
        let mut stmt = tx.prepare("INSERT INTO t VALUES (?, ?)")?;
        for i in 0..n {
            stmt.execute(params![i as i64, format!("data{}", i)])?;
        }
    }
    tx.commit()?;
    let sqlite_mem_time = start.elapsed();
    let sqlite_mem_ops = n as f64 / sqlite_mem_time.as_secs_f64();
    
    // ToonDB
    let toon_path = temp_dir.path().join("toon_single");
    let db = create_toondb(&toon_path)?;
    
    let start = Instant::now();
    db.begin()?;
    for i in 0..n {
        db.put(&format!("t/{}", i), format!("{{\"id\":{},\"data\":\"data{}\"}}", i, i).as_bytes())?;
    }
    db.commit()?;
    let toon_time = start.elapsed();
    let toon_ops = n as f64 / toon_time.as_secs_f64();
    
    println!("  SQLite File:     {:>10.0} ops/sec ({:>8.2}ms)", sqlite_ops, sqlite_time.as_secs_f64() * 1000.0);
    println!("  SQLite Memory:   {:>10.0} ops/sec ({:>8.2}ms)", sqlite_mem_ops, sqlite_mem_time.as_secs_f64() * 1000.0);
    println!("  ToonDB:          {:>10.0} ops/sec ({:>8.2}ms)", toon_ops, toon_time.as_secs_f64() * 1000.0);
    println!("  ToonDB vs File:  {:.1}%", (toon_ops / sqlite_ops) * 100.0);
    
    Ok(())
}

fn benchmark_batch_sizes(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let batch_sizes = [100, 500, 1000, 5000, 10000];
    let total = 50_000;
    
    println!("  {:>8} | {:>14} | {:>14} | {:>8}", "Batch", "SQLite File", "ToonDB", "Winner");
    println!("  ---------|----------------|----------------|----------");
    
    for batch_size in batch_sizes {
        let n_batches = total / batch_size;
        
        // SQLite
        let sqlite_path = temp_dir.path().join(format!("sqlite_batch_{}.db", batch_size));
        let mut sqlite = SqliteConnection::open(&sqlite_path)?;
        sqlite.pragma_update(None, "journal_mode", "WAL")?;
        sqlite.pragma_update(None, "synchronous", "NORMAL")?;
        sqlite.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, data TEXT)", [])?;
        
        let start = Instant::now();
        for batch in 0..n_batches {
            let tx = sqlite.transaction()?;
            {
                let mut stmt = tx.prepare("INSERT INTO t VALUES (?, ?)")?;
                for i in 0..batch_size {
                    let id = batch * batch_size + i;
                    stmt.execute(params![id as i64, format!("d{}", id)])?;
                }
            }
            tx.commit()?;
        }
        let sqlite_time = start.elapsed();
        let sqlite_ops = total as f64 / sqlite_time.as_secs_f64();
        
        // ToonDB
        let toon_path = temp_dir.path().join(format!("toon_batch_{}", batch_size));
        let db = create_toondb(&toon_path)?;
        
        let start = Instant::now();
        for batch in 0..n_batches {
            db.begin()?;
            for i in 0..batch_size {
                let id = batch * batch_size + i;
                db.put(&format!("t/{}", id), format!("{{\"id\":{},\"data\":\"d{}\"}}", id, id).as_bytes())?;
            }
            db.commit()?;
        }
        let toon_time = start.elapsed();
        let toon_ops = total as f64 / toon_time.as_secs_f64();
        
        let winner = if toon_ops > sqlite_ops { "ToonDB ✓" } else { "SQLite" };
        println!("  {:>8} | {:>12.0}/s | {:>12.0}/s | {}", batch_size, sqlite_ops, toon_ops, winner);
    }
    
    Ok(())
}

fn benchmark_point_lookups(temp_dir: &TempDir, n: usize, lookups: usize) -> Result<(), Box<dyn std::error::Error>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let lookup_ids: Vec<i64> = (0..lookups).map(|_| rng.gen_range(0..n) as i64).collect();
    
    // SQLite
    let sqlite_path = temp_dir.path().join("sqlite_lookup.db");
    let mut sqlite = SqliteConnection::open(&sqlite_path)?;
    sqlite.pragma_update(None, "journal_mode", "WAL")?;
    sqlite.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, data TEXT)", [])?;
    
    {
        let tx = sqlite.transaction()?;
        {
            let mut stmt = tx.prepare("INSERT INTO t VALUES (?, ?)")?;
            for i in 0..n {
                stmt.execute(params![i as i64, format!("data{}", i)])?;
            }
        }
        tx.commit()?;
    }
    
    let start = Instant::now();
    for &id in &lookup_ids {
        let _: String = sqlite.query_row("SELECT data FROM t WHERE id = ?", params![id], |r| r.get(0))?;
    }
    let sqlite_time = start.elapsed();
    let sqlite_ops = lookups as f64 / sqlite_time.as_secs_f64();
    
    // ToonDB
    let toon_path = temp_dir.path().join("toon_lookup");
    let db = create_toondb(&toon_path)?;
    
    db.begin()?;
    for i in 0..n {
        db.put(&format!("t/{}", i), format!("{{\"id\":{},\"data\":\"data{}\"}}", i, i).as_bytes())?;
    }
    db.commit()?;
    
    let start = Instant::now();
    for &id in &lookup_ids {
        let _ = db.get(&format!("t/{}", id));
    }
    let toon_time = start.elapsed();
    let toon_ops = lookups as f64 / toon_time.as_secs_f64();
    
    println!("  SQLite:  {:>10.0} ops/sec ({:>8.2}ms)", sqlite_ops, sqlite_time.as_secs_f64() * 1000.0);
    println!("  ToonDB:  {:>10.0} ops/sec ({:>8.2}ms)", toon_ops, toon_time.as_secs_f64() * 1000.0);
    
    let ratio = toon_ops / sqlite_ops;
    if ratio >= 1.0 {
        println!("  Winner:  ToonDB {:.1}x FASTER ✓", ratio);
    } else {
        println!("  ToonDB at {:.1}% of SQLite", ratio * 100.0);
    }
    
    Ok(())
}

fn benchmark_full_scan(temp_dir: &TempDir, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    // SQLite
    let sqlite_path = temp_dir.path().join("sqlite_scan.db");
    let mut sqlite = SqliteConnection::open(&sqlite_path)?;
    sqlite.pragma_update(None, "journal_mode", "WAL")?;
    sqlite.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, data TEXT)", [])?;
    
    {
        let tx = sqlite.transaction()?;
        {
            let mut stmt = tx.prepare("INSERT INTO t VALUES (?, ?)")?;
            for i in 0..n {
                stmt.execute(params![i as i64, format!("data{}", i)])?;
            }
        }
        tx.commit()?;
    }
    
    let start = Instant::now();
    let mut stmt = sqlite.prepare("SELECT * FROM t")?;
    let mut count = 0;
    let iter = stmt.query_map([], |_row| { Ok(()) })?;
    for _ in iter { count += 1; }
    let sqlite_time = start.elapsed();
    let sqlite_ops = count as f64 / sqlite_time.as_secs_f64();
    
    // ToonDB
    let toon_path = temp_dir.path().join("toon_scan");
    let db = create_toondb(&toon_path)?;
    
    db.begin()?;
    for i in 0..n {
        db.put(&format!("t/{}", i), format!("{{\"id\":{},\"data\":\"data{}\"}}", i, i).as_bytes())?;
    }
    db.commit()?;
    
    let start = Instant::now();
    let mut count = 0;
    if let Ok(iter) = db.scan("t/") {
        for _ in iter {
            count += 1;
        }
    }
    let toon_time = start.elapsed();
    let toon_ops = count as f64 / toon_time.as_secs_f64();
    
    println!("  SQLite:  {:>10.0} rows/sec ({:>8.2}ms) [{} rows]", sqlite_ops, sqlite_time.as_secs_f64() * 1000.0, n);
    println!("  ToonDB:  {:>10.0} rows/sec ({:>8.2}ms) [{} rows]", toon_ops, toon_time.as_secs_f64() * 1000.0, count);
    println!("  ToonDB at {:.1}% of SQLite scan speed", (toon_ops / sqlite_ops) * 100.0);
    
    Ok(())
}
