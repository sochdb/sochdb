// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use rusqlite::{params, Connection as SqliteConnection};
use std::time::Instant;
use tempfile::TempDir;
use toondb::prelude::*;

/// Pre-generated test data to eliminate allocation overhead from timing
struct TestData {
    /// Pre-formatted strings for SQLite
    names: Vec<String>,
    emails: Vec<String>,
    /// Pre-created ToonValues for ToonDB
    toon_values: Vec<[ToonValue; 4]>,
    /// Pre-formatted JSON for raw API
    json_values: Vec<String>,
    /// Pre-formatted keys for raw API
    keys: Vec<String>,
}

impl TestData {
    fn generate(n: usize) -> Self {
        let names: Vec<String> = (0..n).map(|i| format!("User {}", i)).collect();
        let emails: Vec<String> = (0..n).map(|i| format!("user{}@example.com", i)).collect();

        let toon_values: Vec<[ToonValue; 4]> = (0..n)
            .map(|i| {
                [
                    ToonValue::Int(i as i64),
                    ToonValue::Text(format!("User {}", i)),
                    ToonValue::Text(format!("user{}@example.com", i)),
                    ToonValue::Int((i % 100) as i64),
                ]
            })
            .collect();

        let json_values: Vec<String> = (0..n)
            .map(|i| {
                format!(
                "{{\"id\":{},\"name\":\"User {}\",\"email\":\"user{}@example.com\",\"score\":{}}}",
                i, i, i, i % 100
            )
            })
            .collect();

        let keys: Vec<String> = (0..n).map(|i| format!("users/{}", i)).collect();

        Self {
            names,
            emails,
            toon_values,
            json_values,
            keys,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100_000;
    let batch_size = 1000; // Used for some benchmarks
    println!("Benchmarking with {} records...", n);

    // Pre-generate ALL test data before any timing begins
    println!("\nPre-generating test data (not timed)...");
    let test_data = TestData::generate(n);
    println!("Test data ready: {} records pre-allocated", n);

    let temp_dir = TempDir::new()?;
    let toon_path = temp_dir.path().join("toondb_data");
    let sqlite_path = temp_dir.path().join("sqlite.db");

    // --- SQLite Benchmark ---
    println!("\n--- SQLite (File) ---");
    let mut sqlite_conn = SqliteConnection::open(&sqlite_path)?;

    // Optimization for SQLite to be competitive
    sqlite_conn.pragma_update(None, "journal_mode", "WAL")?;
    sqlite_conn.pragma_update(None, "synchronous", "NORMAL")?;

    sqlite_conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, score INTEGER)",
        [],
    )?;

    // Insert - using pre-generated data
    let start = Instant::now();
    let tx = sqlite_conn.transaction()?;
    {
        let mut stmt =
            tx.prepare("INSERT INTO users (id, name, email, score) VALUES (?, ?, ?, ?)")?;
        for i in 0..n {
            // Use pre-generated strings instead of format!() in hot path
            stmt.execute(params![
                i,
                &test_data.names[i],
                &test_data.emails[i],
                i % 100
            ])?;
        }
    }
    tx.commit()?;
    let sqlite_insert_time = start.elapsed();
    println!("Insert: {:.2?}", sqlite_insert_time);
    println!(
        "Insert Rate: {:.0} ops/sec",
        n as f64 / sqlite_insert_time.as_secs_f64()
    );

    // Read
    let start = Instant::now();
    let mut stmt = sqlite_conn.prepare("SELECT * FROM users")?;
    let user_iter = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?,
        ))
    })?;
    let mut count = 0;
    for _ in user_iter {
        count += 1;
    }
    let sqlite_read_time = start.elapsed();
    println!("Read (Scan): {:.2?} ({} rows)", sqlite_read_time, count);
    println!(
        "Read Rate: {:.0} ops/sec",
        count as f64 / sqlite_read_time.as_secs_f64()
    );

    // --- SQLite (In-Memory) ---
    println!("\n--- SQLite (In-Memory) ---");
    let mut sqlite_mem_conn = SqliteConnection::open_in_memory()?;

    sqlite_mem_conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, score INTEGER)",
        [],
    )?;

    // Insert - using pre-generated data
    let start = Instant::now();
    let tx = sqlite_mem_conn.transaction()?;
    {
        let mut stmt =
            tx.prepare("INSERT INTO users (id, name, email, score) VALUES (?, ?, ?, ?)")?;
        for i in 0..n {
            stmt.execute(params![
                i,
                &test_data.names[i],
                &test_data.emails[i],
                i % 100
            ])?;
        }
    }
    tx.commit()?;
    let sqlite_mem_insert_time = start.elapsed();
    println!("Insert: {:.2?}", sqlite_mem_insert_time);
    println!(
        "Insert Rate: {:.0} ops/sec",
        n as f64 / sqlite_mem_insert_time.as_secs_f64()
    );

    // Read
    let start = Instant::now();
    let mut stmt = sqlite_mem_conn.prepare("SELECT * FROM users")?;
    let user_iter = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?,
        ))
    })?;
    let mut count = 0;
    for _ in user_iter {
        count += 1;
    }
    let sqlite_mem_read_time = start.elapsed();
    println!("Read (Scan): {:.2?} ({} rows)", sqlite_mem_read_time, count);
    println!(
        "Read Rate: {:.0} ops/sec",
        count as f64 / sqlite_mem_read_time.as_secs_f64()
    );

    // --- ToonDB (Embedded WAL) ---
    println!("\n--- ToonDB (Embedded WAL) ---");
    // Use EmbeddedConnection which wraps the Database kernel (SQLite-style persistence)
    let embedded_path = temp_dir.path().join("toondb_embedded");

    // Enable group commit for stable throughput (amortized fsync)
    let config = toondb_storage::database::DatabaseConfig {
        group_commit: true,
        ..Default::default()
    };
    let embedded_conn =
        toondb::EmbeddedConnection::open_with_config(&embedded_path, config)?;

    // Register table
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
    embedded_conn.register_table(schema)?;

    // Insert - using pre-generated data
    let start = Instant::now();
    let mut total_inserted = 0;

    // Single transaction like SQLite (for fair comparison)
    embedded_conn.begin()?;

    for i in 0..n {
        // Use slice-based zero-allocation API for fair comparison with SQLite
        // SQLite's prepared statement doesn't build HashMaps either
        let row_data = &test_data.toon_values[i];
        let values: [Option<&ToonValue>; 4] = [
            Some(&row_data[0]),
            Some(&row_data[1]),
            Some(&row_data[2]),
            Some(&row_data[3]),
        ];

        embedded_conn.insert_row_slice("users", i as u64, &values)?;
        total_inserted += 1;
    }

    embedded_conn.commit()?;

    let embedded_insert_time = start.elapsed();
    println!("Insert: {:.2?}", embedded_insert_time);
    println!(
        "Insert Rate: {:.0} ops/sec",
        total_inserted as f64 / embedded_insert_time.as_secs_f64()
    );

    // Read (Scan)
    let start = Instant::now();
    // Use query builder
    let result = embedded_conn.query("users").execute()?;
    // Count rows
    let count = result.rows_scanned;

    let embedded_read_time = start.elapsed();
    println!("Read (Scan): {:.2?} ({} rows)", embedded_read_time, count);
    println!(
        "Read Rate: {:.0} ops/sec",
        count as f64 / embedded_read_time.as_secs_f64()
    );

    // --- ToonDB (In-Memory) ---
    println!("\n--- ToonDB (In-Memory) ---");
    // Using InMemoryConnection (alias for ToonConnection)
    // This is purely in-memory, no WAL, comparable to SQLite In-Memory
    let toon_mem_conn = toondb::InMemoryConnection::open(&toon_path)?; // Path is ignored for in-memory or used as identifier

    // Register table
    // Note: InMemoryConnection might need explicit registration depending on impl
    // The previous run showed it worked with registration.
    toon_mem_conn.register_table(
        "users",
        &[
            (
                "id".to_string(),
                toondb::connection::FieldType::Int64,
            ),
            (
                "name".to_string(),
                toondb::connection::FieldType::Text,
            ),
            (
                "email".to_string(),
                toondb::connection::FieldType::Text,
            ),
            (
                "score".to_string(),
                toondb::connection::FieldType::Int64,
            ),
        ],
    )?;

    // Insert - using pre-generated data
    let start = Instant::now();
    let mut total_inserted = 0;

    for chunk_start in (0..n).step_by(batch_size) {
        let end = (chunk_start + batch_size).min(n);
        let mut builder = toon_mem_conn.insert_into("users");

        for i in chunk_start..end {
            // Use pre-generated values (avoids format! allocation)
            let row_data = &test_data.toon_values[i];
            builder = builder
                .set("id", row_data[0].clone())
                .set("name", row_data[1].clone())
                .set("email", row_data[2].clone())
                .set("score", row_data[3].clone());

            if i < end - 1 {
                builder = builder.row();
            }
        }
        builder.execute()?;
        total_inserted += end - chunk_start;
    }

    let toon_mem_insert_time = start.elapsed();
    println!("Insert: {:.2?}", toon_mem_insert_time);
    println!(
        "Insert Rate: {:.0} ops/sec",
        total_inserted as f64 / toon_mem_insert_time.as_secs_f64()
    );

    // Read
    let start = Instant::now();
    let rows = toon_mem_conn.find("users").all()?;
    let toon_mem_read_time = start.elapsed();
    println!(
        "Read (Scan): {:.2?} ({} rows)",
        toon_mem_read_time,
        rows.len()
    );
    println!(
        "Read Rate: {:.0} ops/sec",
        rows.len() as f64 / toon_mem_read_time.as_secs_f64()
    );

    // --- Raw DurableStorage Layer ---
    println!("\n--- Raw DurableStorage (Isolated) ---");
    {
        use toondb_storage::durable_storage::DurableStorage;

        let raw_path = temp_dir.path().join("raw_durable");
        std::fs::create_dir_all(&raw_path).ok();

        // Use group commit like Database default
        let storage = DurableStorage::open_with_group_commit(&raw_path)?;
        storage.set_sync_mode(1); // NORMAL mode

        let start = Instant::now();

        // Pre-generate keys/values for DurableStorage benchmark
        let durable_keys: Vec<String> = (0..n).map(|i| format!("key_{:08}", i)).collect();
        let durable_values: Vec<String> = (0..n).map(|i| format!("value_for_key_{}", i)).collect();

        // Same pattern as embedded: 100 transactions of 1000 writes each
        for chunk_start in (0..n).step_by(batch_size) {
            let end = (chunk_start + batch_size).min(n);
            let txn_id = storage.begin_transaction()?;

            for i in chunk_start..end {
                // Use pre-generated keys/values
                storage.write_refs(
                    txn_id,
                    durable_keys[i].as_bytes(),
                    durable_values[i].as_bytes(),
                )?;
            }

            storage.commit(txn_id)?;
        }

        let raw_insert_time = start.elapsed();
        println!("Insert (100 txns × 1000): {:.2?}", raw_insert_time);
        println!(
            "Insert Rate: {:.0} ops/sec",
            n as f64 / raw_insert_time.as_secs_f64()
        );
    }

    // --- ToonDB via put_raw (minimal overhead) ---
    println!("\n--- ToonDB (put_raw - minimal overhead, single txn) ---");
    {
        use toondb_storage::database::{Database, DatabaseConfig};

        let raw_path = temp_dir.path().join("raw_db");
        let config = DatabaseConfig {
            group_commit: true, // Enable group commit for stable throughput
            ..Default::default()
        };
        let db = Database::open_with_config(&raw_path, config)?;

        let start = Instant::now();

        // Single transaction like SQLite
        let txn = db.begin_transaction()?;

        for i in 0..n {
            // Use pre-generated keys and JSON values
            let key = &test_data.keys[i];
            let value = &test_data.json_values[i];
            db.put_raw(txn, key.as_bytes(), value.as_bytes())?;
        }

        db.commit(txn)?;

        let raw_insert_time = start.elapsed();
        println!("Insert (1 txn × 100k): {:.2?}", raw_insert_time);
        println!(
            "Insert Rate: {:.0} ops/sec",
            n as f64 / raw_insert_time.as_secs_f64()
        );
    }

    // --- ToonDB insert_row_slice (zero-allocation API) ---
    println!("\n--- ToonDB (insert_row_slice - zero alloc) ---");
    {
        use toondb_storage::database::{
            ColumnDef, ColumnType, Database, DatabaseConfig, TableSchema,
        };

        let slice_path = temp_dir.path().join("slice_db");
        let config = DatabaseConfig {
            group_commit: true, // Enable group commit for stable throughput
            ..Default::default()
        };
        let db = Database::open_with_config(&slice_path, config)?;

        // Register table schema
        let schema = TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                },
                ColumnDef {
                    name: "name".to_string(),
                    col_type: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "email".to_string(),
                    col_type: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "score".to_string(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                },
            ],
        };
        db.register_table(schema.clone())?;

        // Register second table for single-txn test
        let mut schema2 = schema.clone();
        schema2.name = "users2".to_string();
        db.register_table(schema2)?;

        let start = Instant::now();

        // Use pre-generated ToonValues (no allocation in hot path)
        for chunk_start in (0..n).step_by(batch_size) {
            let end = (chunk_start + batch_size).min(n);
            let txn = db.begin_transaction()?;

            for i in chunk_start..end {
                // Use pre-generated values - ZERO allocation here
                let row = &test_data.toon_values[i];
                let values: &[Option<&ToonValue>] =
                    &[Some(&row[0]), Some(&row[1]), Some(&row[2]), Some(&row[3])];
                db.insert_row_slice(txn, "users", i as u64, values)?;
            }

            db.commit(txn)?;
        }

        let slice_insert_time = start.elapsed();
        println!("Insert (100 txns × 1000): {:.2?}", slice_insert_time);
        println!(
            "Insert Rate: {:.0} ops/sec",
            n as f64 / slice_insert_time.as_secs_f64()
        );

        // Profile single-transaction mode for fair comparison with SQLite
        {
            let single_txn_start = Instant::now();
            let txn = db.begin_transaction()?;
            for i in 0..n {
                let row = &test_data.toon_values[i];
                let values: &[Option<&ToonValue>] =
                    &[Some(&row[0]), Some(&row[1]), Some(&row[2]), Some(&row[3])];
                db.insert_row_slice(txn, "users2", (n + i) as u64, values)?;
            }
            db.commit(txn)?;
            let single_txn_time = single_txn_start.elapsed();
            println!("Insert (1 txn × {}): {:.2?}", n, single_txn_time);
            println!(
                "Single-txn Rate: {:.0} ops/sec",
                n as f64 / single_txn_time.as_secs_f64()
            );
        }
    }

    // --- ToonDB Fast Mode (no ordered index) ---
    println!("\n--- ToonDB Fast Mode (no ordered index) ---");
    {
        use toondb_storage::database::{
            ColumnDef, ColumnType, Database, DatabaseConfig, TableSchema,
        };
        use toondb_storage::index_policy::IndexPolicy;

        let fast_path = temp_dir.path().join("fast_db");
        let config = DatabaseConfig {
            group_commit: true, // Enable group commit for stable throughput
            default_index_policy: IndexPolicy::WriteOptimized, // FAST MODE
            ..Default::default()
        };
        let db = Database::open_with_config(&fast_path, config)?;

        // Register table schema
        let schema = TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                },
                ColumnDef {
                    name: "name".to_string(),
                    col_type: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "email".to_string(),
                    col_type: ColumnType::Text,
                    nullable: false,
                },
                ColumnDef {
                    name: "score".to_string(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                },
            ],
        };
        db.register_table(schema)?;

        let start = Instant::now();
        let txn = db.begin_transaction()?;
        for i in 0..n {
            let row = &test_data.toon_values[i];
            let values: &[Option<&ToonValue>] =
                &[Some(&row[0]), Some(&row[1]), Some(&row[2]), Some(&row[3])];
            db.insert_row_slice(txn, "users", i as u64, values)?;
        }
        db.commit(txn)?;
        let fast_time = start.elapsed();
        println!("Insert (1 txn × {}): {:.2?}", n, fast_time);
        println!(
            "Insert Rate: {:.0} ops/sec",
            n as f64 / fast_time.as_secs_f64()
        );
    }

    // --- Detailed Profiling ---
    println!("\n--- PROFILING: Component Breakdown ---");
    {
        use toondb_storage::durable_storage::DurableStorage;

        let profile_path = temp_dir.path().join("profile_db");
        std::fs::create_dir_all(&profile_path).ok();
        let storage = DurableStorage::open(&profile_path)?;
        storage.set_sync_mode(1);

        let profile_n = 100_000;

        // Pre-generate data
        let keys: Vec<Vec<u8>> = (0..profile_n)
            .map(|i| format!("key_{:08}", i).into_bytes())
            .collect();
        let values: Vec<Vec<u8>> = (0..profile_n)
            .map(|i| format!("value_{}", i).into_bytes())
            .collect();

        // Single large transaction
        let txn_id = storage.begin_transaction()?;

        // Time write phase
        let write_start = Instant::now();
        for i in 0..profile_n {
            storage.write_refs(txn_id, &keys[i], &values[i])?;
        }
        let write_time = write_start.elapsed();

        // Time commit phase
        let commit_start = Instant::now();
        storage.commit(txn_id)?;
        let commit_time = commit_start.elapsed();

        let total = write_time + commit_time;
        println!(
            "Write phase:  {:>8.2?} ({:>5.0} ns/op) - {:.1}%",
            write_time,
            write_time.as_nanos() as f64 / profile_n as f64,
            write_time.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
        );
        println!(
            "Commit phase: {:>8.2?} ({:>5.0} ns/op) - {:.1}%",
            commit_time,
            commit_time.as_nanos() as f64 / profile_n as f64,
            commit_time.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
        );
        println!(
            "Total:        {:>8.2?} ({:>5.0} ns/op)",
            total,
            total.as_nanos() as f64 / profile_n as f64
        );
        println!(
            "Throughput:   {:.0} ops/sec",
            profile_n as f64 / total.as_secs_f64()
        );

        // Also test with ordered index disabled (saves ~134 ns/op)
        println!("\n--- No-SkipMap Mode (ordered_index=false) ---");
        let noindex_path = temp_dir.path().join("profile_db_noindex");
        std::fs::create_dir_all(&noindex_path).ok();
        let storage_fast = DurableStorage::open_with_config(&noindex_path, false)?;
        storage_fast.set_sync_mode(1);

        let txn_id2 = storage_fast.begin_transaction()?;
        let write_start2 = Instant::now();
        for i in 0..profile_n {
            storage_fast.write_refs(txn_id2, &keys[i], &values[i])?;
        }
        let write_time2 = write_start2.elapsed();

        let commit_start2 = Instant::now();
        storage_fast.commit(txn_id2)?;
        let commit_time2 = commit_start2.elapsed();

        let total2 = write_time2 + commit_time2;
        println!(
            "Write phase:  {:>8.2?} ({:>5.0} ns/op) - {:.1}%",
            write_time2,
            write_time2.as_nanos() as f64 / profile_n as f64,
            write_time2.as_nanos() as f64 / total2.as_nanos() as f64 * 100.0
        );
        println!(
            "Commit phase: {:>8.2?} ({:>5.0} ns/op) - {:.1}%",
            commit_time2,
            commit_time2.as_nanos() as f64 / profile_n as f64,
            commit_time2.as_nanos() as f64 / total2.as_nanos() as f64 * 100.0
        );
        println!(
            "Total:        {:>8.2?} ({:>5.0} ns/op)",
            total2,
            total2.as_nanos() as f64 / profile_n as f64
        );
        println!(
            "Throughput:   {:.0} ops/sec",
            profile_n as f64 / total2.as_secs_f64()
        );
        println!(
            "Speedup:      {:.1}%",
            (1.0 - total2.as_secs_f64() / total.as_secs_f64()) * 100.0
        );
    }

    // --- Profile write_refs components ---
    println!("\n--- PROFILING: write_refs() Breakdown ---");
    {
        use crossbeam_skiplist::SkipMap;
        use dashmap::DashMap;
        use std::collections::HashSet;
        use toondb_storage::txn_wal::TxnWalBuffer;

        let profile_n = 100_000usize;

        // Pre-generate data
        let keys: Vec<Vec<u8>> = (0..profile_n)
            .map(|i| format!("key_{:08}", i).into_bytes())
            .collect();
        let values: Vec<Vec<u8>> = (0..profile_n)
            .map(|i| format!("value_{}", i).into_bytes())
            .collect();

        // Test 1: Just TxnWalBuffer append (no locks, no storage)
        let mut buffer = TxnWalBuffer::new(1);
        let buf_start = Instant::now();
        for i in 0..profile_n {
            buffer.append(&keys[i], &values[i]);
        }
        let buf_time = buf_start.elapsed();

        // Test 2: DashMap entry access pattern
        let map: DashMap<u64, Vec<u8>> = DashMap::new();
        map.insert(1, Vec::new());
        let dash_start = Instant::now();
        for _ in 0..profile_n {
            map.entry(1).or_default();
        }
        let dash_time = dash_start.elapsed();

        // Test 3: HashSet insert (simulating write_set tracking)
        let mut set: HashSet<Vec<u8>> = HashSet::new();
        let set_start = Instant::now();
        for key in keys.iter().take(profile_n) {
            set.insert(key.clone());
        }
        let set_time = set_start.elapsed();

        // Test 4: SkipMap insert (simulating memtable ordered index)
        let skipmap: SkipMap<Vec<u8>, ()> = SkipMap::new();
        let skip_start = Instant::now();
        for key in keys.iter().take(profile_n) {
            skipmap.insert(key.clone(), ());
        }
        let skip_time = skip_start.elapsed();

        // Test 5: DashMap insert (simulating memtable data)
        let datamap: DashMap<Vec<u8>, Vec<u8>> = DashMap::new();
        let data_start = Instant::now();
        for (key, value) in keys.iter().zip(values.iter()).take(profile_n) {
            datamap.insert(key.clone(), value.clone());
        }
        let data_time = data_start.elapsed();

        // Test 6: Vec clone overhead
        let clone_start = Instant::now();
        for (key, value) in keys.iter().zip(values.iter()).take(profile_n) {
            let _ = key.clone();
            let _ = value.clone();
        }
        let clone_time = clone_start.elapsed();

        println!(
            "TxnWalBuffer.append():  {:>8.2?} ({:>5.0} ns/op)",
            buf_time,
            buf_time.as_nanos() as f64 / profile_n as f64
        );
        println!(
            "DashMap.entry():        {:>8.2?} ({:>5.0} ns/op)",
            dash_time,
            dash_time.as_nanos() as f64 / profile_n as f64
        );
        println!(
            "HashSet.insert(clone):  {:>8.2?} ({:>5.0} ns/op)",
            set_time,
            set_time.as_nanos() as f64 / profile_n as f64
        );
        println!(
            "SkipMap.insert(clone):  {:>8.2?} ({:>5.0} ns/op)",
            skip_time,
            skip_time.as_nanos() as f64 / profile_n as f64
        );
        println!(
            "DashMap.insert(clone):  {:>8.2?} ({:>5.0} ns/op)",
            data_time,
            data_time.as_nanos() as f64 / profile_n as f64
        );
        println!(
            "Vec clone (key+value):  {:>8.2?} ({:>5.0} ns/op)",
            clone_time,
            clone_time.as_nanos() as f64 / profile_n as f64
        );

        // Test 7: Bloom filter insert
        use toondb_storage::durable_storage::SsiBloomFilter;
        let mut bloom = SsiBloomFilter::new(profile_n);
        let bloom_start = Instant::now();
        for key in keys.iter().take(profile_n) {
            bloom.insert(key);
        }
        let bloom_time = bloom_start.elapsed();
        println!(
            "BloomFilter.insert():   {:>8.2?} ({:>5.0} ns/op)",
            bloom_time,
            bloom_time.as_nanos() as f64 / profile_n as f64
        );

        // Test 8: Mutex + Vec push (simulating dirty_list)
        let dirty_mutex: parking_lot::Mutex<Vec<Vec<u8>>> = parking_lot::Mutex::new(Vec::new());
        let dirty_start = Instant::now();
        for key in keys.iter().take(profile_n) {
            dirty_mutex.lock().push(key.clone());
        }
        let dirty_time = dirty_start.elapsed();
        println!(
            "DirtyList (mutex+push): {:>8.2?} ({:>5.0} ns/op)",
            dirty_time,
            dirty_time.as_nanos() as f64 / profile_n as f64
        );

        let estimated_total =
            buf_time + dash_time + set_time + skip_time + data_time + bloom_time + dirty_time;
        println!(
            "\nEstimated write_refs:   {:>8.2?} ({:>5.0} ns/op)",
            estimated_total,
            estimated_total.as_nanos() as f64 / profile_n as f64
        );
    }

    // --- Batch Write API Benchmark ---
    println!("\n--- PROFILING: put_batch vs individual writes ---");
    {
        use toondb_storage::database::{Database, DatabaseConfig};
        use toondb_storage::index_policy::IndexPolicy;

        let batch_path = temp_dir.path().join("batch_test_db");
        let config = DatabaseConfig {
            group_commit: true, // Enable for stable throughput
            default_index_policy: IndexPolicy::WriteOptimized, // Fast mode
            ..Default::default()
        };
        let _db = Database::open_with_config(&batch_path, config)?;

        let profile_n = 100_000;
        let batch_sizes = [100, 500, 1000, 5000];

        // Pre-generate key-value pairs
        let kvs: Vec<(Vec<u8>, Vec<u8>)> = (0..profile_n)
            .map(|i| {
                (
                    format!("batch_key_{:08}", i).into_bytes(),
                    format!("batch_value_{}", i).into_bytes(),
                )
            })
            .collect();

        // Test 1: Individual writes (baseline)
        {
            let individual_path = temp_dir.path().join("individual_test_db");
            let db_ind = Database::open_with_config(&individual_path, DatabaseConfig {
                group_commit: true, // Enable for stable throughput
                default_index_policy: IndexPolicy::WriteOptimized,
                ..Default::default()
            })?;

            let txn = db_ind.begin_transaction()?;
            let start = Instant::now();
            for (k, v) in &kvs {
                db_ind.put(txn, k, v)?;
            }
            db_ind.commit(txn)?;
            let individual_time = start.elapsed();

            println!(
                "Individual put():       {:>8.2?} ({:>5.0} ns/op) - {:.0} ops/sec",
                individual_time,
                individual_time.as_nanos() as f64 / profile_n as f64,
                profile_n as f64 / individual_time.as_secs_f64()
            );
        }

        // Test 2: Batched writes with various batch sizes
        for &batch_size in &batch_sizes {
            let batch_path_i = temp_dir.path().join(format!("batch_test_db_{}", batch_size));
            let db_batch = Database::open_with_config(&batch_path_i, DatabaseConfig {
                group_commit: true, // Enable for stable throughput
                default_index_policy: IndexPolicy::WriteOptimized,
                ..Default::default()
            })?;

            let txn = db_batch.begin_transaction()?;
            let start = Instant::now();

            for chunk in kvs.chunks(batch_size) {
                let refs: Vec<(&[u8], &[u8])> = chunk
                    .iter()
                    .map(|(k, v)| (k.as_slice(), v.as_slice()))
                    .collect();
                db_batch.put_batch(txn, &refs)?;
            }

            db_batch.commit(txn)?;
            let batch_time = start.elapsed();

            println!(
                "put_batch(size={}): {:>8.2?} ({:>5.0} ns/op) - {:.0} ops/sec",
                batch_size,
                batch_time,
                batch_time.as_nanos() as f64 / profile_n as f64,
                profile_n as f64 / batch_time.as_secs_f64()
            );
        }
    }

    // Cleanup happens automatically via TempDir
    Ok(())
}
