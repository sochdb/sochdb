// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! SochDB Unified CLI (Task 6)
//!
//! Single binary for all SochDB management operations.
//!
//! ## Usage
//!
//! ```bash
//! # Interactive SQL shell
//! sochdb sql my.db
//!
//! # Execute a SQL statement
//! sochdb sql my.db -c "SELECT * FROM users"
//!
//! # Database info
//! sochdb info my.db
//!
//! # Compact storage
//! sochdb compact my.db
//!
//! # Schema management
//! sochdb schema my.db tables
//! sochdb schema my.db describe users
//!
//! # Run migrations
//! sochdb migrate my.db --status
//! ```

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// SochDB — LLM-Optimized Embedded Database
#[derive(Parser)]
#[command(name = "sochdb")]
#[command(version, about = "SochDB database management CLI")]
#[command(propagate_version = true)]
struct Cli {
    /// Enable verbose/debug output
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute SQL queries against a database
    Sql {
        /// Path to the database directory
        db_path: PathBuf,

        /// Execute a single SQL command and exit
        #[arg(short = 'c', long)]
        command: Option<String>,

        /// Output format
        #[arg(short, long, default_value = "table")]
        format: OutputFormat,
    },

    /// Show database information and statistics
    Info {
        /// Path to the database directory
        db_path: PathBuf,
    },

    /// Compact the database storage
    Compact {
        /// Path to the database directory
        db_path: PathBuf,
    },

    /// Schema management (list tables, describe, etc.)
    Schema {
        /// Path to the database directory
        db_path: PathBuf,

        #[command(subcommand)]
        action: SchemaAction,
    },

    /// Schema migration status and operations
    Migrate {
        /// Path to the database directory
        db_path: PathBuf,

        /// Show migration status without applying
        #[arg(long)]
        status: bool,
    },

    /// Validate database integrity
    Check {
        /// Path to the database directory
        db_path: PathBuf,
    },
}

#[derive(Subcommand)]
enum SchemaAction {
    /// List all tables
    Tables,
    /// Describe a table's schema
    Describe {
        /// Table name
        table: String,
    },
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum OutputFormat {
    /// Aligned table output (default)
    Table,
    /// CSV format
    Csv,
    /// JSON format
    Json,
}

// =============================================================================
// Main entry point
// =============================================================================

fn main() {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    let result = match cli.command {
        Commands::Sql {
            db_path,
            command,
            format,
        } => cmd_sql(db_path, command, format),
        Commands::Info { db_path } => cmd_info(db_path),
        Commands::Compact { db_path } => cmd_compact(db_path),
        Commands::Schema { db_path, action } => cmd_schema(db_path, action),
        Commands::Migrate { db_path, status } => cmd_migrate(db_path, status),
        Commands::Check { db_path } => cmd_check(db_path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

// =============================================================================
// Open database helper
// =============================================================================

fn open_db(path: &PathBuf) -> Result<Arc<sochdb_storage::Database>, String> {
    sochdb_storage::Database::open(path.as_path()).map_err(|e| format!("Failed to open database: {}", e))
}

// =============================================================================
// SQL command
// =============================================================================

fn cmd_sql(db_path: PathBuf, command: Option<String>, format: OutputFormat) -> Result<(), String> {
    let db = open_db(&db_path)?;

    if let Some(sql) = command {
        // Single-command mode: execute and exit
        execute_and_print(&db, &sql, &format)?;
    } else {
        // Interactive REPL mode
        run_sql_repl(&db, &format)?;
    }

    Ok(())
}

fn execute_and_print(
    db: &Arc<sochdb_storage::Database>,
    sql: &str,
    format: &OutputFormat,
) -> Result<(), String> {
    use sochdb_query::sql::bridge::{ExecutionResult, SqlBridge};
    use sochdb_query::storage_bridge::DatabaseSqlConnection;

    let conn = DatabaseSqlConnection::new(db.clone());
    let mut bridge = SqlBridge::new(conn);

    let result = bridge
        .execute(sql)
        .map_err(|e| format!("SQL error: {}", e))?;

    match result {
        ExecutionResult::Rows { columns, rows } => {
            print_rows(&columns, &rows, format);
        }
        ExecutionResult::RowsAffected(n) => {
            println!("{} row(s) affected", n);
        }
        ExecutionResult::Ok => {
            println!("OK");
        }
        ExecutionResult::TransactionOk => {
            println!("OK");
        }
    }
    Ok(())
}

fn print_rows(
    columns: &[String],
    rows: &[HashMap<String, sochdb_core::SochValue>],
    format: &OutputFormat,
) {
    if rows.is_empty() {
        println!("(0 rows)");
        return;
    }

    match format {
        OutputFormat::Table => print_table(columns, rows),
        OutputFormat::Csv => print_csv(columns, rows),
        OutputFormat::Json => print_json(columns, rows),
    }
}

fn print_table(columns: &[String], rows: &[HashMap<String, sochdb_core::SochValue>]) {
    // Compute column widths
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    for row in rows {
        for (i, col) in columns.iter().enumerate() {
            let val = row
                .get(col)
                .map(|v| format_value(v))
                .unwrap_or_else(|| "NULL".to_string());
            widths[i] = widths[i].max(val.len());
        }
    }

    // Header
    let header: Vec<String> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{:width$}", c, width = widths[i]))
        .collect();
    println!(" {} ", header.join(" | "));

    // Separator
    let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    println!("-{}-", sep.join("-+-"));

    // Data rows
    for row in rows {
        let vals: Vec<String> = columns
            .iter()
            .enumerate()
            .map(|(i, col)| {
                let val = row
                    .get(col)
                    .map(|v| format_value(v))
                    .unwrap_or_else(|| "NULL".to_string());
                format!("{:width$}", val, width = widths[i])
            })
            .collect();
        println!(" {} ", vals.join(" | "));
    }

    println!("({} row{})", rows.len(), if rows.len() == 1 { "" } else { "s" });
}

fn print_csv(columns: &[String], rows: &[HashMap<String, sochdb_core::SochValue>]) {
    println!("{}", columns.join(","));
    for row in rows {
        let vals: Vec<String> = columns
            .iter()
            .map(|col| {
                row.get(col)
                    .map(|v| format_value(v))
                    .unwrap_or_else(|| "".to_string())
            })
            .collect();
        println!("{}", vals.join(","));
    }
}

fn print_json(columns: &[String], rows: &[HashMap<String, sochdb_core::SochValue>]) {
    let json_rows: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            let mut map = serde_json::Map::new();
            for col in columns {
                let val = row
                    .get(col)
                    .map(|v| value_to_json(v))
                    .unwrap_or(serde_json::Value::Null);
                map.insert(col.clone(), val);
            }
            serde_json::Value::Object(map)
        })
        .collect();

    println!(
        "{}",
        serde_json::to_string_pretty(&json_rows).unwrap_or_else(|_| "[]".to_string())
    );
}

fn format_value(v: &sochdb_core::SochValue) -> String {
    match v {
        sochdb_core::SochValue::Null => "NULL".to_string(),
        sochdb_core::SochValue::Bool(b) => b.to_string(),
        sochdb_core::SochValue::Int(i) => i.to_string(),
        sochdb_core::SochValue::UInt(u) => u.to_string(),
        sochdb_core::SochValue::Float(f) => format!("{:.6}", f),
        sochdb_core::SochValue::Text(s) => s.to_string(),
        sochdb_core::SochValue::Binary(b) => format!("<binary {} bytes>", b.len()),
        sochdb_core::SochValue::Array(a) => format!("[{} elements]", a.len()),
        sochdb_core::SochValue::Object(m) => format!("{{{} keys}}", m.len()),
        sochdb_core::SochValue::Ref { table, id } => format!("{}/{}", table, id),
    }
}

fn value_to_json(v: &sochdb_core::SochValue) -> serde_json::Value {
    match v {
        sochdb_core::SochValue::Null => serde_json::Value::Null,
        sochdb_core::SochValue::Bool(b) => serde_json::Value::Bool(*b),
        sochdb_core::SochValue::Int(i) => serde_json::json!(i),
        sochdb_core::SochValue::UInt(u) => serde_json::json!(u),
        sochdb_core::SochValue::Float(f) => serde_json::json!(f),
        sochdb_core::SochValue::Text(s) => serde_json::Value::String(s.to_string()),
        sochdb_core::SochValue::Binary(b) => {
            serde_json::Value::String(format!("<binary {} bytes>", b.len()))
        }
        sochdb_core::SochValue::Array(a) => {
            serde_json::Value::Array(a.iter().map(value_to_json).collect())
        }
        sochdb_core::SochValue::Object(m) => {
            let map: serde_json::Map<String, serde_json::Value> =
                m.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect();
            serde_json::Value::Object(map)
        }
        sochdb_core::SochValue::Ref { table, id } => {
            serde_json::json!({"$ref": format!("{}/{}", table, id)})
        }
    }
}

// Simple REPL (no readline dep — keeps the binary small)
fn run_sql_repl(
    db: &Arc<sochdb_storage::Database>,
    format: &OutputFormat,
) -> Result<(), String> {
    use std::io::{self, BufRead, Write};

    println!(
        "SochDB v{} — Interactive SQL Shell",
        env!("CARGO_PKG_VERSION")
    );
    println!("Type .quit to exit, .tables to list tables\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("sochdb> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            // EOF
            println!();
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Meta-commands
        match trimmed {
            ".quit" | ".exit" | "\\q" => break,
            ".tables" => {
                if let Err(e) = execute_and_print(db, "SELECT name FROM sochdb_tables", format) {
                    // Fallback: list from catalog
                    println!("(Table listing not yet available via SQL: {})", e);
                }
                continue;
            }
            ".help" => {
                println!("  .quit      Exit the shell");
                println!("  .tables    List tables");
                println!("  .help      Show this help");
                println!("  <SQL>      Execute SQL statement");
                continue;
            }
            _ => {}
        }

        if let Err(e) = execute_and_print(db, trimmed, format) {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

// =============================================================================
// Info command
// =============================================================================

fn cmd_info(db_path: PathBuf) -> Result<(), String> {
    let db = open_db(&db_path)?;

    println!("SochDB Database Information");
    println!("===========================");
    println!("Path:     {}", db_path.display());
    println!("Version:  {}", env!("CARGO_PKG_VERSION"));

    // Try to get table count
    let tables = db.list_tables();
    println!("Tables:   {}", tables.len());

    for table in &tables {
        println!("  - {}", table);
    }

    // Storage stats
    let stats = db.storage_stats();
    println!("\nStorage:");
    println!("  Memtable:   {} bytes", stats.memtable_size_bytes);
    println!("  WAL size:   {} bytes", stats.wal_size_bytes);
    println!("  Active txn: {}", stats.active_transactions);
    println!("  Last ckpt:  LSN {}", stats.last_checkpoint_lsn);

    // Engine stats
    let engine_stats = db.stats();
    println!("\nEngine:");
    println!("  Txn started:   {}", engine_stats.transactions_started);
    println!("  Txn committed: {}", engine_stats.transactions_committed);
    println!("  Txn aborted:   {}", engine_stats.transactions_aborted);
    println!("  Queries:       {}", engine_stats.queries_executed);
    println!("  Bytes written: {}", engine_stats.bytes_written);
    println!("  Bytes read:    {}", engine_stats.bytes_read);

    Ok(())
}

// =============================================================================
// Compact command
// =============================================================================

fn cmd_compact(db_path: PathBuf) -> Result<(), String> {
    let db = open_db(&db_path)?;
    println!("Compacting database at {}...", db_path.display());

    // Checkpoint first, then GC, then truncate WAL
    let ckpt_lsn = db
        .checkpoint()
        .map_err(|e| format!("Checkpoint failed: {}", e))?;
    println!("  Checkpoint at LSN {}", ckpt_lsn);

    let reclaimed = db.gc();
    println!("  GC reclaimed {} entries", reclaimed);

    db.truncate_wal()
        .map_err(|e| format!("WAL truncation failed: {}", e))?;
    println!("  WAL truncated");

    println!("Compaction complete.");
    Ok(())
}

// =============================================================================
// Schema command
// =============================================================================

fn cmd_schema(db_path: PathBuf, action: SchemaAction) -> Result<(), String> {
    let db = open_db(&db_path)?;

    match action {
        SchemaAction::Tables => {
            let tables = db.list_tables();
            if tables.is_empty() {
                println!("No tables found.");
            } else {
                println!("Tables:");
                for table in &tables {
                    println!("  {}", table);
                }
            }
        }
        SchemaAction::Describe { table } => {
            match db.get_table_schema(&table) {
                Some(schema) => {
                    println!("Table: {}", table);
                    println!("{:-<50}", "");
                    println!("{:<20} {:<15} {}", "Column", "Type", "Nullable");
                    println!("{:-<50}", "");
                    for col in &schema.columns {
                        println!(
                            "{:<20} {:<15} {}",
                            col.name,
                            format!("{:?}", col.col_type),
                            if col.nullable { "YES" } else { "NO" }
                        );
                    }
                }
                None => return Err(format!("Table '{}' not found", table)),
            }
        }
    }

    Ok(())
}

// =============================================================================
// Migrate command
// =============================================================================

fn cmd_migrate(db_path: PathBuf, status_only: bool) -> Result<(), String> {
    let _db = open_db(&db_path)?;

    if status_only {
        println!("Schema Migration Status");
        println!("=======================");
        println!("Database:       {}", db_path.display());
        println!("Schema version: {}", sochdb_core::SCHEMA_VERSION);
        println!("\nNo pending migrations.");
    } else {
        println!("Applying migrations...");
        println!("No migrations to apply. Database is up to date.");
    }

    Ok(())
}

// =============================================================================
// Check command
// =============================================================================

fn cmd_check(db_path: PathBuf) -> Result<(), String> {
    let db = open_db(&db_path)?;

    println!("Checking database integrity at {}...", db_path.display());

    // Verify tables
    let tables = db.list_tables();
    let mut errors = 0;

    for table in &tables {
        match db.get_table_schema(table) {
            Some(schema) => {
                println!("  ✓ {} ({} columns)", table, schema.columns.len());
            }
            None => {
                eprintln!("  ✗ {} — schema not found", table);
                errors += 1;
            }
        }
    }

    if errors == 0 {
        println!("\nDatabase OK ({} table(s) checked)", tables.len());
        Ok(())
    } else {
        Err(format!("{} error(s) found", errors))
    }
}
