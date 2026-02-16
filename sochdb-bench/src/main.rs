//! SochDB Comparative Benchmark Runner
//!
//! Usage:
//!   sochdb-bench --all                    # run all workloads
//!   sochdb-bench --oltp --analytics       # run selected workloads
//!   sochdb-bench --scale 10 --export json # 100K ops, export JSON
//!   sochdb-bench --skip duckdb            # skip a database

use clap::Parser;
use colored::Colorize;
use sochdb_bench::adapters::duckdb_adapter::DuckDbAdapter;
use sochdb_bench::adapters::sochdb_adapter::SochDbAdapter;
use sochdb_bench::adapters::sqlite_adapter::SqliteAdapter;
use sochdb_bench::report;
use sochdb_bench::workloads::{self, WorkloadConfig};
use sochdb_bench::{BenchDb, BenchResult, BenchSuite, SystemInfo, WorkloadResult};
use std::path::Path;
use tempfile::TempDir;

#[derive(Parser, Debug)]
#[command(name = "sochdb-bench", about = "SochDB comparative benchmark suite")]
struct Cli {
    /// Run all workloads.
    #[arg(long)]
    all: bool,

    /// Run OLTP workloads (seq write, seq read, rand read, batch write, delete).
    #[arg(long)]
    oltp: bool,

    /// Run analytics workloads (bulk insert, queries).
    #[arg(long)]
    analytics: bool,

    /// Run vector workloads (insert, search).
    #[arg(long)]
    vector: bool,

    /// Run mixed workload (80/20 read/write).
    #[arg(long)]
    mixed: bool,

    /// Number of operations per workload.
    #[arg(long, default_value = "10000")]
    scale: usize,

    /// Vector dimension.
    #[arg(long, default_value = "128")]
    dim: usize,

    /// Top-k for vector search.
    #[arg(long, default_value = "10")]
    k: usize,

    /// Export directory for CSV + JSON results.
    #[arg(long)]
    export: Option<String>,

    /// Skip databases (comma-separated: sochdb, sqlite, duckdb, lancedb).
    #[arg(long, value_delimiter = ',')]
    skip: Vec<String>,

    /// Value size in bytes for KV workloads.
    #[arg(long, default_value = "256")]
    value_size: usize,
}

fn main() -> BenchResult<()> {
    let cli = Cli::parse();

    // If no workload flags, default to --all.
    let run_all = cli.all || (!cli.oltp && !cli.analytics && !cli.vector && !cli.mixed);
    let run_oltp = run_all || cli.oltp;
    let run_analytics = run_all || cli.analytics;
    let run_vector = run_all || cli.vector;
    let run_mixed = run_all || cli.mixed;

    let cfg = WorkloadConfig {
        scale: cli.scale,
        value_size: cli.value_size,
        batch_size: 1000,
        dim: cli.dim,
        k: cli.k,
    };

    let skip: Vec<String> = cli.skip.iter().map(|s| s.to_lowercase()).collect();

    println!(
        "\n{}",
        "╔══════════════════════════════════════════════════════╗"
            .bold()
            .blue()
    );
    println!(
        "{}",
        "║     SochDB Comparative Benchmark Suite              ║"
            .bold()
            .blue()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════╝"
            .bold()
            .blue()
    );
    println!(
        "  Scale: {} ops  ValueSize: {}B  VecDim: {}  TopK: {}",
        cfg.n(),
        cfg.value_size,
        cfg.dim,
        cfg.k
    );

    let mut suite = BenchSuite {
        system_info: SystemInfo::collect(),
        results: Vec::new(),
    };

    // Create temporary directories for each database.
    let tmp = TempDir::new()?;

    // Build database adapters.
    let mut databases: Vec<Box<dyn BenchDb>> = Vec::new();

    if !skip.contains(&"sochdb".to_string()) {
        match SochDbAdapter::new(tmp.path()) {
            Ok(db) => databases.push(Box::new(db)),
            Err(e) => eprintln!("  {} SochDB: {}", "SKIP".yellow(), e),
        }
    }
    if !skip.contains(&"sqlite".to_string()) {
        match SqliteAdapter::new(tmp.path()) {
            Ok(db) => databases.push(Box::new(db)),
            Err(e) => eprintln!("  {} SQLite: {}", "SKIP".yellow(), e),
        }
    }
    if !skip.contains(&"duckdb".to_string()) {
        match DuckDbAdapter::new(tmp.path()) {
            Ok(db) => databases.push(Box::new(db)),
            Err(e) => eprintln!("  {} DuckDB: {}", "SKIP".yellow(), e),
        }
    }

    #[cfg(feature = "lancedb-bench")]
    {
        if !skip.contains(&"lancedb".to_string()) {
            use sochdb_bench::adapters::lancedb_adapter::LanceDbAdapter;
            match LanceDbAdapter::new(tmp.path()) {
                Ok(db) => databases.push(Box::new(db)),
                Err(e) => eprintln!("  {} LanceDB: {}", "SKIP".yellow(), e),
            }
        }
    }

    if databases.is_empty() {
        return Err(sochdb_bench::BenchError::Config(
            "No databases to benchmark. Check --skip flags.".into(),
        ));
    }

    println!("  Databases: {}", databases.iter().map(|d| d.name()).collect::<Vec<_>>().join(", "));

    // ── OLTP ──
    if run_oltp {
        println!("\n{}", "▶ OLTP Workloads".bold().green());
        run_workload_across(&mut databases, &cfg, &mut suite.results, "oltp_seq_write", |db, cfg| {
            workloads::oltp_sequential_writes(db, cfg)
        })?;
        run_workload_across(&mut databases, &cfg, &mut suite.results, "oltp_seq_read", |db, cfg| {
            workloads::oltp_sequential_reads(db, cfg)
        })?;
        run_workload_across(&mut databases, &cfg, &mut suite.results, "oltp_rand_read", |db, cfg| {
            workloads::oltp_random_reads(db, cfg)
        })?;
        run_workload_across(&mut databases, &cfg, &mut suite.results, "oltp_batch_write", |db, cfg| {
            workloads::oltp_batch_write(db, cfg)
        })?;
        run_workload_across(&mut databases, &cfg, &mut suite.results, "oltp_delete", |db, cfg| {
            workloads::oltp_deletes(db, cfg)
        })?;
    }

    // ── Analytics ──
    if run_analytics {
        println!("\n{}", "▶ Analytics Workloads".bold().green());
        run_workload_across(&mut databases, &cfg, &mut suite.results, "analytics_bulk_insert", |db, cfg| {
            workloads::analytics_bulk_insert(db, cfg)
        })?;
        run_workload_across(&mut databases, &cfg, &mut suite.results, "analytics_queries", |db, cfg| {
            workloads::analytics_queries(db, cfg)
        })?;
    }

    // ── Vector ──
    if run_vector {
        println!("\n{}", "▶ Vector Workloads".bold().green());
        run_workload_across(&mut databases, &cfg, &mut suite.results, "vector_insert", |db, cfg| {
            workloads::vector_insert(db, cfg)
        })?;
        run_workload_across(&mut databases, &cfg, &mut suite.results, "vector_search", |db, cfg| {
            workloads::vector_search(db, cfg)
        })?;
    }

    // ── Mixed ──
    if run_mixed {
        println!("\n{}", "▶ Mixed Workloads".bold().green());
        run_workload_across(&mut databases, &cfg, &mut suite.results, "mixed_80r_20w", |db, cfg| {
            workloads::mixed_read_heavy(db, cfg)
        })?;
    }

    // ── Storage Efficiency ──
    println!("\n{}", "▶ Storage Efficiency".bold().green());
    run_workload_across(&mut databases, &cfg, &mut suite.results, "storage_efficiency", |db, cfg| {
        workloads::storage_efficiency(db, cfg)
    })?;

    // ── Teardown ──
    for db in &mut databases {
        let _ = db.teardown();
    }

    // ── Report ──
    report::print_suite(&suite);

    // ── Export ──
    if let Some(ref dir) = cli.export {
        let export_dir = Path::new(dir);
        std::fs::create_dir_all(export_dir)?;
        report::export_csv(&suite, &export_dir.join("benchmark_results.csv"))?;
        report::export_json(&suite, &export_dir.join("benchmark_results.json"))?;
    }

    Ok(())
}

/// Run a single workload function across all databases, collecting results.
fn run_workload_across(
    databases: &mut [Box<dyn BenchDb>],
    cfg: &WorkloadConfig,
    results: &mut Vec<WorkloadResult>,
    label: &str,
    workload_fn: fn(&mut dyn BenchDb, &WorkloadConfig) -> BenchResult<WorkloadResult>,
) -> BenchResult<()> {
    use std::io::Write;
    print!("  {} ... ", label);
    let _ = std::io::stdout().flush();
    let mut workload_results = Vec::new();

    for db in databases.iter_mut() {
        match workload_fn(db.as_mut(), cfg) {
            Ok(mut r) => {
                if let Ok(size) = db.db_size_bytes() {
                    r.extra.insert("db_size_bytes".into(), size.to_string());
                }
                print!("{}:{:.0} ops/s  ", db.name(), r.throughput);
                let _ = std::io::stdout().flush();
                workload_results.push(r);
            }
            Err(e) => {
                print!("{}: ERR({})  ", db.name(), e);
                let _ = std::io::stdout().flush();
            }
        }
    }
    println!();

    results.extend(workload_results);
    Ok(())
}
