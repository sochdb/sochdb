//! Pretty-print benchmark results with comparison tables, CSV and JSON export.

use crate::{BenchSuite, WorkloadResult};
use colored::Colorize;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::collections::HashMap;
use std::path::Path;

// ────────────────────────────────────────────────────────────────────────────────
// Terminal output
// ────────────────────────────────────────────────────────────────────────────────

/// Print a comparison table for one workload across all databases.
pub fn print_workload_comparison(workload: &str, results: &[WorkloadResult]) {
    if results.is_empty() {
        return;
    }

    println!(
        "\n{}",
        format!("━━━ {} ━━━", workload).bold().cyan()
    );

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    table.set_header(vec![
        "Database",
        "Ops",
        "Throughput",
        "p50 (μs)",
        "p99 (μs)",
        "p99.9 (μs)",
        "Mean (μs)",
        "DB Size",
    ]);

    // Find best throughput for highlighting.
    let best_throughput = results
        .iter()
        .map(|r| r.throughput)
        .fold(0.0f64, f64::max);

    for r in results {
        let is_best = (r.throughput - best_throughput).abs() < 0.01 && r.throughput > 0.0;
        let name = if is_best {
            format!("★ {}", r.db_name)
        } else {
            r.db_name.clone()
        };

        let name_cell = if is_best {
            Cell::new(name).fg(Color::Green)
        } else {
            Cell::new(name)
        };

        let tp_cell = if is_best {
            Cell::new(format_throughput(r.throughput)).fg(Color::Green)
        } else {
            Cell::new(format_throughput(r.throughput))
        };

        let size_str = r.extra.get("db_size_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .map(format_bytes)
            .unwrap_or_else(|| "-".to_string());

        table.add_row(vec![
            name_cell,
            Cell::new(format_count(r.ops)),
            tp_cell,
            Cell::new(format!("{:.1}", r.p50_us)),
            Cell::new(format!("{:.1}", r.p99_us)),
            Cell::new(format!("{:.1}", r.p999_us)),
            Cell::new(format!("{:.1}", r.mean_us)),
            Cell::new(size_str),
        ]);
    }

    println!("{table}");

    // Print extra fields if any (excluding db_size_bytes shown as column).
    for r in results {
        let filtered: Vec<String> = r
            .extra
            .iter()
            .filter(|(k, _)| k.as_str() != "db_size_bytes")
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        if !filtered.is_empty() {
            println!(
                "  {} {}",
                r.db_name.dimmed(),
                filtered.join(", ").dimmed()
            );
        }
    }
}

/// Print the full benchmark suite report.
pub fn print_suite(suite: &BenchSuite) {
    println!(
        "\n{}",
        "╔══════════════════════════════════════════════════════════════╗"
            .bold()
            .blue()
    );
    println!(
        "{}",
        "║          SochDB Comparative Benchmark Report               ║"
            .bold()
            .blue()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════╝"
            .bold()
            .blue()
    );

    println!(
        "  OS: {}  Arch: {}  CPUs: {}  Time: {}",
        suite.system_info.os,
        suite.system_info.arch,
        suite.system_info.cpus,
        suite.system_info.timestamp
    );

    // Group results by workload.
    let mut by_workload: HashMap<String, Vec<WorkloadResult>> = HashMap::new();
    // Preserve ordering by tracking insertion order.
    let mut workload_order: Vec<String> = Vec::new();

    for r in &suite.results {
        if !by_workload.contains_key(&r.workload) {
            workload_order.push(r.workload.clone());
        }
        by_workload
            .entry(r.workload.clone())
            .or_default()
            .push(r.clone());
    }

    for wl in &workload_order {
        if let Some(results) = by_workload.get(wl) {
            print_workload_comparison(wl, results);
        }
    }

    // Summary: wins per database.
    println!(
        "\n{}",
        "── Summary: Wins by Database ──".bold().yellow()
    );
    let mut wins: HashMap<String, usize> = HashMap::new();
    for (_, results) in &by_workload {
        if let Some(best) = results
            .iter()
            .filter(|r| r.throughput > 0.0)
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
        {
            *wins.entry(best.db_name.clone()).or_default() += 1;
        }
    }
    let mut win_list: Vec<_> = wins.into_iter().collect();
    win_list.sort_by(|a, b| b.1.cmp(&a.1));
    for (db, count) in &win_list {
        println!("  {} {} wins", format!("{:>12}", db).bold(), count);
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// CSV export
// ────────────────────────────────────────────────────────────────────────────────

pub fn export_csv(suite: &BenchSuite, path: &Path) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;

    wtr.write_record([
        "database",
        "workload",
        "ops",
        "total_secs",
        "throughput_ops_sec",
        "p50_us",
        "p99_us",
        "p999_us",
        "mean_us",
        "db_size_bytes",
    ])?;

    for r in &suite.results {
        let db_size = r.extra.get("db_size_bytes").cloned().unwrap_or_default();
        wtr.write_record([
            &r.db_name,
            &r.workload,
            &r.ops.to_string(),
            &format!("{:.6}", r.total_secs),
            &format!("{:.2}", r.throughput),
            &format!("{:.2}", r.p50_us),
            &format!("{:.2}", r.p99_us),
            &format!("{:.2}", r.p999_us),
            &format!("{:.2}", r.mean_us),
            &db_size,
        ])?;
    }

    wtr.flush()?;
    println!("  CSV exported to {}", path.display());
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────────
// JSON export
// ────────────────────────────────────────────────────────────────────────────────

pub fn export_json(suite: &BenchSuite, path: &Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(suite)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, json)?;
    println!("  JSON exported to {}", path.display());
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ────────────────────────────────────────────────────────────────────────────────

fn format_throughput(t: f64) -> String {
    if t >= 1_000_000.0 {
        format!("{:.2}M", t / 1_000_000.0)
    } else if t >= 1_000.0 {
        format!("{:.1}K", t / 1_000.0)
    } else {
        format!("{:.0}", t)
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.1} GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1_024 {
        format!("{:.1} KB", b as f64 / 1_024.0)
    } else {
        format!("{} B", b)
    }
}
