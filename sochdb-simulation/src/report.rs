//! Terminal reporting for simulation results.

use crate::engine::{OpResult, ScenarioResult};
use colored::Colorize;
use comfy_table::{Cell, Color, Table};

pub fn print_scenario_result(result: &ScenarioResult) {
    println!(
        "\n{} {} [{}]",
        "Scenario:".bold(),
        result.scenario_name.bold(),
        result.topology.cyan()
    );
    println!("  Simulated {} total operations", result.total_simulated_ops);

    let mut table = Table::new();
    table.set_header(vec![
        "Workload",
        "Ops",
        "Throughput",
        "p50 (μs)",
        "p99 (μs)",
        "Bottleneck",
    ]);

    for op in &result.operations {
        table.add_row(vec![
            Cell::new(&op.workload),
            Cell::new(format_ops(op.ops)),
            Cell::new(format_throughput(op.throughput_ops_sec)),
            Cell::new(format!("{:.2}", op.p50_us)),
            Cell::new(format!("{:.2}", op.p99_us)),
            Cell::new(&op.bottleneck).fg(Color::Cyan),
        ]);
    }

    println!("{table}");
}

pub fn print_component_breakdown(op: &OpResult) {
    println!("\n{} {}", "Component breakdown:".bold(), op.workload.underline());
    let mut table = Table::new();
    table.set_header(vec!["Component", "Latency (μs)", "% of Total"]);

    for c in &op.component_breakdown {
        table.add_row(vec![
            Cell::new(&c.component),
            Cell::new(format!("{:.2}", c.latency_us)),
            Cell::new(format!("{:.1}%", c.pct_of_total)),
        ]);
    }

    println!("{table}");
}

pub fn print_topology_comparison(standalone: &OpResult, distributed: &OpResult) {
    let overhead_pct =
        (distributed.mean_latency_us - standalone.mean_latency_us) / standalone.mean_latency_us * 100.0;

    println!(
        "\n{} {} — distributed adds {:.1}% latency overhead",
        "Topology comparison:".bold(),
        standalone.workload,
        overhead_pct
    );

    let mut table = Table::new();
    table.set_header(vec!["Topology", "Throughput", "p50 (μs)", "p99 (μs)"]);

    for op in [standalone, distributed] {
        table.add_row(vec![
            Cell::new(&op.topology),
            Cell::new(format_throughput(op.throughput_ops_sec)),
            Cell::new(format!("{:.2}", op.p50_us)),
            Cell::new(format!("{:.2}", op.p99_us)),
        ]);
    }

    println!("{table}");
}

fn format_ops(ops: u64) -> String {
    if ops >= 1_000_000 {
        format!("{:.1}M", ops as f64 / 1_000_000.0)
    } else if ops >= 1_000 {
        format!("{:.1}K", ops as f64 / 1_000.0)
    } else {
        ops.to_string()
    }
}

fn format_throughput(tps: f64) -> String {
    if tps >= 1_000_000.0 {
        format!("{:.2}M ops/s", tps / 1_000_000.0)
    } else if tps >= 1_000.0 {
        format!("{:.1}K ops/s", tps / 1_000.0)
    } else {
        format!("{:.0} ops/s", tps)
    }
}