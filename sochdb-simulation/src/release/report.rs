//! Release gate scorecard reporting.

use crate::release::gate::{GateCategory, GatePriority};
use crate::release::validator::{GateResult, GateStatus, ReleaseScorecard};
use colored::Colorize;
use comfy_table::{Cell, Color, Table};

pub fn print_release_banner() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════╗\n\
         ║  SochDB Release Quality Simulation                       ║\n\
         ║  Storage · Concurrency · FFI · Packaging · Performance  ║\n\
         ╚══════════════════════════════════════════════════════════╝"
            .cyan()
    );
}

pub fn print_release_scorecard(card: &ReleaseScorecard) {
    let ready = if card.release_ready {
        "RELEASE READY".green().bold()
    } else {
        "NOT RELEASE READY".red().bold()
    };

    println!(
        "\n{} {} — {} pass, {} warn, {} fail, {} skip, {} manual",
        "Release Scorecard:".bold(),
        ready,
        card.passed,
        card.warned,
        card.failed,
        card.skipped,
        card.manual,
    );

    if card.blocker_failures > 0 {
        println!(
            "  {} {} blocker gate(s) failed",
            "✗".red(),
            card.blocker_failures
        );
    }
    if card.blocker_unverified > 0 {
        println!(
            "  {} {} blocker gate(s) need --validate or manual sign-off",
            "○".yellow(),
            card.blocker_unverified
        );
    }

    // Group by category
    for cat in GateCategory::all() {
        let cat_results: Vec<&GateResult> = card
            .results
            .iter()
            .filter(|r| r.category == cat.name())
            .collect();
        if cat_results.is_empty() {
            continue;
        }

        let cat_pass = cat_results
            .iter()
            .filter(|r| r.status == GateStatus::Pass)
            .count();
        let cat_fail = cat_results
            .iter()
            .filter(|r| r.status == GateStatus::Fail)
            .count();

        println!(
            "\n{} {} ({}/{} pass)",
            "▸".cyan(),
            cat.display_name().bold(),
            cat_pass,
            cat_results.len()
        );

        let mut table = Table::new();
        table.set_header(vec!["Gate", "Priority", "Status", "Duration", "Message"]);

        for r in &cat_results {
            table.add_row(vec![
                Cell::new(&r.title),
                Cell::new(priority_label(r.priority)).fg(priority_color(r.priority)),
                Cell::new(status_label(r.status)).fg(status_color(r.status)),
                Cell::new(format!("{:.0}ms", r.duration_ms)),
                Cell::new(truncate(&r.message, 60)),
            ]);
        }

        if cat_fail > 0 {
            println!("{table}");
        } else {
            // Compact summary when all pass
            for r in &cat_results {
                println!(
                    "    {} {} {}",
                    status_icon(r.status),
                    r.title.dimmed(),
                    truncate(&r.message, 40).dimmed()
                );
            }
        }
    }
}

pub fn print_release_checklist(gates: &[crate::release::gate::ReleaseGate]) {
    print_release_banner();
    println!("\n{}\n", "Full release checklist (10 areas)".bold());

    for cat in GateCategory::all() {
        let cat_gates: Vec<_> = gates.iter().filter(|g| g.category == cat.name()).collect();
        println!("{}", cat.display_name().bold().underline());
        for g in cat_gates {
            let pri = match g.priority {
                GatePriority::Blocker => "[BLOCKER]".red(),
                GatePriority::Warning => "[warn]".yellow(),
                GatePriority::Advisory => "[info]".dimmed(),
            };
            println!("  {} {} — {}", pri, g.title, g.description.dimmed());
        }
        println!();
    }

    println!(
        "{}",
        "Run with --validate for live checks (static + tests).\n\
         Run with --validate --full for stress tests, audit, clippy."
            .dimmed()
    );
}

fn status_label(s: GateStatus) -> &'static str {
    match s {
        GateStatus::Pass => "PASS",
        GateStatus::Warn => "WARN",
        GateStatus::Fail => "FAIL",
        GateStatus::Skip => "SKIP",
        GateStatus::Manual => "MANUAL",
    }
}

fn status_icon(s: GateStatus) -> colored::ColoredString {
    match s {
        GateStatus::Pass => "✓".green(),
        GateStatus::Warn => "⚠".yellow(),
        GateStatus::Fail => "✗".red(),
        GateStatus::Skip => "○".dimmed(),
        GateStatus::Manual => "☐".blue(),
    }
}

fn status_color(s: GateStatus) -> Color {
    match s {
        GateStatus::Pass => Color::Green,
        GateStatus::Warn => Color::Yellow,
        GateStatus::Fail => Color::Red,
        GateStatus::Skip => Color::DarkGrey,
        GateStatus::Manual => Color::Blue,
    }
}

fn priority_label(p: GatePriority) -> &'static str {
    match p {
        GatePriority::Blocker => "blocker",
        GatePriority::Warning => "warning",
        GatePriority::Advisory => "advisory",
    }
}

fn priority_color(p: GatePriority) -> Color {
    match p {
        GatePriority::Blocker => Color::Red,
        GatePriority::Warning => Color::Yellow,
        GatePriority::Advisory => Color::DarkGrey,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}
