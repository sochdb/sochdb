//! sochdb-sim — SochDB simulation environment CLI.

use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use sochdb_simulation::{
    release::{self, gate::GatePriority},
    report,
    scenario::Scenario,
    topology::Topology,
    ExpectedStore, ReleaseValidator, Scorer, SimulationEngine,
};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "sochdb-sim",
    about = "SochDB simulation environment — standalone & distributed topology modeling",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run simulation scenario(s) and score against expected targets.
    Run {
        /// Scenario ID (or "all").
        #[arg(short, long, default_value = "all")]
        scenario: String,

        /// Deployment topology filter.
        #[arg(short, long, value_enum)]
        mode: Option<SimMode>,

        /// Random seed for Monte Carlo jitter.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Export results + scorecards to JSON.
        #[arg(short, long)]
        export: Option<PathBuf>,

        /// Show per-component latency breakdown.
        #[arg(long)]
        breakdown: bool,

        /// Compare standalone vs distributed for KV read.
        #[arg(long)]
        compare: bool,
    },

    /// List available scenarios and expected target files.
    List,

    /// Show component registry with base latencies.
    Components,

    /// Run full release-quality gate simulation (all 10 areas).
    Release {
        /// Filter by category (storage, concurrency, ffi, packaging, ci, perf, compat, security, release).
        #[arg(short, long)]
        category: Option<String>,

        /// Filter by priority (blocker, warning, advisory).
        #[arg(short, long)]
        priority: Option<ReleasePriority>,

        /// Run live validation (cargo test, static checks, clippy).
        #[arg(long)]
        validate: bool,

        /// Include slow checks (stress tests, cargo audit, --ignored).
        #[arg(long)]
        full: bool,

        /// Show checklist only, do not evaluate.
        #[arg(long)]
        checklist: bool,

        /// Export release scorecard to JSON.
        #[arg(short, long)]
        export: Option<PathBuf>,

        /// Workspace root (default: auto-detect from CARGO_MANIFEST_DIR).
        #[arg(long)]
        workspace: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ReleasePriority {
    Blocker,
    Warning,
    Advisory,
}

#[derive(Clone, Copy, ValueEnum)]
enum SimMode {
    Standalone,
    Distributed,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            scenario,
            mode,
            seed,
            export,
            breakdown,
            compare,
        } => run_simulation(&scenario, mode, seed, export.as_deref(), breakdown, compare),
        Commands::List => list_all(),
        Commands::Components => show_components(),
        Commands::Release {
            category,
            priority,
            validate,
            full,
            checklist,
            export,
            workspace,
        } => run_release(
            category.as_deref(),
            priority,
            validate,
            full,
            checklist,
            export.as_deref(),
            workspace.as_deref(),
        ),
    }
}

fn run_release(
    category: Option<&str>,
    priority: Option<ReleasePriority>,
    validate: bool,
    full: bool,
    checklist_only: bool,
    export_path: Option<&Path>,
    workspace: Option<&Path>,
) {
    let gate_file = release::load_gates();
    let gates = gate_file.gates;

    if checklist_only {
        release::report::print_release_checklist(&gates);
        return;
    }

    release::report::print_release_banner();

    let pri_filter = priority.map(|p| match p {
        ReleasePriority::Blocker => GatePriority::Blocker,
        ReleasePriority::Warning => GatePriority::Warning,
        ReleasePriority::Advisory => GatePriority::Advisory,
    });

    let filtered: Vec<_> = release::filter_gates(&gates, category, pri_filter)
        .into_iter()
        .cloned()
        .collect();

    println!(
        "\n{} Evaluating {} release gates{}",
        "▸".cyan(),
        filtered.len(),
        if validate {
            if full {
                " (live + full)"
            } else {
                " (live)"
            }
        } else {
            " (fast: static blockers + perf simulation)"
        }
    );

    let workspace_root = workspace
        .map(Path::to_path_buf)
        .unwrap_or_else(find_workspace_root);

    let validator = ReleaseValidator::new(workspace_root)
        .with_live_validation(validate)
        .with_full(full);

    let scorecard = validator.run_gates(&filtered);
    release::report::print_release_scorecard(&scorecard);

    if let Some(path) = export_path {
        std::fs::write(path, serde_json::to_string_pretty(&scorecard).unwrap()).unwrap();
        println!("\n{} Exported to {}", "✓".green(), path.display());
    }

    if !scorecard.release_ready {
        std::process::exit(1);
    }
}

fn find_workspace_root() -> PathBuf {
    // sochdb-simulation/ → workspace root is parent
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(&manifest).to_path_buf()
}

fn run_simulation(
    scenario_id: &str,
    mode: Option<SimMode>,
    seed: u64,
    export_path: Option<&std::path::Path>,
    breakdown: bool,
    compare: bool,
) {
    print_banner();

    let scenarios = resolve_scenarios(scenario_id, mode);
    if scenarios.is_empty() {
        eprintln!("{} No scenarios matched '{}'", "Error:".red(), scenario_id);
        std::process::exit(1);
    }

    let mut engine = SimulationEngine::new(seed);
    let store = ExpectedStore::load_defaults();
    let scorer = Scorer::new(store);
    let mut all_results = Vec::new();
    let mut all_scorecards = Vec::new();

    for scenario in &scenarios {
        let result = engine.run_scenario(scenario);
        report::print_scenario_result(&result);

        if breakdown {
            for op in &result.operations {
                report::print_component_breakdown(op);
            }
        }

        let card = scorer.score_scenario(&result);
        scorer.print_scorecard(&card);

        all_results.push(result);
        all_scorecards.push(card);
    }

    if compare {
        run_topology_comparison(&mut engine);
    }

    if let Some(path) = export_path {
        let payload = serde_json::json!({
            "results": all_results,
            "scorecards": all_scorecards,
        });
        std::fs::write(path, serde_json::to_string_pretty(&payload).unwrap()).unwrap();
        println!("\n{} Exported to {}", "✓".green(), path.display());
    }

    let any_fail = all_scorecards
        .iter()
        .any(|c| c.overall_grade == sochdb_simulation::Grade::Fail);
    if any_fail {
        std::process::exit(1);
    }
}

fn run_topology_comparison(engine: &mut SimulationEngine) {
    use sochdb_simulation::{component::SimEnvironment, topology::Operation};

    let env = SimEnvironment::default();
    let standalone =
        engine.simulate_operation(Topology::Standalone, Operation::PointRead, 10_000, &env);
    let distributed =
        engine.simulate_operation(Topology::Distributed, Operation::GrpcKvGet, 10_000, &env);
    report::print_topology_comparison(&standalone, &distributed);
}

fn resolve_scenarios(id: &str, mode: Option<SimMode>) -> Vec<Scenario> {
    let all = Scenario::all();
    let filtered: Vec<Scenario> = if id == "all" {
        all
    } else if let Some(s) = Scenario::by_id(id) {
        vec![s]
    } else {
        Vec::new()
    };

    match mode {
        Some(SimMode::Standalone) => filtered
            .into_iter()
            .filter(|s| s.topology == Topology::Standalone)
            .collect(),
        Some(SimMode::Distributed) => filtered
            .into_iter()
            .filter(|s| s.topology == Topology::Distributed)
            .collect(),
        None => filtered,
    }
}

fn list_all() {
    print_banner();
    println!("{}\n", "Scenarios:".bold());
    for s in Scenario::all() {
        println!(
            "  {} {} [{}] — {}",
            "•".cyan(),
            s.id.bold(),
            s.topology.name().cyan(),
            s.description.dimmed()
        );
        for op in &s.operations {
            println!(
                "      - {} ({} ops)",
                format!("{:?}", op.operation).dimmed(),
                op.ops
            );
        }
    }

    println!("\n{}\n", "Expected target files:".bold());
    let store = ExpectedStore::load_defaults();
    for f in store.all_files() {
        println!(
            "  {} {} — {} targets ({})",
            "•".cyan(),
            f.source.dimmed(),
            f.targets.len(),
            f.topology
        );
    }

    let gate_file = release::load_gates();
    println!("\n{}\n", "Release gates:".bold());
    println!(
        "  {} {} gates across 10 release-quality areas",
        "•".cyan(),
        gate_file.gates.len()
    );
    println!(
        "  Run: {} or {}",
        "sochdb-sim release --checklist".cyan(),
        "sochdb-sim release --validate".cyan()
    );
}

fn show_components() {
    use sochdb_simulation::component::Component;

    print_banner();
    println!("{}\n", "Component registry:".bold());

    let mut table = comfy_table::Table::new();
    table.set_header(vec!["Component", "Base Latency (μs)", "Max Throughput"]);

    let components = [
        Component::EmbeddedConnection,
        Component::GrpcClient,
        Component::GrpcServer,
        Component::NetworkRoundTrip,
        Component::MvccCoordinator,
        Component::WalWriter,
        Component::MemtableLookup,
        Component::ColumnarCache,
        Component::HnswIndex,
        Component::VamanaIndex,
        Component::BruteForceScan,
        Component::FusionPipeline,
        Component::ContextBuilder,
        Component::TokenBudgetEngine,
        Component::ToonEncoder,
        Component::McpServer,
        Component::TemporalGraph,
    ];

    for c in components {
        table.add_row(vec![
            c.display_name().into(),
            format!("{:.2}", c.base_latency_us()),
            format_throughput(c.max_throughput()),
        ]);
    }

    println!("{table}");
}

fn format_throughput(tps: f64) -> String {
    if tps >= 1_000_000.0 {
        format!("{:.1}M ops/s", tps / 1_000_000.0)
    } else if tps >= 1_000.0 {
        format!("{:.1}K ops/s", tps / 1_000.0)
    } else {
        format!("{:.0} ops/s", tps)
    }
}

fn print_banner() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════╗\n\
         ║  SochDB Simulation Environment                   ║\n\
         ║  Standalone (embedded) · Distributed (gRPC)    ║\n\
         ╚══════════════════════════════════════════════════╝"
            .cyan()
    );
}
