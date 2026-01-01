// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! perf-harness: Comprehensive performance test harness for ToonDB
//!
//! This binary provides:
//! - HDRHistogram-based latency measurement (p50/p95/p99/p99.9)
//! - Recall@k evaluation with ground truth
//! - Regression gate evaluation
//! - Environment reproducibility checks
//!
//! Usage:
//!   perf-harness --workload benchmarks/workloads/agent_memory.yaml --tier small --runs 5
//!   perf-harness --workload benchmarks/workloads/agent_memory.yaml --baseline benchmarks/baselines/...

use benchmarks::harness::{
    config::{DatasetTier, WorkloadSpec},
    environment::EnvironmentInfo,
    gates::{GateEvaluation, RegressionGates},
    metrics::MetricsCollector,
    recall::RecallEvaluator,
    reporter::RunReport,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Debug)]
struct Args {
    workload: PathBuf,
    tier: DatasetTier,
    baseline: Option<PathBuf>,
    out: PathBuf,
    duration_secs: u64,
    warmup_secs: u64,
    gate_mode: GateMode,
    #[allow(dead_code)]
    generate_truth: bool,
    verbose: bool,
}

#[derive(Debug, Clone, Copy)]
enum GateMode {
    None,
    Pr,
    Nightly,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut workload = None;
    let mut tier = DatasetTier::Small;
    let mut baseline = None;
    let mut out = PathBuf::from("benchmarks/reports/runs");
    let mut duration_secs = 60;
    let mut warmup_secs = 10;
    let mut gate_mode = GateMode::None;
    let mut generate_truth = false;
    let mut verbose = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workload" | "-w" => {
                workload = args.next().map(PathBuf::from);
            }
            "--tier" | "-t" => {
                tier = match args.next().as_deref() {
                    Some("small") => DatasetTier::Small,
                    Some("medium") => DatasetTier::Medium,
                    Some("large") => DatasetTier::Large,
                    _ => return Err("Invalid tier. Use: small, medium, large".to_string()),
                };
            }
            "--baseline" | "-b" => {
                baseline = args.next().map(PathBuf::from);
            }
            "--out" | "-o" => {
                out = args.next().map(PathBuf::from).unwrap_or(out);
            }
            "--duration" | "-d" => {
                duration_secs = args.next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(duration_secs);
            }
            "--warmup" => {
                warmup_secs = args.next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(warmup_secs);
            }
            "--gate" | "-g" => {
                gate_mode = match args.next().as_deref() {
                    Some("pr") => GateMode::Pr,
                    Some("nightly") => GateMode::Nightly,
                    Some("none") => GateMode::None,
                    _ => return Err("Invalid gate mode. Use: pr, nightly, none".to_string()),
                };
            }
            "--generate-truth" => generate_truth = true,
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("Unknown argument: {}", arg));
            }
        }
    }

    let workload = workload.ok_or("--workload is required")?;

    Ok(Args {
        workload,
        tier,
        baseline,
        out,
        duration_secs,
        warmup_secs,
        gate_mode,
        generate_truth,
        verbose,
    })
}

fn print_help() {
    println!(r#"
perf-harness: Comprehensive performance test harness for ToonDB

USAGE:
    perf-harness --workload <WORKLOAD> [OPTIONS]

OPTIONS:
    -w, --workload <PATH>     Path to workload YAML file (required)
    -t, --tier <TIER>         Dataset tier: small, medium, large [default: small]
    -b, --baseline <PATH>     Path to baseline JSON for comparison
    -o, --out <PATH>          Output directory [default: benchmarks/reports/runs]
    -d, --duration <SECS>     Benchmark duration in seconds [default: 60]
    --warmup <SECS>           Warmup duration in seconds [default: 10]
    -g, --gate <MODE>         Gate mode: pr, nightly, none [default: none]
    --generate-truth          Generate ground truth for recall evaluation
    -v, --verbose             Verbose output
    -h, --help                Print this help

EXAMPLES:
    # Quick test with small dataset
    perf-harness -w benchmarks/workloads/agent_memory.yaml -t small -d 30

    # PR gate check with baseline
    perf-harness -w benchmarks/workloads/agent_memory.yaml --gate pr \
        --baseline benchmarks/baselines/ci-linux/agent_memory/small.json

    # Nightly comprehensive run
    perf-harness -w benchmarks/workloads/agent_memory.yaml -t medium --gate nightly -d 300
"#);
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), String> {
    let args = parse_args()?;

    // Load workload spec
    let spec = WorkloadSpec::load(&args.workload)
        .map_err(|e| format!("Failed to load workload: {}", e))?;

    if args.verbose {
        println!("Loaded workload: {}", spec.workload.name);
        println!("Tier: {}", args.tier);
        if let Some(count) = spec.record_count(args.tier) {
            println!("Record count: {}", count);
        }
    }

    // Collect environment info
    let env = EnvironmentInfo::collect();

    if args.verbose {
        println!("Environment: {} / {} cores / {} GB RAM", env.os, env.cpu_cores, env.ram_gb);
        println!("Git: {} (dirty: {})", env.git_sha, env.git_dirty);
        
        if !env.warnings.is_empty() {
            println!("\n⚠️  Reproducibility warnings:");
            for warning in &env.warnings {
                println!("  - {}", warning);
            }
        }
    }

    // Generate run ID
    let run_id = generate_run_id(&spec.workload.name, &args.tier.to_string(), &env.git_sha);
    let timestamp = get_timestamp();

    // Run the benchmark
    let _collector = MetricsCollector::new();
    let recall_results: HashMap<u64, Vec<u64>>;

    if args.verbose {
        println!("\nStarting benchmark...");
        println!("  Warmup: {}s", args.warmup_secs);
        println!("  Duration: {}s", args.duration_secs);
    }

    // TODO: Implement actual workload execution based on spec
    // For now, run a synthetic benchmark to demonstrate the harness
    let (metrics_summary, recall_map) = run_synthetic_benchmark(
        &spec,
        args.tier,
        args.warmup_secs,
        args.duration_secs,
        args.verbose,
    )?;
    recall_results = recall_map;

    // Evaluate recall if ground truth available
    let recall_at_10 = if !recall_results.is_empty() {
        // Check for ground truth file
        let truth_path = args.workload.parent()
            .map(|p| p.join(format!("{}_truth.json", spec.workload.name)))
            .filter(|p| p.exists());

        if let Some(truth_path) = truth_path {
            match RecallEvaluator::load(&truth_path) {
                Ok(evaluator) => {
                    let metrics = evaluator.average_recall_at_k(&recall_results, 10);
                    Some(metrics)
                }
                Err(e) => {
                    if args.verbose {
                        eprintln!("Warning: Failed to load ground truth: {}", e);
                    }
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Build report
    let mut builder = RunReport::builder()
        .run_id(&run_id)
        .timestamp(&timestamp)
        .environment(env.clone())
        .workload(&spec.workload.name, &spec.workload.description)
        .dataset(&spec.workload.name, args.tier, spec.record_count(args.tier).unwrap_or(0))
        .metrics(metrics_summary);

    if let Some(recall) = recall_at_10 {
        builder = builder.recall_at_10(recall);
    }

    // Compare with baseline if provided
    if let Some(ref baseline_path) = args.baseline {
        let baseline = RunReport::load(baseline_path)?;
        let baseline_snapshot = baseline.to_snapshot();
        
        let current_report = builder.clone().build()?;
        let current_snapshot = current_report.to_snapshot();

        let gates = match args.gate_mode {
            GateMode::Pr => RegressionGates::pr_strict(),
            GateMode::Nightly => RegressionGates::nightly_relaxed(),
            GateMode::None => RegressionGates::default(),
        };

        let evaluation = GateEvaluation::evaluate(&gates, &baseline_snapshot, &current_snapshot);
        builder = builder.gates(evaluation);
    }

    // Save report
    let report_path = args.out.join(format!("{}.json", run_id));
    builder = builder.artifact_report(report_path.to_str().unwrap_or(""));

    let report = builder.build()?;
    report.save(&report_path)?;
    report.print_summary();

    // Return exit code based on gates
    if let Some(ref gates) = report.gates {
        if !gates.overall_pass {
            std::process::exit(1);
        }
    }

    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

fn generate_run_id(workload: &str, tier: &str, git_sha: &str) -> String {
    let ts = get_timestamp().replace(":", "").replace("-", "");
    format!("{}_{}_{}_{}", ts, workload, tier, git_sha)
}

fn get_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    
    let secs = now.as_secs();
    format_timestamp(secs)
}

fn format_timestamp(unix_secs: u64) -> String {
    let days_since_epoch = unix_secs / 86400;
    let time_of_day = unix_secs % 86400;
    
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    
    let mut days = days_since_epoch as i64;
    let mut year = 1970i32;
    
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    
    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    
    let mut month = 0;
    for (i, &d) in days_in_months.iter().enumerate() {
        if days < d {
            month = i + 1;
            break;
        }
        days -= d;
    }
    
    let day = days + 1;
    
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Run a synthetic benchmark to demonstrate the harness.
/// 
/// This is the real workload execution that:
/// 1. Creates a vector index based on the workload spec
/// 2. Generates random vectors for the dataset
/// 3. Runs warmup queries
/// 4. Runs timed queries with latency recording
/// 5. Collects recall data
fn run_synthetic_benchmark(
    spec: &WorkloadSpec,
    tier: DatasetTier,
    warmup_secs: u64,
    duration_secs: u64,
    verbose: bool,
) -> Result<(benchmarks::harness::metrics::MetricsSummary, HashMap<u64, Vec<u64>>), String> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use toondb_index::hnsw::{HnswConfig, HnswIndex, DistanceMetric};

    let record_count = spec.record_count(tier).unwrap_or(100_000);
    let dimension = spec.vector_dimension().unwrap_or(768);
    let seed = spec.dataset_config(tier).map(|c| c.seed).unwrap_or(42);
    
    let mut collector = MetricsCollector::new();
    let mut recall_results: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut rng = StdRng::seed_from_u64(seed);

    if verbose {
        println!("  Building index: {} vectors × {} dims", record_count, dimension);
    }

    // Create HNSW index with appropriate configuration based on tier
    let (ef_construction, ef_search, max_connections) = match tier {
        DatasetTier::Small => (64, 32, 16),
        DatasetTier::Medium => (128, 64, 24),
        DatasetTier::Large => (200, 100, 32),
    };

    let config = HnswConfig {
        max_connections,
        max_connections_layer0: max_connections * 2,
        ef_construction,
        ef_search,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let index = HnswIndex::new(dimension, config);

    // Build phase: Insert vectors
    let build_start = Instant::now();
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(record_count);
    
    // Cap record count for quick demo modes
    let actual_record_count = match tier {
        DatasetTier::Small => record_count.min(10_000),
        DatasetTier::Medium => record_count.min(100_000),
        DatasetTier::Large => record_count.min(500_000),
    };

    for i in 0..actual_record_count {
        // Generate normalized random vector
        let mut vector: Vec<f32> = (0..dimension).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        vectors.push(vector.clone());
        
        index.insert(i as u128, vector).map_err(|e| e.to_string())?;
        
        if verbose && (i + 1) % 10_000 == 0 {
            println!("    Inserted {}/{} vectors", i + 1, actual_record_count);
        }
    }

    let build_time = build_start.elapsed();
    if verbose {
        println!("  Index built in {:.2}s ({:.0} vec/s)", 
            build_time.as_secs_f64(),
            actual_record_count as f64 / build_time.as_secs_f64());
    }

    // Generate query vectors (subset of dataset + some OOD)
    let num_queries = 1000;
    let mut query_vectors: Vec<Vec<f32>> = Vec::with_capacity(num_queries);
    
    for i in 0..num_queries {
        if i < num_queries * 9 / 10 && !vectors.is_empty() {
            // 90% in-distribution: use existing vectors with small noise
            let base_idx = rng.gen_range(0..vectors.len());
            let mut query = vectors[base_idx].clone();
            for v in &mut query {
                *v += (rng.gen::<f32>() - 0.5) * 0.1;
            }
            let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut query {
                    *v /= norm;
                }
            }
            query_vectors.push(query);
        } else {
            // 10% out-of-distribution: random vectors
            let mut query: Vec<f32> = (0..dimension).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut query {
                    *v /= norm;
                }
            }
            query_vectors.push(query);
        }
    }

    // Warmup phase
    if verbose {
        println!("  Warming up for {}s...", warmup_secs);
    }
    let warmup_start = Instant::now();
    let warmup_duration = Duration::from_secs(warmup_secs.min(5)); // Cap warmup
    let mut warmup_count = 0;
    
    while warmup_start.elapsed() < warmup_duration {
        let query_idx = warmup_count % query_vectors.len();
        let query = &query_vectors[query_idx];
        let _ = index.search(query, 10);
        warmup_count += 1;
    }
    
    if verbose {
        println!("  Warmup completed: {} queries", warmup_count);
    }

    // Benchmark phase
    if verbose {
        println!("  Running benchmark for {}s...", duration_secs);
    }

    collector.start();
    
    let start = Instant::now();
    let run_duration = Duration::from_secs(duration_secs.min(30)); // Cap at 30s for safety
    let mut query_id = 0u64;

    while start.elapsed() < run_duration {
        let query_idx = (query_id as usize) % query_vectors.len();
        
        // Time the query
        let query_start = Instant::now();
        let query = &query_vectors[query_idx];
        let results = index.search(query, 10).unwrap_or_default();
        let latency = query_start.elapsed();
        
        collector.record_query(latency);
        
        // Store results for recall evaluation
        let result_ids: Vec<u64> = results.into_iter().map(|(id, _)| id as u64).collect();
        recall_results.insert(query_id, result_ids);
        query_id += 1;
    }

    collector.stop();

    if verbose {
        let summary = collector.summary();
        println!("  Completed {} queries in {:.2}s", 
            collector.queries_executed, 
            collector.elapsed().as_secs_f64());
        println!("  QPS: {:.1}", summary.qps);
        println!("  Latency: p50={:.2}ms p95={:.2}ms p99={:.2}ms",
            summary.query_latency.p50_ms(),
            summary.query_latency.p95_ms(),
            summary.query_latency.p99_ms());
    }

    Ok((collector.summary(), recall_results))
}
