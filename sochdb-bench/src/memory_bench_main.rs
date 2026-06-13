//! Memory benchmark CLI: LoComo, LongMemEval, BEAM with exact token accounting.

use clap::Parser;
use sochdb_memory::MemoryStore;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sochdb-memory-bench")]
struct Args {
    #[arg(long, default_value = "locomo")]
    dataset: String,
    #[arg(long)]
    data: PathBuf,
    #[arg(long, default_value = "default")]
    namespace: String,
    #[arg(long, default_value = "memory_bench_report.json")]
    output: PathBuf,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let store = MemoryStore::with_defaults();

    let report = match args.dataset.as_str() {
        "locomo" => {
            let doc_map = sochdb_bench::memory_bench::locomo::ingest_conversations(
                &store,
                &args.namespace,
                &args.data,
            )?;
            let questions =
                sochdb_bench::memory_bench::locomo::load_questions(&args.data, &doc_map)?;
            let lags = std::collections::HashMap::new();
            sochdb_bench::memory_bench::scoring::run_retrieval_suite(
                &store,
                &args.namespace,
                "locomo",
                &questions,
                &lags,
            )
        }
        "longmemeval" => {
            sochdb_bench::memory_bench::longmemeval::ingest_haystacks(
                &store,
                &args.namespace,
                &args.data,
            )?;
            let questions = sochdb_bench::memory_bench::longmemeval::load_questions(&args.data)?;
            let lags = std::collections::HashMap::new();
            sochdb_bench::memory_bench::scoring::run_retrieval_suite(
                &store,
                &args.namespace,
                "longmemeval",
                &questions,
                &lags,
            )
        }
        "beam" => {
            let questions = sochdb_bench::memory_bench::beam::load_questions(&args.data)?;
            let lags = std::collections::HashMap::new();
            sochdb_bench::memory_bench::scoring::run_retrieval_suite(
                &store,
                &args.namespace,
                "beam",
                &questions,
                &lags,
            )
        }
        other => return Err(format!("unknown dataset: {other}")),
    };

    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&args.output, json).map_err(|e| e.to_string())?;
    println!(
        "dataset={} questions={} recall@5={:.1}% recall@10={:.1}% mrr={:.1}% median_tokens={} p50_us={} saved={}",
        report.dataset,
        report.questions,
        report.recall_at_5 * 100.0,
        report.recall_at_10 * 100.0,
        report.mrr * 100.0,
        report.median_tokens,
        report.p50_retrieval_us,
        args.output.display()
    );
    Ok(())
}
