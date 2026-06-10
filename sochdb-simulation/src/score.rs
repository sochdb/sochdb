//! Score simulated results against expected targets.

use crate::engine::{OpResult, ScenarioResult};
use crate::expected::{ExpectedStore, ExpectedTarget, TargetUnit};
use colored::Colorize;
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    Pass,
    Warn,
    Fail,
    NoTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    pub workload: String,
    pub grade: Grade,
    pub metric: String,
    pub simulated: f64,
    pub expected: Option<f64>,
    pub delta_pct: Option<f64>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    pub scenario_id: String,
    pub results: Vec<ScoredResult>,
    pub pass_count: usize,
    pub warn_count: usize,
    pub fail_count: usize,
    pub no_target_count: usize,
    pub overall_grade: Grade,
}

pub struct Scorer {
    store: ExpectedStore,
}

impl Scorer {
    pub fn new(store: ExpectedStore) -> Self {
        Self { store }
    }

    pub fn score_scenario(&self, result: &ScenarioResult) -> Scorecard {
        let mut scored = Vec::new();

        for op in &result.operations {
            let mut op_scored = false;

            if let Some(target) = self.store.target_by_workload(&op.workload) {
                scored.extend(self.score_op(op, target));
                op_scored = true;
            }

            if op.workload == "context_query" {
                if let Some(qs) = op.quality_score {
                    for metric in ["retrieval_recall_at_5", "retrieval_mrr", "retrieval_ndcg_at_5"] {
                        if let Some(t) = self.store.target_by_workload(metric) {
                            scored.push(self.score_quality(metric, qs, t));
                            op_scored = true;
                        }
                    }
                }
            }

            if !op_scored {
                scored.push(ScoredResult {
                    workload: op.workload.clone(),
                    grade: Grade::NoTarget,
                    metric: "none".into(),
                    simulated: op.throughput_ops_sec,
                    expected: None,
                    delta_pct: None,
                    message: "No expected target defined".into(),
                });
            }
        }

        let pass_count = scored.iter().filter(|s| s.grade == Grade::Pass).count();
        let warn_count = scored.iter().filter(|s| s.grade == Grade::Warn).count();
        let fail_count = scored.iter().filter(|s| s.grade == Grade::Fail).count();
        let no_target_count = scored.iter().filter(|s| s.grade == Grade::NoTarget).count();

        let overall_grade = if fail_count > 0 {
            Grade::Fail
        } else if warn_count > 0 {
            Grade::Warn
        } else if pass_count > 0 {
            Grade::Pass
        } else {
            Grade::NoTarget
        };

        Scorecard {
            scenario_id: result.scenario_id.clone(),
            results: scored,
            pass_count,
            warn_count,
            fail_count,
            no_target_count,
            overall_grade,
        }
    }

    fn score_op(&self, op: &OpResult, target: &ExpectedTarget) -> Vec<ScoredResult> {
        let mut results = Vec::new();

        if let Some(expected_tp) = target.throughput_ops_sec {
            results.push(score_metric(
                &op.workload,
                "throughput_ops_sec",
                op.throughput_ops_sec,
                expected_tp,
                target.tolerance_pct,
                TargetUnit::Throughput,
            ));
        }
        if let Some(expected_p50) = target.p50_us {
            results.push(score_metric(
                &op.workload,
                "p50_us",
                op.p50_us,
                expected_p50,
                target.tolerance_pct,
                TargetUnit::LatencyCeiling,
            ));
        }
        if let Some(expected_p99) = target.p99_us {
            results.push(score_metric(
                &op.workload,
                "p99_us",
                op.p99_us,
                expected_p99,
                target.tolerance_pct,
                TargetUnit::LatencyCeiling,
            ));
        }
        if let Some(expected_score) = target.score {
            let simulated = op.quality_score.unwrap_or(0.0);
            results.push(score_metric(
                &op.workload,
                "quality_score",
                simulated,
                expected_score,
                target.tolerance_pct,
                target.unit_kind(),
            ));
        }

        results
    }

    fn score_quality(&self, workload: &str, simulated: f64, target: &ExpectedTarget) -> ScoredResult {
        let expected = target.score.unwrap_or(0.0);
        score_metric(workload, "quality", simulated, expected, target.tolerance_pct, TargetUnit::Ratio)
    }

    pub fn print_scorecard(&self, card: &Scorecard) {
        let grade_str = match card.overall_grade {
            Grade::Pass => "PASS".green().bold(),
            Grade::Warn => "WARN".yellow().bold(),
            Grade::Fail => "FAIL".red().bold(),
            Grade::NoTarget => "NO TARGET".dimmed(),
        };

        println!(
            "\n{} {} — {} pass, {} warn, {} fail, {} no-target",
            "Scorecard:".bold(),
            grade_str,
            card.pass_count,
            card.warn_count,
            card.fail_count,
            card.no_target_count,
        );

        let mut table = Table::new();
        table.set_header(vec!["Workload", "Grade", "Metric", "Simulated", "Expected", "Δ%"]);

        for r in &card.results {
            let grade_cell = match r.grade {
                Grade::Pass => Cell::new("✓ PASS").fg(Color::Green),
                Grade::Warn => Cell::new("⚠ WARN").fg(Color::Yellow),
                Grade::Fail => Cell::new("✗ FAIL").fg(Color::Red),
                Grade::NoTarget => Cell::new("—").fg(Color::DarkGrey),
            };

            let delta = r
                .delta_pct
                .map(|d| format!("{d:+.1}%"))
                .unwrap_or_else(|| "—".into());
            let expected = r
                .expected
                .map(|e| format!("{e:.2}"))
                .unwrap_or_else(|| "—".into());

            table.add_row(vec![
                Cell::new(&r.workload),
                grade_cell,
                Cell::new(&r.metric),
                Cell::new(format!("{:.2}", r.simulated)),
                Cell::new(expected),
                Cell::new(delta),
            ]);
        }

        println!("{table}");
    }
}

fn score_metric(
    workload: &str,
    metric: &str,
    simulated: f64,
    expected: f64,
    tolerance_pct: f64,
    unit: TargetUnit,
) -> ScoredResult {
    let delta_pct = if expected > 0.0 {
        (simulated - expected) / expected * 100.0
    } else {
        0.0
    };

    let grade = match unit {
        TargetUnit::Throughput | TargetUnit::ThroughputFloor | TargetUnit::Ratio | TargetUnit::RatioFloor => {
            if simulated >= expected * (1.0 - tolerance_pct / 100.0) {
                if simulated >= expected {
                    Grade::Pass
                } else {
                    Grade::Warn
                }
            } else {
                Grade::Fail
            }
        }
        TargetUnit::LatencyCeiling | TargetUnit::Milliseconds => {
            if simulated <= expected * (1.0 + tolerance_pct / 100.0) {
                if simulated <= expected {
                    Grade::Pass
                } else {
                    Grade::Warn
                }
            } else {
                Grade::Fail
            }
        }
    };

    let message = match grade {
        Grade::Pass => format!("{metric} within target"),
        Grade::Warn => format!("{metric} close to boundary ({delta_pct:+.1}%)"),
        Grade::Fail => format!("{metric} missed target ({delta_pct:+.1}%)"),
        Grade::NoTarget => "no target".into(),
    };

    ScoredResult {
        workload: workload.into(),
        grade,
        metric: metric.into(),
        simulated,
        expected: Some(expected),
        delta_pct: Some(delta_pct),
        message,
    }
}

pub fn export_json(scorecards: &[Scorecard], path: &std::path::Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(scorecards)?;
    std::fs::write(path, json)
}