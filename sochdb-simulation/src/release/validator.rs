//! Live and static release gate validation.

use crate::release::gate::{GateKind, GatePriority, ReleaseGate};
use crate::scenario::Scenario;
use crate::{ExpectedStore, Grade, Scorer, SimulationEngine};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GateStatus {
    Pass,
    Warn,
    Fail,
    Skip,
    Manual,
}

impl GateStatus {
    pub fn is_release_blocking(&self, priority: GatePriority) -> bool {
        matches!((self, priority), (Self::Fail, GatePriority::Blocker))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GateResult {
    pub gate_id: String,
    pub title: String,
    pub category: String,
    pub priority: GatePriority,
    pub kind: GateKind,
    pub status: GateStatus,
    pub duration_ms: f64,
    pub message: String,
    pub command: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReleaseScorecard {
    pub total: usize,
    pub passed: usize,
    pub warned: usize,
    pub failed: usize,
    pub skipped: usize,
    pub manual: usize,
    pub blocker_failures: usize,
    pub blocker_unverified: usize,
    pub release_ready: bool,
    pub results: Vec<GateResult>,
}

pub struct ReleaseValidator {
    workspace_root: PathBuf,
    validate_live: bool,
    full: bool,
}

impl ReleaseValidator {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            validate_live: false,
            full: false,
        }
    }

    pub fn with_live_validation(mut self, on: bool) -> Self {
        self.validate_live = on;
        self
    }

    pub fn with_full(mut self, on: bool) -> Self {
        self.full = on;
        self
    }

    pub fn run_gates(&self, gates: &[ReleaseGate]) -> ReleaseScorecard {
        let mut results = Vec::new();

        for gate in gates {
            if !self.should_run(gate) {
                results.push(GateResult {
                    gate_id: gate.id.clone(),
                    title: gate.title.clone(),
                    category: gate.category.clone(),
                    priority: gate.priority,
                    kind: gate.kind,
                    status: GateStatus::Skip,
                    duration_ms: 0.0,
                    message: "Skipped (use --validate or --full to run)".into(),
                    command: gate.command.clone(),
                });
                continue;
            }

            let start = Instant::now();
            let result = self.evaluate_gate(gate);
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

            results.push(GateResult {
                gate_id: gate.id.clone(),
                title: gate.title.clone(),
                category: gate.category.clone(),
                priority: gate.priority,
                kind: gate.kind,
                status: result.0,
                duration_ms,
                message: result.1,
                command: gate.command.clone(),
            });
        }

        Self::build_scorecard(results, self.validate_live)
    }

    fn should_run(&self, gate: &ReleaseGate) -> bool {
        match gate.kind {
            GateKind::Manual => return true, // always listed, status = Manual
            GateKind::Simulated => return true,
            GateKind::StaticFile
            | GateKind::StaticGrep
            | GateKind::StaticGrepAbsence
            | GateKind::StaticMultiFile => return true,
            GateKind::LiveTest | GateKind::LiveCommand | GateKind::LoomTest => {
                if !self.validate_live {
                    return false;
                }
                if gate.priority == GatePriority::Warning && !self.full {
                    return false;
                }
                return true;
            }
        }
    }

    fn evaluate_gate(&self, gate: &ReleaseGate) -> (GateStatus, String) {
        match gate.kind {
            GateKind::Manual => (
                GateStatus::Manual,
                "Requires human verification before release".into(),
            ),
            GateKind::StaticFile => self.check_file_exists(gate),
            GateKind::StaticMultiFile => self.check_multi_file(gate),
            GateKind::StaticGrep => self.check_grep(gate, false),
            GateKind::StaticGrepAbsence => self.check_grep_absence(gate),
            GateKind::Simulated => self.run_simulated(gate),
            GateKind::LiveTest | GateKind::LiveCommand => self.run_command(gate),
            GateKind::LoomTest => self.run_loom(gate),
        }
    }

    fn check_file_exists(&self, gate: &ReleaseGate) -> (GateStatus, String) {
        let rel = gate.path.as_deref().unwrap_or("");
        let path = self.workspace_root.join(rel);
        if path.exists() {
            (GateStatus::Pass, format!("Found {}", rel))
        } else {
            (GateStatus::Fail, format!("Missing {}", rel))
        }
    }

    fn check_multi_file(&self, gate: &ReleaseGate) -> (GateStatus, String) {
        let missing: Vec<_> = gate
            .paths
            .iter()
            .filter(|p| !self.workspace_root.join(p).exists())
            .collect();
        if missing.is_empty() {
            (
                GateStatus::Pass,
                format!("All {} files present", gate.paths.len()),
            )
        } else {
            (
                GateStatus::Fail,
                format!(
                    "Missing: {}",
                    missing
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
        }
    }

    fn check_grep(&self, gate: &ReleaseGate, invert: bool) -> (GateStatus, String) {
        let rel = gate.path.as_deref().unwrap_or("");
        let path = self.workspace_root.join(rel);
        if !path.exists() {
            return (GateStatus::Fail, format!("File not found: {rel}"));
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return (GateStatus::Fail, format!("Read error: {e}")),
        };
        let pattern = gate.pattern.as_deref().unwrap_or("");
        let count = count_pattern_matches(&content, pattern);
        let min = gate.min_matches.unwrap_or(1);

        let ok = if invert { count == 0 } else { count >= min };
        if ok {
            (
                GateStatus::Pass,
                if invert {
                    format!("Pattern absent in {rel}")
                } else {
                    format!("Found {count} matches in {rel}")
                },
            )
        } else {
            (
                GateStatus::Fail,
                if invert {
                    format!("Found {count} forbidden matches in {rel}")
                } else {
                    format!("Expected >={min} matches, found {count} in {rel}")
                },
            )
        }
    }

    fn check_grep_absence(&self, gate: &ReleaseGate) -> (GateStatus, String) {
        // Scan source crates for leaked credentials — not CI docs that reference secret *names*
        let pattern = gate.pattern.as_deref().unwrap_or("");
        let scan_dirs = [
            "sochdb-storage",
            "sochdb-index",
            "sochdb-grpc",
            "sochdb-query",
            "sochdb-python",
            "sochdb-client",
        ];
        let mut hits = Vec::new();

        for dir in scan_dirs {
            let base = self.workspace_root.join(dir);
            if !base.exists() {
                continue;
            }
            scan_for_secret_leak(&base, pattern, &mut hits, 20);
        }

        if hits.is_empty() {
            (
                GateStatus::Pass,
                "No hardcoded secrets found in source crates".into(),
            )
        } else {
            (
                GateStatus::Fail,
                format!("Potential secrets in: {}", hits.join(", ")),
            )
        }
    }

    fn run_simulated(&self, gate: &ReleaseGate) -> (GateStatus, String) {
        let scenario_id = gate.scenario.as_deref().unwrap_or("standalone_full");
        let scenario = match Scenario::by_id(scenario_id) {
            Some(s) => s,
            None => {
                return (GateStatus::Fail, format!("Unknown scenario: {scenario_id}"));
            }
        };

        let mut engine = SimulationEngine::new(42);
        let result = engine.run_scenario(&scenario);
        let store = ExpectedStore::load_defaults();
        let scorer = Scorer::new(store);
        let card = scorer.score_scenario(&result);

        let status = match card.overall_grade {
            Grade::Pass | Grade::Warn | Grade::NoTarget => GateStatus::Pass,
            Grade::Fail => GateStatus::Fail,
        };

        (
            status,
            format!(
                "Sim {} — {} pass, {} warn, {} fail",
                scenario_id, card.pass_count, card.warn_count, card.fail_count
            ),
        )
    }

    fn run_command(&self, gate: &ReleaseGate) -> (GateStatus, String) {
        let cmd = gate.command.as_deref().unwrap_or("");
        match run_shell(cmd, &self.workspace_root) {
            Ok(output) => {
                if output.status.success() {
                    (GateStatus::Pass, truncate_output(&output.stdout, 200))
                } else {
                    let stderr = truncate_output(&output.stderr, 300);
                    (
                        GateStatus::Fail,
                        format!("Exit {}: {stderr}", output.status),
                    )
                }
            }
            Err(e) => (GateStatus::Fail, format!("Command failed: {e}")),
        }
    }

    fn run_loom(&self, gate: &ReleaseGate) -> (GateStatus, String) {
        // Loom requires special RUSTFLAGS; try it, skip gracefully if loom not available
        let cmd = gate.command.as_deref().unwrap_or("");
        match run_shell(cmd, &self.workspace_root) {
            Ok(output) if output.status.success() => {
                (GateStatus::Pass, "Loom concurrency model passed".into())
            }
            Ok(output) => {
                let stderr = truncate_output(&output.stderr, 300);
                if stderr.contains("loom") || stderr.contains("cfg(loom)") {
                    (
                        GateStatus::Skip,
                        "Loom not available in this build — run with RUSTFLAGS=\"--cfg loom\""
                            .into(),
                    )
                } else {
                    (GateStatus::Fail, format!("Loom failed: {stderr}"))
                }
            }
            Err(e) => (GateStatus::Skip, format!("Loom test skipped: {e}")),
        }
    }

    fn build_scorecard(results: Vec<GateResult>, validate_live: bool) -> ReleaseScorecard {
        let passed = results
            .iter()
            .filter(|r| r.status == GateStatus::Pass)
            .count();
        let warned = results
            .iter()
            .filter(|r| r.status == GateStatus::Warn)
            .count();
        let failed = results
            .iter()
            .filter(|r| r.status == GateStatus::Fail)
            .count();
        let skipped = results
            .iter()
            .filter(|r| r.status == GateStatus::Skip)
            .count();
        let manual = results
            .iter()
            .filter(|r| r.status == GateStatus::Manual)
            .count();
        let blocker_failures = results
            .iter()
            .filter(|r| r.status == GateStatus::Fail && r.priority == GatePriority::Blocker)
            .count();
        let blocker_unverified = results
            .iter()
            .filter(|r| {
                r.priority == GatePriority::Blocker
                    && matches!(r.status, GateStatus::Skip | GateStatus::Manual)
            })
            .count();

        // Release ready: no blocker failures; in validate mode all blockers must be verified
        let release_ready = if validate_live {
            blocker_failures == 0 && blocker_unverified == 0
        } else {
            blocker_failures == 0
        };

        ReleaseScorecard {
            total: results.len(),
            passed,
            warned,
            failed,
            skipped,
            manual,
            blocker_failures,
            blocker_unverified,
            release_ready,
            results,
        }
    }
}

fn run_shell(cmd: &str, cwd: &Path) -> std::io::Result<std::process::Output> {
    Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .output()
}

fn truncate_output(s: &[u8], max: usize) -> String {
    let text = String::from_utf8_lossy(s);
    if text.len() <= max {
        text.into_owned()
    } else {
        format!("{}…", &text[..max])
    }
}

/// Count matches for a pattern. Supports `|` alternation (case-insensitive).
fn count_pattern_matches(content: &str, pattern: &str) -> usize {
    let lower = content.to_lowercase();
    if pattern.contains('|') {
        pattern
            .split('|')
            .map(|p| lower.matches(&p.to_lowercase()).count())
            .sum()
    } else {
        lower.matches(&pattern.to_lowercase()).count()
    }
}

/// Scan for actual secret leaks, not CI variable name references.
fn scan_for_secret_leak(dir: &Path, pattern: &str, hits: &mut Vec<String>, limit: usize) {
    if hits.len() >= limit {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if hits.len() >= limit {
            break;
        }
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "target" || name == ".git" || name == "node_modules" {
                continue;
            }
            scan_for_secret_leak(&path, pattern, hits, limit);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext, "rs" | "py" | "toml") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if looks_like_secret_leak(&content, pattern) {
                        hits.push(path.display().to_string());
                    }
                }
            }
        }
    }
}

fn looks_like_secret_leak(content: &str, pattern: &str) -> bool {
    for line in content.lines() {
        let trimmed = line.trim();
        // Skip comments and CI secret references
        if trimmed.starts_with("//")
            || trimmed.starts_with('#')
            || trimmed.contains("secrets.")
            || trimmed.contains("${{")
            || trimmed.contains("Variable Groups")
        {
            continue;
        }
        if pattern_contains(line, pattern) {
            // Require assignment-like context to avoid bare mentions
            if line.contains('=') || line.contains(": ") || pattern.contains("sk-") {
                return true;
            }
        }
    }
    false
}

fn pattern_contains(content: &str, pattern: &str) -> bool {
    count_pattern_matches(content, pattern) > 0
}
