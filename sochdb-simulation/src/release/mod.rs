//! Release-quality simulation and validation.
//!
//! Covers all 10 release areas for a database engine / SDK release:
//!
//! 1. Storage correctness (WAL, crash recovery, checkpoints)
//! 2. Concurrency & MVCC (Loom, stress tests)
//! 3. FFI & SDK safety (panic firewall, clean errors)
//! 4. Packaging quality (wheels, npm, crates, version sync)
//! 5. CI / release gate (fmt, clippy -D warnings, full tests)
//! 6. Performance thresholds (benchmark regression)
//! 7. Backward compatibility (format version rejection)
//! 8. Security & supply chain (audit, deny, no secrets)
//! 9. Release operations (dry-run, checksums, notes)
//! 10. Performance simulation (standalone + distributed)

pub mod gate;
pub mod report;
pub mod validator;

use gate::ReleaseGateFile;
use std::path::Path;

pub fn load_gates() -> ReleaseGateFile {
    serde_json::from_str(include_str!("../../expected/release_gates.json"))
        .expect("valid release_gates.json")
}

pub fn load_gates_from_dir(path: &Path) -> std::io::Result<ReleaseGateFile> {
    let content = std::fs::read_to_string(path.join("release_gates.json"))?;
    Ok(serde_json::from_str(&content)?)
}

pub fn filter_gates<'a>(
    gates: &'a [gate::ReleaseGate],
    category: Option<&str>,
    priority: Option<gate::GatePriority>,
) -> Vec<&'a gate::ReleaseGate> {
    let cat_filter = category.and_then(|c| c.parse::<gate::GateCategory>().ok());
    gates
        .iter()
        .filter(|g| {
            cat_filter.map(|cf| g.category == cf.name()).unwrap_or(true)
                && priority.map(|p| g.priority == p).unwrap_or(true)
        })
        .collect()
}
