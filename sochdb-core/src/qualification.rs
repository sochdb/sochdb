// SPDX-License-Identifier: AGPL-3.0-or-later
//
// SochDB - A unified database for AI-native applications
// Copyright (C) 2025 Sushanth Reddy Vanagala
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Evidence-based capability promotion.
//!
//! [`capability`](crate::capability) states what this build claims. It is an
//! honest document, but it is only a document: `Maturity::Supported` is a value
//! anyone can type, and the build passes either way. A consumer deciding
//! whether to enable a SochDB capability has nothing to check the claim
//! against, so in practice it checks whether the server answers a health probe
//! -- which measures that a process is running and nothing else.
//!
//! This module supplies the missing half. A maturity level names the gates it
//! requires; a build presents an [`EvidenceBundle`]; [`qualify`] decides. The
//! decision fails closed at every step.
//!
//! # Absent evidence is failing evidence
//!
//! The single rule that matters. A gate that was never run is
//! [`GateOutcome::NotRun`] and is treated exactly like a gate that failed,
//! because the alternative -- treating an unmeasured property as satisfied --
//! makes every gate optional the moment someone forgets to wire one up, and
//! forgetting is silent. A suite that skips its policy-isolation test reports
//! green, which is the worst possible reading of that particular skip.
//!
//! Everything else follows from it: a bundle whose commit does not match the
//! build is not evidence about this build; a bundle missing a required gate is
//! not sufficient; a gate whose measurement is unorderable did not measure
//! anything.
//!
//! # Metric definitions are shared, not parallel
//!
//! [`Metric`] carries its own definition and its own direction of improvement,
//! and both the gate thresholds and runtime telemetry name the same values.
//! Benchmark evidence measured one way and production metrics measured another
//! way cannot be compared, and the divergence is invisible: both are floats,
//! both have plausible names, and the dashboard agrees with itself.

use std::collections::BTreeMap;
use std::fmt;

/// A property a build must demonstrate.
///
/// The list is closed, so adding a category is a change every maturity table
/// has to acknowledge rather than a string somebody may or may not use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateKind {
    /// The wire contract is compatible with the version it claims.
    ProtocolCompatibility,
    /// Search returns what exact search returns.
    VectorCorrectness,
    /// Approximate search finds enough of the true neighbours.
    Recall,
    /// Filters admit exactly the rows they should.
    FilterCorrectness,
    /// No result crosses a policy or tenant boundary.
    PolicyIsolation,
    /// State survives a restart.
    RestartRecovery,
    /// Acknowledged writes are not lost.
    Durability,
    /// Cancelled work stops and releases what it held.
    Cancellation,
    /// Behaviour holds under sustained concurrency.
    Load,
    /// Memory and file descriptors stay inside their budget.
    ResourceBounds,
    /// Hybrid ranking is measurably better than either lane alone.
    HybridRelevance,
    /// The cache never serves across a scope boundary.
    CacheIsolation,
    /// Distributed results equal centralized results.
    DistributedMerge,
}

impl GateKind {
    /// Every gate, in a fixed order.
    pub const ALL: &'static [GateKind] = &[
        GateKind::ProtocolCompatibility,
        GateKind::VectorCorrectness,
        GateKind::Recall,
        GateKind::FilterCorrectness,
        GateKind::PolicyIsolation,
        GateKind::RestartRecovery,
        GateKind::Durability,
        GateKind::Cancellation,
        GateKind::Load,
        GateKind::ResourceBounds,
        GateKind::HybridRelevance,
        GateKind::CacheIsolation,
        GateKind::DistributedMerge,
    ];

    /// A stable identifier, used in evidence and in metric labels.
    pub const fn id(self) -> &'static str {
        match self {
            GateKind::ProtocolCompatibility => "protocol_compatibility",
            GateKind::VectorCorrectness => "vector_correctness",
            GateKind::Recall => "recall",
            GateKind::FilterCorrectness => "filter_correctness",
            GateKind::PolicyIsolation => "policy_isolation",
            GateKind::RestartRecovery => "restart_recovery",
            GateKind::Durability => "durability",
            GateKind::Cancellation => "cancellation",
            GateKind::Load => "load",
            GateKind::ResourceBounds => "resource_bounds",
            GateKind::HybridRelevance => "hybrid_relevance",
            GateKind::CacheIsolation => "cache_isolation",
            GateKind::DistributedMerge => "distributed_merge",
        }
    }

    /// Whether a failure here is a security failure rather than a quality one.
    ///
    /// The distinction drives scheduling, not strictness. A security gate has
    /// to be fast enough to sit in the merge gate, because "we run it nightly"
    /// means authorization leaks are merged and live for up to a day. Quality
    /// gates can be scheduled.
    pub const fn is_security(self) -> bool {
        matches!(
            self,
            GateKind::PolicyIsolation | GateKind::CacheIsolation | GateKind::FilterCorrectness
        )
    }
}

impl fmt::Display for GateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Which direction counts as better for a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Higher is better: recall, throughput.
    HigherIsBetter,
    /// Lower is better: latency, unauthorized-result rate.
    LowerIsBetter,
}

/// A measurable quantity, carrying the definition it is measured under.
///
/// The definition is attached to the metric rather than written in a document
/// beside it, because the failure this prevents is a benchmark and a dashboard
/// measuring subtly different things while agreeing on a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Metric {
    /// Fraction of true k nearest neighbours returned.
    RecallAtK,
    /// Normalised discounted cumulative gain at k.
    NdcgAtK,
    /// Mean reciprocal rank of the first relevant result.
    MeanReciprocalRank,
    /// Median end-to-end latency in milliseconds.
    LatencyP50Ms,
    /// 95th percentile end-to-end latency in milliseconds.
    LatencyP95Ms,
    /// 99th percentile end-to-end latency in milliseconds.
    LatencyP99Ms,
    /// Sustained completed queries per second.
    QueriesPerSecond,
    /// Resident bytes per stored vector.
    BytesPerVector,
    /// Bytes written during index build per byte of input.
    BuildAmplification,
    /// Seconds from process start to serving.
    RecoverySeconds,
    /// Fraction of results a requester was not entitled to.
    UnauthorizedResultRate,
    /// Fraction of results absent from a centralized baseline, or present in it
    /// and missing here.
    ResultDivergence,
}

impl Metric {
    /// A stable identifier shared by evidence and telemetry.
    pub const fn id(self) -> &'static str {
        match self {
            Metric::RecallAtK => "recall_at_k",
            Metric::NdcgAtK => "ndcg_at_k",
            Metric::MeanReciprocalRank => "mrr",
            Metric::LatencyP50Ms => "latency_p50_ms",
            Metric::LatencyP95Ms => "latency_p95_ms",
            Metric::LatencyP99Ms => "latency_p99_ms",
            Metric::QueriesPerSecond => "queries_per_second",
            Metric::BytesPerVector => "bytes_per_vector",
            Metric::BuildAmplification => "build_amplification",
            Metric::RecoverySeconds => "recovery_seconds",
            Metric::UnauthorizedResultRate => "unauthorized_result_rate",
            Metric::ResultDivergence => "result_divergence",
        }
    }

    /// How the quantity is defined, precisely enough to reimplement.
    pub const fn definition(self) -> &'static str {
        match self {
            Metric::RecallAtK => {
                "|returned ids intersect true k nearest| / k, averaged unweighted over the query \
                 set; ties in the ground truth resolved by ascending id"
            }
            Metric::NdcgAtK => {
                "DCG at k over graded relevance with log2(rank + 1) discount, divided by ideal DCG \
                 at k; queries with no relevant document score 0, not excluded"
            }
            Metric::MeanReciprocalRank => {
                "mean of 1 / rank of the first relevant result, 1-based; queries with no relevant \
                 result contribute 0"
            }
            Metric::LatencyP50Ms => {
                "50th percentile of client-observed wall-clock milliseconds from request write to \
                 last response byte, nearest-rank, including queueing"
            }
            Metric::LatencyP95Ms => {
                "95th percentile of client-observed wall-clock milliseconds from request write to \
                 last response byte, nearest-rank, including queueing"
            }
            Metric::LatencyP99Ms => {
                "99th percentile of client-observed wall-clock milliseconds from request write to \
                 last response byte, nearest-rank, including queueing"
            }
            Metric::QueriesPerSecond => {
                "completed queries divided by wall-clock seconds of the measurement window, \
                 excluding warm-up; failed and cancelled queries are not completions"
            }
            Metric::BytesPerVector => {
                "resident set size attributable to the index divided by the number of stored \
                 vectors, at steady state after build"
            }
            Metric::BuildAmplification => {
                "total bytes written to durable storage during build divided by the byte size of \
                 the input vectors"
            }
            Metric::RecoverySeconds => {
                "wall-clock seconds from process start to the first successful query at the \
                 pre-restart generation"
            }
            Metric::UnauthorizedResultRate => {
                "results the requester was not entitled to, divided by all results returned; the \
                 numerator counts rows, not responses"
            }
            Metric::ResultDivergence => {
                "symmetric difference between distributed and centralized exact result ids, \
                 divided by k"
            }
        }
    }

    /// Which direction is an improvement.
    pub const fn direction(self) -> Direction {
        match self {
            Metric::RecallAtK
            | Metric::NdcgAtK
            | Metric::MeanReciprocalRank
            | Metric::QueriesPerSecond => Direction::HigherIsBetter,
            Metric::LatencyP50Ms
            | Metric::LatencyP95Ms
            | Metric::LatencyP99Ms
            | Metric::BytesPerVector
            | Metric::BuildAmplification
            | Metric::RecoverySeconds
            | Metric::UnauthorizedResultRate
            | Metric::ResultDivergence => Direction::LowerIsBetter,
        }
    }

    /// Whether `measured` satisfies `required` for this metric.
    ///
    /// Direction comes from the metric, so a threshold cannot be compared the
    /// wrong way round by a caller who assumed higher was better. An
    /// unorderable measurement never satisfies a threshold: `NaN` compares
    /// false against everything, and the comparison is written so that lands in
    /// the failing branch rather than the passing one.
    pub fn satisfies(self, measured: f64, required: f64) -> bool {
        if measured.is_nan() || required.is_nan() {
            return false;
        }
        match self.direction() {
            Direction::HigherIsBetter => measured >= required,
            Direction::LowerIsBetter => measured <= required,
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// A threshold a gate must clear.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Threshold {
    /// What is measured.
    pub metric: Metric,
    /// The value that must be met or beaten.
    pub required: f64,
}

impl Threshold {
    /// Name a threshold.
    pub const fn new(metric: Metric, required: f64) -> Self {
        Self { metric, required }
    }
}

/// What a gate observed.
#[derive(Debug, Clone, PartialEq)]
pub enum GateOutcome {
    /// The gate ran and cleared its threshold.
    Passed { metric: Metric, measured: f64 },
    /// The gate ran and did not clear its threshold.
    Failed {
        metric: Metric,
        measured: f64,
        required: f64,
    },
    /// The gate ran and could not complete.
    Errored { detail: String },
    /// The gate did not run.
    ///
    /// Not a separate, softer state. This exists so that a bundle can say
    /// "absent" explicitly instead of a consumer inferring it from a missing
    /// map entry, and it is refused exactly as a failure is.
    NotRun,
}

impl GateOutcome {
    /// Whether this outcome permits promotion.
    pub fn is_pass(&self) -> bool {
        matches!(self, GateOutcome::Passed { .. })
    }

    /// A stable label for metrics.
    pub fn label(&self) -> &'static str {
        match self {
            GateOutcome::Passed { .. } => "passed",
            GateOutcome::Failed { .. } => "failed",
            GateOutcome::Errored { .. } => "errored",
            GateOutcome::NotRun => "not_run",
        }
    }
}

/// Which build the evidence is about.
///
/// Compared exactly against the build being promoted. Evidence about a
/// neighbouring commit is not weaker evidence about this one, it is evidence
/// about something else, and the whole point of a qualification gate is that it
/// refuses to reason about a build it did not measure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildIdentity {
    /// The exact source commit, full length.
    pub commit: String,
    /// The digest of the image the evidence was produced from.
    pub image_digest: String,
}

impl BuildIdentity {
    /// Name a build.
    pub fn new(commit: impl Into<String>, image_digest: impl Into<String>) -> Self {
        Self {
            commit: commit.into(),
            image_digest: image_digest.into(),
        }
    }

    /// Whether this identity is specific enough to be evidence.
    ///
    /// An abbreviated commit or a floating tag identifies a set of builds
    /// rather than one, and evidence about a set is not evidence about the
    /// member you are running.
    pub fn is_pinned(&self) -> bool {
        self.commit.len() == 40
            && self.commit.chars().all(|c| c.is_ascii_hexdigit())
            && self.image_digest.starts_with("sha256:")
            && self.image_digest.len() == 71
    }
}

/// Everything a build offers in support of a promotion.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceBundle {
    /// Which build this is about.
    pub build: BuildIdentity,
    /// What each gate observed.
    pub outcomes: BTreeMap<GateKind, GateOutcome>,
}

impl EvidenceBundle {
    /// An empty bundle for a build.
    ///
    /// Empty means every gate is [`GateOutcome::NotRun`], which qualifies for
    /// nothing above `LibraryOnly`. That is the correct starting point: a build
    /// that has demonstrated nothing has demonstrated nothing.
    pub fn new(build: BuildIdentity) -> Self {
        Self {
            build,
            outcomes: BTreeMap::new(),
        }
    }

    /// Record an outcome.
    #[must_use]
    pub fn with(mut self, gate: GateKind, outcome: GateOutcome) -> Self {
        self.outcomes.insert(gate, outcome);
        self
    }

    /// Record a measurement, deciding pass or fail from the threshold.
    ///
    /// The comparison happens here rather than at the call site, so a suite
    /// cannot report a pass it did not earn by comparing the wrong way round.
    #[must_use]
    pub fn measured(mut self, gate: GateKind, threshold: Threshold, measured: f64) -> Self {
        let outcome = if threshold.metric.satisfies(measured, threshold.required) {
            GateOutcome::Passed {
                metric: threshold.metric,
                measured,
            }
        } else {
            GateOutcome::Failed {
                metric: threshold.metric,
                measured,
                required: threshold.required,
            }
        };
        self.outcomes.insert(gate, outcome);
        self
    }

    /// What a gate observed. Absent entries read as [`GateOutcome::NotRun`].
    pub fn outcome(&self, gate: GateKind) -> GateOutcome {
        self.outcomes
            .get(&gate)
            .cloned()
            .unwrap_or(GateOutcome::NotRun)
    }

    /// A digest over the build identity and every gate outcome.
    ///
    /// Covers all thirteen gates, not just the recorded ones, so that removing
    /// a failing gate from the bundle changes the digest instead of quietly
    /// producing a cleaner-looking one. Every field is length-prefixed.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        push(&mut hasher, self.build.commit.as_bytes());
        push(&mut hasher, self.build.image_digest.as_bytes());
        for gate in GateKind::ALL {
            push(&mut hasher, gate.id().as_bytes());
            let outcome = self.outcome(*gate);
            push(&mut hasher, outcome.label().as_bytes());
            match outcome {
                GateOutcome::Passed { metric, measured } => {
                    push(&mut hasher, metric.id().as_bytes());
                    hasher.update(&measured.to_bits().to_le_bytes());
                }
                GateOutcome::Failed {
                    metric,
                    measured,
                    required,
                } => {
                    push(&mut hasher, metric.id().as_bytes());
                    hasher.update(&measured.to_bits().to_le_bytes());
                    hasher.update(&required.to_bits().to_le_bytes());
                }
                GateOutcome::Errored { detail } => push(&mut hasher, detail.as_bytes()),
                GateOutcome::NotRun => {}
            }
        }
        *hasher.finalize().as_bytes()
    }
}

/// Length-prefix a field before it joins the digest.
fn push(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

/// Why a promotion was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The evidence is about a different build.
    WrongBuild {
        expected: BuildIdentity,
        offered: BuildIdentity,
    },
    /// The build identity is not specific enough to be evidence.
    UnpinnedBuild,
    /// A required gate did not pass.
    GateNotPassed {
        gate: GateKind,
        outcome: GateOutcome,
    },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::WrongBuild { expected, offered } => write!(
                f,
                "evidence is about commit {} but the build is commit {}",
                offered.commit, expected.commit
            ),
            Refusal::UnpinnedBuild => write!(
                f,
                "build identity is not a full commit and an image digest, so it names a set of \
                 builds rather than one"
            ),
            Refusal::GateNotPassed { gate, outcome } => match outcome {
                GateOutcome::NotRun => write!(f, "the {gate} gate was not run"),
                GateOutcome::Errored { detail } => {
                    write!(f, "the {gate} gate could not complete: {detail}")
                }
                GateOutcome::Failed {
                    metric,
                    measured,
                    required,
                } => write!(
                    f,
                    "the {gate} gate measured {metric} at {measured} against a requirement of \
                     {required}"
                ),
                GateOutcome::Passed { .. } => write!(f, "the {gate} gate passed"),
            },
        }
    }
}

impl std::error::Error for Refusal {}

/// The gates a maturity level requires.
///
/// `LibraryOnly` and `Experimental` require nothing, because neither claims
/// anything: the first says the code exists, the second says it may vanish.
///
/// `Preview` requires only what applies to everything that is served at all:
/// that it speaks the protocol it claims, and that it does not leak across a
/// policy boundary. The retrieval-specific gates are deliberately not here. An
/// earlier version of this table demanded vector and filter correctness of
/// every `Preview` capability, and the manifest test immediately failed on
/// `embedding.maintenance.incremental` -- a capability that performs no vector
/// search and evaluates no filters. That is a false failure, and a gate that
/// fires wrongly is worse than no gate, because the standard response is to
/// switch it off.
///
/// `Supported` requires that it keeps working, which is where durability,
/// recovery, load and resource bounds enter, and where the security gates
/// become non-negotiable because `Supported` is the level a consumer is
/// entitled to build on. Gates that cannot apply to a stateless capability are
/// filtered by [`applicable_gates`] rather than being listed conditionally
/// here.
pub fn required_gates(maturity: crate::capability::Maturity) -> &'static [GateKind] {
    use crate::capability::Maturity;
    match maturity {
        Maturity::LibraryOnly | Maturity::Experimental => &[],
        Maturity::Preview => &[GateKind::ProtocolCompatibility, GateKind::PolicyIsolation],
        Maturity::Supported => &[
            GateKind::ProtocolCompatibility,
            GateKind::VectorCorrectness,
            GateKind::Recall,
            GateKind::FilterCorrectness,
            GateKind::PolicyIsolation,
            GateKind::RestartRecovery,
            GateKind::Durability,
            GateKind::Cancellation,
            GateKind::Load,
            GateKind::ResourceBounds,
            GateKind::CacheIsolation,
        ],
    }
}

/// Decide whether a build may claim a maturity level.
///
/// Returns every refusal rather than the first. A suite that fixes one gate,
/// reruns, and discovers a second is a slow way to learn what is missing, and
/// the slowness is itself an argument for lowering the bar.
/// The gates that can meaningfully apply to a capability with this durability.
///
/// A stateless capability has no state to lose across a restart and nothing to
/// make durable, so demanding durability and restart-recovery evidence of it
/// asks for a measurement that does not exist. Every other gate applies to
/// everything.
///
/// This filters, it never adds. A capability cannot escape a gate by declaring
/// a durability that suits it -- the durability it declares is the same one the
/// manifest validates and that clients negotiate against, so understating it to
/// dodge a gate means understating it to every consumer as well.
pub fn applicable_gates(
    durability: crate::capability::Durability,
    gates: &[GateKind],
) -> Vec<GateKind> {
    use crate::capability::Durability;
    gates
        .iter()
        .copied()
        .filter(|gate| {
            !(durability == Durability::Stateless
                && matches!(gate, GateKind::Durability | GateKind::RestartRecovery))
        })
        .collect()
}

/// Decide whether a capability may claim the maturity it declares.
///
/// The required set is the maturity baseline narrowed to what the capability's
/// durability makes measurable.
pub fn qualify_capability(
    capability: &crate::capability::Capability,
    build: &BuildIdentity,
    evidence: &EvidenceBundle,
) -> Result<[u8; 32], Vec<Refusal>> {
    let required = applicable_gates(capability.durability, required_gates(capability.maturity));
    qualify_against(&required, build, evidence)
}

pub fn qualify(
    maturity: crate::capability::Maturity,
    build: &BuildIdentity,
    evidence: &EvidenceBundle,
) -> Result<[u8; 32], Vec<Refusal>> {
    qualify_against(required_gates(maturity), build, evidence)
}

fn qualify_against(
    required: &[GateKind],
    build: &BuildIdentity,
    evidence: &EvidenceBundle,
) -> Result<[u8; 32], Vec<Refusal>> {
    let mut refusals = Vec::new();

    // A level that demands nothing is granted without inspecting the evidence,
    // so a build with no bundle at all can still be honest about being a
    // library. Levels that demand nothing are the ones that claim nothing.
    if required.is_empty() {
        return Ok(evidence.digest());
    }

    if !build.is_pinned() {
        refusals.push(Refusal::UnpinnedBuild);
    }
    if &evidence.build != build {
        refusals.push(Refusal::WrongBuild {
            expected: build.clone(),
            offered: evidence.build.clone(),
        });
    }

    for gate in required {
        let outcome = evidence.outcome(*gate);
        if !outcome.is_pass() {
            refusals.push(Refusal::GateNotPassed {
                gate: *gate,
                outcome,
            });
        }
    }

    if refusals.is_empty() {
        Ok(evidence.digest())
    } else {
        Err(refusals)
    }
}

/// The highest level this evidence supports for this build.
///
/// Walks down from `Supported`, so a bundle that is one gate short of
/// `Supported` still qualifies for `Preview` if it covers `Preview`'s gates.
/// That is the useful answer: it tells an operator where the build actually is
/// rather than only that it is not where they hoped.
pub fn highest_qualified(
    build: &BuildIdentity,
    evidence: &EvidenceBundle,
) -> crate::capability::Maturity {
    use crate::capability::Maturity;
    for level in [Maturity::Supported, Maturity::Preview] {
        if qualify(level, build, evidence).is_ok() {
            return level;
        }
    }
    Maturity::LibraryOnly
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{Maturity, manifest};

    fn build() -> BuildIdentity {
        BuildIdentity::new(
            "0123456789abcdef0123456789abcdef01234567",
            "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        )
    }

    fn passing(gates: &[GateKind]) -> EvidenceBundle {
        let mut bundle = EvidenceBundle::new(build());
        for gate in gates {
            bundle = bundle.with(
                *gate,
                GateOutcome::Passed {
                    metric: Metric::RecallAtK,
                    measured: 1.0,
                },
            );
        }
        bundle
    }

    /// The rule the whole module exists for. A gate that was never run is
    /// refused exactly as a gate that failed, because treating an unmeasured
    /// property as satisfied makes every gate optional the moment one is
    /// forgotten -- and forgetting is silent.
    #[test]
    fn a_gate_that_was_never_run_blocks_promotion() {
        let mut gates = required_gates(Maturity::Supported).to_vec();
        let dropped = gates.pop().expect("supported requires gates");
        let evidence = passing(&gates);
        let refusals = qualify(Maturity::Supported, &build(), &evidence).expect_err("refused");
        assert_eq!(
            refusals,
            vec![Refusal::GateNotPassed {
                gate: dropped,
                outcome: GateOutcome::NotRun
            }]
        );
    }

    /// An empty bundle qualifies for nothing that claims anything.
    #[test]
    fn a_build_that_has_demonstrated_nothing_qualifies_for_nothing() {
        let evidence = EvidenceBundle::new(build());
        assert!(qualify(Maturity::Supported, &build(), &evidence).is_err());
        assert!(qualify(Maturity::Preview, &build(), &evidence).is_err());
        assert_eq!(
            highest_qualified(&build(), &evidence),
            Maturity::LibraryOnly
        );
    }

    /// A level that claims nothing needs no evidence, so an honest library
    /// build is not obstructed by a gate it has no reason to run.
    #[test]
    fn a_level_that_claims_nothing_needs_no_evidence() {
        let evidence = EvidenceBundle::new(BuildIdentity::new("dirty", "none"));
        assert!(qualify(Maturity::LibraryOnly, &build(), &evidence).is_ok());
        assert!(qualify(Maturity::Experimental, &build(), &evidence).is_ok());
    }

    /// Evidence about a neighbouring commit is not weaker evidence about this
    /// one. It is evidence about something else.
    #[test]
    fn evidence_about_another_commit_is_refused() {
        let mut evidence = passing(required_gates(Maturity::Supported));
        evidence.build.commit = "fedcba9876543210fedcba9876543210fedcba98".into();
        let refusals = qualify(Maturity::Supported, &build(), &evidence).expect_err("refused");
        assert!(
            refusals
                .iter()
                .any(|r| matches!(r, Refusal::WrongBuild { .. }))
        );
    }

    /// An abbreviated commit or a floating tag names a set of builds, and
    /// evidence about a set says nothing about the member you are running.
    #[test]
    fn an_unpinned_build_cannot_be_qualified() {
        let loose = BuildIdentity::new("0123456", "latest");
        assert!(!loose.is_pinned());
        let mut evidence = passing(required_gates(Maturity::Supported));
        evidence.build = loose.clone();
        let refusals = qualify(Maturity::Supported, &loose, &evidence).expect_err("refused");
        assert!(refusals.contains(&Refusal::UnpinnedBuild));
    }

    #[test]
    fn a_full_commit_with_a_digest_is_pinned() {
        assert!(build().is_pinned());
        assert!(!BuildIdentity::new("z".repeat(40), "sha256:0").is_pinned());
    }

    /// Every refusal is reported, so a suite learns everything that is missing
    /// in one run rather than one gate per attempt.
    #[test]
    fn every_reason_for_refusal_is_reported_at_once() {
        let evidence = EvidenceBundle::new(BuildIdentity::new("short", "latest"));
        let refusals = qualify(
            Maturity::Supported,
            &BuildIdentity::new("short", "latest"),
            &evidence,
        )
        .expect_err("refused");
        assert_eq!(
            refusals.len(),
            1 + required_gates(Maturity::Supported).len(),
            "unpinned, plus one per missing gate"
        );
    }

    #[test]
    fn a_complete_bundle_qualifies_and_returns_its_digest() {
        let evidence = passing(required_gates(Maturity::Supported));
        let digest = qualify(Maturity::Supported, &build(), &evidence).expect("qualified");
        assert_eq!(digest, evidence.digest());
        assert_eq!(highest_qualified(&build(), &evidence), Maturity::Supported);
    }

    /// A bundle one gate short of `Supported` still reports where it actually
    /// is, rather than only that it is not where someone hoped.
    #[test]
    fn a_partial_bundle_reports_the_level_it_does_reach() {
        let evidence = passing(required_gates(Maturity::Preview));
        assert_eq!(highest_qualified(&build(), &evidence), Maturity::Preview);
        assert!(qualify(Maturity::Supported, &build(), &evidence).is_err());
    }

    /// Removing a failing gate from a bundle changes its digest instead of
    /// producing a cleaner-looking one, because the digest covers all thirteen
    /// gates and not only the recorded ones.
    #[test]
    fn deleting_a_failing_gate_does_not_produce_a_cleaner_digest() {
        let failing = EvidenceBundle::new(build()).with(
            GateKind::Recall,
            GateOutcome::Failed {
                metric: Metric::RecallAtK,
                measured: 0.4,
                required: 0.9,
            },
        );
        let removed = EvidenceBundle::new(build());
        assert_ne!(failing.digest(), removed.digest());
    }

    /// Two measurements that render identically but differ must not digest the
    /// same, which is precisely where a divergence matters.
    #[test]
    fn measurements_reach_the_digest_by_bits_not_by_rendering() {
        let a = EvidenceBundle::new(build()).with(
            GateKind::Recall,
            GateOutcome::Passed {
                metric: Metric::RecallAtK,
                measured: 1.0,
            },
        );
        let b = EvidenceBundle::new(build()).with(
            GateKind::Recall,
            GateOutcome::Passed {
                metric: Metric::RecallAtK,
                measured: 1.0 + f64::EPSILON,
            },
        );
        assert_eq!(format!("{:.6}", 1.0), format!("{:.6}", 1.0 + f64::EPSILON));
        assert_ne!(a.digest(), b.digest());
    }

    /// The direction of improvement comes from the metric, so a caller who
    /// assumed higher was better cannot record a pass by comparing a latency
    /// the wrong way round.
    #[test]
    fn a_threshold_cannot_be_compared_the_wrong_way_round() {
        let slow = EvidenceBundle::new(build()).measured(
            GateKind::Load,
            Threshold::new(Metric::LatencyP99Ms, 50.0),
            500.0,
        );
        assert!(!slow.outcome(GateKind::Load).is_pass());

        let fast = EvidenceBundle::new(build()).measured(
            GateKind::Load,
            Threshold::new(Metric::LatencyP99Ms, 50.0),
            10.0,
        );
        assert!(fast.outcome(GateKind::Load).is_pass());
    }

    /// A measurement that is not a number did not measure anything.
    #[test]
    fn an_unorderable_measurement_never_satisfies_a_threshold() {
        assert!(!Metric::RecallAtK.satisfies(f64::NAN, 0.0));
        assert!(!Metric::LatencyP99Ms.satisfies(f64::NAN, f64::INFINITY));
        assert!(!Metric::RecallAtK.satisfies(1.0, f64::NAN));
        let bundle = EvidenceBundle::new(build()).measured(
            GateKind::Recall,
            Threshold::new(Metric::RecallAtK, 0.9),
            f64::NAN,
        );
        assert!(!bundle.outcome(GateKind::Recall).is_pass());
    }

    #[test]
    fn a_threshold_met_exactly_passes() {
        assert!(Metric::RecallAtK.satisfies(0.9, 0.9));
        assert!(Metric::LatencyP99Ms.satisfies(50.0, 50.0));
        assert!(!Metric::RecallAtK.satisfies(0.9 - f64::EPSILON, 0.9));
    }

    /// A gate that could not complete is not a gate that passed. An errored
    /// suite is the most tempting thing to wave through, because it looks like
    /// infrastructure rather than a defect.
    #[test]
    fn a_gate_that_errored_blocks_promotion() {
        let mut evidence = passing(required_gates(Maturity::Supported));
        evidence = evidence.with(
            GateKind::Durability,
            GateOutcome::Errored {
                detail: "runner ran out of disk".into(),
            },
        );
        let refusals = qualify(Maturity::Supported, &build(), &evidence).expect_err("refused");
        assert_eq!(refusals.len(), 1);
        assert!(matches!(
            &refusals[0],
            Refusal::GateNotPassed {
                gate: GateKind::Durability,
                outcome: GateOutcome::Errored { .. }
            }
        ));
    }

    /// Security gates are required for anything a consumer may build on, and
    /// they are marked so they can be scheduled into the merge gate rather than
    /// run nightly -- nightly means a leak is merged and live for a day.
    #[test]
    fn every_security_gate_is_required_before_supported() {
        let required = required_gates(Maturity::Supported);
        for gate in GateKind::ALL.iter().filter(|g| g.is_security()) {
            assert!(required.contains(gate), "{gate} must gate Supported");
        }
    }

    /// Policy isolation is required even for `Preview`. A build that leaks
    /// across tenants is not a preview of anything anyone should try.
    #[test]
    fn policy_isolation_gates_even_preview() {
        assert!(required_gates(Maturity::Preview).contains(&GateKind::PolicyIsolation));
    }

    /// Every metric carries a definition specific enough to reimplement, so a
    /// benchmark and a dashboard cannot measure different things under one
    /// name.
    #[test]
    fn every_metric_defines_itself() {
        let metrics = [
            Metric::RecallAtK,
            Metric::NdcgAtK,
            Metric::MeanReciprocalRank,
            Metric::LatencyP50Ms,
            Metric::LatencyP95Ms,
            Metric::LatencyP99Ms,
            Metric::QueriesPerSecond,
            Metric::BytesPerVector,
            Metric::BuildAmplification,
            Metric::RecoverySeconds,
            Metric::UnauthorizedResultRate,
            Metric::ResultDivergence,
        ];
        for metric in metrics {
            assert!(
                metric.definition().len() > 40,
                "{metric} needs a definition, not a restatement of its name"
            );
        }
    }

    #[test]
    fn gate_and_metric_identifiers_are_unique() {
        let mut ids: Vec<&str> = GateKind::ALL.iter().map(|g| g.id()).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count);
    }

    /// The claim this repository makes, measured against the machinery that
    /// would have to back it.
    ///
    /// No evidence bundle exists in this tree. So every capability above
    /// `Experimental` is currently making a claim it cannot support, and this
    /// records exactly which ones rather than describing the situation in a
    /// comment. That is uncomfortable and it is the point: the gate was built
    /// to be told this, and the honest reading is that `Preview` in this
    /// manifest presently means "we believe it works", not "we measured it".
    ///
    /// The count is pinned so the gap cannot grow silently. Adding a `Preview`
    /// capability without evidence fails here and forces the number up
    /// deliberately; wiring up a gate suite lets it come down.
    #[test]
    fn every_capability_claiming_more_than_experimental_currently_lacks_evidence() {
        let no_evidence = EvidenceBundle::new(build());
        let manifest = manifest();
        let unbacked: Vec<&str> = manifest
            .capabilities
            .iter()
            .filter(|c| qualify_capability(c, &build(), &no_evidence).is_err())
            .map(|c| c.name.as_str())
            .collect();

        for capability in &manifest.capabilities {
            let claims_nothing = matches!(
                capability.maturity,
                Maturity::LibraryOnly | Maturity::Experimental
            );
            assert_eq!(
                claims_nothing,
                !unbacked.contains(&capability.name.as_str()),
                "{} claims {:?}",
                capability.name,
                capability.maturity
            );
        }

        assert_eq!(
            unbacked.len(),
            9,
            "unbacked Preview claims: {unbacked:?} -- raise this only deliberately"
        );
    }

    /// Nothing claims `Supported`, which is the strongest statement the
    /// manifest can make and the one no build here has earned.
    #[test]
    fn nothing_in_this_build_claims_supported() {
        assert!(
            manifest()
                .capabilities
                .iter()
                .all(|c| c.maturity != Maturity::Supported)
        );
    }

    /// A stateless capability has no state to lose, so demanding restart and
    /// durability evidence of it asks for a measurement that does not exist.
    #[test]
    fn stateless_capabilities_are_not_asked_for_durability_evidence() {
        use crate::capability::Durability;
        let full = required_gates(Maturity::Supported);
        let stateless = applicable_gates(Durability::Stateless, full);
        assert!(!stateless.contains(&GateKind::Durability));
        assert!(!stateless.contains(&GateKind::RestartRecovery));
        assert!(stateless.contains(&GateKind::PolicyIsolation));

        let durable = applicable_gates(Durability::Durable, full);
        assert_eq!(
            durable.len(),
            full.len(),
            "nothing is filtered for a durable capability"
        );
    }

    /// Filtering never adds a gate, so a capability cannot acquire a
    /// requirement by declaring an unusual durability -- and it cannot dodge
    /// one either without understating its durability to every consumer.
    #[test]
    fn applicability_only_ever_removes_gates() {
        use crate::capability::Durability;
        for durability in [
            Durability::Stateless,
            Durability::Ephemeral,
            Durability::Durable,
        ] {
            let filtered = applicable_gates(durability, required_gates(Maturity::Supported));
            assert!(
                filtered
                    .iter()
                    .all(|g| required_gates(Maturity::Supported).contains(g))
            );
        }
    }
}
