// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The capability manifest: what a SochDB build actually promises.
//!
//! A module existing in this repository is not the same thing as a production
//! contract. Several subsystems here are compiled to prevent rot but are not
//! reachable from any served endpoint; others are reachable but hold their
//! state only in process memory. A client that infers what it may use from a
//! version string, a feature name, or the presence of a symbol will eventually
//! infer wrongly, and the failure will look like a correctness bug in the
//! client rather than a missing promise on the server.
//!
//! So a build publishes an explicit, machine-readable manifest, and clients
//! negotiate against it instead of guessing.
//!
//! # Why the maturity ladder is enforced rather than documented
//!
//! Capability tables written as prose drift, and they drift in one direction:
//! toward optimism, because promoting a line in a table is easy and retracting
//! one is embarrassing. [`CapabilityManifest::validate`] therefore refuses to
//! build a manifest whose claims contradict each other, and the repository's
//! own manifest is validated by a test. The rule that does most of the work is
//! that [`Maturity::Supported`] requires [`Durability::Durable`]: a capability
//! whose searchable state disappears on restart cannot be a production promise,
//! however well it performs while the process happens to be alive.
//!
//! The visible consequence is that this build currently advertises nothing as
//! `Supported`. That is the intended output, not a gap in the manifest.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The negotiated wire contract, versioned independently of the crate version.
///
/// Crate versions move for reasons that have nothing to do with the contract —
/// a dependency bump, a performance fix — so a client cannot use them to decide
/// what it may send.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ContractVersion {
    pub major: u32,
    pub minor: u32,
}

impl ContractVersion {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Whether a server offering `self` can serve a client needing `required`.
    ///
    /// Equal majors, and a server minor at least the client's. Minor versions
    /// add optional fields the client may ignore, so a newer server serves an
    /// older client; a newer client may reference fields an older server never
    /// learned to read, so the reverse is refused.
    pub const fn accepts(self, required: ContractVersion) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl fmt::Display for ContractVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// How much of a promise a capability carries.
///
/// Ordered, so a client can require "at least this much".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Maturity {
    /// Present in the codebase but not reachable through any served endpoint.
    /// Callable by an embedding application that links the crate directly.
    LibraryOnly,
    /// Behind the `experimental` feature. No compatibility promise; may change
    /// or vanish without a contract version bump.
    Experimental,
    /// Wired end to end and covered by tests, but without the durability,
    /// recall, or recovery qualification that `Supported` requires.
    Preview,
    /// Qualified for production: durable, covered by integration and restart
    /// evidence, and stable across the contract's major version.
    Supported,
}

impl Maturity {
    /// Whether this build may be relied upon by a first-party production
    /// integration.
    pub const fn is_production(self) -> bool {
        matches!(self, Maturity::Supported)
    }
}

impl fmt::Display for Maturity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Maturity::LibraryOnly => "library-only",
            Maturity::Experimental => "experimental",
            Maturity::Preview => "preview",
            Maturity::Supported => "supported",
        };
        f.write_str(s)
    }
}

/// Whether a capability's state survives a restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Durability {
    /// State is rebuilt deterministically from a persisted manifest.
    Durable,
    /// State exists only in process memory and is lost on restart. A caller
    /// must be able to reconstruct it and must not treat results as
    /// authoritative across a restart boundary.
    Ephemeral,
    /// The capability holds no state, so restart is not meaningful.
    Stateless,
}

/// A single advertised capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    /// Stable identifier. Clients match on this, never on the description.
    pub name: String,
    pub contract_version: ContractVersion,
    pub maturity: Maturity,
    pub durability: Durability,
    /// What the capability promises when used within its limits.
    pub guarantees: Vec<String>,
    /// Known boundaries a caller must respect or design around. A `Preview` or
    /// better capability must state at least one, because a capability with no
    /// known limits usually means nobody has looked.
    pub limits: Vec<String>,
}

impl Capability {
    pub fn new(
        name: impl Into<String>,
        contract_version: ContractVersion,
        maturity: Maturity,
        durability: Durability,
    ) -> Self {
        Self {
            name: name.into(),
            contract_version,
            maturity,
            durability,
            guarantees: Vec::new(),
            limits: Vec::new(),
        }
    }

    pub fn guarantee(mut self, text: impl Into<String>) -> Self {
        self.guarantees.push(text.into());
        self
    }

    pub fn limit(mut self, text: impl Into<String>) -> Self {
        self.limits.push(text.into());
        self
    }
}

/// A claim inside a manifest that contradicts another claim in the same manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// Two capabilities share a name, so negotiation would be ambiguous.
    DuplicateName(String),
    /// `Supported` was claimed for state that does not survive restart.
    SupportedButEphemeral(String),
    /// `Supported` or `Preview` was claimed without stating any limit.
    UnboundedClaim(String),
    /// A capability was advertised with no guarantee, which promises nothing
    /// while appearing to promise something.
    NoGuarantee(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::DuplicateName(n) => {
                write!(f, "capability `{n}` is declared more than once")
            }
            ManifestError::SupportedButEphemeral(n) => write!(
                f,
                "capability `{n}` claims `supported` but its state is ephemeral; \
                 a promise that does not survive a restart is not a production promise"
            ),
            ManifestError::UnboundedClaim(n) => write!(
                f,
                "capability `{n}` is advertised at `preview` or above without \
                 stating a single limit"
            ),
            ManifestError::NoGuarantee(n) => {
                write!(f, "capability `{n}` is advertised without any guarantee")
            }
        }
    }
}

impl std::error::Error for ManifestError {}

/// Why a client requirement could not be met.
///
/// Distinct variants because a client's response differs: an absent capability
/// means the plan is impossible here, while insufficient maturity may be
/// acceptable in a development environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationFailure {
    Absent {
        name: String,
    },
    ContractTooOld {
        name: String,
        offered: ContractVersion,
        required: ContractVersion,
    },
    InsufficientMaturity {
        name: String,
        offered: Maturity,
        required: Maturity,
    },
}

impl fmt::Display for NegotiationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NegotiationFailure::Absent { name } => {
                write!(f, "capability `{name}` is not offered by this build")
            }
            NegotiationFailure::ContractTooOld {
                name,
                offered,
                required,
            } => write!(
                f,
                "capability `{name}` offers contract {offered} but {required} is required"
            ),
            NegotiationFailure::InsufficientMaturity {
                name,
                offered,
                required,
            } => write!(
                f,
                "capability `{name}` is {offered} but {required} is required"
            ),
        }
    }
}

impl std::error::Error for NegotiationFailure {}

/// What a client needs in order to run a given plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirement {
    pub name: String,
    pub minimum_contract: ContractVersion,
    pub minimum_maturity: Maturity,
}

impl Requirement {
    pub fn new(
        name: impl Into<String>,
        minimum_contract: ContractVersion,
        minimum_maturity: Maturity,
    ) -> Self {
        Self {
            name: name.into(),
            minimum_contract,
            minimum_maturity,
        }
    }
}

/// Everything a build promises, plus the identity of the build that promised it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityManifest {
    /// Crate version of the server. Diagnostic only — never negotiate on it.
    pub server_version: String,
    /// Source revision, when the build system supplied one.
    pub build_revision: Option<String>,
    pub capabilities: Vec<Capability>,
}

impl CapabilityManifest {
    /// Build a manifest, refusing internally inconsistent claims.
    pub fn new(
        server_version: impl Into<String>,
        build_revision: Option<String>,
        capabilities: Vec<Capability>,
    ) -> Result<Self, ManifestError> {
        let manifest = Self {
            server_version: server_version.into(),
            build_revision,
            capabilities,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Check that the manifest does not contradict itself.
    ///
    /// This is the whole point of the type. A manifest is only useful if a
    /// reader can trust that its levels mean what they say, and the cheapest
    /// way to keep that true is to make the inconsistent state unrepresentable
    /// at the point of construction rather than reviewable in a document.
    pub fn validate(&self) -> Result<(), ManifestError> {
        let mut seen: Vec<&str> = Vec::with_capacity(self.capabilities.len());
        for cap in &self.capabilities {
            if seen.contains(&cap.name.as_str()) {
                return Err(ManifestError::DuplicateName(cap.name.clone()));
            }
            seen.push(&cap.name);

            if cap.maturity == Maturity::Supported && cap.durability == Durability::Ephemeral {
                return Err(ManifestError::SupportedButEphemeral(cap.name.clone()));
            }
            if cap.guarantees.is_empty() {
                return Err(ManifestError::NoGuarantee(cap.name.clone()));
            }
            if cap.maturity >= Maturity::Preview && cap.limits.is_empty() {
                return Err(ManifestError::UnboundedClaim(cap.name.clone()));
            }
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Capability> {
        self.capabilities.iter().find(|c| c.name == name)
    }

    /// Check every requirement, reporting *all* failures rather than the first.
    ///
    /// A caller deciding whether to fall back wants the complete picture in one
    /// round trip; returning only the first failure turns one negotiation into
    /// a sequence of them.
    pub fn negotiate(&self, requirements: &[Requirement]) -> Result<(), Vec<NegotiationFailure>> {
        let mut failures = Vec::new();
        for req in requirements {
            match self.get(&req.name) {
                None => failures.push(NegotiationFailure::Absent {
                    name: req.name.clone(),
                }),
                Some(cap) => {
                    if !cap.contract_version.accepts(req.minimum_contract) {
                        failures.push(NegotiationFailure::ContractTooOld {
                            name: req.name.clone(),
                            offered: cap.contract_version,
                            required: req.minimum_contract,
                        });
                    }
                    if cap.maturity < req.minimum_maturity {
                        failures.push(NegotiationFailure::InsufficientMaturity {
                            name: req.name.clone(),
                            offered: cap.maturity,
                            required: req.minimum_maturity,
                        });
                    }
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

/// Stable capability names. Clients should reference these constants rather
/// than string literals so a rename is a compile error somewhere.
pub mod names {
    pub const RETRIEVAL_V2: &str = "retrieval.protocol.v2";
    pub const GOVERNED_PUSHDOWN: &str = "retrieval.governed.pushdown";
    pub const EMBEDDING_MAINTENANCE: &str = "embedding.maintenance.incremental";
    pub const HYBRID_PROFILE: &str = "retrieval.hybrid.profile";
    pub const HNSW_SEARCH: &str = "vector.hnsw.search";
    pub const VECTOR_INGEST_BATCH: &str = "vector.ingest.batch";
    pub const METADATA_PREFILTERED_SEARCH: &str = "vector.search.metadata_prefilter";
    pub const BM25: &str = "text.bm25";
    pub const HYBRID_RRF: &str = "retrieval.hybrid.rrf";
    pub const EXACT_RERANK: &str = "retrieval.rerank.exact";
    pub const QUANTIZED_SEGMENT_SEARCH: &str = "vector.search.quantized_segment";
    pub const GRAPH: &str = "graph.service";
    pub const CONTEXT_ASSEMBLY: &str = "context.assembly";
    pub const SEMANTIC_CACHE: &str = "cache.semantic";
}

/// The contract version this build speaks for retrieval.
pub const RETRIEVAL_CONTRACT: ContractVersion = ContractVersion::new(1, 0);

/// The capabilities this build actually offers.
///
/// Every level here is a statement about what is wired and qualified *today*,
/// not about what the code could do. Nothing is `Supported`, because the served
/// vector index holds its state in a process-local map with no persistence
/// path: restart it and the index is gone. Promotion is gated on the durability
/// work, not on review.
pub fn manifest() -> CapabilityManifest {
    let capabilities = vec![
        Capability::new(
            names::HYBRID_PROFILE,
            RETRIEVAL_CONTRACT,
            Maturity::LibraryOnly,
            Durability::Stateless,
        )
        .guarantee("a ranking is reproducible from the profile digest that produced it")
        .guarantee("ties break on document id, so input order cannot change the output")
        .guarantee("every result reports its per-lane rank, raw score and contribution")
        .guarantee("candidates are the union of the lanes, never the intersection")
        .limit(
            "a fused score is a ranking device, not a probability or relevance, \
             and is not comparable across queries or profiles",
        )
        .limit("no reranker is implemented; the profile only records which one was named")
        .limit("not exposed over the wire, so this is a library capability only"),
        Capability::new(
            names::EMBEDDING_MAINTENANCE,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Stateless,
        )
        .guarantee("replaying a maintenance run creates no duplicate chunks or vectors")
        .guarantee("only chunks whose content changed are re-embedded")
        .guarantee("a model, dimension or chunking-policy change forces a full rebuild")
        .guarantee("every record carries source, chunk, policy and model lineage")
        .guarantee("models are identified as provider:name@revision; bare names are refused")
        .limit("planning is in-memory; the caller supplies and persists the record set")
        .limit("chunking itself is not implemented here, only its policy revision")
        .limit(
            "the outbox is an at-least-once queue, so a crash between applying \
             and acknowledging replays the batch",
        ),
        Capability::new(
            names::GOVERNED_PUSHDOWN,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Stateless,
        )
        .guarantee("a filter is split so that pushed AND residual is the original filter")
        .guarantee("a clause that cannot be fully evaluated is never partially pushed")
        .guarantee(
            "an unpushable filter is refused, scanned exactly, or overfetched, never dropped",
        )
        .guarantee("a row missing a governed field never satisfies a predicate on it")
        .guarantee("a short result set reports whether the budget or the index ran out")
        .limit("pushdown covers equality and set membership only; everything else is residual")
        .limit(
            "selectivity is supplied by the caller and is not estimated from \
             index statistics",
        )
        .limit("the caller remains the authority on whether a candidate is authorised"),
        Capability::new(
            names::RETRIEVAL_V2,
            ContractVersion::new(2, 0),
            Maturity::Preview,
            Durability::Ephemeral,
        )
        .guarantee("indexes are addressed by a stable object id, never by name")
        .guarantee("vector identifiers are 128 bits end to end")
        .guarantee("a pinned index generation is honoured exactly or the request is refused")
        .guarantee("every request carries a scoped, expiring capability that is verified")
        .guarantee("a replayed ingest batch is recognised and not applied twice")
        .guarantee("every response carries a digest of the answer it committed to")
        .limit(
            "the record of applied operations is held in memory, so a batch \
             replayed across a restart is applied again",
        )
        .limit(
            "the searchable watermark reports the published generation, not an \
             ingestion timestamp",
        )
        .limit("filter IR expresses only conjunctive exact-match predicates"),
        Capability::new(
            names::HNSW_SEARCH,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Ephemeral,
        )
        .guarantee("approximate k-nearest-neighbour search with a caller-supplied ef")
        .guarantee("results ordered by ascending distance, ties broken by vector id")
        .limit(
            "durability is opt-in: without a configured vector data directory \
             the index is held in process memory and does not survive restart",
        )
        .limit(
            "with a data directory configured, a restart restores the index as \
             of the last published generation, not as of the last insert",
        )
        .limit("recall depends on ef and is not asserted by a conformance gate"),
        Capability::new(
            names::VECTOR_INGEST_BATCH,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Ephemeral,
        )
        .guarantee("batched insert of vectors with optional metadata")
        .limit(
            "the recovery point is the last published generation, so inserts \
             acknowledged after it are lost in an unclean stop; the window is \
             bounded by the configured checkpoint interval, not by zero",
        )
        .limit(
            "without a configured vector data directory no insert is durable \
             at all and every acknowledged write is lost on restart",
        )
        .limit("replaying a batch inserts it again, so ingestion is not idempotent")
        .limit("vector ids are uint64 on the wire"),
        Capability::new(
            names::METADATA_PREFILTERED_SEARCH,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Ephemeral,
        )
        .guarantee("search restricted by exact-match metadata predicates")
        .limit(
            "filter selectivity is not estimated, so a highly selective filter \
                may exhaust the candidate list before k results are found",
        ),
        Capability::new(
            names::BM25,
            RETRIEVAL_CONTRACT,
            Maturity::LibraryOnly,
            Durability::Ephemeral,
        )
        .guarantee("BM25 scoring over an in-process inverted index"),
        Capability::new(
            names::HYBRID_RRF,
            RETRIEVAL_CONTRACT,
            Maturity::LibraryOnly,
            Durability::Ephemeral,
        )
        .guarantee("reciprocal-rank fusion of dense and lexical result lists"),
        Capability::new(
            names::EXACT_RERANK,
            RETRIEVAL_CONTRACT,
            Maturity::LibraryOnly,
            Durability::Stateless,
        )
        .guarantee("exact distance recomputation over a candidate list"),
        Capability::new(
            names::QUANTIZED_SEGMENT_SEARCH,
            RETRIEVAL_CONTRACT,
            Maturity::Experimental,
            Durability::Ephemeral,
        )
        .guarantee("search over quantized segments"),
        Capability::new(
            names::GRAPH,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Ephemeral,
        )
        .guarantee("node and edge storage with adjacency traversal")
        .limit("graph state is held in process memory and does not survive restart"),
        Capability::new(
            names::CONTEXT_ASSEMBLY,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Ephemeral,
        )
        .guarantee("assembly of retrieved fragments into a token-budgeted context")
        .limit(
            "token budgeting uses an approximate tokenizer, so the emitted \
                budget is an estimate rather than a bound",
        ),
        Capability::new(
            names::SEMANTIC_CACHE,
            RETRIEVAL_CONTRACT,
            Maturity::Preview,
            Durability::Ephemeral,
        )
        .guarantee("similarity-keyed reuse of previous results")
        .limit(
            "entries are not partitioned by an authorization scope, so the \
                cache must not be shared across principals with different \
                visibility",
        ),
    ];

    CapabilityManifest {
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        build_revision: option_env!("SOCHDB_BUILD_REVISION").map(str::to_string),
        capabilities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(maturity: Maturity, durability: Durability) -> Capability {
        Capability::new("x", ContractVersion::new(1, 0), maturity, durability)
            .guarantee("g")
            .limit("l")
    }

    #[test]
    fn a_newer_server_serves_an_older_client_but_not_the_reverse() {
        let server = ContractVersion::new(1, 3);
        assert!(server.accepts(ContractVersion::new(1, 0)));
        assert!(server.accepts(ContractVersion::new(1, 3)));
        assert!(
            !server.accepts(ContractVersion::new(1, 4)),
            "a client needing a field this server never learned to read must be refused"
        );
        assert!(
            !server.accepts(ContractVersion::new(2, 0)),
            "a major change is not compatible in either direction"
        );
        assert!(!ContractVersion::new(2, 0).accepts(ContractVersion::new(1, 0)));
    }

    /// The rule that keeps the manifest honest: a promise that evaporates on
    /// restart is not a production promise, whatever else is true of it.
    #[test]
    fn supported_cannot_be_claimed_for_ephemeral_state() {
        let err = CapabilityManifest::new(
            "0.0.0",
            None,
            vec![sample(Maturity::Supported, Durability::Ephemeral)],
        )
        .unwrap_err();
        assert_eq!(err, ManifestError::SupportedButEphemeral("x".into()));

        assert!(
            CapabilityManifest::new(
                "0.0.0",
                None,
                vec![sample(Maturity::Supported, Durability::Durable)],
            )
            .is_ok()
        );
    }

    #[test]
    fn an_advertised_capability_must_promise_and_bound_something() {
        let no_guarantee = Capability::new(
            "x",
            ContractVersion::new(1, 0),
            Maturity::Preview,
            Durability::Durable,
        )
        .limit("l");
        assert_eq!(
            CapabilityManifest::new("0.0.0", None, vec![no_guarantee]).unwrap_err(),
            ManifestError::NoGuarantee("x".into())
        );

        let no_limit = Capability::new(
            "x",
            ContractVersion::new(1, 0),
            Maturity::Preview,
            Durability::Durable,
        )
        .guarantee("g");
        assert_eq!(
            CapabilityManifest::new("0.0.0", None, vec![no_limit]).unwrap_err(),
            ManifestError::UnboundedClaim("x".into())
        );

        // Below preview, an unstated limit is expected rather than suspicious.
        let library_only = Capability::new(
            "x",
            ContractVersion::new(1, 0),
            Maturity::LibraryOnly,
            Durability::Ephemeral,
        )
        .guarantee("g");
        assert!(CapabilityManifest::new("0.0.0", None, vec![library_only]).is_ok());
    }

    #[test]
    fn duplicate_names_are_refused_because_negotiation_would_be_ambiguous() {
        let err = CapabilityManifest::new(
            "0.0.0",
            None,
            vec![
                sample(Maturity::Preview, Durability::Durable),
                sample(Maturity::Supported, Durability::Durable),
            ],
        )
        .unwrap_err();
        assert_eq!(err, ManifestError::DuplicateName("x".into()));
    }

    #[test]
    fn negotiation_reports_every_failure_not_merely_the_first() {
        let manifest = manifest();
        let failures = manifest
            .negotiate(&[
                Requirement::new("does.not.exist", RETRIEVAL_CONTRACT, Maturity::Preview),
                Requirement::new(names::BM25, RETRIEVAL_CONTRACT, Maturity::Supported),
                Requirement::new(
                    names::HNSW_SEARCH,
                    ContractVersion::new(1, 9),
                    Maturity::Preview,
                ),
            ])
            .unwrap_err();

        assert_eq!(
            failures.len(),
            3,
            "a caller deciding whether to fall back needs the whole picture at once"
        );
        assert!(failures.iter().any(|f| matches!(
            f,
            NegotiationFailure::Absent { name } if name == "does.not.exist"
        )));
        assert!(failures.iter().any(|f| matches!(
            f,
            NegotiationFailure::InsufficientMaturity { name, .. } if name == names::BM25
        )));
        assert!(failures.iter().any(|f| matches!(
            f,
            NegotiationFailure::ContractTooOld { name, .. } if name == names::HNSW_SEARCH
        )));
    }

    #[test]
    fn a_met_requirement_negotiates_cleanly() {
        let manifest = manifest();
        assert!(
            manifest
                .negotiate(&[Requirement::new(
                    names::HNSW_SEARCH,
                    RETRIEVAL_CONTRACT,
                    Maturity::Preview
                )])
                .is_ok()
        );
    }

    /// The published manifest is held to the same rules as any other.
    #[test]
    fn the_published_manifest_is_internally_consistent() {
        manifest().validate().expect("published manifest");
    }

    /// This build serves vector search from a process-local map with no
    /// persistence path, so no retrieval capability may claim production
    /// maturity. When the durability work lands, this test is the thing that
    /// has to be updated deliberately — which is the point.
    #[test]
    fn this_build_claims_no_production_capability() {
        let manifest = manifest();
        let claimed: Vec<&str> = manifest
            .capabilities
            .iter()
            .filter(|c| c.maturity.is_production())
            .map(|c| c.name.as_str())
            .collect();
        assert!(
            claimed.is_empty(),
            "no capability may be `supported` while served state is ephemeral, \
             but these claim it: {claimed:?}"
        );
    }

    #[test]
    fn the_manifest_round_trips_as_json_for_non_rust_clients() {
        let manifest = manifest();
        let encoded = serde_json::to_string(&manifest).expect("encode");
        let decoded: CapabilityManifest = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded, manifest);
        decoded
            .validate()
            .expect("a decoded manifest is still checked");
    }
}
