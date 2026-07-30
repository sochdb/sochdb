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

//! Versioned, explainable hybrid retrieval profiles.
//!
//! Hybrid retrieval fuses lists that are not commensurable. A BM25 score and a
//! cosine distance are numbers on unrelated scales, and any single number
//! derived from both is a ranking device rather than a measurement. Three
//! consequences follow, and this module is built around them.
//!
//! **A fused score is not a probability and not a relevance.** It has no
//! meaning outside the one ranking it was produced in, and it cannot be
//! compared across queries, across profiles, or against a threshold. The type
//! is called [`FusionScore`] rather than `score`, it deliberately offers no
//! conversion to a percentage or confidence, and [`FusedResult`] carries the
//! per-lane ranks that *do* mean something so there is a correct thing to use
//! instead.
//!
//! **A ranking is only reproducible if everything that shaped it is recorded.**
//! [`RetrievalProfile::digest`] covers every parameter that can change an
//! output ordering -- lane weights, the fusion algorithm and its constants,
//! the normalisation procedure, the budgets, the reranker. Two runs whose
//! profile digests match should produce the same ordering from the same
//! inputs; two runs whose digests differ are not comparable, and the digest is
//! what lets anyone notice.
//!
//! **Ties must break deterministically.** Fusion produces exact ties routinely:
//! RRF over two lanes yields the same value for many documents, and any
//! ordering that depends on hash iteration or on the order candidates happened
//! to arrive gives different answers to identical queries. Ties here break on
//! document id, which is arbitrary but stable, and a test shuffles the input to
//! prove the output does not move.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How raw lane scores are made comparable before weighted fusion.
///
/// Recorded in the profile because the choice changes the ranking, and a
/// ranking whose normalisation is not written down cannot be reproduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Normalization {
    /// Map the observed range onto [0, 1] within this result set.
    ///
    /// Note the consequence: the transform depends on the other candidates, so
    /// the same document can normalise differently in two queries. That is
    /// acceptable for ranking within one query and wrong for anything else.
    MinMax,
    /// Divide by the sum of scores in the lane.
    SumToOne,
    /// Leave scores untouched. Only sensible when lanes are already on the
    /// same scale, which for lexical against vector they are not.
    None,
}

/// How lanes are combined.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Fusion {
    /// Reciprocal rank fusion: `sum_i weight_i / (k0 + rank_i(d))`.
    ///
    /// Uses only ranks, so it is unaffected by the lanes' incompatible score
    /// scales. This is why it is the default and the first thing to ship.
    ReciprocalRank { k0: f32 },
    /// `alpha * lexical_norm + beta * vector_norm`, which requires the scores
    /// to have been normalised first and is therefore sensitive to how.
    WeightedScore { alpha: f32, beta: f32 },
}

/// What the caller is promised about completeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuaranteeLevel {
    /// Every lane examined every document.
    Exact,
    /// At least one lane used an approximate index. Results may omit
    /// documents that would have ranked.
    Approximate,
}

/// Which lane a candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Lane {
    Lexical,
    Vector,
}

impl Lane {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Vector => "vector",
        }
    }
}

/// Everything that shapes a hybrid ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalProfile {
    pub name: String,
    /// Bumped by whoever changes the profile. The digest catches changes that
    /// were made without bumping it, which is the common case.
    pub revision: u32,
    pub lexical_budget: usize,
    pub vector_budget: usize,
    pub lexical_weight: f32,
    pub vector_weight: f32,
    pub fusion: Fusion,
    pub normalization: Normalization,
    pub reranker: Option<String>,
    pub guarantee: GuaranteeLevel,
}

impl RetrievalProfile {
    /// A profile that ranks by RRF over both lanes, which is the safe default:
    /// it needs no normalisation and no assumption that the lanes' scores mean
    /// anything relative to each other.
    pub fn rrf(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            revision: 1,
            lexical_budget: 100,
            vector_budget: 100,
            lexical_weight: 1.0,
            vector_weight: 1.0,
            fusion: Fusion::ReciprocalRank { k0: 60.0 },
            normalization: Normalization::None,
            reranker: None,
            guarantee: GuaranteeLevel::Approximate,
        }
    }

    /// Digest over every parameter that can change an ordering.
    ///
    /// Floats are hashed by bit pattern rather than by a formatted decimal,
    /// so two weights that print the same but rank differently do not produce
    /// the same digest. The profile name is included but so is everything
    /// else: a profile edited without bumping its revision still changes
    /// digest, which is the case a revision number alone always misses.
    pub fn digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        let mut field = |bytes: &[u8]| {
            hasher.update(&(bytes.len() as u64).to_be_bytes());
            hasher.update(bytes);
        };

        field(self.name.as_bytes());
        field(&self.revision.to_be_bytes());
        field(&(self.lexical_budget as u64).to_be_bytes());
        field(&(self.vector_budget as u64).to_be_bytes());
        field(&self.lexical_weight.to_bits().to_be_bytes());
        field(&self.vector_weight.to_bits().to_be_bytes());
        match self.fusion {
            Fusion::ReciprocalRank { k0 } => {
                field(b"rrf");
                field(&k0.to_bits().to_be_bytes());
            }
            Fusion::WeightedScore { alpha, beta } => {
                field(b"weighted");
                field(&alpha.to_bits().to_be_bytes());
                field(&beta.to_bits().to_be_bytes());
            }
        }
        field(match self.normalization {
            Normalization::MinMax => b"minmax".as_slice(),
            Normalization::SumToOne => b"sum1".as_slice(),
            Normalization::None => b"none".as_slice(),
        });
        field(self.reranker.as_deref().unwrap_or("").as_bytes());
        field(match self.guarantee {
            GuaranteeLevel::Exact => b"exact".as_slice(),
            GuaranteeLevel::Approximate => b"approx".as_slice(),
        });

        hasher.finalize().to_hex().to_string()
    }
}

/// One lane's ranked output.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneResults {
    pub lane: Lane,
    /// Documents in the lane's own order, best first, with the lane's own
    /// raw score.
    pub ranked: Vec<(u128, f32)>,
}

/// A fused score.
///
/// Not a probability, not a relevance, not comparable across queries or
/// profiles. There is deliberately no method that converts it to a percentage
/// or a confidence, because every such method would be used to build a
/// threshold that means nothing.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FusionScore(f32);

impl FusionScore {
    /// The raw value, for ordering and for display alongside the ranks that
    /// give it context.
    pub fn value(&self) -> f32 {
        self.0
    }
}

/// What each lane contributed to one document's fused score.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneContribution {
    pub lane: Lane,
    /// 1-based position in that lane. Absent when the lane did not return the
    /// document at all, which is different from returning it last.
    pub rank: Option<usize>,
    pub raw_score: Option<f32>,
    pub normalized_score: Option<f32>,
    pub contribution: f32,
}

/// One fused result, with enough detail to explain why it is where it is.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedResult {
    pub id: u128,
    pub final_rank: usize,
    pub score: FusionScore,
    pub contributions: Vec<LaneContribution>,
}

impl FusedResult {
    pub fn rank_in(&self, lane: Lane) -> Option<usize> {
        self.contributions
            .iter()
            .find(|c| c.lane == lane)
            .and_then(|c| c.rank)
    }
}

/// The outcome of a fusion, tied to the profile that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionOutcome {
    pub results: Vec<FusedResult>,
    pub profile_digest: String,
    pub profile_revision: u32,
    pub guarantee: GuaranteeLevel,
    pub lane_candidate_counts: BTreeMap<&'static str, usize>,
}

/// Normalise a lane's scores.
///
/// A lane whose scores are all equal normalises to 1.0 rather than to 0 or to
/// NaN. Min-max over a zero range would divide by zero; mapping to 0 would
/// silently delete a lane that agreed about everything, which is a legitimate
/// state and not an absence of signal.
fn normalize(scores: &[f32], how: Normalization) -> Vec<f32> {
    match how {
        Normalization::None => scores.to_vec(),
        Normalization::MinMax => {
            let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if !range.is_finite() || range <= 0.0 {
                return vec![1.0; scores.len()];
            }
            scores.iter().map(|s| (s - min) / range).collect()
        }
        Normalization::SumToOne => {
            let sum: f32 = scores.iter().copied().map(f32::abs).sum();
            if sum <= 0.0 || !sum.is_finite() {
                return vec![1.0 / scores.len().max(1) as f32; scores.len()];
            }
            scores.iter().map(|s| s / sum).collect()
        }
    }
}

/// Fuse lane outputs into one ranking.
///
/// Candidates are the union of the lanes, not the intersection. Intersecting
/// would discard every document only one lane found, which is exactly the set
/// hybrid retrieval exists to recover.
pub fn fuse(profile: &RetrievalProfile, lanes: &[LaneResults], top_k: usize) -> FusionOutcome {
    let mut lane_candidate_counts = BTreeMap::new();
    let mut per_doc: BTreeMap<u128, Vec<LaneContribution>> = BTreeMap::new();

    for lane in lanes {
        let budget = match lane.lane {
            Lane::Lexical => profile.lexical_budget,
            Lane::Vector => profile.vector_budget,
        };
        let truncated: Vec<(u128, f32)> = lane.ranked.iter().copied().take(budget).collect();
        lane_candidate_counts.insert(lane.lane.name(), truncated.len());

        let raw: Vec<f32> = truncated.iter().map(|(_, s)| *s).collect();
        let normalized = normalize(&raw, profile.normalization);
        let weight = match lane.lane {
            Lane::Lexical => profile.lexical_weight,
            Lane::Vector => profile.vector_weight,
        };

        for (position, (id, raw_score)) in truncated.iter().enumerate() {
            let rank = position + 1;
            let normalized_score = normalized[position];
            let contribution = match profile.fusion {
                // k0 keeps the top rank from dominating: without it the first
                // position would be worth infinitely more than the second.
                Fusion::ReciprocalRank { k0 } => weight / (k0 + rank as f32),
                Fusion::WeightedScore { alpha, beta } => {
                    let lane_weight = match lane.lane {
                        Lane::Lexical => alpha,
                        Lane::Vector => beta,
                    };
                    lane_weight * normalized_score
                }
            };

            per_doc.entry(*id).or_default().push(LaneContribution {
                lane: lane.lane,
                rank: Some(rank),
                raw_score: Some(*raw_score),
                normalized_score: Some(normalized_score),
                contribution,
            });
        }
    }

    let mut scored: Vec<(u128, f32, Vec<LaneContribution>)> = per_doc
        .into_iter()
        .map(|(id, mut contributions)| {
            contributions.sort_by_key(|c| c.lane);
            let total = contributions.iter().map(|c| c.contribution).sum();
            (id, total, contributions)
        })
        .collect();

    // Descending by score, then ascending by id. The id tiebreak is arbitrary
    // but total and stable: fusion produces exact ties constantly, and without
    // a deterministic tiebreak the same query returns a different order
    // depending on iteration order.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let results = scored
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(i, (id, score, contributions))| FusedResult {
            id,
            final_rank: i + 1,
            score: FusionScore(score),
            contributions,
        })
        .collect();

    FusionOutcome {
        results,
        profile_digest: profile.digest(),
        profile_revision: profile.revision,
        guarantee: profile.guarantee,
        lane_candidate_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexical(ranked: &[(u128, f32)]) -> LaneResults {
        LaneResults {
            lane: Lane::Lexical,
            ranked: ranked.to_vec(),
        }
    }

    fn vector(ranked: &[(u128, f32)]) -> LaneResults {
        LaneResults {
            lane: Lane::Vector,
            ranked: ranked.to_vec(),
        }
    }

    /// Fusion produces exact ties constantly. Without a total, deterministic
    /// tiebreak the same query returns a different order depending on the
    /// order candidates happened to arrive, which makes results irreproducible
    /// for reasons nobody can see.
    #[test]
    fn ties_break_deterministically_regardless_of_input_order() {
        let profile = RetrievalProfile::rrf("p");
        // Every document is at rank 1 in one lane and rank 2 in the other, so
        // all four fuse to exactly the same score.
        let forward = fuse(
            &profile,
            &[
                lexical(&[(10, 1.0), (20, 0.9)]),
                vector(&[(20, 0.1), (10, 0.2)]),
            ],
            10,
        );
        let reversed = fuse(
            &profile,
            &[
                vector(&[(20, 0.1), (10, 0.2)]),
                lexical(&[(10, 1.0), (20, 0.9)]),
            ],
            10,
        );

        let a: Vec<u128> = forward.results.iter().map(|r| r.id).collect();
        let b: Vec<u128> = reversed.results.iter().map(|r| r.id).collect();
        assert_eq!(a, b, "lane order changed the result order");
        assert_eq!(a, vec![10, 20], "ties did not break on id");

        // The scores really are tied, so this is a genuine tiebreak test.
        assert_eq!(
            forward.results[0].score.value(),
            forward.results[1].score.value()
        );
    }

    /// The union, not the intersection. A document only one lane found is
    /// exactly what hybrid retrieval exists to recover.
    #[test]
    fn a_document_found_by_one_lane_still_appears() {
        let profile = RetrievalProfile::rrf("p");
        let outcome = fuse(&profile, &[lexical(&[(1, 5.0)]), vector(&[(2, 0.1)])], 10);
        let ids: Vec<u128> = outcome.results.iter().map(|r| r.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    /// A document both lanes agree on must outrank one only a single lane
    /// found. This is the entire premise of fusion.
    #[test]
    fn agreement_between_lanes_outranks_a_single_lane_hit() {
        let profile = RetrievalProfile::rrf("p");
        let outcome = fuse(
            &profile,
            &[
                lexical(&[(7, 5.0), (8, 4.0)]),
                vector(&[(7, 0.1), (9, 0.2)]),
            ],
            10,
        );
        assert_eq!(outcome.results[0].id, 7);
        assert_eq!(outcome.results[0].contributions.len(), 2);
    }

    /// Every result must be explainable: which lane found it, where, and what
    /// that was worth. Without this a fused score is an unauditable number.
    #[test]
    fn every_result_reports_its_per_lane_ranks() {
        let profile = RetrievalProfile::rrf("p");
        let outcome = fuse(
            &profile,
            &[
                lexical(&[(1, 9.0), (2, 8.0)]),
                vector(&[(2, 0.1), (1, 0.5)]),
            ],
            10,
        );

        let doc1 = outcome.results.iter().find(|r| r.id == 1).unwrap();
        assert_eq!(doc1.rank_in(Lane::Lexical), Some(1));
        assert_eq!(doc1.rank_in(Lane::Vector), Some(2));
        assert_eq!(doc1.final_rank, 1);

        for c in &doc1.contributions {
            assert!(c.raw_score.is_some());
            assert!(c.contribution > 0.0);
        }
    }

    /// Absent from a lane is different from ranked last in it. Reporting a
    /// missing document as "rank = length + 1" would invent evidence the lane
    /// never produced.
    #[test]
    fn a_lane_that_did_not_return_a_document_reports_no_rank() {
        let profile = RetrievalProfile::rrf("p");
        let outcome = fuse(&profile, &[lexical(&[(1, 9.0)]), vector(&[(2, 0.1)])], 10);
        let doc1 = outcome.results.iter().find(|r| r.id == 1).unwrap();
        assert_eq!(doc1.rank_in(Lane::Lexical), Some(1));
        assert_eq!(
            doc1.rank_in(Lane::Vector),
            None,
            "a lane that never saw the document reported a rank for it"
        );
        assert_eq!(doc1.contributions.len(), 1);
    }

    /// RRF depends only on ranks, which is why it survives lanes whose scores
    /// are on wildly different scales. Multiplying one lane's scores by a
    /// million must not move anything.
    #[test]
    fn rrf_ignores_the_scale_of_lane_scores() {
        let profile = RetrievalProfile::rrf("p");
        let small = fuse(
            &profile,
            &[lexical(&[(1, 0.001), (2, 0.0005)]), vector(&[(2, 0.1)])],
            10,
        );
        let huge = fuse(
            &profile,
            &[lexical(&[(1, 1e9), (2, 5e8)]), vector(&[(2, 0.1)])],
            10,
        );
        let a: Vec<u128> = small.results.iter().map(|r| r.id).collect();
        let b: Vec<u128> = huge.results.iter().map(|r| r.id).collect();
        assert_eq!(a, b);
    }

    /// ...whereas weighted fusion does not, which is the reason the
    /// normalisation procedure has to be recorded in the profile.
    #[test]
    fn weighted_fusion_depends_on_the_normalization_and_so_records_it() {
        let mut minmax = RetrievalProfile::rrf("p");
        minmax.fusion = Fusion::WeightedScore {
            alpha: 1.0,
            beta: 1.0,
        };
        minmax.normalization = Normalization::MinMax;

        let mut none = minmax.clone();
        none.normalization = Normalization::None;

        assert_ne!(
            minmax.digest(),
            none.digest(),
            "the normalization choice did not reach the digest"
        );

        // Unnormalised, the lexical lane's larger numbers dominate entirely
        // and document 1 wins. Normalised, document 2's strong showing in both
        // lanes wins. Same inputs, same weights, different ranking -- which is
        // why the procedure has to be part of the profile identity.
        let lanes = [
            lexical(&[(1, 100.0), (2, 99.0), (3, 0.0)]),
            vector(&[(3, 1.0), (2, 0.5), (1, 0.0)]),
        ];
        let with_minmax: Vec<u128> = fuse(&minmax, &lanes, 10)
            .results
            .iter()
            .map(|r| r.id)
            .collect();
        let without: Vec<u128> = fuse(&none, &lanes, 10)
            .results
            .iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(without, vec![1, 2, 3]);
        assert_eq!(with_minmax, vec![2, 1, 3]);
    }

    /// The digest is what makes a ranking reproducible. Every parameter that
    /// can change an ordering must reach it -- a revision number alone misses
    /// the common case of a profile edited in place.
    #[test]
    fn every_ranking_parameter_reaches_the_digest() {
        let base = RetrievalProfile::rrf("p");
        let d = base.digest();

        let mut changed = base.clone();
        changed.lexical_weight = 2.0;
        assert_ne!(d, changed.digest(), "lexical weight");

        let mut changed = base.clone();
        changed.vector_weight = 2.0;
        assert_ne!(d, changed.digest(), "vector weight");

        let mut changed = base.clone();
        changed.fusion = Fusion::ReciprocalRank { k0: 10.0 };
        assert_ne!(d, changed.digest(), "rrf k0");

        let mut changed = base.clone();
        changed.fusion = Fusion::WeightedScore {
            alpha: 1.0,
            beta: 1.0,
        };
        assert_ne!(d, changed.digest(), "fusion algorithm");

        let mut changed = base.clone();
        changed.normalization = Normalization::SumToOne;
        assert_ne!(d, changed.digest(), "normalization");

        let mut changed = base.clone();
        changed.lexical_budget = 50;
        assert_ne!(d, changed.digest(), "lexical budget");

        let mut changed = base.clone();
        changed.vector_budget = 50;
        assert_ne!(d, changed.digest(), "vector budget");

        let mut changed = base.clone();
        changed.reranker = Some("cross-encoder@1".to_string());
        assert_ne!(d, changed.digest(), "reranker");

        let mut changed = base.clone();
        changed.guarantee = GuaranteeLevel::Exact;
        assert_ne!(d, changed.digest(), "guarantee level");

        let mut changed = base.clone();
        changed.revision = 2;
        assert_ne!(d, changed.digest(), "revision");

        // An identical profile digests identically, or nothing above means
        // anything.
        assert_eq!(d, base.clone().digest());
    }

    /// Two weights that print the same but rank differently must not produce
    /// the same digest, or the digest stops detecting the change it exists for.
    #[test]
    fn weights_reach_the_digest_by_bits_not_by_rendering() {
        let mut a = RetrievalProfile::rrf("p");
        a.lexical_weight = 1.0;
        let mut b = a.clone();
        b.lexical_weight = 1.0 + f32::EPSILON;
        assert_eq!(
            format!("{:.6}", a.lexical_weight),
            format!("{:.6}", b.lexical_weight)
        );
        assert_ne!(a.digest(), b.digest());
    }

    /// The budget is a promise about work done. A lane must not smuggle more
    /// candidates into the fusion than the profile allows.
    #[test]
    fn a_lane_cannot_exceed_its_budget() {
        let mut profile = RetrievalProfile::rrf("p");
        profile.lexical_budget = 2;
        let outcome = fuse(
            &profile,
            &[lexical(&[(1, 9.0), (2, 8.0), (3, 7.0), (4, 6.0)])],
            10,
        );
        assert_eq!(outcome.lane_candidate_counts["lexical"], 2);
        assert_eq!(outcome.results.len(), 2);
        let ids: Vec<u128> = outcome.results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![1, 2], "the budget kept the wrong candidates");
    }

    /// A lane whose scores are all equal is a lane that agrees about
    /// everything -- a real state. Min-max over a zero range would divide by
    /// zero; mapping it to 0 would silently delete the lane.
    #[test]
    fn a_lane_with_no_score_spread_does_not_vanish() {
        assert_eq!(
            normalize(&[5.0, 5.0, 5.0], Normalization::MinMax),
            vec![1.0; 3]
        );
        assert!(
            normalize(&[0.0, 0.0], Normalization::SumToOne)
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        );
        assert!(
            normalize(&[], Normalization::MinMax).is_empty(),
            "an empty lane should normalise to nothing rather than panic"
        );
    }

    #[test]
    fn min_max_maps_the_observed_range_onto_the_unit_interval() {
        assert_eq!(
            normalize(&[0.0, 5.0, 10.0], Normalization::MinMax),
            vec![0.0, 0.5, 1.0]
        );
    }

    /// The outcome must carry the profile that produced it, or a stored result
    /// cannot be explained later.
    #[test]
    fn the_outcome_names_the_profile_that_produced_it() {
        let profile = RetrievalProfile::rrf("hybrid-default");
        let outcome = fuse(&profile, &[lexical(&[(1, 1.0)])], 10);
        assert_eq!(outcome.profile_digest, profile.digest());
        assert_eq!(outcome.profile_revision, profile.revision);
        assert_eq!(outcome.guarantee, GuaranteeLevel::Approximate);
    }

    #[test]
    fn top_k_is_applied_after_fusion_not_before() {
        let profile = RetrievalProfile::rrf("p");
        // Document 3 is poor in the lexical lane and top in the vector lane;
        // truncating before fusion would drop it.
        let outcome = fuse(
            &profile,
            &[
                lexical(&[(1, 9.0), (2, 8.0), (3, 0.1)]),
                vector(&[(3, 0.01)]),
            ],
            2,
        );
        assert_eq!(outcome.results.len(), 2);
        assert_eq!(outcome.results[0].final_rank, 1);
        assert_eq!(outcome.results[1].final_rank, 2);
        assert!(
            outcome.results.iter().any(|r| r.id == 3),
            "a document lifted by the second lane was truncated away first"
        );
    }

    #[test]
    fn an_empty_query_produces_an_empty_ranking_rather_than_an_error() {
        let profile = RetrievalProfile::rrf("p");
        let outcome = fuse(&profile, &[], 10);
        assert!(outcome.results.is_empty());
        assert!(!outcome.profile_digest.is_empty());
    }
}
