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

//! Bounded global top-k over shards.
//!
//! [`shard_topology`](crate::shard_topology) decides *which* shards to ask.
//! This decides what to do with what comes back, which is where distributed
//! retrieval usually stops matching the single-node answer it claims to
//! approximate.
//!
//! # Why asking every shard for k is not enough
//!
//! The obvious plan -- ask `S` shards for their local top-`k`, merge, keep `k`
//! -- is correct only when the true global top-`k` is spread evenly. It is not.
//! If one shard holds `k` of the global best, every other shard's `k` results
//! were wasted and nothing was lost. But if a shard holds `k + 1` of them, its
//! `k + 1`-th is dropped locally and can never be recovered by the merge. The
//! coordinator cannot detect this from the results alone: the merged list looks
//! complete, is the right length, and is wrong.
//!
//! Three things fix it, and all three are needed:
//!
//! - **Oversampling.** Ask each shard for `k' = ceil(k × factor)` so a skewed
//!   shard has room to return more than its even share.
//! - **A bound per shard.** A shard reports the best score it did *not* return.
//!   If every shard's unreturned best is worse than the current global `k`-th,
//!   the answer is provably complete and no further work is needed. This is the
//!   only signal that distinguishes "we have the right answer" from "we have an
//!   answer".
//! - **Expansion.** When the bound says the answer might be incomplete, ask the
//!   shards that could still contribute for more. Bounded, and only those
//!   shards.
//!
//! # Memory
//!
//! The merge holds `O(k + S)`: a heap of at most `k` results plus one cursor per
//! shard. It does not hold `S × k'` at once. Merging by concatenating and
//! sorting is `O(S × k')` in memory and is the shape that fails first when the
//! fan-out grows, so the heap is not an optimisation here.
//!
//! # What this module does not do
//!
//! It performs no I/O. Shards are supplied as [`ShardResponse`] values by the
//! caller, so replica selection, timeouts and transport belong to the layer
//! above. That keeps every rule here testable without a cluster, including the
//! failure rules, which are the ones least likely to be exercised otherwise.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use crate::shard_topology::ShardId;

/// Default oversampling factor.
///
/// Two is a deliberate compromise. Skew tolerance rises with the factor, but so
/// does the work every shard does whether or not it is the skewed one. The
/// bound is what actually protects recall; oversampling only reduces how often
/// an expansion round is needed.
pub const DEFAULT_OVERSAMPLING: f32 = 2.0;

/// The most expansion rounds a query will run before returning what it has.
///
/// A bound is required. Without one a pathological distribution -- or a shard
/// that keeps reporting an optimistic bound -- turns one query into an
/// unbounded conversation.
pub const MAX_EXPANSION_ROUNDS: usize = 3;

/// The identity of the data a query runs against.
///
/// Carried explicitly, and compared on every response, because a shard that has
/// moved to a newer generation is not a slow shard, it is a shard answering a
/// different question. Merging its results produces an answer that corresponds
/// to no single state of the data, and nothing downstream can detect that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuerySnapshot {
    /// The published index generation.
    pub generation: u64,
    /// The source data snapshot the generation was built from.
    pub source_version: u64,
}

impl QuerySnapshot {
    /// Name a snapshot.
    pub fn new(generation: u64, source_version: u64) -> Self {
        Self {
            generation,
            source_version,
        }
    }
}

/// One scored candidate from a shard.
///
/// `score` follows the convention of [`crate::shard_topology`]: lower is
/// better, as for a squared L2 distance. A similarity would have to be negated
/// before it arrives here. That is stated once, in one place, because a mixed
/// convention across shards is a bug the merge cannot see -- every result would
/// still be a float and the ordering would simply be wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Stable vector row identity. Not a position, not a table name.
    pub id: u128,
    /// Distance, lower is better.
    pub score: f32,
    /// The shard that returned it.
    pub shard: ShardId,
}

impl Candidate {
    /// Name a candidate.
    pub fn new(id: u128, score: f32, shard: ShardId) -> Self {
        Self { id, score, shard }
    }

    /// Total order over candidates: score ascending, then id, then shard.
    ///
    /// Total, not partial. Ties are common -- duplicated rows across replicas,
    /// quantised scores, exact-duplicate vectors -- and a comparison that
    /// returns different orders for equal scores makes an identical query
    /// return different answers on different runs, for reasons nobody can see
    /// from the outside. Ties break on id, which is arbitrary but stable.
    ///
    /// NaN sorts last. A shard that returns NaN has failed at something, and
    /// the safe place for the result is the end of the list, not wherever an
    /// inconsistent comparator happens to leave it.
    fn cmp_total(&self, other: &Self) -> Ordering {
        match (self.score.is_nan(), other.score.is_nan()) {
            (true, true) => (self.id, self.shard).cmp(&(other.id, other.shard)),
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self
                .score
                .partial_cmp(&other.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| (self.id, self.shard).cmp(&(other.id, other.shard))),
        }
    }
}

/// A candidate ordered so the *worst* is greatest, for a bounded max-heap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Worst(Candidate);

impl Eq for Worst {}

impl Ord for Worst {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_total(&other.0)
    }
}

impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// What one shard returned.
#[derive(Debug, Clone)]
pub struct ShardResponse {
    /// Which shard.
    pub shard: ShardId,
    /// The snapshot it answered against.
    pub snapshot: QuerySnapshot,
    /// Its local results, best first.
    pub candidates: Vec<Candidate>,
    /// The best score this shard holds that it did *not* return, if any.
    ///
    /// `None` means the shard returned everything it had that could match, so
    /// it can contribute nothing further. That is a stronger statement than an
    /// empty candidate list and the two must not be conflated: a shard with
    /// nothing left is finished, a shard that merely returned no results this
    /// round may still be holding better ones behind a filter.
    pub next_best_score: Option<f32>,
    /// Whether the shard exhausted its own search budget.
    ///
    /// A shard that stopped early cannot honestly claim its `next_best_score`
    /// bounds everything it holds, so this suppresses the completeness proof.
    pub budget_exhausted: bool,
}

impl ShardResponse {
    /// A complete response: the shard searched to exhaustion.
    pub fn complete(shard: ShardId, snapshot: QuerySnapshot, candidates: Vec<Candidate>) -> Self {
        Self {
            shard,
            snapshot,
            candidates,
            next_best_score: None,
            budget_exhausted: false,
        }
    }

    /// A truncated response: the shard held back results at `next_best`.
    pub fn truncated(
        shard: ShardId,
        snapshot: QuerySnapshot,
        candidates: Vec<Candidate>,
        next_best: f32,
    ) -> Self {
        Self {
            shard,
            snapshot,
            candidates,
            next_best_score: Some(next_best),
            budget_exhausted: false,
        }
    }
}

/// Why a shard contributed nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardFailure {
    /// No replica of the shard answered.
    Unreachable,
    /// The shard answered against a different snapshot.
    StaleSnapshot {
        expected: QuerySnapshot,
        observed: QuerySnapshot,
    },
    /// The shard returned a candidate it does not own.
    ///
    /// Not pedantry. A merge that accepts misattributed candidates cannot
    /// deduplicate correctly, and a shard confused about its own identity is
    /// the shape of a routing or configuration fault that should surface loudly
    /// rather than degrade recall quietly.
    Misattributed { claimed: ShardId },
}

impl std::fmt::Display for ShardFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardFailure::Unreachable => write!(f, "no replica answered"),
            ShardFailure::StaleSnapshot { expected, observed } => write!(
                f,
                "answered against generation {} source {} but the query runs at generation {} source {}",
                observed.generation,
                observed.source_version,
                expected.generation,
                expected.source_version
            ),
            ShardFailure::Misattributed { claimed } => {
                write!(f, "returned a candidate belonging to shard {claimed}")
            }
        }
    }
}

/// How complete the returned answer is.
///
/// Three states, not a boolean, because "we proved this is right", "we ran out
/// of rounds" and "a shard is missing" have different operational meanings and
/// only one of them is a healthy query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Completeness {
    /// Every shard's unreturned best is worse than the k-th result, so this is
    /// the true global top-k over the shards that answered.
    Proven,
    /// The bound could not be established within the round budget. The answer
    /// is plausible and unproven.
    RoundsExhausted { rounds: usize },
    /// One or more shards contributed nothing, so results they hold are absent.
    ShardsMissing { count: usize },
}

impl Completeness {
    /// Whether the result is a proven global top-k.
    pub fn is_proven(&self) -> bool {
        matches!(self, Completeness::Proven)
    }
}

/// Which shards could still improve the answer, and what to ask them for.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpansionRequest {
    /// Shards whose unreturned best beats the current k-th result.
    pub shards: Vec<ShardId>,
    /// How many candidates to ask each for.
    pub want: usize,
    /// The score to beat. A shard need not return anything worse.
    pub threshold: f32,
}

/// The outcome of a merge.
#[derive(Debug, Clone)]
pub struct MergeOutcome {
    /// The global top-k, best first.
    pub results: Vec<Candidate>,
    /// Whether that is provably the top-k.
    pub completeness: Completeness,
    /// Shards that contributed nothing, and why.
    pub failures: BTreeMap<ShardId, ShardFailure>,
    /// What to ask for next, if anything.
    pub expansion: Option<ExpansionRequest>,
    /// Counters for the query.
    pub stats: MergeStats,
}

impl MergeOutcome {
    /// Whether the answer can be returned as a true global top-k.
    pub fn is_exact(&self) -> bool {
        self.completeness.is_proven() && self.failures.is_empty()
    }
}

/// Observability for one merge.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeStats {
    /// Shards whose responses were accepted.
    pub shards_merged: usize,
    /// Candidates offered by those shards.
    pub candidates_seen: usize,
    /// Candidates dropped as duplicates of an id already held.
    pub duplicates_dropped: usize,
    /// Candidates never entering the heap because they were worse than the
    /// k-th result already held.
    pub candidates_pruned: usize,
    /// Shards that answered against the wrong snapshot.
    pub stale_shards: usize,
}

/// How to run a merge.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// How many results the caller wants.
    pub k: usize,
    /// How many each shard is asked for, as a multiple of `k`.
    pub oversampling: f32,
    /// The snapshot every shard must agree on.
    pub snapshot: QuerySnapshot,
    /// The most expansion rounds permitted.
    pub max_rounds: usize,
}

impl MergeConfig {
    /// A configuration with the default oversampling and round budget.
    pub fn new(k: usize, snapshot: QuerySnapshot) -> Self {
        Self {
            k,
            oversampling: DEFAULT_OVERSAMPLING,
            snapshot,
            max_rounds: MAX_EXPANSION_ROUNDS,
        }
    }

    /// How many candidates to request from each shard.
    ///
    /// Never fewer than `k`. An oversampling factor below one would ask each
    /// shard for less than the answer size, which cannot be right for any
    /// distribution, so it is clamped rather than trusted.
    pub fn per_shard_k(&self) -> usize {
        if self.k == 0 {
            return 0;
        }
        let factor = if self.oversampling.is_nan() || self.oversampling < 1.0 {
            1.0
        } else {
            self.oversampling
        };
        let scaled = (self.k as f32 * factor).ceil();
        if scaled >= usize::MAX as f32 {
            usize::MAX
        } else {
            (scaled as usize).max(self.k)
        }
    }
}

/// A bounded merge that holds at most `k` results and one cursor per shard.
#[derive(Debug)]
pub struct GlobalMerge {
    config: MergeConfig,
    /// Max-heap of the best `k` seen, worst at the top so it can be evicted.
    heap: BinaryHeap<Worst>,
    /// Ids already admitted, so a row present on two replicas is counted once.
    seen: BTreeSet<u128>,
    /// Each shard's unreturned best, for the completeness bound.
    bounds: BTreeMap<ShardId, f32>,
    /// Shards that cannot contribute further.
    finished: BTreeSet<ShardId>,
    /// Shards that could not be proven exhausted.
    unbounded: BTreeSet<ShardId>,
    failures: BTreeMap<ShardId, ShardFailure>,
    stats: MergeStats,
    rounds: usize,
}

impl GlobalMerge {
    /// Start a merge.
    pub fn new(config: MergeConfig) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(config.k.min(1024)),
            config,
            seen: BTreeSet::new(),
            bounds: BTreeMap::new(),
            finished: BTreeSet::new(),
            unbounded: BTreeSet::new(),
            failures: BTreeMap::new(),
            stats: MergeStats::default(),
            rounds: 0,
        }
    }

    /// Record a shard that no replica answered for.
    pub fn record_unreachable(&mut self, shard: ShardId) {
        self.failures.insert(shard, ShardFailure::Unreachable);
    }

    /// Offer one shard's response to the merge.
    ///
    /// Rejects the whole response rather than part of it when the shard is
    /// stale or misattributed. Taking the acceptable half of a response from a
    /// shard that is demonstrably confused would mix snapshots inside one
    /// answer, which is the failure the snapshot check exists to prevent.
    pub fn accept(&mut self, response: ShardResponse) -> Result<(), ShardFailure> {
        if response.snapshot != self.config.snapshot {
            let failure = ShardFailure::StaleSnapshot {
                expected: self.config.snapshot,
                observed: response.snapshot,
            };
            self.failures.insert(response.shard, failure.clone());
            self.stats.stale_shards += 1;
            return Err(failure);
        }
        if let Some(bad) = response
            .candidates
            .iter()
            .find(|c| c.shard != response.shard)
        {
            let failure = ShardFailure::Misattributed { claimed: bad.shard };
            self.failures.insert(response.shard, failure.clone());
            return Err(failure);
        }

        // A shard that answers is no longer a failure. A replica retry that
        // succeeds must clear the earlier unreachable mark, or a query that
        // recovered would still report shards missing and refuse to call itself
        // exact.
        self.failures.remove(&response.shard);
        self.stats.shards_merged += 1;

        for candidate in response.candidates {
            self.stats.candidates_seen += 1;
            self.offer(candidate);
        }

        match (response.next_best_score, response.budget_exhausted) {
            (_, true) => {
                self.unbounded.insert(response.shard);
                self.bounds.remove(&response.shard);
                self.finished.remove(&response.shard);
            }
            (Some(bound), false) => {
                self.unbounded.remove(&response.shard);
                self.finished.remove(&response.shard);
                self.bounds.insert(response.shard, bound);
            }
            (None, false) => {
                self.unbounded.remove(&response.shard);
                self.bounds.remove(&response.shard);
                self.finished.insert(response.shard);
            }
        }
        Ok(())
    }

    /// Admit one candidate if it improves the answer.
    fn offer(&mut self, candidate: Candidate) {
        if self.config.k == 0 {
            self.stats.candidates_pruned += 1;
            return;
        }
        // Deduplicate before pruning. The same row on two replicas may arrive
        // with slightly different scores if the replicas quantised
        // independently; keeping the first admitted occurrence makes the result
        // depend on nothing but the id set, which is what makes it repeatable.
        if self.seen.contains(&candidate.id) {
            self.stats.duplicates_dropped += 1;
            return;
        }
        if self.heap.len() < self.config.k {
            self.seen.insert(candidate.id);
            self.heap.push(Worst(candidate));
            return;
        }
        // Safe: the heap is non-empty because k > 0 and len == k.
        let worst = self.heap.peek().expect("heap holds k > 0 results").0;
        if candidate.cmp_total(&worst) == Ordering::Less {
            self.heap.pop();
            self.seen.remove(&worst.id);
            self.seen.insert(candidate.id);
            self.heap.push(Worst(candidate));
        } else {
            self.stats.candidates_pruned += 1;
        }
    }

    /// The score of the current k-th result, if the heap is full.
    ///
    /// `None` while fewer than `k` results are held: there is no threshold to
    /// prune against, because every shard could still contribute something that
    /// belongs in the answer.
    fn kth_score(&self) -> Option<f32> {
        if self.heap.len() < self.config.k {
            None
        } else {
            self.heap.peek().map(|w| w.0.score)
        }
    }

    /// Shards whose unreturned best could still beat the k-th result.
    fn shards_that_could_improve(&self) -> Vec<ShardId> {
        let Some(kth) = self.kth_score() else {
            // Not yet k results, so anything any unfinished shard holds
            // belongs in the answer.
            let mut shards: Vec<ShardId> = self
                .bounds
                .keys()
                .chain(self.unbounded.iter())
                .copied()
                .collect();
            shards.sort_unstable();
            shards.dedup();
            return shards;
        };
        let mut shards: Vec<ShardId> = self
            .bounds
            .iter()
            .filter(|(_, bound)| **bound < kth)
            .map(|(shard, _)| *shard)
            .chain(self.unbounded.iter().copied())
            .collect();
        shards.sort_unstable();
        shards.dedup();
        shards
    }

    /// Note that an expansion round has been issued.
    pub fn begin_round(&mut self) {
        self.rounds += 1;
    }

    /// How many expansion rounds have been issued.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// Finish and produce the answer.
    pub fn finish(mut self) -> MergeOutcome {
        // Both of these read the heap, so they are taken before it is drained.
        // Computing them afterwards is silently wrong rather than loudly wrong:
        // an empty heap has no k-th score, so every shard looks capable of
        // improving the answer, no query can ever prove itself complete, and
        // every expansion goes out with an infinite threshold that asks each
        // shard to return everything it has. The two failing tests that found
        // this were the completeness assertions, not the result assertions --
        // the returned results were correct throughout.
        let improvable = self.shards_that_could_improve();
        let kth = self.kth_score();

        let mut results: Vec<Candidate> = self.heap.drain().map(|w| w.0).collect();
        results.sort_by(|a, b| a.cmp_total(b));
        let completeness = if !self.failures.is_empty() {
            Completeness::ShardsMissing {
                count: self.failures.len(),
            }
        } else if improvable.is_empty() {
            Completeness::Proven
        } else {
            Completeness::RoundsExhausted {
                rounds: self.rounds,
            }
        };

        // Only offer an expansion while rounds remain. Returning one the caller
        // is not permitted to run would invite an unbounded loop in exactly the
        // situation the round budget exists to stop.
        let expansion = if improvable.is_empty() || self.rounds >= self.config.max_rounds {
            None
        } else {
            Some(ExpansionRequest {
                shards: improvable,
                want: self.config.per_shard_k(),
                threshold: kth.unwrap_or(f32::INFINITY),
            })
        };

        MergeOutcome {
            results,
            completeness,
            failures: self.failures,
            expansion,
            stats: self.stats,
        }
    }
}

/// Merge one round of responses into a global top-k.
///
/// A convenience over [`GlobalMerge`] for the common case where every response
/// is available at once. The incremental interface exists for the case that is
/// not true, which is most real fan-outs.
pub fn merge_round(config: MergeConfig, responses: Vec<ShardResponse>) -> MergeOutcome {
    let mut merge = GlobalMerge::new(config);
    for response in responses {
        let _ = merge.accept(response);
    }
    merge.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SNAP: QuerySnapshot = QuerySnapshot {
        generation: 7,
        source_version: 3,
    };

    fn shard(id: ShardId, scores: &[(u128, f32)]) -> ShardResponse {
        ShardResponse::complete(
            id,
            SNAP,
            scores
                .iter()
                .map(|&(vid, s)| Candidate::new(vid, s, id))
                .collect(),
        )
    }

    fn ids(outcome: &MergeOutcome) -> Vec<u128> {
        outcome.results.iter().map(|c| c.id).collect()
    }

    /// The defect the bound exists for, demonstrated rather than described.
    ///
    /// Shard 0 holds four of the global best five. Asked for its local top-3 it
    /// returns three, and its fourth -- which belongs in the global answer --
    /// is lost. The merged list is the right length and looks complete. The
    /// bound is the only thing that reveals otherwise.
    #[test]
    fn a_skewed_shard_loses_results_the_merge_alone_cannot_detect() {
        let config = MergeConfig::new(5, SNAP);
        let skewed = ShardResponse::truncated(
            0,
            SNAP,
            vec![
                Candidate::new(1, 0.1, 0),
                Candidate::new(2, 0.2, 0),
                Candidate::new(3, 0.3, 0),
            ],
            0.4,
        );
        let other = shard(1, &[(10, 0.9), (11, 1.0)]);
        let outcome = merge_round(config, vec![skewed, other]);

        assert_eq!(ids(&outcome), vec![1, 2, 3, 10, 11]);
        // The k-th result scores 1.0 and shard 0 is still holding a 0.4, so
        // this answer is not the global top-5 and says so.
        assert!(!outcome.is_exact());
        let expansion = outcome.expansion.expect("shard 0 can still improve this");
        assert_eq!(expansion.shards, vec![0]);
        assert_eq!(expansion.threshold, 1.0);
    }

    /// The same query once shard 0 has been asked again. Its withheld results
    /// displace the weaker ones and the bound now proves the answer.
    #[test]
    fn expansion_recovers_the_lost_results_and_proves_the_answer() {
        let mut merge = GlobalMerge::new(MergeConfig::new(5, SNAP));
        merge
            .accept(ShardResponse::truncated(
                0,
                SNAP,
                vec![
                    Candidate::new(1, 0.1, 0),
                    Candidate::new(2, 0.2, 0),
                    Candidate::new(3, 0.3, 0),
                ],
                0.4,
            ))
            .unwrap();
        merge.accept(shard(1, &[(10, 0.9), (11, 1.0)])).unwrap();

        merge.begin_round();
        merge
            .accept(shard(0, &[(4, 0.4), (5, 0.5), (6, 2.0)]))
            .unwrap();

        let outcome = merge.finish();
        assert_eq!(ids(&outcome), vec![1, 2, 3, 4, 5]);
        assert!(outcome.is_exact(), "{:?}", outcome.completeness);
        assert!(outcome.expansion.is_none());
    }

    /// A shard whose withheld best is worse than the k-th cannot improve the
    /// answer, so no expansion is requested even though it withheld results.
    #[test]
    fn a_shard_holding_only_worse_results_is_not_asked_again() {
        let config = MergeConfig::new(2, SNAP);
        let a = ShardResponse::truncated(0, SNAP, vec![Candidate::new(1, 0.1, 0)], 5.0);
        let b = ShardResponse::truncated(1, SNAP, vec![Candidate::new(2, 0.2, 1)], 6.0);
        let outcome = merge_round(config, vec![a, b]);
        assert!(outcome.is_exact());
        assert!(outcome.expansion.is_none());
    }

    /// A shard that returned everything it has contributes no bound at all, and
    /// must not be confused with one that merely returned nothing this round.
    #[test]
    fn a_finished_shard_and_an_empty_round_are_different_states() {
        let finished = merge_round(MergeConfig::new(2, SNAP), vec![shard(0, &[(1, 0.1)])]);
        assert!(finished.expansion.is_none());

        let holding = merge_round(
            MergeConfig::new(2, SNAP),
            vec![ShardResponse::truncated(0, SNAP, vec![], 0.1)],
        );
        assert!(
            holding.expansion.is_some(),
            "a shard holding a 0.1 behind a filter can still contribute"
        );
    }

    /// The same row reachable through two replicas appears once.
    #[test]
    fn a_row_present_on_two_replicas_is_returned_once() {
        let config = MergeConfig::new(3, SNAP);
        let a = shard(0, &[(1, 0.1), (2, 0.2)]);
        let b = ShardResponse::complete(
            0,
            SNAP,
            vec![Candidate::new(1, 0.1, 0), Candidate::new(3, 0.3, 0)],
        );
        let outcome = merge_round(config, vec![a, b]);
        assert_eq!(ids(&outcome), vec![1, 2, 3]);
        assert_eq!(outcome.stats.duplicates_dropped, 1);
    }

    /// Deduplication keeps the first admitted occurrence, so replicas that
    /// quantised independently cannot change the id set that comes back.
    #[test]
    fn a_duplicate_with_a_different_score_does_not_change_the_result_set() {
        let config = MergeConfig::new(2, SNAP);
        let a = shard(0, &[(1, 0.10), (2, 0.50)]);
        let b = ShardResponse::complete(0, SNAP, vec![Candidate::new(1, 0.11, 0)]);
        let outcome = merge_round(config, vec![a, b]);
        assert_eq!(ids(&outcome), vec![1, 2]);
        assert_eq!(outcome.results[0].score, 0.10);
    }

    /// A shard on another generation is answering a different question, so its
    /// whole response is rejected -- not the part that happens to look fine.
    #[test]
    fn a_shard_on_another_generation_is_rejected_entirely() {
        let mut merge = GlobalMerge::new(MergeConfig::new(3, SNAP));
        let stale = ShardResponse::complete(
            1,
            QuerySnapshot::new(6, 3),
            vec![Candidate::new(9, 0.01, 1)],
        );
        let err = merge.accept(stale).expect_err("rejected");
        assert!(matches!(err, ShardFailure::StaleSnapshot { .. }));
        merge.accept(shard(0, &[(1, 0.5)])).unwrap();
        let outcome = merge.finish();
        assert_eq!(ids(&outcome), vec![1], "the 0.01 must not appear");
        assert_eq!(outcome.stats.stale_shards, 1);
        assert!(!outcome.is_exact());
    }

    /// A generation match with a different source version is still a mismatch.
    /// Two indexes built at the same generation number over different data are
    /// not interchangeable.
    #[test]
    fn a_matching_generation_over_different_source_data_is_still_stale() {
        let mut merge = GlobalMerge::new(MergeConfig::new(3, SNAP));
        let err = merge
            .accept(ShardResponse::complete(1, QuerySnapshot::new(7, 4), vec![]))
            .expect_err("rejected");
        assert!(matches!(err, ShardFailure::StaleSnapshot { .. }));
    }

    /// A shard returning a candidate labelled with another shard's id is a
    /// routing fault. Accepting it would break deduplication accounting and
    /// hide the fault.
    #[test]
    fn a_misattributed_candidate_rejects_the_response() {
        let mut merge = GlobalMerge::new(MergeConfig::new(3, SNAP));
        let confused = ShardResponse::complete(0, SNAP, vec![Candidate::new(1, 0.1, 4)]);
        let err = merge.accept(confused).expect_err("rejected");
        assert_eq!(err, ShardFailure::Misattributed { claimed: 4 });
    }

    /// Losing a shard is reported, not silently absorbed into a shorter list.
    #[test]
    fn an_unreachable_shard_is_reported_rather_than_hidden() {
        let mut merge = GlobalMerge::new(MergeConfig::new(3, SNAP));
        merge.accept(shard(0, &[(1, 0.1), (2, 0.2)])).unwrap();
        merge.record_unreachable(2);
        let outcome = merge.finish();
        assert_eq!(ids(&outcome), vec![1, 2]);
        assert_eq!(
            outcome.completeness,
            Completeness::ShardsMissing { count: 1 }
        );
        assert!(!outcome.is_exact());
        assert_eq!(outcome.failures.get(&2), Some(&ShardFailure::Unreachable));
    }

    /// A retry that reaches another replica clears the failure. Without this a
    /// query that recovered would still refuse to call itself exact.
    #[test]
    fn a_successful_replica_retry_clears_the_failure() {
        let mut merge = GlobalMerge::new(MergeConfig::new(2, SNAP));
        merge.record_unreachable(1);
        merge.accept(shard(0, &[(1, 0.1)])).unwrap();
        merge.begin_round();
        merge.accept(shard(1, &[(2, 0.2)])).unwrap();
        let outcome = merge.finish();
        assert!(outcome.failures.is_empty());
        assert!(outcome.is_exact());
        assert_eq!(ids(&outcome), vec![1, 2]);
    }

    /// A shard that stopped on its own budget cannot bound what it holds, so
    /// its optimistic-looking response must not prove the answer.
    #[test]
    fn a_shard_that_ran_out_of_budget_cannot_prove_completeness() {
        let mut merge = GlobalMerge::new(MergeConfig::new(1, SNAP));
        merge
            .accept(ShardResponse {
                shard: 0,
                snapshot: SNAP,
                candidates: vec![Candidate::new(1, 0.1, 0)],
                next_best_score: None,
                budget_exhausted: true,
            })
            .unwrap();
        let outcome = merge.finish();
        assert!(!outcome.completeness.is_proven());
        assert_eq!(
            outcome.expansion.map(|e| e.shards),
            Some(vec![0]),
            "the shard is asked again because it cannot vouch for what it skipped"
        );
    }

    /// Identical inputs in any arrival order produce an identical answer.
    #[test]
    fn shard_arrival_order_does_not_change_the_answer() {
        let build = || {
            vec![
                shard(0, &[(1, 0.5), (4, 0.5)]),
                shard(1, &[(2, 0.5), (5, 0.5)]),
                shard(2, &[(3, 0.5), (6, 0.5)]),
            ]
        };
        let forward = merge_round(MergeConfig::new(3, SNAP), build());
        let mut reversed = build();
        reversed.reverse();
        let backward = merge_round(MergeConfig::new(3, SNAP), reversed);
        assert!(
            forward.results.iter().all(|c| c.score == 0.5),
            "test premise: every candidate ties on score"
        );
        assert_eq!(ids(&forward), ids(&backward));
        assert_eq!(ids(&forward), vec![1, 2, 3]);
    }

    /// A shard returning NaN has failed at something. The result sorts last
    /// rather than wherever an inconsistent comparator leaves it.
    #[test]
    fn an_unorderable_score_sorts_last_instead_of_corrupting_the_order() {
        let config = MergeConfig::new(3, SNAP);
        let a = shard(0, &[(1, f32::NAN), (2, 0.9)]);
        let b = shard(1, &[(3, 0.1)]);
        let outcome = merge_round(config, vec![a, b]);
        assert_eq!(ids(&outcome), vec![3, 2, 1]);
    }

    /// Even when every score is NaN the order is total and repeatable.
    #[test]
    fn an_all_nan_round_still_returns_a_stable_order() {
        let build = || {
            vec![
                shard(0, &[(3, f32::NAN)]),
                shard(1, &[(1, f32::NAN)]),
                shard(2, &[(2, f32::NAN)]),
            ]
        };
        let forward = merge_round(MergeConfig::new(3, SNAP), build());
        let mut reversed = build();
        reversed.reverse();
        let backward = merge_round(MergeConfig::new(3, SNAP), reversed);
        assert_eq!(ids(&forward), ids(&backward));
    }

    /// The heap holds k, not S times k'. Ten shards of a thousand results each
    /// leave three results held.
    #[test]
    fn memory_is_bounded_by_k_not_by_the_fan_out() {
        let mut merge = GlobalMerge::new(MergeConfig::new(3, SNAP));
        for s in 0..10u32 {
            let candidates: Vec<Candidate> = (0..1_000u128)
                .map(|i| Candidate::new(s as u128 * 1_000 + i, i as f32, s))
                .collect();
            merge
                .accept(ShardResponse::complete(s, SNAP, candidates))
                .unwrap();
        }
        assert_eq!(merge.heap.len(), 3);
        let outcome = merge.finish();
        assert_eq!(outcome.results.len(), 3);
        assert_eq!(outcome.stats.candidates_seen, 10_000);
        assert!(outcome.stats.candidates_pruned > 9_000);
    }

    /// The k-th result is the true k-th across all shards, which is what makes
    /// the distributed answer equal the single-node one.
    #[test]
    fn the_distributed_answer_equals_a_centralized_exact_baseline() {
        let all: Vec<(u128, f32)> = (0..60u128).map(|i| (i, (i as f32 * 37.0) % 60.0)).collect();
        let mut baseline = all.clone();
        baseline.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        let expected: Vec<u128> = baseline.iter().take(10).map(|(id, _)| *id).collect();

        let mut merge = GlobalMerge::new(MergeConfig::new(10, SNAP));
        for s in 0..3u32 {
            let mut mine: Vec<(u128, f32)> = all
                .iter()
                .filter(|(id, _)| (*id as u32) % 3 == s)
                .copied()
                .collect();
            mine.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let candidates: Vec<Candidate> = mine
                .iter()
                .map(|&(id, sc)| Candidate::new(id, sc, s))
                .collect();
            merge
                .accept(ShardResponse::complete(s, SNAP, candidates))
                .unwrap();
        }
        let outcome = merge.finish();
        assert!(outcome.is_exact());
        assert_eq!(ids(&outcome), expected);
    }

    /// Expansion stops. A shard that keeps reporting an optimistic bound cannot
    /// turn one query into an unbounded conversation.
    #[test]
    fn expansion_is_bounded_by_the_round_budget() {
        let mut config = MergeConfig::new(1, SNAP);
        config.max_rounds = 2;
        let mut merge = GlobalMerge::new(config);
        for _ in 0..5 {
            merge.begin_round();
            merge
                .accept(ShardResponse::truncated(
                    0,
                    SNAP,
                    vec![Candidate::new(1, 0.5, 0)],
                    0.0,
                ))
                .unwrap();
        }
        let outcome = merge.finish();
        assert!(outcome.expansion.is_none(), "no further round is offered");
        assert_eq!(
            outcome.completeness,
            Completeness::RoundsExhausted { rounds: 5 }
        );
    }

    /// Before k results are held there is no threshold, so every unfinished
    /// shard is still a candidate for expansion.
    #[test]
    fn an_underfull_answer_asks_every_unfinished_shard() {
        let config = MergeConfig::new(10, SNAP);
        let a = ShardResponse::truncated(0, SNAP, vec![Candidate::new(1, 0.1, 0)], 99.0);
        let b = ShardResponse::truncated(1, SNAP, vec![Candidate::new(2, 0.2, 1)], 99.0);
        let outcome = merge_round(config, vec![a, b]);
        let expansion = outcome.expansion.expect("under k, so keep asking");
        assert_eq!(expansion.shards, vec![0, 1]);
        assert_eq!(expansion.threshold, f32::INFINITY);
    }

    #[test]
    fn oversampling_never_asks_for_less_than_k() {
        let mut config = MergeConfig::new(10, SNAP);
        config.oversampling = 0.5;
        assert_eq!(config.per_shard_k(), 10);
        config.oversampling = f32::NAN;
        assert_eq!(config.per_shard_k(), 10);
        config.oversampling = 2.5;
        assert_eq!(config.per_shard_k(), 25);
    }

    #[test]
    fn a_zero_k_query_returns_nothing_without_panicking() {
        let outcome = merge_round(MergeConfig::new(0, SNAP), vec![shard(0, &[(1, 0.1)])]);
        assert!(outcome.results.is_empty());
        assert_eq!(MergeConfig::new(0, SNAP).per_shard_k(), 0);
    }

    #[test]
    fn a_query_that_reached_no_shards_is_not_exact() {
        let outcome = merge_round(MergeConfig::new(3, SNAP), vec![]);
        assert!(outcome.results.is_empty());
        assert!(outcome.is_exact(), "no shards were expected, none failed");

        let mut merge = GlobalMerge::new(MergeConfig::new(3, SNAP));
        merge.record_unreachable(0);
        let outcome = merge.finish();
        assert!(
            !outcome.is_exact(),
            "a shard was expected and did not answer"
        );
    }

    /// Evicting a result removes its id, so a better copy of a row that was
    /// pushed out can be admitted again rather than being permanently barred.
    #[test]
    fn an_evicted_id_can_be_admitted_again() {
        let mut merge = GlobalMerge::new(MergeConfig::new(1, SNAP));
        merge.accept(shard(0, &[(1, 5.0)])).unwrap();
        merge.accept(shard(1, &[(2, 1.0)])).unwrap();
        merge.accept(shard(2, &[(1, 0.5)])).unwrap();
        let outcome = merge.finish();
        assert_eq!(ids(&outcome), vec![1]);
        assert_eq!(outcome.results[0].score, 0.5);
    }

    #[test]
    fn statistics_account_for_every_candidate_offered() {
        let config = MergeConfig::new(2, SNAP);
        let a = shard(0, &[(1, 0.1), (2, 0.2), (3, 9.0)]);
        let b = shard(1, &[(1, 0.1)]);
        let outcome = merge_round(config, vec![a, b]);
        assert_eq!(outcome.stats.candidates_seen, 4);
        assert_eq!(outcome.stats.duplicates_dropped, 1);
        assert_eq!(outcome.stats.candidates_pruned, 1);
        assert_eq!(outcome.stats.shards_merged, 2);
        assert_eq!(
            outcome.results.len()
                + outcome.stats.duplicates_dropped
                + outcome.stats.candidates_pruned,
            outcome.stats.candidates_seen
        );
    }
}
