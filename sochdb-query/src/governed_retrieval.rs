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

//! Fail-closed filter pushdown and candidate budgeting for governed retrieval.
//!
//! The problem this exists to prevent has one shape. A caller compiles a row
//! policy into a filter, the engine can push down most of it, and the part it
//! cannot push is quietly dropped. Every remaining check passes, the query is
//! fast, the results look right -- and they contain rows the policy excluded.
//! Nothing fails. That is what makes it dangerous.
//!
//! Two properties make that impossible here.
//!
//! The first is that the filter is in conjunctive normal form. A CNF filter is
//! an AND of clauses, so any partition of the clauses into two sets satisfies
//! `pushed AND residual == original` by construction. Splitting is sound
//! precisely because nothing is being rewritten -- clauses are only being
//! assigned to one side or the other. `split_conjunction` never discards a
//! clause, and a test asserts every input clause lands on exactly one side.
//!
//! The second is that the residual cannot be ignored. Planning returns a
//! [`RetrievalPlan`] enum whose partial variant carries the residual, so a
//! caller cannot obtain a pushed filter without also being handed the part
//! that was not pushed. There is no accessor that returns only the pushed
//! filter. When no strategy is acceptable the planner returns an error rather
//! than a weakened filter.
//!
//! The budgeting exists for a related reason. Filtering after candidate
//! generation means a search for `k` results can return fewer than `k` once
//! the residual is applied, and returning a short list without saying so
//! reads as "there is no more data". The escalation loop grows the candidate
//! budget geometrically and reports precisely why it stopped.

use crate::filter_ir::{Disjunction, FilterAtom, FilterIR};
use std::collections::BTreeSet;

/// Smallest selectivity used when sizing a budget.
///
/// A selectivity estimate of zero would demand an infinite budget. Clamping to
/// this instead means a filter believed to match nothing still produces a
/// finite plan that can discover it was wrong.
pub const MIN_SELECTIVITY: f64 = 1e-4;

/// Default multiplier applied to the ideal candidate count.
///
/// The estimate is an estimate. Fetching exactly `k / s` candidates succeeds
/// only when the estimate is perfect or pessimistic, and being wrong costs a
/// whole extra round trip.
pub const DEFAULT_SAFETY_FACTOR: f64 = 2.0;

/// Growth applied on each escalation round.
pub const DEFAULT_GROWTH: f64 = 4.0;

/// Why a clause could not be pushed down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedClause {
    /// Position of the clause in the original filter, so the reason can be
    /// traced back to what the caller wrote.
    pub clause_index: usize,
    pub reason: &'static str,
    /// Fields the clause constrains, for diagnostics.
    pub fields: Vec<String>,
}

/// What a retrieval engine can evaluate during candidate generation.
pub trait PushdownCapability {
    /// Whether an atom can be evaluated by the engine.
    ///
    /// Returning `None` means supported; returning a reason means it is not.
    /// The reason is required rather than optional because an unsupported
    /// filter that cannot say why is one nobody will ever fix.
    fn unsupported_reason(&self, atom: &FilterAtom) -> Option<&'static str>;
}

/// The capability of an engine that can evaluate exact-match and set-membership
/// predicates on metadata, which is what the v2 filter IR carries today.
#[derive(Debug, Clone, Default)]
pub struct EqualityAndSetMembership;

impl PushdownCapability for EqualityAndSetMembership {
    fn unsupported_reason(&self, atom: &FilterAtom) -> Option<&'static str> {
        match atom {
            FilterAtom::Eq { .. } | FilterAtom::In { .. } => None,
            FilterAtom::True | FilterAtom::False => None,
            // Negation is deliberately excluded at this stage. A NOT that is
            // pushed down incorrectly does not narrow the result set, it widens
            // it, and widening is the direction that leaks.
            FilterAtom::Ne { .. } => Some("negated equality is not pushed down"),
            FilterAtom::NotIn { .. } => Some("negated set membership is not pushed down"),
            FilterAtom::Range { .. } => Some("range predicates are not pushed down"),
            FilterAtom::Prefix { .. } => Some("prefix predicates are not pushed down"),
            FilterAtom::Contains { .. } => Some("substring predicates are not pushed down"),
            FilterAtom::HasTag { .. } => Some("tag predicates are not pushed down"),
        }
    }
}

/// The result of partitioning a CNF filter by what an engine can evaluate.
///
/// This type is deliberately not constructible with a residual that has been
/// discarded: both halves are always present, and `pushed AND residual` is
/// always equivalent to the filter that went in.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitFilter {
    pushed: FilterIR,
    residual: FilterIR,
    unsupported: Vec<UnsupportedClause>,
}

impl SplitFilter {
    pub fn is_complete(&self) -> bool {
        self.unsupported.is_empty()
    }

    pub fn pushed(&self) -> &FilterIR {
        &self.pushed
    }

    pub fn residual(&self) -> &FilterIR {
        &self.residual
    }

    pub fn unsupported(&self) -> &[UnsupportedClause] {
        &self.unsupported
    }
}

/// Partition a CNF filter into the part an engine can evaluate and the part it
/// cannot.
///
/// A clause is pushed only when *every* atom in it is supported. This is not a
/// conservative nicety: a clause is a disjunction, so dropping one alternative
/// from `(a OR b)` and pushing `(a)` excludes rows that satisfy `b`, which
/// silently loses authorised results. The reverse -- pushing `(a OR b OR
/// anything)` -- would admit rows the policy excludes. Neither is acceptable,
/// so a clause moves whole or not at all.
pub fn split_conjunction<C: PushdownCapability>(filter: &FilterIR, capability: &C) -> SplitFilter {
    let mut pushed = Vec::new();
    let mut residual = Vec::new();
    let mut unsupported = Vec::new();

    for (index, clause) in filter.clauses.iter().enumerate() {
        let blocker = clause
            .atoms
            .iter()
            .find_map(|atom| capability.unsupported_reason(atom));

        match blocker {
            None => pushed.push(clause.clone()),
            Some(reason) => {
                unsupported.push(UnsupportedClause {
                    clause_index: index,
                    reason,
                    fields: clause
                        .atoms
                        .iter()
                        .filter_map(|a| a.field().map(str::to_string))
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                });
                residual.push(clause.clone());
            }
        }
    }

    SplitFilter {
        pushed: FilterIR { clauses: pushed },
        residual: FilterIR { clauses: residual },
        unsupported,
    }
}

/// What a caller is willing to do about a filter that cannot be fully pushed.
///
/// There is no variant meaning "ignore it". That is the point of the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualStrategy {
    /// Fetch more candidates than needed and apply the residual locally.
    Overfetch,
    /// Abandon the index and scan exactly.
    ExactScan,
    /// Refuse the query.
    Reject,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GovernanceError {
    /// The filter could not be fully pushed and the caller chose to refuse
    /// rather than overfetch or scan.
    UnsupportedFilter { unsupported: Vec<UnsupportedClause> },
    /// The filter cannot match anything.
    Unsatisfiable,
    /// `k` was zero.
    EmptyRequest,
}

impl std::fmt::Display for GovernanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFilter { unsupported } => {
                write!(f, "filter cannot be fully pushed down: ")?;
                for (i, u) in unsupported.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "clause {} ({})", u.clause_index, u.reason)?;
                }
                Ok(())
            }
            Self::Unsatisfiable => write!(f, "filter cannot match any row"),
            Self::EmptyRequest => write!(f, "k must be at least 1"),
        }
    }
}

impl std::error::Error for GovernanceError {}

/// How candidate generation should proceed.
///
/// Every variant that leaves work undone carries that work with it. There is
/// no way to pattern-match out a pushed filter and forget the residual, and no
/// `Partial` variant without a residual field.
#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalPlan {
    /// The engine can evaluate the entire filter. Its answer is authoritative
    /// and `k` candidates are enough.
    FullyPushed { filter: FilterIR, budget: Budget },
    /// The engine evaluates part of the filter; the caller must apply the rest
    /// to every candidate before counting it as a result.
    Partial {
        pushed: FilterIR,
        residual: FilterIR,
        unsupported: Vec<UnsupportedClause>,
        budget: Budget,
    },
    /// Nothing useful can be pushed; the caller must evaluate the whole filter
    /// over an exact scan.
    ExactScan {
        filter: FilterIR,
        unsupported: Vec<UnsupportedClause>,
    },
}

/// Candidate budget and how it grows.
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    pub initial: usize,
    pub max: usize,
    pub growth: f64,
}

/// Initial candidate count.
///
/// `C0 = min(max_candidates, ceil(k / max(s, eps)) * safety_factor)`
///
/// The clamp on selectivity is what stops a pessimistic estimate from asking
/// for an unbounded scan, and the clamp on `max_candidates` is what stops the
/// whole thing from becoming one.
pub fn initial_candidates(k: usize, selectivity: f64, safety_factor: f64, max: usize) -> usize {
    let s = selectivity.max(MIN_SELECTIVITY);
    let ideal = (k as f64 / s).ceil() * safety_factor.max(1.0);
    // Saturate rather than wrap: an f64 beyond usize range casts to an
    // unspecified value, and this number decides how much work is done.
    let ideal = if ideal.is_finite() && ideal < usize::MAX as f64 {
        ideal as usize
    } else {
        usize::MAX
    };
    ideal.clamp(k.max(1), max.max(k.max(1)))
}

/// Plan candidate generation for a governed query.
pub fn plan_retrieval<C: PushdownCapability>(
    filter: &FilterIR,
    capability: &C,
    strategy: ResidualStrategy,
    k: usize,
    estimated_selectivity: f64,
    max_candidates: usize,
) -> Result<RetrievalPlan, GovernanceError> {
    if k == 0 {
        return Err(GovernanceError::EmptyRequest);
    }
    if filter.is_none() {
        // A filter that matches nothing is not an error the caller should have
        // to distinguish from an empty result by inspecting counts.
        return Err(GovernanceError::Unsatisfiable);
    }

    let split = split_conjunction(filter, capability);

    if split.is_complete() {
        return Ok(RetrievalPlan::FullyPushed {
            filter: split.pushed,
            // The engine applies the whole filter, so every candidate it
            // returns is a result and no overfetch is needed.
            budget: Budget {
                initial: k,
                max: max_candidates.max(k),
                growth: DEFAULT_GROWTH,
            },
        });
    }

    match strategy {
        ResidualStrategy::Reject => Err(GovernanceError::UnsupportedFilter {
            unsupported: split.unsupported,
        }),
        ResidualStrategy::ExactScan => Ok(RetrievalPlan::ExactScan {
            filter: filter.clone(),
            unsupported: split.unsupported,
        }),
        ResidualStrategy::Overfetch => {
            // Nothing pushed means the index contributes only its ordering.
            // That is still useful for a nearest-neighbour query, but the
            // caller should know the filter is doing none of the narrowing.
            Ok(RetrievalPlan::Partial {
                budget: Budget {
                    initial: initial_candidates(
                        k,
                        estimated_selectivity,
                        DEFAULT_SAFETY_FACTOR,
                        max_candidates,
                    ),
                    max: max_candidates.max(k),
                    growth: DEFAULT_GROWTH,
                },
                pushed: split.pushed,
                residual: split.residual,
                unsupported: split.unsupported,
            })
        }
    }
}

/// Why an escalation loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exhaustion {
    /// Enough authorised results were found.
    Satisfied,
    /// The candidate budget was reached first. Fewer than `k` results were
    /// found, and more may exist.
    BudgetExhausted,
    /// The index ran out of candidates. Fewer than `k` results exist.
    IndexExhausted,
}

/// What actually happened, in enough detail to tell a short result set caused
/// by an exhausted budget from one caused by there being nothing more.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalEvidence {
    pub requested_k: usize,
    pub rounds: usize,
    pub candidates_examined: usize,
    pub authorized: usize,
    pub rejected: usize,
    pub estimated_selectivity: f64,
    pub observed_selectivity: Option<f64>,
    pub exhaustion: Exhaustion,
    pub unsupported: Vec<UnsupportedClause>,
}

impl RetrievalEvidence {
    /// Whether the caller may treat this as a complete answer.
    ///
    /// A budget-exhausted result is short because the search gave up, not
    /// because the data ran out. Presenting it as complete turns a resource
    /// limit into a factual claim that no more matching rows exist.
    pub fn is_complete(&self) -> bool {
        matches!(
            self.exhaustion,
            Exhaustion::Satisfied | Exhaustion::IndexExhausted
        )
    }
}

/// Run a governed search, escalating the candidate budget until enough
/// authorised results are found or a stated limit is reached.
///
/// `generate` is asked for a candidate list of a given size; returning fewer
/// than requested means the index is exhausted. `authorize` decides whether a
/// candidate survives the residual filter and any check the engine could not
/// perform. It is applied to every candidate, in every round -- there is no
/// path that returns a candidate the caller has not approved.
pub fn escalating_search<T, G, A>(
    budget: &Budget,
    k: usize,
    estimated_selectivity: f64,
    unsupported: Vec<UnsupportedClause>,
    mut generate: G,
    mut authorize: A,
) -> (Vec<T>, RetrievalEvidence)
where
    G: FnMut(usize) -> Vec<T>,
    A: FnMut(&T) -> bool,
{
    let mut size = budget.initial.max(k).min(budget.max.max(k));
    let mut rounds = 0usize;
    let mut authorized: Vec<T> = Vec::new();
    let examined: usize;
    let rejected: usize;
    let exhaustion: Exhaustion;

    loop {
        rounds += 1;
        let candidates = generate(size);
        let produced = candidates.len();

        // Each round regenerates from scratch rather than continuing, because
        // an ANN index has no cursor: asking for more means asking for a wider
        // search. Counting only this round's candidates keeps the observed
        // selectivity a property of one measurement rather than a mixture.
        let mut round_rejected = 0usize;
        authorized.clear();

        for candidate in candidates {
            if authorize(&candidate) {
                authorized.push(candidate);
            } else {
                round_rejected += 1;
            }
        }

        if authorized.len() >= k {
            examined = produced;
            rejected = round_rejected;
            exhaustion = Exhaustion::Satisfied;
            break;
        }
        if produced < size {
            examined = produced;
            rejected = round_rejected;
            // The index returned fewer than asked for, so there is nothing
            // left to find. A larger budget cannot help.
            exhaustion = Exhaustion::IndexExhausted;
            break;
        }
        if size >= budget.max {
            examined = produced;
            rejected = round_rejected;
            exhaustion = Exhaustion::BudgetExhausted;
            break;
        }

        let next = (size as f64 * budget.growth.max(1.5)).ceil() as usize;
        size = next.min(budget.max).max(size + 1);
    }

    authorized.truncate(k);

    (
        authorized,
        RetrievalEvidence {
            requested_k: k,
            rounds,
            candidates_examined: examined,
            authorized: examined - rejected,
            rejected,
            estimated_selectivity,
            observed_selectivity: if examined == 0 {
                None
            } else {
                Some((examined - rejected) as f64 / examined as f64)
            },
            exhaustion,
            unsupported,
        },
    )
}

/// Evaluate a CNF filter against a metadata lookup.
///
/// An absent field does not satisfy a predicate. The alternative -- treating
/// a missing attribute as matching -- makes a row with no tenant label visible
/// to every tenant, which is the worst possible default.
pub fn matches<F>(filter: &FilterIR, lookup: F) -> bool
where
    F: Fn(&str) -> Option<crate::filter_ir::FilterValue>,
{
    filter
        .clauses
        .iter()
        .all(|clause| clause_matches(clause, &lookup))
}

fn clause_matches<F>(clause: &Disjunction, lookup: &F) -> bool
where
    F: Fn(&str) -> Option<crate::filter_ir::FilterValue>,
{
    // An empty disjunction is an empty OR, which is false. It cannot arise
    // from a well-formed filter, and treating it as true would turn a
    // malformed policy into an unrestricted one.
    clause.atoms.iter().any(|atom| atom_matches(atom, lookup))
}

fn atom_matches<F>(atom: &FilterAtom, lookup: &F) -> bool
where
    F: Fn(&str) -> Option<crate::filter_ir::FilterValue>,
{
    match atom {
        FilterAtom::True => true,
        FilterAtom::False => false,
        FilterAtom::Eq { field, value } => {
            lookup(field).map(|v| v.eq_match(value)).unwrap_or(false)
        }
        FilterAtom::Ne { field, value } => {
            // A missing field does not satisfy `!=` either. This is not the
            // SQL three-valued rule; it is the deliberate choice that an
            // unlabelled row never passes a policy predicate.
            lookup(field).map(|v| !v.eq_match(value)).unwrap_or(false)
        }
        FilterAtom::In { field, values } => lookup(field)
            .map(|v| values.iter().any(|c| v.eq_match(c)))
            .unwrap_or(false),
        FilterAtom::NotIn { field, values } => lookup(field)
            .map(|v| !values.iter().any(|c| v.eq_match(c)))
            .unwrap_or(false),
        FilterAtom::Range {
            field,
            min,
            max,
            min_inclusive,
            max_inclusive,
        } => {
            let Some(v) = lookup(field) else {
                return false;
            };
            if let Some(min) = min {
                match v.partial_cmp(min) {
                    Some(std::cmp::Ordering::Less) => return false,
                    Some(std::cmp::Ordering::Equal) if !min_inclusive => return false,
                    // Values that cannot be ordered against the bound (a string
                    // against a number, say) do not satisfy the range. Treating
                    // an incomparable value as in-range would let a mistyped
                    // field bypass the bound entirely.
                    None => return false,
                    _ => {}
                }
            }
            if let Some(max) = max {
                match v.partial_cmp(max) {
                    Some(std::cmp::Ordering::Greater) => return false,
                    Some(std::cmp::Ordering::Equal) if !max_inclusive => return false,
                    None => return false,
                    _ => {}
                }
            }
            true
        }
        FilterAtom::Prefix { field, prefix } => match lookup(field) {
            Some(crate::filter_ir::FilterValue::String(s)) => s.starts_with(prefix),
            _ => false,
        },
        FilterAtom::Contains { field, substring } => match lookup(field) {
            Some(crate::filter_ir::FilterValue::String(s)) => s.contains(substring),
            _ => false,
        },
        FilterAtom::HasTag { tag } => match lookup("tags") {
            Some(crate::filter_ir::FilterValue::String(s)) => s.split(',').any(|t| t.trim() == tag),
            _ => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_ir::FilterValue;
    use std::collections::HashMap;

    fn tenant_clause(value: &str) -> Disjunction {
        Disjunction::single(FilterAtom::eq("tenant", value))
    }

    fn range_clause() -> Disjunction {
        Disjunction::single(FilterAtom::range("score", Some(0.5.into()), None))
    }

    fn lookup_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<FilterValue> + use<> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |field: &str| map.get(field).map(|v| FilterValue::String(v.clone()))
    }

    /// The property that makes splitting sound at all: nothing is created and
    /// nothing is lost. Every clause of the input appears exactly once across
    /// the two halves.
    #[test]
    fn splitting_conserves_every_clause() {
        let filter = FilterIR {
            clauses: vec![
                tenant_clause("acme"),
                range_clause(),
                Disjunction::single(FilterAtom::eq("region", "eu")),
                Disjunction::single(FilterAtom::Prefix {
                    field: "path".to_string(),
                    prefix: "/pub".to_string(),
                }),
            ],
        };
        let split = split_conjunction(&filter, &EqualityAndSetMembership);

        assert_eq!(
            split.pushed().clauses.len() + split.residual().clauses.len(),
            filter.clauses.len(),
            "clauses were lost or duplicated"
        );
        for clause in &filter.clauses {
            let in_pushed = split.pushed().clauses.contains(clause);
            let in_residual = split.residual().clauses.contains(clause);
            assert!(
                in_pushed ^ in_residual,
                "clause {:?} is in both halves or neither",
                clause
            );
        }
        assert!(!split.is_complete());
        assert_eq!(split.unsupported().len(), 2);
    }

    /// The defect this module exists to prevent, stated as a test. If the
    /// unsupported half were dropped, the pushed filter alone would admit a
    /// row that the full filter excludes.
    #[test]
    fn dropping_the_residual_would_admit_a_forbidden_row() {
        let filter = FilterIR {
            clauses: vec![
                tenant_clause("acme"),
                Disjunction::single(FilterAtom::Ne {
                    field: "classification".to_string(),
                    value: FilterValue::String("secret".to_string()),
                }),
            ],
        };
        let split = split_conjunction(&filter, &EqualityAndSetMembership);

        let secret_row = lookup_from(&[("tenant", "acme"), ("classification", "secret")]);

        assert!(
            matches(split.pushed(), &secret_row),
            "test premise: the pushed half alone admits the row"
        );
        assert!(
            !matches(&filter, &secret_row),
            "the full filter must exclude the row"
        );
        assert!(
            !matches(split.residual(), &secret_row),
            "the residual is what excludes it, so it cannot be dropped"
        );
    }

    /// A clause moves whole or not at all. Pushing the supported alternative of
    /// `(a OR b)` would exclude rows satisfying only `b`, losing authorised
    /// results; a test that only checked for leaks would miss this direction.
    #[test]
    fn a_partially_supported_disjunction_is_not_split_apart() {
        let filter = FilterIR {
            clauses: vec![Disjunction::new(vec![
                FilterAtom::eq("tenant", "acme"),
                FilterAtom::Prefix {
                    field: "path".to_string(),
                    prefix: "/public".to_string(),
                },
            ])],
        };
        let split = split_conjunction(&filter, &EqualityAndSetMembership);

        assert!(
            split.pushed().clauses.is_empty(),
            "half a disjunction was pushed"
        );
        assert_eq!(split.residual().clauses.len(), 1);

        let public_row = lookup_from(&[("tenant", "other"), ("path", "/public/x")]);
        assert!(
            matches(&filter, &public_row),
            "the row satisfies the unsupported alternative"
        );
    }

    #[test]
    fn a_fully_supported_filter_is_pushed_whole() {
        let filter = FilterIR {
            clauses: vec![
                tenant_clause("acme"),
                Disjunction::single(FilterAtom::in_set("region", vec!["eu".into(), "us".into()])),
            ],
        };
        let split = split_conjunction(&filter, &EqualityAndSetMembership);
        assert!(split.is_complete());
        assert!(split.residual().clauses.is_empty());
        assert_eq!(split.pushed().clauses.len(), 2);
    }

    /// A caller that refuses to overfetch gets an error naming what it could
    /// not push, not a plan that quietly does less.
    #[test]
    fn rejecting_is_an_available_answer_and_says_why() {
        let filter = FilterIR {
            clauses: vec![tenant_clause("acme"), range_clause()],
        };
        let error = plan_retrieval(
            &filter,
            &EqualityAndSetMembership,
            ResidualStrategy::Reject,
            10,
            0.5,
            10_000,
        )
        .expect_err("an unpushable filter produced a plan under Reject");

        assert!(error.to_string().contains("range"));
        match error {
            GovernanceError::UnsupportedFilter { unsupported } => {
                assert_eq!(unsupported.len(), 1);
                assert_eq!(unsupported[0].clause_index, 1);
                assert!(unsupported[0].reason.contains("range"));
                assert_eq!(unsupported[0].fields, vec!["score".to_string()]);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    /// The partial plan is the only way to get a pushed filter when the split
    /// is incomplete, and it always carries the residual with it.
    #[test]
    fn a_partial_plan_cannot_be_obtained_without_its_residual() {
        let filter = FilterIR {
            clauses: vec![tenant_clause("acme"), range_clause()],
        };
        let plan = plan_retrieval(
            &filter,
            &EqualityAndSetMembership,
            ResidualStrategy::Overfetch,
            10,
            0.5,
            10_000,
        )
        .unwrap();

        match plan {
            RetrievalPlan::Partial {
                pushed,
                residual,
                unsupported,
                budget,
            } => {
                assert_eq!(pushed.clauses.len(), 1);
                assert_eq!(residual.clauses.len(), 1, "the residual came back empty");
                assert_eq!(unsupported.len(), 1);
                // k / 0.5 * 2 = 40
                assert_eq!(budget.initial, 40);
            }
            other => panic!("expected a partial plan, got {other:?}"),
        }
    }

    #[test]
    fn a_fully_pushed_plan_does_not_overfetch() {
        let filter = FilterIR {
            clauses: vec![tenant_clause("acme")],
        };
        match plan_retrieval(
            &filter,
            &EqualityAndSetMembership,
            ResidualStrategy::Overfetch,
            10,
            0.01,
            10_000,
        )
        .unwrap()
        {
            RetrievalPlan::FullyPushed { budget, .. } => {
                assert_eq!(
                    budget.initial, 10,
                    "a fully pushed filter needs no extra candidates"
                );
            }
            other => panic!("expected a fully pushed plan, got {other:?}"),
        }
    }

    #[test]
    fn an_unsatisfiable_filter_is_reported_rather_than_planned() {
        assert_eq!(
            plan_retrieval(
                &FilterIR::none(),
                &EqualityAndSetMembership,
                ResidualStrategy::Overfetch,
                10,
                0.5,
                10_000,
            ),
            Err(GovernanceError::Unsatisfiable)
        );
    }

    /// The budget formula, including the two clamps that keep it finite.
    #[test]
    fn the_initial_budget_follows_the_formula_and_stays_bounded() {
        // ceil(10 / 0.5) * 2
        assert_eq!(initial_candidates(10, 0.5, 2.0, 100_000), 40);
        // ceil(5 / 0.25) * 2
        assert_eq!(initial_candidates(5, 0.25, 2.0, 100_000), 40);
        // A selectivity of zero clamps to MIN_SELECTIVITY instead of diverging.
        assert_eq!(initial_candidates(1, 0.0, 1.0, 100_000), 10_000);
        // ...and the maximum still wins.
        assert_eq!(initial_candidates(1, 0.0, 1.0, 500), 500);
        // Never fewer than k, even when the estimate says everything matches.
        assert_eq!(initial_candidates(20, 1.0, 1.0, 100_000), 20);
        // A max below k cannot make the search unable to return k.
        assert_eq!(initial_candidates(20, 1.0, 1.0, 5), 20);
        // Absurd inputs saturate rather than wrapping to something small.
        assert!(initial_candidates(usize::MAX, 1e-300, 1e300, usize::MAX) > 0);
    }

    /// A short result set caused by an exhausted budget must be distinguishable
    /// from one caused by there being nothing more. Reporting the first as
    /// complete turns a resource limit into a claim about the data.
    #[test]
    fn an_exhausted_budget_is_not_reported_as_a_complete_answer() {
        let budget = Budget {
            initial: 10,
            max: 40,
            growth: 2.0,
        };
        // Every candidate is rejected, and the generator never runs dry.
        let (results, evidence) = escalating_search(
            &budget,
            5,
            0.5,
            vec![],
            |n| (0..n).collect::<Vec<usize>>(),
            |_| false,
        );

        assert!(results.is_empty());
        assert_eq!(evidence.exhaustion, Exhaustion::BudgetExhausted);
        assert!(
            !evidence.is_complete(),
            "a budget-limited empty result claimed to be complete"
        );
        assert!(evidence.rounds > 1, "the budget never escalated");
        assert_eq!(evidence.candidates_examined, 40);
        assert_eq!(evidence.observed_selectivity, Some(0.0));
    }

    /// The opposite case: the index really did run out, so the short answer is
    /// the whole answer.
    #[test]
    fn an_exhausted_index_is_a_complete_answer() {
        let budget = Budget {
            initial: 10,
            max: 1_000,
            growth: 2.0,
        };
        let (results, evidence) = escalating_search(
            &budget,
            5,
            0.5,
            vec![],
            // Only three rows exist.
            |n| (0..n.min(3)).collect::<Vec<usize>>(),
            |_| true,
        );
        assert_eq!(results.len(), 3);
        assert_eq!(evidence.exhaustion, Exhaustion::IndexExhausted);
        assert!(evidence.is_complete());
    }

    /// Escalation must actually help: a filter far more selective than
    /// estimated should still reach k by widening.
    #[test]
    fn escalation_recovers_from_an_optimistic_selectivity_estimate() {
        let budget = Budget {
            initial: 10,
            max: 10_000,
            growth: 4.0,
        };
        // One row in fifty is authorised, but the estimate said one in two.
        let (results, evidence) = escalating_search(
            &budget,
            5,
            0.5,
            vec![],
            |n| (0..n).collect::<Vec<usize>>(),
            |c| c % 50 == 0,
        );
        assert_eq!(results.len(), 5);
        assert_eq!(evidence.exhaustion, Exhaustion::Satisfied);
        assert!(evidence.rounds > 1);
        let observed = evidence.observed_selectivity.unwrap();
        assert!(
            observed < evidence.estimated_selectivity,
            "observed {observed} should be below the estimate"
        );
    }

    /// Authorisation is applied to every candidate in every round. A candidate
    /// the caller rejected must never appear in the output.
    #[test]
    fn no_rejected_candidate_reaches_the_caller() {
        let budget = Budget {
            initial: 100,
            max: 100,
            growth: 2.0,
        };
        let (results, evidence) = escalating_search(
            &budget,
            10,
            0.5,
            vec![],
            |n| (0..n).collect::<Vec<usize>>(),
            |c| c % 2 == 0,
        );
        assert_eq!(results.len(), 10);
        assert!(
            results.iter().all(|c| c % 2 == 0),
            "a rejected candidate was returned"
        );
        assert_eq!(evidence.rejected, 50);
        assert_eq!(evidence.observed_selectivity, Some(0.5));
    }

    #[test]
    fn a_zero_k_request_is_refused() {
        assert_eq!(
            plan_retrieval(
                &FilterIR::all(),
                &EqualityAndSetMembership,
                ResidualStrategy::Overfetch,
                0,
                0.5,
                100,
            ),
            Err(GovernanceError::EmptyRequest)
        );
    }

    /// A row missing the field a policy constrains must not pass. If an absent
    /// attribute matched, an unlabelled row would be visible to every tenant.
    #[test]
    fn a_row_without_the_governed_field_never_passes() {
        let filter = FilterIR {
            clauses: vec![tenant_clause("acme")],
        };
        let unlabelled = lookup_from(&[("other", "x")]);
        assert!(!matches(&filter, &unlabelled));

        // Including for negation, where SQL's three-valued logic would give a
        // different and much more dangerous answer.
        let ne = FilterIR {
            clauses: vec![Disjunction::single(FilterAtom::Ne {
                field: "tenant".to_string(),
                value: FilterValue::String("evil".to_string()),
            })],
        };
        assert!(
            !matches(&ne, &unlabelled),
            "a row with no tenant satisfied a tenant predicate"
        );
    }

    /// An empty disjunction is an empty OR and therefore false. Treating it as
    /// true would turn a malformed policy into no policy.
    #[test]
    fn an_empty_clause_matches_nothing() {
        let filter = FilterIR {
            clauses: vec![Disjunction::new(vec![])],
        };
        assert!(!matches(&filter, lookup_from(&[("tenant", "acme")])));
    }

    #[test]
    fn an_empty_filter_matches_everything() {
        assert!(matches(&FilterIR::all(), lookup_from(&[])));
    }

    /// A value that cannot be compared to the bound is outside the range. If
    /// an incomparable value were treated as in-range, a mistyped field would
    /// bypass the bound entirely.
    #[test]
    fn an_incomparable_value_is_not_inside_a_range() {
        let filter = FilterIR {
            clauses: vec![Disjunction::single(FilterAtom::range(
                "score",
                Some(FilterValue::Float64(0.5)),
                Some(FilterValue::Float64(1.0)),
            ))],
        };
        assert!(!matches(&filter, lookup_from(&[("score", "not-a-number")])));
    }
}
