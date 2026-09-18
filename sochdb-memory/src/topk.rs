//! Bounded top-k selection.
//!
//! Every retrieval lane in this crate ends the same way: score a set of
//! candidates and keep the best `k`. The obvious spelling — collect all `N`
//! scores, sort them, truncate to `k` — costs `O(N log N)` time and `O(N)`
//! memory to produce `k` answers, and `k` is typically a handful while `N` is
//! the whole namespace.
//!
//! A bounded max-heap of the `k` best seen so far costs `O(N log k)` and holds
//! `O(k)`. The scoring itself is untouched; this only removes the sort of the
//! results that were always going to be discarded.
//!
//! # Ordering
//!
//! Selection uses a *total* order — score descending, then id ascending — for a
//! reason beyond tidiness. Candidates arrive from a `HashMap`, whose iteration
//! order varies between runs, so a comparator that reports ties as equal lets
//! the same query over the same data return different documents on different
//! runs. Breaking ties on id is arbitrary but stable, which is the property
//! that matters.
//!
//! NaN sorts last. A NaN score means scoring failed, and the honest place for
//! such a candidate is behind every candidate that has a real score.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A scored candidate ordered so the *worst* compares greatest.
///
/// That inversion is what makes a [`BinaryHeap`] — a max-heap — usable as a
/// bounded "keep the best k": its root is the weakest retained candidate, so
/// admitting a better one is a `peek`, a `pop` and a `push`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Worst {
    score: f32,
    id: u64,
}

impl Eq for Worst {}

impl Ord for Worst {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.score.is_nan(), other.score.is_nan()) {
            (true, true) => self.id.cmp(&other.id),
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // Reversed: a lower score is a worse candidate, hence "greater".
            (false, false) => other
                .score
                .partial_cmp(&self.score)
                .unwrap_or(Ordering::Equal)
                // Not reversed: on equal scores the higher id is the worse one,
                // so the retained order is id-ascending.
                .then_with(|| self.id.cmp(&other.id)),
        }
    }
}

impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Keep the `k` highest-scoring candidates, best first.
///
/// Consumes the candidates lazily: nothing beyond the `k` retained entries is
/// held. Returns fewer than `k` items only when fewer were supplied.
pub(crate) fn top_k<I>(candidates: I, k: usize) -> Vec<(u64, f32)>
where
    I: IntoIterator<Item = (u64, f32)>,
{
    if k == 0 {
        return Vec::new();
    }

    let mut heap: BinaryHeap<Worst> = BinaryHeap::with_capacity(k);
    for (id, score) in candidates {
        let candidate = Worst { score, id };
        if heap.len() < k {
            heap.push(candidate);
            continue;
        }
        // `Greater` means worse, so `candidate < worst` means strictly better.
        // Ties never displace an incumbent: the total order already decides
        // which of two equal scores wins, and evicting on a tie would only
        // churn the heap.
        if heap.peek().is_some_and(|worst| candidate < *worst) {
            heap.pop();
            heap.push(candidate);
        }
    }

    // `into_sorted_vec` yields ascending in `Worst` order, i.e. best first.
    heap.into_sorted_vec()
        .into_iter()
        .map(|w| (w.id, w.score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_the_k_best_in_descending_score_order() {
        let got = top_k(vec![(1, 0.1), (2, 0.9), (3, 0.5), (4, 0.7)], 2);
        assert_eq!(got, vec![(2, 0.9), (4, 0.7)]);
    }

    #[test]
    fn matches_sort_then_truncate_on_random_input() {
        // The property that matters: the bounded heap is not an approximation.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let scored: Vec<(u64, f32)> = (0..500)
            .map(|id| (id, (next() % 1000) as f32 / 1000.0))
            .collect();

        let mut baseline = scored.clone();
        baseline.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        baseline.truncate(25);

        assert_eq!(top_k(scored, 25), baseline);
    }

    #[test]
    fn ties_break_on_id_so_results_are_reproducible() {
        let forward = top_k(vec![(7, 0.5), (3, 0.5), (9, 0.5)], 2);
        let reversed = top_k(vec![(9, 0.5), (3, 0.5), (7, 0.5)], 2);
        assert_eq!(forward, vec![(3, 0.5), (7, 0.5)]);
        assert_eq!(
            forward, reversed,
            "input order must not change which tied candidates survive"
        );
    }

    #[test]
    fn nan_scores_rank_behind_every_real_score() {
        let got = top_k(vec![(1, f32::NAN), (2, -1.0), (3, 0.0)], 3);
        assert_eq!(got[0], (3, 0.0));
        assert_eq!(got[1], (2, -1.0));
        assert_eq!(got[2].0, 1);
        assert!(got[2].1.is_nan());
    }

    #[test]
    fn nan_is_still_returned_when_it_is_all_there_is() {
        let got = top_k(vec![(1, f32::NAN), (2, f32::NAN)], 1);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 1);
    }

    #[test]
    fn k_larger_than_input_returns_everything_sorted() {
        assert_eq!(
            top_k(vec![(1, 0.2), (2, 0.4)], 10),
            vec![(2, 0.4), (1, 0.2)]
        );
    }

    #[test]
    fn k_zero_returns_nothing() {
        assert!(top_k(vec![(1, 1.0)], 0).is_empty());
    }

    #[test]
    fn empty_input_returns_nothing() {
        assert!(top_k(Vec::new(), 5).is_empty());
    }
}
