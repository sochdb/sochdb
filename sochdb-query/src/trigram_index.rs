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

//! Trigram Index for the Grep Lane (Task 5)
//!
//! A case-folded trigram inverted index that turns regex / substring search
//! from a full corpus scan into a sub-linear candidate lookup, following the
//! Cox / Google Code Search (and Zoekt / livegrep) design:
//!
//! ```text
//! regex ──► required-literal extraction ──► trigram conjunction
//!        ──► posting intersection (candidates) ──► regex verification
//! ```
//!
//! ## Why trigrams
//!
//! A contiguous literal of length `L ≥ 3` contains `L − 2` trigrams, and any
//! document matching that literal **must** contain every one of those
//! trigrams. Intersecting the trigram posting lists therefore yields a
//! candidate superset of the true matches with **no false negatives** — the
//! regex verification stage removes the false positives. Case-folding at index
//! time only ever *widens* the candidate set, so it remains a safe superset
//! regardless of whether the final regex is case-sensitive.
//!
//! ## Identity
//!
//! Postings are keyed on `u64` document ids — the same retrieval-universe key
//! space as [`crate::candidate_gate::AllowedSet`] and the canonical
//! [`crate::unified_fusion::DocId`] — so grep candidates intersect directly
//! with the filter gate and fuse directly with the other lanes.

use std::collections::HashMap;

/// Retrieval-universe document id (matches `AllowedSet` / fusion `DocId`).
pub type DocId = u64;

/// A case-folded trigram key.
pub type Trigram = (char, char, char);

/// Extract the ordered, de-duplicated set of case-folded trigrams in `text`.
///
/// Returns an empty vector for inputs shorter than three characters.
pub fn trigrams_of(text: &str) -> Vec<Trigram> {
    let chars: Vec<char> = text.chars().flat_map(|c| c.to_lowercase()).collect();
    if chars.len() < 3 {
        return Vec::new();
    }
    let mut out: Vec<Trigram> = chars.windows(3).map(|w| (w[0], w[1], w[2])).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// A case-folded trigram inverted index over a document corpus.
///
/// Stores the original document text alongside the postings so the grep
/// executor can verify candidates with the real (case-sensitive) regex.
#[derive(Default)]
pub struct TrigramIndex {
    /// Trigram → sorted, de-duplicated document ids containing it.
    postings: HashMap<Trigram, Vec<DocId>>,
    /// Document id → original text (verification source).
    docs: HashMap<DocId, String>,
}

impl TrigramIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Number of distinct trigrams currently indexed.
    pub fn vocab_size(&self) -> usize {
        self.postings.len()
    }

    /// Whether `doc_id` is present.
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.docs.contains_key(&doc_id)
    }

    /// The stored text for `doc_id`, if present.
    pub fn doc_text(&self, doc_id: DocId) -> Option<&str> {
        self.docs.get(&doc_id).map(|s| s.as_str())
    }

    /// Iterate over every indexed `(doc_id, text)` pair.
    pub fn documents(&self) -> impl Iterator<Item = (DocId, &str)> {
        self.docs.iter().map(|(id, t)| (*id, t.as_str()))
    }

    /// Insert (or replace) a document.
    ///
    /// Re-inserting an existing id first removes the stale postings so the
    /// index stays a pure function of the current document set.
    pub fn insert(&mut self, doc_id: DocId, text: &str) {
        if self.docs.contains_key(&doc_id) {
            self.remove(doc_id);
        }
        for tri in trigrams_of(text) {
            let postings = self.postings.entry(tri).or_default();
            if let Err(idx) = postings.binary_search(&doc_id) {
                postings.insert(idx, doc_id);
            }
        }
        self.docs.insert(doc_id, text.to_string());
    }

    /// Remove a document and all of its trigram postings.
    ///
    /// Posting lists that become empty are dropped entirely (no vocabulary
    /// leak), so `vocab_size` reflects only live trigrams.
    pub fn remove(&mut self, doc_id: DocId) -> bool {
        let Some(text) = self.docs.remove(&doc_id) else {
            return false;
        };
        for tri in trigrams_of(&text) {
            if let Some(postings) = self.postings.get_mut(&tri) {
                if let Ok(idx) = postings.binary_search(&doc_id) {
                    postings.remove(idx);
                }
                if postings.is_empty() {
                    self.postings.remove(&tri);
                }
            }
        }
        true
    }

    /// Candidate documents that contain **all** of `required` trigrams.
    ///
    /// Returns the sorted intersection of the posting lists. An empty
    /// `required` slice yields an empty result (the caller decides whether to
    /// full-scan in that case). If any required trigram is absent the result is
    /// empty (the conjunction cannot be satisfied).
    pub fn candidates(&self, required: &[Trigram]) -> Vec<DocId> {
        if required.is_empty() {
            return Vec::new();
        }

        // Gather posting lists; bail early if any required trigram is unseen.
        let mut lists: Vec<&Vec<DocId>> = Vec::with_capacity(required.len());
        for tri in required {
            match self.postings.get(tri) {
                Some(list) => lists.push(list),
                None => return Vec::new(),
            }
        }

        // Intersect smallest-first to minimise work.
        lists.sort_by_key(|l| l.len());
        let mut acc: Vec<DocId> = lists[0].clone();
        for list in &lists[1..] {
            acc = sorted_intersect(&acc, list);
            if acc.is_empty() {
                break;
            }
        }
        acc
    }
}

/// Intersect two ascending, de-duplicated id slices with a linear merge.
fn sorted_intersect(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigrams_basic() {
        assert_eq!(trigrams_of("ab"), Vec::<Trigram>::new());
        assert_eq!(trigrams_of("abc"), vec![('a', 'b', 'c')]);
        // Case folded + sorted + de-duplicated.
        assert_eq!(trigrams_of("AAA"), vec![('a', 'a', 'a')]);
    }

    #[test]
    fn test_insert_and_candidates() {
        let mut idx = TrigramIndex::new();
        idx.insert(1, "hello world");
        idx.insert(2, "help me");
        idx.insert(3, "world peace");

        // "hel" is in docs 1 and 2.
        let c = idx.candidates(&trigrams_of("hel"));
        assert_eq!(c, vec![1, 2]);

        // "world" is in docs 1 and 3.
        let c = idx.candidates(&trigrams_of("world"));
        assert_eq!(c, vec![1, 3]);

        // A trigram present in nobody yields no candidates.
        assert!(idx.candidates(&trigrams_of("xyz")).is_empty());
    }

    #[test]
    fn test_candidates_never_drop_a_true_match() {
        // The core safety property: every document that actually contains the
        // literal must appear in the candidate set.
        let mut idx = TrigramIndex::new();
        let corpus = [
            (1, "fn parse_query() {}"),
            (2, "let parser = build();"),
            (3, "totally unrelated text"),
            (4, "PARSE in caps"),
        ];
        for (id, t) in corpus {
            idx.insert(id, t);
        }
        let cands = idx.candidates(&trigrams_of("parse"));
        // Docs 1 and 4 (case-folded) contain "parse"; both must be candidates.
        assert!(cands.contains(&1));
        assert!(cands.contains(&4));
        assert!(!cands.contains(&3));
    }

    #[test]
    fn test_remove_is_clean() {
        let mut idx = TrigramIndex::new();
        idx.insert(1, "alpha beta");
        idx.insert(2, "alpha gamma");
        let vocab_before = idx.vocab_size();

        assert!(idx.remove(1));
        assert!(!idx.contains(1));
        assert_eq!(idx.len(), 1);

        // "beta" was unique to doc 1 → its trigrams are gone.
        assert!(idx.candidates(&trigrams_of("beta")).is_empty());
        // "alpha" survives in doc 2.
        assert_eq!(idx.candidates(&trigrams_of("alpha")), vec![2]);

        // Removing the last doc leaves a clean, empty index.
        assert!(idx.remove(2));
        assert!(idx.is_empty());
        assert_eq!(idx.vocab_size(), 0);
        assert!(vocab_before > 0);
    }

    #[test]
    fn test_reinsert_replaces() {
        let mut idx = TrigramIndex::new();
        idx.insert(1, "before");
        idx.insert(1, "after change");
        assert_eq!(idx.len(), 1);
        // Old trigrams gone, new ones present.
        assert!(idx.candidates(&trigrams_of("before")).is_empty());
        assert_eq!(idx.candidates(&trigrams_of("change")), vec![1]);
    }
}
