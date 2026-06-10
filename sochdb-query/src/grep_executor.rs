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

//! Grep Lane Executor (Task 5)
//!
//! Exact regex / substring search as a first-class retrieval lane, built on the
//! trigram candidate index. The pipeline is:
//!
//! ```text
//! regex ──► required-literal extraction ──► trigram conjunction
//!       ──► trigram posting intersection (candidate DocIds)
//!       ──► ∩ AllowedSet            (filter pushdown BEFORE verification)
//!       ──► regex verification      (linear-time, finite-automaton engine)
//!       ──► ranked hits  OR  candidate gate
//! ```
//!
//! ## Correctness over speed
//!
//! Trigram pre-filtering is only ever used when the executor can *prove* the
//! extracted literals are mandatory (present in every possible match). For any
//! pattern it cannot prove this for — alternation, groups, character classes,
//! or no literal run of length ≥ 3 — it falls back to an explicit, bounded
//! full scan rather than risk a false negative. The full-scan path is capped by
//! `max_scan`; exceeding the cap is reported as
//! [`GrepError::DegeneratePattern`] instead of silently returning partial
//! results.
//!
//! ## Verification engine
//!
//! Verification uses the `regex` crate, a finite-automaton engine with
//! guaranteed linear-time matching, so adversarial patterns cannot turn the
//! verify stage into a catastrophic-backtracking DoS.
//!
//! ## Two fusion modes
//!
//! Grep produces a *set*, but RRF consumes *ranked lists*. Both shapes are
//! supported:
//! - [`GrepMode::Rank`] scores each hit by specificity-weighted, TF-saturated,
//!   length-pivoted relevance (BM25-flavored over the pattern's literal terms)
//!   so it can plug into RRF as a third ranked lane **without** the
//!   short-document / common-term bias of raw match density.
//! - [`GrepMode::Gate`] returns the matching documents as an
//!   [`AllowedSet`] to intersect into the other lanes (the
//!   "find the function that contains X" cascade), via [`GrepResults::into_allowed_set`].

use regex::Regex;

use crate::candidate_gate::AllowedSet;
use crate::trigram_index::{DocId, Trigram, TrigramIndex, trigrams_of};

/// Default cap on documents verified by a degenerate (no-trigram) full scan.
pub const DEFAULT_MAX_SCAN: usize = 100_000;

/// BM25-style term-frequency saturation constant for grep `Rank` scoring.
/// Bounds the marginal value of additional matches of the same term.
const GREP_K1: f32 = 1.2;

/// BM25-style length-normalization (pivot) constant for grep `Rank` scoring.
/// `0.0` disables length normalization; `1.0` applies it fully.
const GREP_B: f32 = 0.75;

/// How grep results should be consumed by the fusion layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrepMode {
    /// Produce a ranked list (for RRF as a third lane).
    Rank,
    /// Produce a candidate gate (intersect into the other lanes).
    Gate,
}

/// A single grep match.
#[derive(Debug, Clone, PartialEq)]
pub struct GrepHit {
    /// Matching document id.
    pub doc_id: DocId,
    /// Rank score (higher is better): specificity-weighted, TF-saturated,
    /// length-pivoted relevance over the pattern's literal terms.
    pub score: f32,
    /// Number of (non-overlapping) matches in the document.
    pub match_count: usize,
}

/// The outcome of a grep search.
#[derive(Debug, Clone)]
pub struct GrepResults {
    /// Ranked hits, best first.
    pub hits: Vec<GrepHit>,
    /// Whether the trigram index was used (`true`) or a full scan ran (`false`).
    pub used_index: bool,
}

impl GrepResults {
    /// The matching document ids as an [`AllowedSet`] for gate / cascade fusion.
    pub fn into_allowed_set(self) -> AllowedSet {
        AllowedSet::from_iter(self.hits.into_iter().map(|h| h.doc_id))
    }
}

/// Errors the grep lane can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrepError {
    /// The pattern is not a valid regular expression.
    InvalidRegex(String),
    /// The pattern yields no usable trigram and the corpus exceeds the scan
    /// budget, so it is rejected rather than scanned partially.
    DegeneratePattern { corpus: usize, max_scan: usize },
}

impl std::fmt::Display for GrepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrepError::InvalidRegex(e) => write!(f, "invalid regex: {e}"),
            GrepError::DegeneratePattern { corpus, max_scan } => write!(
                f,
                "degenerate pattern (no indexable literal) over a corpus of {corpus} documents \
                 exceeds the scan budget of {max_scan}"
            ),
        }
    }
}

impl std::error::Error for GrepError {}

/// The grep executor: plans and runs regex search over a [`TrigramIndex`].
pub struct GrepExecutor<'a> {
    index: &'a TrigramIndex,
    max_scan: usize,
}

impl<'a> GrepExecutor<'a> {
    /// Create an executor over `index` with the default scan budget.
    pub fn new(index: &'a TrigramIndex) -> Self {
        Self {
            index,
            max_scan: DEFAULT_MAX_SCAN,
        }
    }

    /// Override the full-scan document budget for degenerate patterns.
    pub fn with_max_scan(mut self, max_scan: usize) -> Self {
        self.max_scan = max_scan;
        self
    }

    /// Run a grep search.
    ///
    /// `allowed` is applied as a pushdown filter **before** regex verification,
    /// preserving the same `result ⊆ allowed` invariant the other lanes honor.
    /// `limit` caps the number of returned hits (0 = unlimited).
    pub fn search(
        &self,
        pattern: &str,
        allowed: &AllowedSet,
        limit: usize,
        mode: GrepMode,
    ) -> Result<GrepResults, GrepError> {
        let re = Regex::new(pattern).map_err(|e| GrepError::InvalidRegex(e.to_string()))?;

        if allowed.is_empty() {
            return Ok(GrepResults {
                hits: Vec::new(),
                used_index: false,
            });
        }

        // ---- Plan: candidate document set (safe superset, no false negatives) ----
        //
        // While planning we also capture each term's document-frequency estimate
        // (its trigram-candidate count) so the ranking stage can compute IDF
        // without a second pass over the postings.
        //
        // A leading whole-pattern inline-flag group like `(?i)` is stripped
        // *for literal extraction only* (the compiled `re` above keeps the
        // flag), so a case-insensitive alternation still drives the index and
        // IDF instead of degrading to a full scan.
        let extract = strip_leading_inline_flags(pattern);
        let (terms, is_alternation) = literal_terms(extract);
        let mut term_df: Vec<(String, usize)> = Vec::new();
        let (candidates, used_index): (Vec<DocId>, bool) = if terms.is_empty() {
            // No provably-mandatory literal (complex regex): bounded full scan.
            if self.index.len() > self.max_scan {
                return Err(GrepError::DegeneratePattern {
                    corpus: self.index.len(),
                    max_scan: self.max_scan,
                });
            }
            (self.index.documents().map(|(id, _)| id).collect(), false)
        } else if is_alternation {
            // Alternation `a|b|c`: a match contains *some* branch, so the
            // candidate set is the UNION of each branch's trigram candidates
            // (Cox AND-of-ORs, union form). Every branch is trigram-indexable
            // here (guaranteed by `literal_alternation`), so this stays a safe
            // superset and the previously full-scanned `|` patterns now use the
            // index. Each branch's candidate count doubles as its df estimate.
            let mut union: Vec<DocId> = Vec::new();
            for term in &terms {
                let branch = self.index.candidates(&trigrams_of(term));
                term_df.push((term.to_lowercase(), branch.len().max(1)));
                union.extend(branch);
            }
            union.sort_unstable();
            union.dedup();
            (union, true)
        } else {
            // Conjunction of mandatory literals: AND of all their trigrams.
            let mut trigrams: Vec<Trigram> = Vec::new();
            for term in &terms {
                let df = self.index.candidates(&trigrams_of(term)).len().max(1);
                term_df.push((term.to_lowercase(), df));
                trigrams.extend(trigrams_of(term));
            }
            trigrams.sort_unstable();
            trigrams.dedup();
            (self.index.candidates(&trigrams), true)
        };

        // ---- Gate mode: membership only, no ranking ----
        if mode == GrepMode::Gate {
            let mut hits: Vec<GrepHit> = Vec::new();
            for doc_id in candidates {
                if !allowed.contains(doc_id) {
                    continue;
                }
                if let Some(text) = self.index.doc_text(doc_id) {
                    if re.is_match(text) {
                        hits.push(GrepHit {
                            doc_id,
                            score: 1.0,
                            match_count: 1,
                        });
                    }
                }
            }
            hits.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
            if limit > 0 && hits.len() > limit {
                hits.truncate(limit);
            }
            return Ok(GrepResults { hits, used_index });
        }

        // ---- Rank mode: specificity-weighted, TF-saturated, length-pivoted ----
        //
        // The old score was raw match density (`matches / doc_len`), which is
        // IDF-blind (a hit on a common word counts as much as a rare one),
        // linear in raw match count (50 hits == 50x one hit), and explodes for
        // short documents — so it injected noise into RRF. The corrected score
        // is BM25-flavored over the grep's literal terms:
        //
        //   idf(t)   = ln(1 + (N - df + 0.5)/(df + 0.5))         // term rarity
        //   tf_sat   = c / (c + k1)                              // saturating TF
        //   raw(d)   = SUM_t idf(t) * tf_sat(count_t(d))
        //   score(d) = raw(d) / (1 - b + b*len_d/avg_len)        // pivoted length
        //
        // `df` is estimated index-locally as the trigram-candidate count of the
        // term (a tight upper bound on its true document frequency), captured
        // during planning above, so no extra corpus statistics are needed.
        // Verification still uses the full regex, so the hit *set* is unchanged
        // — only the ranking improves.
        let n = self.index.len().max(1) as f32;
        let term_idf: Vec<(String, f32)> = term_df
            .iter()
            .map(|(t, df)| {
                let dff = *df as f32;
                let idf = (1.0 + (n - dff + 0.5) / (dff + 0.5)).ln();
                (t.clone(), idf.max(0.0))
            })
            .collect();

        struct Pending {
            doc_id: DocId,
            len: f32,
            raw: f32,
            match_count: usize,
        }
        let mut pending: Vec<Pending> = Vec::new();
        let mut total_len = 0.0f32;
        // Reused per-term match-count buffer (alternation path) to avoid a
        // per-document allocation.
        let mut counts: Vec<u32> = vec![0; term_idf.len()];
        for doc_id in candidates {
            if !allowed.contains(doc_id) {
                continue;
            }
            let Some(text) = self.index.doc_text(doc_id) else {
                continue;
            };

            // Single regex pass over the document. For an alternation each match
            // is exactly one branch literal, so we attribute it to its term in
            // the SAME pass — no extra per-term substring scans, no allocation.
            let mut match_count = 0usize;
            if is_alternation {
                for c in counts.iter_mut() {
                    *c = 0;
                }
                for m in re.find_iter(text) {
                    match_count += 1;
                    let ms = m.as_str();
                    for (i, (term_lc, _)) in term_idf.iter().enumerate() {
                        if eq_ci_ascii(ms, term_lc) {
                            counts[i] += 1;
                            break;
                        }
                    }
                }
            } else {
                match_count = re.find_iter(text).count();
            }
            if match_count == 0 {
                continue;
            }

            let len = text.chars().count().max(1) as f32;
            let raw = if term_idf.is_empty() {
                // Complex pattern with no literal terms to weight: saturate the
                // raw regex match count so a flood of matches can't dominate.
                tf_saturate(match_count as f32)
            } else if is_alternation {
                // Per-branch counts already attributed in the single pass above.
                let mut s = 0.0f32;
                for (i, (_, idf)) in term_idf.iter().enumerate() {
                    if counts[i] > 0 {
                        s += idf * tf_saturate(counts[i] as f32);
                    }
                }
                s
            } else {
                // Conjunction / complex literal terms (rare): the whole-pattern
                // matches can't be attributed per term, so scan each mandatory
                // term once (allocation-free, ASCII case-insensitive).
                let mut s = 0.0f32;
                for (term_lc, idf) in &term_idf {
                    let c = count_ci_ascii(text, term_lc);
                    if c > 0 {
                        s += idf * tf_saturate(c as f32);
                    }
                }
                s
            };
            total_len += len;
            pending.push(Pending {
                doc_id,
                len,
                raw,
                match_count,
            });
        }

        let avg_len = if pending.is_empty() {
            1.0
        } else {
            (total_len / pending.len() as f32).max(1.0)
        };

        let mut hits: Vec<GrepHit> = pending
            .into_iter()
            .map(|p| {
                let norm = 1.0 - GREP_B + GREP_B * (p.len / avg_len);
                GrepHit {
                    doc_id: p.doc_id,
                    score: if norm > 0.0 { p.raw / norm } else { p.raw },
                    match_count: p.match_count,
                }
            })
            .collect();

        // Rank: relevance descending, doc_id ascending as a stable tiebreak.
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        if limit > 0 && hits.len() > limit {
            hits.truncate(limit);
        }

        Ok(GrepResults { hits, used_index })
    }
}

/// BM25-style saturating term frequency: `count / (count + k1)`, in `[0, 1)`.
fn tf_saturate(count: f32) -> f32 {
    count / (count + GREP_K1)
}

/// Count non-overlapping, ASCII case-insensitive occurrences of `needle`
/// (already lowercased) in `hay`, without allocating a lowercased copy.
///
/// Non-ASCII bytes are compared as-is (no Unicode case folding); since this
/// only feeds the *ranking* signal of documents the full regex already
/// verified, that approximation never affects correctness.
fn count_ci_ascii(hay: &str, needle: &str) -> usize {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return 0;
    }
    let last = h.len() - n.len();
    let mut count = 0;
    let mut i = 0;
    while i <= last {
        let mut k = 0;
        while k < n.len() && h[i + k].to_ascii_lowercase() == n[k] {
            k += 1;
        }
        if k == n.len() {
            count += 1;
            i += n.len(); // non-overlapping
        } else {
            i += 1;
        }
    }
    count
}

/// ASCII case-insensitive equality. `b` is assumed already lowercased.
fn eq_ci_ascii(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .all(|(x, y)| x.to_ascii_lowercase() == y)
}

/// Strip a leading whole-pattern inline-flag group (e.g. `(?i)`, `(?ims)`,
/// `(?i-u)`) so the remainder can be parsed for mandatory literals. Only a pure
/// flag setter — alphabetic flags plus an optional `-` toggle, immediately
/// closed by `)` with no `:` scoping — is stripped; scoped groups like
/// `(?i:...)` are left intact (returns the original pattern). The compiled
/// regex still carries the flag, so this only affects literal extraction,
/// never matching semantics.
fn strip_leading_inline_flags(pattern: &str) -> &str {
    if let Some(rest) = pattern.strip_prefix("(?") {
        if let Some(close) = rest.find(')') {
            let flags = &rest[..close];
            if !flags.is_empty() && flags.bytes().all(|b| b.is_ascii_alphabetic() || b == b'-') {
                return &rest[close + 1..];
            }
        }
    }
    pattern
}

/// Literal terms used for BOTH trigram planning and specificity scoring,
/// together with a flag indicating whether they came from a top-level
/// alternation (union plan) versus a conjunction (AND plan).
///
/// - Top-level literal alternation `a|b|c` → `(vec!["a","b","c"], true)`.
/// - Mandatory-literal conjunction (e.g. `parse.*query`) → `(runs, false)`.
/// - Anything else (char classes, groups, no ≥3 literal run) → `(vec![], false)`
///   so the caller falls back to a bounded full scan.
fn literal_terms(pattern: &str) -> (Vec<String>, bool) {
    if let Some(branches) = literal_alternation(pattern) {
        (branches, true)
    } else if let Some(runs) = required_literals(pattern) {
        (runs, false)
    } else {
        (Vec::new(), false)
    }
}

/// If `pattern` is a top-level alternation of plain literals — every `|` is at
/// the top level (no grouping/classes) and each branch reduces to a single
/// mandatory literal of length ≥ 3 — return the per-branch literals. Otherwise
/// `None`.
///
/// This is conservative: a branch that is too short or contains a wildcard
/// (multiple runs) disqualifies the whole alternation, so the union plan it
/// drives is always a safe trigram superset of the regex's true matches.
fn literal_alternation(pattern: &str) -> Option<Vec<String>> {
    if !pattern.contains('|') {
        return None;
    }
    // Any grouping/class could scope a `|`, so only treat `|` as top-level when
    // none are present.
    if pattern.contains(['(', ')', '[', ']', '{', '}']) {
        return None;
    }
    let mut branches: Vec<String> = Vec::new();
    for raw in pattern.split('|') {
        let lits = required_literals(raw)?;
        // A clean term branch is exactly one mandatory literal run.
        if lits.len() != 1 {
            return None;
        }
        branches.push(lits.into_iter().next().unwrap());
    }
    if branches.is_empty() {
        None
    } else {
        Some(branches)
    }
}

/// Extract the mandatory trigram conjunction for `pattern`, or `None` if the
/// pattern is too complex to prove a mandatory literal (caller must full-scan).
///
/// Safety contract: a returned trigram set is **required** — every document
/// matching `pattern` contains all of them — so intersecting their postings can
/// never drop a true match. When that cannot be proven, this returns `None`.
pub fn required_trigrams(pattern: &str) -> Option<Vec<Trigram>> {
    let literals = required_literals(pattern)?;
    let mut trigrams: Vec<Trigram> = Vec::new();
    for lit in &literals {
        trigrams.extend(trigrams_of(lit));
    }
    if trigrams.is_empty() {
        return None;
    }
    trigrams.sort_unstable();
    trigrams.dedup();
    Some(trigrams)
}

/// Extract literal runs that must appear in every match of `pattern`.
///
/// Conservative by design: it bails out (returns `None`) on any construct that
/// can make a literal optional or contextual — alternation `|`, groups `( )`,
/// character classes `[ ]`, counted repetition `{ }` — so the only literals it
/// ever reports are unconditionally mandatory. `*` and `?` make the *preceding*
/// character optional, so that character is trimmed from its run; `+` keeps it
/// (one-or-more still requires one). Only runs of length ≥ 3 (trigram-indexable)
/// are returned.
fn required_literals(pattern: &str) -> Option<Vec<String>> {
    let mut runs: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = pattern.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            // Constructs that defeat "mandatory literal" reasoning → full scan.
            '|' | '(' | ')' | '[' | ']' | '{' | '}' => return None,
            '\\' => match chars.next() {
                // Escaped ASCII-alnum is a class (\d, \w, \s, \b, ...): a separator.
                Some(n) if n.is_ascii_alphanumeric() => flush(&mut cur, &mut runs),
                // Escaped punctuation is a literal character (\., \+, \\, ...).
                Some(n) => cur.push(n),
                None => {}
            },
            // `*` / `?`: the preceding char becomes optional → drop it.
            '*' | '?' => {
                cur.pop();
                flush(&mut cur, &mut runs);
            }
            // Wildcard / anchors / `+` end the current literal run but keep it.
            '.' | '^' | '$' | '+' => flush(&mut cur, &mut runs),
            _ => cur.push(c),
        }
    }
    flush(&mut cur, &mut runs);

    let mandatory: Vec<String> = runs
        .into_iter()
        .filter(|r| r.chars().count() >= 3)
        .collect();
    if mandatory.is_empty() {
        None
    } else {
        Some(mandatory)
    }
}

/// Move a completed literal run into `runs` if non-empty.
fn flush(cur: &mut String, runs: &mut Vec<String>) {
    if !cur.is_empty() {
        runs.push(std::mem::take(cur));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index() -> TrigramIndex {
        let mut idx = TrigramIndex::new();
        idx.insert(1, "fn parse_query(input: &str) -> Query");
        idx.insert(2, "let parser = build();");
        idx.insert(3, "// completely unrelated comment");
        idx.insert(4, "error: connection timeout occurred");
        idx.insert(5, "PARSE_MODE constant");
        idx
    }

    #[test]
    fn test_required_literals_extraction() {
        // Pure literal → mandatory.
        assert_eq!(required_literals("parse"), Some(vec!["parse".to_string()]));
        // Wildcard splits into two mandatory runs.
        assert_eq!(
            required_literals("parse.*query"),
            Some(vec!["parse".to_string(), "query".to_string()])
        );
        // Escaped dot is a literal, so the whole thing is one contiguous literal.
        assert_eq!(
            required_literals(r"config\.toml"),
            Some(vec!["config.toml".to_string()])
        );
        // `?` drops the optional preceding char: "color"/"colour".
        assert_eq!(required_literals("colou?r"), Some(vec!["colo".to_string()]));
        // Alternation / groups / classes → cannot prove a mandatory literal.
        assert_eq!(required_literals("cat|dog"), None);
        assert_eq!(required_literals("(foo)bar"), None);
        assert_eq!(required_literals("a[bc]def"), None);
        // No literal run of length ≥ 3.
        assert_eq!(required_literals("a.b"), None);
    }

    #[test]
    fn test_grep_substring_uses_index() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        let res = exec
            .search("parse", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        assert!(res.used_index, "a pure literal must use the trigram index");
        let ids: Vec<DocId> = res.hits.iter().map(|h| h.doc_id).collect();
        // Docs 1 (parse_query) and 2 (parser) contain the lowercase substring
        // "parse"; doc 5 is PARSE (uppercase) and must NOT match a
        // case-sensitive search; doc 3 is unrelated.
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&5));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn test_grep_case_insensitive_pattern() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        // (?i) makes verification case-insensitive; the trigram pre-filter is a
        // safe superset, so doc 5 (PARSE) must now appear.
        let res = exec
            .search("(?i)parse", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        let ids: Vec<DocId> = res.hits.iter().map(|h| h.doc_id).collect();
        assert!(ids.contains(&5));
    }

    #[test]
    fn test_grep_regex_with_wildcard() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        // Both "parse" and "query" are mandatory; only doc 1 has both.
        let res = exec
            .search("parse.*query", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        assert!(res.used_index);
        let ids: Vec<DocId> = res.hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_allowed_set_pushdown() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        // Restrict to docs {2} — even though doc 1 also matches "parse", the
        // gate must exclude it: result ⊆ allowed.
        let allowed = AllowedSet::from_iter([2u64]);
        let res = exec.search("parse", &allowed, 0, GrepMode::Rank).unwrap();
        let ids: Vec<DocId> = res.hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn test_gate_mode_to_allowed_set() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        let res = exec
            .search("parse", &AllowedSet::All, 0, GrepMode::Gate)
            .unwrap();
        let gate = res.into_allowed_set();
        assert!(gate.contains(1));
        assert!(gate.contains(2));
        assert!(!gate.contains(3));
    }

    #[test]
    fn test_invalid_regex_errors() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        let err = exec
            .search("(unclosed", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap_err();
        assert!(matches!(err, GrepError::InvalidRegex(_)));
    }

    #[test]
    fn test_degenerate_pattern_rejected_over_budget() {
        let idx = build_index();
        // Budget of 1, corpus of 5, pattern "a." has no indexable trigram.
        let exec = GrepExecutor::new(&idx).with_max_scan(1);
        let err = exec
            .search("a.", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap_err();
        assert!(matches!(err, GrepError::DegeneratePattern { .. }));
    }

    #[test]
    fn test_degenerate_pattern_scans_within_budget() {
        let idx = build_index();
        // Same degenerate pattern, but the budget covers the corpus → full scan.
        let exec = GrepExecutor::new(&idx).with_max_scan(1000);
        let res = exec
            .search("er.", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        assert!(!res.used_index, "degenerate pattern must full-scan");
        // "er" followed by any char appears in "parser"/"error"/... — at least
        // one hit, proving the scan path actually verifies.
        assert!(!res.hits.is_empty());
    }

    // ---- Alternation planning (Cox AND-of-ORs, union form) ----

    #[test]
    fn test_literal_alternation_extraction() {
        // Clean top-level literal alternation.
        assert_eq!(
            literal_alternation("parse|timeout"),
            Some(vec!["parse".to_string(), "timeout".to_string()])
        );
        // Not an alternation.
        assert_eq!(literal_alternation("parse"), None);
        // Grouping could scope the `|` → not provably top-level.
        assert_eq!(literal_alternation("(parse|query)x"), None);
        // A branch shorter than a trigram disqualifies the whole alternation.
        assert_eq!(literal_alternation("parse|ab"), None);
        // A branch with a wildcard is multiple runs → disqualified.
        assert_eq!(literal_alternation("parse|foo.*bar"), None);
    }

    #[test]
    fn test_strip_leading_inline_flags() {
        // Whole-pattern flag setters are stripped for literal extraction.
        assert_eq!(
            strip_leading_inline_flags("(?i)parse|timeout"),
            "parse|timeout"
        );
        assert_eq!(strip_leading_inline_flags("(?ims)parse"), "parse");
        // Disable-toggle flags (ASCII-only case folding) are also stripped.
        assert_eq!(strip_leading_inline_flags("(?i-u)parse|x"), "parse|x");
        // Scoped groups must be left intact (they constrain `|` scope).
        assert_eq!(strip_leading_inline_flags("(?i:parse|x)y"), "(?i:parse|x)y");
        // No flag group → returned unchanged.
        assert_eq!(strip_leading_inline_flags("parse|timeout"), "parse|timeout");
        assert_eq!(strip_leading_inline_flags("(parse)"), "(parse)");
    }

    #[test]
    fn test_case_insensitive_alternation_uses_index() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        // `(?i)` must still drive the trigram index + union, and now match the
        // uppercase PARSE in doc 5 that the case-sensitive variant skipped.
        let res = exec
            .search("(?i)parse|timeout", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        assert!(
            res.used_index,
            "flagged alternation must still use the index"
        );
        let ids: Vec<DocId> = res.hits.iter().map(|h| h.doc_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&4));
        assert!(
            ids.contains(&5),
            "case-insensitive match must include PARSE"
        );
    }

    #[test]
    fn test_alternation_uses_index_and_unions_branches() {
        let idx = build_index();
        let exec = GrepExecutor::new(&idx);
        // `parse|timeout` previously full-scanned (required_literals bailed on
        // `|`); now it must use the trigram index and union both branches.
        let res = exec
            .search("parse|timeout", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        assert!(res.used_index, "literal alternation must use the index");
        let ids: Vec<DocId> = res.hits.iter().map(|h| h.doc_id).collect();
        // Docs 1 & 2 contain "parse" (lowercase); doc 4 contains "timeout".
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&4));
        // Doc 5 is uppercase PARSE → case-sensitive regex must not match it.
        assert!(!ids.contains(&5));
    }

    // ---- Ranking: specificity / saturation / length pivot ----

    #[test]
    fn test_rank_prefers_rarer_term_over_common_frequent_term() {
        // "alpha" is common (df = 8); "zeta" is rare (df = 1). A single hit on
        // the rare term must outrank four hits on the common term — the exact
        // pathology the old `matches / doc_len` density score got backwards.
        let mut idx = TrigramIndex::new();
        idx.insert(1, "alpha alpha alpha alpha");
        for i in 2..=8u64 {
            idx.insert(i, "alpha context");
        }
        idx.insert(9, "zeta marker present here");

        let exec = GrepExecutor::new(&idx);
        let res = exec
            .search("alpha|zeta", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        assert!(res.used_index);
        // The top-ranked hit is the rare-term document, not the match-stuffed
        // common-term one.
        assert_eq!(res.hits.first().map(|h| h.doc_id), Some(9));
        let score_rare = res.hits.iter().find(|h| h.doc_id == 9).unwrap().score;
        let score_common = res.hits.iter().find(|h| h.doc_id == 1).unwrap().score;
        assert!(
            score_rare > score_common,
            "rare-term doc {score_rare} must outrank frequent common-term doc {score_common}"
        );
    }

    #[test]
    fn test_rank_saturates_repeated_matches() {
        // Two docs of equal length hit the same (equally rare) term; one has
        // many more matches. With length held constant, TF saturation means the
        // high-count doc scores higher, but far less than linearly.
        let mut idx = TrigramIndex::new();
        // 1 match, padded to the same char length as doc 2 (47 chars).
        idx.insert(1, "zebra xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
        // 8 matches, 47 chars.
        idx.insert(2, "zebra zebra zebra zebra zebra zebra zebra zebra");
        let exec = GrepExecutor::new(&idx);
        let res = exec
            .search("zebra", &AllowedSet::All, 0, GrepMode::Rank)
            .unwrap();
        let s1 = res.hits.iter().find(|h| h.doc_id == 1).unwrap().score;
        let s2 = res.hits.iter().find(|h| h.doc_id == 2).unwrap().score;
        // 8x the matches must score higher, but nowhere near 8x (saturation).
        assert!(s2 > s1, "more matches should still score higher");
        assert!(
            s2 < 4.0 * s1,
            "saturation must keep 8x matches well under 8x score (got {s2} vs {s1})"
        );
    }
}
