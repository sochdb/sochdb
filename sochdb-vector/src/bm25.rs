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

//! BM25 Scoring for Lexical Search (Task 4)
//!
//! This module implements BM25 (Best Matching 25) scoring for keyword search.
//! BM25 is the standard ranking function for lexical retrieval, balancing:
//! - Term frequency (TF): How often a term appears in a document
//! - Inverse document frequency (IDF): How rare a term is across all documents
//! - Document length normalization: Penalizing very long documents
//!
//! ## BM25 Formula
//!
//! ```text
//! score(q, d) = Σ IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * |d|/avgdl))
//! ```
//!
//! Where:
//! - `TF(t,d)` = term frequency of term t in document d
//! - `IDF(t)` = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
//! - `N` = total number of documents
//! - `df(t)` = number of documents containing term t
//! - `|d|` = length of document d
//! - `avgdl` = average document length
//! - `k1` = term frequency saturation parameter (typically 1.2)
//! - `b` = length normalization parameter (typically 0.75)

use std::collections::HashMap;

// ============================================================================
// BM25 Configuration
// ============================================================================

/// BM25 scoring parameters
#[derive(Debug, Clone, Copy)]
pub struct BM25Config {
    /// Term frequency saturation parameter (k1)
    /// Higher values give more weight to term frequency
    /// Typical range: 1.2 - 2.0
    pub k1: f32,

    /// Length normalization parameter (b)
    /// 0.0 = no length normalization
    /// 1.0 = full length normalization
    /// Typical value: 0.75
    pub b: f32,

    /// Minimum IDF to filter out very common terms
    pub min_idf: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_idf: 0.0,
        }
    }
}

impl BM25Config {
    /// Lucene-style BM25 parameters
    pub fn lucene() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_idf: 0.0,
        }
    }

    /// Elasticsearch-style parameters
    pub fn elasticsearch() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_idf: 0.0,
        }
    }

    /// Parameters optimized for short queries
    pub fn short_queries() -> Self {
        Self {
            k1: 1.5,
            b: 0.5, // Less length normalization
            min_idf: 0.0,
        }
    }
}

// ============================================================================
// BM25 Scorer
// ============================================================================

/// BM25 scorer for a document collection
pub struct BM25Scorer {
    /// Configuration
    config: BM25Config,

    /// Total number of documents
    num_docs: usize,

    /// Total token length across all documents. The average document length is
    /// derived from this on read so it can never drift out of sync.
    total_len: usize,

    /// Document frequency for each term
    doc_freqs: HashMap<String, usize>,
}

impl BM25Scorer {
    /// Create a new BM25 scorer
    pub fn new(config: BM25Config) -> Self {
        Self {
            config,
            num_docs: 0,
            total_len: 0,
            doc_freqs: HashMap::new(),
        }
    }

    /// Build the scorer from a collection of documents
    pub fn build<I, D, T>(documents: I, config: BM25Config) -> Self
    where
        I: IntoIterator<Item = D>,
        D: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let mut scorer = Self::new(config);
        let mut total_len = 0usize;
        let mut num_docs = 0usize;
        let mut doc_freqs: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            num_docs += 1;
            let mut seen_terms: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let mut doc_len = 0usize;

            for token in doc {
                let term = token.as_ref().to_lowercase();
                if !term.is_empty() {
                    seen_terms.insert(term);
                    doc_len += 1;
                }
            }

            total_len += doc_len;

            for term in seen_terms {
                *doc_freqs.entry(term).or_insert(0) += 1;
            }
        }

        scorer.num_docs = num_docs;
        scorer.total_len = total_len;
        scorer.doc_freqs = doc_freqs;

        scorer
    }

    /// Average document length, derived from running totals.
    ///
    /// Computed on read so it can never drift out of sync with the corpus
    /// (avgdl changes for every term on every insert and delete).
    #[inline]
    pub fn avg_doc_len(&self) -> f32 {
        if self.num_docs > 0 {
            self.total_len as f32 / self.num_docs as f32
        } else {
            0.0
        }
    }

    /// The scoring configuration this scorer was built with.
    #[inline]
    pub fn config(&self) -> BM25Config {
        self.config
    }

    /// Compute IDF from a document frequency and corpus size.
    ///
    /// Single source of truth for IDF, used by every scoring path (batch and
    /// incremental). The `+ 1` floor (Robertson-Sparck Jones with smoothing)
    /// keeps IDF strictly positive even for terms in more than half the corpus.
    #[inline]
    fn compute_idf(&self, df: usize, n: usize) -> f32 {
        let n = n as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Get IDF for a term.
    ///
    /// Computed lazily from the current `(df, N)` so it is always consistent
    /// with the live corpus; unknown terms use `df = 0` (maximum IDF). Terms
    /// whose IDF falls below `min_idf` contribute nothing.
    pub fn idf(&self, term: &str) -> f32 {
        let df = self
            .doc_freqs
            .get(&term.to_lowercase())
            .copied()
            .unwrap_or(0);
        let idf = self.compute_idf(df, self.num_docs);
        if idf < self.config.min_idf { 0.0 } else { idf }
    }

    /// Score a document for a query
    pub fn score<I, T>(&self, query_terms: I, doc_terms: &[T], doc_len: usize) -> f32
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str> + std::hash::Hash + Eq,
    {
        // Build term frequency map for document
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for term in doc_terms {
            *tf.entry(term.as_ref()).or_insert(0) += 1;
        }

        let k1 = self.config.k1;
        let b = self.config.b;
        let dl = doc_len as f32;
        let avgdl = self.avg_doc_len();

        let mut score = 0.0f32;

        for query_term in query_terms {
            let term = query_term.as_ref().to_lowercase();
            let term_str = term.as_str();

            // Get TF for this term in the document
            let term_tf = *tf.get(term_str).unwrap_or(&0) as f32;
            if term_tf == 0.0 {
                continue;
            }

            // Get IDF
            let idf = self.idf(&term);

            // BM25 scoring formula
            let numerator = term_tf * (k1 + 1.0);
            let denominator = term_tf + k1 * (1.0 - b + b * dl / avgdl);

            score += idf * numerator / denominator;
        }

        score
    }

    /// Score a document given precomputed term frequencies
    #[inline]
    pub fn score_with_tf(
        &self,
        query_terms: &[String],
        doc_tf: &HashMap<String, usize>,
        doc_len: usize,
    ) -> f32 {
        self.score_tf_lookup(query_terms, doc_len, |term| {
            *doc_tf.get(term).unwrap_or(&0) as f32
        })
    }

    /// Score a document directly from a `u32`-valued term-frequency map.
    ///
    /// Identical math to [`score_with_tf`](Self::score_with_tf) but lets callers
    /// whose postings already store `u32` frequencies (the inverted index) score
    /// without cloning the whole `term_freqs` map into a `usize`-valued copy on
    /// every query.
    #[inline]
    pub fn score_with_tf_u32(
        &self,
        query_terms: &[String],
        doc_tf: &HashMap<String, u32>,
        doc_len: usize,
    ) -> f32 {
        self.score_tf_lookup(query_terms, doc_len, |term| {
            *doc_tf.get(term).unwrap_or(&0) as f32
        })
    }

    /// Shared BM25 scoring core: sums the per-term contribution using a caller
    /// supplied term-frequency lookup, so the formula has exactly one definition.
    #[inline]
    fn score_tf_lookup<F>(&self, query_terms: &[String], doc_len: usize, mut tf_of: F) -> f32
    where
        F: FnMut(&str) -> f32,
    {
        let k1 = self.config.k1;
        let b = self.config.b;
        let dl = doc_len as f32;
        let avgdl = self.avg_doc_len();

        let mut score = 0.0f32;

        for term in query_terms {
            let term_tf = tf_of(term);
            if term_tf == 0.0 {
                continue;
            }

            let idf = self.idf(term);
            let numerator = term_tf * (k1 + 1.0);
            let denominator = term_tf + k1 * (1.0 - b + b * dl / avgdl);

            score += idf * numerator / denominator;
        }

        score
    }

    /// Update stats when adding a document
    pub fn add_document<I, T>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut doc_len = 0usize;

        for token in tokens {
            let term = token.as_ref().to_lowercase();
            if !term.is_empty() {
                seen.insert(term);
                doc_len += 1;
            }
        }

        // Update running totals (average document length is derived from these
        // on read, so it never goes stale).
        self.num_docs += 1;
        self.total_len += doc_len;

        // Update document frequencies. IDF is computed lazily at query time from
        // (df, N), so there is no cache to keep in sync here.
        for term in seen {
            *self.doc_freqs.entry(term).or_insert(0) += 1;
        }
    }

    /// Update stats when removing a document.
    ///
    /// Inverse of [`add_document`](Self::add_document): pass the document's
    /// unique terms and its token length. Keeps `num_docs`, `total_len`, and
    /// `doc_freqs` consistent so IDF and avgdl never drift under deletion, and
    /// drops terms whose document frequency reaches zero (no vocabulary leak).
    pub fn remove_document<'a, I>(&mut self, unique_terms: I, doc_len: usize)
    where
        I: IntoIterator<Item = &'a str>,
    {
        if self.num_docs == 0 {
            return;
        }
        self.num_docs -= 1;
        self.total_len = self.total_len.saturating_sub(doc_len);

        for term in unique_terms {
            let term = term.to_lowercase();
            if let Some(df) = self.doc_freqs.get_mut(&term) {
                *df -= 1;
                if *df == 0 {
                    self.doc_freqs.remove(&term);
                }
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> BM25Stats {
        BM25Stats {
            num_docs: self.num_docs,
            avg_doc_len: self.avg_doc_len(),
            vocab_size: self.doc_freqs.len(),
        }
    }
}

/// BM25 scorer statistics
#[derive(Debug, Clone)]
pub struct BM25Stats {
    pub num_docs: usize,
    pub avg_doc_len: f32,
    pub vocab_size: usize,
}

// ============================================================================
// Simple Tokenizer
// ============================================================================

/// Simple whitespace + lowercase tokenizer
///
/// For MVP, this is sufficient. Can be replaced with more sophisticated
/// tokenizers (stemming, stopwords, etc.) later.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Tokenize with minimal normalization
pub fn tokenize_minimal(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty() && s.len() > 1) // Filter single chars
        .collect()
}

/// Tokenize query (keeps original for exact matching, adds lowercase)
pub fn tokenize_query(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for part in text.split_whitespace() {
        let lower = part.to_lowercase();
        tokens.push(lower);
    }
    tokens
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let docs = vec![
            vec!["hello", "world"],
            vec!["hello", "there"],
            vec!["goodbye", "world"],
        ];

        let scorer = BM25Scorer::build(docs.iter().map(|d| d.iter()), BM25Config::default());

        assert_eq!(scorer.num_docs, 3);
        assert!((scorer.avg_doc_len() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_bm25_idf() {
        let docs = vec![
            vec!["common", "common", "rare"],
            vec!["common", "other"],
            vec!["common", "another"],
        ];

        let scorer = BM25Scorer::build(docs.iter().map(|d| d.iter()), BM25Config::default());

        // "common" appears in all 3 docs, "rare" in only 1
        let idf_common = scorer.idf("common");
        let idf_rare = scorer.idf("rare");

        // Rare terms should have higher IDF
        assert!(idf_rare > idf_common);
    }

    #[test]
    fn test_bm25_scoring() {
        let docs = vec![
            vec!["the", "quick", "brown", "fox"],
            vec!["the", "lazy", "dog"],
            vec!["quick", "quick", "quick"], // High TF for "quick"
        ];

        let scorer = BM25Scorer::build(docs.iter().map(|d| d.iter()), BM25Config::default());

        // Score doc 3 for "quick"
        let score = scorer.score(vec!["quick"], &["quick", "quick", "quick"], 3);

        assert!(score > 0.0);

        // Score doc 1 for "quick"
        let score1 = scorer.score(vec!["quick"], &["the", "quick", "brown", "fox"], 4);

        // Doc 3 should score higher (more occurrences of "quick")
        assert!(score > score1);
    }

    #[test]
    fn test_tokenize() {
        let text = "Hello, World! This is a test.";
        let tokens = tokenize(text);

        assert_eq!(tokens, vec!["hello,", "world!", "this", "is", "a", "test."]);
    }

    #[test]
    fn test_tokenize_minimal() {
        let text = "Hello, World! This is a test.";
        let tokens = tokenize_minimal(text);

        // Single chars and punctuation filtered
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // Single char
    }

    #[test]
    fn test_add_document() {
        let mut scorer = BM25Scorer::new(BM25Config::default());

        scorer.add_document(vec!["hello", "world"]);
        assert_eq!(scorer.num_docs, 1);

        scorer.add_document(vec!["hello", "there", "friend"]);
        assert_eq!(scorer.num_docs, 2);

        // Average should be (2 + 3) / 2 = 2.5
        assert!((scorer.avg_doc_len() - 2.5).abs() < 0.001);
    }
    #[test]
    fn test_build_equals_incremental() {
        // A corpus built in batch must score identically to the same corpus
        // built one document at a time: scoring is a pure function of corpus
        // content, not of insertion order. Equality is exact (bit-for-bit)
        // because both paths derive IDF/avgdl from identical integer totals.
        let docs: Vec<Vec<&str>> = vec![
            vec!["the", "quick", "brown", "fox"],
            vec!["the", "lazy", "dog", "sleeps"],
            vec!["quick", "quick", "brown", "dog"],
            vec!["the", "fox", "and", "the", "dog"],
        ];

        let batch = BM25Scorer::build(docs.iter().map(|d| d.iter()), BM25Config::default());

        let mut incremental = BM25Scorer::new(BM25Config::default());
        for d in &docs {
            incremental.add_document(d.iter().copied());
        }

        // Corpus-level stats are identical.
        assert_eq!(batch.num_docs, incremental.num_docs);
        assert_eq!(batch.total_len, incremental.total_len);
        assert_eq!(
            batch.avg_doc_len().to_bits(),
            incremental.avg_doc_len().to_bits()
        );

        // IDF is identical for every term in the vocabulary.
        for term in [
            "the", "quick", "brown", "fox", "lazy", "dog", "sleeps", "and",
        ] {
            assert_eq!(
                batch.idf(term).to_bits(),
                incremental.idf(term).to_bits(),
                "IDF mismatch for term {term:?}"
            );
        }

        // And full BM25 scores match.
        let doc = ["quick", "quick", "brown", "dog"];
        assert_eq!(
            batch.score(vec!["quick", "dog"], &doc, doc.len()).to_bits(),
            incremental
                .score(vec!["quick", "dog"], &doc, doc.len())
                .to_bits(),
        );
    }
}
