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

//! Inverted Index for Lexical Search (Task 4)
//!
//! This module implements an inverted index for BM25-based lexical search.
//!
//! ## Structure
//!
//! ```text
//! Term → PostingList
//! ┌────────────────────────────────────────────────────────────────┐
//! │  "hello" → [(doc_1, tf=2, positions=[0,5]), (doc_3, tf=1, ...)]│
//! │  "world" → [(doc_1, tf=1, positions=[1]), (doc_2, tf=3, ...)] │
//! │  ...                                                           │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Query Execution
//!
//! 1. Tokenize query
//! 2. Look up posting lists for each query term
//! 3. Score documents using BM25
//! 4. Return top-K results

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use parking_lot::RwLock;

use crate::bm25::{BM25Config, BM25Scorer, tokenize_minimal};

// ============================================================================
// Types
// ============================================================================

/// Document ID type
pub type DocId = u64;

/// Term position in document
pub type Position = u32;

/// Term frequency
pub type TermFreq = u32;

/// A `(score, doc_id)` pair ordered by BM25 score for bounded top-k selection.
///
/// `f32` is not `Ord`, so ordering uses `f32::total_cmp` with `doc_id` as a
/// deterministic tiebreaker. Wrapped in `Reverse` to drive a min-heap that
/// retains the `k` highest-scoring documents.
struct ScoredDoc {
    score: f32,
    doc_id: DocId,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ScoredDoc {}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ============================================================================
// Posting List
// ============================================================================

/// A posting for a single document
#[derive(Debug, Clone)]
pub struct Posting {
    /// Document ID
    pub doc_id: DocId,

    /// Term frequency in this document
    pub term_freq: TermFreq,

    /// Positions of the term in the document (optional)
    pub positions: Option<Vec<Position>>,
}

impl Posting {
    /// Create a new posting
    pub fn new(doc_id: DocId, term_freq: TermFreq) -> Self {
        Self {
            doc_id,
            term_freq,
            positions: None,
        }
    }

    /// Create with positions
    pub fn with_positions(doc_id: DocId, positions: Vec<Position>) -> Self {
        Self {
            doc_id,
            term_freq: positions.len() as TermFreq,
            positions: Some(positions),
        }
    }
}

/// A posting list for a term
#[derive(Debug, Clone, Default)]
pub struct PostingList {
    /// Postings sorted by doc_id for efficient merge
    postings: Vec<Posting>,
}

impl PostingList {
    /// Create a new empty posting list
    pub fn new() -> Self {
        Self {
            postings: Vec::new(),
        }
    }

    /// Add a posting (maintains sorted order)
    pub fn add(&mut self, posting: Posting) {
        match self
            .postings
            .binary_search_by_key(&posting.doc_id, |p| p.doc_id)
        {
            Ok(idx) => {
                // Update existing
                self.postings[idx] = posting;
            }
            Err(idx) => {
                // Insert at correct position
                self.postings.insert(idx, posting);
            }
        }
    }

    /// Get posting for a document
    pub fn get(&self, doc_id: DocId) -> Option<&Posting> {
        self.postings
            .binary_search_by_key(&doc_id, |p| p.doc_id)
            .ok()
            .map(|idx| &self.postings[idx])
    }

    /// Number of documents containing this term
    pub fn doc_freq(&self) -> usize {
        self.postings.len()
    }

    /// Iterate over postings
    pub fn iter(&self) -> impl Iterator<Item = &Posting> {
        self.postings.iter()
    }

    /// Get all document IDs
    pub fn doc_ids(&self) -> Vec<DocId> {
        self.postings.iter().map(|p| p.doc_id).collect()
    }
}

// ============================================================================
// Document Info
// ============================================================================

/// Information about an indexed document
#[derive(Debug, Clone)]
pub struct DocumentInfo {
    /// Document length (in tokens)
    pub length: u32,

    /// Term frequencies
    pub term_freqs: HashMap<String, TermFreq>,
}

// ============================================================================
// Inverted Index
// ============================================================================

/// Inverted index for lexical search
pub struct InvertedIndex {
    /// Term to posting list mapping
    index: RwLock<HashMap<String, PostingList>>,

    /// Document info (for BM25 scoring)
    docs: RwLock<HashMap<DocId, DocumentInfo>>,

    /// BM25 scorer
    scorer: RwLock<BM25Scorer>,

    /// Next document ID
    next_doc_id: RwLock<DocId>,

    /// Whether to store positions
    store_positions: bool,
}

impl InvertedIndex {
    /// Create a new inverted index
    pub fn new(config: BM25Config) -> Self {
        Self {
            index: RwLock::new(HashMap::new()),
            docs: RwLock::new(HashMap::new()),
            scorer: RwLock::new(BM25Scorer::new(config)),
            next_doc_id: RwLock::new(0),
            store_positions: false,
        }
    }

    /// Enable position storage (for phrase queries)
    pub fn with_positions(mut self) -> Self {
        self.store_positions = true;
        self
    }

    /// Index a document
    ///
    /// Returns the assigned document ID.
    pub fn add_document(&self, text: &str) -> DocId {
        let tokens = tokenize_minimal(text);
        self.add_document_tokens(&tokens)
    }

    /// Index a document with specific ID
    pub fn add_document_with_id(&self, doc_id: DocId, text: &str) {
        let tokens = tokenize_minimal(text);
        self.add_document_tokens_with_id(doc_id, &tokens);
    }

    /// Remove every document, returning the index to its freshly-created state
    /// while preserving the BM25 configuration and position-storage setting.
    pub fn clear(&self) {
        let config = self.scorer.read().config();
        self.index.write().clear();
        self.docs.write().clear();
        *self.scorer.write() = BM25Scorer::new(config);
        *self.next_doc_id.write() = 0;
    }

    /// Rebuild the entire index from an authoritative `(doc_id, text)` source.
    ///
    /// ## Durability contract (Task 7)
    ///
    /// This index is an in-memory derived structure: it holds no WAL and is not
    /// itself crash-durable. The committed **document store is the source of
    /// truth**, and lexical search agrees with it only up to the last rebuild.
    /// The supported recovery model is therefore:
    ///
    /// 1. Documents commit through the durable storage path (WAL/MVCC).
    /// 2. On restart, the lexical index is reconstructed from the committed
    ///    document store via this method — an O(corpus) bounded, deterministic
    ///    pass (it [`clear`](Self::clear)s first, so the result is a pure
    ///    function of the input, independent of any prior in-memory state).
    /// 3. Only then is lexical search served.
    ///
    /// Because scores are order-independent (IDF is derived from `(df, N)` and
    /// avgdl from running totals, never cached), a rebuilt index is byte-for-byte
    /// equivalent in ranking to the index that produced the documents, so a
    /// query returns the same result set across a crash/restart boundary
    /// relative to one committed snapshot.
    pub fn rebuild_from_documents<'a, I>(&self, documents: I)
    where
        I: IntoIterator<Item = (DocId, &'a str)>,
    {
        self.clear();
        let mut max_id: Option<DocId> = None;
        for (doc_id, text) in documents {
            self.add_document_with_id(doc_id, text);
            max_id = Some(max_id.map_or(doc_id, |m| m.max(doc_id)));
        }
        // Resume auto-id allocation past the highest rebuilt id so a subsequent
        // add_document cannot collide with a restored document.
        if let Some(m) = max_id {
            *self.next_doc_id.write() = m + 1;
        }
    }

    /// Index a document from tokens
    pub fn add_document_tokens(&self, tokens: &[String]) -> DocId {
        let doc_id = {
            let mut next = self.next_doc_id.write();
            let id = *next;
            *next += 1;
            id
        };

        self.add_document_tokens_with_id(doc_id, tokens);
        doc_id
    }

    /// Index a document from tokens with specific ID
    pub fn add_document_tokens_with_id(&self, doc_id: DocId, tokens: &[String]) {
        // Build term frequencies and positions
        let mut term_freqs: HashMap<String, TermFreq> = HashMap::new();
        let mut term_positions: HashMap<String, Vec<Position>> = HashMap::new();

        for (pos, token) in tokens.iter().enumerate() {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
            if self.store_positions {
                term_positions
                    .entry(token.clone())
                    .or_default()
                    .push(pos as Position);
            }
        }

        // Update index
        {
            let mut index = self.index.write();
            for (term, tf) in &term_freqs {
                let posting = if self.store_positions {
                    Posting::with_positions(
                        doc_id,
                        term_positions.get(term).cloned().unwrap_or_default(),
                    )
                } else {
                    Posting::new(doc_id, *tf)
                };

                index.entry(term.clone()).or_default().add(posting);
            }
        }

        // Update document info
        {
            let mut docs = self.docs.write();
            docs.insert(
                doc_id,
                DocumentInfo {
                    length: tokens.len() as u32,
                    term_freqs,
                },
            );
        }

        // Update BM25 scorer
        {
            let mut scorer = self.scorer.write();
            scorer.add_document(tokens.iter().map(|s| s.as_str()));
        }
    }

    /// Remove a document from the index
    pub fn remove_document(&self, doc_id: DocId) -> bool {
        let doc_info = {
            let mut docs = self.docs.write();
            docs.remove(&doc_id)
        };

        if let Some(info) = doc_info {
            // Remove this doc's postings, and drop any posting list that becomes
            // empty so the vocabulary does not leak under churn.
            {
                let mut index = self.index.write();
                for term in info.term_freqs.keys() {
                    let now_empty = if let Some(posting_list) = index.get_mut(term) {
                        posting_list.postings.retain(|p| p.doc_id != doc_id);
                        posting_list.postings.is_empty()
                    } else {
                        false
                    };
                    if now_empty {
                        index.remove(term);
                    }
                }
            }

            // Keep the BM25 scorer's corpus statistics (N, total length, df)
            // consistent with the removal so IDF does not drift.
            {
                let mut scorer = self.scorer.write();
                scorer.remove_document(
                    info.term_freqs.keys().map(|s| s.as_str()),
                    info.length as usize,
                );
            }
            true
        } else {
            false
        }
    }

    /// Search the index
    ///
    /// Returns document IDs with scores, sorted by score descending.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(DocId, f32)> {
        let query_tokens = tokenize_minimal(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        self.search_tokens(&query_tokens, limit)
    }

    /// Search with pre-tokenized query
    pub fn search_tokens(&self, query_tokens: &[String], limit: usize) -> Vec<(DocId, f32)> {
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let index = self.index.read();
        let docs = self.docs.read();
        let scorer = self.scorer.read();

        // Collect candidate documents (union of posting lists)
        let mut candidates: HashSet<DocId> = HashSet::new();
        for token in query_tokens {
            if let Some(posting_list) = index.get(token) {
                for posting in posting_list.iter() {
                    candidates.insert(posting.doc_id);
                }
            }
        }

        // Score candidates, keeping only the best `limit` via a bounded
        // min-heap (O(C log k)) instead of scoring into a Vec and sorting the
        // whole thing (O(C log C)). Scoring reads the document's `u32` term
        // frequencies directly — no per-candidate map clone.
        let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(limit + 1);
        for doc_id in candidates {
            let Some(doc_info) = docs.get(&doc_id) else {
                continue;
            };
            let score = scorer.score_with_tf_u32(
                query_tokens,
                &doc_info.term_freqs,
                doc_info.length as usize,
            );
            if score <= 0.0 {
                continue;
            }
            if limit == 0 {
                continue;
            }
            heap.push(Reverse(ScoredDoc { score, doc_id }));
            if heap.len() > limit {
                heap.pop();
            }
        }

        // Drain the heap and order results by score descending, breaking ties
        // by ascending doc_id so the ranking is fully deterministic (and
        // independent of candidate HashMap iteration order — required for the
        // rebuild-from-store durability contract to be reproducible).
        let mut results: Vec<(DocId, f32)> = heap
            .into_iter()
            .map(|Reverse(sd)| (sd.doc_id, sd.score))
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        results
    }

    /// Get posting list for a term
    pub fn get_posting_list(&self, term: &str) -> Option<PostingList> {
        self.index.read().get(&term.to_lowercase()).cloned()
    }

    /// Get document count
    pub fn num_documents(&self) -> usize {
        self.docs.read().len()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.index.read().len()
    }

    /// Get document info
    pub fn get_document_info(&self, doc_id: DocId) -> Option<DocumentInfo> {
        self.docs.read().get(&doc_id).cloned()
    }

    /// Check if a document exists
    pub fn has_document(&self, doc_id: DocId) -> bool {
        self.docs.read().contains_key(&doc_id)
    }
}

// ============================================================================
// Inverted Index Builder
// ============================================================================

/// Builder for batch index construction
pub struct InvertedIndexBuilder {
    config: BM25Config,
    store_positions: bool,
}

impl InvertedIndexBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: BM25Config::default(),
            store_positions: false,
        }
    }

    /// Set BM25 configuration
    pub fn with_config(mut self, config: BM25Config) -> Self {
        self.config = config;
        self
    }

    /// Enable position storage
    pub fn with_positions(mut self) -> Self {
        self.store_positions = true;
        self
    }

    /// Build index from documents
    pub fn build<I>(self, documents: I) -> InvertedIndex
    where
        I: IntoIterator<Item = (DocId, String)>,
    {
        let index = if self.store_positions {
            InvertedIndex::new(self.config).with_positions()
        } else {
            InvertedIndex::new(self.config)
        };

        for (doc_id, text) in documents {
            index.add_document_with_id(doc_id, &text);
        }

        index
    }
}

impl Default for InvertedIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posting_list() {
        let mut list = PostingList::new();

        list.add(Posting::new(1, 2));
        list.add(Posting::new(3, 1));
        list.add(Posting::new(2, 3));

        assert_eq!(list.doc_freq(), 3);

        // Should be sorted by doc_id
        let ids = list.doc_ids();
        assert_eq!(ids, vec![1, 2, 3]);

        // Get specific posting
        let p = list.get(2).unwrap();
        assert_eq!(p.term_freq, 3);
    }

    #[test]
    fn test_add_document() {
        let index = InvertedIndex::new(BM25Config::default());

        let doc1 = index.add_document("hello world");
        let doc2 = index.add_document("hello there");

        assert_eq!(doc1, 0);
        assert_eq!(doc2, 1);
        assert_eq!(index.num_documents(), 2);

        // Check posting list
        let hello_list = index.get_posting_list("hello").unwrap();
        assert_eq!(hello_list.doc_freq(), 2);
    }

    #[test]
    fn test_search() {
        let index = InvertedIndex::new(BM25Config::default());

        index.add_document("the quick brown fox jumps over the lazy dog");
        index.add_document("quick quick quick fox"); // High TF for "quick"
        index.add_document("lazy lazy lazy dog"); // High TF for "lazy"

        // Search for "quick"
        let results = index.search("quick", 10);
        assert!(!results.is_empty());

        // Doc with highest TF for "quick" should score highest
        assert_eq!(results[0].0, 1); // doc_id 1 has "quick quick quick"
    }

    #[test]
    fn test_search_multi_term() {
        let index = InvertedIndex::new(BM25Config::default());

        index.add_document("apple banana cherry");
        index.add_document("apple banana");
        index.add_document("apple");

        // Multi-term query
        let results = index.search("apple banana cherry", 10);

        // Doc with most terms should score highest
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_search_topk_bound_matches_full_sort() {
        // The bounded min-heap path must return exactly the top `limit`
        // documents in descending score order, identical to scoring every
        // candidate and sorting the whole list.
        let index = InvertedIndex::new(BM25Config::default());
        for i in 0..20 {
            // Vary the term frequency so scores are distinct and ordered.
            let body = std::iter::repeat("alpha")
                .take(i + 1)
                .collect::<Vec<_>>()
                .join(" ");
            index.add_document(&format!("{body} doc{i}"));
        }

        let limit = 5;
        let topk = index.search("alpha", limit);
        assert_eq!(topk.len(), limit, "must return exactly `limit` results");

        // Scores must be non-increasing.
        for w in topk.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results must be sorted by score descending"
            );
        }

        // Compare against the unbounded ranking: the bounded top-k must be the
        // length-`limit` prefix of the full ranking.
        let full = index.search("alpha", 1000);
        let full_prefix: Vec<u64> = full.iter().take(limit).map(|(id, _)| *id).collect();
        let topk_ids: Vec<u64> = topk.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            topk_ids, full_prefix,
            "bounded top-k must equal full-sort prefix"
        );
    }

    #[test]
    fn test_rebuild_reproduces_index() {
        // Durability contract: rebuilding from the committed (doc_id, text)
        // store must reproduce a ranking-identical index, regardless of prior
        // in-memory state (insert order, churn, etc.).
        let corpus: Vec<(u64, &str)> = vec![
            (10, "the quick brown fox"),
            (11, "the lazy dog sleeps"),
            (12, "quick foxes jump high"),
            (13, "lazy dogs and quick cats"),
        ];

        // Reference index built by direct id-keyed inserts.
        let reference = InvertedIndex::new(BM25Config::default());
        for (id, text) in &corpus {
            reference.add_document_with_id(*id, text);
        }

        // A churned index: insert extra docs, remove some, then rebuild from
        // the authoritative corpus.
        let rebuilt = InvertedIndex::new(BM25Config::default());
        rebuilt.add_document_with_id(99, "noise document that should vanish");
        rebuilt.add_document_with_id(98, "more transient noise quick fox");
        rebuilt.remove_document(99);
        rebuilt.rebuild_from_documents(corpus.iter().map(|(id, t)| (*id, *t)));

        // Same document membership.
        assert!(!rebuilt.has_document(99));
        assert!(!rebuilt.has_document(98));
        for (id, _) in &corpus {
            assert!(rebuilt.has_document(*id));
        }

        // Same ranking for representative queries.
        for q in ["quick", "lazy dog", "fox", "the quick brown"] {
            assert_eq!(
                rebuilt.search(q, 10),
                reference.search(q, 10),
                "rebuilt ranking diverges for query {q:?}",
            );
        }

        // Auto-id allocation resumes past the highest restored id.
        let next = rebuilt.add_document("brand new quick doc");
        assert_eq!(next, 14, "auto-id must resume one past max restored id");
    }

    #[test]
    fn test_remove_document() {
        let index = InvertedIndex::new(BM25Config::default());

        let doc1 = index.add_document("hello world");
        let doc2 = index.add_document("hello there");

        assert!(index.has_document(doc1));
        assert!(index.remove_document(doc1));
        assert!(!index.has_document(doc1));

        // "hello" should still exist (in doc2)
        let hello_list = index.get_posting_list("hello").unwrap();
        assert_eq!(hello_list.doc_freq(), 1);

        // "world" was only in the removed doc: its posting list is dropped
        // entirely (no empty-posting vocabulary leak).
        assert!(index.get_posting_list("world").is_none());
    }

    #[test]
    fn test_add_remove_equals_never_added() {
        // Adding a document and then removing it must leave the index scoring
        // identically to one where the document was never added.
        let with_removed = InvertedIndex::new(BM25Config::default());
        with_removed.add_document_with_id(1, "the quick brown fox");
        with_removed.add_document_with_id(2, "lazy dog sleeps all day");
        let transient = with_removed.add_document("ephemeral zebra quagga");
        assert!(with_removed.remove_document(transient));

        let never_added = InvertedIndex::new(BM25Config::default());
        never_added.add_document_with_id(1, "the quick brown fox");
        never_added.add_document_with_id(2, "lazy dog sleeps all day");

        // Corpus stats match (num docs and vocabulary size).
        assert_eq!(with_removed.num_documents(), never_added.num_documents());
        assert_eq!(with_removed.vocab_size(), never_added.vocab_size());

        // The removed document's unique terms leave no vocabulary residue.
        assert!(with_removed.get_posting_list("zebra").is_none());
        assert!(with_removed.get_posting_list("quagga").is_none());

        // Scores for shared queries are bit-for-bit identical.
        for q in ["quick", "dog", "fox sleeps"] {
            let a = with_removed.search(q, 10);
            let b = never_added.search(q, 10);
            assert_eq!(a.len(), b.len(), "result-count mismatch for {q:?}");
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.0, y.0, "doc_id mismatch for {q:?}");
                assert_eq!(x.1.to_bits(), y.1.to_bits(), "score mismatch for {q:?}");
            }
        }
    }

    #[test]
    fn test_builder() {
        let documents = vec![
            (0, "hello world".to_string()),
            (1, "hello there".to_string()),
            (2, "goodbye world".to_string()),
        ];

        let index = InvertedIndexBuilder::new()
            .with_config(BM25Config::lucene())
            .build(documents);

        assert_eq!(index.num_documents(), 3);
        assert!(index.vocab_size() > 0);
    }

    #[test]
    fn test_positions() {
        let index = InvertedIndex::new(BM25Config::default()).with_positions();

        let doc_id = index.add_document("hello world hello");

        let hello_list = index.get_posting_list("hello").unwrap();
        let posting = hello_list.get(doc_id).unwrap();

        assert_eq!(posting.positions, Some(vec![0, 2]));
    }
}
