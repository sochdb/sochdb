// Copyright 2025 ToonDB Authors
//
// Licensed under the Apache License, Version 2.0

//! Recall@k evaluation with ground truth for vector search quality measurement.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Query ID type.
pub type QueryId = u64;

/// Document/Vector ID type.
pub type DocId = u64;

/// Ground truth entry for a single query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthEntry {
    /// True nearest neighbor IDs, ordered by distance (closest first)
    pub neighbors: Vec<DocId>,
    /// Optional distances for nDCG calculation
    #[serde(default)]
    pub distances: Vec<f32>,
}

/// Ground truth file format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthFile {
    /// Version for compatibility
    pub version: String,
    /// Dataset this truth was computed for
    pub dataset: String,
    /// k value used for truth computation
    pub k: usize,
    /// Number of queries
    pub num_queries: usize,
    /// Query ID -> truth mapping
    pub queries: HashMap<QueryId, GroundTruthEntry>,
}

/// Recall@k evaluator.
///
/// Computes recall and optionally nDCG against pre-computed ground truth.
pub struct RecallEvaluator {
    ground_truth: HashMap<QueryId, GroundTruthEntry>,
    dataset_name: String,
    truth_k: usize,
}

impl RecallEvaluator {
    /// Load ground truth from a JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read ground truth file: {}", e))?;
        
        let truth: GroundTruthFile = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse ground truth JSON: {}", e))?;
        
        Ok(Self {
            ground_truth: truth.queries,
            dataset_name: truth.dataset,
            truth_k: truth.k,
        })
    }

    /// Create from pre-loaded ground truth.
    pub fn from_truth(truth: HashMap<QueryId, GroundTruthEntry>, dataset: &str, k: usize) -> Self {
        Self {
            ground_truth: truth,
            dataset_name: dataset.to_string(),
            truth_k: k,
        }
    }

    /// Compute recall@k for a single query.
    ///
    /// Recall@k = |retrieved ∩ relevant| / min(k, |relevant|)
    ///
    /// # Arguments
    /// * `query_id` - The query identifier
    /// * `results` - Retrieved document IDs (in order, but order doesn't matter for recall)
    /// * `k` - Number of results to consider
    ///
    /// # Returns
    /// Recall value between 0.0 and 1.0, or None if query not in ground truth.
    pub fn recall_at_k(&self, query_id: QueryId, results: &[DocId], k: usize) -> Option<f64> {
        let truth = self.ground_truth.get(&query_id)?;
        
        // Take top-k from results
        let retrieved: std::collections::HashSet<DocId> = results.iter().take(k).copied().collect();
        
        // Take top-k from truth (or all if less than k)
        let relevant_k = k.min(truth.neighbors.len());
        let relevant: std::collections::HashSet<DocId> = 
            truth.neighbors.iter().take(relevant_k).copied().collect();
        
        if relevant.is_empty() {
            return Some(1.0); // No relevant docs = vacuously correct
        }
        
        let intersection = retrieved.intersection(&relevant).count();
        Some(intersection as f64 / relevant.len() as f64)
    }

    /// Compute average recall@k across all queries.
    pub fn average_recall_at_k(&self, results: &HashMap<QueryId, Vec<DocId>>, k: usize) -> RecallMetrics {
        let mut recalls = Vec::new();
        let mut missing = 0;

        for (query_id, query_results) in results {
            match self.recall_at_k(*query_id, query_results, k) {
                Some(recall) => recalls.push(recall),
                None => missing += 1,
            }
        }

        if recalls.is_empty() {
            return RecallMetrics {
                k,
                count: 0,
                missing,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
            };
        }

        let count = recalls.len();
        let mean = recalls.iter().sum::<f64>() / count as f64;
        let min = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = recalls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let variance = recalls.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        RecallMetrics {
            k,
            count,
            missing,
            mean,
            min,
            max,
            std_dev,
        }
    }

    /// Compute nDCG@k for a single query (if distances are available).
    ///
    /// nDCG accounts for the ranking order, not just set overlap.
    pub fn ndcg_at_k(&self, query_id: QueryId, results: &[DocId], k: usize) -> Option<f64> {
        let truth = self.ground_truth.get(&query_id)?;
        
        if truth.distances.is_empty() {
            return None; // Need distances for nDCG
        }
        
        // Create relevance score from distances (closer = more relevant)
        // Using 1 / (1 + distance) as relevance
        let mut id_to_relevance: HashMap<DocId, f64> = HashMap::new();
        for (i, &id) in truth.neighbors.iter().enumerate() {
            if i < truth.distances.len() {
                let relevance = 1.0 / (1.0 + truth.distances[i] as f64);
                id_to_relevance.insert(id, relevance);
            }
        }

        // DCG of retrieved results
        let mut dcg = 0.0;
        for (i, &doc_id) in results.iter().take(k).enumerate() {
            if let Some(&rel) = id_to_relevance.get(&doc_id) {
                dcg += rel / (2.0_f64.ln() * ((i + 2) as f64).ln()); // log2(i+2)
            }
        }

        // Ideal DCG (from ground truth order)
        let mut idcg = 0.0;
        let mut sorted_rels: Vec<f64> = id_to_relevance.values().copied().collect();
        sorted_rels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        for (i, rel) in sorted_rels.iter().take(k).enumerate() {
            idcg += rel / (2.0_f64.ln() * ((i + 2) as f64).ln());
        }

        if idcg == 0.0 {
            Some(1.0) // Perfect score if no ideal DCG
        } else {
            Some(dcg / idcg)
        }
    }

    /// Get the number of queries in ground truth.
    pub fn num_queries(&self) -> usize {
        self.ground_truth.len()
    }

    /// Get the dataset name.
    pub fn dataset_name(&self) -> &str {
        &self.dataset_name
    }

    /// Get the k value used for truth computation.
    pub fn truth_k(&self) -> usize {
        self.truth_k
    }
}

/// Recall metrics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallMetrics {
    pub k: usize,
    pub count: usize,
    pub missing: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
}

/// Compute ground truth using brute-force exact search.
///
/// This is expensive but necessary for generating correct ground truth.
pub fn compute_ground_truth(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> HashMap<QueryId, GroundTruthEntry> {
    let mut truth = HashMap::new();

    for (query_idx, query) in queries.iter().enumerate() {
        // Compute distances to all vectors
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist = euclidean_distance(query, vec);
                (idx, dist)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        let neighbors: Vec<DocId> = distances.iter().take(k).map(|(idx, _)| *idx as DocId).collect();
        let dists: Vec<f32> = distances.iter().take(k).map(|(_, d)| *d).collect();

        truth.insert(query_idx as QueryId, GroundTruthEntry {
            neighbors,
            distances: dists,
        });
    }

    truth
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Save ground truth to a JSON file.
pub fn save_ground_truth(
    truth: &HashMap<QueryId, GroundTruthEntry>,
    dataset: &str,
    k: usize,
    path: &Path,
) -> Result<(), String> {
    let file = GroundTruthFile {
        version: "1.0".to_string(),
        dataset: dataset.to_string(),
        k,
        num_queries: truth.len(),
        queries: truth.clone(),
    };

    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| format!("Failed to serialize ground truth: {}", e))?;

    fs::write(path, json)
        .map_err(|e| format!("Failed to write ground truth file: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let mut truth = HashMap::new();
        truth.insert(0, GroundTruthEntry {
            neighbors: vec![1, 2, 3, 4, 5],
            distances: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        });

        let evaluator = RecallEvaluator::from_truth(truth, "test", 5);

        // Perfect recall
        let results = vec![1, 2, 3, 4, 5];
        assert_eq!(evaluator.recall_at_k(0, &results, 5), Some(1.0));

        // 60% recall (3 of 5)
        let results = vec![1, 2, 3, 100, 101];
        assert_eq!(evaluator.recall_at_k(0, &results, 5), Some(0.6));

        // 0% recall
        let results = vec![100, 101, 102, 103, 104];
        assert_eq!(evaluator.recall_at_k(0, &results, 5), Some(0.0));
    }

    #[test]
    fn test_compute_ground_truth() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let queries = vec![
            vec![0.0, 0.0], // Closest to vector 0
        ];

        let truth = compute_ground_truth(&vectors, &queries, 3);
        
        assert_eq!(truth.len(), 1);
        let entry = truth.get(&0).unwrap();
        assert_eq!(entry.neighbors[0], 0); // Vector 0 should be closest
    }
}
