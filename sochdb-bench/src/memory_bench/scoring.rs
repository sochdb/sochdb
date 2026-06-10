use serde::{Deserialize, Serialize};
use sochdb_memory::{MemoryQuery, MemoryStore, QueryLanes};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    pub question_id: String,
    pub category: String,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub mrr: f32,
    pub exact_context_tokens: usize,
    pub retrieval_latency_us: u64,
    pub ingestion_lag_us: u64,
    pub lanes_used: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBenchReport {
    pub dataset: String,
    pub questions: usize,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub mrr: f32,
    pub median_tokens: usize,
    pub p50_retrieval_us: u64,
    pub p95_retrieval_us: u64,
    pub median_ingestion_lag_us: u64,
    pub per_category: std::collections::HashMap<String, CategoryStats>,
    pub per_question: Vec<QuestionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    pub count: usize,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub mrr: f32,
}

#[derive(Debug, Clone)]
pub struct BenchQuestion {
    pub id: String,
    pub category: String,
    pub query: String,
    pub gold_doc_ids: Vec<u64>,
    pub context_text: String,
}

pub fn recall_any(retrieved: &[u64], gold: &[u64], k: usize) -> f32 {
    let top: HashSet<_> = retrieved.iter().take(k).copied().collect();
    if gold.iter().any(|g| top.contains(g)) {
        1.0
    } else {
        0.0
    }
}

pub fn mrr(retrieved: &[u64], gold: &[u64]) -> f32 {
    for (i, id) in retrieved.iter().enumerate() {
        if gold.contains(id) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

pub fn run_retrieval_suite(
    store: &MemoryStore,
    namespace: &str,
    dataset: &str,
    questions: &[BenchQuestion],
    ingestion_lags: &std::collections::HashMap<String, u64>,
) -> MemoryBenchReport {
    let mut per_question = Vec::new();
    let mut latencies = Vec::new();

    for q in questions {
        let mq = MemoryQuery {
            namespace: namespace.to_string(),
            query: q.query.clone(),
            as_of: None,
            lanes: QueryLanes::lexical_only(),
            k: 20,
        };
        let result = store.query(&mq);
        let retrieved: Vec<u64> = result.hits.iter().map(|h| h.doc_id).collect();
        latencies.push(result.query_latency_us);

        let exact_tokens = sochdb_query::count_tokens_exact(&q.context_text);
        let ingestion_lag = ingestion_lags.get(&q.id).copied().unwrap_or(0);

        per_question.push(QuestionResult {
            question_id: q.id.clone(),
            category: q.category.clone(),
            recall_at_5: recall_any(&retrieved, &q.gold_doc_ids, 5),
            recall_at_10: recall_any(&retrieved, &q.gold_doc_ids, 10),
            mrr: mrr(&retrieved, &q.gold_doc_ids),
            exact_context_tokens: exact_tokens,
            retrieval_latency_us: result.query_latency_us,
            ingestion_lag_us: ingestion_lag,
            lanes_used: result
                .lanes_used
                .iter()
                .map(|l| format!("{l:?}"))
                .collect(),
        });
    }

    let n = per_question.len().max(1) as f32;
    let mut per_category: std::collections::HashMap<String, Vec<&QuestionResult>> =
        std::collections::HashMap::new();
    for r in &per_question {
        per_category.entry(r.category.clone()).or_default().push(r);
    }

    let cat_stats: std::collections::HashMap<String, CategoryStats> = per_category
        .iter()
        .map(|(cat, rows)| {
            let c = rows.len().max(1) as f32;
            (
                cat.clone(),
                CategoryStats {
                    count: rows.len(),
                    recall_at_5: rows.iter().map(|r| r.recall_at_5).sum::<f32>() / c,
                    recall_at_10: rows.iter().map(|r| r.recall_at_10).sum::<f32>() / c,
                    mrr: rows.iter().map(|r| r.mrr).sum::<f32>() / c,
                },
            )
        })
        .collect();

    latencies.sort_unstable();
    let p50 = latencies.get(latencies.len() / 2).copied().unwrap_or(0);
    let p95 = latencies
        .get((latencies.len() as f32 * 0.95) as usize)
        .copied()
        .unwrap_or(0);

    let mut token_counts: Vec<usize> = per_question.iter().map(|r| r.exact_context_tokens).collect();
    token_counts.sort_unstable();
    let median_tokens = token_counts.get(token_counts.len() / 2).copied().unwrap_or(0);

    let mut lags: Vec<u64> = per_question.iter().map(|r| r.ingestion_lag_us).collect();
    lags.sort_unstable();
    let median_lag = lags.get(lags.len() / 2).copied().unwrap_or(0);

    MemoryBenchReport {
        dataset: dataset.to_string(),
        questions: per_question.len(),
        recall_at_5: per_question.iter().map(|r| r.recall_at_5).sum::<f32>() / n,
        recall_at_10: per_question.iter().map(|r| r.recall_at_10).sum::<f32>() / n,
        mrr: per_question.iter().map(|r| r.mrr).sum::<f32>() / n,
        median_tokens,
        p50_retrieval_us: p50,
        p95_retrieval_us: p95,
        median_ingestion_lag_us: median_lag,
        per_category: cat_stats,
        per_question,
    }
}