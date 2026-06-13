//! LoComo dataset loader for memory benchmarks.

use super::scoring::BenchQuestion;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

const SKIP_CATEGORIES: &[u32] = &[5];

#[derive(Debug, Deserialize)]
struct LocomoItem {
    sample_id: String,
    conversation: LocomoConversation,
    qa: Vec<LocomoQa>,
}

#[derive(Debug, Deserialize)]
struct LocomoConversation {
    #[serde(flatten)]
    sessions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct LocomoQa {
    question: String,
    category: u32,
    evidence: Vec<String>,
}

fn session_keys(conv: &LocomoConversation) -> Vec<String> {
    let mut keys: Vec<String> = conv
        .sessions
        .keys()
        .filter(|k| k.starts_with("session_") && !k.ends_with("_date_time"))
        .cloned()
        .collect();
    keys.sort_by_key(|k| {
        k.split('_')
            .last()
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(0)
    });
    keys
}

pub fn ingest_conversations(
    store: &sochdb_memory::MemoryStore,
    namespace: &str,
    path: &Path,
) -> Result<HashMap<String, u64>, String> {
    let raw: Vec<LocomoItem> =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

    let mut doc_map = HashMap::new();
    for item in &raw {
        for sk in session_keys(&item.conversation) {
            let turns = item.conversation.sessions.get(&sk);
            let text = turns
                .and_then(|v| serde_json::to_string(v).ok())
                .unwrap_or_default();
            let wr = store
                .write_episode(sochdb_memory::EpisodeWrite {
                    namespace: namespace.to_string(),
                    text,
                    t_valid_from: None,
                    metadata: Some(serde_json::json!({
                        "sample_id": item.sample_id,
                        "session": sk,
                    })),
                })
                .map_err(|e| e.to_string())?;
            let key = format!("{}_{}", item.sample_id, sk);
            doc_map.insert(key, wr.episode_id.0);
        }
    }
    Ok(doc_map)
}

pub fn load_questions(
    path: &Path,
    doc_map: &HashMap<String, u64>,
) -> Result<Vec<BenchQuestion>, String> {
    let raw: Vec<LocomoItem> =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    for item in raw {
        for (qi, qa) in item.qa.iter().enumerate() {
            if SKIP_CATEGORIES.contains(&qa.category) {
                continue;
            }
            let gold: Vec<u64> = qa
                .evidence
                .iter()
                .flat_map(|e| e.split(';'))
                .filter_map(|part| {
                    let part = part.trim();
                    let session_token = part.split(':').next()?.replace('D', "");
                    if session_token.is_empty()
                        || !session_token.chars().all(|c| c.is_ascii_digit())
                    {
                        return None;
                    }
                    let sk = format!("session_{}", session_token);
                    let key = format!("{}_{}", item.sample_id, sk);
                    doc_map.get(&key).copied()
                })
                .collect();

            let cat = match qa.category {
                1 => "multi-hop",
                2 => "temporal",
                3 => "common-sense",
                4 => "single-hop",
                _ => "other",
            };

            out.push(BenchQuestion {
                id: format!("{}_q{}", item.sample_id, qi),
                category: cat.to_string(),
                query: qa.question.clone(),
                gold_doc_ids: gold,
                context_text: qa.question.clone(),
            });
        }
    }
    Ok(out)
}
