//! LongMemEval-S dataset loader.

use super::scoring::BenchQuestion;
use serde::Deserialize;
use std::path::Path;

const ABSTENTION: &[&str] = &[
    "single-session-user_abs",
    "multi-session_abs",
    "knowledge-update_abs",
    "temporal-reasoning_abs",
];

#[derive(Debug, Deserialize)]
struct LongMemEntry {
    question_id: String,
    question: String,
    question_type: String,
    answer_session_ids: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<HaystackTurn>>,
}

#[derive(Debug, Deserialize)]
struct HaystackTurn {
    role: String,
    content: String,
}

pub fn load_questions(path: &Path) -> Result<Vec<BenchQuestion>, String> {
    let raw: Vec<LongMemEntry> =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

    Ok(raw
        .into_iter()
        .filter(|e| !ABSTENTION.contains(&e.question_type.as_str()))
        .map(|e| {
            let gold: Vec<u64> = e
                .answer_session_ids
                .iter()
                .map(|s| hash_id(s))
                .collect();
            BenchQuestion {
                id: e.question_id,
                category: e.question_type,
                query: e.question,
                gold_doc_ids: gold,
                context_text: String::new(),
            }
        })
        .collect())
}

pub fn ingest_haystacks(
    store: &sochdb_memory::MemoryStore,
    namespace: &str,
    path: &Path,
) -> Result<(), String> {
    let raw: Vec<LongMemEntry> =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

    for entry in raw {
        for (sid, turns) in entry
            .haystack_session_ids
            .iter()
            .zip(entry.haystack_sessions.iter())
        {
            let text = turns
                .iter()
                .map(|t| format!("{}: {}", t.role, t.content))
                .collect::<Vec<_>>()
                .join("\n");
            store
                .write_episode(sochdb_memory::EpisodeWrite {
                    namespace: namespace.to_string(),
                    text,
                    t_valid_from: None,
                    metadata: Some(serde_json::json!({
                        "session_id": sid,
                        "doc_id": hash_id(sid),
                    })),
                })
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn hash_id(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}