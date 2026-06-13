//! BEAM benchmark loader (stub — loads JSON array of QA items).

use super::scoring::BenchQuestion;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct BeamItem {
    id: String,
    question: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    gold_sessions: Vec<String>,
    #[serde(default)]
    context: String,
}

pub fn load_questions(path: &Path) -> Result<Vec<BenchQuestion>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw: Vec<BeamItem> =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

    Ok(raw
        .into_iter()
        .map(|e| BenchQuestion {
            id: e.id,
            category: if e.category.is_empty() {
                "beam".to_string()
            } else {
                e.category
            },
            query: e.question,
            gold_doc_ids: e.gold_sessions.iter().map(|s| hash_id(s)).collect(),
            context_text: e.context,
        })
        .collect())
}

fn hash_id(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}
