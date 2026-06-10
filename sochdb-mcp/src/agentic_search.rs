//! Agentic search tools: indexed grep with line-window delivery.

use regex::Regex;
use serde_json::{Value, json};
use sochdb_query::trigram_index::{TrigramIndex, trigrams_of};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub struct LineHit {
    pub doc_id: String,
    pub line: usize,
    pub snippet: String,
}

/// In-memory document store for agentic grep/peek/expand.
pub struct AgenticCorpus {
    docs: RwLock<HashMap<String, String>>,
    index: RwLock<TrigramIndex>,
    next_id: RwLock<u64>,
}

impl Default for AgenticCorpus {
    fn default() -> Self {
        Self::new()
    }
}

impl AgenticCorpus {
    pub fn new() -> Self {
        Self {
            docs: RwLock::new(HashMap::new()),
            index: RwLock::new(TrigramIndex::new()),
            next_id: RwLock::new(1),
        }
    }

    pub fn upsert(&self, doc_id: Option<&str>, text: &str) -> String {
        let id = doc_id.map(|s| s.to_string()).unwrap_or_else(|| {
            let mut n = self.next_id.write().unwrap();
            let id = format!("doc:{n}");
            *n += 1;
            id
        });
        let numeric_id = hash_id(&id);
        self.index.write().unwrap().insert(numeric_id, text);
        self.docs.write().unwrap().insert(id.clone(), text.to_string());
        id
    }

    pub fn grep(&self, pattern: &str, scope: Option<&str>, limit: usize) -> Result<Vec<LineHit>, String> {
        let re = Regex::new(pattern).map_err(|e| e.to_string())?;
        let docs = self.docs.read().unwrap();
        let index = self.index.read().unwrap();

        let candidates: Vec<(String, String)> = if let Some(lit) = extract_literal(pattern) {
            let tris = trigrams_of(&lit);
            let ids = index.candidates(&tris);
            ids.into_iter()
                .filter_map(|nid| {
                    docs.iter()
                        .find(|(k, _)| hash_id(k) == nid)
                        .map(|(k, v)| (k.clone(), v.clone()))
                })
                .collect()
        } else {
            docs.iter()
                .filter(|(k, _)| scope.map(|s| k.contains(s)).unwrap_or(true))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        let mut hits = Vec::new();
        for (doc_id, text) in candidates {
            if let Some(scope) = scope {
                if !doc_id.contains(scope) {
                    continue;
                }
            }
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    hits.push(LineHit {
                        doc_id: doc_id.clone(),
                        line: i + 1,
                        snippet: line.chars().take(200).collect(),
                    });
                    if hits.len() >= limit {
                        return Ok(hits);
                    }
                }
            }
        }
        Ok(hits)
    }

    pub fn peek(&self, doc_id: &str, start: usize, end: usize) -> Result<String, String> {
        let docs = self.docs.read().unwrap();
        let text = docs.get(doc_id).ok_or_else(|| format!("doc not found: {doc_id}"))?;
        let lines: Vec<&str> = text.lines().collect();
        let s = start.saturating_sub(1);
        let e = end.min(lines.len());
        Ok(lines[s..e].join("\n"))
    }

    pub fn expand(&self, doc_id: &str, line: usize, window: usize) -> Result<String, String> {
        let start = line.saturating_sub(window);
        let end = line + window;
        self.peek(doc_id, start, end)
    }

    pub fn tool_definitions() -> Vec<Value> {
        vec![
            json!({
                "name": "sochdb_grep",
                "description": "Indexed grep over corpus with line-anchored hits (sublinear via trigram lane).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" },
                        "scope": { "type": "string", "description": "Optional doc_id prefix filter" },
                        "limit": { "type": "integer", "default": 50 }
                    },
                    "required": ["pattern"]
                }
            }),
            json!({
                "name": "sochdb_peek",
                "description": "Read a line range from a document.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_id": { "type": "string" },
                        "start_line": { "type": "integer" },
                        "end_line": { "type": "integer" }
                    },
                    "required": ["doc_id", "start_line", "end_line"]
                }
            }),
            json!({
                "name": "sochdb_expand",
                "description": "Expand ±N lines around a hit for agentic iteration.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_id": { "type": "string" },
                        "line": { "type": "integer" },
                        "window": { "type": "integer", "default": 5 }
                    },
                    "required": ["doc_id", "line"]
                }
            }),
        ]
    }
}

fn hash_id(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn extract_literal(pattern: &str) -> Option<String> {
    let run: String = pattern
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    if run.len() >= 3 {
        Some(run)
    } else {
        None
    }
}