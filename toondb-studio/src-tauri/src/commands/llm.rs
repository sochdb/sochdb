//! LLM Commands - OpenAI-compatible chat completion with MCP tools
//!
//! Supports:
//! - OpenAI API
//! - Azure OpenAI 
//! - OpenAI-compatible endpoints (Ollama, LMStudio, vLLM, etc.)

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tauri::State;

use crate::state::AppState;

/// LLM Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Provider type: "openai", "azure", "custom"
    pub provider: String,
    /// API key
    pub api_key: String,
    /// Custom endpoint URL (required for azure/custom)
    pub endpoint: Option<String>,
    /// Model name
    pub model: String,
    /// Azure API version (required for azure)
    pub azure_api_version: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            api_key: String::new(),
            endpoint: None,
            model: "gpt-4o-mini".to_string(),
            azure_api_version: None,
        }
    }
}

/// Chat message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub tool_results: Option<Vec<ToolResult>>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Save LLM configuration
#[tauri::command]
pub async fn save_llm_config(
    app: tauri::AppHandle,
    config: LlmConfig,
) -> Result<(), String> {
    use tauri_plugin_store::StoreExt;
    
    let store = app.store("llm_config.json")
        .map_err(|e| format!("Failed to open store: {}", e))?;
    
    store.set("config", json!(config));
    store.save().map_err(|e| format!("Failed to save: {}", e))?;
    
    Ok(())
}

/// Get LLM configuration
#[tauri::command]
pub async fn get_llm_config(
    app: tauri::AppHandle,
) -> Result<Option<LlmConfig>, String> {
    use tauri_plugin_store::StoreExt;
    
    let store = app.store("llm_config.json")
        .map_err(|e| format!("Failed to open store: {}", e))?;
    
    match store.get("config") {
        Some(value) => {
            let config: LlmConfig = serde_json::from_value(value.clone())
                .map_err(|e| format!("Failed to parse config: {}", e))?;
            Ok(Some(config))
        }
        None => Ok(None),
    }
}

/// Clear LLM configuration
#[tauri::command]
pub async fn clear_llm_config(
    app: tauri::AppHandle,
) -> Result<(), String> {
    use tauri_plugin_store::StoreExt;
    
    let store = app.store("llm_config.json")
        .map_err(|e| format!("Failed to open store: {}", e))?;
    
    store.clear();
    store.save().map_err(|e| format!("Failed to save: {}", e))?;
    
    Ok(())
}

/// Test LLM connection
#[tauri::command]
pub async fn test_llm_connection(
    app: tauri::AppHandle,
) -> Result<String, String> {
    let config = get_llm_config(app).await?
        .ok_or("No LLM configuration found")?;
    
    if config.api_key.is_empty() {
        return Err("API key is required".to_string());
    }
    
    // Build endpoint URL
    let url = match config.provider.as_str() {
        "openai" => {
            // Use custom endpoint if provided, otherwise default to OpenAI
            if let Some(ref endpoint) = config.endpoint {
                if !endpoint.is_empty() {
                    format!("{}/chat/completions", endpoint.trim_end_matches('/'))
                } else {
                    "https://api.openai.com/v1/chat/completions".to_string()
                }
            } else {
                "https://api.openai.com/v1/chat/completions".to_string()
            }
        },
        "azure" => {
            let endpoint = config.endpoint.as_ref()
                .ok_or("Azure endpoint is required")?;
            let default_version = "2024-02-01".to_string();
            let api_version = config.azure_api_version.as_ref()
                .unwrap_or(&default_version);
            format!("{}/openai/deployments/{}/chat/completions?api-version={}", 
                endpoint.trim_end_matches('/'), 
                config.model,
                api_version
            )
        },
        "custom" => {
            let endpoint = config.endpoint.as_ref()
                .ok_or("Custom endpoint is required")?;
            format!("{}/chat/completions", endpoint.trim_end_matches('/'))
        },
        _ => return Err(format!("Unknown provider: {}", config.provider)),
    };
    
    // Build request
    let client = reqwest::Client::new();
    let mut req = client.post(&url)
        .header("Content-Type", "application/json");
    
    // Add auth header
    if config.provider == "azure" {
        req = req.header("api-key", &config.api_key);
    } else {
        req = req.header("Authorization", format!("Bearer {}", config.api_key));
    }
    
    // Simple test message
    let body = json!({
        "model": config.model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5
    });
    
    let response = req.json(&body).send().await
        .map_err(|e| format!("Network error: {}", e))?;
    
    if response.status().is_success() {
        Ok(format!("✓ Connected to {} using model {}", config.provider, config.model))
    } else {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        Err(format!("API error ({}): {}", status, body))
    }
}

/// Chat completion with MCP tool support
#[tauri::command]
pub async fn chat_completion(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    messages: Vec<ChatMessage>,
) -> Result<ChatResponse, String> {
    let config = get_llm_config(app.clone()).await?
        .ok_or("No LLM configuration found")?;
    
    if config.api_key.is_empty() {
        return Err("API key is required".to_string());
    }
    
    // Get MCP tools (only for native OpenAI API, not custom endpoints)
    let is_native_openai = config.provider == "openai" && 
        config.endpoint.as_ref().map(|e| e.is_empty()).unwrap_or(true);
    let tools = if is_native_openai {
        get_openai_tools(&state).await?
    } else {
        vec![]  // Skip tools for custom endpoints - may not support function calling
    };
    
    // Build endpoint URL
    let url = match config.provider.as_str() {
        "openai" => {
            // Use custom endpoint if provided, otherwise default to OpenAI
            if let Some(ref endpoint) = config.endpoint {
                if !endpoint.is_empty() {
                    format!("{}/chat/completions", endpoint.trim_end_matches('/'))
                } else {
                    "https://api.openai.com/v1/chat/completions".to_string()
                }
            } else {
                "https://api.openai.com/v1/chat/completions".to_string()
            }
        },
        "azure" => {
            let endpoint = config.endpoint.as_ref()
                .ok_or("Azure endpoint is required")?;
            let default_version = "2024-02-01".to_string();
            let api_version = config.azure_api_version.as_ref()
                .unwrap_or(&default_version);
            format!("{}/openai/deployments/{}/chat/completions?api-version={}", 
                endpoint.trim_end_matches('/'), 
                config.model,
                api_version
            )
        },
        "custom" => {
            let endpoint = config.endpoint.as_ref()
                .ok_or("Custom endpoint is required")?;
            format!("{}/chat/completions", endpoint.trim_end_matches('/'))
        },
        _ => return Err(format!("Unknown provider: {}", config.provider)),
    };
    
    // Build request body
    let mut body = json!({
        "model": config.model,
        "messages": messages,
    });
    
    // Add tools only if available (native OpenAI)
    if !tools.is_empty() {
        body["tools"] = json!(tools);
        body["tool_choice"] = json!("auto");
    }
    
    // Make request
    let client = reqwest::Client::new();
    let mut req = client.post(&url)
        .header("Content-Type", "application/json");
    
    if config.provider == "azure" {
        req = req.header("api-key", &config.api_key);
    } else {
        req = req.header("Authorization", format!("Bearer {}", config.api_key));
    }
    
    let response = req.json(&body).send().await
        .map_err(|e| format!("Network error: {}", e))?;
    
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error ({}): {}", status, body));
    }
    
    let response_json: Value = response.json().await
        .map_err(|e| format!("Failed to parse response: {}", e))?;
    
    // Parse response
    let choice = response_json["choices"].get(0)
        .ok_or("No choices in response")?;
    
    let message = &choice["message"];
    let content = message["content"].as_str().map(|s| s.to_string());
    let role = message["role"].as_str().unwrap_or("assistant").to_string();
    
    // Check for tool calls
    let tool_calls: Option<Vec<ToolCall>> = if let Some(calls) = message["tool_calls"].as_array() {
        Some(calls.iter().filter_map(|tc| {
            Some(ToolCall {
                id: tc["id"].as_str()?.to_string(),
                call_type: tc["type"].as_str().unwrap_or("function").to_string(),
                function: FunctionCall {
                    name: tc["function"]["name"].as_str()?.to_string(),
                    arguments: tc["function"]["arguments"].as_str()?.to_string(),
                },
            })
        }).collect())
    } else {
        None
    };
    
    // Execute tool calls if present
    let tool_results = if let Some(ref calls) = tool_calls {
        let mut results = Vec::new();
        for call in calls {
            let args: Value = serde_json::from_str(&call.function.arguments)
                .unwrap_or(json!({}));
            
            // Call MCP tool
            let mcp_lock = state.mcp_server.read().await;
            if let Some(mcp) = mcp_lock.as_ref() {
                let mcp_req = toondb_mcp::RpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id: Value::Number(1.into()),
                    method: "tools/call".to_string(),
                    params: json!({
                        "name": call.function.name,
                        "arguments": args,
                    }),
                };
                
                let mcp_resp = mcp.dispatch(&mcp_req);
                let result_text = if let Some(result) = mcp_resp.result {
                    if let Some(content) = result["content"].as_array() {
                        content.get(0)
                            .and_then(|c| c["text"].as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| serde_json::to_string(&result).unwrap_or_default())
                    } else {
                        serde_json::to_string(&result).unwrap_or_default()
                    }
                } else if let Some(err) = mcp_resp.error {
                    format!("Error: {}", err.message)
                } else {
                    "No result".to_string()
                };
                
                results.push(ToolResult {
                    tool_name: call.function.name.clone(),
                    result: result_text,
                });
            }
        }
        Some(results)
    } else {
        None
    };
    
    // Parse usage
    let usage = response_json["usage"].as_object().map(|u| Usage {
        prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
    });
    
    Ok(ChatResponse {
        message: ChatMessage {
            role,
            content,
            tool_calls,
            tool_call_id: None,
        },
        tool_results,
        usage,
    })
}

/// Convert MCP tools to OpenAI function format
async fn get_openai_tools(state: &State<'_, Arc<AppState>>) -> Result<Vec<Value>, String> {
    let mcp_lock = state.mcp_server.read().await;
    let mcp = match mcp_lock.as_ref() {
        Some(m) => m,
        None => return Ok(vec![]),
    };
    
    // Get tools list
    let req = toondb_mcp::RpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Value::Number(1.into()),
        method: "tools/list".to_string(),
        params: Value::Null,
    };
    
    let resp = mcp.dispatch(&req);
    
    let tools_list = match resp.result {
        Some(result) => result["tools"].as_array().cloned().unwrap_or_default(),
        None => return Ok(vec![]),
    };
    
    // Convert to OpenAI format
    let openai_tools: Vec<Value> = tools_list.iter().map(|tool| {
        json!({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description").unwrap_or(&json!("")),
                "parameters": tool.get("inputSchema").unwrap_or(&json!({"type": "object", "properties": {}}))
            }
        })
    }).collect();
    
    Ok(openai_tools)
}
