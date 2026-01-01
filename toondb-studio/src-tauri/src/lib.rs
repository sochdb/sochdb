//! ToonDB Studio - Database Administration Tool
//! 
//! A Tauri 2.0 desktop application for managing ToonDB databases.
//! Similar to DBeaver, TablePlus, MongoDB Compass - but for ToonDB.

mod commands;
mod connection;
mod parser;
pub mod policy;
mod state;

use std::sync::Arc;

pub use policy::{CommandCategory, CommandPolicy, ConnectionMode, PolicyCheck, UserRole};
pub use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize unified application state
    let app_state = Arc::new(AppState::new());
    
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            // Database commands
            commands::admin::connect,
            commands::admin::disconnect,
            commands::admin::get_stats,
            commands::admin::checkpoint,
            commands::admin::gc,
            commands::admin::compact,
            // Policy commands
            commands::admin::get_policy,
            commands::admin::set_policy,
            commands::admin::check_command,
            commands::admin::get_allowed_commands,
            // Query commands
            commands::query::execute_query,
            commands::query::explain_query,
            // Schema commands
            commands::schema::get_schema,
            commands::schema::get_table_info,
            // Vector commands
            commands::vector::vector_search,
            commands::vector::get_vector_columns,
            // MCP commands
            commands::mcp::mcp_list_tools,
            commands::mcp::mcp_call_tool,
            commands::mcp::get_service_status,
            // LLM commands
            commands::llm::save_llm_config,
            commands::llm::get_llm_config,
            commands::llm::clear_llm_config,
            commands::llm::test_llm_connection,
            commands::llm::chat_completion,
            // Context Engineering commands
            commands::context::list_context_recipes,
            commands::context::get_context_recipe,
            commands::context::save_context_recipe,
            commands::context::explain_context,
            commands::context::get_session_budget,
            commands::context::bind_recipe_to_session,
            commands::context::list_recipe_sessions,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
