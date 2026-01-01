//! Command Policy System for ToonDB Studio
//!
//! Provides capability/policy-driven command exposure based on:
//! - Connection type (local, remote, read-only)
//! - User role (admin, viewer, developer)
//! - Database state (connected, healthy, etc.)

use std::collections::HashSet;
use serde::{Deserialize, Serialize};

/// Command categories for grouping related operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandCategory {
    /// Read-only database queries
    Query,
    /// Schema introspection
    Schema,
    /// Database administration (checkpoint, GC, compact)
    Admin,
    /// Vector search operations
    Vector,
    /// MCP tool operations
    Mcp,
    /// LLM integration commands
    Llm,
    /// Connection management
    Connection,
}

/// User role that determines command permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum UserRole {
    /// Read-only access - can query and view schema
    Viewer,
    /// Developer access - query, schema, vector search
    #[default]
    Developer,
    /// Full admin access - all commands including checkpoint/GC/compact
    Admin,
}

/// Connection mode affects which commands are safe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ConnectionMode {
    /// Local embedded database - full access
    #[default]
    LocalEmbedded,
    /// Read-only connection - no mutations
    ReadOnly,
    /// Remote server connection - may have latency
    Remote,
}

/// Policy configuration for command access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPolicy {
    /// User's role
    pub role: UserRole,
    /// Connection mode
    pub mode: ConnectionMode,
    /// Explicitly allowed commands (overrides role)
    pub allowed_commands: HashSet<String>,
    /// Explicitly denied commands (overrides role)
    pub denied_commands: HashSet<String>,
    /// Whether admin operations require confirmation
    pub require_admin_confirmation: bool,
}

impl Default for CommandPolicy {
    fn default() -> Self {
        Self {
            role: UserRole::Developer,
            mode: ConnectionMode::LocalEmbedded,
            allowed_commands: HashSet::new(),
            denied_commands: HashSet::new(),
            require_admin_confirmation: true,
        }
    }
}

impl CommandPolicy {
    /// Create a new policy with the given role
    pub fn with_role(role: UserRole) -> Self {
        Self {
            role,
            ..Default::default()
        }
    }

    /// Create a read-only policy
    pub fn read_only() -> Self {
        Self {
            role: UserRole::Viewer,
            mode: ConnectionMode::ReadOnly,
            ..Default::default()
        }
    }

    /// Create an admin policy
    pub fn admin() -> Self {
        Self {
            role: UserRole::Admin,
            require_admin_confirmation: false,
            ..Default::default()
        }
    }

    /// Check if a command category is allowed
    pub fn is_category_allowed(&self, category: CommandCategory) -> bool {
        match self.role {
            UserRole::Viewer => matches!(category, CommandCategory::Query | CommandCategory::Schema | CommandCategory::Connection),
            UserRole::Developer => matches!(
                category,
                CommandCategory::Query
                    | CommandCategory::Schema
                    | CommandCategory::Vector
                    | CommandCategory::Mcp
                    | CommandCategory::Llm
                    | CommandCategory::Connection
            ),
            UserRole::Admin => true, // Admin can do everything
        }
    }

    /// Check if a specific command is allowed
    pub fn is_command_allowed(&self, command: &str) -> bool {
        // Check explicit deny list first
        if self.denied_commands.contains(command) {
            return false;
        }

        // Check explicit allow list
        if self.allowed_commands.contains(command) {
            return true;
        }

        // Check based on role and command category
        let category = Self::get_command_category(command);
        if !self.is_category_allowed(category) {
            return false;
        }

        // Additional checks for read-only mode
        if self.mode == ConnectionMode::ReadOnly {
            return !Self::is_mutating_command(command);
        }

        true
    }

    /// Check if this command requires confirmation
    pub fn requires_confirmation(&self, command: &str) -> bool {
        if !self.require_admin_confirmation {
            return false;
        }

        // Admin operations that require confirmation
        matches!(
            command,
            "checkpoint" | "gc" | "compact" | "disconnect"
        )
    }

    /// Get the category for a command
    pub fn get_command_category(command: &str) -> CommandCategory {
        match command {
            "execute_query" | "explain_query" => CommandCategory::Query,
            "get_schema" | "get_table_info" => CommandCategory::Schema,
            "checkpoint" | "gc" | "compact" | "get_stats" => CommandCategory::Admin,
            "vector_search" | "get_vector_columns" => CommandCategory::Vector,
            "mcp_list_tools" | "mcp_call_tool" | "get_service_status" => CommandCategory::Mcp,
            "save_llm_config" | "get_llm_config" | "clear_llm_config" | "test_llm_connection" | "chat_completion" => CommandCategory::Llm,
            "connect" | "disconnect" => CommandCategory::Connection,
            _ => CommandCategory::Query, // Default to most permissive category
        }
    }

    /// Check if a command is mutating (writes data)
    pub fn is_mutating_command(command: &str) -> bool {
        matches!(
            command,
            "checkpoint" | "gc" | "compact" | "mcp_call_tool" | "save_llm_config" | "clear_llm_config"
        )
    }

    /// Get list of allowed commands for current policy
    pub fn get_allowed_commands(&self) -> Vec<&'static str> {
        let all_commands = [
            // Connection
            "connect", "disconnect",
            // Query
            "execute_query", "explain_query",
            // Schema
            "get_schema", "get_table_info",
            // Admin
            "checkpoint", "gc", "compact", "get_stats",
            // Vector
            "vector_search", "get_vector_columns",
            // MCP
            "mcp_list_tools", "mcp_call_tool", "get_service_status",
            // LLM
            "save_llm_config", "get_llm_config", "clear_llm_config", "test_llm_connection", "chat_completion",
        ];

        all_commands
            .iter()
            .filter(|cmd| self.is_command_allowed(cmd))
            .copied()
            .collect()
    }
}

/// Policy result with optional confirmation requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCheck {
    /// Whether the command is allowed
    pub allowed: bool,
    /// Whether confirmation is required
    pub requires_confirmation: bool,
    /// Reason if denied
    pub reason: Option<String>,
}

impl PolicyCheck {
    pub fn allowed() -> Self {
        Self {
            allowed: true,
            requires_confirmation: false,
            reason: None,
        }
    }

    pub fn allowed_with_confirmation() -> Self {
        Self {
            allowed: true,
            requires_confirmation: true,
            reason: None,
        }
    }

    pub fn denied(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            requires_confirmation: false,
            reason: Some(reason.into()),
        }
    }
}

/// Check if a command is allowed under the given policy
pub fn check_command(policy: &CommandPolicy, command: &str) -> PolicyCheck {
    if !policy.is_command_allowed(command) {
        let category = CommandPolicy::get_command_category(command);
        return PolicyCheck::denied(format!(
            "Command '{}' (category: {:?}) not allowed for role {:?}",
            command, category, policy.role
        ));
    }

    if policy.requires_confirmation(command) {
        PolicyCheck::allowed_with_confirmation()
    } else {
        PolicyCheck::allowed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewer_policy() {
        let policy = CommandPolicy::with_role(UserRole::Viewer);
        
        // Viewers can query and view schema
        assert!(policy.is_command_allowed("execute_query"));
        assert!(policy.is_command_allowed("get_schema"));
        assert!(policy.is_command_allowed("connect"));
        
        // Viewers cannot do admin operations
        assert!(!policy.is_command_allowed("checkpoint"));
        assert!(!policy.is_command_allowed("gc"));
        assert!(!policy.is_command_allowed("compact"));
        
        // Viewers cannot use vector search by default
        assert!(!policy.is_command_allowed("vector_search"));
    }

    #[test]
    fn test_developer_policy() {
        let policy = CommandPolicy::with_role(UserRole::Developer);
        
        // Developers can query, schema, vector
        assert!(policy.is_command_allowed("execute_query"));
        assert!(policy.is_command_allowed("get_schema"));
        assert!(policy.is_command_allowed("vector_search"));
        assert!(policy.is_command_allowed("mcp_call_tool"));
        
        // Developers cannot do admin operations
        assert!(!policy.is_command_allowed("checkpoint"));
        assert!(!policy.is_command_allowed("gc"));
    }

    #[test]
    fn test_admin_policy() {
        let policy = CommandPolicy::admin();
        
        // Admins can do everything
        assert!(policy.is_command_allowed("execute_query"));
        assert!(policy.is_command_allowed("checkpoint"));
        assert!(policy.is_command_allowed("gc"));
        assert!(policy.is_command_allowed("compact"));
        assert!(policy.is_command_allowed("vector_search"));
    }

    #[test]
    fn test_read_only_mode() {
        let policy = CommandPolicy::read_only();
        
        // Read-only can query
        assert!(policy.is_command_allowed("execute_query"));
        assert!(policy.is_command_allowed("get_schema"));
        
        // Read-only cannot mutate
        assert!(!policy.is_command_allowed("checkpoint"));
        assert!(!policy.is_command_allowed("mcp_call_tool"));
    }

    #[test]
    fn test_explicit_deny() {
        let mut policy = CommandPolicy::admin();
        policy.denied_commands.insert("checkpoint".to_string());
        
        // Explicitly denied even for admin
        assert!(!policy.is_command_allowed("checkpoint"));
        assert!(policy.is_command_allowed("gc"));
    }

    #[test]
    fn test_explicit_allow() {
        let mut policy = CommandPolicy::with_role(UserRole::Viewer);
        policy.allowed_commands.insert("vector_search".to_string());
        
        // Explicitly allowed even for viewer
        assert!(policy.is_command_allowed("vector_search"));
    }
}
