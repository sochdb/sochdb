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

//! Namespace-Scoped Query API (Task 2)
//!
//! This module enforces **mandatory namespace scoping** at the type level,
//! making cross-workspace data leakage impossible by construction.
//!
//! ## The Problem
//!
//! When namespace/tenant scoping is treated as an optional filter parameter,
//! developers can accidentally:
//! - Query across workspaces by forgetting to add the namespace filter
//! - Reuse a handle across workspaces in local-first scenarios
//! - Mix data from different tenants in multi-tenant deployments
//!
//! ## The Solution
//!
//! Make `namespace` a **required part of the query identity**, not an
//! optional filter. The type system enforces:
//!
//! 1. `Namespace` is required in every query request
//! 2. `Namespace` must be validated against the capability token
//! 3. "No namespace" is not a valid state
//!
//! ## Multi-Namespace Queries
//!
//! For legitimate multi-namespace queries, use `NamespaceScope::Multiple`
//! which requires explicit authorization for each namespace.
//!
//! ## Example
//!
//! ```ignore
//! // This compiles - namespace is required
//! let query = ScopedQuery::new(
//!     Namespace::new("production"),
//!     QueryOp::VectorSearch { ... }
//! );
//!
//! // This won't compile - no namespace
//! let query = ScopedQuery::new(QueryOp::VectorSearch { ... });  // ERROR!
//! ```

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::filter_ir::{AuthScope, FilterIR, FilterBuilder};

// ============================================================================
// Namespace - Opaque, Validated Identifier
// ============================================================================

/// A validated namespace identifier
///
/// This is an opaque type that can only be constructed via validation,
/// preventing accidental use of invalid namespace strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Namespace(String);

impl Namespace {
    /// Maximum length for a namespace identifier
    pub const MAX_LENGTH: usize = 256;
    
    /// Create a new namespace (validates format)
    ///
    /// # Validation Rules
    /// - Non-empty
    /// - Max 256 characters
    /// - Alphanumeric, underscores, hyphens, and periods only
    /// - Cannot start with a period or hyphen
    pub fn new(name: impl Into<String>) -> Result<Self, NamespaceError> {
        let name = name.into();
        Self::validate(&name)?;
        Ok(Self(name))
    }
    
    /// Create without validation (for internal use only)
    ///
    /// # Safety
    /// Caller must ensure the name is valid.
    #[allow(dead_code)]
    pub(crate) fn new_unchecked(name: impl Into<String>) -> Self {
        Self(name.into())
    }
    
    /// Validate a namespace string
    fn validate(name: &str) -> Result<(), NamespaceError> {
        Self::validate_name(name)
    }

    /// Validate a name string (public, reusable for database/table names too)
    pub fn validate_name(name: &str) -> Result<(), NamespaceError> {
        if name.is_empty() {
            return Err(NamespaceError::Empty);
        }
        
        if name.len() > Self::MAX_LENGTH {
            return Err(NamespaceError::TooLong {
                length: name.len(),
                max: Self::MAX_LENGTH,
            });
        }
        
        // Check first character
        let first = name.chars().next().unwrap();
        if first == '.' || first == '-' {
            return Err(NamespaceError::InvalidStart(first));
        }
        
        // Check all characters
        for (i, ch) in name.chars().enumerate() {
            if !ch.is_alphanumeric() && ch != '_' && ch != '-' && ch != '.' {
                return Err(NamespaceError::InvalidChar { ch, position: i });
            }
        }
        
        Ok(())
    }
    
    /// Get the namespace as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Convert to owned string
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Namespace {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Errors that can occur when creating a namespace
#[derive(Debug, Clone, thiserror::Error)]
pub enum NamespaceError {
    #[error("namespace cannot be empty")]
    Empty,
    
    #[error("namespace too long: {length} > {max}")]
    TooLong { length: usize, max: usize },
    
    #[error("namespace cannot start with '{0}'")]
    InvalidStart(char),
    
    #[error("invalid character '{ch}' at position {position}")]
    InvalidChar { ch: char, position: usize },
}

// ============================================================================
// Namespace Scope - Single or Multiple
// ============================================================================

/// Scope for a query - either single namespace or explicitly multiple
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NamespaceScope {
    /// Query within a single namespace (most common)
    Single(Namespace),
    
    /// Query across multiple namespaces (requires explicit authorization)
    Multiple(Vec<Namespace>),
}

impl NamespaceScope {
    /// Create a single-namespace scope
    pub fn single(ns: Namespace) -> Self {
        Self::Single(ns)
    }
    
    /// Create a multi-namespace scope
    pub fn multiple(namespaces: Vec<Namespace>) -> Result<Self, NamespaceError> {
        if namespaces.is_empty() {
            return Err(NamespaceError::Empty);
        }
        Ok(Self::Multiple(namespaces))
    }
    
    /// Get all namespaces in this scope
    pub fn namespaces(&self) -> Vec<&Namespace> {
        match self {
            Self::Single(ns) => vec![ns],
            Self::Multiple(nss) => nss.iter().collect(),
        }
    }
    
    /// Check if a namespace is in this scope
    pub fn contains(&self, ns: &Namespace) -> bool {
        match self {
            Self::Single(single) => single == ns,
            Self::Multiple(multiple) => multiple.contains(ns),
        }
    }
    
    /// Validate against an auth scope
    pub fn validate_against(&self, auth: &AuthScope) -> Result<(), ScopeError> {
        for ns in self.namespaces() {
            if !auth.is_namespace_allowed(ns.as_str()) {
                return Err(ScopeError::NamespaceNotAllowed(ns.clone()));
            }
        }
        Ok(())
    }
    
    /// Convert to filter IR clauses
    pub fn to_filter_ir(&self) -> FilterIR {
        match self {
            Self::Single(ns) => FilterBuilder::new()
                .namespace(ns.as_str())
                .build(),
            Self::Multiple(nss) => {
                use crate::filter_ir::{FilterAtom, FilterValue};
                FilterIR::from_atom(FilterAtom::in_set(
                    "namespace",
                    nss.iter()
                        .map(|ns| FilterValue::String(ns.as_str().to_string()))
                        .collect(),
                ))
            }
        }
    }
}

impl fmt::Display for NamespaceScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single(ns) => write!(f, "{}", ns),
            Self::Multiple(nss) => {
                let names: Vec<_> = nss.iter().map(|ns| ns.as_str()).collect();
                write!(f, "[{}]", names.join(", "))
            }
        }
    }
}

/// Errors related to namespace scope
#[derive(Debug, Clone, thiserror::Error)]
pub enum ScopeError {
    #[error("namespace not allowed: {0}")]
    NamespaceNotAllowed(Namespace),
    
    #[error("auth scope expired")]
    AuthExpired,
    
    #[error("insufficient capabilities for this operation")]
    InsufficientCapabilities,
}

// ============================================================================
// Scoped Query - Query with Mandatory Namespace
// ============================================================================

/// A query that is always scoped to a namespace
///
/// This type makes cross-workspace queries impossible by construction.
/// Every query MUST specify a namespace scope.
#[derive(Debug, Clone)]
pub struct ScopedQuery<Q> {
    /// The namespace scope (mandatory)
    scope: NamespaceScope,
    
    /// The underlying query operation
    query: Q,
    
    /// User-provided filters (in addition to namespace)
    filters: FilterIR,
}

impl<Q> ScopedQuery<Q> {
    /// Create a new scoped query
    ///
    /// The namespace scope is required - this is the key invariant.
    pub fn new(scope: NamespaceScope, query: Q) -> Self {
        Self {
            scope,
            query,
            filters: FilterIR::all(),
        }
    }
    
    /// Create a single-namespace query (convenience)
    pub fn in_namespace(namespace: Namespace, query: Q) -> Self {
        Self::new(NamespaceScope::Single(namespace), query)
    }
    
    /// Add user filters
    pub fn with_filters(mut self, filters: FilterIR) -> Self {
        self.filters = filters;
        self
    }
    
    /// Get the namespace scope
    pub fn scope(&self) -> &NamespaceScope {
        &self.scope
    }
    
    /// Get the underlying query
    pub fn query(&self) -> &Q {
        &self.query
    }
    
    /// Get user filters
    pub fn filters(&self) -> &FilterIR {
        &self.filters
    }
    
    /// Compute the effective filter (namespace + user filters)
    ///
    /// This is the filter that will be passed to executors.
    pub fn effective_filter(&self) -> FilterIR {
        self.scope.to_filter_ir().and(self.filters.clone())
    }
    
    /// Validate this query against an auth scope
    pub fn validate(&self, auth: &AuthScope) -> Result<(), ScopeError> {
        // Check auth expiry
        if auth.is_expired() {
            return Err(ScopeError::AuthExpired);
        }
        
        // Check namespace access
        self.scope.validate_against(auth)?;
        
        Ok(())
    }
    
    /// Extract the query, consuming self
    pub fn into_query(self) -> Q {
        self.query
    }
}

// ============================================================================
// Query Request - Complete Request with Auth
// ============================================================================

/// A complete query request with authentication
///
/// This is the type that crosses API boundaries. It bundles:
/// - The scoped query (with mandatory namespace)
/// - The auth scope (with capability token)
///
/// This makes it impossible to execute a query without proper auth.
#[derive(Debug, Clone)]
pub struct QueryRequest<Q> {
    /// The scoped query
    query: ScopedQuery<Q>,
    
    /// The auth scope (from capability token)
    auth: Arc<AuthScope>,
}

impl<Q> QueryRequest<Q> {
    /// Create a new query request
    ///
    /// # Validation
    /// This validates the query scope against the auth scope at construction time.
    pub fn new(query: ScopedQuery<Q>, auth: Arc<AuthScope>) -> Result<Self, ScopeError> {
        query.validate(&auth)?;
        Ok(Self { query, auth })
    }
    
    /// Get the scoped query
    pub fn query(&self) -> &ScopedQuery<Q> {
        &self.query
    }
    
    /// Get the auth scope
    pub fn auth(&self) -> &AuthScope {
        &self.auth
    }
    
    /// Compute the complete effective filter
    ///
    /// This combines:
    /// 1. Auth scope constraints (mandatory)
    /// 2. Namespace scope constraints (mandatory)  
    /// 3. User-provided filters (optional)
    pub fn effective_filter(&self) -> FilterIR {
        self.auth.to_filter_ir()
            .and(self.query.effective_filter())
    }
    
    /// Get the namespace scope
    pub fn namespace_scope(&self) -> &NamespaceScope {
        self.query.scope()
    }
}

// ============================================================================
// Convenience Constructors
// ============================================================================

/// Create a namespace (shorthand)
pub fn ns(name: &str) -> Result<Namespace, NamespaceError> {
    Namespace::new(name)
}

/// Create a single-namespace scope (shorthand)
pub fn scope(name: &str) -> Result<NamespaceScope, NamespaceError> {
    Ok(NamespaceScope::Single(Namespace::new(name)?))
}

// ============================================================================
// Database Tier — namespace > database > table (P3.1)
// ============================================================================

/// A database within a namespace.
///
/// Mirrors SurrealDB's three-tier hierarchy: `namespace > database > table`.
/// Each namespace can contain multiple databases, providing logical grouping
/// and isolation of tables within the same namespace.
///
/// ## Example
///
/// ```text
/// namespace "production"
///   ├─ database "app"
///   │   ├─ table "users"
///   │   └─ table "posts"
///   └─ database "analytics"
///       ├─ table "events"
///       └─ table "sessions"
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatabaseId {
    /// Parent namespace
    pub namespace: String,
    /// Database name within the namespace
    pub name: String,
}

impl DatabaseId {
    /// Maximum length for a database identifier
    pub const MAX_LENGTH: usize = 256;

    /// Create a new database identifier.
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Result<Self, NamespaceError> {
        let namespace = namespace.into();
        let name = name.into();
        Namespace::validate_name(&name)?;
        Ok(Self { namespace, name })
    }

    /// Return the fully qualified name: `namespace/database`
    pub fn qualified_name(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }
}

impl fmt::Display for DatabaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.namespace, self.name)
    }
}

/// A fully qualified table path: `namespace/database/table`.
///
/// Used to address tables unambiguously across the entire hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QualifiedTable {
    pub namespace: String,
    pub database: String,
    pub table: String,
}

impl QualifiedTable {
    /// Create a new qualified table path.
    pub fn new(
        namespace: impl Into<String>,
        database: impl Into<String>,
        table: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            database: database.into(),
            table: table.into(),
        }
    }

    /// Return the fully qualified name: `namespace/database/table`
    pub fn qualified_name(&self) -> String {
        format!("{}/{}/{}", self.namespace, self.database, self.table)
    }

    /// Return the storage key prefix for this table.
    /// All row keys under this table are prefixed with this string.
    pub fn storage_prefix(&self) -> String {
        format!("{}:{}:{}", self.namespace, self.database, self.table)
    }
}

impl fmt::Display for QualifiedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.namespace, self.database, self.table)
    }
}

/// Registry tracking the namespace → database → table hierarchy.
///
/// Provides O(1) lookups and enforces naming constraints.
#[derive(Debug, Clone, Default)]
pub struct NamespaceRegistry {
    /// namespace_name → set of database names
    databases: std::collections::HashMap<String, std::collections::HashSet<String>>,
    /// (namespace, database) → set of table names
    tables: std::collections::HashMap<(String, String), std::collections::HashSet<String>>,
}

impl NamespaceRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a namespace (idempotent).
    pub fn create_namespace(&mut self, namespace: &str) -> Result<(), NamespaceError> {
        Namespace::validate_name(namespace)?;
        self.databases.entry(namespace.to_string()).or_default();
        Ok(())
    }

    /// Create a database within a namespace.
    pub fn create_database(&mut self, namespace: &str, database: &str) -> Result<(), NamespaceError> {
        Namespace::validate_name(database)?;
        let dbs = self.databases.entry(namespace.to_string()).or_default();
        dbs.insert(database.to_string());
        self.tables.entry((namespace.to_string(), database.to_string())).or_default();
        Ok(())
    }

    /// Register a table within a namespace/database.
    pub fn create_table(&mut self, namespace: &str, database: &str, table: &str) -> Result<(), NamespaceError> {
        Namespace::validate_name(table)?;
        // Ensure parent database exists
        let dbs = self.databases.entry(namespace.to_string()).or_default();
        dbs.insert(database.to_string());
        let tables = self.tables.entry((namespace.to_string(), database.to_string())).or_default();
        tables.insert(table.to_string());
        Ok(())
    }

    /// List databases in a namespace.
    pub fn list_databases(&self, namespace: &str) -> Vec<&str> {
        self.databases.get(namespace)
            .map(|dbs| dbs.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// List tables in a database.
    pub fn list_tables(&self, namespace: &str, database: &str) -> Vec<&str> {
        self.tables.get(&(namespace.to_string(), database.to_string()))
            .map(|tables| tables.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Check if a namespace exists.
    pub fn namespace_exists(&self, namespace: &str) -> bool {
        self.databases.contains_key(namespace)
    }

    /// Check if a database exists within a namespace.
    pub fn database_exists(&self, namespace: &str, database: &str) -> bool {
        self.databases.get(namespace)
            .map(|dbs| dbs.contains(database))
            .unwrap_or(false)
    }

    /// Check if a table exists within a namespace/database.
    pub fn table_exists(&self, namespace: &str, database: &str, table: &str) -> bool {
        self.tables.get(&(namespace.to_string(), database.to_string()))
            .map(|tables| tables.contains(table))
            .unwrap_or(false)
    }

    /// Drop a database and all its tables.
    pub fn drop_database(&mut self, namespace: &str, database: &str) -> bool {
        self.tables.remove(&(namespace.to_string(), database.to_string()));
        self.databases.get_mut(namespace)
            .map(|dbs| dbs.remove(database))
            .unwrap_or(false)
    }

    /// Drop a table from a database.
    pub fn drop_table(&mut self, namespace: &str, database: &str, table: &str) -> bool {
        self.tables.get_mut(&(namespace.to_string(), database.to_string()))
            .map(|tables| tables.remove(table))
            .unwrap_or(false)
    }

    /// Drop a namespace and all its databases/tables.
    pub fn drop_namespace(&mut self, namespace: &str) -> bool {
        if !self.databases.contains_key(namespace) {
            return false;
        }
        // Remove all tables under this namespace
        let db_names: Vec<String> = self.databases.get(namespace)
            .map(|dbs| dbs.iter().cloned().collect())
            .unwrap_or_default();
        for db in &db_names {
            self.tables.remove(&(namespace.to_string(), db.clone()));
        }
        self.databases.remove(namespace);
        true
    }

    /// Resolve a qualified table path to check it exists.
    pub fn resolve_table(&self, qualified: &QualifiedTable) -> bool {
        self.table_exists(&qualified.namespace, &qualified.database, &qualified.table)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_namespace_validation() {
        // Valid
        assert!(Namespace::new("production").is_ok());
        assert!(Namespace::new("my_namespace").is_ok());
        assert!(Namespace::new("project-123").is_ok());
        assert!(Namespace::new("v1.0.0").is_ok());
        
        // Invalid
        assert!(Namespace::new("").is_err()); // Empty
        assert!(Namespace::new("-starts-with-dash").is_err());
        assert!(Namespace::new(".starts-with-dot").is_err());
        assert!(Namespace::new("has spaces").is_err());
        assert!(Namespace::new("has@symbol").is_err());
    }
    
    #[test]
    fn test_namespace_scope_single() {
        let ns = Namespace::new("production").unwrap();
        let scope = NamespaceScope::single(ns.clone());
        
        assert!(scope.contains(&ns));
        assert!(!scope.contains(&Namespace::new("staging").unwrap()));
    }
    
    #[test]
    fn test_namespace_scope_multiple() {
        let ns1 = Namespace::new("prod").unwrap();
        let ns2 = Namespace::new("staging").unwrap();
        let scope = NamespaceScope::multiple(vec![ns1.clone(), ns2.clone()]).unwrap();
        
        assert!(scope.contains(&ns1));
        assert!(scope.contains(&ns2));
        assert!(!scope.contains(&Namespace::new("dev").unwrap()));
    }
    
    #[test]
    fn test_scope_to_filter_ir() {
        let scope = NamespaceScope::single(Namespace::new("production").unwrap());
        let filter = scope.to_filter_ir();
        
        assert!(filter.constrains_field("namespace"));
        assert_eq!(filter.clauses.len(), 1);
    }
    
    #[test]
    fn test_scoped_query_effective_filter() {
        let ns = Namespace::new("production").unwrap();
        let user_filter = FilterBuilder::new()
            .eq("source", "documents")
            .build();
        
        let query: ScopedQuery<()> = ScopedQuery::in_namespace(ns, ())
            .with_filters(user_filter);
        
        let effective = query.effective_filter();
        assert!(effective.constrains_field("namespace"));
        assert!(effective.constrains_field("source"));
    }
    
    #[test]
    fn test_query_request_validation() {
        let ns = Namespace::new("production").unwrap();
        let query: ScopedQuery<()> = ScopedQuery::in_namespace(ns, ());
        
        // Auth allows production
        let auth = Arc::new(AuthScope::for_namespace("production"));
        assert!(QueryRequest::new(query.clone(), auth).is_ok());
        
        // Auth only allows staging
        let auth2 = Arc::new(AuthScope::for_namespace("staging"));
        assert!(QueryRequest::new(query, auth2).is_err());
    }
    
    #[test]
    fn test_query_request_effective_filter() {
        let ns = Namespace::new("production").unwrap();
        let query: ScopedQuery<()> = ScopedQuery::in_namespace(ns, ())
            .with_filters(FilterBuilder::new().eq("type", "article").build());
        
        let auth = Arc::new(
            AuthScope::for_namespace("production")
                .with_tenant("acme")
        );
        
        let request = QueryRequest::new(query, auth).unwrap();
        let effective = request.effective_filter();
        
        // Should have: namespace (from scope) + tenant_id (from auth) + type (from user)
        assert!(effective.constrains_field("namespace"));
        assert!(effective.constrains_field("tenant_id"));
        assert!(effective.constrains_field("type"));
    }

    // ======== Database Tier Tests (P3.1) ========

    #[test]
    fn test_database_id_creation() {
        let db = DatabaseId::new("production", "app").unwrap();
        assert_eq!(db.namespace, "production");
        assert_eq!(db.name, "app");
        assert_eq!(db.qualified_name(), "production/app");
    }

    #[test]
    fn test_qualified_table() {
        let qt = QualifiedTable::new("production", "app", "users");
        assert_eq!(qt.qualified_name(), "production/app/users");
        assert_eq!(qt.storage_prefix(), "production:app:users");
    }

    #[test]
    fn test_namespace_registry_basic() {
        let mut reg = NamespaceRegistry::new();
        reg.create_namespace("prod").unwrap();
        assert!(reg.namespace_exists("prod"));
        assert!(!reg.namespace_exists("staging"));
    }

    #[test]
    fn test_namespace_registry_databases() {
        let mut reg = NamespaceRegistry::new();
        reg.create_namespace("prod").unwrap();
        reg.create_database("prod", "app").unwrap();
        reg.create_database("prod", "analytics").unwrap();
        assert!(reg.database_exists("prod", "app"));
        assert!(reg.database_exists("prod", "analytics"));
        assert!(!reg.database_exists("prod", "logs"));
        let dbs = reg.list_databases("prod");
        assert_eq!(dbs.len(), 2);
    }

    #[test]
    fn test_namespace_registry_tables() {
        let mut reg = NamespaceRegistry::new();
        reg.create_table("prod", "app", "users").unwrap();
        reg.create_table("prod", "app", "posts").unwrap();
        assert!(reg.table_exists("prod", "app", "users"));
        assert!(reg.table_exists("prod", "app", "posts"));
        assert!(!reg.table_exists("prod", "app", "comments"));
        // database was auto-created
        assert!(reg.database_exists("prod", "app"));
    }

    #[test]
    fn test_namespace_registry_drop() {
        let mut reg = NamespaceRegistry::new();
        reg.create_table("prod", "app", "users").unwrap();
        reg.create_table("prod", "app", "posts").unwrap();
        reg.create_table("prod", "analytics", "events").unwrap();

        // Drop one table
        assert!(reg.drop_table("prod", "app", "users"));
        assert!(!reg.table_exists("prod", "app", "users"));
        assert!(reg.table_exists("prod", "app", "posts"));

        // Drop a database
        assert!(reg.drop_database("prod", "app"));
        assert!(!reg.database_exists("prod", "app"));
        assert!(!reg.table_exists("prod", "app", "posts"));

        // analytics still exists
        assert!(reg.table_exists("prod", "analytics", "events"));

        // Drop entire namespace
        assert!(reg.drop_namespace("prod"));
        assert!(!reg.namespace_exists("prod"));
        assert!(!reg.table_exists("prod", "analytics", "events"));
    }

    #[test]
    fn test_qualified_table_resolve() {
        let mut reg = NamespaceRegistry::new();
        reg.create_table("prod", "app", "users").unwrap();
        let qt = QualifiedTable::new("prod", "app", "users");
        assert!(reg.resolve_table(&qt));
        let missing = QualifiedTable::new("prod", "app", "absent");
        assert!(!reg.resolve_table(&missing));
    }
}
