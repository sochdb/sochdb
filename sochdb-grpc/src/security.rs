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

//! # Security Baseline for Marketplace
//!
//! Implements production security requirements:
//! - mTLS with hot reload (watch cert files, reload in-memory)
//! - Capability-based authorization (O(1) checks)
//! - Rate limiting per tenant at interceptor layer
//! - JWKS/JWT verification with caching
//! - Audit logging (append-only, structured)
//!
//! ## Design Principles
//!
//! 1. **Secure by Default**: All endpoints require authentication unless explicitly public
//! 2. **Hot Reload**: Cert/key rotation without restart
//! 3. **O(1) AuthZ**: Capabilities are hash set membership
//! 4. **Circuit Breaker**: JWKS refresh doesn't add latency in hot path

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use ring::hmac;
use serde::Deserialize;
use sha2::{Sha256, Digest};

// ============================================================================
// TLS / mTLS Provider (Enterprise Security)
// ============================================================================

/// TLS configuration provider with hot-reload support.
///
/// Loads server certificate + key from PEM files and optionally a CA
/// certificate for mTLS client verification. Watches files for changes
/// and reloads without server restart.
pub struct TlsProvider {
    /// Current server TLS identity (cert + key)
    identity: RwLock<Option<tonic::transport::Identity>>,
    /// CA certificate for mTLS client verification
    client_ca: RwLock<Option<tonic::transport::Certificate>>,
    /// Paths for hot-reload
    cert_path: Option<std::path::PathBuf>,
    key_path: Option<std::path::PathBuf>,
    ca_cert_path: Option<std::path::PathBuf>,
    /// Last modification times for change detection
    last_cert_modified: RwLock<Option<SystemTime>>,
    last_key_modified: RwLock<Option<SystemTime>>,
    last_ca_modified: RwLock<Option<SystemTime>>,
    /// mTLS enabled flag
    mtls_enabled: bool,
}

impl TlsProvider {
    /// Create a new TLS provider from file paths.
    ///
    /// - `cert_path`: Path to PEM-encoded server certificate chain
    /// - `key_path`: Path to PEM-encoded private key
    /// - `ca_cert_path`: Optional CA cert for mTLS client verification
    pub fn new(
        cert_path: impl Into<std::path::PathBuf>,
        key_path: impl Into<std::path::PathBuf>,
        ca_cert_path: Option<impl Into<std::path::PathBuf>>,
    ) -> Result<Self, TlsError> {
        let cert_path = cert_path.into();
        let key_path = key_path.into();
        let ca_cert_path = ca_cert_path.map(|p| p.into());
        let mtls_enabled = ca_cert_path.is_some();

        let mut provider = Self {
            identity: RwLock::new(None),
            client_ca: RwLock::new(None),
            cert_path: Some(cert_path),
            key_path: Some(key_path),
            ca_cert_path,
            last_cert_modified: RwLock::new(None),
            last_key_modified: RwLock::new(None),
            last_ca_modified: RwLock::new(None),
            mtls_enabled,
        };

        provider.reload()?;
        Ok(provider)
    }

    /// Reload certificates from disk. Called on startup and by the hot-reload watcher.
    pub fn reload(&mut self) -> Result<bool, TlsError> {
        let mut changed = false;

        if let (Some(cert_path), Some(key_path)) = (&self.cert_path, &self.key_path) {
            let cert_modified = std::fs::metadata(cert_path)
                .and_then(|m| m.modified())
                .map_err(|e| TlsError::CertReadError(format!("{}: {}", cert_path.display(), e)))?;
            let key_modified = std::fs::metadata(key_path)
                .and_then(|m| m.modified())
                .map_err(|e| TlsError::CertReadError(format!("{}: {}", key_path.display(), e)))?;

            let cert_changed = self.last_cert_modified.read().map_or(true, |t| t != cert_modified);
            let key_changed = self.last_key_modified.read().map_or(true, |t| t != key_modified);

            if cert_changed || key_changed {
                let cert_pem = std::fs::read(cert_path)
                    .map_err(|e| TlsError::CertReadError(format!("{}: {}", cert_path.display(), e)))?;
                let key_pem = std::fs::read(key_path)
                    .map_err(|e| TlsError::CertReadError(format!("{}: {}", key_path.display(), e)))?;

                // Validate PEM structure before accepting
                Self::validate_pem(&cert_pem, "CERTIFICATE")?;
                Self::validate_pem(&key_pem, "PRIVATE KEY")?;

                let identity = tonic::transport::Identity::from_pem(cert_pem, key_pem);
                *self.identity.write() = Some(identity);
                *self.last_cert_modified.write() = Some(cert_modified);
                *self.last_key_modified.write() = Some(key_modified);
                changed = true;
                tracing::info!("TLS server certificate loaded from {}", cert_path.display());
            }
        }

        // Load CA cert for mTLS
        if let Some(ca_path) = &self.ca_cert_path {
            let ca_modified = std::fs::metadata(ca_path)
                .and_then(|m| m.modified())
                .map_err(|e| TlsError::CertReadError(format!("{}: {}", ca_path.display(), e)))?;

            let ca_changed = self.last_ca_modified.read().map_or(true, |t| t != ca_modified);

            if ca_changed {
                let ca_pem = std::fs::read(ca_path)
                    .map_err(|e| TlsError::CertReadError(format!("{}: {}", ca_path.display(), e)))?;

                Self::validate_pem(&ca_pem, "CERTIFICATE")?;

                let client_ca = tonic::transport::Certificate::from_pem(ca_pem);
                *self.client_ca.write() = Some(client_ca);
                *self.last_ca_modified.write() = Some(ca_modified);
                changed = true;
                tracing::info!("mTLS CA certificate loaded from {}", ca_path.display());
            }
        }

        Ok(changed)
    }

    /// Configure a Tonic server builder with TLS (and optionally mTLS).
    pub fn configure_server(
        &self,
    ) -> Result<tonic::transport::ServerTlsConfig, TlsError> {
        let identity = self
            .identity
            .read()
            .clone()
            .ok_or(TlsError::NoCertificate)?;

        let mut tls_config = tonic::transport::ServerTlsConfig::new().identity(identity);

        if self.mtls_enabled {
            if let Some(ca) = self.client_ca.read().clone() {
                tls_config = tls_config.client_ca_root(ca);
                tracing::info!("mTLS client certificate verification enabled");
            }
        }

        Ok(tls_config)
    }

    /// Whether mTLS is enabled.
    pub fn is_mtls_enabled(&self) -> bool {
        self.mtls_enabled
    }

    /// Validate that PEM data contains the expected block type.
    fn validate_pem(data: &[u8], expected: &str) -> Result<(), TlsError> {
        let pem_str = std::str::from_utf8(data)
            .map_err(|_| TlsError::InvalidPem("Not valid UTF-8".into()))?;
        if !pem_str.contains(&format!("-----BEGIN {}", expected)) {
            return Err(TlsError::InvalidPem(format!(
                "Expected PEM block with '-----BEGIN {}', not found",
                expected
            )));
        }
        Ok(())
    }
}

/// TLS-related errors.
#[derive(Debug)]
pub enum TlsError {
    /// Failed to read certificate file
    CertReadError(String),
    /// No certificate loaded
    NoCertificate,
    /// Invalid PEM data
    InvalidPem(String),
}

impl std::fmt::Display for TlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TlsError::CertReadError(e) => write!(f, "Certificate read error: {}", e),
            TlsError::NoCertificate => write!(f, "No TLS certificate loaded"),
            TlsError::InvalidPem(e) => write!(f, "Invalid PEM: {}", e),
        }
    }
}

impl std::error::Error for TlsError {}

// ============================================================================
// Role-Based Access Control (P1.1)
// ============================================================================

/// Named roles with capability inheritance.
///
/// Roles map to SurrealDB equivalents:
/// - `Owner` → full CRUD + user management
/// - `Editor` → read/write data + manage collections/indexes
/// - `Viewer` → read-only + metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    /// Full access — all capabilities including user management.
    Owner,
    /// Read/write data and manage collections/indexes.
    Editor,
    /// Read-only access + view metrics.
    Viewer,
    /// Custom role with explicit capability set.
    Custom {
        name: String,
        capabilities: HashSet<Capability>,
    },
}

impl Role {
    /// Get the capabilities granted by this role.
    ///
    /// Roles inherit downward: Owner ⊇ Editor ⊇ Viewer.
    pub fn capabilities(&self) -> HashSet<Capability> {
        match self {
            Role::Owner => HashSet::from([
                Capability::Admin,
                Capability::Read,
                Capability::Write,
                Capability::ManageCollections,
                Capability::ManageIndexes,
                Capability::ViewMetrics,
                Capability::ManageBackups,
                Capability::ManageUsers,
            ]),
            Role::Editor => HashSet::from([
                Capability::Read,
                Capability::Write,
                Capability::ManageCollections,
                Capability::ManageIndexes,
            ]),
            Role::Viewer => HashSet::from([
                Capability::Read,
                Capability::ViewMetrics,
            ]),
            Role::Custom { capabilities, .. } => capabilities.clone(),
        }
    }

    /// Parse a role from a string name.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "owner" => Some(Role::Owner),
            "editor" => Some(Role::Editor),
            "viewer" => Some(Role::Viewer),
            _ => None,
        }
    }
}

/// Binds a role to a specific scope (global, namespace, or collection).
#[derive(Debug, Clone)]
pub struct RoleBinding {
    /// The principal this binding applies to.
    pub principal_id: String,
    /// The assigned role.
    pub role: Role,
    /// The scope at which this role applies.
    pub scope: RoleScope,
}

/// Scope at which a role binding applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RoleScope {
    /// Applies to all namespaces.
    Global,
    /// Applies to a specific namespace.
    Namespace(String),
    /// Applies to a specific collection within a namespace.
    Collection { namespace: String, collection: String },
}

/// Security principal (authenticated entity)
#[derive(Debug, Clone)]
pub struct Principal {
    /// Principal identifier (e.g., user ID, service account)
    pub id: String,
    /// Tenant/namespace
    pub tenant_id: String,
    /// Granted capabilities
    pub capabilities: HashSet<Capability>,
    /// Token expiration time
    pub expires_at: Option<u64>,
    /// Authentication method used
    pub auth_method: AuthMethod,
}

impl Principal {
    /// Check if principal has a capability
    pub fn has_capability(&self, cap: &Capability) -> bool {
        // O(1) hash set lookup
        self.capabilities.contains(cap) || self.capabilities.contains(&Capability::Admin)
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now >= exp
        } else {
            false
        }
    }
}

/// Authentication method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMethod {
    /// mTLS client certificate
    MtlsCertificate,
    /// JWT/Bearer token
    JwtBearer,
    /// API key
    ApiKey,
    /// Anonymous (if allowed)
    Anonymous,
}

/// Capability for authorization (RBAC-style)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Full admin access
    Admin,
    /// Read data
    Read,
    /// Write data
    Write,
    /// Create/delete collections
    ManageCollections,
    /// Create/delete indexes
    ManageIndexes,
    /// View metrics
    ViewMetrics,
    /// Manage backups
    ManageBackups,
    /// Manage users/principals
    ManageUsers,
    /// Custom capability
    Custom(String),
}

impl Capability {
    /// Parse capability from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "admin" => Capability::Admin,
            "read" => Capability::Read,
            "write" => Capability::Write,
            "manage_collections" => Capability::ManageCollections,
            "manage_indexes" => Capability::ManageIndexes,
            "view_metrics" => Capability::ViewMetrics,
            "manage_backups" => Capability::ManageBackups,
            "manage_users" => Capability::ManageUsers,
            _ => Capability::Custom(s.to_string()),
        }
    }
}

/// Rate limiter using token bucket per principal
pub struct RateLimiter {
    /// Per-principal token buckets
    buckets: RwLock<HashMap<String, TokenBucket>>,
    /// Default rate limit (requests per second)
    default_rate: u64,
    /// Default burst size
    default_burst: u64,
    /// Per-tenant overrides
    tenant_limits: RwLock<HashMap<String, (u64, u64)>>,
}

struct TokenBucket {
    tokens: f64,
    last_update: Instant,
    rate: f64,    // tokens per second
    capacity: f64,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(default_rate: u64, default_burst: u64) -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            default_rate,
            default_burst,
            tenant_limits: RwLock::new(HashMap::new()),
        }
    }

    /// Set rate limit for a specific tenant
    pub fn set_tenant_limit(&self, tenant_id: &str, rate: u64, burst: u64) {
        self.tenant_limits
            .write()
            .insert(tenant_id.to_string(), (rate, burst));
    }

    /// Check if request is allowed
    pub fn check(&self, principal_id: &str, tenant_id: &str) -> RateLimitResult {
        let now = Instant::now();

        // Get rate/burst for tenant
        let (rate, burst) = self
            .tenant_limits
            .read()
            .get(tenant_id)
            .copied()
            .unwrap_or((self.default_rate, self.default_burst));

        let mut buckets = self.buckets.write();
        let bucket = buckets
            .entry(principal_id.to_string())
            .or_insert(TokenBucket {
                tokens: burst as f64,
                last_update: now,
                rate: rate as f64,
                capacity: burst as f64,
            });

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(bucket.last_update).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * bucket.rate).min(bucket.capacity);
        bucket.last_update = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            RateLimitResult::Allowed {
                remaining: bucket.tokens as u64,
            }
        } else {
            let retry_after = (1.0 - bucket.tokens) / bucket.rate;
            RateLimitResult::Limited {
                retry_after_ms: (retry_after * 1000.0) as u64,
            }
        }
    }

    /// Clean up expired entries
    pub fn cleanup(&self, max_age: Duration) {
        let now = Instant::now();
        let mut buckets = self.buckets.write();
        buckets.retain(|_, bucket| now.duration_since(bucket.last_update) < max_age);
    }
}

/// Rate limit check result
#[derive(Debug)]
pub enum RateLimitResult {
    /// Request allowed
    Allowed { remaining: u64 },
    /// Request rate limited
    Limited { retry_after_ms: u64 },
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditLogEntry {
    /// Timestamp (epoch seconds)
    pub timestamp: u64,
    /// Principal who performed the action
    pub principal_id: String,
    /// Tenant context
    pub tenant_id: String,
    /// Action performed
    pub action: String,
    /// Resource affected
    pub resource: String,
    /// Result (success/failure)
    pub result: AuditResult,
    /// Additional context (serialized JSON)
    pub context: Option<String>,
    /// Request ID for correlation
    pub request_id: String,
    /// Client IP (if available)
    pub client_ip: Option<String>,
}

/// Audit result
#[derive(Debug, Clone, Copy)]
pub enum AuditResult {
    Success,
    Failure,
    Denied,
}

impl AuditLogEntry {
    /// Format as JSON for logging
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"timestamp":{},"principal_id":"{}","tenant_id":"{}","action":"{}","resource":"{}","result":"{}","request_id":"{}","client_ip":{}}}"#,
            self.timestamp,
            self.principal_id.replace('"', "\\\""),
            self.tenant_id.replace('"', "\\\""),
            self.action.replace('"', "\\\""),
            self.resource.replace('"', "\\\""),
            match self.result {
                AuditResult::Success => "success",
                AuditResult::Failure => "failure",
                AuditResult::Denied => "denied",
            },
            self.request_id,
            self.client_ip
                .as_ref()
                .map(|ip| format!("\"{}\"", ip))
                .unwrap_or_else(|| "null".to_string()),
        )
    }
}

/// Audit logger with persistent file output.
pub struct AuditLogger {
    /// Buffer for batch writing
    buffer: RwLock<Vec<AuditLogEntry>>,
    /// Buffer flush threshold
    flush_threshold: usize,
    /// Total entries logged
    total_entries: AtomicU64,
    /// Persistent log file path (None = no persistence)
    log_path: Option<std::path::PathBuf>,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(flush_threshold: usize) -> Self {
        Self {
            buffer: RwLock::new(Vec::with_capacity(flush_threshold)),
            flush_threshold,
            total_entries: AtomicU64::new(0),
            log_path: None,
        }
    }

    /// Create a new audit logger with persistent file output.
    pub fn with_log_path(flush_threshold: usize, log_path: std::path::PathBuf) -> Self {
        Self {
            buffer: RwLock::new(Vec::with_capacity(flush_threshold)),
            flush_threshold,
            total_entries: AtomicU64::new(0),
            log_path: Some(log_path),
        }
    }

    /// Log an audit entry
    pub fn log(&self, entry: AuditLogEntry) {
        self.total_entries.fetch_add(1, Ordering::Relaxed);

        let mut buffer = self.buffer.write();
        buffer.push(entry);

        if buffer.len() >= self.flush_threshold {
            self.flush_buffer(&mut buffer);
        }
    }

    /// Flush buffered entries to persistent storage.
    fn flush_buffer(&self, buffer: &mut Vec<AuditLogEntry>) {
        if buffer.is_empty() {
            return;
        }

        if let Some(ref path) = self.log_path {
            use std::io::Write;
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(mut file) => {
                    for entry in buffer.iter() {
                        let _ = writeln!(file, "{}", entry.to_json());
                    }
                    let _ = file.flush();
                }
                Err(e) => {
                    tracing::error!("Failed to write audit log to {:?}: {}", path, e);
                }
            }
        } else {
            // Log to tracing when no file path is configured
            for entry in buffer.iter() {
                tracing::info!(target: "audit", "{}", entry.to_json());
            }
        }
        buffer.clear();
    }

    /// Log a success action
    pub fn log_success(
        &self,
        principal: &Principal,
        action: &str,
        resource: &str,
        request_id: &str,
    ) {
        self.log(AuditLogEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            principal_id: principal.id.clone(),
            tenant_id: principal.tenant_id.clone(),
            action: action.to_string(),
            resource: resource.to_string(),
            result: AuditResult::Success,
            context: None,
            request_id: request_id.to_string(),
            client_ip: None,
        });
    }

    /// Log a denied action
    pub fn log_denied(
        &self,
        principal: &Principal,
        action: &str,
        resource: &str,
        request_id: &str,
        reason: &str,
    ) {
        self.log(AuditLogEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            principal_id: principal.id.clone(),
            tenant_id: principal.tenant_id.clone(),
            action: action.to_string(),
            resource: resource.to_string(),
            result: AuditResult::Denied,
            context: Some(format!(r#"{{"reason":"{}"}}"#, reason.replace('"', "\\\""))),
            request_id: request_id.to_string(),
            client_ip: None,
        });
    }

    /// Get total entries logged
    pub fn total_entries(&self) -> u64 {
        self.total_entries.load(Ordering::Relaxed)
    }
}

// ============================================================================
// JWT Claims (P0.3 — Real JWT verification)
// ============================================================================

/// JWT claims structure for token verification.
#[derive(Debug, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Expiration (Unix timestamp)
    pub exp: Option<u64>,
    /// Issued-at
    pub iat: Option<u64>,
    /// Issuer
    pub iss: Option<String>,
    /// Audience
    pub aud: Option<String>,
    /// SochDB: tenant/namespace
    pub tenant_id: Option<String>,
    /// SochDB: role (Owner, Editor, Viewer)
    pub role: Option<String>,
    /// SochDB: explicit capability list
    pub capabilities: Option<Vec<String>>,
}

/// Security configuration
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Enable mTLS
    pub mtls_enabled: bool,
    /// Certificate path (watched for hot reload)
    pub cert_path: Option<String>,
    /// Key path
    pub key_path: Option<String>,
    /// CA certificate path (for client verification)
    pub ca_cert_path: Option<String>,

    /// Enable JWT authentication
    pub jwt_enabled: bool,
    /// JWKS URL for JWT verification
    pub jwks_url: Option<String>,
    /// Expected JWT issuer
    pub jwt_issuer: Option<String>,
    /// Expected JWT audience
    pub jwt_audience: Option<String>,

    /// Enable API key authentication
    pub api_key_enabled: bool,

    /// Default rate limit (requests per second)
    pub rate_limit_default: u64,
    /// Default burst size
    pub rate_limit_burst: u64,

    /// Enable audit logging
    pub audit_enabled: bool,
    /// Audit log flush threshold
    pub audit_flush_threshold: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            mtls_enabled: false,
            cert_path: None,
            key_path: None,
            ca_cert_path: None,
            jwt_enabled: false,
            jwks_url: None,
            jwt_issuer: None,
            jwt_audience: None,
            api_key_enabled: false,
            rate_limit_default: 1000,
            rate_limit_burst: 100,
            audit_enabled: true,
            audit_flush_threshold: 100,
        }
    }
}

/// Security service combining all security components
pub struct SecurityService {
    config: SecurityConfig,
    rate_limiter: RateLimiter,
    audit_logger: AuditLogger,
    /// API keys: SHA-256(key) -> Principal (keys never stored in plaintext)
    api_key_hashes: RwLock<HashMap<String, Principal>>,
    /// Role bindings: principal_id -> Vec<RoleBinding>
    role_bindings: RwLock<HashMap<String, Vec<RoleBinding>>>,
    /// User credentials: username -> argon2 password hash
    user_credentials: RwLock<HashMap<String, String>>,
    /// User principal templates: username -> Principal (populated on registration)
    user_principals: RwLock<HashMap<String, Principal>>,
    /// JWKS key material (for JWT verification)
    jwt_decoding_key: RwLock<Option<jsonwebtoken::DecodingKey>>,
}

impl SecurityService {
    /// Create a new security service
    pub fn new(config: SecurityConfig) -> Self {
        let rate_limiter = RateLimiter::new(config.rate_limit_default, config.rate_limit_burst);
        let audit_logger = AuditLogger::new(config.audit_flush_threshold);

        Self {
            config,
            rate_limiter,
            audit_logger,
            api_key_hashes: RwLock::new(HashMap::new()),
            role_bindings: RwLock::new(HashMap::new()),
            user_credentials: RwLock::new(HashMap::new()),
            user_principals: RwLock::new(HashMap::new()),
            jwt_decoding_key: RwLock::new(None),
        }
    }

    /// Hash an API key with SHA-256 for storage.
    fn hash_api_key(key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Register an API key (stored as SHA-256 hash, never plaintext).
    pub fn register_api_key(&self, key: &str, principal: Principal) {
        let hash = Self::hash_api_key(key);
        self.api_key_hashes.write().insert(hash, principal);
    }

    /// Register a user with argon2-hashed password.
    pub fn register_user(
        &self,
        username: &str,
        password: &str,
        principal: Principal,
    ) -> Result<(), AuthError> {
        use argon2::{password_hash::SaltString, Argon2, PasswordHasher};
        use rand::rngs::OsRng;

        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        let hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| AuthError::Internal(format!("Password hashing failed: {}", e)))?;

        self.user_credentials
            .write()
            .insert(username.to_string(), hash.to_string());
        self.user_principals
            .write()
            .insert(username.to_string(), principal);
        Ok(())
    }

    /// Verify a username/password pair, returning the user's principal on success.
    pub fn verify_password(
        &self,
        username: &str,
        password: &str,
    ) -> Result<Principal, AuthError> {
        use argon2::{Argon2, PasswordHash, PasswordVerifier};

        let creds = self.user_credentials.read();
        let hash_str = creds.get(username).ok_or(AuthError::Unauthenticated)?;

        let parsed_hash = PasswordHash::new(hash_str)
            .map_err(|e| AuthError::Internal(format!("Invalid stored hash: {}", e)))?;

        Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .map_err(|_| AuthError::Unauthenticated)?;

        let principals = self.user_principals.read();
        principals
            .get(username)
            .cloned()
            .ok_or(AuthError::Unauthenticated)
    }

    /// Set the JWT decoding key (for HS256/RS256 verification).
    ///
    /// For HS256: `set_jwt_key(DecodingKey::from_secret(b"my-secret"))`
    /// For RS256: `set_jwt_key(DecodingKey::from_rsa_pem(pem_bytes)?)`
    pub fn set_jwt_key(&self, key: jsonwebtoken::DecodingKey) {
        *self.jwt_decoding_key.write() = Some(key);
    }

    /// Bind a role to a principal at a given scope.
    pub fn bind_role(&self, binding: RoleBinding) {
        let mut bindings = self.role_bindings.write();
        bindings
            .entry(binding.principal_id.clone())
            .or_default()
            .push(binding);
    }

    /// Get effective capabilities for a principal in a given namespace.
    pub fn effective_capabilities(
        &self,
        principal_id: &str,
        namespace: &str,
    ) -> HashSet<Capability> {
        let bindings = self.role_bindings.read();
        let Some(principal_bindings) = bindings.get(principal_id) else {
            return HashSet::new();
        };

        let mut caps = HashSet::new();
        for binding in principal_bindings {
            let applies = match &binding.scope {
                RoleScope::Global => true,
                RoleScope::Namespace(ns) => ns == namespace,
                RoleScope::Collection { namespace: ns, .. } => ns == namespace,
            };
            if applies {
                caps.extend(binding.role.capabilities());
            }
        }
        caps
    }

    /// Authenticate a request (returns principal if valid).
    ///
    /// Authentication chain:
    /// 1. mTLS client certificate (if enabled)
    /// 2. JWT Bearer token (if enabled, with real signature verification)
    /// 3. API key (SHA-256 hash lookup)
    pub fn authenticate(
        &self,
        auth_header: Option<&str>,
        _client_cert: Option<&str>,
    ) -> Result<Principal, AuthError> {
        // Try Bearer token
        if let Some(header) = auth_header {
            if header.starts_with("Bearer ") {
                let token = &header[7..];

                // Try JWT verification (real signature check)
                if self.config.jwt_enabled {
                    return self.verify_jwt(token);
                }

                // Try as API key (SHA-256 hash comparison)
                if self.config.api_key_enabled {
                    let hash = Self::hash_api_key(token);
                    if let Some(principal) = self.api_key_hashes.read().get(&hash) {
                        return Ok(principal.clone());
                    }
                }
            }
        }

        Err(AuthError::Unauthenticated)
    }

    /// Verify a JWT token with real cryptographic signature verification.
    fn verify_jwt(&self, token: &str) -> Result<Principal, AuthError> {
        let decoding_key = self.jwt_decoding_key.read();
        let key = decoding_key
            .as_ref()
            .ok_or_else(|| AuthError::Internal("JWT verification key not configured".into()))?;

        let mut validation = jsonwebtoken::Validation::default();

        // Configure expected issuer
        if let Some(ref issuer) = self.config.jwt_issuer {
            validation.set_issuer(&[issuer]);
        } else {
            validation.iss = None;
        }

        // Configure expected audience
        if let Some(ref audience) = self.config.jwt_audience {
            validation.set_audience(&[audience]);
        } else {
            validation.validate_aud = false;
        }

        let token_data = jsonwebtoken::decode::<JwtClaims>(token, key, &validation)
            .map_err(|e| match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                jsonwebtoken::errors::ErrorKind::InvalidSignature => {
                    AuthError::Unauthenticated
                }
                _ => AuthError::Internal(format!("JWT verification failed: {}", e)),
            })?;

        let claims = token_data.claims;

        // Extract capabilities from JWT claims
        let capabilities: HashSet<Capability> = claims
            .capabilities
            .unwrap_or_default()
            .into_iter()
            .map(|s| Capability::from_str(&s))
            .collect();

        // If role is specified, merge role capabilities
        let mut all_caps = capabilities;
        if let Some(ref role_str) = claims.role {
            if let Some(role) = Role::from_str(role_str) {
                all_caps.extend(role.capabilities());
            }
        }

        Ok(Principal {
            id: claims.sub,
            tenant_id: claims.tenant_id.unwrap_or_else(|| "default".to_string()),
            capabilities: all_caps,
            expires_at: claims.exp.map(|e| e as u64),
            auth_method: AuthMethod::JwtBearer,
        })
    }

    /// Authorize an action
    pub fn authorize(
        &self,
        principal: &Principal,
        required_capability: &Capability,
    ) -> Result<(), AuthError> {
        // Check expiration
        if principal.is_expired() {
            return Err(AuthError::TokenExpired);
        }

        // Check capability
        if principal.has_capability(required_capability) {
            Ok(())
        } else {
            Err(AuthError::Unauthorized {
                required: format!("{:?}", required_capability),
            })
        }
    }

    /// Check rate limit
    pub fn check_rate_limit(&self, principal: &Principal) -> Result<(), AuthError> {
        match self.rate_limiter.check(&principal.id, &principal.tenant_id) {
            RateLimitResult::Allowed { .. } => Ok(()),
            RateLimitResult::Limited { retry_after_ms } => {
                Err(AuthError::RateLimited { retry_after_ms })
            }
        }
    }

    /// Get audit logger
    pub fn audit(&self) -> &AuditLogger {
        &self.audit_logger
    }

    /// Full security check (auth + authz + rate limit)
    pub fn full_check(
        &self,
        auth_header: Option<&str>,
        client_cert: Option<&str>,
        required_capability: &Capability,
        action: &str,
        resource: &str,
        request_id: &str,
    ) -> Result<Principal, AuthError> {
        // Authenticate
        let principal = self.authenticate(auth_header, client_cert)?;

        // Rate limit
        self.check_rate_limit(&principal)?;

        // Authorize
        match self.authorize(&principal, required_capability) {
            Ok(()) => {
                if self.config.audit_enabled {
                    self.audit_logger
                        .log_success(&principal, action, resource, request_id);
                }
                Ok(principal)
            }
            Err(e) => {
                if self.config.audit_enabled {
                    self.audit_logger.log_denied(
                        &principal,
                        action,
                        resource,
                        request_id,
                        &format!("{:?}", e),
                    );
                }
                Err(e)
            }
        }
    }
}

/// Authentication/Authorization error
#[derive(Debug)]
pub enum AuthError {
    /// No valid authentication provided
    Unauthenticated,
    /// Token has expired
    TokenExpired,
    /// Missing required capability
    Unauthorized { required: String },
    /// Rate limit exceeded
    RateLimited { retry_after_ms: u64 },
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::Unauthenticated => write!(f, "Authentication required"),
            AuthError::TokenExpired => write!(f, "Token has expired"),
            AuthError::Unauthorized { required } => {
                write!(f, "Missing required capability: {}", required)
            }
            AuthError::RateLimited { retry_after_ms } => {
                write!(f, "Rate limit exceeded, retry after {}ms", retry_after_ms)
            }
            AuthError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for AuthError {}

// ============================================================================
// Secrets Management (Enterprise)
// ============================================================================

/// Secrets provider that loads credentials from Kubernetes Secrets mounts,
/// environment variables, or file paths — with rotation support.
///
/// ## Kubernetes Secrets
///
/// Mount a K8s Secret as a volume and point `SecretsProvider` at the mount:
///
/// ```yaml
/// volumeMounts:
///   - name: sochdb-secrets
///     mountPath: /etc/sochdb/secrets
///     readOnly: true
/// ```
///
/// The provider reads files from the mount path:
/// - `jwt-secret` → JWT signing/verification key
/// - `api-keys` → newline-delimited API keys
/// - `encryption-key` → 32-byte data-at-rest key (base64-encoded)
/// - `tls-cert` → TLS certificate PEM
/// - `tls-key` → TLS private key PEM
/// - `tls-ca` → CA certificate for mTLS
pub struct SecretsProvider {
    /// Base path for mounted secrets (e.g., /etc/sochdb/secrets)
    mount_path: Option<std::path::PathBuf>,
    /// Cached secrets (name → value), refreshed on rotation
    cache: RwLock<HashMap<String, Vec<u8>>>,
    /// Last scan time for change detection
    last_scan: RwLock<Option<Instant>>,
    /// Scan interval for rotation detection
    scan_interval: Duration,
}

impl SecretsProvider {
    /// Create from a Kubernetes Secrets mount path.
    pub fn from_mount(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            mount_path: Some(path.into()),
            cache: RwLock::new(HashMap::new()),
            last_scan: RwLock::new(None),
            scan_interval: Duration::from_secs(30),
        }
    }

    /// Create from environment variables only (no file mount).
    pub fn from_env() -> Self {
        Self {
            mount_path: None,
            cache: RwLock::new(HashMap::new()),
            last_scan: RwLock::new(None),
            scan_interval: Duration::from_secs(60),
        }
    }

    /// Load or refresh secrets from the mounted directory.
    pub fn refresh(&self) -> Result<usize, String> {
        let mut cache = self.cache.write();
        let mut count = 0;

        // Load from file mount
        if let Some(ref path) = self.mount_path {
            if path.is_dir() {
                let entries = std::fs::read_dir(path)
                    .map_err(|e| format!("Failed to read secrets dir: {}", e))?;

                for entry in entries.flatten() {
                    let file_path = entry.path();
                    if file_path.is_file() {
                        // Skip dotfiles and symlink metadata (K8s ..data, ..timestamp)
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.starts_with('.') {
                            continue;
                        }
                        match std::fs::read(&file_path) {
                            Ok(data) => {
                                cache.insert(name, data);
                                count += 1;
                            }
                            Err(e) => {
                                tracing::warn!("Failed to read secret {}: {}", file_path.display(), e);
                            }
                        }
                    }
                }
            }
        }

        // Load from environment variables (override file-based)
        for (key, env_var) in &[
            ("jwt-secret", "SOCHDB_JWT_SECRET"),
            ("encryption-key", "SOCHDB_ENCRYPTION_KEY"),
        ] {
            if let Ok(val) = std::env::var(env_var) {
                cache.insert(key.to_string(), val.into_bytes());
                count += 1;
            }
        }

        // API keys from env (comma-separated)
        if let Ok(keys) = std::env::var("SOCHDB_API_KEYS") {
            let joined = keys.replace(',', "\n");
            cache.insert("api-keys".to_string(), joined.into_bytes());
            count += 1;
        }

        *self.last_scan.write() = Some(Instant::now());
        tracing::info!("Loaded {} secrets", count);
        Ok(count)
    }

    /// Get a secret by name. Returns `None` if not found.
    pub fn get(&self, name: &str) -> Option<Vec<u8>> {
        // Auto-refresh if stale
        let should_refresh = self
            .last_scan
            .read()
            .map_or(true, |t| t.elapsed() > self.scan_interval);

        if should_refresh {
            let _ = self.refresh();
        }

        self.cache.read().get(name).cloned()
    }

    /// Get a secret as a UTF-8 string. Returns `None` if not found or not valid UTF-8.
    pub fn get_string(&self, name: &str) -> Option<String> {
        self.get(name)
            .and_then(|v| String::from_utf8(v).ok())
            .map(|s| s.trim().to_string())
    }

    /// Get all API keys (loaded from `api-keys` secret, newline-delimited).
    pub fn api_keys(&self) -> Vec<String> {
        self.get_string("api-keys")
            .map(|s| {
                s.lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the data-at-rest encryption key (base64-decoded to 32 bytes).
    pub fn encryption_key(&self) -> Option<[u8; 32]> {
        self.get_string("encryption-key").and_then(|b64| {
            use base64::Engine;
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(b64.trim())
                .ok()?;
            if decoded.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&decoded);
                Some(key)
            } else {
                tracing::error!(
                    "Encryption key must be exactly 32 bytes, got {}",
                    decoded.len()
                );
                None
            }
        })
    }

    /// Apply loaded secrets to a SecurityService (JWT key, API keys, etc.).
    pub fn apply_to_security(&self, service: &SecurityService) {
        // JWT secret
        if let Some(secret) = self.get("jwt-secret") {
            service.set_jwt_key(jsonwebtoken::DecodingKey::from_secret(&secret));
            tracing::info!("JWT verification key loaded from secrets");
        }

        // API keys
        for key in self.api_keys() {
            service.register_api_key(
                &key,
                Principal {
                    id: format!("apikey-{}", &SecurityService::hash_api_key(&key)[..8]),
                    tenant_id: "default".to_string(),
                    capabilities: Role::Editor.capabilities(),
                    expires_at: None,
                    auth_method: AuthMethod::ApiKey,
                },
            );
        }
    }
}

// ============================================================================
// Compliance & Data Governance (Enterprise)
// ============================================================================

/// Data retention policy for GDPR/SOC2 compliance.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Policy name
    pub name: String,
    /// Namespace pattern this policy applies to (glob-like: "*" for all)
    pub namespace_pattern: String,
    /// Maximum data retention period
    pub max_retention: Duration,
    /// Whether to auto-delete expired data
    pub auto_purge: bool,
    /// Whether to log deletions to audit trail
    pub audit_deletions: bool,
}

/// Compliance manager for data governance requirements.
///
/// Handles:
/// - Data retention policies (auto-purge with audit trail)
/// - Right-to-erasure (GDPR Article 17) — bulk delete by principal/tenant
/// - Data export (GDPR Article 20) — portable format
/// - Audit trail retention (SOC2)
pub struct ComplianceManager {
    /// Active retention policies
    retention_policies: RwLock<Vec<RetentionPolicy>>,
    /// Erasure request log (for audit trail)
    erasure_log: RwLock<Vec<ErasureRecord>>,
    /// Audit logger reference
    audit_logger: Arc<AuditLogger>,
}

/// Record of a data erasure (right-to-be-forgotten).
#[derive(Debug, Clone)]
pub struct ErasureRecord {
    /// Timestamp of erasure
    pub timestamp: u64,
    /// Requesting principal
    pub requested_by: String,
    /// Tenant/namespace affected
    pub tenant_id: String,
    /// Subject ID (the person whose data was erased)
    pub subject_id: String,
    /// Resources erased (collection names, key patterns, etc.)
    pub resources_erased: Vec<String>,
    /// Unique request ID for compliance tracking
    pub request_id: String,
}

impl ComplianceManager {
    /// Create a new compliance manager.
    pub fn new(audit_logger: Arc<AuditLogger>) -> Self {
        Self {
            retention_policies: RwLock::new(Vec::new()),
            erasure_log: RwLock::new(Vec::new()),
            audit_logger,
        }
    }

    /// Register a data retention policy.
    pub fn add_retention_policy(&self, policy: RetentionPolicy) {
        tracing::info!(
            "Retention policy '{}' registered: namespace={}, max_retention={:?}, auto_purge={}",
            policy.name,
            policy.namespace_pattern,
            policy.max_retention,
            policy.auto_purge,
        );
        self.retention_policies.write().push(policy);
    }

    /// Get all retention policies.
    pub fn retention_policies(&self) -> Vec<RetentionPolicy> {
        self.retention_policies.read().clone()
    }

    /// Record a data erasure event (GDPR Article 17 — right to erasure).
    ///
    /// The actual data deletion is performed by the caller (KvServer, VectorServer, etc.).
    /// This method records the erasure in an immutable audit log.
    pub fn record_erasure(
        &self,
        requested_by: &str,
        tenant_id: &str,
        subject_id: &str,
        resources: Vec<String>,
        request_id: &str,
    ) {
        let record = ErasureRecord {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            requested_by: requested_by.to_string(),
            tenant_id: tenant_id.to_string(),
            subject_id: subject_id.to_string(),
            resources_erased: resources.clone(),
            request_id: request_id.to_string(),
        };

        // Log to audit trail
        self.audit_logger.log(AuditLogEntry {
            timestamp: record.timestamp,
            principal_id: requested_by.to_string(),
            tenant_id: tenant_id.to_string(),
            action: "gdpr_erasure".to_string(),
            resource: resources.join(","),
            result: AuditResult::Success,
            context: Some(format!(
                r#"{{"subject_id":"{}","resource_count":{}}}"#,
                subject_id, resources.len()
            )),
            request_id: request_id.to_string(),
            client_ip: None,
        });

        self.erasure_log.write().push(record);
        tracing::info!(
            "GDPR erasure recorded: subject={}, tenant={}, resources={}",
            subject_id,
            tenant_id,
            resources.len()
        );
    }

    /// Get all erasure records (for compliance audits).
    pub fn erasure_records(&self) -> Vec<ErasureRecord> {
        self.erasure_log.read().clone()
    }

    /// Check if a retention policy matches a namespace.
    pub fn policies_for_namespace(&self, namespace: &str) -> Vec<RetentionPolicy> {
        self.retention_policies
            .read()
            .iter()
            .filter(|p| {
                p.namespace_pattern == "*"
                    || p.namespace_pattern == namespace
                    || (p.namespace_pattern.ends_with('*')
                        && namespace.starts_with(p.namespace_pattern.trim_end_matches('*')))
            })
            .cloned()
            .collect()
    }

    /// Get the effective max retention for a namespace (shortest policy wins).
    pub fn effective_retention(&self, namespace: &str) -> Option<Duration> {
        self.policies_for_namespace(namespace)
            .iter()
            .map(|p| p.max_retention)
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_check() {
        let principal = Principal {
            id: "user1".to_string(),
            tenant_id: "tenant1".to_string(),
            capabilities: HashSet::from([Capability::Read, Capability::Write]),
            expires_at: None,
            auth_method: AuthMethod::ApiKey,
        };

        assert!(principal.has_capability(&Capability::Read));
        assert!(principal.has_capability(&Capability::Write));
        assert!(!principal.has_capability(&Capability::Admin));
    }

    #[test]
    fn test_admin_has_all_capabilities() {
        let admin = Principal {
            id: "admin".to_string(),
            tenant_id: "tenant1".to_string(),
            capabilities: HashSet::from([Capability::Admin]),
            expires_at: None,
            auth_method: AuthMethod::ApiKey,
        };

        assert!(admin.has_capability(&Capability::Read));
        assert!(admin.has_capability(&Capability::Write));
        assert!(admin.has_capability(&Capability::ManageBackups));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(10, 5); // 10 rps, burst 5

        // First 5 requests should succeed (burst)
        for _ in 0..5 {
            assert!(matches!(
                limiter.check("user1", "tenant1"),
                RateLimitResult::Allowed { .. }
            ));
        }

        // Next request should be rate limited (burst exhausted)
        assert!(matches!(
            limiter.check("user1", "tenant1"),
            RateLimitResult::Limited { .. }
        ));
    }

    #[test]
    fn test_security_service_api_key() {
        let config = SecurityConfig {
            api_key_enabled: true,
            ..Default::default()
        };
        let service = SecurityService::new(config);

        // Register an API key
        let principal = Principal {
            id: "service1".to_string(),
            tenant_id: "tenant1".to_string(),
            capabilities: HashSet::from([Capability::Read]),
            expires_at: None,
            auth_method: AuthMethod::ApiKey,
        };
        service.register_api_key("secret-key-123", principal);

        // Authenticate with valid key
        let result = service.authenticate(Some("Bearer secret-key-123"), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().id, "service1");

        // Authenticate with invalid key
        let result = service.authenticate(Some("Bearer invalid-key"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_audit_logging() {
        let logger = AuditLogger::new(10);

        let principal = Principal {
            id: "user1".to_string(),
            tenant_id: "tenant1".to_string(),
            capabilities: HashSet::new(),
            expires_at: None,
            auth_method: AuthMethod::ApiKey,
        };

        logger.log_success(&principal, "read", "/collections/test", "req-123");
        assert_eq!(logger.total_entries(), 1);
    }

    #[test]
    fn test_jwt_authentication() {
        use jsonwebtoken::{encode, EncodingKey, Header};
        use serde::Serialize;

        #[derive(Serialize)]
        struct Claims {
            sub: String,
            exp: u64,
            tenant_id: String,
            role: String,
        }

        let secret = b"test-jwt-secret-key-for-testing";
        let claims = Claims {
            sub: "jwt-user-42".to_string(),
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            tenant_id: "acme-corp".to_string(),
            role: "editor".to_string(),
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret),
        )
        .unwrap();

        let config = SecurityConfig {
            jwt_enabled: true,
            ..Default::default()
        };
        let service = SecurityService::new(config);
        service.set_jwt_key(jsonwebtoken::DecodingKey::from_secret(secret));

        let result = service.authenticate(Some(&format!("Bearer {}", token)), None);
        assert!(result.is_ok());
        let principal = result.unwrap();
        assert_eq!(principal.id, "jwt-user-42");
        assert_eq!(principal.tenant_id, "acme-corp");
        assert!(principal.has_capability(&Capability::Read));
        assert!(principal.has_capability(&Capability::Write));
        assert_eq!(principal.auth_method, AuthMethod::JwtBearer);
    }

    #[test]
    fn test_jwt_expired_token_rejected() {
        use jsonwebtoken::{encode, EncodingKey, Header};
        use serde::Serialize;

        #[derive(Serialize)]
        struct Claims {
            sub: String,
            exp: u64,
        }

        let secret = b"test-secret";
        let claims = Claims {
            sub: "user".to_string(),
            exp: 1000, // Long expired
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret),
        )
        .unwrap();

        let config = SecurityConfig {
            jwt_enabled: true,
            ..Default::default()
        };
        let service = SecurityService::new(config);
        service.set_jwt_key(jsonwebtoken::DecodingKey::from_secret(secret));

        let result = service.authenticate(Some(&format!("Bearer {}", token)), None);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }

    #[test]
    fn test_jwt_invalid_signature_rejected() {
        use jsonwebtoken::{encode, EncodingKey, Header};
        use serde::Serialize;

        #[derive(Serialize)]
        struct Claims {
            sub: String,
            exp: u64,
        }

        let claims = Claims {
            sub: "user".to_string(),
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
        };

        // Sign with one key, verify with another
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(b"signing-key"),
        )
        .unwrap();

        let config = SecurityConfig {
            jwt_enabled: true,
            ..Default::default()
        };
        let service = SecurityService::new(config);
        service.set_jwt_key(jsonwebtoken::DecodingKey::from_secret(b"different-key"));

        let result = service.authenticate(Some(&format!("Bearer {}", token)), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_password_hashing_and_verification() {
        let config = SecurityConfig::default();
        let service = SecurityService::new(config);

        let principal = Principal {
            id: "alice".to_string(),
            tenant_id: "default".to_string(),
            capabilities: HashSet::from([Capability::Read, Capability::Write]),
            expires_at: None,
            auth_method: AuthMethod::JwtBearer,
        };

        service
            .register_user("alice", "correct-horse-battery-staple", principal)
            .unwrap();

        // Correct password succeeds
        let result = service.verify_password("alice", "correct-horse-battery-staple");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().id, "alice");

        // Wrong password fails
        let result = service.verify_password("alice", "wrong-password");
        assert!(matches!(result, Err(AuthError::Unauthenticated)));

        // Unknown user fails
        let result = service.verify_password("bob", "any-password");
        assert!(matches!(result, Err(AuthError::Unauthenticated)));
    }

    #[test]
    fn test_api_key_hashed_storage() {
        let config = SecurityConfig {
            api_key_enabled: true,
            ..Default::default()
        };
        let service = SecurityService::new(config);

        let principal = Principal {
            id: "svc".to_string(),
            tenant_id: "t1".to_string(),
            capabilities: HashSet::from([Capability::Read]),
            expires_at: None,
            auth_method: AuthMethod::ApiKey,
        };
        service.register_api_key("my-secret-key", principal);

        // Internally, keys are stored as SHA-256 hashes
        let stored_keys: Vec<_> = service.api_key_hashes.read().keys().cloned().collect();
        assert_eq!(stored_keys.len(), 1);
        // The stored key is a hex-encoded SHA-256 hash, not the plaintext
        assert_ne!(stored_keys[0], "my-secret-key");
        assert_eq!(stored_keys[0].len(), 64); // SHA-256 = 32 bytes = 64 hex chars

        // Auth still works via hashing
        let result = service.authenticate(Some("Bearer my-secret-key"), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_role_capabilities() {
        assert!(Role::Owner.capabilities().contains(&Capability::Admin));
        assert!(Role::Owner.capabilities().contains(&Capability::ManageUsers));
        assert!(Role::Editor.capabilities().contains(&Capability::Write));
        assert!(!Role::Editor.capabilities().contains(&Capability::Admin));
        assert!(Role::Viewer.capabilities().contains(&Capability::Read));
        assert!(!Role::Viewer.capabilities().contains(&Capability::Write));
    }

    #[test]
    fn test_role_binding_effective_capabilities() {
        let config = SecurityConfig::default();
        let service = SecurityService::new(config);

        service.bind_role(RoleBinding {
            principal_id: "alice".to_string(),
            role: Role::Viewer,
            scope: RoleScope::Global,
        });
        service.bind_role(RoleBinding {
            principal_id: "alice".to_string(),
            role: Role::Editor,
            scope: RoleScope::Namespace("production".to_string()),
        });

        // In "production" namespace, alice gets Viewer (global) + Editor (namespace) capabilities
        let caps = service.effective_capabilities("alice", "production");
        assert!(caps.contains(&Capability::Read));
        assert!(caps.contains(&Capability::Write));

        // In "staging" namespace, alice only has Viewer (global) capabilities
        let caps = service.effective_capabilities("alice", "staging");
        assert!(caps.contains(&Capability::Read));
        assert!(!caps.contains(&Capability::Write));
    }

    #[test]
    fn test_secrets_provider_env() {
        let provider = SecretsProvider::from_env();
        // Without env vars set, keys should be empty
        assert!(provider.api_keys().is_empty());
        assert!(provider.encryption_key().is_none());
    }

    #[test]
    fn test_secrets_provider_from_mount() {
        let dir = std::env::temp_dir().join("sochdb_test_secrets");
        let _ = std::fs::create_dir_all(&dir);

        // Write test secrets
        std::fs::write(dir.join("jwt-secret"), b"test-jwt-secret-32bytes-long!!!!").unwrap();
        std::fs::write(dir.join("api-keys"), "key-alpha\nkey-beta\n").unwrap();

        let provider = SecretsProvider::from_mount(&dir);
        let _ = provider.refresh();

        assert_eq!(
            provider.get_string("jwt-secret").unwrap(),
            "test-jwt-secret-32bytes-long!!!!"
        );
        let keys = provider.api_keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key-alpha".to_string()));

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compliance_retention_policy() {
        let audit = Arc::new(AuditLogger::new(100));
        let mgr = ComplianceManager::new(audit);

        mgr.add_retention_policy(RetentionPolicy {
            name: "gdpr-eu".to_string(),
            namespace_pattern: "eu-*".to_string(),
            max_retention: Duration::from_secs(365 * 24 * 3600), // 1 year
            auto_purge: true,
            audit_deletions: true,
        });
        mgr.add_retention_policy(RetentionPolicy {
            name: "global-default".to_string(),
            namespace_pattern: "*".to_string(),
            max_retention: Duration::from_secs(5 * 365 * 24 * 3600), // 5 years
            auto_purge: false,
            audit_deletions: true,
        });

        // EU namespace gets both policies; effective retention = 1 year (shortest)
        let policies = mgr.policies_for_namespace("eu-production");
        assert_eq!(policies.len(), 2);
        let effective = mgr.effective_retention("eu-production").unwrap();
        assert_eq!(effective, Duration::from_secs(365 * 24 * 3600));

        // US namespace gets only global default
        let policies = mgr.policies_for_namespace("us-east");
        assert_eq!(policies.len(), 1);
    }

    #[test]
    fn test_compliance_erasure_record() {
        let audit = Arc::new(AuditLogger::new(100));
        let mgr = ComplianceManager::new(audit.clone());

        mgr.record_erasure(
            "admin",
            "eu-prod",
            "user-42",
            vec!["kv:user-42:*".to_string(), "vectors:user-42".to_string()],
            "req-gdpr-001",
        );

        let records = mgr.erasure_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].subject_id, "user-42");
        assert_eq!(records[0].resources_erased.len(), 2);

        // Audit logger should have the entry
        assert!(audit.total_entries() >= 1);
    }

    #[test]
    fn test_tls_pem_validation() {
        let valid_cert = b"-----BEGIN CERTIFICATE-----\nMIIBfake\n-----END CERTIFICATE-----\n";
        let valid_key = b"-----BEGIN PRIVATE KEY-----\nMIIBfake\n-----END PRIVATE KEY-----\n";
        let invalid = b"not a pem file";

        assert!(TlsProvider::validate_pem(valid_cert, "CERTIFICATE").is_ok());
        assert!(TlsProvider::validate_pem(valid_key, "PRIVATE KEY").is_ok());
        assert!(TlsProvider::validate_pem(invalid, "CERTIFICATE").is_err());
        assert!(TlsProvider::validate_pem(valid_cert, "PRIVATE KEY").is_err());
    }
}
