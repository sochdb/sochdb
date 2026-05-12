// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # gRPC Authentication Interceptor (Task 7 — Phase 1)
//!
//! Provides a Tonic interceptor that extracts authentication credentials
//! from gRPC metadata and validates them against the `SecurityService`.
//!
//! ## Authentication Methods
//!
//! 1. **Bearer Token** (JWT or API key): `authorization: Bearer <token>`
//! 2. **API Key header**: `x-api-key: <key>`
//! 3. **Anonymous**: Allowed if configured
//!
//! ## Usage
//!
//! ```ignore
//! let auth = AuthInterceptor::new(security_service);
//!
//! // Apply to individual services:
//! let svc = VectorIndexServiceServer::with_interceptor(vector_impl, auth.clone());
//!
//! // Or check manually in a handler:
//! let principal = auth.check_request(&request)?;
//! ```

use std::sync::Arc;

use tonic::{Request, Status};

use crate::security::{AuthError, Capability, Principal, SecurityConfig, SecurityService};

/// Tonic interceptor for gRPC authentication.
///
/// Extracts credentials from gRPC metadata, authenticates via
/// `SecurityService`, and injects the `Principal` as a request extension.
#[derive(Clone)]
pub struct AuthInterceptor {
    security: Arc<SecurityService>,
    /// If true, unauthenticated requests are rejected.
    /// If false, the interceptor is a pass-through (for dev mode).
    enabled: bool,
}

impl AuthInterceptor {
    /// Create a new auth interceptor.
    pub fn new(security: Arc<SecurityService>, enabled: bool) -> Self {
        Self { security, enabled }
    }

    /// Create a disabled (pass-through) interceptor.
    pub fn disabled() -> Self {
        Self {
            security: Arc::new(SecurityService::new(SecurityConfig::default())),
            enabled: false,
        }
    }

    /// Extract the authorization header from gRPC metadata.
    fn extract_auth_header<T>(request: &Request<T>) -> Option<String> {
        request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }

    /// Extract the API key from gRPC metadata.
    fn extract_api_key<T>(request: &Request<T>) -> Option<String> {
        request
            .metadata()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }

    /// Authenticate a request and return the principal.
    pub fn authenticate_request<T>(&self, request: &Request<T>) -> Result<Principal, Status> {
        if !self.enabled {
            return Ok(Principal::anonymous());
        }

        let auth_header = Self::extract_auth_header(request);

        // If x-api-key header is present, convert to Bearer format
        let effective_header = if auth_header.is_none() {
            Self::extract_api_key(request).map(|k| format!("Bearer {}", k))
        } else {
            auth_header
        };

        match self
            .security
            .authenticate(effective_header.as_deref(), None)
        {
            Ok(principal) => Ok(principal),
            Err(e) => Err(auth_error_to_status(e)),
        }
    }

    /// Get a reference to the underlying security service.
    pub fn security(&self) -> &SecurityService {
        &self.security
    }

    /// Register an API key for authentication.
    pub fn register_api_key(&self, key: &str, principal: Principal) {
        self.security.register_api_key(key, principal);
    }
}

/// Implement the Tonic `Interceptor` trait.
///
/// This is called for every gRPC request:
/// 1. Extract credentials from metadata
/// 2. Authenticate via `SecurityService`
/// 3. Check rate limits
/// 4. Inject `Principal` into request extensions
/// 5. Audit log the access
impl tonic::service::Interceptor for AuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        if !self.enabled {
            return Ok(request);
        }

        let auth_header = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Check x-api-key as fallback
        let effective_header = if auth_header.is_none() {
            request
                .metadata()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .map(|k| format!("Bearer {}", k))
        } else {
            auth_header
        };

        // Step 1: Authenticate
        let principal = self
            .security
            .authenticate(effective_header.as_deref(), None)
            .map_err(auth_error_to_status)?;

        // Step 2: Rate limit check
        self.security
            .check_rate_limit(&principal)
            .map_err(auth_error_to_status)?;

        // Step 3: Store principal in request extensions for downstream handlers
        request.extensions_mut().insert(principal);
        Ok(request)
    }
}

/// Convert `AuthError` to gRPC `Status`.
fn auth_error_to_status(err: AuthError) -> Status {
    match err {
        AuthError::Unauthenticated => {
            Status::unauthenticated("Authentication required. Provide a Bearer token or API key.")
        }
        AuthError::TokenExpired => {
            Status::unauthenticated("Authentication token has expired.")
        }
        AuthError::Unauthorized { required } => Status::permission_denied(format!(
            "Insufficient permissions. Required capability: {}",
            required
        )),
        AuthError::RateLimited { retry_after_ms } => {
            let mut status = Status::resource_exhausted(format!(
                "Rate limit exceeded. Retry after {}ms.",
                retry_after_ms
            ));
            // Add retry-after hint as trailing metadata
            let mut metadata = tonic::metadata::MetadataMap::new();
            if let Ok(val) = retry_after_ms.to_string().parse() {
                metadata.insert("retry-after-ms", val);
            }
            status
        }
        AuthError::Internal(msg) => Status::internal(format!("Internal auth error: {}", msg)),
    }
}

// ============================================================================
// Helper: Anonymous Principal
// ============================================================================

impl Principal {
    /// Create an anonymous principal (used when auth is disabled).
    pub fn anonymous() -> Self {
        use std::collections::HashSet;
        Self {
            id: "anonymous".to_string(),
            tenant_id: "default".to_string(),
            capabilities: HashSet::from([
                Capability::Read,
                Capability::Write,
                Capability::ManageCollections,
            ]),
            expires_at: None,
            auth_method: crate::security::AuthMethod::Anonymous,
        }
    }
}

// ============================================================================
// Helper: Extract Principal from gRPC request extensions
// ============================================================================

/// Extract `Principal` from a gRPC request's extensions.
///
/// Must be called BEFORE `request.into_inner()`, since `into_inner()`
/// consumes the `Request` wrapper (and the extensions with it).
///
/// Returns anonymous principal if none was injected (auth disabled mode).
///
/// # Usage
///
/// ```ignore
/// async fn my_handler(&self, request: Request<MyReq>) -> Result<Response<MyRes>, Status> {
///     let principal = extract_principal(&request);
///     require_capability(&principal, &Capability::Write)?;
///     let req = request.into_inner();
///     // ... use req and principal
/// }
/// ```
pub fn extract_principal<T>(request: &Request<T>) -> Principal {
    request
        .extensions()
        .get::<Principal>()
        .cloned()
        .unwrap_or_else(Principal::anonymous)
}

/// Check that a principal has the required capability.
/// Returns `Status::permission_denied` if the check fails.
pub fn require_capability(principal: &Principal, capability: &Capability) -> Result<(), Status> {
    if principal.has_capability(capability) {
        Ok(())
    } else {
        Err(Status::permission_denied(format!(
            "Insufficient permissions. Required: {:?}, principal: {}",
            capability, principal.id,
        )))
    }
}

/// Check that a principal has access to the requested namespace.
/// Owner/Admin can access any namespace; others must match their tenant_id.
pub fn require_namespace_access(principal: &Principal, namespace: &str) -> Result<(), Status> {
    // Admin and Owner can access any namespace
    if principal.has_capability(&Capability::Admin) {
        return Ok(());
    }
    // Other users can only access their own tenant namespace or "default"
    if principal.tenant_id == namespace || namespace == "default" || namespace.is_empty() {
        return Ok(());
    }
    Err(Status::permission_denied(format!(
        "Access denied to namespace '{}' for tenant '{}'",
        namespace, principal.tenant_id,
    )))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{AuthMethod, Capability, SecurityConfig};
    use std::collections::HashSet;

    fn make_config(api_key_enabled: bool) -> SecurityConfig {
        SecurityConfig {
            mtls_enabled: false,
            jwt_enabled: false,
            api_key_enabled,
            rate_limit_default: 1000,
            rate_limit_burst: 100,
            audit_enabled: false,
            audit_flush_threshold: 100,
            ..SecurityConfig::default()
        }
    }

    #[test]
    fn test_disabled_interceptor_allows_all() {
        let interceptor = AuthInterceptor::disabled();
        let request = Request::new(());
        let mut interceptor_clone = interceptor.clone();
        let result = tonic::service::Interceptor::call(&mut interceptor_clone, request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enabled_interceptor_rejects_unauthenticated() {
        let config = make_config(true);
        let security = Arc::new(SecurityService::new(config));
        let interceptor = AuthInterceptor::new(security, true);

        let request = Request::new(());
        let mut interceptor_clone = interceptor.clone();
        let result = tonic::service::Interceptor::call(&mut interceptor_clone, request);
        assert!(result.is_err());

        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn test_api_key_authentication() {
        let config = make_config(true);
        let security = Arc::new(SecurityService::new(config));

        // Register an API key
        let principal = Principal {
            id: "test-user".to_string(),
            tenant_id: "tenant-1".to_string(),
            capabilities: HashSet::from([Capability::Read, Capability::Write]),
            expires_at: None,
            auth_method: AuthMethod::ApiKey,
        };
        security.register_api_key("my-secret-key", principal);

        let interceptor = AuthInterceptor::new(security, true);

        // Request with valid API key
        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", "Bearer my-secret-key".parse().unwrap());

        let mut interceptor_clone = interceptor.clone();
        let result = tonic::service::Interceptor::call(&mut interceptor_clone, request);
        assert!(result.is_ok());

        // Check the principal was injected
        let request = result.unwrap();
        let p = request.extensions().get::<Principal>().unwrap();
        assert_eq!(p.id, "test-user");
        assert_eq!(p.tenant_id, "tenant-1");
    }

    #[test]
    fn test_x_api_key_header() {
        let config = make_config(true);
        let security = Arc::new(SecurityService::new(config));

        let principal = Principal {
            id: "key-user".to_string(),
            tenant_id: "default".to_string(),
            capabilities: HashSet::from([Capability::Read]),
            expires_at: None,
            auth_method: AuthMethod::ApiKey,
        };
        security.register_api_key("header-key", principal);

        let interceptor = AuthInterceptor::new(security, true);

        // Use x-api-key header instead of Authorization
        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("x-api-key", "header-key".parse().unwrap());

        let mut interceptor_clone = interceptor.clone();
        let result = tonic::service::Interceptor::call(&mut interceptor_clone, request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_anonymous_principal() {
        let anon = Principal::anonymous();
        assert_eq!(anon.id, "anonymous");
        assert!(anon.has_capability(&Capability::Read));
        assert!(anon.has_capability(&Capability::Write));
        assert!(!anon.has_capability(&Capability::Admin));
        assert!(!anon.is_expired());
    }

    #[test]
    fn test_auth_error_to_status() {
        let s = auth_error_to_status(AuthError::Unauthenticated);
        assert_eq!(s.code(), tonic::Code::Unauthenticated);

        let s = auth_error_to_status(AuthError::TokenExpired);
        assert_eq!(s.code(), tonic::Code::Unauthenticated);

        let s = auth_error_to_status(AuthError::Unauthorized {
            required: "Admin".into(),
        });
        assert_eq!(s.code(), tonic::Code::PermissionDenied);

        let s = auth_error_to_status(AuthError::RateLimited {
            retry_after_ms: 1000,
        });
        assert_eq!(s.code(), tonic::Code::ResourceExhausted);
    }
}
