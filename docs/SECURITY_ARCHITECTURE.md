# SochDB Security Architecture & Compliance Roadmap

## Security Architecture Overview

### Authentication

SochDB supports multiple authentication methods:

| Method | Module | Status |
|--------|--------|--------|
| API Key (Bearer) | `auth_interceptor.rs` | ✅ Implemented |
| JWT/OIDC (JWKS) | `security.rs` | ✅ Implemented |
| mTLS client certs | `security.rs` (TlsProvider) | ✅ Implemented |
| Anonymous (dev mode) | `auth_interceptor.rs` | ✅ Implemented |
| Cloud IAM (Azure AD, AWS IAM, GCP IAM) | — | 🔲 Planned |

### Authorization

| Feature | Module | Status |
|---------|--------|--------|
| Capability-based RBAC | `security.rs` (Principal, Capability) | ✅ Implemented |
| Per-tenant resource isolation | `tenant_isolation.rs` | ✅ Implemented |
| Rate limiting per tenant | `security.rs` | ✅ Implemented |
| Namespace-level ACLs | — | 🔲 Planned |

### Encryption

| Feature | Status |
|---------|--------|
| TLS 1.3 in transit | ✅ Implemented (TlsProvider with hot-reload) |
| mTLS with CA verification | ✅ Implemented |
| Data at rest encryption | 🔲 Planned (AES-256-GCM, key from K8s Secret / Key Vault) |
| WAL encryption | 🔲 Planned |

### Audit Logging

| Feature | Module | Status |
|---------|--------|--------|
| Structured audit log | `security.rs` (AuditLog) | ✅ Implemented |
| Append-only log format | `security.rs` | ✅ Implemented |
| Log export (SIEM integration) | — | 🔲 Planned |

### Network Security

| Feature | Status |
|---------|--------|
| Kubernetes NetworkPolicy | ✅ Helm chart template |
| Pod-to-pod encryption (mTLS) | ✅ Implemented |
| Ingress TLS termination | ✅ Helm chart (cert-manager) |
| Egress control | ✅ NetworkPolicy template |

### Container Security

| Feature | Status |
|---------|--------|
| Non-root container user | ✅ Dockerfile (USER sochdb) |
| Read-only root filesystem | ✅ Helm values (readOnlyRootFilesystem: true) |
| Drop all capabilities | ✅ Helm values (drop: ALL) |
| No privilege escalation | ✅ Helm values |
| Trivy vulnerability scan | ✅ CI pipeline |
| SBOM generation (SPDX) | ✅ CI pipeline |
| Signed container images | 🔲 Planned (cosign) |

### Secrets Management

| Provider | Status |
|----------|--------|
| Kubernetes Secrets | ✅ Helm chart (security.secrets) |
| Azure Key Vault (CSI driver) | 🔲 Planned |
| AWS Secrets Manager | 🔲 Planned |
| GCP Secret Manager | 🔲 Planned |
| HashiCorp Vault | 🔲 Planned |

---

## Compliance Roadmap

### SOC 2 Type II Readiness

| Control Area | Status | Notes |
|-------------|--------|-------|
| CC1: Control Environment | 🟡 In Progress | Security policy docs, SECURITY.md |
| CC2: Communication | 🟡 In Progress | CONTRIBUTING.md, CODE_OF_CONDUCT.md |
| CC3: Risk Assessment | 🔲 Planned | Threat model, dependency audit |
| CC5: Control Activities | 🟡 In Progress | CI/CD gates, code review required |
| CC6: Logical Access | ✅ Implemented | RBAC, API key auth, mTLS |
| CC7: System Operations | 🟡 In Progress | Prometheus alerts, health checks |
| CC8: Change Management | 🟡 In Progress | Git-based, PR required, CI checks |
| CC9: Risk Mitigation | 🟡 In Progress | Vulnerability scanning in CI |

### GDPR Readiness

| Requirement | Status |
|-------------|--------|
| Data processing records | 🔲 Planned |
| Right to erasure | ✅ Delete API exists |
| Data portability | 🟡 Partial (export via gRPC) |
| Privacy by design | ✅ Tenant isolation, namespace separation |
| Data breach notification | 🔲 Planned (process) |
| DPO designation | 🔲 Planned |

### HIPAA Readiness (for healthcare customers)

| Safeguard | Status |
|-----------|--------|
| Access controls | ✅ RBAC + API key auth |
| Audit controls | ✅ Audit logging |
| Transmission security | ✅ TLS/mTLS |
| Encryption at rest | 🔲 Planned |
| Integrity controls | ✅ WAL + checksums |
| BAA template | 🔲 Planned |

---

## Vulnerability Disclosure

See [SECURITY.md](../SECURITY.md) for vulnerability reporting procedures.

**Response SLAs:**
- Critical (RCE, auth bypass): Patch within 24 hours
- High (data leak, privilege escalation): Patch within 72 hours
- Medium: Patch within 2 weeks
- Low: Next scheduled release

## Dependency Policy

- `cargo audit` runs on every CI build
- `cargo deny` checks licenses and advisories
- Trivy scans container images for CVEs
- Dependencies reviewed before adding to workspace
- No `unsafe` in application code without explicit review
