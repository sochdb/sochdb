# SochDB Privacy Policy

*Effective Date: 2026-04-28*

## Overview

SochDB is self-hosted software deployed in the customer's own infrastructure.
SochDB (the company) does not access, collect, or process customer data.

## Data Processing

### Self-Hosted Deployments

When you deploy SochDB on your own infrastructure (Kubernetes, bare metal, or cloud VMs):

- **No data leaves your infrastructure**. SochDB runs entirely within your environment.
- **No telemetry**: SochDB does not phone home, collect usage metrics, or transmit any data externally.
- **No cloud dependencies**: The database operates fully offline after installation.
- **Your data, your control**: All vectors, metadata, keys, and configuration remain on your storage.

### Cloud Marketplace Deployments

When deploying via Azure/AWS/GCP Marketplace:

- Marketplace metering data (pod count, uptime) is shared with the cloud provider for billing.
- No vector data, queries, or results are shared with SochDB or the cloud provider.
- Metering uses standard cloud marketplace APIs only.

## Data You Provide to Us

If you contact SochDB for support, sales, or create an account:

- **Account information**: Name, email, company (for support/sales communication)
- **Support tickets**: Technical details you share for troubleshooting
- **Usage data**: Anonymized, aggregated feature usage (opt-in only, not in self-hosted)

## Data Retention

- Self-hosted: Determined entirely by your data lifecycle policies
- Account/support data: Retained while your account is active + 90 days
- Deleted upon request

## Security

See [SECURITY_ARCHITECTURE.md](../SECURITY_ARCHITECTURE.md) for technical security details.

- All data in transit encrypted via TLS 1.3
- Data at rest encryption available (AES-256)
- Access control via API keys, JWT/OIDC, mTLS
- Audit logging for all data access

## Your Rights

Under GDPR, CCPA, and similar regulations:

- **Access**: Request a copy of any personal data we hold
- **Deletion**: Request deletion of your account and personal data
- **Portability**: Export your data via gRPC API at any time
- **Correction**: Update your account information
- **Objection**: Opt out of any data processing

## Contact

- Privacy inquiries: privacy@sochdb.dev
- Data protection officer: dpo@sochdb.dev
- Mailing address: [To be added]

## Changes

We may update this policy. Changes will be posted at https://sochdb.dev/privacy
with 30 days notice for material changes.
