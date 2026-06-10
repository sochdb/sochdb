# SochDB Pricing

## Plans

### Community (Free)

- Single node deployment
- Up to 1M vectors
- Community support (GitHub Issues)
- All core features included
- Apache 2.0 / AGPL license

### Professional ($99/node/month)

- Up to 3 nodes
- Up to 50M vectors per node
- Email support (48h response)
- mTLS, RBAC, audit logging
- Prometheus + Grafana dashboards
- WAL archiving (S3/GCS/Azure Blob)
- 99.5% availability SLA

### Enterprise (Custom Pricing)

- Unlimited nodes
- 1B+ vectors
- Dedicated support engineer (4h response)
- Custom SLA (up to 99.99%)
- SSO/SAML integration
- SOC 2 compliance artifacts
- Private cloud deployment
- Priority bug fixes
- Architecture review
- Training and onboarding

## Marketplace Pricing

### Azure Marketplace
- Free tier: $0/month (single node, 1M vectors)
- Professional: $99/node/month (BYOL)
- Enterprise: Per-core metered ($0.05/core-hour)

### AWS Marketplace
- Free tier: $0/month
- Hourly: $0.15/node-hour
- Annual: $99/node/month (annual commitment, 17% savings)

### GCP Marketplace
- Free tier: $0/month
- Professional: $99/node/month
- Enterprise: Custom (contact sales)

## Trial

All paid plans include a 14-day free trial with full features.
No credit card required for the free tier.

## Volume Discounts

| Nodes | Discount |
|-------|----------|
| 1-5   | Standard |
| 6-20  | 10%      |
| 21-50 | 20%      |
| 50+   | Custom   |

## Metering Dimensions

For metered plans, usage is tracked on:
- **Node-hours**: Active SochDB pods × hours
- **Storage GB-hours**: Persistent volume usage
- **Vectors stored**: Peak vector count per billing period
- **API calls**: gRPC requests per billing period

## Contact

- Sales: sales@sochdb.dev
- Enterprise inquiries: enterprise@sochdb.dev
