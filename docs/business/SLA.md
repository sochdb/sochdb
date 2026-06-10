# SochDB Service Level Agreement (SLA)

*Effective Date: 2026-04-28*

## Definitions

- **Monthly Uptime Percentage**: (Total minutes − Downtime minutes) / Total minutes × 100
- **Downtime**: Period where the SochDB gRPC health check fails for ≥1 consecutive minute
- **Scheduled Maintenance**: Pre-announced downtime with ≥24 hours notice

## Service Levels

### Community Plan
- **Target**: Best effort
- **No SLA guarantee**
- Support via GitHub Issues only

### Professional Plan
- **Monthly Uptime Target**: 99.5%
- **Support Response Time**: 48 hours (business days)
- **Scheduled Maintenance Window**: Sundays 02:00–06:00 UTC

### Enterprise Plan
- **Monthly Uptime Target**: 99.9% (negotiable to 99.99%)
- **Support Response Time**:
  - Critical (service down): 1 hour
  - High (degraded): 4 hours
  - Medium: 1 business day
  - Low: 2 business days
- **Scheduled Maintenance**: Coordinated with customer

## Service Credits

If SochDB fails to meet the SLA, credits are applied:

| Monthly Uptime | Credit |
|---------------|--------|
| 99.0% – 99.5% | 10% of monthly fee |
| 95.0% – 99.0% | 25% of monthly fee |
| < 95.0% | 50% of monthly fee |

### Credit Request Process
1. Submit credit request within 30 days of the incident
2. Include affected time range and evidence (monitoring data)
3. Credits applied to next billing cycle
4. Credits do not exceed 50% of monthly fee

## Exclusions

SLA does not apply to:
- Scheduled maintenance windows
- Customer-caused outages (misconfiguration, resource exhaustion)
- Force majeure events
- Free/Community tier
- Beta or preview features
- Third-party infrastructure failures (cloud provider outages)

## Monitoring

Customers can independently verify uptime via:
- Prometheus `/metrics` endpoint
- gRPC health check (`grpc_health_probe`)
- Kubernetes readiness probe status
- StatusPage: https://status.sochdb.dev (planned)

## Support Channels

| Channel | Community | Professional | Enterprise |
|---------|-----------|-------------|------------|
| GitHub Issues | ✅ | ✅ | ✅ |
| Email | ❌ | ✅ | ✅ |
| Slack | ❌ | ❌ | ✅ |
| Phone | ❌ | ❌ | ✅ |
| Dedicated engineer | ❌ | ❌ | ✅ |

## Contact

- Support: support@sochdb.dev
- Enterprise SLA inquiries: enterprise@sochdb.dev
