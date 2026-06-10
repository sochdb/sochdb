// SPDX-License-Identifier: AGPL-3.0-or-later
// Kubernetes Custom Resource Definitions for SochDB
//
// Defines the SochDBCluster CRD for declarative cluster management.
// Feature-gated: compile with `--features k8s` to enable kube-derive.

/// Spec for a SochDB cluster (mirrors the K8s CRD spec).
///
/// When the `k8s` feature is enabled, this derives `CustomResource` from kube-rs,
/// generating the full CRD schema. Without the feature, it's a plain Rust struct.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "k8s", derive(
    serde::Serialize, serde::Deserialize,
    schemars::JsonSchema,
    kube::CustomResource,
))]
#[cfg_attr(feature = "k8s", kube(
    group = "sochdb.io",
    version = "v1alpha1",
    kind = "SochDBCluster",
    namespaced,
    status = "SochDBClusterStatus",
    shortname = "sdb",
))]
pub struct SochDBClusterSpec {
    /// Number of replicas (nodes) in the cluster.
    pub replicas: u32,
    /// Docker image to use.
    pub image: String,
    /// Storage configuration.
    pub storage: StorageSpec,
    /// Resource requests/limits per node.
    pub resources: ResourceSpec,
    /// gRPC server configuration.
    pub grpc: GrpcSpec,
    /// Monitoring configuration.
    pub monitoring: Option<MonitoringSpec>,
}

/// Storage configuration for cluster nodes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "k8s", derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema))]
pub struct StorageSpec {
    /// Storage class name (e.g., "gp3", "local-path").
    pub storage_class: String,
    /// Size of the persistent volume (e.g., "100Gi").
    pub size: String,
}

/// Resource requests and limits.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "k8s", derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema))]
pub struct ResourceSpec {
    pub cpu_request: String,
    pub cpu_limit: String,
    pub memory_request: String,
    pub memory_limit: String,
}

/// gRPC server configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "k8s", derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema))]
pub struct GrpcSpec {
    pub port: u16,
    pub tls_enabled: bool,
}

/// Monitoring configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "k8s", derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema))]
pub struct MonitoringSpec {
    pub prometheus_enabled: bool,
    pub metrics_port: u16,
}

/// Status of the SochDB cluster.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "k8s", derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema))]
pub struct SochDBClusterStatus {
    pub phase: String,
    pub ready_replicas: u32,
    pub current_version: String,
    pub conditions: Vec<ClusterCondition>,
    pub observed_generation: i64,
}

/// A condition on the cluster.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "k8s", derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema))]
pub struct ClusterCondition {
    pub condition_type: String,
    pub status: String,
    pub reason: String,
    pub message: String,
    pub last_transition_time: String,
}

impl Default for SochDBClusterSpec {
    fn default() -> Self {
        Self {
            replicas: 3,
            image: "ghcr.io/sochdb/sochdb:latest".to_string(),
            storage: StorageSpec {
                storage_class: "gp3".to_string(),
                size: "100Gi".to_string(),
            },
            resources: ResourceSpec {
                cpu_request: "2".to_string(),
                cpu_limit: "4".to_string(),
                memory_request: "4Gi".to_string(),
                memory_limit: "8Gi".to_string(),
            },
            grpc: GrpcSpec {
                port: 6334,
                tls_enabled: false,
            },
            monitoring: Some(MonitoringSpec {
                prometheus_enabled: true,
                metrics_port: 9090,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_spec() {
        let spec = SochDBClusterSpec::default();
        assert_eq!(spec.replicas, 3);
        assert_eq!(spec.grpc.port, 6334);
        assert!(spec.monitoring.is_some());
    }

    #[test]
    fn test_status_default() {
        let status = SochDBClusterStatus::default();
        assert_eq!(status.ready_replicas, 0);
        assert!(status.phase.is_empty());
    }
}
