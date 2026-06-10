// SPDX-License-Identifier: AGPL-3.0-or-later
// Kubernetes Reconciler for SochDBCluster CRD
//
// Watches SochDBCluster resources and reconciles the desired state:
// - Creates/updates StatefulSet for the SochDB cluster nodes
// - Creates headless Service for pod discovery
// - Creates ConfigMap for server configuration
// - Manages rolling updates and scale up/down
//
// Feature-gated behind `k8s`.

use crate::crd::SochDBClusterSpec;

/// Reconciler actions produced by the reconciliation loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileAction {
    /// Create the StatefulSet (first deploy).
    CreateStatefulSet,
    /// Scale the StatefulSet to the desired replica count.
    Scale { current: u32, desired: u32 },
    /// Update the StatefulSet image (rolling upgrade).
    RollingUpdate { from_image: String, to_image: String },
    /// Update the ConfigMap.
    UpdateConfig,
    /// No changes needed.
    NoOp,
}

/// Compute the reconciliation action given current vs desired state.
///
/// This is the pure-logic core of the reconciler, decoupled from the
/// Kubernetes API for testability.
pub fn plan_reconciliation(
    desired: &SochDBClusterSpec,
    current_replicas: Option<u32>,
    current_image: Option<&str>,
) -> Vec<ReconcileAction> {
    let mut actions = Vec::new();

    match current_replicas {
        None => {
            // StatefulSet doesn't exist yet
            actions.push(ReconcileAction::CreateStatefulSet);
        }
        Some(current) if current != desired.replicas => {
            actions.push(ReconcileAction::Scale {
                current,
                desired: desired.replicas,
            });
        }
        _ => {}
    }

    if let Some(current_img) = current_image {
        if current_img != desired.image {
            actions.push(ReconcileAction::RollingUpdate {
                from_image: current_img.to_string(),
                to_image: desired.image.clone(),
            });
        }
    }

    if actions.is_empty() {
        actions.push(ReconcileAction::NoOp);
    }

    actions
}

/// Generate the StatefulSet spec as a serde_json::Value.
///
/// This produces the Kubernetes StatefulSet manifest that the operator
/// would apply via the API server.
pub fn build_statefulset_manifest(
    name: &str,
    namespace: &str,
    spec: &SochDBClusterSpec,
) -> String {
    format!(
        r#"apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: sochdb
    app.kubernetes.io/managed-by: sochdb-operator
spec:
  replicas: {replicas}
  serviceName: {name}-headless
  selector:
    matchLabels:
      app.kubernetes.io/name: sochdb
      app.kubernetes.io/instance: {name}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: sochdb
        app.kubernetes.io/instance: {name}
    spec:
      containers:
        - name: sochdb
          image: {image}
          ports:
            - containerPort: {grpc_port}
              name: grpc
          resources:
            requests:
              cpu: "{cpu_request}"
              memory: "{memory_request}"
            limits:
              cpu: "{cpu_limit}"
              memory: "{memory_limit}"
          volumeMounts:
            - name: data
              mountPath: /data
          livenessProbe:
            tcpSocket:
              port: {grpc_port}
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            tcpSocket:
              port: {grpc_port}
            initialDelaySeconds: 5
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        storageClassName: {storage_class}
        accessModes: [ReadWriteOnce]
        resources:
          requests:
            storage: {storage_size}
"#,
        name = name,
        namespace = namespace,
        replicas = spec.replicas,
        image = spec.image,
        grpc_port = spec.grpc.port,
        cpu_request = spec.resources.cpu_request,
        cpu_limit = spec.resources.cpu_limit,
        memory_request = spec.resources.memory_request,
        memory_limit = spec.resources.memory_limit,
        storage_class = spec.storage.storage_class,
        storage_size = spec.storage.size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_deploy() {
        let spec = SochDBClusterSpec::default();
        let actions = plan_reconciliation(&spec, None, None);
        assert_eq!(actions, vec![ReconcileAction::CreateStatefulSet]);
    }

    #[test]
    fn test_scale_up() {
        let spec = SochDBClusterSpec { replicas: 5, ..Default::default() };
        let actions = plan_reconciliation(&spec, Some(3), Some(&spec.image));
        assert_eq!(actions, vec![ReconcileAction::Scale { current: 3, desired: 5 }]);
    }

    #[test]
    fn test_rolling_update() {
        let spec = SochDBClusterSpec {
            image: "ghcr.io/sochdb/sochdb:v2.1.0".to_string(),
            ..Default::default()
        };
        let actions = plan_reconciliation(&spec, Some(3), Some("ghcr.io/sochdb/sochdb:v2.0.0"));
        assert!(actions.contains(&ReconcileAction::RollingUpdate {
            from_image: "ghcr.io/sochdb/sochdb:v2.0.0".to_string(),
            to_image: "ghcr.io/sochdb/sochdb:v2.1.0".to_string(),
        }));
    }

    #[test]
    fn test_no_op() {
        let spec = SochDBClusterSpec::default();
        let actions = plan_reconciliation(&spec, Some(3), Some(&spec.image));
        assert_eq!(actions, vec![ReconcileAction::NoOp]);
    }

    #[test]
    fn test_manifest_generation() {
        let spec = SochDBClusterSpec::default();
        let manifest = build_statefulset_manifest("my-sochdb", "default", &spec);
        assert!(manifest.contains("kind: StatefulSet"));
        assert!(manifest.contains("replicas: 3"));
        assert!(manifest.contains("ghcr.io/sochdb/sochdb:latest"));
    }
}
