// SPDX-License-Identifier: AGPL-3.0-or-later
//! SochDB Kubernetes Operator
//!
//! CRD-based lifecycle management for SochDB clusters on Kubernetes.
//! Feature-gated behind `k8s` to avoid pulling heavy dependencies by default.

pub mod crd;
pub mod reconciler;
