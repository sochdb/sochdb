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

//! Serves the capability manifest so clients can negotiate rather than guess.
//!
//! The manifest itself lives in `sochdb_core::capability`, which validates its
//! own claims. This module is only the wire adapter: it translates the
//! canonical Rust form into protobuf and refuses to serve a manifest that does
//! not pass validation, so an inconsistent manifest fails loudly at the server
//! instead of silently licensing a client to do something unsupported.

use sochdb_core::capability::{self, CapabilityManifest, Durability, Maturity};
use tonic::{Request, Response, Status};

use crate::proto::capability_service_server::{CapabilityService, CapabilityServiceServer};
use crate::proto::{
    CapabilityContractVersion, CapabilityDescriptor, CapabilityDurability,
    CapabilityManifest as ProtoManifest, CapabilityMaturity, GetCapabilitiesRequest,
};

/// Serves the build's capability manifest.
pub struct CapabilityServer {
    manifest: CapabilityManifest,
}

impl CapabilityServer {
    /// Build a server over this build's published manifest.
    pub fn new() -> Self {
        Self {
            manifest: capability::manifest(),
        }
    }

    /// Build a server over a specific manifest.
    ///
    /// Rejects a manifest that contradicts itself. Serving one would be worse
    /// than serving none: a client would negotiate successfully against a claim
    /// the server cannot keep.
    pub fn with_manifest(manifest: CapabilityManifest) -> Result<Self, capability::ManifestError> {
        manifest.validate()?;
        Ok(Self { manifest })
    }

    pub fn into_service(self) -> CapabilityServiceServer<Self> {
        CapabilityServiceServer::new(self)
    }

    fn to_proto(&self) -> ProtoManifest {
        ProtoManifest {
            server_version: self.manifest.server_version.clone(),
            build_revision: self.manifest.build_revision.clone().unwrap_or_default(),
            capabilities: self
                .manifest
                .capabilities
                .iter()
                .map(|c| CapabilityDescriptor {
                    name: c.name.clone(),
                    contract_version: Some(CapabilityContractVersion {
                        major: c.contract_version.major,
                        minor: c.contract_version.minor,
                    }),
                    maturity: maturity_to_proto(c.maturity) as i32,
                    durability: durability_to_proto(c.durability) as i32,
                    guarantees: c.guarantees.clone(),
                    limits: c.limits.clone(),
                })
                .collect(),
        }
    }
}

impl Default for CapabilityServer {
    fn default() -> Self {
        Self::new()
    }
}

fn maturity_to_proto(m: Maturity) -> CapabilityMaturity {
    match m {
        Maturity::LibraryOnly => CapabilityMaturity::LibraryOnly,
        Maturity::Experimental => CapabilityMaturity::Experimental,
        Maturity::Preview => CapabilityMaturity::Preview,
        Maturity::Supported => CapabilityMaturity::Supported,
    }
}

fn durability_to_proto(d: Durability) -> CapabilityDurability {
    match d {
        Durability::Durable => CapabilityDurability::Durable,
        Durability::Ephemeral => CapabilityDurability::Ephemeral,
        Durability::Stateless => CapabilityDurability::Stateless,
    }
}

#[tonic::async_trait]
impl CapabilityService for CapabilityServer {
    async fn get_capabilities(
        &self,
        _request: Request<GetCapabilitiesRequest>,
    ) -> Result<Response<ProtoManifest>, Status> {
        Ok(Response::new(self.to_proto()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sochdb_core::capability::{Capability, ContractVersion, names};

    #[tokio::test]
    async fn the_served_manifest_matches_the_canonical_one() {
        let server = CapabilityServer::new();
        let served = server
            .get_capabilities(Request::new(GetCapabilitiesRequest {}))
            .await
            .expect("get_capabilities")
            .into_inner();

        let canonical = capability::manifest();
        assert_eq!(served.capabilities.len(), canonical.capabilities.len());
        assert_eq!(served.server_version, canonical.server_version);

        for expected in &canonical.capabilities {
            let actual = served
                .capabilities
                .iter()
                .find(|c| c.name == expected.name)
                .unwrap_or_else(|| panic!("`{}` was not served", expected.name));
            assert_eq!(
                actual.maturity,
                maturity_to_proto(expected.maturity) as i32,
                "`{}` was served at a different maturity than it claims",
                expected.name
            );
            assert_eq!(actual.guarantees, expected.guarantees);
            assert_eq!(actual.limits, expected.limits);
        }
    }

    /// Every enum value must survive the wire. An unmapped variant would arrive
    /// as the zero value, which is `UNSPECIFIED` — a client would read that as
    /// "unknown" and could reasonably fall back, so a mapping mistake would
    /// silently downgrade a capability rather than fail.
    #[tokio::test]
    async fn no_capability_is_served_as_unspecified() {
        let server = CapabilityServer::new();
        let served = server
            .get_capabilities(Request::new(GetCapabilitiesRequest {}))
            .await
            .expect("get_capabilities")
            .into_inner();

        for cap in &served.capabilities {
            assert_ne!(
                cap.maturity,
                CapabilityMaturity::Unspecified as i32,
                "`{}` serialized to an unspecified maturity",
                cap.name
            );
            assert_ne!(
                cap.durability,
                CapabilityDurability::Unspecified as i32,
                "`{}` serialized to an unspecified durability",
                cap.name
            );
            assert!(
                cap.contract_version.is_some(),
                "`{}` was served without a contract version",
                cap.name
            );
        }
    }

    /// A self-contradictory manifest must be refused at construction rather
    /// than served, because a client that negotiates against it would be told
    /// it may rely on something the server cannot deliver.
    #[test]
    fn an_inconsistent_manifest_is_refused_rather_than_served() {
        let dishonest = CapabilityManifest {
            server_version: "0.0.0".to_string(),
            build_revision: None,
            capabilities: vec![
                Capability::new(
                    names::HNSW_SEARCH,
                    ContractVersion::new(1, 0),
                    Maturity::Supported,
                    Durability::Ephemeral,
                )
                .guarantee("g")
                .limit("l"),
            ],
        };

        assert!(
            CapabilityServer::with_manifest(dishonest).is_err(),
            "a manifest claiming production maturity for ephemeral state must \
             not reach a client"
        );
    }

    #[tokio::test]
    async fn served_guarantees_and_limits_are_not_dropped_in_translation() {
        let server = CapabilityServer::new();
        let served = server
            .get_capabilities(Request::new(GetCapabilitiesRequest {}))
            .await
            .expect("get_capabilities")
            .into_inner();

        let hnsw = served
            .capabilities
            .iter()
            .find(|c| c.name == names::HNSW_SEARCH)
            .expect("hnsw search is advertised");
        assert!(
            hnsw.limits
                .iter()
                .any(|l| l.contains("does not survive restart")),
            "the ephemerality of the served index must reach the client, since \
             it is the reason the capability is not production-ready"
        );
    }
}
