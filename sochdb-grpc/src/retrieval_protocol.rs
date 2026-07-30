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

//! Pure logic of the v2 retrieval protocol.
//!
//! Everything here is deliberately free of tonic and of server state, because
//! these are the rules that decide whether a request is allowed to run at all.
//! Rules that live inside a request handler get tested through the handler, and
//! testing them through the handler means the negative cases -- the expired
//! token, the widened scope, the replayed batch -- are the ones that quietly do
//! not get written.
//!
//! Four things are defined here:
//!
//! * how a 128-bit identifier crosses a wire that has no 128-bit integer;
//! * what a retrieval capability grants and how it is verified;
//! * how a write operation is named so that a retry is recognisable;
//! * what a response commits to, as a digest.

use crate::proto;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// Contract version this build speaks.
///
/// The compatibility relation is the same one the capability manifest uses:
/// equal major, server minor at least the client's. A newer server serves an
/// older client; the reverse is refused rather than attempted.
pub const CONTRACT_MAJOR: u32 = 2;
pub const CONTRACT_MINOR: u32 = 0;

/// Longest life a capability token may be issued with.
///
/// A grant that outlives the decision behind it is a grant that keeps working
/// after the answer changed. Five minutes is long enough to survive a retry
/// storm and short enough that a leaked token is not a standing key.
pub const MAX_TOKEN_LIFETIME_MS: u64 = 5 * 60 * 1000;

#[derive(Debug, PartialEq, Eq)]
pub enum ProtocolError {
    /// The client speaks a major version this server does not implement.
    IncompatibleMajor { server: u32, client: u32 },
    /// The client requires a minor version newer than this server.
    ServerTooOld { server: u32, client: u32 },
    /// No capability token was presented.
    MissingCapability,
    /// The signature does not match the scope it covers.
    InvalidSignature,
    /// The token was valid but has expired.
    Expired { expired_at_ms: u64, now_ms: u64 },
    /// The token was issued for longer than any token may live.
    LifetimeTooLong { requested_ms: u64, maximum_ms: u64 },
    /// The token grants a different index than the request addresses.
    WrongIndex { granted: String, requested: String },
    /// The token was issued to a different tenant than the caller.
    WrongTenant { granted: String, caller: String },
    /// The token grants a different operation than the request performs.
    WrongOperation,
    /// The token was issued under a different policy scope.
    PolicyScopeMismatch { granted: String, requested: String },
    /// The caller's deadline has already passed.
    DeadlineExceeded { deadline_ms: u64, now_ms: u64 },
    /// The request pinned a generation the server cannot answer from.
    StaleGeneration { required: u64, active: u64 },
    /// The index object id is not bound to any index.
    UnboundIndex { object_id: u128 },
    /// The request is internally inconsistent.
    Malformed(String),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompatibleMajor { server, client } => write!(
                f,
                "contract major {} is not implemented by this server (major {})",
                client, server
            ),
            Self::ServerTooOld { server, client } => write!(
                f,
                "client requires contract minor {}, server speaks {}",
                client, server
            ),
            Self::MissingCapability => write!(f, "no retrieval capability was presented"),
            Self::InvalidSignature => write!(f, "capability signature does not match its scope"),
            Self::Expired {
                expired_at_ms,
                now_ms,
            } => write!(
                f,
                "capability expired at {}ms, now {}ms",
                expired_at_ms, now_ms
            ),
            Self::LifetimeTooLong {
                requested_ms,
                maximum_ms,
            } => write!(
                f,
                "capability lifetime {}ms exceeds the maximum {}ms",
                requested_ms, maximum_ms
            ),
            Self::WrongIndex { granted, requested } => write!(
                f,
                "capability grants index '{}', request addresses '{}'",
                granted, requested
            ),
            Self::WrongTenant { granted, caller } => write!(
                f,
                "capability was issued to tenant '{}', caller is '{}'",
                granted, caller
            ),
            Self::WrongOperation => write!(f, "capability does not grant this operation"),
            Self::PolicyScopeMismatch { granted, requested } => write!(
                f,
                "capability was issued under policy scope '{}', request declares '{}'",
                granted, requested
            ),
            Self::DeadlineExceeded {
                deadline_ms,
                now_ms,
            } => write!(
                f,
                "deadline {}ms already passed at {}ms",
                deadline_ms, now_ms
            ),
            Self::StaleGeneration { required, active } => write!(
                f,
                "request requires index generation {}, active generation is {}",
                required, active
            ),
            Self::UnboundIndex { object_id } => {
                write!(
                    f,
                    "index object {:#034x} is not bound to any index",
                    object_id
                )
            }
            Self::Malformed(m) => write!(f, "malformed request: {}", m),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// Split a 128-bit value into the wire representation.
pub fn to_wire(value: u128) -> proto::Uint128 {
    proto::Uint128 {
        high: (value >> 64) as u64,
        low: value as u64,
    }
}

/// Reassemble a 128-bit value from the wire representation.
pub fn from_wire(value: &proto::Uint128) -> u128 {
    ((value.high as u128) << 64) | (value.low as u128)
}

/// Everything a capability token authorises. The signature covers exactly this
/// and nothing else, so any field a verifier does not include here is a field
/// an attacker may change freely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityScope {
    pub index_key: String,
    pub tenant_id: String,
    pub expires_at_unix_ms: u64,
    pub policy_scope_digest: String,
    pub operation: i32,
}

impl CapabilityScope {
    /// Canonical bytes covered by the signature.
    ///
    /// Fields are length-prefixed rather than joined by a separator. With a
    /// separator, a tenant named `a` with index `b:c` and a tenant named `a:b`
    /// with index `c` produce the same bytes, and one token verifies for both.
    fn canonical(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for field in [
            self.index_key.as_bytes(),
            self.tenant_id.as_bytes(),
            self.policy_scope_digest.as_bytes(),
        ] {
            out.extend_from_slice(&(field.len() as u64).to_be_bytes());
            out.extend_from_slice(field);
        }
        out.extend_from_slice(&self.expires_at_unix_ms.to_be_bytes());
        out.extend_from_slice(&self.operation.to_be_bytes());
        out
    }
}

/// Issues and verifies retrieval capabilities.
///
/// The key never leaves the process. A token is a statement by whoever holds
/// the key that a specific caller may do a specific thing to a specific index
/// until a specific time; it is not an identity and carries no privileges of
/// its own.
#[derive(Clone)]
pub struct CapabilityIssuer {
    key: Vec<u8>,
}

impl CapabilityIssuer {
    pub fn new(key: impl Into<Vec<u8>>) -> Self {
        Self { key: key.into() }
    }

    fn tag(&self, scope: &CapabilityScope) -> Vec<u8> {
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .expect("HMAC accepts keys of any length; this cannot fail");
        mac.update(&scope.canonical());
        mac.finalize().into_bytes().to_vec()
    }

    /// Issue a token for a scope.
    pub fn issue(&self, scope: CapabilityScope) -> proto::RetrievalCapabilityToken {
        let signature = self.tag(&scope);
        proto::RetrievalCapabilityToken {
            index_key: scope.index_key,
            tenant_id: scope.tenant_id,
            expires_at_unix_ms: scope.expires_at_unix_ms,
            policy_scope_digest: scope.policy_scope_digest,
            operation: scope.operation,
            signature,
        }
    }

    /// Verify a token against what the request actually asks for.
    ///
    /// The order matters: the signature is checked before anything is read from
    /// the token as if it were true. Checking expiry first would mean an
    /// attacker learns whether a forged token's expiry was plausible.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        token: Option<&proto::RetrievalCapabilityToken>,
        index_key: &str,
        caller_tenant: &str,
        operation: proto::RetrievalOperation,
        policy_scope_digest: &str,
        now_ms: u64,
    ) -> Result<(), ProtocolError> {
        let token = token.ok_or(ProtocolError::MissingCapability)?;

        let scope = CapabilityScope {
            index_key: token.index_key.clone(),
            tenant_id: token.tenant_id.clone(),
            expires_at_unix_ms: token.expires_at_unix_ms,
            policy_scope_digest: token.policy_scope_digest.clone(),
            operation: token.operation,
        };

        let expected = self.tag(&scope);
        // Constant-time comparison: a byte-by-byte early exit leaks how much of
        // a guessed signature was right, which is enough to forge one.
        if !constant_time_eq(&expected, &token.signature) {
            return Err(ProtocolError::InvalidSignature);
        }

        if token.expires_at_unix_ms <= now_ms {
            return Err(ProtocolError::Expired {
                expired_at_ms: token.expires_at_unix_ms,
                now_ms,
            });
        }
        let remaining = token.expires_at_unix_ms - now_ms;
        if remaining > MAX_TOKEN_LIFETIME_MS {
            return Err(ProtocolError::LifetimeTooLong {
                requested_ms: remaining,
                maximum_ms: MAX_TOKEN_LIFETIME_MS,
            });
        }
        if token.index_key != index_key {
            return Err(ProtocolError::WrongIndex {
                granted: token.index_key.clone(),
                requested: index_key.to_string(),
            });
        }
        if token.tenant_id != caller_tenant {
            return Err(ProtocolError::WrongTenant {
                granted: token.tenant_id.clone(),
                caller: caller_tenant.to_string(),
            });
        }
        if token.operation != operation as i32 {
            return Err(ProtocolError::WrongOperation);
        }
        if token.policy_scope_digest != policy_scope_digest {
            return Err(ProtocolError::PolicyScopeMismatch {
                granted: token.policy_scope_digest.clone(),
                requested: policy_scope_digest.to_string(),
            });
        }
        Ok(())
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Check the negotiated contract version.
pub fn check_contract(
    version: Option<&proto::RetrievalContractVersion>,
) -> Result<(), ProtocolError> {
    // An absent version means a client that predates versioning. There is no
    // such client, so treating absence as "assume current" would only ever
    // help a malformed request pass.
    let version = version
        .ok_or_else(|| ProtocolError::Malformed("contract_version is required".to_string()))?;
    if version.major != CONTRACT_MAJOR {
        return Err(ProtocolError::IncompatibleMajor {
            server: CONTRACT_MAJOR,
            client: version.major,
        });
    }
    if version.minor > CONTRACT_MINOR {
        return Err(ProtocolError::ServerTooOld {
            server: CONTRACT_MINOR,
            client: version.minor,
        });
    }
    Ok(())
}

/// Check a caller-supplied deadline.
///
/// A deadline of 0 means the caller did not set one. That is allowed, because
/// forcing every caller to compute one would push them all to a large constant,
/// which is worse than an honest absence.
pub fn check_deadline(deadline_unix_ms: u64, now_ms: u64) -> Result<(), ProtocolError> {
    if deadline_unix_ms != 0 && deadline_unix_ms <= now_ms {
        return Err(ProtocolError::DeadlineExceeded {
            deadline_ms: deadline_unix_ms,
            now_ms,
        });
    }
    Ok(())
}

/// Check that the active generation can satisfy what the request pinned.
///
/// A required generation of 0 means "whatever is current". Any other value must
/// match exactly. Answering from a *newer* generation is refused as firmly as
/// answering from an older one: a caller that pinned a generation did so to
/// join these results to something else derived from that same generation, and
/// a newer answer breaks that join just as thoroughly.
pub fn check_generation(required: u64, active: u64) -> Result<(), ProtocolError> {
    if required != 0 && required != active {
        return Err(ProtocolError::StaleGeneration { required, active });
    }
    Ok(())
}

/// Name a write operation.
///
/// Two batches with the same index, generation, snapshot and sequence are the
/// same batch by definition, so they must produce the same name; anything else
/// differing must produce a different one. This is what lets a caller that does
/// not know whether its last request arrived simply send it again.
pub fn operation_id(
    index_object_id: u128,
    generation: u64,
    source_snapshot_id: &str,
    batch_sequence: u64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(index_object_id.to_be_bytes());
    hasher.update(generation.to_be_bytes());
    hasher.update((source_snapshot_id.len() as u64).to_be_bytes());
    hasher.update(source_snapshot_id.as_bytes());
    hasher.update(batch_sequence.to_be_bytes());
    hex::encode(hasher.finalize())
}

/// Digest of what a response commits to.
///
/// This is not a checksum of the bytes on the wire. It covers the facts that
/// would make the same query return something different -- which index, which
/// generation, which snapshot, and which candidates came back in which order --
/// so that two parties can compare answers without exchanging them.
pub fn evidence_digest(
    index_object_id: u128,
    generation: u64,
    source_snapshot_id: &str,
    candidates: &[(u128, f32)],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(index_object_id.to_be_bytes());
    hasher.update(generation.to_be_bytes());
    hasher.update((source_snapshot_id.len() as u64).to_be_bytes());
    hasher.update(source_snapshot_id.as_bytes());
    hasher.update((candidates.len() as u64).to_be_bytes());
    for (id, distance) in candidates {
        hasher.update(id.to_be_bytes());
        // Bit pattern rather than a formatted decimal: a decimal rendering
        // makes two genuinely different distances collide once they round the
        // same way, which is exactly when a divergence matters most.
        hasher.update(distance.to_bits().to_be_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Current wall clock in milliseconds since the epoch.
pub fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(index: &str, tenant: &str, expires: u64) -> CapabilityScope {
        CapabilityScope {
            index_key: index.to_string(),
            tenant_id: tenant.to_string(),
            expires_at_unix_ms: expires,
            policy_scope_digest: "policy-a".to_string(),
            operation: proto::RetrievalOperation::Search as i32,
        }
    }

    fn ok(
        issuer: &CapabilityIssuer,
        token: &proto::RetrievalCapabilityToken,
    ) -> Result<(), ProtocolError> {
        issuer.verify(
            Some(token),
            "t:docs",
            "t",
            proto::RetrievalOperation::Search,
            "policy-a",
            1_000,
        )
    }

    /// v1 carried ids as uint64 while the index uses u128, which is how the
    /// truncation bug in the v1 search path became possible in the first place.
    /// v2 must round-trip the whole range, including values that have nothing
    /// left once the high half is dropped.
    #[test]
    fn the_full_128_bit_range_survives_the_wire() {
        for value in [
            0u128,
            1,
            u64::MAX as u128,
            u64::MAX as u128 + 1,
            1u128 << 127,
            u128::MAX,
            // A value whose low 64 bits are zero: truncation would turn this
            // into 0 and it would still look like a plausible id.
            0xdead_beef_u128 << 64,
        ] {
            assert_eq!(
                from_wire(&to_wire(value)),
                value,
                "id {} did not survive",
                value
            );
        }
    }

    #[test]
    fn a_freshly_issued_token_verifies() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("t:docs", "t", 100_000));
        assert_eq!(ok(&issuer, &token), Ok(()));
    }

    /// The point of signing the scope is that the scope cannot be widened. Each
    /// of these is an attempt to take a legitimately issued token and use it for
    /// something it was not issued for.
    #[test]
    fn a_token_cannot_be_edited_to_widen_its_scope() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());

        let mut wider_index = issuer.issue(scope("t:docs", "t", 100_000));
        wider_index.index_key = "t:secrets".to_string();
        assert_eq!(
            ok(&issuer, &wider_index),
            Err(ProtocolError::InvalidSignature)
        );

        let mut other_tenant = issuer.issue(scope("t:docs", "t", 100_000));
        other_tenant.tenant_id = "victim".to_string();
        assert_eq!(
            ok(&issuer, &other_tenant),
            Err(ProtocolError::InvalidSignature)
        );

        let mut longer_life = issuer.issue(scope("t:docs", "t", 100_000));
        longer_life.expires_at_unix_ms = u64::MAX;
        assert_eq!(
            ok(&issuer, &longer_life),
            Err(ProtocolError::InvalidSignature)
        );

        let mut more_operations = issuer.issue(scope("t:docs", "t", 100_000));
        more_operations.operation = proto::RetrievalOperation::Ingest as i32;
        assert_eq!(
            ok(&issuer, &more_operations),
            Err(ProtocolError::InvalidSignature)
        );

        let mut other_policy = issuer.issue(scope("t:docs", "t", 100_000));
        other_policy.policy_scope_digest = "policy-b".to_string();
        assert_eq!(
            ok(&issuer, &other_policy),
            Err(ProtocolError::InvalidSignature)
        );
    }

    /// If the signed bytes were a plain concatenation, a token for tenant `a`
    /// on index `b:c` would verify as a token for tenant `a:b` on index `c`.
    /// Length prefixes are the reason it does not.
    #[test]
    fn field_boundaries_cannot_be_shifted_between_fields() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let a = CapabilityScope {
            index_key: "b:c".to_string(),
            tenant_id: "a".to_string(),
            expires_at_unix_ms: 100_000,
            policy_scope_digest: "p".to_string(),
            operation: 1,
        };
        let b = CapabilityScope {
            index_key: "c".to_string(),
            tenant_id: "a:b".to_string(),
            expires_at_unix_ms: 100_000,
            policy_scope_digest: "p".to_string(),
            operation: 1,
        };
        assert_ne!(
            issuer.issue(a).signature,
            issuer.issue(b).signature,
            "two different scopes produced the same signature"
        );
    }

    #[test]
    fn a_token_signed_by_someone_else_is_refused() {
        let mine = CapabilityIssuer::new(b"mine".to_vec());
        let theirs = CapabilityIssuer::new(b"theirs".to_vec());
        let token = theirs.issue(scope("t:docs", "t", 100_000));
        assert_eq!(ok(&mine, &token), Err(ProtocolError::InvalidSignature));
    }

    #[test]
    fn an_expired_token_is_refused() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("t:docs", "t", 500));
        assert_eq!(
            ok(&issuer, &token),
            Err(ProtocolError::Expired {
                expired_at_ms: 500,
                now_ms: 1_000
            })
        );
    }

    /// A token that expires exactly now has expired. Treating the boundary as
    /// still valid means a token is usable for one more tick after the moment
    /// it was said to stop being usable.
    #[test]
    fn a_token_expiring_this_instant_is_already_expired() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("t:docs", "t", 1_000));
        assert!(matches!(
            ok(&issuer, &token),
            Err(ProtocolError::Expired { .. })
        ));
    }

    /// A validly signed token with a year-long life is not an acceptable token.
    /// Without this check, whoever holds the key can mint a standing credential
    /// by accident.
    #[test]
    fn a_token_that_lives_too_long_is_refused_even_though_it_verifies() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("t:docs", "t", 1_000 + MAX_TOKEN_LIFETIME_MS + 1));
        assert_eq!(
            ok(&issuer, &token),
            Err(ProtocolError::LifetimeTooLong {
                requested_ms: MAX_TOKEN_LIFETIME_MS + 1,
                maximum_ms: MAX_TOKEN_LIFETIME_MS
            })
        );
    }

    /// A correctly signed token still only grants what it says. This is the
    /// cross-tenant case: tenant `b` presents its own genuine token while
    /// asking about tenant `a`'s index.
    #[test]
    fn a_genuine_token_does_not_authorise_another_tenants_index() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("b:docs", "b", 100_000));
        let result = issuer.verify(
            Some(&token),
            "a:docs",
            "b",
            proto::RetrievalOperation::Search,
            "policy-a",
            1_000,
        );
        assert_eq!(
            result,
            Err(ProtocolError::WrongIndex {
                granted: "b:docs".to_string(),
                requested: "a:docs".to_string()
            })
        );
    }

    /// A stolen token used by a different caller is refused even for the index
    /// it names, because the token is bound to who it was issued to.
    #[test]
    fn a_stolen_token_does_not_work_for_a_different_caller() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("a:docs", "a", 100_000));
        let result = issuer.verify(
            Some(&token),
            "a:docs",
            "thief",
            proto::RetrievalOperation::Search,
            "policy-a",
            1_000,
        );
        assert!(matches!(result, Err(ProtocolError::WrongTenant { .. })));
    }

    /// A read grant must not perform a write.
    #[test]
    fn a_search_token_cannot_ingest() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("t:docs", "t", 100_000));
        let result = issuer.verify(
            Some(&token),
            "t:docs",
            "t",
            proto::RetrievalOperation::Ingest,
            "policy-a",
            1_000,
        );
        assert_eq!(result, Err(ProtocolError::WrongOperation));
    }

    /// A grant issued while one policy was in force must not survive a change
    /// to that policy. Without this the fastest path to reading newly forbidden
    /// data is a token minted just before the rule changed.
    #[test]
    fn a_token_does_not_survive_a_policy_change() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        let token = issuer.issue(scope("t:docs", "t", 100_000));
        let result = issuer.verify(
            Some(&token),
            "t:docs",
            "t",
            proto::RetrievalOperation::Search,
            "policy-b",
            1_000,
        );
        assert!(matches!(
            result,
            Err(ProtocolError::PolicyScopeMismatch { .. })
        ));
    }

    #[test]
    fn a_missing_token_is_refused_rather_than_treated_as_unrestricted() {
        let issuer = CapabilityIssuer::new(b"k".to_vec());
        assert_eq!(
            issuer.verify(
                None,
                "t:docs",
                "t",
                proto::RetrievalOperation::Search,
                "policy-a",
                1_000
            ),
            Err(ProtocolError::MissingCapability)
        );
    }

    #[test]
    fn contract_negotiation_follows_the_manifest_relation() {
        let v = |major, minor| proto::RetrievalContractVersion { major, minor };
        assert_eq!(check_contract(Some(&v(2, 0))), Ok(()));
        // A server may serve a client that needs less than it offers.
        assert_eq!(check_contract(Some(&v(2, 0))), Ok(()));
        // But not one that needs more.
        assert_eq!(
            check_contract(Some(&v(2, 7))),
            Err(ProtocolError::ServerTooOld {
                server: 0,
                client: 7
            })
        );
        assert!(matches!(
            check_contract(Some(&v(1, 0))),
            Err(ProtocolError::IncompatibleMajor { .. })
        ));
        assert!(matches!(
            check_contract(Some(&v(3, 0))),
            Err(ProtocolError::IncompatibleMajor { .. })
        ));
        assert!(matches!(
            check_contract(None),
            Err(ProtocolError::Malformed(_))
        ));
    }

    #[test]
    fn a_passed_deadline_is_refused_and_an_absent_one_is_allowed() {
        assert_eq!(check_deadline(0, 1_000), Ok(()));
        assert_eq!(check_deadline(2_000, 1_000), Ok(()));
        assert!(matches!(
            check_deadline(999, 1_000),
            Err(ProtocolError::DeadlineExceeded { .. })
        ));
        // Exactly now is already too late: the work has not started yet.
        assert!(matches!(
            check_deadline(1_000, 1_000),
            Err(ProtocolError::DeadlineExceeded { .. })
        ));
    }

    /// A caller that pinned a generation wants that generation. Silently
    /// answering from a different one is the failure this exists to prevent,
    /// and it is a failure in both directions.
    #[test]
    fn a_pinned_generation_is_not_satisfied_by_a_different_one() {
        assert_eq!(check_generation(0, 7), Ok(()));
        assert_eq!(check_generation(7, 7), Ok(()));
        assert_eq!(
            check_generation(6, 7),
            Err(ProtocolError::StaleGeneration {
                required: 6,
                active: 7
            })
        );
        assert_eq!(
            check_generation(8, 7),
            Err(ProtocolError::StaleGeneration {
                required: 8,
                active: 7
            })
        );
    }

    /// The property that makes a retry safe: same inputs, same name.
    #[test]
    fn the_same_batch_always_gets_the_same_operation_id() {
        let a = operation_id(42, 3, "snap-1", 9);
        let b = operation_id(42, 3, "snap-1", 9);
        assert_eq!(a, b);
    }

    /// ...and the property that makes it useful: anything genuinely different
    /// gets a different name, so distinct batches are never mistaken for
    /// retries of each other and silently dropped.
    #[test]
    fn any_difference_produces_a_different_operation_id() {
        let base = operation_id(42, 3, "snap-1", 9);
        for other in [
            operation_id(43, 3, "snap-1", 9),
            operation_id(42, 4, "snap-1", 9),
            operation_id(42, 3, "snap-2", 9),
            operation_id(42, 3, "snap-1", 10),
        ] {
            assert_ne!(base, other);
        }
    }

    /// Length-prefixing the snapshot id matters here for the same reason it
    /// does in the token: without it, snapshot `ab` at sequence 0 and snapshot
    /// `a` at some sequence whose bytes begin with `b` could collide, and a
    /// collision here means a real batch is discarded as a duplicate.
    #[test]
    fn a_snapshot_id_cannot_absorb_the_following_field() {
        assert_ne!(
            operation_id(1, 0, "ab", 0),
            operation_id(1, 0, "a", u64::from_be_bytes(*b"b\0\0\0\0\0\0\0"))
        );
    }

    #[test]
    fn the_evidence_digest_covers_order_and_distance() {
        let base = evidence_digest(1, 2, "snap", &[(10, 0.5), (11, 0.75)]);
        assert_eq!(
            base,
            evidence_digest(1, 2, "snap", &[(10, 0.5), (11, 0.75)])
        );
        // Reordering is a different answer even with the same members.
        assert_ne!(
            base,
            evidence_digest(1, 2, "snap", &[(11, 0.75), (10, 0.5)])
        );
        // A different generation answering with the same ids is still a
        // different answer, because it cannot be joined to the same snapshot.
        assert_ne!(
            base,
            evidence_digest(1, 3, "snap", &[(10, 0.5), (11, 0.75)])
        );
        assert_ne!(
            base,
            evidence_digest(1, 2, "other", &[(10, 0.5), (11, 0.75)])
        );
    }

    /// Two distances that render identically at any fixed precision must not
    /// produce the same digest, or a divergence disappears exactly where it
    /// matters most -- between two answers that look the same in a log.
    #[test]
    fn distances_are_digested_by_bits_not_by_rendering() {
        let a = 1.0f32;
        let b = 1.0f32 + f32::EPSILON;
        assert_ne!(a, b, "test premise no longer holds");
        assert_eq!(
            format!("{:.6}", a),
            format!("{:.6}", b),
            "these were chosen because they render the same"
        );
        assert_ne!(
            evidence_digest(1, 1, "s", &[(1, a)]),
            evidence_digest(1, 1, "s", &[(1, b)])
        );
    }

    #[test]
    fn constant_time_eq_still_compares_correctly() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(constant_time_eq(b"", b""));
    }
}
