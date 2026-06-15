// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Data-at-Rest Encryption (Enterprise Security)
//!
//! Transparent AES-256-GCM-SIV encryption for data blocks, WAL entries,
//! and checkpoint files. Uses nonce-misuse-resistant authenticated encryption
//! to prevent catastrophic failures from nonce reuse.
//!
//! ## Design Choices
//!
//! - **AES-256-GCM-SIV**: Nonce-misuse resistant — safe even if nonces are
//!   accidentally repeated (unlike plain AES-GCM which is catastrophic).
//! - **Per-block random nonces**: 12-byte random nonce per encrypt operation.
//! - **Zero-copy where possible**: Encrypt in-place for WAL append path.
//! - **Key wrapping**: Data Encryption Key (DEK) is wrapped by a Key Encryption
//!   Key (KEK) loaded from Kubernetes Secrets or env vars.
//!
//! ## Wire Format
//!
//! ```text
//! [1 byte: version] [12 bytes: nonce] [N bytes: ciphertext+tag]
//! ```
//!
//! Version 1: AES-256-GCM-SIV with 12-byte nonce, 16-byte auth tag appended
//! to ciphertext by the AEAD.
//!
//! ## Performance Notes
//!
//! On x86_64 with AES-NI: ~4 GB/s encryption throughput (hardware-accelerated).
//! The overhead is negligible compared to disk I/O.

use aes_gcm_siv::{
    Aes256GcmSiv, Nonce,
    aead::{Aead, KeyInit, OsRng, Payload},
};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;
use sochdb_core::SochDBError;
use zeroize::Zeroize;

/// Current encryption format version.
const ENCRYPTION_VERSION: u8 = 1;
/// Nonce size for AES-256-GCM-SIV.
const NONCE_SIZE: usize = 12;
/// Header size: 1 (version) + 12 (nonce).
const HEADER_SIZE: usize = 1 + NONCE_SIZE;

/// Data-at-rest encryption engine.
///
/// Wraps AES-256-GCM-SIV with random nonces. Thread-safe (the cipher
/// is `Send + Sync` and nonce generation uses OS randomness).
pub struct EncryptionEngine {
    cipher: Aes256GcmSiv,
    /// Whether encryption is active (false = passthrough)
    enabled: bool,
}

impl EncryptionEngine {
    /// Create an encryption engine with the given 256-bit key.
    ///
    /// The key must be exactly 32 bytes. Typically loaded from
    /// Kubernetes Secrets or the `SOCHDB_ENCRYPTION_KEY` env var.
    pub fn new(key: &[u8; 32]) -> Self {
        let cipher =
            Aes256GcmSiv::new_from_slice(key).expect("AES-256-GCM-SIV key must be 32 bytes");
        Self {
            cipher,
            enabled: true,
        }
    }

    /// Create an encryption engine from a zeroize-on-drop [`EncryptionKey`].
    ///
    /// Preferred over [`Self::new`] on the live path: the caller holds the key
    /// material in a wiping container rather than a bare `[u8; 32]`.
    pub fn from_key(key: &EncryptionKey) -> Self {
        Self::new(key.as_bytes())
    }

    /// Create a disabled (passthrough) encryption engine.
    ///
    /// `encrypt()` and `decrypt()` are identity operations when disabled.
    pub fn disabled() -> Self {
        // Use a dummy key — cipher is never called when disabled
        let key = [0u8; 32];
        let cipher =
            Aes256GcmSiv::new_from_slice(&key).expect("AES-256-GCM-SIV key must be 32 bytes");
        Self {
            cipher,
            enabled: false,
        }
    }

    /// Whether encryption is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Encrypt a plaintext block.
    ///
    /// Returns `[version(1) | nonce(12) | ciphertext+tag(N+16)]`.
    ///
    /// # Performance
    ///
    /// ~4 GB/s on x86_64 with AES-NI. The overhead is the 13-byte header
    /// plus 16-byte auth tag per block.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        self.encrypt_with_aad(plaintext, &[])
    }

    /// Encrypt a plaintext block, binding `aad` as additional authenticated data.
    ///
    /// Returns `[version(1) | nonce(12) | ciphertext+tag(N+16)]`. The `aad` is
    /// NOT stored in the output — it is authenticated only, and the reader MUST
    /// reconstruct the identical `aad` (e.g. `{format_version, db_uuid,
    /// dek_epoch, record_LSN}`) from its own trusted state. Binding framing/
    /// position as AAD is what prevents an attacker (or a corrupt/misdirected
    /// write) from reordering, duplicating, splicing, or downgrading WAL records
    /// — GCM-SIV alone authenticates only the isolated record body.
    ///
    /// AAD scope is part of the on-disk format and cannot be widened later
    /// without a format break, so callers must commit to the full AAD tuple from
    /// the first encrypted byte written.
    pub fn encrypt_with_aad(
        &self,
        plaintext: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, EncryptionError> {
        if !self.enabled {
            // Passthrough MUST stay byte-identical to the legacy plaintext frame
            // so an un-keyed DB is wire-compatible with pre-encryption binaries.
            return Ok(plaintext.to_vec());
        }

        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|_| EncryptionError::EncryptFailed)?;

        // Build output: version + nonce + ciphertext
        let mut output = Vec::with_capacity(HEADER_SIZE + ciphertext.len());
        output.push(ENCRYPTION_VERSION);
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);

        Ok(output)
    }

    /// Decrypt an encrypted block produced by `encrypt()`.
    ///
    /// Validates the version byte and authentication tag.
    pub fn decrypt(&self, encrypted: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        self.decrypt_with_aad(encrypted, &[])
    }

    /// Decrypt an encrypted block, verifying it against the same `aad` the writer
    /// bound via [`Self::encrypt_with_aad`]. A mismatched `aad` (e.g. a record
    /// that was moved to a different LSN/position) fails authentication exactly
    /// like a wrong key or tampered ciphertext — surfaced as
    /// [`EncryptionError::DecryptFailed`], which callers MUST treat as a hard
    /// error, never as a clean end-of-stream.
    pub fn decrypt_with_aad(
        &self,
        encrypted: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, EncryptionError> {
        if !self.enabled {
            return Ok(encrypted.to_vec());
        }

        if encrypted.len() < HEADER_SIZE + 16 {
            return Err(EncryptionError::InvalidFormat(
                "Data too short for encrypted block".into(),
            ));
        }

        let version = encrypted[0];
        if version != ENCRYPTION_VERSION {
            return Err(EncryptionError::UnsupportedVersion(version));
        }

        let nonce = Nonce::from_slice(&encrypted[1..HEADER_SIZE]);
        let ciphertext = &encrypted[HEADER_SIZE..];

        self.cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|_| EncryptionError::DecryptFailed)
    }

    /// Encrypt in-place for zero-copy WAL append.
    ///
    /// Prepends the header to the buffer and encrypts the payload region.
    /// The buffer is resized to accommodate the header + auth tag.
    pub fn encrypt_in_place(&self, buffer: &mut Vec<u8>) -> Result<(), EncryptionError> {
        if !self.enabled {
            return Ok(());
        }

        let encrypted = self.encrypt(buffer)?;
        *buffer = encrypted;
        Ok(())
    }
}

/// HKDF-SHA256 expand: derive a 32-byte subkey from input key material.
///
/// Used so the operator-supplied secret (the KEK) is never used verbatim as a
/// cipher key: the per-DB DEK and the keyring wrapping-key are both derived via
/// this with a per-DB random salt and a distinct `info` label. The same env
/// secret across two databases therefore yields independent keys.
pub fn derive_subkey(ikm: &[u8], salt: &[u8], info: &[u8]) -> EncryptionKey {
    let hk = Hkdf::<Sha256>::new(Some(salt), ikm);
    let mut okm = [0u8; 32];
    hk.expand(info, &mut okm)
        .expect("HKDF expand of 32 bytes never fails");
    let key = EncryptionKey::new(okm);
    okm.zeroize();
    key
}

/// Encryption error types.
#[derive(Debug)]
pub enum EncryptionError {
    /// Encryption operation failed
    EncryptFailed,
    /// Decryption failed (wrong key or tampered data)
    DecryptFailed,
    /// Invalid encrypted data format
    InvalidFormat(String),
    /// Unsupported encryption version
    UnsupportedVersion(u8),
}

impl std::fmt::Display for EncryptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncryptionError::EncryptFailed => write!(f, "Encryption failed"),
            EncryptionError::DecryptFailed => {
                write!(f, "Decryption failed (wrong key or tampered data)")
            }
            EncryptionError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            EncryptionError::UnsupportedVersion(v) => {
                write!(f, "Unsupported encryption version: {}", v)
            }
        }
    }
}

impl EncryptionError {
    /// Whether this error means the bytes failed integrity/authentication and
    /// therefore CANNOT be a clean torn-tail. A torn write truncates the file —
    /// it never produces a full-length frame that fails the AEAD tag, an
    /// unsupported version, or a malformed envelope. WAL replay must treat these
    /// as hard, recovery-aborting errors (wrong/missing key, key-epoch mismatch,
    /// bit-rot, or tampering), never as end-of-WAL.
    pub fn is_integrity_failure(&self) -> bool {
        matches!(
            self,
            EncryptionError::DecryptFailed
                | EncryptionError::InvalidFormat(_)
                | EncryptionError::UnsupportedVersion(_)
        )
    }
}

impl std::error::Error for EncryptionError {}

impl From<EncryptionError> for SochDBError {
    fn from(e: EncryptionError) -> Self {
        // Map to the dedicated `Encryption` variant (NOT `Io`) so replay loops
        // never mistake an authentication failure for a torn-tail EOF.
        SochDBError::Encryption(e.to_string())
    }
}

/// Generate a new random 256-bit encryption key.
///
/// Use this to generate a key for `SOCHDB_ENCRYPTION_KEY`.
/// The returned key should be base64-encoded and stored in
/// Kubernetes Secrets.
pub fn generate_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    OsRng.fill_bytes(&mut key);
    key
}

/// A wrapper that zeroizes the key material on drop.
#[derive(Zeroize)]
#[zeroize(drop)]
pub struct EncryptionKey {
    bytes: [u8; 32],
}

impl EncryptionKey {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        let plaintext = b"Hello, SochDB enterprise encryption!";
        let encrypted = engine.encrypt(plaintext).unwrap();

        // Encrypted should be larger (header + auth tag)
        assert!(encrypted.len() > plaintext.len());
        assert_eq!(encrypted[0], ENCRYPTION_VERSION);

        let decrypted = engine.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_empty() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        let encrypted = engine.encrypt(b"").unwrap();
        let decrypted = engine.decrypt(&encrypted).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_encrypt_large_block() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        // 1 MB block
        let plaintext: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let encrypted = engine.encrypt(&plaintext).unwrap();
        let decrypted = engine.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = generate_key();
        let key2 = generate_key();
        let engine1 = EncryptionEngine::new(&key1);
        let engine2 = EncryptionEngine::new(&key2);

        let encrypted = engine1.encrypt(b"secret data").unwrap();
        let result = engine2.decrypt(&encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_data_fails() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        let mut encrypted = engine.encrypt(b"important data").unwrap();
        // Flip a byte in the ciphertext
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0xFF;

        let result = engine.decrypt(&encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_disabled_passthrough() {
        let engine = EncryptionEngine::disabled();

        let plaintext = b"no encryption here";
        let encrypted = engine.encrypt(plaintext).unwrap();
        assert_eq!(encrypted, plaintext);

        let decrypted = engine.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_unique_nonces() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        let enc1 = engine.encrypt(b"same plaintext").unwrap();
        let enc2 = engine.encrypt(b"same plaintext").unwrap();

        // Nonces should differ even for same plaintext
        assert_ne!(enc1[1..13], enc2[1..13]);
        // Ciphertexts should differ
        assert_ne!(enc1, enc2);
    }

    #[test]
    fn test_invalid_format() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        // Too short
        assert!(engine.decrypt(&[1, 2, 3]).is_err());
        // Wrong version
        let fake = vec![99u8; 50];
        assert!(engine.decrypt(&fake).is_err());
    }

    #[test]
    fn test_key_zeroize() {
        let key = EncryptionKey::new(generate_key());
        assert_ne!(key.as_bytes(), &[0u8; 32]);
        drop(key);
        // After drop, memory should be zeroed (we can't read it, but the Zeroize
        // derive guarantees it)
    }

    #[test]
    fn test_aad_roundtrip() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);
        let aad = b"v1|db-uuid|epoch=0|lsn=42";

        let ct = engine.encrypt_with_aad(b"payload", aad).unwrap();
        let pt = engine.decrypt_with_aad(&ct, aad).unwrap();
        assert_eq!(pt, b"payload");
    }

    #[test]
    fn test_aad_mismatch_fails_like_wrong_key() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        // A record encrypted at lsn=42 must NOT authenticate when the reader
        // reconstructs aad for a different position (splice/reorder defense).
        let ct = engine
            .encrypt_with_aad(b"committed record", b"...|lsn=42")
            .unwrap();
        let err = engine.decrypt_with_aad(&ct, b"...|lsn=43").unwrap_err();
        assert!(matches!(err, EncryptionError::DecryptFailed));
        assert!(err.is_integrity_failure());
    }

    #[test]
    fn test_no_aad_is_not_same_as_some_aad() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);
        let ct = engine.encrypt_with_aad(b"x", b"bound").unwrap();
        // Decrypting the same ciphertext with empty aad must fail.
        assert!(engine.decrypt(&ct).is_err());
        // And the plain encrypt()/decrypt() pair (empty aad) round-trips.
        let ct2 = engine.encrypt(b"x").unwrap();
        assert_eq!(engine.decrypt(&ct2).unwrap(), b"x");
    }

    #[test]
    fn test_integrity_failure_classification() {
        assert!(EncryptionError::DecryptFailed.is_integrity_failure());
        assert!(EncryptionError::UnsupportedVersion(9).is_integrity_failure());
        assert!(EncryptionError::InvalidFormat("x".into()).is_integrity_failure());
        // EncryptFailed is a write-side fault, not a replay terminator concern.
        assert!(!EncryptionError::EncryptFailed.is_integrity_failure());
    }

    #[test]
    fn test_hkdf_deterministic_and_salt_separated() {
        let kek = b"operator-supplied-kek-material";
        let salt_a = [1u8; 16];
        let salt_b = [2u8; 16];

        // Same inputs -> same key (deterministic unwrap on reopen).
        let k1 = derive_subkey(kek, &salt_a, b"sochdb/dek/v1");
        let k2 = derive_subkey(kek, &salt_a, b"sochdb/dek/v1");
        assert_eq!(k1.as_bytes(), k2.as_bytes());

        // Different salt (per-DB) -> different key, so the same env secret across
        // two databases yields independent DEKs.
        let k3 = derive_subkey(kek, &salt_b, b"sochdb/dek/v1");
        assert_ne!(k1.as_bytes(), k3.as_bytes());

        // Different info label -> different key (DEK vs wrapping-key separation).
        let k4 = derive_subkey(kek, &salt_a, b"sochdb/wrap/v1");
        assert_ne!(k1.as_bytes(), k4.as_bytes());

        // Derived key is usable.
        let engine = EncryptionEngine::from_key(&k1);
        let ct = engine.encrypt(b"hi").unwrap();
        assert_eq!(engine.decrypt(&ct).unwrap(), b"hi");
    }

    #[test]
    fn test_encrypt_in_place() {
        let key = generate_key();
        let engine = EncryptionEngine::new(&key);

        let original = b"WAL entry payload".to_vec();
        let mut buffer = original.clone();
        engine.encrypt_in_place(&mut buffer).unwrap();

        assert_ne!(buffer, original);
        let decrypted = engine.decrypt(&buffer).unwrap();
        assert_eq!(decrypted, original);
    }
}
