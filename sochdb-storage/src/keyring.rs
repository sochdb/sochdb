// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Keyring — KEK/DEK envelope for data-at-rest encryption (Task 3B)
//!
//! The keyring is the per-database key-management substrate that sits in front
//! of [`crate::encryption::EncryptionEngine`]. It exists so that:
//!
//! - The operator-supplied secret (the **KEK**, e.g. `SOCHDB_ENCRYPTION_KEY` or
//!   an embedded `ConnectionConfig.encryption` key) is **never** used verbatim
//!   as the cipher key. Instead a random per-DB **DEK** is generated, and the
//!   KEK only wraps it. Rotating the KEK is then a cheap re-wrap — it does NOT
//!   require re-encrypting any data.
//! - A wrong or missing key is caught **fail-closed at open**, before a single
//!   WAL byte is read, via an authenticated descriptor MAC and a DEK canary —
//!   never silently degrading to a plaintext read or an "empty" database.
//! - A `key_epoch` is reserved from the first encrypted byte, so future DEK
//!   rotation is expressible on disk without a format break.
//!
//! ## On-disk descriptor (`<db_dir>/keyring.json`)
//!
//! The **presence** of this file with `encrypted=true` is the source of truth
//! that a database is encrypted. A plaintext DB has no keyring file at all
//! (preserving byte-compatibility with pre-3B binaries). All byte fields are
//! hex-encoded. The whole descriptor is authenticated by `mac` =
//! HMAC-SHA256(HKDF(KEK, salt, "keyring-mac"), canonical-fields), so an attacker
//! or a bad rollback cannot flip `encrypted` to `false` to force a downgrade.
//!
//! ```text
//! { format_version, encrypted, db_uuid, kek_source_id, key_epoch,
//!   salt, wrapped_dek, canary, mac }
//! ```
//!
//! - `wrapped_dek` = AEAD(HKDF(KEK,salt,"wrap")).encrypt(DEK, aad=wrap_aad)
//! - `canary`      = AEAD(DEK).encrypt(CANARY_TOKEN, aad=canary_aad)
//!
//! On open we (1) verify the MAC with the KEK, (2) unwrap the DEK, (3) decrypt
//! the canary with the DEK. Any failure ⇒ hard error (wrong/missing KEK or
//! tampering). Only after all three pass is the WAL touched.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use zeroize::Zeroize;

use crate::encryption::{EncryptionEngine, EncryptionKey, derive_subkey, generate_key};
use sochdb_core::{Result, SochDBError};

type HmacSha256 = Hmac<Sha256>;

/// Current keyring descriptor format version.
const KEYRING_FORMAT_VERSION: u32 = 1;
/// Keyring file name within the database directory.
pub const KEYRING_FILE_NAME: &str = "keyring.json";
/// Fixed plaintext token sealed under the DEK to detect a wrong key at open.
const CANARY_TOKEN: &[u8] = b"sochdb-keyring-canary-v1";
/// HKDF info labels — keep these stable; they are part of the on-disk contract.
const INFO_WRAP: &[u8] = b"sochdb/keyring/wrap/v1";
const INFO_MAC: &[u8] = b"sochdb/keyring/mac/v1";

/// The resolved encryption state for an opened database.
///
/// `Plaintext` carries a disabled engine (identity passthrough, byte-identical
/// legacy frames). `Encrypted` carries the live DEK-backed engine plus the
/// `db_uuid` and `key_epoch` that the WAL binds into every record's AAD.
pub enum EncryptionState {
    Plaintext,
    Encrypted(ActiveEncryption),
}

impl std::fmt::Debug for EncryptionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately does NOT print key material.
        match self {
            EncryptionState::Plaintext => write!(f, "EncryptionState::Plaintext"),
            EncryptionState::Encrypted(a) => write!(
                f,
                "EncryptionState::Encrypted {{ db_uuid: {}, key_epoch: {} }}",
                hex::encode(a.db_uuid),
                a.key_epoch
            ),
        }
    }
}

impl EncryptionState {
    /// Whether at-rest encryption is active for this database.
    pub fn is_encrypted(&self) -> bool {
        matches!(self, EncryptionState::Encrypted(_))
    }

    /// The engine to hand to the WAL (disabled passthrough if plaintext).
    pub fn engine(&self) -> Arc<EncryptionEngine> {
        match self {
            EncryptionState::Plaintext => Arc::new(EncryptionEngine::disabled()),
            EncryptionState::Encrypted(a) => a.engine.clone(),
        }
    }

    /// 16-byte DB identity bound into WAL AAD (all-zero for plaintext, unused).
    pub fn db_uuid(&self) -> [u8; 16] {
        match self {
            EncryptionState::Plaintext => [0u8; 16],
            EncryptionState::Encrypted(a) => a.db_uuid,
        }
    }

    /// Active DEK epoch (0 for plaintext, unused).
    pub fn key_epoch(&self) -> u32 {
        match self {
            EncryptionState::Plaintext => 0,
            EncryptionState::Encrypted(a) => a.key_epoch,
        }
    }
}

/// Live encryption context for an encrypted database.
pub struct ActiveEncryption {
    pub engine: Arc<EncryptionEngine>,
    pub db_uuid: [u8; 16],
    pub key_epoch: u32,
}

/// On-disk keyring descriptor (hex-encoded byte fields).
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyringFile {
    format_version: u32,
    encrypted: bool,
    db_uuid: String,
    kek_source_id: String,
    key_epoch: u32,
    salt: String,
    wrapped_dek: String,
    canary: String,
    mac: String,
}

/// Resolve the encryption state for a database directory.
///
/// Contract (the file's presence + `encrypted` flag is the source of truth):
/// - **keyring present, `encrypted=true`**: `kek` is REQUIRED. We verify the
///   descriptor MAC, unwrap the DEK, and check the canary. Any failure (wrong
///   key, missing key, tamper) is a hard, fail-closed error — we never fall
///   back to a disabled engine. Returns `Encrypted`.
/// - **keyring present, `encrypted=false`**: plaintext DB. Returns `Plaintext`.
/// - **keyring absent + `kek = Some`**: a *new* encrypted DB. Generates a DEK,
///   wraps it, writes the keyring atomically. `allow_create` MUST be true (the
///   caller asserts this is not an existing plaintext DB). Returns `Encrypted`.
/// - **keyring absent + `kek = None`**: legacy/plaintext DB. Returns `Plaintext`.
pub fn load_or_init(
    db_dir: &Path,
    kek: Option<&EncryptionKey>,
    source_id: &str,
    allow_create: bool,
) -> Result<EncryptionState> {
    let path = keyring_path(db_dir);

    if path.exists() {
        let file: KeyringFile = read_keyring(&path)?;
        if file.format_version != KEYRING_FORMAT_VERSION {
            return Err(SochDBError::Encryption(format!(
                "unsupported keyring format version {} (expected {})",
                file.format_version, KEYRING_FORMAT_VERSION
            )));
        }
        // A keyring file is ONLY ever written for an encrypted database, so its
        // mere presence means a KEK is required. Resolve it fail-closed BEFORE
        // honoring the `encrypted` flag — otherwise an attacker could flip
        // `encrypted` to false (or drop the env key) to force a plaintext
        // downgrade. The descriptor MAC (verified inside `open_encrypted`, and
        // re-checked below for the plaintext-marker case) cannot be recomputed
        // without the KEK, so a forged `encrypted=false` is rejected.
        let kek = kek.ok_or_else(|| {
            SochDBError::Encryption(
                "database has a keyring (encryption configured) but no \
                 encryption key was provided (set the KEK, e.g. \
                 SOCHDB_ENCRYPTION_KEY); refusing to open"
                    .to_string(),
            )
        })?;
        // Authenticate the descriptor with the KEK first. This rejects a forged
        // `encrypted=false` downgrade, since the MAC covers the `encrypted`
        // field and cannot be reforged without the KEK.
        verify_mac(&file, kek)?;
        if !file.encrypted {
            // MAC-authenticated plaintext marker (not written by current code,
            // but honored if it ever authenticates — defensive forward-compat).
            return Ok(EncryptionState::Plaintext);
        }
        open_encrypted(file, kek)
    } else if let Some(kek) = kek {
        if !allow_create {
            return Err(SochDBError::Encryption(
                "an encryption key was provided for a database that has no \
                 keyring (existing plaintext data must be migrated explicitly, \
                 not encrypted in place); refusing to open"
                    .to_string(),
            ));
        }
        create_encrypted(db_dir, &path, kek, source_id)
    } else {
        Ok(EncryptionState::Plaintext)
    }
}

fn keyring_path(db_dir: &Path) -> PathBuf {
    db_dir.join(KEYRING_FILE_NAME)
}

fn read_keyring(path: &Path) -> Result<KeyringFile> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|e| SochDBError::Encryption(format!("malformed keyring: {e}")))
}

/// Build the canonical, deterministic byte string the MAC authenticates.
/// Length-prefixed fields so no two distinct descriptors collide.
fn mac_input(file: &KeyringFile) -> Vec<u8> {
    let mut out = Vec::new();
    let mut push = |b: &[u8]| {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    };
    push(&file.format_version.to_le_bytes());
    push(&[file.encrypted as u8]);
    push(file.db_uuid.as_bytes());
    push(file.kek_source_id.as_bytes());
    push(&file.key_epoch.to_le_bytes());
    push(file.salt.as_bytes());
    push(file.wrapped_dek.as_bytes());
    push(file.canary.as_bytes());
    out
}

fn compute_mac(mac_key: &EncryptionKey, file: &KeyringFile) -> Vec<u8> {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(mac_key.as_bytes())
        .expect("HMAC accepts any key length");
    mac.update(&mac_input(file));
    mac.finalize().into_bytes().to_vec()
}

/// AAD binding the wrapped DEK to this DB identity + epoch + KEK source, so a
/// wrapped DEK cannot be spliced into a different DB/epoch under a shared KEK.
fn wrap_aad(db_uuid: &[u8; 16], epoch: u32, source_id: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(16 + 4 + source_id.len());
    aad.extend_from_slice(db_uuid);
    aad.extend_from_slice(&epoch.to_le_bytes());
    aad.extend_from_slice(source_id.as_bytes());
    aad
}

fn canary_aad(db_uuid: &[u8; 16], epoch: u32) -> Vec<u8> {
    let mut aad = Vec::with_capacity(16 + 4);
    aad.extend_from_slice(db_uuid);
    aad.extend_from_slice(&epoch.to_le_bytes());
    aad
}

fn create_encrypted(
    db_dir: &Path,
    path: &Path,
    kek: &EncryptionKey,
    source_id: &str,
) -> Result<EncryptionState> {
    let mut db_uuid = [0u8; 16];
    {
        use rand::RngCore;
        rand::rngs::OsRng.fill_bytes(&mut db_uuid);
    }
    let mut salt = [0u8; 16];
    {
        use rand::RngCore;
        rand::rngs::OsRng.fill_bytes(&mut salt);
    }
    let epoch: u32 = 0;

    // Random per-DB DEK; this is what actually encrypts data.
    let dek = EncryptionKey::new(generate_key());

    // Wrap the DEK under a wrapping key derived from the KEK.
    let wrap_key = derive_subkey(kek.as_bytes(), &salt, INFO_WRAP);
    let wrap_engine = EncryptionEngine::from_key(&wrap_key);
    let wrapped_dek =
        wrap_engine.encrypt_with_aad(dek.as_bytes(), &wrap_aad(&db_uuid, epoch, source_id))?;

    // Seal a canary under the DEK so a wrong key is detected at open.
    let dek_engine = EncryptionEngine::from_key(&dek);
    let canary = dek_engine.encrypt_with_aad(CANARY_TOKEN, &canary_aad(&db_uuid, epoch))?;

    let mut file = KeyringFile {
        format_version: KEYRING_FORMAT_VERSION,
        encrypted: true,
        db_uuid: hex::encode(db_uuid),
        kek_source_id: source_id.to_string(),
        key_epoch: epoch,
        salt: hex::encode(salt),
        wrapped_dek: hex::encode(&wrapped_dek),
        canary: hex::encode(&canary),
        mac: String::new(),
    };
    let mac_key = derive_subkey(kek.as_bytes(), &salt, INFO_MAC);
    file.mac = hex::encode(compute_mac(&mac_key, &file));

    write_keyring_atomic(db_dir, path, &file)?;

    Ok(EncryptionState::Encrypted(ActiveEncryption {
        engine: Arc::new(dek_engine),
        db_uuid,
        key_epoch: epoch,
    }))
}

/// Verify the descriptor MAC with the KEK. A wrong KEK or any tampering of an
/// authenticated field (e.g. `encrypted` flipped to false, epoch altered) fails
/// here — the MAC cannot be recomputed without the KEK.
fn verify_mac(file: &KeyringFile, kek: &EncryptionKey) -> Result<()> {
    let salt = decode_fixed::<16>(&file.salt, "salt")?;
    let mac_key = derive_subkey(kek.as_bytes(), &salt, INFO_MAC);
    let actual = hex::decode(&file.mac)
        .map_err(|_| SochDBError::Encryption("malformed keyring mac".into()))?;
    // Use HMAC's own vetted constant-time tag verification rather than a
    // hand-rolled compare, so the constant-time property is not at the mercy of
    // optimizer transforms.
    let mut mac = <HmacSha256 as Mac>::new_from_slice(mac_key.as_bytes())
        .expect("HMAC accepts any key length");
    mac.update(&mac_input(file));
    mac.verify_slice(&actual).map_err(|_| {
        SochDBError::Encryption(
            "keyring authentication failed: wrong encryption key or tampered \
             keyring; refusing to open"
                .to_string(),
        )
    })
}

fn open_encrypted(file: KeyringFile, kek: &EncryptionKey) -> Result<EncryptionState> {
    // MAC is already verified by the caller (load_or_init).
    let salt = decode_fixed::<16>(&file.salt, "salt")?;
    let db_uuid = decode_fixed::<16>(&file.db_uuid, "db_uuid")?;
    let epoch = file.key_epoch;

    // Unwrap the DEK.
    let wrap_key = derive_subkey(kek.as_bytes(), &salt, INFO_WRAP);
    let wrap_engine = EncryptionEngine::from_key(&wrap_key);
    let wrapped_dek = hex::decode(&file.wrapped_dek)
        .map_err(|_| SochDBError::Encryption("malformed wrapped_dek".into()))?;
    let mut dek_bytes = wrap_engine
        .decrypt_with_aad(
            &wrapped_dek,
            &wrap_aad(&db_uuid, epoch, &file.kek_source_id),
        )
        .map_err(|_| {
            SochDBError::Encryption(
                "failed to unwrap data key: wrong encryption key; refusing to open".into(),
            )
        })?;
    if dek_bytes.len() != 32 {
        dek_bytes.zeroize();
        return Err(SochDBError::Encryption(
            "unwrapped DEK is not 32 bytes".into(),
        ));
    }
    // Move the plaintext DEK into the zeroize-on-drop wrapper and wipe the
    // transient heap/stack copies the AEAD left behind — the DEK decrypts ALL
    // data, so it must not linger in freed memory / swap / core dumps.
    let mut dek_arr = [0u8; 32];
    dek_arr.copy_from_slice(&dek_bytes);
    dek_bytes.zeroize();
    let dek = EncryptionKey::new(dek_arr);
    dek_arr.zeroize();

    // Canary check: prove the DEK actually decrypts data before touching WAL.
    let dek_engine = EncryptionEngine::from_key(&dek);
    let canary = hex::decode(&file.canary)
        .map_err(|_| SochDBError::Encryption("malformed canary".into()))?;
    let token = dek_engine
        .decrypt_with_aad(&canary, &canary_aad(&db_uuid, epoch))
        .map_err(|_| {
            SochDBError::Encryption(
                "canary decryption failed: wrong encryption key; refusing to open".into(),
            )
        })?;
    if token != CANARY_TOKEN {
        return Err(SochDBError::Encryption(
            "canary token mismatch; refusing to open".into(),
        ));
    }

    Ok(EncryptionState::Encrypted(ActiveEncryption {
        engine: Arc::new(dek_engine),
        db_uuid,
        key_epoch: epoch,
    }))
}

fn decode_fixed<const N: usize>(hexstr: &str, what: &str) -> Result<[u8; N]> {
    let v = hex::decode(hexstr)
        .map_err(|_| SochDBError::Encryption(format!("malformed keyring {what}")))?;
    if v.len() != N {
        return Err(SochDBError::Encryption(format!(
            "keyring {what} wrong length: {} != {N}",
            v.len()
        )));
    }
    let mut a = [0u8; N];
    a.copy_from_slice(&v);
    Ok(a)
}

/// Atomically persist the keyring: write temp, fsync file, rename, fsync dir.
fn write_keyring_atomic(db_dir: &Path, path: &Path, file: &KeyringFile) -> Result<()> {
    fs::create_dir_all(db_dir)?;
    let json = serde_json::to_vec_pretty(file)
        .map_err(|e| SochDBError::Encryption(format!("serialize keyring: {e}")))?;
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&json)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    // fsync the directory so the rename is durable. On Unix a real I/O error
    // here must surface (the keyring's durability rests on this — a lost rename
    // can orphan the DEK). On non-Unix, opening a directory handle isn't
    // supported, so this is best-effort there.
    #[cfg(unix)]
    {
        let dir = fs::File::open(db_dir)?;
        dir.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn kek(seed: u8) -> EncryptionKey {
        EncryptionKey::new([seed; 32])
    }

    #[test]
    fn plaintext_when_no_key_and_no_file() {
        let dir = tempdir().unwrap();
        let st = load_or_init(dir.path(), None, "test", true).unwrap();
        assert!(!st.is_encrypted());
        assert!(!dir.path().join(KEYRING_FILE_NAME).exists());
    }

    #[test]
    fn create_then_reopen_roundtrips_dek() {
        let dir = tempdir().unwrap();
        let st = load_or_init(dir.path(), Some(&kek(7)), "env", true).unwrap();
        assert!(st.is_encrypted());
        let uuid1 = st.db_uuid();
        // Engine actually encrypts.
        let ct = st.engine().encrypt(b"secret").unwrap();
        assert_ne!(ct, b"secret");

        // Reopen with the SAME kek -> same DEK -> can decrypt the ciphertext.
        let st2 = load_or_init(dir.path(), Some(&kek(7)), "env", false).unwrap();
        assert!(st2.is_encrypted());
        assert_eq!(st2.db_uuid(), uuid1);
        assert_eq!(st2.engine().decrypt(&ct).unwrap(), b"secret");
    }

    #[test]
    fn reopen_with_wrong_key_fails_closed() {
        let dir = tempdir().unwrap();
        load_or_init(dir.path(), Some(&kek(1)), "env", true).unwrap();
        let err = load_or_init(dir.path(), Some(&kek(2)), "env", false).unwrap_err();
        // Must be a hard encryption error, NOT a silent plaintext/empty open.
        assert!(matches!(err, SochDBError::Encryption(_)));
    }

    #[test]
    fn reopen_encrypted_without_key_fails_closed() {
        let dir = tempdir().unwrap();
        load_or_init(dir.path(), Some(&kek(1)), "env", true).unwrap();
        let err = load_or_init(dir.path(), None, "env", true).unwrap_err();
        assert!(matches!(err, SochDBError::Encryption(_)));
    }

    #[test]
    fn forging_encrypted_false_is_rejected_by_mac() {
        let dir = tempdir().unwrap();
        load_or_init(dir.path(), Some(&kek(9)), "env", true).unwrap();
        let path = dir.path().join(KEYRING_FILE_NAME);

        // Attacker flips encrypted -> false to force a plaintext downgrade.
        let mut file: KeyringFile = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        file.encrypted = false;
        fs::write(&path, serde_json::to_vec_pretty(&file).unwrap()).unwrap();

        // MAC is verified BEFORE the encrypted flag is honored, and the MAC
        // covers `encrypted`, so the forgery is rejected fail-closed — never a
        // silent plaintext/empty open.
        let err = load_or_init(dir.path(), Some(&kek(9)), "env", false).unwrap_err();
        assert!(matches!(err, SochDBError::Encryption(_)));
    }

    #[test]
    fn keyring_present_but_no_key_fails_even_if_flag_says_plaintext() {
        let dir = tempdir().unwrap();
        load_or_init(dir.path(), Some(&kek(4)), "env", true).unwrap();
        let path = dir.path().join(KEYRING_FILE_NAME);
        let mut file: KeyringFile = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        file.encrypted = false; // forged
        fs::write(&path, serde_json::to_vec_pretty(&file).unwrap()).unwrap();

        // No key supplied + keyring present ⇒ refuse (presence implies encryption).
        let err = load_or_init(dir.path(), None, "env", false).unwrap_err();
        assert!(matches!(err, SochDBError::Encryption(_)));
    }

    #[test]
    fn tampering_authenticated_field_is_rejected() {
        let dir = tempdir().unwrap();
        load_or_init(dir.path(), Some(&kek(5)), "env", true).unwrap();
        let path = dir.path().join(KEYRING_FILE_NAME);

        let mut file: KeyringFile = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        // Tamper an authenticated field while keeping encrypted=true.
        file.key_epoch = 999;
        fs::write(&path, serde_json::to_vec_pretty(&file).unwrap()).unwrap();

        let err = load_or_init(dir.path(), Some(&kek(5)), "env", false).unwrap_err();
        assert!(matches!(err, SochDBError::Encryption(_)));
    }

    #[test]
    fn key_provided_for_existing_plaintext_db_without_create_fails() {
        let dir = tempdir().unwrap();
        // Simulate an existing plaintext DB: a wal.log but no keyring.
        fs::write(dir.path().join("wal.log"), b"legacy").unwrap();
        let err = load_or_init(dir.path(), Some(&kek(3)), "env", false).unwrap_err();
        assert!(matches!(err, SochDBError::Encryption(_)));
    }
}
