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

//! Loom model-checking of the lock-free MVCC coordination protocols.
//!
//! ## Why this exists
//!
//! `ConcurrentMvcc` (sochdb-storage/src/mvcc_concurrent.rs) coordinates readers
//! and the single writer through lock-free atomics in an mmap'd metadata page.
//! Ordinary unit tests run one interleaving and cannot surface memory-ordering
//! or compare-exchange races. [`loom`] exhaustively explores *every* legal
//! thread interleaving and memory ordering, turning "passed once" into "cannot
//! race under the C11 memory model".
//!
//! ## What is verified
//!
//! These tests re-encode the exact compare-exchange sequences from the
//! production code — they are deliberately byte-for-byte faithful to the real
//! algorithms so that a bug in the protocol would reproduce here:
//!
//! 1. [`ReaderSlot::try_claim`] — concurrent registration of *distinct* owners
//!    on one free slot. Invariant: at most one claims it (no double-allocation),
//!    and the slot ends owned by exactly the winner.
//! 2. The hybrid-logical-clock CAS advance (`Timestamp` generation). Invariant:
//!    concurrent advances are **unique** (no two callers get the same stamp) and
//!    **monotonic with no lost update** (the final value counts every advance).
//!
//! Loom cannot model the *cross-process* dimension of `ConcurrentMvcc` (the
//! `writer_lock` is keyed by pid and mmap-shared across processes); it models
//! the intra-process thread interleavings of the same atomic protocols, which
//! is precisely where compare-exchange ordering bugs live.
//!
//! ## Running
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test -p sochdb-storage --test loom_concurrency
//! ```
//!
//! Without `--cfg loom` this file compiles to nothing, so it adds zero cost to
//! the normal build/test cycle.

#![cfg(loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Faithful reproduction of `ReaderSlot::try_claim`'s claim sequence:
/// load the current owner, reject if held by another owner, otherwise CAS it to
/// ours. Returns whether this owner won the slot.
fn try_claim(slot: &AtomicU32, my_pid: u32) -> bool {
    let current_pid = slot.load(Ordering::Acquire);

    // Only claim if free (0) or already ours — matches production guard.
    if current_pid != 0 && current_pid != my_pid {
        return false;
    }

    slot.compare_exchange(current_pid, my_pid, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

/// Faithful reproduction of the HLC `compare_exchange` advance loop: read the
/// last value and publish `last + 1`, retrying on contention. (The production
/// clock packs physical/logical halves; the lost-update / uniqueness invariant
/// it relies on is exactly this monotonic CAS, which is what we model.)
fn advance(ts: &AtomicU64) -> u64 {
    loop {
        let last = ts.load(Ordering::Acquire);
        let next = last + 1;
        if ts
            .compare_exchange(last, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return next;
        }
        loom::hint::spin_loop();
    }
}

/// Two distinct owners race to claim one free reader slot. Under *every*
/// interleaving exactly one must win, and the slot must end owned by the winner.
#[test]
fn reader_slot_claim_is_exclusive() {
    loom::model(|| {
        let slot = Arc::new(AtomicU32::new(0));

        let s1 = slot.clone();
        let t1 = loom::thread::spawn(move || try_claim(&s1, 1));

        let s2 = slot.clone();
        let t2 = loom::thread::spawn(move || try_claim(&s2, 2));

        let won1 = t1.join().unwrap();
        let won2 = t2.join().unwrap();

        // Exactly one owner may claim a single free slot — never both.
        assert!(
            won1 ^ won2,
            "double-claim or no-claim: won1={won1}, won2={won2}"
        );

        // The slot must be owned by precisely the winner.
        let owner = slot.load(Ordering::Acquire);
        let expected = if won1 { 1 } else { 2 };
        assert_eq!(owner, expected, "slot owner does not match the winner");
    });
}

/// Two threads each take one timestamp concurrently. Under every interleaving
/// the two stamps must be unique, and the final clock value must account for
/// both advances (no lost update).
#[test]
fn hlc_advance_is_unique_and_monotonic() {
    loom::model(|| {
        let ts = Arc::new(AtomicU64::new(0));

        let a = ts.clone();
        let t1 = loom::thread::spawn(move || advance(&a));

        let b = ts.clone();
        let t2 = loom::thread::spawn(move || advance(&b));

        let v1 = t1.join().unwrap();
        let v2 = t2.join().unwrap();

        // No two callers may receive the same timestamp.
        assert_ne!(v1, v2, "duplicate timestamp handed to two callers");

        // Every advance is reflected in the final value (no lost update).
        let final_ts = ts.load(Ordering::Acquire);
        assert_eq!(final_ts, 2, "lost a timestamp advance: final={final_ts}");
        assert_eq!(v1.max(v2), 2, "highest stamp must equal the final clock");
        assert!(v1 >= 1 && v2 >= 1, "timestamps must be strictly positive");
    });
}
