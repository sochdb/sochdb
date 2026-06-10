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

//! # FFI Panic Firewall
//!
//! A Rust panic that unwinds across an `extern "C"` boundary into a foreign
//! frame is undefined behavior (under `panic = "unwind"`) or aborts the whole
//! host process (under `panic = "abort"`). Neither is acceptable for a library:
//! bad input must surface as a typed/sentinel error, not kill the host.
//!
//! Every `extern "C"` entry point in this crate wraps its body in
//! [`ffi_guard!`], which uses `catch_unwind` to stop any panic at the boundary
//! and return a stable [`FfiDefault`] sentinel for the function's return type.
//! With this firewall at *every* boundary, building with `panic = "unwind"` is
//! sound — a caught panic becomes an error code rather than a process abort.

use std::os::raw::c_int;

/// A stable, safe error sentinel returned from an FFI entry point when its body
/// panics. Implemented for every C return type used at the boundary.
pub trait FfiDefault {
    /// The value to return to the C caller when the Rust body panicked.
    fn ffi_default() -> Self;
}

impl FfiDefault for () {
    #[inline]
    fn ffi_default() -> Self {}
}

impl<T> FfiDefault for *mut T {
    #[inline]
    fn ffi_default() -> Self {
        std::ptr::null_mut()
    }
}

impl<T> FfiDefault for *const T {
    #[inline]
    fn ffi_default() -> Self {
        std::ptr::null()
    }
}

impl FfiDefault for c_int {
    #[inline]
    fn ffi_default() -> Self {
        -1
    }
}

impl FfiDefault for usize {
    #[inline]
    fn ffi_default() -> Self {
        0
    }
}

impl FfiDefault for u64 {
    #[inline]
    fn ffi_default() -> Self {
        0
    }
}

impl FfiDefault for i64 {
    #[inline]
    fn ffi_default() -> Self {
        -1
    }
}

impl FfiDefault for f32 {
    #[inline]
    fn ffi_default() -> Self {
        0.0
    }
}

/// Wrap an FFI entry-point body so that no panic can cross the C ABI.
///
/// On a caught panic, returns the [`FfiDefault`] sentinel for the function's
/// return type instead of unwinding into foreign frames. `AssertUnwindSafe` is
/// required because raw pointers captured from the C caller are not
/// `UnwindSafe`; this is sound because the firewall converts the panic into a
/// value return and performs no further work on the poisoned state.
macro_rules! ffi_guard {
    ($body:block) => {{
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(move || $body)) {
            ::std::result::Result::Ok(__ffi_ok) => __ffi_ok,
            ::std::result::Result::Err(_) => $crate::ffi_firewall::FfiDefault::ffi_default(),
        }
    }};
}
