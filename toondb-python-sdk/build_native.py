#!/usr/bin/env python3
"""
Build script for ToonDB Python SDK with bundled native binaries and FFI libraries.

This script:
1. Builds the Rust toondb-bulk binary for the current platform
2. Builds FFI libraries (libtoondb_storage, libtoondb_index) for the current platform
3. Copies binaries to src/toondb/_bin/<platform>/
4. Copies libraries to src/toondb/lib/<platform>/
5. Allows building wheels with bundled native code

Usage:
    python build_native.py          # Build for current platform
    python build_native.py --all    # Build for all target platforms (requires cross)
    python build_native.py --clean  # Remove bundled binaries and libraries
    python build_native.py --libs   # Build only FFI libraries (skip toondb-bulk)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# Supported target triples for cross-compilation
TARGETS = {
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
    ("darwin", "aarch64"): "aarch64-apple-darwin",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
}

# FFI libraries to bundle
FFI_LIBS = {
    "toondb-storage": {
        "darwin": "libtoondb_storage.dylib",
        "linux": "libtoondb_storage.so",
        "windows": "toondb_storage.dll",
    },
    "toondb-index": {
        "darwin": "libtoondb_index.dylib",
        "linux": "libtoondb_index.so",
        "windows": "toondb_index.dll",
    },
}


def get_platform_dir() -> str:
    """Get the platform directory name for the current system."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Normalize machine names
    if machine in ("x86_64", "amd64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "aarch64"
    
    return f"{system}-{machine}"


def get_os_name() -> str:
    """Get normalized OS name."""
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    elif system.startswith("linux"):
        return "linux"
    elif system.startswith(("win", "mingw", "msys", "cygwin")):
        return "windows"
    return system


def get_binary_name() -> str:
    """Get the binary name for the current platform."""
    if platform.system().lower() == "windows":
        return "toondb-bulk.exe"
    return "toondb-bulk"


def find_workspace_root() -> Path:
    """Find the ToonDB workspace root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "Cargo.toml").exists():
            with open(current / "Cargo.toml") as f:
                if "[workspace]" in f.read():
                    return current
        current = current.parent
    raise RuntimeError("Could not find ToonDB workspace root")


def build_binary(target: str | None = None, release: bool = True) -> Path:
    """Build the toondb-bulk binary."""
    workspace = find_workspace_root()
    
    cmd = ["cargo", "build", "-p", "toondb-tools"]
    if release:
        cmd.append("--release")
    if target:
        cmd.extend(["--target", target])
    
    print(f"Building toondb-bulk: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workspace, check=True)
    
    # Find the built binary
    if target:
        binary_dir = workspace / "target" / target / ("release" if release else "debug")
    else:
        binary_dir = workspace / "target" / ("release" if release else "debug")
    
    binary_name = get_binary_name()
    binary_path = binary_dir / binary_name
    
    if not binary_path.exists():
        raise RuntimeError(f"Binary not found: {binary_path}")
    
    return binary_path


def build_ffi_libraries(target: str | None = None, release: bool = True) -> dict[str, Path]:
    """Build FFI libraries (libtoondb_storage, libtoondb_index)."""
    workspace = find_workspace_root()
    os_name = get_os_name()
    
    # Build storage library
    cmd = ["cargo", "build", "-p", "toondb-storage"]
    if release:
        cmd.append("--release")
    if target:
        cmd.extend(["--target", target])
    
    print(f"Building toondb-storage: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workspace, check=True)
    
    # Build index library
    cmd = ["cargo", "build", "-p", "toondb-index"]
    if release:
        cmd.append("--release")
    if target:
        cmd.extend(["--target", target])
    
    print(f"Building toondb-index: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workspace, check=True)
    
    # Find built libraries
    if target:
        lib_dir = workspace / "target" / target / ("release" if release else "debug")
    else:
        lib_dir = workspace / "target" / ("release" if release else "debug")
    
    libs = {}
    for crate_name, lib_names in FFI_LIBS.items():
        lib_name = lib_names.get(os_name)
        if lib_name:
            lib_path = lib_dir / lib_name
            if lib_path.exists():
                libs[crate_name] = lib_path
            else:
                print(f"Warning: Library not found: {lib_path}")
    
    return libs


def install_binary(binary_path: Path, target_platform: str | None = None) -> Path:
    """Install binary to the package _bin directory."""
    pkg_dir = Path(__file__).parent / "src" / "toondb" / "_bin"
    
    if target_platform is None:
        target_platform = get_platform_dir()
    
    dest_dir = pkg_dir / target_platform
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = dest_dir / binary_path.name
    print(f"Installing binary: {binary_path} -> {dest_path}")
    
    shutil.copy2(binary_path, dest_path)
    
    # Make executable on Unix
    if platform.system() != "Windows":
        os.chmod(dest_path, 0o755)
    
    return dest_path


def install_ffi_libraries(libs: dict[str, Path], target_platform: str | None = None) -> list[Path]:
    """Install FFI libraries to the package lib directory."""
    pkg_dir = Path(__file__).parent / "src" / "toondb" / "lib"
    
    if target_platform is None:
        target_platform = get_platform_dir()
    
    dest_dir = pkg_dir / target_platform
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    installed = []
    for crate_name, lib_path in libs.items():
        dest_path = dest_dir / lib_path.name
        print(f"Installing library: {lib_path} -> {dest_path}")
        shutil.copy2(lib_path, dest_path)
        
        # Make executable on Unix (shared libs need this)
        if platform.system() != "Windows":
            os.chmod(dest_path, 0o755)
        
        installed.append(dest_path)
    
    return installed


def clean() -> None:
    """Remove all bundled binaries and libraries."""
    pkg_base = Path(__file__).parent / "src" / "toondb"
    
    bin_dir = pkg_base / "_bin"
    if bin_dir.exists():
        print(f"Removing: {bin_dir}")
        shutil.rmtree(bin_dir)
    
    lib_dir = pkg_base / "lib"
    if lib_dir.exists():
        print(f"Removing: {lib_dir}")
        shutil.rmtree(lib_dir)
    
    print("✓ Cleaned bundled binaries and libraries")


def build_current(libs_only: bool = False) -> None:
    """Build and install binaries + libraries for current platform."""
    platform_dir = get_platform_dir()
    
    # Build FFI libraries first
    print("\n=== Building FFI Libraries ===")
    libs = build_ffi_libraries()
    install_ffi_libraries(libs)
    print(f"✓ Installed {len(libs)} libraries for {platform_dir}")
    
    if not libs_only:
        # Build toondb-bulk binary
        print("\n=== Building toondb-bulk Binary ===")
        binary = build_binary()
        install_binary(binary)
        print(f"✓ Installed {binary.name} for {platform_dir}")
    
    print(f"\n✓ Build complete for {platform_dir}")


def build_all() -> None:
    """Build for all supported platforms using cross-compilation."""
    # Check for cross-rs
    if shutil.which("cross") is None:
        print("Warning: 'cross' not found. Install with: cargo install cross")
        print("Falling back to native build only.")
        build_current()
        return
    
    current_system = platform.system().lower()
    current_machine = platform.machine().lower()
    if current_machine in ("x86_64", "amd64"):
        current_machine = "x86_64"
    elif current_machine in ("arm64", "aarch64"):
        current_machine = "aarch64"
    
    for (system, machine), target in TARGETS.items():
        # Skip if cross-compiling across OS (needs special setup)
        if system != current_system:
            print(f"Skipping {system}-{machine} (cross-OS compilation not configured)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Building for {system}-{machine} ({target})")
        print('='*60)
        
        platform_dir = f"{system}-{machine}"
        
        try:
            # Build and install FFI libraries
            libs = build_ffi_libraries(target=target)
            install_ffi_libraries(libs, platform_dir)
            
            # Build and install binary
            binary = build_binary(target=target)
            install_binary(binary, platform_dir)
            
            print(f"✓ Build complete for {platform_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to build for {target}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ToonDB native binaries and FFI libraries")
    parser.add_argument("--all", action="store_true", help="Build for all platforms")
    parser.add_argument("--clean", action="store_true", help="Remove bundled binaries and libraries")
    parser.add_argument("--debug", action="store_true", help="Build debug instead of release")
    parser.add_argument("--libs", action="store_true", help="Build only FFI libraries (skip toondb-bulk)")
    
    args = parser.parse_args()
    
    if args.clean:
        clean()
    elif args.all:
        build_all()
    else:
        build_current(libs_only=args.libs)


if __name__ == "__main__":
    main()
