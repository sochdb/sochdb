#!/usr/bin/env python3
"""Single-source version bumper for the SochDB monorepo.

The release version lives in exactly ONE authoritative place:

    Cargo.toml  ->  [workspace.package] version

Every workspace member crate inherits it via ``version.workspace = true``.
But the version also has to appear in places that *cannot* inherit from the
Cargo workspace -- the excluded ``sochdb-python`` crate, internal path-dependency
pins, the Python/npm SDK manifests, the Helm chart and the cloud marketplace
listings. Those are what drift out of sync by hand.

This script keeps them all aligned with one command:

    python update_version.py 2.0.3     # set the release everywhere
    python update_version.py            # re-sync to whatever root Cargo.toml says
    python update_version.py --check    # CI guard: fail if anything is out of sync

It deliberately does NOT touch components that are versioned on their own
release train (``deploy/operator``, ``examples/rust``) or the local-only 0.1.0
helper crates (``sochdb-bench``, ``sochdb-simulation``, ``test_analytics_rust``),
nor historical CHANGELOG / RELEASE_NOTES entries.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# The one authoritative declaration. Reading/writing here defines the release.
ROOT_CARGO = ROOT / "Cargo.toml"
ROOT_VERSION_RE = re.compile(
    r"(\[workspace\.package\][^\[]*?\bversion\s*=\s*\")(\d+\.\d+\.\d+)(\")",
    re.DOTALL,
)


def read_canonical() -> str:
    text = ROOT_CARGO.read_text(encoding="utf-8")
    m = ROOT_VERSION_RE.search(text)
    if not m:
        sys.exit("error: could not find [workspace.package] version in Cargo.toml")
    return m.group(2)


def rules_for(version: str) -> list[tuple[Path, list[tuple[re.Pattern[str], str]]]]:
    """(file, [(pattern, replacement)]) edits that pin `version`.

    Replacements use a capturing group 1 for the prefix and group 3 (where
    present) for the suffix, so only the version token itself is rewritten and
    the rule is idempotent.
    """
    v = version
    return [
        # --- Cargo: the one source of truth + the excluded crate ---
        (ROOT_CARGO, [(ROOT_VERSION_RE, rf"\g<1>{v}\g<3>")]),
        (
            ROOT / "sochdb-python" / "Cargo.toml",
            [(re.compile(r'(?m)^(version\s*=\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        (
            ROOT / "sochdb-python" / "pyproject.toml",
            [(re.compile(r'(?m)^(version\s*=\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        # --- Internal path-dependency pins across every Cargo.toml ---
        # (applied separately to all manifests below; listed here as a no-op
        #  anchor so the count report includes them)
        # --- Python packaging ---
        (
            ROOT / "sochdb-python" / "python" / "sochdb" / "__init__.py",
            [(re.compile(r'(__version__\s*=\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        (
            ROOT / "sochdb-sdk" / "python" / "pyproject.toml",
            [(re.compile(r'(?m)^(version\s*=\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        (
            ROOT / "sochdb-sdk" / "python" / "sochdb_sdk" / "__init__.py",
            [(re.compile(r'(__version__\s*=\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        (
            ROOT / "sochdb-sdk" / "python" / "sochdb_sdk" / "client.py",
            [(re.compile(r'(__version__\s*=\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        # --- Node SDK ---
        (
            ROOT / "sochdb-sdk" / "node" / "package.json",
            [(re.compile(r'("version"\s*:\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        # --- Helm chart + dev values ---
        (
            ROOT / "deploy" / "helm" / "sochdb" / "Chart.yaml",
            [
                (re.compile(r'(?m)^(version:\s*)(\d+\.\d+\.\d+)\s*$'), rf"\g<1>{v}"),
                (re.compile(r'(?m)^(appVersion:\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>"),
            ],
        ),
        (
            ROOT / "deploy" / "helm" / "sochdb" / "values-minikube.yaml",
            [(re.compile(r'(tag:\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>")],
        ),
        # --- Cloud marketplace listings (config values + docker-tag examples) ---
        (
            ROOT / "deploy" / "marketplace" / "gcp" / "schema.yaml",
            [
                (re.compile(r'(publishedVersion:\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>"),
                (re.compile(r'(SochDB\s+)(\d+\.\d+\.\d+)(\s+\u2014)'), rf"\g<1>{v}\g<3>"),
                (re.compile(r'(sochdb-grpc:)(\d+\.\d+\.\d+)'), rf"\g<1>{v}"),
            ],
        ),
        (
            ROOT / "deploy" / "marketplace" / "aws" / "product.yaml",
            [
                (re.compile(r'(tag:\s*")(\d+\.\d+\.\d+)(")'), rf"\g<1>{v}\g<3>"),
                (re.compile(r'(sochdb-grpc:)(\d+\.\d+\.\d+)'), rf"\g<1>{v}"),
            ],
        ),
        (
            ROOT / "deploy" / "marketplace" / "azure" / "offer.yaml",
            [
                (re.compile(r'(--version\s+)(\d+\.\d+\.\d+)'), rf"\g<1>{v}"),
                (re.compile(r'(sochdb-grpc:)(\d+\.\d+\.\d+)'), rf"\g<1>{v}"),
            ],
        ),
    ]


# Internal path-dependency version pins, e.g.
#   sochdb-core = { version = "2.0.0", path = "../sochdb-core" }
#   sochdb      = { version = "2.0.0", path = "../sochdb-client" }
# These appear in many manifests and must track the release version.
INTERNAL_DEP_RE = re.compile(
    r'(\bsochdb(?:-[\w-]+)?\s*=\s*\{[^}]*?\bversion\s*=\s*")(\d+\.\d+\.\d+)(")'
)
# Manifests whose own [package] version is intentionally independent and must
# never be rewritten by the internal-dep pass touching their package stanza.
SKIP_DEP_FILES = {
    ROOT / "deploy" / "operator" / "Cargo.toml",
    ROOT / "examples" / "rust" / "Cargo.toml",
}


def apply_file(path: Path, edits: list[tuple[re.Pattern[str], str]], *, check: bool) -> int:
    if not path.exists():
        print(f"  skip (missing): {path.relative_to(ROOT)}")
        return 0
    text = path.read_text(encoding="utf-8")
    new = text
    total = 0
    for pattern, repl in edits:
        new, n = pattern.subn(repl, new)
        total += n
    if new != text and not check:
        path.write_text(new, encoding="utf-8")
    return total


def bump_internal_deps(version: str, *, check: bool) -> tuple[int, int]:
    repl = rf"\g<1>{version}\g<3>"
    changed_files = 0
    changed_lines = 0
    for cargo in sorted(ROOT.rglob("Cargo.toml")):
        if any(p in ("target", ".venv", "node_modules") for p in cargo.parts) or cargo in SKIP_DEP_FILES:
            continue
        text = cargo.read_text(encoding="utf-8")
        new, n = INTERNAL_DEP_RE.subn(repl, text)
        if n and new != text:
            changed_lines += n
            changed_files += 1
            if not check:
                cargo.write_text(new, encoding="utf-8")
    return changed_files, changed_lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?", help="New version, e.g. 2.0.3. Omit to re-sync to root Cargo.toml.")
    ap.add_argument("--check", action="store_true", help="Report drift without writing (exit 1 if any).")
    args = ap.parse_args()

    if args.version:
        if not SEMVER.match(args.version):
            sys.exit(f"error: '{args.version}' is not a valid X.Y.Z version")
        version = args.version
    else:
        version = read_canonical()

    mode = "checking" if args.check else f"setting"
    print(f"{mode} version -> {version}\n")

    drift = 0
    for path, edits in rules_for(version):
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        n = apply_file(path, edits, check=args.check)
        after = path.read_text(encoding="utf-8") if path.exists() else ""
        # Count how many of the targeted tokens are NOT yet at `version`.
        stale = sum(
            1
            for pattern, _ in edits
            for m in pattern.finditer(before)
            if version not in m.group(0)
        )
        drift += stale
        tag = "ok" if stale == 0 else ("DRIFT" if args.check else "fixed")
        if path.exists():
            print(f"  [{tag:5}] {path.relative_to(ROOT)}  ({n} pins)")

    df, dl = bump_internal_deps(version, check=args.check)
    print(f"  [{'ok' if dl == 0 else ('DRIFT' if args.check else 'fixed'):5}] internal path-dep pins: {dl} in {df} manifests")

    print()
    if args.check:
        if drift:
            print(f"OUT OF SYNC: {drift} declaration(s) differ from {version}. Run: python update_version.py {version}")
            sys.exit(1)
        print(f"All declarations match {version}.")
    else:
        print(f"Done. Canonical source: Cargo.toml [workspace.package] version = \"{version}\".")
        print("Workspace member crates inherit it via `version.workspace = true`.")


if __name__ == "__main__":
    main()
