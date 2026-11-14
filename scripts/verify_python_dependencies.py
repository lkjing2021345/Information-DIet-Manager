"""Compare hashed locks, an isolated environment and complete pip-audit reports.

Uses only the standard library. Audit the application and installer separately
with pip-audit --disable-pip --require-hashes, then pass both JSON reports here.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any, Iterable

PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)")
HASH = re.compile(r"--hash=sha256:[a-f0-9]{64}")


class DependencyVerificationError(ValueError):
    """A lock, environment or audit does not cover the intended package set."""


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> dict[str, str]:
    """Parse this project's deliberately restricted, fully hashed lock syntax."""
    pins: dict[str, str] = {}
    current: str | None = None
    hashes = 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "--only-binary :all:":
            continue
        parts = line.removesuffix("\\").strip().split()
        match = PIN.fullmatch(parts[0]) if parts else None
        if match:
            if current is not None and hashes == 0:
                raise DependencyVerificationError(f"Missing SHA-256 hashes for {current}")
            current = normalized_name(match[1])
            if current in pins:
                raise DependencyVerificationError(f"Duplicate locked package: {current}")
            pins[current], hashes = match[2], 0
            parts = parts[1:]
        elif current is None:
            raise DependencyVerificationError(f"Unsupported lock syntax at line {number}")
        if any(HASH.fullmatch(part) is None for part in parts):
            raise DependencyVerificationError(f"Unsupported lock syntax at line {number}")
        hashes += len(parts)
    if not pins:
        raise DependencyVerificationError("The lock file contains no pinned packages")
    if hashes == 0:
        raise DependencyVerificationError(f"Missing SHA-256 hashes for {current}")
    return pins


def read_direct_inputs(path: Path) -> dict[str, str]:
    """Read exact direct pins and local -r includes; reject unsafe/loose inputs."""
    base = path.resolve().parent
    visiting: set[Path] = set()
    result: dict[str, str] = {}

    def read(current: Path) -> None:
        current = current.resolve()
        if not current.is_relative_to(base):
            raise DependencyVerificationError("Requirement inputs must remain inside their source directory")
        if current in visiting:
            raise DependencyVerificationError("Requirement inputs contain an include cycle")
        visiting.add(current)
        for number, raw in enumerate(current.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                target = line[3:].strip()
                if not target or "://" in target or Path(target).is_absolute():
                    raise DependencyVerificationError("Requirement includes must use relative local paths")
                read(current.parent / target)
                continue
            match = PIN.fullmatch(line)
            if match is None:
                raise DependencyVerificationError(f"Direct requirements must be exact pins (line {number})")
            name, version = normalized_name(match[1]), match[2]
            if name in result and result[name] != version:
                raise DependencyVerificationError(f"Conflicting direct requirement: {name}")
            result[name] = version
        visiting.remove(current)

    read(path)
    if not result:
        raise DependencyVerificationError("Requirement inputs contain no pinned packages")
    return result


def package_set(records: Iterable[dict[str, Any]], *, audit: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            raise DependencyVerificationError("Package records must include a name and version; skipped packages are forbidden")
        name, version = normalized_name(row["name"]), row["version"]
        if PIN.fullmatch(f"{name}=={version}") is None:
            raise DependencyVerificationError("Invalid package name or version")
        if name in result:
            raise DependencyVerificationError(f"Duplicate package record: {name}")
        if audit and ("skip_reason" in row or not isinstance(row.get("vulns"), list)):
            raise DependencyVerificationError(f"Package was not fully audited: {name}")
        if audit and row["vulns"]:
            raise DependencyVerificationError(f"Known vulnerabilities found for {name}")
        result[name] = version
    return result


def audit_set(report: Any) -> dict[str, str]:
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        raise DependencyVerificationError("Audit JSON must contain a dependencies array")
    return package_set(report["dependencies"], audit=True)


def compare(expected: dict[str, str], actual: dict[str, str], label: str) -> None:
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    versions = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
    if missing or extra or versions:
        raise DependencyVerificationError(f"{label} mismatch: missing={missing}, extra={extra}, wrong_version={versions}")


def verify(
    *, application: dict[str, str], installer: dict[str, str], installed: list[dict[str, Any]],
    application_audit: Any, installer_audit: Any, runtime: dict[str, str] | None = None,
    direct: dict[str, str] | None = None,
) -> dict[str, Any]:
    if application.keys() & installer.keys():
        raise DependencyVerificationError("Application and installer locks must not overlap")
    if runtime is not None:
        compare(runtime, {name: version for name, version in application.items() if name in runtime}, "Runtime/test shared packages")
    if direct is not None:
        compare(direct, {name: version for name, version in application.items() if name in direct}, "Direct input/lock pins")
    compare({**application, **installer}, package_set(installed), "Installed environment")
    compare(application, audit_set(application_audit), "Application audit coverage")
    compare(installer, audit_set(installer_audit), "Installer audit coverage")
    return {"status": "verified", "application_packages": len(application), "installer_packages": len(installer),
            "runtime_packages": len(runtime) if runtime is not None else None,
            "skipped_packages": 0, "known_vulnerabilities": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--installer-lock", type=Path, required=True)
    parser.add_argument("--audit", "--audit-json", type=Path, required=True)
    parser.add_argument("--installer-audit", "--installer-audit-json", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--input", type=Path, help="Direct requirements input, including local -r files")
    parser.add_argument("--installed-json", type=Path, help="Optional pip list --format=json from the target isolated environment")
    args = parser.parse_args()
    try:
        installed = (json.loads(args.installed_json.read_text(encoding="utf-8-sig")) if args.installed_json else
                     [{"name": dist.metadata["Name"], "version": dist.version} for dist in importlib.metadata.distributions()])
        if not isinstance(installed, list):
            raise DependencyVerificationError("Installed-package JSON must be an array")
        result = verify(
            application=parse_lock(args.lock.read_text(encoding="utf-8-sig")),
            installer=parse_lock(args.installer_lock.read_text(encoding="utf-8-sig")),
            installed=installed,
            application_audit=json.loads(args.audit.read_text(encoding="utf-8-sig")),
            installer_audit=json.loads(args.installer_audit.read_text(encoding="utf-8-sig")),
            runtime=parse_lock(args.runtime_lock.read_text(encoding="utf-8-sig")) if args.runtime_lock else None,
            direct=read_direct_inputs(args.input) if args.input else None,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Dependency verification failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
