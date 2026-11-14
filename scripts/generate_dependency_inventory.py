"""Generate an offline, lock-derived inventory, not a distribution SBOM.

Only the fixed repository inputs below are read. No installed environment,
registry, model assets or Git state is queried. --revision is a caller label.
"""
from __future__ import annotations

import argparse
import base64
from datetime import date
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit

if __package__:
    from .verify_python_dependencies import PIN, compare, normalized_name, parse_lock
else:
    from verify_python_dependencies import PIN, compare, normalized_name, parse_lock

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
GENERATOR_VERSION = "1.1.0"
PYTHON_LOCKS = {
    "runtime": "requirements/locks/runtime-windows-py312.txt",
    "test": "requirements/locks/test-windows-py312.txt",
    "toolchain": "requirements/locks/toolchain-windows-py312.txt",
    "installer": "requirements/locks/installer.txt",
}
DIRECT_INPUTS = {
    "runtime": "requirements-runtime.txt",
    "test": "requirements-test.txt",
    "toolchain": "requirements/locks/toolchain.in",
}
NPM_ROOTS = {"frontend": "frontend", "extension-test": "chrome-extension"}
VENDOR_PATH = "chrome-extension/readability.js"
PROVENANCE_PATH = "chrome-extension/readability.provenance.json"
EXTENSION_MANIFEST = "chrome-extension/manifest.json"
INPUT_PATHS = tuple(sorted({
    *PYTHON_LOCKS.values(), *DIRECT_INPUTS.values(), VENDOR_PATH, PROVENANCE_PATH, EXTENSION_MANIFEST,
    *(f"{directory}/{filename}" for directory in NPM_ROOTS.values()
      for filename in ("package.json", "package-lock.json")),
}))
NPM_NAME = r"(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*"
NPM_PATH = re.compile(rf"(?:node_modules/{NPM_NAME}/)*node_modules/({NPM_NAME})")
NPM_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
ROOT_DEPENDENCIES = ("dependencies", "devDependencies", "optionalDependencies")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _json(text: str, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in items:
            _require(name not in result, f"Duplicate JSON key in {label}")
            result[name] = value
        return result

    def invalid_constant(_: str) -> None:
        raise ValueError(f"Non-standard JSON number in {label}")

    return _object(json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant), label)


def _read_inputs(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    result = {}
    for name in INPUT_PATHS:
        path = (root / name).resolve()
        _require(path.is_relative_to(root), f"Input escapes repository: {name}")
        result[name] = path.read_bytes()
    return result


def _direct_inputs(texts: dict[str, str], name: str) -> dict[str, str]:
    """Resolve only fingerprinted direct inputs; never open an include target."""
    result: dict[str, str] = {}
    visiting: set[str] = set()

    def read(path: str) -> None:
        _require(path in DIRECT_INPUTS.values(), "Requirement include is not a fixed inventory input")
        _require(path not in visiting, "Requirement include cycle")
        visiting.add(path)
        for line in texts[path].splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                target = line[3:].strip()
                _require(bool(target) and not any(part in (".", "..") for part in target.split("/"))
                         and "\\" not in target and ":" not in target and not target.startswith("/"),
                         "Requirement includes must be fixed relative paths")
                read(str(PurePosixPath(path).parent / target))
                continue
            match = PIN.fullmatch(line)
            _require(match is not None, "Direct requirements must be exact pins")
            package, version = normalized_name(match[1]), match[2]
            _require(package not in result or result[package] == version, "Conflicting direct requirement")
            result[package] = version
        visiting.remove(path)

    read(name)
    _require(bool(result), "Direct requirements contain no packages")
    return result


def _python_packages(text: str, direct: dict[str, str]) -> list[dict[str, Any]]:
    pins = parse_lock(text)  # Reuse the project's strict pinned/hash grammar.
    compare(direct, {name: pins[name] for name in direct if name in pins}, "Direct input/lock pins")
    hashes: dict[str, set[str]] = {name: set() for name in pins}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "--only-binary :all:":
            continue
        parts = line.removesuffix("\\").strip().split()
        _require(bool(parts), "Unsupported empty lock continuation")
        match = PIN.fullmatch(parts[0])
        if match:
            current, parts = normalized_name(match[1]), parts[1:]
        for part in parts:
            hashes[current].add(part.removeprefix("--hash=sha256:"))
    return [{"name": name, "version": pins[name], "allowed_sha256": sorted(hashes[name]),
             "direct": name in direct} for name in sorted(pins)]


def _version(value: Any) -> tuple[int, int, int]:
    _require(isinstance(value, str) and NPM_VERSION.fullmatch(value) is not None,
             "Only stable x.y.z npm versions are supported")
    return tuple(int(part) for part in value.split("."))


def _satisfies(version: str, declaration: str) -> bool:
    actual = _version(version)
    if not declaration.startswith("^"):
        return actual == _version(declaration)
    lower = _version(declaration[1:])
    major, minor, patch = lower
    upper = (major + 1, 0, 0) if major else ((0, minor + 1, 0) if minor else (0, 0, patch + 1))
    return lower <= actual < upper


def _npm_dependencies(obj: dict[str, Any], label: str) -> dict[str, dict[str, str]]:
    for key in ("peerDependencies", "workspaces", "overrides", "bundledDependencies", "bundleDependencies"):
        _require(key not in obj, f"Unsupported npm root declaration: {key}")
    result = {}
    for key in ROOT_DEPENDENCIES:
        declarations = _object(obj.get(key, {}), f"{label}.{key}")
        for name, declaration in declarations.items():
            _require(re.fullmatch(NPM_NAME, name) is not None and isinstance(declaration, str),
                     "Invalid npm direct dependency")
            # Validate even before looking up the resolved package.
            _version(declaration.removeprefix("^"))
        result[key] = declarations
    return result


def _npm_inventory(manifest: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    _require(type(lock.get("lockfileVersion")) is int and lock["lockfileVersion"] == 3,
             "Only npm lockfileVersion 3 is supported")
    packages = _object(lock.get("packages"), "npm packages")
    root = _object(packages.get(""), "npm lock root")
    _require(isinstance(manifest.get("name"), str) and re.fullmatch(NPM_NAME, manifest["name"]) is not None,
             "npm manifest needs a valid name")
    _version(manifest.get("version"))
    for key in ("name", "version"):
        _require(manifest[key] == lock.get(key) == root.get(key), f"npm root {key} mismatch")
    direct = _npm_dependencies(manifest, "manifest")
    _require(direct == _npm_dependencies(root, "lock root"), "npm manifest/lock root dependencies mismatch")
    records = []
    for path, raw in sorted(packages.items()):
        if path == "":
            continue
        match = NPM_PATH.fullmatch(path)
        _require(match is not None, "Unsupported npm package path")
        row = _object(raw, "npm package")
        name = match[1]
        _require("link" not in row and row.get("name", name) == name, "npm links/aliases are unsupported")
        _version(row.get("version"))
        resolved = row.get("resolved")
        _require(isinstance(resolved, str), "npm package needs resolved URL")
        url = urlsplit(resolved)
        _require(url.scheme == "https" and url.netloc == "registry.npmjs.org" and bool(url.path)
                 and not url.query and not url.fragment, "Only public npm registry tarball URLs are supported")
        integrity = row.get("integrity")
        _require(isinstance(integrity, str) and integrity.startswith("sha512-"), "npm package needs sha512 integrity")
        try:
            digest = base64.b64decode(integrity[7:], validate=True)
        except ValueError as exc:
            raise ValueError("Invalid npm sha512 integrity") from exc
        _require(len(digest) == 64 and base64.b64encode(digest).decode("ascii") == integrity[7:],
                 "Invalid npm sha512 integrity")
        record = {"path": path, "name": name, "version": row["version"], "resolved": resolved,
                  "integrity": integrity}
        for key in ("dev", "optional", "devOptional", "hasInstallScript"):
            value = row.get(key, False)
            _require(type(value) is bool, f"npm {key} must be boolean")
            record[key] = value
        for key in ("os", "cpu", "libc"):
            if key in row:
                value = row[key]
                _require(isinstance(value, list) and all(isinstance(v, str) and v for v in value),
                         f"npm {key} must be an array of strings")
                record[key] = value
        if "engines" in row:
            engines = _object(row["engines"], "npm engines")
            _require(all(isinstance(value, str) and value for value in engines.values()), "Invalid npm engines")
            record["engines"] = engines
        if "license" in row:
            _require(isinstance(row["license"], str), "npm license declaration must be a string")
            record["license_declaration"] = row["license"]
        records.append(record)
    _require(bool(records), "npm lock contains no packages")
    by_path = {row["path"]: row for row in records}
    for declarations in direct.values():
        for name, declaration in declarations.items():
            path = f"node_modules/{name}"
            _require(path in by_path, "npm direct package missing from lock")
            _require(_satisfies(by_path[path]["version"], declaration), "npm direct version does not satisfy manifest")
    return {"root": {"name": manifest["name"], "version": manifest["version"]},
            "direct_declarations": direct, "packages": records}


def _source_match(metadata: dict[str, Any], raw: bytes) -> dict[str, Any]:
    """Check only the bytes against reviewed metadata, never fetch its URL."""
    fields = {"schema_version", "local_path", "repository", "upstream_path", "upstream_commit",
              "upstream_raw_url", "reviewed_on", "comparison_policy", "files", "release_version",
              "upstream_package_version", "verified_license"}
    _require(set(metadata) == fields, "Provenance must have exactly the supported fields")
    _require(type(metadata["schema_version"]) is int and metadata["schema_version"] == 1,
             "Unsupported provenance schema version")
    for field, expected in (("local_path", VENDOR_PATH), ("repository", "https://github.com/mozilla/readability"),
                            ("upstream_path", "Readability.js"), ("comparison_policy", "lf-or-crlf-only")):
        _require(metadata[field] == expected, f"Unsupported provenance {field}")
    commit = metadata["upstream_commit"]
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
             "Provenance commit must be a full lowercase 40-hex identifier")
    _require(metadata["upstream_raw_url"] == f"https://raw.githubusercontent.com/mozilla/readability/{commit}/Readability.js",
             "Provenance raw URL must match the immutable commit and path")
    reviewed_on = metadata["reviewed_on"]
    _require(isinstance(reviewed_on, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", reviewed_on) is not None,
             "Invalid provenance review date")
    date.fromisoformat(reviewed_on)
    _require(metadata["release_version"] is None and metadata["verified_license"] is None,
             "This provenance schema cannot assert a verified release or license")
    _version(metadata["upstream_package_version"])
    files = _object(metadata["files"], "provenance files")
    _require(set(files) == {"lf", "crlf"}, "Provenance requires exactly LF and CRLF fingerprints")
    for fingerprint in files.values():
        fingerprint = _object(fingerprint, "provenance fingerprint")
        _require(set(fingerprint) == {"sha256", "size_bytes"}, "Invalid provenance fingerprint fields")
        _require(isinstance(fingerprint["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint["sha256"]) is not None,
                 "Invalid provenance SHA-256")
        _require(type(fingerprint["size_bytes"]) is int and fingerprint["size_bytes"] > 0,
                 "Provenance byte size must be a positive integer")

    def matches(data: bytes, kind: str) -> bool:
        return len(data) == files[kind]["size_bytes"] and hashlib.sha256(data).hexdigest() == files[kind]["sha256"]

    if matches(raw, "lf"):
        method, lf = "byte-identical", raw
    elif matches(raw, "crlf"):
        method, lf = "crlf-only", raw.replace(b"\r\n", b"\n")
    else:
        raise ValueError("Vendored Readability bytes do not match reviewed provenance")
    _require(not lf.startswith(b"\xef\xbb\xbf") and b"\r" not in lf and b"\n" in lf,
             "Provenance accepts LF bytes without BOM or bare CR")
    crlf = lf.replace(b"\n", b"\r\n")
    _require(matches(lf, "lf") and matches(crlf, "crlf"), "Provenance LF/CRLF fingerprints are inconsistent")
    _require(method != "crlf-only" or raw == crlf, "Provenance permits only complete CRLF-to-LF conversion")
    return {
        "method": method, "metadata": PROVENANCE_PATH, "repository": metadata["repository"],
        "upstream_path": metadata["upstream_path"], "upstream_commit": commit,
        "upstream_raw_url": metadata["upstream_raw_url"], "reviewed_on": reviewed_on,
        "upstream_lf": files["lf"], "upstream_package_version_declaration": metadata["upstream_package_version"],
        "release_version": None,
    }


def build_inventory(root: Path, revision: str) -> dict[str, Any]:
    _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None,
             "revision must be a full 40-character hexadecimal label")
    inputs = _read_inputs(root)
    texts = {name: data.decode("utf-8-sig") for name, data in inputs.items()}
    python = {}
    for scope, lock in PYTHON_LOCKS.items():
        direct = _direct_inputs(texts, DIRECT_INPUTS[scope]) if scope in DIRECT_INPUTS else {}
        python[scope] = {"lock": lock, "direct_input": DIRECT_INPUTS.get(scope),
                         "packages": _python_packages(texts[lock], direct)}
    package_maps = {scope: {row["name"]: row for row in data["packages"]} for scope, data in python.items()}
    for name, row in package_maps["runtime"].items():
        other = package_maps["test"].get(name)
        _require(other is not None and row["version"] == other["version"]
                 and row["allowed_sha256"] == other["allowed_sha256"], "Runtime must be a same-version/hash subset of test")
    _require(set(package_maps["installer"]) == {"pip"}, "Installer lock must contain only pip")
    for scope in ("runtime", "test", "toolchain"):
        _require(not (package_maps[scope].keys() & package_maps["installer"].keys()), "Installer and other locks must not overlap")
    npm = {}
    for scope, directory in NPM_ROOTS.items():
        manifest, lock = f"{directory}/package.json", f"{directory}/package-lock.json"
        npm[scope] = {"manifest": manifest, "lock": lock,
                      **_npm_inventory(_json(texts[manifest], manifest), _json(texts[lock], lock))}
        if scope == "extension-test":
            declarations = npm[scope]["direct_declarations"]
            _require(not declarations["dependencies"] and not declarations["optionalDependencies"],
                     "extension-test scope permits only devDependencies")
    extension = _json(texts[EXTENSION_MANIFEST], EXTENSION_MANIFEST)
    scripts = extension.get("content_scripts")
    _require(isinstance(scripts, list) and all(isinstance(entry, dict) and isinstance(entry.get("js"), list)
             and all(isinstance(js, str) for js in entry["js"]) for entry in scripts), "Invalid extension content_scripts")
    _require(any("readability.js" in entry["js"] for entry in scripts), "Vendored readability.js is not referenced by manifest")
    source_match = _source_match(_json(texts[PROVENANCE_PATH], PROVENANCE_PATH), inputs[VENDOR_PATH])
    header = re.match(r"\s*(/\*.*?\*/\s*/\*.*?\*/)", texts[VENDOR_PATH], flags=re.DOTALL)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "idm-offline-dependency-inventory", "version": GENERATOR_VERSION},
        "revision_label": revision.lower(),
        "evidence": {"kind": "lock-derived inventory", "platform_scope": "Windows CPython 3.12 AMD64; npm all locked platforms",
                     "revision_semantics": "Caller-supplied label; not a Git lookup, signature or clean-checkout attestation",
                     "hash_semantics": "Raw input bytes and allowed package hashes; not proof of the actually installed artifacts",
                     "python_runtime_subset_of_test": True,
                     "npm_scope": "Lock package paths and direct declarations; not a resolved transitive graph or built dist contents",
                     "license_status": "Declarations only; no project license choice or license verification",
                     "source_match_semantics": "Offline byte correspondence to reviewed metadata; not a signature, original download channel or unique source commit proof",
                     "excluded": ["installed environments", "optional model/analysis/training environments", "model weights and tokenizers",
                                  "browser binaries", "operating system and interpreter binaries", "personal data", "distribution file inventory"]},
        "inputs": [{"path": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(inputs.items())],
        "python": python, "npm": npm,
        "vendored_files": [{"path": VENDOR_PATH, "sha256": hashlib.sha256(inputs[VENDOR_PATH]).hexdigest(),
                            "referenced_by": EXTENSION_MANIFEST, "version": None, "verified_source": None,
                            "verified_license": None, "source_match": source_match,
                            "header_declarations": header[1] if header else None}],
    }


def write_inventory(inventory: dict[str, Any], output: Path, *, root: Path) -> None:
    """Replace only after complete validation/serialization and a finished write."""
    output = output.resolve()
    protected = {(root / name).resolve() for name in INPUT_PATHS}
    protected.update((root / "scripts" / name).resolve() for name in
                     ("generate_dependency_inventory.py", "verify_python_dependencies.py"))
    _require(output not in protected, "Output must not overwrite an inventory input or generator")
    data = (json.dumps(inventory, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Explicit full 40-hex caller revision label (not independently verified)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inventory = build_inventory(ROOT, args.revision)
        write_inventory(inventory, args.output, root=ROOT)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Dependency inventory failed: {exc}\n")
    print("Dependency inventory written (lock-derived; not a distribution SBOM).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
