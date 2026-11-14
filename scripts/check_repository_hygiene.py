"""Reject local runtime artifacts in the Git index without opening their contents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


BLOCKED_DIRECTORIES = {
    "node_modules": "installed dependencies",
    ".venv": "local Python environment",
    "venv": "local Python environment",
    "env": "local Python environment",
    "env.bak": "local Python environment",
    "venv.bak": "local Python environment",
    "__pycache__": "Python cache",
    ".pytest_cache": "test cache",
    ".mypy_cache": "type-checker cache",
    ".ruff_cache": "linter cache",
    ".hypothesis": "test cache",
    ".tox": "test environment",
    ".nox": "test environment",
    ".cache": "local cache",
    "build": "generated build output",
    "dist": "generated build output",
    "dist-ssr": "generated build output",
    "output": "local generated output",
    "coverage": "coverage output",
    "htmlcov": "coverage output",
    "playwright-report": "browser-test output",
    "test-results": "test output",
    "logs": "runtime logs",
    "checkpoints": "model checkpoints",
}
BLOCKED_PREFIXES = {
    "src/lsj/src/algorithms/models/": "downloaded or trained models",
    "src/lsj/src/training_data/": "local training data",
    "models/": "downloaded or trained models",
}
MODEL_SUFFIXES = {
    ".safetensors", ".pt", ".pth", ".onnx", ".ckpt", ".model",
    ".pkl", ".pickle", ".h5", ".hdf5",
}
DATABASE_NAME = re.compile(r"\.(?:db|sqlite|sqlite3)(?:-(?:wal|shm|journal)|\.lock)?$")
ENV_TEMPLATES = {".env.example"}
LEGACY_ROOT_NPM = {"package.json", "package-lock.json", "npm-shrinkwrap.json"}


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str


def violation_reason(path: str) -> str | None:
    """Classify an indexed path, not its payload; comparisons are Windows-safe."""
    normalized = path.replace("\\", "/").casefold()
    parts = normalized.split("/")
    name = parts[-1]
    if normalized in LEGACY_ROOT_NPM:
        return "obsolete root npm entry point; use frontend/ or chrome-extension/"
    for part in parts[:-1]:
        if part in BLOCKED_DIRECTORIES:
            return BLOCKED_DIRECTORIES[part]
    for prefix, reason in BLOCKED_PREFIXES.items():
        if normalized.startswith(prefix):
            return reason
    if DATABASE_NAME.search(name):
        return "runtime database or its journal/lock file"
    if name in {"credentials.json", "credentials.json.new"} or name.endswith(
        (".credentials.json", ".credentials.json.new")
    ):
        return "local credentials"
    if name not in ENV_TEMPLATES and (name.endswith(".env") or ".env." in name):
        return "environment configuration; only the exact .env.example template is allowed"
    suffix = Path(name).suffix
    if suffix in MODEL_SUFFIXES or name == "pytorch_model.bin":
        return "model artifact"
    if suffix in {".pyc", ".pyo", ".log", ".tmp", ".temp"}:
        return "generated cache, log or temporary file"
    if name == ".coverage" or name.startswith(".coverage."):
        return "coverage output"
    return None


def indexed_paths(repository: Path) -> list[str]:
    """Use the index so ignored artifacts forcibly staged with -f are still checked."""
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repository}", "-C", str(repository),
             "ls-files", "--cached", "-z"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot read the Git index: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Cannot read the Git index: {detail}")
    # Git's -z protocol avoids quotePath encoding and preserves spaces/non-ASCII names.
    return [entry.decode("utf-8", errors="surrogateescape")
            for entry in result.stdout.split(b"\0") if entry]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        paths = indexed_paths(args.repo.resolve())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    violations = [Violation(path, reason) for path in sorted(paths)
                  if (reason := violation_reason(path)) is not None]
    if violations:
        print(f"Repository hygiene failed: {len(violations)} prohibited indexed paths.", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.path!r}: {violation.reason}", file=sys.stderr)
        print("Remove these paths from the index while preserving needed local files. "
              "This check never deletes files.", file=sys.stderr)
        return 1
    print(f"Repository hygiene passed: {len(paths)} indexed paths checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
