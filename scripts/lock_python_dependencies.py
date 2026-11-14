"""Regenerate the supported Windows CPython 3.12 AMD64 dependency locks.

Run in an isolated maintainer environment installed from the installer and
toolchain hash locks. pip==26.2.1 and pip-tools==7.6.1 remain the generator pair.
Application users install the checked-in locks and do not need pip-tools.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

if __package__:
    from .verify_python_dependencies import compare, package_set, parse_lock
else:
    from verify_python_dependencies import compare, package_set, parse_lock

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSIONS = {"pip": "26.2.1", "pip-tools": "7.6.1"}


def validate_generator() -> None:
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
            or sys.platform != "win32" or sysconfig.get_platform() != "win-amd64"):
        raise RuntimeError("Generate these locks on Windows CPython 3.12 AMD64")
    for name, version in GENERATOR_VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Install the installer and toolchain hash locks; missing {name}=={version}") from exc
        if actual != version:
            raise RuntimeError(f"Expected {name}=={version}; found {actual}")
    toolchain = parse_lock((ROOT / "requirements/locks/toolchain-windows-py312.txt").read_text(encoding="utf-8"))
    installer = parse_lock((ROOT / "requirements/locks/installer.txt").read_text(encoding="utf-8"))
    if toolchain.keys() & installer.keys():
        raise RuntimeError("Toolchain and installer locks must not overlap")
    installed = package_set({"name": dist.metadata["Name"], "version": dist.version}
                            for dist in importlib.metadata.distributions())
    compare({**toolchain, **installer}, installed, "Generator environment")


def generator_environment(root: Path, *, upgrade: bool, scope: str = "application") -> dict[str, str]:
    # Reject accidental extra indexes, local paths and credentials inherited from
    # a developer's pip configuration. The committed inputs use official PyPI.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PIP_") and key != "CUSTOM_COMPILE_COMMAND"}
    env.update({
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_CACHE_DIR": str(root / "output" / "python-lock-cache" / "pip"),
        "PIP_TOOLS_CACHE_DIR": str(root / "output" / "python-lock-cache" / "compile"),
        "PYTHONUTF8": "1",
        "CUSTOM_COMPILE_COMMAND": "python scripts/lock_python_dependencies.py"
        + (f" --scope {scope}" if scope != "application" else "") + (" --upgrade" if upgrade else ""),
    })
    return env


def compile_commands(*, upgrade: bool, scope: str = "application") -> list[list[str]]:
    common = [
        "--generate-hashes", "--allow-unsafe", "--strip-extras", "--resolver=backtracking",
        "--no-config", "--newline=lf", "--index-url=https://pypi.org/simple",
        "--no-emit-index-url", "--no-emit-trusted-host", "--pip-args=--only-binary=:all:", "--quiet",
    ]
    if upgrade:
        common.append("--upgrade")
    commands = []
    for application_scope in (() if scope == "toolchain" else ("runtime", "test")):
        command = [sys.executable, "-m", "piptools", "compile", f"requirements-{application_scope}.txt",
                   f"--output-file=requirements/locks/{application_scope}-windows-py312.txt", *common]
        if application_scope == "test":
            command.append("--constraint=requirements/locks/runtime-windows-py312.txt")
        commands.append(command)
    if scope in {"toolchain", "all"}:
        # pip is required by pip-tools but is pinned/hashed/audited separately
        # in installer.txt. Override the unsafe set so setuptools is INCLUDED;
        # no other transitive package is excluded from the toolchain lock.
        tool_options = [option for option in common if option != "--allow-unsafe"]
        commands.append([sys.executable, "-m", "piptools", "compile", "requirements/locks/toolchain.in",
                         "--output-file=requirements/locks/toolchain-windows-py312.txt", *tool_options,
                         "--no-allow-unsafe", "--unsafe-package=pip", "--constraint=requirements/locks/installer.txt"])
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upgrade", action="store_true", help="Explicitly re-resolve transitive dependencies")
    parser.add_argument("--scope", choices=("application", "toolchain", "all"), default="application")
    args = parser.parse_args()
    try:
        validate_generator()
        for command in compile_commands(upgrade=args.upgrade, scope=args.scope):
            subprocess.run(command, cwd=ROOT, env=generator_environment(ROOT, upgrade=args.upgrade, scope=args.scope), check=True)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Dependency lock generation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
