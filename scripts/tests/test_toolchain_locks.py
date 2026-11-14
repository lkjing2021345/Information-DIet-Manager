"""Maintainer tools must be reproducible without weakening application checks."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import lock_python_dependencies as generator
from scripts import verify_python_dependencies as checker


def locked(pins):
    return "\n".join(f"{name}=={version} --hash=sha256:{'a' * 64}" for name, version in pins.items())


class ToolchainLockTests(unittest.TestCase):
    def test_toolchain_generation_only_excludes_the_separately_constrained_installer(self):
        commands = generator.compile_commands(upgrade=False, scope="toolchain")
        self.assertEqual(len(commands), 1)
        command = commands[0]
        for required in ("--generate-hashes", "--no-allow-unsafe", "--unsafe-package=pip",
                         "--constraint=requirements/locks/installer.txt", "--pip-args=--only-binary=:all:",
                         "--index-url=https://pypi.org/simple"):
            self.assertIn(required, command)
        self.assertNotIn("--allow-unsafe", command)
        self.assertEqual([arg for arg in command if arg.startswith("--unsafe-package")], ["--unsafe-package=pip"])
        self.assertNotIn("--upgrade", command)
        self.assertIn("--upgrade", generator.compile_commands(upgrade=True, scope="toolchain")[0])

    def test_application_default_remains_separate_and_test_is_runtime_constrained(self):
        commands = generator.compile_commands(upgrade=False)
        self.assertEqual(len(commands), 2)
        self.assertIn("requirements-runtime.txt", commands[0])
        self.assertIn("requirements-test.txt", commands[1])
        self.assertIn("--constraint=requirements/locks/runtime-windows-py312.txt", commands[1])
        self.assertEqual(len(generator.compile_commands(upgrade=False, scope="all")), 3)

    def test_toolchain_header_reproduces_the_explicit_generation_scope(self):
        env = generator.generator_environment(Path("synthetic-root"), upgrade=True, scope="toolchain")
        self.assertEqual(env["CUSTOM_COMPILE_COMMAND"], "python scripts/lock_python_dependencies.py --scope toolchain --upgrade")

    def test_generator_refuses_missing_extra_and_version_drift_in_transitives(self):
        tools = {"pip-tools": "7.6.1", "pip-audit": "2.10.1", "setuptools": "84.0.0"}
        installer = {"pip": "26.2.1"}
        expected = {**tools, **installer}
        with tempfile.TemporaryDirectory(prefix="idm-toolchain-") as directory:
            root = Path(directory)
            locks = root / "requirements" / "locks"
            locks.mkdir(parents=True)
            (locks / "toolchain-windows-py312.txt").write_text(locked(tools), encoding="utf-8")
            (locks / "installer.txt").write_text(locked(installer), encoding="utf-8")
            with (patch.object(generator, "ROOT", root), patch.object(generator.sys, "platform", "win32"),
                  patch.object(generator.sys, "version_info", (3, 12)),
                  patch.object(generator.sysconfig, "get_platform", return_value="win-amd64"),
                  patch.object(generator.importlib.metadata, "version", side_effect=expected.__getitem__)):
                for installed, passes in ((expected, True), ({**expected, "unexpected": "1.0"}, False),
                                          ({key: value for key, value in expected.items() if key != "setuptools"}, False),
                                          ({**expected, "setuptools": "83.0.0"}, False)):
                    distributions = [SimpleNamespace(metadata={"Name": name}, version=version) for name, version in installed.items()]
                    with self.subTest(installed=installed), patch.object(generator.importlib.metadata, "distributions", return_value=distributions):
                        if passes:
                            generator.validate_generator()
                        else:
                            with self.assertRaisesRegex(checker.DependencyVerificationError, "Generator environment"):
                                generator.validate_generator()

    def test_tool_audit_cannot_omit_a_build_tool_or_duplicate_the_installer(self):
        tools = {"pip-tools": "7.6.1", "pip-audit": "2.10.1", "setuptools": "84.0.0"}
        installer = {"pip": "26.2.1"}
        audit = lambda pins: {"dependencies": [{"name": name, "version": version, "vulns": []} for name, version in pins.items()]}
        args = dict(application=tools, installer=installer,
                    installed=[{"name": name, "version": version} for name, version in {**tools, **installer}.items()],
                    application_audit=audit(tools), installer_audit=audit(installer))
        self.assertEqual(checker.verify(**args)["application_packages"], 3)
        with self.assertRaisesRegex(checker.DependencyVerificationError, "Application audit coverage.*setuptools"):
            checker.verify(**{**args, "application_audit": audit({name: version for name, version in tools.items() if name != "setuptools"})})
        with self.assertRaisesRegex(checker.DependencyVerificationError, "must not overlap"):
            checker.verify(**{**args, "application": {**tools, **installer}})


if __name__ == "__main__":
    unittest.main()
