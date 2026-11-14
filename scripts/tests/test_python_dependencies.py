"""Regression tests for complete audit coverage and immutable dependency inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import lock_python_dependencies as generator
from scripts import verify_python_dependencies as checker

PROJECT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT / "scripts" / "verify_python_dependencies.py"
HASH = "a" * 64


def locked(pins):
    return "\n".join(f"{name}=={version} --hash=sha256:{HASH}" for name, version in pins.items())


def audit(pins):
    return {"dependencies": [{"name": name, "version": version, "vulns": []} for name, version in pins.items()]}


class DependencyLockTests(unittest.TestCase):
    def setUp(self):
        self.application = {"pytest": "9.1.1", "packaging": "26.3"}
        self.installer = {"pip": "26.2.1"}
        self.args = dict(
            application=self.application, installer=self.installer,
            installed=[{"name": name, "version": version} for name, version in {**self.application, **self.installer}.items()],
            application_audit=audit(self.application), installer_audit=audit(self.installer),
        )

    def test_complete_matching_sets_pass(self):
        self.assertEqual(checker.verify(**self.args)["status"], "verified")

    def test_missing_transitive_packaging_is_rejected_even_if_other_packages_have_no_vulnerabilities(self):
        self.args["application_audit"] = audit({"pytest": "9.1.1"})
        with self.assertRaisesRegex(checker.DependencyVerificationError, "Application audit coverage.*packaging"):
            checker.verify(**self.args)

    def test_same_count_wrong_names_or_versions_cannot_pass(self):
        for substitute in ({"pytest": "9.1.1", "unrelated": "26.3"}, {"pytest": "9.1.1", "packaging": "26.2"}):
            with self.subTest(substitute=substitute), self.assertRaises(checker.DependencyVerificationError):
                checker.verify(**{**self.args, "application_audit": audit(substitute)})

    def test_skips_malformed_results_duplicates_and_vulnerabilities_are_rejected(self):
        for row in (
            {"name": "packaging", "skip_reason": "synthetic unavailable index"},
            {"name": "packaging", "version": "26.3", "vulns": [], "skip_reason": "skipped"},
            {"name": "packaging", "version": "26.3", "vulns": None},
            {"name": "packaging", "version": "26.3", "vulns": [{"id": "SYNTHETIC-1"}]},
        ):
            with self.subTest(row=row), self.assertRaises(checker.DependencyVerificationError):
                checker.audit_set({"dependencies": [row]})
        with self.assertRaisesRegex(checker.DependencyVerificationError, "Duplicate"):
            checker.audit_set({"dependencies": audit(self.application)["dependencies"] * 2})

    def test_dirty_environment_and_wrong_installer_are_rejected(self):
        for installed in (
            self.args["installed"] + [{"name": "setuptools", "version": "84.0.0"}],
            [row for row in self.args["installed"] if row["name"] != "pip"],
            [{**row, "version": "24.2"} if row["name"] == "pip" else row for row in self.args["installed"]],
        ):
            with self.subTest(installed=installed), self.assertRaisesRegex(checker.DependencyVerificationError, "Installed environment"):
                checker.verify(**{**self.args, "installed": installed})

    def test_installer_needs_its_own_complete_clean_audit(self):
        with self.assertRaisesRegex(checker.DependencyVerificationError, "Installer audit coverage"):
            checker.verify(**{**self.args, "installer_audit": {"dependencies": []}})

    def test_name_aliases_normalize_but_duplicate_aliases_do_not_hide(self):
        self.assertEqual(checker.package_set([{"name": "Typing_Extensions", "version": "4.16.0"}]), {"typing-extensions": "4.16.0"})
        with self.assertRaises(checker.DependencyVerificationError):
            checker.package_set([{"name": name, "version": "4.16.0"} for name in ("Typing_Extensions", "typing.extensions")])

    def test_changed_direct_pins_and_inconsistent_runtime_pins_fail(self):
        for extra in ({"direct": {"pytest": "9.2.0"}}, {"runtime": {"packaging": "26.2"}}, {"direct": {"new-package": "1.0"}}):
            with self.subTest(extra=extra), self.assertRaises(checker.DependencyVerificationError):
                checker.verify(**self.args, **extra)

    def test_lock_parser_requires_pins_and_hashes_for_every_package(self):
        self.assertEqual(checker.parse_lock(locked(self.application)), self.application)
        for text in (
            "", "pytest>=9", "pytest==9.1.1", "-r unlocked.txt", "--index-url https://example.invalid",
            f"pytest==9.1.1 --hash=md5:{HASH}", f"pytest==9.1.1 --hash=sha256:abc",
            "pytest==9.1.1\n" + locked({"packaging": "26.3"}),
            locked({"typing-extensions": "4.16.0"}) + "\n" + locked({"Typing_Extensions": "4.16.0"}),
        ):
            with self.subTest(text=text), self.assertRaises(checker.DependencyVerificationError):
                checker.parse_lock(text)

    def test_multiline_generated_hashes_and_comments_are_supported(self):
        content = f"# generated\n--only-binary :all:\npytest==9.1.1 \\\n    --hash=sha256:{HASH} \\\n    --hash=sha256:{'b' * 64}\n    # via -r requirements-test.txt\n"
        self.assertEqual(checker.parse_lock(content), {"pytest": "9.1.1"})

    def test_input_includes_are_local_pinned_and_acyclic(self):
        with tempfile.TemporaryDirectory(prefix="idm-inputs-") as directory:
            root = Path(directory)
            runtime = root / "runtime.txt"
            tests = root / "test.txt"
            runtime.write_text("packaging==26.3\n", encoding="utf-8")
            tests.write_text("-r runtime.txt\npytest==9.1.1\n", encoding="utf-8")
            self.assertEqual(checker.read_direct_inputs(tests), self.application)
            for source in ("-r test.txt", "-r ../outside.txt", "-r https://example.invalid/x", "pytest>=9", "-r runtime.txt\npackaging==26.2"):
                tests.write_text(source, encoding="utf-8")
                with self.subTest(source=source), self.assertRaises(checker.DependencyVerificationError):
                    checker.read_direct_inputs(tests)

    def test_cli_checks_the_target_snapshot_and_rejects_the_omitted_dependency(self):
        with tempfile.TemporaryDirectory(prefix="idm-lock-cli-") as directory:
            root = Path(directory)
            (root / "app.txt").write_text(locked(self.application), encoding="utf-8")
            (root / "installer.txt").write_text(locked(self.installer), encoding="utf-8")
            for name, value in (("installed", self.args["installed"]), ("audit", self.args["application_audit"]), ("installer-audit", self.args["installer_audit"])):
                (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            command = [sys.executable, str(CHECKER), "--lock", str(root / "app.txt"), "--installer-lock", str(root / "installer.txt"),
                       "--installed-json", str(root / "installed.json"), "--audit-json", str(root / "audit.json"),
                       "--installer-audit-json", str(root / "installer-audit.json")]
            passed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            (root / "audit.json").write_text(json.dumps(audit({"pytest": "9.1.1"})), encoding="utf-8")
            failed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(failed.returncode, 1)
            self.assertIn("packaging", failed.stderr)

    def test_generator_does_not_inherit_private_indexes_or_requirement_injection(self):
        with patch.dict(os.environ, {"PIP_EXTRA_INDEX_URL": "https://example.invalid/private", "PIP_CONSTRAINT": "private.txt", "PIP_FIND_LINKS": "private-wheels"}):
            environment = generator.generator_environment(Path("synthetic-root"), upgrade=False)
        self.assertNotIn("PIP_EXTRA_INDEX_URL", environment)
        self.assertNotIn("PIP_CONSTRAINT", environment)
        self.assertNotIn("PIP_FIND_LINKS", environment)
        self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(environment["CUSTOM_COMPILE_COMMAND"], "python scripts/lock_python_dependencies.py")

    def test_generator_rejects_unvalidated_platform_before_resolving(self):
        with patch.object(generator.sys, "platform", "linux"), self.assertRaisesRegex(RuntimeError, "Windows"):
            generator.validate_generator()


if __name__ == "__main__":
    unittest.main()
