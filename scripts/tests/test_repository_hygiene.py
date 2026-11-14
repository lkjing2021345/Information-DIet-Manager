"""Exercise the hygiene CLI against real disposable Git indexes and synthetic files."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT / "scripts" / "check_repository_hygiene.py"


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="idm-hygiene-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.git("init", "--quiet")

    def git(self, *args):
        return subprocess.run(
            ["git", "-c", f"safe.directory={self.repo}", "-C", str(self.repo), *args],
            capture_output=True, check=True, timeout=30,
        )

    def write(self, relative, content="synthetic fixture\n"):
        target = self.repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def stage(self, paths):
        for path in paths:
            self.write(path)
        self.git("add", "--force", "--", *paths)

    def check(self, *, repository=None, env=None):
        environment = dict(os.environ if env is None else env)
        environment["PYTHONUTF8"] = "1"
        return subprocess.run(
            [sys.executable, str(CHECKER), "--repo", str(repository or self.repo)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=environment, timeout=30,
        )

    def assert_rejected(self, paths):
        self.stage(paths)
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        for path in paths:
            self.assertIn(repr(path), result.stderr)

    def test_source_templates_and_synthetic_text_fixtures_pass(self):
        self.stage([
            "frontend/.env.example", "frontend/package.json", "frontend/package-lock.json",
            "chrome-extension/package.json", "chrome-extension/package-lock.json",
            "src/hyh/models.py", "src/hyh/schema.sql", "tests/fixtures/synthetic.csv",
            "tests/fixtures/synthetic.json", "docs/数据说明 文档.md",
        ])
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("10 indexed paths checked", result.stdout)

    def test_forced_add_bypasses_ignore_but_not_the_gate(self):
        self.write(".gitignore", (PROJECT / ".gitignore").read_text(encoding="utf-8"))
        paths = ["node_modules/lib/index.js", "frontend/node_modules/lib/index.js",
                 "src/hyh/data/private.sqlite3", ".pytest_cache/state"]
        for path in paths:
            self.write(path)
            self.git("check-ignore", "--quiet", "--", path)
        self.git("add", "--force", "--", *paths)
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stderr)
        for path in paths:
            self.assertIn(repr(path), result.stderr)

    def test_database_files_and_companions_are_rejected(self):
        self.assert_rejected([
            "data/local.db", "data/local.sqlite", "data/local.sqlite3",
            "data/local.sqlite3-wal", "data/local.sqlite3-shm", "data/local.db-journal",
            "data/local.sqlite3.lock", "资料 目录/本地 记录.sqlite3",
        ])

    def test_environment_names_require_the_exact_template_name(self):
        self.assert_rejected([
            ".env", "frontend/.env.production", "frontend/.env.local",
            "frontend/.env.production.example", "foo.env.dataexample", "staging.env",
            "idm.credentials.json", "idm.credentials.json.new", "credentials.json",
        ])

    def test_caches_environments_and_generated_outputs_are_rejected(self):
        self.assert_rejected([
            ".venv/pyvenv.cfg", "venv/pyvenv.cfg", "env/pyvenv.cfg",
            "src/__pycache__/module.pyc", ".mypy_cache/state", ".ruff_cache/state",
            ".hypothesis/example", ".tox/py/config", ".nox/check/config", ".cache/tool/state",
            "frontend/dist/index.html", "frontend/dist-ssr/server.js", "build/app.js",
            "output/playwright/run/result.json", "coverage/result.json", "htmlcov/index.html",
            "playwright-report/index.html", "test-results/result.json", "logs/app.log",
            ".coverage", ".coverage.worker", "temporary.tmp",
        ])

    def test_downloaded_models_and_local_training_data_are_rejected(self):
        self.assert_rejected([
            "models/config.json", "src/lsj/src/algorithms/models/config.json",
            "src/lsj/src/training_data/records.json", "checkpoints/config.json",
            "weights.safetensors", "weights.pt", "weights.pth", "weights.onnx",
            "weights.ckpt", "classifier.pkl", "classifier.pickle", "weights.model",
            "weights.h5", "weights.hdf5", "pytorch_model.bin",
        ])

    def test_obsolete_root_npm_entry_points_are_rejected(self):
        self.assert_rejected(["package.json", "package-lock.json", "npm-shrinkwrap.json"])

    def test_windows_case_variants_cannot_bypass_the_gate(self):
        self.assert_rejected(["Node_Modules/lib/source.js", "private.SQLITE3", ".ENV.PRODUCTION"])

    def test_untracked_local_artifacts_are_not_read_or_blocked(self):
        self.stage(["README.md"])
        self.write("local.sqlite3", "not a real database")
        self.write("node_modules/local/index.js")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 indexed paths checked", result.stdout)

    def test_cached_removal_passes_and_preserves_the_local_file(self):
        self.stage(["src/hyh/data/private.sqlite3"])
        self.assertEqual(self.check().returncode, 1)
        self.git("rm", "--cached", "--", "src/hyh/data/private.sqlite3")
        self.assertTrue((self.repo / "src/hyh/data/private.sqlite3").is_file())
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_non_repository_fails_instead_of_reporting_clean(self):
        with tempfile.TemporaryDirectory(prefix="idm-not-a-repo-") as directory:
            result = self.check(repository=Path(directory))
        self.assertEqual(result.returncode, 2)
        self.assertIn("Cannot read the Git index", result.stderr)
        self.assertNotIn("passed", result.stdout)

    def test_missing_git_fails_instead_of_reporting_clean(self):
        empty_path = self.repo / "empty-command-path"
        empty_path.mkdir()
        environment = os.environ.copy()
        environment["PATH"] = str(empty_path)
        result = self.check(env=environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Cannot read the Git index", result.stderr)
        self.assertNotIn("passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
