"""Real CLI argument handling with owned paths and synthetic capability keys."""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/rotate_local_keys.py"
ORIGINAL = {"version": 1, "admin_token": "a" * 43, "collector_token": "b" * 43}


class RotateLocalKeysCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="idm-rotation-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = self.root / "synthetic.sqlite3"
        self.credentials = self.database.with_suffix(".credentials.json")
        self.env = {**os.environ, "IDM_DB_PATH": str(self.database), "PYTHONUTF8": "1"}
        for name in ("IDM_ADMIN_TOKEN", "IDM_COLLECTOR_TOKEN", "IDM_FRONTEND_ORIGINS"):
            self.env.pop(name, None)

    def seed_credentials(self):
        content = (json.dumps(ORIGINAL, indent=2) + "\n").encode("utf-8")
        self.credentials.write_bytes(content)
        return content

    def command(self, arguments=(), *, env=None, python_options=()):
        result = subprocess.run([sys.executable, *python_options, str(SCRIPT), *arguments], cwd=self.root,
                                env=env or self.env, capture_output=True, text=True,
                                encoding="utf-8", timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.assertNotIn(ORIGINAL["admin_token"], result.stdout + result.stderr)
        self.assertNotIn(ORIGINAL["collector_token"], result.stdout + result.stderr)
        return result

    def test_help_preserves_existing_credentials_and_has_no_lock_side_effect(self):
        for argument in ("--help", "-h"):
            with self.subTest(argument=argument):
                before = self.seed_credentials()
                result = self.command([argument])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertNotIn("Keys rotated:", result.stdout)
                self.assertEqual(self.credentials.read_bytes(), before)
                self.assertFalse(self.database.with_suffix(".lock").exists())
                self.assertFalse(self.database.exists())

    def test_unknown_option_and_positional_argument_preserve_existing_credentials(self):
        for argument in ("--unrecognized-option", "unexpected-positional-value"):
            with self.subTest(argument=argument):
                before = self.seed_credentials()
                result = self.command([argument])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("unrecognized arguments", result.stderr)
                self.assertNotIn("Keys rotated:", result.stdout)
                self.assertEqual(self.credentials.read_bytes(), before)
                self.assertFalse(self.database.with_suffix(".lock").exists())

    def test_help_and_invalid_arguments_do_not_create_a_new_configured_directory(self):
        for number, (argument, expected) in enumerate((("--help", 0), ("--unrecognized-option", 2))):
            with self.subTest(argument=argument):
                parent = self.root / f"not-created-{number}"
                env = {**self.env, "IDM_DB_PATH": str(parent / "synthetic.sqlite3")}
                result = self.command([argument], env=env)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertFalse(parent.exists())

    def test_argument_parsing_precedes_invalid_database_and_environment_key_configuration(self):
        for database in ("", "   ", str(self.root)):
            for argument, expected in (("--help", 0), ("--unrecognized-option", 2)):
                with self.subTest(database=database, argument=argument):
                    env = {**self.env, "IDM_DB_PATH": database, "IDM_ADMIN_TOKEN": "invalid-synthetic-key"}
                    before = set(self.root.iterdir())
                    result = self.command([argument], env=env)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertIn("usage:", result.stdout + result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(set(self.root.iterdir()), before)

    def test_help_and_unknown_arguments_work_without_site_packages(self):
        for argument, expected in (("--help", 0), ("--unrecognized-option", 2)):
            with self.subTest(argument=argument):
                env = {**self.env, "IDM_DB_PATH": ""}
                result = self.command([argument], env=env, python_options=["-S"])
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertIn("usage:", result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(self.credentials.exists())
                self.assertFalse(self.database.with_suffix(".lock").exists())

    def test_normal_rotation_changes_both_keys_and_preserves_database(self):
        self.seed_credentials()
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("CREATE TABLE synthetic(value TEXT)")
            connection.execute("INSERT INTO synthetic VALUES ('Owned synthetic record')")
            connection.commit()
        before = self.database.read_bytes()
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Keys rotated:", result.stdout)
        current = json.loads(self.credentials.read_bytes())
        self.assertEqual(current["version"], 1)
        self.assertNotEqual(current["admin_token"], current["collector_token"])
        for name in ("admin_token", "collector_token"):
            self.assertNotEqual(current[name], ORIGINAL[name])
            self.assertNotIn(current[name], result.stdout + result.stderr)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertFalse(self.credentials.with_name(self.credentials.name + ".new").exists())

    def test_environment_keys_and_running_process_lock_still_block_rotation(self):
        from src.hyh.security import process_ownership

        before = self.seed_credentials()
        result = self.command(env={**self.env, "IDM_ADMIN_TOKEN": ORIGINAL["admin_token"],
                                   "IDM_COLLECTOR_TOKEN": ORIGINAL["collector_token"]})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Environment keys are configured", result.stderr)
        self.assertEqual(self.credentials.read_bytes(), before)
        self.assertFalse(self.database.with_suffix(".lock").exists())
        with process_ownership(self.database):
            result = self.command()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another IDM process", result.stderr)
            self.assertEqual(self.credentials.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
