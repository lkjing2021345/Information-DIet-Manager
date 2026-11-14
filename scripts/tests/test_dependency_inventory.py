"""Offline inventory contracts using a synthetic repository, never environments/models."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import generate_dependency_inventory as inventory

PROJECT = Path(__file__).resolve().parents[2]
REVISION = "1a" * 20
HASH_A, HASH_B = "a" * 64, "b" * 64
INTEGRITY = "sha512-" + base64.b64encode(b"x" * 64).decode("ascii")


def provenance(lf):
    commit = "2b" * 20
    return {
        "schema_version": 1, "local_path": inventory.VENDOR_PATH,
        "repository": "https://github.com/mozilla/readability", "upstream_path": "Readability.js",
        "upstream_commit": commit,
        "upstream_raw_url": f"https://raw.githubusercontent.com/mozilla/readability/{commit}/Readability.js",
        "reviewed_on": "2026-10-03", "comparison_policy": "lf-or-crlf-only",
        "files": {kind: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                  for kind, data in (("lf", lf), ("crlf", lf.replace(b"\n", b"\r\n")))},
        "release_version": None, "upstream_package_version": "0.0.0", "verified_license": None,
    }


def lock(pins, hashes=(HASH_A,)):
    return "\n".join(f"{name}=={version} " + " ".join(f"--hash=sha256:{value}" for value in hashes)
                     for name, version in pins.items()) + "\n"


def npm_fixture(name):
    manifest = {"name": name, "version": "1.0.0", "dependencies": {"sample": "^1.2.0"}}
    package = {"version": "1.2.3", "resolved": "https://registry.npmjs.org/sample/-/sample-1.2.3.tgz",
               "integrity": INTEGRITY, "optional": True, "os": ["darwin"], "cpu": ["arm64"],
               "libc": ["musl"], "engines": {"node": ">=20"}, "license": "MIT"}
    locked = {"name": name, "version": "1.0.0", "lockfileVersion": 3,
              "packages": {"": copy.deepcopy(manifest), "node_modules/sample": package}}
    return manifest, locked


class DependencyInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="idm-inventory-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for path in inventory.INPUT_PATHS:
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        self.put(inventory.PYTHON_LOCKS["runtime"], lock({"Alpha_Pkg": "1.2.3"}, (HASH_B, HASH_A)))
        self.put(inventory.PYTHON_LOCKS["test"], lock({"alpha-pkg": "1.2.3"}, (HASH_A, HASH_B)) + lock({"pytest": "9.1.1"}))
        self.put(inventory.PYTHON_LOCKS["toolchain"], lock({"tool": "1.0.0"}))
        self.put(inventory.PYTHON_LOCKS["installer"], lock({"pip": "26.2.1"}))
        self.put(inventory.DIRECT_INPUTS["runtime"], "alpha-pkg==1.2.3\n")
        self.put(inventory.DIRECT_INPUTS["test"], "-r requirements-runtime.txt\npytest==9.1.1\n")
        self.put(inventory.DIRECT_INPUTS["toolchain"], "tool==1.0.0\n")
        for directory in inventory.NPM_ROOTS.values():
            manifest, locked = npm_fixture(directory)
            if directory == "chrome-extension":
                manifest["devDependencies"] = manifest.pop("dependencies")
                locked["packages"][""] = copy.deepcopy(manifest)
                locked["packages"]["node_modules/sample"]["dev"] = True
            self.put_json(f"{directory}/package.json", manifest)
            self.put_json(f"{directory}/package-lock.json", locked)
        self.put_json(inventory.EXTENSION_MANIFEST, {"content_scripts": [{"js": ["readability.js", "content.js"]}]})
        self.put(inventory.VENDOR_PATH, "/* Synthetic license declaration */\n/* Based on ancestor 1.7.1 */\nfunction Readability() {}\n")
        self.put_json(inventory.PROVENANCE_PATH, provenance((self.root / inventory.VENDOR_PATH).read_bytes()))

    def put(self, name, text):
        (self.root / name).write_bytes(text.encode("utf-8"))

    def put_json(self, name, value):
        self.put(name, json.dumps(value, ensure_ascii=False))

    def read_json(self, name="frontend/package-lock.json"):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def build(self):
        return inventory.build_inventory(self.root, REVISION)

    def test_inventory_preserves_distinct_scopes_hashes_and_unverified_vendor(self):
        result = self.build()
        self.assertEqual(set(result["python"]), {"runtime", "test", "toolchain", "installer"})
        self.assertEqual(result["python"]["runtime"]["packages"], [
            {"name": "alpha-pkg", "version": "1.2.3", "allowed_sha256": [HASH_A, HASH_B], "direct": True}])
        self.assertEqual(len(result["python"]["test"]["packages"]), 2)
        npm = result["npm"]["frontend"]
        self.assertEqual(npm["direct_declarations"]["dependencies"]["sample"], "^1.2.0")
        self.assertEqual(npm["packages"][0]["version"], "1.2.3")
        self.assertEqual(npm["packages"][0]["os"], ["darwin"])
        self.assertEqual(npm["packages"][0]["cpu"], ["arm64"])
        self.assertEqual(npm["packages"][0]["libc"], ["musl"])
        self.assertTrue(npm["packages"][0]["optional"])
        vendor = result["vendored_files"][0]
        self.assertIsNone(vendor["version"])
        self.assertIsNone(vendor["verified_source"])
        self.assertIsNone(vendor["verified_license"])
        self.assertIn("ancestor 1.7.1", vendor["header_declarations"])
        self.assertEqual(vendor["source_match"]["method"], "byte-identical")
        self.assertEqual(vendor["source_match"]["metadata"], inventory.PROVENANCE_PATH)
        self.assertEqual(vendor["source_match"]["upstream_package_version_declaration"], "0.0.0")
        self.assertIn("not a Git lookup", result["evidence"]["revision_semantics"])
        self.assertIn("Offline byte correspondence", result["evidence"]["source_match_semantics"])

    def test_reads_only_fixed_inputs_and_fingerprints_exact_bytes(self):
        original = Path.read_bytes
        seen = []

        def reader(path):
            seen.append(path.relative_to(self.root).as_posix())
            return original(path)

        with patch.object(Path, "read_bytes", reader), patch("importlib.metadata.distributions", side_effect=AssertionError("No environment scan")):
            result = self.build()
        self.assertEqual(sorted(seen), list(inventory.INPUT_PATHS))
        self.assertEqual(result["inputs"], [{"path": path, "sha256": hashlib.sha256(original(self.root / path)).hexdigest()}
                                           for path in inventory.INPUT_PATHS])

    def test_unknown_includes_are_rejected_without_opening_them(self):
        for include in ("extra.txt", "../outside.txt", "/absolute.txt", "https://example.invalid/secret", "..\\outside.txt"):
            with self.subTest(include=include):
                self.put(inventory.DIRECT_INPUTS["test"], f"-r {include}\n")
                with self.assertRaisesRegex(ValueError, "Requirement include"):
                    self.build()

    def test_include_cycles_conflicts_and_loose_direct_pins_fail(self):
        for content in ("-r requirements-test.txt", "-r requirements-runtime.txt\nalpha-pkg==9.0.0", "pytest>=9.0"):
            with self.subTest(content=content):
                self.put(inventory.DIRECT_INPUTS["test"], content)
                with self.assertRaises(ValueError):
                    self.build()

    def test_direct_inputs_must_be_present_and_exact_in_each_lock(self):
        for scope in inventory.DIRECT_INPUTS:
            path = inventory.DIRECT_INPUTS[scope]
            original = (self.root / path).read_text(encoding="utf-8")
            for extra in ("\nmissing==1.0.0", original.replace("1.2.3", "8.0.0").replace("9.1.1", "8.0.0").replace("1.0.0", "8.0.0")):
                with self.subTest(scope=scope, extra=extra):
                    self.put(path, original + extra if extra.startswith("\n") else extra)
                    with self.assertRaises(ValueError):
                        self.build()
            self.put(path, original)

    def test_runtime_test_subset_requires_versions_and_hashes_to_match(self):
        for runtime, tests in (
            (lock({"alpha-pkg": "1.2.3"}, (HASH_A, HASH_B)), lock({"pytest": "9.1.1"})),
            (lock({"alpha-pkg": "1.2.3", "transitive": "1.0.0"}, (HASH_A, HASH_B)),
             lock({"alpha-pkg": "1.2.3"}, (HASH_A, HASH_B)) + lock({"pytest": "9.1.1", "transitive": "2.0.0"})),
            (lock({"alpha-pkg": "1.2.3"}, (HASH_A, HASH_B)), lock({"alpha-pkg": "1.2.3", "pytest": "9.1.1"}, (HASH_A,))),
        ):
            with self.subTest(tests=tests):
                self.put(inventory.PYTHON_LOCKS["runtime"], runtime)
                self.put(inventory.PYTHON_LOCKS["test"], tests)
                with self.assertRaises(ValueError):
                    self.build()

    def test_installer_is_only_pip_and_does_not_overlap_other_scopes(self):
        self.put(inventory.PYTHON_LOCKS["installer"], lock({"pip": "26.2.1", "tool": "1.0.0"}))
        with self.assertRaises(ValueError):
            self.build()
        self.put(inventory.PYTHON_LOCKS["installer"], lock({"pip": "26.2.1"}))
        self.put(inventory.PYTHON_LOCKS["toolchain"], lock({"tool": "1.0.0", "pip": "26.2.1"}))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.build()

    def test_lock_parser_rejects_unhashed_loose_duplicate_and_unsupported_entries(self):
        for content in ("tool==1.0.0", "tool>=1", lock({"tool": "1.0.0"}) * 2,
                        "tool==1.0.0 --hash=sha256:bad", lock({"tool": "1.0.0"}) + "-r hidden.txt",
                        lock({"tool": "1.0.0"}) + "\\\n"):
            with self.subTest(content=content):
                self.put(inventory.PYTHON_LOCKS["toolchain"], content)
                with self.assertRaises(ValueError):
                    self.build()

    def test_duplicate_json_keys_and_non_standard_numbers_are_not_silently_accepted(self):
        for path, text in (("frontend/package.json", '{"name":"x","name":"y"}'),
                           ("frontend/package-lock.json", '{"packages":{"node_modules/x":{},"node_modules/x":{}}}'),
                           (inventory.EXTENSION_MANIFEST, '{"content_scripts":[],"extra":NaN}')):
            original = (self.root / path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.put(path, text)
                with self.assertRaises(ValueError):
                    self.build()
            self.put(path, original)

    def test_npm_root_name_version_dependency_and_format_drift_fail(self):
        original = self.read_json()
        variants = []
        for key, value in (("name", "wrong"), ("version", "2.0.0"), ("lockfileVersion", 2), ("lockfileVersion", True), ("packages", [])):
            variants.append({**original, key: value})
        for key in ("name", "version", "dependencies"):
            value = copy.deepcopy(original)
            del value["packages"][""][key]
            variants.append(value)
        for value in variants:
            with self.subTest(value=value):
                self.put_json("frontend/package-lock.json", value)
                with self.assertRaises(ValueError):
                    self.build()

    def test_npm_direct_package_must_exist_and_satisfy_the_supported_declaration(self):
        original = self.read_json()
        for version in (None, "1.1.9", "2.0.0", "1.2.3-beta", True):
            value = copy.deepcopy(original)
            if version is None:
                del value["packages"]["node_modules/sample"]
            else:
                value["packages"]["node_modules/sample"]["version"] = version
            with self.subTest(version=version):
                self.put_json("frontend/package-lock.json", value)
                with self.assertRaises(ValueError):
                    self.build()

    def test_exact_and_caret_root_ranges_are_supported_but_new_syntax_is_explicitly_rejected(self):
        for declaration, version, accepted in (("1.2.3", "1.2.3", True), ("1.2.3", "1.2.4", False),
                ("^0.2.0", "0.2.5", True), ("^0.2.0", "0.3.0", False), ("^0.0.2", "0.0.2", True),
                ("^0.0.2", "0.0.3", False), ("~1.2.0", "1.2.3", False), ("*", "1.2.3", False)):
            with self.subTest(declaration=declaration, version=version):
                manifest, locked = npm_fixture("frontend")
                manifest["dependencies"]["sample"] = declaration
                locked["packages"][""] = copy.deepcopy(manifest)
                locked["packages"]["node_modules/sample"]["version"] = version
                self.put_json("frontend/package.json", manifest)
                self.put_json("frontend/package-lock.json", locked)
                if accepted:
                    self.build()
                else:
                    with self.assertRaises(ValueError):
                        self.build()

    def test_npm_package_required_metadata_and_types_are_validated(self):
        original = self.read_json()
        changes = [(key, None) for key in ("version", "resolved", "integrity")]
        changes += [("integrity", "sha512-bad"), ("integrity", "sha1-" + "a" * 40),
                    ("resolved", "https://key@registry.npmjs.org/x"), ("resolved", "file:local-package"),
                    ("dev", 1), ("optional", "true"), ("os", "darwin"), ("cpu", [False]),
                    ("engines", {"node": None}), ("license", []), ("link", True), ("name", "alias")]
        for key, bad in changes:
            with self.subTest(key=key, bad=bad):
                value = copy.deepcopy(original)
                if bad is None:
                    del value["packages"]["node_modules/sample"][key]
                else:
                    value["packages"]["node_modules/sample"][key] = bad
                self.put_json("frontend/package-lock.json", value)
                with self.assertRaises(ValueError):
                    self.build()

    def test_package_installation_paths_preserve_duplicate_names_and_versions(self):
        value = self.read_json()
        value["packages"]["node_modules/parent/node_modules/sample"] = copy.deepcopy(value["packages"]["node_modules/sample"])
        self.put_json("frontend/package-lock.json", value)
        rows = self.build()["npm"]["frontend"]["packages"]
        self.assertEqual([row["path"] for row in rows], ["node_modules/parent/node_modules/sample", "node_modules/sample"])
        self.assertEqual([row["name"] for row in rows], ["sample", "sample"])

    def test_extension_runtime_dependencies_cannot_be_mislabelled_as_test_inventory(self):
        for group in ("dependencies", "optionalDependencies"):
            manifest = self.read_json("chrome-extension/package.json")
            for existing in inventory.ROOT_DEPENDENCIES:
                manifest.pop(existing, None)
            manifest[group] = {"sample": "^1.2.0"}
            locked = self.read_json("chrome-extension/package-lock.json")
            locked["packages"][""] = copy.deepcopy(manifest)
            locked["packages"]["node_modules/sample"]["dev"] = False
            self.put_json("chrome-extension/package.json", manifest)
            self.put_json("chrome-extension/package-lock.json", locked)
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "only devDependencies"):
                self.build()

    def test_malformed_npm_paths_cannot_escape_or_alias_component_identity(self):
        for path in ("../node_modules/sample", "node_modules/sample/../other", "node_modules\\sample", "node_modules/@scope", "random"):
            value = self.read_json()
            row = value["packages"].pop("node_modules/sample")
            value["packages"][path] = row
            self.put_json("frontend/package-lock.json", value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.build()
            manifest, locked = npm_fixture("frontend")
            self.put_json("frontend/package-lock.json", locked)

    def test_vendor_reference_and_header_change_do_not_invent_provenance(self):
        self.put(inventory.VENDOR_PATH, "function Readability() {}\n")
        self.put_json(inventory.PROVENANCE_PATH, provenance((self.root / inventory.VENDOR_PATH).read_bytes()))
        vendor = self.build()["vendored_files"][0]
        self.assertIsNone(vendor["header_declarations"])
        self.assertIsNone(vendor["version"])
        self.put_json(inventory.EXTENSION_MANIFEST, {"content_scripts": [{"js": ["content.js"]}]})
        with self.assertRaises(ValueError):
            self.build()

    def test_lf_and_crlf_source_matches_keep_actual_raw_hash_distinct(self):
        source = self.root / inventory.VENDOR_PATH
        lf = source.read_bytes()
        first = self.build()
        source.write_bytes(lf.replace(b"\n", b"\r\n"))
        second = self.build()
        vendor_lf, vendor_crlf = first["vendored_files"][0], second["vendored_files"][0]
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(len(first["inputs"]), 14)
        self.assertEqual(vendor_lf["source_match"]["method"], "byte-identical")
        self.assertEqual(vendor_crlf["source_match"]["method"], "crlf-only")
        self.assertEqual(vendor_lf["source_match"]["upstream_lf"], vendor_crlf["source_match"]["upstream_lf"])
        self.assertNotEqual(vendor_lf["sha256"], vendor_crlf["sha256"])
        self.assertEqual(vendor_crlf["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertNotEqual(first["inputs"], second["inputs"])

    def test_source_changes_fail_before_replacing_old_output_or_updating_metadata(self):
        source = self.root / inventory.VENDOR_PATH
        lf = source.read_bytes()
        metadata_before = (self.root / inventory.PROVENANCE_PATH).read_bytes()
        output = self.root / "inventory.json"
        inventory.write_inventory(self.build(), output, root=self.root)
        before = output.read_bytes()
        invalid_sources = (lf.replace(b"Readability", b"Other_Name", 1), b"\xef\xbb\xbf" + lf,
                           lf.replace(b"\n", b"\r"), lf.replace(b"\n", b"\r\n", 1), lf.rstrip(b"\n"), b"")
        for raw in invalid_sources:
            with self.subTest(raw=raw):
                source.write_bytes(raw)
                with patch.object(inventory, "ROOT", self.root), self.assertRaises(SystemExit) as error:
                    inventory.main(["--revision", REVISION, "--output", str(output)])
                self.assertEqual(error.exception.code, 1)
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual((self.root / inventory.PROVENANCE_PATH).read_bytes(), metadata_before)
                self.assertEqual(list(self.root.glob(".inventory.json.*.tmp")), [])

    def test_provenance_fields_are_closed_typed_and_cannot_claim_release_or_license(self):
        original = self.read_json(inventory.PROVENANCE_PATH)
        variants = []
        for key in original:
            value = copy.deepcopy(original)
            del value[key]
            variants.append(value)
        for key, bad in (("schema_version", True), ("schema_version", 2), ("local_path", "../private.js"),
                         ("repository", "https://example.invalid/readability"), ("upstream_path", "../Readability.js"),
                         ("upstream_commit", "main"), ("upstream_commit", "G" * 40),
                         ("upstream_raw_url", original["upstream_raw_url"].replace("2b" * 20, "3c" * 20)),
                         ("upstream_raw_url", "file:private.js"), ("reviewed_on", "2026-02-30"),
                         ("reviewed_on", True), ("comparison_policy", "ignore-all-whitespace"),
                         ("release_version", "0.6.0"), ("verified_license", "Apache-2.0"),
                         ("upstream_package_version", None), ("extra", "unsupported")):
            variants.append({**original, key: bad})
        for value in variants:
            with self.subTest(value=value):
                self.put_json(inventory.PROVENANCE_PATH, value)
                with self.assertRaises(ValueError):
                    self.build()

    def test_provenance_fingerprints_validate_both_newline_forms_and_reject_bool_sizes(self):
        original = self.read_json(inventory.PROVENANCE_PATH)
        variants = []
        for kind in ("lf", "crlf"):
            for key, bad in (("sha256", HASH_A), ("sha256", "bad"), ("sha256", False),
                             ("size_bytes", True), ("size_bytes", 0), ("size_bytes", -1),
                             ("size_bytes", 1.5), ("size_bytes", "12"), ("extra", 1)):
                value = copy.deepcopy(original)
                value["files"][kind][key] = bad
                variants.append(value)
            value = copy.deepcopy(original)
            del value["files"][kind]
            variants.append(value)
        variants.append({**original, "files": []})
        for value in variants:
            with self.subTest(value=value):
                self.put_json(inventory.PROVENANCE_PATH, value)
                with self.assertRaises(ValueError):
                    self.build()

    def test_matching_raw_fingerprint_does_not_allow_unapproved_normalization(self):
        source = self.root / inventory.VENDOR_PATH
        lf = source.read_bytes()
        for invalid in (b"\xef\xbb\xbf" + lf, lf.replace(b"\n", b"\r")):
            with self.subTest(invalid=invalid):
                source.write_bytes(invalid)
                self.put_json(inventory.PROVENANCE_PATH, provenance(invalid))
                with self.assertRaises(ValueError):
                    self.build()
        mixed = lf.replace(b"\n", b"\r\n", 1)
        value = provenance(lf)
        value["files"]["crlf"] = {"sha256": hashlib.sha256(mixed).hexdigest(), "size_bytes": len(mixed)}
        source.write_bytes(mixed)
        self.put_json(inventory.PROVENANCE_PATH, value)
        with self.assertRaises(ValueError):
            self.build()

    def test_provenance_input_fingerprint_tracks_review_metadata_and_cannot_be_overwritten(self):
        before = self.build()
        original = self.read_json(inventory.PROVENANCE_PATH)
        updated = {**original, "reviewed_on": "2026-10-02"}
        self.put_json(inventory.PROVENANCE_PATH, updated)
        after = self.build()
        before_hashes = {row["path"]: row["sha256"] for row in before["inputs"]}
        after_hashes = {row["path"]: row["sha256"] for row in after["inputs"]}
        self.assertNotEqual(before_hashes[inventory.PROVENANCE_PATH], after_hashes[inventory.PROVENANCE_PATH])
        self.assertEqual(before_hashes[inventory.VENDOR_PATH], after_hashes[inventory.VENDOR_PATH])
        source_bytes = (self.root / inventory.PROVENANCE_PATH).read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            inventory.write_inventory(after, self.root / inventory.PROVENANCE_PATH, root=self.root)
        self.assertEqual((self.root / inventory.PROVENANCE_PATH).read_bytes(), source_bytes)

    def test_duplicate_or_invalid_provenance_json_preserves_old_inventory(self):
        output = self.root / "inventory.json"
        inventory.write_inventory(self.build(), output, root=self.root)
        before = output.read_bytes()
        for text in ('{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}', '{broken}'):
            with self.subTest(text=text):
                self.put(inventory.PROVENANCE_PATH, text)
                with patch.object(inventory, "ROOT", self.root), self.assertRaises(SystemExit) as error:
                    inventory.main(["--revision", REVISION, "--output", str(output)])
                self.assertEqual(error.exception.code, 1)
                self.assertEqual(output.read_bytes(), before)

    def test_missing_input_fails_instead_of_partial_inventory(self):
        (self.root / inventory.VENDOR_PATH).unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()

    def test_revision_requires_full_hex_and_is_only_normalized_label(self):
        for revision in ("HEAD", "abc123", "x" * 40, "a" * 41, "a" * 39, None):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                inventory.build_inventory(self.root, revision)
        result = inventory.build_inventory(self.root, REVISION.upper())
        self.assertEqual(result["revision_label"], REVISION)

    def test_same_input_generates_identical_bytes_and_raw_byte_changes_are_fingerprinted(self):
        first, second = self.root / "first.json", self.root / "second.json"
        inventory.write_inventory(self.build(), first, root=self.root)
        inventory.write_inventory(self.build(), second, root=self.root)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertNotIn(b"timestamp", first.read_bytes())
        before = self.build()
        source = self.root / inventory.DIRECT_INPUTS["runtime"]
        source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
        after = self.build()
        self.assertEqual(before["python"], after["python"])
        self.assertNotEqual(before["inputs"], after["inputs"])

    def test_output_cannot_replace_any_input_or_generator(self):
        result = self.build()
        for name in (*inventory.INPUT_PATHS, "scripts/generate_dependency_inventory.py", "scripts/verify_python_dependencies.py"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "overwrite"):
                inventory.write_inventory(result, self.root / name, root=self.root)

    def test_invalid_input_cli_preserves_previous_output(self):
        output = self.root / "inventory.json"
        output.write_bytes(b"previous output\n")
        self.put("frontend/package.json", "{bad json}")
        with patch.object(inventory, "ROOT", self.root), self.assertRaises(SystemExit) as error:
            inventory.main(["--revision", REVISION, "--output", str(output)])
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(output.read_bytes(), b"previous output\n")

    def test_replace_failure_preserves_previous_output_and_removes_only_own_temp(self):
        output = self.root / "inventory.json"
        output.write_bytes(b"previous output\n")
        unrelated = self.root / ".inventory.json.unrelated.tmp"
        unrelated.write_bytes(b"do not remove")
        with patch.object(inventory.os, "replace", side_effect=PermissionError("synthetic locked destination")):
            with self.assertRaises(PermissionError):
                inventory.write_inventory(self.build(), output, root=self.root)
        self.assertEqual(output.read_bytes(), b"previous output\n")
        self.assertEqual(list(self.root.glob(".inventory.json.*.tmp")), [unrelated])

    def test_write_or_serialization_failure_does_not_replace_existing_output(self):
        output = self.root / "inventory.json"
        output.write_bytes(b"previous output\n")
        with patch.object(inventory.os, "fsync", side_effect=OSError("synthetic full disk")):
            with self.assertRaises(OSError):
                inventory.write_inventory(self.build(), output, root=self.root)
        self.assertEqual(output.read_bytes(), b"previous output\n")
        self.assertEqual(list(self.root.glob(".inventory.json.*.tmp")), [])
        with self.assertRaises(ValueError):
            inventory.write_inventory({"invalid": float("nan")}, output, root=self.root)
        self.assertEqual(output.read_bytes(), b"previous output\n")

    def test_real_repository_cli_uses_its_own_root_and_produces_all_fixed_inputs(self):
        first, second = self.root / "actual-first.json", self.root / "actual-second.json"
        command = [sys.executable, str(PROJECT / "scripts/generate_dependency_inventory.py"), "--revision", REVISION]
        for output in (first, second):
            result = subprocess.run([*command, "--output", str(output)], cwd=self.root, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        actual = json.loads(first.read_bytes())
        self.assertEqual([row["path"] for row in actual["inputs"]], list(inventory.INPUT_PATHS))
        self.assertEqual(actual["vendored_files"][0]["path"], inventory.VENDOR_PATH)
        self.assertEqual(actual["vendored_files"][0]["source_match"]["upstream_commit"],
                         "d7949dc47dd9ed9ee1d3b34ffdcf3bce28cde435")
        self.assertIn(actual["vendored_files"][0]["source_match"]["method"], ("byte-identical", "crlf-only"))
        self.assertEqual(set(actual["npm"]), {"frontend", "extension-test"})


if __name__ == "__main__":
    unittest.main()
