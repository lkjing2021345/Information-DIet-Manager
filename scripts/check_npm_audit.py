"""Apply an explicit, expiring policy to an npm audit JSON report.

The quality workflow must keep reporting known dependency advisories without
turning every run into an uninformative red build. This checker only accepts
advisories listed in the repository policy, and rejects unknown or expired
entries. It never changes the dependency tree.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any, Iterable


class AuditPolicyError(ValueError):
    """Raised when an audit report is not covered by the explicit policy."""


def _advisories(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("url"), str) or isinstance(value.get("source"), int):
            yield value
        for nested in value.values():
            yield from _advisories(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _advisories(nested)


def _findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if "vulnerabilities" not in report:
        raise AuditPolicyError("npm audit report is missing the vulnerabilities object")
    vulnerabilities = report["vulnerabilities"]
    if not isinstance(vulnerabilities, dict):
        raise AuditPolicyError("npm audit report has an invalid vulnerabilities object")
    def inherited_advisories(package: str, visiting: set[str] | None = None) -> list[dict[str, Any]]:
        visiting = set() if visiting is None else visiting
        if package in visiting:
            return []
        visiting.add(package)
        entry = vulnerabilities.get(package)
        if not isinstance(entry, dict):
            return []
        result = list(_advisories(entry.get("via", [])))
        for reference in entry.get("via", []):
            if isinstance(reference, str):
                result.extend(inherited_advisories(reference, visiting))
        return result

    for package, entry in vulnerabilities.items():
        if not isinstance(entry, dict):
            raise AuditPolicyError(f"npm audit entry for {package!r} is invalid")
        package_advisories = inherited_advisories(package)
        if not package_advisories:
            package_advisories = [{"name": package, "severity": entry.get("severity")}]
        for advisory in package_advisories:
            finding = dict(advisory)
            finding.setdefault("name", package)
            finding.setdefault("severity", entry.get("severity"))
            findings.append(finding)
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for finding in findings:
        key = (
            str(finding.get("url", "")),
            str(finding.get("source", "")),
            str(finding.get("name", "")),
        )
        unique[key] = finding
    return list(unique.values())


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditPolicyError(f"Cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditPolicyError(f"JSON root in {path} must be an object")
    return value


def check(report: dict[str, Any], policy: dict[str, Any], *, today: date | None = None) -> list[str]:
    """Return human-readable accepted findings or raise for a policy breach."""
    today = today or date.today()
    if type(policy.get("schema_version")) is not int or policy["schema_version"] != 1:
        raise AuditPolicyError("Policy schema_version must be 1")
    allowed = policy.get("allowed", [])
    if not isinstance(allowed, list):
        raise AuditPolicyError("Policy allowed must be a list")
    by_url: dict[str, dict[str, Any]] = {}
    for item in allowed:
        url = item.get("url") if isinstance(item, dict) else None
        if not isinstance(url, str) or not url.startswith("https://github.com/advisories/"):
            raise AuditPolicyError("Every policy entry must contain a GitHub Advisory URL")
        if url in by_url:
            raise AuditPolicyError(f"Duplicate policy URL: {url}")
        expiry = item.get("expires")
        if not isinstance(expiry, str):
            raise AuditPolicyError(f"Policy entry {url} must contain an ISO expiry date")
        try:
            expires = date.fromisoformat(expiry)
        except ValueError as exc:
            raise AuditPolicyError(f"Policy expiry is not ISO formatted: {expiry}") from exc
        if expires < today:
            raise AuditPolicyError(f"Policy entry expired on {expiry}: {url}")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise AuditPolicyError(f"Policy entry {url} must contain a non-empty reason")
        by_url[url] = item

    accepted: list[str] = []
    rejected: list[str] = []
    for finding in _findings(report):
        url = finding.get("url")
        if not isinstance(url, str) or url not in by_url:
            rejected.append(f"{finding.get('name', 'unknown')} ({url or 'no advisory URL'})")
            continue
        accepted.append(f"{finding.get('name', 'unknown')} ({url})")
    if rejected:
        raise AuditPolicyError("Untracked npm audit findings: " + ", ".join(sorted(set(rejected))))
    return sorted(set(accepted))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="npm audit JSON report")
    parser.add_argument("--policy", type=Path, required=True, help="repository advisory policy")
    args = parser.parse_args(argv)
    try:
        accepted = check(_load(args.input), _load(args.policy))
    except AuditPolicyError as exc:
        parser.error(str(exc))
    if accepted:
        print("Accepted by explicit npm audit policy:")
        for finding in accepted:
            print(f"- {finding}")
    else:
        print("npm audit reported no vulnerabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
