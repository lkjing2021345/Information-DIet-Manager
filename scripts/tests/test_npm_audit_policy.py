from __future__ import annotations

from datetime import date
import unittest

from scripts.check_npm_audit import AuditPolicyError, check


ADVISORY = "https://github.com/advisories/GHSA-example"


def policy(*entries: dict[str, str]) -> dict[str, object]:
    return {"schema_version": 1, "allowed": list(entries)}


def allowed(url: str, expires: str = "2026-12-31") -> dict[str, str]:
    return {"url": url, "expires": expires, "reason": "synthetic test policy"}


def report(*urls: str) -> dict[str, object]:
    return {
        "vulnerabilities": {
            "demo": {
                "severity": "high",
                "via": [{"source": 1, "url": url, "severity": "high"} for url in urls],
            }
        }
    }


class NpmAuditPolicyTests(unittest.TestCase):
    def test_empty_report_is_accepted(self) -> None:
        self.assertEqual(check({"vulnerabilities": {}}, policy()), [])

    def test_malformed_report_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "missing the vulnerabilities"):
            check({}, policy())

    def test_policy_schema_version_is_required(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "schema_version"):
            check({"vulnerabilities": {}}, {"allowed": []})

    def test_policy_schema_version_must_be_one(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "schema_version"):
            check({"vulnerabilities": {}}, {"schema_version": 2, "allowed": []})

    def test_policy_reason_is_required(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "non-empty reason"):
            check(
                report(ADVISORY),
                policy({"url": ADVISORY, "expires": "2026-12-31"}),
            )

    def test_policy_reason_cannot_be_blank(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "non-empty reason"):
            check(
                report(ADVISORY),
                policy({"url": ADVISORY, "expires": "2026-12-31", "reason": "  "}),
            )

    def test_explicit_advisory_is_accepted_before_expiry(self) -> None:
        accepted = check(
            report(ADVISORY),
            policy(allowed(ADVISORY)),
            today=date(2026, 11, 26),
        )
        self.assertEqual(accepted, [f"demo ({ADVISORY})"])

    def test_unknown_advisory_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "Untracked npm audit findings"):
            check(report(ADVISORY), policy())

    def test_expired_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "expired"):
            check(
                report(ADVISORY),
                policy(allowed(ADVISORY, "2026-11-25")),
                today=date(2026, 11, 26),
            )

    def test_duplicate_findings_are_reported_once(self) -> None:
        accepted = check(
            report(ADVISORY, ADVISORY),
            policy(allowed(ADVISORY)),
            today=date(2026, 11, 26),
        )
        self.assertEqual(accepted, [f"demo ({ADVISORY})"])

    def test_transitive_package_reference_inherits_advisory(self) -> None:
        audit = {
            "vulnerabilities": {
                "vue": {"severity": "high", "via": ["@vue/server-renderer"]},
                "@vue/server-renderer": {
                    "severity": "high",
                    "via": [{"source": 1, "url": ADVISORY, "severity": "high"}],
                },
            }
        }
        accepted = check(
            audit,
            policy(allowed(ADVISORY)),
            today=date(2026, 11, 26),
        )
        self.assertEqual(
            accepted,
            [f"@vue/server-renderer ({ADVISORY})", f"vue ({ADVISORY})"],
        )
