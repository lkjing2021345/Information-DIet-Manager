from __future__ import annotations

from datetime import date
import unittest

from scripts.check_npm_audit import AuditPolicyError, check


ADVISORY = "https://github.com/advisories/GHSA-example"


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
        self.assertEqual(check({"vulnerabilities": {}}, {"allowed": []}), [])

    def test_malformed_report_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "missing the vulnerabilities"):
            check({}, {"allowed": []})

    def test_explicit_advisory_is_accepted_before_expiry(self) -> None:
        accepted = check(
            report(ADVISORY),
            {"allowed": [{"url": ADVISORY, "expires": "2026-12-31"}]},
            today=date(2026, 11, 26),
        )
        self.assertEqual(accepted, [f"demo ({ADVISORY})"])

    def test_unknown_advisory_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "Untracked npm audit findings"):
            check(report(ADVISORY), {"allowed": []})

    def test_expired_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(AuditPolicyError, "expired"):
            check(
                report(ADVISORY),
                {"allowed": [{"url": ADVISORY, "expires": "2026-11-25"}]},
                today=date(2026, 11, 26),
            )

    def test_duplicate_findings_are_reported_once(self) -> None:
        accepted = check(
            report(ADVISORY, ADVISORY),
            {"allowed": [{"url": ADVISORY, "expires": "2026-12-31"}]},
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
            {"allowed": [{"url": ADVISORY, "expires": "2026-12-31"}]},
            today=date(2026, 11, 26),
        )
        self.assertEqual(
            accepted,
            [f"@vue/server-renderer ({ADVISORY})", f"vue ({ADVISORY})"],
        )
