#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

RUNTIME_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNTIME_ROOT / "b14_frozen" / "scripts"))
import coverage_analysis as coverage


def _row(body: str, variant: str, body_id: str | None = None) -> dict[str, object]:
    return {
        "collection": "Test Bodies",
        "body": body,
        "body_id": body_id or body.casefold().replace("+", "-plus"),
        "variant": variant,
        "variant_id": variant.casefold().replace(" ", "-"),
    }


ROWS = [
    _row("Dionys", "Medium"),
    _row("Dionys", "Large"),
    _row("Muse+", "Medium", "muse-plus"),
    _row("Muse+", "Large", "muse-plus"),
]


def test_selected_option_body_family_is_authoritative_before_geometry():
    narrowed, report, authoritative = coverage._source_hint_candidates(ROWS, {"option": "Dionys Large", "group": "Body", "mod": "Dress"})
    assert [(x["body"], x["variant"]) for x in narrowed] == [("Dionys", "Large")]
    assert report["method"] == "explicit-option-body-family"
    assert authoritative is narrowed[0]


def test_body_family_name_without_variant_keeps_only_that_family_for_geometry():
    narrowed, report, authoritative = coverage._source_hint_candidates(ROWS, {"option": "Dionys", "group": "Body", "mod": "Dress"})
    assert {x["body"] for x in narrowed} == {"Dionys"}
    assert len(narrowed) == 2
    assert report["method"] == "explicit-option-body-family"
    assert authoritative is None


def test_group_body_name_is_used_when_option_has_no_family():
    narrowed, report, authoritative = coverage._source_hint_candidates(ROWS, {"option": "Large", "group": "Muse+ Body", "mod": "Dress"})
    assert [(x["body"], x["variant"]) for x in narrowed] == [("Muse+", "Large")]
    assert report["method"] == "explicit-group-body-family"
    assert authoritative is narrowed[0]


def test_mod_name_is_only_a_fallback():
    narrowed, report, authoritative = coverage._source_hint_candidates(ROWS, {"option": "Default", "group": "Body", "mod": "Dionys Conversion"})
    assert {x["body"] for x in narrowed} == {"Dionys"}
    assert report["method"] == "explicit-mod-body-family"
    assert authoritative is None


def test_no_body_name_does_not_force_a_family():
    narrowed, report, authoritative = coverage._source_hint_candidates(ROWS, {"option": "Default", "group": "Body", "mod": "Pretty Dress"})
    assert len(narrowed) == len(ROWS)
    assert authoritative is None
    assert report.get("used") is False


def main():
    test_selected_option_body_family_is_authoritative_before_geometry()
    test_body_family_name_without_variant_keeps_only_that_family_for_geometry()
    test_group_body_name_is_used_when_option_has_no_family()
    test_mod_name_is_only_a_fallback()
    test_no_body_name_does_not_force_a_family()
    print("source body name hint tests: PASS")


if __name__ == "__main__":
    main()
