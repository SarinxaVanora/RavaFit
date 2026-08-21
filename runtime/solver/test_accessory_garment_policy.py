#!/usr/bin/env python3
from __future__ import annotations

import production_b14 as prod


def test_accessory_paths_are_always_fit_only():
    for suffix, slot in {
        "ear": "Earrings",
        "nek": "Necklace",
        "wrs": "Wrists",
        "rir": "Right Ring",
        "ril": "Left Ring",
    }.items():
        spec = {
            "game_path": f"chara/accessory/a0001/model/c0201a0001_{suffix}.mdl",
            "transplant_target_body": True,
        }
        fit_only, transplant, container_slot = prod._resolve_output_policy(spec, source_contains_body=True)
        assert fit_only is True
        assert transplant is False
        assert container_slot == slot


def test_accessory_body_detection_is_independent_from_output_authority():
    spec = {
        "game_path": "chara/accessory/a0001/model/c0201a0001_wrs.mdl",
        "source_contains_body": True,
        "transplant_target_body": True,
    }
    fit_only, transplant, container_slot = prod._resolve_output_policy(spec, source_contains_body=True)
    assert fit_only is True
    assert transplant is False
    assert container_slot == "Wrists"


def test_normal_equipment_keeps_requested_body_authority():
    spec = {
        "game_path": "chara/equipment/e0001/model/c0201e0001_top.mdl",
        "transplant_target_body": True,
    }
    fit_only, transplant, container_slot = prod._resolve_output_policy(spec, source_contains_body=True)
    assert fit_only is False
    assert transplant is True
    assert container_slot is None


def test_explicit_fit_only_also_protects_non_accessory_solver_sources():
    spec = {
        "game_path": "chara/equipment/e0001/model/c0201e0001_top.mdl",
        "fit_only": True,
        "transplant_target_body": True,
    }
    fit_only, transplant, container_slot = prod._resolve_output_policy(spec, source_contains_body=True)
    assert fit_only is True
    assert transplant is False
    assert container_slot is None


def main():
    test_accessory_paths_are_always_fit_only()
    test_accessory_body_detection_is_independent_from_output_authority()
    test_normal_equipment_keeps_requested_body_authority()
    test_explicit_fit_only_also_protects_non_accessory_solver_sources()
    print("accessory garment policy tests: PASS")


if __name__ == "__main__":
    main()
