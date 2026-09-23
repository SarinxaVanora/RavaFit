#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

RUNTIME_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNTIME_ROOT / "b14_frozen" / "scripts"))
import coverage_analysis as coverage


class _FakeGlb:
    def __init__(self, *_args, **_kwargs):
        names = ["j_kosi", "j_asi_a_l", "j_asi_b_l", "j_asi_d_l", "j_sebo_b", "j_te_l", "j_sebo_a"]
        vertices = np.asarray([
            [-0.20, 0.30, 0.00], [0.20, 0.30, 0.00], [0.00, 0.50, 0.00],
            [0.92, 0.10, 0.00], [1.08, 0.10, 0.00], [1.00, 0.22, 0.00],
        ], dtype=np.float64)
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        weights = np.zeros((len(vertices), len(names)), dtype=np.float64)
        weights[:, names.index("j_sebo_a")] = 0.72
        weights[:, names.index("j_te_l")] = 0.28
        self._data = {
            "V": vertices,
            "F": faces,
            "W": weights,
            "joint_names": names,
            "material": "chara/equipment/test/garment.mtrl",
        }

    def mesh_names(self):
        return ["jacket"]

    def data(self, _name):
        return self._data


def _fake_skeleton_positions(_glb, joint_names):
    positions = {
        "j_kosi": [0.0, 0.0, 0.0],
        "j_asi_a_l": [0.0, -0.45, 0.0],
        "j_asi_b_l": [0.0, -0.85, 0.0],
        "j_asi_d_l": [0.0, -1.25, 0.0],
        "j_sebo_b": [0.0, 0.50, 0.0],
        "j_te_l": [1.0, 0.14, 0.0],
        "j_sebo_a": [0.0, 0.30, 0.0],
    }
    return np.asarray([positions[name] for name in joint_names], dtype=np.float64), None, None


def test_wrist_container_can_resolve_as_chest_with_hands_support_and_no_body():
    original_glb = coverage.GLB
    original_skeleton = coverage.skeleton_global_positions
    coverage.GLB = _FakeGlb
    coverage.skeleton_global_positions = _fake_skeleton_positions
    try:
        result = coverage.analyse_coverage(Path("synthetic.glb"), "chara/accessory/a0042/model/c0201a0042_wrs.mdl")
    finally:
        coverage.GLB = original_glb
        coverage.skeleton_global_positions = original_skeleton

    assert result["container_slot"] == "Wrists", result
    assert result["primary_slot"] == "Chest", result
    assert result["source_contains_body"] is False, result
    assert result["slots"]["Chest"]["primary"] is True, result
    assert result["slots"]["Hands"]["recommended"] is True, result


def main():
    test_wrist_container_can_resolve_as_chest_with_hands_support_and_no_body()
    print("accessory coverage analysis tests: PASS")


if __name__ == "__main__":
    main()
