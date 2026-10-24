"""Regression: lost jiggle-bone weights must be observable before normalisation."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

RBODY = Path(__file__).resolve().parents[1] / "rbody"
if str(RBODY) not in sys.path:
    sys.path.insert(0, str(RBODY))

from body_rig_audit import audit_body_rig_projection, compare_body_rig_audits
from rbody_v3_core import body_surface_view


def _example_body():
    # A real body can have a weighted physics bone absent from the outfit skin.
    # The original bone tables remain unchanged; only the solver's rig projection
    # loses the extra influence.
    j = np.asarray([
        [1, 0, 2, 0],
        [0, 2, 0, 0],
        [1, 0, 2, 0],
    ], dtype=np.uint16)
    w = np.asarray([
        [0.6, 0.4, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ], dtype=np.float32)
    return {
        "positions": np.asarray([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32),
        "normals": np.tile(np.asarray([[0,0,1]], dtype=np.float32), (3,1)),
        "uv0": np.asarray([[0,0],[1,0],[0,1]], dtype=np.float32),
        "joints": j, "weights": w,
        "indices": np.asarray([0,1,2], dtype=np.uint32),
        "joint_names": ["j_spine", "j_physics", "j_unused"],
        "mesh_records": [{"vertex_offset":0,"vertex_count":3,"index_offset":0,"index_count":3}],
        "materials": ["chara/human/c0201/obj/body/b0001/material/v0001/mt_c0201b0001_a.mtrl"],
    }


def test_real_projection_preserves_weighted_bone_loss_as_audit_not_geometry_change():
    body = _example_body()
    projected = body_surface_view(body, rig_joint_names=["j_spine"], surface_mode="body")
    audit = projected["rig_projection_audit"]

    assert audit["changes_applied"] is False
    assert audit["weighted_body_vertices"] == 3
    assert audit["missing_weighted_joint_count"] == 1
    assert [row["name"] for row in audit["missing_weighted_joints"]] == ["j_physics"]
    assert audit["vertices_with_lost_weight"] == 2
    assert audit["vertices_fully_lost"] == 0
    np.testing.assert_allclose(audit["total_missing_weight_mass"], 1.1, atol=1e-6)
    np.testing.assert_allclose(audit["missing_weight_mass_fraction"], 1.1/3.0, atol=1e-6)
    np.testing.assert_allclose(audit["lost_weight_max_fraction"], 0.6, atol=1e-6)

    # Existing solver behaviour stays byte-for-byte in principle: it drops
    # absent weight and normalises what survives. This test documents the bug;
    # the diagnostic makes the loss observable, it does not fix weights yet.
    np.testing.assert_allclose(projected["W"], np.ones((3,1)), atol=1e-6)
    np.testing.assert_array_equal(body["weights"], _example_body()["weights"])


def test_same_body_with_full_skeleton_has_zero_projection_loss():
    body = _example_body()
    projected = body_surface_view(body, rig_joint_names=body["joint_names"], surface_mode="body")
    audit = projected["rig_projection_audit"]
    assert audit["missing_weighted_joint_count"] == 0
    assert audit["vertices_with_lost_weight"] == 0
    assert audit["total_missing_weight_mass"] == 0
    np.testing.assert_allclose(projected["W"][:,0], [0.4,1.0,0.5], atol=1e-6)
    np.testing.assert_allclose(projected["W"][:,1], [0.6,0.0,0.5], atol=1e-6)


def test_unweighted_missing_bone_does_not_trigger_false_alarm():
    body = _example_body()
    result = audit_body_rig_projection(body["joints"], body["weights"],
        body["joint_names"], ["j_spine", "j_physics"])
    assert result["missing_weighted_joint_count"] == 0
    assert result["weighted_body_joint_count"] == 2
    assert result["total_missing_weight_mass"] == 0
    assert result["missing_weighted_joints"] == []


def test_source_target_joint_distribution_is_explicit_even_when_bone_is_missing():
    body = _example_body()
    source = body_surface_view(body, rig_joint_names=["j_spine"], surface_mode="body")
    # Target uses the same original body, but changes its relative jiggle weighting.
    target_body = _example_body()
    target_body["weights"][0, :2] = [0.9, 0.1]
    target = body_surface_view(target_body, rig_joint_names=["j_spine"], surface_mode="body")
    comparison = compare_body_rig_audits(source["rig_projection_audit"], target["rig_projection_audit"])
    assert comparison["changes_applied"] is False
    assert comparison["bones_missing_from_outfit_rig"] == ["j_physics"]
    physics = next(row for row in comparison["bones"] if row["name"] == "j_physics")
    assert physics["target_minus_source_mass_fraction"] > 0
    assert physics["missing_from_outfit_rig"] is True
    assert comparison["newly_weighted_target_bones"] == []
    assert compare_body_rig_audits(None, target["rig_projection_audit"]) is None
