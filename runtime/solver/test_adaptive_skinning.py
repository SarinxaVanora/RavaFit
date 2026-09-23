from __future__ import annotations

import numpy as np

import adaptive_skinning


class _Prod:
    pass


class _Source:
    def __init__(self, name, data, capacity):
        self._name = name
        self._data = data
        attrs = {"JOINTS_0": 0, "WEIGHTS_0": 1}
        if capacity == 8:
            attrs.update({"JOINTS_1": 2, "WEIGHTS_1": 3})
        self.js = {"meshes": [{"name": name, "primitives": [{"attributes": attrs}]}]}

    def mesh_names(self):
        return [self._name]

    def data(self, name):
        if name != self._name:
            raise KeyError(name)
        return self._data


def _base_prod(delta, report):
    prod = _Prod()

    def preserve(raw_positions, source_positions, source_weights, source_joint_names, cache, behavior, effective_behavior, labels, classes, raw_to_weld):
        return np.asarray(source_weights, dtype=np.float64).copy(), {
            "mode": "source_authored_skinning_preserved",
            "exact_source_weight_preserve": True,
        }

    def verified(points, garment_weights, garment_joint_names, cache):
        return None if delta is None else np.asarray(delta, dtype=np.float64).copy(), dict(report)

    prod._retarget_garment_skinning = preserve
    prod._verified_body_skin_delta_at_points = verified
    adaptive_skinning.install_adaptive_body_skinning(prod)
    return prod


def _invoke(prod, weights, names, cache, points=None, behavior="body_following_flexible_layer"):
    count = len(weights)
    if points is None:
        points = np.zeros((count, 3), dtype=np.float64)
    else:
        points = np.asarray(points, dtype=np.float64)
    labels = np.zeros(count, dtype=np.int64)
    return prod._retarget_garment_skinning(
        points, points, np.asarray(weights, dtype=np.float64), list(names), cache,
        behavior, behavior, labels, {0: "cloth"}, np.arange(count),
    )


def _paired_cache(source_point, target_point, body_weights=(.2, .8)):
    source_point = np.asarray(source_point, dtype=np.float64).reshape(1, 3)
    target_point = np.asarray(target_point, dtype=np.float64).reshape(1, 3)
    weights = np.asarray([body_weights], dtype=np.float64)
    return {
        "names": ["j_body_a", "j_body_b"],
        "X": source_point,
        "Y": target_point,
        "BW": weights.copy(),
        "target_correspondence_W": weights.copy(),
    }


def test_same_body_field_preserves_authored_weights_exactly_when_geometry_evidence_is_unavailable():
    source = np.asarray([[0.6, 0.3, 0.1]])
    prod = _base_prod(None, {"enabled": True, "verified_body_delta": False, "reason": "same field"})
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_cloth"], {"names": ["j_body_a", "j_body_b"]})
    np.testing.assert_array_equal(solved, source)
    assert stage["exact_source_weight_preserve"] is True
    assert stage["retargeted_vertices"] == 0
    assert stage["motion_skinning_revision"] == 3


def test_identical_paired_body_geometry_and_field_preserve_close_garment_exactly():
    source = np.asarray([[0.80, 0.10, 0.10]])
    points = np.asarray([[0.0, 0.0, .001]])
    cache = _paired_cache([0., 0., 0.], [0., 0., 0.])
    prod = _base_prod(None, {"enabled": True, "verified_body_delta": False, "reason": "same field"})
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_cloth"], cache, points=points)
    np.testing.assert_array_equal(solved, source)
    assert stage["exact_source_weight_preserve"] is True
    assert stage["retargeted_vertices"] == 0


def test_same_weight_field_but_changed_target_shape_adapts_close_body_mass_to_target_motion():
    # This is the case the old revision missed: two body variants can publish identical
    # body weights while the actual support surface moves materially. A tight garment that
    # was fitted onto that new surface must move with the new body's local deformation blend.
    source = np.asarray([[0.80, 0.10, 0.10]])
    points = np.asarray([[0.0, 0.0, .001]])
    cache = _paired_cache([0., 0., 0.], [0., 0., .010], body_weights=(.2, .8))
    prod = _base_prod(None, {"enabled": True, "verified_body_delta": False, "reason": "same weight field"})
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_cloth"], cache, points=points)
    np.testing.assert_allclose(solved, [[.18, .72, .10]], atol=1e-12)
    assert solved[0, 2] == source[0, 2]
    assert stage["geometry_driven_vertices"] == 1
    assert stage["geometry_motion_authority_p95"] > .99
    assert stage["preserved_non_body_weights_exact"] is True


def test_changed_target_shape_does_not_reweight_stand_off_structure():
    source = np.asarray([[0.80, 0.10, 0.10]])
    points = np.asarray([[0.0, 0.0, .001]])
    cache = _paired_cache([0., 0., 0.], [0., 0., .010], body_weights=(.2, .8))
    prod = _base_prod(None, {"enabled": True, "verified_body_delta": False, "reason": "same weight field"})
    solved, stage = _invoke(
        prod, source, ["j_body_a", "j_body_b", "j_cloth"], cache, points=points,
        behavior="stand_off_structured_shell",
    )
    np.testing.assert_array_equal(solved, source)
    assert stage["exact_source_weight_preserve"] is True


def test_local_change_below_quantisation_floor_preserves_source_exactly():
    source = np.asarray([[0.55, 0.35, 0.10]])
    delta = np.asarray([[-0.001, 0.001]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.asarray([0, 1]),
        "local_delta_l1": np.asarray([0.002]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_cloth"], {"names": ["j_body_a", "j_body_b"]})
    np.testing.assert_array_equal(solved, source)
    assert stage["exact_source_weight_preserve"] is True
    assert stage["retargeted_vertices"] == 0


def test_changed_body_field_applies_full_delta_to_existing_body_mass():
    source = np.asarray([[0.55, 0.35, 0.10]])
    delta = np.asarray([[-0.20, 0.20]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.asarray([0, 1]),
        "local_delta_l1": np.asarray([0.40]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_cloth"], {"names": ["j_body_a", "j_body_b"]})
    np.testing.assert_allclose(solved, [[0.37, 0.53, 0.10]], atol=1e-12)
    assert solved[0, 2] == source[0, 2]
    assert stage["full_delta_vertices"] == 1
    assert stage["motion_response_residual_l1_max"] < 1e-12
    assert stage["preserved_non_body_weights_exact"] is True


def test_target_body_may_add_body_influences_without_evicting_cloth():
    source = np.asarray([[0.90, 0.0, 0.0, 0.10]])
    delta = np.asarray([[-0.40, 0.20, 0.20]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.asarray([0, 1, 2]),
        "local_delta_l1": np.asarray([0.80]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_body_c", "j_cloth"], {"names": ["j_body_a", "j_body_b", "j_body_c"]})
    np.testing.assert_allclose(solved, [[0.54, 0.18, 0.18, 0.10]], atol=1e-12)
    assert np.count_nonzero(solved[0, :3] > 1e-8) == 3
    assert solved[0, 3] == source[0, 3]
    assert stage["max_final_active_influences"] == 4
    assert stage["capacity_limited_vertices"] == 0
    assert stage["motion_response_residual_l1_max"] < 1e-12


def test_body_blend_is_capacity_limited_without_sacrificing_non_body_influence():
    source = np.asarray([[0.90, 0.0, 0.0, 0.0, 0.10]])
    delta = np.asarray([[-0.45, 0.15, 0.15, 0.15]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.asarray([0, 1, 2, 3]),
        "local_delta_l1": np.asarray([0.90]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    solved, stage = _invoke(
        prod, source,
        ["j_body_a", "j_body_b", "j_body_c", "j_body_d", "j_cloth"],
        {"names": ["j_body_a", "j_body_b", "j_body_c", "j_body_d"]},
    )
    assert solved[0, 4] == 0.10
    assert np.count_nonzero(solved[0, :4] > 1e-8) == 3
    assert abs(float(solved[0, :4].sum()) - 0.90) < 1e-12
    assert abs(float(solved.sum()) - 1.0) < 1e-12
    assert stage["capacity_limited_vertices"] == 1
    assert stage["motion_response_residual_l1_max"] > 0.0
    assert stage["source_influence_capacity"] == 4
    assert stage["source_influence_capacity_authority"] == "conservative_authored_usage_fallback"


def test_registered_eight_slot_primitive_can_use_free_second_set_for_target_motion():
    source_weights = np.asarray([[0.50, 0.25, 0.15, 0.0, 0.0, 0.0, 0.10]])
    names = ["j_body_a", "j_body_b", "j_body_c", "j_body_d", "j_body_e", "j_body_f", "j_cloth"]
    points = np.asarray([[0.01, 0.02, 0.03]])
    cache = {"names": names[:6]}
    source = _Source("mesh 1", {"V": points, "W": source_weights, "joint_names": names}, 8)
    registered = adaptive_skinning.register_source_influence_capacities(cache, source)
    assert registered["meshes"][0]["capacity"] == 8

    delta = np.asarray([[-0.30, 0.0, 0.0, 0.10, 0.10, 0.10]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.arange(6, dtype=np.int64),
        "local_delta_l1": np.asarray([0.60]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    solved, stage = _invoke(prod, source_weights, names, cache, points=points)

    assert stage["source_influence_capacity"] == 8
    assert stage["source_influence_capacity_authority"] == "source_gltf_primitive"
    assert solved[0, -1] == 0.10
    assert np.count_nonzero(solved[0, :-1] > 1e-8) == 6
    assert stage["capacity_limited_vertices"] == 0
    assert stage["motion_response_residual_l1_max"] < 1e-12


def test_registered_four_slot_primitive_stays_four_even_if_target_wants_more():
    source_weights = np.asarray([[0.60, 0.20, 0.10, 0.0, 0.10]])
    names = ["j_body_a", "j_body_b", "j_body_c", "j_body_d", "j_cloth"]
    points = np.asarray([[0.02, 0.03, 0.04]])
    cache = {"names": names[:4]}
    adaptive_skinning.register_source_influence_capacities(cache, _Source("mesh 2", {"V": points, "W": source_weights, "joint_names": names}, 4))
    delta = np.asarray([[-0.30, 0.0, 0.0, 0.30]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.arange(4, dtype=np.int64),
        "local_delta_l1": np.asarray([0.60]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    solved, stage = _invoke(prod, source_weights, names, cache, points=points)
    assert stage["source_influence_capacity"] == 4
    assert stage["source_influence_capacity_authority"] == "source_gltf_primitive"
    assert solved[0, -1] == 0.10
    assert np.count_nonzero(solved[0] > 1e-8) <= 4
    assert stage["capacity_limited_vertices"] == 1


def test_required_missing_target_joint_fails_instead_of_guessing():
    source = np.asarray([[0.90, 0.10]])
    delta = np.asarray([[-0.40, 0.40]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.asarray([0, 1]),
        "local_delta_l1": np.asarray([0.80]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    try:
        _invoke(prod, source, ["j_body_a", "j_cloth"], {"names": ["j_body_a", "j_missing_target"]})
    except ValueError as ex:
        assert "absent from the garment skin" in str(ex)
    else:
        raise AssertionError("A required target-body joint missing from the garment should not be approximated silently")
