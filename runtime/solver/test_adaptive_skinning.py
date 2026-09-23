from __future__ import annotations

import numpy as np

import adaptive_skinning


class _Prod:
    pass


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


def _invoke(prod, weights, names, cache):
    count = len(weights)
    points = np.zeros((count, 3), dtype=np.float64)
    labels = np.zeros(count, dtype=np.int64)
    return prod._retarget_garment_skinning(
        points, points, np.asarray(weights, dtype=np.float64), list(names), cache,
        "body_following_flexible_layer", "body_following_flexible_layer", labels, {0: "cloth"}, np.arange(count),
    )


def test_same_body_field_preserves_authored_weights_exactly():
    source = np.asarray([[0.6, 0.3, 0.1]])
    prod = _base_prod(None, {"enabled": True, "verified_body_delta": False, "reason": "same field"})
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_cloth"], {"names": ["j_body_a", "j_body_b"]})
    np.testing.assert_array_equal(solved, source)
    assert stage["exact_source_weight_preserve"] is True
    assert stage["retargeted_vertices"] == 0


def test_changed_body_field_adapts_only_existing_body_mass():
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
    assert solved[0, 0] < source[0, 0]
    assert solved[0, 1] > source[0, 1]
    assert solved[0, 2] == source[0, 2]
    assert abs(float(solved.sum()) - float(source.sum())) < 1e-9
    assert abs(float(solved[0, :2].sum()) - float(source[0, :2].sum())) < 1e-9
    assert stage["preserved_non_body_weights_exact"] is True
    assert stage["retargeted_vertices"] == 1


def test_changed_body_field_does_not_add_more_body_influences_than_source():
    source = np.asarray([[0.9, 0.0, 0.0, 0.1]])
    delta = np.asarray([[-0.4, 0.2, 0.2]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.asarray([0, 1, 2]),
        "local_delta_l1": np.asarray([0.8]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    solved, stage = _invoke(prod, source, ["j_body_a", "j_body_b", "j_body_c", "j_cloth"], {"names": ["j_body_a", "j_body_b", "j_body_c"]})
    assert np.count_nonzero(solved[0, :3] > 1e-8) == 1
    assert solved[0, 3] == source[0, 3]
    assert stage["influence_budget_preserved"] is True


def test_required_missing_target_joint_fails_instead_of_guessing():
    source = np.asarray([[0.9, 0.1]])
    delta = np.asarray([[-0.4, 0.4]])
    report = {
        "enabled": True,
        "verified_body_delta": True,
        "body_supported_cache_columns": np.asarray([0, 1]),
        "local_delta_l1": np.asarray([0.8]),
        "quantisation_floor": 0.0035,
    }
    prod = _base_prod(delta, report)
    try:
        _invoke(prod, source, ["j_body_a", "j_cloth"], {"names": ["j_body_a", "j_missing_target"]})
    except ValueError as ex:
        assert "absent from the garment skin" in str(ex)
    else:
        raise AssertionError("A required target-body joint missing from the garment should not be approximated silently")
