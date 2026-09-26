from __future__ import annotations

import numpy as np

import body_skin_delta_guard


class _Prod:
    pass


def _install(delta, support_distance, *, cache_names=("a", "b")):
    prod = _Prod()

    def verified(points, garment_weights, garment_joint_names, cache):
        values = np.asarray(delta, dtype=np.float64).copy()
        return values, {
            "enabled": True,
            "verified_body_delta": True,
            "body_supported_cache_columns": np.arange(values.shape[1], dtype=np.int64),
            "local_delta_l1": np.abs(values).sum(axis=1),
            "support_distance": np.asarray(support_distance, dtype=np.float64),
            "support_distance_p95_mm": float(np.percentile(support_distance, 95) * 1000.0),
            "quantisation_floor": 0.0035,
        }

    prod._verified_body_skin_delta_at_points = verified
    body_skin_delta_guard.install_complete_body_delta_visibility(prod)
    return prod, {"names": list(cache_names)}


def test_body_skin_delta_is_full_near_tapered_then_zero_for_remote_cloth():
    delta = np.asarray([
        [-0.20, 0.20],
        [-0.20, 0.20],
        [-0.20, 0.20],
    ])
    prod, cache = _install(delta, [0.010, 0.0275, 0.180])
    gated, report = prod._verified_body_skin_delta_at_points(
        np.zeros((3, 3)), np.asarray([[0.6, 0.4]] * 3), ["a", "b"], cache,
    )

    np.testing.assert_allclose(gated[0], delta[0], atol=1e-12)
    np.testing.assert_allclose(gated[1], delta[1] * 0.5, atol=1e-12)
    np.testing.assert_array_equal(gated[2], np.zeros(2))
    np.testing.assert_allclose(report["local_delta_l1"], [0.40, 0.20, 0.0], atol=1e-12)
    assert report["body_support_distance_gate"] is True
    assert report["body_support_full_vertices"] == 1
    assert report["body_support_partial_vertices"] == 1
    assert report["body_support_source_exact_remote_vertices"] == 1


def test_remote_target_only_joint_is_visible_but_has_zero_motion_authority():
    delta = np.asarray([[-0.40, 0.40]])
    prod, cache = _install(delta, [0.180], cache_names=("a", "target_only"))
    gated, report = prod._verified_body_skin_delta_at_points(
        np.zeros((1, 3)), np.asarray([[1.0]]), ["a"], cache,
    )

    np.testing.assert_array_equal(gated, np.zeros_like(delta))
    assert report["complete_body_delta_visibility"] is True
    assert report["garment_zero_augmented_body_joint_columns"] == 1
    assert report["body_support_source_exact_remote_vertices"] == 1
