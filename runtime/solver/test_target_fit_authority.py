from __future__ import annotations

import numpy as np

import target_fit_authority as authority


class _Prod:
    pass


def test_full_target_collision_ignores_candidate_cutaway_holes():
    cache = {
        "slot_pairs": [{
            "slot": "Legs",
            "target_literal_V": np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
            "target_literal_F": np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        }],
        "_ravafit_source_body_suppression": {
            "native_suppression": [{"slot": "Legs", "native_mesh": 0, "triangles": [1]}],
            "_collision_triangles": np.asarray([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]),
        },
        "_ravafit_target_collision_triangles": np.zeros((0, 3, 3)),
    }
    authority.enforce_full_target_collision(cache)
    collision = cache["_ravafit_source_body_suppression"]["_collision_triangles"]
    assert collision.shape == (2, 3, 3)
    assert cache["_ravafit_source_body_suppression"]["native_suppression"][0]["triangles"] == [1]
    assert "_ravafit_target_collision_triangles" not in cache


def test_close_constructed_shell_uses_macro_support_without_local_relief():
    prod = _Prod()
    calls = []

    def original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        calls.append(behavior)
        return np.asarray(mapped) + 1.0, {"enabled": True}

    prod._apply_target_relief_correction = original
    authority.install_close_shell_macro_authority(prod)
    mapped = np.asarray([[1., 2., 3.]])
    result, report = prod._apply_target_relief_correction(
        mapped, np.zeros((0, 3), dtype=np.int64), mapped, np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), {},
        "constructed_close_shell", {"source_clearance_median_mm": 3.0},
    )
    np.testing.assert_array_equal(result, mapped)
    assert report["enabled"] is False
    assert calls == []


def test_non_close_shell_keeps_existing_relief_path():
    prod = _Prod()
    calls = []

    def original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        calls.append(behavior)
        return np.asarray(mapped) + 1.0, {"enabled": True}

    prod._apply_target_relief_correction = original
    authority.install_close_shell_macro_authority(prod)
    mapped = np.asarray([[1., 2., 3.]])
    result, report = prod._apply_target_relief_correction(
        mapped, np.zeros((0, 3), dtype=np.int64), mapped, np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), {},
        "stand_off_structured_shell", {"source_clearance_median_mm": 3.0},
    )
    np.testing.assert_array_equal(result, mapped + 1.0)
    assert report["enabled"] is True
    assert calls == ["stand_off_structured_shell"]
