from __future__ import annotations

import numpy as np

import target_fit_authority as authority


class _Prod:
    pass


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


def test_constructed_shell_outside_close_threshold_keeps_existing_relief_path():
    prod = _Prod()
    calls = []

    def original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        calls.append(behavior)
        return np.asarray(mapped) + 2.0, {"enabled": True}

    prod._apply_target_relief_correction = original
    authority.install_close_shell_macro_authority(prod)
    mapped = np.asarray([[1., 2., 3.]])
    result, report = prod._apply_target_relief_correction(
        mapped, np.zeros((0, 3), dtype=np.int64), mapped, np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), {},
        "constructed_close_shell", {"source_clearance_median_mm": 8.0},
    )
    np.testing.assert_array_equal(result, mapped + 2.0)
    assert report["enabled"] is True
    assert calls == ["constructed_close_shell"]
