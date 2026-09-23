from __future__ import annotations

import numpy as np

import target_fit_authority as authority


class _Prod:
    pass


def _cache(z_source=0.0, z_target=1.0):
    return {
        "X": np.asarray([[0., 0., z_source]]),
        "Y": np.asarray([[0., 0., z_target]]),
        "NS": np.asarray([[0., 0., 1.]]),
        "NT": np.asarray([[0., 0., 1.]]),
    }


def test_close_constructed_shell_restores_authored_source_clearance_when_too_loose():
    prod = _Prod()
    calls = []

    def original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        calls.append(behavior)
        return np.asarray(mapped) + 1.0, {"enabled": True}

    prod._apply_target_relief_correction = original
    authority.install_close_shell_macro_authority(prod)

    # Source garment was authored 1 mm above its support body.  The target macro
    # solve has left it 6 mm above the corresponding target support surface.
    source = np.asarray([[0., 0., .001]])
    mapped = np.asarray([[0., 0., 1.006]])
    result, report = prod._apply_target_relief_correction(
        source, np.zeros((0, 3), dtype=np.int64), mapped,
        np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), _cache(),
        "constructed_close_shell", {"source_clearance_median_mm": 1.0},
    )

    np.testing.assert_allclose(result, [[0., 0., 1.001]], atol=1e-9)
    assert report["mode"] == "source_clearance_target_fit"
    assert report["before_target_clearance_p50_mm"] == 6.0
    assert abs(report["after_target_clearance_p50_mm"] - 1.0) < 1e-9
    assert calls == []


def test_close_constructed_shell_restores_authored_source_clearance_when_too_tight():
    prod = _Prod()
    prod._apply_target_relief_correction = lambda *args: (np.asarray(args[2]), {"enabled": True})
    authority.install_close_shell_macro_authority(prod)

    source = np.asarray([[0., 0., .0015]])
    mapped = np.asarray([[0., 0., 1.0001]])
    result, report = prod._apply_target_relief_correction(
        source, np.zeros((0, 3), dtype=np.int64), mapped,
        np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), _cache(),
        "constructed_close_shell", {"source_clearance_median_mm": 1.5},
    )

    np.testing.assert_allclose(result, [[0., 0., 1.0015]], atol=1e-9)
    assert report["mode"] == "source_clearance_target_fit"


def test_close_flexible_layer_uses_same_source_clearance_authority():
    prod = _Prod()
    prod._apply_target_relief_correction = lambda *args: (np.asarray(args[2]) + 5.0, {"enabled": True})
    authority.install_close_shell_macro_authority(prod)

    source = np.asarray([[0., 0., .002]])
    mapped = np.asarray([[0., 0., 1.005]])
    result, report = prod._apply_target_relief_correction(
        source, np.zeros((0, 3), dtype=np.int64), mapped,
        np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), _cache(),
        "body_following_flexible_layer", {"source_clearance_median_mm": 2.0},
    )

    np.testing.assert_allclose(result, [[0., 0., 1.002]], atol=1e-9)
    assert report["behavior"] == "body_following_flexible_layer"


def test_close_fit_changes_only_target_normal_component():
    prod = _Prod()
    prod._apply_target_relief_correction = lambda *args: (np.asarray(args[2]) + 3.0, {"enabled": True})
    authority.install_close_shell_macro_authority(prod)

    source = np.asarray([[.02, -.01, .001]])
    mapped = np.asarray([[.37, -.22, 1.006]])
    result, _ = prod._apply_target_relief_correction(
        source, np.zeros((0, 3), dtype=np.int64), mapped,
        np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), _cache(),
        "constructed_close_shell", {"source_clearance_median_mm": 1.0},
    )

    assert result[0, 0] == mapped[0, 0]
    assert result[0, 1] == mapped[0, 1]
    assert abs(result[0, 2] - 1.001) < 1e-9


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
        "constructed_close_shell", {"source_clearance_median_mm": 12.0},
    )
    np.testing.assert_array_equal(result, mapped + 2.0)
    assert report["enabled"] is True
    assert calls == ["constructed_close_shell"]


def test_close_shell_without_correspondence_keeps_macro_fit_without_imprinting_relief():
    prod = _Prod()
    calls = []

    def original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        calls.append(behavior)
        return np.asarray(mapped) + 9.0, {"enabled": True}

    prod._apply_target_relief_correction = original
    authority.install_close_shell_macro_authority(prod)
    mapped = np.asarray([[1., 2., 3.]])
    result, report = prod._apply_target_relief_correction(
        mapped, np.zeros((0, 3), dtype=np.int64), mapped,
        np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), {},
        "constructed_close_shell", {"source_clearance_median_mm": 2.0},
    )
    np.testing.assert_array_equal(result, mapped)
    assert report["enabled"] is False
    assert calls == []
