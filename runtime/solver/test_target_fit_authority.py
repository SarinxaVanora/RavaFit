from __future__ import annotations

import numpy as np

import target_fit_authority as authority


class _Prod:
    pass


def _install_fake_coupled_prod(source_distance=.001, contact_shift=(0., 0., 0.)):
    prod = _Prod()
    contact_shift = np.asarray(contact_shift, dtype=np.float64)

    def relief_original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        return np.asarray(mapped) + 9.0, {"enabled": True, "original": True}

    def support_frame(source, cache):
        count = len(source)
        contact = np.asarray(source, dtype=np.float64) + np.asarray([0., 0., 1.]) + contact_shift
        normal = np.tile(np.asarray([[0., 0., 1.]]), (count, 1))
        distance = np.full(count, source_distance)
        return contact, normal, distance, np.zeros(count, dtype=np.int64)

    def topology_guard(source, before, proposed, faces):
        return np.asarray(proposed), 1.0, {"ok": True}, {"ok": True}

    def clearance_original(positions, contexts, cache, margin=.00065, enforce_support=True):
        return {k: np.asarray(v).copy() for k, v in positions.items()}, set(), {"enabled": True, "original": True}

    prod._apply_target_relief_correction = relief_original
    prod._coupled_support_frame = support_frame
    prod._coupled_topology_safe_alpha = topology_guard
    prod._coupled_target_clearance_guard = clearance_original
    authority.install_close_shell_macro_authority(prod)
    return prod


def _context(behaviour="constructed_close_shell", clearance_mm=1.0):
    source = np.asarray([
        [-.01, 0., 0.],
        [.01, 0., 0.],
        [0., .01, 0.],
    ])
    return {
        "data": {"V": source, "F": np.asarray([[0, 1, 2]], dtype=np.int64)},
        "behavior": behaviour,
        "effective_behavior": behaviour,
        "features": {"source_clearance_median_mm": clearance_mm},
    }


def test_close_constructed_relief_still_uses_macro_support_only():
    prod = _install_fake_coupled_prod()
    mapped = np.asarray([[1., 2., 3.]])
    result, report = prod._apply_target_relief_correction(
        mapped, np.zeros((0, 3), dtype=np.int64), mapped,
        np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), {},
        "constructed_close_shell", {"source_clearance_median_mm": 3.0},
    )
    np.testing.assert_array_equal(result, mapped)
    assert report["enabled"] is False


def test_non_close_relief_keeps_existing_path():
    prod = _install_fake_coupled_prod()
    mapped = np.asarray([[1., 2., 3.]])
    result, report = prod._apply_target_relief_correction(
        mapped, np.zeros((0, 3), dtype=np.int64), mapped,
        np.ones((1, 1)), np.zeros((1, 1), dtype=np.int64), {},
        "stand_off_structured_shell", {"source_clearance_median_mm": 3.0},
    )
    np.testing.assert_array_equal(result, mapped + 9.0)
    assert report["original"] is True


def test_late_coupled_guard_pulls_loose_close_shell_back_to_authored_spacing():
    prod = _install_fake_coupled_prod(source_distance=.001)
    context = _context("constructed_close_shell", 1.0)
    current = np.asarray(context["data"]["V"]).copy()
    current[:, 2] = 1.006

    result, changed, report = prod._coupled_target_clearance_guard(
        {"cup": current}, {"cup": context}, {}, margin=.00065, enforce_support=True,
    )

    expected = 1.0 + .001 + .00015
    np.testing.assert_allclose(result["cup"][:, 2], expected, atol=1e-9)
    assert "cup" in changed
    fit = report["source_authored_close_fit"]
    assert fit["adjusted_mesh_count"] == 1
    assert fit["meshes"][0]["move_p95_mm"] > 4.0


def test_close_shell_follows_target_frame_tangentially_without_destroying_local_form():
    prod = _install_fake_coupled_prod(source_distance=.001, contact_shift=(.010, 0., 0.))
    context = _context("constructed_close_shell", 1.0)
    source = np.asarray(context["data"]["V"]).copy()
    current = source + np.asarray([0., 0., 1.001])
    before_edges = np.asarray([
        current[1] - current[0],
        current[2] - current[1],
        current[0] - current[2],
    ])

    result, changed, report = prod._coupled_target_clearance_guard(
        {"cup": current}, {"cup": context}, {}, margin=.00065, enforce_support=True,
    )

    solved = result["cup"]
    after_edges = np.asarray([
        solved[1] - solved[0],
        solved[2] - solved[1],
        solved[0] - solved[2],
    ])
    assert "cup" in changed
    # Ten millimetres of paired target-frame translation should be followed to within
    # the deliberate 0.15 mm numerical tolerance.
    shift = np.mean(solved - current, axis=0)
    assert 0.00980 < float(shift[0]) < 0.01001
    assert abs(float(shift[1])) < 1e-10
    np.testing.assert_allclose(after_edges, before_edges, atol=1e-10)
    fit = report["source_authored_close_fit"]["meshes"][0]
    assert fit["target_frame_error_p95_mm"] <= .151


def test_late_coupled_guard_does_not_pull_already_close_shell_further_in():
    prod = _install_fake_coupled_prod(source_distance=.001)
    context = _context("constructed_close_shell", 1.0)
    current = np.asarray(context["data"]["V"]).copy()
    current[:, 2] = 1.0011

    result, changed, report = prod._coupled_target_clearance_guard(
        {"cup": current}, {"cup": context}, {}, margin=.00065, enforce_support=True,
    )

    np.testing.assert_array_equal(result["cup"], current)
    assert "cup" not in changed
    assert report["source_authored_close_fit"]["adjusted_mesh_count"] == 0


def test_literal_floor_prevents_copying_an_unsafe_tiny_source_gap():
    prod = _install_fake_coupled_prod(source_distance=.0001)
    context = _context("constructed_close_shell", .1)
    current = np.asarray(context["data"]["V"]).copy()
    current[:, 2] = 1.004

    result, changed, report = prod._coupled_target_clearance_guard(
        {"cup": current}, {"cup": context}, {}, margin=.00065, enforce_support=True,
    )

    expected = 1.0 + .00070 + .00015
    np.testing.assert_allclose(result["cup"][:, 2], expected, atol=1e-9)
    assert "cup" in changed
    assert report["source_authored_close_fit"]["meshes"][0]["clearance_floor_mm"] == .7


def test_body_following_layer_uses_same_full_target_frame_rule():
    prod = _install_fake_coupled_prod(source_distance=.002)
    context = _context("body_following_flexible_layer", 2.0)
    current = np.asarray(context["data"]["V"]).copy()
    current[:, 2] = 1.007

    result, changed, _ = prod._coupled_target_clearance_guard(
        {"stocking": current}, {"stocking": context}, {}, margin=.00065, enforce_support=False,
    )

    assert "stocking" in changed
    assert float(np.max(result["stocking"][:, 2])) < 1.0023


def test_stand_off_structure_is_never_shrink_wrapped_by_late_guard():
    prod = _install_fake_coupled_prod(source_distance=.001, contact_shift=(.010, 0., 0.))
    context = _context("stand_off_structured_shell", 1.0)
    current = np.asarray(context["data"]["V"]).copy()
    current[:, 2] = 1.020

    result, changed, report = prod._coupled_target_clearance_guard(
        {"bow": current}, {"bow": context}, {}, margin=.00065, enforce_support=True,
    )

    np.testing.assert_array_equal(result["bow"], current)
    assert "bow" not in changed
    assert report["source_authored_close_fit"]["adjusted_mesh_count"] == 0


def test_close_component_above_clearance_classification_threshold_is_left_alone():
    prod = _install_fake_coupled_prod(source_distance=.001, contact_shift=(.010, 0., 0.))
    context = _context("constructed_close_shell", 12.0)
    current = np.asarray(context["data"]["V"]).copy()
    current[:, 2] = 1.020

    result, changed, _ = prod._coupled_target_clearance_guard(
        {"structured": current}, {"structured": context}, {}, margin=.00065, enforce_support=True,
    )

    np.testing.assert_array_equal(result["structured"], current)
    assert "structured" not in changed
