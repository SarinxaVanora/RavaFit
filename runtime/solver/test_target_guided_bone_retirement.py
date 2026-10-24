"""Target-guided retirement of source-body bones missing from the target MDL."""
from __future__ import annotations

import numpy as np
import pytest

from target_guided_bone_retirement import remap_unavailable_body_bones, install_target_guided_bone_retirement


def _cache(missing=("j_phys",), target=("j_chest", "j_spine"), names=("j_phys", "j_chest", "j_spine", "j_cloth")):
    return {
        "names": list(names),
        "_ravafit_source_body_joints_absent_from_target": list(missing),
        "_ravafit_target_body_weighted_joints": list(target),
    }


def _sampler(blend, distance=0.001):
    def sample(positions, cache, garment_weights, garment_names):
        values = np.asarray(blend, dtype=np.float64)
        assert values.shape == (len(cache["names"]),)
        return np.tile(values, (len(positions), 1)), np.full(len(positions), distance)
    return sample


def _run(weights, names, cache, sample):
    values = np.asarray(weights, dtype=np.float64)
    points = np.zeros((len(values), 3), dtype=np.float64)
    return remap_unavailable_body_bones(points, points, values, names, cache, sample)


def test_retire_body_jiggle_weights_into_target_chest_spine_field():
    # The source garment uses a weighted physics joint the target body MDL lacks.
    # The target itself proves a local 70% chest / 30% spine anatomical blend.
    names = ["j_phys", "j_chest", "j_spine", "j_cloth"]
    before = [[0.6, 0.3, 0.0, 0.1]]
    result, report = _run(before, names, _cache(), _sampler([0, 0.7, 0.3, 0]))
    np.testing.assert_allclose(result, [[0.0, 0.72, 0.18, 0.1]], atol=1e-12)
    assert report["enabled"] and report["affected_vertices"] == 1
    assert report["retired_joint_names"] == ["j_phys"]
    assert report["reassigned_weight_mass"] == pytest.approx(0.6)
    assert report["geometry_modified"] is False


def test_fully_weighted_absent_bone_is_recovered_using_target_body():
    names = ["j_phys", "j_chest", "j_spine", "j_cloth"]
    result, report = _run([[1.0, 0.0, 0.0, 0.0]], names, _cache(), _sampler([0, 0.25, 0.75, 0]))
    np.testing.assert_allclose(result, [[0, 0.25, 0.75, 0]], atol=1e-12)
    assert report["capacity_limited_vertices"] == 0


def test_preserve_bilateral_thigh_bridge_and_cloth_weight_exactly():
    names = ["j_phys", "j_thigh_l", "j_thigh_r", "j_pelvis", "j_cloth"]
    # Original mesh supports four weights. The two authored thighs and cloth
    # must remain untouched even though the physics bone is retired.
    source = [[0.2, 0.2, 0.2, 0.0, 0.4]]
    cache = _cache(target=("j_thigh_l", "j_thigh_r", "j_pelvis"), names=names)
    result, report = _run(source, names, cache, _sampler([0, 0.1, 0.1, 0.8, 0]))
    np.testing.assert_allclose(result, [[0, 0.22, 0.22, 0.16, 0.4]], atol=1e-12)
    assert report["capacity_limited_vertices"] == 0
    assert result[0, 1] >= source[0][1] and result[0, 2] >= source[0][2]
    assert result[0, 4] == source[0][4]


def test_capacity_prevents_additional_bones_without_evicting_authored_weights():
    names = ["j_phys", "j_chest", "j_spine", "j_other", "j_cloth", "j_new"]
    source = [[0.25, 0.25, 0.0, 0.25, 0.25, 0.0]]
    cache = _cache(target=("j_chest", "j_spine", "j_new"), names=names)
    result, report = _run(source, names, cache, _sampler([0, 0.5, 0.2, 0, 0, 0.3]))
    # Three authored slots remain (including j_other and cloth): only one
    # additional donor fits, but chest remains available and gains mass.
    np.testing.assert_allclose(result, [[0, 0.4125, 0, 0.25, 0.25, 0.0875]], atol=1e-12)
    assert np.count_nonzero(result[0] > 1e-8) == 4
    assert report["capacity_limited_vertices"] == 1


def test_no_supported_target_bone_refuses_unjustified_reweight():
    names = ["j_phys", "j_cloth"]
    cache = _cache(target=("j_nonexistent",), names=("j_phys", "j_cloth"))
    with pytest.raises(ValueError, match="no weighted bones"):
        _run([[0.5, 0.5]], names, cache, _sampler([0, 0]))


def test_target_sampler_with_no_local_body_support_refuses():
    names = ["j_phys", "j_chest", "j_spine", "j_cloth"]
    with pytest.raises(ValueError, match="no supported anatomical"):
        _run([[1, 0, 0, 0]], names, _cache(), _sampler([0, 0, 0, 0]))


def test_non_body_garment_detail_and_other_untouched_weights_are_not_removed():
    # No source-body evidence that a garment-only ornament joint is obsolete.
    names = ["j_ornament", "j_chest", "j_cloth"]
    cache = _cache(missing=("j_phys",), target=("j_chest",), names=names)
    source = np.asarray([[0.4, 0.5, 0.1]], dtype=np.float64)
    solved, report = _run(source, names, cache, _sampler([0, 1, 0]))
    np.testing.assert_array_equal(solved, source)
    assert report["enabled"] is False


def test_target_body_does_not_mark_existing_named_source_bones_for_removal():
    # When the per-conversion original body-bone set says none are absent,
    # preserve the entire authored skin exactly.
    names = ["j_phys", "j_chest"]
    source = np.asarray([[0.8, 0.2]], dtype=np.float64)
    cache = _cache(missing=(), target=("j_chest",), names=names)
    solved, report = _run(source, names, cache, _sampler([0, 1]))
    np.testing.assert_array_equal(solved, source)
    assert report["enabled"] is False


def test_wrapped_adaptive_result_gets_retired_and_reports_new_stage():
    class Prod:
        pass
    prod = Prod()
    names = ["j_phys", "j_chest", "j_spine", "j_cloth"]
    def original(*args):
        return np.asarray(args[2], dtype=np.float64).copy(), {"retargeted_vertices": 0, "exact_source_weight_preserve": True}
    prod._retarget_garment_skinning = original
    prod._target_skin_weights_at_points = _sampler([0, 0.75, 0.25, 0])
    install_target_guided_bone_retirement(prod)
    points = np.asarray([[0, 0, 0]], dtype=np.float64)
    source = np.asarray([[0.6, 0.3, 0, 0.1]], dtype=np.float64)
    result, stage = prod._retarget_garment_skinning(
        points, points, source, names, _cache(), "flexible", "flexible", [], {}, [],
    )
    np.testing.assert_allclose(result, [[0, 0.75, 0.15, 0.1]], atol=1e-12)
    assert stage["retargeted_vertices"] == 1
    assert stage["exact_source_weight_preserve"] is False
    assert stage["target_guided_unavailable_body_bones"]["enabled"]
    install_target_guided_bone_retirement(prod)
    assert prod._ravafit_target_guided_body_bone_retirement_installed
