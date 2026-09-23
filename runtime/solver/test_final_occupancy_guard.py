from __future__ import annotations
import numpy as np
import final_occupancy_guard as guard


class _Source:
    def __init__(self):
        self._data = {
            "mesh": {
                "V": np.asarray([[0., 0., -.001], [1., 0., -.001], [0., 1., -.001]]),
                "F": np.asarray([[0, 1, 2]], dtype=np.int64),
            }
        }

    def data(self, name):
        return self._data[name]


class _Prod:
    pass


def _plane_occupancy(points, triangles, k=48, exact_band=.002):
    points = np.asarray(points, float)
    nearest = points.copy()
    nearest[:, 2] = 0.0
    normals = np.tile([[0., 0., 1.]], (len(points), 1))
    signed = points[:, 2].copy()
    return nearest, normals, signed, np.abs(signed), np.zeros(len(points), dtype=np.int64)


def test_dense_face_sampling_catches_interior_witness_vertices_and_centroid_miss():
    prod = _Prod()

    def occupancy(points, triangles, k=48, exact_band=.002):
        points = np.asarray(points, float)
        hit = (np.abs(points[:, 0] - .5) < 1e-10) & (np.abs(points[:, 1] - .25) < 1e-10)
        signed = np.where(hit, -.001, .002)
        return points, np.tile([[0., 0., 1.]], (len(points), 1)), signed, np.abs(signed), np.zeros(len(points), dtype=np.int64)

    prod._nearest_literal_occupancy = occupancy
    source = _Source()
    report = guard._dense_penetration_report(
        prod, source, {"mesh": source.data("mesh")["V"].copy()}, np.zeros((1, 3, 3)), .00012,
    )
    assert report["penetrating_samples"] == 1
    assert report["meshes"][0]["samples"] == 16


def test_dense_finalizer_repairs_residual_body_poke_instead_of_rejecting():
    prod = _Prod()
    source = _Source()
    base = {"mesh": source.data("mesh")["V"].copy()}
    prod._finalize_modded_coupled_solution = lambda source, cache, positions, skinning, records, contexts, rms: (positions, skinning, records, {})
    prod._target_fit_collision_triangles = lambda cache: np.zeros((1, 3, 3))
    prod._triangles_from_surface = lambda v, f: np.asarray(v)[np.asarray(f)]
    prod._nearest_literal_occupancy = _plane_occupancy
    prod._preserve_final_source_shared_seams = lambda source, positions: (positions, {"enabled": True})

    def clear(source, positions, target, support, **kwargs):
        # Keep the unit test focused on the new dense face-interior corrector rather than
        # re-implementing production's connected-cloth clearance solver in the mock.
        return {name: np.asarray(value, float).copy() for name, value in positions.items()}, {"mock": True}

    prod._final_target_body_clearance = clear
    cache = {
        "target_support_V": np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        "target_support_F": np.asarray([[0, 1, 2]], dtype=np.int64),
    }
    guard.install_dense_final_target_occupancy(prod)
    solved, _, _, stats = prod._finalize_modded_coupled_solution(source, cache, base, {}, {}, {}, 0.)
    final_report = stats["dense_final_target_occupancy"]["validation"]
    assert final_report["penetrating_samples"] == 0
    assert np.min(solved["mesh"][:, 2]) > 0.0
    assert stats["dense_final_target_occupancy"]["repair"]["converged"] is True


def test_clear_input_stays_unchanged():
    prod = _Prod()
    source = _Source()
    clear_positions = {"mesh": source.data("mesh")["V"].copy() + [0., 0., .003]}
    prod._finalize_modded_coupled_solution = lambda source, cache, positions, skinning, records, contexts, rms: (positions, skinning, records, {})
    prod._target_fit_collision_triangles = lambda cache: np.zeros((1, 3, 3))
    prod._triangles_from_surface = lambda v, f: np.asarray(v)[np.asarray(f)]
    prod._nearest_literal_occupancy = _plane_occupancy
    prod._final_target_body_clearance = lambda source, positions, target, support, **kwargs: (positions, {"mock": True})
    cache = {
        "target_support_V": np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        "target_support_F": np.asarray([[0, 1, 2]], dtype=np.int64),
    }
    guard.install_dense_final_target_occupancy(prod)
    solved, _, _, stats = prod._finalize_modded_coupled_solution(source, cache, clear_positions, {}, {}, {}, 0.)
    np.testing.assert_allclose(solved["mesh"], clear_positions["mesh"])
    assert stats["dense_final_target_occupancy"]["validation"]["penetrating_samples"] == 0
