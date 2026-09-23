from __future__ import annotations

import numpy as np

import production_b14 as p


class _Source:
    def __init__(self, meshes):
        self._meshes = meshes

    def data(self, name: str):
        return self._meshes[name]


def _grid(x0: float, x1: float, y0: float, y1: float, z: float, nx: int = 4, ny: int = 4):
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    vertices = np.asarray([[x, y, z] for y in ys for x in xs], dtype=np.float64)
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _body_triangles(z: float = 0.0):
    vertices = np.asarray([[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices[faces]


def test_repeated_cross_mesh_source_seam_is_rejoined_after_independent_fit():
    left, left_faces = _grid(-.10, 0.0, -.05, .05, .010)
    right, right_faces = _grid(0.0, .10, -.05, .05, .010)
    source = _Source({"left": {"V": left, "F": left_faces}, "right": {"V": right, "F": right_faces}})

    solved_left = left.copy()
    solved_right = right.copy()
    left_seam = np.isclose(left[:, 0], 0.0)
    right_seam = np.isclose(right[:, 0], 0.0)
    solved_left[left_seam, 0] -= .002
    solved_right[right_seam, 0] += .002

    out, report = p._preserve_final_source_shared_seams(
        source, {"left": solved_left, "right": solved_right},
        tolerance_m=.000001, minimum_pair_witnesses=3)

    assert report["enabled"]
    assert report["discovery"]["accepted_pair_count"] == 4
    assert report["seam_error_p95_before_mm"] > 3.9
    assert report["seam_error_p95_after_mm"] < 1e-6
    assert np.max(np.linalg.norm(out["left"][left_seam] - out["right"][right_seam], axis=1)) < 1e-9
    assert np.array_equal(out["left"][np.isclose(left[:, 0], -.10)], solved_left[np.isclose(left[:, 0], -.10)])
    assert np.array_equal(out["right"][np.isclose(right[:, 0], .10)], solved_right[np.isclose(right[:, 0], .10)])


def test_isolated_cross_mesh_contact_does_not_become_seam_authority():
    left, left_faces = _grid(-.10, 0.0, -.05, .05, .010)
    right, right_faces = _grid(.02, .12, -.05, .05, .010)
    right[0] = left[-1]
    source = _Source({"left": {"V": left, "F": left_faces}, "right": {"V": right, "F": right_faces}})
    solved = {"left": left.copy(), "right": right.copy()}
    solved["right"][0] += np.asarray([.004, 0.0, 0.0])

    out, report = p._preserve_final_source_shared_seams(
        source, solved, tolerance_m=.000001, minimum_pair_witnesses=3)

    assert not report["enabled"]
    assert np.array_equal(out["left"], solved["left"])
    assert np.array_equal(out["right"], solved["right"])


def test_source_boundary_copies_within_one_mesh_remain_joined():
    left, left_faces = _grid(-.10, 0., -.05, .05, .01)
    right, right_faces = _grid(0., .10, -.05, .05, .01)
    vertices = np.vstack([left, right])
    faces = np.vstack([left_faces, right_faces + len(left)])
    solved = vertices.copy()
    solved[:len(left), 2] += .004
    solved[len(left):, 2] -= .004
    source = _Source({"cup_and_strap": {"V": vertices, "F": faces}})

    out, report = p._preserve_final_source_shared_seams(source, {"cup_and_strap": solved})

    a = np.flatnonzero(np.isclose(left[:, 0], 0.))
    b = np.flatnonzero(np.isclose(right[:, 0], 0.)) + len(left)
    np.testing.assert_allclose(out["cup_and_strap"][a], out["cup_and_strap"][b], atol=1e-12)
    assert report["seam_error_p95_after_mm"] < 1e-6


def test_three_mesh_seam_junction_shares_one_displacement():
    vertices, faces = _grid(-.1, .1, -.05, .05, .01)
    source = _Source({name: {"V": vertices.copy(), "F": faces} for name in ["a", "b", "c"]})
    solved = {name: vertices + [0., 0., z] for name, z in [("a", .003), ("b", -.004), ("c", .009)]}

    out, report = p._preserve_final_source_shared_seams(source, solved)

    np.testing.assert_allclose(out["a"], out["b"], atol=1e-12)
    np.testing.assert_allclose(out["a"], out["c"], atol=1e-12)
    assert report["seam_error_p95_after_mm"] < 1e-6


def test_clearance_repairs_within_mesh_seam_as_one_surface():
    left, left_faces = _grid(-.1, 0., -.05, .05, .001)
    right, right_faces = _grid(0., .1, -.05, .05, .001)
    vertices=np.vstack([left,right]);faces=np.vstack([left_faces,right_faces+len(left)])
    source=_Source({"garment":{"V":vertices,"F":faces}})
    fitted=vertices.copy();fitted[:len(left),2]=-.003;fitted[len(left):,2]=.001
    body=_body_triangles()

    out,report=p._final_target_body_clearance(source,{"garment":fitted},body,body)

    a=np.flatnonzero(np.isclose(left[:,0],0.));b=np.flatnonzero(np.isclose(right[:,0],0.))+len(left)
    np.testing.assert_allclose(out["garment"][a],out["garment"][b],atol=1e-12)
    assert np.min(out["garment"][:,2]) >= .000099
    assert report["source_seams"]["shared_displacement"]



def test_shared_seam_is_garment_only_and_preserves_authored_offset():
    left, left_faces = _grid(-.10, 0.0, -.05, .05, .010)
    right, right_faces = _grid(.00004, .10004, -.05, .05, .010)
    source = _Source({"left": {"V": left, "F": left_faces}, "right": {"V": right, "F": right_faces}})
    solved_left = left.copy(); solved_right = right.copy()
    seam_left = np.isclose(left[:, 0], 0.0)
    seam_right = np.isclose(right[:, 0], .00004)
    solved_left[seam_left, 2] += .004
    solved_right[seam_right, 2] -= .003

    out, report = p._preserve_final_source_shared_seams(
        source, {"left": solved_left, "right": solved_right}, tolerance_m=.00005, minimum_pair_witnesses=3)

    assert report["enabled"]
    assert "body" not in " ".join(report.keys()).casefold()
    delta = out["left"][seam_left] - out["right"][seam_right]
    expected = left[seam_left] - right[seam_right]
    np.testing.assert_allclose(delta, expected, atol=1e-12)

def test_final_source_authored_clearance_repairs_small_true_penetration_including_face_interiors():
    garment, faces = _grid(-.10, .10, -.10, .10, .001, nx=5, ny=5)
    source = _Source({"garment": {"V": garment, "F": faces}})
    target = garment.copy()
    target[:, 2] -= .00130  # 0.30 mm into the target body.
    body = _body_triangles()

    out, report = p._final_source_authored_body_clearance(
        source, {"garment": target}, body, body, body, margin_m=.00005, maximum_vertex_move_m=.00150)

    assert report["enabled"]
    assert report["affected_face_count"] > 0
    assert report["maximum_move_mm"] < 1.0
    assert float(np.min(out["garment"][:, 2])) >= -.00003


def test_final_clearance_never_inflates_already_positive_clearance():
    garment, faces = _grid(-.10, .10, -.10, .10, .001, nx=5, ny=5)
    source = _Source({"garment": {"V": garment, "F": faces}})
    proposed = garment.copy()
    proposed[:, 2] = .00020  # Closer than source, but still outside the target body.
    body = _body_triangles()

    out, report = p._final_source_authored_body_clearance(
        source, {"garment": proposed}, body, body, body, margin_m=.00005)

    assert np.array_equal(out["garment"], proposed)
    assert report["meshes"]["garment"]["affected_faces"] == 0


def test_final_clearance_catches_ridge_between_midpoint_and_centroid_probes():
    garment = np.asarray([[0., 0., .001], [.04, 0., .001], [0., .04, .001]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    body, body_faces = _grid(-.01, .05, -.01, .05, 0., nx=25, ny=25)
    radius2 = np.sum((body[:, :2] - [.01, .01]) ** 2, axis=1)
    body[:, 2] = .003 * np.exp(-radius2 / .0012 ** 2)
    triangles = body[body_faces]
    old_bary = np.asarray([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
                           [.5,.5,0.],[.5,0.,.5],[0.,.5,.5],[1/3,1/3,1/3]])
    old_samples = old_bary @ garment
    assert np.min(p._nearest_surface_reference_chunked(old_samples, triangles)[2]) > 0
    quarter = np.asarray([[.5, .25, .25]])
    assert p._nearest_surface_reference_chunked(quarter @ garment, triangles)[2][0] < 0
    source = _Source({"garment": {"V": garment, "F": faces}})

    out, report = p._final_target_body_clearance(source, {"garment": garment},
                                                triangles, _body_triangles())

    assert report["affected_face_count"] == 1
    assert p._nearest_surface_reference_chunked(quarter @ out["garment"], triangles)[2][0] >= 0


def test_final_clearance_uses_target_body_even_when_source_outfit_removed_that_anatomy():
    garment, faces = _grid(-.10, .10, -.10, .10, .001, nx=5, ny=5)
    source = _Source({"garment": {"V": garment, "F": faces}})
    proposed = garment.copy()
    proposed[:, 2] = -.0020
    # Source-body topology is deliberately absent/far away. The selected target body is still final
    # occupancy authority, so the garment must clear it instead of inheriting the source hole.
    embedded_source_body = _body_triangles(z=-.050)
    target_body = _body_triangles(z=0.0)

    out, report = p._final_source_authored_body_clearance(
        source, {"garment": proposed}, embedded_source_body, target_body, target_body, margin_m=.00005)

    assert report["meshes"]["garment"]["affected_faces"] > 0
    assert float(np.min(out["garment"][:, 2])) >= .000049
    assert report["unresolved_penetrating_face_count"] == 0


def test_final_clearance_caps_deep_literal_relief_instead_of_reshaping_garment():
    garment, faces = _grid(-.10, .10, -.10, .10, .001, nx=5, ny=5)
    source = _Source({"garment": {"V": garment, "F": faces}})
    proposed = garment.copy()
    proposed[:, 2] = -.010
    body = _body_triangles()

    out, report = p._final_source_authored_body_clearance(
        source, {"garment": proposed}, body, body, body, margin_m=.00005, maximum_vertex_move_m=.00150)

    moved = np.linalg.norm(out["garment"] - proposed, axis=1)
    assert float(np.max(moved)) <= .0015001
    assert report["unresolved_penetrating_face_count"] > 0


def test_strict_source_contact_uses_embedded_body_from_suppression_plan():
    embedded = _body_triangles(z=-.050)
    catalogue = _body_triangles(z=0.0)
    cache = {
        "_ravafit_source_body_suppression": {"_source_collision_triangles": embedded},
        "slot_pairs": [{"source_literal_V": catalogue.reshape(-1, 3), "source_literal_F": np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)}],
    }
    actual = p._strict_source_literal_body_triangles(cache, catalogue)
    assert np.allclose(actual, embedded)


def test_final_clearance_and_shared_seam_compose_without_reopening_boundary():
    left, left_faces = _grid(-.10, 0.0, -.05, .05, .001, nx=4, ny=4)
    right, right_faces = _grid(0.0, .10, -.05, .05, .001, nx=4, ny=4)
    source = _Source({"left": {"V": left, "F": left_faces}, "right": {"V": right, "F": right_faces}})
    body = _body_triangles()
    solved_left = left.copy(); solved_right = right.copy()
    solved_left[:, 2] -= .00120; solved_right[:, 2] -= .00120
    solved_left[np.isclose(left[:, 0], 0.0), 0] -= .0015
    solved_right[np.isclose(right[:, 0], 0.0), 0] += .0015
    positions = {"left": solved_left, "right": solved_right}

    positions, _ = p._final_source_authored_body_clearance(source, positions, body, body, body, margin_m=.00005)
    positions, report = p._preserve_final_source_shared_seams(source, positions, tolerance_m=.000001, minimum_pair_witnesses=3)

    left_seam = np.isclose(left[:, 0], 0.0); right_seam = np.isclose(right[:, 0], 0.0)
    assert report["seam_error_p95_after_mm"] < 1e-6
    assert np.max(np.linalg.norm(positions["left"][left_seam] - positions["right"][right_seam], axis=1)) < 1e-9
    assert float(np.min(positions["left"][:, 2])) >= -.00008
    assert float(np.min(positions["right"][:, 2])) >= -.00008


def test_source_shared_seam_skinning_preserves_authored_pair_relationship():
    left, left_faces = _grid(-.10, 0.0, -.05, .05, .010)
    right, right_faces = _grid(0.0, .10, -.05, .05, .010)
    left_w = np.zeros((len(left), 3), dtype=np.float64)
    right_w = np.zeros((len(right), 3), dtype=np.float64)
    left_w[:, 0] = .60; left_w[:, 1] = .40
    right_w[:, 0] = .55; right_w[:, 1] = .45
    source = _Source({
        "left": {"V": left, "F": left_faces, "W": left_w, "joint_names": ["pelvis", "left", "right"]},
        "right": {"V": right, "F": right_faces, "W": right_w, "joint_names": ["pelvis", "left", "right"]},
    })
    solved = {"left": left.copy(), "right": right.copy()}
    # Deliberately divergent target-retargeted seam weights that would open in pose.
    a = left_w.copy(); b = right_w.copy()
    a[:, 0] = .20; a[:, 1] = .80
    b[:, 0] = .85; b[:, 1] = .15
    skinning = {
        "left": {"weights": a, "joint_names": ["pelvis", "left", "right"], "stage": {}},
        "right": {"weights": b, "joint_names": ["pelvis", "left", "right"], "stage": {}},
    }

    out, report = p._preserve_source_shared_seam_skinning(source, solved, skinning, tolerance_m=.000001, minimum_pair_witnesses=3)

    assert report["enabled"]
    assert report["pair_weight_delta_error_l1_p95_after"] < report["pair_weight_delta_error_l1_p95_before"]
    left_seam = np.flatnonzero(np.isclose(left[:, 0], 0.0))
    right_seam = np.flatnonzero(np.isclose(right[:, 0], 0.0))
    for ai, bi in zip(left_seam, right_seam):
        source_delta = left_w[ai] - right_w[bi]
        solved_delta = out["left"]["weights"][ai] - out["right"]["weights"][bi]
        np.testing.assert_allclose(solved_delta, source_delta, atol=1e-12)


def test_final_local_shell_bridge_does_not_touch_macro_shape_when_literal_matches_support(monkeypatch):
    garment, faces = _grid(-.10, .10, -.10, .10, .010, nx=5, ny=5)
    source = _Source({"garment": {"V": garment, "F": faces, "W": np.ones((len(garment), 1)), "joint_names": ["j_kosi"], "UV": None, "material": "cloth", "name": "garment"}})
    proposed = garment.copy(); proposed[:, 2] += .006

    def fake_bridge(source_vertices, faces_arg, fitted_vertices, source_body, target_body, **kwargs):
        candidate = np.asarray(fitted_vertices, dtype=np.float64).copy(); candidate[:, 2] -= .002
        return candidate, {"changed_vertex_count": len(candidate)}

    monkeypatch.setattr(p, "_bridge_new_local_curvature", fake_bridge)
    body = _body_triangles(z=0.0)
    out, report = p._apply_final_local_shell_bridge(source, {"garment": proposed}, body, body, body, max_passes=3)

    np.testing.assert_allclose(out["garment"], proposed, atol=0.0)
    assert report["changed_mesh_count"] == 0
    assert report["meshes"]["garment"]["passes"][0]["target_only_detail_seed_vertices"] == 0


def test_final_local_shell_bridge_requires_literal_target_detail_relative_to_support(monkeypatch):
    garment, faces = _grid(-.10, .10, -.10, .10, .010, nx=5, ny=5)
    source = _Source({"garment": {"V": garment, "F": faces, "W": np.ones((len(garment), 1)), "joint_names": ["j_kosi"], "UV": None, "material": "cloth", "name": "garment"}})
    proposed = garment.copy()

    def fake_bridge(source_vertices, faces_arg, fitted_vertices, source_body, target_body, **kwargs):
        candidate = np.asarray(fitted_vertices, dtype=np.float64).copy(); candidate[:, 2] -= .001
        return candidate, {"changed_vertex_count": len(candidate)}

    monkeypatch.setattr(p, "_bridge_new_local_curvature", fake_bridge)
    literal = _body_triangles(z=0.0); support = _body_triangles(z=-.002)
    out, report = p._apply_final_local_shell_bridge(source, {"garment": proposed}, literal, literal, support, max_passes=1, detail_gate_m=.00075)

    assert report["changed_mesh_count"] == 1
    assert report["meshes"]["garment"]["passes"][0]["target_only_detail_seed_vertices"] == len(garment)
    assert float(np.max(np.linalg.norm(out["garment"] - proposed, axis=1))) > .0009


def test_source_proven_seam_anchors_are_never_dropped_by_topology_guard(monkeypatch):
    left, left_faces = _grid(-.10, 0.0, -.05, .05, .010)
    right, right_faces = _grid(0.0, .10, -.05, .05, .010)
    source = _Source({"left": {"V": left, "F": left_faces}, "right": {"V": right, "F": right_faces}})
    solved_left = left.copy(); solved_right = right.copy()
    solved_left[np.isclose(left[:, 0], 0.0), 0] -= .004
    solved_right[np.isclose(right[:, 0], 0.0), 0] += .004

    calls = {"n": 0}
    def forced_warning(source_vertices, candidate_vertices, faces):
        calls["n"] += 1
        # Baseline looks healthy; every modified candidate looks catastrophically worse.  The hard
        # seam anchors must still win even though the diagnostic topology guard objects.
        if calls["n"] % 3 == 1:
            return {"flip_fraction": 0.0, "area_p01": 1.0, "edge_p99": 1.0}
        return {"flip_fraction": 1.0, "area_p01": 0.0, "edge_p99": 99.0}

    monkeypatch.setattr(p, "_source_relative_topology_summary", forced_warning)
    out, report = p._preserve_final_source_shared_seams(source, {"left": solved_left, "right": solved_right}, tolerance_m=.000001, minimum_pair_witnesses=3)

    left_seam = np.isclose(left[:, 0], 0.0); right_seam = np.isclose(right[:, 0], 0.0)
    assert report["enabled"]
    assert report["seam_error_p95_after_mm"] < 1e-6
    assert np.max(np.linalg.norm(out["left"][left_seam] - out["right"][right_seam], axis=1)) < 1e-9
