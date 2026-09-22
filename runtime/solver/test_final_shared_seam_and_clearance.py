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
        source, {"left": solved_left, "right": solved_right}, None,
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
        source, solved, None, tolerance_m=.000001, minimum_pair_witnesses=3)

    assert not report["enabled"]
    assert np.array_equal(out["left"], solved["left"])
    assert np.array_equal(out["right"], solved["right"])


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


def test_final_clearance_does_not_resurrect_body_missing_from_source_outfit():
    garment, faces = _grid(-.10, .10, -.10, .10, .001, nx=5, ny=5)
    source = _Source({"garment": {"V": garment, "F": faces}})
    proposed = garment.copy()
    proposed[:, 2] = -.0020
    # The actual embedded source body is far away: this garment region had no body under it.
    embedded_source_body = _body_triangles(z=-.050)
    target_body = _body_triangles(z=0.0)

    out, report = p._final_source_authored_body_clearance(
        source, {"garment": proposed}, embedded_source_body, target_body, target_body, margin_m=.00005)

    assert np.array_equal(out["garment"], proposed)
    assert report["meshes"]["garment"]["affected_faces"] == 0


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
    positions, report = p._preserve_final_source_shared_seams(source, positions, body, source_body_triangles=body, tolerance_m=.000001, minimum_pair_witnesses=3, body_margin_m=.00005)

    left_seam = np.isclose(left[:, 0], 0.0); right_seam = np.isclose(right[:, 0], 0.0)
    assert report["seam_error_p95_after_mm"] < 1e-6
    assert np.max(np.linalg.norm(positions["left"][left_seam] - positions["right"][right_seam], axis=1)) < 1e-9
    assert float(np.min(positions["left"][:, 2])) >= -.00008
    assert float(np.min(positions["right"][:, 2])) >= -.00008
