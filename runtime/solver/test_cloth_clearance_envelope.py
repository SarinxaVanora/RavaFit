import numpy as np

import production_b14 as p
from cloth_clearance_envelope import clearance_envelope, accept_envelope_step


def _cup(n=25):
    xy = np.linspace(-.072, .072, n)
    vertices = np.asarray([[x, y, .060 - (x*x+y*y)/.18] for y in xy for x in xy])
    faces = []
    for y in range(n-1):
        for x in range(n-1):
            a = y*n+x
            faces.extend([[a, a+1, a+n+1], [a, a+n+1, a+n]])
    return vertices, np.asarray(faces, dtype=np.int64)


def test_main_body_relief_is_bridged_without_flattening_the_cup():
    body, faces = _cup()
    garment = body + [0., 0., .002]
    radius = np.linalg.norm(body[:, :2], axis=1)
    body[:, 2] += .006 * np.exp(-.5*(radius/.003)**2)
    class Source:
        def data(self, name):
            return {"V": garment, "F": faces}

    # The relief is embedded in the main body: support == literal collision mesh.
    result, report = p._final_target_body_clearance(Source(), {"cup": garment}, body[faces], body[faces])
    fitted = result["cup"]
    apex = np.argmin(radius)
    ring = (radius >= .010) & (radius <= .013)
    displacement = fitted[:, 2] - garment[:, 2]
    assert report["unresolved_penetrating_face_count"] == 0
    assert fitted[apex, 2] > body[apex, 2]
    # At least 60% of the apex correction reaches the surrounding 10-13 mm ring,
    # rather than producing the body's isolated 3 mm wide protrusion in the cloth.
    assert np.min(displacement[ring]) >= .60 * displacement[apex]
    # The broad convex cup and distant construction remain the reference shape.
    np.testing.assert_allclose(fitted[radius > .065], garment[radius > .065], rtol=0., atol=.000002)
    np.testing.assert_array_equal(fitted[radius > .090], garment[radius > .090])
    assert fitted[apex, 2] - np.mean(fitted[radius > .065, 2]) > .024
    assert report["meshes"]["cup"]["cloth_envelope"]["enabled"]


def test_envelope_does_not_jump_to_an_adjacent_disconnected_layer():
    vertices, faces = _cup(13)
    vertices = np.vstack([vertices, vertices + [0., 0., .001]])
    faces = np.vstack([faces, faces + len(vertices)//2])
    correction = np.zeros_like(vertices)
    correction[84, 2] = .003

    result, report = clearance_envelope(vertices, faces, correction)

    assert report["envelope_vertices"] > 1
    np.testing.assert_array_equal(result[len(vertices)//2:], 0.)
    assert np.max(np.linalg.norm(result, axis=1)) <= .008


def test_envelope_leaves_noncolliding_authored_relief_exactly_unchanged():
    vertices, faces = _cup()
    vertices[:, 2] += .004*np.sin(vertices[:, 0]*400.)
    correction = np.zeros_like(vertices)

    result, report = clearance_envelope(vertices, faces, correction)

    np.testing.assert_array_equal(result, correction)
    assert not report["enabled"]


def test_contact_spread_uses_physical_size_instead_of_mesh_ring_count():
    heights = []
    for resolution in (13, 25):
        vertices, faces = _cup(resolution)
        apex = np.argmin(np.linalg.norm(vertices[:, :2], axis=1))
        correction = np.zeros_like(vertices)
        correction[apex, 2] = .003
        result, _ = clearance_envelope(vertices, faces, correction)
        sample = np.argmin(np.linalg.norm(vertices[:, :2]-[.024, 0.], axis=1))
        heights.append(np.linalg.norm(result[sample]))
    np.testing.assert_allclose(heights[0], heights[1], rtol=.02)


def test_unsafe_local_triangle_does_not_cancel_other_envelope_repairs():
    vertices=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                         [2.,0.,0.],[3.,0.,0.],[2.,1.,0.]])
    faces=np.asarray([[0,1,2],[3,4,5]])
    proposed=vertices.copy();proposed[1,0]=-2.;proposed[3:,2]=.003

    result,report=accept_envelope_step(vertices,faces,proposed)

    assert np.cross(result[1]-result[0],result[2]-result[0])[2]>0.
    np.testing.assert_array_equal(result[3:],proposed[3:])
    assert report['limited_vertices']==3
