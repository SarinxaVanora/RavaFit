import numpy as np

from body_visibility import covered_body_faces


def test_cloth_outside_body_does_not_remove_body():
    body = np.asarray([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    cloth = body + [0., 0., .001]
    assert not covered_body_faces(body, cloth)[0]


def test_body_poking_through_cloth_can_authorise_cutout():
    body = np.asarray([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    cloth = body - [0., 0., .001]
    assert covered_body_faces(body, cloth)[0]


def test_partial_or_open_coverage_cannot_authorise_cutout():
    body = np.asarray([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]],
                       [[.01, 0., 0.], [.02, 0., 0.], [.02, .01, 0.]]])
    cloth = body[:1] - [0., 0., .001]
    np.testing.assert_array_equal(covered_body_faces(body, cloth), [True, False])


def test_far_clothing_cannot_authorise_cutout():
    body = np.asarray([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    assert not covered_body_faces(body, body - [0., 0., .020])[0]
