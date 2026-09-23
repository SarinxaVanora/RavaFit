import numpy as np

from body_visibility import covered_body_faces


def test_nearby_opening_does_not_remove_visible_body():
    body = np.asarray([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]],
                       [[.01, 0., 0.], [.02, 0., 0.], [.02, .01, 0.]]])
    cloth = body[:1] + [0., 0., .001]
    np.testing.assert_array_equal(covered_body_faces(body, cloth), [True, False])


def test_a_cutout_requires_coverage_of_corners_not_just_centre():
    body = np.asarray([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    cloth = (body-body.mean(axis=1, keepdims=True))*.7 + body.mean(axis=1, keepdims=True) + [0., 0., .001]
    assert not covered_body_faces(body, cloth)[0]


def test_clothing_behind_body_or_far_away_cannot_authorise_a_cutout():
    body = np.asarray([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
    assert not covered_body_faces(body, body-[0., 0., .001])[0]
    assert not covered_body_faces(body, body+[0., 0., .04])[0]
