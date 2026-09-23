"""Conservative output-body cutouts, evaluated only after garment fitting."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def covered_body_faces(body_triangles, garment_triangles, maximum_gap_m=.030):
    """Require cloth outward of every corner and centre of a proposed cutout.

    Callers supply only faces already supported by authored source cutaways.
    Nearest cloth alone is insufficient: a nearby neckline or leg opening must
    not remove visible skin. Cast short outward rays instead. Missing a candidate
    conservatively retains the body; this function never moves either surface.
    """
    body = np.asarray(body_triangles, dtype=np.float64)
    cloth = np.asarray(garment_triangles, dtype=np.float64)
    covered = np.zeros(len(body), dtype=bool)
    if not len(body) or not len(cloth):
        return covered
    e1, e2 = cloth[:, 1]-cloth[:, 0], cloth[:, 2]-cloth[:, 0]
    tree = cKDTree(cloth.mean(axis=1))
    # All corners plus the centre must lie beneath cloth. This retains a ring
    # of body triangles around a fitted opening rather than expanding its hole.
    bary = np.vstack([np.eye(3), np.full((1, 3), 1/3)])
    for start in range(0, len(body), 256):
        triangles = body[start:start+256]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-15)
        points = np.einsum('bk,fkj->fbj', bary, triangles).reshape(-1, 3)
        directions = np.repeat(normals, len(bary), axis=0)
        _, indices = tree.query(points, k=min(96, len(cloth)))
        indices = np.asarray(indices).reshape(len(points), -1)
        first, second = e1[indices], e2[indices]
        cross = np.cross(directions[:, None, :], second)
        determinant = np.einsum('nki,nki->nk', first, cross)
        usable = np.abs(determinant) > 1e-14
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=usable)
        offset = points[:, None, :]-cloth[indices, 0]
        u = np.einsum('nki,nki->nk', offset, cross)*inverse
        q = np.cross(offset, first)
        v = np.einsum('ni,nki->nk', directions, q)*inverse
        distance = np.einsum('nki,nki->nk', second, q)*inverse
        hits = usable & (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1.+1e-8)
        hits &= (distance >= -1e-7) & (distance <= maximum_gap_m)
        covered[start:start+len(triangles)] = np.all(np.any(hits, axis=1).reshape(-1, len(bary)), axis=1) & (lengths > 1e-14)
    return covered
