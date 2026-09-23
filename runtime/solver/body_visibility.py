"""Conservative output-body cutouts, evaluated only after garment fitting."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _nearest_ray_hits(points, directions, cloth, edge1, edge2, tree, maximum_gap_m):
    _, indices = tree.query(points, k=min(96, len(cloth)))
    indices = np.asarray(indices).reshape(len(points), -1)
    first, second = edge1[indices], edge2[indices]
    cross = np.cross(directions[:, None, :], second)
    determinant = np.einsum("nki,nki->nk", first, cross)
    usable = np.abs(determinant) > 1e-14
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=usable)
    offset = points[:, None, :] - cloth[indices, 0]
    u = np.einsum("nki,nki->nk", offset, cross) * inverse
    q = np.cross(offset, first)
    v = np.einsum("ni,nki->nk", directions, q) * inverse
    distance = np.einsum("nki,nki->nk", second, q) * inverse
    hits = usable & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8)
    hits &= (distance >= 1e-7) & (distance <= maximum_gap_m)
    return np.min(np.where(hits, distance, np.inf), axis=1)


def covered_body_faces(body_triangles, garment_triangles, maximum_gap_m=.008):
    """Return candidate target-body faces that are inside the final fitted garment.

    Source-body omission only proposes candidates. A target face is suppressible only
    when all three corners and the face centre see the nearest fitted garment surface
    on the inward side of the body face. Clothing that simply sits outside the body,
    or an opening boundary with incomplete coverage, leaves the target body intact.
    """
    body = np.asarray(body_triangles, dtype=np.float64)
    cloth = np.asarray(garment_triangles, dtype=np.float64)
    accepted = np.zeros(len(body), dtype=bool)
    if not len(body) or not len(cloth):
        return accepted

    edge1, edge2 = cloth[:, 1] - cloth[:, 0], cloth[:, 2] - cloth[:, 0]
    tree = cKDTree(cloth.mean(axis=1))
    bary = np.vstack([np.eye(3), np.full((1, 3), 1.0 / 3.0)])

    for start in range(0, len(body), 256):
        triangles = body[start:start + 256]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-15)
        points = np.einsum("bk,fkj->fbj", bary, triangles).reshape(-1, 3)
        outward_direction = np.repeat(normals, len(bary), axis=0)
        outward = _nearest_ray_hits(points, outward_direction, cloth, edge1, edge2, tree, maximum_gap_m)
        inward = _nearest_ray_hits(points, -outward_direction, cloth, edge1, edge2, tree, maximum_gap_m)
        witness = np.isfinite(inward) & (inward + 1e-6 < outward)
        accepted[start:start + len(triangles)] = np.all(witness.reshape(-1, len(bary)), axis=1) & (lengths > 1e-14)

    return accepted
