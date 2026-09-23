"""Spread collision displacement over cloth without smoothing authored geometry."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def clearance_envelope(vertices, faces, displacement, *, radius_m=.060, maximum_move_m=.008):
    """Build a local, outward displacement envelope on connected garment topology.

    A contact point supplies a required height, not a new cloth shape. Compact
    radial kernels carry that height to nearby cloth; their zero slope at the
    centre avoids reproducing a narrow anatomical point. Distances follow mesh
    edges, so nearby disconnected ribbons and opposite layers cannot receive it.
    Only the correction is filtered: the fitted cups, folds and seams remain the
    reference surface. The caller must audit literal-body clearance afterwards.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    D = np.asarray(displacement, dtype=np.float64)
    if not len(F) or not np.any(np.linalg.norm(D, axis=1) > 1e-8):
        return D.copy(), {"enabled": False, "reason": "no contact displacement"}
    edges = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    graph = coo_matrix((np.tile(lengths, 2),
                        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(len(V), len(V))).tocsr()
    normals = np.zeros_like(V)
    area = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for corner in range(3):
        np.add.at(normals, F[:, corner], area)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    magnitude = np.linalg.norm(D, axis=1)
    projection = np.einsum("ij,ij->i", D, normals)
    # Keep unusual/inward/near-tangential repairs in the caller's original field.
    eligible = (magnitude > 1e-8) & (projection > .5 * magnitude)
    height = np.zeros(len(V))
    height[eligible] = np.minimum(magnitude[eligible] ** 2 / projection[eligible], maximum_move_m)
    envelope = height.copy()
    seeds = np.flatnonzero(eligible)
    # Bound the dense distance batches; never allocate contacts x all vertices.
    batch_size = max(1, min(32, 2_000_000 // len(V)))
    for start in range(0, len(seeds), batch_size):
        ids = seeds[start:start+batch_size]
        distances = dijkstra(graph, directed=False, indices=ids, limit=radius_m)
        r = np.minimum(distances / radius_m, 1.)
        values = height[ids, None] * (1. - r) ** 4 * (1. + 4. * r)
        envelope = np.maximum(envelope, np.max(values, axis=0))
    result = D.copy()
    use = ((magnitude <= 1e-8) | eligible) & (envelope > 1e-8)
    result[use] = normals[use] * envelope[use, None]
    return result, {"enabled": bool(np.any(use)), "radius_mm": radius_m * 1000.,
                    "contact_vertices": int(np.count_nonzero(eligible)),
                    "envelope_vertices": int(np.count_nonzero(use))}
