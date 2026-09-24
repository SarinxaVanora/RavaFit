from __future__ import annotations

import numpy as np


def _edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    out = np.zeros_like(vertices)
    if not len(faces):
        return out
    area = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    np.add.at(out, faces[:, 0], area)
    np.add.at(out, faces[:, 1], area)
    np.add.at(out, faces[:, 2], area)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def macro_body_correspondence(source: np.ndarray, literal_target: np.ndarray, faces: np.ndarray, *, radius_m: float = .022) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Low-pass the *body displacement field*, never the body or garment itself.

    B14 needs broad source->target anatomical change. Literal target grooves, nipples,
    genital clefts and other small surface relief remain collision geometry and must not
    become fitting targets. Diffusion is performed only across source-body topology so
    nearby but disconnected/opposite surfaces never contaminate one another.
    """
    source = np.asarray(source, dtype=np.float64)
    literal_target = np.asarray(literal_target, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if source.shape != literal_target.shape or source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("Macro body correspondence requires matching Nx3 source/target vertices.")
    edges = _edges(faces)
    if not len(source) or not len(edges):
        return literal_target.copy(), _vertex_normals(literal_target, faces), {"enabled": False, "reason": "no usable body topology"}

    edge_lengths = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    finite_edges = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 1e-7)]
    median_edge = float(np.median(finite_edges)) if len(finite_edges) else .004
    iterations = int(np.clip(np.ceil((float(radius_m) / max(median_edge, 1e-5)) ** 2), 6, 40))

    displacement = literal_target - source
    current = displacement.copy()
    degree = np.zeros(len(source), dtype=np.float64)
    np.add.at(degree, edges[:, 0], 1.0)
    np.add.at(degree, edges[:, 1], 1.0)
    for _ in range(iterations):
        accum = np.zeros_like(current)
        np.add.at(accum, edges[:, 0], current[edges[:, 1]])
        np.add.at(accum, edges[:, 1], current[edges[:, 0]])
        neighbour = current.copy()
        valid = degree > 0.0
        neighbour[valid] = accum[valid] / degree[valid, None]
        current = .58 * current + .42 * neighbour

    macro = source + current
    normals = _vertex_normals(macro, faces)
    rejected = displacement - current
    rejected_len = np.linalg.norm(rejected, axis=1)
    literal_len = np.linalg.norm(displacement, axis=1)
    return macro, normals, {
        "enabled": True,
        "policy": "B14 receives topology-low-pass source-to-target body motion; literal target relief remains collision-only",
        "vertices": int(len(source)),
        "edges": int(len(edges)),
        "radius_mm": float(radius_m * 1000.0),
        "median_source_edge_mm": float(median_edge * 1000.0),
        "iterations": int(iterations),
        "literal_move_p95_mm": float(np.percentile(literal_len, 95) * 1000.0),
        "rejected_local_relief_p50_mm": float(np.percentile(rejected_len, 50) * 1000.0),
        "rejected_local_relief_p95_mm": float(np.percentile(rejected_len, 95) * 1000.0),
        "rejected_local_relief_max_mm": float(np.max(rejected_len, initial=0.0) * 1000.0),
    }
