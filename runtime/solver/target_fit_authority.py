"""Target-fit guards for close garments.

Close clothing has two different authorities that must never be confused:

* the smooth paired target support frame owns macro fit, position and authored spacing;
* the complete literal target body owns occupancy only.

A close garment therefore follows the target body's low-frequency frame in all three
axes while retaining its own high-frequency construction. Literal nipples, grooves,
folds and genital relief are never allowed to become garment-shaping targets.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _close_authority(source_distance: np.ndarray) -> np.ndarray:
    """Full authority for skin-close cloth, fading out before stand-off structure."""
    distance = np.asarray(source_distance, dtype=np.float64)
    t = np.clip((distance - .006) / .012, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 - smooth


def _smooth_tangent_field(values: np.ndarray, faces: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Low-pass only the macro tangential retarget displacement, never garment geometry."""
    values = np.asarray(values, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not len(values) or not len(faces) or iterations <= 0:
        return values.copy()
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    current = values.copy()
    for _ in range(iterations):
        accum = np.zeros_like(current)
        counts = np.zeros(len(current), dtype=np.float64)
        np.add.at(accum, edges[:, 0], current[edges[:, 1]])
        np.add.at(accum, edges[:, 1], current[edges[:, 0]])
        np.add.at(counts, edges[:, 0], 1.0)
        np.add.at(counts, edges[:, 1], 1.0)
        mask = counts > 0.0
        neighbour = current.copy()
        neighbour[mask] = accum[mask] / counts[mask, None]
        current = .62 * current + .38 * neighbour
    return current


def _fit_close_garment_to_target_frame(prod: Any, positions: dict[str, np.ndarray], contexts: dict[str, dict[str, Any]], cache: dict[str, Any], margin: float) -> tuple[dict[str, np.ndarray], set[str], dict[str, Any]]:
    """Align close cloth to the full paired target support frame.

    The previous guard corrected only support-normal distance. That could leave a cup at
    approximately the source body's width/position even after fitting to a differently
    shaped breast. Here the exact source->target support correspondence supplies a complete
    per-vertex target frame. Normal displacement is authoritative; tangential displacement
    is low-pass filtered so macro width/position follows the target without erasing authored
    cup curvature, folds, seams or other local garment construction.
    """
    support_frame = getattr(prod, "_coupled_support_frame", None)
    topology_guard = getattr(prod, "_coupled_topology_safe_alpha", None)
    if not callable(support_frame):
        return positions, set(), {"enabled": False, "reason": "coupled support frame unavailable"}

    out = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    changed: set[str] = set()
    reports: list[dict[str, Any]] = []
    tolerance = .00015
    clearance_floor = max(.00070, float(margin))

    for name, context in contexts.items():
        if name not in out:
            continue
        behaviour = str(context.get("behavior") or "").casefold()
        effective = str(context.get("effective_behavior") or behaviour).casefold()
        features = context.get("features") or {}
        median_clearance_mm = float(features.get("source_clearance_median_mm", 999.0))
        close = {"constructed_close_shell", "body_following_flexible_layer"}
        if behaviour not in close and effective not in close:
            continue
        if median_clearance_mm > 8.0:
            continue

        data = context.get("data") or {}
        source = np.asarray(data.get("V", []), dtype=np.float64)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        current = out[name]
        if source.shape != current.shape or source.ndim != 2 or source.shape[1:] != (3,):
            continue

        frame = support_frame(source, cache)
        if frame is None:
            continue
        contact, normal, source_distance, _face_index = frame
        contact = np.asarray(contact, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        source_distance = np.asarray(source_distance, dtype=np.float64).reshape(-1)
        if contact.shape != source.shape or normal.shape != source.shape or source_distance.shape != (len(source),):
            continue
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)

        finite = np.isfinite(source_distance) & np.all(np.isfinite(contact), axis=1) & np.all(np.isfinite(normal), axis=1)
        authority = _close_authority(source_distance)
        authority[~finite] = 0.0
        desired_distance = np.clip(source_distance, clearance_floor, .030)
        ideal = contact + normal * desired_distance[:, None]
        error = ideal - current

        # Keep source-authored normal spacing. Tangential movement is the target's macro
        # width/position change, so low-pass that displacement field rather than flattening
        # the garment itself.
        normal_scalar = np.einsum("ij,ij->i", error, normal)
        normal_move = normal_scalar[:, None] * normal
        tangent_move = error - normal_move
        tangent_move = _smooth_tangent_field(tangent_move, faces, iterations=5)
        move = normal_move + tangent_move

        magnitude = np.linalg.norm(move, axis=1)
        beyond = magnitude > tolerance
        scale = np.zeros(len(move), dtype=np.float64)
        scale[beyond] = (magnitude[beyond] - tolerance) / np.maximum(magnitude[beyond], 1e-12)
        move *= (scale * authority)[:, None]

        move_length = np.linalg.norm(move, axis=1)
        too_large = move_length > .020
        move[too_large] *= (.020 / np.maximum(move_length[too_large], 1e-12))[:, None]
        active = np.linalg.norm(move, axis=1) > 1e-7
        if not np.any(active):
            current_projection = np.einsum("ij,ij->i", current - contact, normal)
            reports.append({
                "mesh": name,
                "active_vertices": 0,
                "source_clearance_p50_mm": float(np.median(source_distance[finite]) * 1000.0) if np.any(finite) else None,
                "target_clearance_p50_mm": float(np.median(current_projection[finite]) * 1000.0) if np.any(finite) else None,
            })
            continue

        proposed = current + move
        safe = proposed
        alpha = 1.0
        topology = None
        edge = None
        if len(faces) and callable(topology_guard):
            try:
                safe, alpha, topology, edge = topology_guard(source, current, proposed, faces)
            except Exception:
                reports.append({"mesh": name, "active_vertices": int(np.count_nonzero(active)), "skipped": "topology guard failed"})
                continue

        moved = np.linalg.norm(safe - current, axis=1)
        if not np.any(moved > 1e-7):
            continue
        out[name] = safe
        changed.add(name)
        final_projection = np.einsum("ij,ij->i", safe - contact, normal)
        current_projection = np.einsum("ij,ij->i", current - contact, normal)
        target_error = np.linalg.norm(safe - ideal, axis=1)
        reports.append({
            "mesh": name,
            "active_vertices": int(np.count_nonzero(active)),
            "moved_vertices": int(np.count_nonzero(moved > 1e-7)),
            "source_clearance_p50_mm": float(np.median(source_distance[finite]) * 1000.0) if np.any(finite) else None,
            "before_target_clearance_p50_mm": float(np.median(current_projection[active]) * 1000.0),
            "after_target_clearance_p50_mm": float(np.median(final_projection[active]) * 1000.0),
            "target_frame_error_p95_mm": float(np.percentile(target_error[active], 95) * 1000.0),
            "move_p95_mm": float(np.percentile(moved[active], 95) * 1000.0),
            "move_max_mm": float(np.max(moved[active]) * 1000.0),
            "tolerance_mm": float(tolerance * 1000.0),
            "clearance_floor_mm": float(clearance_floor * 1000.0),
            "topology_alpha": float(alpha),
            "topology": topology,
            "edge": edge,
        })

    return out, changed, {
        "enabled": True,
        "adjusted_mesh_count": int(len(changed)),
        "meshes": reports,
        "policy": "close cloth follows the complete paired target macro frame plus source-authored support spacing; tangential correction is low-pass displacement only, literal body detail remains collision-only, and the existing final seam stage remains untouched",
    }


def install_close_shell_macro_authority(prod: Any) -> None:
    """Install close-garment relief and full target-frame authority."""
    if getattr(prod, "_ravafit_close_shell_macro_authority_installed", False):
        return

    relief_original = getattr(prod, "_apply_target_relief_correction", None)
    if callable(relief_original):
        def relief_guard(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
            behaviour = str(behavior or "").casefold()
            clearance_mm = float((features or {}).get("source_clearance_median_mm", 999.0))
            if behaviour == "constructed_close_shell" and clearance_mm <= 4.50:
                return np.asarray(mapped, dtype=np.float64), {
                    "enabled": False,
                    "reason": "close constructed cloth uses smooth target macro support; literal target detail remains collision authority",
                    "source_clearance_median_mm": clearance_mm,
                }
            return relief_original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features)
        prod._apply_target_relief_correction = relief_guard

    clearance_original = getattr(prod, "_coupled_target_clearance_guard", None)
    if callable(clearance_original):
        def clearance_guard(positions, contexts, cache, margin=.00065, enforce_support=True):
            fitted, fitted_meshes, fitted_report = _fit_close_garment_to_target_frame(prod, positions, contexts, cache, float(margin))
            result, changed, report = clearance_original(fitted, contexts, cache, margin=margin, enforce_support=enforce_support)
            changed = set(changed) | set(fitted_meshes)
            merged = dict(report or {})
            merged["source_authored_close_fit"] = fitted_report
            return result, changed, merged
        prod._coupled_target_clearance_guard = clearance_guard

    prod._ravafit_close_shell_macro_authority_installed = True
