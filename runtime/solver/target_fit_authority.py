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

try:
    from cloth_clearance_envelope import clearance_envelope, accept_envelope_step
except Exception:  # pragma: no cover - production runtime always carries this module.
    clearance_envelope = None
    accept_envelope_step = None


def _close_authority(source_distance: np.ndarray, behavior: str) -> np.ndarray:
    """Per-vertex target-frame authority derived from authored support, not mesh medians."""
    distance = np.asarray(source_distance, dtype=np.float64)
    behaviour = str(behavior or "").casefold()
    if behaviour == "constructed_close_shell":
        # Structured cups/panels remain target-supported through normal authored volume.
        lo, hi = .018, .040
    else:
        lo, hi = .006, .018
    t = np.clip((distance - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 - smooth


def _smooth_tangent_field(values: np.ndarray, faces: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Low-pass a displacement field over garment topology, never garment geometry itself."""
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


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices)
    if not len(faces):
        return normals
    area = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], area)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals


def _close_behaviour(context: dict[str, Any]) -> str | None:
    behaviour = str(context.get("behavior") or "").casefold()
    effective = str(context.get("effective_behavior") or behaviour).casefold()
    close = {"constructed_close_shell", "body_following_flexible_layer"}
    if behaviour in close:
        return behaviour
    if effective in close:
        return effective
    return None


def _fit_close_garment_to_target_frame(prod: Any, positions: dict[str, np.ndarray], contexts: dict[str, dict[str, Any]], cache: dict[str, Any], margin: float) -> tuple[dict[str, np.ndarray], set[str], dict[str, Any]]:
    """Align close cloth to the paired *smooth* target support frame.

    Flexible cloth receives only a low-frequency displacement field, so source-authored
    bridging over local anatomical relief survives a body change. Constructed shells keep
    their local form while following the target macro frame.
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
        base_behaviour = _close_behaviour(context)
        if base_behaviour is None:
            continue

        features = context.get("features") or {}
        median_clearance_mm = float(features.get("source_clearance_median_mm", 999.0))
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
        authority = _close_authority(source_distance, base_behaviour)
        authority[~finite] = 0.0
        desired_distance = np.clip(source_distance, clearance_floor, .045)
        ideal = contact + normal * desired_distance[:, None]
        error = ideal - current

        if base_behaviour == "body_following_flexible_layer":
            # This is the key anti-embossing rule: move the panel with the target's
            # broad support field, never with individual anatomical peaks/grooves.
            move = _smooth_tangent_field(error, faces, iterations=12)
            relief_rejected = error - move
        else:
            # Structured close shells keep source curvature/volume. Macro normal movement
            # follows the target; tangential movement is filtered to avoid flattening cups.
            normal_scalar = np.einsum("ij,ij->i", error, normal)
            normal_move = normal_scalar[:, None] * normal
            tangent_move = _smooth_tangent_field(error - normal_move, faces, iterations=5)
            move = normal_move + tangent_move
            relief_rejected = np.zeros_like(move)

        magnitude = np.linalg.norm(move, axis=1)
        beyond = magnitude > tolerance
        scale = np.zeros(len(move), dtype=np.float64)
        scale[beyond] = (magnitude[beyond] - tolerance) / np.maximum(magnitude[beyond], 1e-12)
        move *= (scale * authority)[:, None]

        move_length = np.linalg.norm(move, axis=1)
        too_large = move_length > .025
        move[too_large] *= (.025 / np.maximum(move_length[too_large], 1e-12))[:, None]
        active = np.linalg.norm(move, axis=1) > 1e-7
        if not np.any(active):
            current_projection = np.einsum("ij,ij->i", current - contact, normal)
            reports.append({
                "mesh": name,
                "active_vertices": 0,
                "behavior": base_behaviour,
                "source_clearance_mesh_p50_mm": median_clearance_mm,
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
        rejected = np.linalg.norm(relief_rejected, axis=1)
        reports.append({
            "mesh": name,
            "behavior": base_behaviour,
            "active_vertices": int(np.count_nonzero(active)),
            "moved_vertices": int(np.count_nonzero(moved > 1e-7)),
            "source_clearance_mesh_p50_mm": median_clearance_mm,
            "source_clearance_p50_mm": float(np.median(source_distance[finite]) * 1000.0) if np.any(finite) else None,
            "before_target_clearance_p50_mm": float(np.median(current_projection[active]) * 1000.0),
            "after_target_clearance_p50_mm": float(np.median(final_projection[active]) * 1000.0),
            "target_frame_error_p95_mm": float(np.percentile(target_error[active], 95) * 1000.0),
            "move_p95_mm": float(np.percentile(moved[active], 95) * 1000.0),
            "move_max_mm": float(np.max(moved[active]) * 1000.0),
            "rejected_local_relief_p95_mm": float(np.percentile(rejected[active], 95) * 1000.0),
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
        "policy": "paired smooth target support owns macro fit; source garment owns local authored shape; literal anatomy is never a garment-shaping target",
    }


def _shape_preserving_literal_clearance(prod: Any, before: dict[str, np.ndarray], literal: dict[str, np.ndarray], contexts: dict[str, dict[str, Any]], cache: dict[str, Any]) -> tuple[dict[str, np.ndarray], set[str], dict[str, Any]]:
    """Convert literal-body collision repairs into outward cloth envelopes.

    The production collision solver may legitimately detect that a vertex must move, but its
    raw per-vertex displacement is not allowed to become a new anatomical surface sample. For
    close garments we retain only the outward requirement, spread it across connected cloth,
    and explicitly reject inward/tangential micro-relief. This is what prevents a target-body
    cleft, nipple or fold from being embossed into otherwise smooth source-authored cloth.
    """
    out = {name: np.asarray(value, dtype=np.float64).copy() for name, value in literal.items()}
    changed: set[str] = set()
    reports: list[dict[str, Any]] = []
    support_frame = getattr(prod, "_coupled_support_frame", None)

    if not callable(clearance_envelope) or not callable(accept_envelope_step):
        return out, changed, {"enabled": False, "reason": "cloth clearance envelope unavailable"}

    for name, context in contexts.items():
        if name not in before or name not in out:
            continue
        behaviour = _close_behaviour(context)
        if behaviour is None:
            continue
        base = np.asarray(before[name], dtype=np.float64)
        raw = np.asarray(out[name], dtype=np.float64)
        if base.shape != raw.shape or base.ndim != 2 or base.shape[1:] != (3,):
            continue
        data = context.get("data") or {}
        source = np.asarray(data.get("V", []), dtype=np.float64)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        if source.shape != base.shape or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            continue

        delta = raw - base
        raw_length = np.linalg.norm(delta, axis=1)
        if not np.any(raw_length > 1e-9):
            continue

        normals = _vertex_normals(base, faces)
        if callable(support_frame):
            try:
                frame = support_frame(source, cache)
                if frame is not None:
                    support_normal = np.asarray(frame[1], dtype=np.float64)
                    support_normal /= np.maximum(np.linalg.norm(support_normal, axis=1, keepdims=True), 1e-12)
                    if support_normal.shape == normals.shape:
                        flip = np.einsum("ij,ij->i", normals, support_normal) < 0.0
                        normals[flip] *= -1.0
            except Exception:
                pass

        projection = np.einsum("ij,ij->i", delta, normals)
        outward = np.maximum(projection, 0.0)
        outward_delta = normals * outward[:, None]
        rejected = delta - outward_delta

        radius = .060 if behaviour == "body_following_flexible_layer" else .045
        maximum = .012 if behaviour == "body_following_flexible_layer" else .010
        envelope_delta, envelope_report = clearance_envelope(base, faces, outward_delta, radius_m=radius, maximum_move_m=maximum)

        # Never reduce a collision solver's proven outward requirement at a contact seed.
        envelope_projection = np.einsum("ij,ij->i", envelope_delta, normals)
        deficit = np.maximum(outward - envelope_projection, 0.0)
        envelope_delta = envelope_delta + normals * deficit[:, None]

        proposed, topology_report = accept_envelope_step(base, faces, base + envelope_delta)
        moved = np.linalg.norm(proposed - base, axis=1)
        out[name] = proposed
        if np.any(moved > 1e-9):
            changed.add(name)
        reports.append({
            "mesh": name,
            "behavior": behaviour,
            "raw_collision_vertices": int(np.count_nonzero(raw_length > 1e-9)),
            "raw_move_p95_mm": float(np.percentile(raw_length, 95) * 1000.0),
            "outward_seed_vertices": int(np.count_nonzero(outward > 1e-9)),
            "rejected_non_outward_p95_mm": float(np.percentile(np.linalg.norm(rejected, axis=1), 95) * 1000.0),
            "enveloped_move_p95_mm": float(np.percentile(moved, 95) * 1000.0),
            "envelope": envelope_report,
            "topology": topology_report,
        })

    # Every coupled clearance call is followed by a seam/attachment projection. This turns
    # authored joins into a continuing invariant instead of repairing them and then allowing
    # the next collision pass to split them again.
    seam_report: dict[str, Any] = {"enabled": False, "reason": "authored weld seam projector unavailable"}
    preserve_welds = getattr(prod, "_preserve_authored_weld_splits", None)
    if callable(preserve_welds):
        try:
            projected, seam_changed, seam_report = preserve_welds(out, contexts)
            out = {name: np.asarray(value, dtype=np.float64) for name, value in projected.items()}
            changed.update(str(name) for name in seam_changed)
        except Exception as ex:
            seam_report = {"enabled": False, "reason": f"authored weld seam projection failed: {type(ex).__name__}: {ex}"}

    return out, changed, {
        "enabled": True,
        "meshes": reports,
        "seams": seam_report,
        "policy": "literal body supplies occupancy only; close-garment collision is outward-only and spread over connected cloth before authored seams are re-projected",
    }


def install_close_shell_macro_authority(prod: Any) -> None:
    """Install source-shape authority for close garments."""
    if getattr(prod, "_ravafit_close_shell_macro_authority_installed", False):
        return

    relief_original = getattr(prod, "_apply_target_relief_correction", None)
    if callable(relief_original):
        def relief_guard(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
            behaviour = str(behavior or "").casefold()
            clearance_mm = float((features or {}).get("source_clearance_median_mm", 999.0))
            if behaviour == "constructed_close_shell":
                return np.asarray(mapped, dtype=np.float64), {
                    "enabled": False,
                    "reason": "constructed close cloth keeps source local form and uses paired smooth macro support; literal target detail remains occupancy-only",
                    "source_clearance_median_mm": clearance_mm,
                }
            return relief_original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features)
        prod._apply_target_relief_correction = relief_guard

    clearance_original = getattr(prod, "_coupled_target_clearance_guard", None)
    if callable(clearance_original):
        def clearance_guard(positions, contexts, cache, margin=.00065, enforce_support=True):
            fitted, fitted_meshes, fitted_report = _fit_close_garment_to_target_frame(prod, positions, contexts, cache, float(margin))
            literal, changed, report = clearance_original(fitted, contexts, cache, margin=margin, enforce_support=enforce_support)
            shaped, shaped_meshes, shaped_report = _shape_preserving_literal_clearance(prod, fitted, literal, contexts, cache)
            changed = set(changed) | set(fitted_meshes) | set(shaped_meshes)
            merged = dict(report or {})
            merged["source_authored_close_fit"] = fitted_report
            merged["shape_preserving_literal_clearance"] = shaped_report
            return shaped, changed, merged
        prod._coupled_target_clearance_guard = clearance_guard

    prod._ravafit_close_shell_macro_authority_installed = True
