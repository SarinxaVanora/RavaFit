"""Target-fit guards for close garments.

Two rules live here, outside frozen B14:

1. Close constructed cloth should not inherit literal target micro-relief during the
   early relief stage. Target macro shape still drives the fit and literal anatomy
   still drives final collision.
2. The late coupled finalizer must preserve the source-authored garment/body spacing
   in both directions. The production guard already pushes cloth outward when it is
   too close, but historically did not pull cloth back in when a target solve was too
   loose. That asymmetry is what this module closes.
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


def _pull_close_garment_to_authored_clearance(prod: Any, positions: dict[str, np.ndarray], contexts: dict[str, dict[str, Any]], cache: dict[str, Any], margin: float) -> tuple[dict[str, np.ndarray], set[str], dict[str, Any]]:
    """Pull only demonstrably loose close cloth toward its source-authored body spacing.

    ``production_b14._coupled_support_frame`` gives the exact source->target support
    correspondence for each untouched source garment vertex. Its returned distance is
    therefore the authored source-body spacing we want to preserve. We alter only the
    target support-normal component; tangential garment construction stays untouched.
    """
    support_frame = getattr(prod, "_coupled_support_frame", None)
    topology_guard = getattr(prod, "_coupled_topology_safe_alpha", None)
    if not callable(support_frame):
        return positions, set(), {"enabled": False, "reason": "coupled support frame unavailable"}

    out = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    changed: set[str] = set()
    reports: list[dict[str, Any]] = []
    tolerance = max(.00045, float(margin) * .70)

    for name, context in contexts.items():
        if name not in out:
            continue
        behaviour = str(context.get("behavior") or "").casefold()
        effective = str(context.get("effective_behavior") or behaviour).casefold()
        features = context.get("features") or {}
        median_clearance_mm = float(features.get("source_clearance_median_mm", 999.0))
        if behaviour not in {"constructed_close_shell", "body_following_flexible_layer"} and effective not in {"constructed_close_shell", "body_following_flexible_layer"}:
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

        current_projection = np.einsum("ij,ij->i", current - contact, normal)
        finite = np.isfinite(source_distance) & np.isfinite(current_projection)
        authority = _close_authority(source_distance)
        authority[~finite] = 0.0

        # The source distance is already a positive Euclidean support spacing. Keep a
        # small collision floor, but do not preserve accidental source penetration.
        desired = np.clip(source_distance, max(float(margin), .00035), .030)
        excess = current_projection - (desired + tolerance)
        pull = np.maximum(excess, 0.0) * authority
        pull = np.minimum(pull, .0080)
        active = pull > 1e-7
        if not np.any(active):
            reports.append({
                "mesh": name,
                "active_vertices": 0,
                "source_clearance_p50_mm": float(np.median(source_distance[finite]) * 1000.0) if np.any(finite) else None,
                "target_clearance_p50_mm": float(np.median(current_projection[finite]) * 1000.0) if np.any(finite) else None,
            })
            continue

        proposed = current - pull[:, None] * normal
        safe = proposed
        alpha = 1.0
        topology = None
        edge = None
        if len(faces) and callable(topology_guard):
            try:
                safe, alpha, topology, edge = topology_guard(source, current, proposed, faces)
            except Exception:
                # Correctness over bravado: if the shared topology veto cannot certify
                # this inward correction, leave this mesh alone and let normal collision
                # handling continue unchanged.
                reports.append({"mesh": name, "active_vertices": int(np.count_nonzero(active)), "skipped": "topology guard failed"})
                continue

        moved = np.linalg.norm(safe - current, axis=1)
        if not np.any(moved > 1e-7):
            continue
        out[name] = safe
        changed.add(name)
        final_projection = np.einsum("ij,ij->i", safe - contact, normal)
        reports.append({
            "mesh": name,
            "active_vertices": int(np.count_nonzero(active)),
            "moved_vertices": int(np.count_nonzero(moved > 1e-7)),
            "source_clearance_p50_mm": float(np.median(source_distance[finite]) * 1000.0) if np.any(finite) else None,
            "before_target_clearance_p50_mm": float(np.median(current_projection[active]) * 1000.0),
            "after_target_clearance_p50_mm": float(np.median(final_projection[active]) * 1000.0),
            "move_p95_mm": float(np.percentile(moved[active], 95) * 1000.0),
            "move_max_mm": float(np.max(moved[active]) * 1000.0),
            "tolerance_mm": float(tolerance * 1000.0),
            "topology_alpha": float(alpha),
            "topology": topology,
            "edge": edge,
        })

    return out, changed, {
        "enabled": True,
        "adjusted_mesh_count": int(len(changed)),
        "meshes": reports,
        "policy": "close cloth may be pulled inward only toward its source-authored support spacing; tangential authored shape and later literal collision remain authoritative",
    }


def install_close_shell_macro_authority(prod: Any) -> None:
    """Install close-garment relief and late source-clearance authority."""
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
            tightened, tightened_meshes, tightened_report = _pull_close_garment_to_authored_clearance(prod, positions, contexts, cache, float(margin))
            result, changed, report = clearance_original(tightened, contexts, cache, margin=margin, enforce_support=enforce_support)
            changed = set(changed) | set(tightened_meshes)
            merged = dict(report or {})
            merged["source_authored_close_fit"] = tightened_report
            return result, changed, merged
        prod._coupled_target_clearance_guard = clearance_guard

    prod._ravafit_close_shell_macro_authority_installed = True
