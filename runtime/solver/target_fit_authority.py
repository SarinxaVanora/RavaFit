"""Target-fit guards for skin-close garments.

The production solver owns the large-scale source-to-target mapping, structural
assembly, seams and literal target collision.  This module supplies one additional
piece of authoring evidence that must survive a body conversion: the distance a
skin-close garment was actually authored from its source support body.

That distance is transferred through the paired source/target support field.  This
is deliberately bidirectional: a solve that is too loose is pulled back toward the
authored clearance, while a solve that is too tight is pushed outward.  Only the
normal component is changed, so cup/panel construction and tangential authored shape
remain under the existing solver's authority.  Literal target anatomy still wins the
later collision pass.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    length = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(length, 1e-12), out=np.zeros_like(values), where=length > 1e-12)


def _blend(values: np.ndarray, weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(values, dtype=np.float64)[indices] * weights[:, :, None], axis=1)


def _close_authority(source_radial: np.ndarray) -> np.ndarray:
    """Use exact clearance for truly body-close cloth and fade out toward stand-off structure."""
    radial = np.asarray(source_radial, dtype=np.float64)
    # Full authority to 6 mm, smooth fade to zero by 18 mm.  This is deliberately
    # per-vertex: a close cup can be fitted while a nearby bow/trim keeps its authored
    # stand-off shape even when they share one mesh.
    t = np.clip((radial - .006) / .012, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 - smooth


def _source_clearance_fit(prod: Any, source_vertices, faces, mapped, blend, blend_ids, cache) -> tuple[np.ndarray, dict[str, Any]] | None:
    source = np.asarray(source_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mapped = np.asarray(mapped, dtype=np.float64)
    blend = np.asarray(blend, dtype=np.float64)
    blend_ids = np.asarray(blend_ids, dtype=np.int64)

    if source.ndim != 2 or source.shape[1:] != (3,) or mapped.shape != source.shape:
        return None
    if blend.ndim != 2 or blend_ids.shape != blend.shape or len(blend) != len(source):
        return None

    X = np.asarray(cache.get("X", []), dtype=np.float64)
    Y = np.asarray(cache.get("Y", []), dtype=np.float64)
    NS = np.asarray(cache.get("NS", []), dtype=np.float64)
    NT = np.asarray(cache.get("NT", []), dtype=np.float64)
    if X.ndim != 2 or X.shape[1:] != (3,) or Y.shape != X.shape or NS.shape != X.shape or NT.shape != X.shape:
        return None
    if not len(X) or np.any(blend_ids < 0) or np.any(blend_ids >= len(X)):
        return None

    source_anchor = _blend(X, blend, blend_ids)
    target_anchor = _blend(Y, blend, blend_ids)
    source_normal = _normalise_rows(_blend(NS, blend, blend_ids))
    target_normal = _normalise_rows(_blend(NT, blend, blend_ids))

    source_radial = np.einsum("ij,ij->i", source - source_anchor, source_normal)
    current_radial = np.einsum("ij,ij->i", mapped - target_anchor, target_normal)

    finite = np.isfinite(source_radial) & np.isfinite(current_radial)
    normal_ok = (np.linalg.norm(source_normal, axis=1) > .5) & (np.linalg.norm(target_normal, axis=1) > .5)
    # A modest negative source witness can come from numerical correspondence noise,
    # but large negative/far witnesses are not trustworthy garment-clearance evidence.
    witness_ok = finite & normal_ok & (source_radial > -.0025) & (source_radial < .030)
    if not np.any(witness_ok):
        return None

    authority = _close_authority(np.maximum(source_radial, 0.0))
    authority[~witness_ok] = 0.0

    # Preserve the authored source clearance, but never request actual penetration.
    # 0.35 mm is only a support-frame floor; the production literal-body clearance
    # pass remains final collision authority and can add more where anatomy requires it.
    desired_radial = np.clip(source_radial, .00035, .030)
    requested_radial = (desired_radial - current_radial) * authority
    requested_radial = np.clip(requested_radial, -.012, .012)
    proposed = mapped + requested_radial[:, None] * target_normal

    topology_alpha = 1.0
    safe = proposed
    topology = None
    edge = None
    topology_guard = getattr(prod, "_coupled_topology_safe_alpha", None)
    if len(faces) and callable(topology_guard):
        try:
            safe, topology_alpha, topology, edge = topology_guard(source, mapped, proposed, faces)
        except Exception:
            # The existing production solver will perform its own later topology guards.
            # If this optional guard cannot evaluate the mesh, do not risk a new mutation.
            return None

    final_radial = np.einsum("ij,ij->i", safe - target_anchor, target_normal)
    moved = np.linalg.norm(safe - mapped, axis=1)
    active = authority > .05
    report = {
        "enabled": True,
        "mode": "source_clearance_target_fit",
        "policy": "skin-close garment follows target macro support while preserving its authored source-body normal clearance; literal target anatomy remains final collision authority",
        "witness_vertices": int(np.count_nonzero(witness_ok)),
        "active_vertices": int(np.count_nonzero(active)),
        "source_clearance_p50_mm": float(np.median(source_radial[witness_ok]) * 1000.0),
        "source_clearance_p95_mm": float(np.percentile(source_radial[witness_ok], 95) * 1000.0),
        "before_target_clearance_p50_mm": float(np.median(current_radial[active]) * 1000.0) if np.any(active) else None,
        "after_target_clearance_p50_mm": float(np.median(final_radial[active]) * 1000.0) if np.any(active) else None,
        "requested_move_p95_mm": float(np.percentile(np.abs(requested_radial[active]), 95) * 1000.0) if np.any(active) else 0.0,
        "accepted_move_p95_mm": float(np.percentile(moved[active], 95) * 1000.0) if np.any(active) else 0.0,
        "topology_alpha": float(topology_alpha),
        "topology": topology,
        "edge": edge,
    }
    return safe, report


def install_close_shell_macro_authority(prod: Any) -> None:
    """Install source-clearance authority for genuinely skin-close garment layers."""
    if getattr(prod, "_ravafit_close_shell_macro_authority_installed", False):
        return
    original = getattr(prod, "_apply_target_relief_correction", None)
    if not callable(original):
        return

    def guarded(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        behaviour = str(behavior or "").casefold()
        clearance_mm = float((features or {}).get("source_clearance_median_mm", 999.0))
        close_behaviour = behaviour in {"constructed_close_shell", "body_following_flexible_layer"}
        if close_behaviour and clearance_mm <= 8.0:
            fitted = _source_clearance_fit(prod, source_vertices, faces, mapped, blend, blend_ids, cache)
            if fitted is not None:
                result, report = fitted
                report["behavior"] = behaviour
                report["source_clearance_median_mm"] = clearance_mm
                return result, report
            # If correspondence evidence is unavailable, preserving the already-computed
            # macro fit is safer than imprinting literal local anatomy into close cloth.
            return np.asarray(mapped, dtype=np.float64), {
                "enabled": False,
                "reason": "close garment has no reliable source-clearance witness; kept macro support without local target-relief imprint",
                "behavior": behaviour,
                "source_clearance_median_mm": clearance_mm,
            }
        return original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features)

    prod._apply_target_relief_correction = guarded
    prod._ravafit_close_shell_macro_authority_installed = True
