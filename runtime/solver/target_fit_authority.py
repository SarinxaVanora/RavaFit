"""Small runtime guards for final target-body fit authority.

These guards deliberately sit outside the frozen B14 implementation. They do not
change authored garment skinning or seam logic; they only ensure the complete
selected target body remains collision authority and keep close constructed cloth
on the smooth macro support field rather than imprinting local body relief.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def enforce_full_target_collision(cache: dict[str, Any]) -> dict[str, Any]:
    """Keep source-cutaway evidence out of the fitting collision surface.

    Source omission may still propose post-fit output-body suppression, but it may
    never punch holes into the target collision surface used to fit the garment.
    """
    triangles: list[np.ndarray] = []
    for pair in cache.get("slot_pairs", []):
        vertices = np.asarray(pair.get("target_literal_V", []), dtype=np.float64)
        faces = np.asarray(pair.get("target_literal_F", []), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
            continue
        if not len(vertices) or not len(faces):
            continue
        if int(np.min(faces, initial=0)) < 0 or int(np.max(faces, initial=-1)) >= len(vertices):
            continue
        triangles.append(vertices[faces])

    full = np.vstack(triangles) if triangles else np.zeros((0, 3, 3), dtype=np.float64)
    plan = cache.get("_ravafit_source_body_suppression")
    if isinstance(plan, dict):
        plan = dict(plan)
        plan["_collision_triangles"] = full
        plan["fit_collision_policy"] = "complete selected target body; source cutaways are post-fit output candidates only"
        cache["_ravafit_source_body_suppression"] = plan

    cache.pop("_ravafit_target_collision_triangles", None)
    return cache


def install_close_shell_macro_authority(prod: Any) -> None:
    """Prevent local target relief from flattening close constructed shells.

    The normal macro/body mapping still fits the garment to the selected target,
    and literal target geometry remains available to the final collision guards.
    This only blocks the local relief field for genuinely close constructed cloth.
    """
    if getattr(prod, "_ravafit_close_shell_macro_authority_installed", False):
        return
    original = getattr(prod, "_apply_target_relief_correction", None)
    if not callable(original):
        return

    def guarded(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        behaviour = str(behavior or "").casefold()
        clearance_mm = float((features or {}).get("source_clearance_median_mm", 999.0))
        if behaviour == "constructed_close_shell" and clearance_mm <= 4.50:
            return np.asarray(mapped, dtype=np.float64), {
                "enabled": False,
                "reason": "close constructed cloth uses smooth target macro support; literal target detail remains collision authority",
                "source_clearance_median_mm": clearance_mm,
            }
        return original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features)

    prod._apply_target_relief_correction = guarded
    prod._ravafit_close_shell_macro_authority_installed = True


def prepare_worker(prod: Any, cache: dict[str, Any]) -> dict[str, Any]:
    """Apply the non-destructive fit guards used by garment workers/finalisers."""
    install_close_shell_macro_authority(prod)
    return enforce_full_target_collision(cache)
