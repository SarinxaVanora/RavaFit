from __future__ import annotations

from typing import Any

import numpy as np


def _target_surfaces(prod: Any, cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    collision_fn = getattr(prod, "_target_fit_collision_triangles", None)
    triangles_fn = getattr(prod, "_triangles_from_surface", None)
    if not callable(collision_fn) or not callable(triangles_fn):
        return None
    collision = np.asarray(collision_fn(cache), dtype=np.float64)
    support = cache.get("_ravafit_target_support_triangles")
    if support is None:
        support_v = np.asarray(cache.get("target_support_V", []), dtype=np.float64)
        support_f = np.asarray(cache.get("target_support_F", []), dtype=np.int64)
        if support_v.ndim != 2 or support_v.shape[1:] != (3,) or support_f.ndim != 2 or support_f.shape[1:] != (3,):
            return None
        support = triangles_fn(support_v, support_f)
        cache["_ravafit_target_support_triangles"] = support
    support = np.asarray(support, dtype=np.float64)
    if collision.ndim != 3 or collision.shape[1:] != (3, 3) or not len(collision):
        return None
    if support.ndim != 3 or support.shape[1:] != (3, 3) or not len(support):
        return None
    return collision, support


def _dense_penetration_report(prod: Any, source: Any, positions: dict[str, np.ndarray], target_triangles: np.ndarray, margin: float) -> dict[str, Any]:
    occupancy = getattr(prod, "_nearest_literal_occupancy", None)
    if not callable(occupancy):
        return {"enabled": False, "reason": "literal occupancy evaluator unavailable", "penetrating_samples": 0}

    # Vertices/centroids are not enough: a narrow target-body ridge can pass through
    # the interior of a garment triangle while both remain clear. Sample a quarter-grid
    # over every final face and include the centroid as an independent witness.
    bary = np.asarray(
        [(i / 4.0, j / 4.0, (4 - i - j) / 4.0) for i in range(5) for j in range(5 - i)]
        + [(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    total_bad = 0
    worst = None
    threshold = float(margin) - .00003

    for name in sorted(positions):
        data = source.data(name)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        vertices = np.asarray(positions[name], dtype=np.float64)
        if not len(faces) or vertices.ndim != 2 or vertices.shape[1:] != (3,):
            continue
        samples = np.einsum("bk,fkj->fbj", bary, vertices[faces]).reshape(-1, 3)
        _, _, signed, _, _ = occupancy(samples, target_triangles, k=48, exact_band=float(margin) + .0020)
        signed = np.asarray(signed, dtype=np.float64)
        bad = signed < threshold
        count = int(np.count_nonzero(bad))
        minimum = float(np.min(signed)) if len(signed) else None
        if minimum is not None:
            worst = minimum if worst is None else min(worst, minimum)
        total_bad += count
        rows.append({
            "mesh": name,
            "faces": int(len(faces)),
            "samples": int(len(samples)),
            "penetrating_samples": count,
            "minimum_signed_mm": minimum * 1000.0 if minimum is not None else None,
        })

    return {
        "enabled": True,
        "margin_mm": float(margin * 1000.0),
        "penetrating_samples": int(total_bad),
        "minimum_signed_mm": worst * 1000.0 if worst is not None else None,
        "meshes": rows,
        "policy": "dense face-interior samples must remain outside the complete selected target-body collision surface",
    }


def install_dense_final_target_occupancy(prod: Any) -> None:
    """Make dense literal-target occupancy the last geometry authority of the coupled path."""
    if getattr(prod, "_ravafit_dense_final_target_occupancy_installed", False):
        return
    original = getattr(prod, "_finalize_modded_coupled_solution", None)
    clear = getattr(prod, "_final_target_body_clearance", None)
    if not callable(original) or not callable(clear):
        return

    def guarded(source, cache, positions, skinning, records, contexts, local_affine_quality_rms_mm):
        candidate, skinning_out, records_out, stats = original(
            source, cache, positions, skinning, records, contexts, local_affine_quality_rms_mm
        )
        surfaces = _target_surfaces(prod, cache)
        if surfaces is None:
            raise ValueError("Final target-body occupancy validation cannot resolve the selected target collision/support surfaces.")
        target_collision, target_support = surfaces

        # Run after the normal coupled finalizer has completed seams/layers/unilateral
        # reconstruction. The existing final clearance routine is seam-aware, so target
        # occupancy can be final authority without reopening a proven shared boundary.
        corrected, clearance_report = clear(
            source,
            candidate,
            target_collision,
            target_support,
            margin_m=.00035,
            maximum_vertex_move_m=.00800,
            maximum_step_m=.00125,
            max_passes=10,
        )
        before = {name: np.asarray(candidate[name], dtype=np.float64) for name in candidate}
        candidate = {name: np.asarray(value, dtype=np.float64) for name, value in corrected.items()}
        changed = [
            name for name in candidate
            if name in before and np.any(np.linalg.norm(candidate[name] - before[name], axis=1) > 1e-8)
        ]
        for name in changed:
            records_out.setdefault(name, {})["dense_final_target_occupancy_applied"] = True

        validation = _dense_penetration_report(prod, source, candidate, target_collision, margin=.00012)
        if not validation.get("enabled", False):
            raise ValueError(f"Final target-body occupancy validation unavailable: {validation.get('reason', 'unknown reason')}")
        if int(validation.get("penetrating_samples", 0)) > 0:
            worst = validation.get("minimum_signed_mm")
            raise ValueError(
                "RavaFit refused to emit a model with target body still penetrating garment face interiors: "
                f"{int(validation['penetrating_samples'])} dense sample(s), worst signed clearance {worst} mm."
            )

        merged_stats = dict(stats or {})
        merged_stats["dense_final_target_occupancy"] = {
            "changed_meshes": changed,
            "changed_mesh_count": int(len(changed)),
            "clearance": clearance_report,
            "validation": validation,
        }
        return candidate, skinning_out, records_out, merged_stats

    prod._finalize_modded_coupled_solution = guarded
    prod._ravafit_dense_final_target_occupancy_installed = True
