from __future__ import annotations

from typing import Any
import numpy as np

import core_fit_authority
import final_occupancy_guard
from source_standoff_authority import authored_clearance_floor

_STRICT_MARGIN_M = .00035
_STANDOFF_FLOOR_M = .00055
_STANDOFF_PAD_M = .00015


def _triangles(prod: Any, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = getattr(prod, "_triangles_from_surface", None)
    if callable(fn):
        return np.asarray(fn(vertices, faces), dtype=np.float64)
    return np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]


def prepare_strict_cache(prod: Any, cache: dict[str, Any]) -> dict[str, Any]:
    """Make strict B14 consume macro support while preserving literal target occupancy.

    The historical lane owns the real Duskwing RBODY->RBODY solve.  Its shaping surface must
    therefore be the low-frequency source->target body field, not literal target anatomy.  The
    untouched target body is kept separately and remains the only collision/occupancy authority.
    """
    core_report = core_fit_authority.prepare_cache(prod, cache)
    if not bool(core_report.get("enabled", False)):
        raise ValueError(f"Strict B14 core-fit authority could not prepare the body cache: {core_report.get('reason', 'unknown reason')}")

    X = np.asarray(cache.get("X", []), dtype=np.float64)
    Y = np.asarray(cache.get("Y", []), dtype=np.float64)
    source_v = np.asarray(cache.get("_ravafit_strict_source_surface_V", []), dtype=np.float64)
    source_f = np.asarray(cache.get("_ravafit_strict_source_surface_F", []), dtype=np.int64)
    literal_target_v = np.asarray(cache.get("_ravafit_strict_target_surface_V", []), dtype=np.float64)
    literal_target_f = np.asarray(cache.get("_ravafit_strict_target_surface_F", []), dtype=np.int64)

    valid_source = source_v.shape == X.shape and source_f.ndim == 2 and source_f.shape[1:] == (3,) and len(source_f)
    valid_target = literal_target_v.ndim == 2 and literal_target_v.shape[1:] == (3,) and literal_target_f.ndim == 2 and literal_target_f.shape[1:] == (3,) and len(literal_target_f)
    if not valid_source or Y.shape != X.shape:
        raise ValueError("Strict B14 core-fit authority has no source-topology macro surface.")
    if not valid_target:
        raise ValueError("Strict B14 core-fit authority has no literal target collision surface.")

    # Freeze untouched literal target occupancy before replacing strict B14's shaping surface.
    cache["_ravafit_literal_strict_target_surface_V"] = literal_target_v.copy()
    cache["_ravafit_literal_strict_target_surface_F"] = literal_target_f.copy()
    cache["_ravafit_target_collision_triangles"] = _triangles(prod, literal_target_v, literal_target_f)

    # Strict B14 must shape against the macro correspondence on SOURCE topology.  This prevents a
    # target crotch cleft, nipple, groove, rib or other local relief from becoming garment geometry.
    cache["_ravafit_strict_target_surface_V"] = Y.copy()
    cache["_ravafit_strict_target_surface_F"] = source_f.copy()
    cache["source_support_V"] = source_v.copy()
    cache["source_support_F"] = source_f.copy()
    cache["target_support_V"] = Y.copy()
    cache["target_support_F"] = source_f.copy()

    for key in (
        "_ravafit_local_affines", "_ravafit_source_support_triangles", "_ravafit_target_support_triangles",
        "_ravafit_strict_source_surface_triangles", "_ravafit_strict_target_surface_triangles",
        "_ravafit_garment_support_proxy", "_ravafit_structural_source_support_triangles",
        "_ravafit_structural_target_support_triangles",
    ):
        cache.pop(key, None)

    displacement = np.linalg.norm(Y - X, axis=1)
    report = {
        "enabled": True,
        "strict_lane": True,
        "core": core_report,
        "macro_vertices": int(len(Y)),
        "literal_collision_vertices": int(len(literal_target_v)),
        "macro_displacement_p50_mm": float(np.percentile(displacement, 50) * 1000.0),
        "macro_displacement_p95_mm": float(np.percentile(displacement, 95) * 1000.0),
        "policy": "strict B14 shapes against macro body correspondence on source topology; untouched target anatomy is occupancy-only",
    }
    cache["_ravafit_strict_core_fit_report"] = report
    return report


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _shell_raw_components(prod: Any, data: dict[str, Any], source_triangles: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    weld = getattr(prod, "weld_mesh", None)
    infer = getattr(prod, "infer_shell_behavior", None)
    if not callable(weld) or not callable(infer):
        return np.zeros(len(data.get("V", [])), dtype=np.int64), {0: "shell"}
    w = weld(data)
    _, _, labels, classes, _ = infer(w, source_triangles)
    raw_to_weld = np.asarray(w.get("raw_to_weld", []), dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(raw_to_weld) != len(data.get("V", [])) or not len(labels):
        return np.zeros(len(data.get("V", [])), dtype=np.int64), {0: "shell"}
    return labels[raw_to_weld], {int(k): str(v) for k, v in classes.items()}


def enforce_source_standoff(prod: Any, source: Any, cache: dict[str, Any], positions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Restore source-authored cloth/body spacing against strict B14's macro target frame."""
    nearest = getattr(prod, "_b14_nearest_surface", None)
    safe = getattr(prod, "_coupled_topology_safe_alpha", None)
    if not callable(nearest):
        raise ValueError("Strict B14 standoff authority requires nearest-surface support.")

    source_v = np.asarray(cache.get("_ravafit_strict_source_surface_V", []), dtype=np.float64)
    source_f = np.asarray(cache.get("_ravafit_strict_source_surface_F", []), dtype=np.int64)
    target_v = np.asarray(cache.get("_ravafit_strict_target_surface_V", []), dtype=np.float64)
    target_f = np.asarray(cache.get("_ravafit_strict_target_surface_F", []), dtype=np.int64)
    if not len(source_v) or not len(source_f) or not len(target_v) or not len(target_f):
        raise ValueError("Strict B14 standoff authority could not resolve macro source/target support surfaces.")
    source_tri = _triangles(prod, source_v, source_f)
    target_tri = _triangles(prod, target_v, target_f)

    out = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    reports: list[dict[str, Any]] = []
    total_moved = 0
    maximum_move = 0.0
    maximum_shortfall = 0.0

    for name in sorted(out):
        data = source.data(name)
        source_vertices = np.asarray(data.get("V", []), dtype=np.float64)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        before = out[name].copy()
        if source_vertices.shape != before.shape or not len(faces):
            continue
        raw_components, classes = _shell_raw_components(prod, data, source_tri)
        proposal = before.copy()
        component_reports = []

        for component in sorted(set(int(value) for value in raw_components.tolist())):
            if str(classes.get(component, "shell")).casefold() != "shell":
                continue
            ids = np.flatnonzero(raw_components == component)
            if len(ids) < 12:
                continue

            _, _, source_signed, source_distance, _ = nearest(source_vertices[ids], source_tri, k=32)
            source_distance = np.asarray(source_distance, dtype=np.float64)
            source_signed = np.asarray(source_signed, dtype=np.float64)
            finite = source_distance[np.isfinite(source_distance)]
            if not len(finite):
                continue
            median = float(np.median(finite))
            if median > .030:
                component_reports.append({"component": component, "vertices": int(len(ids)), "skipped": "source component is not body-supported", "source_clearance_median_mm": median * 1000.0})
                continue

            robust_cap = float(np.clip(np.percentile(finite, 97) + .0010, .0020, .0300))
            desired = authored_clearance_floor(source_distance, _STANDOFF_FLOOR_M, robust_cap)
            desired = np.minimum(desired + _STANDOFF_PAD_M, robust_cap)

            _, target_normals, target_signed, _, _ = nearest(before[ids], target_tri, k=32)
            target_normals = _normalise_rows(target_normals)
            target_signed = np.asarray(target_signed, dtype=np.float64)
            # Only source vertices that were actually outside/at the source body participate in the
            # standoff floor. Literal target occupancy still handles any inside/outside ambiguity.
            source_valid = source_signed >= -1e-5
            deficit = np.where(source_valid, np.maximum(desired - target_signed, 0.0), 0.0)
            step = np.minimum(deficit, .015)[:, None] * target_normals
            proposal[ids] += step

            component_reports.append({
                "component": component,
                "vertices": int(len(ids)),
                "source_clearance_p05_mm": float(np.percentile(source_distance, 5) * 1000.0),
                "source_clearance_median_mm": median * 1000.0,
                "desired_clearance_p05_mm": float(np.percentile(desired, 5) * 1000.0),
                "target_clearance_before_p05_mm": float(np.percentile(target_signed, 5) * 1000.0),
                "required_move_p95_mm": float(np.percentile(np.linalg.norm(step, axis=1), 95) * 1000.0),
            })
            maximum_shortfall = max(maximum_shortfall, float(np.max(deficit, initial=0.0)))

        topology = {"accepted_alpha": 1.0}
        candidate = proposal
        if callable(safe) and np.any(np.linalg.norm(proposal - before, axis=1) > 1e-10):
            candidate, alpha, _, _ = safe(source_vertices, before, proposal, faces)
            candidate = np.asarray(candidate, dtype=np.float64)
            topology = {"accepted_alpha": float(alpha)}
        out[name] = candidate
        movement = np.linalg.norm(candidate - before, axis=1)
        moved = int(np.count_nonzero(movement > 1e-8))
        total_moved += moved
        maximum_move = max(maximum_move, float(np.max(movement, initial=0.0)))
        reports.append({
            "mesh": name,
            "moved_vertices": moved,
            "move_p95_mm": float(np.percentile(movement, 95) * 1000.0) if len(movement) else 0.0,
            "move_max_mm": float(np.max(movement, initial=0.0) * 1000.0),
            "topology": topology,
            "components": component_reports,
        })

    return out, {
        "enabled": True,
        "moved_vertices": int(total_moved),
        "maximum_move_mm": float(maximum_move * 1000.0),
        "maximum_requested_shortfall_mm": float(maximum_shortfall * 1000.0),
        "meshes": reports,
        "policy": "source-authored body spacing is a minimum macro-frame standoff; target anatomy may only force additional outward clearance",
    }


def finalize_strict_solution(prod: Any, source: Any, cache: dict[str, Any], positions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Solve source standoff, source seams and literal target occupancy as one fixed point."""
    preserve_seams = getattr(prod, "_preserve_final_source_shared_seams", None)
    clear = getattr(prod, "_final_target_body_clearance", None)
    collision_fn = getattr(prod, "_target_fit_collision_triangles", None)
    if not callable(clear) or not callable(collision_fn):
        raise ValueError("Strict B14 final authority requires target clearance helpers.")

    target_collision = np.asarray(collision_fn(cache), dtype=np.float64)
    target_support = _triangles(
        prod,
        np.asarray(cache["_ravafit_strict_target_surface_V"], dtype=np.float64),
        np.asarray(cache["_ravafit_strict_target_surface_F"], dtype=np.int64),
    )
    candidate = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    passes: list[dict[str, Any]] = []
    converged = False

    for pass_index in range(12):
        before = {name: value.copy() for name, value in candidate.items()}
        candidate, standoff = enforce_source_standoff(prod, source, cache, candidate)
        if callable(preserve_seams):
            candidate, seams = preserve_seams(source, candidate)
            candidate = {name: np.asarray(value, dtype=np.float64) for name, value in candidate.items()}
        else:
            seams = {"enabled": False, "reason": "source shared-seam synchronizer unavailable"}

        validation = final_occupancy_guard._dense_penetration_report(prod, source, candidate, target_collision, _STRICT_MARGIN_M)
        if int(validation.get("penetrating_samples", 0)):
            candidate, occupancy = final_occupancy_guard._repair_until_dense_clear(
                prod, source, candidate, target_collision, target_support, clear, margin=_STRICT_MARGIN_M,
            )
        else:
            occupancy = {"converged": True, "validation": validation, "passes": []}

        if callable(preserve_seams):
            candidate, final_seams = preserve_seams(source, candidate)
            candidate = {name: np.asarray(value, dtype=np.float64) for name, value in candidate.items()}
        else:
            final_seams = seams
        final_validation = final_occupancy_guard._dense_penetration_report(prod, source, candidate, target_collision, _STRICT_MARGIN_M)
        max_delta = max((float(np.max(np.linalg.norm(candidate[name] - before[name], axis=1), initial=0.0)) for name in candidate), default=0.0)
        passes.append({
            "pass": int(pass_index + 1),
            "standoff": standoff,
            "seams": seams,
            "occupancy": occupancy,
            "final_seams": final_seams,
            "final_validation": final_validation,
            "max_iteration_move_mm": float(max_delta * 1000.0),
        })
        if int(final_validation.get("penetrating_samples", 0)) == 0 and max_delta <= .00001:
            converged = True
            break

    # A final dry-run of the standoff projection proves a seam repair has not silently collapsed the
    # authored body gap again.  We refuse to publish rather than choose between a split seam and clip.
    would_be, standoff_check = enforce_source_standoff(prod, source, cache, candidate)
    standoff_residual = max((float(np.max(np.linalg.norm(would_be[name] - candidate[name], axis=1), initial=0.0)) for name in candidate), default=0.0)
    final_validation = final_occupancy_guard._dense_penetration_report(prod, source, candidate, target_collision, _STRICT_MARGIN_M)
    if int(final_validation.get("penetrating_samples", 0)):
        raise ValueError("Strict B14 final output still intersects the selected target body; RavaFit refused to publish it.")
    if standoff_residual > .00005:
        raise ValueError(
            f"Strict B14 could not satisfy source-authored standoff and source seams simultaneously; residual standoff correction {standoff_residual * 1000.0:.3f} mm."
        )

    return candidate, {
        "enabled": True,
        "converged": bool(converged),
        "passes": passes,
        "standoff_residual_mm": float(standoff_residual * 1000.0),
        "standoff_check": standoff_check,
        "final_validation": final_validation,
        "minimum_dense_clearance_mm": float(_STRICT_MARGIN_M * 1000.0),
        "policy": "strict B14 output must simultaneously preserve source-authored standoff, source-proven seams and literal target occupancy",
    }
