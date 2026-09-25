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


def _component_local_faces(faces: np.ndarray, ids: np.ndarray, vertex_count: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    lookup = np.full(int(vertex_count), -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids), dtype=np.int64)
    mask = np.all(lookup[faces] >= 0, axis=1)
    return lookup[faces[mask]]


def enforce_source_standoff(prod: Any, source: Any, cache: dict[str, Any], positions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Restore only source-proven body-supported garment spacing against the macro target frame.

    Far straps, collars, skirts and other non-body-supported vertices must never inherit their raw
    nearest-body distance as a clearance target. Close cloth does retain its authored spacing, and
    the correction is applied in small topology-safe component-local steps so one difficult vertex
    cannot veto an otherwise valid garment-wide standoff correction.
    """
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
    maximum_requested = 0.0
    maximum_remaining = 0.0

    for name in sorted(out):
        data = source.data(name)
        source_vertices = np.asarray(data.get("V", []), dtype=np.float64)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        before_mesh = out[name].copy()
        if source_vertices.shape != before_mesh.shape or not len(faces):
            continue

        raw_components, classes = _shell_raw_components(prod, data, source_tri)
        candidate = before_mesh.copy()
        component_reports: list[dict[str, Any]] = []

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
            # One connected shell can contain close cups plus a collar tens of millimetres away.
            # Classify support per vertex; only source-proven close body-supported vertices inherit
            # their authored body spacing.
            support_limit = float(np.clip(max(.0040, median * 4.0), .0040, .0120))
            source_valid = np.isfinite(source_distance) & (source_signed >= -1e-5) & (source_distance <= support_limit)
            supported = int(np.count_nonzero(source_valid))
            minimum_supported = max(12, int(np.ceil(len(ids) * .02)))
            if supported < minimum_supported:
                component_reports.append({
                    "component": component,
                    "vertices": int(len(ids)),
                    "supported_vertices": supported,
                    "skipped": "too little source-proven close body support",
                    "source_clearance_median_mm": median * 1000.0,
                    "support_limit_mm": support_limit * 1000.0,
                })
                continue

            valid_distance = source_distance[source_valid]
            robust_cap = float(np.clip(np.percentile(valid_distance, 99) + .00050, .0010, support_limit))
            desired = authored_clearance_floor(source_distance, _STANDOFF_FLOOR_M, robust_cap)
            desired = np.minimum(desired + _STANDOFF_PAD_M, robust_cap)

            _, target_normals, target_signed, _, _ = nearest(candidate[ids], target_tri, k=32)
            target_normals = _normalise_rows(target_normals)
            target_signed = np.asarray(target_signed, dtype=np.float64)
            deficit = np.where(source_valid, np.maximum(desired - target_signed, 0.0), 0.0)
            maximum_requested = max(maximum_requested, float(np.max(deficit, initial=0.0)))

            # Small bounded steps converge reliably and keep the topology guard local to the affected
            # authored component instead of letting one difficult vertex veto the whole mesh.
            requested_step = np.minimum(deficit, .00150)[:, None] * target_normals
            local_before = candidate[ids].copy()
            local_proposal = local_before + requested_step
            topology = {"accepted_alpha": 1.0}
            local_faces = _component_local_faces(faces, ids, len(source_vertices))
            if callable(safe) and len(local_faces) and np.any(np.linalg.norm(requested_step, axis=1) > 1e-10):
                local_candidate, alpha, _, _ = safe(source_vertices[ids], local_before, local_proposal, local_faces)
                local_candidate = np.asarray(local_candidate, dtype=np.float64)
                topology = {"accepted_alpha": float(alpha)}
            else:
                local_candidate = local_proposal
            candidate[ids] = local_candidate

            _, _, after_signed, _, _ = nearest(candidate[ids], target_tri, k=32)
            after_signed = np.asarray(after_signed, dtype=np.float64)
            remaining = np.where(source_valid, np.maximum(desired - after_signed, 0.0), 0.0)
            maximum_remaining = max(maximum_remaining, float(np.max(remaining, initial=0.0)))
            moved = np.linalg.norm(local_candidate - local_before, axis=1)
            component_reports.append({
                "component": component,
                "vertices": int(len(ids)),
                "supported_vertices": supported,
                "support_limit_mm": support_limit * 1000.0,
                "source_clearance_p05_mm": float(np.percentile(valid_distance, 5) * 1000.0),
                "source_clearance_median_mm": float(np.median(valid_distance) * 1000.0),
                "desired_clearance_p05_mm": float(np.percentile(desired[source_valid], 5) * 1000.0),
                "target_clearance_before_p05_mm": float(np.percentile(target_signed[source_valid], 5) * 1000.0),
                "requested_shortfall_p95_mm": float(np.percentile(deficit[source_valid], 95) * 1000.0),
                "remaining_shortfall_p95_mm": float(np.percentile(remaining[source_valid], 95) * 1000.0),
                "remaining_shortfall_max_mm": float(np.max(remaining, initial=0.0) * 1000.0),
                "move_p95_mm": float(np.percentile(moved, 95) * 1000.0),
                "topology": topology,
            })

        out[name] = candidate
        movement = np.linalg.norm(candidate - before_mesh, axis=1)
        moved = int(np.count_nonzero(movement > 1e-8))
        total_moved += moved
        maximum_move = max(maximum_move, float(np.max(movement, initial=0.0)))
        reports.append({
            "mesh": name,
            "moved_vertices": moved,
            "move_p95_mm": float(np.percentile(movement, 95) * 1000.0) if len(movement) else 0.0,
            "move_max_mm": float(np.max(movement, initial=0.0) * 1000.0),
            "components": component_reports,
        })

    return out, {
        "enabled": True,
        "moved_vertices": int(total_moved),
        "maximum_move_mm": float(maximum_move * 1000.0),
        "maximum_requested_shortfall_mm": float(maximum_requested * 1000.0),
        "maximum_remaining_shortfall_mm": float(maximum_remaining * 1000.0),
        "meshes": reports,
        "policy": "only source-proven close body-supported cloth keeps authored macro-frame standoff; far straps/collars/skirt regions are excluded; corrections are small component-local topology-safe steps",
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
        if int(final_validation.get("penetrating_samples", 0)) == 0 and float(standoff.get("maximum_remaining_shortfall_mm", 0.0)) <= .05 and max_delta <= .00001:
            converged = True
            break

    # A final dry-run of the standoff projection proves a seam repair has not silently collapsed the
    # authored body gap again.  We refuse to publish rather than choose between a split seam and clip.
    would_be, standoff_check = enforce_source_standoff(prod, source, cache, candidate)
    projected_residual = max((float(np.max(np.linalg.norm(would_be[name] - candidate[name], axis=1), initial=0.0)) for name in candidate), default=0.0)
    measured_residual = float(standoff_check.get("maximum_remaining_shortfall_mm", 0.0)) / 1000.0
    standoff_residual = max(projected_residual, measured_residual)
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
