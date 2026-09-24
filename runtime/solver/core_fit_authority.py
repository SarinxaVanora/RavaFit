from __future__ import annotations

from typing import Any
import numpy as np

from macro_body_authority import macro_body_correspondence
from source_standoff_authority import authored_clearance_floor


def _normalise_weights(values: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = out.sum(axis=1, keepdims=True)
    good = total[:, 0] > 1e-12
    out[good] /= total[good]
    return out


def _macro_from_combined_cache(cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    X = np.asarray(cache.get("X", []), dtype=np.float64)
    Y = np.asarray(cache.get("Y", []), dtype=np.float64)
    if X.ndim != 2 or X.shape[1:] != (3,) or X.shape != Y.shape or len(X) < 3:
        return None

    for key_v, key_f in (
        ("_ravafit_strict_source_surface_V", "_ravafit_strict_source_surface_F"),
        ("source_surface_V", "source_surface_F"),
    ):
        V = np.asarray(cache.get(key_v, []), dtype=np.float64)
        F = np.asarray(cache.get(key_f, []), dtype=np.int64)
        if V.shape == X.shape and F.ndim == 2 and F.shape[1:] == (3,) and len(F) and int(np.max(F)) < len(X):
            if float(np.max(np.linalg.norm(V - X, axis=1), initial=0.0)) <= 2e-5:
                macro, normals, report = macro_body_correspondence(X, Y, F, radius_m=.032)
                return macro, normals, {"mode": key_f, **report}

    pairs = list(cache.get("slot_pairs") or [])
    if not pairs:
        return None
    macro_rows = []
    normal_rows = []
    reports = []
    for pair in pairs:
        PX = np.asarray(pair.get("X", []), dtype=np.float64)
        PY = np.asarray(pair.get("Y", []), dtype=np.float64)
        PV = np.asarray(pair.get("source_literal_V", []), dtype=np.float64)
        PF = np.asarray(pair.get("source_literal_F", []), dtype=np.int64)
        if PX.shape != PY.shape or PX.ndim != 2 or PX.shape[1:] != (3,) or len(PX) < 3:
            return None
        if PV.shape != PX.shape or PF.ndim != 2 or PF.shape[1:] != (3,) or not len(PF) or int(np.max(PF)) >= len(PX):
            return None
        if float(np.max(np.linalg.norm(PV - PX, axis=1), initial=0.0)) > 2e-5:
            return None
        macro, normals, report = macro_body_correspondence(PX, PY, PF, radius_m=.032)
        macro_rows.append(macro)
        normal_rows.append(normals)
        reports.append({"slot": str(pair.get("slot") or "Body"), **report})
    macro = np.vstack(macro_rows)
    normals = np.vstack(normal_rows)
    if macro.shape != X.shape:
        return None
    return macro, normals, {"enabled": True, "mode": "slot_pairs", "slots": reports}


def _macro_support_surface(cache: dict[str, Any]) -> dict[str, Any]:
    source = np.asarray(cache.get("source_support_V", []), dtype=np.float64)
    target = np.asarray(cache.get("target_support_V", []), dtype=np.float64)
    source_faces = np.asarray(cache.get("source_support_F", []), dtype=np.int64)
    target_faces = np.asarray(cache.get("target_support_F", []), dtype=np.int64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 3:
        return {"enabled": False, "reason": "paired support surfaces do not share a vertex domain"}
    if source_faces.shape != target_faces.shape or source_faces.ndim != 2 or source_faces.shape[1:] != (3,) or not len(source_faces) or not np.array_equal(source_faces, target_faces):
        return {"enabled": False, "reason": "paired support surfaces do not share topology"}
    macro, _, report = macro_body_correspondence(source, target, source_faces, radius_m=.032)
    cache["_ravafit_literal_target_support_V"] = target.copy()
    cache["target_support_V"] = macro
    return {"enabled": True, **report}


def _recalibrate_target_weight_correspondence(prod: Any, cache: dict[str, Any], literal_correspondence: np.ndarray) -> dict[str, Any]:
    sampler = getattr(prod, "_target_skin_weights_at_points", None)
    BW = np.asarray(cache.get("BW", []), dtype=np.float64)
    names = list(cache.get("names") or [])
    existing = np.asarray(cache.get("target_correspondence_W", []), dtype=np.float64)
    if not callable(sampler) or BW.ndim != 2 or BW.shape[0] != len(literal_correspondence) or BW.shape[1] != len(names):
        return {"enabled": False, "reason": "paired target skin sampler unavailable"}
    try:
        sampled, distance = sampler(literal_correspondence, cache, source_weights=BW, source_joint_names=names)
        sampled = _normalise_weights(sampled)
    except Exception as ex:
        return {"enabled": False, "reason": f"paired target skin sampling failed: {type(ex).__name__}: {ex}"}
    if sampled.shape != BW.shape or not np.isfinite(sampled).all():
        return {"enabled": False, "reason": "paired target skin sampler returned incompatible weights"}

    cache["target_correspondence_W"] = sampled
    source_delta = np.abs(sampled - BW).sum(axis=1)
    old_delta = np.abs(existing - BW).sum(axis=1) if existing.shape == BW.shape else np.zeros(len(BW), dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    return {
        "enabled": True,
        "policy": "target body skin field is sampled only at paired source->target body correspondence points; garment proximity never chooses weights",
        "vertices": int(len(BW)),
        "old_body_delta_l1_p95": float(np.percentile(old_delta, 95)) if len(old_delta) else 0.0,
        "paired_body_delta_l1_p50": float(np.percentile(source_delta, 50)) if len(source_delta) else 0.0,
        "paired_body_delta_l1_p95": float(np.percentile(source_delta, 95)) if len(source_delta) else 0.0,
        "paired_body_delta_l1_max": float(np.max(source_delta, initial=0.0)),
        "target_surface_sample_distance_p95_mm": float(np.percentile(distance, 95) * 1000.0) if len(distance) else 0.0,
    }


def prepare_cache(prod: Any, cache: dict[str, Any]) -> dict[str, Any]:
    """Install the actual fitting hierarchy into an already-built production body cache."""
    if cache.get("_ravafit_core_fit_cache_revision") == 2:
        return dict(cache.get("_ravafit_core_fit_report") or {})

    X = np.asarray(cache.get("X", []), dtype=np.float64)
    literal = np.asarray(cache.get("Y", []), dtype=np.float64).copy()
    report: dict[str, Any] = {"enabled": False, "revision": 2}
    macro_result = _macro_from_combined_cache(cache)
    if macro_result is None:
        report["reason"] = "body cache has no source-topology correspondence suitable for macro filtering"
    else:
        macro, normals, macro_report = macro_result
        if macro.shape == X.shape and normals.shape == X.shape:
            cache["_ravafit_literal_correspondence_Y"] = literal
            cache["Y"] = macro
            cache["NT"] = normals
            correction = np.asarray(cache.get("target_relief_C", []), dtype=np.float64)
            if correction.shape == X.shape:
                cache["target_relief_C"] = np.zeros_like(correction)
            report.update({"enabled": True, "macro_body": macro_report})

    report["macro_support_surface"] = _macro_support_surface(cache)
    skin_report = _recalibrate_target_weight_correspondence(prod, cache, literal) if literal.shape == X.shape else {"enabled": False, "reason": "literal paired body positions unavailable"}
    report["target_skin_field"] = skin_report
    cache["_ravafit_core_fit_cache_revision"] = 2
    cache["_ravafit_core_fit_report"] = report
    for key in (
        "_ravafit_local_affines", "_ravafit_source_support_triangles", "_ravafit_target_support_triangles",
        "_ravafit_target_collision_triangles", "_ravafit_garment_support_proxy",
        "_ravafit_structural_source_support_triangles", "_ravafit_structural_target_support_triangles",
    ):
        cache.pop(key, None)
    return report


def _component_local_faces(faces: np.ndarray, ids: np.ndarray, vertex_count: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    lookup = np.full(int(vertex_count), -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids), dtype=np.int64)
    mask = np.all(lookup[faces] >= 0, axis=1)
    return lookup[faces[mask]]


def install_runtime_authority(prod: Any) -> None:
    """Preserve authored standoff on the functions the production solver actually calls."""
    if getattr(prod, "_ravafit_core_fit_runtime_authority_installed", False):
        return

    relief_original = getattr(prod, "_apply_target_relief_correction", None)
    if callable(relief_original):
        def relief_guard(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
            if bool((cache.get("_ravafit_core_fit_report") or {}).get("enabled")):
                return np.asarray(mapped, dtype=np.float64), {
                    "enabled": False,
                    "reason": "target local relief is collision-only; macro body correspondence already owns fitting",
                    "core_fit_authority": True,
                }
            return relief_original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features)
        prod._apply_target_relief_correction = relief_guard

    support_original = getattr(prod, "_support_frame_clearance_guard", None)
    nearest = getattr(prod, "_b14_nearest_surface", None)
    if callable(support_original) and callable(nearest):
        def support_guard(w, mapped, blend, blend_ids, cache, labels, classes, source_body_triangles, minimum_clearance=.00035):
            base, base_report = support_original(w, mapped, blend, blend_ids, cache, labels, classes, source_body_triangles, minimum_clearance=minimum_clearance)
            source = np.asarray(w["V"], dtype=np.float64)
            faces = np.asarray(w["F"], dtype=np.int64)
            base_array = np.asarray(base, dtype=np.float64)
            out = base_array.copy()
            blend_arr = np.asarray(blend, dtype=np.float64)
            blend_ids_arr = np.asarray(blend_ids, dtype=np.int64)
            labels_arr = np.asarray(labels, dtype=np.int64)
            Y = np.asarray(cache.get("Y", []), dtype=np.float64)
            NT = np.asarray(cache.get("NT", []), dtype=np.float64)
            if source.shape != out.shape or Y.ndim != 2 or NT.shape != Y.shape or blend_ids_arr.ndim != 2:
                return out, {"base": base_report, "source_standoff": {"enabled": False, "reason": "incompatible support arrays"}}

            target_anchor = np.sum(Y[blend_ids_arr] * blend_arr[:, :, None], axis=1)
            target_normal = np.sum(NT[blend_ids_arr] * blend_arr[:, :, None], axis=1)
            target_normal /= np.maximum(np.linalg.norm(target_normal, axis=1, keepdims=True), 1e-12)
            reports = []
            adjusted = 0
            for component in sorted(set(int(x) for x in labels_arr.tolist())):
                if str(classes.get(component, "shell")).casefold() != "shell":
                    continue
                ids = np.flatnonzero(labels_arr == component)
                if len(ids) < 12:
                    continue
                _, _, _, source_distance, _ = nearest(source[ids], source_body_triangles, k=32)
                source_distance = np.asarray(source_distance, dtype=np.float64)
                finite = source_distance[np.isfinite(source_distance)]
                if not len(finite):
                    continue
                median = float(np.median(finite))
                if median > .030:
                    reports.append({"component": component, "vertices": int(len(ids)), "skipped": "loose/far component", "source_clearance_median_mm": median * 1000.0})
                    continue

                robust_cap = float(np.clip(np.percentile(finite, 97) + .0010, .0020, .0300))
                desired = authored_clearance_floor(source_distance, max(float(minimum_clearance), .00055), robust_cap)
                desired = np.minimum(desired + .00015, robust_cap)
                before_radial = np.einsum("ij,ij->i", out[ids] - target_anchor[ids], target_normal[ids])
                deficit = np.maximum(desired - before_radial, 0.0)
                topology = {"accepted_alpha": 1.0}
                if np.any(deficit > 1e-8):
                    step = np.minimum(deficit, .015)[:, None] * target_normal[ids]
                    local_faces = _component_local_faces(faces, ids, len(source))
                    proposal = out[ids] + step
                    safe_fn = getattr(prod, "_coupled_topology_safe_alpha", None)
                    if callable(safe_fn) and len(local_faces):
                        try:
                            safe, alpha, _, _ = safe_fn(source[ids], out[ids], proposal, local_faces)
                            proposal = np.asarray(safe, dtype=np.float64)
                            topology = {"accepted_alpha": float(alpha)}
                        except Exception as ex:
                            topology = {"accepted_alpha": 0.0, "error": f"{type(ex).__name__}: {ex}"}
                    out[ids] = proposal

                after_radial = np.einsum("ij,ij->i", out[ids] - target_anchor[ids], target_normal[ids])
                shortfall = desired - after_radial
                moved = np.linalg.norm(out[ids] - base_array[ids], axis=1)
                adjusted += int(np.count_nonzero(moved > 1e-8))
                reports.append({
                    "component": component,
                    "vertices": int(len(ids)),
                    "source_clearance_p05_mm": float(np.percentile(source_distance, 5) * 1000.0),
                    "source_clearance_median_mm": median * 1000.0,
                    "desired_clearance_p05_mm": float(np.percentile(desired, 5) * 1000.0),
                    "before_radial_p05_mm": float(np.percentile(before_radial, 5) * 1000.0),
                    "after_radial_p05_mm": float(np.percentile(after_radial, 5) * 1000.0),
                    "remaining_shortfall_p95_mm": float(max(0.0, np.percentile(shortfall, 95)) * 1000.0),
                    "moved_vertices": int(np.count_nonzero(moved > 1e-8)),
                    "move_p95_mm": float(np.percentile(moved, 95) * 1000.0),
                    "topology": topology,
                })
            return out, {
                "base": base_report,
                "source_standoff": {
                    "enabled": True,
                    "policy": "source-authored garment/body spacing is a minimum target-frame clearance; literal target may push outward but never shrink-wrap inward",
                    "adjusted_vertices": int(adjusted),
                    "components": reports,
                },
            }
        prod._support_frame_clearance_guard = support_guard

    prod._ravafit_core_fit_runtime_authority_installed = True
