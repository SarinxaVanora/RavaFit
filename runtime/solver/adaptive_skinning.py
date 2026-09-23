"""Evidence-driven garment skinning adaptation.

RavaFit's target is not merely a garment that occupies the target bind pose; it should
move as though it had been authored for that target body.

Authority is deliberately local:

* same paired body geometry + same paired body deformation field -> preserve the source
  garment skinning exactly;
* changed paired body deformation field -> carry the proven source->target body-weight
  delta into the garment's body-supported mass;
* materially changed paired body geometry under genuinely close cloth -> let the local
  target-body blend progressively become motion authority for that body-supported mass;
* garment/cloth/secondary influences remain exact structural authority;
* target body blends may use more body influences only when the source GLTF primitive
  actually has encoder capacity for them.

This is not nearest-target weight transfer.  Geometry change is measured on the paired
source/target body correspondence, never by sampling whichever target triangle happens
to be nearest after the garment has already moved.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

_CAPACITY_CACHE_KEY = "_ravafit_source_influence_capacities_v2"


def _public_delta_report(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {"enabled": False, "reason": "no body correspondence report"}
    out: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            out[key] = value
        elif isinstance(value, np.generic):
            out[key] = value.item()
    return out


def _smoothstep(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if hi <= lo:
        return (values >= hi).astype(np.float64)
    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def source_skin_capacity_key(positions: np.ndarray, joint_names: list[str] | tuple[str, ...]) -> str:
    """Stable identity for one untouched garment skin payload."""
    values = np.ascontiguousarray(np.asarray(positions, dtype="<f4"))
    h = hashlib.sha256()
    h.update(np.asarray(values.shape, dtype="<i8").tobytes())
    h.update(values.tobytes())
    h.update(b"\0JOINTS\0")
    h.update("\0".join(str(name) for name in joint_names).encode("utf-8"))
    return h.hexdigest()


def _primitive_influence_capacity(source_js: dict[str, Any], mesh_name: str) -> int | None:
    matches = [mesh for mesh in (source_js or {}).get("meshes", []) if str(mesh.get("name") or "") == str(mesh_name)]
    if len(matches) != 1:
        return None
    primitives = matches[0].get("primitives") or []
    if len(primitives) != 1:
        return None
    attrs = primitives[0].get("attributes") or {}
    pairs = 0
    for suffix in ("0", "1"):
        has_joints = f"JOINTS_{suffix}" in attrs
        has_weights = f"WEIGHTS_{suffix}" in attrs
        if has_joints != has_weights:
            raise ValueError(f"{mesh_name} has incomplete JOINTS_{suffix}/WEIGHTS_{suffix} source skinning attributes.")
        if has_joints:
            pairs += 1
    return 4 * pairs if pairs else None


def register_source_influence_capacities(cache: dict[str, Any], source: Any) -> dict[str, Any]:
    """Record the real 4/8 influence capacity of each source GLTF primitive."""
    registry = dict(cache.get(_CAPACITY_CACHE_KEY) or {})
    rows = []
    source_js = getattr(source, "js", None)
    mesh_names = list(source.mesh_names()) if callable(getattr(source, "mesh_names", None)) else []
    for name in mesh_names:
        data = source.data(name)
        capacity = _primitive_influence_capacity(source_js, name)
        if capacity not in (4, 8):
            rows.append({"mesh": str(name), "capacity": None, "reason": "source primitive capacity unavailable"})
            continue
        positions = np.asarray(data.get("V", []), dtype=np.float64)
        names = list(data.get("joint_names") or [])
        weights = np.asarray(data.get("W", []), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or weights.ndim != 2 or len(weights) != len(positions):
            rows.append({"mesh": str(name), "capacity": None, "reason": "source skin payload unavailable"})
            continue
        active = int(np.max(np.count_nonzero(weights > 1e-8, axis=1), initial=0))
        if active > capacity:
            raise ValueError(f"{name} has {active} active source influences but its GLTF primitive can encode only {capacity}.")
        key = source_skin_capacity_key(positions, names)
        previous = registry.get(key)
        if previous is not None and int(previous) != capacity:
            raise ValueError(f"Conflicting source influence capacities for garment skin {name}: {previous} vs {capacity}.")
        registry[key] = int(capacity)
        rows.append({"mesh": str(name), "capacity": int(capacity), "max_authored_active": active})
    cache[_CAPACITY_CACHE_KEY] = registry
    return {"registered": int(len(registry)), "meshes": rows}


def _source_capacity(weights: np.ndarray, source_positions: np.ndarray, joint_names: list[str], cache: dict[str, Any]) -> tuple[int, str]:
    """Resolve actual source capacity, with a conservative fallback for legacy/in-process paths."""
    weights = np.asarray(weights, dtype=np.float64)
    maximum = int(np.max(np.count_nonzero(weights > 1e-8, axis=1), initial=0)) if weights.ndim == 2 else 0
    try:
        key = source_skin_capacity_key(source_positions, joint_names)
        registered = (cache.get(_CAPACITY_CACHE_KEY) or {}).get(key)
        if registered is not None:
            capacity = int(registered)
            if capacity not in (4, 8):
                raise ValueError(f"Invalid registered GLTF influence capacity {capacity}.")
            if maximum > capacity:
                raise ValueError(f"Source skin uses {maximum} active influences but registered primitive capacity is {capacity}.")
            return capacity, "source_gltf_primitive"
    except ValueError:
        raise
    except Exception:
        pass
    return (8 if maximum > 4 else 4), "conservative_authored_usage_fallback"


def _pack_body_mass_to_budget(ideal_body: np.ndarray, source_body: np.ndarray, body_mass: np.ndarray,
                              non_body_active: np.ndarray, capacity: int, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the ideal target-body blend into slots left after exact garment influences."""
    ideal_body = np.asarray(ideal_body, dtype=np.float64)
    source_body = np.asarray(source_body, dtype=np.float64)
    body_mass = np.asarray(body_mass, dtype=np.float64)
    non_body_active = np.asarray(non_body_active, dtype=np.int64)
    active = np.asarray(active, dtype=bool)
    packed = source_body.copy()
    capacity_limited = np.zeros(len(packed), dtype=bool)

    for row in np.flatnonzero(active):
        available = max(0, int(capacity) - int(non_body_active[row]))
        source_slots = int(np.count_nonzero(source_body[row] > 1e-8))
        if available < source_slots:
            raise ValueError(
                f"Adaptive garment skinning has only {available} body influence slot(s) after preserving authored "
                f"non-body influences, but the source vertex already requires {source_slots}."
            )
        if available <= 0 or body_mass[row] <= 1e-12:
            continue
        values = np.maximum(ideal_body[row], 0.0)
        positive = np.flatnonzero(values > 1e-10)
        if len(positive) > available:
            capacity_limited[row] = True
            order = np.argsort(-values, kind="stable")
            keep = order[:available]
            reduced = np.zeros_like(values)
            reduced[keep] = values[keep]
            values = reduced
        total = float(values.sum())
        if total <= 1e-12:
            continue
        packed[row] = values * (float(body_mass[row]) / total)
    return packed, capacity_limited


def _paired_body_support_field(points: np.ndarray, garment_weights: np.ndarray, garment_joint_names: list[str], cache: dict[str, Any]) -> dict[str, Any] | None:
    """Sample paired source/target body geometry and target motion at source garment points.

    This deliberately follows the same source-side evidence shape as production's verified
    body-delta sampler.  The target is reached only through X->Y correspondence; target-space
    nearest-neighbour sampling is never used as reweight authority.
    """
    p = np.asarray(points, dtype=np.float64)
    wg = np.asarray(garment_weights, dtype=np.float64)
    x = np.asarray(cache.get("X", []), dtype=np.float64)
    y = np.asarray(cache.get("Y", []), dtype=np.float64)
    bw = np.asarray(cache.get("BW", []), dtype=np.float64)
    tw = np.asarray(cache.get("target_correspondence_W", []), dtype=np.float64)
    cache_names = list(cache.get("names") or [])
    garment_names = list(garment_joint_names)
    if (p.ndim != 2 or p.shape[1:] != (3,) or x.ndim != 2 or x.shape[1:] != (3,) or
            y.shape != x.shape or bw.ndim != 2 or tw.shape != bw.shape or len(bw) != len(x) or
            bw.shape[1] != len(cache_names) or wg.ndim != 2 or len(wg) != len(p) or
            wg.shape[1] != len(garment_names) or not len(x)):
        return None

    garment_index = {name: index for index, name in enumerate(garment_names)}
    common_cache = np.asarray([i for i, name in enumerate(cache_names) if name in garment_index], dtype=np.int64)
    if not len(common_cache):
        return None

    signature = np.zeros((len(p), len(cache_names)), dtype=np.float64)
    for ci in common_cache.tolist():
        signature[:, ci] = wg[:, garment_index[cache_names[ci]]]
    body_mass = signature.sum(axis=1)
    has_body = body_mass > 1e-8
    signature[has_body] /= body_mass[has_body, None]

    k = min(32, len(x))
    tree = cKDTree(x)
    distance, index = tree.query(p, k=k)
    if distance.ndim == 1:
        distance = distance[:, None]
        index = index[:, None]
    alignment = np.einsum("nk,nqk->nq", signature, bw[index], optimize=True)
    score = distance + .015 * (1.0 - alignment)
    side = np.sign(p[:, 0])[:, None]
    body_side = np.sign(x[index][:, :, 0])
    score += np.where((np.abs(p[:, 0, None]) > .020) & (side != body_side), .080, 0.0)
    take = min(12, score.shape[1])
    order = np.argpartition(score, take - 1, axis=1)[:, :take]
    selected = np.take_along_axis(index, order, axis=1)
    selected_score = np.take_along_axis(score, order, axis=1)
    selected_distance = np.take_along_axis(distance, order, axis=1)
    relative = selected_score - selected_score.min(axis=1, keepdims=True)
    blend = np.exp(-relative / .0035)
    blend /= np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)

    local_source_weights = np.sum(bw[selected] * blend[:, :, None], axis=1)
    local_target_weights = np.sum(tw[selected] * blend[:, :, None], axis=1)
    local_source_point = np.sum(x[selected] * blend[:, :, None], axis=1)
    local_target_point = np.sum(y[selected] * blend[:, :, None], axis=1)
    support_distance = np.sum(selected_distance * blend, axis=1)
    shape_change = np.linalg.norm(local_target_point - local_source_point, axis=1)
    shape_change[~has_body] = 0.0
    support_distance[~has_body] = np.inf
    return {
        "source_weights": local_source_weights,
        "target_weights": local_target_weights,
        "source_point": local_source_point,
        "target_point": local_target_point,
        "shape_change": shape_change,
        "support_distance": support_distance,
        "body_mass": body_mass,
        "body_supported_cache_columns": common_cache,
    }


def _geometry_motion_authority(field: dict[str, Any] | None, behavior: str, effective_behavior: str) -> np.ndarray:
    if field is None:
        return np.zeros(0, dtype=np.float64)
    shape_change = np.asarray(field["shape_change"], dtype=np.float64)
    support_distance = np.asarray(field["support_distance"], dtype=np.float64)
    behaviour = str(behavior or "").casefold()
    effective = str(effective_behavior or behaviour).casefold()
    close = {"constructed_close_shell", "body_following_flexible_layer"}
    if behaviour not in close and effective not in close:
        return np.zeros(len(shape_change), dtype=np.float64)

    # Below 0.20 mm the paired target is effectively the same body surface for skinning
    # purposes.  By 2.5 mm of real source->target body movement, close cloth should move
    # with target-body motion rather than retaining a source-body blend merely because the
    # two bodies happened to publish identical weight values at corresponding vertices.
    shape_gate = _smoothstep(shape_change, .00020, .00250)
    if behaviour == "body_following_flexible_layer" or effective == "body_following_flexible_layer":
        close_gate = 1.0 - _smoothstep(support_distance, .0050, .0160)
    else:
        close_gate = 1.0 - _smoothstep(support_distance, .0040, .0120)
    return np.clip(shape_gate * close_gate, 0.0, 1.0)


def install_adaptive_body_skinning(prod: Any) -> None:
    """Install correspondence-driven, target-authored motion adaptation."""
    if getattr(prod, "_ravafit_adaptive_body_skinning_installed", False):
        return
    original = getattr(prod, "_retarget_garment_skinning", None)
    delta_helper = getattr(prod, "_verified_body_skin_delta_at_points", None)
    if not callable(original) or not callable(delta_helper):
        return

    def adaptive(raw_positions, source_positions, source_weights, source_joint_names, cache, behavior, effective_behavior, labels, classes, raw_to_weld):
        preserved, base_stage = original(
            raw_positions, source_positions, source_weights, source_joint_names, cache,
            behavior, effective_behavior, labels, classes, raw_to_weld,
        )
        preserved = np.asarray(preserved, dtype=np.float64)
        source_positions_array = np.asarray(source_positions, dtype=np.float64)
        joint_names = list(source_joint_names)
        capacity, capacity_authority = _source_capacity(preserved, source_positions_array, joint_names, cache)
        cache_names = list(cache.get("names") or [])
        garment_index = {name: index for index, name in enumerate(joint_names)}

        paired = _paired_body_support_field(source_positions_array, preserved, joint_names, cache)
        geometry_authority = _geometry_motion_authority(paired, behavior, effective_behavior)
        if paired is None:
            common_cache = np.asarray([i for i, name in enumerate(cache_names) if name in garment_index], dtype=np.int64)
        else:
            common_cache = np.asarray(paired["body_supported_cache_columns"], dtype=np.int64)

        delta, delta_report = delta_helper(source_positions_array, preserved, joint_names, cache)
        public_delta = _public_delta_report(delta_report)
        delta_verified = delta is not None and bool((delta_report or {}).get("verified_body_delta", False))
        if delta is None:
            delta = np.zeros((len(preserved), len(cache_names)), dtype=np.float64)
        else:
            delta = np.asarray(delta, dtype=np.float64)
            if delta.shape != (len(preserved), len(cache_names)):
                raise ValueError(f"Adaptive garment skinning delta shape {delta.shape} does not match {(len(preserved), len(cache_names))}.")

        report_columns = np.asarray((delta_report or {}).get("body_supported_cache_columns", []), dtype=np.int64)
        body_supported_cache = np.unique(np.concatenate((common_cache, report_columns))) if (len(common_cache) or len(report_columns)) else np.zeros(0, dtype=np.int64)
        mapped_pairs = [(int(ci), garment_index[cache_names[int(ci)]]) for ci in body_supported_cache.tolist()
                        if 0 <= int(ci) < len(cache_names) and cache_names[int(ci)] in garment_index]
        if not mapped_pairs:
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_no_local_body_support",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
                "motion_skinning_revision": 3,
                "source_influence_capacity": int(capacity),
                "source_influence_capacity_authority": capacity_authority,
            })
            return preserved, stage

        mapped_cache = np.asarray([row[0] for row in mapped_pairs], dtype=np.int64)
        body_columns = np.asarray([row[1] for row in mapped_pairs], dtype=np.int64)
        missing_cache = np.asarray([int(ci) for ci in report_columns.tolist()
                                    if 0 <= int(ci) < len(cache_names) and cache_names[int(ci)] not in garment_index], dtype=np.int64)

        quantisation_floor = float((delta_report or {}).get("quantisation_floor", 0.0035))
        local_l1 = np.asarray((delta_report or {}).get("local_delta_l1", np.abs(delta).sum(axis=1)), dtype=np.float64)
        if local_l1.shape != (len(preserved),):
            local_l1 = np.abs(delta).sum(axis=1)
        body_mass = preserved[:, body_columns].sum(axis=1)
        delta_active = delta_verified & (local_l1 > quantisation_floor) & (body_mass > 1e-8)
        if np.isscalar(delta_active):
            delta_active = np.full(len(preserved), bool(delta_active), dtype=bool)
        geometry_active = geometry_authority > 1e-5 if len(geometry_authority) == len(preserved) else np.zeros(len(preserved), dtype=bool)
        active = (delta_active | geometry_active) & (body_mass > 1e-8)

        if not np.any(active):
            stage = dict(base_stage)
            shape = np.asarray(paired["shape_change"], dtype=np.float64) if paired is not None else np.zeros(len(preserved))
            stage.update({
                "mode": "source_authored_skinning_preserved_same_target_region",
                "policy": "paired target geometry and deformation support are equivalent locally; preserve authored garment skinning exactly",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
                "motion_skinning_revision": 3,
                "source_influence_capacity": int(capacity),
                "source_influence_capacity_authority": capacity_authority,
                "paired_shape_change_p95_mm": float(np.percentile(shape, 95) * 1000.0) if len(shape) else 0.0,
                "geometry_motion_authority_p95": 0.0,
                "motion_response_residual_l1_mean": 0.0,
                "motion_response_residual_l1_p95": 0.0,
                "motion_response_residual_l1_max": 0.0,
                "capacity_limited_vertices": 0,
            })
            return preserved, stage

        if len(missing_cache) and delta_verified:
            missing_positive = np.clip(delta[:, missing_cache], 0.0, None).sum(axis=1) * body_mass
            missing_active = delta_active & (missing_positive > max(quantisation_floor, 1.0 / 255.0))
            if np.any(missing_active):
                missing_names = [cache_names[int(ci)] for ci in missing_cache.tolist()
                                 if np.any(delta_active & (np.clip(delta[:, int(ci)], 0.0, None) * body_mass > max(quantisation_floor, 1.0 / 255.0)))]
                shown = ", ".join(missing_names[:12])
                extra = f" (+{len(missing_names)-12} more)" if len(missing_names) > 12 else ""
                raise ValueError(
                    "Target body deformation requires body joint(s) absent from the garment skin: "
                    f"{shown}{extra}. RavaFit will not approximate the missing motion using unrelated bones."
                )

        source_body = preserved[:, body_columns].copy()
        body_delta = delta[:, mapped_cache]
        delta_ideal = np.maximum(source_body + body_delta * body_mass[:, None], 0.0)
        delta_total = delta_ideal.sum(axis=1)
        valid_delta = delta_total > 1e-12
        delta_ideal[valid_delta] *= (body_mass[valid_delta] / delta_total[valid_delta])[:, None]
        delta_ideal[~valid_delta] = source_body[~valid_delta]

        ideal_body = delta_ideal.copy()
        target_direct_valid = np.zeros(len(preserved), dtype=bool)
        if paired is not None and len(geometry_authority) == len(preserved):
            target_full = np.asarray(paired["target_weights"], dtype=np.float64)
            target_mapped = np.maximum(target_full[:, mapped_cache], 0.0)
            target_total = target_mapped.sum(axis=1)
            target_direct_valid = target_total > 1e-8
            target_direct = target_mapped.copy()
            target_direct[target_direct_valid] *= (body_mass[target_direct_valid] / target_total[target_direct_valid])[:, None]
            alpha = np.where(target_direct_valid, geometry_authority, 0.0)
            ideal_body = delta_ideal * (1.0 - alpha[:, None]) + target_direct * alpha[:, None]
        else:
            alpha = np.zeros(len(preserved), dtype=np.float64)

        ideal_total = ideal_body.sum(axis=1)
        valid_ideal = active & (ideal_total > 1e-12)
        ideal_body[valid_ideal] *= (body_mass[valid_ideal] / ideal_total[valid_ideal])[:, None]
        invalid_ideal = active & ~valid_ideal
        if np.any(invalid_ideal):
            ideal_body[invalid_ideal] = source_body[invalid_ideal]
            active[invalid_ideal] = False

        body_column_set = set(body_columns.tolist())
        non_body_columns = np.asarray([i for i in range(preserved.shape[1]) if i not in body_column_set], dtype=np.int64)
        non_body_active = np.count_nonzero(preserved[:, non_body_columns] > 1e-8, axis=1) if len(non_body_columns) else np.zeros(len(preserved), dtype=np.int64)
        packed_body, capacity_limited = _pack_body_mass_to_budget(ideal_body, source_body, body_mass, non_body_active, capacity, active)

        candidate = preserved.copy()
        candidate[:, body_columns] = np.where(active[:, None], packed_body, source_body)
        if len(non_body_columns) and not np.array_equal(candidate[:, non_body_columns], preserved[:, non_body_columns]):
            raise ValueError("Adaptive garment skinning modified authored non-body/cloth influences.")
        source_totals = preserved.sum(axis=1)
        candidate_totals = candidate.sum(axis=1)
        if np.any(~np.isfinite(candidate)) or np.any(np.abs(candidate_totals - source_totals) > 1e-7):
            raise ValueError("Adaptive garment skinning failed to preserve finite per-vertex weight mass.")

        motion_residual = np.abs(candidate[:, body_columns] - ideal_body).sum(axis=1)
        motion_residual[~active] = 0.0
        weight_delta = np.abs(candidate - preserved).sum(axis=1)
        changed = weight_delta > 1e-10
        candidate_influences = np.count_nonzero(candidate > 1e-8, axis=1)
        if int(np.max(candidate_influences, initial=0)) > capacity:
            raise ValueError(f"Adaptive garment skinning exceeded the source GLTF influence capacity of {capacity}.")

        if not np.any(changed):
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_after_motion_check",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
                "motion_skinning_revision": 3,
                "source_influence_capacity": int(capacity),
                "source_influence_capacity_authority": capacity_authority,
                "motion_response_residual_l1_mean": float(np.mean(motion_residual)),
                "motion_response_residual_l1_p95": float(np.percentile(motion_residual, 95)) if len(motion_residual) else 0.0,
                "motion_response_residual_l1_max": float(np.max(motion_residual, initial=0.0)),
                "capacity_limited_vertices": int(np.count_nonzero(capacity_limited)),
            })
            return preserved, stage

        newly_used_body = 0
        for column in body_columns.tolist():
            if not np.any(preserved[:, column] > 1e-8) and np.any(candidate[:, column] > 1e-8):
                newly_used_body += 1
        shape = np.asarray(paired["shape_change"], dtype=np.float64) if paired is not None else np.zeros(len(preserved))
        support_distance = np.asarray(paired["support_distance"], dtype=np.float64) if paired is not None else np.full(len(preserved), np.inf)
        stage = dict(base_stage)
        stage.update({
            "mode": "target_authored_motion_body_skinning",
            "policy": "preserve source rig only where both paired target shape and deformation field are equivalent; for materially changed close-body geometry, adapt only body-supported mass toward paired target-body motion while preserving garment/cloth/secondary influences exactly",
            "vertices": int(len(candidate)),
            "retargeted_vertices": int(np.count_nonzero(changed)),
            "exact_source_weight_preserve": False,
            "adaptive_body_skinning": True,
            "motion_skinning_revision": 3,
            "preserved_non_body_weights_exact": True,
            "body_supported_joint_count": int(len(body_columns)),
            "newly_activated_body_joint_columns": int(newly_used_body),
            "source_influence_capacity": int(capacity),
            "source_influence_capacity_authority": capacity_authority,
            "max_final_active_influences": int(np.max(candidate_influences, initial=0)),
            "capacity_limited_vertices": int(np.count_nonzero(capacity_limited)),
            "weight_delta_l1_mean": float(np.mean(weight_delta)),
            "weight_delta_l1_p95": float(np.percentile(weight_delta, 95)) if len(weight_delta) else 0.0,
            "weight_delta_l1_max": float(np.max(weight_delta, initial=0.0)),
            "motion_response_residual_l1_mean": float(np.mean(motion_residual)),
            "motion_response_residual_l1_p95": float(np.percentile(motion_residual, 95)) if len(motion_residual) else 0.0,
            "motion_response_residual_l1_max": float(np.max(motion_residual, initial=0.0)),
            "full_delta_vertices": int(np.count_nonzero(delta_active)),
            "geometry_driven_vertices": int(np.count_nonzero(geometry_active)),
            "paired_shape_change_p95_mm": float(np.percentile(shape, 95) * 1000.0) if len(shape) else 0.0,
            "source_support_distance_p95_mm": float(np.percentile(support_distance[np.isfinite(support_distance)], 95) * 1000.0) if np.any(np.isfinite(support_distance)) else None,
            "geometry_motion_authority_mean": float(np.mean(alpha)) if len(alpha) else 0.0,
            "geometry_motion_authority_p95": float(np.percentile(alpha, 95)) if len(alpha) else 0.0,
            "body_weight_delta": public_delta,
        })
        return candidate, stage

    prod._retarget_garment_skinning = adaptive
    prod._ravafit_adaptive_body_skinning_installed = True
