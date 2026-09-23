"""Evidence-driven garment skinning adaptation.

The goal is for a converted garment to deform as though it had been authored for the
selected target body.  The paired source->target body skin field is the only body
skinning authority:

* equivalent body deformation field -> preserve source garment skinning exactly;
* genuine paired-body field change -> apply the full corresponding change to only the
  garment's body-supported weight mass;
* garment/cloth/secondary influences remain exact structural authority;
* target body blends may use more body influences than the source vertex when the
  source GLTF primitive actually has room for them.

This is deliberately not nearest-target weight transfer.  Geometry moving through a
weight gradient is not proof that its rig should change.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

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
    """Record the real 4/8 influence capacity of each source GLTF primitive.

    Workers call this before solving.  It lets the adaptive skinning stage use a second
    JOINTS/WEIGHTS set when the source primitive genuinely owns one, even if every
    untouched source vertex happened to use four or fewer non-zero influences.
    """
    registry = dict(cache.get(_CAPACITY_CACHE_KEY) or {})
    rows = []
    source_js = getattr(source, "js", None)
    mesh_names = list(source.mesh_names()) if callable(getattr(source, "mesh_names", None)) else []
    for name in mesh_names:
        try:
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
                # A hash collision or duplicate authored payload with contradictory
                # primitive layout must never silently choose the larger capacity.
                raise ValueError(f"Conflicting source influence capacities for garment skin {name}: {previous} vs {capacity}.")
            registry[key] = int(capacity)
            rows.append({"mesh": str(name), "capacity": int(capacity), "max_authored_active": active})
        except Exception:
            raise
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


def install_adaptive_body_skinning(prod: Any) -> None:
    """Install correspondence-driven, motion-faithful garment body skinning."""
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

        delta, delta_report = delta_helper(source_positions_array, preserved, joint_names, cache)
        public_delta = _public_delta_report(delta_report)
        if delta is None or not bool((delta_report or {}).get("verified_body_delta", False)):
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_same_body_field",
                "policy": "paired source/target body deformation support is equivalent; preserve authored garment skinning exactly",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
                "motion_skinning_revision": 2,
                "source_influence_capacity": int(capacity),
                "source_influence_capacity_authority": capacity_authority,
                "motion_response_residual_l1_mean": 0.0,
                "motion_response_residual_l1_p95": 0.0,
                "motion_response_residual_l1_max": 0.0,
                "capacity_limited_vertices": 0,
            })
            return preserved, stage

        delta = np.asarray(delta, dtype=np.float64)
        cache_names = list(cache.get("names") or [])
        if delta.shape != (len(preserved), len(cache_names)):
            raise ValueError(f"Adaptive garment skinning delta shape {delta.shape} does not match {(len(preserved), len(cache_names))}.")

        body_supported_cache = np.asarray((delta_report or {}).get("body_supported_cache_columns", []), dtype=np.int64)
        if body_supported_cache.ndim != 1 or not len(body_supported_cache):
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_no_local_body_support",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
                "motion_skinning_revision": 2,
                "source_influence_capacity": int(capacity),
                "source_influence_capacity_authority": capacity_authority,
            })
            return preserved, stage

        garment_index = {name: index for index, name in enumerate(joint_names)}
        mapped_pairs = [(int(ci), garment_index[cache_names[int(ci)]]) for ci in body_supported_cache.tolist()
                        if 0 <= int(ci) < len(cache_names) and cache_names[int(ci)] in garment_index]
        if not mapped_pairs:
            raise ValueError("Target body requires a skin-weight adaptation but the garment exposes none of the corresponding body joints.")
        mapped_cache = np.asarray([row[0] for row in mapped_pairs], dtype=np.int64)
        body_columns = np.asarray([row[1] for row in mapped_pairs], dtype=np.int64)
        missing_cache = np.asarray([int(ci) for ci in body_supported_cache.tolist()
                                    if 0 <= int(ci) < len(cache_names) and cache_names[int(ci)] not in garment_index], dtype=np.int64)

        local_l1 = np.asarray((delta_report or {}).get("local_delta_l1", np.abs(delta).sum(axis=1)), dtype=np.float64)
        if local_l1.shape != (len(preserved),):
            local_l1 = np.abs(delta).sum(axis=1)
        quantisation_floor = float((delta_report or {}).get("quantisation_floor", 0.0035))
        body_mass = preserved[:, body_columns].sum(axis=1)
        active = (local_l1 > quantisation_floor) & (body_mass > 1e-8)
        if not np.any(active):
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_local_body_field_equivalent",
                "policy": "local paired body weight change is below the encoder/noise floor",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
                "motion_skinning_revision": 2,
                "source_influence_capacity": int(capacity),
                "source_influence_capacity_authority": capacity_authority,
                "motion_response_residual_l1_mean": 0.0,
                "motion_response_residual_l1_p95": 0.0,
                "motion_response_residual_l1_max": 0.0,
                "capacity_limited_vertices": 0,
            })
            return preserved, stage

        if len(missing_cache):
            missing_positive = np.clip(delta[:, missing_cache], 0.0, None).sum(axis=1) * body_mass
            missing_active = active & (missing_positive > max(quantisation_floor, 1.0 / 255.0))
            if np.any(missing_active):
                missing_names = [cache_names[int(ci)] for ci in missing_cache.tolist()
                                 if np.any(active & (np.clip(delta[:, int(ci)], 0.0, None) * body_mass > max(quantisation_floor, 1.0 / 255.0)))]
                shown = ", ".join(missing_names[:12])
                extra = f" (+{len(missing_names)-12} more)" if len(missing_names) > 12 else ""
                raise ValueError(
                    "Target body deformation requires body joint(s) absent from the garment skin: "
                    f"{shown}{extra}. RavaFit will not approximate the missing motion using unrelated bones."
                )

        source_body = preserved[:, body_columns].copy()
        body_delta = delta[:, mapped_cache]
        ideal_body = np.maximum(source_body + body_delta * body_mass[:, None], 0.0)
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
                "motion_skinning_revision": 2,
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
        stage = dict(base_stage)
        stage.update({
            "mode": "motion_compatible_body_correspondence_skinning",
            "policy": "preserve source rig exactly where body deformation support is equivalent; otherwise apply the full paired body-weight delta while preserving authored garment-specific influences and the real source GLTF encoder capacity",
            "vertices": int(len(candidate)),
            "retargeted_vertices": int(np.count_nonzero(changed)),
            "exact_source_weight_preserve": False,
            "adaptive_body_skinning": True,
            "motion_skinning_revision": 2,
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
            "full_delta_vertices": int(np.count_nonzero(active)),
            "body_weight_delta": public_delta,
        })
        return candidate, stage

    prod._retarget_garment_skinning = adaptive
    prod._ravafit_adaptive_body_skinning_installed = True
