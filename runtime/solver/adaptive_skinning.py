"""Evidence-driven garment skinning adaptation.

The goal is not to preserve weights for their own sake.  The goal is for the converted
garment to deform as though it had been authored for the selected target body.

The paired source->target body skin field is the only body-skinning authority here:

* when that field is equivalent under a garment region, the source garment skin is
  preserved exactly;
* when it genuinely changes, the *full* corresponding body-weight change is applied
  to the garment's body-supported weight mass;
* garment/cloth/secondary influences that are not part of the body support field are
  preserved exactly;
* body influences may become more complex than the source vertex when the target body
  actually needs that blend, but never at the expense of authored non-body influences
  or the GLTF/XIV per-vertex influence budget.

This is deliberately different from nearest-target weight transfer.  Moving a garment
vertex across a weight gradient is not evidence that its rig should change.  The paired
anatomical body correspondence must prove the change first.
"""
from __future__ import annotations

from typing import Any

import numpy as np


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


def _source_capacity_hint(weights: np.ndarray) -> int:
    """Return a conservative encoder capacity without inventing extra GLTF slots.

    A source vertex that already carries >4 influences proves the primitive has the
    second JOINTS/WEIGHTS set and can encode eight.  Otherwise assume four.  The final
    writer still validates the real primitive capacity; this conservative hint prevents
    adaptive skinning from relying on an unproven second influence set.
    """
    weights = np.asarray(weights, dtype=np.float64)
    maximum = int(np.max(np.count_nonzero(weights > 1e-8, axis=1), initial=0)) if weights.ndim == 2 else 0
    return 8 if maximum > 4 else 4


def _pack_body_mass_to_budget(ideal_body: np.ndarray, source_body: np.ndarray, body_mass: np.ndarray,
                              non_body_active: np.ndarray, capacity: int, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the ideal target-body blend into the remaining per-vertex influence slots.

    Authored non-body weights are handled by the caller and are never candidates for
    eviction.  Only body-supported influences compete for the remaining slots.
    """
    ideal_body = np.asarray(ideal_body, dtype=np.float64)
    source_body = np.asarray(source_body, dtype=np.float64)
    body_mass = np.asarray(body_mass, dtype=np.float64)
    non_body_active = np.asarray(non_body_active, dtype=np.int64)
    active = np.asarray(active, dtype=bool)
    packed = np.asarray(source_body, dtype=np.float64).copy()
    capacity_limited = np.zeros(len(packed), dtype=bool)

    for row in np.flatnonzero(active):
        available = max(0, int(capacity) - int(non_body_active[row]))
        source_slots = int(np.count_nonzero(source_body[row] > 1e-8))
        # The source itself was encodable.  If this ever trips, the inferred capacity
        # is inconsistent with the untouched source skin and we must not silently lose it.
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
            # A malformed/degenerate target delta must not erase a valid source body
            # influence.  Keep the authored body distribution in that exceptional row.
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
            })
            return preserved, stage

        garment_index = {name: index for index, name in enumerate(joint_names)}
        mapped_pairs = [(int(ci), garment_index[cache_names[int(ci)]]) for ci in body_supported_cache.tolist()
                        if 0 <= int(ci) < len(cache_names) and cache_names[int(ci)] in garment_index]
        if not mapped_pairs:
            raise ValueError("Target body requires a skin-weight adaptation but the garment exposes none of the corresponding body joints.")

        mapped_cache = np.asarray([row[0] for row in mapped_pairs], dtype=np.int64)
        body_columns = np.asarray([row[1] for row in mapped_pairs], dtype=np.int64)

        local_l1 = np.asarray((delta_report or {}).get("local_delta_l1", np.abs(delta).sum(axis=1)), dtype=np.float64)
        if local_l1.shape != (len(preserved),):
            local_l1 = np.abs(delta).sum(axis=1)
        quantisation_floor = float((delta_report or {}).get("quantisation_floor", 0.0035))

        # Once correspondence proves a local rigging difference above the noise floor,
        # apply the full body-field delta.  A partial alpha blend leaves the garment with
        # a different motion response from the target body and is exactly the kind of
        # mismatch that turns a clean bind pose into body poke-through during animation.
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
                "motion_response_residual_l1_mean": 0.0,
                "motion_response_residual_l1_p95": 0.0,
                "motion_response_residual_l1_max": 0.0,
                "capacity_limited_vertices": 0,
            })
            return preserved, stage

        source_body = preserved[:, body_columns].copy()
        body_delta = delta[:, mapped_cache]
        ideal_body = source_body + body_delta * body_mass[:, None]
        ideal_body = np.maximum(ideal_body, 0.0)
        ideal_total = ideal_body.sum(axis=1)
        valid_ideal = active & (ideal_total > 1e-12)
        ideal_body[valid_ideal] *= (body_mass[valid_ideal] / ideal_total[valid_ideal])[:, None]
        # If an exceptional row collapsed after clipping, treat it as non-adaptable rather
        # than manufacturing weights.  The source body blend remains the safe authority.
        invalid_ideal = active & ~valid_ideal
        if np.any(invalid_ideal):
            ideal_body[invalid_ideal] = source_body[invalid_ideal]
            active[invalid_ideal] = False

        body_column_set = set(body_columns.tolist())
        non_body_columns = np.asarray([i for i in range(preserved.shape[1]) if i not in body_column_set], dtype=np.int64)
        non_body_active = np.count_nonzero(preserved[:, non_body_columns] > 1e-8, axis=1) if len(non_body_columns) else np.zeros(len(preserved), dtype=np.int64)
        capacity = _source_capacity_hint(preserved)
        packed_body, capacity_limited = _pack_body_mass_to_budget(
            ideal_body, source_body, body_mass, non_body_active, capacity, active,
        )

        candidate = preserved.copy()
        candidate[active[:, None] & np.isin(np.arange(candidate.shape[1])[None, :], body_columns)] = candidate[active[:, None] & np.isin(np.arange(candidate.shape[1])[None, :], body_columns)]
        candidate[:, body_columns] = np.where(active[:, None], packed_body, source_body)

        # Authored garment/cloth/secondary influences are never collateral damage from
        # body retargeting.  Their exact values and total mass remain untouched.
        if len(non_body_columns) and not np.array_equal(candidate[:, non_body_columns], preserved[:, non_body_columns]):
            raise ValueError("Adaptive garment skinning modified authored non-body/cloth influences.")

        source_totals = preserved.sum(axis=1)
        candidate_totals = candidate.sum(axis=1)
        if np.any(~np.isfinite(candidate)) or np.any(np.abs(candidate_totals - source_totals) > 1e-7):
            raise ValueError("Adaptive garment skinning failed to preserve finite per-vertex weight mass.")

        # This is the important movement diagnostic: how far the encodable garment body
        # response remains from the ideal paired target-body response after respecting
        # cloth influences and the proven source primitive capacity.
        motion_residual = np.abs(candidate[:, body_columns] - ideal_body).sum(axis=1)
        motion_residual[~active] = 0.0
        weight_delta = np.abs(candidate - preserved).sum(axis=1)
        changed = weight_delta > 1e-10

        if not np.any(changed):
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_after_motion_check",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
                "motion_skinning_revision": 2,
                "motion_response_residual_l1_mean": float(np.mean(motion_residual)),
                "motion_response_residual_l1_p95": float(np.percentile(motion_residual, 95)) if len(motion_residual) else 0.0,
                "motion_response_residual_l1_max": float(np.max(motion_residual, initial=0.0)),
                "capacity_limited_vertices": int(np.count_nonzero(capacity_limited)),
                "source_influence_capacity_hint": int(capacity),
            })
            return preserved, stage

        newly_used_body = 0
        for column in body_columns.tolist():
            if not np.any(preserved[:, column] > 1e-8) and np.any(candidate[:, column] > 1e-8):
                newly_used_body += 1

        candidate_influences = np.count_nonzero(candidate > 1e-8, axis=1)
        if int(np.max(candidate_influences, initial=0)) > capacity:
            raise ValueError("Adaptive garment skinning exceeded the conservative source influence capacity.")

        stage = dict(base_stage)
        stage.update({
            "mode": "motion_compatible_body_correspondence_skinning",
            "policy": "preserve source rig exactly where body deformation support is equivalent; otherwise apply the full paired body-weight delta while preserving authored garment-specific influences and encoder capacity",
            "vertices": int(len(candidate)),
            "retargeted_vertices": int(np.count_nonzero(changed)),
            "exact_source_weight_preserve": False,
            "adaptive_body_skinning": True,
            "motion_skinning_revision": 2,
            "preserved_non_body_weights_exact": True,
            "body_supported_joint_count": int(len(body_columns)),
            "newly_activated_body_joint_columns": int(newly_used_body),
            "source_influence_capacity_hint": int(capacity),
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
