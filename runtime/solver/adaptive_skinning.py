"""Evidence-driven garment skinning adaptation.

RavaFit should make a garment behave as though it had been authored for the selected
body without needlessly destroying good source rigging.  The source garment therefore
remains exact authority wherever the paired source/target body deformation field is
equivalent.  Where that body field genuinely changes, only the garment's body-supported
weight mass is adapted; cloth/garment-specific influences remain untouched.
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


def _smooth_gate(values: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = max(float(floor), 1e-8)
    hi = max(lo * 3.0, 2.0 / 255.0)
    t = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def install_adaptive_body_skinning(prod: Any) -> None:
    """Replace exact-preserve-only skinning with local correspondence-driven adaptation.

    The production helper ``_verified_body_skin_delta_at_points`` is deliberately used
    as the sole evidence source.  It compares paired source/target body weights at the
    same anatomical correspondence, so moving garment geometry through a weight gradient
    is never mistaken for proof that a garment needs reweighting.
    """
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
                "policy": "preserve authored garment skinning exactly wherever paired source/target body deformation support is equivalent",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
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
        alpha = _smooth_gate(local_l1, quantisation_floor)

        body_mass = preserved[:, body_columns].sum(axis=1)
        active = (alpha > 0.0) & (body_mass > 1e-8)
        if not np.any(active):
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_local_body_field_equivalent",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
            })
            return preserved, stage

        # If the target needs positive weight on a body joint that is not present in the
        # garment skin at all, do not silently fake a conversion.  A future skin-expansion
        # path can add such a joint explicitly; until then correctness wins over guessing.
        if len(missing_cache):
            missing_positive = np.clip(delta[:, missing_cache], 0.0, None).sum(axis=1) * body_mass * alpha
            if float(np.max(missing_positive, initial=0.0)) > max(quantisation_floor, 1.0 / 255.0):
                missing_names = [cache_names[int(ci)] for ci in missing_cache.tolist()
                                 if float(np.max(np.clip(delta[:, int(ci)], 0.0, None), initial=0.0)) > quantisation_floor]
                shown = ", ".join(missing_names[:12])
                extra = f" (+{len(missing_names)-12} more)" if len(missing_names) > 12 else ""
                raise ValueError(
                    "Target body deformation requires body joint(s) absent from the garment skin: "
                    f"{shown}{extra}. RavaFit will not approximate this by corrupting existing garment weights."
                )

        candidate = preserved.copy()
        body_delta = delta[:, mapped_cache]
        proposed_body = preserved[:, body_columns] + body_delta * body_mass[:, None] * alpha[:, None]
        proposed_body = np.maximum(proposed_body, 0.0)

        # Preserve the exact authored amount of non-body/cloth influence.  Only the
        # distribution inside the existing body-supported mass is allowed to change.
        proposed_total = proposed_body.sum(axis=1)
        valid = active & (proposed_total > 1e-12)
        proposed_body[valid] *= (body_mass[valid] / proposed_total[valid])[:, None]

        # Never increase a vertex's total influence complexity.  This guarantees that
        # repacking cannot evict authored cloth/garment-specific influences merely because
        # the target body uses a different blend of body bones.  Body bones may replace
        # one another inside the original body-influence budget.
        source_body_active = np.count_nonzero(preserved[:, body_columns] > 1e-8, axis=1)
        for row in np.flatnonzero(valid):
            slots = int(source_body_active[row])
            if slots <= 0:
                continue
            values = proposed_body[row]
            if np.count_nonzero(values > 1e-8) > slots:
                order = np.argsort(-values, kind="stable")
                keep = order[:slots]
                reduced = np.zeros_like(values)
                reduced[keep] = values[keep]
                total = float(reduced.sum())
                if total <= 1e-12:
                    continue
                reduced *= float(body_mass[row]) / total
                proposed_body[row] = reduced
            candidate[row, body_columns] = proposed_body[row]

        weight_delta = np.abs(candidate - preserved).sum(axis=1)
        changed = weight_delta > 1e-10
        if not np.any(changed):
            stage = dict(base_stage)
            stage.update({
                "mode": "source_authored_skinning_preserved_after_adaptive_check",
                "retargeted_vertices": 0,
                "exact_source_weight_preserve": True,
                "body_weight_delta": public_delta,
                "adaptive_body_skinning": True,
            })
            return preserved, stage

        # Non-body columns must be bit-for-bit untouched before downstream packing.
        body_column_set = set(body_columns.tolist())
        non_body_columns = np.asarray([i for i in range(preserved.shape[1]) if i not in body_column_set], dtype=np.int64)
        if len(non_body_columns) and not np.array_equal(candidate[:, non_body_columns], preserved[:, non_body_columns]):
            raise ValueError("Adaptive garment skinning modified authored non-body/cloth influences.")

        totals = candidate.sum(axis=1)
        if np.any(~np.isfinite(candidate)) or np.any(np.abs(totals - preserved.sum(axis=1)) > 1e-7):
            raise ValueError("Adaptive garment skinning failed to preserve finite per-vertex weight mass.")

        newly_used = 0
        for column in body_columns.tolist():
            if not np.any(preserved[:, column] > 1e-8) and np.any(candidate[:, column] > 1e-8):
                newly_used += 1

        stage = dict(base_stage)
        stage.update({
            "mode": "adaptive_body_correspondence_skinning",
            "policy": "preserve source-authored garment rigging exactly where body deformation support is equivalent; otherwise adapt only body-supported mass from paired source-to-target body correspondence",
            "vertices": int(len(candidate)),
            "retargeted_vertices": int(np.count_nonzero(changed)),
            "exact_source_weight_preserve": False,
            "adaptive_body_skinning": True,
            "preserved_non_body_weights_exact": True,
            "body_supported_joint_count": int(len(body_columns)),
            "new_target_joint_slots": int(newly_used),
            "influence_budget_preserved": True,
            "weight_delta_l1_mean": float(np.mean(weight_delta)),
            "weight_delta_l1_p95": float(np.percentile(weight_delta, 95)) if len(weight_delta) else 0.0,
            "weight_delta_l1_max": float(np.max(weight_delta, initial=0.0)),
            "adaptation_alpha_mean": float(np.mean(alpha[active])) if np.any(active) else 0.0,
            "adaptation_alpha_p95": float(np.percentile(alpha[active], 95)) if np.any(active) else 0.0,
            "body_weight_delta": public_delta,
        })
        return candidate, stage

    prod._retarget_garment_skinning = adaptive
    prod._ravafit_adaptive_body_skinning_installed = True
