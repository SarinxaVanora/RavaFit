"""Retire unavailable *body-authored* garment bones using the selected target body's skin.

This is intentionally not a general nearest-body reskin. It operates only on original
source body joints absent from every selected target body's original MDL bone table,
and only when the garment itself uses those joints. All other authored garment
influences are unchanged. The target body supplies the replacement joint blend,
sampled at the garment's final fitted positions; the source body never does.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from adaptive_skinning import _source_capacity

_CACHE_MISSING = "_ravafit_source_body_joints_absent_from_target"
_CACHE_TARGET = "_ravafit_target_body_weighted_joints"


def remap_unavailable_body_bones(final_positions, source_positions, weights, joint_names, cache, sampler):
    W = np.asarray(weights, dtype=np.float64)
    P = np.asarray(final_positions, dtype=np.float64)
    names = list(joint_names)
    missing = set(cache.get(_CACHE_MISSING) or [])
    if W.ndim != 2 or W.shape != (len(P), len(names)):
        raise ValueError("Unsupported-body-bone remap received incompatible garment skin data.")
    if not missing:
        return W, {"enabled": False, "reason": "no source-body-weighted joints absent from selected target MDLs"}
    absent = [i for i, name in enumerate(names) if name in missing]
    if not absent:
        return W, {"enabled": False, "reason": "affected source body joints are not used by this garment"}
    affected = np.flatnonzero(W[:, absent].sum(axis=1) > 1e-8)
    if not len(affected):
        return W, {"enabled": False, "reason": "garment has no positive weights on unavailable body joints"}

    cache_names = list(cache.get("names") or [])
    target_weighted = set(cache.get(_CACHE_TARGET) or [])
    eligible = [(i, names.index(name)) for i, name in enumerate(cache_names)
                if name in target_weighted and name in names and name not in missing]
    if not eligible:
        raise ValueError("Selected target body exposes no weighted bones supported by this garment; cannot retire missing body bones.")

    # The production sampler uses the actual target body mesh and its own skin
    # weights, including its side/leg disambiguation.  Sampling the final
    # garment location is essential: the source shape may differ significantly.
    target_weights, distance = sampler(P[affected], cache, W[affected], names)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if target_weights.shape != (len(affected), len(cache_names)) or distance.shape != (len(affected),):
        raise ValueError("Selected target-body skin witness has an invalid shape.")
    if not np.all(np.isfinite(target_weights)) or np.any(target_weights < -1e-10) or not np.all(np.isfinite(distance)):
        raise ValueError("Selected target-body skin witness contains invalid weights or distances.")

    result = W.copy()
    cap, authority = _source_capacity(W, np.asarray(source_positions, dtype=np.float64), names, cache)
    chosen_columns = np.asarray([row[0] for row in eligible], dtype=np.int64)
    garment_columns = np.asarray([row[1] for row in eligible], dtype=np.int64)
    missing_mass = W[affected][:, absent].sum(axis=1)
    result[np.ix_(affected, absent)] = 0.0
    truncated = 0
    for row, original_row in enumerate(affected.tolist()):
        source_mass = float(missing_mass[row])
        target_blend = np.maximum(target_weights[row, chosen_columns], 0.0)
        donor_available = target_blend > 1e-8
        if not np.any(donor_available):
            raise ValueError(
                "Target body has no supported anatomical skin-weight witness for garment vertex "
                f"{original_row}; cannot safely transfer missing bone mass."
            )
        # Never evict an authored cloth, detail, pelvis/thigh bridge or other
        # still-supported influence. Only vacant GLTF slots can add donor bones.
        existing = np.flatnonzero(result[original_row] > 1e-8)
        spare = int(cap) - len(existing)
        if spare < 0:
            raise ValueError(f"Retiring missing bones found {len(existing)} surviving influences but only {cap} GLTF slots.")
        ranked = np.argsort(-target_blend, kind="stable")
        donors = [int(i) for i in ranked if donor_available[i] and garment_columns[i] in existing]
        novel = [int(i) for i in ranked if donor_available[i] and garment_columns[i] not in existing]
        if len(novel) > spare:
            truncated += 1
        donors.extend(novel[:spare])
        if not donors:
            raise ValueError(
                f"Garment vertex {original_row} has no available GLTF influence slots or existing target-body donors; "
                "cannot safely retire its absent bone."
            )
        blend = target_blend[donors]
        blend = blend / float(blend.sum())
        result[original_row, garment_columns[donors]] += source_mass * blend

    before = W.sum(axis=1)
    after = result.sum(axis=1)
    if np.any(np.abs(after - before) > 1e-7) or np.any(~np.isfinite(result)) or np.any(result < -1e-10):
        raise ValueError("Retiring unavailable body bones changed the authored garment's total skin weight.")
    if np.any(result[:, absent] > 1e-8):
        raise ValueError("Retiring unavailable body bones left residual weight on the unsupported joints.")
    if np.any(np.count_nonzero(result > 1e-8, axis=1) > cap):
        raise ValueError("Retiring unavailable body bones exceeded source GLTF influence capacity.")

    return result, {
        "enabled": True,
        "policy": "zero source-body bones absent from selected target MDL; move only their original weight mass onto locally sampled target-body supported bones; preserve all other authored influences",
        "affected_vertices": int(len(affected)),
        "retired_joint_names": [names[i] for i in absent if np.any(W[:, i] > 1e-8)],
        "reassigned_weight_mass": float(np.sum(missing_mass)),
        "capacity_limited_vertices": int(truncated),
        "source_influence_capacity": int(cap),
        "source_influence_capacity_authority": authority,
        "target_witness_distance_p95_mm": float(np.percentile(distance, 95) * 1000.0),
        "target_witness_distance_max_mm": float(np.max(distance) * 1000.0),
        "nonmissing_source_weights_preserved_except_target_guided_addition": True,
        "geometry_modified": False,
    }


def install_target_guided_bone_retirement(prod: Any) -> None:
    if getattr(prod, "_ravafit_target_guided_body_bone_retirement_installed", False):
        return
    original = getattr(prod, "_retarget_garment_skinning", None)
    if not callable(original):
        raise AttributeError("Production has no skinning finalizer to wrap.")
    sampler = getattr(prod, "_target_skin_weights_at_points", None)

    def retarget(raw_positions, source_positions, source_weights, source_joint_names, cache,
                 behavior, effective_behavior, labels, classes, raw_to_weld):
        solved, stage = original(raw_positions, source_positions, source_weights, source_joint_names,
                                 cache, behavior, effective_behavior, labels, classes, raw_to_weld)
        if not cache.get(_CACHE_MISSING):
            return solved, stage
        if not callable(sampler):
            raise ValueError("RavaFit cannot sample the selected target body's actual skin weights.")
        corrected, report = remap_unavailable_body_bones(
            raw_positions, source_positions, solved, source_joint_names, cache, sampler,
        )
        stage = dict(stage)
        stage["target_guided_unavailable_body_bones"] = report
        if report.get("enabled"):
            stage["exact_source_weight_preserve"] = False
            stage["retargeted_vertices"] = max(int(stage.get("retargeted_vertices", 0)), int(report["affected_vertices"]))
        return corrected, stage

    prod._retarget_garment_skinning = retarget
    prod._ravafit_target_guided_body_bone_retirement_installed = True
