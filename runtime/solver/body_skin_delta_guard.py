"""Expose only physically relevant paired-body deformation to adaptive garment skinning.

Production body correspondence may contain body joints that the untouched garment skin does not
currently name. The delta detector must still see those target-only columns, but a body-weight
change is motion authority only where the untouched garment is actually supported by the source
body. Remote dress/skirt/ornament vertices retain their authored skinning exactly.

This module never transfers nearest-body weights and never invents a replacement bone.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_BODY_SUPPORT_FULL_M = .020
_BODY_SUPPORT_NONE_M = .035


def _gate_delta_by_source_body_support(delta, report):
    if delta is None:
        return delta, report
    report = dict(report or {})
    if not bool(report.get("verified_body_delta", False)):
        return delta, report

    values = np.asarray(delta, dtype=np.float64)
    support = np.asarray(report.get("support_distance", []), dtype=np.float64).reshape(-1)
    if values.ndim != 2 or support.shape != (len(values),) or np.any(~np.isfinite(support)):
        report["body_support_distance_gate"] = False
        report["body_support_distance_gate_reason"] = "per-vertex source-body support distance unavailable"
        return values, report

    linear = np.clip((_BODY_SUPPORT_NONE_M - support) / max(_BODY_SUPPORT_NONE_M - _BODY_SUPPORT_FULL_M, 1e-12), 0.0, 1.0)
    gain = linear * linear * (3.0 - 2.0 * linear)
    gated = values * gain[:, None]
    local_l1 = np.abs(gated).sum(axis=1)

    report["local_delta_l1"] = local_l1
    report["local_delta_l1_p95"] = float(np.percentile(local_l1, 95)) if len(local_l1) else 0.0
    report["local_delta_l1_max"] = float(np.max(local_l1, initial=0.0))
    report["body_support_distance_gate"] = True
    report["body_support_full_mm"] = _BODY_SUPPORT_FULL_M * 1000.0
    report["body_support_zero_mm"] = _BODY_SUPPORT_NONE_M * 1000.0
    report["body_support_full_vertices"] = int(np.count_nonzero(gain >= 1.0 - 1e-12))
    report["body_support_partial_vertices"] = int(np.count_nonzero((gain > 1e-12) & (gain < 1.0 - 1e-12)))
    report["body_support_source_exact_remote_vertices"] = int(np.count_nonzero(gain <= 1e-12))
    report["policy"] = (
        "evaluate the complete paired-body deformation field, including target-only body joints, then apply it only "
        "where the untouched garment is source-body-supported; remote cloth keeps authored source skinning and "
        "adaptive skinning must fail rather than guess a missing required body joint"
    )
    return gated, report


def install_complete_body_delta_visibility(prod: Any) -> None:
    if getattr(prod, "_ravafit_complete_body_delta_visibility_installed", False):
        return
    original = getattr(prod, "_verified_body_skin_delta_at_points", None)
    if not callable(original):
        return

    def guarded(points, garment_weights, garment_joint_names, cache):
        weights = np.asarray(garment_weights, dtype=np.float64)
        names = list(garment_joint_names)
        cache_names = list(cache.get("names") or [])
        missing: list[str] = []

        if weights.ndim == 2 and weights.shape[1] == len(names) and cache_names:
            missing = [name for name in cache_names if name not in names]

        if missing:
            augmented = np.zeros((len(weights), len(names) + len(missing)), dtype=np.float64)
            augmented[:, :len(names)] = weights
            delta, report = original(points, augmented, names + missing, cache)
            report = dict(report or {})
            report["garment_zero_augmented_body_joint_columns"] = int(len(missing))
            report["garment_zero_augmented_body_joint_names"] = list(missing)
            report["complete_body_delta_visibility"] = True
        else:
            delta, report = original(points, garment_weights, garment_joint_names, cache)
            report = dict(report or {})
            report["garment_zero_augmented_body_joint_columns"] = 0
            report["complete_body_delta_visibility"] = True

        return _gate_delta_by_source_body_support(delta, report)

    prod._verified_body_skin_delta_at_points = guarded
    prod._ravafit_complete_body_delta_visibility_installed = True
