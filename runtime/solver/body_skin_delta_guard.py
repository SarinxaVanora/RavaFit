"""Expose the complete paired-body deformation field to adaptive garment skinning.

Production body correspondence is allowed to contain body joints that the untouched garment
skin does not currently name. The delta detector must still see those target-only columns;
otherwise a real body deformation change can be silently filtered out before adaptive_skinning
has a chance to either apply it through an existing garment joint or fail safely.

This module never transfers nearest-body weights and never invents a replacement bone.
"""
from __future__ import annotations

from typing import Any

import numpy as np


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
        if weights.ndim != 2 or weights.shape[1] != len(names) or not cache_names:
            return original(points, garment_weights, garment_joint_names, cache)

        missing = [name for name in cache_names if name not in names]
        if not missing:
            return original(points, weights, names, cache)

        augmented = np.zeros((len(weights), len(names) + len(missing)), dtype=np.float64)
        augmented[:, :len(names)] = weights
        delta, report = original(points, augmented, names + missing, cache)
        report = dict(report or {})
        report["garment_zero_augmented_body_joint_columns"] = int(len(missing))
        report["garment_zero_augmented_body_joint_names"] = list(missing)
        report["complete_body_delta_visibility"] = True
        report["policy"] = (
            "evaluate the complete paired-body deformation field, including target-only body joints; "
            "adaptive skinning may use only joints genuinely exposed by the garment skin and must fail rather than guess"
        )
        return delta, report

    prod._verified_body_skin_delta_at_points = guarded
    prod._ravafit_complete_body_delta_visibility_installed = True
