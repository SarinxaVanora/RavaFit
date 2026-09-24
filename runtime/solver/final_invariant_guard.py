"""Final coupled invariants for garment output.

The existing dense occupancy stage correctly treats literal target anatomy as occupancy-only.
This module adds the missing fixed-point contract: source-authored shared seams/attachments and
literal-body clearance must both be true at the same time. A clearance pass is never allowed to
be the final mutation after a seam projection, and a seam projection is never allowed to hide a
new body intersection.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import final_occupancy_guard

_FINAL_MARGIN_M = .00035


def install_final_garment_invariants(prod: Any) -> None:
    if getattr(prod, "_ravafit_final_garment_invariants_installed", False):
        return
    original = getattr(prod, "_finalize_modded_coupled_solution", None)
    preserve_seams = getattr(prod, "_preserve_final_source_shared_seams", None)
    clear = getattr(prod, "_final_target_body_clearance", None)
    if not callable(original) or not callable(clear):
        return

    def guarded(source, cache, positions, skinning, records, contexts, local_affine_quality_rms_mm):
        candidate, skinning_out, records_out, stats = original(
            source, cache, positions, skinning, records, contexts, local_affine_quality_rms_mm,
        )
        candidate = {name: np.asarray(value, dtype=np.float64).copy() for name, value in candidate.items()}
        surfaces = final_occupancy_guard._target_surfaces(prod, cache)
        if surfaces is None:
            raise ValueError("Final seam/clearance invariant cannot resolve target collision/support surfaces.")
        target_collision, target_support = surfaces

        passes: list[dict[str, Any]] = []
        converged = False
        for pass_index in range(12):
            if callable(preserve_seams):
                candidate, seam_report = preserve_seams(source, candidate)
                candidate = {name: np.asarray(value, dtype=np.float64) for name, value in candidate.items()}
            else:
                seam_report = {"enabled": False, "reason": "source shared-seam synchronizer unavailable"}

            validation = final_occupancy_guard._dense_penetration_report(
                prod, source, candidate, target_collision, margin=_FINAL_MARGIN_M,
            )
            remaining = int(validation.get("penetrating_samples", 0))
            passes.append({
                "pass": int(pass_index + 1),
                "seams": seam_report,
                "validation_after_seams": validation,
            })
            if remaining == 0:
                converged = True
                break

            candidate, repair_report = final_occupancy_guard._repair_until_dense_clear(
                prod, source, candidate, target_collision, target_support, clear, margin=_FINAL_MARGIN_M,
            )
            passes[-1]["occupancy_repair"] = repair_report

        # The only acceptable emitted state is one checked *after* the final seam projection.
        if callable(preserve_seams):
            candidate, final_seam_report = preserve_seams(source, candidate)
            candidate = {name: np.asarray(value, dtype=np.float64) for name, value in candidate.items()}
        else:
            final_seam_report = {"enabled": False, "reason": "source shared-seam synchronizer unavailable"}
        final_validation = final_occupancy_guard._dense_penetration_report(
            prod, source, candidate, target_collision, margin=_FINAL_MARGIN_M,
        )
        if int(final_validation.get("penetrating_samples", 0)) != 0:
            raise ValueError(
                "Final source-seam and target-body clearance constraints could not be satisfied together. "
                "RavaFit refused to emit a model with a split join or a knowingly clipping garment."
            )

        merged = dict(stats or {})
        merged["final_garment_invariants"] = {
            "converged": bool(converged),
            "passes": passes,
            "final_seams": final_seam_report,
            "final_validation": final_validation,
            "minimum_dense_clearance_mm": float(_FINAL_MARGIN_M * 1000.0),
            "policy": "source-authored shared seams/attachments and literal target occupancy are simultaneous hard invariants; literal anatomy never supplies garment shape",
        }
        return candidate, skinning_out, records_out, merged

    prod._finalize_modded_coupled_solution = guarded
    prod._ravafit_final_garment_invariants_installed = True
