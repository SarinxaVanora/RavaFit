from __future__ import annotations

from functools import wraps
from typing import Any

import adaptive_skinning
import body_skin_delta_guard
import core_fit_authority
import strict_fit_authority


def _install_motion_authority(production: Any, cache: dict, source: Any) -> dict:
    """Install paired-body motion adaptation before the solve freezes skinning."""
    body_skin_delta_guard.install_complete_body_delta_visibility(production)
    capacity = adaptive_skinning.register_source_influence_capacities(cache, source)
    adaptive_skinning.install_adaptive_body_skinning(production)
    return capacity


def install_core_fit_entrypoint(production: Any) -> None:
    """Ensure every production garment lane receives the same fit authority.

    Duskwing's RBODY->RBODY conversion uses the strict historical B14 lane, while other
    conversions can use the modern in-process or isolated-worker lane.  Authority is therefore
    installed around BOTH production entrypoints; no route may silently fall back to the old fit.
    """
    if getattr(production, "_ravafit_core_fit_entrypoint_installed", False):
        return

    original = getattr(production, "_solve_garment_meshes", None)
    if not callable(original):
        raise AttributeError("production solver exposes no callable _solve_garment_meshes entrypoint")

    @wraps(original)
    def solve_with_core_fit(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        report = core_fit_authority.prepare_cache(production, cache)
        core_fit_authority.install_runtime_authority(production)
        capacity = _install_motion_authority(production, cache, source)
        result = original(source, cache, body_mesh_names, mesh_filter, *args, **kwargs)
        try:
            stats = result[3]
            if isinstance(stats, dict):
                stats["core_fit_entrypoint"] = report
                stats["source_influence_capacity_registry"] = capacity
        except Exception:
            pass
        return result

    production._solve_garment_meshes = solve_with_core_fit

    strict_original = getattr(production, "_solve_strict_b14_layers", None)
    if callable(strict_original):
        @wraps(strict_original)
        def strict_with_core_fit(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
            # This is the actual same-race RBODY lane used by the Duskwing top and dwn models.
            # Fail closed if macro/literal separation cannot be established: returning the old
            # strict result would recreate the byte-identical false-success we just diagnosed.
            report = strict_fit_authority.prepare_strict_cache(production, cache)
            core_fit_authority.install_runtime_authority(production)
            capacity = _install_motion_authority(production, cache, source)
            positions, skinning, records, stats = strict_original(source, cache, body_mesh_names, mesh_filter, *args, **kwargs)
            positions, final_report = strict_fit_authority.finalize_strict_solution(production, source, cache, positions)
            stats = dict(stats or {})
            stats["strict_core_fit_entrypoint"] = report
            stats["strict_final_authority"] = final_report
            stats["source_influence_capacity_registry"] = capacity
            return positions, skinning, records, stats

        production._solve_strict_b14_layers = strict_with_core_fit

    production._ravafit_core_fit_entrypoint_installed = True
