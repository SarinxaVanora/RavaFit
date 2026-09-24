from __future__ import annotations

from functools import wraps
from typing import Any

import core_fit_authority


def install_core_fit_entrypoint(production: Any) -> None:
    """Ensure every in-process garment solve receives core fit authority.

    Multi-mesh isolated workers already install the authority inside their own
    process. The production fast path for a single authored garment mesh does
    not spawn those workers, so install the same preparation around the common
    _solve_garment_meshes entrypoint used by direct, filtered and fallback
    solves as well.
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
        result = original(source, cache, body_mesh_names, mesh_filter, *args, **kwargs)
        try:
            stats = result[3]
            if isinstance(stats, dict):
                stats["core_fit_entrypoint"] = report
        except Exception:
            pass
        return result

    production._solve_garment_meshes = solve_with_core_fit
    production._ravafit_core_fit_entrypoint_installed = True
