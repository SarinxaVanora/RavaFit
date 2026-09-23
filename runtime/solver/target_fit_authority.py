"""Narrow runtime guard for close constructed garment shells.

The production solver already keeps the complete selected target anatomy as fitting
collision authority and preserves authored garment skinning.  This module therefore
only closes the remaining relief-policy gap: close constructed cloth should follow
the smooth target macro support while literal local anatomy remains collision-only.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def install_close_shell_macro_authority(prod: Any) -> None:
    """Prevent local target relief from flattening close constructed shells.

    The ordinary source-to-target macro mapping remains untouched, so target breast,
torso and limb volume still drive the fit.  Literal target geometry is also still
used by the existing final collision guards.  Only the local relief imprinting pass
is skipped for genuinely close constructed cloth.
    """
    if getattr(prod, "_ravafit_close_shell_macro_authority_installed", False):
        return
    original = getattr(prod, "_apply_target_relief_correction", None)
    if not callable(original):
        return

    def guarded(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features):
        behaviour = str(behavior or "").casefold()
        clearance_mm = float((features or {}).get("source_clearance_median_mm", 999.0))
        if behaviour == "constructed_close_shell" and clearance_mm <= 4.50:
            return np.asarray(mapped, dtype=np.float64), {
                "enabled": False,
                "reason": "close constructed cloth uses smooth target macro support; literal target detail remains collision authority",
                "source_clearance_median_mm": clearance_mm,
            }
        return original(source_vertices, faces, mapped, blend, blend_ids, cache, behavior, features)

    prod._apply_target_relief_correction = guarded
    prod._ravafit_close_shell_macro_authority_installed = True
