from __future__ import annotations

import numpy as np


def authored_clearance_floor(source_distance: np.ndarray, minimum_clearance: float = .00035, maximum_clearance: float = .030) -> np.ndarray:
    """Return the minimum target support-frame clearance authored by the source garment.

    Literal target anatomy may push the garment farther outward, but never inward below
    source-proven garment/body spacing. This is intentionally unsigned: it describes the
    physical gap authored into clothing, independent of local normal orientation quirks.
    """
    distance = np.asarray(source_distance, dtype=np.float64).reshape(-1)
    out = np.full(len(distance), float(minimum_clearance), dtype=np.float64)
    finite = np.isfinite(distance)
    out[finite] = np.clip(distance[finite], float(minimum_clearance), float(maximum_clearance))
    return out


def outward_standoff_deficit(current_radial: np.ndarray, source_distance: np.ndarray, minimum_clearance: float = .00035, maximum_clearance: float = .030) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current_radial, dtype=np.float64).reshape(-1)
    desired = authored_clearance_floor(source_distance, minimum_clearance, maximum_clearance)
    if current.shape != desired.shape:
        raise ValueError("Current radial clearance and source distance must have the same shape.")
    return np.maximum(desired - current, 0.0), desired
