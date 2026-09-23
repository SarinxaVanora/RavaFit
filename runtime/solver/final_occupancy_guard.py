from __future__ import annotations
from typing import Any
import numpy as np

_DENSE_BARY = np.asarray(
    [(i / 4.0, j / 4.0, (4 - i - j) / 4.0) for i in range(5) for j in range(5 - i)]
    + [(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)],
    dtype=np.float64,
)


def _target_surfaces(prod: Any, cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    collision_fn = getattr(prod, "_target_fit_collision_triangles", None)
    triangles_fn = getattr(prod, "_triangles_from_surface", None)
    if not callable(collision_fn) or not callable(triangles_fn):
        return None
    collision = np.asarray(collision_fn(cache), dtype=np.float64)
    support = cache.get("_ravafit_target_support_triangles")
    if support is None:
        v = np.asarray(cache.get("target_support_V", []), dtype=np.float64)
        f = np.asarray(cache.get("target_support_F", []), dtype=np.int64)
        if v.ndim != 2 or v.shape[1:] != (3,) or f.ndim != 2 or f.shape[1:] != (3,):
            return None
        support = triangles_fn(v, f)
        cache["_ravafit_target_support_triangles"] = support
    support = np.asarray(support, dtype=np.float64)
    if collision.ndim != 3 or collision.shape[1:] != (3, 3) or not len(collision):
        return None
    if support.ndim != 3 or support.shape[1:] != (3, 3) or not len(support):
        return None
    return collision, support


def _dense_penetration_report(prod: Any, source: Any, positions: dict[str, np.ndarray], target_triangles: np.ndarray, margin: float) -> dict[str, Any]:
    occupancy = getattr(prod, "_nearest_literal_occupancy", None)
    if not callable(occupancy):
        return {"enabled": False, "reason": "literal occupancy evaluator unavailable", "penetrating_samples": 0}
    rows = []
    total = 0
    worst = None
    threshold = float(margin) - .00003
    for name in sorted(positions):
        data = source.data(name)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        vertices = np.asarray(positions[name], dtype=np.float64)
        if not len(faces) or vertices.ndim != 2 or vertices.shape[1:] != (3,):
            continue
        samples = np.einsum("bk,fkj->fbj", _DENSE_BARY, vertices[faces]).reshape(-1, 3)
        _, _, signed, _, _ = occupancy(samples, target_triangles, k=48, exact_band=float(margin) + .002)
        signed = np.asarray(signed, dtype=np.float64)
        bad = signed < threshold
        count = int(np.count_nonzero(bad))
        minimum = float(np.min(signed)) if len(signed) else None
        if minimum is not None:
            worst = minimum if worst is None else min(worst, minimum)
        total += count
        rows.append({
            "mesh": name,
            "faces": int(len(faces)),
            "samples": int(len(samples)),
            "penetrating_samples": count,
            "minimum_signed_mm": minimum * 1000 if minimum is not None else None,
        })
    return {
        "enabled": True,
        "margin_mm": float(margin * 1000),
        "penetrating_samples": int(total),
        "minimum_signed_mm": worst * 1000 if worst is not None else None,
        "meshes": rows,
        "policy": "dense face-interior samples are repaired until they remain outside the complete selected target-body collision surface",
    }


def _clip_rows(values: np.ndarray, maximum: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if maximum <= 0.0 or not len(values):
        return values
    lengths = np.linalg.norm(values, axis=1)
    scale = np.ones(len(values), dtype=np.float64)
    mask = lengths > maximum
    scale[mask] = maximum / np.maximum(lengths[mask], 1e-12)
    return values * scale[:, None]


def _dense_face_repair_pass(prod: Any, source: Any, positions: dict[str, np.ndarray], target_triangles: np.ndarray,
                            margin: float, maximum_vertex_step: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Project dense face-interior witnesses out of the target body.

    The normal clearance solver works on garment vertices and connected neighbourhoods. A narrow
    target ridge can still cross the interior of a large garment triangle while all three vertices
    are clear. This pass treats each dense interior witness as a linear barycentric constraint and
    distributes the smallest normal-space correction back to that triangle's three vertices.
    """
    occupancy = getattr(prod, "_nearest_literal_occupancy", None)
    if not callable(occupancy):
        raise ValueError("Dense target-body repair requires the literal occupancy evaluator.")

    out = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    changed = set()
    witness_count = 0
    worst_before = None

    for name in sorted(out):
        data = source.data(name)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        vertices = out[name]
        if not len(faces) or vertices.ndim != 2 or vertices.shape[1:] != (3,):
            continue

        samples = np.einsum("bk,fkj->fbj", _DENSE_BARY, vertices[faces]).reshape(-1, 3)
        nearest, normals, signed, _, _ = occupancy(samples, target_triangles, k=48, exact_band=float(margin) + .003)
        nearest = np.asarray(nearest, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        signed = np.asarray(signed, dtype=np.float64)
        bad = np.flatnonzero(signed < float(margin) - .00003)
        if not len(bad):
            continue

        witness_count += int(len(bad))
        local_worst = float(np.min(signed[bad]))
        worst_before = local_worst if worst_before is None else min(worst_before, local_worst)
        pass_start = vertices.copy()

        # Deepest witnesses first gives the projection a stable outward direction before
        # shallower constraints refine the same patch.
        order = bad[np.argsort(signed[bad], kind="stable")]
        sample_count = len(_DENSE_BARY)
        for flat_index in order.tolist():
            face_index = int(flat_index // sample_count)
            bary_index = int(flat_index % sample_count)
            tri = faces[face_index]
            bary = _DENSE_BARY[bary_index]
            sample = bary @ vertices[tri]

            # Re-evaluate the witness after earlier corrections in this same pass. This is
            # deliberately local/serial: it avoids applying a stale penetration vector after
            # a neighbouring witness has already moved the triangle.
            near, normal, current_signed, _, _ = occupancy(sample[None, :], target_triangles, k=48, exact_band=float(margin) + .003)
            near = np.asarray(near, dtype=np.float64)[0]
            normal = np.asarray(normal, dtype=np.float64)[0]
            current_signed = float(np.asarray(current_signed, dtype=np.float64)[0])
            if current_signed >= float(margin) - .00003:
                continue
            normal_length = float(np.linalg.norm(normal))
            if normal_length <= 1e-12:
                continue
            normal = normal / normal_length
            desired_sample = near + normal * (float(margin) + .00004)
            displacement = desired_sample - sample
            outward = float(np.dot(displacement, normal))
            if outward <= 0.0:
                displacement = normal * max(float(margin) - current_signed + .00004, .00004)
            if np.linalg.norm(displacement) > .004:
                displacement *= .004 / max(float(np.linalg.norm(displacement)), 1e-12)

            denom = float(np.dot(bary, bary))
            if denom <= 1e-12:
                continue
            for corner in range(3):
                vertices[int(tri[corner])] += displacement * (float(bary[corner]) / denom)

        total_delta = _clip_rows(vertices - pass_start, maximum_vertex_step)
        out[name] = pass_start + total_delta
        if np.any(np.linalg.norm(total_delta, axis=1) > 1e-10):
            changed.add(name)

    return out, {
        "changed_meshes": sorted(changed),
        "dense_witnesses_repaired": int(witness_count),
        "worst_signed_before_mm": worst_before * 1000 if worst_before is not None else None,
        "maximum_vertex_step_mm": float(maximum_vertex_step * 1000),
    }


def _repair_until_dense_clear(prod: Any, source: Any, positions: dict[str, np.ndarray], target_collision: np.ndarray,
                              target_support: np.ndarray, clear: Any, margin: float = .00012) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Turn dense occupancy from a rejection condition into a corrective invariant."""
    candidate = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    preserve_seams = getattr(prod, "_preserve_final_source_shared_seams", None)
    passes = []

    # Ordinary residuals should disappear in the first few iterations. The second phase is
    # intentionally more permissive for difficult body changes, but it still moves only along
    # proven target-body occupancy corrections rather than adding generic garment padding.
    for pass_index in range(48):
        validation = _dense_penetration_report(prod, source, candidate, target_collision, margin)
        if not validation.get("enabled", False):
            raise ValueError(f"Final target-body occupancy validation unavailable: {validation.get('reason', 'unknown reason')}")
        remaining = int(validation.get("penetrating_samples", 0))
        if remaining == 0:
            return candidate, {"passes": passes, "validation": validation, "converged": True}

        step = .00125 if pass_index < 12 else (.00200 if pass_index < 28 else .00300)
        repaired, direct_report = _dense_face_repair_pass(
            prod, source, candidate, target_collision, margin=max(float(margin), .00016), maximum_vertex_step=step,
        )
        if callable(preserve_seams):
            repaired, seam_report = preserve_seams(source, repaired)
        else:
            seam_report = {"enabled": False, "reason": "source seam synchronizer unavailable"}

        # Re-run the established seam-aware connected-cloth clearance after every dense
        # interior projection. This spreads the correction over the fitted garment and keeps
        # literal target occupancy authoritative without shrink-wrapping local anatomy.
        repaired, clearance_report = clear(
            source, repaired, target_collision, target_support,
            margin_m=max(.00035, float(margin) + .00012),
            maximum_vertex_move_m=.012 if pass_index < 28 else .020,
            maximum_step_m=step,
            max_passes=12 if pass_index < 28 else 20,
        )
        candidate = {name: np.asarray(value, dtype=np.float64) for name, value in repaired.items()}
        passes.append({
            "pass": int(pass_index + 1),
            "penetrating_samples_before": remaining,
            "direct": direct_report,
            "seams": seam_report,
            "clearance": clearance_report,
        })

    # This is a structural/numerical impossibility guard, not a normal clipping outcome. Valid
    # conversions are expected to converge above; we never intentionally emit a known-clipping
    # garment merely to avoid an exception.
    final_validation = _dense_penetration_report(prod, source, candidate, target_collision, margin)
    if int(final_validation.get("penetrating_samples", 0)):
        raise ValueError(
            "Dense target-body repair did not converge after 48 corrective passes. "
            "The garment was not emitted because returning a knowingly clipping model would violate RavaFit's correctness invariant."
        )
    return candidate, {"passes": passes, "validation": final_validation, "converged": True}


def install_dense_final_target_occupancy(prod: Any) -> None:
    if getattr(prod, "_ravafit_dense_final_target_occupancy_installed", False):
        return
    original = getattr(prod, "_finalize_modded_coupled_solution", None)
    clear = getattr(prod, "_final_target_body_clearance", None)
    if not callable(original) or not callable(clear):
        return

    def guarded(source, cache, positions, skinning, records, contexts, local_affine_quality_rms_mm):
        candidate, skinning_out, records_out, stats = original(
            source, cache, positions, skinning, records, contexts, local_affine_quality_rms_mm,
        )
        surfaces = _target_surfaces(prod, cache)
        if surfaces is None:
            raise ValueError("Final target-body occupancy repair cannot resolve the selected target collision/support surfaces.")
        target_collision, target_support = surfaces

        # First use the established connected-cloth/seam-aware solver. Dense face-interior repair
        # is only invoked for the residual cases that vertex/contact sampling cannot see.
        corrected, clearance_report = clear(
            source, candidate, target_collision, target_support,
            margin_m=.00035, maximum_vertex_move_m=.008, maximum_step_m=.00125, max_passes=10,
        )
        before = {name: np.asarray(value, dtype=np.float64) for name, value in candidate.items()}
        candidate = {name: np.asarray(value, dtype=np.float64) for name, value in corrected.items()}

        initial_validation = _dense_penetration_report(prod, source, candidate, target_collision, margin=.00012)
        repair_report = {"passes": [], "validation": initial_validation, "converged": True}
        if int(initial_validation.get("penetrating_samples", 0)) > 0:
            candidate, repair_report = _repair_until_dense_clear(
                prod, source, candidate, target_collision, target_support, clear, margin=.00012,
            )

        changed = [
            name for name in candidate
            if name in before and np.any(np.linalg.norm(candidate[name] - before[name], axis=1) > 1e-8)
        ]
        for name in changed:
            records_out.setdefault(name, {})["dense_final_target_occupancy_applied"] = True

        merged = dict(stats or {})
        merged["dense_final_target_occupancy"] = {
            "changed_meshes": changed,
            "changed_mesh_count": len(changed),
            "clearance": clearance_report,
            "initial_validation": initial_validation,
            "repair": repair_report,
            "validation": repair_report.get("validation", initial_validation),
            "policy": "repair residual target-body penetration to a correct output; never use rejection as the ordinary clipping strategy",
        }
        return candidate, skinning_out, records_out, merged

    prod._finalize_modded_coupled_solution = guarded
    prod._ravafit_dense_final_target_occupancy_installed = True
