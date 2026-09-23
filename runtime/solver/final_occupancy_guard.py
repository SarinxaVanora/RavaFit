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
        "policy": "literal target anatomy is occupancy authority only; residual garment repair bridges local relief using the smooth target support field",
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


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _bad_face_components(faces: np.ndarray, bad_faces: np.ndarray) -> list[np.ndarray]:
    """Connected penetrating face patches, using shared garment vertices as adjacency."""
    bad_faces = np.asarray(sorted({int(value) for value in np.asarray(bad_faces).reshape(-1).tolist()}), dtype=np.int64)
    if not len(bad_faces):
        return []
    vertex_to_faces: dict[int, list[int]] = {}
    bad_set = set(bad_faces.tolist())
    for face_index in bad_faces.tolist():
        for vertex in np.asarray(faces[face_index], dtype=np.int64).tolist():
            vertex_to_faces.setdefault(int(vertex), []).append(int(face_index))
    components: list[np.ndarray] = []
    remaining = set(bad_set)
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            face_index = stack.pop()
            for vertex in np.asarray(faces[face_index], dtype=np.int64).tolist():
                for neighbour in vertex_to_faces.get(int(vertex), []):
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
                        component.append(neighbour)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _dense_face_repair_pass(prod: Any, source: Any, positions: dict[str, np.ndarray], target_collision: np.ndarray,
                            target_support: np.ndarray, margin: float, maximum_vertex_step: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bridge dense face-interior body penetration without embossing literal anatomy.

    The literal selected body answers only whether garment surface is occupied. It never supplies
    the correction shape. Penetrating faces are grouped into connected patches and translated along
    the smooth target-support normal field by the deepest required clearance in that patch. This is
    intentionally different from projecting each witness onto the literal body: doing that imprints
    nipples, genital folds, grooves and other local anatomy into close clothing.
    """
    occupancy = getattr(prod, "_nearest_literal_occupancy", None)
    if not callable(occupancy):
        raise ValueError("Dense target-body repair requires the occupancy evaluator.")

    out = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    changed = set()
    witness_count = 0
    patch_count = 0
    worst_before = None

    for name in sorted(out):
        data = source.data(name)
        faces = np.asarray(data.get("F", []), dtype=np.int64)
        vertices = out[name]
        if not len(faces) or vertices.ndim != 2 or vertices.shape[1:] != (3,):
            continue

        samples_by_face = np.einsum("bk,fkj->fbj", _DENSE_BARY, vertices[faces])
        samples = samples_by_face.reshape(-1, 3)
        _, _, signed, _, _ = occupancy(samples, target_collision, k=48, exact_band=float(margin) + .003)
        signed = np.asarray(signed, dtype=np.float64).reshape(len(faces), len(_DENSE_BARY))
        bad_mask = signed < float(margin) - .00003
        bad_faces = np.flatnonzero(np.any(bad_mask, axis=1))
        if not len(bad_faces):
            continue

        witness_count += int(np.count_nonzero(bad_mask))
        local_worst = float(np.min(signed[bad_mask]))
        worst_before = local_worst if worst_before is None else min(worst_before, local_worst)
        pass_start = vertices.copy()
        accumulated = np.zeros_like(vertices)
        counts = np.zeros(len(vertices), dtype=np.float64)

        for component in _bad_face_components(faces, bad_faces):
            component_mask = bad_mask[component]
            required = np.maximum(float(margin) - signed[component] + .00004, 0.0)
            required = float(np.max(required[component_mask], initial=0.0))
            if required <= 0.0:
                continue
            required = min(required, .0040)
            component_vertices = np.unique(faces[component].reshape(-1))
            points = vertices[component_vertices]

            # The support proxy is the garment-shaping authority. Its local normal follows target
            # macro anatomy while bridging high-frequency literal details that fabric should not trace.
            _, support_normals, _, _, _ = occupancy(points, target_support, k=48, exact_band=.006)
            support_normals = _normalise_rows(np.asarray(support_normals, dtype=np.float64))
            _, literal_normals, _, _, _ = occupancy(points, target_collision, k=48, exact_band=.006)
            literal_normals = _normalise_rows(np.asarray(literal_normals, dtype=np.float64))
            flip = np.einsum("ij,ij->i", support_normals, literal_normals) < 0.0
            support_normals[flip] *= -1.0

            # If a support normal is numerically unusable, fall back to the fitted garment patch
            # normal, oriented outward by the literal body. This still avoids literal micro-relief.
            bad_normal = np.linalg.norm(support_normals, axis=1) < .5
            if np.any(bad_normal):
                tris = vertices[faces[component]]
                face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
                face_normals = _normalise_rows(face_normals)
                fallback = np.mean(face_normals, axis=0)
                if np.linalg.norm(fallback) <= 1e-12:
                    fallback = np.mean(literal_normals, axis=0)
                fallback = fallback / max(float(np.linalg.norm(fallback)), 1e-12)
                if float(np.dot(fallback, np.mean(literal_normals, axis=0))) < 0.0:
                    fallback *= -1.0
                support_normals[bad_normal] = fallback

            delta = support_normals * required
            accumulated[component_vertices] += delta
            counts[component_vertices] += 1.0
            patch_count += 1

        active = counts > 0.0
        if not np.any(active):
            continue
        delta = np.zeros_like(vertices)
        delta[active] = accumulated[active] / counts[active, None]
        delta = _clip_rows(delta, maximum_vertex_step)
        out[name] = pass_start + delta
        if np.any(np.linalg.norm(delta, axis=1) > 1e-10):
            changed.add(name)

    return out, {
        "changed_meshes": sorted(changed),
        "dense_witnesses_repaired": int(witness_count),
        "connected_relief_patches": int(patch_count),
        "worst_signed_before_mm": worst_before * 1000 if worst_before is not None else None,
        "maximum_vertex_step_mm": float(maximum_vertex_step * 1000),
        "repair_direction": "smooth_target_support_normal",
        "literal_surface_used_for_detection_only": True,
        "policy": "bridge target micro-relief as connected cloth patches; never project garment vertices onto literal anatomical relief",
    }


def _repair_until_dense_clear(prod: Any, source: Any, positions: dict[str, np.ndarray], target_collision: np.ndarray,
                              target_support: np.ndarray, clear: Any, margin: float = .00012) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Turn dense occupancy into a corrective invariant while preserving cloth-like support shape."""
    candidate = {name: np.asarray(value, dtype=np.float64).copy() for name, value in positions.items()}
    preserve_seams = getattr(prod, "_preserve_final_source_shared_seams", None)
    passes = []

    for pass_index in range(48):
        validation = _dense_penetration_report(prod, source, candidate, target_collision, margin)
        if not validation.get("enabled", False):
            raise ValueError(f"Final target-body occupancy validation unavailable: {validation.get('reason', 'unknown reason')}")
        remaining = int(validation.get("penetrating_samples", 0))
        if remaining == 0:
            return candidate, {"passes": passes, "validation": validation, "converged": True}

        step = .00125 if pass_index < 12 else (.00200 if pass_index < 28 else .00300)
        repaired, direct_report = _dense_face_repair_pass(
            prod, source, candidate, target_collision, target_support,
            margin=max(float(margin), .00016), maximum_vertex_step=step,
        )
        if callable(preserve_seams):
            repaired, seam_report = preserve_seams(source, repaired)
        else:
            seam_report = {"enabled": False, "reason": "source seam synchronizer unavailable"}

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
            "policy": "repair literal target occupancy using smooth garment-support shape; literal anatomy detects penetration but never imprints its local relief into cloth",
        }
        return candidate, skinning_out, records_out, merged

    prod._finalize_modded_coupled_solution = guarded
    prod._ravafit_dense_final_target_occupancy_installed = True
