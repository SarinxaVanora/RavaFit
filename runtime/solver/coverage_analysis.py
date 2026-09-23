from __future__ import annotations

from pathlib import Path
from typing import Any

import re
import numpy as np

from ffxiv_lobofit import GLB, skeleton_global_positions

_SLOT_SUFFIX = {
    "_top.mdl": "Chest",
    "_dwn.mdl": "Legs",
    "_glv.mdl": "Hands",
    "_sho.mdl": "Feet",
}
_ACCESSORY_SUFFIX = {
    "_ear.mdl": "Earrings",
    "_nek.mdl": "Necklace",
    "_wrs.mdl": "Wrists",
    "_rir.mdl": "Right Ring",
    "_ril.mdl": "Left Ring",
}


def _model_slot(game_path: str) -> str | None:
    path = str(game_path or "").replace("\\", "/").casefold()
    for suffix, slot in _SLOT_SUFFIX.items():
        if path.endswith(suffix):
            return slot
    if any(path.endswith(suffix) for suffix in _ACCESSORY_SUFFIX):
        return None
    raise ValueError(f"Could not determine XIV equipment/accessory slot from model path: {game_path}")


def _container_slot(game_path: str) -> str:
    path = str(game_path or "").replace("\\", "/").casefold()
    for suffix, slot in _SLOT_SUFFIX.items():
        if path.endswith(suffix): return slot
    for suffix, slot in _ACCESSORY_SUFFIX.items():
        if path.endswith(suffix): return slot
    raise ValueError(f"Could not determine XIV equipment/accessory slot from model path: {game_path}")


def _bone_region(name: str) -> str | None:
    """Map XIV skeleton names to broad body-library regions."""
    n = str(name or "").casefold()

    if n.startswith((
        "j_asi_a_", "j_asi_b_", "j_asi_c_",
        "iv_daitai", "ya_daitai", "iv_shiri", "ya_shiri",
        "j_sk_", "n_hiza",
    )):
        return "Legs"

    # Feet stay separate from legs because XIV treats them as a separate slot.
    if n.startswith(("j_asi_d_", "j_asi_e_", "iv_asi_", "ya_asi_")):
        return "Feet"

    if n.startswith((
        "j_te_", "j_hito_", "j_ko_", "j_kusu_", "j_naka_", "j_oya_",
        "iv_hito_", "iv_ko_", "iv_kusu_", "iv_naka_", "iv_oya_",
    )) and "asi_" not in n:
        return "Hands"

    if n.startswith((
        "j_sebo_", "j_mune_", "j_sako_", "j_ude_", "j_kata_", "j_kubi",
        "iv_mune", "iv_sako", "iv_ude", "ya_mune", "ya_sako", "ya_ude",
    )):
        return "Chest"

    return None


def _first_data(glb: GLB):
    for name in glb.mesh_names():
        if not name:
            continue
        try:
            data = glb.data(name)
        except Exception:
            continue
        if len(data["V"]) and len(data["F"]):
            return data
    raise ValueError("Selected GLB contains no skinned triangle geometry to analyse.")


def _joint_position_map(glb: GLB, joint_names: list[str]) -> dict[str, np.ndarray]:
    positions, _, _ = skeleton_global_positions(glb, joint_names)
    return {str(name): np.asarray(positions[i], dtype=np.float64) for i, name in enumerate(joint_names)}


def _mean_joint_y(joints: dict[str, np.ndarray], names: tuple[str, ...]) -> float | None:
    values = [float(joints[name][1]) for name in names if name in joints]
    return float(np.mean(values)) if values else None


def _joint_points(joints: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    values = [joints[name] for name in names if name in joints]
    return np.asarray(values, dtype=np.float64) if values else np.empty((0, 3), dtype=np.float64)


def _is_obvious_embedded_body_material(material: str | None) -> bool:
    # Ignore embedded body meshes when measuring garment coverage.
    value = str(material or "").replace("\\", "/").casefold()
    marker = value.rfind("/mt_c")
    if marker < 0:
        marker = value.find("mt_c")
    if marker < 0:
        return False
    tail = value[marker + (1 if value[marker] == "/" else 0):]
    return len(tail) >= 14 and tail.startswith("mt_c") and tail[4:8].isdigit() and tail[8] == "b" and tail[9:13].isdigit()


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def analyse_coverage(glb_path: str | Path, game_path: str) -> dict[str, Any]:
    """Infer which body regions can materially affect the selected garment."""
    source = GLB(Path(glb_path))
    primary_hint = _model_slot(game_path)
    container_slot = _container_slot(game_path)
    first = _first_data(source)
    joints = _joint_position_map(source, list(first["joint_names"]))

    # Build landmarks from the exported rig, not fixed world-space heights.
    pelvis_y = _mean_joint_y(joints, ("j_kosi", "j_sebo_a"))
    thigh_y = _mean_joint_y(joints, ("j_asi_a_l", "j_asi_a_r"))
    knee_y = _mean_joint_y(joints, ("j_asi_b_l", "j_asi_b_r"))
    ankle_y = _mean_joint_y(joints, ("j_asi_d_l", "j_asi_d_r"))
    spine_b_y = _mean_joint_y(joints, ("j_sebo_b",))

    hand_points = _joint_points(joints, ("j_te_l", "j_te_r"))
    foot_points = _joint_points(joints, ("j_asi_d_l", "j_asi_d_r", "j_asi_e_l", "j_asi_e_r"))

    total_area = 0.0
    region_mass = {slot: 0.0 for slot in ("Chest", "Legs", "Hands", "Feet")}
    lower_entry_area = 0.0
    lower_deep_area = 0.0
    upper_torso_area = 0.0
    hand_near_area = 0.0
    foot_near_area = 0.0
    analysed_meshes = 0

    # Use the hip/upper-thigh midpoint without letting a waist seam count as legs.
    leg_entry_y = None if pelvis_y is None or thigh_y is None else float((pelvis_y + thigh_y) * 0.5)
    deep_leg_y = None if thigh_y is None else float(thigh_y + 0.02)
    upper_torso_y = None
    if pelvis_y is not None and spine_b_y is not None:
        upper_torso_y = float(pelvis_y + (spine_b_y - pelvis_y) * 0.45)

    mesh_rows: list[tuple[str, dict[str, Any]]] = []
    body_like_rows: list[tuple[str, dict[str, Any]]] = []
    for name in source.mesh_names():
        if not name:
            continue
        try:
            data = source.data(name)
        except Exception:
            # Non-skinned helper nodes are not body-context evidence.
            continue
        row = (name, data)
        if _is_obvious_embedded_body_material(data.get("material")):
            body_like_rows.append(row)
        else:
            mesh_rows.append(row)

    # If everything looks body-like, analyse it rather than returning no evidence.
    analysed_rows = mesh_rows if mesh_rows else body_like_rows

    for name, data in analysed_rows:
        vertices = np.asarray(data["V"], dtype=np.float64)
        faces = np.asarray(data["F"], dtype=np.int64)
        weights = np.asarray(data["W"], dtype=np.float64)
        if not len(vertices) or not len(faces):
            continue

        analysed_meshes += 1
        tri = vertices[faces]
        tri_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
        if not np.any(tri_area > 0):
            continue
        vertex_area = np.zeros(len(vertices), dtype=np.float64)
        for corner in range(3):
            np.add.at(vertex_area, faces[:, corner], tri_area / 3.0)
        area = float(np.sum(vertex_area))
        total_area += area

        names = list(data["joint_names"])
        for slot in region_mass:
            ids = [i for i, joint_name in enumerate(names) if _bone_region(joint_name) == slot]
            if ids:
                region_mass[slot] += float(np.sum(vertex_area * np.sum(weights[:, ids], axis=1)))

        centres = np.mean(tri, axis=1)
        if leg_entry_y is not None:
            lower_entry_area += float(np.sum(tri_area[centres[:, 1] < leg_entry_y]))
        if deep_leg_y is not None:
            lower_deep_area += float(np.sum(tri_area[centres[:, 1] < deep_leg_y]))
        if upper_torso_y is not None:
            upper_torso_area += float(np.sum(tri_area[centres[:, 1] > upper_torso_y]))

        if len(hand_points):
            distances = np.min(np.linalg.norm(centres[:, None, :] - hand_points[None, :, :], axis=2), axis=1)
            hand_near_area += float(np.sum(tri_area[distances < 0.075]))
        if len(foot_points):
            distances = np.min(np.linalg.norm(centres[:, None, :] - foot_points[None, :, :], axis=2), axis=1)
            foot_near_area += float(np.sum(tri_area[distances < 0.105]))

    if total_area <= 1e-12:
        raise ValueError("Selected GLB has no measurable skinned surface area.")

    mass = {slot: float(value / total_area) for slot, value in region_mass.items()}
    lower_entry = float(lower_entry_area / total_area)
    lower_deep = float(lower_deep_area / total_area)
    upper_torso = float(upper_torso_area / total_area)
    hand_near = float(hand_near_area / total_area)
    foot_near = float(foot_near_area / total_area)

    # These are evidence scores, not garment-type rules; stay conservative.
    scores = {
        "Chest": max(mass["Chest"] / 0.12, upper_torso / 0.18),
        "Legs": max(mass["Legs"] / 0.10, lower_entry / 0.08, lower_deep / 0.025),
        "Hands": max(mass["Hands"] / 0.10, hand_near / 0.085),
        "Feet": max(mass["Feet"] / 0.10, foot_near / 0.040),
    }
    # Accessory slot names do not determine fitting support; use rig and spatial coverage.
    primary = primary_hint or max(scores, key=lambda slot: (scores[slot], mass[slot]))

    result: dict[str, Any] = {}
    for slot in ("Chest", "Legs", "Hands", "Feet"):
        if slot == primary:
            inferred_accessory = primary_hint is None
            result[slot] = {
                "recommended": True,
                "primary": True,
                "confidence": 1.0 if not inferred_accessory else _clamp01(max(0.60, scores[slot])),
                "reason": "Primary XIV model slot" if not inferred_accessory else f"Primary body support inferred for {container_slot} accessory geometry",
            }
            continue

        confidence = _clamp01(scores[slot])
        recommended = confidence >= 0.58
        reasons: list[str] = []
        if slot == "Legs":
            if lower_entry >= 0.018:
                reasons.append("garment reaches the hip / upper-thigh envelope")
            if lower_deep >= 0.004:
                reasons.append("garment extends below the upper-thigh boundary")
            if mass["Legs"] >= 0.025:
                reasons.append("lower-body / skirt bone influence")
        elif slot == "Chest":
            if upper_torso >= 0.08:
                reasons.append("garment extends into the torso envelope")
            if mass["Chest"] >= 0.04:
                reasons.append("torso / arm bone influence")
        elif slot == "Hands":
            if hand_near >= 0.035:
                reasons.append("garment reaches the wrist / hand envelope")
            if mass["Hands"] >= 0.02:
                reasons.append("hand / finger bone influence")
        elif slot == "Feet":
            if foot_near >= 0.015:
                reasons.append("garment reaches the ankle / foot envelope")
            if mass["Feet"] >= 0.02:
                reasons.append("ankle / foot bone influence")

        if not reasons:
            reason = "No strong interaction detected"
        elif recommended:
            reason = " + ".join(reasons[:2])
        else:
            reason = "Possible: " + " + ".join(reasons[:2])

        result[slot] = {
            "recommended": bool(recommended),
            "primary": False,
            "confidence": confidence,
            "reason": reason,
        }

    return {
        "ok": True,
        "primary_slot": primary,
        "container_slot": container_slot,
        "source_contains_body": bool(body_like_rows),
        "slots": result,
        "metrics": {
            "mesh_count": analysed_meshes,
            "ignored_embedded_body_meshes": len(body_like_rows) if mesh_rows else 0,
            "surface_area": total_area,
            "region_weight_mass": mass,
            "lower_entry_area_fraction": lower_entry,
            "lower_deep_area_fraction": lower_deep,
            "upper_torso_area_fraction": upper_torso,
            "hand_near_area_fraction": hand_near,
            "foot_near_area_fraction": foot_near,
            "landmarks": {
                "pelvis_y": pelvis_y,
                "thigh_y": thigh_y,
                "knee_y": knee_y,
                "ankle_y": ankle_y,
            },
        },
    }


def _normalise_material(value: str | None) -> str:
    return str(value or "").replace("\\", "/").strip().casefold()


_BODY_PACKAGE_MATERIAL_RE = re.compile(r"(?:^|/)mt_c\d{4}b(?P<body>\d{4})_(?P<name>[^/]+?)\.mtrl$")


def _body_package_material_key(material: str | None) -> tuple[str, str] | None:
    value = _normalise_material(material)
    match = _BODY_PACKAGE_MATERIAL_RE.search(value)
    return None if match is None else (match.group("body"), match.group("name"))


def _body_material_sets_overlap(left: set[str], right: set[str]) -> bool:
    if left.intersection(right):
        return True
    left_keys = {_body_package_material_key(value) for value in left}
    right_keys = {_body_package_material_key(value) for value in right}
    left_keys.discard(None)
    right_keys.discard(None)
    return bool(left_keys.intersection(right_keys))


def _sample_points(vertices: np.ndarray, limit: int = 4096) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) <= limit:
        return vertices
    ids = np.linspace(0, len(vertices) - 1, limit, dtype=np.int64)
    return vertices[ids]


def _shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return a scale-aware symmetric surface distance for aligned XIV body geometry."""
    from scipy.spatial import cKDTree

    a = _sample_points(a)
    b = _sample_points(b)
    if len(a) < 16 or len(b) < 16:
        return float("inf")
    ta = cKDTree(a)
    tb = cKDTree(b)
    da = tb.query(a, k=1)[0]
    db = ta.query(b, k=1)[0]
    diag = max(float(np.linalg.norm(np.maximum(a.max(axis=0), b.max(axis=0)) - np.minimum(a.min(axis=0), b.min(axis=0)))), 1e-6)
    distance = float(np.median(da) + np.median(db) + 0.25 * (np.percentile(da, 90) + np.percentile(db, 90)))
    return distance / diag


def _interface_distance(primary_vertices: np.ndarray, candidate_vertices: np.ndarray, target_slot: str) -> float:
    """Compare only the anatomical boundary shared by adjacent body slots."""
    from scipy.spatial import cKDTree

    a = np.asarray(primary_vertices, dtype=np.float64)
    b = np.asarray(candidate_vertices, dtype=np.float64)
    if len(a) < 16 or len(b) < 16:
        return float("inf")

    target = str(target_slot).casefold()
    if target == "legs":
        a_cut = float(np.quantile(a[:, 1], 0.34))
        b_cut = float(np.quantile(b[:, 1], 0.72))
        aa = a[a[:, 1] <= a_cut]
        bb = b[b[:, 1] >= b_cut]
    elif target == "hands":
        ax = np.abs(a[:, 0])
        bx = np.abs(b[:, 0])
        aa = a[ax >= np.quantile(ax, 0.80)]
        bb = b[bx >= np.quantile(bx, 0.55)]
    elif target == "feet":
        a_cut = float(np.quantile(a[:, 1], 0.25))
        b_cut = float(np.quantile(b[:, 1], 0.75))
        aa = a[a[:, 1] <= a_cut]
        bb = b[b[:, 1] >= b_cut]
    else:
        return _shape_distance(a, b)

    aa = _sample_points(aa, 2500)
    bb = _sample_points(bb, 2500)
    if len(aa) < 16 or len(bb) < 16:
        return float("inf")
    ta = cKDTree(aa)
    tb = cKDTree(bb)
    da = tb.query(aa, k=1)[0]
    db = ta.query(bb, k=1)[0]
    diag = max(float(np.linalg.norm(np.maximum(aa.max(axis=0), bb.max(axis=0)) - np.minimum(aa.min(axis=0), bb.min(axis=0)))), 1e-6)
    return float(np.quantile(da, 0.25) + np.quantile(db, 0.25) + 0.25 * (np.median(da) + np.median(db))) / diag


def _preferred_candidate_label(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rank(row: dict[str, Any]):
        variant = str(row.get("variant") or "")
        lowered = variant.casefold()
        noisy = sum(token in lowered for token in ("the clothes", "emperor", "required files"))
        return noisy, len(variant), variant.casefold()
    return sorted(rows, key=rank)[0]


def _variant_tokens(value: str | None) -> set[str]:
    import re

    tokens = {x for x in re.split(r"[^a-z0-9]+", str(value or "").casefold()) if len(x) >= 2}
    # Ignore packaging noise, but keep body/size labels that can distinguish variants.
    return tokens.difference({"the", "and", "with", "without", "required", "files", "file", "races", "race", "emperor", "core", "add", "ons"})


def _slot_surface_points(source: GLB, rows: list[dict[str, Any]], slot: str, *, direct_body: bool = False) -> np.ndarray:
    """Extract the garment surface that provides evidence for one body region."""
    collected: list[np.ndarray] = []
    fallback_vertices: list[np.ndarray] = []
    fallback_points: list[np.ndarray] = []

    for data in rows:
        vertices = np.asarray(data.get("V", []), dtype=np.float64)
        weights = np.asarray(data.get("W", []), dtype=np.float64)
        names = list(data.get("joint_names", []))
        if len(vertices) == 0:
            continue
        fallback_vertices.append(vertices)

        ids = [i for i, joint_name in enumerate(names) if _bone_region(joint_name) == slot]
        if ids and weights.ndim == 2 and weights.shape[0] == len(vertices):
            region_weight = np.sum(weights[:, ids], axis=1)
            threshold = 0.42 if direct_body else 0.035
            selected = vertices[region_weight >= threshold]
            if len(selected):
                collected.append(selected)

        try:
            positions, _, _ = skeleton_global_positions(source, names)
            points = np.asarray([positions[i] for i, joint_name in enumerate(names) if _bone_region(joint_name) == slot], dtype=np.float64)
            if len(points):
                fallback_points.append(points)
        except Exception:
            pass

    if collected:
        points = np.vstack(collected)
        if len(points) >= (48 if direct_body else 72):
            return _sample_points(points, 6000)

    if direct_body or not fallback_vertices or not fallback_points:
        return np.empty((0, 3), dtype=np.float64)

    vertices = np.vstack(fallback_vertices)
    anchors = np.vstack(fallback_points)
    if len(vertices) < 16 or len(anchors) == 0:
        return np.empty((0, 3), dtype=np.float64)

    from scipy.spatial import cKDTree

    tree = cKDTree(anchors)
    distance = tree.query(vertices, k=1)[0]
    # Scale the contact threshold from the rig and keep only the nearest third of the garment.
    anchor_extent = max(float(np.linalg.norm(anchors.max(axis=0) - anchors.min(axis=0))), 0.08)
    threshold = max(0.055, min(0.22, anchor_extent * (0.75 if slot in ("Hands", "Feet") else 0.95)))
    mask = distance <= threshold
    if np.count_nonzero(mask) < 72:
        cutoff = float(np.quantile(distance, min(0.35, max(72 / max(len(distance), 1), 0.08))))
        mask = distance <= cutoff
    return _sample_points(vertices[mask], 6000) if np.count_nonzero(mask) >= 24 else np.empty((0, 3), dtype=np.float64)


def _garment_candidate_score(garment_points: np.ndarray, candidate_ref: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Score how plausibly an RBODY surface fits the garment's authored envelope."""
    from scipy.spatial import cKDTree

    garment = _sample_points(np.asarray(garment_points, dtype=np.float64), 5000)
    body = np.asarray(candidate_ref.get("V", []), dtype=np.float64)
    normals = np.asarray(candidate_ref.get("N", []), dtype=np.float64)
    if len(garment) < 24 or len(body) < 24:
        return float("inf"), {}

    body_tree = cKDTree(body)
    garment_to_body, index = body_tree.query(garment, k=1)
    diag = max(float(np.linalg.norm(body.max(axis=0) - body.min(axis=0))), 0.03)

    # Score only the candidate-body region the garment can actually observe.
    gmin = garment.min(axis=0)
    gmax = garment.max(axis=0)
    gextent = np.maximum(gmax - gmin, 0.02)
    margin = np.maximum(0.02, gextent * 0.12)
    overlap = np.all((body >= gmin - margin) & (body <= gmax + margin), axis=1)
    body_overlap = body[overlap]
    if len(body_overlap) < 24:
        body_overlap = body
    body_overlap = _sample_points(body_overlap, 5000)
    garment_tree = cKDTree(garment)
    body_to_garment = garment_tree.query(body_overlap, k=1)[0]

    g50 = float(np.median(garment_to_body))
    b50 = float(np.median(body_to_garment))
    g85 = float(np.percentile(garment_to_body, 85))
    b85 = float(np.percentile(body_to_garment, 85))
    p50 = (g50 + b50) / (2.0 * diag)
    p85 = (g85 + b85) / (2.0 * diag)
    gmad = float(np.median(np.abs(garment_to_body - g50)))
    bmad = float(np.median(np.abs(body_to_garment - b50)))
    mad = (gmad + bmad) / (2.0 * diag)

    penetration = 0.0
    off_normal = 0.0
    if normals.shape == body.shape:
        delta = garment - body[index]
        n = normals[index]
        signed = np.sum(delta * n, axis=1)
        tolerance = max(0.0015, diag * 0.0015)
        penetration = float(np.mean(signed < -tolerance))
        cosine = signed / np.maximum(garment_to_body, 1e-7)
        off_normal = float(np.mean(cosine < -0.05))

    contact_limit = max(0.035, diag * 0.050)
    contact_fraction = float(np.mean(garment_to_body <= contact_limit))
    score = p50 + 0.35 * p85 + 0.45 * mad + 1.00 * penetration + 0.12 * off_normal
    return float(score), {
        "p50": float(p50),
        "p85": float(p85),
        "mad": float(mad),
        "penetration": penetration,
        "contact_fraction": contact_fraction,
        "overlap_body_vertices": float(len(body_overlap)),
    }

def _cross_slot_interface_distance(anchor_ref: dict[str, Any], anchor_slot: str, candidate_ref: dict[str, Any], candidate_slot: str) -> float:
    a = np.asarray(anchor_ref.get("V", []), dtype=np.float64)
    b = np.asarray(candidate_ref.get("V", []), dtype=np.float64)
    if len(a) < 16 or len(b) < 16:
        return float("inf")
    pair = (str(anchor_slot), str(candidate_slot))
    if pair == ("Chest", "Legs"):
        return _interface_distance(a, b, "Legs")
    if pair == ("Legs", "Chest"):
        return _interface_distance(b, a, "Legs")
    if pair == ("Chest", "Hands"):
        return _interface_distance(a, b, "Hands")
    if pair == ("Hands", "Chest"):
        return _interface_distance(b, a, "Hands")
    if pair == ("Legs", "Feet"):
        return _interface_distance(a, b, "Feet")
    if pair == ("Feet", "Legs"):
        return _interface_distance(b, a, "Feet")
    return float("inf")



_HINT_SIZE_ALIASES = {
    "xs": "xs", "extra-small": "xs", "extrasmall": "xs",
    "s": "small", "sm": "small", "small": "small",
    "m": "medium", "med": "medium", "medium": "medium",
    "l": "large", "lg": "large", "large": "large",
    "xl": "xl", "xxl": "xxl",
    "nb": "neobelly", "neo-belly": "neobelly",
    "wc": "watermelon", "sc": "skull",
}
_HINT_GENERIC = {
    "top", "bottom", "chest", "legs", "leg", "body", "size", "sizes", "option", "options",
    "model", "models", "outfit", "clothes", "clothing", "smallclothes", "the", "required", "files",
}
_PENILE_HINTS = {"penis", "cock", "dick", "cut", "uncut", "erect", "flaccid", "shaft"}


def _body_hint_tokens(value: Any) -> set[str]:
    import re
    raw = re.findall(r"[a-z0-9]+", str(value or "").casefold().replace("neo belly", "neobelly"))
    out: set[str] = set()
    for token in raw:
        token = _HINT_SIZE_ALIASES.get(token, token)
        if token and token not in _HINT_GENERIC:
            out.add(token)
    return out


def _candidate_hint_tokens(row: dict[str, Any]) -> set[str]:
    values = (row.get("collection"), row.get("body"), row.get("body_id"), row.get("variant"), row.get("variant_id"))
    out: set[str] = set()
    for value in values:
        out.update(_body_hint_tokens(value))
    return out


def _candidate_family_tokens(row: dict[str, Any]) -> set[str]:
    """Tokens that identify the body family, deliberately excluding variant/collection labels."""
    values = (row.get("body"), row.get("body_id"))
    out: set[str] = set()
    for value in values:
        out.update(_body_hint_tokens(value))
    return {token for token in out if len(token) >= 3 and token not in _PENILE_HINTS}


def _hint_token_matches_family(hint_token: str, family_token: str) -> bool:
    if hint_token == family_token:
        return True
    # Body packs are not perfectly consistent about short/long family labels (for example
    # Dionys vs Dionysus). Prefix matching is safe once the distinctive portion is >= 5 chars.
    if min(len(hint_token), len(family_token)) >= 5:
        return hint_token.startswith(family_token) or family_token.startswith(hint_token)
    return False


def _explicit_family_rows(rows: list[dict[str, Any]], hint_text: Any) -> list[dict[str, Any]]:
    hint_tokens = {token for token in _body_hint_tokens(hint_text) if len(token) >= 3 and token not in _PENILE_HINTS}
    if not hint_tokens:
        return []
    matched: list[dict[str, Any]] = []
    for row in rows:
        family_tokens = _candidate_family_tokens(row)
        if any(_hint_token_matches_family(hint, family) for hint in hint_tokens for family in family_tokens):
            matched.append(row)
    return matched


def _is_penile_candidate(row: dict[str, Any]) -> bool:
    return bool(_candidate_hint_tokens(row).intersection(_PENILE_HINTS))


def _source_hint_candidates(rows: list[dict[str, Any]], source_context: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Use Penumbra's selected option/group/mod names before expensive body-geometry scoring.

    An explicit body-family name in the selected option is a constraint, not a weak vote. This
    prevents visually similar body families (for example Dionys and Muse) from defeating what
    the mod author actually labelled. Group and mod names are progressively weaker fallbacks.
    Geometry still chooses between variants inside the identified family when the label does not
    uniquely identify a size/shape.
    """
    context = dict(source_context or {})
    option_text = str(context.get("option") or "")
    group_text = str(context.get("group") or "")
    mod_text = str(context.get("mod") or "")
    option_tokens = _body_hint_tokens(option_text)
    group_tokens = _body_hint_tokens(group_text)
    mod_tokens = _body_hint_tokens(mod_text)
    context_tokens = option_tokens | group_tokens | mod_tokens
    if not rows:
        return rows, {"used": False, "reason": "no candidates"}, None

    candidate_count_before = len(rows)

    # Genital body variants are considered only when the selected naming context provides
    # explicit evidence. This is deliberately independent of family-name narrowing.
    has_penile_hint = bool(context_tokens.intersection(_PENILE_HINTS))
    if not has_penile_hint:
        non_penile = [row for row in rows if not _is_penile_candidate(row)]
        if non_penile:
            rows = non_penile

    # Selected option is strongest, then its group, then the mod name. Stop at the first scope
    # that explicitly names a known candidate body family; lower-priority text must not undo it.
    family_scope = ""
    family_text = ""
    family_rows: list[dict[str, Any]] = []
    for scope, text in (("option", option_text), ("group", group_text), ("mod", mod_text)):
        matches = _explicit_family_rows(rows, text)
        if matches:
            family_scope, family_text, family_rows = scope, text, matches
            rows = matches
            break

    if not context_tokens:
        return rows, {
            "used": bool(len(rows)), "authoritative": False,
            "method": "anatomy-compatible-default" if not has_penile_hint else "no-name-hint",
            "penile_anatomy_explicit": has_penile_hint,
            "candidate_count_before": candidate_count_before,
            "candidate_count_after": len(rows),
        }, None

    # First try the variant label inside the already-selected family. Do not compare the whole
    # option text to a variant name because body-family words are intentionally present too.
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        tokens = _candidate_hint_tokens(row)
        variant_tokens = _body_hint_tokens(row.get("variant")) | _body_hint_tokens(row.get("variant_id"))
        family_tokens = _candidate_family_tokens(row)
        overlap_option = option_tokens.intersection(tokens)
        overlap_group = group_tokens.intersection(tokens)
        overlap_mod = mod_tokens.intersection(family_tokens)
        score = 0.0
        for token in overlap_option:
            score += 2.0 if token in {"small", "medium", "large", "xs", "xl", "a", "b", "c", "gen"} else 4.0
        for token in overlap_group:
            score += 0.5 if token in {"small", "medium", "large", "xs", "xl"} else 1.0
        for token in overlap_mod:
            score += 0.35

        # Within an explicitly named family, a matching size/variant token is strong evidence.
        variant_specific = {t for t in option_tokens if t in variant_tokens and t not in family_tokens}
        score += 5.0 * len(variant_specific)

        # Only penalise distinctive option tokens when no explicit family was found. Once a body
        # family is named, unrelated words like "default", "normal", etc. must not defeat it.
        if not family_rows:
            distinctive = {t for t in option_tokens if t not in {"small", "medium", "large", "xs", "xl", "a", "b", "c", "gen"}}
            if distinctive and not distinctive.issubset(tokens):
                score -= 8.0
        scored.append((score, row))

    best_score = max((score for score, _ in scored), default=0.0)
    if family_rows:
        narrowed = [row for score, row in scored if score >= best_score - 0.25]
        # If only the body family was named, do not pretend a variant is authoritative. Let the
        # geometry scorer resolve the shape from the candidates in that correct family.
        option_variant_tokens = option_tokens - set().union(*(_candidate_family_tokens(row) for row in rows))
        meaningful_variant_tokens = {t for t in option_variant_tokens if t in {"small", "medium", "large", "xs", "xl", "xxl", "a", "b", "c", "gen", "neobelly", "watermelon", "skull"}}
        authoritative = narrowed[0] if len(narrowed) == 1 and meaningful_variant_tokens else None
        return narrowed, {
            "used": True,
            "authoritative": authoritative is not None,
            "method": f"explicit-{family_scope}-body-family",
            "scope": family_scope,
            "hint": family_text,
            "option": option_text,
            "group": group_text,
            "mod": mod_text,
            "penile_anatomy_explicit": has_penile_hint,
            "candidate_count_before": candidate_count_before,
            "candidate_count_after": len(narrowed),
            "best_score": float(best_score),
        }, authoritative

    if best_score < 4.0:
        return rows, {"used": False, "reason": "option-name hint was too weak", "option": option_text}, None

    narrowed = [row for score, row in scored if score >= best_score - 0.25]
    if not has_penile_hint:
        non_penile = [row for row in narrowed if not _is_penile_candidate(row)]
        if non_penile:
            narrowed = non_penile

    authoritative = narrowed[0] if len(narrowed) == 1 and best_score >= 8.0 else None
    return narrowed, {
        "used": True,
        "authoritative": authoritative is not None,
        "method": "option-name-token-narrowing",
        "option": option_text,
        "group": group_text,
        "mod": mod_text,
        "tokens": sorted(context_tokens),
        "penile_anatomy_explicit": has_penile_hint,
        "candidate_count_before": candidate_count_before,
        "candidate_count_after": len(narrowed),
        "best_score": float(best_score),
    }, authoritative


def detect_source_bodies(glb_path: str | Path, game_path: str, candidates: list[dict[str, Any]], required_slots: list[str] | tuple[str, ...] | None = None, preferred_race_code: str | None = None, source_context: dict[str, Any] | None = None, primary_slot: str | None = None) -> dict[str, Any]:
    """Infer source-body context for the whole selected garment."""
    from rbody_v3_loader import RBodyV3

    source = GLB(Path(glb_path))
    primary_slot = str(primary_slot or _model_slot(game_path) or "")
    if primary_slot not in ("Chest", "Legs", "Hands", "Feet"):
        raise ValueError("Accessory source-body detection requires the primary body-support slot from coverage analysis.")
    required = {str(x) for x in (required_slots or [primary_slot]) if str(x) in ("Chest", "Legs", "Hands", "Feet")}
    required.add(primary_slot)

    body_rows: list[dict[str, Any]] = []
    garment_rows: list[dict[str, Any]] = []
    body_materials: set[str] = set()
    for name in source.mesh_names():
        if not name:
            continue
        try:
            data = source.data(name)
        except Exception:
            continue
        if len(np.asarray(data.get("V", []))) == 0:
            continue
        material = _normalise_material(data.get("material"))
        if _is_obvious_embedded_body_material(material):
            body_rows.append(data)
            body_materials.add(material)
        else:
            garment_rows.append(data)

    candidates_by_slot = {slot: [c for c in candidates if str(c.get("slot")) == slot] for slot in required}
    hint_diagnostics: dict[str, Any] = {}
    authoritative_hints: dict[str, dict[str, Any]] = {}
    for slot in list(candidates_by_slot):
        narrowed, hint_report, authoritative = _source_hint_candidates(candidates_by_slot[slot], source_context)
        candidates_by_slot[slot] = narrowed
        hint_diagnostics[slot] = hint_report
        if authoritative is not None:
            authoritative_hints[slot] = authoritative
    loaders: dict[str, RBodyV3] = {}
    ref_cache: dict[tuple[str, str, str, str, str | None], dict[str, Any]] = {}

    def loader_for(path: str) -> RBodyV3:
        full = str(Path(path).resolve())
        if full not in loaders:
            loaders[full] = RBodyV3(full)
        return loaders[full]

    def candidate_identity(row: dict[str, Any]) -> tuple[RBodyV3, str | None, str, dict[str, Any]]:
        lib = loader_for(str(row["rbody"]))
        race_codes = {str(x) for x in row.get("race_codes", [])}
        race = str(preferred_race_code) if preferred_race_code is not None and str(preferred_race_code) in race_codes else None
        body_key = row.get("body_id") or row["body"]
        variant_key = row.get("variant_id") or row["variant"]
        payload_id = str(lib.resolve_payload_id(body_key, row["slot"], variant_key, race_code=race))
        return lib, race, payload_id, lib.payload_meta(payload_id)

    def candidate_ref(row: dict[str, Any]) -> tuple[dict[str, Any], str | None, str, dict[str, Any]]:
        lib, race, payload_id, meta = candidate_identity(row)
        body_key = row.get("body_id") or row["body"]
        variant_key = row.get("variant_id") or row["variant"]
        key = (str(Path(row["rbody"]).resolve()), str(body_key), str(row["slot"]), str(variant_key), race)
        if key not in ref_cache:
            ref_cache[key] = lib.reference(body_key, row["slot"], variant_key, race_code=race)
        return ref_cache[key], race, payload_id, meta

    matches: list[dict[str, Any]] = []
    matched_rows: dict[str, dict[str, Any]] = {}
    matched_refs: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}

    try:
        for hinted_slot, chosen in authoritative_hints.items():
            try:
                chosen_ref, _, _, _ = candidate_ref(chosen)
            except Exception:
                chosen_ref = None
            matches.append({
                "slot": hinted_slot,
                "rbody": str(Path(chosen["rbody"]).resolve()),
                "collection": chosen.get("collection"),
                "body_id": chosen.get("body_id"),
                "body": chosen.get("body"),
                "variant_id": chosen.get("variant_id"),
                "variant": chosen.get("variant"),
                "confidence": 1.0,
                "reason": f"Penumbra selection name directly identifies {chosen.get('body')} / {chosen.get('variant')}",
            })
            matched_rows[hinted_slot] = chosen
            if chosen_ref is not None:
                matched_refs[hinted_slot] = chosen_ref
            diagnostics[hinted_slot] = {"method": "option-name", "confidence": 1.0, "hint": hint_diagnostics.get(hinted_slot, {})}

        for direct_slot in sorted(required, key=lambda s: (s != primary_slot, ("Chest", "Legs", "Hands", "Feet").index(s))):
            if direct_slot in matched_rows:
                continue
            slot_candidates = candidates_by_slot.get(direct_slot, [])
            if direct_slot not in diagnostics and hint_diagnostics.get(direct_slot, {}).get("used"):
                diagnostics[direct_slot] = {"method": "option-name-narrowing", "hint": hint_diagnostics[direct_slot]}
            if not body_rows or not slot_candidates:
                continue

            source_vertices = _slot_surface_points(source, body_rows, direct_slot, direct_body=True)
            if len(source_vertices) < 24:
                # Fall back to the primary slot for odd exports, but do not infer extra regions from it.
                if direct_slot != primary_slot:
                    continue
                source_vertices = np.vstack([np.asarray(row["V"], dtype=np.float64) for row in body_rows])
            source_vertices = _sample_points(np.asarray(source_vertices, dtype=np.float64), 6000)
            if len(source_vertices) < 16:
                continue

            source_min = source_vertices.min(axis=0)
            source_max = source_vertices.max(axis=0)
            source_extent = np.maximum(source_max - source_min, 0.03)
            source_centre = (source_min + source_max) * 0.5

            metadata_ranked: list[tuple[float, dict[str, Any], str, str | None]] = []
            seen_payloads: dict[str, list[dict[str, Any]]] = {}
            for row in slot_candidates:
                try:
                    _, race, payload_id, meta = candidate_identity(row)
                except Exception:
                    continue
                seen_payloads.setdefault(payload_id, []).append(row)
                cmin = np.asarray(meta.get("bounds_min", [0, 0, 0]), dtype=np.float64)
                cmax = np.asarray(meta.get("bounds_max", [0, 0, 0]), dtype=np.float64)
                cextent = np.maximum(cmax - cmin, 0.03)
                ccentre = (cmin + cmax) * 0.5
                # Metadata only narrows the list; geometry makes the final call.
                extent_error = float(np.linalg.norm(np.minimum(np.abs(cextent - source_extent) / np.maximum(source_extent, cextent), 4.0)))
                centre_error = float(np.linalg.norm((ccentre - source_centre) / np.maximum(source_extent, cextent)))
                candidate_materials = {_normalise_material(x) for x in meta.get("body_surface_materials", []) if x}
                material_match = _body_material_sets_overlap(body_materials, candidate_materials)
                metadata_score = 0.55 * extent_error + 0.20 * centre_error + (0.0 if material_match else 0.80)
                metadata_ranked.append((metadata_score, row, payload_id, race))

            best_meta_by_payload: dict[str, tuple[float, dict[str, Any], str | None]] = {}
            for score, row, payload_id, race in sorted(metadata_ranked, key=lambda x: x[0]):
                best_meta_by_payload.setdefault(payload_id, (score, row, race))

            from scipy.spatial import cKDTree
            source_tree = cKDTree(source_vertices)
            detailed: list[dict[str, Any]] = []
            # Direct embedded-body evidence beats metadata, so score the full catalogue when needed.
            for payload_id, (metadata_score, row, _) in best_meta_by_payload.items():
                try:
                    ref, _, _, meta = candidate_ref(row)
                    candidate_vertices = _sample_points(np.asarray(ref["V"], dtype=np.float64), 6000)
                except Exception:
                    continue
                if len(candidate_vertices) < 16:
                    continue
                candidate_tree = cKDTree(candidate_vertices)
                source_to_candidate = candidate_tree.query(source_vertices, k=1)[0]
                candidate_to_source = source_tree.query(candidate_vertices, k=1)[0]
                diag = max(float(np.linalg.norm(np.maximum(source_vertices.max(axis=0), candidate_vertices.max(axis=0))
                                                - np.minimum(source_vertices.min(axis=0), candidate_vertices.min(axis=0)))), 1e-6)
                # Source-to-candidate matters most because hidden source geometry may have been deleted.
                shape_score = float(
                    np.median(source_to_candidate)
                    + 0.30 * np.mean(source_to_candidate)
                    + 0.12 * np.percentile(source_to_candidate, 90)
                    + 0.04 * np.median(candidate_to_source)
                ) / diag
                candidate_materials = {_normalise_material(x) for x in meta.get("body_surface_materials", []) if x}
                material_match = _body_material_sets_overlap(body_materials, candidate_materials)
                total = shape_score + 0.001 * metadata_score + (0.0 if material_match else 0.08)
                detailed.append({
                    "total": total,
                    "shape": shape_score,
                    "material_match": material_match,
                    "row": row,
                    "payload": payload_id,
                    "ref": ref,
                })

            detailed.sort(key=lambda x: x["total"])
            if not detailed:
                continue
            best = detailed[0]
            next_distinct = next((x for x in detailed[1:] if x["payload"] != best["payload"]), None)
            next_score = float(next_distinct["total"]) if next_distinct is not None else float("inf")
            relative_gap = (next_score - float(best["total"])) / max(float(best["total"]), 0.001) if np.isfinite(next_score) else 1.0
            absolute_confidence = max(0.0, min(1.0, (0.035 - float(best["shape"])) / 0.033))
            gap_confidence = max(0.0, min(1.0, relative_gap / 0.30))
            confidence = float(0.68 * absolute_confidence + 0.32 * gap_confidence)
            credible = float(best["shape"]) <= 0.025 and (bool(best["material_match"]) or float(best["shape"]) <= 0.008)

            best_family = (str(best["row"].get("collection", "")).casefold(), str(best["row"].get("body_id", "")).casefold())
            near = [x for x in detailed
                    if (str(x["row"].get("collection", "")).casefold(), str(x["row"].get("body_id", "")).casefold()) == best_family
                    and float(x["total"]) <= float(best["total"]) + max(0.00002, float(best["total"]) * 0.02)]
            near_rows: list[dict[str, Any]] = []
            for x in near:
                near_rows.extend(seen_payloads.get(x["payload"], [x["row"]]))
            chosen = _preferred_candidate_label(near_rows or seen_payloads.get(best["payload"], [best["row"]]))
            approximate = len({x["payload"] for x in near}) > 1

            diagnostics[direct_slot] = {
                "method": "embedded-body-region",
                "hint": hint_diagnostics.get(direct_slot, {}),
                "shape_score": float(best["shape"]),
                "material_match": bool(best["material_match"]),
                "confidence": confidence,
                "observed_vertices": int(len(source_vertices)),
                "approximate_variant": approximate,
            }

            if direct_slot != primary_slot and not credible:
                continue

            try:
                chosen_ref, _, _, _ = candidate_ref(chosen)
            except Exception:
                chosen_ref = best["ref"]
            matches.append({
                "slot": direct_slot,
                "rbody": str(Path(chosen["rbody"]).resolve()),
                "collection": chosen.get("collection"),
                "body_id": chosen.get("body_id"),
                "body": chosen.get("body"),
                "variant_id": chosen.get("variant_id"),
                "variant": chosen.get("variant"),
                "confidence": confidence,
                "reason": (f"Embedded {direct_slot} body geometry directly matches {chosen.get('body')} / {chosen.get('variant')}"
                           + (" (hidden variant detail is ambiguous)" if approximate else "")),
            })
            matched_rows[direct_slot] = chosen
            matched_refs[direct_slot] = chosen_ref

        # Anything still unresolved is inferred from the garment envelope.
        for slot in sorted(required, key=lambda s: (s != primary_slot, ("Chest", "Legs", "Hands", "Feet").index(s))):
            if slot in matched_rows:
                continue
            # Do not replace unresolved embedded-body evidence with a weaker envelope guess.
            if slot == primary_slot and body_rows:
                continue
            slot_candidates = candidates_by_slot.get(slot, [])
            if not slot_candidates or not garment_rows:
                continue
            evidence = _slot_surface_points(source, garment_rows, slot)
            if len(evidence) < 24:
                diagnostics[slot] = {"method": "garment-envelope", "reason": "insufficient garment surface evidence"}
                continue

            anchor_slot = next((s for s in (primary_slot, "Chest", "Legs", "Hands", "Feet") if s in matched_rows and s != slot), None)
            anchor_row = matched_rows.get(anchor_slot) if anchor_slot else None
            anchor_ref = matched_refs.get(anchor_slot) if anchor_slot else None
            anchor_tokens = _variant_tokens(anchor_row.get("variant")) if anchor_row else set()

            # Collapse aliases to payloads and score the cheap/high-value candidates first.
            payload_rows: dict[str, list[dict[str, Any]]] = {}
            payload_order: list[tuple[tuple[float, ...], str, dict[str, Any]]] = []
            evidence_min = evidence.min(axis=0)
            evidence_max = evidence.max(axis=0)
            evidence_extent = np.maximum(evidence_max - evidence_min, 0.03)
            for row in slot_candidates:
                try:
                    _, race, payload_id, meta = candidate_identity(row)
                except Exception:
                    continue
                payload_rows.setdefault(payload_id, []).append(row)
                if any(existing[1] == payload_id for existing in payload_order):
                    continue
                same_body = bool(anchor_row and str(row.get("body_id", "")).casefold() == str(anchor_row.get("body_id", "")).casefold()
                                 and str(row.get("collection", "")).casefold() == str(anchor_row.get("collection", "")).casefold())
                race_preferred = race is not None
                cmin = np.asarray(meta.get("bounds_min", [0, 0, 0]), dtype=np.float64)
                cmax = np.asarray(meta.get("bounds_max", [0, 0, 0]), dtype=np.float64)
                separation = np.maximum(np.maximum(evidence_min - cmax, cmin - evidence_max), 0.0)
                separation_score = float(np.linalg.norm(separation / evidence_extent))
                payload_order.append(((0.0 if same_body else 1.0, 0.0 if race_preferred else 1.0, separation_score, str(row.get("body", "")).casefold(), str(row.get("variant", "")).casefold()), payload_id, row))

            scored: list[dict[str, Any]] = []
            for _, payload_id, row in sorted(payload_order, key=lambda x: x[0])[:192]:
                try:
                    ref, _, _, _ = candidate_ref(row)
                    raw_score, fit = _garment_candidate_score(evidence, ref)
                except Exception:
                    continue
                if not np.isfinite(raw_score):
                    continue

                adjusted = float(raw_score)
                same_body = bool(anchor_row and str(row.get("body_id", "")).casefold() == str(anchor_row.get("body_id", "")).casefold()
                                 and str(row.get("collection", "")).casefold() == str(anchor_row.get("collection", "")).casefold())
                same_collection = bool(anchor_row and str(row.get("collection", "")).casefold() == str(anchor_row.get("collection", "")).casefold())
                if same_body:
                    adjusted -= 0.014
                elif same_collection:
                    adjusted -= 0.004

                shared_tokens = anchor_tokens.intersection(_variant_tokens(row.get("variant")))
                adjusted -= min(0.009, 0.003 * len(shared_tokens))

                interface = float("inf")
                if anchor_ref is not None and anchor_slot is not None:
                    try:
                        interface = _cross_slot_interface_distance(anchor_ref, anchor_slot, ref, slot)
                    except Exception:
                        interface = float("inf")
                    if np.isfinite(interface):
                        adjusted += min(0.012, 0.045 * interface)

                scored.append({
                    "adjusted": max(adjusted, 0.0),
                    "raw": float(raw_score),
                    "row": row,
                    "payload": payload_id,
                    "ref": ref,
                    "fit": fit,
                    "same_body": same_body,
                    "shared_tokens": sorted(shared_tokens),
                    "interface": interface,
                })

            if not scored:
                continue
            scored.sort(key=lambda x: x["adjusted"])

            global_best = scored[0]
            same_body_candidates = [x for x in scored if x["same_body"]]
            family_preferred = False
            if same_body_candidates:
                same_body_best = min(same_body_candidates, key=lambda x: x["raw"])
                allowed_gap = max(0.030, min(0.060, float(same_body_best["raw"]) * 0.35))
                if float(same_body_best["raw"]) <= 0.23 and float(same_body_best["raw"]) <= float(global_best["raw"]) + allowed_gap:
                    best = same_body_best
                    family_preferred = best is not global_best
                else:
                    best = global_best
            else:
                best = global_best

            best_family = (str(best["row"].get("collection", "")).casefold(), str(best["row"].get("body_id", "")).casefold())
            next_family = min((x for x in scored if x is not best and (str(x["row"].get("collection", "")).casefold(), str(x["row"].get("body_id", "")).casefold()) != best_family), key=lambda x: x["adjusted"], default=None)
            next_family_score = float(next_family["adjusted"]) if next_family else float("inf")
            family_gap = (next_family_score - float(best["adjusted"])) / max(float(best["adjusted"]), 0.02) if np.isfinite(next_family_score) else 1.0

            absolute_confidence = max(0.0, min(1.0, (0.24 - float(best["raw"])) / 0.20))
            gap_confidence = max(0.0, min(1.0, family_gap / 0.24))
            contact_confidence = max(0.0, min(1.0, float(best["fit"].get("contact_fraction", 0.0)) / 0.45))
            confidence = float(0.48 * absolute_confidence + 0.34 * gap_confidence + 0.18 * contact_confidence)
            if best["same_body"]:
                confidence = min(1.0, confidence + 0.10)

            # If the visible geometry cannot distinguish sibling payloads, choose a stable representative.
            near = [x for x in scored
                    if (str(x["row"].get("collection", "")).casefold(), str(x["row"].get("body_id", "")).casefold()) == best_family
                    and float(x["adjusted"]) <= float(best["adjusted"]) + max(0.0045, float(best["adjusted"]) * 0.055)]
            if anchor_tokens and near:
                best_token_overlap = max(len(x["shared_tokens"]) for x in near)
                if best_token_overlap > 0:
                    near = [x for x in near if len(x["shared_tokens"]) == best_token_overlap]
            near_rows: list[dict[str, Any]] = []
            for x in near:
                near_rows.extend(payload_rows.get(x["payload"], [x["row"]]))
            chosen = _preferred_candidate_label(near_rows or payload_rows.get(best["payload"], [best["row"]]))
            approximate = len({x["payload"] for x in near}) > 1

            diagnostics[slot] = {
                "method": "garment-envelope",
                "score": float(best["raw"]),
                "adjusted_score": float(best["adjusted"]),
                "confidence": confidence,
                "evidence_vertices": int(len(evidence)),
                "family_gap": float(family_gap),
                "contact_fraction": float(best["fit"].get("contact_fraction", 0.0)),
                "approximate_variant": approximate,
                "direct_family_preferred": family_preferred,
            }

            # Relax confidence only when direct body evidence and garment fit agree on the same family.
            accepted = confidence >= (0.34 if best["same_body"] else 0.46) and float(best["raw"]) <= 0.23
            if not accepted:
                continue

            qualifier = "approximate variant; hidden differences are not visible to the garment" if approximate else "geometry match"
            matches.append({
                "slot": slot,
                "rbody": str(Path(chosen["rbody"]).resolve()),
                "collection": chosen.get("collection"),
                "body_id": chosen.get("body_id"),
                "body": chosen.get("body"),
                "variant_id": chosen.get("variant_id"),
                "variant": chosen.get("variant"),
                "confidence": confidence,
                "reason": f"Garment extends into {slot}; its authored envelope best fits {chosen.get('body')} / {chosen.get('variant')} ({qualifier})",
            })
            matched_rows[slot] = chosen
            # Keep the exact scored reference for neighbouring-region context.
            matched_refs[slot] = best["ref"]

        return {
            "ok": True,
            "matches": matches,
            "embedded_body_materials": sorted(body_materials),
            "required_slots": sorted(required),
            "diagnostics": diagnostics,
            "strategy": "embedded-body-when-present + garment-envelope-for-missing-regions",
        }
    finally:
        for loader in loaders.values():
            loader.close()
