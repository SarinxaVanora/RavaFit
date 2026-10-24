"""Non-mutating audit of weighted body joints lost when mapping onto an outfit rig.

The selected body retains its original MDL bone names and weights.  Converting those
weights into the garment GLB skin's joint columns can drop genuinely weighted
joints, and subsequent normalisation conceals that loss.  Measure it before the
projection; this module must never alter source or target weights.
"""
from __future__ import annotations

import numpy as np


def audit_body_rig_projection(joints, weights, body_joint_names, outfit_joint_names):
    """Summarise every originally weighted body bone and its projection visibility."""
    J = np.asarray(joints, dtype=np.int64)
    W = np.asarray(weights, dtype=np.float64)
    names = [str(name) for name in body_joint_names]
    outfit_names = [str(name) for name in outfit_joint_names]
    if J.ndim != 2 or W.shape != J.shape:
        raise ValueError(f"Body joint/weight arrays disagree: {J.shape}/{W.shape}.")
    if np.any(~np.isfinite(W)) or np.any(W < 0):
        raise ValueError("Body skin has nonfinite or negative weights; rig projection cannot be audited.")
    valid = (J >= 0) & (J < len(names)) & (W > 0)
    if np.any((W > 0) & ~valid):
        raise ValueError("Body skin has a weighted joint index outside its source bone table.")

    active = J[valid]
    mass = W[valid]
    by_joint = np.bincount(active, weights=mass, minlength=len(names))
    occurrences = np.bincount(active, minlength=len(names))
    name_set = set(outfit_names)
    retained = np.asarray([name in name_set for name in names], dtype=bool)
    # Measure before the normalisation of the surviving columns.
    total_by_vertex = np.sum(np.where(valid, W, 0.0), axis=1)
    missing_mask = valid & ~retained[np.clip(J, 0, max(len(names) - 1, 0))]
    missing_by_vertex = np.sum(np.where(missing_mask, W, 0.0), axis=1)
    good = total_by_vertex > 1e-12
    lost_fraction = np.zeros(len(J), dtype=np.float64)
    lost_fraction[good] = missing_by_vertex[good] / total_by_vertex[good]
    originally_weighted = int(np.count_nonzero(good))
    affected = (lost_fraction > 1e-8) & good

    per_joint = [
        {
            "name": names[i],
            "retained_in_outfit_rig": bool(retained[i]),
            "weight_mass": float(by_joint[i]),
            "weighted_vertex_entries": int(occurrences[i]),
        }
        for i in range(len(names)) if by_joint[i] > 1e-10
    ]
    per_joint.sort(key=lambda row: (-row["weight_mass"], row["name"]))
    missing = [row for row in per_joint if not row["retained_in_outfit_rig"]]
    all_mass = float(np.sum(by_joint))
    lost_mass = float(np.sum(by_joint[~retained]))
    return {
        "mode": "pre_normalisation_body_rig_projection_audit",
        "original_body_joint_count": len(names),
        "original_body_joint_names": names,
        "outfit_rig_joint_count": len(outfit_names),
        "weighted_body_joint_count": len(per_joint),
        "missing_weighted_joint_count": len(missing),
        "weighted_body_vertices": originally_weighted,
        "vertices_with_lost_weight": int(np.count_nonzero(affected)),
        "vertices_fully_lost": int(np.count_nonzero(good & (lost_fraction >= 1.0 - 1e-8))),
        "total_original_weight_mass": all_mass,
        "total_missing_weight_mass": lost_mass,
        "missing_weight_mass_fraction": lost_mass / all_mass if all_mass > 1e-12 else 0.0,
        "lost_weight_mean_fraction": float(np.mean(lost_fraction[good])) if originally_weighted else 0.0,
        "lost_weight_p95_fraction": float(np.percentile(lost_fraction[good], 95)) if originally_weighted else 0.0,
        "lost_weight_max_fraction": float(np.max(lost_fraction, initial=0.0)),
        "missing_weighted_joints": missing,
        "weighted_body_joints": per_joint,
        "changes_applied": False,
    }


def compare_body_rig_audits(source, target):
    """Compare source and target bone-weight distributions without normalising away absent joints.

    The bodies need not share topology or vertex counts. Mass fractions indicate
    how heavily each body uses a joint *overall*, not a vertex correspondence or
    proof of a local skinning change.
    """
    if source is None or target is None:
        return None
    source_rows = {row["name"]: row for row in source["weighted_body_joints"]}
    target_rows = {row["name"]: row for row in target["weighted_body_joints"]}
    source_mass = float(source["total_original_weight_mass"])
    target_mass = float(target["total_original_weight_mass"])
    bones = []
    for name in sorted(set(source_rows) | set(target_rows)):
        s = source_rows.get(name)
        t = target_rows.get(name)
        sv = float(s["weight_mass"]) / source_mass if s and source_mass > 1e-12 else 0.0
        tv = float(t["weight_mass"]) / target_mass if t and target_mass > 1e-12 else 0.0
        bones.append({
            "name": name,
            "source_mass_fraction": sv,
            "target_mass_fraction": tv,
            "target_minus_source_mass_fraction": tv - sv,
            "weighted_in_source": s is not None,
            "weighted_in_target": t is not None,
            "missing_from_outfit_rig": not (s or t)["retained_in_outfit_rig"],
        })
    bones.sort(key=lambda row: (-abs(row["target_minus_source_mass_fraction"]), row["name"]))
    return {
        "mode": "body_joint_distribution_comparison_not_vertex_correspondence",
        "source_weighted_bone_count": len(source_rows),
        "target_weighted_bone_count": len(target_rows),
        "newly_weighted_target_bones": [x["name"] for x in bones if not x["weighted_in_source"]],
        "source_only_weighted_bones": [x["name"] for x in bones if not x["weighted_in_target"]],
        "bones_missing_from_outfit_rig": [x["name"] for x in bones if x["missing_from_outfit_rig"]],
        "bones": bones,
        "changes_applied": False,
    }
