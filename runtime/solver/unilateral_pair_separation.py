from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import heapq

import numpy as np

_EPS = 1.0e-12


@dataclass(frozen=True)
class UnilateralPairSeparationConfig:
    """Generic source-authority guard for disconnected left/right leg structures.

    Distances are metres.  The guard never freezes source-space placement: it scales the authored
    inner separation by the candidate's solved outer span, then corrects only the source-medial
    portion of each component when that scaled gap has collapsed.
    """

    centreline_x: float = 0.0
    min_component_vertices: int = 8
    min_component_faces: int = 4
    min_dominant_side_fraction: float = 0.88
    max_opposite_side_fraction: float = 0.08
    centreline_tolerance_m: float = 0.00035
    min_source_inner_gap_m: float = 0.00075
    min_source_outer_span_m: float = 0.006
    preserve_ratio: float = 0.94
    absolute_tolerance_m: float = 0.001
    max_corrective_vertex_movement_m: float = 0.010
    lateral_scale_min: float = 0.65
    lateral_scale_max: float = 1.60
    medial_fraction: float = 0.48
    medial_falloff_power: float = 1.65
    pair_score_limit: float = 2.75
    preserve_proximal_structure: bool = True
    proximal_zone_fraction: float = 0.075
    proximal_zone_min_m: float = 0.018
    proximal_zone_max_m: float = 0.045
    proximal_core_fraction: float = 0.46
    proximal_falloff_power: float = 1.35
    proximal_scale_min: float = 0.65
    proximal_scale_max: float = 1.38
    proximal_max_vertex_correction_m: float = 0.012
    require_leg_evidence: bool = False
    min_leg_mass: float = 0.30
    min_leg_dominance: float = 0.18


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a)); rb = self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _connected_components(vertex_count: int, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    F = np.asarray(faces, dtype=np.int64)
    if vertex_count <= 0 or F.size == 0:
        return []
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("faces must be an Mx3 triangle index array")
    if int(np.min(F)) < 0 or int(np.max(F)) >= vertex_count:
        raise ValueError("faces contain out-of-range vertex indices")
    uf = _UnionFind(vertex_count)
    for a, b, c in F.tolist():
        uf.union(a, b); uf.union(b, c); uf.union(c, a)
    used = np.unique(F.reshape(-1))
    by_root: dict[int, list[int]] = {}
    for vi in used.tolist():
        by_root.setdefault(uf.find(int(vi)), []).append(int(vi))
    face_roots = np.asarray([uf.find(int(face[0])) for face in F], dtype=np.int64)
    result = []
    for root, ids in by_root.items():
        vids = np.asarray(sorted(ids), dtype=np.int64)
        fids = np.flatnonzero(face_roots == int(root)).astype(np.int64)
        result.append((vids, fids))
    return result


def _safe_log_ratio(a: float, b: float) -> float:
    return abs(float(np.log(max(a, _EPS) / max(b, _EPS))))


def _boundary_vertex_groups(faces: np.ndarray) -> list[np.ndarray]:
    """Return connected open-boundary vertex groups for one connected component.

    The groups are inferred only from authored triangle topology.  Closed components simply return no
    groups and fall back to whole-component measurements.
    """
    F = np.asarray(faces, dtype=np.int64)
    if F.size == 0:
        return []
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in F.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(min(u, v)), int(max(u, v)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return []
    adjacency: dict[int, set[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    groups: list[np.ndarray] = []
    seen: set[int] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]; seen.add(start); vertices: list[int] = []
        while stack:
            current = stack.pop(); vertices.append(int(current))
            for nxt in adjacency.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt); stack.append(int(nxt))
        groups.append(np.asarray(sorted(vertices), dtype=np.int64))
    return groups


def _source_separation_witness(source: np.ndarray, component_faces: np.ndarray, vids: np.ndarray) -> tuple[np.ndarray, str]:
    """Choose a source-proven measurement witness without garment-specific assumptions.

    Long unilateral shells commonly have several open boundaries.  Their globally most-medial vertex
    can belong to an unrelated distal edge, seam or authored projection and is therefore not a stable
    witness for the proximal separation that keeps the two shells on their own sides.  When multiple
    authored boundary groups exist, use the highest-Y group (FFXIV model-space proximal leg end).
    Compact/open patches with a single boundary group keep their complete authored component witness,
    and closed components fall back to the full component.
    """
    groups = _boundary_vertex_groups(component_faces)
    if len(groups) < 2:
        return np.asarray(vids, dtype=np.int64), "whole_component"
    viable = [group for group in groups if len(group) >= 3]
    if len(viable) < 2:
        return np.asarray(vids, dtype=np.int64), "whole_component"
    selected = max(viable, key=lambda group: float(np.mean(source[group, 1])))
    return np.asarray(selected, dtype=np.int64), "proximal_boundary_loop"


def _component_descriptor(
    source: np.ndarray,
    faces: np.ndarray,
    vids: np.ndarray,
    fids: np.ndarray,
    cfg: UnilateralPairSeparationConfig,
    left_leg_mass: np.ndarray | None,
    right_leg_mass: np.ndarray | None,
) -> dict[str, Any] | None:
    if len(vids) < int(cfg.min_component_vertices) or len(fids) < int(cfg.min_component_faces):
        return None
    P = source[vids]
    rel = P[:, 0] - float(cfg.centreline_x)
    tol = float(cfg.centreline_tolerance_m)
    neg = float(np.mean(rel < -tol)); pos = float(np.mean(rel > tol)); centre = max(0.0, 1.0 - neg - pos)
    dominant_fraction = max(neg, pos)
    opposite_fraction = min(neg, pos)
    if dominant_fraction < float(cfg.min_dominant_side_fraction) or opposite_fraction > float(cfg.max_opposite_side_fraction):
        return None
    side = -1 if neg > pos else 1
    # A component with substantial source occupancy on the centreline is a bridge/neutral structure,
    # not a unilateral shell, even if its centroid happens to sit on one side.
    if centre > (1.0 - float(cfg.min_dominant_side_fraction)):
        return None

    rig_side = None; leg_strength = None; leg_dominance = None
    if left_leg_mass is not None and right_leg_mass is not None:
        lm = float(np.mean(left_leg_mass[vids])); rm = float(np.mean(right_leg_mass[vids]))
        leg_strength = max(lm, rm); leg_dominance = abs(lm - rm)
        if leg_strength >= float(cfg.min_leg_mass) and leg_dominance >= float(cfg.min_leg_dominance):
            rig_side = -1 if lm > rm else 1
    if bool(cfg.require_leg_evidence) and rig_side is None:
        return None

    extent = np.ptp(P, axis=0)
    centroid = np.mean(P, axis=0)
    witness_ids, witness_mode = _source_separation_witness(source, np.asarray(faces, dtype=np.int64)[fids], vids)
    return {
        "vertex_ids": vids,
        "face_ids": fids,
        "side": int(side),
        "rig_side": rig_side,
        "leg_strength": leg_strength,
        "leg_dominance": leg_dominance,
        "centroid": centroid,
        "extent": extent,
        "vertex_count": int(len(vids)),
        "face_count": int(len(fids)),
        "separation_vertex_ids": witness_ids,
        "separation_witness_mode": witness_mode,
        "separation_vertex_count": int(len(witness_ids)),
        "x_min": float(np.min(P[:, 0])),
        "x_max": float(np.max(P[:, 0])),
    }


def _pair_score(left: dict[str, Any], right: dict[str, Any], cfg: UnilateralPairSeparationConfig) -> float:
    if left["side"] != -1 or right["side"] != 1:
        return float("inf")
    if left.get("rig_side") is not None and right.get("rig_side") is not None and left["rig_side"] == right["rig_side"]:
        return float("inf")
    c = float(cfg.centreline_x)
    mirrored = np.asarray(left["centroid"], dtype=np.float64).copy(); mirrored[0] = 2.0 * c - mirrored[0]
    right_centroid = np.asarray(right["centroid"], dtype=np.float64)
    scale = max(float(np.max(np.r_[left["extent"], right["extent"]])), 0.010)
    centroid_term = float(np.linalg.norm(mirrored - right_centroid) / scale)
    extent_l = np.maximum(np.asarray(left["extent"], dtype=np.float64), 1.0e-5)
    extent_r = np.maximum(np.asarray(right["extent"], dtype=np.float64), 1.0e-5)
    extent_term = float(np.mean(np.abs(np.log(extent_l / extent_r))))
    count_term = _safe_log_ratio(float(left["vertex_count"]), float(right["vertex_count"]))
    # Vertical/depth agreement is important when several mirrored bands exist at different heights.
    yz_scale = max(float(np.linalg.norm(np.maximum(extent_l[1:], extent_r[1:]))), 0.008)
    yz_term = float(np.linalg.norm(np.asarray(left["centroid"])[1:] - right_centroid[1:]) / yz_scale)
    return 0.95 * centroid_term + 0.75 * extent_term + 0.35 * count_term + 0.85 * yz_term


def _medial_weights(
    source_x: np.ndarray,
    side: int,
    cfg: UnilateralPairSeparationConfig,
    *,
    witness_x: np.ndarray | None = None,
) -> np.ndarray:
    """Smoothly weight the medial side while preserving the solved outer envelope.

    When a topology-derived separation witness is available, its authored lateral envelope defines
    what "inner" and "outer" mean.  This prevents an unrelated distal projection from diluting
    the proximal band's medial weight.  Vertices outside that witness envelope are simply clamped;
    the correction remains a smooth X-only medial falloff rather than a rigid component translation.
    """
    x = np.asarray(source_x, dtype=np.float64)
    basis = x if witness_x is None else np.asarray(witness_x, dtype=np.float64)
    if basis.size == 0:
        basis = x
    lo = float(np.min(basis)); hi = float(np.max(basis)); span = hi - lo
    if span <= _EPS:
        return np.ones(len(x), dtype=np.float64)
    if side < 0:
        inward = (x - lo) / span
    else:
        inward = (hi - x) / span
    inward = np.clip(inward, 0.0, 1.0)
    width = float(np.clip(cfg.medial_fraction, 0.05, 1.0))
    start = 1.0 - width
    t = np.clip((inward - start) / max(width, _EPS), 0.0, 1.0)
    return np.power(t, max(float(cfg.medial_falloff_power), 0.25))



def _component_geodesic_distances(source: np.ndarray, faces: np.ndarray, vids: np.ndarray, fids: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Source-topology geodesic distance from an authored witness into one component."""
    vids=np.asarray(vids,dtype=np.int64); allowed=set(int(v) for v in vids.tolist())
    adjacency={int(v):[] for v in vids.tolist()}
    for a,b,c in np.asarray(faces,dtype=np.int64)[np.asarray(fids,dtype=np.int64)].tolist():
        for u,v in ((a,b),(b,c),(c,a)):
            u=int(u);v=int(v)
            if u not in allowed or v not in allowed:continue
            length=float(np.linalg.norm(source[u]-source[v]))
            adjacency[u].append((v,length));adjacency[v].append((u,length))
    distances={int(v):float("inf") for v in vids.tolist()};queue=[]
    for seed in np.asarray(seeds,dtype=np.int64).tolist():
        if int(seed) in distances:
            distances[int(seed)]=0.0;heapq.heappush(queue,(0.0,int(seed)))
    while queue:
        dist,u=heapq.heappop(queue)
        if dist!=distances[u]:continue
        for v,w in adjacency.get(u,()):
            nd=dist+w
            if nd<distances[v]:
                distances[v]=nd;heapq.heappush(queue,(nd,v))
    return np.asarray([distances[int(v)] for v in vids.tolist()],dtype=np.float64)


def _fit_similarity(source_points: np.ndarray, target_points: np.ndarray, weights: np.ndarray, scale_min: float, scale_max: float, fixed_scale: float | None = None) -> tuple[np.ndarray,float,np.ndarray,float]:
    """Orientation-preserving weighted similarity for row-vector XYZ points."""
    X=np.asarray(source_points,dtype=np.float64);Y=np.asarray(target_points,dtype=np.float64);w=np.asarray(weights,dtype=np.float64).reshape(-1)
    if len(X)<3 or X.shape!=Y.shape or len(w)!=len(X):
        return np.eye(3),1.0,np.zeros(3),float("inf")
    w=np.maximum(w,1.0e-8);w=w/float(np.sum(w));mx=np.sum(X*w[:,None],axis=0);my=np.sum(Y*w[:,None],axis=0)
    Xc=X-mx;Yc=Y-my;cov=(Xc*w[:,None]).T@Yc
    U,S,Vt=np.linalg.svd(cov,full_matrices=False);R=U@Vt
    sign=1.0
    if np.linalg.det(R)<0.0:
        U[:,-1]*=-1.0;R=U@Vt;sign=-1.0
    denom=float(np.sum(w*np.sum(Xc*Xc,axis=1)))
    scale=float((np.sum(S[:-1])+sign*S[-1])/max(denom,_EPS)) if len(S)>=3 else 1.0
    if fixed_scale is not None and np.isfinite(float(fixed_scale)):
        scale=float(fixed_scale)
    scale=float(np.clip(scale,float(scale_min),float(scale_max)))
    translation=my-scale*(mx@R)
    predicted=scale*(X@R)+translation
    rms=float(np.sqrt(np.sum(w*np.sum((predicted-Y)**2,axis=1))))
    return R,scale,translation,rms


def _preserve_proximal_component_structure(source: np.ndarray, solved: np.ndarray, faces: np.ndarray, comp: dict[str,Any], cfg: UnilateralPairSeparationConfig, *, preferred_scale: float | None = None) -> tuple[np.ndarray,dict[str,Any]]:
    """Project only the source-proximal zone onto the best target-relative similarity frame.

    This preserves the authored opening/band structure without freezing world-space placement or the
    rest of the unilateral shell.  The transform is inferred from the candidate solve itself; source
    geometry contributes structure only, never target placement.
    """
    if not bool(cfg.preserve_proximal_structure) or comp.get("separation_witness_mode")!="proximal_boundary_loop":
        return solved,{"enabled":False,"reason":"no source-proven proximal boundary witness","moved_vertices":0}
    vids=np.asarray(comp["vertex_ids"],dtype=np.int64);fids=np.asarray(comp["face_ids"],dtype=np.int64);seeds=np.asarray(comp["separation_vertex_ids"],dtype=np.int64)
    y_extent=float(np.ptp(source[vids,1]));depth=float(np.clip(y_extent*float(cfg.proximal_zone_fraction),float(cfg.proximal_zone_min_m),float(cfg.proximal_zone_max_m)))
    distances=_component_geodesic_distances(source,faces,vids,fids,seeds);zone=distances<=depth
    if int(np.count_nonzero(zone))<8:
        return solved,{"enabled":False,"reason":"insufficient proximal geodesic zone","moved_vertices":0,"zone_depth_mm":depth*1000.0}
    local_ids=vids[zone];d=distances[zone];core=max(0.0,min(0.95,float(cfg.proximal_core_fraction)))
    t=np.clip((1.0-d/depth)/max(1.0-core,_EPS),0.0,1.0);blend=np.where(d<=depth*core,1.0,np.power(t,max(float(cfg.proximal_falloff_power),0.25)))
    fit_weights=np.maximum(0.12,blend)
    R,scale,translation,fit_rms=_fit_similarity(source[local_ids],solved[local_ids],fit_weights,float(cfg.proximal_scale_min),float(cfg.proximal_scale_max),fixed_scale=preferred_scale)
    predicted=scale*(source[local_ids]@R)+translation
    # Preserve the actual target-refit outer envelope.  The recovered source shape is positioned from
    # that solved outer witness rather than recentred in source/world space.
    local_lookup={int(v):i for i,v in enumerate(local_ids.tolist())};seed_local=np.asarray([local_lookup[int(v)] for v in seeds.tolist() if int(v) in local_lookup],dtype=np.int64)
    outer_shift_x=0.0
    if len(seed_local):
        if int(comp.get("side",0))<0:
            outer_shift_x=float(np.min(solved[seeds,0])-np.min(predicted[seed_local,0]))
        else:
            outer_shift_x=float(np.max(solved[seeds,0])-np.max(predicted[seed_local,0]))
        predicted[:,0]+=outer_shift_x
    delta=(predicted-solved[local_ids])*blend[:,None];length=np.linalg.norm(delta,axis=1);cap=float(cfg.proximal_max_vertex_correction_m)
    capped=length>cap
    if np.any(capped):delta[capped]*=(cap/np.maximum(length[capped],_EPS))[:,None]
    out=solved.copy();out[local_ids]+=delta;move=np.linalg.norm(delta,axis=1);moved=move>1.0e-12
    before=np.linalg.norm((scale*(source[local_ids]@R)+translation)-solved[local_ids],axis=1)
    after=np.linalg.norm((scale*(source[local_ids]@R)+translation)-out[local_ids],axis=1)
    return out,{"enabled":True,"zone_depth_mm":depth*1000.0,"zone_vertices":int(len(local_ids)),"core_vertices":int(np.count_nonzero(d<=depth*core)),"similarity_scale":float(scale),"outer_envelope_anchor_shift_mm":float(outer_shift_x*1000.0),"fit_rms_mm":fit_rms*1000.0,"moved_vertices":int(np.count_nonzero(moved)),"maximum_correction_mm":float(np.max(move,initial=0.0)*1000.0),"pre_shape_error_p95_mm":float(np.percentile(before,95)*1000.0),"post_shape_error_p95_mm":float(np.percentile(after,95)*1000.0)}

def _pair_measurements(source: np.ndarray, solved: np.ndarray, left: dict[str, Any], right: dict[str, Any], cfg: UnilateralPairSeparationConfig) -> dict[str, float]:
    li = np.asarray(left.get("separation_vertex_ids", left["vertex_ids"]), dtype=np.int64)
    ri = np.asarray(right.get("separation_vertex_ids", right["vertex_ids"]), dtype=np.int64)
    source_inner_gap = float(np.min(source[ri, 0]) - np.max(source[li, 0]))
    source_outer_span = float(np.max(source[ri, 0]) - np.min(source[li, 0]))
    solved_inner_gap = float(np.min(solved[ri, 0]) - np.max(solved[li, 0]))
    solved_outer_span = float(np.max(solved[ri, 0]) - np.min(solved[li, 0]))
    lateral_scale = float(np.clip(solved_outer_span / max(source_outer_span, _EPS), float(cfg.lateral_scale_min), float(cfg.lateral_scale_max)))
    expected_inner_gap = max(0.0, source_inner_gap * lateral_scale)
    required_inner_gap = max(expected_inner_gap * float(cfg.preserve_ratio), expected_inner_gap - float(cfg.absolute_tolerance_m), 0.0)
    return {
        "source_inner_gap_m": source_inner_gap,
        "source_outer_span_m": source_outer_span,
        "solved_inner_gap_m": solved_inner_gap,
        "solved_outer_span_m": solved_outer_span,
        "lateral_scale": lateral_scale,
        "expected_inner_gap_m": expected_inner_gap,
        "required_inner_gap_m": required_inner_gap,
    }


def preserve_source_unilateral_pair_separation(
    source_vertices: np.ndarray,
    solved_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    left_leg_mass: np.ndarray | None = None,
    right_leg_mass: np.ndarray | None = None,
    config: UnilateralPairSeparationConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore target-scaled source separation for source-proven disconnected unilateral pairs.

    The function is intentionally garment-name agnostic.  Pair identity comes from untouched source
    topology, source-side occupancy, mirrored structural similarity and (when supplied) unilateral leg
    skinning evidence.  Connected centreline bridges are never split or translated.
    """
    cfg = config or UnilateralPairSeparationConfig()
    source = np.asarray(source_vertices, dtype=np.float64)
    solved = np.asarray(solved_vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if source.ndim != 2 or source.shape[1] != 3 or solved.shape != source.shape:
        raise ValueError("source_vertices and solved_vertices must be matching Nx3 arrays")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(solved)):
        raise ValueError("unilateral separation received non-finite vertices")
    if left_leg_mass is not None:
        left_leg_mass = np.asarray(left_leg_mass, dtype=np.float64).reshape(-1)
        if len(left_leg_mass) != len(source): raise ValueError("left_leg_mass length mismatch")
    if right_leg_mass is not None:
        right_leg_mass = np.asarray(right_leg_mass, dtype=np.float64).reshape(-1)
        if len(right_leg_mass) != len(source): raise ValueError("right_leg_mass length mismatch")
    if (left_leg_mass is None) != (right_leg_mass is None):
        raise ValueError("left_leg_mass and right_leg_mass must be supplied together")

    components = []
    for index, (vids, fids) in enumerate(_connected_components(len(source), F)):
        descriptor = _component_descriptor(source, F, vids, fids, cfg, left_leg_mass, right_leg_mass)
        if descriptor is not None:
            descriptor["component_index"] = int(index)
            components.append(descriptor)
    negatives = [c for c in components if c["side"] < 0]
    positives = [c for c in components if c["side"] > 0]

    candidates = []
    for left in negatives:
        for right in positives:
            score = _pair_score(left, right, cfg)
            if np.isfinite(score) and score <= float(cfg.pair_score_limit):
                metrics = _pair_measurements(source, solved, left, right, cfg)
                if metrics["source_inner_gap_m"] < float(cfg.min_source_inner_gap_m):
                    continue
                if metrics["source_outer_span_m"] < float(cfg.min_source_outer_span_m):
                    continue
                candidates.append((float(score), int(left["component_index"]), int(right["component_index"]), left, right))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    used: set[int] = set(); pairs = []
    for score, _, _, left, right in candidates:
        if left["component_index"] in used or right["component_index"] in used:
            continue
        used.add(int(left["component_index"])); used.add(int(right["component_index"]))
        pairs.append((score, left, right))

    out = solved.copy(); reports = []; moved_mask = np.zeros(len(out), dtype=bool); maximum_move = 0.0
    for score, left, right in pairs:
        metrics = _pair_measurements(source, out, left, right, cfg)
        deficit = float(metrics["required_inner_gap_m"] - metrics["solved_inner_gap_m"])
        report = {
            "left_component": int(left["component_index"]),
            "right_component": int(right["component_index"]),
            "pair_score": float(score),
            "left_witness_mode": str(left.get("separation_witness_mode", "whole_component")),
            "right_witness_mode": str(right.get("separation_witness_mode", "whole_component")),
            "left_witness_vertices": int(left.get("separation_vertex_count", len(left["vertex_ids"]))),
            "right_witness_vertices": int(right.get("separation_vertex_count", len(right["vertex_ids"]))),
            **{k.replace("_m", "_mm"): float(v) * 1000.0 for k, v in metrics.items() if k.endswith("_m")},
            "lateral_scale": float(metrics["lateral_scale"]),
            "moved_vertices": 0,
            "maximum_correction_mm": 0.0,
            "status": "already_preserved" if deficit <= 1.0e-9 else "corrected",
        }
        if deficit > 1.0e-9:
            li = left["vertex_ids"]; ri = right["vertex_ids"]
            left_witness = np.asarray(left.get("separation_vertex_ids", li), dtype=np.int64)
            right_witness = np.asarray(right.get("separation_vertex_ids", ri), dtype=np.int64)
            lw = _medial_weights(source[li, 0], -1, cfg, witness_x=source[left_witness, 0])
            rw = _medial_weights(source[ri, 0], 1, cfg, witness_x=source[right_witness, 0])
            left_inner_global = int(left_witness[int(np.argmax(source[left_witness, 0]))])
            right_inner_global = int(right_witness[int(np.argmin(source[right_witness, 0]))])
            left_local = {int(v): i for i, v in enumerate(np.asarray(li, dtype=np.int64).tolist())}
            right_local = {int(v): i for i, v in enumerate(np.asarray(ri, dtype=np.int64).tolist())}
            witness_weight = float(lw[left_local[left_inner_global]] + rw[right_local[right_inner_global]])
            amplitude = min(deficit / max(witness_weight, _EPS), float(cfg.max_corrective_vertex_movement_m))
            lmove = amplitude * lw; rmove = amplitude * rw
            out[li, 0] -= lmove; out[ri, 0] += rmove
            lm = lmove > 1.0e-12; rm = rmove > 1.0e-12
            moved_mask[li[lm]] = True; moved_mask[ri[rm]] = True
            pair_max = max(float(np.max(lmove, initial=0.0)), float(np.max(rmove, initial=0.0)))
            maximum_move = max(maximum_move, pair_max)
            report["moved_vertices"] = int(np.count_nonzero(lm) + np.count_nonzero(rm))
            report["maximum_correction_mm"] = pair_max * 1000.0
            final_metrics = _pair_measurements(source, out, left, right, cfg)
            report["final_inner_gap_mm"] = float(final_metrics["solved_inner_gap_m"] * 1000.0)
            report["remaining_gap_deficit_mm"] = float(max(0.0, final_metrics["required_inner_gap_m"] - final_metrics["solved_inner_gap_m"]) * 1000.0)
            if report["remaining_gap_deficit_mm"] > 1.0e-5:
                report["status"] = "correction_capped"
        else:
            report["final_inner_gap_mm"] = float(metrics["solved_inner_gap_m"] * 1000.0)
            report["remaining_gap_deficit_mm"] = 0.0

        # A legal left/right gap is not sufficient: the proximal opening itself can still be
        # squashed, twisted or sheared by the body/layer solve.  Reproject only a short geodesic
        # source-proven proximal zone into the candidate's best local similarity frame.
        proximal_reports={}
        for label,component in (("left",left),("right",right)):
            before_component=out.copy()
            out,proximal=_preserve_proximal_component_structure(source,out,F,component,cfg,preferred_scale=float(metrics["lateral_scale"]))
            proximal_reports[label]=proximal
            component_move=np.linalg.norm(out-before_component,axis=1)
            component_changed=component_move>1.0e-12
            moved_mask[component_changed]=True
            maximum_move=max(maximum_move,float(np.max(component_move,initial=0.0)))
        report["proximal_structure"]=proximal_reports

        # Proximal similarity preservation is allowed to change shape but not to violate the pair's
        # target-scaled medial floor.  Reassert the gap with the original smooth medial falloff.
        final_metrics=_pair_measurements(source,out,left,right,cfg)
        final_deficit=float(final_metrics["required_inner_gap_m"]-final_metrics["solved_inner_gap_m"])
        if final_deficit>1.0e-9:
            li=left["vertex_ids"];ri=right["vertex_ids"];left_witness=np.asarray(left.get("separation_vertex_ids",li),dtype=np.int64);right_witness=np.asarray(right.get("separation_vertex_ids",ri),dtype=np.int64)
            if left.get("separation_witness_mode")=="proximal_boundary_loop" and right.get("separation_witness_mode")=="proximal_boundary_loop":
                # Keep the actual solved outer envelope fixed.  Any residual floor correction is
                # medial within the opening and fades geodesically down the shell.
                ldepth=float(np.clip(float(np.ptp(source[li,1]))*float(cfg.proximal_zone_fraction),float(cfg.proximal_zone_min_m),float(cfg.proximal_zone_max_m)))
                rdepth=float(np.clip(float(np.ptp(source[ri,1]))*float(cfg.proximal_zone_fraction),float(cfg.proximal_zone_min_m),float(cfg.proximal_zone_max_m)))
                ld=_component_geodesic_distances(source,F,li,left["face_ids"],left_witness);rd=_component_geodesic_distances(source,F,ri,right["face_ids"],right_witness)
                lgeo=np.power(np.clip(1.0-ld/max(ldepth,_EPS),0.0,1.0),max(float(cfg.proximal_falloff_power),0.25));rgeo=np.power(np.clip(1.0-rd/max(rdepth,_EPS),0.0,1.0),max(float(cfg.proximal_falloff_power),0.25))
                lmed=_medial_weights(source[li,0],-1,cfg,witness_x=source[left_witness,0]);rmed=_medial_weights(source[ri,0],1,cfg,witness_x=source[right_witness,0]);lw=lgeo*lmed;rw=rgeo*rmed
                lmove=np.zeros(len(li),dtype=np.float64);rmove=np.zeros(len(ri),dtype=np.float64);remaining=float(cfg.max_corrective_vertex_movement_m)
                for _ in range(8):
                    now=_pair_measurements(source,out,left,right,cfg);deficit=float(now["required_inner_gap_m"]-now["solved_inner_gap_m"])
                    if deficit<=1.0e-9 or remaining<=1.0e-12:break
                    left_current=int(left_witness[int(np.argmax(out[left_witness,0]))]);right_current=int(right_witness[int(np.argmin(out[right_witness,0]))])
                    left_local={int(v):i for i,v in enumerate(np.asarray(li,dtype=np.int64).tolist())};right_local={int(v):i for i,v in enumerate(np.asarray(ri,dtype=np.int64).tolist())}
                    witness_weight=float(lw[left_local[left_current]]+rw[right_local[right_current]])
                    if witness_weight<=_EPS:break
                    amplitude=min(deficit/max(witness_weight,_EPS),remaining);step_l=amplitude*lw;step_r=amplitude*rw;out[li,0]-=step_l;out[ri,0]+=step_r;lmove+=step_l;rmove+=step_r;remaining-=amplitude
            else:
                lw=_medial_weights(source[li,0],-1,cfg,witness_x=source[left_witness,0]);rw=_medial_weights(source[ri,0],1,cfg,witness_x=source[right_witness,0])
                left_inner_global=int(left_witness[int(np.argmax(source[left_witness,0]))]);right_inner_global=int(right_witness[int(np.argmin(source[right_witness,0]))])
                left_local={int(v):i for i,v in enumerate(np.asarray(li,dtype=np.int64).tolist())};right_local={int(v):i for i,v in enumerate(np.asarray(ri,dtype=np.int64).tolist())}
                witness_weight=float(lw[left_local[left_inner_global]]+rw[right_local[right_inner_global]]);amplitude=min(final_deficit/max(witness_weight,_EPS),float(cfg.max_corrective_vertex_movement_m));lmove=amplitude*lw;rmove=amplitude*rw
            out[li,0]-=lmove;out[ri,0]+=rmove;moved_mask[li[lmove>1.0e-12]]=True;moved_mask[ri[rmove>1.0e-12]]=True;maximum_move=max(maximum_move,float(np.max(np.r_[lmove,rmove],initial=0.0)))
            final_metrics=_pair_measurements(source,out,left,right,cfg)
        report["final_inner_gap_mm"]=float(final_metrics["solved_inner_gap_m"]*1000.0)
        report["remaining_gap_deficit_mm"]=float(max(0.0,final_metrics["required_inner_gap_m"]-final_metrics["solved_inner_gap_m"])*1000.0)
        proximal_moved=sum(int(v.get("moved_vertices",0)) for v in proximal_reports.values())
        if proximal_moved>0 and report["status"]=="already_preserved":report["status"]="proximal_structure_corrected"
        report["moved_vertices"]=int(report.get("moved_vertices",0))+proximal_moved
        report["maximum_correction_mm"]=max(float(report.get("maximum_correction_mm",0.0)),max((float(v.get("maximum_correction_mm",0.0)) for v in proximal_reports.values()),default=0.0))
        reports.append(report)

    return out, {
        "enabled": bool(pairs),
        "policy": "source-proven disconnected unilateral pairs preserve target-scaled centreline separation plus a short source-proximal similarity structure zone; outer/lower target refit remains authoritative",
        "component_count": int(len(components)),
        "pair_count": int(len(pairs)),
        "moved_vertices": int(np.count_nonzero(moved_mask)),
        "maximum_correction_mm": float(maximum_move * 1000.0),
        "require_leg_evidence": bool(cfg.require_leg_evidence),
        "pairs": reports,
    }
