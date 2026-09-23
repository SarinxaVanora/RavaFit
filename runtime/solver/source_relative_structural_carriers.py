from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

_EPS = 1.0e-12


@dataclass(frozen=True)
class StructuralCarrierConfig:
    """Generic source-structure preservation driven by the already-fitted garment.

    Two source-proven structures are handled here:
    * broad thin carried bands/cuffs inherit one local similarity frame from the fitted host shell;
    * compact disconnected authored assemblies (chains, nested ornaments, stud groups, etc.) share one
      similarity frame per source-proximity cluster so internal containment/spacing cannot drift.

    Source geometry supplies shape/layout only.  Target placement is always inferred from the current
    fitted garment.  No garment/body/material names or mesh IDs are consulted.
    """

    # Broad thin carried band/cuff inference.
    band_min_vertices: int = 40
    band_max_vertices: int = 512
    band_min_major_extent_m: float = 0.080
    band_min_middle_extent_m: float = 0.050
    band_max_minor_extent_m: float = 0.060
    band_max_minor_to_middle_ratio: float = 0.80
    band_host_min_vertices: int = 300
    band_host_vertex_ratio: float = 2.0
    band_source_median_clearance_m: float = 0.040
    band_source_p90_clearance_m: float = 0.060
    band_anchor_vertices: int = 300
    band_scale_min: float = 0.65
    band_scale_max: float = 1.40
    band_max_host_fit_rms_m: float = 0.020
    band_min_shape_error_m: float = 0.00035
    band_max_vertex_correction_m: float = 0.030
    band_local_neighbors: int = 24
    band_local_blend: float = 1.00
    band_local_blend_min: float = 0.45
    band_source_witness_tolerance_m: float = 0.00025
    band_source_witness_min_vertices: int = 8

    # Compact hardware seated onto a source-proven broad band/strap.
    hardware_min_vertices: int = 16
    hardware_max_vertices: int = 260
    hardware_max_extent_m: float = 0.100
    hardware_host_min_vertices: int = 150
    hardware_host_min_major_extent_m: float = 0.080
    hardware_host_max_minor_extent_m: float = 0.070
    hardware_source_min_gap_m: float = 0.0015
    hardware_source_p10_gap_m: float = 0.0045
    hardware_anchor_vertices: int = 120
    hardware_scale_min: float = 0.65
    hardware_scale_max: float = 1.40
    hardware_max_vertex_correction_m: float = 0.040

    # Compact authored assembly inference.
    assembly_min_component_vertices: int = 8
    assembly_max_component_vertices: int = 300
    assembly_max_component_extent_m: float = 0.060
    assembly_source_gap_m: float = 0.006
    assembly_min_components: int = 2
    assembly_min_vertices: int = 16
    assembly_max_span_m: float = 0.38
    assembly_scale_min: float = 0.65
    assembly_scale_max: float = 1.40
    assembly_min_layout_error_m: float = 0.00035
    assembly_max_fit_rms_m: float = 0.025
    assembly_max_vertex_correction_m: float = 0.035


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
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


def _components(vertex_count: int, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    F = np.asarray(faces, dtype=np.int64)
    if vertex_count <= 0 or F.size == 0:
        return []
    uf = _UnionFind(vertex_count)
    for a, b, c in F.tolist():
        uf.union(a, b); uf.union(b, c); uf.union(c, a)
    used = np.unique(F.reshape(-1))
    groups: dict[int, list[int]] = {}
    for vi in used.tolist():
        groups.setdefault(uf.find(int(vi)), []).append(int(vi))
    face_roots = np.asarray([uf.find(int(face[0])) for face in F], dtype=np.int64)
    return [
        (np.asarray(sorted(ids), dtype=np.int64), np.flatnonzero(face_roots == int(root)).astype(np.int64))
        for root, ids in groups.items()
    ]


def _fit_similarity(X: np.ndarray, Y: np.ndarray, scale_min: float, scale_max: float) -> tuple[np.ndarray, float, np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    if len(X) < 3 or X.shape != Y.shape:
        return np.eye(3), 1.0, np.zeros(3), float("inf")
    mx = X.mean(axis=0); my = Y.mean(axis=0); Xc = X - mx; Yc = Y - my
    cov = Xc.T @ Yc / float(len(X))
    U, S, Vt = np.linalg.svd(cov, full_matrices=False); R = U @ Vt; sign = 1.0
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0; R = U @ Vt; sign = -1.0
    denom = float(np.mean(np.sum(Xc * Xc, axis=1)))
    scale = float((np.sum(S[:-1]) + sign * S[-1]) / max(denom, _EPS)) if len(S) >= 3 else 1.0
    scale = float(np.clip(scale, float(scale_min), float(scale_max)))
    translation = my - scale * (mx @ R)
    predicted = scale * (X @ R) + translation
    rms = float(np.sqrt(np.mean(np.sum((predicted - Y) ** 2, axis=1))))
    return R, scale, translation, rms




def _fit_weighted_affine(X: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64); w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(X) < 4 or X.shape != Y.shape or len(w) != len(X):
        return np.eye(3), np.zeros(3), float("inf")
    W = np.sqrt(np.maximum(w, _EPS))[:, None]
    XA = np.hstack([X, np.ones((len(X), 1), dtype=np.float64)])
    try:
        coef, *_ = np.linalg.lstsq(XA * W, Y * W, rcond=None)
    except np.linalg.LinAlgError:
        return np.eye(3), np.zeros(3), float("inf")
    A = np.asarray(coef[:3, :], dtype=np.float64)
    t = np.asarray(coef[3, :], dtype=np.float64)
    pred = X @ A + t
    rms = float(np.sqrt(np.mean(np.sum((pred - Y) ** 2, axis=1))))
    return A, t, rms


def _predict_local_host_field(points: np.ndarray, host_source: np.ndarray, host_solved: np.ndarray, tree: cKDTree, neighbors: int) -> tuple[np.ndarray, float]:
    P = np.asarray(points, dtype=np.float64)
    k = max(4, min(int(neighbors), len(host_source)))
    out = np.zeros_like(P)
    rms_values = []
    for i, p in enumerate(P):
        dist, idx = tree.query(p, k=k, workers=1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        if len(idx) < 4:
            out[i] = p
            continue
        weights = 1.0 / np.maximum(dist, 1.0e-4)
        weights /= max(np.sum(weights), _EPS)
        A, t, rms = _fit_weighted_affine(host_source[idx], host_solved[idx], weights)
        out[i] = p @ A + t
        rms_values.append(rms)
    return out, float(np.median(rms_values)) if rms_values else float("inf")

def _descriptor(index: int, vids: np.ndarray, fids: np.ndarray, source: np.ndarray, vertex_group_ids: np.ndarray | None = None) -> dict[str, Any]:
    P = source[vids]; extent = np.ptp(P, axis=0)
    group_id = None
    if vertex_group_ids is not None:
        groups = np.unique(np.asarray(vertex_group_ids, dtype=np.int64)[vids])
        if len(groups) == 1:
            group_id = int(groups[0])
    return {
        "index": int(index), "group_id": group_id, "vids": np.asarray(vids, dtype=np.int64), "fids": np.asarray(fids, dtype=np.int64),
        "vertex_count": int(len(vids)), "face_count": int(len(fids)), "centroid": np.mean(P, axis=0),
        "extent": extent, "max_extent": float(np.max(extent)), "span": float(np.linalg.norm(extent)),
        "bbox_min": np.min(P, axis=0), "bbox_max": np.max(P, axis=0),
    }


def _bbox_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    delta = np.maximum(0.0, np.maximum(a["bbox_min"] - b["bbox_max"], b["bbox_min"] - a["bbox_max"]))
    return float(np.linalg.norm(delta))


def _cap_delta(delta: np.ndarray, cap: float) -> np.ndarray:
    out = np.asarray(delta, dtype=np.float64).copy(); length = np.linalg.norm(out, axis=1); over = length > float(cap)
    if np.any(over):
        out[over] *= (float(cap) / np.maximum(length[over], _EPS))[:, None]
    return out


def _preserve_carried_bands(source: np.ndarray, solved: np.ndarray, components: list[dict[str, Any]], cfg: StructuralCarrierConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = solved.copy(); reports: list[dict[str, Any]] = []
    hosts = [c for c in components if c["vertex_count"] >= int(cfg.band_host_min_vertices)]
    trees = {c["index"]: cKDTree(source[c["vids"]]) for c in hosts}
    for band in components:
        n = int(band["vertex_count"]); ordered = np.sort(np.asarray(band["extent"], dtype=np.float64))
        if n < int(cfg.band_min_vertices) or n > int(cfg.band_max_vertices):
            continue
        if ordered[2] < float(cfg.band_min_major_extent_m) or ordered[1] < float(cfg.band_min_middle_extent_m) or ordered[0] > float(cfg.band_max_minor_extent_m):
            continue
        if ordered[0] / max(float(ordered[1]), _EPS) > float(cfg.band_max_minor_to_middle_ratio):
            continue
        # A disconnected authored band should inherit from its own source mesh when that mesh
        # already contains a valid larger carrier.  A merely closer component from another mesh can
        # be a neighbouring layer/detail rather than the band's actual host and will bend the band
        # around the wrong deformation field.  Cross-mesh fallback remains legal when the source
        # mesh genuinely has no usable carrier.
        eligible = []
        for host in hosts:
            if host["index"] == band["index"] or host["vertex_count"] < max(int(cfg.band_host_min_vertices), int(np.ceil(n * float(cfg.band_host_vertex_ratio)))):
                continue
            distances, _ = trees[host["index"]].query(source[band["vids"]], k=1, workers=1)
            median = float(np.median(distances)); p90 = float(np.percentile(distances, 90))
            if median > float(cfg.band_source_median_clearance_m) or p90 > float(cfg.band_source_p90_clearance_m):
                continue
            score = median + 0.35 * p90
            eligible.append((score, host, median, p90))
        if not eligible:
            continue
        same_group = [row for row in eligible if band.get("group_id") is not None and row[1].get("group_id") == band.get("group_id")]
        best = min(same_group or eligible, key=lambda row: row[0])
        _, host, median, p90 = best; host_ids = np.asarray(host["vids"], dtype=np.int64); tree = trees[host["index"]]
        k = min(len(host_ids), max(24, int(cfg.band_anchor_vertices)))
        _, nearest = tree.query(np.asarray(band["centroid"], dtype=np.float64), k=k, workers=1)
        anchors = np.unique(host_ids[np.asarray(nearest, dtype=np.int64).reshape(-1)])
        if len(anchors) < 12:
            continue
        R, scale, translation, fit_rms = _fit_similarity(source[anchors], out[anchors], float(cfg.band_scale_min), float(cfg.band_scale_max))
        if not np.isfinite(fit_rms) or fit_rms > float(cfg.band_max_host_fit_rms_m):
            reports.append({"band_component":band["index"],"host_component":host["index"],"status":"carrier_fit_rejected","carrier_fit_rms_mm":fit_rms*1000.0,"moved_vertices":0})
            continue
        ids = np.asarray(band["vids"], dtype=np.int64)
        predicted_global = scale * (source[ids] @ R) + translation
        predicted_local, local_rms = _predict_local_host_field(source[ids], source[host_ids], out[host_ids], tree, int(cfg.band_local_neighbors))
        if np.isfinite(local_rms):
            base_blend = float(np.clip(cfg.band_local_blend, 0.0, 1.0))
            min_blend = float(np.clip(cfg.band_local_blend_min, 0.0, base_blend))
            support_dist, _ = tree.query(source[ids], k=1, workers=1)
            support_dist = np.asarray(support_dist, dtype=np.float64).reshape(-1)
            d0 = float(np.percentile(support_dist, 5))
            d1 = float(np.percentile(support_dist, 90))
            denom = max(d1 - d0, _EPS)
            support_t = np.clip((support_dist - d0) / denom, 0.0, 1.0)
            # Closest support-side vertices must truly seat onto the solved host; farther outer/profile
            # vertices retain more authored structure.  This creates an actual layered cuff/belt solve
            # instead of a nearly unchanged broad blend.
            blend = base_blend - (base_blend - min_blend) * np.square(support_t)
            support_threshold = float(np.percentile(support_dist, 35))
            support_mask = support_dist <= support_threshold
            blend[support_mask] = 1.0
            predicted = predicted_global * (1.0 - blend[:, None]) + predicted_local * blend[:, None]
            blend_report = float(np.median(blend))
        else:
            blend = np.zeros(len(ids), dtype=np.float64)
            blend_report = 0.0
            predicted = predicted_global

        # Exact source witness closure: when a disconnected authored band has vertices coincident
        # with its host in the untouched source, that is explicit topology evidence that the seam is
        # attached.  Carry the already-fitted band shape, but smoothly translate it so those source
        # witnesses land on the corresponding solved host vertices.  This is stronger and more
        # authoritative than proximity blending, and prevents a cuff from remaining visibly detached
        # even though its host shell has genuinely refit.
        source_gap, source_nn = tree.query(source[ids], k=1, workers=1)
        source_gap = np.asarray(source_gap, dtype=np.float64).reshape(-1)
        source_nn = np.asarray(source_nn, dtype=np.int64).reshape(-1)
        witness_mask = source_gap <= float(cfg.band_source_witness_tolerance_m)
        witness_count = int(np.count_nonzero(witness_mask))
        if witness_count >= int(cfg.band_source_witness_min_vertices):
            witness_local = np.flatnonzero(witness_mask)
            witness_source = source[ids[witness_local]]
            witness_target = out[host_ids[source_nn[witness_local]]]
            witness_residual = witness_target - predicted[witness_local]
            witness_tree = cKDTree(witness_source)
            k_witness = min(4, witness_count)
            dist_w, idx_w = witness_tree.query(source[ids], k=k_witness, workers=1)
            dist_w = np.asarray(dist_w, dtype=np.float64)
            idx_w = np.asarray(idx_w, dtype=np.int64)
            if k_witness == 1:
                dist_w = dist_w[:, None]; idx_w = idx_w[:, None]
            weights = 1.0 / np.maximum(dist_w, 1.0e-5)
            weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), _EPS)
            correction = np.sum(witness_residual[idx_w] * weights[:, :, None], axis=1)
            predicted = predicted + correction
            predicted[witness_local] = witness_target
        error = np.linalg.norm(predicted - out[ids], axis=1); p95 = float(np.percentile(error, 95))
        if p95 < float(cfg.band_min_shape_error_m):
            reports.append({"band_component":band["index"],"host_component":host["index"],"status":"already_carrier_relative","carrier_scale":scale,"shape_error_p95_mm":p95*1000.0,"moved_vertices":0})
            continue
        delta = _cap_delta(predicted - out[ids], float(cfg.band_max_vertex_correction_m)); out[ids] += delta
        move = np.linalg.norm(delta, axis=1); moved = move > 1.0e-12
        post = np.linalg.norm(predicted - out[ids], axis=1)
        reports.append({
            "band_component":band["index"],"host_component":host["index"],"status":"carrier_relative_band_corrected",
            "carrier_scale":scale,"carrier_fit_rms_mm":fit_rms*1000.0,"local_host_field_rms_mm":local_rms*1000.0 if np.isfinite(local_rms) else None,
            "local_host_field_blend":blend_report,"local_host_field_blend_min":float(np.min(blend)) if len(blend) else 0.0,
            "local_host_field_blend_max":float(np.max(blend)) if len(blend) else 0.0,"source_witness_vertices":witness_count,
            "source_clearance_median_mm":median*1000.0,"source_clearance_p90_mm":p90*1000.0,
            "shape_error_p95_mm":p95*1000.0,"post_error_p95_mm":float(np.percentile(post,95)*1000.0),
            "moved_vertices":int(np.count_nonzero(moved)),"maximum_correction_mm":float(np.max(move,initial=0.0)*1000.0),
        })
    return out, reports


def _compact_assembly_groups(source: np.ndarray, components: list[dict[str, Any]], cfg: StructuralCarrierConfig) -> list[list[dict[str, Any]]]:
    candidates = [c for c in components if int(cfg.assembly_min_component_vertices) <= c["vertex_count"] <= int(cfg.assembly_max_component_vertices) and c["max_extent"] <= float(cfg.assembly_max_component_extent_m)]
    if not candidates:
        return []
    index_by_component = {c["index"]: i for i, c in enumerate(candidates)}; uf = _UnionFind(len(candidates)); trees = {c["index"]: cKDTree(source[c["vids"]]) for c in candidates}
    gap = float(cfg.assembly_source_gap_m)
    for ia, a in enumerate(candidates):
        for ib in range(ia + 1, len(candidates)):
            b = candidates[ib]
            if _bbox_gap(a, b) > gap:
                continue
            if float(np.linalg.norm(a["centroid"] - b["centroid"])) > float(a["span"] + b["span"] + 3.0 * gap):
                continue
            if len(a["vids"]) <= len(b["vids"]):
                nearest = trees[b["index"]].query(source[a["vids"]], k=1, workers=1)[0]
            else:
                nearest = trees[a["index"]].query(source[b["vids"]], k=1, workers=1)[0]
            if float(np.min(nearest, initial=float("inf"))) <= gap:
                uf.union(ia, ib)
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, comp in enumerate(candidates):
        groups.setdefault(uf.find(i), []).append(comp)
    accepted = []
    for group in groups.values():
        if len(group) < int(cfg.assembly_min_components):
            continue
        ids = np.concatenate([c["vids"] for c in group]); P = source[ids]
        if len(ids) < int(cfg.assembly_min_vertices) or float(np.max(np.ptp(P, axis=0))) > float(cfg.assembly_max_span_m):
            continue
        accepted.append(sorted(group, key=lambda c: int(c["index"])))
    return accepted


def _preserve_compact_assemblies(source: np.ndarray, solved: np.ndarray, components: list[dict[str, Any]], cfg: StructuralCarrierConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = solved.copy(); reports: list[dict[str, Any]] = []
    for group in _compact_assembly_groups(source, components, cfg):
        ids = np.unique(np.concatenate([c["vids"] for c in group]).astype(np.int64)); component_ids = [int(c["index"]) for c in group]
        R, scale, translation, fit_rms = _fit_similarity(source[ids], out[ids], float(cfg.assembly_scale_min), float(cfg.assembly_scale_max))
        predicted = scale * (source[ids] @ R) + translation; error = np.linalg.norm(predicted - out[ids], axis=1); p95 = float(np.percentile(error, 95))
        if not np.isfinite(fit_rms) or fit_rms > float(cfg.assembly_max_fit_rms_m):
            reports.append({"component_ids":component_ids,"component_count":len(group),"status":"assembly_fit_rejected","fit_rms_mm":fit_rms*1000.0,"layout_error_p95_mm":p95*1000.0,"moved_vertices":0}); continue
        if p95 < float(cfg.assembly_min_layout_error_m):
            reports.append({"component_ids":component_ids,"component_count":len(group),"status":"already_coherent","carrier_scale":scale,"fit_rms_mm":fit_rms*1000.0,"layout_error_p95_mm":p95*1000.0,"moved_vertices":0}); continue
        delta = _cap_delta(predicted - out[ids], float(cfg.assembly_max_vertex_correction_m)); out[ids] += delta
        move = np.linalg.norm(delta, axis=1); moved = move > 1.0e-12; post = np.linalg.norm(predicted - out[ids], axis=1)
        reports.append({
            "component_ids":component_ids,"component_count":len(group),"vertex_count":int(len(ids)),"status":"coherent_assembly_corrected",
            "carrier_scale":scale,"fit_rms_mm":fit_rms*1000.0,"layout_error_p95_mm":p95*1000.0,"post_error_p95_mm":float(np.percentile(post,95)*1000.0),
            "source_span_mm":float(np.max(np.ptp(source[ids],axis=0))*1000.0),"moved_vertices":int(np.count_nonzero(moved)),"maximum_correction_mm":float(np.max(move,initial=0.0)*1000.0),
        })
    return out, reports




def _seat_compact_hardware_on_bands(source: np.ndarray, solved: np.ndarray, components: list[dict[str, Any]], cfg: StructuralCarrierConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Seat compact rigid/source-authored hardware onto the solved broad band that carried it.

    Proximity alone is not enough to make unrelated decorations rigid peers.  Here we require strong
    source support evidence: a compact component must sit within a few millimetres of a broad band in
    the untouched source.  The hardware then receives one similarity transform inferred from that
    band's already-solved local neighbourhood, preserving its authored shape while correcting its
    placement/orientation on the fitted strap/cuff.
    """
    out = solved.copy(); reports: list[dict[str, Any]] = []
    hosts = []
    for c in components:
        ordered = np.sort(np.asarray(c["extent"], dtype=np.float64))
        if c["vertex_count"] < int(cfg.hardware_host_min_vertices):
            continue
        if ordered[2] < float(cfg.hardware_host_min_major_extent_m) or ordered[0] > float(cfg.hardware_host_max_minor_extent_m):
            continue
        hosts.append(c)
    if not hosts:
        return out, reports
    trees = {c["index"]: cKDTree(source[c["vids"]]) for c in hosts}
    host_ids_by_index = {c["index"]: np.asarray(c["vids"], dtype=np.int64) for c in hosts}
    for hardware in components:
        n = int(hardware["vertex_count"])
        if n < int(cfg.hardware_min_vertices) or n > int(cfg.hardware_max_vertices) or float(hardware["max_extent"]) > float(cfg.hardware_max_extent_m):
            continue
        eligible = []
        hw_ids = np.asarray(hardware["vids"], dtype=np.int64)
        for host in hosts:
            if host["index"] == hardware["index"]:
                continue
            d, _ = trees[host["index"]].query(source[hw_ids], k=1, workers=1)
            d = np.asarray(d, dtype=np.float64)
            min_gap = float(np.min(d, initial=float("inf"))); p10 = float(np.percentile(d, 10))
            if min_gap > float(cfg.hardware_source_min_gap_m) or p10 > float(cfg.hardware_source_p10_gap_m):
                continue
            eligible.append((p10 + 0.25 * min_gap, host, min_gap, p10))
        if not eligible:
            continue
        same_group = [row for row in eligible if hardware.get("group_id") is not None and row[1].get("group_id") == hardware.get("group_id")]
        _, host, min_gap, p10 = min(same_group or eligible, key=lambda row: row[0])
        host_ids = host_ids_by_index[host["index"]]; tree = trees[host["index"]]
        k = min(len(host_ids), max(16, int(cfg.hardware_anchor_vertices)))
        _, nearest = tree.query(np.asarray(hardware["centroid"], dtype=np.float64), k=k, workers=1)
        anchors = np.unique(host_ids[np.asarray(nearest, dtype=np.int64).reshape(-1)])
        if len(anchors) < 8:
            continue
        R, scale, translation, fit_rms = _fit_similarity(source[anchors], out[anchors], float(cfg.hardware_scale_min), float(cfg.hardware_scale_max))
        if not np.isfinite(fit_rms):
            continue
        predicted = scale * (source[hw_ids] @ R) + translation
        delta = _cap_delta(predicted - out[hw_ids], float(cfg.hardware_max_vertex_correction_m))
        move = np.linalg.norm(delta, axis=1)
        if float(np.max(move, initial=0.0)) <= 1.0e-12:
            continue
        out[hw_ids] += delta
        reports.append({
            "hardware_component": int(hardware["index"]), "host_component": int(host["index"]),
            "source_min_gap_mm": min_gap * 1000.0, "source_p10_gap_mm": p10 * 1000.0,
            "carrier_scale": scale, "carrier_fit_rms_mm": fit_rms * 1000.0,
            "moved_vertices": int(np.count_nonzero(move > 1.0e-12)),
            "maximum_correction_mm": float(np.max(move, initial=0.0) * 1000.0),
            "status": "surface_seated_hardware_corrected",
        })
    return out, reports

def preserve_source_relative_structural_carriers(source_vertices: np.ndarray, solved_vertices: np.ndarray, faces: np.ndarray, *, config: StructuralCarrierConfig | None = None, vertex_group_ids: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = config or StructuralCarrierConfig(); source = np.asarray(source_vertices, dtype=np.float64); solved = np.asarray(solved_vertices, dtype=np.float64); F = np.asarray(faces, dtype=np.int64)
    if source.ndim != 2 or source.shape[1] != 3 or solved.shape != source.shape:
        raise ValueError("source_vertices and solved_vertices must be matching Nx3 arrays")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(solved)):
        raise ValueError("structural carrier preservation received non-finite vertices")
    groups = None if vertex_group_ids is None else np.asarray(vertex_group_ids, dtype=np.int64)
    if groups is not None and groups.shape != (len(source),):
        raise ValueError(f"vertex_group_ids must have shape {(len(source),)}, got {groups.shape}")
    descriptors = [_descriptor(i, vids, fids, source, groups) for i, (vids, fids) in enumerate(_components(len(source), F))]
    after_bands, band_reports = _preserve_carried_bands(source, solved, descriptors, cfg)
    after_assemblies, assembly_reports = _preserve_compact_assemblies(source, after_bands, descriptors, cfg)
    out, hardware_reports = _seat_compact_hardware_on_bands(source, after_assemblies, descriptors, cfg)
    move = np.linalg.norm(out - solved, axis=1); corrected_bands = [r for r in band_reports if r.get("status") == "carrier_relative_band_corrected"]; corrected_assemblies = [r for r in assembly_reports if r.get("status") == "coherent_assembly_corrected"]; corrected_hardware = [r for r in hardware_reports if r.get("status") == "surface_seated_hardware_corrected"]
    return out, {
        "enabled": bool(corrected_bands or corrected_assemblies or corrected_hardware),
        "policy": "source-proven broad bands inherit one solved host similarity frame; nearby compact authored components share one coherent source-relative carrier frame so shape, spacing and containment survive target refit",
        "component_count": len(descriptors), "corrected_band_count": len(corrected_bands), "corrected_assembly_count": len(corrected_assemblies), "corrected_hardware_count": len(corrected_hardware),
        "moved_vertices": int(np.count_nonzero(move > 1.0e-12)), "maximum_correction_mm": float(np.max(move, initial=0.0) * 1000.0),
        "bands": band_reports, "assemblies": assembly_reports, "hardware": hardware_reports,
    }
