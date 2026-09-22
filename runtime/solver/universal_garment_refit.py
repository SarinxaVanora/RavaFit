from __future__ import annotations

"""Universal coherent post-B14 garment refit.

The contract is intentionally small:

* source garment geometry is construction/detail authority;
* the selected target body is fit authority;
* source->target body motion is transferred as a *continuous low-frequency field*;
* source clearance controls how strongly a garment follows that field;
* spatially attached disconnected pieces share the same field, so straps/rings/trims do not drift;
* far/rigid structures over a nearly-identical body mapping remain authored geometry;
* literal target geometry is collision authority only and any correction is spatially distributed.

There are no slot, outfit, body, bra, breast, boot, stocking, or Hot Topic conditions here.
"""

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_EPS = 1e-12


@dataclass(frozen=True)
class UniversalRefitConfig:
    # Source-clearance envelope.  Close garments follow the target strongly; stand-off pieces fade
    # continuously rather than switching between hand-authored garment categories.
    full_follow_clearance_m: float = 0.0030
    zero_follow_clearance_m: float = 0.0320
    clearance_power: float = 1.15

    # Spatial attachment edges join disconnected authored pieces that physically meet in the source.
    attachment_radius_m: float = 0.0045
    attachment_confirm_radius_m: float = 0.00125
    attachment_min_witnesses: int = 3
    attachment_tiny_component_radius_m: float = 0.00055
    attachment_k: int = 12
    attachment_weight: float = 0.85

    # The body correspondence is smoothed, never the garment surface.  This preserves folds, seams,
    # rings, trim and authored local detail while filtering nipple/rib/muscle-scale body relief.
    field_iterations: int = 18
    field_alpha: float = 0.42

    # Moderately close / stand-off authored structures should not receive a noisy per-vertex mix of
    # B14 and body-following geometry.  Once source clearance proves that the garment is not a
    # skin-like layer, preserve the clean source construction and carry it through a smoother,
    # component-coherent body displacement field.
    structured_clearance_m: float = 0.0060
    structured_field_iterations: int = 30
    structured_field_alpha: float = 0.40
    structured_follow_iterations: int = 14
    structured_follow_alpha: float = 0.42
    structured_component_follow_blend: float = 0.78
    structured_component_min_vertices: int = 24
    structured_offset_scale_exponent: float = 0.50
    structured_offset_scale_min: float = 0.65
    structured_offset_scale_max: float = 1.35
    structured_direct_affine_clearance_m: float = 0.045

    # Body-independent authored construction: if a sufficiently stand-off source structure already
    # clears the selected target, there is no fit problem to solve.  Preserve the source exactly.
    # This is deliberately inferred from geometry/contact rather than garment names or controls.
    authored_standoff_noop_clearance_m: float = 0.0150
    authored_standoff_noop_target_p01_m: float = 0.00025
    # Exact source no-op requires strong evidence that the garment is genuinely body-independent:
    # the target body must have changed substantially *relative to the garment's authored clearance*
    # while the exact source construction still clears it.  This prevents loose/stand-off boots,
    # gloves and other body-following equipment from being frozen merely because the target is smaller.
    authored_standoff_noop_min_motion_clearance_ratio: float = 1.50

    # Shape-preserving authored assemblies.  Jewellery, rigid harness pieces, ornament clusters and
    # other multi-component constructions can sit closer than a jacket while still being designed
    # as one rigid authored object.  If exact source is already almost valid on the target, do not
    # reshape thousands of vertices: preserve the complete source shape and solve only the smallest
    # coherent rigid placement needed to clear shallow contact.
    authored_rigid_min_components: int = 20
    authored_rigid_min_clearance_m: float = 0.0050
    authored_rigid_max_penetrating_fraction: float = 0.015
    authored_rigid_max_initial_penetration_m: float = 0.0020
    authored_rigid_clearance_margin_m: float = 0.00010
    authored_rigid_max_translation_m: float = 0.00250
    authored_rigid_projection_iterations: int = 512
    authored_rigid_requery_iterations: int = 4
    # Raw disconnected-component count is not rigidity evidence in FFXIV assets: seam splits,
    # decorative islands and triangle storage can create thousands of components in ordinary cloth.
    # A true rigid accessory/assembly must also have compact skinning authority: almost all vertices
    # are carried by at most two dominant bones and the weight field is spatially coherent.
    authored_rigid_min_top2_dominant_fraction: float = 0.90
    authored_rigid_max_weight_deviation_p95: float = 0.90

    # If the body underneath a stand-off structure moves by less than the garment's own clearance,
    # there is no evidence that the structure needs a visible refit.  This is continuous and generic.
    motion_clearance_ratio: float = 0.75
    near_identity_motion_floor_m: float = 0.0060

    # Literal-body collision is distributed through the same assembly graph instead of vertex-pushed.
    collision_margin_m: float = 0.00012
    collision_diffusion_iterations: int = 8
    collision_alpha: float = 0.55
    collision_max_move_m: float = 0.0020
    collision_final_direct_cap_m: float = 0.00025

    # Hard safety budgets.  If coherent refit cannot satisfy them, blend monotonically back toward B14.
    min_triangle_area_ratio: float = 0.32
    max_blend_back_steps: int = 10


def _component_labels(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    F = np.asarray(faces, dtype=np.int64)
    if vertex_count <= 0:
        return np.zeros(0, dtype=np.int64)
    if not len(F):
        return np.arange(vertex_count, dtype=np.int64)
    e = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    rows = np.r_[e[:, 0], e[:, 1]]
    cols = np.r_[e[:, 1], e[:, 0]]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(vertex_count, vertex_count)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64)


def _graph(vertices: np.ndarray, faces: np.ndarray, cfg: UniversalRefitConfig) -> tuple[csr_matrix, np.ndarray, list[tuple[int, int, float]]]:
    """Topology graph plus source-proven cross-component physical attachments.

    Spatial proximity alone is not attachment evidence.  Bilateral pieces such as stockings can
    legitimately pass within a few millimetres while remaining independent authored structures.
    A cross-component bridge is therefore admitted only when the source contains repeated near-
    coincident witnesses for that same component pair (or a tiny hardware component is essentially
    touching its parent).
    """
    P=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);n=len(P)
    labels=_component_labels(n,F)
    edge_weight: dict[tuple[int,int],float]={}
    if len(F):
        for a,b in np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])):
            a=int(a);b=int(b)
            if a==b:continue
            key=(a,b) if a<b else (b,a);edge_weight[key]=max(edge_weight.get(key,0.0),1.0)

    attachments: list[tuple[int,int,float]]=[]
    unique_labels=np.unique(labels)
    if n>1 and len(unique_labels)>1:
        sizes={int(lab):int(np.count_nonzero(labels==lab)) for lab in unique_labels}
        tree=cKDTree(P);k=min(max(2,int(cfg.attachment_k)),n)
        distances,neighbours=tree.query(P,k=k)
        if k==1:distances=distances[:,None];neighbours=neighbours[:,None]
        pair_candidates: dict[tuple[int,int],list[tuple[int,int,float]]]={}
        for i in range(n):
            for d,j in zip(np.asarray(distances[i]).reshape(-1)[1:],np.asarray(neighbours[i]).reshape(-1)[1:]):
                j=int(j);d=float(d)
                if d>cfg.attachment_radius_m:break
                if i==j or labels[i]==labels[j]:continue
                ca=int(labels[i]);cb=int(labels[j]);pair=(ca,cb) if ca<cb else (cb,ca)
                a,b=(i,j) if i<j else (j,i)
                pair_candidates.setdefault(pair,[]).append((a,b,d))

        for pair,raw in pair_candidates.items():
            # Deduplicate vertex pairs, keeping the closest observation.
            best: dict[tuple[int,int],float]={}
            for a,b,d in raw:
                key=(a,b);best[key]=min(best.get(key,float('inf')),float(d))
            witnesses=[(a,b,d) for (a,b),d in best.items() if d<=cfg.attachment_confirm_radius_m]
            min_distance=min(best.values(),default=float('inf'))
            tiny=min(sizes.get(pair[0],0),sizes.get(pair[1],0))<=8
            confirmed=len(witnesses)>=int(cfg.attachment_min_witnesses) or (tiny and min_distance<=cfg.attachment_tiny_component_radius_m)
            if not confirmed:continue
            accepted=witnesses if witnesses else [(a,b,d) for (a,b),d in best.items() if d<=cfg.attachment_tiny_component_radius_m]
            # Do not create a dense zipper between components.  A few closest source witnesses are
            # enough for field continuity and preserve authored separation elsewhere.
            accepted=sorted(accepted,key=lambda x:x[2])[:max(int(cfg.attachment_min_witnesses),8)]
            for a,b,d in accepted:
                w=float(cfg.attachment_weight*np.clip(1.0-d/max(cfg.attachment_confirm_radius_m,_EPS),0.20,1.0))
                edge_weight[(a,b)]=max(edge_weight.get((a,b),0.0),w);attachments.append((a,b,d))

    if not edge_weight:return csr_matrix((n,n),dtype=np.float64),labels,attachments
    rows=[];cols=[];data=[]
    for (a,b),w in edge_weight.items():rows.extend((a,b));cols.extend((b,a));data.extend((w,w))
    return coo_matrix((data,(rows,cols)),shape=(n,n),dtype=np.float64).tocsr(),labels,attachments


def _diffuse(values: np.ndarray, adjacency: csr_matrix, iterations: int, alpha: float) -> np.ndarray:
    V = np.asarray(values, dtype=np.float64).copy()
    if not len(V) or adjacency.nnz == 0 or iterations <= 0:
        return V
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    for _ in range(int(iterations)):
        avg = adjacency @ V
        avg /= np.maximum(degree[:, None], _EPS)
        isolated = degree <= _EPS
        if np.any(isolated):
            avg[isolated] = V[isolated]
        V = (1.0 - float(alpha)) * V + float(alpha) * avg
    return V


def _topology_ok(reference: np.ndarray, candidate: np.ndarray, faces: np.ndarray, min_area_ratio: float) -> tuple[bool, int, float]:
    R=np.asarray(reference,dtype=np.float64); C=np.asarray(candidate,dtype=np.float64); F=np.asarray(faces,dtype=np.int64)
    if not len(F):
        return True, 0, 1.0
    ar=np.cross(R[F[:,1]]-R[F[:,0]], R[F[:,2]]-R[F[:,0]])
    ac=np.cross(C[F[:,1]]-C[F[:,0]], C[F[:,2]]-C[F[:,0]])
    nr=np.linalg.norm(ar,axis=1); nc=np.linalg.norm(ac,axis=1)
    valid=nr>1e-12
    dot=np.einsum("ij,ij->i",ar,ac)
    flips=int(np.count_nonzero(valid & (dot < 0.0)))
    ratios=nc[valid]/np.maximum(nr[valid],_EPS)
    minimum=float(np.min(ratios,initial=1.0))
    return bool(flips==0 and minimum>=float(min_area_ratio)), flips, minimum


def _follow_strength(clearance: np.ndarray, median_clearance: float, body_motion_p95: float, cfg: UniversalRefitConfig) -> np.ndarray:
    c=np.abs(np.asarray(clearance,dtype=np.float64))
    span=max(cfg.zero_follow_clearance_m-cfg.full_follow_clearance_m,_EPS)
    raw=np.clip((cfg.zero_follow_clearance_m-c)/span,0.0,1.0)
    raw=np.where(c<=cfg.full_follow_clearance_m,1.0,raw)
    raw=np.power(raw,float(cfg.clearance_power))

    # Evidence gate: a stand-off structure should not visibly morph when the underlying body motion
    # is smaller than the free space it was already authored to have.  Tight layers are exempt.
    if median_clearance>cfg.full_follow_clearance_m:
        threshold=max(float(cfg.near_identity_motion_floor_m),float(cfg.motion_clearance_ratio)*float(median_clearance))
        if body_motion_p95<=threshold:
            return np.zeros_like(raw)
        evidence=np.clip((body_motion_p95-threshold)/max(threshold,_EPS),0.0,1.0)
        raw*=float(evidence)
    return raw



def fit_body_macro_transform(source_body_vertices: np.ndarray, target_body_vertices: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,float]]:
    """Fit one shared low-frequency affine body-frame transform.

    Every structured garment piece in the same body slot receives this exact transform, so panels,
    collars, trims and disconnected authored parts cannot drift relative to one another.  The polar
    rotation is returned separately so garment stand-off can retain authored direction/clearance
    without inheriting non-uniform body compression verbatim.
    """
    X=np.asarray(source_body_vertices,dtype=np.float64);Y=np.asarray(target_body_vertices,dtype=np.float64)
    if X.shape!=Y.shape or X.ndim!=2 or X.shape[1]!=3 or len(X)<4:
        raise ValueError('source and target body macro vertices must be matching Nx3 arrays')
    design=np.column_stack((X,np.ones(len(X),dtype=np.float64)))
    M,*_=np.linalg.lstsq(design,Y,rcond=None)
    linear=np.asarray(M[:3,:].T,dtype=np.float64)
    U,s,Vt=np.linalg.svd(linear)
    s=np.clip(s,.45,1.35)
    linear=(U*s[None,:])@Vt
    rotation=U@Vt
    sx=np.mean(X,axis=0);sy=np.mean(Y,axis=0);translation=sy-linear@sx
    residual=np.linalg.norm(X@linear.T+translation-Y,axis=1)
    return linear,translation,rotation,{
        'singular_value_min':float(np.min(s)),'singular_value_mid':float(np.median(s)),'singular_value_max':float(np.max(s)),
        'fit_residual_p50_mm':float(np.percentile(residual,50)*1000.0),'fit_residual_p95_mm':float(np.percentile(residual,95)*1000.0),
    }


def _structured_macro_transport(source_vertices: np.ndarray, source_surface_points: np.ndarray, linear: np.ndarray, translation: np.ndarray, rotation: np.ndarray, median_clearance: float, cfg: UniversalRefitConfig) -> np.ndarray:
    """Carry clean source construction through the shared body frame while retaining stand-off.

    The source-body support point follows the full macro body affine.  The garment offset from that
    support is rotated coherently and scales by sqrt(body scale along the offset), which retains more
    authored clearance than the body itself while still adapting to strong source->target size
    changes. Very remote structures use the shared affine directly because nearest-body support is no
    longer meaningful at that distance.
    """
    P=np.asarray(source_vertices,dtype=np.float64);support=np.asarray(source_surface_points,dtype=np.float64)
    if float(median_clearance)>=float(cfg.structured_direct_affine_clearance_m):
        return P@linear.T+translation
    offset=P-support
    rotated=offset@rotation.T
    transformed=offset@linear.T
    olen=np.linalg.norm(offset,axis=1);tlen=np.linalg.norm(transformed,axis=1)
    scale=np.ones(len(P),dtype=np.float64)
    good=olen>1e-9
    scale[good]=np.power(np.clip(tlen[good]/olen[good],.05,4.0),float(cfg.structured_offset_scale_exponent))
    scale=np.clip(scale,float(cfg.structured_offset_scale_min),float(cfg.structured_offset_scale_max))
    return support@linear.T+translation+rotated*scale[:,None]


def _rigid_skinning_evidence(source_weights: np.ndarray | None, cfg: UniversalRefitConfig) -> dict[str, Any]:
    if source_weights is None:
        return {"selected":False,"reason":"no authored skinning evidence supplied"}
    W=np.asarray(source_weights,dtype=np.float64)
    if W.ndim!=2 or not len(W) or W.shape[1]==0:
        return {"selected":False,"reason":"invalid authored skinning evidence"}
    sums=np.sum(W,axis=1);good=sums>1e-10
    if int(np.count_nonzero(good))<max(8,int(.50*len(W))):
        return {"selected":False,"reason":"insufficient weighted vertices for rigid evidence"}
    N=np.zeros_like(W);N[good]=W[good]/sums[good,None]
    dominant=np.argmax(N[good],axis=1)
    _,counts=np.unique(dominant,return_counts=True)
    counts=np.sort(counts)[::-1]
    top2_fraction=float(np.sum(counts[:2]))/float(len(dominant)) if len(dominant) else 0.0
    mean=np.mean(N[good],axis=0)
    deviation=np.sum(np.abs(N[good]-mean[None,:]),axis=1)
    deviation_p95=float(np.percentile(deviation,95)) if len(deviation) else float('inf')
    selected=(top2_fraction>=float(cfg.authored_rigid_min_top2_dominant_fraction)
              and deviation_p95<=float(cfg.authored_rigid_max_weight_deviation_p95))
    return {
        "selected":bool(selected),
        "top2_dominant_fraction":top2_fraction,
        "weight_deviation_p95":deviation_p95,
        "reason":"compact coherent authored skinning supports rigid assembly" if selected else "skinning field is too distributed for rigid assembly",
    }


def _shape_preserving_rigid_clearance(
    source_vertices: np.ndarray,
    target_body_triangles: np.ndarray,
    *,
    nearest_surface_fn: Callable[..., Any],
    component_count: int,
    median_clearance: float,
    source_weights: np.ndarray | None,
    cfg: UniversalRefitConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Preserve a multi-component authored assembly and solve only coherent placement.

    This lane is deliberately conservative.  It is accepted only when exact source is already almost
    valid on the target: few vertices may penetrate, penetration must be shallow, and one tiny rigid
    translation must clear the literal target.  Geometry, component relationships, lengths and local
    detail remain bit-for-bit source apart from that shared translation.
    """
    P=np.asarray(source_vertices,dtype=np.float64)
    target=np.asarray(target_body_triangles,dtype=np.float64)
    if int(component_count)<int(cfg.authored_rigid_min_components):
        return None,{"selected":False,"reason":"insufficient authored component evidence"}
    if float(median_clearance)<float(cfg.authored_rigid_min_clearance_m):
        return None,{"selected":False,"reason":"source is too body-close for rigid authored placement"}
    skinning=_rigid_skinning_evidence(source_weights,cfg)
    if not bool(skinning.get("selected")):
        return None,{"selected":False,"reason":"multi-component topology is not sufficient rigid evidence","skinning":skinning}

    initial=nearest_surface_fn(P,target,k=24)
    initial_signed=np.asarray(initial[2],dtype=np.float64)
    if not len(initial_signed):
        return None,{"selected":False,"reason":"no target contact samples"}
    penetrating=initial_signed<0.0
    penetration_fraction=float(np.count_nonzero(penetrating))/float(len(P))
    maximum_penetration=float(max(0.0,-float(np.min(initial_signed))))
    if penetration_fraction>float(cfg.authored_rigid_max_penetrating_fraction):
        return None,{"selected":False,"reason":"too much of source intersects target for rigid placement","penetrating_fraction":penetration_fraction}
    if maximum_penetration>float(cfg.authored_rigid_max_initial_penetration_m):
        return None,{"selected":False,"reason":"source-target contact is too deep for rigid placement","maximum_penetration_mm":maximum_penetration*1000.0}
    if not np.any(penetrating):
        return P.copy(),{
            "selected":True,"reason":"multi-component authored assembly already clears target; exact source preserved",
            "translation_mm":[0.0,0.0,0.0],"translation_norm_mm":0.0,
            "initial_penetrating_vertices":0,"final_penetrating_vertices":0,
            "initial_maximum_penetration_mm":0.0,"skinning":skinning,
        }

    total=np.zeros(3,dtype=np.float64)
    margin=float(cfg.authored_rigid_clearance_margin_m)
    max_translation=float(cfg.authored_rigid_max_translation_m)
    for _ in range(max(1,int(cfg.authored_rigid_requery_iterations))):
        candidate=P+total[None,:]
        query=nearest_surface_fn(candidate,target,k=24)
        normals=np.asarray(query[1],dtype=np.float64)
        signed=np.asarray(query[2],dtype=np.float64)
        violating=np.flatnonzero(signed<margin-1e-10)
        if not len(violating):
            final_signed=signed
            break
        local=np.zeros(3,dtype=np.float64)
        vn=normals[violating]
        vs=signed[violating]
        for _ in range(max(1,int(cfg.authored_rigid_projection_iterations))):
            value=vs+vn@local-margin
            j=int(np.argmin(value))
            if float(value[j])>=-1e-10:
                break
            normal=vn[j]
            denom=float(np.dot(normal,normal))+_EPS
            local+=(-float(value[j])/denom)*normal
            if float(np.linalg.norm(total+local))>max_translation+1e-12:
                return None,{
                    "selected":False,"reason":"required coherent rigid placement exceeds displacement budget",
                    "required_translation_norm_mm":float(np.linalg.norm(total+local))*1000.0,
                }
        total+=local
    else:
        final_query=nearest_surface_fn(P+total[None,:],target,k=24)
        final_signed=np.asarray(final_query[2],dtype=np.float64)

    if float(np.linalg.norm(total))>max_translation+1e-12:
        return None,{"selected":False,"reason":"coherent rigid placement exceeded displacement budget"}
    final_query=nearest_surface_fn(P+total[None,:],target,k=24)
    final_signed=np.asarray(final_query[2],dtype=np.float64)
    final_penetrating=int(np.count_nonzero(final_signed<0.0))
    if final_penetrating:
        return None,{
            "selected":False,"reason":"coherent rigid placement could not clear literal target",
            "final_penetrating_vertices":final_penetrating,
            "translation_norm_mm":float(np.linalg.norm(total))*1000.0,
        }
    return P+total[None,:],{
        "selected":True,
        "reason":"multi-component authored assembly preserved; shallow target contact cleared by one coherent rigid placement",
        "translation_mm":[float(v*1000.0) for v in total],
        "translation_norm_mm":float(np.linalg.norm(total))*1000.0,
        "initial_penetrating_vertices":int(np.count_nonzero(initial_signed<0.0)),
        "final_penetrating_vertices":0,
        "initial_maximum_penetration_mm":maximum_penetration*1000.0,
        "final_minimum_clearance_mm":float(np.min(final_signed))*1000.0,
    }

def coherent_universal_refit(
    source_vertices: np.ndarray,
    faces: np.ndarray,
    b14_vertices: np.ndarray,
    correspondence_vertices: np.ndarray,
    source_body_triangles: np.ndarray,
    target_body_triangles: np.ndarray,
    *,
    body_motion_p95_m: float,
    nearest_surface_fn: Callable[..., Any],
    macro_transform: tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,float]] | None = None,
    source_weights: np.ndarray | None = None,
    allow_authored_standoff_noop: bool = True,
    config: UniversalRefitConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit one authored mesh coherently without garment/slot-specific rules."""
    cfg=config or UniversalRefitConfig()
    P=np.asarray(source_vertices,dtype=np.float64)
    F=np.asarray(faces,dtype=np.int64)
    B=np.asarray(b14_vertices,dtype=np.float64)
    C=np.asarray(correspondence_vertices,dtype=np.float64)
    if P.shape!=B.shape or B.shape!=C.shape:
        raise ValueError("source, B14 and correspondence vertices must have identical shape")
    if not len(P):
        return B.copy(), {"selected":False,"reason":"empty geometry"}

    source_query=nearest_surface_fn(P,np.asarray(source_body_triangles,dtype=np.float64),k=24)
    source_support_points=np.asarray(source_query[0],dtype=np.float64)
    source_clear=np.asarray(source_query[2],dtype=np.float64)
    median_clear=float(np.median(np.abs(source_clear)))
    early_labels=_component_labels(len(P),F)
    early_component_count=int(len(np.unique(early_labels)))

    rigid_candidate,rigid_report=_shape_preserving_rigid_clearance(
        P,target_body_triangles,nearest_surface_fn=nearest_surface_fn,component_count=early_component_count,
        median_clearance=median_clear,source_weights=source_weights,cfg=cfg)
    if rigid_candidate is not None and bool(rigid_report.get("selected")):
        move=np.linalg.norm(rigid_candidate-B,axis=1)
        rigid_report.update({
            "source_clearance_median_mm":median_clear*1000.0,
            "body_motion_p95_mm":float(body_motion_p95_m)*1000.0,
            "component_count":early_component_count,
            "attachment_edge_count":0,
            "changed_vertex_count":int(np.count_nonzero(move>1e-12)),
            "authored_shape_preserved":True,
            "construction_mode":"shape-preserving coherent rigid placement",
        })
        return np.asarray(rigid_candidate,dtype=np.float64),rigid_report

    # A stand-off authored structure that already clears the target is already a valid fit.  Do not
    # resize, smooth or reinterpret it merely because the underlying body morphology changed.  A
    # larger target naturally falls through to the normal refit path as soon as this clearance test
    # fails.  This rule is control-free and applies equally to jackets, collars, cages, boots, armour
    # and any other sufficiently stand-off construction.
    no_op_motion_threshold=max(float(cfg.near_identity_motion_floor_m),float(cfg.authored_standoff_noop_min_motion_clearance_ratio)*median_clear)
    if bool(allow_authored_standoff_noop) and median_clear>=float(cfg.authored_standoff_noop_clearance_m) and float(body_motion_p95_m)>=no_op_motion_threshold:
        source_on_target=nearest_surface_fn(P,np.asarray(target_body_triangles,dtype=np.float64),k=24)
        target_signed=np.asarray(source_on_target[2],dtype=np.float64)
        target_p01=float(np.percentile(target_signed,1)) if len(target_signed) else float('inf')
        if target_p01>=float(cfg.authored_standoff_noop_target_p01_m):
            move=np.linalg.norm(P-B,axis=1)
            labels=early_labels
            return P.copy(), {
                "selected":bool(np.any(move>1e-12)),
                "reason":"stand-off authored construction already clears target; exact source preserved",
                "source_clearance_median_mm":median_clear*1000.0,
                "target_source_clearance_p01_mm":target_p01*1000.0,
                "body_motion_p95_mm":float(body_motion_p95_m)*1000.0,
                "no_op_motion_threshold_mm":no_op_motion_threshold*1000.0,
                "component_count":int(len(np.unique(labels))),
                "attachment_edge_count":0,
                "changed_vertex_count":int(np.count_nonzero(move>1e-12)),
                "authored_source_restored":True,
                "already_fits_target":True,
            }

    follow=_follow_strength(source_clear,median_clear,float(body_motion_p95_m),cfg)

    adjacency, labels, attachments=_graph(P,F,cfg)
    component_count=int(len(np.unique(labels)))

    low_motion_follow=bool(float(np.max(follow,initial=0.0))<=1e-6)

    # Construction stays in source geometry; only the body-driven displacement field is diffused.
    raw_field=C-P
    if median_clear>=float(cfg.structured_clearance_m) and macro_transform is not None:
        # Structured outer clothing follows one shared body *frame*, not the literal body surface.
        # This preserves authored panels/collars/trim while still resizing the complete assembly for
        # the selected target.  Stand-off is retained more strongly than body compression itself.
        linear,translation,rotation,macro_report=macro_transform
        candidate=_structured_macro_transport(P,source_support_points,linear,translation,rotation,median_clear,cfg)
        coherent_follow=np.ones_like(follow)
        construction_mode='shared body-frame structured transport'
    elif median_clear>=float(cfg.structured_clearance_m):
        # Compatibility fallback for callers that do not yet provide the slot-wide macro transform.
        macro_field=_diffuse(raw_field,adjacency,cfg.structured_field_iterations,cfg.structured_field_alpha)
        coherent_follow=_diffuse(follow[:,None],adjacency,cfg.structured_follow_iterations,cfg.structured_follow_alpha)[:,0]
        for component in np.unique(labels):
            ids=np.flatnonzero(labels==component)
            if len(ids)<int(cfg.structured_component_min_vertices):continue
            scalar=float(np.median(coherent_follow[ids]));blend=float(cfg.structured_component_follow_blend)
            coherent_follow[ids]=(1.0-blend)*coherent_follow[ids]+blend*scalar
        candidate=P+coherent_follow[:,None]*macro_field
        construction_mode='source-authored structured transport fallback'
        macro_report={}
    else:
        macro_field=_diffuse(raw_field,adjacency,cfg.field_iterations,cfg.field_alpha)
        body_authored=P+macro_field
        candidate=B+follow[:,None]*(body_authored-B)
        coherent_follow=follow
        construction_mode='tight body-following transport'
        macro_report={}

    # Literal-body contact is a constraint, not a sculpting surface.  Distribute the required
    # correction through the authored assembly graph before applying it.
    target_query=nearest_surface_fn(candidate,np.asarray(target_body_triangles,dtype=np.float64),k=24)
    normals=np.asarray(target_query[1],dtype=np.float64)
    signed=np.asarray(target_query[2],dtype=np.float64)
    required=np.maximum(0.0,float(cfg.collision_margin_m)-signed)
    contact_seed=normals*required[:,None]
    distributed=_diffuse(contact_seed,adjacency,cfg.collision_diffusion_iterations,cfg.collision_alpha)
    seed_mag=np.linalg.norm(contact_seed,axis=1)
    dist_mag=np.linalg.norm(distributed,axis=1)
    # Keep original contact vertices at least as safe as their distributed field predicts, but cap
    # the whole correction.  Neighbours receive only the smooth distributed portion.
    correction=distributed
    need=seed_mag>dist_mag
    correction[need]=contact_seed[need]
    cmag=np.linalg.norm(correction,axis=1)
    correction*=np.minimum(1.0,float(cfg.collision_max_move_m)/np.maximum(cmag,_EPS))[:,None]
    candidate+=correction

    # One tiny direct residual guard is permitted only after distributed correction.  This is too
    # small to define garment shape; it merely prevents numerical overlap.
    residual=nearest_surface_fn(candidate,np.asarray(target_body_triangles,dtype=np.float64),k=24)
    rnorm=np.asarray(residual[1],dtype=np.float64); rsigned=np.asarray(residual[2],dtype=np.float64)
    rneed=np.maximum(0.0,float(cfg.collision_margin_m)-rsigned)
    direct=np.minimum(rneed,float(cfg.collision_final_direct_cap_m))
    candidate+=rnorm*direct[:,None]

    # Preserve source/B14 topology.  If needed, monotonically blend back toward B14; never invent
    # a local repair that could alter construction identity.
    accepted=candidate.copy(); blend=1.0; topology_ok,flips,min_ratio=_topology_ok(B,accepted,F,cfg.min_triangle_area_ratio)
    steps=0
    while not topology_ok and steps<int(cfg.max_blend_back_steps):
        blend*=0.75
        accepted=B+blend*(candidate-B)
        topology_ok,flips,min_ratio=_topology_ok(B,accepted,F,cfg.min_triangle_area_ratio)
        steps+=1
    if not topology_ok:
        accepted=B.copy(); blend=0.0

    move=np.linalg.norm(accepted-B,axis=1)
    attachment_delta=[]
    for a,b,d0 in attachments:
        d1=float(np.linalg.norm(accepted[a]-accepted[b]))
        attachment_delta.append(abs(d1-float(d0)))

    final_query=nearest_surface_fn(accepted,np.asarray(target_body_triangles,dtype=np.float64),k=24)
    final_signed=np.asarray(final_query[2],dtype=np.float64)
    return accepted, {
        "selected":bool(np.any(move>1e-12)),
        "policy":"source construction + continuous low-frequency body field + source-clearance authority + distributed literal collision",
        "source_clearance_median_mm":median_clear*1000.0,
        "source_clearance_p95_mm":float(np.percentile(np.abs(source_clear),95))*1000.0,
        "body_motion_p95_mm":float(body_motion_p95_m)*1000.0,
        "follow_strength_p50":float(np.percentile(follow,50)),
        "follow_strength_p95":float(np.percentile(follow,95)),
        "effective_follow_strength_p50":float(np.percentile(coherent_follow,50)),
        "effective_follow_strength_p95":float(np.percentile(coherent_follow,95)),
        "construction_mode":construction_mode,
        "body_macro_transform":macro_report,
        "component_count":component_count,
        "attachment_edge_count":len(attachments),
        "attachment_length_error_p95_mm":float(np.percentile(attachment_delta,95)*1000.0) if attachment_delta else 0.0,
        "candidate_penetrations_before_distributed_contact":int(np.count_nonzero(signed<0.0)),
        "final_penetrating_vertices":int(np.count_nonzero(final_signed<-1e-9)),
        "changed_vertex_count":int(np.count_nonzero(move>1e-12)),
        "displacement_p50_mm":float(np.percentile(move,50))*1000.0,
        "displacement_p95_mm":float(np.percentile(move,95))*1000.0,
        "displacement_max_mm":float(np.max(move,initial=0.0))*1000.0,
        "topology_blend":float(blend),
        "triangle_flips":int(flips),
        "minimum_triangle_area_ratio":float(min_ratio),
    }
