from __future__ import annotations

"""Generic authored-structure constraints for RavaFit's LoBoMap production lane.

This module is deliberately independent of the frozen strict-B14 implementation.  It
starts from untouched garment topology plus the coherent LoBoMap initialisation and
adds structure *inside* the local residual solve:

- connectivity is authoritative for authored components;
- components are grouped into stable garment layers from material continuity, source
  body support/clearance, surface-region overlap and bilateral peer evidence;
- rigid pieces are projected by SE(3) only (no scale/shear);
- semi-rigid/flexible pieces keep the differential structure of the coherent target
  initialisation through local edge constraints;
- compatible close parallel components can retain an inferred thin-wall separation;
- overlapping layers retain their source body-relative ordering;
- final body contact uses exact closest points on target *triangles*, not body points.

No garment names, body names, mod names or hand-fitted/control geometry are consumed.
"""

from dataclasses import dataclass
from hashlib import sha1
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point as triangle_closest_point

from lobomap_production import LoBoMapGarment, LoBoMapMeshInput
from lobomap_refinement import LocalFitContext, build_local_fit_context

_EPS=1e-12


def _unit(v: np.ndarray, fallback: Sequence[float]=(0.0,0.0,1.0)) -> np.ndarray:
    a=np.asarray(v,dtype=np.float64);n=float(np.linalg.norm(a))
    if n>_EPS:return a/n
    f=np.asarray(fallback,dtype=np.float64);return f/max(float(np.linalg.norm(f)),_EPS)


def _stable_id(prefix: str, values: Iterable[object]) -> str:
    payload="|".join(str(x) for x in values).encode("utf-8",errors="replace")
    return f"{prefix}-{sha1(payload).hexdigest()[:16]}"


def _faces_for_ids(faces: np.ndarray, ids: np.ndarray, vertex_count: int) -> tuple[np.ndarray,np.ndarray]:
    mask=np.zeros(int(vertex_count),dtype=bool);mask[np.asarray(ids,dtype=np.int64)]=True
    selected=np.flatnonzero(np.all(mask[np.asarray(faces,dtype=np.int64)],axis=1))
    if not len(selected):return selected,np.empty((0,3),dtype=np.int64)
    lookup=np.full(int(vertex_count),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64)
    return selected,lookup[np.asarray(faces,dtype=np.int64)[selected]]


def _component_labels(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    F=np.asarray(faces,dtype=np.int64);n=int(vertex_count)
    if not len(F):return np.arange(n,dtype=np.int64)
    edges=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]))
    rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
    graph=coo_matrix((np.ones(len(rows),dtype=np.uint8),(rows,cols)),shape=(n,n)).tocsr()
    _,labels=connected_components(graph,directed=False,return_labels=True)
    return labels.astype(np.int64)


def _boundary_fraction(vertex_count: int, local_faces: np.ndarray) -> float:
    F=np.asarray(local_faces,dtype=np.int64);n=int(vertex_count)
    if n<=0:return 0.0
    if not len(F):return 1.0
    edges=np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True);bedges=unique[counts==1]
    return float(len(np.unique(bedges))/n) if len(bedges) else 0.0


def _principal_geometry(points: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    P=np.asarray(points,dtype=np.float64)
    if len(P)<3:return np.asarray([0.0,0.0,0.0]),np.eye(3,dtype=np.float64)
    Q=P-np.mean(P,axis=0);_,s,vt=np.linalg.svd(Q,full_matrices=False)
    if len(s)<3:s=np.pad(s,(0,3-len(s)))
    scale=max(float(s[0]),_EPS);return np.asarray([float(s[0]),float(s[1])/scale,float(s[2])/scale]),vt.T


def _classify_component(vertex_count: int, extent_max: float, middle_ratio: float, thin_ratio: float, boundary_fraction: float, clearance_std: float) -> str:
    """Geometry-only structural class; thresholds are physical-scale, not garment-type."""
    n=int(vertex_count);e=float(extent_max);thin=float(thin_ratio);middle=float(middle_ratio);boundary=float(boundary_fraction);spread=float(clearance_std)
    # Tiny dense hardware/details should not be stretched simply because they carry skin weights.
    if n<=24 or (e<=0.018 and n<=640 and spread<=0.006):return "rigid"
    # Narrow ribbons, straps, bands and small constructed shells benefit from a strong
    # coherent frame but still need some target-body adaptation.
    if (thin<=0.10 and e<=0.20) or (boundary>=0.18 and e<=0.16) or (middle<=0.20 and e<=0.12):return "semi_rigid"
    return "flexible"


@dataclass(frozen=True)
class GarmentComponent:
    stable_id: str
    mesh_name: str
    component_index: int
    material: str | None
    vertex_ids: np.ndarray
    face_ids: np.ndarray
    local_faces: np.ndarray
    vertex_count: int
    triangle_count: int
    centroid: np.ndarray
    extent: np.ndarray
    support_centroid: np.ndarray
    support_min: np.ndarray
    support_max: np.ndarray
    clearance_median_m: float
    clearance_p95_m: float
    clearance_std_m: float
    boundary_fraction: float
    middle_ratio: float
    thin_ratio: float
    principal_normal: np.ndarray
    structural_class: str

    def __post_init__(self):
        object.__setattr__(self,"vertex_ids",np.asarray(self.vertex_ids,dtype=np.int64))
        object.__setattr__(self,"face_ids",np.asarray(self.face_ids,dtype=np.int64))
        object.__setattr__(self,"local_faces",np.asarray(self.local_faces,dtype=np.int64))
        for name in ("centroid","extent","support_centroid","support_min","support_max","principal_normal"):
            object.__setattr__(self,name,np.asarray(getattr(self,name),dtype=np.float64))


@dataclass(frozen=True)
class LayerOrderRelation:
    other_layer_id: str
    relation: str
    source_gap_m: float
    support_overlap: float


@dataclass(frozen=True)
class ThinWallRelation:
    component_a: str
    component_b: str
    source_centroid_delta: np.ndarray
    initial_distance_hint_m: float

    def __post_init__(self):object.__setattr__(self,"source_centroid_delta",np.asarray(self.source_centroid_delta,dtype=np.float64))


@dataclass(frozen=True)
class GarmentLayer:
    stable_id: str
    material: str | None
    component_ids: tuple[str,...]
    mesh_names: tuple[str,...]
    structural_class: str
    source_clearance_median_m: float
    source_clearance_p95_m: float
    support_min: np.ndarray
    support_max: np.ndarray
    ordering: tuple[LayerOrderRelation,...]=()
    thin_walls: tuple[ThinWallRelation,...]=()

    def __post_init__(self):
        object.__setattr__(self,"support_min",np.asarray(self.support_min,dtype=np.float64));object.__setattr__(self,"support_max",np.asarray(self.support_max,dtype=np.float64))


@dataclass(frozen=True)
class GarmentStructure:
    components: tuple[GarmentComponent,...]
    layers: tuple[GarmentLayer,...]
    symmetry_origin: np.ndarray
    symmetry_normal: np.ndarray

    def __post_init__(self):
        object.__setattr__(self,"symmetry_origin",np.asarray(self.symmetry_origin,dtype=np.float64));object.__setattr__(self,"symmetry_normal",_unit(self.symmetry_normal))

    @property
    def component_by_id(self) -> dict[str,GarmentComponent]:return {row.stable_id:row for row in self.components}

    @property
    def layer_by_id(self) -> dict[str,GarmentLayer]:return {row.stable_id:row for row in self.layers}


@dataclass(frozen=True)
class StructuralSolveConfig:
    semi_rigid_projection: float = 0.38
    semi_rigid_edge_stiffness: float = 0.58
    flexible_edge_stiffness: float = 0.50
    edge_iterations: int = 5
    ordering_strength: float = 0.75
    ordering_min_fraction: float = 0.25
    ordering_max_shift_m: float = 0.003
    thin_wall_strength: float = 0.70
    collision_margin_m: float = 0.00035
    collision_vertex_neighbors: int = 5
    collision_max_push_m: float = 0.006
    collision_iterations: int = 4
    rigid_clearance_translation_strength: float = 0.85
    rigid_clearance_translation_max_m: float = 0.006
    topology_area_floor: float = 0.12

    def __post_init__(self):
        for name in ("semi_rigid_projection","semi_rigid_edge_stiffness","flexible_edge_stiffness","ordering_strength","ordering_min_fraction","thin_wall_strength","rigid_clearance_translation_strength"):
            v=float(getattr(self,name))
            if not 0.0<=v<=1.0:raise ValueError(f"{name} must be in [0,1]")
        if self.edge_iterations<0 or self.collision_iterations<1:raise ValueError("structural/collision iteration counts are invalid")
        if self.ordering_max_shift_m<0 or self.collision_margin_m<0 or self.collision_max_push_m<=0 or self.rigid_clearance_translation_max_m<0:raise ValueError("structural distances are invalid")
        if not 0.0<=self.topology_area_floor<=1.0:raise ValueError("topology_area_floor must be in [0,1]")
        if self.collision_vertex_neighbors<1:raise ValueError("collision_vertex_neighbors must be positive")


def _infer_symmetry_plane(body_vertices: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Choose the PCA mirror plane whose reflection best matches the untouched body."""
    X=np.asarray(body_vertices,dtype=np.float64);origin=np.median(X,axis=0)
    Q=X-origin;cov=(Q.T@Q)/max(len(Q),1);_,vec=np.linalg.eigh(cov)
    # Work on a deterministic sample so this remains cheap for large RBODY libraries.
    step=max(1,len(X)//2000);sample=X[::step];tree=cKDTree(X)
    best=None
    for axis in vec.T:
        n=_unit(axis);d=(sample-origin)@n;reflected=sample-2.0*d[:,None]*n
        dist,_=tree.query(reflected,k=1,workers=-1);score=float(np.percentile(dist,80))
        if best is None or score<best[0]:best=(score,n)
    assert best is not None
    n=best[1]
    # Stable sign: first materially non-zero coordinate is positive.
    for value in n:
        if abs(float(value))>1e-8:
            if value<0:n=-n
            break
    return origin,n


def _aabb_overlap(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, margin: float=0.0) -> float:
    lo=np.maximum(np.asarray(a0)-margin,np.asarray(b0)-margin);hi=np.minimum(np.asarray(a1)+margin,np.asarray(b1)+margin)
    inter=np.maximum(hi-lo,0.0);iv=float(np.prod(inter))
    ea=np.maximum(np.asarray(a1)-np.asarray(a0),1e-5);eb=np.maximum(np.asarray(b1)-np.asarray(b0),1e-5)
    denom=max(min(float(np.prod(ea)),float(np.prod(eb))),1e-12)
    return float(np.clip(iv/denom,0.0,1.0))


def _mirrored_peer(a: GarmentComponent, b: GarmentComponent, origin: np.ndarray, normal: np.ndarray) -> bool:
    ca=a.support_centroid;ref=ca-2.0*float(np.dot(ca-origin,normal))*normal
    span=max(float(np.max(a.extent)),float(np.max(b.extent)),0.005)
    centre_error=float(np.linalg.norm(ref-b.support_centroid))
    size_ratio=max(float(a.vertex_count),float(b.vertex_count))/max(min(float(a.vertex_count),float(b.vertex_count)),1.0)
    extent_ratio=max(float(np.max(a.extent)),float(np.max(b.extent)),1e-6)/max(min(float(np.max(a.extent)),float(np.max(b.extent))),1e-6)
    return centre_error<=max(0.008,0.22*span) and size_ratio<=2.6 and extent_ratio<=2.2


def infer_garment_structure(meshes: Mapping[str,LoBoMapMeshInput], mappings: Mapping[str,LoBoMapGarment], body_cache: Mapping[str,object], *, contexts: Mapping[str,LocalFitContext] | None = None) -> tuple[GarmentStructure,dict[str,LocalFitContext],dict[str,object]]:
    """Infer connected authored components, layers and source ordering generically."""
    X=np.asarray(body_cache["X"],dtype=np.float64);origin,normal=_infer_symmetry_plane(X)
    ctxs=dict(contexts or {});components:list[GarmentComponent]=[]
    for mesh_name in sorted(meshes):
        mesh=meshes[mesh_name]
        if mesh_name not in mappings:raise ValueError(f"{mesh_name}: missing LoBoMap mapping for structure inference")
        ctx=ctxs.get(mesh_name)
        if ctx is None:ctx=build_local_fit_context(mesh,mappings[mesh_name],body_cache);ctxs[mesh_name]=ctx
        labels=_component_labels(mesh.vertex_count,mesh.faces)
        for ci in sorted(np.unique(labels).tolist()):
            ids=np.flatnonzero(labels==ci);face_ids,local_faces=_faces_for_ids(mesh.faces,ids,mesh.vertex_count);P=mesh.vertices[ids];A=ctx.source_anchor[ids];C=ctx.source_clearance[ids]
            centroid=np.mean(P,axis=0);extent=np.ptp(P,axis=0);principal,axes=_principal_geometry(P);middle=float(principal[1]);thin=float(principal[2])
            structural=_classify_component(len(ids),float(np.max(extent)),middle,thin,_boundary_fraction(len(ids),local_faces),float(np.std(C)))
            raw=mesh.raw_vertex_ids[ids]
            stable=_stable_id("component",(mesh.material or "",mesh_name,ci,int(raw.min()) if len(raw) else -1,int(raw.max()) if len(raw) else -1,len(raw),sha1(raw.tobytes()).hexdigest()[:12]))
            components.append(GarmentComponent(
                stable,mesh_name,int(ci),mesh.material,ids,face_ids,local_faces,int(len(ids)),int(len(local_faces)),centroid,extent,
                np.mean(A,axis=0),np.min(A,axis=0),np.max(A,axis=0),float(np.median(C)),float(np.percentile(C,95)),float(np.std(C)),
                _boundary_fraction(len(ids),local_faces),middle,thin,_unit(axes[:,2]),structural))

    count=len(components);rows=[];cols=[]
    # Layer connectivity used to spend most of this stage in ~O(n^2) Python calls.
    # Evaluate the exact same predicates in vectorised material groups; ordering and
    # stable IDs remain unchanged because the original component indices are retained.
    by_material:dict[str|None,list[int]]={}
    for idx,comp in enumerate(components):by_material.setdefault(comp.material,[]).append(idx)
    for indices in by_material.values():
        if len(indices)<2:continue
        group=np.asarray(indices,dtype=np.int64);gi,gj=np.triu_indices(len(group),k=1);ii=group[gi];jj=group[gj]
        med=np.asarray([x.clearance_median_m for x in components],dtype=np.float64);std=np.asarray([x.clearance_std_m for x in components],dtype=np.float64)
        keep=np.abs(med[ii]-med[jj])<=np.maximum(0.0035,1.25*(std[ii]+std[jj])+0.0015)
        if not np.any(keep):continue
        ii=ii[keep];jj=jj[keep]
        smin=np.vstack([x.support_min for x in components]);smax=np.vstack([x.support_max for x in components]);cent=np.vstack([x.support_centroid for x in components])
        extmax=np.asarray([float(np.max(x.extent)) for x in components],dtype=np.float64);vcount=np.asarray([float(x.vertex_count) for x in components],dtype=np.float64);meshname=np.asarray([x.mesh_name for x in components],dtype=object)
        lo=np.maximum(smin[ii]-0.006,smin[jj]-0.006);hi=np.minimum(smax[ii]+0.006,smax[jj]+0.006);inter=np.maximum(hi-lo,0.0);iv=np.prod(inter,axis=1)
        ea=np.maximum(smax[ii]-smin[ii],1e-5);eb=np.maximum(smax[jj]-smin[jj],1e-5);denom=np.maximum(np.minimum(np.prod(ea,axis=1),np.prod(eb,axis=1)),1e-12);overlap=np.clip(iv/denom,0.0,1.0)
        ca=cent[ii];ref=ca-2.0*np.einsum("nc,c->n",ca-origin,normal)[:,None]*normal[None,:];mirror_span=np.maximum(np.maximum(extmax[ii],extmax[jj]),0.005)
        centre_error=np.linalg.norm(ref-cent[jj],axis=1);size_ratio=np.maximum(vcount[ii],vcount[jj])/np.maximum(np.minimum(vcount[ii],vcount[jj]),1.0);extent_ratio=np.maximum(np.maximum(extmax[ii],extmax[jj]),1e-6)/np.maximum(np.minimum(extmax[ii],extmax[jj]),1e-6)
        mirror=(centre_error<=np.maximum(0.008,0.22*mirror_span))&(size_ratio<=2.6)&(extent_ratio<=2.2)
        distance=np.linalg.norm(cent[ii]-cent[jj],axis=1);local_span=np.maximum(np.maximum(extmax[ii],extmax[jj]),0.01);local_peer=(meshname[ii]==meshname[jj])&(distance<=np.maximum(0.018,0.35*local_span))
        linked=(overlap>=0.06)|mirror|local_peer
        if np.any(linked):
            a=ii[linked].tolist();b=jj[linked].tolist();rows.extend([x for pair in zip(a,b) for x in pair]);cols.extend([x for pair in zip(b,a) for x in pair])
    graph=coo_matrix((np.ones(len(rows),dtype=np.uint8),(rows,cols)),shape=(count,count)).tocsr() if count else coo_matrix((0,0)).tocsr()
    _,layer_labels=connected_components(graph,directed=False,return_labels=True) if count else (0,np.empty(0,dtype=np.int64))

    provisional=[]
    for li in sorted(np.unique(layer_labels).tolist()):
        members=[components[i] for i in np.flatnonzero(layer_labels==li)]
        material=members[0].material if members else None;component_ids=tuple(sorted(x.stable_id for x in members));mesh_names=tuple(sorted(set(x.mesh_name for x in members)))
        weight=np.asarray([x.vertex_count for x in members],dtype=np.float64);weight/=max(float(weight.sum()),1.0)
        clearance=float(np.sum(weight*np.asarray([x.clearance_median_m for x in members])))
        p95=float(np.max([x.clearance_p95_m for x in members])) if members else 0.0
        class_weight={k:sum(x.vertex_count for x in members if x.structural_class==k) for k in ("rigid","semi_rigid","flexible")};structural=max(class_weight,key=lambda k:(class_weight[k],{"flexible":2,"semi_rigid":1,"rigid":0}[k]))
        stable=_stable_id("layer",(material or "",)+component_ids)
        provisional.append((stable,material,members,component_ids,mesh_names,structural,clearance,p95,np.min(np.vstack([x.support_min for x in members]),axis=0),np.max(np.vstack([x.support_max for x in members]),axis=0)))

    # Source body-relative ordering is recorded only where support regions materially overlap.
    ordering:dict[str,list[LayerOrderRelation]]={row[0]:[] for row in provisional}
    for i,a in enumerate(provisional):
        for b in provisional[i+1:]:
            overlap=_aabb_overlap(a[8],a[9],b[8],b[9],margin=0.004)
            if overlap<0.08:continue
            gap=float(b[6]-a[6])
            if abs(gap)<0.0006:relation_ab=relation_ba="peer"
            elif gap>0:relation_ab,relation_ba="inside","outside"
            else:relation_ab,relation_ba="outside","inside"
            ordering[a[0]].append(LayerOrderRelation(b[0],relation_ab,abs(gap),overlap));ordering[b[0]].append(LayerOrderRelation(a[0],relation_ba,abs(gap),overlap))

    # Thin-wall relation: close, parallel, overlapping components in the same inferred layer.
    thin_by_layer:dict[str,list[ThinWallRelation]]={row[0]:[] for row in provisional}
    for row in provisional:
        members=row[2]
        for i,a in enumerate(members):
            for b in members[i+1:]:
                if a.structural_class=="rigid" or b.structural_class=="rigid":continue
                overlap=_aabb_overlap(a.support_min,a.support_max,b.support_min,b.support_max,margin=0.003)
                if overlap<0.12:continue
                d=b.centroid-a.centroid;distance=float(np.linalg.norm(d));span=max(min(float(np.max(a.extent)),float(np.max(b.extent))),0.005)
                parallel=abs(float(np.dot(a.principal_normal,b.principal_normal)))
                aligned=max(abs(float(np.dot(_unit(d),a.principal_normal))),abs(float(np.dot(_unit(d),b.principal_normal)))) if distance>1e-8 else 0.0
                if 0.0002<distance<=min(0.012,0.20*span) and parallel>=0.75 and aligned>=0.45:
                    thin_by_layer[row[0]].append(ThinWallRelation(a.stable_id,b.stable_id,d,distance))

    layers=tuple(GarmentLayer(row[0],row[1],row[3],row[4],row[5],row[6],row[7],row[8],row[9],tuple(sorted(ordering[row[0]],key=lambda x:x.other_layer_id)),tuple(thin_by_layer[row[0]])) for row in provisional)
    structure=GarmentStructure(tuple(components),layers,origin,normal)
    report={
        "revision":"lobomap-authored-structure-v1",
        "component_count":int(len(components)),"layer_count":int(len(layers)),
        "rigid_component_count":int(sum(x.structural_class=="rigid" for x in components)),
        "semi_rigid_component_count":int(sum(x.structural_class=="semi_rigid" for x in components)),
        "flexible_component_count":int(sum(x.structural_class=="flexible" for x in components)),
        "thin_wall_relation_count":int(sum(len(x.thin_walls) for x in layers)),
        "ordering_relation_count":int(sum(len(x.ordering) for x in layers)//2),
        "symmetry_origin":origin.tolist(),"symmetry_normal":normal.tolist(),
        "layers":[{"id":x.stable_id,"material":x.material,"components":len(x.component_ids),"meshes":list(x.mesh_names),"class":x.structural_class,"clearance_median_mm":x.source_clearance_median_m*1000.0,"ordering":len(x.ordering),"thin_walls":len(x.thin_walls)} for x in layers],
    }
    return structure,ctxs,report


def _rigid_fit(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    P=np.asarray(reference,dtype=np.float64);Q=np.asarray(target,dtype=np.float64)
    if P.shape!=Q.shape or P.ndim!=2 or P.shape[1]!=3:raise ValueError("rigid fit point shape mismatch")
    if len(P)<2:return np.eye(3),np.mean(Q-P,axis=0) if len(P) else np.zeros(3)
    cp=np.mean(P,axis=0);cq=np.mean(Q,axis=0);H=(P-cp).T@(Q-cq);u,_,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0.0:vt[-1]*=-1.0;R=vt.T@u.T
    return R,cq-cp@R.T


def _similarity_reference(source: np.ndarray, target: np.ndarray, scale_min: float=0.70, scale_max: float=1.40) -> np.ndarray:
    """Preserve authored shape up to one uniform target-scale before SE(3) solving."""
    P=np.asarray(source,dtype=np.float64);Q=np.asarray(target,dtype=np.float64)
    if P.shape!=Q.shape or not len(P):return P.copy()
    cp=np.mean(P,axis=0);cq=np.mean(Q,axis=0);A=P-cp;B=Q-cq;H=A.T@B;u,_,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0.0:vt[-1]*=-1.0;R=vt.T@u.T
    rotated=A@R.T;den=float(np.sum(rotated*rotated));scale=float(np.sum(rotated*B)/max(den,_EPS));scale=float(np.clip(scale,scale_min,scale_max))
    return cq+scale*rotated


def _unique_edges(local_faces: np.ndarray) -> np.ndarray:
    F=np.asarray(local_faces,dtype=np.int64)
    if not len(F):return np.empty((0,2),dtype=np.int64)
    return np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)


def _edge_project(reference: np.ndarray, candidate: np.ndarray, faces: np.ndarray, stiffness: float, iterations: int) -> np.ndarray:
    if iterations<=0 or stiffness<=0.0:return np.asarray(candidate,dtype=np.float64).copy()
    R=np.asarray(reference,dtype=np.float64);out=np.asarray(candidate,dtype=np.float64).copy();edges=_unique_edges(faces)
    if not len(edges):return out
    target=np.linalg.norm(R[edges[:,0]]-R[edges[:,1]],axis=1)
    valid=target>1e-8;edges=edges[valid];target=target[valid]
    if not len(edges):return out
    for _ in range(int(iterations)):
        vec=out[edges[:,1]]-out[edges[:,0]];length=np.linalg.norm(vec,axis=1);good=length>1e-10
        corr=np.zeros_like(vec);corr[good]=(0.5*float(stiffness)*(1.0-target[good]/length[good]))[:,None]*vec[good]
        accum=np.zeros_like(out);weight=np.zeros(len(out),dtype=np.float64)
        np.add.at(accum,edges[:,0],corr);np.add.at(accum,edges[:,1],-corr);np.add.at(weight,edges[:,0],1.0);np.add.at(weight,edges[:,1],1.0)
        active=weight>0;out[active]+=accum[active]/weight[active,None]
    return out


def _component_centroid(positions: Mapping[str,np.ndarray], component: GarmentComponent) -> np.ndarray:
    return np.mean(np.asarray(positions[component.mesh_name],dtype=np.float64)[component.vertex_ids],axis=0)


def _topology_safe_blend(reference: np.ndarray, candidate: np.ndarray, faces: np.ndarray, area_floor: float) -> tuple[np.ndarray,float,dict[str,float|int]]:
    """Locally roll back only topology-threatening neighbourhoods toward the coherent initialisation."""
    R=np.asarray(reference,dtype=np.float64);C=np.asarray(candidate,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if not len(F):return C.copy(),1.0,{"flips":0,"area_ratio_min":1.0,"repaired_vertices":0}
    rn=np.cross(R[F[:,1]]-R[F[:,0]],R[F[:,2]]-R[F[:,0]]);ra=np.linalg.norm(rn,axis=1);valid=ra>_EPS
    retain=np.ones(len(R),dtype=np.float64);out=C.copy();touched=np.zeros(len(R),dtype=bool)
    for _ in range(12):
        qn=np.cross(out[F[:,1]]-out[F[:,0]],out[F[:,2]]-out[F[:,0]]);qa=np.linalg.norm(qn,axis=1);dot=np.einsum("nc,nc->n",rn,qn)
        ratio=np.ones(len(F),dtype=np.float64);ratio[valid]=qa[valid]/ra[valid]
        bad=valid&((dot<0.0)|(ratio<float(area_floor)))
        if not np.any(bad):
            minimum=float(np.min(ratio[valid])) if np.any(valid) else 1.0
            return out,float(np.percentile(retain,50)),{"flips":0,"area_ratio_min":minimum,"repaired_vertices":int(np.count_nonzero(touched)),"retain_p05":float(np.percentile(retain,5))}
        core=np.unique(F[bad]);touched[core]=True
        # One-ring feathering keeps the repair local without creating a sharp seam.
        incident=np.any(np.isin(F,core),axis=1);ring=np.unique(F[incident]);ring=np.setdiff1d(ring,core,assume_unique=False)
        retain[core]*=.55;retain[ring]=np.minimum(retain[ring],.88)
        out=R+retain[:,None]*(C-R)
    # Deterministic final safety: expand the rollback only as far as necessary.  This
    # loop must terminate because the all-reference state is valid by definition.
    for _ in range(32):
        qn=np.cross(out[F[:,1]]-out[F[:,0]],out[F[:,2]]-out[F[:,0]]);qa=np.linalg.norm(qn,axis=1);dot=np.einsum("nc,nc->n",rn,qn);ratio=np.ones(len(F));ratio[valid]=qa[valid]/ra[valid]
        bad=valid&((dot<0.0)|(ratio<float(area_floor)))
        if not np.any(bad):break
        core=np.unique(F[bad]);touched[core]=True;retain[core]=0.0
        incident=np.any(np.isin(F,core),axis=1);ring=np.setdiff1d(np.unique(F[incident]),core,assume_unique=False);retain[ring]=np.minimum(retain[ring],.70)
        out=R+retain[:,None]*(C-R)
    qn=np.cross(out[F[:,1]]-out[F[:,0]],out[F[:,2]]-out[F[:,0]]);qa=np.linalg.norm(qn,axis=1);dot=np.einsum("nc,nc->n",rn,qn);ratio=np.ones(len(F));ratio[valid]=qa[valid]/ra[valid];flips=int(np.count_nonzero(valid&(dot<0.0)))
    return out,float(np.percentile(retain,50)),{"flips":flips,"area_ratio_min":float(np.min(ratio[valid])) if np.any(valid) else 1.0,"repaired_vertices":int(np.count_nonzero(touched)),"retain_p05":float(np.percentile(retain,5))}


def _source_topology_status(source: np.ndarray, candidate: np.ndarray, faces: np.ndarray, area_floor: float) -> dict[str,object]:
    S=np.asarray(source,dtype=np.float64);Q=np.asarray(candidate,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if not len(F):return {"valid_faces":0,"opposed_faces":0,"area_floor_violations":0,"minimum_signed_area_ratio":1.0}
    sn=np.cross(S[F[:,1]]-S[F[:,0]],S[F[:,2]]-S[F[:,0]]);qn=np.cross(Q[F[:,1]]-Q[F[:,0]],Q[F[:,2]]-Q[F[:,0]])
    sa=np.linalg.norm(sn,axis=1);valid=sa>_EPS;unit=np.zeros_like(sn);unit[valid]=sn[valid]/sa[valid,None]
    signed=np.ones(len(F),dtype=np.float64);signed[valid]=np.einsum("nc,nc->n",qn[valid],unit[valid])/sa[valid]
    opposed=valid&(signed<-1e-10);area_bad=valid&(signed<float(area_floor)-1e-10)
    return {"valid_faces":int(np.count_nonzero(valid)),"opposed_faces":int(np.count_nonzero(opposed)),"area_floor_violations":int(np.count_nonzero(area_bad)),"minimum_signed_area_ratio":float(np.min(signed[valid])) if np.any(valid) else 1.0}


def enforce_source_topology(mesh: LoBoMapMeshInput, candidate: np.ndarray, *, area_floor: float = 0.01, iterations: int = 128, core_strength: float = 0.55, ring_strength: float = 0.12) -> tuple[np.ndarray,dict[str,object]]:
    """Smooth only the deformation field around source-topology violations.

    The untouched authored garment remains the orientation/area authority, but the
    correction is applied to the *target displacement field*.  This is deliberately
    different from blending positions back toward the source body: macro target motion
    is retained while only locally inconsistent differential motion is regularised.
    """
    S=np.asarray(mesh.vertices,dtype=np.float64);Q=np.asarray(candidate,dtype=np.float64).copy();F=np.asarray(mesh.faces,dtype=np.int64)
    if S.shape!=Q.shape:raise ValueError(f"{mesh.name}: source-topology candidate shape mismatch")
    before=_source_topology_status(S,Q,F,area_floor)
    if not len(F) or int(before["area_floor_violations"])==0:
        return Q,{"revision":"lobomap-source-topology-v1","iterations":0,"before":before,"after":before,"moved_vertices":0,"move_p95_mm":0.0,"move_max_mm":0.0}
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0);rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
    adj=coo_matrix((np.ones(len(rows),dtype=np.float64),(rows,cols)),shape=(len(S),len(S))).tocsr();degree=np.asarray(adj.sum(axis=1)).reshape(-1);degree=np.maximum(degree,1.0);avg=adj.multiply((1.0/degree)[:,None]).tocsr()
    D=Q-S;touched=np.zeros(len(S),dtype=bool);steps=0
    for steps in range(1,int(iterations)+1):
        sn=np.cross(S[F[:,1]]-S[F[:,0]],S[F[:,2]]-S[F[:,0]]);qn=np.cross(Q[F[:,1]]-Q[F[:,0]],Q[F[:,2]]-Q[F[:,0]])
        sa=np.linalg.norm(sn,axis=1);valid=sa>_EPS;unit=np.zeros_like(sn);unit[valid]=sn[valid]/sa[valid,None]
        signed=np.ones(len(F),dtype=np.float64);signed[valid]=np.einsum("nc,nc->n",qn[valid],unit[valid])/sa[valid]
        bad=valid&(signed<float(area_floor))
        if not np.any(bad):break
        core=np.unique(F[bad]);touched[core]=True;neighbour=avg@D;updated=D.copy();updated[core]=(1.0-float(core_strength))*D[core]+float(core_strength)*neighbour[core]
        # Feather one topological ring so the local correction cannot create a hard seam.
        core_mask=np.zeros(len(S),dtype=bool);core_mask[core]=True;ring=np.flatnonzero((np.asarray(adj@core_mask.astype(np.float64)).reshape(-1)>0.0)&~core_mask)
        if len(ring):updated[ring]=(1.0-float(ring_strength))*D[ring]+float(ring_strength)*neighbour[ring]
        D=updated;Q=S+D
    # A tiny position-based fallback handles the last numerically stubborn face without
    # expanding the smoothed neighbourhood.  It changes target positions only; source
    # geometry remains immutable and no source-body position is used as a target.
    pbd_steps=0
    for pbd_steps in range(1,97):
        sn=np.cross(S[F[:,1]]-S[F[:,0]],S[F[:,2]]-S[F[:,0]]);sa=np.linalg.norm(sn,axis=1);valid=sa>_EPS;unit=np.zeros_like(sn);unit[valid]=sn[valid]/sa[valid,None]
        qn=np.cross(Q[F[:,1]]-Q[F[:,0]],Q[F[:,2]]-Q[F[:,0]]);ratio=np.ones(len(F),dtype=np.float64);ratio[valid]=np.einsum("nc,nc->n",qn[valid],unit[valid])/sa[valid]
        bad=np.flatnonzero(valid&(ratio<float(area_floor)-1e-10))
        if not len(bad):pbd_steps-=1;break
        fi=int(bad[np.argmin(ratio[bad])]);a,b,c=map(int,F[fi]);p0,p1,p2=Q[a].copy(),Q[b].copy(),Q[c].copy();n=unit[fi]
        current=float(np.dot(np.cross(p1-p0,p2-p0),n));target=float(area_floor)*float(sa[fi])
        g0=np.cross(p1-p2,n);g1=np.cross(p2-p0,n);g2=np.cross(p0-p1,n);den=float(np.dot(g0,g0)+np.dot(g1,g1)+np.dot(g2,g2)+1e-30)
        lam=0.9*(target-current)/den;Q[a]+=lam*g0;Q[b]+=lam*g1;Q[c]+=lam*g2;touched[[a,b,c]]=True
    after=_source_topology_status(S,Q,F,area_floor);move=np.linalg.norm(Q-np.asarray(candidate,dtype=np.float64),axis=1)*1000.0
    return Q,{"revision":"lobomap-source-topology-v2-field+pbd","iterations":int(steps),"pbd_iterations":int(pbd_steps),"before":before,"after":after,"moved_vertices":int(np.count_nonzero(move>1e-9)),"move_p95_mm":float(np.percentile(move,95)) if len(move) else 0.0,"move_max_mm":float(np.max(move)) if len(move) else 0.0}


def enforce_source_topology_batch(meshes: Mapping[str,LoBoMapMeshInput], positions: Mapping[str,np.ndarray], *, area_floor: float = 0.01) -> tuple[dict[str,np.ndarray],dict[str,object]]:
    out={};reports={};before=after=0
    for name,mesh in meshes.items():
        solved,report=enforce_source_topology(mesh,positions[name],area_floor=area_floor);out[name]=solved;reports[name]=report
        before+=int(report["before"]["area_floor_violations"]);after+=int(report["after"]["area_floor_violations"])
    return out,{"revision":"lobomap-source-topology-batch-v1","area_floor":float(area_floor),"violations_before":int(before),"violations_after":int(after),"meshes":reports}


def apply_authored_structure(meshes: Mapping[str,LoBoMapMeshInput], initial_positions: Mapping[str,np.ndarray], candidate_positions: Mapping[str,np.ndarray], structure: GarmentStructure, contexts: Mapping[str,LocalFitContext], *, config: StructuralSolveConfig | None = None) -> tuple[dict[str,np.ndarray],dict[str,object]]:
    """Project a body-fit candidate onto generic authored structure constraints."""
    cfg=config or StructuralSolveConfig();out={name:np.asarray(value,dtype=np.float64).copy() for name,value in candidate_positions.items()};components=structure.component_by_id
    reports=[]
    for comp in structure.components:
        mesh=meshes[comp.mesh_name];ids=comp.vertex_ids;source=mesh.vertices[ids];initial=np.asarray(initial_positions[comp.mesh_name],dtype=np.float64)[ids];candidate=out[comp.mesh_name][ids]
        before=float(np.percentile(np.linalg.norm(candidate-initial,axis=1),95)*1000.0) if len(ids) else 0.0
        if comp.structural_class=="rigid":
            authored=_similarity_reference(source,initial)
            R,t=_rigid_fit(authored,candidate);solved=authored@R.T+t
            # Preserve SE(3) exactly while recovering the component's authored body
            # clearance as far as one coherent translation can explain it.
            ctx=contexts[comp.mesh_name];N=ctx.target_normal[ids];current=np.einsum("nc,nc->n",solved-ctx.target_anchor[ids],N);error=ctx.source_clearance[ids]-current
            if len(ids)>=3 and float(cfg.rigid_clearance_translation_strength)>0.0:
                lhs=N.T@N+np.eye(3,dtype=np.float64)*1e-8;rhs=N.T@error
                shift=float(cfg.rigid_clearance_translation_strength)*np.linalg.solve(lhs,rhs);length=float(np.linalg.norm(shift))
                if length>float(cfg.rigid_clearance_translation_max_m)>0.0:shift*=float(cfg.rigid_clearance_translation_max_m)/length
                solved=solved+shift
            mode="uniform-source-reference+se3+clearance-translation"
        elif comp.structural_class=="semi_rigid":
            R,t=_rigid_fit(initial,candidate);rigid=initial@R.T+t
            solved=(1.0-float(cfg.semi_rigid_projection))*candidate+float(cfg.semi_rigid_projection)*rigid
            solved=_edge_project(initial,solved,comp.local_faces,cfg.semi_rigid_edge_stiffness,cfg.edge_iterations);mode="semi-rigid+edges"
        else:
            solved=_edge_project(initial,candidate,comp.local_faces,cfg.flexible_edge_stiffness,cfg.edge_iterations);mode="edges"
        solved,topology_alpha,topology=_topology_safe_blend(initial,solved,comp.local_faces,cfg.topology_area_floor)
        out[comp.mesh_name][ids]=solved
        reports.append({"component":comp.stable_id,"mesh":comp.mesh_name,"class":comp.structural_class,"vertices":comp.vertex_count,"mode":mode,"input_move_p95_mm":before,"projection_move_p95_mm":float(np.percentile(np.linalg.norm(solved-candidate,axis=1),95)*1000.0) if len(ids) else 0.0,"topology_alpha":topology_alpha,"topology":topology})

    # Preserve inferred thin-wall centroid vectors by equal and opposite component translation.
    thin_adjust=0
    for layer in structure.layers:
        for relation in layer.thin_walls:
            a=components[relation.component_a];b=components[relation.component_b];ca=_component_centroid(out,a);cb=_component_centroid(out,b)
            ia=_component_centroid(initial_positions,a);ib=_component_centroid(initial_positions,b);desired=ib-ia;current=cb-ca;delta=float(cfg.thin_wall_strength)*(desired-current)
            if np.linalg.norm(delta)>0.004:delta=_unit(delta)*0.004
            out[a.mesh_name][a.vertex_ids]-=0.5*delta;out[b.mesh_name][b.vertex_ids]+=0.5*delta;thin_adjust+=1

    # Preserve source layer ordering only when it is actually threatened.  Whole-layer
    # translation avoids pointwise embossing and cannot alter within-layer construction.
    layer_components={layer.stable_id:[components[x] for x in layer.component_ids] for layer in structure.layers}
    layer_lookup=structure.layer_by_id;handled=set();ordering_adjust=[]
    def layer_current_clearance(layer_id: str) -> tuple[float,np.ndarray]:
        values=[];normals=[]
        for comp in layer_components[layer_id]:
            ctx=contexts[comp.mesh_name];P=out[comp.mesh_name][comp.vertex_ids]
            values.append(np.einsum("nc,nc->n",P-ctx.target_anchor[comp.vertex_ids],ctx.target_normal[comp.vertex_ids]));normals.append(ctx.target_normal[comp.vertex_ids])
        v=np.concatenate(values) if values else np.zeros(0);n=np.vstack(normals) if normals else np.asarray([[0.,0.,1.]])
        return (float(np.median(v)) if len(v) else 0.0,_unit(np.mean(n,axis=0)))
    for layer in structure.layers:
        for rel in layer.ordering:
            key=tuple(sorted((layer.stable_id,rel.other_layer_id)))
            if key in handled or rel.relation=="peer":continue
            handled.add(key);other=layer_lookup[rel.other_layer_id]
            outer=layer if rel.relation=="outside" else other;inner=other if rel.relation=="outside" else layer
            outer_c,outer_n=layer_current_clearance(outer.stable_id);inner_c,_=layer_current_clearance(inner.stable_id)
            desired=float(rel.source_gap_m)*float(cfg.ordering_min_fraction);deficit=desired-(outer_c-inner_c)
            if deficit<=0.0:continue
            shift=min(float(cfg.ordering_max_shift_m),float(cfg.ordering_strength)*deficit);delta=outer_n*shift
            for comp in layer_components[outer.stable_id]:out[comp.mesh_name][comp.vertex_ids]+=delta
            ordering_adjust.append({"outer":outer.stable_id,"inner":inner.stable_id,"source_gap_mm":rel.source_gap_m*1000.0,"before_gap_mm":(outer_c-inner_c)*1000.0,"shift_mm":shift*1000.0})

    return out,{"revision":"lobomap-authored-structure-projection-v1","components":reports,"thin_wall_adjustments":thin_adjust,"ordering_adjustments":ordering_adjust}


@dataclass
class TargetCollisionSurface:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    incident_faces: np.ndarray
    tree: cKDTree

    @classmethod
    def build(cls, vertices: np.ndarray, faces: np.ndarray, vertex_normals: np.ndarray) -> "TargetCollisionSurface":
        V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);N=np.asarray(vertex_normals,dtype=np.float64)
        if V.ndim!=2 or V.shape[1]!=3 or N.shape!=V.shape or F.ndim!=2 or F.shape[1]!=3:raise ValueError("target collision surface arrays have invalid shape")
        if len(F) and (int(F.min())<0 or int(F.max())>=len(V)):raise ValueError("target collision faces reference invalid vertices")
        adjacency=[[] for _ in range(len(V))]
        for fi,tri in enumerate(F):
            for v in tri:adjacency[int(v)].append(fi)
        max_degree=max((len(x) for x in adjacency),default=0);incident=np.full((len(V),max(max_degree,1)),-1,dtype=np.int64)
        for i,rows in enumerate(adjacency):
            if rows:incident[i,:len(rows)]=rows
        nn=np.linalg.norm(N,axis=1);good=nn>_EPS;N=N.copy();N[good]/=nn[good,None];N[~good]=[0.,0.,1.]
        return cls(V,F,N,incident,cKDTree(V))


def collect_target_collision_surface(source: object, body_cache: Mapping[str,object], body_mesh_names: Iterable[str]) -> tuple[TargetCollisionSurface,dict[str,object]]:
    """Rebuild exact target body triangles from the source GLB topology + RBODY Y rows."""
    body={str(x) for x in body_mesh_names};chunks=[];faces=[];offset=0;mesh_order=[]
    for raw_name in source.mesh_names():
        name=str(raw_name or "")
        if name not in body:continue
        data=source.data(name);V=np.asarray(data["V"],dtype=np.float64);F=np.asarray(data["F"],dtype=np.int64)
        chunks.append(V);faces.append(F+offset);offset+=len(V);mesh_order.append(name)
    if not chunks:raise ValueError("target collision surface requires at least one supplied source body mesh")
    source_vertices=np.vstack(chunks);X=np.asarray(body_cache["X"],dtype=np.float64);Y=np.asarray(body_cache["Y"],dtype=np.float64);NT=np.asarray(body_cache["NT"],dtype=np.float64)
    if source_vertices.shape!=X.shape or not np.allclose(source_vertices,X,atol=2e-7,rtol=0.0):raise ValueError("source body mesh order/topology does not match authoritative RBODY correspondence X")
    surface=TargetCollisionSurface.build(Y,np.vstack(faces),NT)
    return surface,{"revision":"lobomap-target-triangle-surface-v1","body_meshes":mesh_order,"vertex_count":int(len(Y)),"triangle_count":int(sum(len(x) for x in faces))}


def _closest_target_triangles(points: np.ndarray, surface: TargetCollisionSurface, vertex_neighbors: int, *, chunk_size: int=2048) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    P=np.asarray(points,dtype=np.float64);n=len(P);best_point=np.zeros_like(P);best_normal=np.zeros_like(P);best_distance=np.full(n,np.inf,dtype=np.float64)
    k=min(int(vertex_neighbors),len(surface.vertices));_,nearest=surface.tree.query(P,k=k,workers=-1)
    if k==1:nearest=nearest[:,None]
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n);Q=P[start:end];candidate=surface.incident_faces[nearest[start:end]].reshape(end-start,-1)
        valid=candidate>=0
        # A body face is commonly incident to several of the nearest body vertices, so
        # the legacy slab can contain the same triangle many times.  Pack first
        # occurrences only, preserving historical candidate order/tie behaviour.
        unique=valid.copy()
        for col in range(1,candidate.shape[1]):
            unique[:,col]&=~np.any(candidate[:,col,None]==candidate[:,:col],axis=1)
        counts=np.sum(unique,axis=1);width=max(int(np.max(counts)),1);safe=np.zeros((len(Q),width),dtype=np.int64);packed_valid=np.zeros((len(Q),width),dtype=bool)
        rank=np.cumsum(unique,axis=1)-1;rr,cc=np.nonzero(unique);safe[rr,rank[rr,cc]]=candidate[rr,cc];packed_valid[rr,rank[rr,cc]]=True
        tri=surface.faces[safe];triangles=surface.vertices[tri];repeated=np.repeat(Q[:,None,:],safe.shape[1],axis=1)
        cp=triangle_closest_point(triangles.reshape(-1,3,3),repeated.reshape(-1,3)).reshape(end-start,safe.shape[1],3)
        distance=np.linalg.norm(repeated-cp,axis=2);distance[~packed_valid]=np.inf;choice=np.argmin(distance,axis=1);rows=np.arange(end-start);fid=safe[rows,choice]
        best_point[start:end]=cp[rows,choice];best_distance[start:end]=distance[rows,choice]
        vn=np.mean(surface.vertex_normals[surface.faces[fid]],axis=1);norm=np.linalg.norm(vn,axis=1);good=norm>_EPS;vn[good]/=norm[good,None]
        if np.any(~good):
            T=surface.vertices[surface.faces[fid[~good]]];geom=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);geom_norm=np.linalg.norm(geom,axis=1);geom/=np.maximum(geom_norm[:,None],_EPS);vn[~good]=geom
        best_normal[start:end]=vn
    return best_point,best_normal,best_distance


def apply_target_collision(meshes: Mapping[str,LoBoMapMeshInput], positions: Mapping[str,np.ndarray], structure: GarmentStructure, contexts: Mapping[str,LocalFitContext], surface: TargetCollisionSurface, *, config: StructuralSolveConfig | None = None, initial_positions: Mapping[str,np.ndarray] | None = None) -> tuple[dict[str,np.ndarray],dict[str,object]]:
    """Clear the literal target triangle surface while preserving component topology."""
    cfg=config or StructuralSolveConfig();out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};by_mesh:dict[str,list[GarmentComponent]]={}
    for comp in structure.components:by_mesh.setdefault(comp.mesh_name,[]).append(comp)
    reports=[];total=0;rigid_moved=0;remaining_total=0
    for mesh_name,mesh in meshes.items():
        P=out[mesh_name];cp,normals,distance=_closest_target_triangles(P,surface,cfg.collision_vertex_neighbors);signed=np.einsum("nc,nc->n",P-cp,normals)
        ctx=contexts[mesh_name];eligible=ctx.source_clearance>=-1e-6;deficit=np.where(eligible,np.maximum(float(cfg.collision_margin_m)-signed,0.0),0.0);deficit=np.minimum(deficit,float(cfg.collision_max_push_m))
        corrected=np.zeros(len(P),dtype=bool);topology_limited=0
        for comp in by_mesh.get(mesh_name,[]):
            ids=comp.vertex_ids;local=deficit[ids]
            if not np.any(local>0.0):continue
            pre=out[mesh_name][ids].copy()
            if comp.structural_class=="rigid":
                active=local>0.0;direction=_unit(np.sum(normals[ids][active]*local[active,None],axis=0));denom=normals[ids]@direction
                need=np.where((local>0.0)&(denom>0.15),local/np.maximum(denom,0.15),0.0);shift=min(float(cfg.collision_max_push_m),float(np.max(need)))
                candidate=pre+direction*shift;rigid_moved+=int(shift>0.0)
            else:
                candidate=pre+normals[ids]*local[:,None]
            reference=np.asarray(initial_positions[mesh_name],dtype=np.float64)[ids] if initial_positions is not None else pre
            safe,alpha,topology=_topology_safe_blend(reference,candidate,comp.local_faces,cfg.topology_area_floor)
            if alpha<.999 or int(topology.get("repaired_vertices",0))>0:topology_limited+=1
            # If a semi-rigid piece cannot take the local correction without threatening
            # topology, clear only the *remaining actual penetration* by one coherent
            # translation.  Do not translate the whole piece merely to chase the margin.
            if comp.structural_class=="semi_rigid":
                approx=np.einsum("nc,nc->n",safe-cp[ids],normals[ids]);actual_need=np.maximum(0.00005-approx,0.0)
                actual_need=np.where(ctx.source_clearance[ids]>=-1e-6,actual_need,0.0)
                if np.any(actual_need>0.0):
                    active=actual_need>0.0;direction=_unit(np.sum(normals[ids][active]*actual_need[active,None],axis=0));denom=normals[ids]@direction
                    need=np.where(active&(denom>0.15),actual_need/np.maximum(denom,0.15),0.0);shift=min(float(cfg.collision_max_push_m),float(np.max(need)))
                    if shift>0.0:safe=safe+direction*shift
            out[mesh_name][ids]=safe;corrected[ids]|=local>0.0
        count=int(np.count_nonzero(corrected));total+=count
        # Collision itself may locally re-fold a target patch.  Re-assert the untouched
        # source topology on the displacement field before measuring convergence.  If
        # that regularisation moves anything back into the body, the outer collision
        # loop sees the penetration and performs another contact pass.
        out[mesh_name],source_topology=enforce_source_topology(mesh,out[mesh_name],area_floor=0.01)
        # Verify against literal triangles after both topology guards.  Remaining contacts
        # are reported rather than hidden by a structural rollback.
        post=out[mesh_name];post_cp,post_n,_=_closest_target_triangles(post,surface,cfg.collision_vertex_neighbors);post_signed=np.einsum("nc,nc->n",post-post_cp,post_n)
        actual=eligible&(post_signed<-1e-7);margin_short=eligible&(post_signed<float(cfg.collision_margin_m)-1e-7)
        actual_count=int(np.count_nonzero(actual));margin_count=int(np.count_nonzero(margin_short));remaining_total+=actual_count
        reports.append({"mesh":mesh_name,"candidate_margin_contacts":int(np.count_nonzero(deficit>0.0)),"corrected_vertices":count,"remaining_penetrations":actual_count,"remaining_margin_violations":margin_count,"topology_limited_components":topology_limited,"source_topology":source_topology,"minimum_signed_distance_mm":float(np.min(post_signed[eligible])*1000.0) if np.any(eligible) else None,"maximum_push_mm":float(np.max(deficit)*1000.0) if len(deficit) else 0.0,"nearest_distance_p50_mm":float(np.percentile(distance,50)*1000.0) if len(distance) else 0.0})
    remaining_source_topology=int(sum(int(x["source_topology"]["after"]["area_floor_violations"]) for x in reports))
    return out,{"revision":"lobomap-target-triangle-collision-v4-source-topology-aware","corrected_vertices":int(total),"remaining_penetrations":int(remaining_total),"remaining_source_topology_violations":remaining_source_topology,"remaining_margin_violations":int(sum(x["remaining_margin_violations"] for x in reports)),"rigid_components_translated":int(rigid_moved),"meshes":reports}

