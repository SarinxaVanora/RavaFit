from __future__ import annotations

"""LoBoMap-style production lane for RavaFit.

This module deliberately lives beside ``production_b14.py``.  It does not patch or
replace the trusted strict-B14 lane.  The representation follows the central LoBoFit
idea: compile garment vertices into sparse bone-local coordinates on the source rig,
then decode the *whole garment coherently* on the target rig before any body-fit
refinement is attempted.

RavaFit-specific policy:
- source garment/body and target body are the only solve authorities;
- authored/control meshes are validation-only and never accepted by this module;
- rendered/indexed vertices are expected; dead storage vertices stay outside the solve;
- garment semantics are inferred from geometry/rigging, never from clothing names.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

_EPS = 1e-12


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v=np.asarray(v,dtype=np.float64)
    n=float(np.linalg.norm(v))
    if n>_EPS:return v/n
    if fallback is None:fallback=np.asarray([0.0,1.0,0.0],dtype=np.float64)
    f=np.asarray(fallback,dtype=np.float64);fn=float(np.linalg.norm(f))
    return f/max(fn,_EPS)


def _least_parallel_axis(z: np.ndarray) -> np.ndarray:
    axes=np.eye(3,dtype=np.float64)
    return axes[int(np.argmin(np.abs(axes@z)))]


@dataclass(frozen=True)
class BoneFrameSet:
    """Stable orthonormal local frames in one shared bone-name order.

    ``axes[b, :, 0:3]`` stores x/y/z as columns. ``lengths`` are strictly positive.
    ``parents`` uses frame indices and -1 for a root/unavailable parent.
    """

    bone_names: tuple[str, ...]
    origins: np.ndarray
    axes: np.ndarray
    lengths: np.ndarray
    parents: np.ndarray

    def __post_init__(self):
        count=len(self.bone_names)
        origins=np.asarray(self.origins,dtype=np.float64)
        axes=np.asarray(self.axes,dtype=np.float64)
        lengths=np.asarray(self.lengths,dtype=np.float64)
        parents=np.asarray(self.parents,dtype=np.int64)
        if origins.shape!=(count,3):raise ValueError(f"origins must be {(count,3)}, got {origins.shape}")
        if axes.shape!=(count,3,3):raise ValueError(f"axes must be {(count,3,3)}, got {axes.shape}")
        if lengths.shape!=(count,):raise ValueError(f"lengths must be {(count,)}, got {lengths.shape}")
        if parents.shape!=(count,):raise ValueError(f"parents must be {(count,)}, got {parents.shape}")
        if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(axes)) or not np.all(np.isfinite(lengths)):
            raise ValueError("bone frames contain non-finite values")
        if np.any(lengths<=0.0):raise ValueError("bone frame lengths must be positive")
        gram=np.einsum("bij,bik->bjk",axes,axes)
        if not np.allclose(gram,np.eye(3)[None,:,:],atol=2e-6,rtol=0.0):raise ValueError("bone frame axes are not orthonormal")
        object.__setattr__(self,"origins",origins)
        object.__setattr__(self,"axes",axes)
        object.__setattr__(self,"lengths",lengths)
        object.__setattr__(self,"parents",parents)


@dataclass(frozen=True)
class LoBoMapGarment:
    """Sparse source garment representation in bone-local coordinates.

    Arrays are [vertex, influence] except ``local`` which is [vertex, influence, xyz].
    Keeping only the strongest influences gives bounded memory and fast decode while
    retaining smooth multi-bone attachment.
    """

    bone_names: tuple[str, ...]
    source_vertices: np.ndarray
    influence_bones: np.ndarray
    influence_weights: np.ndarray
    local: np.ndarray

    def __post_init__(self):
        V=np.asarray(self.source_vertices,dtype=np.float64)
        B=np.asarray(self.influence_bones,dtype=np.int32)
        W=np.asarray(self.influence_weights,dtype=np.float64)
        L=np.asarray(self.local,dtype=np.float64)
        if V.ndim!=2 or V.shape[1]!=3:raise ValueError(f"source_vertices must be Nx3, got {V.shape}")
        if B.shape!=W.shape or B.ndim!=2:raise ValueError("influence bone/weight arrays must be matching NxK arrays")
        if B.shape[0]!=len(V) or L.shape!=(B.shape[0],B.shape[1],3):raise ValueError("LoBoMap sparse arrays do not match vertex count")
        if np.any(B<0) or np.any(B>=len(self.bone_names)):raise ValueError("LoBoMap contains invalid bone indices")
        if not np.all(np.isfinite(V)) or not np.all(np.isfinite(W)) or not np.all(np.isfinite(L)):raise ValueError("LoBoMap contains non-finite values")
        if np.any(W<0.0):raise ValueError("LoBoMap weights cannot be negative")
        if not np.allclose(W.sum(axis=1),1.0,atol=2e-8,rtol=0.0):raise ValueError("LoBoMap weights must sum to one per vertex")
        object.__setattr__(self,"source_vertices",V)
        object.__setattr__(self,"influence_bones",B)
        object.__setattr__(self,"influence_weights",W)
        object.__setattr__(self,"local",L)

    @property
    def vertex_count(self) -> int:return int(len(self.source_vertices))

    @property
    def influence_count(self) -> int:return int(self.influence_bones.shape[1])




@dataclass(frozen=True)
class LoBoMapMeshInput:
    """Strict rendered-topology view of one FFXIV/GLB garment mesh.

    ``raw_vertex_ids`` maps solved rendered vertices back to the untouched GLB/MDL
    storage vertex ids.  Dead/unreferenced storage vertices never enter LoBoMap.
    """

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    joint_names: tuple[str, ...]
    raw_vertex_ids: np.ndarray
    raw_vertex_count: int
    material: str | None = None

    def __post_init__(self):
        V=np.asarray(self.vertices,dtype=np.float64);F=np.asarray(self.faces,dtype=np.int64);W=np.asarray(self.weights,dtype=np.float64)
        ids=np.asarray(self.raw_vertex_ids,dtype=np.int64);names=tuple(str(x) for x in self.joint_names)
        if V.ndim!=2 or V.shape[1]!=3:raise ValueError(f"{self.name}: vertices must be Nx3, got {V.shape}")
        if F.ndim!=2 or F.shape[1]!=3:raise ValueError(f"{self.name}: faces must be Mx3, got {F.shape}")
        if W.shape!=(len(V),len(names)):raise ValueError(f"{self.name}: skin weights {W.shape} do not match rendered vertices/joints {(len(V),len(names))}")
        if ids.shape!=(len(V),):raise ValueError(f"{self.name}: raw vertex id map must contain one id per rendered vertex")
        if np.any(ids<0) or np.any(ids>=int(self.raw_vertex_count)):raise ValueError(f"{self.name}: raw vertex id map is outside the authored storage range")
        if len(F) and (int(F.min())<0 or int(F.max())>=len(V)):raise ValueError(f"{self.name}: rendered faces reference vertices outside the compact view")
        if not np.all(np.isfinite(V)) or not np.all(np.isfinite(W)):raise ValueError(f"{self.name}: rendered mesh contains non-finite values")
        if np.any(W<0.0):raise ValueError(f"{self.name}: rendered mesh contains negative skin weights")
        object.__setattr__(self,"vertices",V);object.__setattr__(self,"faces",F);object.__setattr__(self,"weights",W)
        object.__setattr__(self,"joint_names",names);object.__setattr__(self,"raw_vertex_ids",ids)

    @property
    def vertex_count(self) -> int:return int(len(self.vertices))


def _rendered_mesh_view(name: str, data: Mapping[str,object]) -> LoBoMapMeshInput:
    """Compact one GLB mesh to triangle-referenced rendered vertices only."""
    V=np.asarray(data.get("V"),dtype=np.float64);F=np.asarray(data.get("F"),dtype=np.int64);W=np.asarray(data.get("W"),dtype=np.float64)
    joint_names=tuple(str(x) for x in (data.get("joint_names") or ()))
    if V.ndim!=2 or V.shape[1]!=3:raise ValueError(f"{name}: GLB vertex payload is not Nx3")
    if F.ndim!=2 or F.shape[1]!=3:raise ValueError(f"{name}: GLB face payload is not Mx3")
    if W.ndim!=2 or W.shape[0]!=len(V) or W.shape[1]!=len(joint_names):raise ValueError(f"{name}: GLB skin payload does not match vertices/joint names")
    if not len(V) or not len(F):raise ValueError(f"{name}: empty rendered geometry")
    if int(F.min())<0 or int(F.max())>=len(V):raise ValueError(f"{name}: GLB faces reference vertices outside storage")
    ids=np.unique(F.reshape(-1)).astype(np.int64)
    remap=np.full(len(V),-1,dtype=np.int64);remap[ids]=np.arange(len(ids),dtype=np.int64)
    compact_faces=remap[F]
    return LoBoMapMeshInput(str(name),V[ids].copy(),compact_faces,W[ids].copy(),joint_names,ids,int(len(V)),None if data.get("material") is None else str(data.get("material")))


def collect_ffxiv_mesh_batch(source: object, *, body_mesh_names: Iterable[str] = (), mesh_filter: Iterable[str] | None = None) -> tuple[dict[str,LoBoMapMeshInput],dict[str,object]]:
    """Adapt RavaFit's real ``GLB``/mesh-view interface to strict LoBoMap inputs.

    The adapter intentionally knows nothing about garment names or types.  The caller
    supplies body meshes already identified by RavaFit's body-material logic; every
    other skinned, triangle-bearing mesh is compacted to rendered topology.
    """
    if not hasattr(source,"mesh_names") or not hasattr(source,"data"):raise TypeError("LoBoMap FFXIV adapter requires a source exposing mesh_names() and data(name)")
    body={str(x) for x in body_mesh_names};allowed=None if mesh_filter is None else {str(x) for x in mesh_filter}
    meshes:dict[str,LoBoMapMeshInput]={};records:dict[str,dict[str,object]]={}
    for raw_name in source.mesh_names():
        name=str(raw_name or "")
        if not name:
            continue
        if name in body:
            records[name]={"eligible":False,"reason":"source body mesh"};continue
        if allowed is not None and name not in allowed:
            records[name]={"eligible":False,"reason":"excluded by mesh filter"};continue
        try:data=source.data(name)
        except Exception as ex:
            records[name]={"eligible":False,"reason":f"not skinned garment geometry: {ex}"};continue
        try:view=_rendered_mesh_view(name,data)
        except ValueError as ex:
            records[name]={"eligible":False,"reason":str(ex)};continue
        meshes[name]=view
        records[name]={"eligible":True,"rendered_vertices":view.vertex_count,"raw_vertices":int(view.raw_vertex_count),"unreferenced_storage_vertices":int(view.raw_vertex_count-view.vertex_count),"triangles":int(len(view.faces)),"joint_count":int(len(view.joint_names)),"material":view.material}
    if not meshes:raise ValueError("No rendered skinned garment meshes were eligible for LoBoMap initialisation after removing the source body.")
    report={"eligible_mesh_count":int(len(meshes)),"rendered_vertex_count":int(sum(x.vertex_count for x in meshes.values())),"raw_vertex_count":int(sum(x.raw_vertex_count for x in meshes.values())),"meshes":records}
    return meshes,report


@dataclass(frozen=True)
class LoBoMapInitialisationReport:
    vertex_count: int
    influence_count: int
    displacement_p50_mm: float
    displacement_p95_mm: float
    displacement_max_mm: float
    reconstruction_error_max_mm: float


def build_bone_frames(bone_names: Sequence[str], joint_positions: np.ndarray, parents: Sequence[int], *, root_reference: np.ndarray | None = None) -> BoneFrameSet:
    """Construct hierarchical LoBoFit-style frames from skeleton joint positions.

    z follows the local bone segment. x is inherited from the parent and Gram-Schmidt
    projected against z; y completes a right-handed frame. Leaves use their parent
    segment direction, matching the practical FFXIV fallback used by RavaFit.
    """
    names=tuple(str(x) for x in bone_names)
    P=np.asarray(joint_positions,dtype=np.float64)
    par=np.asarray(parents,dtype=np.int64)
    n=len(names)
    if P.shape!=(n,3):raise ValueError(f"joint_positions must be {(n,3)}, got {P.shape}")
    if par.shape!=(n,):raise ValueError(f"parents must be {(n,)}, got {par.shape}")
    if np.any(par>=n) or np.any(par<-1):raise ValueError("parent index outside frame set")

    children=[[] for _ in range(n)]
    for i,p in enumerate(par):
        if p>=0:children[int(p)].append(i)

    z=np.zeros((n,3),dtype=np.float64);lengths=np.ones(n,dtype=np.float64)
    for i in range(n):
        candidates=[]
        for c in children[i]:
            v=P[c]-P[i]
            if np.linalg.norm(v)>1e-8:candidates.append(v)
        if candidates:
            # Prefer the longest immediate child segment; tiny helper bones should not
            # define the macro frame when a normal skeletal child is available.
            v=max(candidates,key=lambda x:float(np.linalg.norm(x)))
        elif par[i]>=0:
            v=P[i]-P[int(par[i])]
        else:
            v=np.asarray([0.0,1.0,0.0],dtype=np.float64)
        lengths[i]=max(float(np.linalg.norm(v)),1e-6);z[i]=_unit(v)

    axes=np.zeros((n,3,3),dtype=np.float64);done=np.zeros(n,dtype=bool)
    root_ref=None if root_reference is None else np.asarray(root_reference,dtype=np.float64)

    def solve(i: int):
        if done[i]:return
        p=int(par[i])
        if p>=0:solve(p)
        zi=z[i]
        if p>=0:
            inherited=axes[p,:,0]
            xi=inherited-zi*float(np.dot(inherited,zi))
            if np.linalg.norm(xi)<1e-8:
                inherited=axes[p,:,2]
                xi=inherited-zi*float(np.dot(inherited,zi))
        else:
            ref=root_ref if root_ref is not None else _least_parallel_axis(zi)
            xi=ref-zi*float(np.dot(ref,zi))
        xi=_unit(xi,_least_parallel_axis(zi));yi=_unit(np.cross(zi,xi));xi=_unit(np.cross(yi,zi))
        axes[i,:,0]=xi;axes[i,:,1]=yi;axes[i,:,2]=zi;done[i]=True

    for i in range(n):solve(i)
    return BoneFrameSet(names,P,axes,lengths,par)


def remap_weights(weights: np.ndarray, source_joint_names: Sequence[str], target_joint_names: Sequence[str]) -> np.ndarray:
    """Remap a dense garment skin matrix by joint name and normalise surviving mass."""
    W=np.asarray(weights,dtype=np.float64)
    source=tuple(str(x) for x in source_joint_names);target=tuple(str(x) for x in target_joint_names)
    if W.ndim!=2 or W.shape[1]!=len(source):raise ValueError("weights do not match source joint names")
    lookup={name:i for i,name in enumerate(target)};out=np.zeros((len(W),len(target)),dtype=np.float64)
    for src,name in enumerate(source):
        dst=lookup.get(name)
        if dst is not None:out[:,dst]+=W[:,src]
    mass=out.sum(axis=1)
    if np.any(mass<=_EPS):
        bad=np.where(mass<=_EPS)[0][:8].tolist();raise ValueError(f"garment vertices lost all LoBoMap bone influence after name remap: {bad}")
    out/=mass[:,None]
    return out


def compile_lobomap(vertices: np.ndarray, weights: np.ndarray, source_joint_names: Sequence[str], source_frames: BoneFrameSet, *, max_influences: int = 4, min_weight: float = 1e-5) -> LoBoMapGarment:
    """Compile an immutable sparse LoBoMap representation from the untouched source."""
    V=np.asarray(vertices,dtype=np.float64)
    if V.ndim!=2 or V.shape[1]!=3:raise ValueError("vertices must be Nx3")
    W=remap_weights(weights,source_joint_names,source_frames.bone_names)
    k=max(1,min(int(max_influences),W.shape[1]))
    order=np.argpartition(-W,k-1,axis=1)[:,:k]
    vals=np.take_along_axis(W,order,axis=1)
    vals=np.where(vals>=float(min_weight),vals,0.0)
    mass=vals.sum(axis=1)
    zero=mass<=_EPS
    if np.any(zero):
        strongest=np.argmax(W[zero],axis=1);order[zero,0]=strongest;vals[zero]=0.0;vals[zero,0]=1.0;mass[zero]=1.0
    vals/=mass[:,None]

    origins=source_frames.origins[order]
    axes=source_frames.axes[order]
    lengths=source_frames.lengths[order]
    delta=V[:,None,:]-origins
    local=np.einsum("nkc,nkcd->nkd",delta,axes)/lengths[:,:,None]
    return LoBoMapGarment(source_frames.bone_names,V,order.astype(np.int32),vals,local)


def decode_lobomap(mapping: LoBoMapGarment, target_frames: BoneFrameSet) -> np.ndarray:
    """Decode all garment vertices coherently on the target skeleton."""
    if target_frames.bone_names!=mapping.bone_names:
        raise ValueError("target LoBoMap frames must use the same bone-name order as the compiled source mapping")
    idx=mapping.influence_bones
    origins=target_frames.origins[idx]
    axes=target_frames.axes[idx]
    lengths=target_frames.lengths[idx]
    mapped=origins+lengths[:,:,None]*np.einsum("nkd,nkcd->nkc",mapping.local,axes)
    return np.sum(mapped*mapping.influence_weights[:,:,None],axis=1)


def initialise_garment(vertices: np.ndarray, weights: np.ndarray, joint_names: Sequence[str], source_frames: BoneFrameSet, target_frames: BoneFrameSet, *, max_influences: int = 4) -> tuple[np.ndarray, LoBoMapGarment, LoBoMapInitialisationReport]:
    """Compile once and produce the coherent target-space garment initialisation."""
    mapping=compile_lobomap(vertices,weights,joint_names,source_frames,max_influences=max_influences)
    # Decode back onto source frames as a strict representation invariant.
    reconstructed=decode_lobomap(mapping,source_frames)
    reconstruction=np.linalg.norm(reconstructed-mapping.source_vertices,axis=1)
    target=decode_lobomap(mapping,target_frames)
    movement=np.linalg.norm(target-mapping.source_vertices,axis=1)
    report=LoBoMapInitialisationReport(
        vertex_count=int(len(target)),influence_count=int(mapping.influence_count),
        displacement_p50_mm=float(np.percentile(movement,50)*1000.0) if len(movement) else 0.0,
        displacement_p95_mm=float(np.percentile(movement,95)*1000.0) if len(movement) else 0.0,
        displacement_max_mm=float(np.max(movement)*1000.0) if len(movement) else 0.0,
        reconstruction_error_max_mm=float(np.max(reconstruction)*1000.0) if len(reconstruction) else 0.0,
    )
    return target,mapping,report


def frame_parents_from_named_hierarchy(bone_names: Sequence[str], parent_by_name: Mapping[str,str|None]) -> np.ndarray:
    """Utility for adapters/tests: convert a named skeleton hierarchy to frame indices."""
    names=tuple(str(x) for x in bone_names);index={name:i for i,name in enumerate(names)};parents=np.full(len(names),-1,dtype=np.int64)
    for i,name in enumerate(names):
        parent=parent_by_name.get(name)
        if parent in index:parents[i]=index[parent]
    return parents


def _weighted_similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray, *, scale_min: float = 0.55, scale_max: float = 1.80) -> tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    """Weighted source->target similarity used by the RBODY/FXXIV frame adapter."""
    X=np.asarray(source,dtype=np.float64);Y=np.asarray(target,dtype=np.float64);w=np.maximum(np.asarray(weights,dtype=np.float64).reshape(-1),0.0)
    if X.shape!=Y.shape or X.ndim!=2 or X.shape[1]!=3 or len(w)!=len(X):raise ValueError("weighted similarity input shape mismatch")
    mass=float(w.sum())
    if mass<=_EPS:raise ValueError("weighted similarity has no support")
    w=w/mass;cx=np.sum(X*w[:,None],axis=0);cy=np.sum(Y*w[:,None],axis=0);A=X-cx;B=Y-cy
    H=(A*w[:,None]).T@B
    u,_,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0.0:
        vt[-1]*=-1.0;R=vt.T@u.T
    src_energy=float(np.sum(w*np.sum(A*A,axis=1)));tgt_energy=float(np.sum(w*np.sum(B*B,axis=1)))
    scale=np.sqrt(max(tgt_energy,_EPS)/max(src_energy,_EPS));scale=float(np.clip(scale,scale_min,scale_max))
    return cx,cy,R,scale


def build_body_correspondence_frames(source_body: np.ndarray, target_body: np.ndarray, body_weights: np.ndarray, bone_names: Sequence[str], *, min_effective_mass: float = 1e-4) -> tuple[BoneFrameSet,BoneFrameSet,dict[str,float]]:
    """Build matching bone-local frames from an RBODY source->target body correspondence.

    FFXIV MDL payloads do not carry the full authoring skeleton hierarchy.  RBODY does,
    however, give us source/target body correspondence in one skin-weight space.  For
    each bone influence we therefore fit one weighted similarity transform over the
    body region owned by that bone.  Expressed as a source and target ``BoneFrameSet``,
    this is algebraically the same decode used by LoBoMap Blending and gives the whole
    garment one coherent, body-aware target initialization without nearest-component
    rediscovery.
    """
    X=np.asarray(source_body,dtype=np.float64);Y=np.asarray(target_body,dtype=np.float64);BW=np.asarray(body_weights,dtype=np.float64)
    names=tuple(str(x) for x in bone_names);n=len(names)
    if X.shape!=Y.shape or X.ndim!=2 or X.shape[1]!=3:raise ValueError("source/target RBODY correspondence must be matching Nx3 arrays")
    if BW.shape!=(len(X),n):raise ValueError(f"body_weights must be {(len(X),n)}, got {BW.shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)) or not np.all(np.isfinite(BW)):raise ValueError("RBODY frame inputs contain non-finite values")

    # A weak global transform is only a fallback for bones with effectively no body
    # support. It is never chosen when the body correspondence actually supports the bone.
    global_w=np.maximum(BW.sum(axis=1),1e-6);gcx,gcy,gR,gs=_weighted_similarity(X,Y,global_w)
    so=np.zeros((n,3),float);to=np.zeros((n,3),float);sa=np.repeat(np.eye(3)[None,:,:],n,axis=0);ta=np.empty((n,3,3),float);sl=np.ones(n,float);tl=np.empty(n,float)
    supported=0;scales=[];residual=[]
    for b in range(n):
        w=np.maximum(BW[:,b],0.0);mass=float(w.sum())
        if mass>=float(min_effective_mass):
            cx,cy,R,s=_weighted_similarity(X,Y,w);supported+=1
            pred=cy+(X-cx)@(s*R).T
            err=np.linalg.norm(pred-Y,axis=1);active=w>max(float(np.max(w))*0.02,1e-6)
            if np.any(active):residual.extend(err[active].tolist())
        else:
            cx,cy,R,s=gcx,gcy,gR,gs
        so[b]=cx;to[b]=cy;ta[b]=R;tl[b]=s;scales.append(s)
    parents=np.full(n,-1,dtype=np.int64)
    source=BoneFrameSet(names,so,sa,sl,parents);target=BoneFrameSet(names,to,ta,tl,parents)
    report={
        "bone_count":float(n),"supported_bone_count":float(supported),
        "bone_scale_p50":float(np.percentile(scales,50)) if scales else 1.0,
        "bone_scale_p95":float(np.percentile(scales,95)) if scales else 1.0,
        "body_similarity_residual_p95_mm":float(np.percentile(residual,95)*1000.0) if residual else 0.0,
    }
    return source,target,report


def initialise_from_body_correspondence(vertices: np.ndarray, garment_weights: np.ndarray, garment_joint_names: Sequence[str], source_body: np.ndarray, target_body: np.ndarray, body_weights: np.ndarray, body_joint_names: Sequence[str], *, max_influences: int = 4) -> tuple[np.ndarray,LoBoMapGarment,LoBoMapInitialisationReport,dict[str,float]]:
    """FFXIV/RBODY entry point for the new coherent initialization lane."""
    source_frames,target_frames,frame_report=build_body_correspondence_frames(source_body,target_body,body_weights,body_joint_names)
    target,mapping,report=initialise_garment(vertices,garment_weights,garment_joint_names,source_frames,target_frames,max_influences=max_influences)
    return target,mapping,report,frame_report


def initialise_mesh_batch(meshes: Mapping[str,Mapping[str,object]|LoBoMapMeshInput], body_cache: Mapping[str,object], *, excluded_meshes: Iterable[str] = (), max_influences: int = 4) -> tuple[dict[str,np.ndarray],dict[str,LoBoMapGarment],dict[str,object]]:
    """Initialise a complete garment batch in one shared LoBoMap frame field.

    All meshes share one source->target frame set derived from the body pair; no
    mesh/component gets to rediscover its own target mapping.  Inputs may be plain
    RavaFit mesh dictionaries or strict :class:`LoBoMapMeshInput` views.
    """
    required=("X","Y","BW","names")
    missing=[key for key in required if key not in body_cache]
    if missing:raise ValueError(f"body_cache missing LoBoMap inputs: {missing}")
    source_frames,target_frames,frame_report=build_body_correspondence_frames(body_cache["X"],body_cache["Y"],body_cache["BW"],body_cache["names"])
    excluded=set(str(x) for x in excluded_meshes);positions={};mappings={};mesh_reports={};started=__import__("time").perf_counter()
    for name,data in meshes.items():
        if str(name) in excluded:continue
        if isinstance(data,LoBoMapMeshInput):
            V=data.vertices;W=data.weights;joint_names=data.joint_names
        else:
            V=np.asarray(data["V"],dtype=np.float64);W=np.asarray(data["W"],dtype=np.float64);joint_names=list(data["joint_names"])
        out,mapping,report=initialise_garment(V,W,joint_names,source_frames,target_frames,max_influences=max_influences)
        positions[str(name)]=out;mappings[str(name)]=mapping;mesh_reports[str(name)]=report.__dict__
    elapsed=__import__("time").perf_counter()-started
    report={"revision":"lobomap-coherent-init-v2-ffxiv-adapter","mesh_count":len(positions),"vertex_count":int(sum(len(v) for v in positions.values())),"elapsed_sec":float(elapsed),"frame_report":frame_report,"meshes":mesh_reports}
    return positions,mappings,report


def initialise_ffxiv_source(source: object, body_cache: Mapping[str,object], *, body_mesh_names: Iterable[str] = (), mesh_filter: Iterable[str] | None = None, max_influences: int = 4) -> tuple[dict[str,np.ndarray],dict[str,LoBoMapGarment],dict[str,LoBoMapMeshInput],dict[str,object]]:
    """Run coherent LoBoMap initialisation directly over RavaFit's production GLB view.

    This is the strict bridge used by the forthcoming refinement lane.  It compacts
    raw XIV storage to rendered topology, preserves an exact raw-id map for final
    restoration, and then performs one shared-frame whole-outfit initialisation.
    """
    meshes,adapter_report=collect_ffxiv_mesh_batch(source,body_mesh_names=body_mesh_names,mesh_filter=mesh_filter)
    positions,mappings,report=initialise_mesh_batch(meshes,body_cache,max_influences=max_influences)
    report=dict(report);report["ffxiv_adapter"]=adapter_report
    return positions,mappings,meshes,report


def restore_rendered_positions(raw_positions: np.ndarray, raw_vertex_ids: np.ndarray, solved_positions: np.ndarray) -> np.ndarray:
    """Restore rendered solved rows into raw storage without touching unreferenced rows."""
    raw=np.asarray(raw_positions);ids=np.asarray(raw_vertex_ids,dtype=np.int64);solved=np.asarray(solved_positions)
    if raw.ndim!=2 or raw.shape[1]!=3 or solved.shape!=(len(ids),3):raise ValueError("raw/rendered position shapes are incompatible")
    if len(ids) and (int(ids.min())<0 or int(ids.max())>=len(raw)):raise ValueError("raw vertex id map is outside position storage")
    if len(np.unique(ids))!=len(ids):raise ValueError("raw vertex id map must be one-to-one")
    out=raw.copy();out[ids]=solved.astype(out.dtype,copy=False);return out


def freeze_ffxiv_solution(source_path: object, output_path: object, meshes: Mapping[str,LoBoMapMeshInput], solved_positions: Mapping[str,np.ndarray], *, rebuild_normals: bool = True, rebuild_tangents: bool = True) -> dict[str,object]:
    """Freeze solved rendered vertices back into untouched XIV GLB storage.

    ``LoBoMapMeshInput.raw_vertex_ids`` is the sole rendered->storage authority. Dead
    storage rows are copied byte-for-byte; JOINTS/WEIGHTS are never rewritten. Derived
    NORMAL/TANGENT rows are refreshed only for rendered vertices so unreferenced storage
    remains untouched as well.
    """
    from pathlib import Path
    import sys
    import trimesh

    runtime_root=Path(__file__).resolve().parent.parent;scripts=runtime_root/"b14_frozen"/"scripts"
    if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
    from glb_patch_legacy import GLBEditor,compute_tangents

    src=Path(source_path);dst=Path(output_path);ed=GLBEditor(src);mesh_reports={}
    for name,mesh in meshes.items():
        if name not in solved_positions:raise ValueError(f"{name}: no solved position payload supplied for GLB freeze")
        _,primitive=ed.mesh_primitive(name);attrs=primitive.get("attributes",{});pos_acc=attrs.get("POSITION")
        if pos_acc is None:raise ValueError(f"{name}: GLB primitive has no POSITION accessor")
        original=ed.accessor(pos_acc);ids=np.asarray(mesh.raw_vertex_ids,dtype=np.int64);solved=np.asarray(solved_positions[name],dtype=np.float64)
        if original.shape!=(mesh.raw_vertex_count,3):raise ValueError(f"{name}: raw POSITION shape {original.shape} does not match adapter storage {(mesh.raw_vertex_count,3)}")
        if solved.shape!=(mesh.vertex_count,3):raise ValueError(f"{name}: solved rendered shape {solved.shape} does not match adapter {(mesh.vertex_count,3)}")
        raw=restore_rendered_positions(original,ids,solved);dead=np.ones(len(raw),dtype=bool);dead[ids]=False
        if not np.array_equal(raw[dead],original[dead]):raise AssertionError(f"{name}: dead raw POSITION storage changed before serialization")
        ed.write_accessor(pos_acc,raw)
        faces=ed.accessor(primitive["indices"]).reshape(-1,3).astype(np.int64)
        if rebuild_normals and "NORMAL" in attrs:
            old_n=ed.accessor(attrs["NORMAL"]);tri=trimesh.Trimesh(vertices=raw,faces=faces,process=False);calc=np.asarray(tri.vertex_normals,dtype=old_n.dtype);new_n=old_n.copy();new_n[ids]=calc[ids];ed.write_accessor(attrs["NORMAL"],new_n)
        if rebuild_tangents and "TANGENT" in attrs and "TEXCOORD_0" in attrs and "NORMAL" in attrs:
            old_t=ed.accessor(attrs["TANGENT"]);uv=ed.accessor(attrs["TEXCOORD_0"]).astype(np.float32);normals=ed.accessor(attrs["NORMAL"]).astype(np.float32)
            calc_t=compute_tangents(raw.astype(np.float32),faces,uv,normals,old_t.astype(np.float32));new_t=old_t.copy();new_t[ids]=calc_t[ids];ed.write_accessor(attrs["TANGENT"],new_t)
        ed.js["accessors"][pos_acc]["min"]=raw.min(axis=0).astype(float).tolist();ed.js["accessors"][pos_acc]["max"]=raw.max(axis=0).astype(float).tolist()
        mesh_reports[name]={"rendered_vertices":int(len(ids)),"raw_vertices":int(len(raw)),"dead_storage_vertices":int(np.count_nonzero(dead))}
    dst.parent.mkdir(parents=True,exist_ok=True);ed.save(dst)

    # Independent reopen audit: storage identity and authored skinning must survive disk.
    before=GLBEditor(src);after=GLBEditor(dst);skin_accessors=("JOINTS_0","JOINTS_1","WEIGHTS_0","WEIGHTS_1");dead_total=0;skin_checked=0
    for name,mesh in meshes.items():
        _,bp=before.mesh_primitive(name);_,ap=after.mesh_primitive(name);battrs=bp.get("attributes",{});aattrs=ap.get("attributes",{});ids=np.asarray(mesh.raw_vertex_ids,dtype=np.int64)
        bpos=before.accessor(battrs["POSITION"]);apos=after.accessor(aattrs["POSITION"]);dead=np.ones(len(bpos),dtype=bool);dead[ids]=False;dead_total+=int(np.count_nonzero(dead))
        if not np.array_equal(bpos[dead],apos[dead]):raise AssertionError(f"{name}: dead raw POSITION storage changed after GLB serialization")
        expected=np.asarray(solved_positions[name],dtype=apos.dtype)
        if not np.array_equal(apos[ids],expected):raise AssertionError(f"{name}: rendered solved POSITION rows did not round-trip exactly at storage precision")
        for semantic in skin_accessors:
            if semantic not in battrs and semantic not in aattrs:continue
            if semantic not in battrs or semantic not in aattrs:raise AssertionError(f"{name}: authored {semantic} accessor presence changed")
            skin_checked+=1
            if not np.array_equal(before.accessor(battrs[semantic]),after.accessor(aattrs[semantic])):raise AssertionError(f"{name}: authored {semantic} payload changed during LoBoMap freeze")
    return {"revision":"lobomap-raw-storage-freeze-v1","mesh_count":int(len(meshes)),"dead_storage_rows_verified":int(dead_total),"skinning_accessors_verified":int(skin_checked),"meshes":mesh_reports,"output_path":str(dst)}
