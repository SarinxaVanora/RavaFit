from __future__ import annotations

"""Localized coarse-to-fine residual refinement for the RavaFit LoBoMap lane.

The coherent LoBoMap decode remains deformation authority.  This module only solves a
smooth *residual* field needed to recover the source garment's body-relative fit style
on the target body correspondence.  It deliberately does not re-model garments in
world/global vertex space.

Current stage:
- body correspondence is chosen in source space and reused on the target;
- skin affinity biases spatial correspondence away from adjacent unrelated limbs;
- source signed normal clearance is the fit-style signal;
- only localized residual displacement is diffused over the authored mesh graph;
- open boundaries retain stronger data fidelity than interiors;
- a final zero-penetration guard applies only where the source garment was on/outside
  its support surface, so authored negative clearance is not silently rewritten.

Multi-layer ordering, rigid SE(3) projection and thin-wall coupling are intentionally
separate later constraints; this stage supplies the generic local residual machinery
those constraints will operate on.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from lobomap_production import LoBoMapGarment, LoBoMapMeshInput

_EPS=1e-12


def _unit_rows(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    V=np.asarray(values,dtype=np.float64)
    n=np.linalg.norm(V,axis=1)
    out=np.zeros_like(V)
    good=n>_EPS
    out[good]=V[good]/n[good,None]
    if np.any(~good):
        if fallback is None:
            out[~good]=np.asarray([0.0,0.0,1.0],dtype=np.float64)
        else:
            F=np.asarray(fallback,dtype=np.float64)
            if F.ndim==1:F=np.repeat(F[None,:],len(V),axis=0)
            fn=np.linalg.norm(F,axis=1);fg=fn>_EPS
            replacement=np.repeat(np.asarray([[0.0,0.0,1.0]],dtype=np.float64),len(V),axis=0)
            replacement[fg]=F[fg]/fn[fg,None]
            out[~good]=replacement[~good]
    return out


@dataclass(frozen=True)
class LoBoMapRefinementConfig:
    body_neighbors: int = 8
    spatial_affinity_floor: float = 0.10
    coarse_iterations: int = 8
    coarse_diffusion: float = 0.58
    fine_iterations: int = 3
    fine_diffusion: float = 0.20
    boundary_diffusion_scale: float = 0.35
    max_residual_m: float | None = 0.060

    def __post_init__(self):
        if self.body_neighbors<1:raise ValueError("body_neighbors must be positive")
        if not 0.0<=self.spatial_affinity_floor<=1.0:raise ValueError("spatial_affinity_floor must be in [0,1]")
        for name,value in (("coarse_diffusion",self.coarse_diffusion),("fine_diffusion",self.fine_diffusion),("boundary_diffusion_scale",self.boundary_diffusion_scale)):
            if not 0.0<=float(value)<=1.0:raise ValueError(f"{name} must be in [0,1]")
        if self.coarse_iterations<0 or self.fine_iterations<0:raise ValueError("refinement iterations cannot be negative")
        if self.max_residual_m is not None and self.max_residual_m<=0.0:raise ValueError("max_residual_m must be positive when supplied")


@dataclass(frozen=True)
class LocalFitContext:
    source_anchor: np.ndarray
    target_anchor: np.ndarray
    source_normal: np.ndarray
    target_normal: np.ndarray
    source_clearance: np.ndarray
    neighbour_indices: np.ndarray
    neighbour_weights: np.ndarray

    def __post_init__(self):
        n=len(np.asarray(self.source_clearance))
        for name,value,shape in (
            ("source_anchor",self.source_anchor,(n,3)),("target_anchor",self.target_anchor,(n,3)),
            ("source_normal",self.source_normal,(n,3)),("target_normal",self.target_normal,(n,3)),
        ):
            arr=np.asarray(value,dtype=np.float64)
            if arr.shape!=shape:raise ValueError(f"{name} must be {shape}, got {arr.shape}")
            if not np.all(np.isfinite(arr)):raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self,name,arr)
        c=np.asarray(self.source_clearance,dtype=np.float64).reshape(-1)
        I=np.asarray(self.neighbour_indices,dtype=np.int64);W=np.asarray(self.neighbour_weights,dtype=np.float64)
        if c.shape!=(n,) or I.ndim!=2 or I.shape[0]!=n or W.shape!=I.shape:raise ValueError("local fit context sparse arrays do not match vertex count")
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(W)):raise ValueError("local fit context contains non-finite values")
        if np.any(W<0.0) or not np.allclose(W.sum(axis=1),1.0,atol=2e-8,rtol=0.0):raise ValueError("local fit context neighbour weights must be normalized")
        object.__setattr__(self,"source_clearance",c);object.__setattr__(self,"neighbour_indices",I);object.__setattr__(self,"neighbour_weights",W)


@dataclass(frozen=True)
class MeshGraph:
    neighbour_average: csr_matrix
    boundary_vertices: np.ndarray


def build_mesh_graph(vertex_count: int, faces: np.ndarray) -> MeshGraph:
    F=np.asarray(faces,dtype=np.int64)
    if F.ndim!=2 or F.shape[1]!=3:raise ValueError("faces must be Mx3")
    n=int(vertex_count)
    if len(F) and (int(F.min())<0 or int(F.max())>=n):raise ValueError("faces reference vertices outside graph")
    if n<=0:raise ValueError("mesh graph requires vertices")
    directed=np.vstack([F[:,[0,1]],F[:,[1,0]],F[:,[1,2]],F[:,[2,1]],F[:,[2,0]],F[:,[0,2]]]) if len(F) else np.empty((0,2),dtype=np.int64)
    if len(directed):
        A=coo_matrix((np.ones(len(directed)),(directed[:,0],directed[:,1])),shape=(n,n)).tocsr();A.sum_duplicates();A.data[:]=1.0
        degree=np.asarray(A.sum(axis=1)).reshape(-1);inv=np.zeros(n,dtype=np.float64);good=degree>0;inv[good]=1.0/degree[good]
        avg=A.multiply(inv[:,None]).tocsr()
    else:
        avg=csr_matrix((n,n),dtype=np.float64)
    edges=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]) if len(F) else np.empty((0,2),dtype=np.int64)
    boundary=np.zeros(n,dtype=bool)
    if len(edges):
        undirected=np.sort(edges,axis=1);unique,counts=np.unique(undirected,axis=0,return_counts=True)
        bedges=unique[counts==1]
        if len(bedges):boundary[np.unique(bedges)]=True
    return MeshGraph(avg,boundary)


def build_local_fit_context(mesh: LoBoMapMeshInput, mapping: LoBoMapGarment, body_cache: Mapping[str,object], *, config: LoBoMapRefinementConfig | None = None) -> LocalFitContext:
    """Compile source-body correspondences once; target uses the exact same body ids."""
    cfg=config or LoBoMapRefinementConfig()
    required=("X","Y","BW","NS","NT","names")
    missing=[key for key in required if key not in body_cache]
    if missing:raise ValueError(f"body_cache missing LoBoMap refinement inputs: {missing}")
    X=np.asarray(body_cache["X"],dtype=np.float64);Y=np.asarray(body_cache["Y"],dtype=np.float64);BW=np.asarray(body_cache["BW"],dtype=np.float64)
    NS=_unit_rows(np.asarray(body_cache["NS"],dtype=np.float64));NT=_unit_rows(np.asarray(body_cache["NT"],dtype=np.float64))
    names=tuple(str(x) for x in body_cache["names"])
    if X.shape!=Y.shape or X.ndim!=2 or X.shape[1]!=3:raise ValueError("body correspondence X/Y must be matching Nx3 arrays")
    if BW.shape!=(len(X),len(names)) or NS.shape!=X.shape or NT.shape!=X.shape:raise ValueError("body weights/normals do not match body correspondence")
    if mapping.bone_names!=names:raise ValueError("LoBoMap mapping and body cache must use the same authoritative bone-name order")
    V=np.asarray(mesh.vertices,dtype=np.float64)
    if len(V)!=mapping.vertex_count:raise ValueError("mesh/mapping vertex counts differ")
    k=min(int(cfg.body_neighbors),len(X));tree=cKDTree(X);dist,idx=tree.query(V,k=k,workers=-1)
    if k==1:dist=dist[:,None];idx=idx[:,None]
    # Skin affinity is computed directly from the sparse LoBoMap influences, avoiding a
    # large dense garment-weight remap while still disambiguating adjacent body regions.
    sparse_bones=mapping.influence_bones[:,None,:]
    selected=BW[idx[:,:,None],sparse_bones]
    affinity=np.sum(selected*mapping.influence_weights[:,None,:],axis=2)
    sigma=np.maximum(dist[:,-1],1e-5)
    spatial=np.exp(-np.square(dist/sigma[:,None]))
    combined=spatial*(float(cfg.spatial_affinity_floor)+(1.0-float(cfg.spatial_affinity_floor))*np.clip(affinity,0.0,1.0))
    mass=combined.sum(axis=1)
    weak=mass<=_EPS
    if np.any(weak):combined[weak,0]=1.0;mass=combined.sum(axis=1)
    weight=combined/mass[:,None]
    sa=np.einsum("nk,nkc->nc",weight,X[idx]);ta=np.einsum("nk,nkc->nc",weight,Y[idx])
    sn=_unit_rows(np.einsum("nk,nkc->nc",weight,NS[idx]),NS[idx[:,0]])
    tn=_unit_rows(np.einsum("nk,nkc->nc",weight,NT[idx]),NT[idx[:,0]])
    clearance=np.einsum("nc,nc->n",V-sa,sn)
    return LocalFitContext(sa,ta,sn,tn,clearance,idx,weight)


def _clearance_error(vertices: np.ndarray, context: LocalFitContext) -> np.ndarray:
    current=np.einsum("nc,nc->n",np.asarray(vertices,dtype=np.float64)-context.target_anchor,context.target_normal)
    return context.source_clearance-current


def _diffuse_residual(raw: np.ndarray, graph: MeshGraph, *, iterations: int, diffusion: float, boundary_scale: float) -> np.ndarray:
    raw=np.asarray(raw,dtype=np.float64);out=raw.copy()
    if iterations<=0 or diffusion<=0.0:return out
    beta=np.full(len(raw),float(diffusion),dtype=np.float64);beta[graph.boundary_vertices]*=float(boundary_scale)
    for _ in range(int(iterations)):
        avg=graph.neighbour_average@out
        isolated=np.asarray(graph.neighbour_average.sum(axis=1)).reshape(-1)<=_EPS
        avg[isolated]=out[isolated]
        out=(1.0-beta[:,None])*raw+beta[:,None]*avg
    return out


def _clamp_residual(delta: np.ndarray, max_m: float | None) -> np.ndarray:
    if max_m is None:return delta
    length=np.linalg.norm(delta,axis=1);scale=np.ones(len(delta),dtype=np.float64);over=length>float(max_m)
    scale[over]=float(max_m)/np.maximum(length[over],_EPS)
    return delta*scale[:,None]


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    F=np.asarray(faces,dtype=np.int64)
    if not len(F):return np.empty((0,2),dtype=np.int64)
    edges=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]])
    return np.unique(np.sort(edges,axis=1),axis=0)


def _edge_strain_percent(reference: np.ndarray, candidate: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if not len(edges):return np.empty(0,dtype=np.float64)
    a=np.linalg.norm(reference[edges[:,0]]-reference[edges[:,1]],axis=1)
    b=np.linalg.norm(candidate[edges[:,0]]-candidate[edges[:,1]],axis=1)
    good=a>_EPS
    return np.abs(b[good]/a[good]-1.0)*100.0


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    F=np.asarray(faces,dtype=np.int64)
    if not len(F):return np.empty(0,dtype=np.float64)
    V=np.asarray(vertices,dtype=np.float64)
    return .5*np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)


def measure_structural_diagnostics(mesh: LoBoMapMeshInput, initial_vertices: np.ndarray, refined_vertices: np.ndarray, graph: MeshGraph | None = None) -> dict[str,object]:
    """Measure deformation quality without changing the solve.

    The source garment remains authored structural authority.  Initial-relative metrics are
    also reported because they expose what the residual stage itself changed, but they are
    diagnostics only: a large target-body shape change can legitimately make the coherent
    LoBoMap initialisation differ substantially from source edge lengths.
    """
    source=np.asarray(mesh.vertices,dtype=np.float64);initial=np.asarray(initial_vertices,dtype=np.float64);refined=np.asarray(refined_vertices,dtype=np.float64)
    if source.shape!=initial.shape or source.shape!=refined.shape:raise ValueError(f"{mesh.name}: structural diagnostic vertex arrays must have matching shapes")
    edges=_mesh_edges(mesh.faces)
    source_initial=_edge_strain_percent(source,initial,edges);source_refined=_edge_strain_percent(source,refined,edges);initial_refined=_edge_strain_percent(initial,refined,edges)
    source_area=_triangle_areas(source,mesh.faces);initial_area=_triangle_areas(initial,mesh.faces);refined_area=_triangle_areas(refined,mesh.faces)
    valid_area=initial_area>_EPS;area_ratio=refined_area[valid_area]/initial_area[valid_area] if np.any(valid_area) else np.empty(0,dtype=np.float64)
    source_valid=source_area>_EPS;source_area_ratio=refined_area[source_valid]/source_area[source_valid] if np.any(source_valid) else np.empty(0,dtype=np.float64)
    F=np.asarray(mesh.faces,dtype=np.int64)
    if len(F):
        ns=np.cross(source[F[:,1]]-source[F[:,0]],source[F[:,2]]-source[F[:,0]])
        ni=np.cross(initial[F[:,1]]-initial[F[:,0]],initial[F[:,2]]-initial[F[:,0]])
        nr=np.cross(refined[F[:,1]]-refined[F[:,0]],refined[F[:,2]]-refined[F[:,0]])
        ns_len=np.linalg.norm(ns,axis=1);ni_len=np.linalg.norm(ni,axis=1);nr_len=np.linalg.norm(nr,axis=1);valid_orientation=(ni_len>_EPS)&(nr_len>_EPS)
        orientation=np.ones(len(F),dtype=np.float64);orientation[valid_orientation]=np.einsum("nc,nc->n",ni[valid_orientation],nr[valid_orientation])/(ni_len[valid_orientation]*nr_len[valid_orientation])
        flips=valid_orientation&(orientation<0.0)
        source_valid=(ns_len>_EPS)&(nr_len>_EPS);source_signed_ratio=np.ones(len(F),dtype=np.float64)
        source_signed_ratio[source_valid]=np.einsum("nc,nc->n",ns[source_valid],nr[source_valid])/(ns_len[source_valid]*ns_len[source_valid])
        source_flips=source_valid&(source_signed_ratio<-1e-10);source_floor_bad=source_valid&(source_signed_ratio<0.01-1e-10)
    else:
        orientation=np.empty(0,dtype=np.float64);flips=np.zeros(0,dtype=bool);source_signed_ratio=np.empty(0,dtype=np.float64);source_flips=np.zeros(0,dtype=bool);source_floor_bad=np.zeros(0,dtype=bool);source_valid=np.zeros(0,dtype=bool)
    g=graph or build_mesh_graph(len(initial),mesh.faces)
    initial_avg=g.neighbour_average@initial;refined_avg=g.neighbour_average@refined
    isolated=np.asarray(g.neighbour_average.sum(axis=1)).reshape(-1)<=_EPS
    initial_avg[isolated]=initial[isolated];refined_avg[isolated]=refined[isolated]
    laplacian_delta=np.linalg.norm((refined-refined_avg)-(initial-initial_avg),axis=1)*1000.0
    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values,q)) if len(values) else 0.0
    return {
        "edge_count":int(len(edges)),
        "source_to_initial_edge_strain_p50_pct":percentile(source_initial,50),
        "source_to_initial_edge_strain_p95_pct":percentile(source_initial,95),
        "source_to_refined_edge_strain_p50_pct":percentile(source_refined,50),
        "source_to_refined_edge_strain_p95_pct":percentile(source_refined,95),
        "initial_to_refined_edge_strain_p50_pct":percentile(initial_refined,50),
        "initial_to_refined_edge_strain_p95_pct":percentile(initial_refined,95),
        "initial_area_ratio_min":float(np.min(area_ratio)) if len(area_ratio) else 1.0,
        "initial_area_ratio_p01":percentile(area_ratio,1) if len(area_ratio) else 1.0,
        "initial_area_ratio_p05":percentile(area_ratio,5) if len(area_ratio) else 1.0,
        "initial_area_ratio_p50":percentile(area_ratio,50) if len(area_ratio) else 1.0,
        "source_area_ratio_p01":percentile(source_area_ratio,1) if len(source_area_ratio) else 1.0,
        "source_area_ratio_p50":percentile(source_area_ratio,50) if len(source_area_ratio) else 1.0,
        "near_degenerate_triangle_count_ratio_lt_0_10":int(np.count_nonzero(area_ratio<.10)),
        # Initial-vs-final normal disagreement is retained as a deformation diagnostic
        # only.  Acceptance authority is the untouched source topology below.
        "triangle_flip_count_vs_initial":int(np.count_nonzero(flips)),
        "triangle_flip_fraction_vs_initial":float(np.count_nonzero(flips)/max(len(orientation),1)),
        "orientation_cosine_p01_vs_initial":percentile(orientation,1) if len(orientation) else 1.0,
        "source_orientation_valid_triangle_count":int(np.count_nonzero(source_valid)),
        "source_triangle_flip_count":int(np.count_nonzero(source_flips)),
        "source_triangle_flip_fraction":float(np.count_nonzero(source_flips)/max(int(np.count_nonzero(source_valid)),1)),
        "source_signed_area_ratio_min":float(np.min(source_signed_ratio[source_valid])) if np.any(source_valid) else 1.0,
        "source_signed_area_ratio_p01":percentile(source_signed_ratio[source_valid],1) if np.any(source_valid) else 1.0,
        "source_area_floor_violation_count_ratio_lt_0_01":int(np.count_nonzero(source_floor_bad)),
        "laplacian_delta_p50_mm":percentile(laplacian_delta,50),
        "laplacian_delta_p95_mm":percentile(laplacian_delta,95),
    }


def refine_mesh_local_residuals(mesh: LoBoMapMeshInput, mapping: LoBoMapGarment, initial_vertices: np.ndarray, body_cache: Mapping[str,object], *, config: LoBoMapRefinementConfig | None = None, context: LocalFitContext | None = None, graph: MeshGraph | None = None) -> tuple[np.ndarray,dict[str,object],LocalFitContext,MeshGraph]:
    """Coarse-to-fine localized residual solve for one rendered garment mesh."""
    cfg=config or LoBoMapRefinementConfig();V0=np.asarray(initial_vertices,dtype=np.float64)
    if V0.shape!=mesh.vertices.shape:raise ValueError(f"{mesh.name}: initial vertices {V0.shape} do not match rendered source {mesh.vertices.shape}")
    if not np.all(np.isfinite(V0)):raise ValueError(f"{mesh.name}: initial LoBoMap geometry contains non-finite values")
    ctx=context or build_local_fit_context(mesh,mapping,body_cache,config=cfg);g=graph or build_mesh_graph(len(V0),mesh.faces)
    before=_clearance_error(V0,ctx)
    raw=before[:,None]*ctx.target_normal
    coarse=_diffuse_residual(raw,g,iterations=cfg.coarse_iterations,diffusion=cfg.coarse_diffusion,boundary_scale=cfg.boundary_diffusion_scale)
    coarse=_clamp_residual(coarse,cfg.max_residual_m);V1=V0+coarse
    remaining=_clearance_error(V1,ctx);fine_raw=remaining[:,None]*ctx.target_normal
    fine=_diffuse_residual(fine_raw,g,iterations=cfg.fine_iterations,diffusion=cfg.fine_diffusion,boundary_scale=cfg.boundary_diffusion_scale)
    total=_clamp_residual(coarse+fine,cfg.max_residual_m);out=V0+total
    # Contact/separation guard: if source clearance was non-negative, never let local
    # target clearance cross the target support surface.  Negative authored source
    # clearance remains untouched rather than being guessed away.
    target_clearance=np.einsum("nc,nc->n",out-ctx.target_anchor,ctx.target_normal)
    guard=(ctx.source_clearance>=0.0)&(target_clearance<0.0)
    if np.any(guard):out[guard]+=(-target_clearance[guard])[:,None]*ctx.target_normal[guard]
    after=_clearance_error(out,ctx);move=np.linalg.norm(out-V0,axis=1)
    abs_before=np.abs(before);abs_after=np.abs(after)
    structural=measure_structural_diagnostics(mesh,V0,out,g)
    report={
        "revision":"lobomap-local-residual-v2-structural-diagnostics",
        "vertex_count":int(len(out)),"triangle_count":int(len(mesh.faces)),"boundary_vertex_count":int(np.count_nonzero(g.boundary_vertices)),
        "clearance_error_before_p50_mm":float(np.percentile(abs_before,50)*1000.0),"clearance_error_before_p95_mm":float(np.percentile(abs_before,95)*1000.0),
        "clearance_error_after_p50_mm":float(np.percentile(abs_after,50)*1000.0),"clearance_error_after_p95_mm":float(np.percentile(abs_after,95)*1000.0),
        "residual_move_p50_mm":float(np.percentile(move,50)*1000.0),"residual_move_p95_mm":float(np.percentile(move,95)*1000.0),"residual_move_max_mm":float(np.max(move)*1000.0),
        "contact_guard_vertices":int(np.count_nonzero(guard)),"body_neighbors":int(ctx.neighbour_indices.shape[1]),
        "structural":structural,
    }
    return out,report,ctx,g


def refine_ffxiv_batch(meshes: Mapping[str,LoBoMapMeshInput], mappings: Mapping[str,LoBoMapGarment], initial_positions: Mapping[str,np.ndarray], body_cache: Mapping[str,object], *, config: LoBoMapRefinementConfig | None = None) -> tuple[dict[str,np.ndarray],dict[str,object]]:
    """Refine the whole outfit without rediscovering a separate global deformation."""
    cfg=config or LoBoMapRefinementConfig();positions={};reports={};started=__import__("time").perf_counter()
    for name,mesh in meshes.items():
        if name not in mappings or name not in initial_positions:raise ValueError(f"{name}: LoBoMap batch is missing mapping or initial position")
        out,report,_,_=refine_mesh_local_residuals(mesh,mappings[name],initial_positions[name],body_cache,config=cfg)
        positions[name]=out;reports[name]=report
    elapsed=__import__("time").perf_counter()-started
    before=[row["clearance_error_before_p95_mm"] for row in reports.values()];after=[row["clearance_error_after_p95_mm"] for row in reports.values()]
    summary={"revision":"lobomap-local-residual-batch-v2-structural-diagnostics","mesh_count":int(len(positions)),"vertex_count":int(sum(len(v) for v in positions.values())),"elapsed_sec":float(elapsed),"mesh_p95_clearance_error_before_max_mm":float(max(before)) if before else 0.0,"mesh_p95_clearance_error_after_max_mm":float(max(after)) if after else 0.0,"meshes":reports}
    return positions,summary


def refine_ffxiv_batch_structured(meshes: Mapping[str,LoBoMapMeshInput], mappings: Mapping[str,LoBoMapGarment], initial_positions: Mapping[str,np.ndarray], body_cache: Mapping[str,object], *, config: LoBoMapRefinementConfig | None = None, structural_config: object | None = None, collision_surface: object | None = None) -> tuple[dict[str,np.ndarray],dict[str,object]]:
    """Run body-fit residuals with authored structure and literal collision constraints.

    Body fit is the data term.  One generic authored-structure projection then restores
    component/layer construction, after which topology-aware target-triangle collision
    iterates only until true penetration is gone (or the explicit iteration cap is hit).
    Structure is not re-run after contact, preventing the two constraints from undoing
    each other.
    """
    from lobomap_structure import StructuralSolveConfig, apply_authored_structure, apply_target_collision, enforce_source_topology_batch, infer_garment_structure
    cfg=config or LoBoMapRefinementConfig();scfg=structural_config if structural_config is not None else StructuralSolveConfig()
    started=__import__("time").perf_counter()
    working_initial,initial_topology_report=enforce_source_topology_batch(meshes,initial_positions,area_floor=0.01)
    structure,contexts,inference_report=infer_garment_structure(meshes,mappings,body_cache)
    graphs={};candidate={};residual_reports={}
    for name,mesh in meshes.items():
        if name not in mappings or name not in working_initial:raise ValueError(f"{name}: structured LoBoMap batch is missing mapping or initial position")
        out,report,ctx,graph=refine_mesh_local_residuals(mesh,mappings[name],working_initial[name],body_cache,config=cfg,context=contexts[name])
        candidate[name]=out;residual_reports[name]=report;contexts[name]=ctx;graphs[name]=graph

    structured,structure_report=apply_authored_structure(meshes,working_initial,candidate,structure,contexts,config=scfg)
    structured,post_structure_topology_report=enforce_source_topology_batch(meshes,structured,area_floor=0.01)
    collision_reports=[]
    if collision_surface is not None:
        for _ in range(int(scfg.collision_iterations)):
            structured,collision_report=apply_target_collision(meshes,structured,structure,contexts,collision_surface,config=scfg,initial_positions=working_initial);collision_reports.append(collision_report)
            if int(collision_report.get("remaining_penetrations",0))==0 and int(collision_report.get("remaining_source_topology_violations",0))==0:break

    mesh_reports={};before=[];after=[]
    for name,mesh in meshes.items():
        ctx=contexts[name];final_error=np.abs(_clearance_error(structured[name],ctx));initial_error=np.abs(_clearance_error(working_initial[name],ctx))
        structural=measure_structural_diagnostics(mesh,working_initial[name],structured[name],graphs[name])
        row={
            "clearance_error_initial_p95_mm":float(np.percentile(initial_error,95)*1000.0) if len(initial_error) else 0.0,
            "clearance_error_final_p50_mm":float(np.percentile(final_error,50)*1000.0) if len(final_error) else 0.0,
            "clearance_error_final_p95_mm":float(np.percentile(final_error,95)*1000.0) if len(final_error) else 0.0,
            "residual":residual_reports[name],"structural":structural,
        }
        mesh_reports[name]=row;before.append(row["clearance_error_initial_p95_mm"]);after.append(row["clearance_error_final_p95_mm"])
    elapsed=__import__("time").perf_counter()-started
    report={
        "revision":"lobomap-structured-residual-v3-source-topology-authoritative",
        "mesh_count":int(len(structured)),"vertex_count":int(sum(len(x) for x in structured.values())),"elapsed_sec":float(elapsed),
        "mesh_p95_clearance_error_initial_max_mm":float(max(before)) if before else 0.0,
        "mesh_p95_clearance_error_final_max_mm":float(max(after)) if after else 0.0,
        "initial_source_topology":initial_topology_report,"structure_inference":inference_report,"structure_projection":structure_report,"post_structure_source_topology":post_structure_topology_report,"collision_passes":collision_reports,
        "remaining_actual_penetrations":int(collision_reports[-1].get("remaining_penetrations",0)) if collision_reports else None,
        "remaining_source_topology_violations":int(collision_reports[-1].get("remaining_source_topology_violations",0)) if collision_reports else int(sum(x["structural"]["source_area_floor_violation_count_ratio_lt_0_01"] for x in mesh_reports.values())),
        "remaining_source_triangle_flips":int(sum(x["structural"]["source_triangle_flip_count"] for x in mesh_reports.values())),
        "remaining_margin_violations":int(collision_reports[-1].get("remaining_margin_violations",0)) if collision_reports else None,
        "meshes":mesh_reports,
    }
    return structured,report



def solve_ffxiv_source_structured(source: object, body_cache: Mapping[str,object], *, body_mesh_names=(), mesh_filter=None, max_influences: int = 4, config: LoBoMapRefinementConfig | None = None, structural_config: object | None = None) -> tuple[dict[str,np.ndarray],dict[str,LoBoMapGarment],dict[str,LoBoMapMeshInput],dict[str,object]]:
    """One-call real FFXIV entry point for coherent init -> structure -> collision.

    The source body mesh names are explicit data supplied by RavaFit's existing body
    identification path.  They are used both to exclude body geometry from the garment
    solve and to rebuild the exact target collision triangles from RBODY correspondence.
    """
    from lobomap_production import initialise_ffxiv_source
    from lobomap_structure import collect_target_collision_surface
    initial,mappings,meshes,initial_report=initialise_ffxiv_source(source,body_cache,body_mesh_names=body_mesh_names,mesh_filter=mesh_filter,max_influences=max_influences)
    collision_surface,collision_surface_report=collect_target_collision_surface(source,body_cache,body_mesh_names)
    final,solve_report=refine_ffxiv_batch_structured(meshes,mappings,initial,body_cache,config=config,structural_config=structural_config,collision_surface=collision_surface)
    report={"revision":"lobomap-ffxiv-structured-solve-v1","initialisation":initial_report,"collision_surface":collision_surface_report,"solve":solve_report}
    return final,mappings,meshes,report
