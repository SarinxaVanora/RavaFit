from __future__ import annotations

"""Minimal evidence-gated local retargeting around Frozen B14 geometry.

Frozen B14 remains geometry authority.  This module may alter only a connected local
component when the untouched source body proves that B14 transferred that component's
support/clearance relationship poorly *and* the existing body correspondence predicts a
materially better relationship.  It never globally conforms or re-solves a garment.

The implementation is garment-name agnostic.  Narrow open ribbon/shell components are
recognised from topology and physical shape, then a low-frequency correction field is
fit to B14 -> correspondence displacement.  B14 edge lengths are re-projected before a
small literal-body collision polish.  All non-selected vertices remain bit-identical.
"""

from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

_EPS=1e-12


@dataclass(frozen=True)
class LocalRetargetConfig:
    min_vertices: int = 24
    min_extent_m: float = 0.045
    max_thin_ratio: float = 0.10
    max_middle_ratio: float = 0.65
    min_boundary_fraction: float = 0.08
    min_b14_clearance_p95_error_m: float = 0.0020
    required_clearance_p95_ratio: float = 0.65
    displacement_bins: int = 12
    edge_iterations: int = 80
    edge_relaxation: float = 0.70
    macro_pull: float = 0.001
    collision_margin_m: float = 0.00015
    collision_iterations: int = 8
    collision_blend: float = 0.35
    collision_max_push_m: float = 0.0015
    max_edge_strain_p95: float = 0.07
    min_area_ratio: float = 0.45

    # Source-proven disconnected attachment continuity.  These gates are deliberately geometric
    # rather than garment-specific: repeated close witnesses prove that two disconnected authored
    # components meet in the source, while compact children (rings, buckles, clips, trim hardware)
    # are allowed to follow the final frame of the larger component they are attached to.
    attachment_search_radius_m: float = 0.0040
    attachment_confirm_radius_m: float = 0.0025
    attachment_min_witnesses: int = 3
    attachment_tiny_extent_m: float = 0.018
    attachment_tiny_radius_m: float = 0.0010
    attachment_local_frame_radius_m: float = 0.012
    attachment_compact_extent_m: float = 0.040
    attachment_compact_relative_extent: float = 0.55
    attachment_compact_relative_vertices: float = 0.40
    attachment_max_rebind_m: float = 0.030
    attachment_pin_iterations: int = 140
    attachment_pin_relaxation: float = 0.55
    attachment_pin_goal_pull: float = 0.018


def _component_labels(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    F=np.asarray(faces,dtype=np.int64)
    if vertex_count<=0:return np.zeros(0,dtype=np.int64)
    if not len(F):return np.arange(vertex_count,dtype=np.int64)
    edges=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]))
    rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
    graph=coo_matrix((np.ones(len(rows),dtype=np.uint8),(rows,cols)),shape=(vertex_count,vertex_count)).tocsr()
    _,labels=connected_components(graph,directed=False,return_labels=True)
    return labels.astype(np.int64)


def _local_faces(faces: np.ndarray, ids: np.ndarray, vertex_count: int) -> np.ndarray:
    ids=np.asarray(ids,dtype=np.int64);mask=np.zeros(vertex_count,dtype=bool);mask[ids]=True
    selected=np.asarray(faces,dtype=np.int64)[np.all(mask[np.asarray(faces,dtype=np.int64)],axis=1)]
    if not len(selected):return np.zeros((0,3),dtype=np.int64)
    lookup=np.full(vertex_count,-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64)
    return lookup[selected]


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    F=np.asarray(faces,dtype=np.int64)
    if not len(F):return np.zeros((0,2),dtype=np.int64)
    return np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)


def _shape_features(vertices: np.ndarray, faces: np.ndarray) -> dict[str,float]:
    P=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if not len(P):return {"max_extent":0.0,"middle_ratio":0.0,"thin_ratio":0.0,"boundary_fraction":0.0}
    centred=P-np.mean(P,axis=0);sv=np.linalg.svd(centred,compute_uv=False);lead=max(float(sv[0]) if len(sv) else 0.0,_EPS)
    edges=np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1) if len(F) else np.zeros((0,2),dtype=np.int64)
    boundary=0.0
    if len(edges):
        unique,count=np.unique(edges,axis=0,return_counts=True);bedges=unique[count==1]
        if len(bedges):boundary=float(len(np.unique(bedges))/len(P))
    return {"max_extent":float(np.max(np.ptp(P,axis=0))),"middle_ratio":float(sv[1]/lead) if len(sv)>1 else 0.0,"thin_ratio":float(sv[2]/lead) if len(sv)>2 else 0.0,"boundary_fraction":boundary}


def _is_local_ribbon_candidate(features: dict[str,float], vertex_count: int, cfg: LocalRetargetConfig) -> bool:
    return int(vertex_count)>=cfg.min_vertices and float(features["max_extent"])>=cfg.min_extent_m and float(features["middle_ratio"])<=cfg.max_middle_ratio and float(features["thin_ratio"])<=cfg.max_thin_ratio and float(features["boundary_fraction"])>=cfg.min_boundary_fraction


def _geodesic_coordinate(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    P=np.asarray(vertices,dtype=np.float64);edges=_unique_edges(faces)
    if len(P)<2 or not len(edges):return np.linspace(0.0,1.0,len(P),dtype=np.float64)
    length=np.linalg.norm(P[edges[:,1]]-P[edges[:,0]],axis=1)
    graph=coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(P),len(P))).tocsr()
    first=dijkstra(graph,indices=0);a=int(np.argmax(first));da=dijkstra(graph,indices=a);b=int(np.argmax(da));db=dijkstra(graph,indices=b)
    return np.asarray(da/np.maximum(da+db,_EPS),dtype=np.float64)


def _low_frequency_displacement(source: np.ndarray, faces: np.ndarray, displacement: np.ndarray, bins: int) -> np.ndarray:
    t=_geodesic_coordinate(source,faces);D=np.asarray(displacement,dtype=np.float64);bins=max(4,int(bins));cuts=np.linspace(0.0,1.0,bins+1);xs=[];ys=[]
    for i in range(bins):
        sel=(t>=cuts[i])&(t<(cuts[i+1] if i<bins-1 else cuts[i+1]+1e-12))
        if np.any(sel):xs.append(float(np.median(t[sel])));ys.append(np.median(D[sel],axis=0))
    if len(xs)<2:return np.repeat(np.median(D,axis=0,keepdims=True),len(D),axis=0)
    x=np.asarray(xs,dtype=np.float64);y=np.asarray(ys,dtype=np.float64);out=np.empty_like(D);order=np.argsort(x);x=x[order];y=y[order]
    for axis in range(3):
        spline=UnivariateSpline(x,y[:,axis],k=min(3,len(x)-1),s=0.0,ext=3)
        out[:,axis]=spline(t)
    return out


def _edge_project(rest: np.ndarray, target: np.ndarray, faces: np.ndarray, cfg: LocalRetargetConfig) -> np.ndarray:
    R=np.asarray(rest,dtype=np.float64);goal=np.asarray(target,dtype=np.float64);U=goal.copy();edges=_unique_edges(faces)
    if not len(edges):return U
    rest_len=np.linalg.norm(R[edges[:,1]]-R[edges[:,0]],axis=1)
    for _ in range(max(0,int(cfg.edge_iterations))):
        vec=U[edges[:,1]]-U[edges[:,0]];cur=np.linalg.norm(vec,axis=1);direction=vec/np.maximum(cur[:,None],_EPS);corr=.5*(cur-rest_len)[:,None]*direction
        accum=np.zeros_like(U);count=np.zeros(len(U),dtype=np.float64);np.add.at(accum,edges[:,0],corr);np.add.at(accum,edges[:,1],-corr);np.add.at(count,edges[:,0],1.0);np.add.at(count,edges[:,1],1.0)
        U+=float(cfg.edge_relaxation)*accum/np.maximum(count[:,None],1.0)
        U+=float(cfg.macro_pull)*(goal-U)
    return U


def _edge_project_with_pins(rest: np.ndarray, target: np.ndarray, faces: np.ndarray, pin_ids: np.ndarray, pin_positions: np.ndarray, *, iterations: int=140, relaxation: float=.55, goal_pull: float=.018) -> np.ndarray:
    """Restore local ribbon edge lengths while keeping source-proven attachment witnesses hard-pinned.

    Pins are the actual ribbon-side witness vertices from the untouched source attachment relation.
    They therefore remain in the final parent frame instead of being allowed to drift away again
    during ordinary edge-length projection.
    """
    R=np.asarray(rest,dtype=np.float64);goal=np.asarray(target,dtype=np.float64);U=goal.copy();F=np.asarray(faces,dtype=np.int64)
    pins=np.asarray(pin_ids,dtype=np.int64).reshape(-1);targets=np.asarray(pin_positions,dtype=np.float64).reshape((-1,3))
    if not len(pins):return U
    # A witness vertex can appear in more than one nearest pair. Average those targets deterministically.
    unique=np.unique(pins);pin_target=np.zeros((len(unique),3),dtype=np.float64)
    for row,vid in enumerate(unique):pin_target[row]=np.mean(targets[pins==vid],axis=0)
    pinned=np.zeros(len(U),dtype=bool);pinned[unique]=True;U[unique]=pin_target
    edges=_unique_edges(F)
    if not len(edges):return U
    rest_len=np.linalg.norm(R[edges[:,1]]-R[edges[:,0]],axis=1)
    free=~pinned
    for _ in range(max(0,int(iterations))):
        vec=U[edges[:,1]]-U[edges[:,0]];cur=np.linalg.norm(vec,axis=1);direction=vec/np.maximum(cur[:,None],_EPS);error=(cur-rest_len)[:,None]*direction
        accum=np.zeros_like(U);count=np.zeros(len(U),dtype=np.float64)
        a=edges[:,0];b=edges[:,1];fa=free[a];fb=free[b]
        both=fa&fb
        if np.any(both):
            np.add.at(accum,a[both], .5*error[both]);np.add.at(accum,b[both],-.5*error[both]);np.add.at(count,a[both],1.0);np.add.at(count,b[both],1.0)
        only_a=fa&~fb
        if np.any(only_a):np.add.at(accum,a[only_a],error[only_a]);np.add.at(count,a[only_a],1.0)
        only_b=~fa&fb
        if np.any(only_b):np.add.at(accum,b[only_b],-error[only_b]);np.add.at(count,b[only_b],1.0)
        movable=np.flatnonzero(free)
        U[movable]+=float(relaxation)*accum[movable]/np.maximum(count[movable,None],1.0)
        U[movable]+=float(goal_pull)*(goal[movable]-U[movable])
        U[unique]=pin_target
    return U


def _edge_strain_p95(reference: np.ndarray, candidate: np.ndarray, faces: np.ndarray) -> float:
    edges=_unique_edges(faces)
    if not len(edges):return 0.0
    a=np.linalg.norm(reference[edges[:,1]]-reference[edges[:,0]],axis=1);b=np.linalg.norm(candidate[edges[:,1]]-candidate[edges[:,0]],axis=1)
    return float(np.percentile(np.abs(b/np.maximum(a,_EPS)-1.0),95))


def _topology_ok(reference: np.ndarray, candidate: np.ndarray, faces: np.ndarray, min_area_ratio: float) -> bool:
    F=np.asarray(faces,dtype=np.int64)
    if not len(F):return True
    ar=np.cross(reference[F[:,1]]-reference[F[:,0]],reference[F[:,2]]-reference[F[:,0]]);ac=np.cross(candidate[F[:,1]]-candidate[F[:,0]],candidate[F[:,2]]-candidate[F[:,0]])
    nr=np.linalg.norm(ar,axis=1);nc=np.linalg.norm(ac,axis=1);valid=nr>1e-12
    if np.any(valid & (np.einsum("ij,ij->i",ar,ac)<0.0)):return False
    if np.any(valid & (nc/np.maximum(nr,_EPS)<float(min_area_ratio))):return False
    return True


def _signed_clearance(nearest_surface_fn: Callable[...,Any], vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    result=nearest_surface_fn(np.asarray(vertices,dtype=np.float64),np.asarray(triangles,dtype=np.float64),k=24)
    return np.asarray(result[2],dtype=np.float64)



def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    out=np.zeros_like(V)
    if not len(F):return out
    tri=V[F];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    for j in range(3):np.add.at(out,F[:,j],fn)
    length=np.linalg.norm(out,axis=1);good=length>_EPS;out[good]/=length[good,None]
    return out


def _topology_neighbours(vertex_count: int, faces: np.ndarray) -> tuple[list[set[int]],np.ndarray]:
    nbr=[set() for _ in range(int(vertex_count))];edge_count={}
    F=np.asarray(faces,dtype=np.int64)
    for a,b,c in F:
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            nbr[u].add(v);nbr[v].add(u);key=(u,v) if u<v else (v,u);edge_count[key]=edge_count.get(key,0)+1
    boundary=np.zeros(int(vertex_count),dtype=bool)
    for (u,v),count in edge_count.items():
        if count==1:boundary[u]=True;boundary[v]=True
    return nbr,boundary


def _boundary_hops(nbr: list[set[int]], boundary: np.ndarray) -> np.ndarray:
    from collections import deque
    hops=np.full(len(nbr),999,dtype=np.int64);q=deque(int(x) for x in np.flatnonzero(boundary))
    for x in q:hops[x]=0
    while q:
        x=q.popleft()
        for nb in nbr[x]:
            if hops[nb]>hops[x]+1:hops[nb]=hops[x]+1;q.append(nb)
    return hops


def _smooth_displacement(displacement: np.ndarray, nbr: list[set[int]], ids: np.ndarray, iterations: int=4, alpha: float=.5) -> np.ndarray:
    D=np.asarray(displacement,dtype=np.float64);out=D.copy();active=np.zeros(len(D),dtype=bool);active[np.asarray(ids,dtype=np.int64)]=True
    for _ in range(max(0,int(iterations))):
        nxt=out.copy()
        for i in np.asarray(ids,dtype=np.int64):
            local=[j for j in nbr[int(i)] if active[j]]
            if local:nxt[i]=(1.0-alpha)*out[i]+alpha*np.mean(out[local],axis=0)
        out=nxt
    return out


def _cluster_points(ids: np.ndarray, vertices: np.ndarray, radius_m: float=.035) -> list[np.ndarray]:
    ids=np.asarray(ids,dtype=np.int64)
    if not len(ids):return []
    P=np.asarray(vertices,dtype=np.float64);labels=np.full(len(ids),-1,dtype=np.int64);group=0
    for i in range(len(ids)):
        if labels[i]>=0:continue
        labels[i]=group;changed=True
        while changed:
            changed=False;members=np.flatnonzero(labels==group)
            for j in range(len(ids)):
                if labels[j]>=0:continue
                if float(np.min(np.linalg.norm(P[ids[j]]-P[ids[members]],axis=1)))<radius_m:
                    labels[j]=group;changed=True
        group+=1
    return [ids[labels==g] for g in range(group)]


def _graph_distance_from_seeds(nbr: list[set[int]], seeds: np.ndarray, component_mask: np.ndarray, boundary_hops: np.ndarray, max_ring: int=7) -> np.ndarray:
    from collections import deque
    dist=np.full(len(nbr),999,dtype=np.int64);q=deque()
    for i in np.asarray(seeds,dtype=np.int64):dist[i]=0;q.append(int(i))
    while q:
        u=q.popleft()
        if dist[u]>=max_ring:continue
        for v in nbr[u]:
            if not component_mask[v] or boundary_hops[v]<3:continue
            if dist[v]>dist[u]+1:dist[v]=dist[u]+1;q.append(v)
    return dist


def _quadratic_bridge_target(base: np.ndarray, seed_group: np.ndarray, nbr: list[set[int]], component_mask: np.ndarray, boundary_hops: np.ndarray, target_normals: np.ndarray, *, pull: float=.62, max_move_m: float=.0032) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    V=np.asarray(base,dtype=np.float64);dist=_graph_distance_from_seeds(nbr,seed_group,component_mask,boundary_hops,7)
    core=np.flatnonzero(dist<=3);fit_ids=np.flatnonzero((dist>=4)&(dist<=7))
    if len(core)<3 or len(fit_ids)<12:return V.copy(),np.zeros(len(V),dtype=bool),{"selected":False,"reason":"insufficient local shell neighbourhood"}
    n=np.mean(np.asarray(target_normals,dtype=np.float64)[core],axis=0);nl=float(np.linalg.norm(n))
    if nl<=_EPS:return V.copy(),np.zeros(len(V),dtype=bool),{"selected":False,"reason":"local target normal is degenerate"}
    n/=nl;center=np.mean(V[fit_ids],axis=0);Q=V[fit_ids]-center;Qp=Q-np.outer(Q@n,n)
    _,_,vh=np.linalg.svd(Qp,full_matrices=False);t1=vh[0]-n*float(np.dot(vh[0],n));tl=float(np.linalg.norm(t1))
    if tl<=_EPS:return V.copy(),np.zeros(len(V),dtype=bool),{"selected":False,"reason":"local shell tangent is degenerate"}
    t1/=tl;t2=np.cross(n,t1);t2/=max(float(np.linalg.norm(t2)),_EPS)
    def _uvh(ids):
        q=V[np.asarray(ids,dtype=np.int64)]-center
        return q@t1,q@t2,q@n
    u,v,h=_uvh(fit_ids);A=np.column_stack((np.ones(len(u)),u,v,u*u,u*v,v*v));coef,*_=np.linalg.lstsq(A,h,rcond=None)
    uc,vc,hc=_uvh(core);Ac=np.column_stack((np.ones(len(uc)),uc,vc,uc*uc,uc*vc,vc*vc));fitted=Ac@coef;residual=hc-fitted
    outer=np.abs(h-A@coef);tolerance=max(float(np.percentile(outer,90)),.00020);excess=np.maximum(residual-tolerance,0.0)
    positive=int(np.count_nonzero(excess>0.0))
    if positive<3:return V.copy(),np.zeros(len(V),dtype=bool),{"selected":False,"reason":"surrounding shell does not prove a local outward artifact","positive_excess_vertices":positive,"tolerance_mm":tolerance*1000.0}
    target=V.copy();requested=np.minimum(excess*float(pull),float(max_move_m));target[core]-=requested[:,None]*n[None,:]
    raw=target-V;changed=np.linalg.norm(raw,axis=1)>1e-12
    # feather one topology ring beyond the core while keeping the fitted outer shell fixed
    ring4=np.flatnonzero(dist==4)
    for i in ring4:
        nearest=core[np.argsort(np.linalg.norm(V[core]-V[i],axis=1))[:8]]
        if len(nearest):target[i]+=0.58*np.mean(raw[nearest],axis=0)
    ring5=np.flatnonzero(dist==5)
    for i in ring5:
        nearest=core[np.argsort(np.linalg.norm(V[core]-V[i],axis=1))[:8]]
        if len(nearest):target[i]+=0.22*np.mean(raw[nearest],axis=0)
    changed|=np.linalg.norm(target-V,axis=1)>1e-12
    return target,changed,{"selected":True,"reason":"new interior convexity exceeds surrounding authored shell","core_vertices":int(len(core)),"fit_vertices":int(len(fit_ids)),"positive_excess_vertices":positive,"tolerance_mm":float(tolerance*1000.0),"residual_p95_mm":float(np.percentile(residual,95)*1000.0),"max_requested_mm":float(np.max(requested)*1000.0 if len(requested) else 0.0)}


def _restore_local_edges(reference: np.ndarray, target: np.ndarray, faces: np.ndarray, changed: np.ndarray, *, iterations: int=80, relaxation: float=.65, target_pull: float=.035) -> tuple[np.ndarray,np.ndarray]:
    R=np.asarray(reference,dtype=np.float64);goal=np.asarray(target,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    allowed=np.asarray(changed,dtype=bool).copy();nbr,_=_topology_neighbours(len(R),F)
    for i in np.flatnonzero(changed):
        for j in nbr[int(i)]:allowed[j]=True
    edges=_unique_edges(F);edges=edges[allowed[edges[:,0]]&allowed[edges[:,1]]]
    if not len(edges):return goal.copy(),allowed
    rest=np.linalg.norm(R[edges[:,1]]-R[edges[:,0]],axis=1);U=goal.copy()
    for _ in range(max(0,int(iterations))):
        vec=U[edges[:,1]]-U[edges[:,0]];cur=np.linalg.norm(vec,axis=1);direction=vec/np.maximum(cur[:,None],_EPS);corr=.5*(cur-rest)[:,None]*direction
        accum=np.zeros_like(U);count=np.zeros(len(U),dtype=np.float64)
        for side,sign in ((0,1.0),(1,-1.0)):
            ids=edges[:,side];valid=allowed[ids];np.add.at(accum,ids[valid],sign*corr[valid]);np.add.at(count,ids[valid],1.0)
        U+=float(relaxation)*accum/np.maximum(count[:,None],1.0);U[allowed]+=float(target_pull)*(goal[allowed]-U[allowed]);U[~allowed]=R[~allowed]
    return U,allowed


def _bridge_new_local_curvature(source_vertices: np.ndarray, faces: np.ndarray, fitted_vertices: np.ndarray, source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, *, nearest_surface_fn: Callable[...,Any], config: LocalRetargetConfig) -> tuple[np.ndarray,dict[str,Any]]:
    """Remove only small new interior convexities that the surrounding fitted shell disproves.

    This is deliberately post-B14 and control-free.  The untouched source garment establishes whether
    a high-frequency deformation is new; the surrounding B14 shell establishes the local surface the
    patch should bridge.  No authored comparison/control garment participates.
    """
    config=config or LocalRetargetConfig()
    P=np.asarray(source_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);base=np.asarray(fitted_vertices,dtype=np.float64);out=base.copy()
    labels=_component_labels(len(P),F);nbr,boundary=_topology_neighbours(len(P),F);hops=_boundary_hops(nbr,boundary);N=_vertex_normals(out,F)
    source_clear=_signed_clearance(nearest_surface_fn,P,source_body_triangles);target_result=nearest_surface_fn(out,target_body_triangles,k=24);target_normals=np.asarray(target_result[1],dtype=np.float64);target_clear=np.asarray(target_result[2],dtype=np.float64)
    reports=[];changed_total=np.zeros(len(P),dtype=bool)
    for component_index in sorted(np.unique(labels).tolist()):
        ids=np.flatnonzero(labels==component_index);lf=_local_faces(F,ids,len(P));features=_shape_features(P[ids],lf)
        row={"component_index":int(component_index),"vertex_count":int(len(ids)),"features":features,"selected":False}
        # Only broad, substantial shells may enter this lane. Narrow/open structures are handled by
        # the ribbon path and rigid/small components remain B14 fixed points.
        if len(ids)<300 or float(features["max_extent"])<.10 or float(features["middle_ratio"])<.45:
            row["reason"]="not a broad supported shell";reports.append(row);continue
        comp_source_clear=np.abs(source_clear[ids]);row["source_clearance_median_mm"]=float(np.median(comp_source_clear)*1000.0)
        if float(np.median(comp_source_clear))>.008:
            row["reason"]="shell is not closely body-supported";reports.append(row);continue
        displacement=out-P;smoothed=_smooth_displacement(displacement,nbr,ids,iterations=4,alpha=.5);normal_residual=np.abs(np.einsum("ij,ij->i",displacement-smoothed,N))
        clearance_error=np.abs(target_clear-source_clear);seed_mask=np.zeros(len(P),dtype=bool)
        seed_mask[ids]=(hops[ids]>=3)&(normal_residual[ids]>=.00125)&(clearance_error[ids]>=.0022)
        seeds=np.flatnonzero(seed_mask);row["strong_seed_vertices"]=int(len(seeds))
        if not len(seeds):
            row["reason"]="no new interior high-frequency deformation";reports.append(row);continue
        groups=_cluster_points(seeds,out,radius_m=.035);component_mask=labels==component_index;component_candidate=out.copy();component_changed=np.zeros(len(P),dtype=bool);group_reports=[]
        for group in groups:
            candidate,changed,group_report=_quadratic_bridge_target(component_candidate,group,nbr,component_mask,hops,target_normals,pull=.62,max_move_m=.0032)
            if bool(group_report.get("selected")):
                component_candidate[changed]=candidate[changed];component_changed|=changed
            group_reports.append(group_report)
        if not np.any(component_changed):
            row["reason"]="candidate seeds were explained by surrounding authored shell";row["groups"]=group_reports;reports.append(row);continue
        restored,allowed=_restore_local_edges(out,component_candidate,F,component_changed,iterations=80,relaxation=.65,target_pull=.035)
        move=np.linalg.norm(restored-out,axis=1);edge_p95=_edge_strain_p95(out[ids],restored[ids],lf);before_pen=int(np.count_nonzero(target_clear[ids]<-1e-9));after_clear=_signed_clearance(nearest_surface_fn,restored[ids],target_body_triangles);after_pen=int(np.count_nonzero(after_clear<-1e-9))
        row.update({"groups":group_reports,"candidate_vertices":int(np.count_nonzero(component_changed)),"affected_vertices":int(np.count_nonzero(move>1e-12)),"displacement_p95_mm":float(np.percentile(move[move>1e-12],95)*1000.0) if np.any(move>1e-12) else 0.0,"displacement_max_mm":float(np.max(move)*1000.0) if len(move) else 0.0,"edge_strain_p95_percent":float(edge_p95*100.0),"penetrating_vertices_before":before_pen,"penetrating_vertices_after":after_pen})
        if edge_p95>.07:row["reason"]="candidate exceeded B14 local edge-strain budget";reports.append(row);continue
        if after_pen>before_pen:row["reason"]="candidate introduced new target-body penetration";reports.append(row);continue
        if not _topology_ok(out[ids],restored[ids],lf,.75):row["reason"]="candidate violated B14 local topology";reports.append(row);continue
        out[allowed]=restored[allowed];changed_total|=allowed;row["selected"]=True;row["reason"]="surrounding B14 shell proves a new local convex artifact; bridged with B14 edge preservation";reports.append(row)
        # Geometry changed, so subsequent components must use refreshed normals/target contact.
        N=_vertex_normals(out,F);target_result=nearest_surface_fn(out,target_body_triangles,k=24);target_normals=np.asarray(target_result[1],dtype=np.float64);target_clear=np.asarray(target_result[2],dtype=np.float64)
    return out,{"policy":"control-free local shell bridge: source proves new detail, surrounding B14 shell proves correction","changed_vertex_count":int(np.count_nonzero(np.linalg.norm(out-base,axis=1)>1e-12)),"components":reports}



def _connected_mask_components(mask: np.ndarray, nbr: list[set[int]]) -> list[np.ndarray]:
    mask=np.asarray(mask,dtype=bool);seen=np.zeros(len(mask),dtype=bool);out=[]
    for start in np.flatnonzero(mask):
        if seen[start]:continue
        stack=[int(start)];seen[start]=True;ids=[]
        while stack:
            u=stack.pop();ids.append(u)
            for v in nbr[u]:
                if mask[v] and not seen[v]:seen[v]=True;stack.append(v)
        out.append(np.asarray(ids,dtype=np.int64))
    return sorted(out,key=len,reverse=True)


def _smooth_component_displacement(displacement: np.ndarray, nbr: list[set[int]], ids: np.ndarray, iterations: int=4, alpha: float=.35) -> np.ndarray:
    D=np.asarray(displacement,dtype=np.float64);out=D.copy();active=np.zeros(len(D),dtype=bool);active[np.asarray(ids,dtype=np.int64)]=True
    for _ in range(max(0,int(iterations))):
        nxt=out.copy()
        for i in np.asarray(ids,dtype=np.int64):
            local=[j for j in nbr[int(i)] if active[j]]
            if local:nxt[i]=(1.0-alpha)*out[i]+alpha*np.mean(out[local],axis=0)
        out=nxt
    return out


def _minimal_literal_collision_fix(vertices: np.ndarray, faces: np.ndarray, active_mask: np.ndarray, target_triangles: np.ndarray, *, nearest_surface_fn: Callable[...,Any], margin_m: float=.00015, max_push_m: float=.00075, passes: int=2) -> np.ndarray:
    U=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64);active=np.asarray(active_mask,dtype=bool);nbr,_=_topology_neighbours(len(U),F);ids=np.flatnonzero(active)
    for _ in range(max(0,int(passes))):
        cp,nrm,signed,_,_=nearest_surface_fn(U[ids],target_triangles,k=24);signed=np.asarray(signed,dtype=np.float64);nrm=np.asarray(nrm,dtype=np.float64);need=np.flatnonzero(signed<float(margin_m))
        if not len(need):break
        corr=np.zeros_like(U)
        for local in need:
            gi=int(ids[local]);amount=min(float(margin_m-signed[local]),float(max_push_m));vec=amount*nrm[local];corr[gi]+=vec
            for nb in nbr[gi]:
                if active[nb]:corr[nb]+=0.15*vec
        U+=corr
    return U


def _fit_broad_supported_shells_to_body(source_vertices: np.ndarray, faces: np.ndarray, fitted_vertices: np.ndarray, correspondence_vertices: np.ndarray, source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, *, nearest_surface_fn: Callable[...,Any]) -> tuple[np.ndarray,dict[str,Any]]:
    """Body-anchored whole-shell fitting for genuinely close broad garment regions.

    The body correspondence supplies the destination and source clearance supplies the tightness.
    Only the low-frequency correspondence field is used, so local anatomical relief cannot emboss
    itself into the garment.  The solve smooths *displacement*, not absolute garment positions:
    neighbouring vertices therefore move coherently while the authored garment construction remains.
    YAB/control meshes never participate.
    """
    P=np.asarray(source_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);base=np.asarray(fitted_vertices,dtype=np.float64);C=np.asarray(correspondence_vertices,dtype=np.float64);out=base.copy()
    labels=_component_labels(len(P),F);nbr,boundary=_topology_neighbours(len(P),F);hops=_boundary_hops(nbr,boundary);source_clear=_signed_clearance(nearest_surface_fn,P,source_body_triangles);reports=[]
    all_edges=_unique_edges(F)
    for component_index in sorted(np.unique(labels).tolist()):
        ids=np.flatnonzero(labels==component_index);lf=_local_faces(F,ids,len(P));features=_shape_features(P[ids],lf);row={"component_index":int(component_index),"vertex_count":int(len(ids)),"features":features,"selected":False}
        if len(ids)<300 or float(features["max_extent"])<.10 or float(features["middle_ratio"])<.45:
            row["reason"]="not a broad supported shell";reports.append(row);continue
        source_abs=np.abs(source_clear[ids]);row["source_clearance_median_mm"]=float(np.median(source_abs)*1000.0)
        if float(np.median(source_abs))>.008:
            row["reason"]="broad shell is not source-tight";reports.append(row);continue
        # Deep interior topology defines independently supported domains.  The immediate authored
        # boundary stays fixed; one feather ring is allowed so deformation cannot hinge at the seam.
        deep=(labels==component_index)&(hops>=2);domains=[d for d in _connected_mask_components(deep,nbr) if len(d)>=max(80,int(len(ids)*.08))]
        if not domains:
            row["reason"]="no substantial supported interior domain";reports.append(row);continue
        low=_smooth_component_displacement(C-P,nbr,ids,iterations=4,alpha=.35);target=P+low;component_candidate=out.copy();domain_reports=[];selected_any=False
        occupied=np.zeros(len(P),dtype=bool)
        for domain in domains:
            active=np.zeros(len(P),dtype=bool);active[domain]=True
            for u in domain:
                for v in nbr[int(u)]:
                    if labels[v]==component_index and hops[v]>=1:active[v]=True
            # Never let neighbouring inferred domains fight over the same feather vertices.
            active&=~occupied;occupied|=active;did=np.flatnonzero(active)
            if len(did)<80:continue
            local_index=np.full(len(P),-1,dtype=np.int64);local_index[did]=np.arange(len(did),dtype=np.int64);clear=np.abs(source_clear[did]);tight=np.exp(-np.square(clear/.006));seam=np.clip((hops[did]-.5)/3.5,0.0,1.0);fit_weight=tight*seam
            fit_scale=20.0;smooth_lambda=60.0;zero_reg=.10;diag=fit_scale*fit_weight+zero_reg;rows=[];cols=[];data=[]
            for a,b in all_edges:
                if labels[a]!=component_index or labels[b]!=component_index:continue
                ia=int(local_index[a]);ib=int(local_index[b])
                if ia>=0 and ib>=0:
                    diag[ia]+=smooth_lambda;diag[ib]+=smooth_lambda;rows.extend((ia,ib));cols.extend((ib,ia));data.extend((-smooth_lambda,-smooth_lambda))
                elif ia>=0:diag[ia]+=smooth_lambda
                elif ib>=0:diag[ib]+=smooth_lambda
            m=len(did);rows.extend(range(m));cols.extend(range(m));data.extend(diag.tolist());K=coo_matrix((data,(rows,cols)),shape=(m,m)).tocsr();desired=target[did]-base[did];rhs=(fit_scale*fit_weight)[:,None]*desired;delta=np.column_stack([spsolve(K,rhs[:,axis]) for axis in range(3)]);mag=np.linalg.norm(delta,axis=1);delta*=np.minimum(1.0,.014/np.maximum(mag,_EPS))[:,None];trial=component_candidate.copy();trial[did]=base[did]+delta
            trial=_minimal_literal_collision_fix(trial,F,active,target_body_triangles,nearest_surface_fn=nearest_surface_fn,margin_m=.00015,max_push_m=.00075,passes=2)
            before_clear=_signed_clearance(nearest_surface_fn,base[did],target_body_triangles);after_clear=_signed_clearance(nearest_surface_fn,trial[did],target_body_triangles);before_error=np.abs(before_clear-source_clear[did]);after_error=np.abs(after_clear-source_clear[did]);before_p95=float(np.percentile(before_error,95));after_p95=float(np.percentile(after_error,95));local_faces=_local_faces(F,did,len(P));edge_p95=_edge_strain_p95(base[did],trial[did],local_faces) if len(local_faces) else 0.0;move=np.linalg.norm(trial[did]-base[did],axis=1)
            drow={"vertices":int(len(did)),"deep_vertices":int(len(domain)),"source_clearance_median_mm":float(np.median(clear)*1000.0),"before_clearance_error_p95_mm":before_p95*1000.0,"after_clearance_error_p95_mm":after_p95*1000.0,"penetrating_vertices_after":int(np.count_nonzero(after_clear<-1e-9)),"displacement_p50_mm":float(np.percentile(move,50)*1000.0),"displacement_p95_mm":float(np.percentile(move,95)*1000.0),"displacement_max_mm":float(np.max(move)*1000.0),"edge_strain_p95_percent":float(edge_p95*100.0),"selected":False}
            if after_p95>=before_p95*.95:drow["reason"]="body-fit did not materially improve source clearance";domain_reports.append(drow);continue
            if np.any(after_clear<-.00001):drow["reason"]="body-fit retained target-body penetration";domain_reports.append(drow);continue
            if not _topology_ok(base[did],trial[did],local_faces,.40):drow["reason"]="body-fit violated local topology";domain_reports.append(drow);continue
            component_candidate[did]=trial[did];drow["selected"]=True;drow["reason"]="source-tight shell follows low-frequency body correspondence with smooth displacement";domain_reports.append(drow);selected_any=True
        row["domains"]=domain_reports
        if selected_any:
            out=component_candidate;row["selected"]=True;row["reason"]="broad source-tight shell refit to target macro body form"
        else:row["reason"]="no body-guided domain passed safety/evidence gates"
        reports.append(row)
    return out,{"policy":"source-tightness drives body following; low-frequency body correspondence drives shape; seams remain authored boundaries","changed_vertex_count":int(np.count_nonzero(np.linalg.norm(out-base,axis=1)>1e-12)),"components":reports}



def _source_attachment_relations(vertices: np.ndarray, faces: np.ndarray, cfg: LocalRetargetConfig) -> tuple[np.ndarray,list[dict[str,Any]],dict[int,dict[str,Any]]]:
    """Infer only source-proven cross-component attachment relations.

    Proximity by itself is not enough: a relation needs repeated close witnesses, except for a tiny
    component that is essentially touching its parent.  This keeps nearby bilateral pieces and
    overlapping garment layers independent while still recognising disconnected authored hardware.
    """
    P=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);labels=_component_labels(len(P),F)
    components={}
    for component_index in np.unique(labels):
        component_index=int(component_index);ids=np.flatnonzero(labels==component_index);lf=_local_faces(F,ids,len(P));features=_shape_features(P[ids],lf);extent=float(np.linalg.norm(np.ptp(P[ids],axis=0))) if len(ids) else 0.0
        components[component_index]={"ids":ids,"vertex_count":int(len(ids)),"extent_m":extent,"features":features,"ribbon":bool(_is_local_ribbon_candidate(features,len(ids),cfg))}
    if len(components)<=1 or len(P)<=1:return labels,[],components
    tree=cKDTree(P);k=min(16,len(P));distances,neighbours=tree.query(P,k=k)
    if k==1:distances=distances[:,None];neighbours=neighbours[:,None]
    candidates={}
    for i in range(len(P)):
        for d,j in zip(np.asarray(distances[i]).reshape(-1)[1:],np.asarray(neighbours[i]).reshape(-1)[1:]):
            j=int(j);d=float(d)
            if d>float(cfg.attachment_search_radius_m):break
            if i==j or labels[i]==labels[j]:continue
            ca,cb=int(labels[i]),int(labels[j]);pair=(ca,cb) if ca<cb else (cb,ca);a,b=(i,j) if i<j else (j,i)
            best=candidates.setdefault(pair,{})
            best[(a,b)]=min(float(best.get((a,b),float('inf'))),d)
    relations=[]
    for pair,best in candidates.items():
        ordered=sorted(((a,b,d) for (a,b),d in best.items()),key=lambda row:row[2]);confirmed=[row for row in ordered if row[2]<=float(cfg.attachment_confirm_radius_m)]
        min_distance=float(ordered[0][2]) if ordered else float('inf');small=min(components[pair[0]]["extent_m"],components[pair[1]]["extent_m"])<=float(cfg.attachment_tiny_extent_m)
        accepted=confirmed if len(confirmed)>=int(cfg.attachment_min_witnesses) else []
        tiny_touch=False
        if not accepted and small and min_distance<=float(cfg.attachment_tiny_radius_m):
            accepted=[row for row in ordered if row[2]<=float(cfg.attachment_tiny_radius_m)];tiny_touch=bool(accepted)
        if len(accepted)<min(int(cfg.attachment_min_witnesses),3) and not tiny_touch:continue
        # A handful of the closest witnesses is enough to define attachment without creating a dense zipper.
        accepted=accepted[:max(int(cfg.attachment_min_witnesses),12)]
        relations.append({"components":pair,"pairs":accepted,"witness_count":int(len(accepted)),"min_distance_m":min_distance,"median_distance_m":float(np.median([row[2] for row in accepted]))})
    return labels,relations,components


def _relation_side_witnesses(relation: dict[str,Any], labels: np.ndarray, component_index: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    own=[];foreign=[];distance=[]
    for a,b,d in relation.get("pairs",[]):
        if int(labels[a])==int(component_index):own.append(int(a));foreign.append(int(b));distance.append(float(d))
        elif int(labels[b])==int(component_index):own.append(int(b));foreign.append(int(a));distance.append(float(d))
    return np.asarray(own,dtype=np.int64),np.asarray(foreign,dtype=np.int64),np.asarray(distance,dtype=np.float64)


def _bind_ribbon_endpoints_to_assembly(source_vertices: np.ndarray, faces: np.ndarray, assembly_vertices: np.ndarray, ribbon_vertices: np.ndarray, component_index: int, labels: np.ndarray, relations: list[dict[str,Any]], cfg: LocalRetargetConfig) -> tuple[np.ndarray,list[dict[str,Any]]]:
    P=np.asarray(source_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);A=np.asarray(assembly_vertices,dtype=np.float64);ids=np.flatnonzero(labels==int(component_index));lf=_local_faces(F,ids,len(P));tcoord=_geodesic_coordinate(P[ids],lf);ribbon=np.asarray(ribbon_vertices,dtype=np.float64).copy();end_rows=[];deltas=[];pin_local=[];pin_targets=[]
    local_lookup=np.full(len(P),-1,dtype=np.int64);local_lookup[ids]=np.arange(len(ids),dtype=np.int64)
    component_relations=[row for row in relations if int(component_index) in row.get("components",())]
    for end_index,end_mask in ((0,tcoord<=.12),(1,tcoord>=.88)):
        e_local=np.flatnonzero(end_mask);eids=ids[e_local]
        if not len(eids):deltas.append(np.zeros(3,dtype=np.float64));end_rows.append({"end":end_index,"bound":False,"reason":"empty endpoint neighbourhood"});continue
        candidates=[]
        endpoint_set=set(int(x) for x in eids.tolist())
        for relation in component_relations:
            own,foreign,distance=_relation_side_witnesses(relation,labels,component_index)
            keep=np.asarray([int(x) in endpoint_set for x in own],dtype=bool)
            if int(np.count_nonzero(keep))<2:continue
            own=own[keep];foreign=foreign[keep];distance=distance[keep];foreign_components=np.unique(labels[foreign])
            if len(foreign_components)!=1:continue
            candidates.append((int(len(own)),float(np.median(distance)),int(foreign_components[0]),own,foreign))
        if not candidates:
            deltas.append(np.zeros(3,dtype=np.float64));end_rows.append({"end":end_index,"bound":False,"reason":"no source-proven endpoint attachment"});continue
        candidates.sort(key=lambda row:(-row[0],row[1]));witness_count,median_distance,attached_component,own,foreign=candidates[0]
        anchor_ids=np.flatnonzero(labels==attached_component);endpoint_center=np.mean(P[own],axis=0);local_distance=np.linalg.norm(P[anchor_ids]-endpoint_center[None,:],axis=1);local_anchor=anchor_ids[local_distance<=float(cfg.attachment_local_frame_radius_m)]
        if len(local_anchor)<6:local_anchor=anchor_ids
        R,tr=_kabsch_rigid_transform(P[local_anchor],A[local_anchor])
        # Preserve the *actual source witness vector* relative to the final parent witness rather than
        # only matching endpoint centroids. This is what makes a strap remain through/against its loop.
        desired_witness=A[foreign]+(P[own]-P[foreign])@R.T
        own_local=local_lookup[own]
        valid=own_local>=0
        own_local=own_local[valid];desired_witness=desired_witness[valid]
        if len(own_local):
            pin_local.extend(int(x) for x in own_local.tolist());pin_targets.extend(np.asarray(desired_witness,dtype=np.float64).tolist())
            delta=np.mean(desired_witness,axis=0)-np.mean(ribbon[own_local],axis=0)
            before_gap=float(np.median(np.linalg.norm(ribbon[own_local]-A[foreign[valid]],axis=1)))
            after_gap=float(np.median(np.linalg.norm(desired_witness-A[foreign[valid]],axis=1)))
        else:
            desired=P[eids]@R.T+tr;delta=np.mean(desired,axis=0)-np.mean(ribbon[e_local],axis=0);before_gap=float('nan');after_gap=float('nan')
        deltas.append(delta);end_rows.append({"end":end_index,"bound":True,"attached_component":attached_component,"witnesses":int(witness_count),"source_gap_mm":median_distance*1000.0,"frame_vertices":int(len(local_anchor)),"endpoint_delta_mm":[float(x*1000.0) for x in delta],"witness_gap_before_mm":before_gap*1000.0,"pinned_source_gap_mm":after_gap*1000.0,"hard_pinned_witnesses":int(len(own_local))})
    if len(deltas)==2:
        s=np.clip(tcoord,0.0,1.0);s=s*s*(3.0-2.0*s);goal=ribbon+(1.0-s[:,None])*deltas[0][None,:]+s[:,None]*deltas[1][None,:]
        project_cfg=LocalRetargetConfig(**{**cfg.__dict__,"edge_iterations":max(int(cfg.edge_iterations),100),"edge_relaxation":min(float(cfg.edge_relaxation),.58),"macro_pull":max(float(cfg.macro_pull),.012)})
        ribbon=_edge_project(ribbon,goal,lf,project_cfg)
        if pin_local:
            ribbon=_edge_project_with_pins(ribbon,ribbon,lf,np.asarray(pin_local,dtype=np.int64),np.asarray(pin_targets,dtype=np.float64),iterations=int(cfg.attachment_pin_iterations),relaxation=float(cfg.attachment_pin_relaxation),goal_pull=float(cfg.attachment_pin_goal_pull))
    return ribbon,end_rows


def _rebind_compact_attachment_children(source_vertices: np.ndarray, faces: np.ndarray, assembly_vertices: np.ndarray, labels: np.ndarray, relations: list[dict[str,Any]], components: dict[int,dict[str,Any]], cfg: LocalRetargetConfig, *, parent_mode: str) -> tuple[np.ndarray,list[dict[str,Any]]]:
    P=np.asarray(source_vertices,dtype=np.float64);out=np.asarray(assembly_vertices,dtype=np.float64).copy();reports=[]
    candidate_by_child={}
    stable_children=set()
    for relation in relations:
        a,b=map(int,relation["components"]);ca,cb=components[a],components[b]
        # Ribbons have their own endpoint-aware deformable lane; this pass is only for compact attached hardware/trim.
        if ca["ribbon"] and cb["ribbon"]:continue
        # Infer the compact child from physical extent first, with vertex count only as supporting evidence.
        if ca["extent_m"]<cb["extent_m"]:child,parent=a,b
        elif cb["extent_m"]<ca["extent_m"]:child,parent=b,a
        else:child,parent=(a,b) if ca["vertex_count"]<=cb["vertex_count"] else (b,a)
        cc,pc=components[child],components[parent]
        if cc["ribbon"]:continue
        compact=cc["extent_m"]<=float(cfg.attachment_compact_extent_m) or cc["extent_m"]<=float(cfg.attachment_compact_relative_extent)*max(pc["extent_m"],_EPS) or cc["vertex_count"]<=float(cfg.attachment_compact_relative_vertices)*max(pc["vertex_count"],1)
        if parent_mode=="ribbon":
            # A deformable ribbon may carry compact closed hardware (rings/sliders/buckles), but it
            # must never become authority for an open shell/panel merely because that panel is short.
            closed_hardware=float(cc["features"].get("boundary_fraction",1.0))<=.25 and float(cc["features"].get("max_extent",1.0))<=.020
            compact=compact and (cc["extent_m"]<=float(cfg.attachment_tiny_extent_m) or (cc["extent_m"]<=.024 and (cc["vertex_count"]<=8 or closed_hardware)))
        if not compact:continue
        if not pc["ribbon"]:stable_children.add(child)
        candidate_by_child.setdefault(child,[]).append((parent,relation))
    for child in sorted(candidate_by_child,key=lambda c:components[c]["extent_m"]):
        options=[]
        for parent,relation in candidate_by_child[child]:
            parent_is_ribbon=bool(components[parent]["ribbon"])
            if parent_mode=="stable" and parent_is_ribbon:continue
            if parent_mode=="ribbon" and (not parent_is_ribbon or child in stable_children):continue
            options.append((parent,relation))
        if not options:continue
        options.sort(key=lambda item:(-int(item[1].get("witness_count",0)),-float(components[item[0]]["extent_m"]),float(item[1].get("median_distance_m",0.0))))
        parent,relation=options[0];child_w,parent_w,_=_relation_side_witnesses(relation,labels,child)
        if not len(child_w):continue
        parent_ids=np.flatnonzero(labels==parent);parent_center=np.mean(P[parent_w],axis=0);local_distance=np.linalg.norm(P[parent_ids]-parent_center[None,:],axis=1);local_anchor=parent_ids[local_distance<=float(cfg.attachment_local_frame_radius_m)]
        if len(local_anchor)<6:local_anchor=parent_ids
        Rparent,tparent=_kabsch_rigid_transform(P[local_anchor],out[local_anchor])
        # Preserve the source-proven child/parent witness vectors against the *actual final* parent
        # witnesses. Compact hardware is source-authoritative rigid geometry, so rebuild its rigid
        # frame from the untouched source rather than carrying any solve-time distortion forward.
        desired=out[parent_w]+(P[child_w]-P[parent_w])@Rparent.T
        before_error=float(np.median(np.linalg.norm(out[child_w]-desired,axis=1)))
        Rchild,tchild=_kabsch_rigid_transform(P[child_w],desired);child_ids=np.flatnonzero(labels==child);trial=P[child_ids]@Rchild.T+tchild;movement=np.linalg.norm(trial-out[child_ids],axis=1);move_p95=float(np.percentile(movement,95)) if len(movement) else 0.0
        if move_p95>float(cfg.attachment_max_rebind_m):
            reports.append({"child_component":int(child),"parent_component":int(parent),"selected":False,"reason":"required compact attachment rebind exceeded safety cap","move_p95_mm":move_p95*1000.0});continue
        trial_w=P[child_w]@Rchild.T+tchild;after_error=float(np.median(np.linalg.norm(trial_w-desired,axis=1)))
        if after_error>before_error+1e-9:
            reports.append({"child_component":int(child),"parent_component":int(parent),"selected":False,"reason":"rebind did not improve source-proven attachment frame"});continue
        source_gap=float(np.median(np.linalg.norm(P[child_w]-P[parent_w],axis=1)));final_gap=float(np.median(np.linalg.norm(trial_w-out[parent_w],axis=1)))
        out[child_ids]=trial;reports.append({"child_component":int(child),"parent_component":int(parent),"selected":True,"parent_mode":parent_mode,"witnesses":int(len(child_w)),"before_frame_error_mm":before_error*1000.0,"after_frame_error_mm":after_error*1000.0,"move_p95_mm":move_p95*1000.0,"source_witness_gap_mm":source_gap*1000.0,"final_witness_gap_mm":final_gap*1000.0,"source_rigid_shape_restored":True})
    return out,reports


def preserve_source_proven_attachment_continuity(source_vertices: np.ndarray, faces: np.ndarray, assembly_vertices: np.ndarray, *, config: LocalRetargetConfig | None=None) -> tuple[np.ndarray,dict[str,Any]]:
    """Final authored-attachment closure after every other garment/layer reconciliation.

    This pass does not refit the garment or consult any control mesh.  It only restores relationships
    that are explicitly proven by the untouched source: compact disconnected children follow the
    final local frame of their stable parent, then narrow ribbons bind their ends to those now-final
    components, then compact hardware whose only parent is a ribbon follows that ribbon.
    """
    cfg=config or LocalRetargetConfig();P=np.asarray(source_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);A=np.asarray(assembly_vertices,dtype=np.float64)
    if P.shape!=A.shape:raise ValueError("source and assembly vertices must have identical shape")
    labels,relations,components=_source_attachment_relations(P,F,cfg)
    if not relations:return A.copy(),{"enabled":False,"reason":"no source-proven disconnected attachments","relation_count":0}
    out,stable_reports=_rebind_compact_attachment_children(P,F,A,labels,relations,components,cfg,parent_mode="stable")
    ribbon_reports=[]
    for component_index in sorted(components):
        info=components[component_index]
        if not info["ribbon"]:continue
        ids=np.asarray(info["ids"],dtype=np.int64);bound,end_rows=_bind_ribbon_endpoints_to_assembly(P,F,out,out[ids],component_index,labels,relations,cfg)
        if any(bool(row.get("bound")) for row in end_rows):out[ids]=bound;ribbon_reports.append({"component_index":int(component_index),"ends":end_rows})
    out,ribbon_child_reports=_rebind_compact_attachment_children(P,F,out,labels,relations,components,cfg,parent_mode="ribbon")
    move=np.linalg.norm(out-A,axis=1)
    return out,{"enabled":True,"policy":"source-proven final attachment closure after garment/layer reconciliation","relation_count":int(len(relations)),"compact_stable_parent":stable_reports,"ribbons":ribbon_reports,"compact_ribbon_parent":ribbon_child_reports,"changed_vertex_count":int(np.count_nonzero(move>1e-12)),"displacement_p95_mm":float(np.percentile(move,95)*1000.0) if len(move) else 0.0,"displacement_max_mm":float(np.max(move,initial=0.0)*1000.0)}

def _kabsch_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    A=np.asarray(source,dtype=np.float64);B=np.asarray(target,dtype=np.float64)
    if len(A)<3 or A.shape!=B.shape:
        return np.eye(3,dtype=np.float64),np.mean(B-A,axis=0) if len(A) else np.zeros(3,dtype=np.float64)
    ca=np.mean(A,axis=0);cb=np.mean(B,axis=0);H=(A-ca).T@(B-cb);u,_,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0.0:vt[-1]*=-1.0;R=vt.T@u.T
    return R,cb-R@ca


def _retarget_ribbon_components_only(source_vertices: np.ndarray, faces: np.ndarray, b14_vertices: np.ndarray, correspondence_vertices: np.ndarray, source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, *, nearest_surface_fn: Callable[...,Any], collision_polish_fn: Callable[...,Any], config: LocalRetargetConfig) -> tuple[np.ndarray,list[dict[str,Any]]]:
    cfg=config;P=np.asarray(source_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);B=np.asarray(b14_vertices,dtype=np.float64);C=np.asarray(correspondence_vertices,dtype=np.float64);out=B.copy();labels=_component_labels(len(P),F);reports=[]
    for component_index in sorted(np.unique(labels).tolist()):
        ids=np.flatnonzero(labels==component_index);lf=_local_faces(F,ids,len(P));features=_shape_features(P[ids],lf);row={"component_index":int(component_index),"vertex_count":int(len(ids)),"features":features,"selected":False}
        if not _is_local_ribbon_candidate(features,len(ids),cfg):row["reason"]="not a narrow open local structure";reports.append(row);continue
        macro_delta=np.linalg.norm(C[ids]-B[ids],axis=1);row["b14_to_correspondence_p95_mm"]=float(np.percentile(macro_delta,95)*1000.0)
        source_clear=_signed_clearance(nearest_surface_fn,P[ids],source_body_triangles);b14_clear=_signed_clearance(nearest_surface_fn,B[ids],target_body_triangles);corr_clear=_signed_clearance(nearest_surface_fn,C[ids],target_body_triangles)
        b14_error=np.abs(b14_clear-source_clear);corr_error=np.abs(corr_clear-source_clear);b14_p95=float(np.percentile(b14_error,95));corr_p95=float(np.percentile(corr_error,95));ratio=corr_p95/max(b14_p95,_EPS)
        row.update({"b14_clearance_error_p95_mm":b14_p95*1000.0,"correspondence_clearance_error_p95_mm":corr_p95*1000.0,"clearance_error_ratio":ratio,"b14_penetrating_vertices":int(np.count_nonzero(b14_clear<0.0)),"correspondence_penetrating_vertices":int(np.count_nonzero(corr_clear<0.0))})
        if b14_p95<cfg.min_b14_clearance_p95_error_m:row["reason"]="B14 relationship already within local tolerance";reports.append(row);continue
        if ratio>cfg.required_clearance_p95_ratio:row["reason"]="correspondence does not materially improve source relationship";reports.append(row);continue
        low=_low_frequency_displacement(P[ids],lf,C[ids]-B[ids],cfg.displacement_bins);goal=B[ids]+low;candidate=_edge_project(B[ids],goal,lf,cfg)
        candidate,collision_report=collision_polish_fn(candidate,lf,target_body_triangles,margin=cfg.collision_margin_m,iterations=cfg.collision_iterations,blend=cfg.collision_blend,max_push=cfg.collision_max_push_m)
        candidate=np.asarray(candidate,dtype=np.float64);edge_p95=_edge_strain_p95(B[ids],candidate,lf);final_clear=_signed_clearance(nearest_surface_fn,candidate,target_body_triangles);final_error=np.abs(final_clear-source_clear);final_p95=float(np.percentile(final_error,95));row.update({"candidate_clearance_error_p95_mm":final_p95*1000.0,"candidate_penetrating_vertices":int(np.count_nonzero(final_clear<-1e-9)),"edge_strain_p95_percent":edge_p95*100.0,"collision":collision_report})
        if final_p95>=b14_p95:row["reason"]="candidate did not improve B14 source relationship";reports.append(row);continue
        if edge_p95>cfg.max_edge_strain_p95:row["reason"]="candidate exceeded B14 local edge-strain budget";reports.append(row);continue
        if not _topology_ok(B[ids],candidate,lf,cfg.min_area_ratio):row["reason"]="candidate violated B14 local topology";reports.append(row);continue
        out[ids]=candidate;row["selected"]=True;row["reason"]="strong local evidence; B14 detail preserved with bounded macro correction";row["moved_vertices"]=int(np.count_nonzero(np.linalg.norm(candidate-B[ids],axis=1)>1e-12));row["displacement_p95_mm"]=float(np.percentile(np.linalg.norm(candidate-B[ids],axis=1),95)*1000.0);reports.append(row)
    return out,reports


def retarget_attached_ribbons_to_assembly(source_vertices: np.ndarray, faces: np.ndarray, b14_vertices: np.ndarray, correspondence_vertices: np.ndarray, assembly_vertices: np.ndarray, source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, *, nearest_surface_fn: Callable[...,Any], collision_polish_fn: Callable[...,Any], config: LocalRetargetConfig | None=None) -> tuple[np.ndarray,dict[str,Any]]:
    """Restore proven narrow-ribbon retargeting after a whole-garment solve, then bind its ends to source-proven attachment frames.

    This is garment-name agnostic.  The ribbon path comes only from source/B14/body correspondence.
    YAB/control geometry is never consulted.  Attachment endpoints are inferred from repeated close
    source witnesses and carried by the final solved assembly, preventing straps from kinking or
    drifting away from rings/cups after the rest of the garment changes.
    """
    cfg=config or LocalRetargetConfig();P=np.asarray(source_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);B=np.asarray(b14_vertices,dtype=np.float64);C=np.asarray(correspondence_vertices,dtype=np.float64);A=np.asarray(assembly_vertices,dtype=np.float64)
    if not (P.shape==B.shape==C.shape==A.shape):raise ValueError("source, B14, correspondence and assembly vertices must have identical shape")
    ribbon_base,reports=_retarget_ribbon_components_only(P,F,B,C,source_body_triangles,target_body_triangles,nearest_surface_fn=nearest_surface_fn,collision_polish_fn=collision_polish_fn,config=cfg)
    out=A.copy();labels,relations,_=_source_attachment_relations(P,F,cfg);selected=[]
    for row in reports:
        if not bool(row.get("selected")):continue
        component_index=int(row["component_index"]);ids=np.flatnonzero(labels==component_index);ribbon,end_rows=_bind_ribbon_endpoints_to_assembly(P,F,out,ribbon_base[ids],component_index,labels,relations,cfg)
        out[ids]=ribbon;row["attachment_binding"]=end_rows;selected.append(component_index)
    changed=np.linalg.norm(out-A,axis=1)>1e-12
    return out,{"policy":"proven narrow-ribbon retarget restored after whole-garment solve; endpoints bound to source-proven final assembly frames","selected_components":selected,"selected_component_count":len(selected),"changed_vertex_count":int(np.count_nonzero(changed)),"components":reports}

def retarget_b14_local_components(source_vertices: np.ndarray, faces: np.ndarray, b14_vertices: np.ndarray, correspondence_vertices: np.ndarray, source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, *, nearest_surface_fn: Callable[...,Any], collision_polish_fn: Callable[...,Any], config: LocalRetargetConfig | None=None) -> tuple[np.ndarray,dict[str,Any]]:
    """Return B14 geometry with only strongly evidenced local ribbon corrections applied."""
    cfg=config or LocalRetargetConfig();P=np.asarray(source_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);B=np.asarray(b14_vertices,dtype=np.float64);C=np.asarray(correspondence_vertices,dtype=np.float64)
    if P.shape!=B.shape or B.shape!=C.shape:raise ValueError("source, B14 and correspondence vertices must have identical shape")
    out=B.copy();labels=_component_labels(len(P),F);reports=[]
    out,reports=_retarget_ribbon_components_only(P,F,B,C,source_body_triangles,target_body_triangles,nearest_surface_fn=nearest_surface_fn,collision_polish_fn=collision_polish_fn,config=cfg)
    ribbon_changed=np.linalg.norm(out-B,axis=1)>1e-12
    body_fitted,body_fit_report=_fit_broad_supported_shells_to_body(P,F,out,C,source_body_triangles,target_body_triangles,nearest_surface_fn=nearest_surface_fn)
    out=np.asarray(body_fitted,dtype=np.float64)
    changed=np.linalg.norm(out-B,axis=1)>1e-12
    return out,{"policy":"Frozen B14 authority with evidence-gated ribbon retarget plus source-tightness body-anchored shell fitting","changed_vertex_count":int(np.count_nonzero(changed)),"unchanged_vertex_count":int(len(B)-np.count_nonzero(changed)),"component_count":int(len(np.unique(labels))),"selected_components":int(sum(bool(r.get("selected")) for r in reports)),"ribbon_changed_vertex_count":int(np.count_nonzero(ribbon_changed)),"components":reports,"body_guided_shell_fit":body_fit_report}
