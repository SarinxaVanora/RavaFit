from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

_EPS=1.0e-12


@dataclass(frozen=True)
class SurfaceRelativeDetailConfig:
    """Generic carrier-frame authority for compact disconnected garment details.

    A detail remains authored geometry, but its placement is inherited from the already-solved nearby
    garment carrier.  No item/material/body names are used.  This prevents compact ornaments from
    independently drifting while still allowing the actual carrier cloth/shell to refit the target.
    """
    min_detail_vertices: int=12
    max_detail_vertices: int=256
    max_detail_extent_m: float=0.050
    min_host_vertices: int=180
    min_host_extent_m: float=0.045
    max_source_median_clearance_m: float=0.016
    max_source_p90_clearance_m: float=0.025
    min_anchor_vertices: int=12
    anchor_vertices: int=72
    scale_min: float=0.72
    scale_max: float=1.38
    max_host_fit_rms_m: float=0.006
    min_layout_error_m: float=0.00020
    max_vertex_correction_m: float=0.020


class _UnionFind:
    def __init__(self,n:int):self.parent=np.arange(n,dtype=np.int64);self.rank=np.zeros(n,dtype=np.int8)
    def find(self,x:int)->int:
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=int(self.parent[x])
        return x
    def union(self,a:int,b:int)->None:
        ra=self.find(int(a));rb=self.find(int(b))
        if ra==rb:return
        if self.rank[ra]<self.rank[rb]:ra,rb=rb,ra
        self.parent[rb]=ra
        if self.rank[ra]==self.rank[rb]:self.rank[ra]+=1


def _components(vertex_count:int,faces:np.ndarray)->list[tuple[np.ndarray,np.ndarray]]:
    F=np.asarray(faces,dtype=np.int64)
    if vertex_count<=0 or F.size==0:return []
    uf=_UnionFind(vertex_count)
    for a,b,c in F.tolist():uf.union(a,b);uf.union(b,c);uf.union(c,a)
    used=np.unique(F.reshape(-1));by={}
    for vi in used.tolist():by.setdefault(uf.find(int(vi)),[]).append(int(vi))
    face_roots=np.asarray([uf.find(int(face[0])) for face in F],dtype=np.int64)
    return [(np.asarray(sorted(ids),dtype=np.int64),np.flatnonzero(face_roots==int(root)).astype(np.int64)) for root,ids in by.items()]


def _fit_similarity(X:np.ndarray,Y:np.ndarray,scale_min:float,scale_max:float)->tuple[np.ndarray,float,np.ndarray,float]:
    X=np.asarray(X,dtype=np.float64);Y=np.asarray(Y,dtype=np.float64)
    if len(X)<3 or X.shape!=Y.shape:return np.eye(3),1.0,np.zeros(3),float("inf")
    mx=X.mean(axis=0);my=Y.mean(axis=0);Xc=X-mx;Yc=Y-my;cov=Xc.T@Yc/float(len(X))
    U,S,Vt=np.linalg.svd(cov,full_matrices=False);R=U@Vt;sign=1.0
    if np.linalg.det(R)<0.0:U[:,-1]*=-1.0;R=U@Vt;sign=-1.0
    denom=float(np.mean(np.sum(Xc*Xc,axis=1)));scale=float((np.sum(S[:-1])+sign*S[-1])/max(denom,_EPS)) if len(S)>=3 else 1.0
    scale=float(np.clip(scale,float(scale_min),float(scale_max)));translation=my-scale*(mx@R);predicted=scale*(X@R)+translation
    rms=float(np.sqrt(np.mean(np.sum((predicted-Y)**2,axis=1))))
    return R,scale,translation,rms


def preserve_surface_relative_detail_layout(source_vertices:np.ndarray,solved_vertices:np.ndarray,faces:np.ndarray,*,config:SurfaceRelativeDetailConfig|None=None)->tuple[np.ndarray,dict[str,Any]]:
    cfg=config or SurfaceRelativeDetailConfig();source=np.asarray(source_vertices,dtype=np.float64);solved=np.asarray(solved_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if source.ndim!=2 or source.shape[1]!=3 or solved.shape!=source.shape:raise ValueError("source_vertices and solved_vertices must be matching Nx3 arrays")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(solved)):raise ValueError("surface-relative detail layout received non-finite vertices")
    comps=[]
    for index,(vids,fids) in enumerate(_components(len(source),F)):
        P=source[vids];extent=np.ptp(P,axis=0);comps.append({"index":int(index),"vids":vids,"fids":fids,"vertex_count":int(len(vids)),"face_count":int(len(fids)),"extent":extent,"max_extent":float(np.max(extent)),"centroid":np.mean(P,axis=0)})
    hosts=[c for c in comps if c["vertex_count"]>=int(cfg.min_host_vertices) and c["max_extent"]>=float(cfg.min_host_extent_m)]
    details=[c for c in comps if int(cfg.min_detail_vertices)<=c["vertex_count"]<=int(cfg.max_detail_vertices) and c["max_extent"]<=float(cfg.max_detail_extent_m)]
    if not hosts or not details:return solved.copy(),{"enabled":False,"reason":"no compatible carrier/detail components","component_count":len(comps),"host_count":len(hosts),"detail_candidate_count":len(details),"moved_vertices":0}
    host_trees={h["index"]:cKDTree(source[h["vids"]]) for h in hosts};out=solved.copy();reports=[];moved_mask=np.zeros(len(source),dtype=bool);maximum=0.0
    for detail in details:
        D=source[detail["vids"]];best=None
        for host in hosts:
            if host["index"]==detail["index"]:continue
            distances,_=host_trees[host["index"]].query(D,k=1,workers=1);median=float(np.median(distances));p90=float(np.percentile(distances,90))
            if median>float(cfg.max_source_median_clearance_m) or p90>float(cfg.max_source_p90_clearance_m):continue
            score=median+0.35*p90
            if best is None or score<best[0]:best=(score,host,median,p90)
        if best is None:continue
        _,host,median_clearance,p90_clearance=best;tree=host_trees[host["index"]];host_ids=np.asarray(host["vids"],dtype=np.int64)
        # Use the carrier vertices actually surrounding the detail plus a small centroid neighbourhood.
        _,nearest=tree.query(D,k=1,workers=1);anchors=np.unique(host_ids[np.asarray(nearest,dtype=np.int64).reshape(-1)])
        k=min(len(host_ids),max(int(cfg.anchor_vertices),int(cfg.min_anchor_vertices)))
        if k>0:
            _,near_centroid=tree.query(np.asarray(detail["centroid"],dtype=np.float64),k=k,workers=1);anchors=np.unique(np.r_[anchors,host_ids[np.asarray(near_centroid,dtype=np.int64).reshape(-1)]])
        if len(anchors)<int(cfg.min_anchor_vertices):continue
        R,scale,translation,fit_rms=_fit_similarity(source[anchors],out[anchors],float(cfg.scale_min),float(cfg.scale_max))
        if not np.isfinite(fit_rms) or fit_rms>float(cfg.max_host_fit_rms_m):
            reports.append({"detail_component":detail["index"],"host_component":host["index"],"status":"carrier_fit_rejected","carrier_fit_rms_mm":fit_rms*1000.0,"source_clearance_median_mm":median_clearance*1000.0,"source_clearance_p90_mm":p90_clearance*1000.0,"moved_vertices":0});continue
        ids=np.asarray(detail["vids"],dtype=np.int64);predicted=scale*(source[ids]@R)+translation;delta=predicted-out[ids];length=np.linalg.norm(delta,axis=1);layout_error=float(np.median(length))
        if layout_error<float(cfg.min_layout_error_m):
            reports.append({"detail_component":detail["index"],"host_component":host["index"],"status":"already_carrier_relative","carrier_fit_rms_mm":fit_rms*1000.0,"carrier_scale":scale,"layout_error_median_mm":layout_error*1000.0,"source_clearance_median_mm":median_clearance*1000.0,"source_clearance_p90_mm":p90_clearance*1000.0,"moved_vertices":0});continue
        cap=float(cfg.max_vertex_correction_m);capped=length>cap
        if np.any(capped):delta[capped]*=(cap/np.maximum(length[capped],_EPS))[:,None]
        out[ids]+=delta;move=np.linalg.norm(delta,axis=1);moved=move>1.0e-12;moved_mask[ids[moved]]=True;maximum=max(maximum,float(np.max(move,initial=0.0)))
        # Similarity-normalised shape error: detail structure should match the carrier-frame source after correction.
        post=np.linalg.norm(predicted-out[ids],axis=1)
        reports.append({"detail_component":detail["index"],"host_component":host["index"],"status":"carrier_relative_corrected","carrier_fit_rms_mm":fit_rms*1000.0,"carrier_scale":scale,"layout_error_median_mm":layout_error*1000.0,"layout_error_p95_mm":float(np.percentile(length,95)*1000.0),"post_error_p95_mm":float(np.percentile(post,95)*1000.0),"source_clearance_median_mm":median_clearance*1000.0,"source_clearance_p90_mm":p90_clearance*1000.0,"moved_vertices":int(np.count_nonzero(moved)),"maximum_correction_mm":float(np.max(move,initial=0.0)*1000.0)})
    corrected=[r for r in reports if r.get("status")=="carrier_relative_corrected"]
    return out,{"enabled":bool(corrected),"policy":"compact disconnected source-proven details inherit the local similarity frame of their already-solved nearby garment carrier; source shape/standoff is preserved without freezing carrier fit","component_count":len(comps),"host_count":len(hosts),"detail_candidate_count":len(details),"matched_detail_count":len(reports),"corrected_detail_count":len(corrected),"moved_vertices":int(np.count_nonzero(moved_mask)),"maximum_correction_mm":maximum*1000.0,"details":reports}
