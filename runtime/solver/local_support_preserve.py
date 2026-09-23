"""Local source-authority for motion-faithful garment refits.

If the paired body support directly underneath a garment region is effectively the
same as the source, the untouched source geometry and authored skinning are already
the correct motion solution.  Other body regions are still free to retarget.

The dense final target-body occupancy guard is installed after this module, so literal
target anatomy can still override source preservation where the target truly differs.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _normalise(values: np.ndarray) -> np.ndarray:
    values=np.asarray(values,dtype=np.float64)
    return np.divide(values,np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12))


def _transition_alpha(faces: np.ndarray, stable: np.ndarray, raw_to_weld: np.ndarray | None, rings: int=4) -> np.ndarray:
    stable=np.asarray(stable,dtype=bool).copy();count=len(stable);faces=np.asarray(faces,dtype=np.int64)
    if count==0:return np.zeros(0,dtype=np.float64)
    weld=np.asarray(raw_to_weld,dtype=np.int64) if raw_to_weld is not None else None
    if weld is not None and weld.shape==(count,):
        for wid in np.unique(weld):
            ids=np.flatnonzero(weld==wid)
            if len(ids)>1:stable[ids]=bool(np.all(stable[ids]))
    if np.all(stable):return np.ones(count,dtype=np.float64)
    if not np.any(stable):return np.zeros(count,dtype=np.float64)
    neighbours=[set() for _ in range(count)]
    for tri in faces:
        a,b,c=(int(x) for x in tri)
        if min(a,b,c)<0 or max(a,b,c)>=count:continue
        neighbours[a].update((b,c));neighbours[b].update((a,c));neighbours[c].update((a,b))
    distance=np.full(count,rings+1,dtype=np.int32);queue:deque[int]=deque()
    for index in np.flatnonzero(~stable):distance[index]=0;queue.append(int(index))
    while queue:
        current=queue.popleft()
        if distance[current]>=rings:continue
        nd=int(distance[current])+1
        for neighbour in neighbours[current]:
            if nd<distance[neighbour]:distance[neighbour]=nd;queue.append(neighbour)
    alpha=np.clip(distance.astype(np.float64)/float(max(rings,1)),0.0,1.0);alpha[~stable]=0.0
    alpha=alpha*alpha*(3.0-2.0*alpha)
    if weld is not None and weld.shape==(count,):
        for wid in np.unique(weld):
            ids=np.flatnonzero(weld==wid)
            if len(ids)>1:alpha[ids]=float(np.min(alpha[ids]))
    return alpha


def _local_equivalence(prod: Any, source_positions: np.ndarray, source_weights: np.ndarray, joint_names: list[str], cache: dict[str,Any]):
    source=np.asarray(source_positions,dtype=np.float64);weights=np.asarray(source_weights,dtype=np.float64);count=len(source)
    support_frame=getattr(prod,"_coupled_support_frame",None);nearest=getattr(prod,"_b14_nearest_surface",None);triangles=getattr(prod,"_triangles_from_surface",None);delta_helper=getattr(prod,"_verified_body_skin_delta_at_points",None)
    if count==0 or not all(callable(fn) for fn in (support_frame,nearest,triangles,delta_helper)):
        return np.zeros(count,dtype=bool),{"enabled":False,"reason":"local support correspondence unavailable"}
    sv=np.asarray(cache.get("source_support_V",[]),dtype=np.float64);sf=np.asarray(cache.get("source_support_F",[]),dtype=np.int64)
    if sv.ndim!=2 or sv.shape[1:]!=(3,) or sf.ndim!=2 or sf.shape[1:]!=(3,) or not len(sf):
        return np.zeros(count,dtype=bool),{"enabled":False,"reason":"source support surface unavailable"}
    frame=support_frame(source,cache)
    if frame is None:return np.zeros(count,dtype=bool),{"enabled":False,"reason":"paired target support frame unavailable"}
    target_contact,target_normal,_,_=frame;target_contact=np.asarray(target_contact,dtype=np.float64);target_normal=_normalise(target_normal)
    source_contact,source_normal,_,_,_=nearest(source,triangles(sv,sf),k=48);source_contact=np.asarray(source_contact,dtype=np.float64);source_normal=_normalise(source_normal)
    if source_contact.shape!=source.shape or target_contact.shape!=source.shape:return np.zeros(count,dtype=bool),{"enabled":False,"reason":"support correspondence shape mismatch"}
    support_move=np.linalg.norm(target_contact-source_contact,axis=1);normal_dot=np.abs(np.einsum("ij,ij->i",source_normal,target_normal))
    _,delta_report=delta_helper(source,weights,list(joint_names),cache);delta_report=delta_report or {};local_delta=np.asarray(delta_report.get("local_delta_l1",np.full(count,np.inf)),dtype=np.float64)
    if local_delta.shape!=(count,):local_delta=np.full(count,np.inf,dtype=np.float64)
    quant=float(delta_report.get("quantisation_floor",.0035));geometry_limit=.00040;normal_limit=float(np.cos(np.deg2rad(3.0)));weight_limit=max(quant,1.0/255.0)
    finite=np.isfinite(support_move)&np.isfinite(normal_dot)&np.isfinite(local_delta);stable=finite&(support_move<=geometry_limit)&(normal_dot>=normal_limit)&(local_delta<=weight_limit)
    valid_delta=local_delta[np.isfinite(local_delta)]
    return stable,{"enabled":True,"vertices":int(count),"equivalent_vertices":int(np.count_nonzero(stable)),"equivalent_fraction":float(np.mean(stable)),"geometry_limit_mm":geometry_limit*1000.0,"normal_limit_degrees":3.0,"weight_delta_limit_l1":weight_limit,"support_move_p50_mm":float(np.percentile(support_move,50)*1000.0),"support_move_p95_mm":float(np.percentile(support_move,95)*1000.0),"body_weight_delta_p95":float(np.percentile(valid_delta,95)) if len(valid_delta) else None,"policy":"exact source authority only where paired support geometry, orientation and deformation field are all equivalent"}


def install_local_support_preservation(prod: Any)->None:
    if getattr(prod,"_ravafit_local_support_preservation_installed",False):return
    original=getattr(prod,"_finalize_modded_coupled_solution",None);weld_mesh=getattr(prod,"weld_mesh",None)
    if not callable(original) or not callable(weld_mesh):return
    def guarded(source,cache,positions,skinning,records,contexts,local_affine_quality_rms_mm):
        candidate,skinning_out,records_out,stats=original(source,cache,positions,skinning,records,contexts,local_affine_quality_rms_mm);candidate={n:np.asarray(v,dtype=np.float64).copy() for n,v in candidate.items()};skinning_out=dict(skinning_out);mesh_reports=[];changed=[]
        for name in sorted(candidate):
            data=source.data(name);source_v=np.asarray(data.get("V",[]),dtype=np.float64);source_f=np.asarray(data.get("F",[]),dtype=np.int64);source_w=np.asarray(data.get("W",[]),dtype=np.float64);joint_names=list(data.get("joint_names") or []);solved_v=candidate[name];skin=skinning_out.get(name) or {};solved_w=np.asarray(skin.get("weights",[]),dtype=np.float64)
            if source_v.shape!=solved_v.shape or source_w.shape!=solved_w.shape or not len(source_v):continue
            stable,report=_local_equivalence(prod,source_v,source_w,joint_names,cache)
            if not report.get("enabled") or not np.any(stable):report["mesh"]=name;mesh_reports.append(report);continue
            try:raw_to_weld=np.asarray(weld_mesh(data).get("raw_to_weld"),dtype=np.int64)
            except Exception:raw_to_weld=None
            alpha=_transition_alpha(source_f,stable,raw_to_weld,rings=4);exact=alpha>=1.0-1e-12;transition=(alpha>1e-12)&~exact
            if not np.any(alpha>1e-12):report["mesh"]=name;mesh_reports.append(report);continue
            new_v=solved_v*(1.0-alpha[:,None])+source_v*alpha[:,None];new_w=solved_w*(1.0-alpha[:,None])+source_w*alpha[:,None];target_total=source_w.sum(axis=1);total=new_w.sum(axis=1);valid=total>1e-12;new_w[valid]*=(target_total[valid]/total[valid])[:,None];new_v[exact]=source_v[exact];new_w[exact]=source_w[exact]
            candidate[name]=new_v;stage=dict(skin.get("stage") or {});authority={"exact_source_vertices":int(np.count_nonzero(exact)),"transition_vertices":int(np.count_nonzero(transition)),"retargeted_vertices":int(np.count_nonzero(alpha<=1e-12))};stage["local_support_source_authority"]=authority;skinning_out[name]={"weights":new_w,"joint_names":list(skin.get("joint_names") or joint_names),"stage":stage};records_out.setdefault(name,{})["local_support_source_authority"]=authority;report.update({"mesh":name,**authority,"exact_source_fraction":float(np.mean(exact))});mesh_reports.append(report);changed.append(name)
        merged=dict(stats or {});merged["local_support_source_authority"]={"enabled":True,"changed_meshes":changed,"changed_mesh_count":len(changed),"meshes":mesh_reports,"policy":"unchanged local body support keeps untouched garment rest geometry and authored skinning; only genuinely changed support is retargeted"};return candidate,skinning_out,records_out,merged
    prod._finalize_modded_coupled_solution=guarded;prod._ravafit_local_support_preservation_installed=True
