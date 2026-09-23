from __future__ import annotations
from typing import Any
import numpy as np


def _target_surfaces(prod: Any, cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    collision_fn=getattr(prod,"_target_fit_collision_triangles",None);triangles_fn=getattr(prod,"_triangles_from_surface",None)
    if not callable(collision_fn) or not callable(triangles_fn):return None
    collision=np.asarray(collision_fn(cache),dtype=np.float64);support=cache.get("_ravafit_target_support_triangles")
    if support is None:
        v=np.asarray(cache.get("target_support_V",[]),dtype=np.float64);f=np.asarray(cache.get("target_support_F",[]),dtype=np.int64)
        if v.ndim!=2 or v.shape[1:]!=(3,) or f.ndim!=2 or f.shape[1:]!=(3,):return None
        support=triangles_fn(v,f);cache["_ravafit_target_support_triangles"]=support
    support=np.asarray(support,dtype=np.float64)
    if collision.ndim!=3 or collision.shape[1:]!=(3,3) or not len(collision) or support.ndim!=3 or support.shape[1:]!=(3,3) or not len(support):return None
    return collision,support


def _dense_penetration_report(prod: Any, source: Any, positions: dict[str,np.ndarray], target_triangles: np.ndarray, margin: float)->dict[str,Any]:
    occupancy=getattr(prod,"_nearest_literal_occupancy",None)
    if not callable(occupancy):return {"enabled":False,"reason":"literal occupancy evaluator unavailable","penetrating_samples":0}
    bary=np.asarray([(i/4.,j/4.,(4-i-j)/4.) for i in range(5) for j in range(5-i)]+[(1/3.,1/3.,1/3.)],dtype=np.float64)
    rows=[];total=0;worst=None;threshold=float(margin)-.00003
    for name in sorted(positions):
        data=source.data(name);F=np.asarray(data.get("F",[]),dtype=np.int64);V=np.asarray(positions[name],dtype=np.float64)
        if not len(F) or V.ndim!=2 or V.shape[1:]!=(3,):continue
        samples=np.einsum("bk,fkj->fbj",bary,V[F]).reshape(-1,3);_,_,signed,_,_=occupancy(samples,target_triangles,k=48,exact_band=float(margin)+.002);signed=np.asarray(signed,dtype=np.float64);bad=signed<threshold;count=int(np.count_nonzero(bad));minimum=float(np.min(signed)) if len(signed) else None
        if minimum is not None:worst=minimum if worst is None else min(worst,minimum)
        total+=count;rows.append({"mesh":name,"faces":int(len(F)),"samples":int(len(samples)),"penetrating_samples":count,"minimum_signed_mm":minimum*1000 if minimum is not None else None})
    return {"enabled":True,"margin_mm":float(margin*1000),"penetrating_samples":int(total),"minimum_signed_mm":worst*1000 if worst is not None else None,"meshes":rows,"policy":"dense face-interior samples must remain outside the complete selected target-body collision surface"}


def install_dense_final_target_occupancy(prod: Any)->None:
    if getattr(prod,"_ravafit_dense_final_target_occupancy_installed",False):return
    original=getattr(prod,"_finalize_modded_coupled_solution",None);clear=getattr(prod,"_final_target_body_clearance",None)
    if not callable(original) or not callable(clear):return
    def guarded(source,cache,positions,skinning,records,contexts,local_affine_quality_rms_mm):
        candidate,skinning_out,records_out,stats=original(source,cache,positions,skinning,records,contexts,local_affine_quality_rms_mm);surfaces=_target_surfaces(prod,cache)
        if surfaces is None:raise ValueError("Final target-body occupancy validation cannot resolve the selected target collision/support surfaces.")
        target_collision,target_support=surfaces;corrected,clearance_report=clear(source,candidate,target_collision,target_support,margin_m=.00035,maximum_vertex_move_m=.008,maximum_step_m=.00125,max_passes=10)
        before={n:np.asarray(v,dtype=np.float64) for n,v in candidate.items()};candidate={n:np.asarray(v,dtype=np.float64) for n,v in corrected.items()};changed=[n for n in candidate if n in before and np.any(np.linalg.norm(candidate[n]-before[n],axis=1)>1e-8)]
        for n in changed:records_out.setdefault(n,{})["dense_final_target_occupancy_applied"]=True
        validation=_dense_penetration_report(prod,source,candidate,target_collision,margin=.00012)
        if not validation.get("enabled",False):raise ValueError(f"Final target-body occupancy validation unavailable: {validation.get('reason','unknown reason')}")
        if int(validation.get("penetrating_samples",0))>0:raise ValueError(f"RavaFit refused to emit a model with target body still penetrating garment face interiors: {int(validation['penetrating_samples'])} dense sample(s), worst signed clearance {validation.get('minimum_signed_mm')} mm.")
        merged=dict(stats or {});merged["dense_final_target_occupancy"]={"changed_meshes":changed,"changed_mesh_count":len(changed),"clearance":clearance_report,"validation":validation};return candidate,skinning_out,records_out,merged
    prod._finalize_modded_coupled_solution=guarded;prod._ravafit_dense_final_target_occupancy_installed=True
