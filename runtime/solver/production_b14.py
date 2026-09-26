from __future__ import annotations

import copy
import ctypes
import hashlib
import gc
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
import trimesh

_MODULE_DIR = Path(__file__).resolve().parent
# Usually lives under Runtime/solver, but tolerate a developer copy in Runtime.
RUNTIME_ROOT = _MODULE_DIR.parent if _MODULE_DIR.name.casefold() == "solver" else _MODULE_DIR
RBODY_TOOLS = RUNTIME_ROOT / "rbody"
B14_SCRIPTS = RUNTIME_ROOT / "b14_frozen" / "scripts"
for p in (RBODY_TOOLS, B14_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rbody_v3_loader import get_cached_rbody, close_cached_rbodies
from rbody_b14_adapter import collect_body_pairs, build_dense_source_proxy

# Frozen B14 is read-only.
from ffxiv_lobofit import (
    GLB,
    weld_mesh,
    solver_body_weights,
    local_body_field_map_soft,
    precompute_body_local_affines,
    skeleton_bone_axes,
    skeleton_global_positions,
    suppress_bone_axis_drift,
    expand_welded,
)
from b14_compat import (
    solve_standoff,
    solve_constructed_close,
    solve_flexible,
    solve_conservative,
    solve_standoff_root_only,
    solve_constructed_close_root_only,
    refine_peer_assembly,
    runtime_self_test as _b14_runtime_self_test,
    reset_runtime_caches as _reset_b14_runtime_caches,
    set_surface_query_cache_enabled as _set_surface_query_cache_enabled,
)
from construction_fields import infer_shell_behavior
from collision_eval import smooth_collision_polish as _b14_residual_collision_polish, nearest_surface as _b14_nearest_surface
from glb_patch_legacy import GLBEditor, compute_tangents
from b14_local_retarget import retarget_b14_local_components, retarget_attached_ribbons_to_assembly, preserve_source_proven_attachment_continuity, _bridge_new_local_curvature
from universal_garment_refit import coherent_universal_refit, fit_body_macro_transform
from unilateral_pair_separation import UnilateralPairSeparationConfig, preserve_source_unilateral_pair_separation
from surface_relative_detail_layout import SurfaceRelativeDetailConfig, preserve_surface_relative_detail_layout
from source_relative_structural_carriers import StructuralCarrierConfig, preserve_source_relative_structural_carriers
from cloth_clearance_envelope import clearance_envelope, accept_envelope_step
from body_visibility import covered_body_faces
from multi_primitive_glb import ProductionGLB, mesh_domains, aggregate_mesh_data

# Never let the production path inherit Frozen B14's first-primitive-only GLB view.
GLB = ProductionGLB

# B14 fits layers. RavaFit only discovers the layers, supplies the correct body correspondence, and preserves their authored relationships afterward.
PRODUCTION_REVISION = "1.1.9-complete-rbody-fit-authority"


def _best_effort_trim_process_memory() -> dict[str, object]:
    """Best-effort parent-process memory trimming before spawning heavy numerical workers."""
    result={"gc_collected":0,"malloc_trim_attempted":False,"malloc_trim_result":None}
    try:
        result["gc_collected"]=int(gc.collect())
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            libc=ctypes.CDLL("libc.so.6")
            trim=getattr(libc,"malloc_trim",None)
            if trim is not None:
                trim.argtypes=[ctypes.c_size_t]
                trim.restype=ctypes.c_int
                result["malloc_trim_attempted"]=True
                result["malloc_trim_result"]=int(trim(0))
    except Exception:
        pass
    return result


def runtime_self_test() -> dict[str, object]:
    result=dict(_b14_runtime_self_test())
    result["race_skeleton_retarget_math"]=_race_retarget_math_self_test()
    result["coherent_coverage_frame_math"]=_coherent_coverage_frame_self_test()
    return result

_MESH_RE = re.compile(r"^mesh\s+(?P<mesh>\d+)(?:[.-](?P<sub>\d+))?$", re.IGNORECASE)
_SLOT_SUFFIX = {
    "_top.mdl": "Chest",
    "_dwn.mdl": "Legs",
    "_glv.mdl": "Hands",
    "_sho.mdl": "Feet",
}
_ACCESSORY_SUFFIX = {
    "_ear.mdl": "Earrings",
    "_nek.mdl": "Necklace",
    "_wrs.mdl": "Wrists",
    "_rir.mdl": "Right Ring",
    "_ril.mdl": "Left Ring",
}


def _normalise_material(name: str) -> str:
    return str(name or "").replace("\\", "/").strip().casefold()


_BODY_PACKAGE_MATERIAL_RE = re.compile(r"(?:^|/)mt_c\d{4}b(?P<body>\d{4})_(?P<name>[^/]+?)\.mtrl$")


def _body_package_material_key(material: str | None) -> tuple[str, str] | None:
    """Canonical body-package material identity, ignoring the race prefix only."""
    value=_normalise_material(material)
    match=_BODY_PACKAGE_MATERIAL_RE.search(value)
    if match is None:return None
    return match.group("body"),match.group("name")


def _material_matches_body_package(material: str | None, body_materials: set[str]) -> bool:
    value=_normalise_material(material)
    if value in body_materials:return True
    key=_body_package_material_key(value)
    if key is None:return False
    return any(_body_package_material_key(candidate)==key for candidate in body_materials)


def _is_obvious_embedded_body_material(material: str | None) -> bool:
    value = _normalise_material(material)
    marker = value.rfind("/mt_c")
    if marker < 0:
        marker = value.find("mt_c")
    if marker < 0:
        return False
    tail = value[marker + (1 if value[marker] == "/" else 0):]
    return len(tail) >= 14 and tail.startswith("mt_c") and tail[4:8].isdigit() and tail[8] == "b" and tail[9:13].isdigit()


def _embedded_body_reference(glb: GLB, slot: str, rig_joint_names: list[str]):
    rows=[];materials=set();records=[];base=0
    target_index={name:i for i,name in enumerate(rig_joint_names)}
    for name in glb.mesh_names():
        if not name:
            continue
        try:
            data=glb.data(name)
        except Exception:
            continue
        material=_normalise_material(data.get("material"))
        if not _is_obvious_embedded_body_material(material):
            continue
        V=np.asarray(data.get("V"),dtype=np.float64);F=np.asarray(data.get("F"),dtype=np.int64)
        if not len(V) or not len(F):
            continue
        UV=data.get("UV");N=data.get("N")
        if UV is None or len(UV)!=len(V):
            raise ValueError(f"Embedded body support mesh {name!r} has no usable UV data.")
        if N is None or len(N)!=len(V):
            raise ValueError(f"Embedded body support mesh {name!r} has no usable normal data.")
        source_names=list(data.get("joint_names") or [])
        source_weights=np.asarray(data.get("W"),dtype=np.float64)
        if source_weights.ndim!=2 or source_weights.shape[0]!=len(V):
            raise ValueError(f"Embedded body support mesh {name!r} has no usable skin weights.")
        W=np.zeros((len(V),len(rig_joint_names)),dtype=np.float64)
        for source_index,bone_name in enumerate(source_names):
            target=target_index.get(bone_name)
            if target is not None and source_index<source_weights.shape[1]:
                W[:,target]+=source_weights[:,source_index]
        total=W.sum(axis=1,keepdims=True)
        good=total[:,0]>1e-12
        if not np.all(good):
            raise ValueError(f"Embedded body support mesh {name!r} has {int(np.count_nonzero(~good))} vertices with no weights in the resolved rig.")
        W/=total
        rows.append((V,F+base,np.asarray(UV,dtype=np.float64),np.asarray(N,dtype=np.float64),W))
        records.append({"mesh_name":name,"material":material,"vertex_offset":base,"vertex_count":int(len(V)),"index_count":int(len(F)*3)})
        materials.add(material);base+=len(V)
    if not rows:
        available=set()
        for name in glb.mesh_names():
            if not name:
                continue
            try:
                available.add(_normalise_material(glb.data(name).get("material")))
            except Exception:
                continue
        raise ValueError(f"The selected {slot} source-support model contains no usable embedded body surface (mt_c####b####). Model materials were {sorted(available)}.")
    return {
        "V":np.vstack([row[0] for row in rows]),
        "F":np.vstack([row[1] for row in rows]),
        "UV":np.vstack([row[2] for row in rows]),
        "N":np.vstack([row[3] for row in rows]),
        "W":np.vstack([row[4] for row in rows]),
        "joint_names":list(rig_joint_names),
        "slot":slot,
        "mesh_records":records,
    },materials


def _embedded_body_materials(glb: GLB) -> set[str]:
    """Return body-material identities actually present in the outfit without using their geometry as fit support."""
    materials:set[str]=set()
    for name in glb.mesh_names():
        if not name:
            continue
        try:
            data=glb.data(name)
        except Exception:
            continue
        material=_normalise_material(data.get("material"))
        if not _is_obvious_embedded_body_material(material):
            continue
        V=np.asarray(data.get("V"),dtype=np.float64);F=np.asarray(data.get("F"),dtype=np.int64)
        if len(V) and len(F):
            materials.add(material)
    return materials


def _model_slot(game_path: str) -> str:
    p = str(game_path or "").replace("\\", "/").casefold()
    for suffix, slot in _SLOT_SUFFIX.items():
        if p.endswith(suffix):
            return slot
    raise ValueError(f"Could not determine XIV equipment slot from model path: {game_path}")


def _accessory_container_slot(game_path: str) -> str | None:
    p = str(game_path or "").replace("\\", "/").casefold()
    for suffix, slot in _ACCESSORY_SUFFIX.items():
        if p.endswith(suffix):
            return slot
    return None


def _resolve_output_policy(spec: dict[str, Any], source_contains_body: bool) -> tuple[bool, bool, str | None]:
    """Resolve body output authority from the authored source, not garment coverage.

    A target-body region may only be transplanted when the source asset actually contains that
    body region.  This makes complete source omission authoritative (for example, fully enclosed
    boots with no authored foot geometry) while leaving partial hidden-region suppression to the
    topology-based source-body suppression pass.
    """
    accessory_slot = _accessory_container_slot(str(spec.get("game_path") or ""))
    fit_only = bool(spec.get("fit_only", False)) or accessory_slot is not None
    requested_transplant = bool(spec.get("transplant_target_body", source_contains_body)) and bool(source_contains_body)
    return fit_only, False if fit_only else requested_transplant, accessory_slot


_PAIR_CACHE_LIMIT = 8
_PAIR_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()


def _node_local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t=np.asarray(node.get("translation",[0.0,0.0,0.0]),dtype=np.float64)
    q=np.asarray(node.get("rotation",[0.0,0.0,0.0,1.0]),dtype=np.float64)
    sc=np.asarray(node.get("scale",[1.0,1.0,1.0]),dtype=np.float64);x,y,z,w=q
    R=np.asarray([
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],
    ],dtype=np.float64)
    M=np.eye(4,dtype=np.float64);M[:3,:3]=R@np.diag(sc);M[:3,3]=t;return M


def _skinned_joint_names(glb: GLB) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    nodes=glb.js.get("nodes",[])
    for skin in glb.js.get("skins",[]):
        for joint in skin.get("joints",[]):
            name=str(nodes[int(joint)].get("name","") or "")
            if name and name not in seen:
                seen.add(name);names.append(name)
    return names


def _global_joint_matrices(glb: GLB, joint_names: list[str]) -> dict[str, np.ndarray]:
    parents={}
    for i,node in enumerate(glb.js.get("nodes",[])):
        for child in node.get("children",[]):parents[int(child)]=i
    cache={}
    def glob(index: int) -> np.ndarray:
        if index in cache:return cache[index]
        M=_node_local_matrix(glb.js["nodes"][index]);M=glob(parents[index])@M if index in parents else M;cache[index]=M;return M
    by_name={node.get("name",""):i for i,node in enumerate(glb.js.get("nodes",[])) if node.get("name")}
    return {name:glob(by_name[name]) for name in joint_names if name in by_name}


def _race_retarget_context(spec: dict[str, Any], joint_names: list[str]) -> dict[str, Any] | None:
    if not bool(spec.get("race_skeleton_retarget")):return None
    source_rest_path=spec.get("source_rest_glb")
    if not source_rest_path:raise ValueError("Race skeleton retarget requested without source_rest_glb")
    source_rest_path=Path(source_rest_path).resolve();target_rest_path=Path(spec["source_glb"]).resolve()
    if not source_rest_path.exists():raise FileNotFoundError(source_rest_path)
    source_rest=GLB(source_rest_path);target_rest=GLB(target_rest_path)

    # Include joints used by every garment mesh, including cloth and IVCS joints.
    source_skin_names=_skinned_joint_names(source_rest);target_skin_names=set(_skinned_joint_names(target_rest))
    common_names=[name for name in source_skin_names if name in target_skin_names]
    source_mats=_global_joint_matrices(source_rest,common_names);target_mats=_global_joint_matrices(target_rest,common_names)
    transforms={};singular=[]
    for name in common_names:
        src=source_mats.get(name);tgt=target_mats.get(name)
        if src is None or tgt is None:continue
        try:transforms[name]=tgt@np.linalg.inv(src)
        except np.linalg.LinAlgError:singular.append(name)
    if not transforms:raise ValueError("Source and target race skeletons expose no common invertible skinned joints")

    requested=list(dict.fromkeys(str(name) for name in joint_names if name))
    missing=[name for name in requested if name not in transforms]
    if missing:
        shown=", ".join(missing[:12]);remainder=f" (+{len(missing)-12} more)" if len(missing)>12 else ""
        raise ValueError(f"Race skeleton retarget cannot resolve {len(missing)} required source joint(s) in both rest skeletons: {shown}{remainder}")
    requested_common=requested
    shift=np.zeros(0,dtype=np.float64)
    if requested_common:
        source_pos,_,_=skeleton_global_positions(source_rest,requested_common);target_pos,_,_=skeleton_global_positions(target_rest,requested_common)
        shift=np.linalg.norm(target_pos-source_pos,axis=1)
    return {
        "transforms":transforms,
        "source_race_code":str(spec.get("source_race_code") or ""),"target_race_code":str(spec.get("target_race_code") or ""),
        "joint_count":int(len(transforms)),"requested_joint_count":int(len(requested)),
        "missing_joint_count":int(len(missing)),"missing_joints":missing[:32],"singular_joint_count":int(len(singular)),
        "joint_shift_p50_mm":float(np.percentile(shift,50)*1000.0) if len(shift) else 0.0,
        "joint_shift_p95_mm":float(np.percentile(shift,95)*1000.0) if len(shift) else 0.0,
        "joint_shift_max_mm":float(np.max(shift)*1000.0) if len(shift) else 0.0,
    }


def _retarget_points_by_skin(points: np.ndarray, weights: np.ndarray, joint_names: list[str], context: dict[str, Any]) -> np.ndarray:
    P=np.asarray(points,dtype=np.float64);W=np.asarray(weights,dtype=np.float64)
    if W.shape!=(len(P),len(joint_names)):raise ValueError(f"Skeleton retarget weights {W.shape} do not match points/joints {(len(P),len(joint_names))}")
    out=np.zeros_like(P);mass=np.zeros(len(P),dtype=np.float64);transforms=context["transforms"]
    for column,name in enumerate(joint_names):
        w=W[:,column]
        if not np.any(w>1e-12):continue
        M=transforms.get(name)
        if M is None:continue
        mapped=P@M[:3,:3].T+M[:3,3];out+=w[:,None]*mapped;mass+=w
    missing=mass<1.0-1e-8
    if np.any(missing):out[missing]+=(1.0-mass[missing])[:,None]*P[missing]
    good=mass>1e-12
    out[~good]=P[~good]
    return out


def _retarget_normals_by_skin(normals: np.ndarray, weights: np.ndarray, joint_names: list[str], context: dict[str, Any]) -> np.ndarray:
    N=np.asarray(normals,dtype=np.float64);W=np.asarray(weights,dtype=np.float64);out=np.zeros_like(N);mass=np.zeros(len(N),dtype=np.float64)
    for column,name in enumerate(joint_names):
        w=W[:,column]
        if not np.any(w>1e-12):continue
        M=context["transforms"].get(name)
        if M is None:continue
        try:A=np.linalg.inv(M[:3,:3]).T
        except np.linalg.LinAlgError:A=M[:3,:3]
        out+=w[:,None]*(N@A.T);mass+=w
    missing=mass<1.0-1e-8
    if np.any(missing):out[missing]+=(1.0-mass[missing])[:,None]*N[missing]
    norm=np.linalg.norm(out,axis=1,keepdims=True);return out/np.maximum(norm,1e-12)


def _race_retarget_math_self_test() -> bool:
    points=np.asarray([[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0]],dtype=np.float64)
    weights=np.asarray([[1.0,0.0],[0.0,1.0],[0.25,0.75]],dtype=np.float64)
    first=np.eye(4,dtype=np.float64);first[:3,3]=[2.0,0.0,0.0]
    second=np.eye(4,dtype=np.float64);second[:3,3]=[0.0,-4.0,0.0]
    context={"transforms":{"a":first,"b":second}}
    mapped=_retarget_points_by_skin(points,weights,["a","b"],context)
    expected=np.asarray([[3.0,0.0,0.0],[0.0,-3.0,0.0],[1.5,-2.0,0.0]],dtype=np.float64)
    if not np.allclose(mapped,expected,rtol=0.0,atol=1e-12):
        raise RuntimeError("Race skeleton retarget point blend changed unexpectedly.")
    normals=_retarget_normals_by_skin(np.asarray([[0.0,0.0,1.0]]*3,dtype=np.float64),weights,["a","b"],context)
    if not np.all(np.isfinite(normals)) or not np.allclose(np.linalg.norm(normals,axis=1),1.0,atol=1e-12):
        raise RuntimeError("Race skeleton retarget normal blend produced an invalid frame.")
    return True


def _retarget_reference(ref: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    out=copy.deepcopy(ref);names=list(out["joint_names"]);W=np.asarray(out["W"],dtype=np.float64)
    out["V"]=_retarget_points_by_skin(out["V"],W,names,context)
    if out.get("N") is not None:out["N"]=_retarget_normals_by_skin(out["N"],W,names,context)
    return out


def _retarget_loaded_source_glb(source: GLB, context: dict[str, Any]) -> dict[str, Any]:
    reports=[]
    for mesh_name in source.mesh_names():
        if not mesh_name:continue
        try:data=source.data(mesh_name)
        except Exception:continue
        names=list(data["joint_names"]);W=np.asarray(data["W"],dtype=np.float64);before=np.asarray(data["V"],dtype=np.float64)
        after=_retarget_points_by_skin(before,W,names,context)
        _,primitive=source.primitive(mesh_name);attrs=primitive["attributes"]
        position_accessor=source.accessor(attrs["POSITION"],False);position_accessor[:]=after.astype(position_accessor.dtype,copy=False)
        if "NORMAL" in attrs and data.get("N") is not None:
            normals=_retarget_normals_by_skin(data["N"],W,names,context);normal_accessor=source.accessor(attrs["NORMAL"],False);normal_accessor[:]=normals.astype(normal_accessor.dtype,copy=False)
        move=np.linalg.norm(after-before,axis=1)
        reports.append({"mesh":mesh_name,"vertices":int(len(before)),"move_p50_mm":float(np.percentile(move,50)*1000.0),"move_p95_mm":float(np.percentile(move,95)*1000.0),"move_max_mm":float(np.max(move)*1000.0)})
    return {"enabled":True,"meshes":reports,"mesh_count":len(reports),**{key:value for key,value in context.items() if key!="transforms"}}


def _rig_names_from_loaded_glb(glb: GLB, mesh_name: str | None = None) -> list[str]:
    """Read rig names without reparsing the source GLB."""
    js=glb.js;mi=None
    if mesh_name is not None:
        for i,m in enumerate(js.get("meshes",[])):
            if m.get("name")==mesh_name:mi=i;break
        if mi is None:raise KeyError(f"No mesh {mesh_name!r}")
    node=None
    for n in js.get("nodes",[]):
        if "skin" not in n or "mesh" not in n:continue
        if mi is None or n["mesh"]==mi:node=n;break
    if node is None:raise ValueError("No skinned mesh found in rig GLB")
    skin=js["skins"][node["skin"]];return [js["nodes"][joint].get("name","") for joint in skin["joints"]]


def _rbody_stamp(path: str) -> tuple[str,int,int]:
    resolved=Path(path).resolve();stat=resolved.stat();return str(resolved),int(stat.st_size),int(stat.st_mtime_ns)


def _pair_cache_key(spec: dict[str, Any], names: list[str]) -> tuple[Any, ...]:
    rows=[]
    top_source_mode=str(spec.get("source_body_mode") or "rbody").casefold()
    source_body_glbs=spec.get("source_body_glbs") or {}
    for row in spec["slots"]:
        target_path=str(Path(row["target"]["rbody"]).resolve())
        source_mode=str(row.get("source_mode") or top_source_mode).casefold()
        if source_mode=="embedded":
            source_stamp=("embedded",_rbody_stamp(str(Path(spec["source_glb"]).resolve())),str(row.get("source_race_code") or ""))
        elif source_mode=="vanilla":
            source_body_glb=source_body_glbs.get(str(row["slot"]))
            if not source_body_glb:
                raise ValueError(f"Vanilla {row['slot']} source support is missing from source_body_glbs.")
            source_stamp=("vanilla",_rbody_stamp(str(Path(source_body_glb).resolve())),str(row.get("source_race_code") or ""))
        else:
            source=row.get("source") or {}
            source_path=str(Path(source["rbody"]).resolve())
            source_stamp=(_rbody_stamp(source_path),str(source.get("body") or ""),str(source.get("variant") or ""),str(row.get("source_race_code",row.get("race_code")) or ""),str(row.get("source_support_surface") or "body"))
        rows.append((
            str(row["slot"]),
            source_stamp,
            _rbody_stamp(target_path),str(row["target"]["body"]),str(row["target"]["variant"]),str(row.get("target_race_code",row.get("race_code")) or ""),str(row.get("target_support_surface") or "body"),bool(row.get("cross_sex_smallclothes_bridge")),
        ))
    retarget_stamp=None
    if bool(spec.get("race_skeleton_retarget")):
        source_rest=spec.get("source_rest_glb")
        target_rest=spec.get("source_glb")
        retarget_stamp=(
            _rbody_stamp(source_rest) if source_rest else None,
            _rbody_stamp(target_rest) if target_rest else None,
            str(spec.get("source_race_code") or ""),
            str(spec.get("target_race_code") or ""),
        )
    source_outfit_stamp=None
    if any(str(row.get("source_mode") or top_source_mode).casefold() in ("vanilla","rbody") for row in spec["slots"]):
        source_glb=spec.get("source_glb")
        source_outfit_stamp=_rbody_stamp(str(Path(source_glb).resolve())) if source_glb else None
    return tuple(names),tuple(rows),retarget_stamp,source_outfit_stamp


def _strict_macro_body_surface(ref: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
    """Return the broad anatomical surface that should shape clothing in the strict B14 lane.

    RBODY body payloads can contain small local detail meshes (pubic detail, nails, claws, etc.) that
    intentionally reuse parts of the body UV atlas.  They are useful collision/detail geometry, but
    feeding them into the historical UV correspondence makes those duplicate UVs compete with the
    actual macro body and can throw garment anchors across limbs.  Select the material group with the
    broadest spatial coverage and keep every record using that material.  This is geometry/material
    driven and has no body-, slot-, or outfit-specific names.
    """
    records=list(ref.get('mesh_records') or [])
    V=np.asarray(ref.get('V'),dtype=np.float64)
    if not records or V.ndim!=2 or V.shape[1]!=3:
        return ref,{"enabled":False,"reason":"no mesh records","selected_records":len(records)}
    groups:dict[str,list[dict[str,Any]]]={}
    for rec in records:
        material=_normalise_material(rec.get('material'))
        groups.setdefault(material,[]).append(rec)
    scored=[]
    for material,recs in groups.items():
        ids=[]
        for rec in recs:
            a=int(rec.get('vertex_offset',0));b=a+int(rec.get('vertex_count',0))
            if 0<=a<b<=len(V):ids.append(np.arange(a,b,dtype=np.int64))
        if not ids:continue
        ids=np.concatenate(ids);pts=V[ids];extent=np.ptp(pts,axis=0);bbox=float(np.prod(np.maximum(extent,1e-6)))
        # Broad coverage is primary; vertex count only breaks near-ties without favouring tiny dense detail.
        scored.append((bbox,int(len(ids)),material,recs,ids,extent))
    if not scored:
        return ref,{"enabled":False,"reason":"no valid material geometry","selected_records":len(records)}
    bbox,_,material,recs,ids,extent=max(scored,key=lambda row:(row[0],row[1]))
    # Rebuild the selected material group into one compact indexed surface.
    out={key:value for key,value in ref.items() if key not in {'V','F','UV','N','W','mesh_records'}}
    chunks=[];faces=[];new_records=[];base=0
    F0=np.asarray(ref.get('F'),dtype=np.int64)
    for rec in recs:
        a=int(rec.get('vertex_offset',0));vc=int(rec.get('vertex_count',0));b=a+vc
        if vc<=0 or not (0<=a<b<=len(V)):continue
        fo=int(rec.get('face_offset',0));fc=int(rec.get('face_count',int(rec.get('index_count',0))//3))
        localF=F0[fo:fo+fc]-a
        if len(localF) and (int(localF.min())<0 or int(localF.max())>=vc):
            continue
        chunks.append((a,b));faces.append(localF+base)
        nr=dict(rec);nr['vertex_offset']=base;nr['face_offset']=sum(int(x.get('face_count',0)) for x in new_records);nr['face_count']=int(len(localF));nr['index_count']=int(len(localF)*3);new_records.append(nr);base+=vc
    if not chunks:
        return ref,{"enabled":False,"reason":"selected material had no usable indexed records","selected_material":material}
    for key in ('V','UV','N','W'):
        arr=np.asarray(ref.get(key))
        out[key]=np.vstack([arr[a:b] for a,b in chunks]) if arr.ndim==2 else np.concatenate([arr[a:b] for a,b in chunks])
    out['F']=np.vstack(faces) if faces else np.empty((0,3),dtype=np.int64);out['mesh_records']=new_records
    # MDL storage can contain thousands of unreferenced/dead vertices. Historical B14 consumed the
    # rendered/indexed topology, so compact the macro body to triangle-referenced vertices here too.
    if len(out['F']):
        ids=np.unique(out['F'].reshape(-1));remap=np.full(len(out['V']),-1,dtype=np.int64);remap[ids]=np.arange(len(ids),dtype=np.int64)
        for key in ('V','UV','N','W'):out[key]=np.asarray(out[key])[ids].copy()
        out['F']=remap[out['F']]
        out['mesh_records']=[{'mesh_index':int(new_records[0].get('mesh_index',0)),'material':material,'vertex_offset':0,'vertex_count':int(len(ids)),'index_count':int(len(out['F'])*3),'face_offset':0,'face_count':int(len(out['F']))}]
    return out,{"enabled":True,"mode":"broadest-material-spatial-coverage+indexed","selected_material":material,"selected_records":len(new_records),"input_records":len(records),"selected_vertices":int(len(out['V'])),"bbox_volume":bbox,"extent":extent.tolist()}


def _strict_uv_map(SV: np.ndarray, SUV: np.ndarray, SN: np.ndarray, SW: np.ndarray, TV: np.ndarray, TUV: np.ndarray, TN: np.ndarray, TW: np.ndarray) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    """Historical k=12 UV map with a narrow rig-space ambiguity repair for duplicated UV islands."""
    tree=cKDTree(TUV);dist,idx=tree.query(SUV,k=min(12,len(TV)));dist=dist if dist.ndim>1 else dist[:,None];idx=idx if idx.ndim>1 else idx[:,None]
    ww=1.0/(dist*dist+1e-7);ww/=np.maximum(ww.sum(axis=1,keepdims=True),1e-12)
    Y=np.sum(TV[idx]*ww[:,:,None],axis=1);NT=np.sum(TN[idx]*ww[:,:,None],axis=1);NT/=np.maximum(np.linalg.norm(NT,axis=1,keepdims=True),1e-12)
    target_weights=np.sum(TW[idx]*ww[:,:,None],axis=1);target_weights=np.maximum(target_weights,0.0);target_weights/=np.maximum(target_weights.sum(axis=1,keepdims=True),1e-12)
    exact_weight_correspondence=bool(SW.shape==TW.shape and SUV.shape==TUV.shape and np.max(np.abs(SUV-TUV),initial=0.0)<=1e-10)
    if exact_weight_correspondence:target_weights=TW.copy()
    repaired=np.zeros(len(SV),dtype=bool)
    if SW.ndim==2 and TW.ndim==2 and SW.shape[1]==TW.shape[1] and SW.shape[1]>0:
        align=np.einsum('nk,nqk->nq',SW,TW[idx]);spread=np.max(align,axis=1)-np.min(align,axis=1)
        wr=ww*np.square(.05+np.maximum(align,0.0));wr/=np.maximum(wr.sum(axis=1,keepdims=True),1e-12)
        Yr=np.sum(TV[idx]*wr[:,:,None],axis=1);NTr=np.sum(TN[idx]*wr[:,:,None],axis=1);NTr/=np.maximum(np.linalg.norm(NTr,axis=1,keepdims=True),1e-12)
        target_weights_r=np.sum(TW[idx]*wr[:,:,None],axis=1);target_weights_r=np.maximum(target_weights_r,0.0);target_weights_r/=np.maximum(target_weights_r.sum(axis=1,keepdims=True),1e-12)
        delta=np.linalg.norm(Yr-Y,axis=1);base_disp=np.linalg.norm(Y-SV,axis=1);rig_disp=np.linalg.norm(Yr-SV,axis=1)
        # Only repair candidates where UV-near vertices have radically different rig identity and the
        # rig-consistent interpolation removes a large implausible jump. Ordinary historical mapping
        # (including the successful Rue chest case) remains byte-for-byte on the pure UV formula.
        repaired=(spread>.75)&(delta>.020)&(base_disp>.060)&(rig_disp<base_disp*.50)
        if np.any(repaired):
            Y[repaired]=Yr[repaired];NT[repaired]=NTr[repaired]
            if not exact_weight_correspondence:target_weights[repaired]=target_weights_r[repaired]
    return Y,NT,{"rig_ambiguity_repaired_vertices":int(np.count_nonzero(repaired)),"rig_ambiguity_max_correction_mm":float(np.max(np.linalg.norm(Yr-Y,axis=1))*1000.0) if 'Yr' in locals() and len(Yr) else 0.0,"_target_weight_correspondence":target_weights}


def _apply_strict_historical_uv_contract(pairs: list[tuple[dict[str,Any],dict[str,Any]]], base_cache: dict[str,Any]) -> dict[str,Any]:
    """Restore the successful frozen-B14 body contract without letting local RBODY detail pollute fit authority."""
    if not pairs:return dict(base_cache)
    rows=[];common_names=None;uv_p95=[];macro_reports=[];strict_source_surfaces=[];strict_target_surfaces=[];ambiguity_reports=[]
    for src0,tgt0 in pairs:
        if bool(src0.get('_cross_sex_smallclothes_bridge')) or bool(tgt0.get('_cross_sex_smallclothes_bridge')):
            out=dict(base_cache);out['_ravafit_strict_b14_contract']=False;out['_ravafit_strict_b14_report']={'enabled':False,'reason':'cross-sex smallclothes bridge requires modern adapter'};return out
        src,src_macro=_strict_macro_body_surface(src0);tgt,tgt_macro=_strict_macro_body_surface(tgt0);macro_reports.append({'slot':str(src.get('slot') or tgt.get('slot') or 'body'),'source':src_macro,'target':tgt_macro})
        SV=np.asarray(src.get('V'),dtype=np.float64);SUV=np.asarray(src.get('UV'),dtype=np.float64);SN=np.asarray(src.get('N'),dtype=np.float64);SW=np.asarray(src.get('W'),dtype=np.float64)
        TV=np.asarray(tgt.get('V'),dtype=np.float64);TUV=np.asarray(tgt.get('UV'),dtype=np.float64);TN=np.asarray(tgt.get('N'),dtype=np.float64);TW=np.asarray(tgt.get('W'),dtype=np.float64)
        sn=list(src.get('joint_names') or []);tn=list(tgt.get('joint_names') or [])
        valid=(SV.ndim==2 and SV.shape[1]==3 and len(SV)>=3 and SUV.shape==(len(SV),2) and SN.shape==(len(SV),3) and SW.ndim==2 and SW.shape[0]==len(SV) and TV.ndim==2 and TV.shape[1]==3 and len(TV)>=3 and TUV.shape==(len(TV),2) and TN.shape==(len(TV),3) and TW.ndim==2 and TW.shape[0]==len(TV) and sn==tn and len(sn)==SW.shape[1] and TW.shape[1]==len(sn))
        if not valid:
            out=dict(base_cache);out['_ravafit_strict_b14_contract']=False;out['_ravafit_strict_b14_report']={'enabled':False,'reason':'body references do not expose compatible UV/normal/rig surfaces'};return out
        if common_names is None:common_names=sn
        elif sn!=common_names:
            out=dict(base_cache);out['_ravafit_strict_b14_contract']=False;out['_ravafit_strict_b14_report']={'enabled':False,'reason':'slot body references do not share one rig list'};return out
        nearest=cKDTree(TUV).query(SUV,k=1)[0];p95=float(np.percentile(nearest,95));uv_p95.append(p95)
        if not np.isfinite(p95) or p95>.025:
            out=dict(base_cache);out['_ravafit_strict_b14_contract']=False;out['_ravafit_strict_b14_report']={'enabled':False,'reason':f'body UV layouts are incompatible (p95={p95:.6f})'};return out
        Y,NT,ambiguity=_strict_uv_map(SV,SUV,SN,SW,TV,TUV,TN,TW);target_correspondence_W=np.asarray(ambiguity.pop('_target_weight_correspondence'),dtype=np.float64);ambiguity_reports.append(ambiguity)
        NS=SN.copy();NS/=np.maximum(np.linalg.norm(NS,axis=1,keepdims=True),1e-12);BW=SW.copy();BW/=np.maximum(BW.sum(axis=1,keepdims=True),1e-12)
        slot=str(src.get('slot') or tgt.get('slot') or 'body');rows.append((SV.copy(),Y,BW,target_correspondence_W,NS,NT,np.asarray([slot]*len(SV),dtype=object)))
        sF_raw=src.get('F');tF_raw=tgt.get('F')
        if sF_raw is not None and tF_raw is not None:
            sF=np.asarray(sF_raw,dtype=np.int64);tF=np.asarray(tF_raw,dtype=np.int64)
            if sF.ndim==2 and sF.shape[1]==3 and tF.ndim==2 and tF.shape[1]==3 and len(sF) and len(tF):
                strict_source_surfaces.append((SV.copy(),sF.copy()));strict_target_surfaces.append((TV.copy(),tF.copy()))
    out=dict(base_cache);out['X']=np.vstack([r[0] for r in rows]);out['Y']=np.vstack([r[1] for r in rows]);out['BW']=np.vstack([r[2] for r in rows]);out['target_correspondence_W']=np.vstack([r[3] for r in rows]);out['NS']=np.vstack([r[4] for r in rows]);out['NT']=np.vstack([r[5] for r in rows]);out['names']=list(common_names or []);out['parts']=np.concatenate([r[6] for r in rows]);out['identity_body_mapping']=bool(np.max(np.linalg.norm(out['Y']-out['X'],axis=1))<1e-9);out['_ravafit_strict_b14_contract']=True
    # Keep macro support separate from complete body-package collision/detail geometry.
    if strict_source_surfaces and len(strict_source_surfaces)==len(strict_target_surfaces):
        sV=[];sF=[];tV=[];tF=[];sb=0;tb=0
        for (sv,sf),(tv,tf) in zip(strict_source_surfaces,strict_target_surfaces):
            sV.append(sv);sF.append(sf+sb);sb+=len(sv);tV.append(tv);tF.append(tf+tb);tb+=len(tv)
        out['_ravafit_strict_source_surface_V']=np.vstack(sV);out['_ravafit_strict_source_surface_F']=np.vstack(sF);out['_ravafit_strict_target_surface_V']=np.vstack(tV);out['_ravafit_strict_target_surface_F']=np.vstack(tF)
    out['_ravafit_strict_b14_report']={'enabled':True,'mode':'historical-k12-inverse-square-uv+macro-support','vertices':int(len(out['X'])),'slot_count':int(len(rows)),'uv_nearest_p95':uv_p95,'macro_support':macro_reports,'ambiguity_repairs':ambiguity_reports,'policy':'broad anatomical support shapes garment; local RBODY detail remains collision-only; no fitted garment/control geometry participates'}
    return out


def _load_pairs(spec: dict[str, Any], names: list[str], source_glb: GLB | None = None):
    cache_key=_pair_cache_key(spec,names);cached=_PAIR_CACHE.get(cache_key)
    if cached is not None:
        _PAIR_CACHE.move_to_end(cache_key);return cached

    pairs=[]
    # These sets identify embedded source-body meshes only; they are not garment classification.
    source_materials: set[str] = set()
    source_materials_by_slot: dict[str, set[str]] = {}
    target_materials_by_slot: dict[str, set[str]] = {}
    target_mesh_materials_by_slot: dict[str, dict[int, dict[str, Any]]] = {}
    payload_details: list[dict[str, Any]] = []
    retarget_context=_race_retarget_context(spec,names)
    embedded_cache: dict[str, tuple[dict[str, Any],set[str]]] = {}
    vanilla_support_cache: dict[str, tuple[dict[str, Any],set[str]]] = {}
    top_source_mode=str(spec.get("source_body_mode") or "rbody").casefold()
    source_body_glbs=spec.get("source_body_glbs") or {}
    vanilla_mode=top_source_mode=="vanilla" or any(str(row.get("source_mode") or top_source_mode).casefold()=="vanilla" for row in spec["slots"])
    actual_outfit_body_materials=_embedded_body_materials(source_glb) if source_glb is not None else set()
    for row in spec["slots"]:
        target_path=str(Path(row["target"]["rbody"]).resolve())
        tgt_lib=get_cached_rbody(target_path)
        target_race=row.get("target_race_code",row.get("race_code"));target_support_surface=str(row.get("target_support_surface") or "body")
        tgt_ref=tgt_lib.reference(row["target"]["body"],row["slot"],row["target"]["variant"],race_code=target_race,rig_joint_names=names,surface_mode=target_support_surface)
        # Cross-sex leg smallclothes provide support shape, not a signed collision solid.
        if bool(row.get("cross_sex_smallclothes_bridge")) and str(row.get("slot") or "").casefold()=="legs" and target_support_surface.casefold()=="smallclothes":
            target_body_collision_ref=tgt_lib.reference(row["target"]["body"],row["slot"],row["target"]["variant"],race_code=target_race,rig_joint_names=names,surface_mode="body")
            tgt_ref["_cross_sex_collision_V"]=np.asarray(target_body_collision_ref["V"],dtype=np.float64).copy()
            tgt_ref["_cross_sex_collision_F"]=np.asarray(target_body_collision_ref["F"],dtype=np.int64).copy()
            tgt_ref["_cross_sex_collision_W"]=np.asarray(target_body_collision_ref["W"],dtype=np.float64).copy()
            tgt_ref["_cross_sex_collision_mode"]="body_only_with_smallclothes_support"
        tgt_meta=tgt_lib.payload_meta(tgt_ref["payload_id"]);tgt_assignments=tgt_lib.mesh_material_assignments(tgt_ref["payload_id"])
        for assignment in tgt_assignments.values():assignment["normalised_material"]=_normalise_material(assignment["material"])
        tgt_mats={assignment["normalised_material"] for assignment in tgt_assignments.values()}
        target_surface_mats={_normalise_material(x) for x in tgt_meta.get("body_surface_materials",[]) if x}

        source_mode=str(row.get("source_mode") or top_source_mode).casefold()
        if source_mode=="embedded":
            if source_glb is None:
                raise ValueError("Embedded source-body mode requires the exported source GLB.")
            cache_slot=str(row["slot"])
            if cache_slot not in embedded_cache:
                embedded_cache[cache_slot]=_embedded_body_reference(source_glb,cache_slot,names)
            src_ref,src_mats=embedded_cache[cache_slot]
            src_meta={"body_surface_materials":sorted(src_mats)}
            source_payload="embedded-vanilla"
            source_race=row.get("source_race_code",row.get("race_code"))
        elif source_mode=="vanilla":
            cache_slot=str(row["slot"])
            source_body_glb=source_body_glbs.get(cache_slot)
            if not source_body_glb:
                raise ValueError(f"Vanilla {cache_slot} source support is missing from source_body_glbs.")
            source_body_path=Path(source_body_glb).resolve()
            if not source_body_path.exists():
                raise FileNotFoundError(source_body_path)
            if cache_slot not in vanilla_support_cache:
                vanilla_support_cache[cache_slot]=_embedded_body_reference(GLB(source_body_path),cache_slot,names)
            literal_src_ref,src_mats=vanilla_support_cache[cache_slot]
            # Vanilla e0000 surfaces are densified before entering the normal body-pair solve.
            src_ref=build_dense_source_proxy(literal_src_ref,tgt_ref)
            src_meta={"body_surface_materials":sorted(src_mats)}
            source_payload=str(src_ref.get("payload_id") or f"dense-xiv-vanilla-e0000:{cache_slot}")
            source_race=row.get("source_race_code",row.get("race_code"))
        else:
            source=row.get("source")
            if not isinstance(source,dict):
                raise ValueError(f"{row['slot']} conversion is missing its RBODY source reference.")
            source_path=str(Path(source["rbody"]).resolve());src_lib=get_cached_rbody(source_path)
            source_race=row.get("source_race_code",row.get("race_code"));source_support_surface=str(row.get("source_support_surface") or "body")
            src_ref=src_lib.reference(source["body"],row["slot"],source["variant"],race_code=source_race,rig_joint_names=names,surface_mode=source_support_surface)
            if retarget_context is not None:
                src_ref=_retarget_reference(src_ref,retarget_context)
            source_payload=src_ref["payload_id"]
            src_meta=src_lib.payload_meta(source_payload);src_assignments=src_lib.mesh_material_assignments(source_payload)
            for assignment in src_assignments.values():assignment["normalised_material"]=_normalise_material(assignment["material"])
            src_mats={assignment["normalised_material"] for assignment in src_assignments.values()}
            # The explicitly selected RBODY owns complete source anatomy. An
            # outfit's embedded body can be cut away, compressed or incomplete;
            # use it only for detach/visibility evidence after the garment fit.
            embedded_source_report={"enabled":False,"mode":"rbody_source_authority",
                                    "reason":"selected RBODY supplies complete source anatomy; embedded outfit body is output visibility evidence only"}
            embedded_source_mats=set(actual_outfit_body_materials)

        if bool(row.get("cross_sex_smallclothes_bridge")) and str(row.get("slot") or "").casefold()=="legs":
            src_ref["_cross_sex_smallclothes_bridge"]=True
            tgt_ref["_cross_sex_smallclothes_bridge"]=True
            literal_source_payload=str(src_ref.get("payload_id") or "source")
            src_ref=build_dense_source_proxy(src_ref,tgt_ref,k=48)
            src_ref["payload_id"]=f"dense-cross-sex:{literal_source_payload}->{tgt_ref.get('payload_id','target')}"
            src_ref["_cross_sex_smallclothes_bridge"]=True
            if isinstance(src_ref.get("_dense_proxy_stats"),dict):
                src_ref["_dense_proxy_stats"]["mode"]="dense_cross_sex_target_topology"
        pairs.append((src_ref,tgt_ref))
        source_surface_mats={_normalise_material(x) for x in src_meta.get("body_surface_materials",[]) if x}
        if source_mode=="vanilla":
            # e0000 is fitting support; outfit materials determine which body fragments are replaced.
            source_materials.update(actual_outfit_body_materials)
            source_materials_by_slot.setdefault(row["slot"],set()).update(actual_outfit_body_materials)
        elif source_mode=="rbody" and embedded_source_mats:
            # Keep embedded-body identification independent of fit authority.
            source_materials.update(embedded_source_mats);source_materials_by_slot.setdefault(row["slot"],set()).update(embedded_source_mats)
        else:
            source_materials.update(src_mats);source_materials_by_slot.setdefault(row["slot"],set()).update(src_mats)
        target_materials_by_slot.setdefault(row["slot"],set()).update(tgt_mats)
        target_mesh_materials_by_slot[row["slot"]]=tgt_assignments
        payload_details.append({
            "slot":row["slot"],"source_mode":source_mode,"source_payload":source_payload,"target_payload":tgt_ref["payload_id"],"source_race_code":source_race,"target_race_code":target_race,"source_support_surface":str(row.get("source_support_surface") or "body"),"target_support_surface":target_support_surface,"cross_sex_smallclothes_bridge":bool(row.get("cross_sex_smallclothes_bridge")),
            "source_materials":sorted(actual_outfit_body_materials if source_mode=="vanilla" else (embedded_source_mats if source_mode=="rbody" and embedded_source_mats else src_mats)),"source_support_materials":sorted(src_mats),"target_materials":sorted(tgt_mats),"source_surface_materials":sorted(source_surface_mats),"target_surface_materials":sorted(target_surface_mats),
            "embedded_source_authority":embedded_source_report if source_mode=="rbody" else {"enabled":source_mode=="embedded","mode":source_mode},
            "target_mesh_material_assignments":[tgt_assignments[index] for index in sorted(tgt_assignments)],
        })
    body_cache=collect_body_pairs(pairs)
    if not vanilla_mode and retarget_context is None:
        body_cache=_apply_strict_historical_uv_contract(pairs,body_cache)
    else:
        body_cache['_ravafit_strict_b14_contract']=False
        body_cache['_ravafit_strict_b14_report']={'enabled':False,'reason':'vanilla dense proxy or race-skeleton retarget uses modern body adapter'}
    if retarget_context is not None:
        body_cache["race_skeleton_retarget"]={key:value for key,value in retarget_context.items() if key!="transforms"}
    result=(names,body_cache,source_materials,source_materials_by_slot,target_materials_by_slot,target_mesh_materials_by_slot,payload_details)
    _PAIR_CACHE[cache_key]=result;_PAIR_CACHE.move_to_end(cache_key)
    while len(_PAIR_CACHE)>_PAIR_CACHE_LIMIT:_PAIR_CACHE.popitem(last=False)
    return result



def _attach_target_auxiliary_obstacles(cache: dict[str,Any], target_glbs_by_slot: dict[str,Path], payload_details: list[dict[str,Any]]):
    """Attach body-package geometry that is not the smooth body support as collision-only auxiliaries.

    This is deliberately material/geometry driven.  The RBODY payload tells us which material is the
    body support; any other exported geometry from the same selected target body package becomes an
    auxiliary obstacle (piercings, body ornaments, etc.).  Auxiliaries may veto garment occupancy but
    never become a direct garment-shaping surface.
    """
    rows=[];all_tri=[]
    detail_by_slot={str(row.get("slot")):row for row in payload_details}
    for slot,path in target_glbs_by_slot.items():
        detail=detail_by_slot.get(str(slot),{})
        surface={_normalise_material(x) for x in detail.get("target_surface_materials",[]) if x}
        package={_normalise_material(x.get("material","")) for x in detail.get("target_mesh_material_assignments",[]) if x.get("material")}
        try:glb=GLB(path)
        except Exception:continue
        for name in glb.mesh_names():
            if not name:continue
            try:data=glb.data(name)
            except Exception:continue
            mat=_normalise_material(data.get("material", ""));V=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64)
            if len(V)<3 or len(F)==0 or mat in surface or (package and mat not in package):continue
            # Split exported auxiliary meshes into actual connected pieces so a bilateral material does
            # not create one enormous artificial collision envelope spanning the body.
            labels=np.arange(len(V),dtype=np.int64)
            parent=np.arange(len(V),dtype=np.int64)
            def find(x):
                while parent[x]!=x:
                    parent[x]=parent[parent[x]];x=int(parent[x])
                return x
            def union(a,b):
                ra,rb=find(int(a)),find(int(b))
                if ra!=rb:parent[rb]=ra
            for a,b,c in F:union(a,b);union(b,c);union(c,a)
            groups={}
            for i in range(len(V)):groups.setdefault(find(i),[]).append(i)
            for ci,values in enumerate(groups.values()):
                ids=np.asarray(values,dtype=np.int64)
                if len(ids)<3:continue
                lut=np.full(len(V),-1,dtype=np.int64);lut[ids]=np.arange(len(ids),dtype=np.int64);mask=np.all(lut[F]>=0,axis=1);LF=lut[F[mask]]
                if len(LF)==0:continue
                P=V[ids].copy();rows.append({"slot":str(slot),"mesh":str(name),"component":int(ci),"material":mat,"V":P,"F":LF.copy()});all_tri.append(P[LF])
    cache["_ravafit_target_auxiliary_components"]=rows
    cache["_ravafit_target_auxiliary_triangles"]=np.concatenate(all_tri,axis=0) if all_tri else np.empty((0,3,3),dtype=np.float64)
    cache["_ravafit_target_auxiliary_report"]={"component_count":len(rows),"triangle_count":int(sum(len(x) for x in all_tri)),"policy":"target body-package non-support geometry is collision-only auxiliary authority"}
    cache.pop("_ravafit_auxiliary_envelope_support",None)
    return cache["_ravafit_target_auxiliary_report"]

def _mesh_node_index(glb: GLB, mesh_index: int) -> int | None:
    for ni, node in enumerate(glb.js.get("nodes", [])):
        if node.get("mesh") == mesh_index:
            return ni
    return None


def _primitive_material_names(glb: GLB, mesh_index: int) -> list[str]:
    out: list[str] = []
    materials = glb.js.get("materials", [])
    for p in glb.js.get("meshes", [])[mesh_index].get("primitives", []):
        mi = p.get("material")
        out.append(_normalise_material(materials[mi].get("name", "")) if mi is not None and 0 <= mi < len(materials) else "")
    return out


def _find_body_meshes(glb: GLB, body_materials: set[str]) -> list[str]:
    if not body_materials:
        raise ValueError("Selected source body exposes no body-package material identities.")
    names = []
    for mi, mesh in enumerate(glb.js.get("meshes", [])):
        mats = _primitive_material_names(glb, mi)
        if any(_material_matches_body_package(m,body_materials) for m in mats):
            names.append(mesh.get("name") or f"mesh {mi}")
    if not names:
        available = sorted({m for mi in range(len(glb.js.get("meshes", []))) for m in _primitive_material_names(glb, mi) if m})
        raise ValueError(
            "Could not identify the embedded source-body mesh from the selected RBODY reference. "
            f"Expected one of {sorted(body_materials)}; model materials were {available}."
        )
    return names




def _subdivide_vanilla_garment_data_once(data: dict[str, Any]) -> dict[str, Any]:
    """Subdivide a temporary vanilla garment solve proxy without altering authored output topology."""
    V=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64)
    if len(V)<3 or len(F)<1:return dict(data)
    N=np.asarray(data.get("N",[]),dtype=np.float64);UV_raw=data.get("UV");UV=None if UV_raw is None else np.asarray(UV_raw,dtype=np.float64)
    W=np.asarray(data.get("W",[]),dtype=np.float64)
    if W.ndim!=2 or W.shape[0]!=len(V):return dict(data)
    vertices=[row.copy() for row in V];normals=[row.copy() for row in N] if len(N)==len(V) else None;uvs=[row.copy() for row in UV] if UV is not None and len(UV)==len(V) else None;weights=[row.copy() for row in W]
    edges:dict[tuple[int,int],int]={}
    def midpoint(a:int,b:int)->int:
        key=(a,b) if a<b else (b,a);existing=edges.get(key)
        if existing is not None:return existing
        index=len(vertices);edges[key]=index;vertices.append((V[a]+V[b])*.5)
        if normals is not None:
            value=(N[a]+N[b])*.5;length=float(np.linalg.norm(value));normals.append(value/length if length>1e-12 else value)
        if uvs is not None:uvs.append((UV[a]+UV[b])*.5)
        value=(W[a]+W[b])*.5;total=float(np.sum(value));weights.append(value/total if total>1e-12 else value)
        return index
    faces=[]
    for a,b,c in F:
        a=int(a);b=int(b);c=int(c);ab=midpoint(a,b);bc=midpoint(b,c);ca=midpoint(c,a)
        faces.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
    result=dict(data);result["V"]=np.asarray(vertices,dtype=np.float64);result["F"]=np.asarray(faces,dtype=np.int64);result["W"]=np.asarray(weights,dtype=np.float64)
    if normals is not None:result["N"]=np.asarray(normals,dtype=np.float64)
    if uvs is not None:result["UV"]=np.asarray(uvs,dtype=np.float64)
    result["_ravafit_original_vertex_count"]=int(len(V));return result


def _vanilla_garment_edge_p95(data: dict[str, Any]) -> float:
    V=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64)
    if len(V)<3 or len(F)<1:return 0.0
    edges=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]));edges=np.sort(edges,axis=1);edges=np.unique(edges,axis=0)
    return float(np.percentile(np.linalg.norm(V[edges[:,0]]-V[edges[:,1]],axis=1),95)) if len(edges) else 0.0


def _indexed_render_view(data: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
    """Return the triangle-referenced view of one mesh and the exact raw expansion map."""
    V=np.asarray(data.get('V',[]));F=np.asarray(data.get('F',[]),dtype=np.int64)
    if V.ndim!=2 or V.shape[1]!=3 or F.ndim!=2 or (len(F) and F.shape[1]!=3):
        raise ValueError(f"Indexed garment view requires V Nx3 and F Mx3, got {V.shape} / {F.shape}")
    if len(F)==0:
        return dict(data),{'raw_vertices':int(len(V)),'indexed_vertices':int(len(V)),'unindexed_vertices':0,'fraction':0.0}
    if int(np.min(F))<0 or int(np.max(F))>=len(V):raise ValueError('Garment indices reference vertices outside the raw storage range.')
    ids=np.unique(F.reshape(-1));remap=np.full(len(V),-1,dtype=np.int64);remap[ids]=np.arange(len(ids),dtype=np.int64)
    out=dict(data);out['V']=np.asarray(data['V'])[ids].copy();out['F']=remap[F]
    for key in ('N','UV','W'):
        value=data.get(key)
        if value is not None and len(value)==len(V):out[key]=np.asarray(value)[ids].copy()
    out['_ravafit_indexed_raw_ids']=ids;out['_ravafit_raw_vertex_count']=int(len(V));out['_ravafit_indexed_contract']=True
    missing=int(len(V)-len(ids));report={'raw_vertices':int(len(V)),'indexed_vertices':int(len(ids)),'unindexed_vertices':missing,'fraction':float(missing/max(len(V),1))}
    return out,report


class _IndexedRenderGarmentSource:
    """Read-only solve view that excludes non-rendered raw storage vertices from garment topology."""
    def __init__(self,source: Any,body_mesh_names:set[str]):
        self.source=source;self.js=source.js;self.body_mesh_names=set(body_mesh_names);self._data={};self.report={}
        for name in source.mesh_names():
            if not name:continue
            data=source.data(name)
            if name in self.body_mesh_names:
                self._data[name]=data;continue
            indexed,report=_indexed_render_view(data);self._data[name]=indexed;self.report[name]=report
    def mesh_names(self):return self.source.mesh_names()
    def data(self,name):return self._data[name]


def _collapse_indexed_render_garment_solution(view: _IndexedRenderGarmentSource, positions: dict[str,np.ndarray], skinning: dict[str,dict[str,Any]]):
    """Expand solved rendered vertices back into the exact authored raw MDL/GLB storage layout."""
    out_pos={};out_skin={}
    for name,solved in positions.items():
        indexed=view.data(name);raw=view.source.data(name);ids=np.asarray(indexed.get('_ravafit_indexed_raw_ids'),dtype=np.int64)
        if ids.ndim!=1:
            out_pos[name]=np.asarray(solved,dtype=np.float64).copy();out_skin[name]=skinning[name];continue
        raw_v=np.asarray(raw['V'],dtype=np.float64).copy();candidate=np.asarray(solved,dtype=np.float64)
        if candidate.shape!=(len(ids),3):raise ValueError(f'Indexed solve for {name} returned {candidate.shape}, expected {(len(ids),3)}')
        raw_v[ids]=candidate;out_pos[name]=raw_v
        payload=skinning.get(name)
        if payload is not None:
            raw_w=np.asarray(raw['W'],dtype=np.float64).copy();solved_w=np.asarray(payload['weights'],dtype=np.float64)
            if solved_w.shape[0]!=len(ids):raise ValueError(f'Indexed skinning for {name} returned {solved_w.shape[0]} rows, expected {len(ids)}')
            raw_w[ids]=solved_w;out_skin[name]={'weights':raw_w,'joint_names':list(raw['joint_names']),'stage':payload.get('stage',{})}
    return out_pos,out_skin


class _DenseVanillaGarmentSource:
    """Read-only solve view that densifies sparse vanilla garment geometry only for B14 fitting."""
    def __init__(self,source: GLB,body_mesh_names:set[str],target_edge_p95:float=.060,max_passes:int=2,max_faces:int=24000):
        self.source=source;self.js=source.js;self._data:dict[str,dict[str,Any]]={};self.original_counts:dict[str,int]={};self.report:dict[str,Any]={}
        for name in source.mesh_names():
            if not name:continue
            data=source.data(name);original_count=int(len(np.asarray(data.get("V",[]))));self.original_counts[name]=original_count
            if name in body_mesh_names or original_count<3 or len(np.asarray(data.get("F",[])))<1:
                self._data[name]=data;continue
            work=dict(data);before_faces=int(len(np.asarray(work.get("F",[]))));before_p95=_vanilla_garment_edge_p95(work);passes=0
            while passes<max_passes and _vanilla_garment_edge_p95(work)>target_edge_p95 and int(len(np.asarray(work.get("F",[]))))*4<=max_faces:
                work=_subdivide_vanilla_garment_data_once(work);passes+=1
            self._data[name]=work
            self.report[name]={"original_vertices":original_count,"solve_vertices":int(len(np.asarray(work.get("V",[])))),"original_faces":before_faces,"solve_faces":int(len(np.asarray(work.get("F",[])))),"edge_p95_before_mm":before_p95*1000.0,"edge_p95_after_mm":_vanilla_garment_edge_p95(work)*1000.0,"subdivision_passes":passes}
    def mesh_names(self):return self.source.mesh_names()
    def data(self,name):return self._data[name]


def _collapse_dense_vanilla_garment_solution(proxy:_DenseVanillaGarmentSource,positions:dict[str,np.ndarray],skinning:dict[str,dict[str,Any]]):
    collapsed_positions={};collapsed_skinning={}
    for name,value in positions.items():
        count=int(proxy.original_counts.get(name,len(value)));collapsed_positions[name]=np.asarray(value,dtype=np.float64)[:count].copy()
    for name,payload in skinning.items():
        count=int(proxy.original_counts.get(name,len(np.asarray(payload.get("weights",[])))));row=dict(payload);row["weights"]=np.asarray(payload["weights"],dtype=np.float64)[:count].copy();collapsed_skinning[name]=row
    return collapsed_positions,collapsed_skinning


def _complete_vanilla_target_body_plan(cache:dict[str,Any],present_slots:list[str]):
    """Vanilla garments may never delete geometry from the selected target body."""
    triangles=[];source_triangles=[];reports=[]
    for pair in cache.get("slot_pairs",[]):
        slot=str(pair.get("slot") or "Body");V=np.asarray(pair.get("target_literal_V",[]),dtype=np.float64);F=np.asarray(pair.get("target_literal_F",[]),dtype=np.int64);SV=np.asarray(pair.get("source_literal_V",[]),dtype=np.float64);SF=np.asarray(pair.get("source_literal_F",[]),dtype=np.int64)
        if len(V) and len(F):triangles.append(V[F])
        if len(SV) and len(SF):source_triangles.append(SV[SF])
        reports.append({"slot":slot,"source_body_present":slot in present_slots,"status":"vanilla full-target authority; suppression disabled","accepted_components":[],"suppressed_target_triangles":0,"native_meshes":[]})
    collision=np.vstack(triangles) if triangles else np.zeros((0,3,3),dtype=np.float64);source_collision=np.vstack(source_triangles) if source_triangles else np.zeros((0,3,3),dtype=np.float64)
    return {"enabled":False,"policy":"vanilla conversions transplant the selected target body complete; garment fitting must accommodate the body","slots":reports,"native_suppression":[],"_collision_triangles":collision,"_source_collision_triangles":source_collision}

def _triangles_from_surface(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Body collision surface is empty.")
    return np.asarray(vertices, np.float64)[np.asarray(faces, np.int64)]


def _sanitise_strict_b14_surface_triangles(triangles: np.ndarray, surface_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only unusable faces before crossing the untouched frozen-B14 boundary.

    Frozen B14 deliberately uses its historical ``trimesh.triangles.closest_point`` path.
    A zero-area/near-collinear support triangle can make that routine divide by zero and return
    NaN even when every input vertex is finite.  Such a face has no usable surface area, so
    dropping it does not change the represented collision surface; it only prevents invalid
    closest-point candidates from entering the historical optimiser.
    """
    tri=np.asarray(triangles,dtype=np.float64)
    if tri.ndim!=3 or tri.shape[1:]!=(3,3):
        raise ValueError(f"{surface_name} strict B14 surface has invalid triangle shape {tri.shape}; expected (N, 3, 3).")
    if len(tri)==0:
        raise ValueError(f"{surface_name} strict B14 surface is empty.")

    finite=np.all(np.isfinite(tri),axis=(1,2))
    keep=finite.copy()
    degenerate=np.zeros(len(tri),dtype=bool)
    finite_ids=np.flatnonzero(finite)
    if len(finite_ids):
        candidate=tri[finite_ids]
        ab=candidate[:,1]-candidate[:,0];ac=candidate[:,2]-candidate[:,0];bc=candidate[:,2]-candidate[:,1]
        edge_sq=np.maximum.reduce((np.sum(ab*ab,axis=1),np.sum(ac*ac,axis=1),np.sum(bc*bc,axis=1)))
        double_area=np.linalg.norm(np.cross(ab,ac),axis=1)
        # Relative to each face's own edge scale.  This rejects collapsed/near-collinear faces
        # without imposing a world-space minimum that could erase legitimately tiny detail.
        local_degenerate=(edge_sq<=0.0)|(double_area<=edge_sq*1.0e-12)
        degenerate[finite_ids]=local_degenerate
        keep[finite_ids]&=~local_degenerate

    cleaned=np.ascontiguousarray(tri[keep],dtype=np.float64)
    if len(cleaned)==0:
        raise ValueError(
            f"{surface_name} strict B14 surface contains no usable triangles after sanitisation "
            f"(input={len(tri)}, non_finite={int(np.sum(~finite))}, degenerate={int(np.sum(degenerate))})."
        )
    report={
        "input_triangle_count":int(len(tri)),
        "output_triangle_count":int(len(cleaned)),
        "dropped_non_finite_triangle_count":int(np.sum(~finite)),
        "dropped_degenerate_triangle_count":int(np.sum(degenerate)),
        "changed":bool(np.any(~keep)),
    }
    return cleaned,report


def _rigid_fit_points(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None):
    source = np.asarray(source, dtype=np.float64); target = np.asarray(target, dtype=np.float64)
    if len(source) != len(target) or len(source) < 3:
        raise ValueError("Rigid preservation requires at least three paired points.")
    w = np.ones(len(source), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) != len(source):
        raise ValueError("Rigid preservation weights do not match point count.")
    w = np.maximum(w, 0.0); w /= max(float(w.sum()), 1e-12)
    cs = (source * w[:, None]).sum(axis=0); ct = (target * w[:, None]).sum(axis=0)
    a = source - cs; b = target - ct
    u, _, vt = np.linalg.svd((a * w[:, None]).T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1.0; r = vt.T @ u.T
    t = ct - cs @ r.T
    return r, t


def _solve_component_rigid_preservation(w: dict[str, Any], base: np.ndarray, labels: np.ndarray):
    """Move connected structured components onto the new body without deforming them."""
    out = np.asarray(w["V"], dtype=np.float64).copy(); details = []
    for component in sorted(set(np.asarray(labels, dtype=np.int64).tolist())):
        ids = np.flatnonzero(labels == component)
        if len(ids) < 3:
            out[ids] = base[ids]; details.append({"component": int(component), "vertices": int(len(ids)), "mode": "direct"}); continue
        r, t = _rigid_fit_points(w["V"][ids], base[ids]); out[ids] = w["V"][ids] @ r.T + t
        details.append({"component": int(component), "vertices": int(len(ids)), "mode": "rigid"})
    return out, {"mode": "rigid_component_preservation", "components": details}




def _mesh_geodesic_from_vertices(vertices: np.ndarray, faces: np.ndarray, seeds: np.ndarray):
    V=np.asarray(vertices,dtype=np.float64); F=np.asarray(faces,dtype=np.int64); seeds=np.asarray(seeds,dtype=np.int64).reshape(-1)
    edges=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])); edges=np.sort(edges,axis=1); edges=np.unique(edges,axis=0); lengths=np.linalg.norm(V[edges[:,0]]-V[edges[:,1]],axis=1)
    row=np.concatenate((edges[:,0],edges[:,1])); col=np.concatenate((edges[:,1],edges[:,0])); val=np.concatenate((lengths,lengths)); graph=coo_matrix((val,(row,col)),shape=(len(V),len(V))).tocsr()
    dist=dijkstra(graph,directed=False,indices=seeds); return np.asarray(dist if dist.ndim==1 else np.min(dist,axis=0),dtype=np.float64)


def _smooth_b14_panel_flow_curve(w: dict[str, Any], solved: np.ndarray, source_surface_vertices: np.ndarray):
    """Smooth a loose panel's B14 motion from its attachment towards the free edge."""
    V=np.asarray(w["V"],dtype=np.float64); F=np.asarray(w["F"],dtype=np.int64); U=np.asarray(solved,dtype=np.float64)
    body_distance=cKDTree(np.asarray(source_surface_vertices,dtype=np.float64)).query(V,k=1)[0]; cutoff=max(float(np.percentile(body_distance,10.0)),1e-6); anchors=np.flatnonzero(body_distance<=cutoff)
    if len(anchors)<3: anchors=np.argsort(body_distance)[:max(3,min(len(body_distance),16))]
    aw=np.exp(-((body_distance[anchors]/cutoff)**2)); r,t=_rigid_fit_points(V[anchors],U[anchors],aw); rigid=V@r.T+t; residual=U-rigid
    geo=_mesh_geodesic_from_vertices(V,F,anchors); finite=np.isfinite(geo); geo[~finite]=float(np.max(geo[finite])) if np.any(finite) else 0.0; scale=max(float(np.percentile(geo,99.5)),1e-6); s=np.clip(geo/scale,0.0,1.0)
    # Near-body motion has priority; distant free edges only shape the curve.
    fit_w=0.20+0.80*np.exp(-((body_distance/max(0.055,float(np.median(body_distance))*1.25))**2))
    smooth=np.zeros_like(residual)
    for axis in range(3):
        coef=np.polynomial.polynomial.polyfit(s,residual[:,axis],deg=3,w=fit_w); smooth[:,axis]=np.polynomial.polynomial.polyval(s,coef)
    local=np.exp(-((s/0.16)**2))[:,None]
    out=rigid+smooth+local*(residual-smooth)
    return out,{"mode":"b14_smooth_flow_curve","anchor_vertices":int(len(anchors)),"geodesic_length_mm":scale*1000.0,"body_distance_p50_mm":float(np.median(body_distance)*1000.0),"polynomial_degree":3}


def _use_b14_smooth_flow_curve(features: dict[str, Any], details: list[dict[str, Any]]) -> bool:
    if int(features.get("component_count",0))!=1 or not details: return False
    extent=np.asarray(details[0].get("extent",[0.0,0.0,0.0]),dtype=np.float64); clearance=float(features.get("source_clearance_median_mm",0.0))
    return 22.0 <= clearance < 80.0 and float(np.max(extent)) >= 0.30


def _solve_anchored_free_panel(w: dict[str, Any], base: np.ndarray, source_surface_vertices: np.ndarray):
    """Move a hanging panel from its body-near attachment while keeping the free drape."""
    source_surface_vertices = np.asarray(source_surface_vertices, dtype=np.float64)
    if len(source_surface_vertices) == 0:
        raise ValueError("Anchored free-panel solve requires source body surface vertices.")
    distance = cKDTree(source_surface_vertices).query(np.asarray(w["V"], dtype=np.float64), k=1)[0]
    cutoff = float(np.percentile(distance, 10.0)); cutoff = max(cutoff, 1e-6)
    anchors = distance <= cutoff
    if int(np.count_nonzero(anchors)) < 3:
        anchors = np.argsort(distance)[:max(3, min(len(distance), 16))]
    anchor_distance = distance[anchors]
    anchor_weights = np.exp(-((anchor_distance / cutoff) ** 2))
    r, t = _rigid_fit_points(w["V"][anchors], base[anchors], anchor_weights)
    rigid = np.asarray(w["V"], dtype=np.float64) @ r.T + t
    sigma = float(np.clip(max(0.040, cutoff * 1.25), 0.040, 0.060))
    alpha = np.exp(-((distance / sigma) ** 2))[:, None]
    out = rigid + alpha * (base - rigid)
    return out, {
        "mode": "anchored_free_panel",
        "anchor_vertices": int(np.count_nonzero(distance <= cutoff)),
        "anchor_cutoff_mm": cutoff * 1000.0,
        "blend_sigma_mm": sigma * 1000.0,
        "body_distance_p50_mm": float(np.median(distance) * 1000.0),
        "body_distance_p95_mm": float(np.percentile(distance, 95) * 1000.0),
    }


def _use_anchored_free_panel(features: dict[str, Any], details: list[dict[str, Any]]) -> bool:
    if int(features.get("component_count", 0)) != 1 or not details:
        return False
    extent = np.asarray(details[0].get("extent", [0.0, 0.0, 0.0]), dtype=np.float64)
    clearance = float(features.get("source_clearance_median_mm", 0.0))
    return clearance >= 80.0 and float(np.max(extent)) >= 0.30


def _peer_shell_component_ids(details: list[dict[str, Any]]) -> list[int]:
    """Find substantial disconnected shells that should solve independently."""
    shells=[d for d in details if str(d.get("kind", "")).casefold()=="shell" and int(d.get("n",0))>=3]
    if len(shells)<2:return []
    largest=max(int(d.get("n",0)) for d in shells)
    largest_extent=max(float(np.max(np.asarray(d.get("extent",[0.0,0.0,0.0]),dtype=np.float64))) for d in shells)
    # Treat tiny dense detached detail as rigid when deforming it would only add noise.
    min_vertices=max(24,int(np.ceil(largest*0.30)))
    min_extent=max(0.012,largest_extent*0.25)
    peers=[int(d["component"]) for d in shells if int(d.get("n",0))>=min_vertices and float(np.max(np.asarray(d.get("extent",[0.0,0.0,0.0]),dtype=np.float64)))>=min_extent]
    return peers if len(peers)>=2 else []



def _fresh_process_peer_shell_group(source: Any, w: dict[str, Any], Wg: np.ndarray, base: np.ndarray, contact: np.ndarray, X: np.ndarray, Y: np.ndarray, BW: np.ndarray, NS: np.ndarray, NT: np.ndarray, names: list[str], axes: np.ndarray, source_tri: np.ndarray, target_tri: np.ndarray, labels: np.ndarray, features: dict[str, Any], behavior: str, peer_ids: list[int]):
    """Solve independent peer roots in isolated workers launched by a clean supervisor."""
    if behavior not in {"constructed_close_shell","stand_off_structured_shell"}:raise ValueError(f"Fresh peer group is unsupported for {behavior}")
    worker=_MODULE_DIR/"peer_shell_worker.py";supervisor=_MODULE_DIR/"peer_group_supervisor.py"
    if not worker.exists() or not supervisor.exists():raise FileNotFoundError(worker if not worker.exists() else supervisor)
    work_dir=Path(tempfile.mkdtemp(prefix="ravafit-peer-group-"));input_path=work_dir/"input.npz";manifest_path=work_dir/"manifest.json";supervisor_report=work_dir/"supervisor.json";started=time.perf_counter();results=[]
    try:
        np.savez(input_path,w_V=np.asarray(w["V"],dtype=np.float64),w_F=np.asarray(w["F"],dtype=np.int64),w_W=np.asarray(w.get("W",np.zeros((len(w["V"]),0))),dtype=np.float64),Wg=np.asarray(Wg,dtype=np.float64),base=np.asarray(base,dtype=np.float64),contact=np.asarray(contact,dtype=np.int64),X=np.asarray(X,dtype=np.float64),Y=np.asarray(Y,dtype=np.float64),BW=np.asarray(BW,dtype=np.float64),NS=np.asarray(NS,dtype=np.float64),NT=np.asarray(NT,dtype=np.float64),axes=np.asarray(axes,dtype=np.float64),source_tri=np.asarray(source_tri,dtype=np.float64),target_tri=np.asarray(target_tri,dtype=np.float64),labels=np.asarray(labels,dtype=np.int64))
        rows=[]
        for peer_id in peer_ids:
            peer_features=dict(features);peer_features["root_component"]=int(peer_id)
            meta_path=work_dir/f"meta-{int(peer_id)}.json";output_path=work_dir/f"output-{int(peer_id)}.npz";stage_path=work_dir/f"stage-{int(peer_id)}.json"
            meta={"source_js":source.js,"w_joint_names":list(w.get("joint_names",[])),"names":list(names),"features":_strict_json_safe(peer_features),"behavior":behavior}
            meta_path.write_text(json.dumps(meta,separators=(",",":")),encoding="utf-8")
            rows.append({"peer_id":int(peer_id),"meta":str(meta_path),"output":str(output_path),"stage":str(stage_path)})
        manifest_path.write_text(json.dumps({"peers":rows},separators=(",",":")),encoding="utf-8")
        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0) if sys.platform.startswith("win") else 0;env=os.environ.copy();env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
        proc=subprocess.Popen([sys.executable,str(supervisor),str(worker),str(input_path),str(manifest_path),str(supervisor_report)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=creationflags,env=env)
        try:stdout,stderr=proc.communicate(timeout=max(90,55*len(peer_ids)))
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate();raise RuntimeError(f"peer supervisor timed out: {(stderr or stdout)[-2000:]}")
        sup=json.loads(supervisor_report.read_text(encoding="utf-8")) if supervisor_report.exists() else {}
        if proc.returncode!=0 or not sup.get("ok",False):raise RuntimeError(f"peer supervisor failed: {sup or (stderr or stdout)[-2000:]}")
        report_map={int(r["peer_id"]):r for r in sup.get("reports",[])}
        for row in rows:
            peer_id=int(row["peer_id"]);output_path=Path(row["output"]);stage_path=Path(row["stage"])
            with np.load(output_path,allow_pickle=False) as out:
                peer_vertices=np.asarray(out["peer_vertices"],dtype=np.int64).copy();peer_positions=np.asarray(out["peer_positions"],dtype=np.float64).copy()
            stage_payload=json.loads(stage_path.read_text(encoding="utf-8"));wr=report_map.get(peer_id,{})
            results.append((peer_id,peer_vertices,peer_positions,stage_payload.get("stage",{}),{"mode":"fresh_process_supervised","wall_sec":wr.get("wall_sec"),"worker_sec":wr.get("worker_sec"),"pid":wr.get("pid"),"attempts":wr.get("attempts",1)}))
        return results,{"mode":"fresh_process_supervised_group","peer_count":len(results),"wall_sec":time.perf_counter()-started,"supervisor_sec":sup.get("elapsed_sec"),"wave_size":1}
    finally:
        shutil.rmtree(work_dir,ignore_errors=True)

def _fresh_process_shell_solve(source: Any, w: dict[str, Any], Wg: np.ndarray, base: np.ndarray, contact: np.ndarray, X: np.ndarray, Y: np.ndarray, BW: np.ndarray, NS: np.ndarray, NT: np.ndarray, names: list[str], axes: np.ndarray, source_tri: np.ndarray, target_tri: np.ndarray, labels: np.ndarray, features: dict[str, Any], behavior: str):
    """Run one heavy Torch shell solve in a fresh process using the exact same B14 function."""
    if behavior not in {"constructed_close_shell","stand_off_structured_shell","body_following_flexible_layer"}:
        raise ValueError(f"Fresh shell solve is unsupported for {behavior}")
    worker=_MODULE_DIR/"shell_solve_worker.py"
    if not worker.exists():raise FileNotFoundError(worker)
    work_dir=Path(tempfile.mkdtemp(prefix="ravafit-shell-")); input_path=work_dir/"input.npz";meta_path=work_dir/"meta.json";output_path=work_dir/"output.npz";stage_path=work_dir/"stage.json";started=time.perf_counter()
    try:
        np.savez(input_path,w_V=np.asarray(w["V"],dtype=np.float64),w_F=np.asarray(w["F"],dtype=np.int64),w_W=np.asarray(w.get("W",np.zeros((len(w["V"]),0))),dtype=np.float64),Wg=np.asarray(Wg,dtype=np.float64),base=np.asarray(base,dtype=np.float64),contact=np.asarray(contact,dtype=np.int64),X=np.asarray(X,dtype=np.float64),Y=np.asarray(Y,dtype=np.float64),BW=np.asarray(BW,dtype=np.float64),NS=np.asarray(NS,dtype=np.float64),NT=np.asarray(NT,dtype=np.float64),axes=np.asarray(axes,dtype=np.float64),source_tri=np.asarray(source_tri,dtype=np.float64),target_tri=np.asarray(target_tri,dtype=np.float64),labels=np.asarray(labels,dtype=np.int64))
        meta={"source_js":source.js,"w_joint_names":list(w.get("joint_names",[])),"names":list(names),"features":_strict_json_safe(features),"behavior":behavior}
        meta_path.write_text(json.dumps(meta,separators=(",",":")),encoding="utf-8")
        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0) if sys.platform.startswith("win") else 0
        worker_env=os.environ.copy();worker_env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
        proc=subprocess.Popen([sys.executable,str(worker),str(input_path),str(meta_path),str(output_path),str(stage_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=creationflags,env=worker_env)
        try:stdout,stderr=proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate();raise RuntimeError(f"shell worker timed out: {(stderr or stdout)[-2000:]}")
        if proc.returncode!=0 or not output_path.exists():raise RuntimeError(f"shell worker exited {proc.returncode}: {(stderr or stdout)[-2000:]}")
        out=np.load(output_path,allow_pickle=False);U=np.asarray(out["U"],dtype=np.float64);ids=np.asarray(out["ids"],dtype=np.int64);F=np.asarray(out["F"],dtype=np.int64)
        stage_payload=json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else {}
        stage=stage_payload.get("stage",{})
        return U,ids,F,{"mode":"fresh_process_exact_shell","solve":stage,"worker":{"wall_sec":time.perf_counter()-started,"worker_sec":stage_payload.get("elapsed_sec"),"pid":stage_payload.get("pid")}}
    finally:
        shutil.rmtree(work_dir,ignore_errors=True)


def _solve_single_shell_fresh_or_legacy(source: Any, w: dict[str, Any], Wg: np.ndarray, base: np.ndarray, contact: np.ndarray, X: np.ndarray, Y: np.ndarray, BW: np.ndarray, NS: np.ndarray, NT: np.ndarray, names: list[str], axes: np.ndarray, source_tri: np.ndarray, target_tri: np.ndarray, labels: np.ndarray, features: dict[str, Any], behavior: str):
    try:
        return _fresh_process_shell_solve(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features,behavior)
    except Exception as ex:
        print(f"[RavaFit perf] fresh single-shell worker failed for {behavior}; using exact in-process fallback: {ex}",file=sys.stderr,flush=True)
        if behavior=="stand_off_structured_shell":
            return solve_standoff(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features)
        if behavior=="body_following_flexible_layer":
            return solve_flexible(w,Wg,base,axes,target_tri,labels,features)
        return solve_constructed_close(source,source,w,Wg,base,X,Y,BW,NS,NT,names,axes,cKDTree(X),source_tri,target_tri,labels,features)


def _solve_peer_shell_group_fresh(source: Any, w: dict[str, Any], Wg: np.ndarray, base: np.ndarray, contact: np.ndarray, X: np.ndarray, Y: np.ndarray, BW: np.ndarray, NS: np.ndarray, NT: np.ndarray, names: list[str], axes: np.ndarray, source_tri: np.ndarray, target_tri: np.ndarray, labels: np.ndarray, features: dict[str, Any], behavior: str, peer_ids: list[int]):
    """Solve every substantial peer shell before running the one shared assembly tail."""
    root_component=int(features.get("root_component",-1));solved=[];reports=[];group_report=None
    try:
        fresh_rows,group_report=_fresh_process_peer_shell_group(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features,behavior,peer_ids)
        for peer_id,peer_vertices,peer_positions,peer_stage,worker_report in fresh_rows:
            solved.append((peer_id,peer_vertices,peer_positions,peer_stage));reports.append({"component":peer_id,"vertices":int(len(peer_vertices)),"stage":peer_stage,"worker":worker_report})
    except Exception as group_error:
        print(f"[RavaFit perf] fresh peer group failed for {behavior}: {group_error}",file=sys.stderr,flush=True)
        # Exact legacy fallback: solve each independent peer with the same root-only functions.
        for peer_id in peer_ids:
            peer_features=dict(features);peer_features["root_component"]=int(peer_id)
            if behavior=="stand_off_structured_shell":candidate,peer_vertices,_,peer_stage=solve_standoff_root_only(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,peer_features)
            else:candidate,peer_vertices,_,peer_stage=solve_constructed_close_root_only(source,source,w,Wg,base,X,Y,BW,NS,NT,names,axes,cKDTree(X),source_tri,target_tri,labels,peer_features)
            peer_vertices=np.asarray(peer_vertices,dtype=np.int64);peer_positions=np.asarray(candidate,dtype=np.float64)[peer_vertices].copy();del candidate
            solved.append((int(peer_id),peer_vertices,peer_positions,peer_stage));reports.append({"component":int(peer_id),"vertices":int(len(peer_vertices)),"stage":peer_stage,"worker":{"mode":"in_process_group_fallback","error":str(group_error)}})
        group_report={"mode":"in_process_group_fallback","error":str(group_error)}
    root_matches=[row for row in solved if row[0]==root_component]
    if not root_matches:raise ValueError(f"Peer-shell group did not solve root component {root_component}")
    _,root_vertices,root_positions,root_stage=root_matches[0]
    axial=.65 if behavior=="stand_off_structured_shell" else .60
    preassembly=suppress_bone_axis_drift(np.asarray(w["V"],dtype=np.float64),np.asarray(base,dtype=np.float64),np.asarray(Wg,dtype=np.float64),np.asarray(axes,dtype=np.float64),axial)
    preassembly[root_vertices]=root_positions
    assembled,kinds,edges,assembly_history=refine_peer_assembly(w,preassembly,labels,behavior,root_component)
    result=np.asarray(assembled,dtype=np.float64).copy()
    for _,vertices,positions,stage in solved:result[vertices]=positions
    stage={"mode":"independent_peer_shells","initial_root":root_stage,"assembly_last":assembly_history[-1] if assembly_history else {"skipped":True,"reason":"no assembly history"},"peers":reports,"worker_group":group_report,"fresh_process_workers":group_report.get("mode")=="fresh_process_group"}
    return result,root_vertices,stage

def _use_fragmented_decorated_shell_frame(behavior: str, features: dict[str, Any], details: list[dict[str, Any]]) -> bool:
    """Avoid a global shell optimiser when one cloth root carries thousands of rigid ornaments.

    This is intentionally a very high threshold.  Ordinary multi-component B14 garments, including
    Dragon, keep their existing shell solve.  The conservative component frame only seeds Runtime 11's
    source-differential structural finaliser; it does not become a replacement fitting algorithm.
    """
    if behavior not in {"stand_off_structured_shell","constructed_close_shell"} or not details:return False
    component_count=int(features.get("component_count",0));root_vertices=int(features.get("root_vertices",0));root_fraction=float(features.get("root_fraction",1.0))
    if component_count<512 or root_vertices<256 or root_fraction>.35:return False
    rigid_count=sum(1 for row in details if str(row.get("kind","")).casefold()=="rigid")
    return rigid_count>=512 and (rigid_count/max(component_count,1))>=.90


def _use_rigid_component_preservation(behavior: str, features: dict[str, Any], details: list[dict[str, Any]]) -> bool:
    if not details:
        return False
    maximum_extent = max(float(np.max(np.asarray(d.get("extent", [0.0, 0.0, 0.0]), dtype=np.float64))) for d in details)
    if behavior == "conservative_component_assembly":
        vertical_extent = max(float(np.asarray(d.get("extent", [0.0, 0.0, 0.0]), dtype=np.float64)[1]) for d in details)
        shells = [d for d in details if str(d.get("kind", "")).casefold() == "shell"]
        if not shells:
            return maximum_extent <= 0.30 and vertical_extent <= 0.12
        # Compact assemblies already clear of the source body are details, not cloth to shrink-wrap.
        clearance = float(features.get("source_clearance_median_mm", 0.0))
        return clearance >= 8.0 and maximum_extent <= 0.30 and vertical_extent <= 0.12
    return behavior == "body_following_flexible_layer" and int(features.get("component_count", 0)) == 1 and float(features.get("boundary_fraction", 1.0)) == 0.0 and maximum_extent <= 0.13


def _residual_surface_clearance_guard(vertices: np.ndarray, faces: np.ndarray, target_triangles: np.ndarray):
    """Apply the final small non-penetration guard."""
    before=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    candidate,report=_b14_residual_collision_polish(before,F,target_triangles,margin=.00035,iterations=10,blend=.50,max_push=.0015,k=24)
    displacement=np.asarray(candidate,dtype=np.float64)-before
    if not np.any(np.linalg.norm(displacement,axis=1)>1e-10):
        report.update({"accepted_alpha":1.0,"orientation_guarded":False})
        return before,report

    a0=np.cross(before[F[:,1]]-before[F[:,0]],before[F[:,2]]-before[F[:,0]])
    n0=np.linalg.norm(a0,axis=1);alpha=1.0;accepted=before
    for _ in range(7):
        trial=before+alpha*displacement
        a=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]])
        n=np.linalg.norm(a,axis=1);dot=np.sum(a0*a,axis=1)/np.maximum(n0*n,1e-24);area=n/np.maximum(n0,1e-12)
        if bool(np.all(dot>0.05) and np.all(area>0.15)):
            accepted=trial;break
        alpha*=0.5
    report.update({"accepted_alpha":float(alpha),"orientation_guarded":bool(alpha<1.0)})
    return accepted,report



def _rigid_component_clearance_guard(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, target_triangles: np.ndarray, max_component_extent: float | None = None, eligible_components: set[int] | None = None):
    """Clear rigid components by translation only."""
    V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64);lab=np.asarray(labels,dtype=np.int64)
    if len(V)==0 or len(F)==0:return V,{"moved_components":0,"unresolved_components":0,"minimum_after_mm":None}
    bary=np.asarray([(i/6.0,j/6.0,(6-i-j)/6.0) for i in range(7) for j in range(7-i)],dtype=np.float64)
    moved=[];unresolved=[];mins=[]
    for component in sorted(set(lab.tolist())):
        if eligible_components is not None and int(component) not in eligible_components: continue
        vids=np.flatnonzero(lab==component)
        if len(vids)<3:continue
        if max_component_extent is not None:
            extent=float(np.max(np.ptp(V[vids],axis=0))) if len(vids) else 0.0
            if extent>float(max_component_extent):continue
        face_ids=np.flatnonzero(np.all(lab[F]==component,axis=1))
        if len(face_ids)==0:continue
        Fc=F[face_ids];total=np.zeros(3,dtype=np.float64);before_min=None
        for outer in range(10):
            samples=np.einsum("bk,fkj->fbj",bary,V[Fc]).reshape(-1,3)
            _,normals,signed,_,_=_b14_nearest_surface(samples,target_triangles,k=32)
            before_min=float(np.min(signed)) if before_min is None else before_min
            desired=np.where(signed<-.00010,.00085,.00050)
            bad=signed<desired-1e-6
            if not np.any(bad):break
            N=np.asarray(normals[bad],dtype=np.float64);d=np.asarray(desired[bad]-signed[bad],dtype=np.float64)
            # Project rigid translation against the local clearance half-spaces.
            step=np.zeros(3,dtype=np.float64)
            for _ in range(12):
                changed=False
                order=np.argsort(d-N@step)[::-1]
                for j in order:
                    need=float(d[j]-np.dot(N[j],step))
                    if need>1e-6:
                        step+=N[j]*need;changed=True
                if not changed:break
            mag=float(np.linalg.norm(step))
            if not np.isfinite(mag) or mag<1e-9:break
            # Cap one clearance correction so a bad contact cannot launch a detail across the mesh.
            if mag>.006:step*=.006/mag
            V[vids]+=step;total+=step
            if float(np.linalg.norm(total))>.018:break
        samples=np.einsum("bk,fkj->fbj",bary,V[Fc]).reshape(-1,3);_,_,final_signed,_,_=_b14_nearest_surface(samples,target_triangles,k=32)
        final_min=float(np.min(final_signed));mins.append(final_min)
        if float(np.linalg.norm(total))>1e-9:moved.append({"component":int(component),"vertices":int(len(vids)),"translation_mm":(total*1000.0).tolist(),"move_mm":float(np.linalg.norm(total)*1000.0),"before_min_mm":float(before_min*1000.0),"after_min_mm":float(final_min*1000.0)})
        if final_min<.00049:unresolved.append({"component":int(component),"after_min_mm":float(final_min*1000.0)})
    return V,{"moved_components":int(len(moved)),"unresolved_components":int(len(unresolved)),"minimum_after_mm":float(min(mins)*1000.0) if mins else None,"moved":moved,"unresolved":unresolved}

def _continuous_surface_clearance_guard(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, source_triangles: np.ndarray, target_triangles: np.ndarray, preserve_source_orientation: bool = True):
    """Catch target-body intersections through garment face interiors."""
    Vsrc=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64)
    if len(V)==0 or len(F)==0:
        return V,{"affected_faces":0,"iterations":0,"sample_min_before_mm":None,"sample_min_after_mm":None,"max_vertex_move_mm":0.0}

    # Sample face interiors as well as vertices so sharp body features cannot poke through a face.
    bary=np.asarray([
        [1/3,1/3,1/3],
        [.5,.5,0.0],[0.0,.5,.5],[.5,0.0,.5],
        [.6,.2,.2],[.2,.6,.2],[.2,.2,.6],
    ],dtype=np.float64)
    source_samples=np.einsum("bk,fkj->fbj",bary,Vsrc[F]).reshape(-1,3)
    _,_,source_signed,source_distance,_=_b14_nearest_surface(source_samples,source_triangles,k=32)
    source_signed=source_signed.reshape(len(F),len(bary));source_distance=source_distance.reshape(len(F),len(bary))

    initial_samples=np.einsum("bk,fkj->fbj",bary,V[F]).reshape(-1,3)
    _,_,initial_signed,_,_=_b14_nearest_surface(initial_samples,target_triangles,k=32)
    initial_signed=initial_signed.reshape(len(F),len(bary))

    source_valid=source_signed>0.0
    significant=initial_signed<-.00010
    shallow=initial_signed<.00050
    # Target-body clearance always wins; source clipping is not permission to clip the target.
    constrained=shallow
    if not np.any(constrained):
        return V,{"affected_faces":0,"iterations":0,"sample_min_before_mm":float(np.min(initial_signed)*1000.0),"sample_min_after_mm":float(np.min(initial_signed)*1000.0),"max_vertex_move_mm":0.0,"source_contact_samples":0}

    desired=np.full_like(source_signed,.00050)
    source_close=source_distance<=.012
    authored=np.maximum(.00085,np.minimum(np.maximum(source_signed,0.0),.00115))
    desired=np.where(significant&source_close,authored,desired)
    desired=np.where(significant&~source_close,.00085,desired)

    active_face_ids=np.where(np.any(constrained,axis=1))[0]
    local_faces=F[active_face_ids];local_active=constrained[active_face_ids];local_desired=desired[active_face_ids]
    edges=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
    neighbours=[[] for _ in range(len(V))]
    for a,b in edges:neighbours[int(a)].append(int(b));neighbours[int(b)].append(int(a))

    original=V.copy()
    baseline_area=np.cross(original[F[:,1]]-original[F[:,0]],original[F[:,2]]-original[F[:,0]])
    baseline_norm=np.linalg.norm(baseline_area,axis=1)
    # Keep a rigid-aligned source copy as a handedness check.
    ref_r, ref_t = _rigid_fit_points(Vsrc, original)
    aligned_source = Vsrc @ ref_r.T + ref_t
    source_ref_area=np.cross(aligned_source[F[:,1]]-aligned_source[F[:,0]],aligned_source[F[:,2]]-aligned_source[F[:,0]])
    source_ref_norm=np.linalg.norm(source_ref_area,axis=1)
    source_baseline_dot=np.ones(len(F),dtype=np.float64)
    source_baseline_valid=(source_ref_norm>1e-12)&(baseline_norm>1e-12)
    source_baseline_dot[source_baseline_valid]=np.sum(source_ref_area[source_baseline_valid]*baseline_area[source_baseline_valid],axis=1)/np.maximum(source_ref_norm[source_baseline_valid]*baseline_norm[source_baseline_valid],1e-30)
    affected_faces=set();iterations=0;minimum_alpha=1.0
    for iteration in range(10):
        samples=np.einsum("bk,fkj->fbj",bary,V[local_faces]).reshape(-1,3)
        _,normals,signed,_,_=_b14_nearest_surface(samples,target_triangles,k=32)
        signed=signed.reshape(len(local_faces),len(bary));normals=normals.reshape(len(local_faces),len(bary),3)
        deficit=np.where(local_active,np.maximum(0.0,local_desired-signed),0.0)
        bad=np.max(deficit,axis=1)>1e-6
        if not np.any(bad):break
        iterations=iteration+1;bad_local=np.where(bad)[0];affected_faces.update(int(active_face_ids[x]) for x in bad_local)
        acc=np.zeros_like(V);weights=np.zeros(len(V),dtype=np.float64)
        for local_fi in bad_local:
            bi=int(np.argmax(deficit[local_fi]));required=float(deficit[local_fi,bi]);normal=normals[local_fi,bi]
            for corner,vi in enumerate(local_faces[local_fi]):
                weight=.35+.65*float(bary[bi,corner]);acc[vi]+=normal*required*weight;weights[vi]+=weight
        displacement=np.zeros_like(V);direct=weights>0;displacement[direct]=acc[direct]/weights[direct,None]
        for _ in range(2):
            smoothed=displacement.copy()
            for vi,nb in enumerate(neighbours):
                if not nb:continue
                avg=np.mean(displacement[nb],axis=0);smoothed[vi]=(.82*displacement[vi]+.18*avg) if direct[vi] else .15*avg
            displacement=smoothed
        magnitude=np.linalg.norm(displacement,axis=1);displacement*=np.minimum(1.0,.0015/np.maximum(magnitude,1e-12))[:,None]

        accepted=V;local_scale=1.0;blocked_faces=0
        guarded=displacement.copy()
        valid=baseline_norm>1e-12
        for _ in range(64):
            trial=V+guarded
            area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);norm=np.linalg.norm(area,axis=1)
            dot=np.ones(len(F),dtype=np.float64);ratio=np.ones(len(F),dtype=np.float64);source_dot=np.ones(len(F),dtype=np.float64)
            dot[valid]=np.sum(baseline_area[valid]*area[valid],axis=1)/np.maximum(baseline_norm[valid]*norm[valid],1e-24)
            ratio[valid]=norm[valid]/np.maximum(baseline_norm[valid],1e-12)
            source_valid=(source_ref_norm>1e-12)&(norm>1e-12)
            source_dot[source_valid]=np.sum(source_ref_area[source_valid]*area[source_valid],axis=1)/np.maximum(source_ref_norm[source_valid]*norm[source_valid],1e-30)
            source_floor=np.minimum(.02,source_baseline_dot-.02) if preserve_source_orientation else np.full(len(F),-2.0,dtype=np.float64)
            bad_faces=valid&((dot<=.05)|(ratio<=.15)|(source_dot<source_floor))
            if not np.any(bad_faces):accepted=trial;break
            blocked_faces=max(blocked_faces,int(np.count_nonzero(bad_faces)))
            blocked_vertices=np.unique(F[bad_faces].reshape(-1))
            # Freeze one ring around unsafe faces so tiny details do not spread the constraint forever.
            expanded=set(int(v) for v in blocked_vertices.tolist())
            for vi in blocked_vertices:
                expanded.update(int(n) for n in neighbours[int(vi)])
            guarded[np.fromiter(expanded,dtype=np.int64)]=0.0
            if not np.any(np.linalg.norm(guarded,axis=1)>1e-12):break
        if blocked_faces>0:
            # 1 means untouched and 0 means fully frozen; kept for report compatibility.
            moved=np.linalg.norm(guarded,axis=1)>1e-12;candidate=np.linalg.norm(displacement,axis=1)>1e-12
            local_scale=float(np.count_nonzero(moved)/max(np.count_nonzero(candidate),1))
        minimum_alpha=min(minimum_alpha,local_scale);V=accepted

    final_samples=np.einsum("bk,fkj->fbj",bary,V[local_faces]).reshape(-1,3)
    _,_,final_signed,_,_=_b14_nearest_surface(final_samples,target_triangles,k=32)
    final_signed=final_signed.reshape(len(local_faces),len(bary));relevant=final_signed[local_active]
    return V,{
        "affected_faces":int(len(affected_faces)),"iterations":int(iterations),
        "sample_min_before_mm":float(np.min(initial_signed[constrained])*1000.0),
        "sample_min_after_mm":float(np.min(relevant)*1000.0) if len(relevant) else None,
        "max_vertex_move_mm":float(np.max(np.linalg.norm(V-original,axis=1))*1000.0),
        "rms_vertex_move_mm":float(np.sqrt(np.mean(np.sum((V-original)**2,axis=1)))*1000.0),
        "source_contact_samples":int(np.count_nonzero(local_active)),"source_contact_faces":int(len(active_face_ids)),
        "minimum_orientation_alpha":float(minimum_alpha),
    }




def _shell_component_collision_guard(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, classes: dict[int, str], target_triangles: np.ndarray, minimum_clearance: float = .00050):
    """Clear deformable shell components with local B14 collision polish."""
    V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64);lab=np.asarray(labels,dtype=np.int64)
    reports=[];original=V.copy();# A modest triangular grid catches narrow anatomical peaks (nipples/labia) between the old
    bary=np.asarray([(i/6.0,j/6.0,(6-i-j)/6.0) for i in range(7) for j in range(7-i)],dtype=np.float64)
    for component in sorted(set(lab.tolist())):
        if str(classes.get(int(component),"")).casefold()!="shell": continue
        ids=np.flatnonzero(lab==component);face_ids=np.flatnonzero(np.all(lab[F]==component,axis=1))
        if len(ids)<3 or len(face_ids)==0: continue
        lookup=np.full(len(V),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64);LF=lookup[F[face_ids]]
        initial=V[ids].copy();a0=np.cross(initial[LF[:,1]]-initial[LF[:,0]],initial[LF[:,2]]-initial[LF[:,0]]);n0=np.linalg.norm(a0,axis=1);valid=n0>1e-14
        current=initial.copy();rounds=[]
        def sample_min(points):
            samples=np.einsum("bk,fkj->fbj",bary,points[LF]).reshape(-1,3);_,_,signed,_,_=_b14_nearest_surface(samples,target_triangles,k=32);return float(np.min(signed))
        before_min=sample_min(current);current_min=before_min
        if before_min>=minimum_clearance-1e-5: continue
        for round_index in range(8):
            if current_min>=minimum_clearance-1e-5: break
            candidate,polish=_b14_residual_collision_polish(current,LF,target_triangles,margin=max(minimum_clearance,.00070),iterations=22,blend=.55,max_push=.0045,k=32)
            displacement=np.asarray(candidate,dtype=np.float64)-current;accepted=current;accepted_alpha=0.0
            for alpha in (1.0,.75,.5,.375,.25,.1875,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625):
                trial=current+alpha*displacement;a=np.cross(trial[LF[:,1]]-trial[LF[:,0]],trial[LF[:,2]]-trial[LF[:,0]]);n=np.linalg.norm(a,axis=1)
                dot=np.ones(len(LF),dtype=np.float64);ratio=np.ones(len(LF),dtype=np.float64);dot[valid]=np.sum(a0[valid]*a[valid],axis=1)/np.maximum(n0[valid]*n[valid],1e-24);ratio[valid]=n[valid]/np.maximum(n0[valid],1e-14)
                if bool(np.all(dot[valid]>.05) and np.all(ratio[valid]>.15)):
                    accepted=trial;accepted_alpha=float(alpha);break
            current=accepted;current_min=sample_min(current);rounds.append({"round":round_index+1,"accepted_alpha":accepted_alpha,"sample_min_mm":current_min*1000.0,"polish":polish})
            if accepted_alpha<=0.0: break
        V[ids]=current;after_min=current_min
        reports.append({"component":int(component),"vertices":int(len(ids)),"faces":int(len(LF)),"before_min_mm":before_min*1000.0,"after_min_mm":after_min*1000.0,"max_vertex_move_mm":float(np.max(np.linalg.norm(current-initial,axis=1))*1000.0),"rounds":rounds})
    moved=np.linalg.norm(V-original,axis=1);mins=[r["after_min_mm"] for r in reports]
    return V,{"affected_components":int(len(reports)),"minimum_after_mm":float(min(mins)) if mins else None,"max_vertex_move_mm":float(np.max(moved)*1000.0) if len(moved) else 0.0,"components":reports}


def _shell_face_clearance_inflation_guard(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, classes: dict[int, str], target_triangles: np.ndarray, margin: float = .00052):
    """Clear the last shell face-interior intersections with a smooth normal offset."""
    V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64);lab=np.asarray(labels,dtype=np.int64);original=V.copy();reports=[]
    bary=np.asarray([(i/6.0,j/6.0,(6-i-j)/6.0) for i in range(7) for j in range(7-i)],dtype=np.float64)
    for component in sorted(set(lab.tolist())):
        if str(classes.get(int(component),"")).casefold()!="shell": continue
        ids=np.flatnonzero(lab==component);face_ids=np.flatnonzero(np.all(lab[F]==component,axis=1))
        if len(ids)<3 or len(face_ids)==0: continue
        lookup=np.full(len(V),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64);LF=lookup[F[face_ids]];cur=V[ids].copy();initial=cur.copy()
        edges=np.unique(np.sort(np.vstack([LF[:,[0,1]],LF[:,[1,2]],LF[:,[2,0]]]),axis=1),axis=0);nbr=[[] for _ in range(len(cur))]
        for a,b in edges:nbr[int(a)].append(int(b));nbr[int(b)].append(int(a))
        a0=np.cross(initial[LF[:,1]]-initial[LF[:,0]],initial[LF[:,2]]-initial[LF[:,0]]);n0=np.linalg.norm(a0,axis=1);valid=n0>1e-14
        rounds=[]
        def face_signed(points):
            samples=np.einsum("bk,fkj->fbj",bary,points[LF]).reshape(-1,3);_,_,signed,_,_=_b14_nearest_surface(samples,target_triangles,k=32);return signed.reshape(len(LF),len(bary))
        signed=face_signed(cur);before=float(np.min(signed))
        if before>=margin-1e-5: continue
        after_round=before
        for ri in range(6):
            deficit=np.maximum(0.0,margin-signed);face_req=np.max(deficit,axis=1)
            if float(np.max(face_req))<=1e-6: break
            req=np.zeros(len(cur),dtype=np.float64)
            for fi in np.flatnonzero(face_req>1e-6):
                req[LF[fi]]=np.maximum(req[LF[fi]],face_req[fi])
            field=req.copy()
            for _ in range(3):
                avg=np.asarray([np.mean(field[n]) if n else field[i] for i,n in enumerate(nbr)],dtype=np.float64)
                field=np.maximum(req,.78*field+.22*avg*.65)
            field=np.clip(field,0.0,.00085)
            _,vn,_,_,_=_b14_nearest_surface(cur,target_triangles,k=32);disp=vn*field[:,None]
            accepted=cur;alpha_used=0.0
            for alpha in (1.0,.75,.5,.375,.25,.125,.0625,.03125,.015625,.0078125,.00390625):
                trial=cur+alpha*disp;a=np.cross(trial[LF[:,1]]-trial[LF[:,0]],trial[LF[:,2]]-trial[LF[:,0]]);nn=np.linalg.norm(a,axis=1);dot=np.ones(len(LF));ratio=np.ones(len(LF));dot[valid]=np.sum(a0[valid]*a[valid],axis=1)/np.maximum(n0[valid]*nn[valid],1e-24);ratio[valid]=nn[valid]/np.maximum(n0[valid],1e-14)
                if bool(np.all(dot[valid]>.02) and np.all(ratio[valid]>.10)):
                    accepted=trial;alpha_used=float(alpha);break
            cur=accepted;signed=face_signed(cur);after_round=float(np.min(signed));rounds.append({"round":ri+1,"alpha":alpha_used,"min_mm":after_round*1000.0,"max_field_mm":float(np.max(field)*1000.0)})
            if alpha_used<=0.0 or after_round>=margin-1e-5: break
        V[ids]=cur;after=after_round;reports.append({"component":int(component),"vertices":int(len(ids)),"faces":int(len(LF)),"before_min_mm":before*1000.0,"after_min_mm":after*1000.0,"max_vertex_move_mm":float(np.max(np.linalg.norm(cur-initial,axis=1))*1000.0),"rounds":rounds})
    moved=np.linalg.norm(V-original,axis=1);mins=[r["after_min_mm"] for r in reports]
    return V,{"affected_components":len(reports),"minimum_after_mm":float(min(mins)) if mins else None,"max_vertex_move_mm":float(np.max(moved)*1000.0) if len(moved) else 0.0,"components":reports}


def _componentwise_surface_clearance_guard(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, source_triangles: np.ndarray, target_triangles: np.ndarray, preserve_source_orientation: bool = True):
    """Run face-interior clearance independently per connected garment component."""
    Vsrc=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64);lab=np.asarray(labels,dtype=np.int64)
    original=V.copy();reports=[]
    for component in sorted(set(lab.tolist())):
        ids=np.flatnonzero(lab==component)
        if len(ids)<3:continue
        face_ids=np.flatnonzero(np.all(lab[F]==component,axis=1))
        if len(face_ids)==0:continue
        lookup=np.full(len(V),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64)
        local_faces=lookup[F[face_ids]]
        out,rep=_continuous_surface_clearance_guard(Vsrc[ids],V[ids],local_faces,source_triangles,target_triangles,preserve_source_orientation)
        V[ids]=out;rep=dict(rep);rep["component"]=int(component);rep["vertices"]=int(len(ids));rep["faces"]=int(len(local_faces));reports.append(rep)
    moved=np.linalg.norm(V-original,axis=1)
    mins=[r.get("sample_min_after_mm") for r in reports if r.get("sample_min_after_mm") is not None]
    befores=[r.get("sample_min_before_mm") for r in reports if r.get("sample_min_before_mm") is not None]
    return V,{
        "affected_faces":int(sum(int(r.get("affected_faces",0)) for r in reports)),
        "affected_components":int(sum(1 for r in reports if int(r.get("affected_faces",0))>0)),
        "sample_min_before_mm":float(min(befores)) if befores else None,
        "sample_min_after_mm":float(min(mins)) if mins else None,
        "max_vertex_move_mm":float(np.max(moved)*1000.0) if len(moved) else 0.0,
        "rms_vertex_move_mm":float(np.sqrt(np.mean(moved*moved))*1000.0) if len(moved) else 0.0,
        "components":reports,
    }


def _skeleton_longitudinal_axis(glb: GLB, joint_names: list[str]) -> np.ndarray:
    """Return a stable pelvis-to-neck axis from the retained XIV skeleton."""
    positions, _, _ = skeleton_global_positions(glb, joint_names)
    by_name = {name: i for i, name in enumerate(joint_names)}
    low = by_name.get("j_kosi")
    high = by_name.get("j_kubi", by_name.get("j_sebo_c"))
    if low is None or high is None:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.asarray(positions[high] - positions[low], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-8 else np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def _boundary_groups(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Return connected open-boundary vertex groups for one shell."""
    V = np.asarray(vertices, dtype=np.float64); F = np.asarray(faces, dtype=np.int64)
    if len(V) == 0 or len(F) == 0:
        return []
    edges = np.sort(np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return []
    neighbours: dict[int, set[int]] = {}
    for a, b in boundary:
        neighbours.setdefault(int(a), set()).add(int(b)); neighbours.setdefault(int(b), set()).add(int(a))
    groups = []; remaining = set(neighbours)
    while remaining:
        seed = remaining.pop(); stack = [seed]; group = [seed]
        while stack:
            cur = stack.pop()
            for nxt in neighbours.get(cur, ()):
                if nxt in remaining:
                    remaining.remove(nxt); stack.append(nxt); group.append(nxt)
        groups.append(np.asarray(group, dtype=np.int64))
    return groups


def _preserve_structured_shell_longitudinal_design(w: dict[str, Any], solved: np.ndarray, labels: np.ndarray, features: dict[str, Any], longitudinal_axis: np.ndarray):
    """Stop substantial open edges sliding lengthwise during a structured-shell fit."""
    V = np.asarray(w["V"], dtype=np.float64); U = np.asarray(solved, dtype=np.float64).copy(); F = np.asarray(w["F"], dtype=np.int64); lab = np.asarray(labels, dtype=np.int64)
    axis = np.asarray(longitudinal_axis, dtype=np.float64); axis /= max(float(np.linalg.norm(axis)), 1e-12)
    root = int(features.get("root_component", 0)); ids = np.flatnonzero(lab == root)
    if len(ids) < 8:
        return U, {"active": False, "reason": "root too small"}
    lookup = np.full(len(V), -1, dtype=np.int64); lookup[ids] = np.arange(len(ids), dtype=np.int64)
    face_ids = np.flatnonzero(np.all(lab[F] == root, axis=1))
    if len(face_ids) == 0:
        return U, {"active": False, "reason": "no root faces"}
    LF = lookup[F[face_ids]]; LV = V[ids]; LU = U[ids]
    groups = _boundary_groups(LV, LF)
    if not groups:
        return U, {"active": False, "reason": "closed shell"}
    # Ignore tiny eyelets/cuts; only real garment openings count as boundaries.
    shell_extent = float(np.max(np.ptp(LV, axis=0)))
    selected = []; fixed = {}
    for group in groups:
        if len(group) < 6:
            continue
        pts = LV[group]; extent = float(np.max(np.ptp(pts, axis=0))) if len(pts) else 0.0
        if extent < max(0.018, shell_extent * 0.10):
            continue
        delta = (LU[group] - LV[group]) @ axis
        excessive = np.abs(delta) > .004
        if int(np.count_nonzero(excessive)) < max(3, int(np.ceil(len(group) * .20))):
            continue
        correction = -delta
        for local_id, amount in zip(group.tolist(), correction.tolist()):
            fixed[int(local_id)] = float(amount)
        selected.append({"vertices": int(len(group)), "extent_mm": extent * 1000.0, "mean_before_mm": float(np.mean(delta) * 1000.0), "max_abs_before_mm": float(np.max(np.abs(delta)) * 1000.0)})
    if not fixed:
        return U, {"active": False, "reason": "no excessive substantial boundary slide", "boundary_count": len(groups)}
    edges = np.unique(np.sort(np.vstack((LF[:, [0, 1]], LF[:, [1, 2]], LF[:, [2, 0]])), axis=1), axis=0)
    neighbours = [[] for _ in range(len(LV))]
    for a, b in edges:
        neighbours[int(a)].append(int(b)); neighbours[int(b)].append(int(a))
    field = np.zeros(len(LV), dtype=np.float64); fixed_mask = np.zeros(len(LV), dtype=bool)
    for idx, amount in fixed.items():
        field[idx] = amount; fixed_mask[idx] = True
    # Harmonic extension carries the edge constraint smoothly through the interior.
    for _ in range(120):
        nxt = field.copy()
        for i, nb in enumerate(neighbours):
            if fixed_mask[i] or not nb:
                continue
            nxt[i] = float(np.mean(field[nb]))
        if float(np.max(np.abs(nxt - field))) < 1e-7:
            field = nxt; break
        field = nxt
    # This is a guard, not a second solver; clamp silly corrections.
    field = np.clip(field, -.060, .060)
    LU = LU + field[:, None] * axis[None, :]
    U[ids] = LU
    after = (LU - LV) @ axis
    boundary_after = np.asarray([after[i] for i in fixed], dtype=np.float64)
    return U, {"active": True, "root_component": root, "selected_boundaries": selected, "fixed_vertices": int(len(fixed)), "max_abs_after_mm": float(np.max(np.abs(boundary_after)) * 1000.0) if len(boundary_after) else 0.0, "harmonic_iterations": 120}


def _dense_target_obstacle_clearance_guard(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, classes: dict[int, str], target_triangles: np.ndarray, longitudinal_axis: np.ndarray | None = None, margin: float = .00050, preserve_source_orientation: bool = True):
    """Run the batched final body-obstacle pass for highly disconnected garments."""
    Vsrc = np.asarray(source_vertices, dtype=np.float64); V = np.asarray(vertices, dtype=np.float64).copy(); F = np.asarray(faces, dtype=np.int64); lab = np.asarray(labels, dtype=np.int64); original = V.copy()
    if len(V) == 0 or len(F) == 0:
        return V, {"affected_components": 0, "penetrating_after": 0, "dense_min_after_mm": None, "max_vertex_move_mm": 0.0, "batched": True}
    face_component = lab[F[:, 0]]
    if not np.all((lab[F[:, 1]] == face_component) & (lab[F[:, 2]] == face_component)):
        raise ValueError("Connected-component labels are inconsistent across garment faces.")
    axis = None if longitudinal_axis is None else np.asarray(longitudinal_axis, dtype=np.float64)
    if axis is not None:
        axis /= max(float(np.linalg.norm(axis)), 1e-12)

    bary = []
    N = 4
    for i in range(N + 1):
        for j in range(N + 1 - i):
            bary.append([i / N, j / N, (N - i - j) / N])
    bary = np.asarray(bary, dtype=np.float64)
    coarse_bary = np.asarray([[1/3,1/3,1/3],[.5,.5,0],[0,.5,.5],[.5,0,.5],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]], dtype=np.float64)

    T = np.asarray(target_triangles, dtype=np.float64); centres = T.mean(axis=1); tree = cKDTree(centres)
    tnorm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); tnorm /= np.maximum(np.linalg.norm(tnorm, axis=1, keepdims=True), 1e-12)
    def nearest(points: np.ndarray, k: int):
        # Chunk large face-sample searches to keep peak memory sane.
        P = np.asarray(points, dtype=np.float64); cp_all = []; normal_all = []; signed_all = []; kk = min(k, len(T))
        for start in range(0, len(P), 6000):
            chunk = P[start:start + 6000]; _, idx = tree.query(chunk, k=kk, workers=-1); idx = idx if idx.ndim > 1 else idx[:, None]
            n = len(chunk); TC = T[idx.reshape(-1)]; Q = np.repeat(chunk, kk, axis=0); C = trimesh.triangles.closest_point(TC, Q).reshape(n, kk, 3)
            dd = np.sum((C - chunk[:, None, :]) ** 2, axis=2); winner = np.argmin(dd, axis=1); cp = C[np.arange(n), winner]; fi = idx[np.arange(n), winner]; normals = tnorm[fi]; signed = np.sum((chunk - cp) * normals, axis=1)
            cp_all.append(cp); normal_all.append(normals); signed_all.append(signed)
        return np.vstack(cp_all), np.vstack(normal_all), np.concatenate(signed_all)

    edges = np.unique(np.sort(np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]])), axis=1), axis=0)
    neighbours = [[] for _ in range(len(V))]
    for a, b in edges:
        neighbours[int(a)].append(int(b)); neighbours[int(b)].append(int(a))
    boundary = np.zeros(len(V), dtype=bool)
    components = sorted(set(int(x) for x in lab.tolist()))
    comp_vertices = {}; comp_faces = {}; comp_kind = {}
    for component in components:
        ids = np.flatnonzero(lab == component); fids = np.flatnonzero(face_component == component); comp_vertices[component] = ids; comp_faces[component] = fids; comp_kind[component] = str(classes.get(component, "shell")).casefold()
        if comp_kind[component] != "rigid" and len(ids) >= 3 and len(fids):
            lookup = np.full(len(V), -1, dtype=np.int64); lookup[ids] = np.arange(len(ids), dtype=np.int64); LF = lookup[F[fids]]
            groups = _boundary_groups(V[ids], LF)
            # Treat tiny closed dense shells as rigid ornaments; open or substantial shells stay deformable.
            extent = float(np.max(np.ptp(V[ids], axis=0))) if len(ids) else 0.0
            if not groups and extent <= .025:
                comp_kind[component] = "rigid"
            else:
                for group in groups:
                    boundary[ids[group]] = True

    base_area = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); base_norm = np.linalg.norm(base_area, axis=1); valid = base_norm > 1e-14
    source_area = np.zeros_like(base_area); source_norm = np.zeros(len(F), dtype=np.float64)
    source_baseline_dot = np.ones(len(F), dtype=np.float64)
    if preserve_source_orientation:
        for component in components:
            ids = comp_vertices[component]; fids = comp_faces[component]
            if len(ids) < 3 or not len(fids): continue
            r, t = _rigid_fit_points(Vsrc[ids], V[ids]); lookup = np.full(len(V), -1, dtype=np.int64); lookup[ids] = np.arange(len(ids), dtype=np.int64); LF = lookup[F[fids]]; aligned = Vsrc[ids] @ r.T + t
            area = np.cross(aligned[LF[:, 1]] - aligned[LF[:, 0]], aligned[LF[:, 2]] - aligned[LF[:, 0]]); source_area[fids] = area; source_norm[fids] = np.linalg.norm(area, axis=1)
        sv = (source_norm > 1e-12) & valid; source_baseline_dot[sv] = np.sum(source_area[sv] * base_area[sv], axis=1) / np.maximum(source_norm[sv] * base_norm[sv], 1e-30)

    def dense_state(points: np.ndarray):
        coarse = np.einsum("bk,fkj->fbj", coarse_bary, points[F]).reshape(-1, 3)
        _, _, coarse_signed = nearest(coarse, 12); coarse_signed = coarse_signed.reshape(len(F), len(coarse_bary))
        candidate = np.min(coarse_signed, axis=1) < .0025
        if not np.any(candidate):
            return np.empty(0, dtype=np.int64), np.empty((0, len(bary))), np.empty((0, len(bary), 3))
        cids = np.flatnonzero(candidate); samples = np.einsum("bk,fkj->fbj", bary, points[F[cids]]).reshape(-1, 3)
        _, normals, signed = nearest(samples, 24)
        return cids, signed.reshape(len(cids), len(bary)), normals.reshape(len(cids), len(bary), 3)

    initial_cids, initial_signed, _ = dense_state(V); initial_min = float(np.min(initial_signed)) if initial_signed.size else float("inf")
    if not initial_signed.size or initial_min >= margin - 1e-6:
        return V, {"affected_components": 0, "penetrating_after": int(np.count_nonzero(initial_signed < 0.0)) if initial_signed.size else 0, "dense_min_after_mm": None if not np.isfinite(initial_min) else initial_min * 1000.0, "max_vertex_move_mm": 0.0, "batched": True, "components": []}

    touched = set(); rounds = []
    def safe_trial(current: np.ndarray, displacement: np.ndarray):
        # Use a global line search here so one tiny triangle cannot trigger repeated local freezes.
        for alpha in (1.0, .75, .5, .35, .25, .15, .08, .04):
            trial = current + alpha * displacement; area = np.cross(trial[F[:, 1]] - trial[F[:, 0]], trial[F[:, 2]] - trial[F[:, 0]]); nn = np.linalg.norm(area, axis=1)
            dot = np.ones(len(F)); ratio = np.ones(len(F)); source_dot = np.ones(len(F)); dot[valid] = np.sum(base_area[valid] * area[valid], axis=1) / np.maximum(base_norm[valid] * nn[valid], 1e-30); ratio[valid] = nn[valid] / np.maximum(base_norm[valid], 1e-16)
            sv = (source_norm > 1e-12) & (nn > 1e-12); source_dot[sv] = np.sum(source_area[sv] * area[sv], axis=1) / np.maximum(source_norm[sv] * nn[sv], 1e-30)
            source_floor = np.minimum(.02, source_baseline_dot - .02) if preserve_source_orientation else np.full(len(F), -2.0, dtype=np.float64)
            if not np.any(valid & ((dot <= .03) | (ratio <= .12) | (source_dot < source_floor))):
                return trial, float(alpha)
        return current, 0.0

    for ri in range(5):
        cids, signed, normals = dense_state(V)
        if not signed.size: break
        deficits = np.maximum(0.0, margin - signed); rows, bis = np.where(deficits > 1e-6)
        if len(rows) == 0: break
        delta = np.zeros_like(V)
        sample_components = face_component[cids[rows]]
        for component in np.unique(sample_components):
            component = int(component)
            if comp_kind[component] != "rigid": continue
            mask = sample_components == component; rr = rows[mask]; bb = bis[mask]; shift = np.zeros(3, dtype=np.float64)
            for _ in range(10):
                changed = False
                for row, bi in zip(rr.tolist(), bb.tolist()):
                    normal = normals[row, bi]; required = float(deficits[row, bi]); short = required - float(np.dot(shift, normal))
                    if short > 1e-7: shift += normal * short; changed = True
                if not changed: break
            mag = float(np.linalg.norm(shift));
            if mag > .004: shift *= .004 / mag
            delta[comp_vertices[component]] += shift; touched.add(component)
        shell_mask = np.asarray([comp_kind[int(c)] != "rigid" for c in sample_components], dtype=bool)
        if np.any(shell_mask):
            rr = rows[shell_mask]; bb = bis[shell_mask]; face = F[cids[rr]]; bw = bary[bb]; ns = normals[rr, bb]; req = deficits[rr, bb]
            acc = np.zeros_like(V); weight_sum = np.zeros(len(V), dtype=np.float64)
            for corner in range(3):
                vi = face[:, corner]; wt = bw[:, corner]; add = ns * (req * wt)[:, None]; np.add.at(acc, vi, add); np.add.at(weight_sum, vi, wt)
            direct = weight_sum > 0; disp = np.zeros_like(V); disp[direct] = acc[direct] / weight_sum[direct, None]
            feather = disp.copy()
            for vi in np.flatnonzero(direct):
                nb = neighbours[int(vi)]
                if nb: feather[vi] = .94 * disp[vi] + .06 * np.mean(disp[nb], axis=0)
            for vi in np.flatnonzero(~direct):
                nb = neighbours[int(vi)]
                active_nb = [n for n in nb if direct[n]]
                if active_nb: feather[vi] = .025 * np.mean(disp[active_nb], axis=0)
            disp = feather
            if axis is not None:
                bvids = np.flatnonzero(boundary & (np.linalg.norm(disp, axis=1) > 1e-12)); disp[bvids] -= np.outer(disp[bvids] @ axis, axis)
            mag = np.linalg.norm(disp, axis=1); disp *= np.minimum(1.0, .00125 / np.maximum(mag, 1e-12))[:, None]; delta += disp
            touched.update(int(x) for x in np.unique(sample_components[shell_mask]).tolist())
        trial, accepted = safe_trial(V, delta); V = trial
        _, check, _ = dense_state(V); current_min = float(np.min(check)) if check.size else float("inf"); rounds.append({"round": ri + 1, "mode": "averaged", "alpha": accepted, "dense_min_mm": None if not np.isfinite(current_min) else current_min * 1000.0})
        if accepted <= 0.0 or current_min >= margin - 1e-6: break

    for pri in range(6):
        cids, signed, normals = dense_state(V)
        if not signed.size: break
        deficits = np.maximum(0.0, margin - signed); rows, bis = np.where(deficits > 1e-6)
        if len(rows) == 0: break
        delta = np.zeros_like(V)
        sample_components = face_component[cids[rows]]
        for component in np.unique(sample_components):
            component = int(component)
            if comp_kind[component] != "rigid": continue
            mask = sample_components == component; rr = rows[mask]; bb = bis[mask]; shift = np.zeros(3, dtype=np.float64)
            for _ in range(12):
                changed = False
                for row, bi in zip(rr.tolist(), bb.tolist()):
                    normal = normals[row, bi]; short = float(deficits[row, bi]) - float(np.dot(shift, normal))
                    if short > 1e-7: shift += normal * short; changed = True
                if not changed: break
            mag = float(np.linalg.norm(shift));
            if mag > .004: shift *= .004 / mag
            delta[comp_vertices[component]] += shift; touched.add(component)
        # Finish shell clearance with projected Gauss-Seidel on the remaining violations.
        shell_pairs = [(int(r), int(b)) for r, b, c in zip(rows.tolist(), bis.tolist(), sample_components.tolist()) if comp_kind[int(c)] != "rigid"]
        for _ in range(10):
            changed = False
            for row, bi in shell_pairs:
                face = F[cids[row]]; normal = normals[row, bi]; b = bary[bi]; required = float(deficits[row, bi]); projected = float(sum(float(b[c]) * np.dot(delta[int(face[c])], normal) for c in range(3)))
                short = required - projected
                if short <= 1e-7: continue
                directions = []; denom = 0.0
                for c in range(3):
                    vi = int(face[c]); direction = normal.copy()
                    if axis is not None and boundary[vi]:
                        transverse = direction - axis * float(np.dot(direction, axis))
                        if float(np.linalg.norm(transverse)) > .20: direction = transverse
                    directions.append(direction); denom += float(b[c] * b[c]) * max(float(np.dot(direction, normal)), 0.0)
                if denom <= 1e-10: continue
                lam = short / denom
                for c in range(3): delta[int(face[c])] += directions[c] * (lam * float(b[c]))
                changed = True
            if not changed: break
        mag = np.linalg.norm(delta, axis=1); delta *= np.minimum(1.0, .00175 / np.maximum(mag, 1e-12))[:, None]
        # Backtrack globally here, then recompute the constraints on the next outer pass.
        accepted = 0.0; candidate = V
        for alpha in (1.0, .75, .5, .35, .25, .15, .08, .04):
            trial = V + alpha * delta; area = np.cross(trial[F[:, 1]] - trial[F[:, 0]], trial[F[:, 2]] - trial[F[:, 0]]); nn = np.linalg.norm(area, axis=1)
            dot = np.ones(len(F)); ratio = np.ones(len(F)); source_dot = np.ones(len(F)); dot[valid] = np.sum(base_area[valid] * area[valid], axis=1) / np.maximum(base_norm[valid] * nn[valid], 1e-30); ratio[valid] = nn[valid] / np.maximum(base_norm[valid], 1e-16)
            sv = (source_norm > 1e-12) & (nn > 1e-12); source_dot[sv] = np.sum(source_area[sv] * area[sv], axis=1) / np.maximum(source_norm[sv] * nn[sv], 1e-30); source_floor = np.minimum(.02, source_baseline_dot - .02) if preserve_source_orientation else np.full(len(F), -2.0, dtype=np.float64)
            if np.any(valid & ((dot <= .03) | (ratio <= .12) | (source_dot < source_floor))): continue
            candidate = trial; accepted = float(alpha); break
        V = candidate; _, check, _ = dense_state(V); current_min = float(np.min(check)) if check.size else float("inf"); rounds.append({"round": pri + 1, "mode": "projected", "alpha": accepted, "constraint_count": int(len(rows)), "dense_min_mm": None if not np.isfinite(current_min) else current_min * 1000.0})
        if accepted <= 0.0 or current_min >= margin - 1e-6: break

    final_cids, final_signed, _ = dense_state(V); final_min = float(np.min(final_signed)) if final_signed.size else float("inf"); component_reports = []
    if final_signed.size:
        for component in components:
            mask = face_component[final_cids] == component
            if not np.any(mask): continue
            vals = final_signed[mask]; component_reports.append({"component": component, "kind": comp_kind[component], "dense_min_after_mm": float(np.min(vals) * 1000.0), "penetrating_after": int(np.count_nonzero(vals < 0.0)), "max_vertex_move_mm": float(np.max(np.linalg.norm(V[comp_vertices[component]] - original[comp_vertices[component]], axis=1)) * 1000.0) if len(comp_vertices[component]) else 0.0})
    moved = np.linalg.norm(V - original, axis=1)
    return V, {"affected_components": int(len(touched)), "penetrating_after": int(np.count_nonzero(final_signed < 0.0)) if final_signed.size else 0, "dense_min_after_mm": None if not np.isfinite(final_min) else final_min * 1000.0, "max_vertex_move_mm": float(np.max(moved) * 1000.0) if len(moved) else 0.0, "batched": True, "rounds": rounds, "components": component_reports}

def _leg_joint_columns(names: list[str]):
    left=[i for i,name in enumerate(names) if name.startswith("j_asi_") and name.endswith("_l")]
    right=[i for i,name in enumerate(names) if name.startswith("j_asi_") and name.endswith("_r")]
    return left,right


def _sample_target_skin_weights(points: np.ndarray, target_v: np.ndarray, target_f: np.ndarray, target_w: np.ndarray):
    points=np.asarray(points,dtype=np.float64);target_v=np.asarray(target_v,dtype=np.float64);target_f=np.asarray(target_f,dtype=np.int64);target_w=np.asarray(target_w,dtype=np.float64)
    target_tri=target_v[target_f]
    closest,_,_,distance,face_index=_b14_nearest_surface(points,target_tri,k=32)
    face_index=np.asarray(face_index,dtype=np.int64)
    tri=target_tri[face_index]
    bary=trimesh.triangles.points_to_barycentric(tri,np.asarray(closest,dtype=np.float64))
    tri_weights=target_w[target_f[face_index]]
    weights=np.einsum("ni,nij->nj",bary,tri_weights)
    weights=np.maximum(weights,0.0);total=weights.sum(axis=1,keepdims=True);good=total[:,0]>1e-12;weights[good]/=total[good]

    if not np.all(good):
        vertex_good=target_w.sum(axis=1)>1e-12
        face_good=np.all(vertex_good[target_f],axis=1)
        if int(np.count_nonzero(face_good))<8:face_good=np.any(vertex_good[target_f],axis=1)
        bad=np.flatnonzero(~good)
        if int(np.count_nonzero(face_good)):
            valid_faces=target_f[face_good];valid_tri=target_v[valid_faces]
            fallback_closest,_,_,fallback_distance,fallback_face=_b14_nearest_surface(points[bad],valid_tri,k=32)
            fallback_face=np.asarray(fallback_face,dtype=np.int64);fallback_tri=valid_tri[fallback_face]
            fallback_bary=trimesh.triangles.points_to_barycentric(fallback_tri,np.asarray(fallback_closest,dtype=np.float64))
            fallback_weights=np.einsum("ni,nij->nj",fallback_bary,target_w[valid_faces[fallback_face]])
            fallback_weights=np.maximum(fallback_weights,0.0);fallback_total=fallback_weights.sum(axis=1,keepdims=True);fallback_good=fallback_total[:,0]>1e-12
            fallback_weights[fallback_good]/=fallback_total[fallback_good]
            recovered=bad[fallback_good];weights[recovered]=fallback_weights[fallback_good];distance[recovered]=np.asarray(fallback_distance,dtype=np.float64)[fallback_good];good[recovered]=True

        remaining=np.flatnonzero(~good)
        if len(remaining) and np.any(vertex_good):
            valid_vertices=np.flatnonzero(vertex_good);tree=cKDTree(target_v[valid_vertices]);vd,vi=tree.query(points[remaining],k=1)
            weights[remaining]=target_w[valid_vertices[np.asarray(vi,dtype=np.int64)]];distance[remaining]=np.asarray(vd,dtype=np.float64);good[remaining]=weights[remaining].sum(axis=1)>1e-12

    return weights,np.asarray(distance,dtype=np.float64)


def _target_skin_weights_at_points(points: np.ndarray, cache: dict[str, Any], source_weights: np.ndarray | None = None, source_joint_names: list[str] | None = None):
    """Sample target skinning without letting one leg borrow the other leg's surface."""
    target_v=np.asarray(cache.get("target_surface_V"),dtype=np.float64);target_f=np.asarray(cache.get("target_surface_F"),dtype=np.int64);target_w=np.asarray(cache.get("target_surface_W"),dtype=np.float64)
    if target_w.ndim!=2 or len(target_w)!=len(target_v):raise ValueError("Target body surface weights are missing or do not match target surface vertices.")
    points=np.asarray(points,dtype=np.float64);weights,distance=_sample_target_skin_weights(points,target_v,target_f,target_w)
    if source_weights is None or source_joint_names is None:return weights,distance

    cache_names=list(cache["names"]);source_joint_names=list(source_joint_names);source=np.asarray(source_weights,dtype=np.float64)
    if source.shape[1]!=len(source_joint_names):raise ValueError("Source garment skin weights do not match the source joint list.")
    source_index={name:i for i,name in enumerate(source_joint_names)}
    missing=[name for name in cache_names if name not in source_index]
    if missing:return weights,distance
    source_cache=np.column_stack([source[:,source_index[name]] for name in cache_names])
    left,right=_leg_joint_columns(cache_names)
    if not left or not right:return weights,distance

    source_left=source_cache[:,left].sum(axis=1);source_right=source_cache[:,right].sum(axis=1)
    left_rows=(source_left>=.55)&((source_left-source_right)>=.35)
    right_rows=(source_right>=.55)&((source_right-source_left)>=.35)
    if not np.any(left_rows|right_rows):return weights,distance

    face_weights=np.mean(target_w[target_f],axis=1)
    face_left=face_weights[:,left].sum(axis=1);face_right=face_weights[:,right].sum(axis=1)
    masks=((face_left>=face_right)&(face_left>.10),(face_right>=face_left)&(face_right>.10))
    for rows,mask in ((left_rows,masks[0]),(right_rows,masks[1])):
        if not np.any(rows) or int(np.count_nonzero(mask))<8:continue
        local_weights,local_distance=_sample_target_skin_weights(points[rows],target_v,target_f[mask],target_w)
        weights[rows]=local_weights;distance[rows]=local_distance
    return weights,distance


def _verified_body_skin_delta_at_points(points: np.ndarray, garment_weights: np.ndarray, garment_joint_names: list[str], cache: dict[str,Any]):
    """Transfer only a body-correspondence-proven source->target skin-weight delta to garment points.

    The field is defined on the *paired body correspondence* (X/BW -> Y/target_correspondence_W), not
    by independently sampling whichever target-body triangle happens to be nearest after refitting.
    This distinction is important: moving garment geometry across a skin-weight gradient is not itself
    evidence that the garment should be reweighted.  If paired source/target body weights are the same,
    the returned delta is exactly zero regardless of geometric movement.
    """
    P=np.asarray(points,dtype=np.float64);Wg=np.asarray(garment_weights,dtype=np.float64)
    X=np.asarray(cache.get("X"),dtype=np.float64);BW=np.asarray(cache.get("BW"),dtype=np.float64);TW=np.asarray(cache.get("target_correspondence_W"),dtype=np.float64)
    cache_names=list(cache.get("names") or []);garment_joint_names=list(garment_joint_names)
    if X.ndim!=2 or X.shape[1:]!=(3,) or BW.ndim!=2 or TW.shape!=BW.shape or len(BW)!=len(X) or BW.shape[1]!=len(cache_names):
        return None,{"enabled":False,"reason":"paired source/target body skin-weight correspondence is unavailable"}
    if Wg.ndim!=2 or len(Wg)!=len(P) or Wg.shape[1]!=len(garment_joint_names):
        return None,{"enabled":False,"reason":"garment skinning does not match garment joint list"}
    source_index={name:i for i,name in enumerate(garment_joint_names)}
    common_cache=np.asarray([i for i,name in enumerate(cache_names) if name in source_index],dtype=np.int64)
    if not len(common_cache):
        return None,{"enabled":False,"reason":"garment and paired body expose no common skin joints"}
    body_supported=(np.sum(BW,axis=0)+np.sum(TW,axis=0))>1e-8
    common_cache=common_cache[body_supported[common_cache]]
    if not len(common_cache):
        return None,{"enabled":False,"reason":"no common body-supported skin joints"}

    body_delta=np.asarray(TW-BW,dtype=np.float64)
    body_delta_l1=np.abs(body_delta).sum(axis=1)
    field_max=float(np.max(body_delta_l1,initial=0.0));field_p95=float(np.percentile(body_delta_l1,95)) if len(body_delta_l1) else 0.0
    # One 8-bit weight step transferred from one bone to another is ~2/255 L1.  The correspondence
    # builders preserve exact same-topology weights, while this small floor suppresses interpolation
    # noise for cross-topology pairs that do not contain a meaningful rigging change.
    field_floor=0.0035
    if field_max<=field_floor:
        return np.zeros((len(P),len(cache_names)),dtype=np.float64),{"enabled":True,"verified_body_delta":False,"reason":"paired body weight field is equivalent within quantisation tolerance","field_delta_l1_p95":field_p95,"field_delta_l1_max":field_max,"quantisation_floor":field_floor}

    signature=np.zeros((len(P),len(cache_names)),dtype=np.float64)
    for ci in common_cache.tolist():signature[:,ci]=Wg[:,source_index[cache_names[ci]]]
    signature_total=signature.sum(axis=1,keepdims=True);has_signature=signature_total[:,0]>1e-8;signature[has_signature]/=signature_total[has_signature]
    if not np.any(has_signature):
        return np.zeros((len(P),len(cache_names)),dtype=np.float64),{"enabled":True,"verified_body_delta":False,"reason":"garment has no body-supported weight mass","field_delta_l1_p95":field_p95,"field_delta_l1_max":field_max,"quantisation_floor":field_floor}

    k=min(32,len(X));tree=cKDTree(X);distance,index=tree.query(P,k=k);distance=distance if distance.ndim>1 else distance[:,None];index=index if index.ndim>1 else index[:,None]
    alignment=np.einsum("nk,nqk->nq",signature,BW[index],optimize=True);score=distance+.015*(1.0-alignment)
    sx=np.sign(P[:,0])[:,None];bx=np.sign(X[index][:,:,0]);score+=np.where((np.abs(P[:,0,None])>.020)&(sx!=bx),.080,0.0)
    take=min(12,score.shape[1]);order=np.argpartition(score,take-1,axis=1)[:,:take];selected=np.take_along_axis(index,order,axis=1);selected_score=np.take_along_axis(score,order,axis=1);selected_distance=np.take_along_axis(distance,order,axis=1)
    rel=selected_score-selected_score.min(axis=1,keepdims=True);blend=np.exp(-rel/.0035);blend/=np.maximum(blend.sum(axis=1,keepdims=True),1e-12)
    transferred=np.sum(body_delta[selected]*blend[:,:,None],axis=1);support_distance=np.sum(selected_distance*blend,axis=1);local_l1=np.abs(transferred).sum(axis=1)
    transferred[~has_signature]=0.0;local_l1[~has_signature]=0.0
    return transferred,{"enabled":True,"verified_body_delta":True,"field_delta_l1_p95":field_p95,"field_delta_l1_max":field_max,"local_delta_l1_p95":float(np.percentile(local_l1,95)) if len(local_l1) else 0.0,"local_delta_l1_max":float(np.max(local_l1,initial=0.0)),"support_distance_p95_mm":float(np.percentile(support_distance,95)*1000.0) if len(support_distance) else 0.0,"support_distance":support_distance,"local_delta_l1":local_l1,"body_supported_cache_columns":common_cache,"quantisation_floor":field_floor}


def _retarget_garment_skinning(raw_positions: np.ndarray, source_positions: np.ndarray, source_weights: np.ndarray, source_joint_names: list[str], cache: dict[str, Any], behavior: str, effective_behavior: str, labels: np.ndarray, classes: dict[int, str], raw_to_weld: np.ndarray):
    """Preserve the untouched garment skinning exactly after geometry refitting.

    Target-body geometry is authoritative for fit and collision.  It is not skinning authority.  A
    source->target body weight-field difference can describe the bodies themselves without proving
    that an authored garment should be reweighted, and broad transfer here was able to destroy cloth,
    stocking, strap and pelvis/thigh deformation relationships.  Until a garment-specific necessity is
    proved independently, the untouched garment weights remain byte-stable structural authority.
    """
    raw_positions=np.asarray(raw_positions,dtype=np.float64);source_positions=np.asarray(source_positions,dtype=np.float64);source=np.asarray(source_weights,dtype=np.float64);source_joint_names=list(source_joint_names)
    if raw_positions.shape!=source_positions.shape:raise ValueError(f"Garment skin preserve position mismatch {raw_positions.shape} vs {source_positions.shape}.")
    if source.ndim!=2 or len(source)!=len(raw_positions):raise ValueError(f"Garment skin weights {source.shape} do not match {len(raw_positions)} garment vertices.")
    if source.shape[1]!=len(source_joint_names):raise ValueError(f"Garment skin weights {source.shape} do not match {len(source_joint_names)} source joints.")
    total=source.sum(axis=1)
    if np.any(~np.isfinite(source)) or np.any(~np.isfinite(total)) or np.any(total<=1e-12):
        raise ValueError(f"Source garment skinning contains {int(np.count_nonzero((~np.isfinite(total))|(total<=1e-12)))} invalid or unweighted vertices.")

    geometric_move=np.linalg.norm(raw_positions-source_positions,axis=1);preserved=source.copy()
    left_columns,right_columns=_leg_joint_columns(source_joint_names)
    if left_columns and right_columns:
        source_left=source[:,left_columns].sum(axis=1);source_right=source[:,right_columns].sum(axis=1);bilateral=(source_left>=.04)&(source_right>=.04)
    else:bilateral=np.zeros(len(source),dtype=bool)
    raw_labels=np.asarray(labels,dtype=np.int64)[np.asarray(raw_to_weld,dtype=np.int64)] if len(raw_to_weld)==len(source) else np.full(len(source),-1,dtype=np.int64)
    rigid=np.asarray([str(classes.get(int(component),"")).casefold()=="rigid" for component in raw_labels],dtype=bool)
    return preserved,{
        "mode":"source_authored_skinning_preserved",
        "policy":"target body controls garment fit/clearance; untouched source garment skinning remains exact structural authority unless a future garment-specific rule proves a reweight is required",
        "vertices":int(len(preserved)),"retargeted_vertices":0,"geometry_rigid_vertices":int(np.count_nonzero(rigid)),"skin_rigid_smoothed_vertices":0,"skin_rigid_smoothed_components":0,
        "preserved_non_body_joint_count":int(len(source_joint_names)),"preserved_non_body_weight_mean":1.0,"bilateral_authored_vertices":int(np.count_nonzero(bilateral)),"bilateral_exact_source_weight_preserve":True,
        "source_influence_capacity":int(np.max(np.count_nonzero(source>1e-8,axis=1),initial=0)),"new_target_joint_slots":0,
        "geometric_move_p50_mm":float(np.median(geometric_move)*1000.0) if len(geometric_move) else 0.0,"geometric_move_p95_mm":float(np.percentile(geometric_move,95)*1000.0) if len(geometric_move) else 0.0,
        "alpha_mean":0.0,"alpha_p95":0.0,"target_distance_p50_mm":0.0,"target_distance_p95_mm":0.0,
        "weight_delta_l1_mean":0.0,"weight_delta_l1_p95":0.0,"weight_delta_l1_max":0.0,"exact_source_weight_preserve":True,
        "body_weight_delta":{"enabled":False,"verified_body_delta":False,"reason":"body skin-weight differences are diagnostic only and do not override authored garment skinning"},
    }

def _preserve_leg_component_separation(source_positions: np.ndarray, solved_positions: np.ndarray, source_weights: np.ndarray, joint_names: list[str], labels: np.ndarray):
    source=np.asarray(source_positions,dtype=np.float64);solved=np.asarray(solved_positions,dtype=np.float64).copy();weights=np.asarray(source_weights,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
    left,right=_leg_joint_columns(list(joint_names))
    if not left or not right:return solved,{"moved_vertices":0,"components":[]}
    left_mass=weights[:,left].sum(axis=1);right_mass=weights[:,right].sum(axis=1);reports=[];moved_total=0
    for component in np.unique(labels):
        ids=np.flatnonzero(labels==component)
        if len(ids)<3:continue
        lm=float(np.mean(left_mass[ids]));rm=float(np.mean(right_mass[ids]));dominant=max(lm,rm)
        if dominant<.55 or abs(lm-rm)<.35:continue
        side_mass=left_mass if lm>rm else right_mass
        strong=ids[side_mass[ids]>=.45]
        if len(strong)<3:strong=ids
        source_side=float(np.median(source[strong,0]));sign=1.0 if source_side>0 else -1.0 if source_side<0 else 0.0
        if sign==0.0:continue
        source_signed=sign*source[ids,0]
        positive=source_signed[source_signed>1e-6]
        if len(positive)<max(3,int(len(ids)*.75)):continue
        source_centre=float(np.median(positive));solved_signed=sign*solved[ids,0]
        solved_positive=solved_signed[solved_signed>1e-6]
        solved_centre=float(np.median(solved_positive)) if len(solved_positive) else source_centre
        lateral_scale=float(np.clip(solved_centre/max(source_centre,1e-6),.65,1.50))
        floor=np.maximum(source_signed*lateral_scale*.40,.0015)
        bad=solved_signed<floor
        if not np.any(bad):continue
        moved=ids[bad];solved[moved,0]=sign*floor[bad];moved_total+=int(len(moved))
        reports.append({"component":int(component),"side":"left" if lm>rm else "right","moved_vertices":int(len(moved)),"lateral_scale":lateral_scale,"minimum_half_gap_mm":float(np.min(floor[bad])*1000.0)})
    return solved,{"moved_vertices":moved_total,"components":reports}


class NonFiniteGeometryError(ValueError):
    """Raised when a solver stage produces NaN/Inf geometry."""


def _assert_finite_stage(mesh_name: str, stage_name: str, value: Any):
    array=np.asarray(value)
    if np.all(np.isfinite(array)):
        return value
    bad=np.argwhere(~np.isfinite(array));first=bad[0].tolist() if len(bad) else None
    raise NonFiniteGeometryError(
        f"RavaFit non-finite geometry in mesh '{mesh_name}' after {stage_name}: "
        f"count={len(bad)}, first_index={first}, shape={array.shape}"
    )


def _finite_stage_call(mesh_name: str, stage_name: str, fn):
    try:
        return fn()
    except NonFiniteGeometryError:
        raise
    except (ValueError, FloatingPointError) as ex:
        text=str(ex).casefold()
        if ("must be finite" in text or "nan or inf" in text or "non-finite" in text
                or "nonfinite" in text):
            raise NonFiniteGeometryError(
                f"RavaFit non-finite geometry reached {stage_name} for mesh '{mesh_name}': {ex}"
            ) from ex
        raise



def _layer_sample_ids(count: int, limit: int = 4096) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    # Classify relations once, but still evaluate every vertex in the affected layered region.
    return np.unique(np.linspace(0, count - 1, limit, dtype=np.int64))


def _expanded_aabb_overlap(a: np.ndarray, b: np.ndarray, margin: float = .012) -> bool:
    if len(a) == 0 or len(b) == 0:
        return False
    amin=np.min(a,axis=0)-margin;amax=np.max(a,axis=0)+margin
    bmin=np.min(b,axis=0)-margin;bmax=np.max(b,axis=0)+margin
    return bool(np.all(amax>=bmin) and np.all(bmax>=amin))


def _layer_directional_evidence(vertices: np.ndarray, other_triangles: np.ndarray, source_body_triangles: np.ndarray, close_distance: float = .010):
    ids=_layer_sample_ids(len(vertices));sample=np.asarray(vertices,dtype=np.float64)[ids]
    closest,_,_,distance,_=_b14_nearest_surface(sample,other_triangles,k=32)
    _,body_normals,_,_,_=_b14_nearest_surface(sample,source_body_triangles,k=32)
    radial=np.sum((sample-closest)*body_normals,axis=1)
    mask=(distance<=close_distance)&np.isfinite(radial)
    minimum=max(48,int(math.ceil(len(sample)*.025)))
    if int(np.count_nonzero(mask))<minimum or float(np.mean(mask))<.035:
        return None
    values=radial[mask]
    return {
        "count":int(len(values)),"fraction":float(np.mean(mask)),
        "positive_fraction":float(np.mean(values>.00015)),"negative_fraction":float(np.mean(values<-.00015)),
        "median":float(np.median(values)),"p25":float(np.percentile(values,25)),
        "close_p95":float(np.percentile(distance[mask],95)),
    }


def _infer_authored_layer_relations(meshes: dict[str, dict[str, Any]], source_body_triangles: np.ndarray):
    """Infer strong source-authored inner/outer garment relationships."""
    names=sorted(meshes)
    relations=[]
    for ai in range(len(names)):
        a=meshes[names[ai]];av=np.asarray(a["V"],dtype=np.float64)
        for bi in range(ai+1,len(names)):
            b=meshes[names[bi]];bv=np.asarray(b["V"],dtype=np.float64)
            if not _expanded_aabb_overlap(av,bv,.012):
                continue
            at=np.asarray(a["V"],dtype=np.float64)[np.asarray(a["F"],dtype=np.int64)]
            bt=np.asarray(b["V"],dtype=np.float64)[np.asarray(b["F"],dtype=np.int64)]
            ab=_layer_directional_evidence(av,bt,source_body_triangles)
            ba=_layer_directional_evidence(bv,at,source_body_triangles)
            if ab is None or ba is None:
                continue
            outer=inner=None;outer_evidence=inner_evidence=None
            if ab["positive_fraction"]>=.90 and ab["median"]>=.00055 and ba["negative_fraction"]>=.90 and ba["median"]<=-.00055:
                outer,inner=names[ai],names[bi];outer_evidence,inner_evidence=ab,ba
            elif ba["positive_fraction"]>=.90 and ba["median"]>=.00055 and ab["negative_fraction"]>=.90 and ab["median"]<=-.00055:
                outer,inner=names[bi],names[ai];outer_evidence,inner_evidence=ba,ab
            if outer is None:
                continue
            outer_mesh=meshes[outer];inner_mesh=meshes[inner];outer_vertices=np.asarray(outer_mesh["V"],dtype=np.float64);inner_triangles=np.asarray(inner_mesh["V"],dtype=np.float64)[np.asarray(inner_mesh["F"],dtype=np.int64)]
            source_closest,_,_,source_distance,_=_b14_nearest_surface(outer_vertices,inner_triangles,k=32);_,source_body_normals,_,_,_=_b14_nearest_surface(outer_vertices,source_body_triangles,k=32);source_radial=np.sum((outer_vertices-source_closest)*source_body_normals,axis=1)
            layer_mask=(source_distance<=.010)&np.isfinite(source_radial)&(source_radial>.00005);minimum_gap=np.where(layer_mask,np.minimum(source_radial,.00045),0.0)
            influence=float(np.clip(max(float(outer_evidence["close_p95"]),float(inner_evidence["close_p95"]))*1.25,.008,.012))
            relations.append({
                "inner":inner,"outer":outer,"minimum_gap_p50_mm":float(np.median(minimum_gap[layer_mask])*1000.0) if np.any(layer_mask) else 0.0,"influence":influence,
                "source_outer_median_mm":float(outer_evidence["median"]*1000.0),
                "source_near_fraction":float(outer_evidence["fraction"]),
                "_source_minimum_gap":minimum_gap,
            })
    return relations


def _mesh_vertex_neighbours(vertex_count: int, faces: np.ndarray):
    F=np.asarray(faces,dtype=np.int64)
    if len(F)==0:return [[] for _ in range(vertex_count)]
    edges=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
    neighbours=[[] for _ in range(vertex_count)]
    for a,b in edges:
        neighbours[int(a)].append(int(b));neighbours[int(b)].append(int(a))
    return neighbours


def _layer_orientation_alpha(vertices: np.ndarray, faces: np.ndarray, displacement: np.ndarray, push: np.ndarray) -> float:
    F=np.asarray(faces,dtype=np.int64);touched=np.any(np.asarray(push)[F]>1e-8,axis=1)
    if not np.any(touched):return 1.0
    FT=F[touched];base=np.asarray(vertices,dtype=np.float64);base_area=np.cross(base[FT[:,1]]-base[FT[:,0]],base[FT[:,2]]-base[FT[:,0]]);base_norm=np.linalg.norm(base_area,axis=1);valid=base_norm>1e-12
    for alpha in (1.0,.75,.5,.35,.25,.15,.08,.04):
        trial=base+alpha*np.asarray(displacement,dtype=np.float64);area=np.cross(trial[FT[:,1]]-trial[FT[:,0]],trial[FT[:,2]]-trial[FT[:,0]]);norm=np.linalg.norm(area,axis=1)
        dot=np.ones(len(FT));ratio=np.ones(len(FT));dot[valid]=np.sum(base_area[valid]*area[valid],axis=1)/np.maximum(base_norm[valid]*norm[valid],1e-30);ratio[valid]=norm[valid]/np.maximum(base_norm[valid],1e-16)
        if not np.any(valid&((dot<=.03)|(ratio<=.12))):return float(alpha)
    return 0.0


def _enforce_layer_relation(outer_vertices: np.ndarray, outer_faces: np.ndarray, inner_vertices: np.ndarray, inner_faces: np.ndarray, target_body_triangles: np.ndarray, minimum_gap: np.ndarray, influence: float, neighbours: list[list[int]], source_outer_vertices: np.ndarray | None = None):
    """Move only the inferred outer layer outwards."""
    V=np.asarray(outer_vertices,dtype=np.float64).copy();inner_tri=np.asarray(inner_vertices,dtype=np.float64)[np.asarray(inner_faces,dtype=np.int64)];total=np.zeros(len(V),dtype=np.float64);rounds=[];minimum_gap=np.asarray(minimum_gap,dtype=np.float64)
    if minimum_gap.shape!=(len(V),):
        return V,{"before_min_mm":None,"after_min_mm":None,"before_violations":0,"after_violations":0,"moved_vertices":0,"max_move_mm":0.0,"rounds":[],"skipped":"source/target vertex correspondence changed before layer guard"}
    before_min=None;before_violations=0
    for round_index in range(3):
        closest,_,_,distance,_=_b14_nearest_surface(V,inner_tri,k=32);_,body_normals,_,_,_=_b14_nearest_surface(V,target_body_triangles,k=32);radial=np.sum((V-closest)*body_normals,axis=1)
        eligible=(distance<=influence)&(minimum_gap>0.0)
        if round_index==0 and np.any(eligible):
            before_min=float(np.min(radial[eligible])*1000.0);before_violations=int(np.count_nonzero(eligible&(radial<minimum_gap-.00002)))
        required=np.where(eligible,np.maximum(0.0,minimum_gap-radial),0.0);required=np.minimum(required,np.maximum(0.0,.006-total))
        if float(np.max(required,initial=0.0))<1e-6:break
        field=required.copy()
        for _ in range(8):
            average=np.asarray([np.mean(field[n]) if n else field[i] for i,n in enumerate(neighbours)],dtype=np.float64);field=np.maximum(required,.78*required+.22*average)
        push=np.minimum(field,.003);push[distance>influence*1.25]=0.0;displacement=body_normals*push[:,None];alpha=_layer_orientation_alpha(V,outer_faces,displacement,push)
        if alpha<=0.0:break
        proposed=V+alpha*displacement
        if source_outer_vertices is not None:
            safe,safe_alpha,_,edge_metric=_coupled_topology_safe_alpha(np.asarray(source_outer_vertices,dtype=np.float64),V,proposed,outer_faces)
            alpha*=safe_alpha;proposed=safe
        else:
            edge_metric=None
        if alpha<=0.0:break
        moved=np.linalg.norm(proposed-V,axis=1);V=proposed;total+=moved;rounds.append({"round":round_index+1,"vertices":int(np.count_nonzero(required>1e-6)),"alpha":float(alpha),"max_requested_mm":float(np.max(required)*1000.0),"edge":edge_metric})
    closest,_,_,distance,_=_b14_nearest_surface(V,inner_tri,k=32);_,body_normals,_,_,_=_b14_nearest_surface(V,target_body_triangles,k=32);radial=np.sum((V-closest)*body_normals,axis=1);eligible=(distance<=influence)&(minimum_gap>0.0)
    after_min=float(np.min(radial[eligible])*1000.0) if np.any(eligible) else None;after_violations=int(np.count_nonzero(eligible&(radial<minimum_gap-.00002))) if np.any(eligible) else 0
    authored=minimum_gap[minimum_gap>0.0]
    return V,{"before_min_mm":before_min,"after_min_mm":after_min,"before_violations":before_violations,"after_violations":after_violations,"moved_vertices":int(np.count_nonzero(total>1e-6)),"max_move_mm":float(np.max(total,initial=0.0)*1000.0),"authored_minimum_p50_mm":float(np.median(authored)*1000.0) if len(authored) else 0.0,"rounds":rounds}


def _preserve_authored_garment_layers(source_meshes: dict[str, dict[str, Any]], positions: dict[str, np.ndarray], source_body_triangles: np.ndarray, target_body_triangles: np.ndarray):
    candidates={name:data for name,data in source_meshes.items() if name in positions and len(data.get("V",[]))>=64 and len(data.get("F",[]))>=32}
    relations=_infer_authored_layer_relations(candidates,source_body_triangles)
    if not relations:return positions,set(),{"relation_count":0,"adjusted_mesh_count":0,"relations":[]}

    # Apply layers inner-to-outer; skip contradictory cycles rather than guessing.
    nodes=sorted({r["inner"] for r in relations}|{r["outer"] for r in relations});indegree={n:0 for n in nodes};edges={n:[] for n in nodes}
    for r in relations:edges[r["inner"]].append(r["outer"]);indegree[r["outer"]]+=1
    queue=sorted([n for n in nodes if indegree[n]==0]);order=[]
    while queue:
        n=queue.pop(0);order.append(n)
        for child in sorted(edges[n]):
            indegree[child]-=1
            if indegree[child]==0:queue.append(child);queue.sort()
    valid=set(order)
    if len(order)!=len(nodes):
        cyclic={n for n in nodes if indegree[n]>0};relations=[r for r in relations if r["inner"] not in cyclic and r["outer"] not in cyclic]
    rank={name:i for i,name in enumerate(order)};relations.sort(key=lambda r:(rank.get(r["outer"],999),rank.get(r["inner"],999),r["inner"],r["outer"]))

    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};changed=set();reports=[];neighbour_cache={name:_mesh_vertex_neighbours(len(out[name]),np.asarray(candidates[name]["F"],dtype=np.int64)) for name in candidates}
    per_relation={}
    for pass_index in range(2):
        for relation in relations:
            inner=relation["inner"];outer=relation["outer"]
            corrected,report=_enforce_layer_relation(out[outer],np.asarray(candidates[outer]["F"],dtype=np.int64),out[inner],np.asarray(candidates[inner]["F"],dtype=np.int64),target_body_triangles,np.asarray(relation["_source_minimum_gap"],dtype=np.float64),float(relation["influence"]),neighbour_cache[outer],np.asarray(candidates[outer]["V"],dtype=np.float64))
            if report["moved_vertices"]>0:
                out[outer]=corrected;changed.add(outer)
            key=(inner,outer);public_relation={k:v for k,v in relation.items() if not k.startswith("_")};entry=per_relation.setdefault(key,{**public_relation,"passes":[]});entry["passes"].append({"pass":pass_index+1,**report})
    reports=list(per_relation.values())
    return out,changed,{"relation_count":len(relations),"adjusted_mesh_count":len(changed),"relations":reports}



def _local_layer_directional_evidence(vertices: np.ndarray, other_triangles: np.ndarray, source_body_triangles: np.ndarray, close_distance: float=.009):
    sample=np.asarray(vertices,dtype=np.float64)
    if len(sample)>2048:sample=sample[np.unique(np.linspace(0,len(sample)-1,2048,dtype=np.int64))]
    if len(sample)<16 or len(other_triangles)==0:return None
    closest,_,_,distance,_=_b14_nearest_surface(sample,other_triangles,k=28);_,body_normals,_,_,_=_b14_nearest_surface(sample,source_body_triangles,k=28);radial=np.sum((sample-closest)*body_normals,axis=1);mask=(distance<=close_distance)&np.isfinite(radial)
    minimum=max(14,int(math.ceil(len(sample)*.04)))
    if int(np.count_nonzero(mask))<minimum or float(np.mean(mask))<.06:return None
    values=radial[mask]
    return {"count":int(len(values)),"fraction":float(np.mean(mask)),"positive_fraction":float(np.mean(values>.00012)),"negative_fraction":float(np.mean(values<-.00012)),"median":float(np.median(values)),"close_p95":float(np.percentile(distance[mask],95))}


def _local_layer_nodes(meshes: dict[str,dict[str,Any]]):
    nodes=[];component_counts={}
    for name in sorted(meshes):
        V=np.asarray(meshes[name].get("V",[]),dtype=np.float64);F=np.asarray(meshes[name].get("F",[]),dtype=np.int64)
        if len(V)<32 or len(F)<16:component_counts[name]=0;continue
        components=_masked_vertex_components(F,np.ones(len(V),dtype=bool));component_counts[name]=len(components)
        for ci,ids in enumerate(components):
            ids=np.asarray(ids,dtype=np.int64)
            if len(ids)<32:continue
            LF=_component_local_faces(F,ids,len(V))
            if len(LF)<16:continue
            P=V[ids];feature=_structural_component_features(P,LF)
            # Very small rigid/detail pieces are handled by the structural attachment lane, not cloth layering.
            if len(ids)<48 and feature["max_extent"]<.040:continue
            nodes.append({"mesh":name,"component":int(ci),"ids":ids,"V":P,"F":LF,"features":feature})
    return nodes,component_counts


def _infer_local_authored_layer_relations(meshes: dict[str,dict[str,Any]], source_body_triangles: np.ndarray):
    nodes,component_counts=_local_layer_nodes(meshes);relations=[]
    for ai,a in enumerate(nodes):
        for b in nodes[ai+1:]:
            # Whole single-component mesh pairs are already handled by the established mesh-level relation pass.
            if a["mesh"]!=b["mesh"] and component_counts.get(a["mesh"],0)<=1 and component_counts.get(b["mesh"],0)<=1:continue
            av=np.asarray(a["V"],dtype=np.float64);bv=np.asarray(b["V"],dtype=np.float64)
            if not _expanded_aabb_overlap(av,bv,.010):continue
            at=av[np.asarray(a["F"],dtype=np.int64)];bt=bv[np.asarray(b["F"],dtype=np.int64)]
            ab=_local_layer_directional_evidence(av,bt,source_body_triangles);ba=_local_layer_directional_evidence(bv,at,source_body_triangles)
            if ab is None or ba is None:continue
            outer=inner=None;oe=ie=None
            if ab["positive_fraction"]>=.84 and ab["median"]>=.00032 and ba["negative_fraction"]>=.84 and ba["median"]<=-.00032:outer,inner,oe,ie=a,b,ab,ba
            elif ba["positive_fraction"]>=.84 and ba["median"]>=.00032 and ab["negative_fraction"]>=.84 and ab["median"]<=-.00032:outer,inner,oe,ie=b,a,ba,ab
            if outer is None:continue
            OV=np.asarray(outer["V"],dtype=np.float64);IT=np.asarray(inner["V"],dtype=np.float64)[np.asarray(inner["F"],dtype=np.int64)]
            closest,_,_,distance,_=_b14_nearest_surface(OV,IT,k=28);_,body_normals,_,_,_=_b14_nearest_surface(OV,source_body_triangles,k=28);radial=np.sum((OV-closest)*body_normals,axis=1);mask=(distance<=.0095)&np.isfinite(radial)&(radial>.00005)
            if int(np.count_nonzero(mask))<max(12,int(len(OV)*.025)):continue
            minimum=np.zeros(len(OV),dtype=np.float64);minimum[mask]=np.clip(radial[mask]*.90,.00012,.00125)
            influence=float(np.clip(max(float(oe["close_p95"]),float(ie["close_p95"]))*1.35,.0055,.0115))
            relations.append({"inner_mesh":inner["mesh"],"inner_component":int(inner["component"]),"inner_ids":inner["ids"],"outer_mesh":outer["mesh"],"outer_component":int(outer["component"]),"outer_ids":outer["ids"],"minimum_gap":minimum,"influence":influence,"source_outer_median_mm":float(oe["median"]*1000.0),"source_near_fraction":float(oe["fraction"])})
    return relations


def _preserve_local_authored_garment_layers(source_meshes: dict[str,dict[str,Any]], positions: dict[str,np.ndarray], source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, relations: list[dict[str,Any]] | None=None, pass_count: int=2, quick_verify: bool=False):
    """Restore source-proven local layer order, including disconnected components inside one mesh.

    Runtime 11 may reuse source-authored relations after clearance.  The source relation graph is immutable,
    so recomputing it after every correction was wasted work and could become pathological on highly
    fragmented XIV garments.  A lightweight verification pass skips relations that remain safely ordered.
    """
    if relations is None:relations=_infer_local_authored_layer_relations(source_meshes,source_body_triangles)
    if not relations:return positions,set(),{"relation_count":0,"adjusted_mesh_count":0,"relations":[],"policy":"component-local source-authored layering"}
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};changed=set();reports=[None]*len(relations);skipped_safe=0
    for pass_index in range(max(1,int(pass_count))):
        for ri,relation in enumerate(relations):
            om=str(relation["outer_mesh"]);im=str(relation["inner_mesh"]);oids=np.asarray(relation["outer_ids"],dtype=np.int64);iids=np.asarray(relation["inner_ids"],dtype=np.int64)
            if om not in out or im not in out:continue
            OF=_component_local_faces(np.asarray(source_meshes[om]["F"],dtype=np.int64),oids,len(source_meshes[om]["V"]));IF=_component_local_faces(np.asarray(source_meshes[im]["F"],dtype=np.int64),iids,len(source_meshes[im]["V"]))
            if len(OF)==0 or len(IF)==0:continue
            minimum=np.asarray(relation["minimum_gap"],dtype=np.float64)
            if quick_verify and pass_index==0:
                sample=np.unique(np.linspace(0,len(oids)-1,min(len(oids),192),dtype=np.int64));OV=out[om][oids][sample];IT=out[im][iids][IF]
                closest,_,_,distance,_=_b14_nearest_surface(OV,IT,k=20);_,body_normals,_,_,_=_b14_nearest_surface(OV,target_body_triangles,k=20);radial=np.sum((OV-closest)*body_normals,axis=1);eligible=(distance<=float(relation["influence"]))&(minimum[sample]>0.0)
                if not np.any(eligible) or int(np.count_nonzero(eligible&(radial<minimum[sample]-.00008)))==0:
                    skipped_safe+=1
                    if reports[ri] is None:reports[ri]={"inner_mesh":im,"inner_component":int(relation["inner_component"]),"outer_mesh":om,"outer_component":int(relation["outer_component"]),"source_outer_median_mm":float(relation["source_outer_median_mm"]),"source_near_fraction":float(relation["source_near_fraction"]),"influence_mm":float(relation["influence"]*1000.0),"passes":[],"quick_verified_safe":True}
                    continue
            neighbours=_mesh_vertex_neighbours(len(oids),OF);before=out[om][oids].copy();corrected,report=_enforce_layer_relation(before,OF,out[im][iids],IF,target_body_triangles,minimum,float(relation["influence"]),neighbours,np.asarray(source_meshes[om]["V"],dtype=np.float64)[oids])
            if report.get("moved_vertices",0)>0:out[om][oids]=corrected;changed.add(om)
            if reports[ri] is None:reports[ri]={"inner_mesh":im,"inner_component":int(relation["inner_component"]),"outer_mesh":om,"outer_component":int(relation["outer_component"]),"source_outer_median_mm":float(relation["source_outer_median_mm"]),"source_near_fraction":float(relation["source_near_fraction"]),"influence_mm":float(relation["influence"]*1000.0),"passes":[report]}
            else:reports[ri].setdefault("passes",[]).append(report)
    reports=[r for r in reports if r is not None];public_pairs=sorted({(str(r["inner_mesh"]),str(r["outer_mesh"])) for r in relations if r["inner_mesh"]!=r["outer_mesh"]})
    return out,changed,{"relation_count":len(relations),"adjusted_mesh_count":len(changed),"relations":reports,"cross_mesh_pairs":public_pairs,"quick_verified_safe":int(skipped_safe),"policy":"component-local source-authored layering including same-mesh disconnected components; immutable relations reused across convergence passes"}

def _boundary_vertex_ids(faces: np.ndarray) -> np.ndarray:
    F=np.asarray(faces,dtype=np.int64)
    if len(F)==0:return np.empty(0,dtype=np.int64)
    edges=np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=unique[counts==1]
    return np.unique(boundary) if len(boundary) else np.empty(0,dtype=np.int64)


def _infer_authored_cross_mesh_relations(source_meshes: dict[str, dict[str, Any]], excluded_pairs: set[frozenset[str]] | None = None):
    """Find separate meshes whose authored open boundaries demonstrably belong together."""
    excluded_pairs=excluded_pairs or set(); names=sorted(source_meshes);relations=[]
    boundary_cache={}
    for name in names:
        data=source_meshes[name];V=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64)
        ids=_boundary_vertex_ids(F);boundary_cache[name]=(V,F,ids)
    for ai,a in enumerate(names):
        AV,AF,aids=boundary_cache[a]
        if len(aids)<4:continue
        for b in names[ai+1:]:
            if frozenset((a,b)) in excluded_pairs:continue
            BV,BF,bids=boundary_cache[b]
            if len(bids)<4:continue
            apts=AV[aids];bpts=BV[bids]
            tree_b=cKDTree(bpts);dab,jab=tree_b.query(apts,k=1)
            tree_a=cKDTree(apts);dba,jba=tree_a.query(bpts,k=1)
            rows=[]
            for local_a,(distance,local_b) in enumerate(zip(np.asarray(dab),np.asarray(jab,dtype=np.int64))):
                local_b=int(local_b)
                if float(distance)>.004 or float(dba[local_b])>.004 or int(jba[local_b])!=local_a:continue
                rows.append((int(aids[local_a]),int(bids[local_b]),float(distance)))
            if not rows:continue
            support_fraction=len(rows)/max(1,min(len(aids),len(bids)))
            distances=np.asarray([row[2] for row in rows],dtype=np.float64)
            exact_fraction=float(np.mean(distances<=.00020))
            if not (len(rows)>=12 and support_fraction>=.04) and not (len(rows)>=8 and exact_fraction>=.75):continue
            relations.append({
                "a":a,"b":b,"pair_count":int(len(rows)),"support_fraction":float(support_fraction),
                "source_gap_p50_mm":float(np.median(distances)*1000.0),"source_gap_p95_mm":float(np.percentile(distances,95)*1000.0),
                "_pairs":np.asarray([[row[0],row[1]] for row in rows],dtype=np.int64),
                "_source_gap":np.asarray([BV[row[1]]-AV[row[0]] for row in rows],dtype=np.float64),
                "_source_distance":distances,
            })
    return relations


def _preserve_authored_cross_mesh_assembly(source_meshes: dict[str, dict[str, Any]], positions: dict[str, np.ndarray], excluded_pairs: set[frozenset[str]] | None = None):
    """Preserve source-proven seam/assembly registration across separate garment meshes."""
    candidates={name:data for name,data in source_meshes.items() if name in positions and len(data.get("V",[]))>=3 and len(data.get("F",[]))>=1}
    relations=_infer_authored_cross_mesh_relations(candidates,excluded_pairs)
    if not relations:return positions,set(),{"relation_count":0,"adjusted_mesh_count":0,"relations":[]}
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};start_positions={name:out[name].copy() for name in candidates};changed=set();passes=[]
    neighbours={name:_mesh_vertex_neighbours(len(np.asarray(candidates[name]["V"])),np.asarray(candidates[name]["F"],dtype=np.int64)) for name in candidates}
    for pass_index in range(5):
        accum={name:np.zeros_like(out[name]) for name in candidates};weight={name:np.zeros(len(out[name]),dtype=np.float64) for name in candidates};pass_errors=[]
        for relation in relations:
            a=relation["a"];b=relation["b"];pairs=np.asarray(relation["_pairs"],dtype=np.int64);source_gap=np.asarray(relation["_source_gap"],dtype=np.float64);source_distance=np.asarray(relation["_source_distance"],dtype=np.float64)
            current_gap=out[b][pairs[:,1]]-out[a][pairs[:,0]];error=current_gap-source_gap;magnitude=np.linalg.norm(error,axis=1);pass_errors.extend(magnitude.tolist())
            scale=np.minimum(1.0,.004/np.maximum(magnitude,1e-12));error*=scale[:,None]
            authority=np.exp(-np.square(source_distance/.004));correction=.5*error*authority[:,None]
            for row,(ia,ib) in enumerate(pairs):
                ia=int(ia);ib=int(ib);w=float(authority[row])
                if w<=1e-6:continue
                accum[a][ia]+=correction[row];weight[a][ia]+=w;accum[b][ib]-=correction[row];weight[b][ib]+=w
        pass_moved=0
        for name in candidates:
            direct=weight[name]>0.0
            if not np.any(direct):continue
            delta=np.zeros_like(out[name]);delta[direct]=accum[name][direct]/weight[name][direct,None]
            feather=np.zeros_like(delta);feather_weight=np.zeros(len(delta),dtype=np.float64)
            for vi in np.flatnonzero(direct):
                for nb in neighbours[name][int(vi)]:
                    if direct[int(nb)]:continue
                    feather[int(nb)]+=delta[int(vi)]*.20;feather_weight[int(nb)]+=.20
            fmask=feather_weight>0.0;delta[fmask]+=feather[fmask]/feather_weight[fmask,None]
            mag=np.linalg.norm(delta,axis=1);delta*=np.minimum(1.0,.002/np.maximum(mag,1e-12))[:,None]
            candidate=out[name]+delta;total=candidate-start_positions[name];total_mag=np.linalg.norm(total,axis=1);total*=np.minimum(1.0,.004/np.maximum(total_mag,1e-12))[:,None];candidate=start_positions[name]+total
            moved=np.linalg.norm(candidate-out[name],axis=1)>1e-7
            if np.any(moved):out[name]=candidate;changed.add(name);pass_moved+=int(np.count_nonzero(moved))
        err=np.asarray(pass_errors,dtype=np.float64)
        passes.append({"pass":pass_index+1,"moved_vertices":pass_moved,"gap_error_p50_mm":float(np.percentile(err,50)*1000.0) if len(err) else 0.0,"gap_error_p95_mm":float(np.percentile(err,95)*1000.0) if len(err) else 0.0})
        if pass_moved==0 or (len(err) and float(np.percentile(err,95))<=.00025):break
    relation_public=[]
    for relation in relations:
        a=relation["a"];b=relation["b"];pairs=np.asarray(relation["_pairs"],dtype=np.int64);source_gap=np.asarray(relation["_source_gap"],dtype=np.float64)
        final_error=np.linalg.norm((out[b][pairs[:,1]]-out[a][pairs[:,0]])-source_gap,axis=1)
        total_move_a=np.linalg.norm(out[a]-start_positions[a],axis=1);total_move_b=np.linalg.norm(out[b]-start_positions[b],axis=1)
        relation_public.append({k:v for k,v in relation.items() if not k.startswith("_")}|{"final_gap_error_p50_mm":float(np.percentile(final_error,50)*1000.0),"final_gap_error_p95_mm":float(np.percentile(final_error,95)*1000.0),"final_gap_error_max_mm":float(np.max(final_error,initial=0.0)*1000.0),"max_mesh_correction_mm":float(max(np.max(total_move_a,initial=0.0),np.max(total_move_b,initial=0.0))*1000.0)})
    return out,changed,{"relation_count":len(relations),"adjusted_mesh_count":len(changed),"relations":relation_public,"passes":passes,"policy":"iterative mutual source-boundary registration; source-mode agnostic; 4 mm cumulative local cap"}


def _compile_authored_weld_split_topology(context: dict[str,Any] | None, vertex_count: int):
    """Compile source-authored seam connectivity once; geometry remains dynamic on every pass."""
    if context is None:return None
    w=context.get("w");data=context.get("data")
    raw_to_weld=None if w is None else np.asarray(w.get("raw_to_weld",[]),dtype=np.int64)
    raw_F=np.asarray((data or {}).get("F",[]),dtype=np.int64);raw_W=np.asarray((data or {}).get("W",[]),dtype=np.float64);source_V=np.asarray((data or {}).get("V",[]),dtype=np.float64)
    if raw_to_weld.shape!=(int(vertex_count),) or raw_W.ndim!=2 or raw_W.shape[0]!=int(vertex_count) or raw_F.ndim!=2 or (len(raw_F) and raw_F.shape[1]!=3):return None
    if source_V.shape!=(int(vertex_count),3):source_V=np.zeros((int(vertex_count),3),dtype=np.float64)
    key=(int(vertex_count),int(len(raw_F)),int(np.asarray(raw_F,dtype=np.int64).sum(dtype=np.int64)),int(np.asarray(raw_to_weld,dtype=np.int64).sum(dtype=np.int64)),int(raw_W.shape[1]))
    cached=context.get("_ravafit_authored_weld_split_topology")
    if isinstance(cached,dict) and cached.get("key")==key:return cached

    raw_neighbours=[set() for _ in range(int(vertex_count))];incident=[[] for _ in range(int(vertex_count))];adjacency=[set() for _ in range(int(vertex_count))]
    for fi,(a,b,c) in enumerate(raw_F):
        a=int(a);b=int(b);c=int(c)
        if min(a,b,c)<0 or max(a,b,c)>=int(vertex_count):continue
        incident[a].append(fi);incident[b].append(fi);incident[c].append(fi)
        adjacency[a].update((b,c));adjacency[b].update((a,c));adjacency[c].update((a,b))
        raw_neighbours[a].update((int(raw_to_weld[b]),int(raw_to_weld[c])))
        raw_neighbours[b].update((int(raw_to_weld[a]),int(raw_to_weld[c])))
        raw_neighbours[c].update((int(raw_to_weld[a]),int(raw_to_weld[b])))
    comp=np.full(int(vertex_count),-1,dtype=np.int64);cc=0
    for seed in range(int(vertex_count)):
        if comp[seed]>=0:continue
        stack=[seed];comp[seed]=cc
        while stack:
            cur=stack.pop()
            for nxt in adjacency[cur]:
                if comp[nxt]<0:comp[nxt]=cc;stack.append(nxt)
        cc+=1

    order=np.argsort(raw_to_weld,kind="stable");sorted_weld=raw_to_weld[order];boundaries=np.r_[0,np.flatnonzero(np.diff(sorted_weld))+1,len(sorted_weld)]
    duplicate_group_count=0;eligible_groups=0;cross_component_rejections=0;weight_rejected_pairs=0;entries=[]
    for group_index,(first,last) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        if last-first<2:continue
        duplicate_group_count+=1;group_ids=order[first:last];by_component={}
        for i in group_ids:by_component.setdefault(int(comp[int(i)]),[]).append(int(i))
        if len(by_component)>1:cross_component_rejections+=sum(len(v) for v in by_component.values())
        any_seam=False
        for component_ids in by_component.values():
            if len(component_ids)<2:continue
            ids=np.asarray(component_ids,dtype=np.int64);parent={int(i):int(i) for i in ids}
            def find(x):
                while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
                return x
            def union(a,b):
                ra,rb=find(a),find(b)
                if ra!=rb:parent[rb]=ra
            for ai in range(len(ids)):
                a=int(ids[ai])
                for bi in range(ai+1,len(ids)):
                    b=int(ids[bi]);shared=raw_neighbours[a]&raw_neighbours[b]
                    if not shared:continue
                    if float(np.sum(np.abs(raw_W[a]-raw_W[b])))>.08:weight_rejected_pairs+=1;continue
                    union(a,b)
            components={}
            for i in ids:components.setdefault(find(int(i)),[]).append(int(i))
            seam_components=[np.asarray(c,dtype=np.int64) for c in components.values() if len(c)>=2]
            if not seam_components:continue
            any_seam=True
            for seam_ids in seam_components:
                face_ids=sorted({fi for vi in seam_ids for fi in incident[int(vi)]})
                if not face_ids:
                    entries.append({"group":group_index,"seam_ids":seam_ids,"face_ids":np.empty(0,dtype=np.int64),"patch_ids":np.empty(0,dtype=np.int64),"LF":np.empty((0,3),dtype=np.int64),"seam_local":np.empty(0,dtype=np.int64)})
                    continue
                face_ids=np.asarray(face_ids,dtype=np.int64);patch_ids=np.unique(raw_F[face_ids].reshape(-1));lookup={int(v):i for i,v in enumerate(patch_ids.tolist())};LF=np.asarray([[lookup[int(x)] for x in tri] for tri in raw_F[face_ids]],dtype=np.int64);seam_local=np.asarray([lookup[int(i)] for i in seam_ids],dtype=np.int64)
                entries.append({"group":group_index,"seam_ids":seam_ids,"face_ids":face_ids,"patch_ids":patch_ids,"LF":LF,"seam_local":seam_local})
        if any_seam:eligible_groups+=1
    cached={"key":key,"raw_F":raw_F,"source_V":source_V,"duplicate_group_count":int(duplicate_group_count),"eligible_group_count":int(eligible_groups),"disconnected_group_count":int(duplicate_group_count-eligible_groups),"cross_component_rejections":int(cross_component_rejections),"weight_rejected_pairs":int(weight_rejected_pairs),"entries":entries}
    context["_ravafit_authored_weld_split_topology"]=cached
    return cached


def _preserve_authored_weld_splits(positions: dict[str, np.ndarray], retarget_contexts: dict[str, dict[str, Any]]):
    """Reconcile genuine render/UV seams while reusing immutable authored seam topology."""
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};changed=set();mesh_reports={};all_splits=[];all_adjustments=[]
    for name in sorted(out):
        context=retarget_contexts.get(name);V=out[name];compiled=_compile_authored_weld_split_topology(context,len(V))
        if compiled is None:
            mesh_reports[name]={"weld_group_count":0,"eligible_seam_group_count":0,"corrected_group_count":0,"corrected_vertices":0,"split_max_mm":0.0,"adjustment_max_mm":0.0,"skipped":"missing authored weld/topology/skinning correspondence"};continue
        source_V=np.asarray(compiled["source_V"],dtype=np.float64)
        if source_V.shape!=V.shape:source_V=V.copy()
        topology_rejected=0;large_split_rejected=0;corrected_groups=0;corrected_vertices=0;split_values=[];adjustment_values=[]
        for entry in compiled["entries"]:
            seam_ids=np.asarray(entry["seam_ids"],dtype=np.int64);pts=V[seam_ids];delta=pts[:,None,:]-pts[None,:,:];split=float(np.max(np.linalg.norm(delta,axis=2),initial=0.0));split_values.append(split)
            if split<=1e-7:continue
            if split>.0080:large_split_rejected+=1;continue
            patch_ids=np.asarray(entry["patch_ids"],dtype=np.int64);LF=np.asarray(entry["LF"],dtype=np.int64);seam_local=np.asarray(entry["seam_local"],dtype=np.int64)
            if len(patch_ids)==0 or len(LF)==0:continue
            centre=np.mean(pts,axis=0);disp=centre-pts;dmag=np.linalg.norm(disp,axis=1);cap=.0020;scale=np.minimum(1.0,cap/np.maximum(dmag,1e-12));target=pts+disp*scale[:,None]
            before=V[patch_ids].copy();proposed=before.copy();proposed[seam_local]=target
            safe,alpha,_,_=_coupled_topology_safe_alpha(source_V[patch_ids],before,proposed,LF)
            if alpha<=0.0:topology_rejected+=1;continue
            move=np.linalg.norm(safe-before,axis=1);V[patch_ids]=safe;actual=move[seam_local]
            corrected_groups+=1;corrected_vertices+=len(seam_ids);adjustment_values.extend(actual.tolist())
        if corrected_groups:changed.add(name)
        all_splits.extend(split_values);all_adjustments.extend(adjustment_values)
        mesh_reports[name]={"weld_group_count":compiled["duplicate_group_count"],"eligible_seam_group_count":compiled["eligible_group_count"],"disconnected_coincident_group_count":compiled["disconnected_group_count"],"cross_component_rejections":compiled["cross_component_rejections"],"weight_rejected_pair_count":compiled["weight_rejected_pairs"],"topology_rejected_group_count":int(topology_rejected),"large_split_rejected_group_count":int(large_split_rejected),"corrected_group_count":int(corrected_groups),"corrected_vertices":int(corrected_vertices),"split_p95_mm":float(np.percentile(split_values,95)*1000.0) if split_values else 0.0,"split_max_mm":float(max(split_values,default=0.0)*1000.0),"adjustment_p95_mm":float(np.percentile(adjustment_values,95)*1000.0) if adjustment_values else 0.0,"adjustment_max_mm":float(max(adjustment_values,default=0.0)*1000.0)}
    return out,changed,{"enabled":True,"authority":"authored raw connectivity + welded neighbourhood + skinning + local topology veto","adjusted_mesh_count":int(len(changed)),"corrected_group_count":int(sum(r.get("corrected_group_count",0) for r in mesh_reports.values())),"corrected_vertices":int(sum(r.get("corrected_vertices",0) for r in mesh_reports.values())),"split_p95_mm":float(np.percentile(all_splits,95)*1000.0) if all_splits else 0.0,"split_max_mm":float(max(all_splits,default=0.0)*1000.0),"adjustment_p95_mm":float(np.percentile(all_adjustments,95)*1000.0) if all_adjustments else 0.0,"adjustment_max_mm":float(max(all_adjustments,default=0.0)*1000.0),"meshes":mesh_reports}

def _weighted_rigid_projection(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    P=np.asarray(source,dtype=np.float64);Q=np.asarray(target,dtype=np.float64);w=np.maximum(np.asarray(weights,dtype=np.float64),0.0)
    total=float(np.sum(w))
    if len(P)<3 or total<=1e-12:return Q.copy()
    w/=total;cp=np.sum(P*w[:,None],axis=0);cq=np.sum(Q*w[:,None],axis=0);A=P-cp;B=Q-cq
    U,_,Vt=np.linalg.svd((A*w[:,None]).T@B);R=Vt.T@U.T
    if np.linalg.det(R)<0:Vt[-1]*=-1;R=Vt.T@U.T
    return A@R.T+cq


def _extreme_macro_body_field(points: np.ndarray, garment_weights: np.ndarray, cache: dict[str, Any]):
    X=np.asarray(cache["X"],dtype=np.float64);Y=np.asarray(cache["Y"],dtype=np.float64);BW=np.asarray(cache["BW"],dtype=np.float64);P=np.asarray(points,dtype=np.float64);Wg=np.asarray(garment_weights,dtype=np.float64)
    body_move=np.linalg.norm(Y-X,axis=1);p95=float(np.percentile(body_move,95)) if len(body_move) else 0.0
    if p95<.055 or len(X)<8:return None,{"enabled":False,"body_move_p95_mm":p95*1000.0}
    span=np.ptp(X,axis=0);sigma=float(np.clip(np.max(span)*.014,.012,.024));k=min(128,len(X));tree=cKDTree(X);distance,index=tree.query(P,k=k)
    if distance.ndim==1:distance=distance[:,None];index=index[:,None]
    # Keep the exact per-vertex macro-transfer mathematics, but never materialise BW[index]
    # for an entire large garment.  On heavily decorated meshes that temporary can be hundreds
    # of megabytes even though each vertex is independent.  Bounded row chunks preserve the
    # same neighbour set, reductions and returned blend/index fields with a fixed peak footprint.
    weight=np.empty_like(distance,dtype=np.float64);mapped=np.empty_like(P,dtype=np.float64);weighted_distance=np.empty(len(P),dtype=np.float64);weighted_alignment=np.empty(len(P),dtype=np.float64)
    chunk_rows=1024
    movement=Y-X
    for start in range(0,len(P),chunk_rows):
        stop=min(len(P),start+chunk_rows);local_index=index[start:stop];local_distance=distance[start:stop];local_points=P[start:stop];local_garment_weights=Wg[start:stop]
        alignment=np.einsum("nk,nqk->nq",local_garment_weights,BW[local_index]);local_weight=np.exp(-((local_distance/max(sigma,1e-6))**2))*np.clip(alignment,.05,1.0)
        cross=(np.abs(local_points[:,0,None])>.020)&(np.sign(local_points[:,0,None])!=np.sign(X[local_index][:,:,0]));local_weight*=np.where(cross,np.exp(-5.0),1.0);local_weight/=np.maximum(local_weight.sum(axis=1,keepdims=True),1e-12)
        weight[start:stop]=local_weight;mapped[start:stop]=local_points+np.sum(movement[local_index]*local_weight[:,:,None],axis=1);weighted_distance[start:stop]=np.sum(local_distance*local_weight,axis=1);weighted_alignment[start:stop]=np.sum(alignment*local_weight,axis=1)
    best=np.argmax(weight,axis=1);contact=index[np.arange(len(P)),best]
    return (mapped,contact,weighted_distance,weighted_alignment,weight,index),{"enabled":True,"body_move_p95_mm":p95*1000.0,"sigma_mm":sigma*1000.0,"samples":int(k),"chunk_rows":int(chunk_rows)}


def _apply_target_relief_correction(source_vertices: np.ndarray, faces: np.ndarray, mapped: np.ndarray, blend: np.ndarray, blend_ids: np.ndarray, cache: dict[str, Any], behavior: str, features: dict[str, Any]):
    correction=np.asarray(cache.get("target_relief_C"),dtype=np.float64)
    if correction.ndim!=2 or correction.shape!=(len(cache["X"]),3):return np.asarray(mapped,dtype=np.float64),{"enabled":False,"reason":"no relief field"}
    blend=np.asarray(blend,dtype=np.float64);blend_ids=np.asarray(blend_ids,dtype=np.int64);source_vertices=np.asarray(source_vertices,dtype=np.float64);faces=np.asarray(faces,dtype=np.int64);mapped=np.asarray(mapped,dtype=np.float64)
    sampled=np.sum(correction[blend_ids]*blend[:,:,None],axis=1)
    base_strength={"stand_off_structured_shell":1.0,"conservative_component_assembly":1.0,"constructed_close_shell":.86,"body_following_flexible_layer":.74}.get(str(behavior),.90)
    clearance=float(features.get("source_clearance_median_mm",0.0));strength=float(np.clip(base_strength+(1.0-base_strength)*np.clip(clearance/35.0,0.0,1.0),0.0,1.0));applied=sampled*strength
    backoff=np.ones(len(mapped),dtype=np.float64);rounds=0;backed=np.zeros(len(mapped),dtype=bool)
    if len(faces):
        source_face=np.cross(source_vertices[faces[:,1]]-source_vertices[faces[:,0]],source_vertices[faces[:,2]]-source_vertices[faces[:,0]]);base_face=np.cross(mapped[faces[:,1]]-mapped[faces[:,0]],mapped[faces[:,2]]-mapped[faces[:,0]]);source_len=np.linalg.norm(source_face,axis=1);base_len=np.linalg.norm(base_face,axis=1);base_alignment=np.einsum("ij,ij->i",source_face,base_face)/np.maximum(source_len*base_len,1e-24);baseline_good=(base_alignment>.05)&(base_len>source_len*.10)
        for _ in range(6):
            trial=mapped+applied*backoff[:,None];face=np.cross(trial[faces[:,1]]-trial[faces[:,0]],trial[faces[:,2]]-trial[faces[:,0]]);length=np.linalg.norm(face,axis=1);alignment=np.einsum("ij,ij->i",source_face,face)/np.maximum(source_len*length,1e-24);bad=baseline_good&((alignment<=.05)|(length<=source_len*.10))
            if not np.any(bad):break
            ids=np.unique(faces[bad].reshape(-1));backoff[ids]*=.5;backed[ids]=True;rounds+=1
    applied*=backoff[:,None];amount=np.linalg.norm(applied,axis=1);out=mapped+applied
    return out,{"enabled":True,"strength":strength,"vertices":int(len(out)),"adjusted_vertices":int(np.count_nonzero(amount>1e-6)),"adjustment_p50_mm":float(np.percentile(amount,50)*1000.0),"adjustment_p95_mm":float(np.percentile(amount,95)*1000.0),"adjustment_max_mm":float(np.max(amount)*1000.0),"topology_backoff_vertices":int(np.count_nonzero(backed)),"topology_backoff_rounds":int(rounds)}




def _component_local_faces(faces: np.ndarray, ids: np.ndarray, vertex_count: int):
    ids=np.asarray(ids,dtype=np.int64);faces=np.asarray(faces,dtype=np.int64)
    if len(ids)<3 or len(faces)==0:return np.empty((0,3),dtype=np.int64)
    lookup=np.full(int(vertex_count),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64)
    mask=np.all(lookup[faces]>=0,axis=1)
    return lookup[faces[mask]]


def _fit_orientation_preserving_affine(source: np.ndarray, target: np.ndarray):
    P=np.asarray(source,dtype=np.float64);Q=np.asarray(target,dtype=np.float64)
    if len(P)<4:return Q.copy(),np.ones(3,dtype=np.float64)
    cp=P.mean(axis=0);cq=Q.mean(axis=0);A=P-cp;B=Q-cq;cov=A.T@A;lam=.008*float(np.trace(cov))/3.0
    M=(B.T@A+lam*np.eye(3))@np.linalg.inv(cov+lam*np.eye(3));u,sv,vt=np.linalg.svd(M);sv=np.clip(sv,.45,3.0);M=u@np.diag(sv)@vt
    if np.linalg.det(M)<0:
        u[:,-1]*=-1;M=u@np.diag(sv)@vt
    return A@M.T+cq,sv


def _extreme_surface_frame_transfer(w: dict[str, Any], mapped: np.ndarray, labels: np.ndarray, classes: dict[int,str], cache: dict[str,Any], enabled: bool, source_body_triangles: np.ndarray):
    """Preserve authored body-relative coverage for extreme refits without preserving the old body silhouette."""
    out=np.asarray(mapped,dtype=np.float64).copy()
    if not enabled:return out,{"enabled":False,"adjusted_vertices":0,"components":[]}
    source=np.asarray(w["V"],dtype=np.float64);faces=np.asarray(w["F"],dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    SV=np.asarray(cache["source_support_V"],dtype=np.float64);SF=np.asarray(cache["source_support_F"],dtype=np.int64);TV=np.asarray(cache["target_support_V"],dtype=np.float64);TF=np.asarray(cache["target_support_F"],dtype=np.int64)
    if len(SF)!=len(TF):return out,{"enabled":False,"adjusted_vertices":0,"reason":"source/target support topology differs","components":[]}
    source_tri=SV[SF];target_tri=TV[TF];reports=[];adjusted=0
    for component in sorted(set(int(x) for x in labels.tolist())):
        if str(classes.get(component,"shell")).casefold()!="shell":continue
        ids=np.flatnonzero(labels==component)
        if len(ids)<20:continue
        closest,source_normal,_,distance,face_index=_b14_nearest_surface(source[ids],source_tri,k=32);clearance=float(np.median(distance))
        if clearance>.025:continue
        face_index=np.asarray(face_index,dtype=np.int64);source_face=source_tri[face_index];target_face=target_tri[face_index]
        bary=trimesh.triangles.points_to_barycentric(source_face,np.asarray(closest,dtype=np.float64));target_contact=np.einsum("ni,nij->nj",bary,target_face)
        target_normal=np.cross(target_face[:,1]-target_face[:,0],target_face[:,2]-target_face[:,0]);target_normal/=np.maximum(np.linalg.norm(target_normal,axis=1,keepdims=True),1e-12)
        # Preserve the garment's signed stand-off from the source surface, not the old body's shape.
        signed_offset=np.einsum("ij,ij->i",source[ids]-closest,source_normal);signed_offset=np.clip(signed_offset,-.005,.030)
        exact=target_contact+signed_offset[:,None]*target_normal
        P=source[ids];macro,sv=_fit_orientation_preserving_affine(P,exact);local_faces=_component_local_faces(faces,ids,len(source));macro_quality=_component_topology_quality(P,macro,local_faces)
        chosen=macro;detail_alpha=0.0;quality=macro_quality
        for alpha in (1.0,.85,.70,.55,.40,.30,.20,.10,0.0):
            trial=macro+alpha*(exact-macro);q=_component_topology_quality(P,trial,local_faces)
            if q["flip_fraction"]<=.0015 and q["orientation_p01"]>.02 and q["area_ratio_p01"]>.08:
                chosen=trial;detail_alpha=float(alpha);quality=q;break
        if macro_quality["flip_fraction"]>.005 or macro_quality["orientation_p01"]<=.01 or macro_quality["area_ratio_p01"]<=.05:
            chosen=out[ids];detail_alpha=-1.0;quality=_component_topology_quality(P,chosen,local_faces)
        else:
            out[ids]=chosen;adjusted+=int(len(ids))
        reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":clearance*1000.0,"detail_alpha":detail_alpha,"affine_singular_values":[float(x) for x in sv],"exact_move_p95_mm":float(np.percentile(np.linalg.norm(exact-P,axis=1),95)*1000.0),"macro_move_p95_mm":float(np.percentile(np.linalg.norm(macro-P,axis=1),95)*1000.0),"quality":quality})
    return out,{"enabled":True,"adjusted_vertices":adjusted,"components":reports,"policy":"extreme close shells preserve source surface-frame coverage on the target, with topology-limited local detail"}


def _extreme_literal_shell_clearance_guard(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, classes: dict[int,str], target_triangles: np.ndarray, enabled: bool, source_weights: np.ndarray | None = None, joint_names: list[str] | None = None, margin: float=.00060):
    """Use coherent cross-section expansion for extreme refits instead of local anatomy-shaped pushes."""
    Vsrc=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64);T=np.asarray(target_triangles,dtype=np.float64)
    if not enabled or len(V)==0 or len(F)==0 or len(T)==0:return V,{"enabled":False,"adjusted_vertices":0,"components":[]}
    reports=[];changed=np.zeros(len(V),dtype=bool);original=V.copy();scales=(1.0,1.02,1.04,1.06,1.08,1.10,1.12,1.15,1.18,1.22,1.26,1.30)
    SW=None if source_weights is None else np.asarray(source_weights,dtype=np.float64);left_cols,right_cols=_leg_joint_columns(list(joint_names or []));left_mass=SW[:,left_cols].sum(axis=1) if SW is not None and left_cols else None;right_mass=SW[:,right_cols].sum(axis=1) if SW is not None and right_cols else None
    for component in sorted(set(int(x) for x in labels.tolist())):
        if str(classes.get(component,"shell")).casefold()!="shell":continue
        ids=np.flatnonzero(labels==component)
        if len(ids)<16:continue
        local_faces=_component_local_faces(F,ids,len(V));P=Vsrc[ids];Q=V[ids];base_quality=_component_topology_quality(P,Q,local_faces);_,_,signed,_,_=_b14_nearest_surface(Q,T,k=28);before_min=float(np.min(signed));before_p01=float(np.percentile(signed,1));before_neg=float(np.mean(signed<0.0))
        if before_min>=margin-.00010:
            reports.append({"component":component,"vertices":int(len(ids)),"scale":1.0,"before_min_mm":before_min*1000.0,"after_min_mm":before_min*1000.0,"before_p01_mm":before_p01*1000.0,"after_p01_mm":before_p01*1000.0,"negative_fraction_before":before_neg,"negative_fraction_after":before_neg,"reason":"already clear"});continue
        leg_dominant=False
        if left_mass is not None and right_mass is not None:
            lm=float(np.mean(left_mass[ids]));rm=float(np.mean(right_mass[ids]));leg_dominant=max(lm,rm)>=.55 and abs(lm-rm)>=.35
        centre=Q.mean(axis=0);best=Q;best_scale=1.0;best_min=before_min;best_p01=before_p01;best_neg=before_neg;best_score=before_p01+.25*before_min-.020*before_neg
        for scale in scales[1:]:
            trial=Q.copy();trial[:,0]=centre[0]+(Q[:,0]-centre[0])*scale;trial[:,2]=centre[2]+(Q[:,2]-centre[2])*scale
            quality=_component_topology_quality(P,trial,local_faces);flip_limit=max(.0040,base_quality["flip_fraction"]+.0015);orientation_floor=min(.03,base_quality["orientation_p01"]-.12)
            if quality["flip_fraction"]>flip_limit or quality["orientation_p01"]<orientation_floor or quality["area_ratio_p01"]<=.04:continue
            _,_,trial_signed,_,_=_b14_nearest_surface(trial,T,k=28);trial_min=float(np.min(trial_signed));trial_p01=float(np.percentile(trial_signed,1));trial_neg=float(np.mean(trial_signed<0.0));score=trial_p01+.25*trial_min-.020*trial_neg
            min_backslide=.0050 if leg_dominant else .00050
            if trial_min<before_min-min_backslide:continue
            if score>best_score+1e-6:
                best=trial;best_scale=float(scale);best_min=trial_min;best_p01=trial_p01;best_neg=trial_neg;best_score=score
            if trial_min>=margin and trial_p01>=margin:break
        if best_scale>1.0:
            V[ids]=best;changed[ids]=np.linalg.norm(best-Q,axis=1)>1e-6
        reports.append({"component":component,"vertices":int(len(ids)),"scale":best_scale,"before_min_mm":before_min*1000.0,"after_min_mm":best_min*1000.0,"before_p01_mm":before_p01*1000.0,"after_p01_mm":best_p01*1000.0,"negative_fraction_before":before_neg,"negative_fraction_after":best_neg,"leg_dominant":bool(leg_dominant),"moved_p95_mm":float(np.percentile(np.linalg.norm(best-Q,axis=1),95)*1000.0),"policy":"orientation-preserving XZ cross-section expansion"})
    return V,{"enabled":True,"adjusted_vertices":int(np.count_nonzero(changed)),"max_vertex_move_mm":float(np.max(np.linalg.norm(V-original,axis=1))*1000.0),"components":reports,"policy":"extreme wrap-around shells expand coherently around their own cross-section; local anatomy is not embossed into cloth"}


def _component_topology_quality(source: np.ndarray, target: np.ndarray, faces: np.ndarray):
    P=np.asarray(source,dtype=np.float64);Q=np.asarray(target,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if len(F)==0:return {"flip_fraction":0.0,"orientation_p01":1.0,"area_ratio_p01":1.0}
    a=np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]);b=np.cross(Q[F[:,1]]-Q[F[:,0]],Q[F[:,2]]-Q[F[:,0]])
    an=np.linalg.norm(a,axis=1);bn=np.linalg.norm(b,axis=1);dot=np.einsum("ij,ij->i",a,b)/np.maximum(an*bn,1e-24);ratio=bn/np.maximum(an,1e-16)
    return {"flip_fraction":float(np.mean(dot<0.0)),"orientation_p01":float(np.percentile(dot,1)),"area_ratio_p01":float(np.percentile(ratio,1))}


def _support_frame_clearance_guard(w: dict[str, Any], mapped: np.ndarray, blend: np.ndarray, blend_ids: np.ndarray, cache: dict[str, Any], labels: np.ndarray, classes: dict[int, str], source_body_triangles: np.ndarray, minimum_clearance: float = .00035):
    source=np.asarray(w["V"],dtype=np.float64);faces=np.asarray(w["F"],dtype=np.int64);mapped=np.asarray(mapped,dtype=np.float64);blend=np.asarray(blend,dtype=np.float64);blend_ids=np.asarray(blend_ids,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    Y=np.asarray(cache["Y"],dtype=np.float64);NT=np.asarray(cache["NT"],dtype=np.float64);target_anchor=np.sum(Y[blend_ids]*blend[:,:,None],axis=1);target_normal=np.sum(NT[blend_ids]*blend[:,:,None],axis=1);target_normal/=np.maximum(np.linalg.norm(target_normal,axis=1,keepdims=True),1e-12)
    out=mapped.copy();reports=[];adjusted=0
    for component in sorted(set(int(x) for x in labels.tolist())):
        if str(classes.get(component,"shell")).casefold()!="shell":continue
        ids=np.flatnonzero(labels==component)
        if len(ids)<20:continue
        _,_,_,source_distance,_=_b14_nearest_surface(source[ids],source_body_triangles,k=24);source_clearance=float(np.median(source_distance))
        if source_clearance>.025:continue
        radial=np.einsum("ij,ij->i",out[ids]-target_anchor[ids],target_normal[ids]);deficit=np.maximum(float(minimum_clearance)-radial,0.0)
        if not np.any(deficit>1e-8):continue
        displacement=np.minimum(deficit,.015)[:,None]*target_normal[ids]
        lookup=np.full(len(source),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64);mask=np.all(lookup[faces]>=0,axis=1);local_faces=lookup[faces[mask]]
        backoff=np.ones(len(ids),dtype=np.float64);backed=np.zeros(len(ids),dtype=bool)
        if len(local_faces):
            P=source[ids];B=out[ids];sf=np.cross(P[local_faces[:,1]]-P[local_faces[:,0]],P[local_faces[:,2]]-P[local_faces[:,0]]);bf=np.cross(B[local_faces[:,1]]-B[local_faces[:,0]],B[local_faces[:,2]]-B[local_faces[:,0]]);sl=np.linalg.norm(sf,axis=1);blen=np.linalg.norm(bf,axis=1);alignment=np.einsum("ij,ij->i",sf,bf)/np.maximum(sl*blen,1e-24);baseline_good=(alignment>.03)&(blen>sl*.10)
            for _ in range(8):
                trial=B+displacement*backoff[:,None];tf=np.cross(trial[local_faces[:,1]]-trial[local_faces[:,0]],trial[local_faces[:,2]]-trial[local_faces[:,0]]);tl=np.linalg.norm(tf,axis=1);ta=np.einsum("ij,ij->i",sf,tf)/np.maximum(sl*tl,1e-24);bad=baseline_good&((ta<=.03)|(tl<=sl*.10))
                if not np.any(bad):break
                bad_ids=np.unique(local_faces[bad].reshape(-1));backoff[bad_ids]*=.5;backed[bad_ids]=True
        accepted=out[ids]+displacement*backoff[:,None];out[ids]=accepted;final_radial=np.einsum("ij,ij->i",accepted-target_anchor[ids],target_normal[ids]);moved=np.linalg.norm(displacement*backoff[:,None],axis=1);count=int(np.count_nonzero(moved>1e-8));adjusted+=count
        reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":source_clearance*1000.0,"adjusted_vertices":count,"backed_off_vertices":int(np.count_nonzero(backed)),"before_negative_fraction":float(np.mean(radial<0.0)),"after_negative_fraction":float(np.mean(final_radial<0.0)),"move_p95_mm":float(np.percentile(moved,95)*1000.0),"minimum_clearance_mm":float(minimum_clearance*1000.0)})
    return out,{"adjusted_vertices":adjusted,"components":reports,"policy":"close garments keep a small positive clearance in the mapped target support frame"}


def _literal_body_envelope_guard(w: dict[str, Any], mapped: np.ndarray, blend: np.ndarray, blend_ids: np.ndarray, cache: dict[str, Any], labels: np.ndarray, classes: dict[int, str], source_body_triangles: np.ndarray, enabled: bool, minimum_clearance: float = .00035):
    if not enabled:
        return np.asarray(mapped,dtype=np.float64),{"enabled":False,"reason":"normal body-range conversion"}
    source=np.asarray(w["V"],dtype=np.float64);faces=np.asarray(w["F"],dtype=np.int64);mapped=np.asarray(mapped,dtype=np.float64);blend=np.asarray(blend,dtype=np.float64);blend_ids=np.asarray(blend_ids,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    Y=np.asarray(cache["Y"],dtype=np.float64);NT=np.asarray(cache["NT"],dtype=np.float64);BW=np.asarray(cache["BW"],dtype=np.float64);literal_v=np.asarray(cache.get("target_surface_V"),dtype=np.float64);literal_w=np.asarray(cache.get("target_surface_W"),dtype=np.float64)
    if literal_v.ndim!=2 or literal_v.shape[1]!=3 or literal_w.ndim!=2 or len(literal_v)!=len(literal_w):
        return mapped.copy(),{"enabled":False,"reason":"literal target surface unavailable"}
    anchor=np.sum(Y[blend_ids]*blend[:,:,None],axis=1);normal=np.sum(NT[blend_ids]*blend[:,:,None],axis=1);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12);expected=np.sum(BW[blend_ids]*blend[:,:,None],axis=1)
    tree=cKDTree(literal_v);k=min(192,len(literal_v));out=mapped.copy();reports=[];adjusted=0
    for component in sorted(set(int(x) for x in labels.tolist())):
        if str(classes.get(component,"shell")).casefold()!="shell":continue
        ids=np.flatnonzero(labels==component)
        if len(ids)<20:continue
        _,_,_,source_distance,_=_b14_nearest_surface(source[ids],source_body_triangles,k=24);source_clearance=float(np.median(source_distance))
        distance,candidate_ids=tree.query(anchor[ids],k=k)
        if distance.ndim==1:distance=distance[:,None];candidate_ids=candidate_ids[:,None]
        offset=literal_v[candidate_ids]-anchor[ids,None,:];radial=np.einsum("nqj,nj->nq",offset,normal[ids]);tangent=offset-radial[:,:,None]*normal[ids,None,:];tangent_distance=np.linalg.norm(tangent,axis=2)
        shared=np.sum(literal_w[candidate_ids],axis=2)>1e-8;alignment=np.einsum("nqk,nk->nq",literal_w[candidate_ids],expected[ids])
        compatible=shared&(alignment>=.30);target_only=(~shared)&(distance<=.060)
        valid=(tangent_distance<=.015)&(compatible|target_only)&np.isfinite(radial)&(radial>=0.0)
        envelope=np.zeros(len(ids),dtype=np.float64);have=np.zeros(len(ids),dtype=bool)
        for row in range(len(ids)):
            values=radial[row,valid[row]]
            if len(values):envelope[row]=max(0.0,float(np.percentile(values,90)));have[row]=True
        if not np.any(have):
            reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":source_clearance*1000.0,"action":"no-compatible-literal-envelope"});continue
        current=np.einsum("ij,ij->i",out[ids]-anchor[ids],normal[ids]);deficit=np.maximum(envelope+float(minimum_clearance)-current,0.0);deficit[~have]=0.0
        env90=float(np.percentile(envelope[have],90));deficit95=float(np.percentile(deficit[have],95));deficit99=float(np.percentile(deficit[have],99));deficit995=float(np.percentile(deficit[have],99.5));stand_off=source_clearance>.025;detail_dominated=env90>.012
        component_normal=np.median(normal[ids],axis=0);component_normal_length=float(np.linalg.norm(component_normal));component_direction=component_normal/max(component_normal_length,1e-12)
        if stand_off:
            if deficit99<=.001 or component_normal_length<=.25:
                reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":source_clearance*1000.0,"literal_envelope_p90_mm":env90*1000.0,"deficit_p95_mm":deficit95*1000.0,"deficit_p99_mm":deficit99*1000.0,"action":"stand-off-already-clear"});continue
            requested=float(np.clip(deficit99,0.0,.025));displacement=np.broadcast_to(requested*component_direction,(len(ids),3)).copy();direction_mode="stand-off-component"
        elif detail_dominated:
            requested=float(np.clip(deficit95,0.0,.012))
            if requested<=1e-6 or component_normal_length<=.25:
                reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":source_clearance*1000.0,"literal_envelope_p90_mm":env90*1000.0,"deficit_p95_mm":deficit95*1000.0,"action":"detail-clearance-unavailable"});continue
            displacement=np.broadcast_to(requested*component_direction,(len(ids),3)).copy();direction_mode="component"
        else:
            requested=float(np.clip(deficit995,0.0,.012))
            if requested<=1e-6:
                reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":source_clearance*1000.0,"literal_envelope_p90_mm":env90*1000.0,"deficit_p95_mm":deficit95*1000.0,"action":"already-clear"});continue
            displacement=requested*normal[ids];direction_mode="support"
        lookup=np.full(len(source),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64);mask=np.all(lookup[faces]>=0,axis=1);local_faces=lookup[faces[mask]];base=out[ids]
        baseline_quality=_component_topology_quality(source[ids],base,local_faces) if len(local_faces) else {"flip_fraction":0.0,"orientation_p01":1.0,"area_ratio_p01":1.0};flip_limit=max(.001,float(baseline_quality["flip_fraction"])+1e-6);orientation_floor=min(.03,float(baseline_quality["orientation_p01"])-.02);area_floor=min(.10,float(baseline_quality["area_ratio_p01"])*.8)
        accepted=base;accepted_alpha=0.0;quality=baseline_quality
        for alpha in (1.0,.8,.6,.4,.2,.1,.05,0.0):
            trial=base+float(alpha)*displacement;q=_component_topology_quality(source[ids],trial,local_faces) if len(local_faces) else {"flip_fraction":0.0,"orientation_p01":1.0,"area_ratio_p01":1.0}
            if q["flip_fraction"]<=flip_limit and q["orientation_p01"]>=orientation_floor and q["area_ratio_p01"]>=area_floor:
                accepted=trial;accepted_alpha=float(alpha);quality=q;break
        out[ids]=accepted;moved=float(requested*accepted_alpha)
        if moved>1e-8:adjusted+=int(len(ids))
        reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":source_clearance*1000.0,"literal_envelope_p90_mm":env90*1000.0,"deficit_p95_mm":deficit95*1000.0,"deficit_p99_mm":deficit99*1000.0,"deficit_p995_mm":deficit995*1000.0,"detail_dominated":detail_dominated,"stand_off":stand_off,"direction_mode":direction_mode,"requested_shift_mm":requested*1000.0,"accepted_alpha":accepted_alpha,"accepted_shift_mm":moved*1000.0,"action":"broad-clearance" if moved>1e-8 else "topology-blocked","topology":quality})
    return out,{"enabled":True,"adjusted_vertices":adjusted,"components":reports,"policy":"extreme refits keep literal target anatomy as collision authority through smooth component clearance, never literal shape transfer"}


def _stabilise_extreme_refit_components(w: dict[str, Any], mapped: np.ndarray, labels: np.ndarray, classes: dict[int, str], source_body_triangles: np.ndarray, enabled: bool = True):
    """Keep extreme close-body refits topology-safe without preserving the old body silhouette."""
    if not enabled:return np.asarray(mapped,dtype=np.float64),{"enabled":False,"adjusted_vertices":0,"components":[],"policy":"inactive outside extreme refits"}
    source=np.asarray(w["V"],dtype=np.float64);mapped=np.asarray(mapped,dtype=np.float64);faces=np.asarray(w["F"],dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    out=mapped.copy();reports=[];adjusted=0
    for component in sorted(set(int(x) for x in labels.tolist())):
        if str(classes.get(component,"shell")).casefold()!="shell":continue
        ids=np.flatnonzero(labels==component)
        if len(ids)<20:continue
        lookup=np.full(len(source),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64);mask=np.all(lookup[faces]>=0,axis=1);local_faces=lookup[faces[mask]]
        if not len(local_faces):continue
        _,_,_,distance,_=_b14_nearest_surface(source[ids],source_body_triangles,k=24);clearance=float(np.median(distance))
        if clearance>.025:continue
        before=_component_topology_quality(source[ids],mapped[ids],local_faces)
        if before["flip_fraction"]<.005 and before["orientation_p01"]>.05:continue

        P=source[ids];Q=mapped[ids];cp=np.mean(P,axis=0);cq=np.mean(Q,axis=0);A=P-cp;B=Q-cq;cov=A.T@A;lam=.015*float(np.trace(cov))/3.0
        affine=(B.T@A+lam*np.eye(3))@np.linalg.inv(cov+lam*np.eye(3));u,sv,vt=np.linalg.svd(affine);sv=np.clip(sv,.45,2.5);affine=u@np.diag(sv)@vt
        if np.linalg.det(affine)<0:
            u[:,-1]*=-1;affine=u@np.diag(sv)@vt
        macro=A@affine.T+cq
        chosen=macro;chosen_alpha=0.0;chosen_quality=_component_topology_quality(P,macro,local_faces)
        for alpha in (1.0,.9,.8,.7,.6,.5,.4,.3,.2,.1,0.0):
            trial=macro+alpha*(Q-macro);quality=_component_topology_quality(P,trial,local_faces)
            if quality["flip_fraction"]<=.001 and quality["orientation_p01"]>.03 and quality["area_ratio_p01"]>.10:
                chosen=trial;chosen_alpha=float(alpha);chosen_quality=quality;break
        out[ids]=chosen;adjusted+=int(len(ids))
        reports.append({"component":component,"vertices":int(len(ids)),"source_clearance_median_mm":clearance*1000.0,"detail_alpha":chosen_alpha,"before":before,"after":chosen_quality,"macro_move_p95_mm":float(np.percentile(np.linalg.norm(macro-P,axis=1),95)*1000.0),"final_move_p95_mm":float(np.percentile(np.linalg.norm(chosen-P,axis=1),95)*1000.0)})
    return out,{"adjusted_vertices":adjusted,"components":reports,"policy":"extreme close-fit components keep target macro shape; unsafe local residual is topology-limited"}

def _preserve_authored_rig_geometry(w: dict[str, Any], mapped: np.ndarray, cache: dict[str, Any], labels: np.ndarray):
    source=np.asarray(w["V"],dtype=np.float64);mapped=np.asarray(mapped,dtype=np.float64);weights=np.asarray(w["W"],dtype=np.float64);joint_names=list(w["joint_names"]);cache_names=list(cache["names"])
    target_surface_weights=np.asarray(cache.get("target_surface_W"),dtype=np.float64);supported_cache=np.sum(target_surface_weights,axis=0)>1e-8
    supported=np.zeros(len(joint_names),dtype=bool);source_index={name:i for i,name in enumerate(joint_names)}
    for ci,name in enumerate(cache_names):
        index=source_index.get(name)
        if index is not None:supported[index]=bool(supported_cache[ci])
    preserved_mass=weights[:,~supported].sum(axis=1) if np.any(~supported) else np.zeros(len(source),dtype=np.float64)
    out=mapped.copy();reports=[];adjusted=0
    for component in np.unique(labels):
        ids=np.flatnonzero(labels==component)
        if len(ids)<3:continue
        local=preserved_mass[ids];mean=float(np.mean(local));p50=float(np.median(local));p90=float(np.percentile(local,90))
        if p90<.20 or mean<.08:continue
        source_tri=cache.get("_ravafit_source_support_triangles")
        if source_tri is None:
            source_tri=_triangles_from_surface(cache["source_support_V"],cache["source_support_F"]);cache["_ravafit_source_support_triangles"]=source_tri
        _,_,_,component_distance,_=_b14_nearest_surface(source[ids],source_tri,k=24);clearance=float(np.median(component_distance))
        fit_scale=float(np.clip((clearance-.012)/.023,0.0,1.0))
        if fit_scale<=1e-6:continue
        body_mass=np.clip(1.0-local,0.0,1.0);anchor=.015+body_mass**3
        structural=_weighted_rigid_projection(source[ids],mapped[ids],anchor)
        authority=np.clip((local-.10)/.70,0.0,1.0)*.94*fit_scale
        candidate=(1.0-authority[:,None])*mapped[ids]+authority[:,None]*structural
        topology_alpha=1.0;topology_before=None;topology_after=None
        lookup=np.full(len(source),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64);face_mask=np.all(lookup[np.asarray(w["F"],dtype=np.int64)]>=0,axis=1);local_faces=lookup[np.asarray(w["F"],dtype=np.int64)[face_mask]]
        if len(local_faces):
            topology_before=_component_topology_quality(source[ids],candidate,local_faces)
            if topology_before["flip_fraction"]>.001 or topology_before["orientation_p01"]<=.03 or topology_before["area_ratio_p01"]<=.10:
                for alpha in (1.0,.9,.8,.7,.6,.5,.4,.3,.2,.1,0.0):
                    trial=structural+alpha*(candidate-structural);quality=_component_topology_quality(source[ids],trial,local_faces)
                    if quality["flip_fraction"]<=.001 and quality["orientation_p01"]>.03 and quality["area_ratio_p01"]>.10:
                        candidate=trial;topology_alpha=float(alpha);topology_after=quality;break
            if topology_after is None:topology_after=_component_topology_quality(source[ids],candidate,local_faces)
        before=np.linalg.norm(mapped[ids]-source[ids],axis=1);after=np.linalg.norm(candidate-source[ids],axis=1)
        out[ids]=candidate;adjusted+=int(np.count_nonzero(authority>.02))
        reports.append({"component":int(component),"vertices":int(len(ids)),"preserved_weight_mean":mean,"preserved_weight_p50":p50,"preserved_weight_p90":p90,"source_clearance_median_mm":clearance*1000.0,"geometry_authority":fit_scale,"adjusted_vertices":int(np.count_nonzero(authority>.02)),"mapped_move_p95_mm":float(np.percentile(before,95)*1000.0),"guarded_move_p95_mm":float(np.percentile(after,95)*1000.0),"topology_detail_alpha":topology_alpha,"topology_before":topology_before,"topology_after":topology_after})
    return out,{"adjusted_vertices":adjusted,"components":reports,"policy":"garment-only bones keep component deformation structural; target body still controls body-supported anchors"}


def _dense_vanilla_expansion_clearance_guard(positions: dict[str, np.ndarray], retarget_contexts: dict[str, dict[str, Any]], cache: dict[str, Any], expansion_threshold: float = .00180, influence: float = .018, margin: float = .00055, per_step_cap: float = .00150, total_cap: float = .00800):
    """Resolve only meaningful outward body expansion for dense vanilla proxies."""
    if not positions or not bool(cache.get('dense_vanilla_source_proxy',False)):
        return {name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()},set(),{"enabled":False,"reason":"not dense vanilla proxy"}
    X=np.asarray(cache.get('X'),dtype=np.float64);Y=np.asarray(cache.get('Y'),dtype=np.float64);NT=np.asarray(cache.get('NT'),dtype=np.float64)
    if X.shape!=Y.shape or X.shape!=NT.shape or X.ndim!=2 or X.shape[1]!=3:
        return {name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()},set(),{"enabled":False,"reason":"dense body field shape mismatch"}
    normal_length=np.linalg.norm(NT,axis=1);valid_normals=normal_length>.5
    expansion=np.einsum('ij,ij->i',Y-X,NT)
    eligible_ids=np.flatnonzero(valid_normals&(expansion>float(expansion_threshold)))
    if len(eligible_ids)==0:
        return {name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()},set(),{"enabled":True,"eligible_body_vertices":0,"iterations":0,"moved_meshes":0,"unresolved_body_vertices":0}

    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()}
    baseline={name:value.copy() for name,value in out.items()}
    mesh_names=[name for name in sorted(out) if name in retarget_contexts]
    face_map=[];neighbours={};baseline_face_area={}
    for name in mesh_names:
        data=retarget_contexts[name]['data'];F=np.asarray(data['F'],dtype=np.int64);V=out[name]
        n=[set() for _ in range(len(V))]
        for a,b,c in F:
            a=int(a);b=int(b);c=int(c);n[a].update((b,c));n[b].update((a,c));n[c].update((a,b))
        neighbours[name]=n
        area=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]) if len(F) else np.zeros((0,3),dtype=np.float64)
        baseline_face_area[name]=area
    if not mesh_names:
        return out,set(),{"enabled":False,"reason":"no retarget contexts"}

    changed=set();iterations=0;initial_bad=None;final_bad=None;maximum_move=0.0
    body_points=Y[eligible_ids];body_normals=NT[eligible_ids]
    for iteration in range(12):
        triangles=[];face_map=[]
        for name in mesh_names:
            F=np.asarray(retarget_contexts[name]['data']['F'],dtype=np.int64)
            if len(F):
                triangles.append(out[name][F]);face_map.extend((name,fi) for fi in range(len(F)))
        if not triangles:break
        garment_tri=np.vstack(triangles)
        closest,garment_normals,_,distance,face_index=_b14_nearest_surface(body_points,garment_tri,k=32)
        clearance=np.einsum('ij,ij->i',closest-body_points,body_normals)
        alignment=np.einsum('ij,ij->i',np.asarray(garment_normals,dtype=np.float64),body_normals)
        bad=(distance<float(influence))&(clearance<float(margin))&(alignment>.15)
        bad_ids=np.flatnonzero(bad)
        if initial_bad is None:initial_bad=int(len(bad_ids))
        final_bad=int(len(bad_ids))
        if len(bad_ids)==0:break
        iterations=iteration+1

        accum={name:np.zeros_like(out[name]) for name in mesh_names};weight_sum={name:np.zeros(len(out[name]),dtype=np.float64) for name in mesh_names}
        for local_index in bad_ids:
            global_face=int(face_index[local_index])
            if global_face<0 or global_face>=len(face_map):continue
            name,fi=face_map[global_face];F=np.asarray(retarget_contexts[name]['data']['F'],dtype=np.int64);tri_ids=F[int(fi)]
            tri=out[name][tri_ids][None,:,:];point=closest[local_index][None,:]
            try:bary=trimesh.triangles.points_to_barycentric(tri,point)[0]
            except Exception:continue
            need=float((margin+.00010)-clearance[local_index]);normal=body_normals[local_index]
            if need<=0.0:continue
            for corner,vertex in enumerate(tri_ids):
                weight=.25+.75*float(bary[corner]);vertex=int(vertex);accum[name][vertex]+=normal*need*weight;weight_sum[name][vertex]+=weight

        for name in mesh_names:
            current=out[name];F=np.asarray(retarget_contexts[name]['data']['F'],dtype=np.int64);weights=weight_sum[name];direct=weights>0.0
            if not np.any(direct):continue
            displacement=np.zeros_like(current);displacement[direct]=accum[name][direct]/weights[direct,None]
            for _ in range(2):
                smoothed=displacement.copy()
                for vertex,nb in enumerate(neighbours[name]):
                    if not nb:continue
                    average=np.mean(displacement[list(nb)],axis=0)
                    smoothed[vertex]=(.84*displacement[vertex]+.16*average) if direct[vertex] else .08*average
                displacement=smoothed
            magnitude=np.linalg.norm(displacement,axis=1);displacement*=np.minimum(1.0,float(per_step_cap)/np.maximum(magnitude,1e-12))[:,None]
            trial=current+displacement
            total=trial-baseline[name];total_mag=np.linalg.norm(total,axis=1);over=total_mag>float(total_cap)
            if np.any(over):trial[over]=baseline[name][over]+total[over]*(float(total_cap)/total_mag[over])[:,None]

            if len(F):
                base_area=baseline_face_area[name];base_norm=np.linalg.norm(base_area,axis=1);area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);area_norm=np.linalg.norm(area,axis=1)
                good=base_norm>1e-12;dot=np.ones(len(F),dtype=np.float64);ratio=np.ones(len(F),dtype=np.float64)
                dot[good]=np.einsum('ij,ij->i',base_area[good],area[good])/np.maximum(base_norm[good]*area_norm[good],1e-24);ratio[good]=area_norm[good]/np.maximum(base_norm[good],1e-12)
                unsafe=good&((dot<=.05)|(ratio<=.15))
                if np.any(unsafe):
                    unsafe_vertices=np.unique(F[unsafe].reshape(-1));trial[unsafe_vertices]=current[unsafe_vertices]
            moved=np.linalg.norm(trial-current,axis=1)>1e-10
            if np.any(moved):changed.add(name);out[name]=trial;maximum_move=max(maximum_move,float(np.max(np.linalg.norm(trial-baseline[name],axis=1))))

    # One last status query, using the same eligibility mask that drove the corrections.
    triangles=[]
    for name in mesh_names:
        F=np.asarray(retarget_contexts[name]['data']['F'],dtype=np.int64)
        if len(F):triangles.append(out[name][F])
    unresolved=0;minimum_after=None
    if triangles:
        garment_tri=np.vstack(triangles);closest,normals,_,distance,_=_b14_nearest_surface(body_points,garment_tri,k=32);clearance=np.einsum('ij,ij->i',closest-body_points,body_normals);alignment=np.einsum('ij,ij->i',np.asarray(normals,dtype=np.float64),body_normals);near=(distance<float(influence))&(alignment>.15);unresolved=int(np.count_nonzero(near&(clearance<float(margin))));minimum_after=float(np.min(clearance[near])*1000.0) if np.any(near) else None
    return out,changed,{"enabled":True,"policy":"clear only outward target expansion relative to dense vanilla proxy; preserve distant authored silhouette","eligible_body_vertices":int(len(eligible_ids)),"initial_violating_body_vertices":int(initial_bad or 0),"unresolved_body_vertices":int(unresolved),"iterations":int(iterations),"moved_meshes":int(len(changed)),"max_vertex_move_mm":float(maximum_move*1000.0),"minimum_near_clearance_after_mm":minimum_after,"expansion_threshold_mm":float(expansion_threshold*1000.0),"influence_mm":float(influence*1000.0),"total_vertex_cap_mm":float(total_cap*1000.0)}





def _smooth_translation_body_field(vertices: np.ndarray, faces: np.ndarray, garment_weights: np.ndarray, X: np.ndarray, Y: np.ndarray, body_weights: np.ndarray, tree: cKDTree, tau: float=.004, iterations: int=10):
    """Cheap topology-friendly body field for genuinely skin-close flexible layers."""
    P=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);Wg=np.asarray(garment_weights,dtype=np.float64)
    if len(P)==0:return P.copy(),{"enabled":False,"reason":"empty garment"}
    k=min(32,len(X));dist,idx=tree.query(P,k=k);dist=dist if dist.ndim>1 else dist[:,None];idx=idx if idx.ndim>1 else idx[:,None]
    align=np.einsum('nk,nqk->nq',Wg,body_weights[idx]);score=dist+.015*(1.0-align)
    sx=np.sign(P[:,0])[:,None];bx=np.sign(X[idx][:,:,0]);score+=np.where((np.abs(P[:,0,None])>.020)&(sx!=bx),.080,0.0)
    terms=min(12,score.shape[1]);order=np.argpartition(score,terms-1,axis=1)[:,:terms];sel=np.take_along_axis(idx,order,axis=1);ss=np.take_along_axis(score,order,axis=1)
    rel=ss-ss.min(axis=1,keepdims=True);weights=np.exp(-rel/max(float(tau),1e-6));weights/=np.maximum(weights.sum(axis=1,keepdims=True),1e-12)
    displacement=np.sum((Y-X)[sel]*weights[:,:,None],axis=1)
    neighbours=_mesh_vertex_neighbours(len(P),F)
    before=displacement.copy()
    for _ in range(max(0,int(iterations))):
        nxt=displacement.copy()
        for vi,nb in enumerate(neighbours):
            if nb:nxt[vi]=.75*displacement[vi]+.25*np.mean(displacement[nb],axis=0)
        displacement=nxt
    out=P+displacement
    return out,{"enabled":True,"mode":"smoothed_translation_body_field","iterations":int(iterations),"tau":float(tau),"raw_move_p95_mm":float(np.percentile(np.linalg.norm(before,axis=1),95)*1000.0),"smoothed_move_p95_mm":float(np.percentile(np.linalg.norm(displacement,axis=1),95)*1000.0),"smoothed_move_max_mm":float(np.max(np.linalg.norm(displacement,axis=1))*1000.0) if len(displacement) else 0.0}


def _raw_component_boundary_fraction(vertices: np.ndarray, faces: np.ndarray, component_ids: np.ndarray) -> float:
    ids=np.asarray(component_ids,dtype=np.int64);F=np.asarray(faces,dtype=np.int64)
    if len(ids)<3 or len(F)==0:return 0.0
    lookup=np.full(len(vertices),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64)
    mask=np.all(lookup[F]>=0,axis=1)
    LF=lookup[F[mask]]
    if len(LF)==0:return 0.0
    edges=np.sort(np.vstack((LF[:,[0,1]],LF[:,[1,2]],LF[:,[2,0]])),axis=1)
    _,cnt=np.unique(edges,axis=0,return_counts=True)
    boundary_edges=np.unique(edges,axis=0)[cnt==1]
    if len(boundary_edges)==0:return 0.0
    return float(len(np.unique(boundary_edges))/max(len(ids),1))


def _similarity_fit_points(source: np.ndarray, target: np.ndarray, scale_min: float=.80, scale_max: float=1.20):
    src=np.asarray(source,dtype=np.float64);dst=np.asarray(target,dtype=np.float64)
    if len(src)<3:return dst.copy(),1.0
    cs=src.mean(axis=0);ct=dst.mean(axis=0);A=src-cs;B=dst-ct
    u,_,vt=np.linalg.svd(A.T@B);R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    Ar=A@R.T;scale=float(np.sum(Ar*B)/max(np.sum(Ar*Ar),1e-12));scale=float(np.clip(scale,scale_min,scale_max))
    return scale*A@R.T+ct,scale


def _narrow_authored_strip_shape_guard(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]], cache: dict[str,Any], margin: float=.00065):
    """Restore distorted thin disconnected garment strips without body-specific semantics."""
    out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()};changed=set();reports=[]
    bary=np.asarray([[1/3,1/3,1/3],[.60,.20,.20],[.20,.60,.20],[.20,.20,.60],[.50,.50,0],[0,.50,.50],[.50,0,.50]],dtype=np.float64)
    for name,context in contexts.items():
        if name not in out:continue
        data=context.get("data") or {};S=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64);V=out[name]
        if len(S)<64 or len(F)==0 or S.shape!=V.shape:continue
        raw_components=_masked_vertex_components(F,np.ones(len(S),dtype=bool))
        for component_index,ids in enumerate(raw_components):
            if len(ids)<64 or len(ids)>1800:continue
            P=S[ids];extent=np.ptp(P,axis=0);max_extent=float(np.max(extent))
            if max_extent>.220:continue
            centred=P-P.mean(axis=0);sv=np.linalg.svd(centred,compute_uv=False)
            if len(sv)<3 or sv[0]<1e-10:continue
            middle_ratio=float(sv[1]/sv[0]);thin_ratio=float(sv[2]/sv[0]);boundary_fraction=_raw_component_boundary_fraction(S,F,ids)
            if boundary_fraction<.45 or middle_ratio>.42 or thin_ratio>.08:continue
            raw_mask=np.zeros(len(S),dtype=bool);raw_mask[ids]=True;face_mask=np.all(raw_mask[F],axis=1);component_faces=F[face_mask]
            if len(component_faces)<8:continue
            edges=np.unique(np.sort(np.vstack((component_faces[:,[0,1]],component_faces[:,[1,2]],component_faces[:,[2,0]])),axis=1),axis=0)
            displacement=V-S;roughness=float(np.percentile(np.linalg.norm(displacement[edges[:,0]]-displacement[edges[:,1]],axis=1),95)) if len(edges) else 0.0
            source_edge=np.linalg.norm(S[edges[:,1]]-S[edges[:,0]],axis=1);current_edge=np.linalg.norm(V[edges[:,1]]-V[edges[:,0]],axis=1);edge_ratio=current_edge/np.maximum(source_edge,1e-9);edge_p99=float(np.percentile(edge_ratio,99)) if len(edge_ratio) else 1.0
            if roughness<=.00150 and edge_p99<=1.60:continue
            # Pick the source/target body pair this raw strip was actually authored nearest to.
            best=None
            probe=P[::max(1,len(P)//256)]
            for pair in cache.get("slot_pairs",[]):
                SV=np.asarray(pair.get("source_literal_V",[]),dtype=np.float64);SF=np.asarray(pair.get("source_literal_F",[]),dtype=np.int64);TV=np.asarray(pair.get("target_literal_V",[]),dtype=np.float64);TF=np.asarray(pair.get("target_literal_F",[]),dtype=np.int64);TSV=np.asarray(pair.get("target_support_V",[]),dtype=np.float64);TSF=np.asarray(pair.get("target_support_F",[]),dtype=np.int64)
                if len(SV)<16 or len(SF)==0 or len(TV)<16 or len(TF)==0 or len(TSV)<16 or len(TSF)==0:continue
                _,_,_,sd,_=_b14_nearest_surface(probe,SV[SF],k=24);median=float(np.median(sd))
                if best is None or median<best[0]:best=(median,str(pair.get("slot") or "Body"),TV[TF],TSV[TSF])
            if best is None or best[0]>.020:continue
            _,slot,target_tri,target_support_tri=best
            coherent,scale=_similarity_fit_points(P,V[ids],scale_min=.75,scale_max=1.30)
            lookup=np.full(len(S),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids),dtype=np.int64);LF=lookup[component_faces]
            # Translate the complete authored strip as one object until exact face samples clear.
            clearance_rounds=0
            for clearance_round in range(12):
                samples=np.einsum("bk,fkj->fbj",bary,coherent[LF]).reshape(-1,3)
                _,normals,signed,_,_,_=_nearest_literal_surface_consistent_with_support(samples,target_tri,target_support_tri,exact_base=True)
                deficit=np.maximum(0.0,float(margin)-signed);bad=deficit>0.0
                if not np.any(bad):break
                nn=normals[bad].copy();ww=deficit[bad];anchor=nn[int(np.argmax(ww))].copy();flip=np.einsum("ij,j->i",nn,anchor)<0.0;nn[flip]*=-1.0
                direction=np.sum(nn*ww[:,None],axis=0);dn=float(np.linalg.norm(direction))
                if dn<1e-12:break
                direction/=dn;projection=normals[bad]@direction;positive=projection>.05
                if not np.any(positive):break
                needed=float(np.max(ww[positive]/np.maximum(projection[positive],.05)));step=min(.0040,needed+.00010);coherent=coherent+direction*step;clearance_rounds=clearance_round+1
            samples=np.einsum("bk,fkj->fbj",bary,coherent[LF]).reshape(-1,3);_,_,final_signed,_,_,_=_nearest_literal_surface_consistent_with_support(samples,target_tri,target_support_tri,exact_base=True)
            final_min=float(np.min(final_signed)) if len(final_signed) else np.inf
            if final_min<float(margin)-.00008:continue
            before_quality=_component_topology_quality(P,V[ids],LF);after_quality=_component_topology_quality(P,coherent,LF)
            before_area=max(float(before_quality["area_ratio_p01"]),1e-6);after_area=max(float(after_quality["area_ratio_p01"]),1e-6)
            area_error_before=abs(float(np.log(before_area)));area_error_after=abs(float(np.log(after_area)))
            if after_quality["flip_fraction"]>before_quality["flip_fraction"]+1e-6 or area_error_after>area_error_before+.03:continue
            correction=np.linalg.norm(coherent-V[ids],axis=1);V=V.copy();V[ids]=coherent;out[name]=V;changed.add(name)
            reports.append({"mesh":name,"raw_component":int(component_index),"slot":slot,"vertices":int(len(ids)),"extent_mm":max_extent*1000.0,"boundary_fraction":boundary_fraction,"middle_ratio":middle_ratio,"thin_ratio":thin_ratio,"roughness_p95_before_mm":roughness*1000.0,"edge_p99_before":edge_p99,"similarity_scale":float(scale),"clearance_rounds":int(clearance_rounds),"sample_min_after_mm":final_min*1000.0,"correction_p95_mm":float(np.percentile(correction,95)*1000.0),"topology_before":before_quality,"topology_after":after_quality})
    return out,changed,{"enabled":bool(reports),"adjusted_mesh_count":len(changed),"component_count":len(reports),"components":reports,"policy":"distorted thin raw garment strips preserve authored similarity shape and clear the literal target as one coherent component"}


def _preserve_authored_ribbon_components(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]]):
    """Restore small open ribbon/tie components by source-derived similarity, not garment names."""
    out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()};changed=set();reports=[]
    for name,context in contexts.items():
        if name not in out:continue
        data=context["data"];w=context["w"];labels=np.asarray(context["labels"],dtype=np.int64);classes=context["classes"];raw_to_weld=np.asarray(w["raw_to_weld"],dtype=np.int64)
        raw_labels=labels[raw_to_weld];src=np.asarray(data["V"],dtype=np.float64);cur=out[name];F=np.asarray(data["F"],dtype=np.int64)
        for component in sorted(set(int(x) for x in labels.tolist())):
            if str(classes.get(component,"shell")).casefold()!="shell":continue
            raw_ids=np.flatnonzero(raw_labels==component)
            if len(raw_ids)<24 or len(raw_ids)>1800:continue
            P=src[raw_ids];extent=np.ptp(P,axis=0);max_extent=float(np.max(extent))
            if max_extent>.16:continue
            centred=P-P.mean(axis=0);sv=np.linalg.svd(centred,compute_uv=False)
            if len(sv)<3 or sv[0]<1e-9:continue
            middle_ratio=float(sv[1]/sv[0]);thin_ratio=float(sv[2]/sv[0])
            boundary_fraction=_raw_component_boundary_fraction(src,F,raw_ids)
            if boundary_fraction<.30 or middle_ratio>.45 or thin_ratio>.24:continue
            coherent,scale=_similarity_fit_points(P,cur[raw_ids])
            before=cur[raw_ids].copy();cur=cur.copy();cur[raw_ids]=coherent;out[name]=cur;changed.add(name)
            reports.append({"mesh":name,"component":int(component),"vertices":int(len(raw_ids)),"boundary_fraction":boundary_fraction,"middle_ratio":middle_ratio,"thin_ratio":thin_ratio,"scale":scale,"alpha":1.0,"correction_p95_mm":float(np.percentile(np.linalg.norm(cur[raw_ids]-before,axis=1),95)*1000.0)})
    return out,changed,{"enabled":True,"adjusted_mesh_count":int(len(changed)),"component_count":int(len(reports)),"components":reports,"policy":"small open ribbon/tie components preserve source-authored shape under solved similarity transform"}


def _vertex_normals_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);N=np.zeros_like(V)
    if len(V)==0 or len(F)==0:return N
    A=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    np.add.at(N,F[:,0],A);np.add.at(N,F[:,1],A);np.add.at(N,F[:,2],A)
    N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
    return N



def _transport_vectors_between_normals(vectors: np.ndarray, source_normals: np.ndarray, target_normals: np.ndarray) -> np.ndarray:
    """Transport local vectors through the shortest source-normal -> target-normal rotation."""
    V=np.asarray(vectors,dtype=np.float64);A=np.asarray(source_normals,dtype=np.float64);B=np.asarray(target_normals,dtype=np.float64)
    if V.shape!=A.shape or V.shape!=B.shape or V.ndim!=2 or V.shape[1]!=3:
        raise ValueError("Normal-frame transport requires matching Nx3 vectors and normals.")
    an=np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-12);bn=np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-12);a=A/an;b=B/bn
    cross=np.cross(a,b);s=np.linalg.norm(cross,axis=1);c=np.clip(np.einsum("ij,ij->i",a,b),-1.0,1.0);out=V.copy();regular=s>1e-9
    if np.any(regular):
        k=cross[regular]/s[regular,None];v=V[regular];cc=c[regular,None];ss=s[regular,None]
        out[regular]=v*cc+np.cross(k,v)*ss+k*np.einsum("ij,ij->i",k,v)[:,None]*(1.0-cc)
    anti=(~regular)&(c<0.0)
    if np.any(anti):
        aa=a[anti];basis=np.zeros_like(aa);choice=np.argmin(np.abs(aa),axis=1);basis[np.arange(len(aa)),choice]=1.0;k=np.cross(aa,basis);k/=np.maximum(np.linalg.norm(k,axis=1,keepdims=True),1e-12);v=V[anti]
        out[anti]=-v+2.0*k*np.einsum("ij,ij->i",k,v)[:,None]
    return out


def _coherent_coverage_vertex_field(vertex_count: int, faces: np.ndarray, barycentric: np.ndarray, errors: np.ndarray, sample_source_normals: np.ndarray, sample_target_normals: np.ndarray, vertex_source_normals: np.ndarray, vertex_target_normals: np.ndarray):
    """Reconcile source-authorised face constraints without averaging across opposite body folds."""
    F=np.asarray(faces,dtype=np.int64);B=np.asarray(barycentric,dtype=np.float64);E=np.asarray(errors,dtype=np.float64);SNS=np.asarray(sample_source_normals,dtype=np.float64).copy();TNS=np.asarray(sample_target_normals,dtype=np.float64).copy();VNS=np.asarray(vertex_source_normals,dtype=np.float64).copy();VNT=np.asarray(vertex_target_normals,dtype=np.float64).copy()
    SNS/=np.maximum(np.linalg.norm(SNS,axis=1,keepdims=True),1e-12);TNS/=np.maximum(np.linalg.norm(TNS,axis=1,keepdims=True),1e-12);VNS/=np.maximum(np.linalg.norm(VNS,axis=1,keepdims=True),1e-12);VNT/=np.maximum(np.linalg.norm(VNT,axis=1,keepdims=True),1e-12)
    if len(F)==0:return np.zeros((int(vertex_count),3),dtype=np.float64),{"contributions":0,"cross_fold_rejected":0,"fallback_vertices":0,"direct_vertices":0}
    vi=F.reshape(-1);corner=B.reshape(-1);err=np.repeat(E,3,axis=0);sns=np.repeat(SNS,3,axis=0);tns=np.repeat(TNS,3,axis=0)
    base=np.where(corner>.04,.15+.85*np.clip(corner,0.0,1.0),0.0)
    src_align=np.einsum("ij,ij->i",sns,VNS[vi]);tgt_align=np.einsum("ij,ij->i",tns,VNT[vi]);score=np.minimum(src_align,tgt_align)
    same_side=(base>0.0)&(score>=.15)
    total=np.zeros(int(vertex_count),dtype=np.float64);accepted=np.zeros(int(vertex_count),dtype=np.float64);np.add.at(total,vi,base);np.add.at(accepted,vi,np.where(same_side,base,0.0))
    needs_fallback=(total>0.0)&(accepted<=0.0);max_score=np.full(int(vertex_count),-np.inf,dtype=np.float64);valid_base=base>0.0
    np.maximum.at(max_score,vi[valid_base],score[valid_base]);fallback=valid_base&needs_fallback[vi]&(score>=max_score[vi]-1e-12);use=same_side|fallback
    gain=np.clip(.25+.75*np.maximum(score,0.0),.10,1.0);w=np.where(use,base*gain,0.0)
    accum=np.zeros((int(vertex_count),3),dtype=np.float64);weight=np.zeros(int(vertex_count),dtype=np.float64);np.add.at(accum,vi,err*w[:,None]);np.add.at(weight,vi,w)
    field=np.zeros_like(accum);direct=weight>0.0;field[direct]=accum[direct]/weight[direct,None]
    return field,{"contributions":int(np.count_nonzero(base>0.0)),"cross_fold_rejected":int(np.count_nonzero((base>0.0)&~use)),"fallback_vertices":int(np.count_nonzero(needs_fallback)),"direct_vertices":int(np.count_nonzero(direct))}


def _coherent_coverage_frame_self_test() -> dict[str, object]:
    faces=np.asarray([[0,1,2],[0,1,2]],dtype=np.int64);bary=np.asarray([[1.0,0.0,0.0],[1.0,0.0,0.0]],dtype=np.float64);errors=np.asarray([[.002,0.0,0.0],[-.006,0.0,0.0]],dtype=np.float64);sns=np.asarray([[0.0,0.0,1.0],[0.0,0.0,-1.0]],dtype=np.float64);vns=np.asarray([[0.0,0.0,1.0]]*3,dtype=np.float64)
    field,report=_coherent_coverage_vertex_field(3,faces,bary,errors,sns,sns,vns,vns)
    if not np.isfinite(field).all() or float(field[0,0])<=0.0019 or int(report.get("cross_fold_rejected",0))<1:
        raise RuntimeError("Coherent coverage frame self-test failed to reject an opposite-fold constraint.")
    transported=_transport_vectors_between_normals(np.asarray([[.002,.003,.004]]),np.asarray([[0.0,0.0,1.0]]),np.asarray([[0.0,1.0,0.0]]))
    if not np.allclose(transported,np.asarray([[.002,.004,-.003]]),atol=1e-12):
        raise RuntimeError("Coverage normal-frame transport self-test failed.")
    return {"ok":True,"cross_fold_rejected":int(report.get("cross_fold_rejected",0))}


def _preserve_source_authored_body_coverage(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, source_body_points: np.ndarray, target_body_points: np.ndarray, source_body_normals: np.ndarray, target_body_normals: np.ndarray, source_limit: float=.0040, margin: float=.00065):
    """Preserve source-proven body coverage for close-fitting garments without anatomy labels."""
    Vsrc=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64)
    X=np.asarray(source_body_points,dtype=np.float64);Y=np.asarray(target_body_points,dtype=np.float64);NS=np.asarray(source_body_normals,dtype=np.float64);NT=np.asarray(target_body_normals,dtype=np.float64)
    if len(V)==0 or len(F)==0 or len(X)<16 or X.shape!=Y.shape or NS.shape!=X.shape or NT.shape!=Y.shape:
        return V,{"enabled":False,"reason":"insufficient corresponding body/garment geometry","source_covered_points":0,"undercovered_before":0,"undercovered_after":0}
    source_tri=Vsrc[F]
    source_contact,source_garment_normal,_,source_distance,source_face_index=_b14_nearest_surface(X,source_tri,k=32)
    garment_alignment=np.abs(np.einsum("ij,ij->i",source_garment_normal,NS));body_alignment=np.einsum("ij,ij->i",NS,NT)
    body_move=np.linalg.norm(Y-X,axis=1)
    covered=(source_distance<=float(source_limit))&(garment_alignment>=.20)&(body_alignment>0.0)&(body_move<=.060)
    covered_ids=np.flatnonzero(covered)
    if len(covered_ids)<8:
        return V,{"enabled":True,"reason":"too little source-proven body coverage","source_covered_points":int(len(covered_ids)),"undercovered_before":0,"undercovered_after":0}
    source_faces=F[np.asarray(source_face_index[covered_ids],dtype=np.int64)]
    source_bary=trimesh.triangles.points_to_barycentric(Vsrc[source_faces],source_contact[covered_ids])
    source_offset=source_contact[covered_ids]-X[covered_ids]
    covered_source_normals=NS[covered_ids]/np.maximum(np.linalg.norm(NS[covered_ids],axis=1,keepdims=True),1e-12);covered_target_normals=NT[covered_ids]/np.maximum(np.linalg.norm(NT[covered_ids],axis=1,keepdims=True),1e-12)
    source_normal_clearance=np.einsum("ij,ij->i",source_offset,covered_source_normals)
    source_normal_clearance=np.clip(source_normal_clearance,max(float(margin),.00045),.0030)
    transported_offset=_transport_vectors_between_normals(source_offset,covered_source_normals,covered_target_normals)
    transported_tangent=transported_offset-covered_target_normals*np.einsum("ij,ij->i",transported_offset,covered_target_normals)[:,None]
    desired_contacts=Y[covered_ids]+transported_tangent+covered_target_normals*source_normal_clearance[:,None]
    own_body_index=np.asarray(cKDTree(X).query(Vsrc,k=1,workers=1)[1],dtype=np.int64)
    vertex_source_normals=NS[own_body_index];vertex_target_normals=NT[own_body_index]
    edges=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0);ea=edges[:,0];eb=edges[:,1]
    degree=np.bincount(np.concatenate([ea,eb]),minlength=len(V)).astype(np.float64) if len(edges) else np.zeros(len(V),dtype=np.float64)
    nbr=[[] for _ in range(len(V))]
    for a,b in edges:
        nbr[int(a)].append(int(b));nbr[int(b)].append(int(a))
    baseline=V.copy();base_area=np.cross(baseline[F[:,1]]-baseline[F[:,0]],baseline[F[:,2]]-baseline[F[:,0]]);base_norm=np.linalg.norm(base_area,axis=1);valid=base_norm>1e-12
    cumulative=np.zeros(len(V),dtype=np.float64);before_under=None;before_p95=None;before_max=None;iterations=0
    reconcile_totals={"contributions":0,"cross_fold_rejected":0,"fallback_vertices":0,"direct_vertices_peak":0}
    for iteration in range(4):
        _,_,_,target_distance,_=_b14_nearest_surface(Y,V[F],k=32)
        covered_distance=target_distance[covered_ids]
        under=covered_distance>np.maximum(source_distance[covered_ids]+.0018,.0050)
        if before_under is None:
            before_under=int(np.count_nonzero(under));before_p95=float(np.percentile(covered_distance,95)*1000.0);before_max=float(np.max(covered_distance)*1000.0)
        if not np.any(under):break
        iterations=iteration+1;ids=covered_ids[under];faces=source_faces[under];bary=source_bary[under]
        desired=desired_contacts[under]
        current_sample=np.einsum("ni,nij->nj",bary,V[faces]);error=desired-current_sample
        error_mag=np.linalg.norm(error,axis=1);error*=np.minimum(1.0,.012/np.maximum(error_mag,1e-12))[:,None]
        field,reconcile=_coherent_coverage_vertex_field(len(V),faces,bary,error,NS[ids],NT[ids],vertex_source_normals,vertex_target_normals)
        direct=np.linalg.norm(field,axis=1)>0.0;active=direct.copy()
        reconcile_totals["contributions"]+=int(reconcile["contributions"]);reconcile_totals["cross_fold_rejected"]+=int(reconcile["cross_fold_rejected"]);reconcile_totals["fallback_vertices"]+=int(reconcile["fallback_vertices"]);reconcile_totals["direct_vertices_peak"]=max(int(reconcile_totals["direct_vertices_peak"]),int(reconcile["direct_vertices"]))
        for _ in range(3):
            if not len(edges):break
            sums=np.zeros_like(V);np.add.at(sums,ea,field[eb]);np.add.at(sums,eb,field[ea]);average=sums/np.maximum(degree[:,None],1.0)
            next_active=active.copy();edge_active=active[ea]|active[eb]
            if np.any(edge_active):next_active[ea[edge_active]]=True;next_active[eb[edge_active]]=True
            nxt=field.copy();stay=next_active&active;grow=next_active&~active
            nxt[stay]=.86*field[stay]+.14*average[stay];nxt[grow]=.10*average[grow]
            field=nxt;active=next_active
        magnitude=np.linalg.norm(field,axis=1);remaining=np.maximum(0.0,.015-cumulative);step_cap=np.minimum(.007,remaining)
        field*=np.minimum(1.0,step_cap/np.maximum(magnitude,1e-12))[:,None]
        trial=V+field
        for _ in range(10):
            area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);area_norm=np.linalg.norm(area,axis=1);dot=np.ones(len(F));ratio=np.ones(len(F))
            dot[valid]=np.einsum("ij,ij->i",base_area[valid],area[valid])/np.maximum(base_norm[valid]*area_norm[valid],1e-24);ratio[valid]=area_norm[valid]/np.maximum(base_norm[valid],1e-12)
            unsafe=valid&((dot<=.02)|(ratio<=.10))
            if not np.any(unsafe):break
            unsafe_vertices=np.unique(F[unsafe].reshape(-1));trial[unsafe_vertices]=V[unsafe_vertices]+.5*(trial[unsafe_vertices]-V[unsafe_vertices])
        step=np.linalg.norm(trial-V,axis=1)
        if float(np.max(step,initial=0.0))<1e-8:break
        cumulative+=step;V=trial
    residual_rounds=[]
    for residual_round in range(8):
        _,_,_,residual_distance,_=_b14_nearest_surface(Y,V[F],k=32);covered_distance=residual_distance[covered_ids]
        threshold=np.maximum(source_distance[covered_ids]+.0018,.0050);under=covered_distance>threshold
        under_rows=np.flatnonzero(under)
        if len(under_rows)==0:break
        moved=0
        for row in under_rows:
            face=np.asarray(source_faces[row],dtype=np.int64);bary=np.asarray(source_bary[row],dtype=np.float64)
            desired=desired_contacts[row]
            current=np.einsum("i,ij->j",bary,V[face]);error=desired-current;em=float(np.linalg.norm(error))
            if em<1e-8:continue
            error*=min(1.0,.00150/em)
            contributing=face[bary>.05];core=set(int(x) for x in (contributing if len(contributing)>=2 else face).tolist())
            rings=[core];seen=set(core);front=set(core)
            for _ in range(8):
                nxt=set()
                for vi in front:nxt.update(int(nb) for nb in nbr[vi] if int(nb) not in seen)
                rings.append(nxt);seen.update(nxt);front=nxt
                if not front:break
            field=np.zeros_like(V);ring_scales=(1.0,.86,.72,.58,.44,.32,.22,.14,.08)
            for ring_index,ring in enumerate(rings):
                if not ring:continue
                scale=ring_scales[min(ring_index,len(ring_scales)-1)];idx=np.fromiter(ring,dtype=np.int64);field[idx]=error*scale
            current_area=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);current_norm=np.linalg.norm(current_area,axis=1);current_valid=current_norm>1e-12
            touched_vertices=np.flatnonzero(np.linalg.norm(field,axis=1)>0.0);vertex_mask=np.zeros(len(V),dtype=bool);vertex_mask[touched_vertices]=True;touched=np.any(vertex_mask[F],axis=1)
            accepted=None
            for alpha in (1.0,.80,.60,.40,.25,.15,.08,.04,.02):
                trial=V+alpha*field;area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);nn=np.linalg.norm(area,axis=1);dot=np.ones(len(F));ratio=np.ones(len(F));dot[current_valid]=np.einsum("ij,ij->i",current_area[current_valid],area[current_valid])/np.maximum(current_norm[current_valid]*nn[current_valid],1e-24);ratio[current_valid]=nn[current_valid]/np.maximum(current_norm[current_valid],1e-12)
                unsafe_current=touched&current_valid&((dot<=.02)|(ratio<=.08))
                if not np.any(unsafe_current):accepted=trial;break
            if accepted is None:continue
            step=np.linalg.norm(accepted-V,axis=1);cumulative+=step;V=accepted;moved+=1
        residual_rounds.append({"round":residual_round+1,"undercovered_before":int(len(under_rows)),"moved_patches":int(moved)})
        if moved==0:break
    _,_,_,final_distance,_=_b14_nearest_surface(Y,V[F],k=32);covered_final=final_distance[covered_ids]
    under_after=covered_final>np.maximum(source_distance[covered_ids]+.0018,.0050)
    return V,{"enabled":True,"policy":"source-proven corresponding body coverage; source-frame tangent transport + same-hemisphere vertex reconciliation + boundary-aware residual patches","source_covered_points":int(len(covered_ids)),"undercovered_before":int(before_under or 0),"undercovered_after":int(np.count_nonzero(under_after)),"iterations":int(iterations),"reconciliation":reconcile_totals,"residual_patch_rounds":residual_rounds,"target_distance_p95_before_mm":before_p95,"target_distance_max_before_mm":before_max,"target_distance_p95_after_mm":float(np.percentile(covered_final,95)*1000.0),"target_distance_max_after_mm":float(np.max(covered_final)*1000.0),"moved_vertices":int(np.count_nonzero(cumulative>1e-7)),"move_p95_mm":float(np.percentile(cumulative,95)*1000.0),"move_max_mm":float(np.max(cumulative,initial=0.0)*1000.0),"source_limit_mm":float(source_limit*1000.0) }



_REFERENCE_SURFACE_INDEX_CACHE: dict[int, tuple[np.ndarray, np.ndarray, cKDTree, np.ndarray]] = {}

def _reference_surface_index(triangles: np.ndarray):
    """Reuse immutable late-stage triangle centre indices inside one solver process."""
    T=np.asarray(triangles,dtype=np.float64);key=id(T);cached=_REFERENCE_SURFACE_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is T:return cached[1],cached[2],cached[3]
    centres=T.mean(axis=1);tree=cKDTree(centres);fn=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-12)
    if len(_REFERENCE_SURFACE_INDEX_CACHE)>=16:_REFERENCE_SURFACE_INDEX_CACHE.pop(next(iter(_REFERENCE_SURFACE_INDEX_CACHE)))
    _REFERENCE_SURFACE_INDEX_CACHE[key]=(T,centres,tree,fn);return centres,tree,fn

def _nearest_surface_reference_chunked(points: np.ndarray, triangles: np.ndarray, k: int=32, chunk_size: int=6000):
    """Exact non-Numba nearest-surface query used for late-stage validation, with reusable surface index."""
    P=np.asarray(points,dtype=np.float64);T=np.asarray(triangles,dtype=np.float64);kk=min(int(k),len(T));_,tree,fn=_reference_surface_index(T)
    cp=np.empty((len(P),3),dtype=np.float64);fi=np.empty(len(P),dtype=np.int64);dist2=np.empty(len(P),dtype=np.float64)
    for start in range(0,len(P),int(chunk_size)):
        stop=min(len(P),start+int(chunk_size));Pc=P[start:stop];_,idx=tree.query(Pc,k=kk,workers=1);idx=idx if idx.ndim>1 else idx[:,None]
        m=len(Pc);Tc=T[idx.reshape(-1)];Q=np.repeat(Pc,kk,axis=0);C=trimesh.triangles.closest_point(Tc,Q).reshape(m,kk,3);dd=np.sum((C-Pc[:,None,:])**2,axis=2);winner=np.argmin(dd,axis=1);rows=np.arange(m)
        cp[start:stop]=C[rows,winner];fi[start:stop]=idx[rows,winner];dist2[start:stop]=dd[rows,winner]
    normals=fn[fi];signed=np.sum((P-cp)*normals,axis=1);distance=np.sqrt(dist2);return cp,normals,signed,distance,fi

def _nearest_literal_occupancy(points: np.ndarray, triangles: np.ndarray, k: int=48, exact_band: float=.0025):
    """Fast compiled literal-body query with exact verification only near/inside the collision band.

    The compiled B14 candidate kernel is used for the broad query.  Samples close enough to affect
    clearance are then rechecked by the exact late-stage reference query, preserving literal occupancy
    authority without allocating huge all-sample trimesh candidate arrays.
    """
    P=np.asarray(points,dtype=np.float64);T=np.asarray(triangles,dtype=np.float64)
    cp,normals,signed,distance,face_index=_b14_nearest_surface(P,T,k=max(32,int(k)))
    verify=np.flatnonzero(np.asarray(signed,dtype=np.float64)<float(exact_band))
    if len(verify):
        ecp,en,es,ed,efi=_nearest_surface_reference_chunked(P[verify],T,k=max(32,min(int(k),64)),chunk_size=2000)
        cp[verify]=ecp;normals[verify]=en;signed[verify]=es;distance[verify]=ed;face_index[verify]=efi
    return cp,normals,signed,distance,face_index

def _nearest_surface_oriented_reference_chunked(points: np.ndarray, triangles: np.ndarray, reference_normals: np.ndarray, k: int=64, min_alignment: float=.10, chunk_size: int=3000):
    """Nearest literal surface constrained to the reference-normal hemisphere, reusing the surface index."""
    P=np.asarray(points,dtype=np.float64);T=np.asarray(triangles,dtype=np.float64);R=np.asarray(reference_normals,dtype=np.float64)
    if P.shape!=R.shape or P.ndim!=2 or P.shape[1]!=3:raise ValueError("Oriented surface query requires matching Nx3 points/reference normals.")
    if len(P)==0:return np.zeros((0,3)),np.zeros((0,3)),np.zeros(0),np.zeros(0),np.zeros(0,dtype=np.int64),np.zeros(0,dtype=bool)
    kk=min(int(k),len(T));_,tree,fn=_reference_surface_index(T);R=R/np.maximum(np.linalg.norm(R,axis=1,keepdims=True),1e-12)
    cp=np.empty((len(P),3),dtype=np.float64);fi=np.empty(len(P),dtype=np.int64);dist2=np.empty(len(P),dtype=np.float64);used=np.zeros(len(P),dtype=bool)
    for start in range(0,len(P),int(chunk_size)):
        stop=min(len(P),start+int(chunk_size));Pc=P[start:stop];Rc=R[start:stop];_,idx=tree.query(Pc,k=kk,workers=1);idx=idx if idx.ndim>1 else idx[:,None]
        m=len(Pc);Tc=T[idx.reshape(-1)];Q=np.repeat(Pc,kk,axis=0);C=trimesh.triangles.closest_point(Tc,Q).reshape(m,kk,3);dd=np.sum((C-Pc[:,None,:])**2,axis=2);align=np.einsum("mkj,mj->mk",fn[idx],Rc);allowed=align>=float(min_alignment);has=np.any(allowed,axis=1);masked=np.where(allowed,dd,np.inf);winner=np.argmin(masked,axis=1);fallback=np.argmin(dd,axis=1);winner=np.where(has,winner,fallback);rows=np.arange(m)
        cp[start:stop]=C[rows,winner];fi[start:stop]=idx[rows,winner];dist2[start:stop]=dd[rows,winner];used[start:stop]=has
    normals=fn[fi];signed=np.sum((P-cp)*normals,axis=1);distance=np.sqrt(dist2);return cp,normals,signed,distance,fi,used

def _nearest_literal_surface_consistent_with_support(points: np.ndarray, literal_triangles: np.ndarray, support_triangles: np.ndarray | None, *, exact_base: bool=False, deep_negative: float=-.00075):
    """Reject an opposite-facing literal winner only for materially negative collision samples."""
    P=np.asarray(points,dtype=np.float64);T=np.asarray(literal_triangles,dtype=np.float64)
    if exact_base:cp,normals,signed,distance,face_index=_nearest_surface_reference_chunked(P,T,k=32)
    else:cp,normals,signed,distance,face_index=_b14_nearest_surface(P,T,k=32)
    if support_triangles is None or len(support_triangles)==0 or len(P)==0:return cp,normals,signed,distance,face_index,0
    suspicious=np.flatnonzero(signed<float(deep_negative))
    if len(suspicious)==0:return cp,normals,signed,distance,face_index,0
    _,support_normals,_,_,_=_nearest_surface_reference_chunked(P[suspicious],np.asarray(support_triangles,dtype=np.float64),k=32)
    agreement=np.einsum("ij,ij->i",normals[suspicious],support_normals);wrong=np.flatnonzero(agreement<.10)
    if len(wrong)==0:return cp,normals,signed,distance,face_index,0
    ids=suspicious[wrong];ocp,on,os,od,ofi,used=_nearest_surface_oriented_reference_chunked(P[ids],T,support_normals[wrong],k=64,min_alignment=.10)
    replace=np.flatnonzero(used)
    if len(replace):
        rid=ids[replace];cp[rid]=ocp[replace];normals[rid]=on[replace];signed[rid]=os[replace];distance[rid]=od[replace];face_index[rid]=ofi[replace]
    return cp,normals,signed,distance,face_index,int(len(replace))


def _clearance_push_normals_with_bisector(support_normals: np.ndarray, literal_normals: np.ndarray, signed: np.ndarray, *, deep_negative: float=-.00120, low_alignment: float=.35, hemisphere_floor: float=-.05):
    """Blend smooth-support and literal collision normals for rare pinned deep collisions."""
    S=np.asarray(support_normals,dtype=np.float64).copy();L=np.asarray(literal_normals,dtype=np.float64).copy();sg=np.asarray(signed,dtype=np.float64).reshape(-1)
    if S.shape!=L.shape or S.ndim!=2 or S.shape[1]!=3 or len(S)!=len(sg):
        raise ValueError('Bisector clearance normals require matching Nx3 support/literal normals and N signed distances.')
    S/=np.maximum(np.linalg.norm(S,axis=1,keepdims=True),1e-12);L/=np.maximum(np.linalg.norm(L,axis=1,keepdims=True),1e-12)
    agreement=np.einsum('ij,ij->i',S,L)
    use=(sg<float(deep_negative))&(agreement<float(low_alignment))&(agreement>float(hemisphere_floor))
    out=S.copy()
    if np.any(use):
        B=S[use]+L[use];bn=np.linalg.norm(B,axis=1,keepdims=True)
        fallback=(bn[:,0]<=1e-12)
        if np.any(~fallback):
            out[np.flatnonzero(use)[~fallback]]=B[~fallback]/bn[~fallback]
    return out,{'used':int(np.count_nonzero(use)),'deep_negative_threshold_mm':float(deep_negative*1000.0),'low_alignment_threshold':float(low_alignment)}


def _source_covered_face_clearance(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, source_triangles: np.ndarray, target_triangles: np.ndarray, margin: float=.00065, target_support_triangles: np.ndarray | None=None, _bounded_retry_budget: int=1):
    """Clear target-body intersections only where the untouched source garment covered the source body."""
    Vsrc=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64)
    if len(V)==0 or len(F)==0:return V,{"affected_faces":0,"iterations":0}
    bary=np.asarray([(i/4.0,j/4.0,(4-i-j)/4.0) for i in range(5) for j in range(5-i)],dtype=np.float64)
    source_samples=np.einsum("bk,fkj->fbj",bary,Vsrc[F]).reshape(-1,3)
    _,_,source_signed,source_distance,_=_b14_nearest_surface(source_samples,source_triangles,k=32)
    source_signed=source_signed.reshape(len(F),len(bary));source_distance=source_distance.reshape(len(F),len(bary))
    # Source is authority: only face samples that were genuinely body-covering may be corrected.
    authored=(source_distance<=.012)&(source_signed>=-.0015)&(source_signed<=.012)
    if not np.any(authored):return V,{"affected_faces":0,"iterations":0,"reason":"no source-authored body contact"}
    desired=np.maximum(float(margin),np.clip(source_signed,.00045,.00150))
    contact_face_ids=np.flatnonzero(np.any(authored,axis=1));contact_authored=authored[contact_face_ids];contact_desired=desired[contact_face_ids]
    face_to_contact=np.full(len(F),-1,dtype=np.int64);face_to_contact[contact_face_ids]=np.arange(len(contact_face_ids),dtype=np.int64)
    original=V.copy();edges=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0);nbr=[[] for _ in range(len(V))]
    for a,b in edges:nbr[int(a)].append(int(b));nbr[int(b)].append(int(a))
    base_area=np.cross(original[F[:,1]]-original[F[:,0]],original[F[:,2]]-original[F[:,0]]);base_norm=np.linalg.norm(base_area,axis=1);valid=base_norm>1e-12
    affected=set();initial_min=None;iterations=0;opposite_facing_literal_rejections=0;bisector_pushes=0
    for ri in range(12):
        sample_grid=np.einsum("bk,fkj->fbj",bary,V[F[contact_face_ids]]);samples=sample_grid.reshape(-1,3);_,literal_normals,signed,_,_,oriented_rejections=_nearest_literal_surface_consistent_with_support(samples,target_triangles,target_support_triangles);opposite_facing_literal_rejections+=int(oriented_rejections);signed=signed.reshape(len(contact_face_ids),len(bary));literal_normals=literal_normals.reshape(len(contact_face_ids),len(bary),3)
        relevant=signed[contact_authored]
        if initial_min is None and len(relevant):initial_min=float(np.min(relevant))
        deficit=np.where(contact_authored,np.maximum(0.0,contact_desired-signed),0.0);bad_local=np.max(deficit,axis=1)>1e-6
        if not np.any(bad_local):break
        iterations=ri+1;acc=np.zeros_like(V);weight=np.zeros(len(V),dtype=np.float64)
        bad_rows=np.flatnonzero(bad_local);bad_ids=contact_face_ids[bad_rows];affected.update(int(x) for x in bad_ids.tolist())
        chosen=np.argmax(deficit[bad_rows],axis=1);need=np.minimum(deficit[bad_rows,chosen],.0018)
        if target_support_triangles is not None and len(target_support_triangles):
            direction_points=sample_grid[bad_rows,chosen]
            _,support_n,_,_,_=_nearest_surface_reference_chunked(direction_points,np.asarray(target_support_triangles,dtype=np.float64),k=32)
            n,bisector_report=_clearance_push_normals_with_bisector(support_n,literal_normals[bad_rows,chosen],signed[bad_rows,chosen])
            bisector_pushes+=int(bisector_report.get('used',0))
        else:n=literal_normals[bad_rows,chosen]
        weights=.30+.70*bary[chosen]                         # [faces,3]
        vertex_ids=F[bad_ids]                               # [faces,3]
        contrib=n[:,None,:]*need[:,None,None]*weights[:,:,None]
        np.add.at(acc,vertex_ids.reshape(-1),contrib.reshape(-1,3));np.add.at(weight,vertex_ids.reshape(-1),weights.reshape(-1))
        direct=weight>0;field=np.zeros_like(V);field[direct]=acc[direct]/weight[direct,None]
        ea=edges[:,0];eb=edges[:,1];degree=np.bincount(np.concatenate([ea,eb]),minlength=len(V)).astype(np.float64)
        active=direct.copy()
        for _ in range(6):
            sums=np.zeros_like(V);np.add.at(sums,ea,field[eb]);np.add.at(sums,eb,field[ea]);avg=sums/np.maximum(degree[:,None],1.0)
            next_active=active.copy();edge_active=active[ea]|active[eb]
            if np.any(edge_active):
                eaa=ea[edge_active];ebb=eb[edge_active];next_active[eaa]=True;next_active[ebb]=True
            nxt=field.copy();stay=next_active&active;grow=next_active&~active
            nxt[stay]=.78*field[stay]+.22*avg[stay];nxt[grow]=.18*avg[grow]
            field=nxt;active=next_active
        mag=np.linalg.norm(field,axis=1);field*=np.minimum(1.0,.0018/np.maximum(mag,1e-12))[:,None]
        trial=V+field;area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);nn=np.linalg.norm(area,axis=1);dot=np.ones(len(F));ratio=np.ones(len(F));dot[valid]=np.einsum("ij,ij->i",base_area[valid],area[valid])/np.maximum(base_norm[valid]*nn[valid],1e-24);ratio[valid]=nn[valid]/np.maximum(base_norm[valid],1e-12);unsafe=valid&((dot<=.08)|(ratio<=.15))
        if np.any(unsafe):
            freeze=np.unique(F[unsafe].reshape(-1));trial[freeze]=V[freeze]
        if np.max(np.linalg.norm(trial-V,axis=1))<1e-9:break
        V=trial
    patch_rounds=[]
    for patch_round in range(4):
        sample_grid=np.einsum("bk,fkj->fbj",bary,V[F[contact_face_ids]]);samples=sample_grid.reshape(-1,3);_,literal_normals,signed,_,_,oriented_rejections=_nearest_literal_surface_consistent_with_support(samples,target_triangles,target_support_triangles);opposite_facing_literal_rejections+=int(oriented_rejections);signed=signed.reshape(len(contact_face_ids),len(bary));literal_normals=literal_normals.reshape(len(contact_face_ids),len(bary),3)
        deficit=np.where(contact_authored,np.maximum(0.0,contact_desired-signed),0.0);bad_rows=np.flatnonzero(np.max(deficit,axis=1)>.00008);bad_faces=contact_face_ids[bad_rows]
        if len(bad_faces)==0:break
        bad_choice=np.argmax(deficit[bad_rows],axis=1)
        if target_support_triangles is not None and len(target_support_triangles):
            _,support_push_normals,_,_,_=_nearest_surface_reference_chunked(sample_grid[bad_rows,bad_choice],np.asarray(target_support_triangles,dtype=np.float64),k=32)
            bad_push_normals,bisector_report=_clearance_push_normals_with_bisector(support_push_normals,literal_normals[bad_rows,bad_choice],signed[bad_rows,bad_choice],deep_negative=-.00080,low_alignment=.45)
            bisector_pushes+=int(bisector_report.get('used',0))
        else:bad_push_normals=literal_normals[bad_rows,bad_choice]
        bad_cr=face_to_contact[bad_faces];bad_dirs=np.asarray(bad_push_normals,dtype=np.float64).copy()
        bad_dirs/=np.maximum(np.linalg.norm(bad_dirs,axis=1)[:,None],1e-12)
        dir_by_face={int(fi):bad_dirs[i] for i,fi in enumerate(bad_faces.tolist())}
        vertex_to_bad={}
        for fi in bad_faces:
            for vi in F[int(fi)]:vertex_to_bad.setdefault(int(vi),[]).append(int(fi))
        remaining=set(int(x) for x in bad_faces.tolist());components=[]
        while remaining:
            seed=remaining.pop();stack=[seed];component=[seed]
            while stack:
                fi=stack.pop();fdir=dir_by_face[fi]
                for vi in F[fi]:
                    for nb_face in vertex_to_bad.get(int(vi),[]):
                        if nb_face not in remaining:continue
                        if float(np.dot(fdir,dir_by_face[nb_face]))<.35:continue
                        remaining.remove(nb_face);stack.append(nb_face);component.append(nb_face)
            components.append(component)
        moved_components=0
        for component in components:
            cf=np.asarray(component,dtype=np.int64);cr=face_to_contact[cf];valid_rows=cr>=0
            if not np.any(valid_rows):continue
            cf=cf[valid_rows];cr=cr[valid_rows];local_def=deficit[cr];chosen=np.argmax(local_def,axis=1);need=local_def[np.arange(len(cf)),chosen]
            row_lookup={int(contact_row):idx for idx,contact_row in enumerate(bad_rows.tolist())};n=np.asarray([bad_push_normals[row_lookup[int(contact_row)]] for contact_row in cr],dtype=np.float64)
            anchor=n[int(np.argmax(need))].copy();anchor/=max(float(np.linalg.norm(anchor)),1e-12)
            aligned=n.copy();flip=np.einsum("ij,j->i",aligned,anchor)<0.0;aligned[flip]*=-1.0
            weight_need=np.maximum(need,1e-9);direction=np.sum(aligned*weight_need[:,None],axis=0);dn=float(np.linalg.norm(direction))
            if dn<1e-10:continue
            direction/=dn;push=float(min(.0045,np.percentile(need,95)+.00045))
            core=set(int(x) for x in np.unique(F[cf].reshape(-1)).tolist());rings=[core];seen=set(core);front=set(core)
            for _ in range(3):
                nxt=set()
                for vi in front:nxt.update(int(nb) for nb in nbr[vi] if int(nb) not in seen)
                rings.append(nxt);seen.update(nxt);front=nxt
            field=np.zeros_like(V)
            for ring_index,ids in enumerate(rings):
                if not ids:continue
                scale=(1.0,.62,.32,.14)[min(ring_index,3)]
                idx=np.fromiter(ids,dtype=np.int64);field[idx]=direction*(push*scale)
            accepted=None;accepted_alpha=0.0
            current_area=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);current_norm=np.linalg.norm(current_area,axis=1);current_valid=current_norm>1e-12
            for alpha in (1.0,.80,.65,.50,.35,.20,.10,.05):
                trial=V+alpha*field;area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);nn=np.linalg.norm(area,axis=1);dot=np.ones(len(F));ratio=np.ones(len(F));dot[current_valid]=np.einsum("ij,ij->i",current_area[current_valid],area[current_valid])/np.maximum(current_norm[current_valid]*nn[current_valid],1e-24);ratio[current_valid]=nn[current_valid]/np.maximum(current_norm[current_valid],1e-12)
                touched=np.zeros(len(F),dtype=bool);touched[cf]=True
                # Include faces incident to the feathered patch when checking safety.
                touched_vertices=np.flatnonzero(np.linalg.norm(field,axis=1)>0.0);vertex_mask=np.zeros(len(V),dtype=bool);vertex_mask[touched_vertices]=True;touched|=np.any(vertex_mask[F],axis=1)
                unsafe_current=touched&current_valid&((dot<=.08)|(ratio<=.20))
                if not np.any(unsafe_current):accepted=trial;accepted_alpha=float(alpha);break
            if accepted is None:continue
            V=accepted;moved_components+=1
        patch_rounds.append({"round":patch_round+1,"stuck_faces":int(len(bad_faces)),"components":int(len(components)),"moved_components":int(moved_components)})
        if moved_components==0:break
    tail_rounds=[]
    for tail_round in range(3):
        sample_grid=np.einsum("bk,fkj->fbj",bary,V[F[contact_face_ids]]);samples=sample_grid.reshape(-1,3);_,literal_normals,signed,_,_,oriented_rejections=_nearest_literal_surface_consistent_with_support(samples,target_triangles,target_support_triangles);opposite_facing_literal_rejections+=int(oriented_rejections);signed=signed.reshape(len(contact_face_ids),len(bary));literal_normals=literal_normals.reshape(len(contact_face_ids),len(bary),3)
        deficit=np.where(contact_authored,np.maximum(0.0,contact_desired-signed),0.0);tail_rows=np.flatnonzero(np.max(deficit,axis=1)>.00018)
        if len(tail_rows)==0:break
        tail_faces=contact_face_ids[tail_rows];tail_choice=np.argmax(deficit[tail_rows],axis=1);tail_need=deficit[tail_rows,tail_choice];tail_signed=signed[tail_rows,tail_choice]
        if target_support_triangles is not None and len(target_support_triangles):
            _,support_tail,_,_,_=_nearest_surface_reference_chunked(sample_grid[tail_rows,tail_choice],np.asarray(target_support_triangles,dtype=np.float64),k=32)
            tail_push_normals,bisector_report=_clearance_push_normals_with_bisector(support_tail,literal_normals[tail_rows,tail_choice],tail_signed,deep_negative=-.00035,low_alignment=.85,hemisphere_floor=-.20)
            bisector_pushes+=int(bisector_report.get('used',0))
        else:
            tail_push_normals=literal_normals[tail_rows,tail_choice]
        tail_dirs=np.asarray(tail_push_normals,dtype=np.float64).copy();tail_dirs/=np.maximum(np.linalg.norm(tail_dirs,axis=1)[:,None],1e-12)
        vertex_to_tail={}
        for fi in tail_faces:
            for vi in F[int(fi)]:vertex_to_tail.setdefault(int(vi),[]).append(int(fi))
        dir_by_face={int(fi):tail_dirs[i] for i,fi in enumerate(tail_faces.tolist())}
        remaining=set(int(x) for x in tail_faces.tolist());components=[]
        while remaining:
            seed=remaining.pop();stack=[seed];component=[seed]
            while stack:
                fi=stack.pop();fdir=dir_by_face[fi]
                for vi in F[fi]:
                    for nb_face in vertex_to_tail.get(int(vi),[]):
                        if nb_face not in remaining:continue
                        if float(np.dot(fdir,dir_by_face[nb_face]))<.10:continue
                        remaining.remove(nb_face);stack.append(nb_face);component.append(nb_face)
            components.append(component)
        moved_components=0
        for component in components:
            cf=np.asarray(component,dtype=np.int64);cr=face_to_contact[cf];valid_rows=cr>=0
            if not np.any(valid_rows):continue
            cf=cf[valid_rows];cr=cr[valid_rows]
            local_def=deficit[cr];chosen=np.argmax(local_def,axis=1);need=local_def[np.arange(len(cf)),chosen];chosen_signed=signed[cr,chosen]
            row_lookup={int(contact_row):idx for idx,contact_row in enumerate(tail_rows.tolist())};n=np.asarray([tail_push_normals[row_lookup[int(contact_row)]] for contact_row in cr],dtype=np.float64)
            anchor=n[int(np.argmax(need))].copy();anchor/=max(float(np.linalg.norm(anchor)),1e-12)
            aligned=n.copy();flip=np.einsum("ij,j->i",aligned,anchor)<0.0;aligned[flip]*=-1.0
            direction=np.sum(aligned*np.maximum(need,1e-9)[:,None],axis=0);dn=float(np.linalg.norm(direction))
            if dn<1e-10:continue
            direction/=dn;push=float(min(.0028,np.percentile(need,90)+.00035))
            core=set(int(x) for x in np.unique(F[cf].reshape(-1)).tolist());rings=[core];seen=set(core);front=set(core)
            for _ in range(2):
                nxt=set()
                for vi in front:nxt.update(int(nb) for nb in nbr[vi] if int(nb) not in seen)
                rings.append(nxt);seen.update(nxt);front=nxt
            field=np.zeros_like(V)
            for ring_index,ids in enumerate(rings):
                if not ids:continue
                scale=(1.0,.45,.18)[min(ring_index,2)]
                idx=np.fromiter(ids,dtype=np.int64);field[idx]=direction*(push*scale)
            current_area=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);current_norm=np.linalg.norm(current_area,axis=1);current_valid=current_norm>1e-12
            base_component_max=float(np.max(local_def))
            accepted=None;accepted_alpha=0.0;accepted_max=base_component_max
            for alpha in (1.0,.85,.70,.55,.40,.25,.15,.08):
                trial=V+alpha*field;area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);nn=np.linalg.norm(area,axis=1);dot=np.ones(len(F));ratio=np.ones(len(F));dot[current_valid]=np.einsum("ij,ij->i",current_area[current_valid],area[current_valid])/np.maximum(current_norm[current_valid]*nn[current_valid],1e-24);ratio[current_valid]=nn[current_valid]/np.maximum(current_norm[current_valid],1e-12)
                touched=np.zeros(len(F),dtype=bool);touched[cf]=True
                touched_vertices=np.flatnonzero(np.linalg.norm(field,axis=1)>0.0);vertex_mask=np.zeros(len(V),dtype=bool);vertex_mask[touched_vertices]=True;touched|=np.any(vertex_mask[F],axis=1)
                unsafe_current=touched&current_valid&((dot<=.08)|(ratio<=.20))
                if np.any(unsafe_current):
                    continue
                trial_grid=np.einsum("bk,fkj->fbj",bary,trial[F[cf]]);trial_samples=trial_grid.reshape(-1,3)
                _,_,trial_signed,_,_,trial_rejections=_nearest_literal_surface_consistent_with_support(trial_samples,target_triangles,target_support_triangles,exact_base=True)
                local_authored=contact_authored[cr];local_desired=contact_desired[cr]
                trial_signed=trial_signed.reshape(len(cf),len(bary));trial_def=np.where(local_authored,np.maximum(0.0,local_desired-trial_signed),0.0)
                trial_max=float(np.max(trial_def)) if len(trial_def) else 0.0
                if trial_max<base_component_max-1e-5:
                    accepted=trial;accepted_alpha=float(alpha);accepted_max=trial_max;opposite_facing_literal_rejections+=int(trial_rejections);break
            if accepted is None:continue
            V=accepted;moved_components+=1
        tail_rounds.append({"round":tail_round+1,"stuck_faces":int(len(tail_faces)),"components":int(len(components)),"moved_components":int(moved_components)})
        if moved_components==0:break
    samples=np.einsum("bk,fkj->fbj",bary,V[F[contact_face_ids]]).reshape(-1,3);_,_,signed,_,_,oriented_rejections=_nearest_literal_surface_consistent_with_support(samples,target_triangles,target_support_triangles,exact_base=True);opposite_facing_literal_rejections+=int(oriented_rejections);signed=signed.reshape(len(contact_face_ids),len(bary));relevant=signed[contact_authored]
    final_min=float(np.min(relevant)) if len(relevant) else None;final_p01=float(np.percentile(relevant,1)) if len(relevant) else None
    report={"affected_faces":len(affected),"iterations":iterations,"patch_rounds":patch_rounds,"tail_rounds":tail_rounds,"opposite_facing_literal_rejections":int(opposite_facing_literal_rejections),"bisector_pushes":int(bisector_pushes),"sample_min_before_mm":initial_min*1000.0 if initial_min is not None else None,"sample_min_after_mm":final_min*1000.0 if final_min is not None else None,"sample_p01_after_mm":final_p01*1000.0 if final_p01 is not None else None,"max_vertex_move_mm":float(np.max(np.linalg.norm(V-original,axis=1))*1000.0),"final_query":"exact_chunked_reference+support_oriented_deep_negative_rescue+bisector+bounded_tail"}
    if final_min is not None and final_p01 is not None and int(_bounded_retry_budget)>0 and final_min<0.0 and final_p01>.00045:
        retry_V,retry_report=_source_covered_face_clearance(source_vertices,V,F,source_triangles,target_triangles,margin=margin,target_support_triangles=target_support_triangles,_bounded_retry_budget=int(_bounded_retry_budget)-1)
        retry_min_mm=retry_report.get('sample_min_after_mm');retry_p01_mm=retry_report.get('sample_p01_after_mm')
        retry_accept=False
        if retry_min_mm is not None and retry_p01_mm is not None:
            retry_accept=(retry_min_mm>(final_min*1000.0)+.05) and (retry_p01_mm>=((final_p01*1000.0)-.05))
        report['bounded_retry']={"attempted":True,"accepted":bool(retry_accept),"remaining_budget":int(_bounded_retry_budget)-1,"base_sample_min_after_mm":final_min*1000.0,"base_sample_p01_after_mm":final_p01*1000.0,"retry_sample_min_after_mm":retry_min_mm,"retry_sample_p01_after_mm":retry_p01_mm}
        if retry_accept:
            return retry_V,retry_report
    return V,report



def _source_relative_topology_summary(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray):
    """Small source-relative topology summary used only to accept/reject structural guards."""
    S=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if len(S)==0 or len(F)==0:return {"flip_fraction":0.0,"edge_p01":1.0,"edge_p99":1.0,"area_p01":1.0}
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
    base_len=np.linalg.norm(S[edges[:,1]]-S[edges[:,0]],axis=1);cur_len=np.linalg.norm(V[edges[:,1]]-V[edges[:,0]],axis=1);ratio=cur_len/np.maximum(base_len,1e-12)
    a0=np.cross(S[F[:,1]]-S[F[:,0]],S[F[:,2]]-S[F[:,0]]);a1=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    n0=np.linalg.norm(a0,axis=1);n1=np.linalg.norm(a1,axis=1);valid=n0>1e-12
    orient=np.ones(len(F));orient[valid]=np.einsum("ij,ij->i",a0[valid],a1[valid])/np.maximum(n0[valid]*n1[valid],1e-24)
    area=np.ones(len(F));area[valid]=n1[valid]/np.maximum(n0[valid],1e-12)
    return {"flip_fraction":float(np.mean(orient[valid]<0.0)) if np.any(valid) else 0.0,"edge_p01":float(np.percentile(ratio,1)) if len(ratio) else 1.0,"edge_p99":float(np.percentile(ratio,99)) if len(ratio) else 1.0,"area_p01":float(np.percentile(area[valid],1)) if np.any(valid) else 1.0}


def _weighted_rigid_transform(source_points: np.ndarray, target_points: np.ndarray, weights: np.ndarray):
    A=np.asarray(source_points,dtype=np.float64);B=np.asarray(target_points,dtype=np.float64);w=np.asarray(weights,dtype=np.float64).reshape(-1)
    if len(A)<3 or A.shape!=B.shape or len(w)!=len(A):raise ValueError("Rigid structural anchor requires matching point sets.")
    w=np.maximum(w,1e-12);w/=float(np.sum(w));ca=np.sum(A*w[:,None],axis=0);cb=np.sum(B*w[:,None],axis=0)
    X=(A-ca)*np.sqrt(w[:,None]);Y=(B-cb)*np.sqrt(w[:,None]);u,_,vt=np.linalg.svd(X.T@Y);R=u@vt
    if float(np.linalg.det(R))<0.0:u[:,-1]*=-1.0;R=u@vt
    return R,cb-ca@R


def _source_support_distances(source_vertices: np.ndarray, source_triangles: np.ndarray):
    if len(source_vertices)==0 or len(source_triangles)==0:return np.full(len(source_vertices),np.inf,dtype=np.float64)
    _,_,_,distance,_=_nearest_surface_reference_chunked(np.asarray(source_vertices,dtype=np.float64),np.asarray(source_triangles,dtype=np.float64),k=32)
    return np.asarray(distance,dtype=np.float64)


def _far_structural_assembly_anchor_guard(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]], cache: dict[str,Any]):
    """Stop body-field extrapolation from ballooning very large stand-off authored assemblies."""
    out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()};changed=set();reports=[]
    source_tri=cache.get("_ravafit_source_support_triangles")
    if source_tri is None:
        source_tri=_triangles_from_surface(cache["source_support_V"],cache["source_support_F"]);cache["_ravafit_source_support_triangles"]=source_tri
    for name,context in contexts.items():
        if name not in out:continue
        behavior=str(context.get("behavior") or "").casefold();features=context.get("features") or {};components=int(features.get("component_count",0));clearance=float(features.get("source_clearance_median_mm",999.0))
        data=context.get("data") or {};S=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64);V=out[name]
        if len(S)<32 or len(F)==0 or S.shape!=V.shape:continue
        extent=float(np.max(np.ptp(S,axis=0)))
        extreme_disconnected=(behavior=="conservative_component_assembly" and components>=100 and clearance>=6.0 and extent>=.65)
        extreme_standoff=(behavior=="stand_off_structured_shell" and components>=50 and clearance>=80.0 and extent>=.50)
        if not (extreme_disconnected or extreme_standoff):continue
        distance=_source_support_distances(S,source_tri);anchors=distance<=.020
        if int(np.count_nonzero(anchors))<24:
            threshold=float(np.percentile(distance,10));anchors=distance<=threshold
        if int(np.count_nonzero(anchors))<12:continue
        aw=np.exp(-np.square(distance[anchors]/.020));R,t=_weighted_rigid_transform(S[anchors],V[anchors],aw);rigid=S@R+t
        blend=np.clip((distance-.040)/.060,0.0,1.0);blend=blend*blend*(3.0-2.0*blend);candidate=V*(1.0-blend[:,None])+rigid*blend[:,None]
        before=_source_relative_topology_summary(S,V,F);after=_source_relative_topology_summary(S,candidate,F)
        if after["flip_fraction"]>before["flip_fraction"]+1e-4 or after["area_p01"]<before["area_p01"]*.92:continue
        if float(np.max(np.linalg.norm(candidate-V,axis=1)))<=1e-10:continue
        out[name]=candidate;changed.add(name);reports.append({"mesh":name,"behavior":behavior,"components":components,"source_clearance_median_mm":clearance,"extent_mm":extent*1000.0,"anchor_vertices":int(np.count_nonzero(anchors)),"far_vertices":int(np.count_nonzero(blend>.95)),"move_p95_before_mm":float(np.percentile(np.linalg.norm(V-S,axis=1),95)*1000.0),"move_p95_after_mm":float(np.percentile(np.linalg.norm(candidate-S,axis=1),95)*1000.0),"topology_before":before,"topology_after":after})
    return out,changed,{"enabled":bool(reports),"adjusted_mesh_count":len(changed),"meshes":reports,"policy":"extreme far authored structures inherit one body-near rigid anchor transform instead of extrapolated local body-field motion"}


def _deformation_roughness_p95(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray):
    S=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if len(S)==0 or len(F)==0:return 0.0
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0);D=V-S
    return float(np.percentile(np.linalg.norm(D[edges[:,0]]-D[edges[:,1]],axis=1),95)) if len(edges) else 0.0


def _smooth_mesh_displacement(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, iterations: int=20, strength: float=.40):
    S=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);D=V-S
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0);neighbours=[[] for _ in range(len(S))]
    for a,b in edges:neighbours[int(a)].append(int(b));neighbours[int(b)].append(int(a))
    X=D.copy();lam=float(np.clip(strength,0.0,.75))
    for _ in range(max(1,int(iterations))):
        nxt=X.copy()
        for i,nb in enumerate(neighbours):
            if nb:nxt[i]=(1.0-lam)*X[i]+lam*np.mean(X[nb],axis=0)
        X=nxt
    return S+X


def _long_panel_authored_shape_guard(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]], cache: dict[str,Any], margin: float=.00065):
    """Preserve the authored low-frequency curve of long attached panels after exact clearance."""
    out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()};changed=set();reports=[]
    source_tri=cache.get("_ravafit_source_support_triangles")
    if source_tri is None:source_tri=_triangles_from_surface(cache["source_support_V"],cache["source_support_F"]);cache["_ravafit_source_support_triangles"]=source_tri
    for name,context in contexts.items():
        if name not in out:continue
        behavior=str(context.get("behavior") or "").casefold();features=context.get("features") or {};components=int(features.get("component_count",0));clearance=float(features.get("source_clearance_median_mm",999.0));data=context.get("data") or {}
        S=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64);V=out[name]
        if behavior!="conservative_component_assembly" or components>2 or clearance>3.0 or len(S)<256 or len(F)==0 or S.shape!=V.shape:continue
        extent=float(np.max(np.ptp(S,axis=0)))
        if extent<.65:continue
        rough_before=_deformation_roughness_p95(S,V,F)
        if rough_before<=.0050:continue
        smoothed=_smooth_mesh_displacement(S,V,F,iterations=20,strength=.40)
        distance=_source_support_distances(S,source_tri);anchors=distance<=.006
        if int(np.count_nonzero(anchors))>=24:
            aw=np.exp(-np.square(distance[anchors]/.006));R,t=_weighted_rigid_transform(S[anchors],smoothed[anchors],aw);rigid=S@R+t
            blend=np.clip((distance-.015)/.030,0.0,1.0);blend=blend*blend*(3.0-2.0*blend);candidate=smoothed*(1.0-blend[:,None])+rigid*blend[:,None]
        else:candidate=smoothed
        before=_source_relative_topology_summary(S,V,F);candidate_topology=_source_relative_topology_summary(S,candidate,F)
        if candidate_topology["flip_fraction"]>=before["flip_fraction"] or candidate_topology["edge_p99"]>before["edge_p99"]:continue
        reclear_reports=[];current=candidate
        for pair in cache.get("slot_pairs",[]):
            SV=np.asarray(pair.get("source_literal_V",[]),dtype=np.float64);SF=np.asarray(pair.get("source_literal_F",[]),dtype=np.int64);TV=np.asarray(pair.get("target_literal_V",[]),dtype=np.float64);TF=np.asarray(pair.get("target_literal_F",[]),dtype=np.int64);TSV=np.asarray(pair.get("target_support_V",[]),dtype=np.float64);TSF=np.asarray(pair.get("target_support_F",[]),dtype=np.int64)
            if len(SV)<16 or len(SF)==0 or len(TV)<16 or len(TF)==0 or len(TSV)<16 or len(TSF)==0:continue
            current,rep=_source_covered_face_clearance(S,current,F,SV[SF],TV[TF],margin=margin,target_support_triangles=TSV[TSF]);reclear_reports.append({"slot":str(pair.get("slot") or "Body"),"face_clearance":rep})
        final_topology=_source_relative_topology_summary(S,current,F);rough_after=_deformation_roughness_p95(S,current,F)
        # Accept against untouched source authority, not against the already-damaged solved state.
        if final_topology["flip_fraction"]>max(before["flip_fraction"]*.40,.0125) or final_topology["edge_p99"]>max(before["edge_p99"]*.65,1.55):continue
        out[name]=current;changed.add(name);reports.append({"mesh":name,"extent_mm":extent*1000.0,"source_clearance_median_mm":clearance,"roughness_p95_before_mm":rough_before*1000.0,"roughness_p95_after_mm":rough_after*1000.0,"topology_before":before,"topology_candidate":candidate_topology,"topology_after":final_topology,"reclearance":reclear_reports})
    return out,changed,{"enabled":bool(reports),"adjusted_mesh_count":len(changed),"meshes":reports,"policy":"long attached panels preserve authored low-frequency curve by smoothing displacement, not source geometry, then exact source-contact reclearance"}

def _modded_source_coverage_jobs(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]], cache: dict[str,Any]):
    jobs=[]
    if not positions or bool(cache.get("dense_vanilla_source_proxy",False)):return jobs
    full_coverage_behaviors={"constructed_close_shell","body_following_flexible_layer"}
    for name,context in contexts.items():
        if name not in positions:continue
        classes=context.get("classes") or {};behavior=str(context.get("behavior") or "").casefold()
        if classes and all(str(kind).casefold()=="rigid" for kind in classes.values()):continue
        source_clearance=float((context.get("features") or {}).get("source_clearance_median_mm",999.0))
        coverage_enabled=behavior in full_coverage_behaviors and source_clearance<=4.5
        clearance_only=(not coverage_enabled) and source_clearance<=9.0
        if not coverage_enabled and not clearance_only:continue
        data=context["data"];F=np.asarray(data["F"],dtype=np.int64);srcV=np.asarray(data["V"],dtype=np.float64)
        if len(F)==0:continue
        pairs=[]
        for pair in cache.get("slot_pairs",[]):
            Xp=np.asarray(pair.get("X",[]),dtype=np.float64);Yp=np.asarray(pair.get("Y",[]),dtype=np.float64);NSp=np.asarray(pair.get("NS",[]),dtype=np.float64);NTp=np.asarray(pair.get("NT",[]),dtype=np.float64)
            SV=np.asarray(pair.get("source_literal_V",[]),dtype=np.float64);SF=np.asarray(pair.get("source_literal_F",[]),dtype=np.int64);TV=np.asarray(pair.get("target_literal_V",[]),dtype=np.float64);TF=np.asarray(pair.get("target_literal_F",[]),dtype=np.int64);TSV=np.asarray(pair.get("target_support_V",[]),dtype=np.float64);TSF=np.asarray(pair.get("target_support_F",[]),dtype=np.int64)
            if len(SV)<16 or len(SF)==0 or len(TV)<16 or len(TF)==0 or len(TSV)<16 or len(TSF)==0 or len(Xp)<16 or Xp.shape!=Yp.shape or NSp.shape!=Xp.shape or NTp.shape!=Yp.shape:continue
            pairs.append((str(pair.get("slot") or "Body"),Xp,Yp,NSp,NTp,SV,SF,TV,TF,TSV,TSF))
        source_limit=float(np.clip(source_clearance*.001*1.15,.0035,.0050))
        if pairs:jobs.append((name,srcV,F,pairs,source_limit,bool(coverage_enabled)))
    return jobs


def _modded_source_coverage_clearance_in_process(positions: dict[str,np.ndarray], jobs: list[tuple], margin: float=.00065):
    out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()};changed=set();reports=[]
    for name,srcV,F,pairs,source_limit,coverage_enabled in jobs:
        current=out[name];mesh_reports=[]
        for slot,Xp,Yp,NSp,NTp,SV,SF,TV,TF,TSV,TSF in pairs:
            if coverage_enabled:
                candidate,coverage_rep=_preserve_source_authored_body_coverage(srcV,current,F,Xp,Yp,NSp,NTp,source_limit=source_limit,margin=margin)
                if np.any(np.linalg.norm(candidate-current,axis=1)>1e-10):current=candidate;changed.add(name)
            else:
                coverage_rep={"enabled":False,"reason":"clearance-only source-contact lane preserves non-close garment geometry"}
            candidate,clearance_rep=_source_covered_face_clearance(srcV,current,F,SV[SF],TV[TF],margin=margin,target_support_triangles=TSV[TSF])
            if int(clearance_rep.get("affected_faces",0))>0:current=candidate;changed.add(name)
            mesh_reports.append({"slot":slot,"body_coverage":coverage_rep,"face_clearance":clearance_rep})
        out[name]=current;reports.append({"mesh":name,"source_limit_mm":float(source_limit*1000.0),"slots":mesh_reports})
    return out,changed,{"enabled":True,"adjusted_mesh_count":len(changed),"meshes":reports,"policy":"source-authored corresponding-body coverage + literal-target face-interior clearance; anatomy-agnostic local topology-safe pushes"}


def _modded_source_coverage_clearance_fresh(positions: dict[str,np.ndarray], jobs: list[tuple], margin: float=.00065):
    worker=_MODULE_DIR/"coverage_clearance_worker.py"
    if not worker.exists():
        out,changed,report=_modded_source_coverage_clearance_in_process(positions,jobs,margin)
        report["worker"]={"mode":"in_process_contact_face_subset_fallback","wall_sec":0.0}
        return out,changed,report
    work_dir=Path(tempfile.mkdtemp(prefix="ravafit-coverage-"));input_path=work_dir/"input.npz";meta_path=work_dir/"meta.json";output_path=work_dir/"output.npz";report_path=work_dir/"report.json";started=time.perf_counter()
    try:
        arrays={};meta={"margin":float(margin),"jobs":[]}
        for job_index,(name,srcV,F,pairs,source_limit,coverage_enabled) in enumerate(jobs):
            jid=f"j{job_index}";arrays[f"{jid}_current"]=np.asarray(positions[name],dtype=np.float64);arrays[f"{jid}_source"]=np.asarray(srcV,dtype=np.float64);arrays[f"{jid}_faces"]=np.asarray(F,dtype=np.int64)
            slot_names=[]
            for slot_index,(slot,Xp,Yp,NSp,NTp,SV,SF,TV,TF,TSV,TSF) in enumerate(pairs):
                slot_names.append(str(slot));prefix=f"{jid}_slot{slot_index}"
                arrays[f"{prefix}_x"]=np.asarray(Xp,dtype=np.float64);arrays[f"{prefix}_y"]=np.asarray(Yp,dtype=np.float64);arrays[f"{prefix}_ns"]=np.asarray(NSp,dtype=np.float64);arrays[f"{prefix}_nt"]=np.asarray(NTp,dtype=np.float64)
                arrays[f"{prefix}_source_v"]=np.asarray(SV,dtype=np.float64);arrays[f"{prefix}_source_f"]=np.asarray(SF,dtype=np.int64);arrays[f"{prefix}_target_v"]=np.asarray(TV,dtype=np.float64);arrays[f"{prefix}_target_f"]=np.asarray(TF,dtype=np.int64);arrays[f"{prefix}_target_support_v"]=np.asarray(TSV,dtype=np.float64);arrays[f"{prefix}_target_support_f"]=np.asarray(TSF,dtype=np.int64)
            meta["jobs"].append({"id":jid,"mesh":name,"source_limit":float(source_limit),"coverage_enabled":bool(coverage_enabled),"slots":slot_names})
        np.savez(input_path,**arrays);meta_path.write_text(json.dumps(meta,separators=(",",":")),encoding="utf-8")
        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0) if sys.platform.startswith("win") else 0;env=os.environ.copy();env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"});proc=subprocess.Popen([sys.executable,str(worker),str(input_path),str(meta_path),str(output_path),str(report_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=creationflags,env=env)
        try:stdout,stderr=proc.communicate(timeout=420)
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate();raise RuntimeError(f"coverage clearance worker timed out: {(stderr or stdout)[-2000:]}")
        if proc.returncode!=0 or not output_path.exists() or not report_path.exists():raise RuntimeError(f"coverage clearance worker exited {proc.returncode}: {(stderr or stdout)[-2000:]}")
        payload=json.loads(report_path.read_text(encoding="utf-8"));z=np.load(output_path,allow_pickle=False);out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()};changed=set()
        for job_index,(name,_srcV,_F,_pairs,_source_limit,_coverage_enabled) in enumerate(jobs):
            jid=f"j{job_index}";candidate=np.asarray(z[f"{jid}_current"],dtype=np.float64)
            if np.any(np.linalg.norm(candidate-out[name],axis=1)>1e-10):changed.add(name)
            out[name]=candidate
        report={"enabled":True,"adjusted_mesh_count":len(changed),"meshes":payload.get("reports",[]),"policy":"source-authored corresponding-body coverage + literal-target face-interior clearance; anatomy-agnostic local topology-safe pushes"}
        report["worker"]={"mode":"fresh_process_exact_source_coverage","wall_sec":time.perf_counter()-started,"worker_sec":payload.get("elapsed_sec")}
        return out,changed,report
    finally:
        shutil.rmtree(work_dir,ignore_errors=True)


def _modded_source_coverage_clearance_guard(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]], cache: dict[str,Any], margin: float=.00065):
    """Exact close-fit finaliser for modded garments in one fresh worker."""
    if not positions or bool(cache.get("dense_vanilla_source_proxy",False)):
        return positions,set(),{"enabled":False,"reason":"vanilla uses its dense post-collapse clearance guard"}
    jobs=_modded_source_coverage_jobs(positions,contexts,cache)
    if not jobs:return positions,set(),{"enabled":True,"adjusted_mesh_count":0,"meshes":[],"policy":"no eligible source-authored close-fit cloth"}
    try:
        return _modded_source_coverage_clearance_fresh(positions,jobs,margin)
    except Exception as ex:
        print(f"[RavaFit perf] fresh coverage clearance worker failed; using exact in-process fallback: {ex}",file=sys.stderr,flush=True)
        started=time.perf_counter();out,changed,report=_modded_source_coverage_clearance_in_process(positions,jobs,margin);report["worker"]={"mode":"in_process_contact_face_subset_fallback","wall_sec":time.perf_counter()-started,"error":str(ex)};return out,changed,report

def _preserve_authored_assembly_fresh(source_meshes: dict[str,dict[str,Any]], positions: dict[str,np.ndarray], source_tri: np.ndarray, target_tri: np.ndarray):
    """Run source-only layer inference plus cross-mesh registration in a clean process."""
    worker=_MODULE_DIR/"assembly_worker.py"
    if not worker.exists():
        p1,layered,layer_stage=_preserve_authored_garment_layers(source_meshes,positions,source_tri,target_tri);pairs={frozenset((str(r.get("inner")),str(r.get("outer")))) for r in layer_stage.get("relations",[]) if r.get("inner") and r.get("outer")};p2,cross,cross_stage=_preserve_authored_cross_mesh_assembly(source_meshes,p1,pairs);return p2,layered,layer_stage,cross,cross_stage
    work_dir=Path(tempfile.mkdtemp(prefix="ravafit-assembly-"));input_path=work_dir/"input.npz";meta_path=work_dir/"meta.json";output_path=work_dir/"output.npz";report_path=work_dir/"report.json";started=time.perf_counter()
    try:
        arrays={"source_tri":np.asarray(source_tri,dtype=np.float64),"target_tri":np.asarray(target_tri,dtype=np.float64)};rows=[]
        for i,name in enumerate(sorted(source_meshes)):
            if name not in positions:continue
            key=f"m{i}";data=source_meshes[name];arrays[f"{key}_source_v"]=np.asarray(data["V"],dtype=np.float64);arrays[f"{key}_faces"]=np.asarray(data["F"],dtype=np.int64);arrays[f"{key}_position"]=np.asarray(positions[name],dtype=np.float64);rows.append({"key":key,"name":name})
        np.savez(input_path,**arrays);meta_path.write_text(json.dumps({"meshes":rows},separators=(",",":")),encoding="utf-8")
        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0) if sys.platform.startswith("win") else 0;env=os.environ.copy();env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"});proc=subprocess.Popen([sys.executable,str(worker),str(input_path),str(meta_path),str(output_path),str(report_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=creationflags,env=env)
        try:stdout,stderr=proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate();raise RuntimeError(f"assembly worker timed out: {(stderr or stdout)[-2000:]}")
        if proc.returncode!=0 or not output_path.exists():raise RuntimeError(f"assembly worker exited {proc.returncode}: {(stderr or stdout)[-2000:]}")
        z=np.load(output_path,allow_pickle=False);payload=json.loads(report_path.read_text(encoding="utf-8"));out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()}
        for row in rows:out[row["name"]]=np.asarray(z[f'{row["key"]}_position'],dtype=np.float64)
        layer_stage=payload.get("layer_stage",{});cross_stage=payload.get("cross_stage",{});layer_stage["worker"]={"mode":"fresh_process_exact_assembly","wall_sec":time.perf_counter()-started,"worker_sec":payload.get("elapsed_sec")};cross_stage["worker"]=layer_stage["worker"]
        return out,set(payload.get("layered",[])),layer_stage,set(payload.get("cross",[])),cross_stage
    except Exception as ex:
        print(f"[RavaFit perf] fresh assembly worker failed; using exact in-process fallback: {ex}",file=sys.stderr,flush=True)
        p1,layered,layer_stage=_preserve_authored_garment_layers(source_meshes,positions,source_tri,target_tri);pairs={frozenset((str(r.get("inner")),str(r.get("outer")))) for r in layer_stage.get("relations",[]) if r.get("inner") and r.get("outer")};p2,cross,cross_stage=_preserve_authored_cross_mesh_assembly(source_meshes,p1,pairs);return p2,layered,layer_stage,cross,cross_stage
    finally:
        shutil.rmtree(work_dir,ignore_errors=True)




def _structural_component_features(vertices: np.ndarray, faces: np.ndarray):
    P=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if len(P)==0:return {"max_extent":0.0,"middle_ratio":0.0,"thin_ratio":0.0,"boundary_fraction":0.0}
    extent=np.ptp(P,axis=0);max_extent=float(np.max(extent));centred=P-P.mean(axis=0);sv=np.linalg.svd(centred,compute_uv=False)
    middle=float(sv[1]/max(float(sv[0]),1e-12)) if len(sv)>1 else 0.0;thin=float(sv[2]/max(float(sv[0]),1e-12)) if len(sv)>2 else 0.0
    boundary=0.0
    if len(F):
        edges=np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1);unique,count=np.unique(edges,axis=0,return_counts=True);be=unique[count==1]
        boundary=float(len(np.unique(be))/max(len(P),1)) if len(be) else 0.0
    return {"max_extent":max_extent,"middle_ratio":middle,"thin_ratio":thin,"boundary_fraction":boundary}


def _structural_laplacian_regularize(source_vertices: np.ndarray, target_vertices: np.ndarray, faces: np.ndarray, strength: float):
    """ARAP-style source differential reconstruction around the supplied macro target.

    The old Runtime-8 implementation preserved differentials from one global affine fit.  That was
    topology-safe but could flatten locally authored shape whenever the target body's deformation field
    varied across the garment (breasts/hips/belly/shoulders are the obvious cases).  Here the B14/support
    result remains a soft positional authority while source edge differentials are reconstructed through
    locally fitted rotations.  No garment/body/anatomy names participate.
    """
    P=np.asarray(source_vertices,dtype=np.float64);Q=np.asarray(target_vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if len(P)<4 or len(F)==0 or P.shape!=Q.shape:return Q.copy()
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
    if len(edges)<3:return Q.copy()
    a=edges[:,0].astype(np.int64);b=edges[:,1].astype(np.int64);n=len(P)
    degree=np.bincount(np.concatenate((a,b)),minlength=n).astype(np.float64)
    # Keep the macro target authoritative.  `strength` only tunes how much local source structure survives.
    structure_weight=float(np.clip(float(strength)/12.0,.70,2.60));anchor_weight=4.0
    rows=np.concatenate((np.arange(n),a,b));cols=np.concatenate((np.arange(n),b,a));vals=np.concatenate((degree,-np.ones(len(a)),-np.ones(len(a))))
    L=csr_matrix((vals,(rows,cols)),shape=(n,n));A=(structure_weight*L+anchor_weight*eye(n,format="csr")).tocsr();X=Q.copy();pedge=P[a]-P[b]
    for _ in range(2):
        xedge=X[a]-X[b]
        covariance=np.zeros((n,3,3),dtype=np.float64);outer=np.einsum("ni,nj->nij",xedge,pedge)
        np.add.at(covariance,a,outer);np.add.at(covariance,b,outer)
        try:
            u,_,vt=np.linalg.svd(covariance,full_matrices=False);rotation=np.einsum("nij,njk->nik",u,vt);det=np.linalg.det(rotation);bad=det<0.0
            if np.any(bad):
                u2=u[bad].copy();u2[:,:,-1]*=-1.0;rotation[bad]=np.einsum("nij,njk->nik",u2,vt[bad])
        except np.linalg.LinAlgError:
            return Q.copy()
        edge_rotation=.5*(rotation[a]+rotation[b]);goal=np.einsum("nij,nj->ni",edge_rotation,pedge);rhs_edge=np.zeros((n,3),dtype=np.float64);np.add.at(rhs_edge,a,goal);np.add.at(rhs_edge,b,-goal)
        rhs=structure_weight*rhs_edge+anchor_weight*Q
        try:X=np.column_stack([spsolve(A,rhs[:,axis]) for axis in range(3)])
        except Exception:return Q.copy()
        if not np.all(np.isfinite(X)):return Q.copy()
    # Never accept a structural reconstruction that worsens source-relative topology/edge pathology.
    accepted,_,_,_=_coupled_topology_safe_alpha(P,Q,X,F)
    return np.asarray(accepted,dtype=np.float64)


def _garment_support_proxy(cache: dict[str,Any]):
    """Build a source-relative multi-scale target support surface for clothing.

    The selected target body owns broad anatomy and silhouette.  Only target-only high-frequency relief
    is attenuated, with an intermediate band kept mostly target-authoritative.  This prevents small local
    anatomy from embossing through close garments without flattening centimetre-scale hips/glutes/bust.
    Literal target geometry remains absolute collision authority elsewhere in the solver.
    """
    cached=cache.get("_ravafit_garment_support_proxy")
    if isinstance(cached,dict) and "V" in cached:return cached
    TV=np.asarray(cache.get("target_support_V"),dtype=np.float64).copy();TF=np.asarray(cache.get("target_support_F"),dtype=np.int64)
    SV=np.asarray(cache.get("source_support_V"),dtype=np.float64);SF=np.asarray(cache.get("source_support_F"),dtype=np.int64)
    if len(TV)==0 or len(TF)==0:
        result={"V":TV,"F":TF.copy(),"report":{"enabled":False,"reason":"missing target support"}};cache["_ravafit_garment_support_proxy"]=result;return result

    raw_edges=np.sort(np.vstack((TF[:,[0,1]],TF[:,[1,2]],TF[:,[2,0]])),axis=1)
    edges=np.unique(raw_edges,axis=0);ea,eb=edges[:,0],edges[:,1]
    deg=np.bincount(np.concatenate((ea,eb)),minlength=len(TV)).astype(np.float64)
    unique_edges,edge_counts=np.unique(raw_edges,axis=0,return_counts=True)
    boundary=np.unique(unique_edges[edge_counts==1]) if np.any(edge_counts==1) else np.empty(0,dtype=np.int64)
    edge_len=np.linalg.norm(TV[ea]-TV[eb],axis=1);median_edge=float(np.median(edge_len[edge_len>1e-8])) if np.any(edge_len>1e-8) else .0025

    def lowpass(V: np.ndarray, iterations: int):
        cur=np.asarray(V,dtype=np.float64).copy()
        for _ in range(max(1,int(iterations))):
            sums=np.zeros_like(cur);np.add.at(sums,ea,cur[eb]);np.add.at(sums,eb,cur[ea]);avg=sums/np.maximum(deg[:,None],1.0);cur=.74*cur+.26*avg
        return cur

    # Diffusion radius grows roughly with sqrt(iterations)*edge length.  Choose bands from physical scale,
    # not anatomy names, so this remains generic across garments and bodies.
    micro_iterations=int(np.clip(round((.006/max(median_edge,1e-5))**2),2,7))
    broad_iterations=int(np.clip(round((.022/max(median_edge,1e-5))**2),14,42))
    target_micro_lp=lowpass(TV,micro_iterations);target_broad=lowpass(TV,broad_iterations)
    target_micro=TV-target_micro_lp;target_meso=target_micro_lp-target_broad
    mode="target-multiscale-fallback";micro_alpha=np.ones(len(TV),dtype=np.float64);meso_alpha=np.ones(len(TV),dtype=np.float64)
    if len(SV)==len(TV) and SF.shape==TF.shape and np.array_equal(SF,TF):
        source_micro_lp=lowpass(SV,micro_iterations);source_broad=lowpass(SV,broad_iterations)
        source_micro=SV-source_micro_lp;source_meso=source_micro_lp-source_broad
        sm=np.linalg.norm(source_micro,axis=1);tm=np.linalg.norm(target_micro,axis=1)
        ss=np.linalg.norm(source_meso,axis=1);ts=np.linalg.norm(target_meso,axis=1)
        micro_alpha=np.clip((sm+.00016)/np.maximum(tm+.00016,1e-12),.12,1.0)
        # Meso curvature carries broad clothing silhouette: retain at least 88% target authority.
        meso_alpha=np.clip((ss+.00045)/np.maximum(ts+.00045,1e-12),.88,1.0)
        for _ in range(3):
            sums=np.zeros(len(micro_alpha),dtype=np.float64);np.add.at(sums,ea,micro_alpha[eb]);np.add.at(sums,eb,micro_alpha[ea]);micro_alpha=.72*micro_alpha+.28*sums/np.maximum(deg,1.0)
        for _ in range(2):
            sums=np.zeros(len(meso_alpha),dtype=np.float64);np.add.at(sums,ea,meso_alpha[eb]);np.add.at(sums,eb,meso_alpha[ea]);meso_alpha=.76*meso_alpha+.24*sums/np.maximum(deg,1.0)
        out=target_broad+target_meso*meso_alpha[:,None]+target_micro*micro_alpha[:,None];mode="source-relative-multiscale-relief"
    else:
        out=target_broad+target_meso+target_micro*.70

    delta=out-TV;mag=np.linalg.norm(delta,axis=1);cap=.0038;delta*=np.minimum(1.0,cap/np.maximum(mag,1e-12))[:,None]
    if len(boundary):delta[boundary]*=.20
    out=TV+delta;move=np.linalg.norm(delta,axis=1)
    report={"enabled":True,"mode":mode,"median_edge_mm":median_edge*1000.0,"micro_iterations":micro_iterations,"broad_iterations":broad_iterations,
        "move_p95_mm":float(np.percentile(move,95)*1000.0),"move_max_mm":float(np.max(move,initial=0.0)*1000.0),
        "micro_authority_p05":float(np.percentile(micro_alpha,5)) if len(micro_alpha) else 1.0,"micro_authority_p50":float(np.percentile(micro_alpha,50)) if len(micro_alpha) else 1.0,
        "meso_authority_p05":float(np.percentile(meso_alpha,5)) if len(meso_alpha) else 1.0,"meso_authority_p50":float(np.percentile(meso_alpha,50)) if len(meso_alpha) else 1.0,
        "boundary_vertices":int(len(boundary)),"policy":"physical-scale support: target broad/macro silhouette preserved, meso curvature mostly target-authoritative, only target-only micro relief attenuated; literal body remains collision authority"}
    result={"V":out,"F":TF.copy(),"report":report};cache["_ravafit_garment_support_proxy"]=result;return result

def _structural_support_frame_target(points: np.ndarray, cache: dict[str,Any]):
    P=np.asarray(points,dtype=np.float64);SV=np.asarray(cache.get("source_support_V"),dtype=np.float64);SF=np.asarray(cache.get("source_support_F"),dtype=np.int64);proxy=_garment_support_proxy(cache);TV=np.asarray(proxy.get("V"),dtype=np.float64);TF=np.asarray(proxy.get("F"),dtype=np.int64)
    if len(P)==0 or len(SV)==0 or len(SF)==0 or len(TV)==0 or len(TF)==0 or len(SF)!=len(TF):return None
    source_tri=cache.get("_ravafit_structural_source_support_triangles")
    if source_tri is None:source_tri=SV[SF];cache["_ravafit_structural_source_support_triangles"]=source_tri
    target_tri=cache.get("_ravafit_structural_target_support_triangles")
    if target_tri is None:target_tri=TV[TF];cache["_ravafit_structural_target_support_triangles"]=target_tri
    closest,_,signed,distance,face_index=_b14_nearest_surface(P,source_tri,k=48)
    source_face=source_tri[face_index];target_face=target_tri[face_index];bary=trimesh.triangles.points_to_barycentric(source_face,closest);contact=np.einsum("ni,nij->nj",bary,target_face)
    normal=np.cross(target_face[:,1]-target_face[:,0],target_face[:,2]-target_face[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    stand=np.clip(np.maximum(np.asarray(signed,dtype=np.float64),.00045),.00045,.020)
    return contact+stand[:,None]*normal,np.asarray(distance,dtype=np.float64),np.asarray(signed,dtype=np.float64)


def _structural_transform_offset(source_triangle: np.ndarray, target_triangle: np.ndarray, offset: np.ndarray):
    S=np.asarray(source_triangle,dtype=np.float64);T=np.asarray(target_triangle,dtype=np.float64);v=np.asarray(offset,dtype=np.float64)
    se1=S[1]-S[0];se2=S[2]-S[0];sn=np.cross(se1,se2);sn/=max(float(np.linalg.norm(sn)),1e-12)
    te1=T[1]-T[0];te2=T[2]-T[0];tn=np.cross(te1,te2);tn/=max(float(np.linalg.norm(tn)),1e-12)
    A0=np.column_stack((se1,se2,sn));A1=np.column_stack((te1,te2,tn))
    try:M=A1@np.linalg.inv(A0)
    except np.linalg.LinAlgError:return v.copy()
    u,sv,vt=np.linalg.svd(M);sv=np.clip(sv,.70,1.40);M=u@np.diag(sv)@vt
    return M@v


def _structural_fit_similarity_from_anchors(source_anchor: np.ndarray, target_anchor: np.ndarray, points: np.ndarray, scale_min: float=.75, scale_max: float=1.30):
    A=np.asarray(source_anchor,dtype=np.float64);B=np.asarray(target_anchor,dtype=np.float64);P=np.asarray(points,dtype=np.float64)
    if len(A)<3:return P.copy(),1.0
    ca=A.mean(axis=0);cb=B.mean(axis=0);AA=A-ca;BB=B-cb;u,_,vt=np.linalg.svd(AA.T@BB);R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    Ar=AA@R.T;scale=float(np.sum(Ar*BB)/max(float(np.sum(Ar*Ar)),1e-12));scale=float(np.clip(scale,scale_min,scale_max))
    return scale*(P-ca)@R.T+cb,scale


def _structural_find_attachment(mesh_name: str, ids: np.ndarray, source_meshes: dict[str,dict[str,Any]], candidate: dict[str,np.ndarray], source_triangles: dict[str,np.ndarray] | None=None):
    source=np.asarray(source_meshes[mesh_name]["V"],dtype=np.float64);P=source[ids];probe=P[::max(1,len(P)//96)];best=None
    # Thousands of authored studs/rings may each ask the same few garment surfaces for an attachment.
    # Reusing the exact immutable triangle arrays lets _reference_surface_index reuse its cKDTree/normals
    # instead of rebuilding a full acceleration structure once per tiny component.  Query mathematics
    # and component-local decisions remain unchanged.
    triangles=source_triangles or {}
    for other,row in source_meshes.items():
        if other==mesh_name or other not in candidate:continue
        SV=np.asarray(row["V"],dtype=np.float64);SF=np.asarray(row["F"],dtype=np.int64)
        if len(SV)<3 or len(SF)==0:continue
        tri=triangles.get(other)
        if tri is None:tri=SV[SF]
        _,_,_,distance,_=_nearest_surface_reference_chunked(probe,tri,k=32);median=float(np.median(distance));p10=float(np.percentile(distance,10));score=.70*p10+.30*median
        if best is None or score<best[0]:best=(score,median,p10,other)
    if best is None or best[2]>.012:return None
    support=best[3];SV=np.asarray(source_meshes[support]["V"],dtype=np.float64);SF=np.asarray(source_meshes[support]["F"],dtype=np.int64);tri=triangles.get(support)
    if tri is None:tri=SV[SF]
    closest,normal,signed,distance,face_index=_nearest_surface_reference_chunked(P,tri,k=32);threshold=min(.010,float(np.percentile(distance,20))+.0025);anchors=np.flatnonzero(distance<=threshold)
    if len(anchors)<3:anchors=np.argsort(distance)[:min(12,len(P))]
    target_v=np.asarray(candidate[support],dtype=np.float64);source_anchor=[];target_anchor=[]
    for ai in anchors:
        face=int(face_index[ai]);st=tri[face];tt=target_v[SF[face]];cp=closest[ai];bary=trimesh.triangles.points_to_barycentric(st[None,:,:],cp[None,:])[0];tc=bary@tt;off=P[ai]-cp
        source_anchor.append(P[ai]);target_anchor.append(tc+_structural_transform_offset(st,tt,off))
    side=float(np.median(np.einsum("ij,ij->i",P-closest,normal)))
    return {"support":support,"source_anchor":np.asarray(source_anchor),"target_anchor":np.asarray(target_anchor),"median_gap":best[1],"p10_gap":best[2],"source_side":1.0 if side>=0.0 else -1.0}


def _structural_oriented_clearance(vertices: np.ndarray, faces: np.ndarray, source_vertices: np.ndarray, target_literal_tri: np.ndarray, target_support_tri: np.ndarray, rigid: bool, margin: float=.00070):
    V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64);S=np.asarray(source_vertices,dtype=np.float64);literal=np.asarray(target_literal_tri,dtype=np.float64);support=np.asarray(target_support_tri,dtype=np.float64)
    if len(V)==0 or len(literal)==0:return V,{"adjusted":False,"iterations":0}
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0) if len(F) else np.empty((0,2),dtype=np.int64);neighbours=[[] for _ in range(len(V))]
    for a,b in edges:neighbours[int(a)].append(int(b));neighbours[int(b)].append(int(a))
    original=V.copy();rejections=0;history=[]
    for iteration in range(6):
        sample_parts=[V];owners=[np.arange(len(V),dtype=np.int64)[:,None]];weights=[np.ones((len(V),1),dtype=np.float64)]
        if len(F):
            cent=V[F].mean(axis=1);sample_parts.append(cent);owners.append(F);weights.append(np.full((len(F),3),1.0/3.0,dtype=np.float64))
        samples=np.vstack(sample_parts);cp,normals,signed,_,_,rej=_nearest_literal_surface_consistent_with_support(samples,literal,support,exact_base=True,deep_negative=-.00035);rejections+=int(rej);need=np.maximum(float(margin)-signed,0.0);bad=need>1e-6
        if not np.any(bad):break
        history.append({"iteration":iteration+1,"min_mm":float(np.min(signed)*1000.0),"p01_mm":float(np.percentile(signed,1)*1000.0),"bad_samples":int(np.count_nonzero(bad))})
        if rigid:
            nn=np.asarray(normals[bad],dtype=np.float64);ww=need[bad];anchor=nn[int(np.argmax(ww))].copy();flip=np.einsum("ij,j->i",nn,anchor)<0.0;nn[flip]*=-1.0;direction=np.sum(nn*ww[:,None],axis=0);dn=float(np.linalg.norm(direction))
            if dn<1e-12:break
            direction/=dn;projection=nn@direction;positive=projection>.10
            if not np.any(positive):break
            step=float(min(.0035,np.max(ww[positive]/np.maximum(projection[positive],.10))+.00005));field=np.broadcast_to(direction*step,V.shape).copy()
        else:
            field=np.zeros_like(V);mass=np.zeros(len(V),dtype=np.float64);offset=0
            vertex_count=len(V);vb=bad[:vertex_count]
            if np.any(vb):field[vb]+=normals[:vertex_count][vb]*need[:vertex_count][vb,None];mass[vb]+=1.0
            if len(F):
                fb=bad[vertex_count:]
                rows=np.flatnonzero(fb)
                if len(rows):
                    contrib=normals[vertex_count:][rows]*need[vertex_count:][rows,None]
                    for corner in range(3):np.add.at(field,F[rows,corner],contrib/3.0);np.add.at(mass,F[rows,corner],1.0/3.0)
            active=mass>0.0;field[active]/=mass[active,None]
            for _ in range(5):
                avg=np.zeros_like(field)
                for i,ns in enumerate(neighbours):
                    if ns:avg[i]=np.mean(field[ns],axis=0)
                grown=np.linalg.norm(avg,axis=1)>1e-8;field=np.where(active[:,None],.82*field+.18*avg,.12*avg);active|=grown
            mag=np.linalg.norm(field,axis=1);field*=np.minimum(1.0,.00150/np.maximum(mag,1e-12))[:,None]
        accepted=None
        for alpha in (1.0,.80,.60,.40,.25,.12,.06):
            trial=V+float(alpha)*field
            if len(F):
                q=_component_topology_quality(S,trial,F);base=_component_topology_quality(S,V,F)
                if q["flip_fraction"]>max(.002,base["flip_fraction"]+.0005) or q["orientation_p01"]<min(.02,base["orientation_p01"]-.03) or q["area_ratio_p01"]<min(.08,base["area_ratio_p01"]*.75):continue
            accepted=trial;break
        if accepted is None:break
        V=accepted
    sample_parts=[V]
    if len(F):sample_parts.append(V[F].mean(axis=1))
    samples=np.vstack(sample_parts);_,_,signed,_,_,rej=_nearest_literal_surface_consistent_with_support(samples,literal,support,exact_base=True,deep_negative=-.00035);rejections+=int(rej)
    move=np.linalg.norm(V-original,axis=1)
    return V,{"adjusted":bool(np.any(move>1e-7)),"iterations":len(history),"history":history,"move_p95_mm":float(np.percentile(move,95)*1000.0) if len(move) else 0.0,"move_max_mm":float(np.max(move)*1000.0) if len(move) else 0.0,"sample_min_mm":float(np.min(signed)*1000.0) if len(signed) else None,"sample_p01_mm":float(np.percentile(signed,1)*1000.0) if len(signed) else None,"opposite_facing_literal_rejections":rejections}


def _finalize_modded_structural_solution(source: Any, cache: dict[str,Any], positions: dict[str,np.ndarray], skinning: dict[str,dict[str,Any]], records: dict[str,Any], contexts: dict[str,dict[str,Any]], local_affine_quality_rms_mm: float):
    """Preserve source-authored garment structure around B14's target-body macro fit.

    B14 remains body-transfer authority. This stage only reconstructs local garment structure,
    carries rigid attachments with their authored garment support, and applies outward-only collision safety.
    """
    source_meshes={name:{"V":np.asarray(contexts[name]["data"]["V"],dtype=np.float64),"F":np.asarray(contexts[name]["data"]["F"],dtype=np.int64)} for name in positions}
    # Immutable source surfaces are shared by every component attachment query in this conversion.
    # Keep one triangle object per mesh so the exact nearest-surface accelerator is actually reusable.
    source_triangles={name:row["V"][row["F"]] for name,row in source_meshes.items() if len(row["F"])}
    candidate={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};component_rows={};detail_meshes=set();classification_counts={"rigid-detail":0,"structured-strip":0,"support-frame-cloth":0,"cloth-shell":0}
    for name in sorted(candidate):
        S=source_meshes[name]["V"];F=source_meshes[name]["F"];R=np.asarray(positions[name],dtype=np.float64);components=_masked_vertex_components(F,np.ones(len(S),dtype=bool));rows=[];rigid_count=0
        for ci,ids in enumerate(components):
            LF=_component_local_faces(F,ids,len(S));P=S[ids];Q=R[ids];feature=_structural_component_features(P,LF);maxe=feature["max_extent"];middle=feature["middle_ratio"];thin=feature["thin_ratio"];boundary=feature["boundary_fraction"]
            rigid=(len(ids)<=650 and maxe<=.055) or (len(ids)<=220 and maxe<=.090 and thin<=.18);strip=(not rigid and maxe<=.34 and (thin<=.10 or boundary>=.28) and middle<=.45)
            if rigid:
                X,_=_similarity_fit_points(P,Q,scale_min=.72,scale_max=1.35);kind="rigid-detail";rigid_count+=1
            elif strip:
                X=_structural_laplacian_regularize(P,Q,LF,20.0);kind="structured-strip"
            else:
                frame=_structural_support_frame_target(P,cache);close=frame is not None and float(np.median(frame[1]))<=.018
                if close:
                    data_target=.10*Q+.90*frame[0];X=_structural_laplacian_regularize(P,data_target,LF,32.0 if maxe<.38 else 24.0);kind="support-frame-cloth"
                else:
                    X=_structural_laplacian_regularize(P,Q,LF,10.0);kind="cloth-shell"
            candidate[name][ids]=X;classification_counts[kind]+=1;rows.append({"component":ci,"ids":ids,"faces":LF,"kind":kind,"features":feature})
        component_rows[name]=rows
        if len(components)>=12 and rigid_count>=8 and rigid_count>=int(np.ceil(.35*len(components))):detail_meshes.add(name)

    attachment_reports=[]
    for name in sorted(detail_meshes):
        for row in component_rows[name]:
            if row["kind"]!="rigid-detail":continue
            ids=row["ids"];attachment=_structural_find_attachment(name,ids,source_meshes,candidate,source_triangles)
            if attachment is None:continue
            P=source_meshes[name]["V"][ids];X,scale=_structural_fit_similarity_from_anchors(attachment["source_anchor"],attachment["target_anchor"],P);candidate[name][ids]=X
            # Preserve source stand-off from the garment support itself; this is a rigid translation only.
            source_p10=float(attachment["p10_gap"]);before_gap=None;after_gap=None
            if source_p10>.00020:
                support=attachment["support"];SF=source_meshes[support]["F"];support_tri=np.asarray(candidate[support],dtype=np.float64)[SF];current=candidate[name][ids].copy();desired=max(.00035,source_p10*.85)
                for _ in range(3):
                    _,normal,_,distance,_=_nearest_surface_reference_chunked(current,support_tri,k=32);p10=float(np.percentile(distance,10));before_gap=p10 if before_gap is None else before_gap;deficit=desired-p10
                    if deficit<=.00003:break
                    near=distance<=np.percentile(distance,35);direction=np.mean(normal[near]*float(attachment["source_side"]),axis=0);dn=float(np.linalg.norm(direction))
                    if dn<1e-12:break
                    current=current+(direction/dn)*min(.0025,deficit+.00005)
                candidate[name][ids]=current;_,_,_,distance,_=_nearest_surface_reference_chunked(current,support_tri,k=32);after_gap=float(np.percentile(distance,10))
            attachment_reports.append({"mesh":name,"component":int(row["component"]),"support":attachment["support"],"vertices":int(len(ids)),"source_p10_gap_mm":source_p10*1000.0,"before_p10_gap_mm":None if before_gap is None else before_gap*1000.0,"after_p10_gap_mm":None if after_gap is None else after_gap*1000.0,"scale":float(scale)})

    target_support_tri=np.asarray(cache.get("_ravafit_target_support_triangles"),dtype=np.float64) if cache.get("_ravafit_target_support_triangles") is not None else _triangles_from_surface(cache["target_support_V"],cache["target_support_F"])
    target_literal_tri=_target_fit_collision_triangles(cache)
    clearance_reports=[];changed=set()
    for name in sorted(candidate):
        S=source_meshes[name]["V"];F=source_meshes[name]["F"];C=candidate[name].copy()
        for row in component_rows[name]:
            ids=row["ids"];LF=row["faces"];X,report=_structural_oriented_clearance(C[ids],LF,S[ids],target_literal_tri,target_support_tri,row["kind"]=="rigid-detail",margin=.00070);C[ids]=X
            if report.get("adjusted"):changed.add(name)
            clearance_reports.append({"mesh":name,"component":int(row["component"]),"kind":row["kind"],**report})
        candidate[name]=C

    # Any structurally reconstructed mesh gets authoritative target-body skinning recomputed from final geometry.
    for name in sorted(candidate):
        context=contexts[name];data=context["data"];w=context["w"];retargeted,stage=_retarget_garment_skinning(candidate[name],data["V"],data["W"],data["joint_names"],cache,context["behavior"],context["effective_behavior"],context["labels"],context["classes"],w["raw_to_weld"]);_assert_finite_stage(name,"structural finalizer skinning retarget",retargeted);skinning[name]={"weights":retargeted,"joint_names":list(data["joint_names"]),"stage":stage};records.setdefault(name,{})["skinning"]=stage;records[name]["structural_finalizer_applied"]=True

    unresolved=[r for r in clearance_reports if r.get("sample_p01_mm") is not None and float(r["sample_p01_mm"])<-.10]
    return candidate,skinning,records,{"local_affine_quality_rms_mm":float(local_affine_quality_rms_mm),"garment_mesh_count":len(candidate),"skinning_retargeted_mesh_count":len(skinning),"structural_finalizer":{"enabled":True,"policy":"B14 macro fit + source differential shape + smoothed support-frame close cloth + source-relative rigid attachments + outward-only literal collision veto","classification_counts":classification_counts,"detail_meshes":sorted(detail_meshes),"attachment_count":len(attachment_reports),"attachments":attachment_reports,"clearance":clearance_reports,"unresolved_p01_collision_components":len(unresolved)},"authored_layer_relations":0,"authored_layer_adjusted_meshes":0,"authored_layer_report":{"enabled":False,"reason":"superseded by structural finalizer"},"authored_cross_mesh_relations":0,"authored_cross_mesh_adjusted_meshes":0,"authored_cross_mesh_report":{"enabled":False,"reason":"superseded by source-relative attachment frames"},"authored_split_seams":{"enabled":False,"reason":"source differential reconstruction preserves authored topology"},"far_authored_structural_anchor":{"enabled":False,"reason":"superseded by structural component solve"},"authored_long_panel_shape_guard":{"enabled":False,"reason":"superseded by structural component solve"},"authored_ribbon_topology":{"enabled":False,"reason":"superseded by rigid/strip structural classes"},"authored_narrow_strip_shape_guard":{"enabled":False,"reason":"superseded by structured-strip solve"},"modded_source_coverage_clearance":{"enabled":False,"reason":"superseded by outward-only structural collision veto"},"dense_vanilla_expansion_clearance":{"enabled":False,"reason":"modded structural finalizer"},"body_support_proxy":{"enabled":True,"mode":"smoothed support-frame cloth with literal-body collision veto","slots":cache.get("slot_stats",[])},"source_body_suppression":_public_source_body_suppression(cache.get("_ravafit_source_body_suppression"))}

def _coupled_support_frame(points: np.ndarray, cache: dict[str,Any], target_vertices: np.ndarray | None=None):
    """Map arbitrary source-space points through the exact source->target support correspondence."""
    P=np.asarray(points,dtype=np.float64);SV=np.asarray(cache.get("source_support_V"),dtype=np.float64);SF=np.asarray(cache.get("source_support_F"),dtype=np.int64);proxy=_garment_support_proxy(cache);TV=np.asarray(proxy.get("V") if target_vertices is None else target_vertices,dtype=np.float64);TF=np.asarray(proxy.get("F"),dtype=np.int64)
    if len(P)==0 or len(SV)==0 or len(SF)==0 or len(TV)==0 or len(TF)==0 or len(SF)!=len(TF):return None
    source_tri=SV[SF];target_tri=TV[TF];closest,source_normal,_,distance,face_index=_b14_nearest_surface(P,source_tri,k=48);source_face=source_tri[face_index];target_face=target_tri[face_index];bary=trimesh.triangles.points_to_barycentric(source_face,closest);contact=np.einsum("ni,nij->nj",bary,target_face);normal=np.cross(target_face[:,1]-target_face[:,0],target_face[:,2]-target_face[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    # Preserve hemisphere orientation from the corresponding source face instead of trusting an unrelated nearest target face.
    sf_normal=np.cross(source_face[:,1]-source_face[:,0],source_face[:,2]-source_face[:,0]);sf_normal/=np.maximum(np.linalg.norm(sf_normal,axis=1,keepdims=True),1e-12);same=np.einsum("ij,ij->i",sf_normal,source_normal)>=0.0;normal[~same]*=-1.0
    return contact,normal,np.asarray(distance,dtype=np.float64),np.asarray(face_index,dtype=np.int64)


def _build_auxiliary_envelope_support(cache: dict[str,Any]):
    """Create a broad support-frame envelope around target-body auxiliary geometry.

    The envelope is deliberately wider than the literal accessory silhouette.  Aux geometry is therefore
    a collision veto (cloth must bridge over it) rather than a sculpting tool (cloth must not emboss it).
    """
    cached=cache.get("_ravafit_auxiliary_envelope_support")
    if isinstance(cached,dict) and "V" in cached:return cached
    proxy=_garment_support_proxy(cache);TV=np.asarray(proxy.get("V"),dtype=np.float64);TF=np.asarray(proxy.get("F"),dtype=np.int64)
    if len(TV)==0 or len(TF)==0:
        result={"V":TV.copy(),"F":TF.copy(),"height":np.zeros(len(TV)),"report":{"enabled":False,"reason":"missing target support"}};cache["_ravafit_auxiliary_envelope_support"]=result;return result
    normals=_vertex_normals_from_faces(TV,TF);tri=TV[TF];tree=cKDTree(TV);height=np.zeros(len(TV),dtype=np.float64);seed=np.zeros(len(TV),dtype=np.float64);components=cache.get("_ravafit_target_auxiliary_components") or [];rows=[]
    for row in components:
        P=np.asarray(row.get("V",[]),dtype=np.float64)
        if len(P)<3:continue
        cp,n,signed,distance,_=_nearest_surface_reference_chunked(P,tri,k=48);positive=np.maximum(np.asarray(signed,dtype=np.float64),0.0);effective=np.maximum(positive,.40*np.asarray(distance,dtype=np.float64));stride=max(1,len(P)//160);used=0;max_h=0.0
        for qi in range(0,len(P),stride):
            h=float(np.clip(effective[qi]+.00070,.00075,.00650))
            if effective[qi]<.00010:continue
            sigma=float(np.clip(max(.0070,2.2*h),.0070,.0140));ids=np.asarray(tree.query_ball_point(cp[qi],r=3.0*sigma),dtype=np.int64)
            if not len(ids):continue
            align=np.einsum("ij,j->i",normals[ids],n[qi]);ids=ids[align>.15]
            if not len(ids):continue
            dist=np.linalg.norm(TV[ids]-cp[qi],axis=1);bump=h*np.exp(-.5*np.square(dist/sigma));seed[ids]=np.maximum(seed[ids],bump);used+=1;max_h=max(max_h,h)
        if used:rows.append({"slot":row.get("slot"),"mesh":row.get("mesh"),"component":row.get("component"),"samples":used,"max_height_mm":max_h*1000.0})
    height=seed.copy()
    if np.any(height>0.0):
        edges=np.unique(np.sort(np.vstack((TF[:,[0,1]],TF[:,[1,2]],TF[:,[2,0]])),axis=1),axis=0);ea,eb=edges[:,0],edges[:,1];deg=np.bincount(np.concatenate((ea,eb)),minlength=len(TV)).astype(np.float64)
        for _ in range(5):
            sums=np.zeros(len(TV),dtype=np.float64);np.add.at(sums,ea,height[eb]);np.add.at(sums,eb,height[ea]);avg=sums/np.maximum(deg,1.0);height=np.maximum(.90*seed,.68*height+.32*avg)
    V=TV+normals*height[:,None];report={"enabled":bool(rows),"auxiliary_components":len(rows),"support_vertices_raised":int(np.count_nonzero(height>.00005)),"height_p95_mm":float(np.percentile(height,95)*1000.0) if len(height) else 0.0,"height_max_mm":float(np.max(height,initial=0.0)*1000.0),"components":rows,"policy":"broad smoothed body-relative auxiliary envelope; auxiliaries veto occupancy without imprinting their literal silhouette"};result={"V":V,"F":TF.copy(),"height":height,"report":report};cache["_ravafit_auxiliary_envelope_support"]=result;return result


def _coupled_edge_metric(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray):
    S=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    if len(F)==0:return {"min":1.0,"p001":1.0,"p01":1.0,"p99":1.0,"p999":1.0,"max":1.0}
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0);a=np.linalg.norm(S[edges[:,1]]-S[edges[:,0]],axis=1);b=np.linalg.norm(V[edges[:,1]]-V[edges[:,0]],axis=1);valid=a>.00015;ratio=b[valid]/np.maximum(a[valid],1e-12)
    if not len(ratio):return {"min":1.0,"p001":1.0,"p01":1.0,"p99":1.0,"p999":1.0,"max":1.0}
    return {"min":float(np.min(ratio)),"p001":float(np.percentile(ratio,.1)),"p01":float(np.percentile(ratio,1)),"p99":float(np.percentile(ratio,99)),"p999":float(np.percentile(ratio,99.9)),"max":float(np.max(ratio))}


def _coupled_topology_safe_alpha(source_vertices: np.ndarray, before: np.ndarray, proposed: np.ndarray, faces: np.ndarray):
    S=np.asarray(source_vertices,dtype=np.float64);B=np.asarray(before,dtype=np.float64);P=np.asarray(proposed,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);base=_source_relative_topology_summary(S,B,F);base_edge=_coupled_edge_metric(S,B,F)
    for alpha in (1.0,.80,.60,.45,.30,.20,.12,.07,.04):
        C=B+alpha*(P-B);m=_source_relative_topology_summary(S,C,F);e=_coupled_edge_metric(S,C,F)
        flip_limit=max(.0125,base["flip_fraction"]+.0025);p99_limit=max(1.90,base_edge["p99"]*1.05);p999_limit=max(2.80,base_edge["p999"]*1.08);max_limit=max(3.25,base_edge["max"]*1.02);p01_floor=min(.42,base_edge["p01"]*.92);p001_floor=min(.25,base_edge["p001"]*.90);min_floor=min(.12,base_edge["min"]*.85);area_floor=min(.12,base["area_p01"]*.70)
        if (m["flip_fraction"]<=flip_limit and e["p99"]<=p99_limit and e["p999"]<=p999_limit and e["max"]<=max_limit
                and e["p01"]>=p01_floor and e["p001"]>=p001_floor and e["min"]>=min_floor and m["area_p01"]>=area_floor):return C,float(alpha),m,e
    return B.copy(),0.0,base,base_edge


def _coupled_component_clearance(source_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray, cache: dict[str,Any], env: dict[str,Any], margin: float=.00065, precomputed: dict[str,np.ndarray] | None=None, enforce_support: bool=True):
    """Clear one authored component with broad patches and a literal-body hemisphere veto.

    Source->target support correspondence is immutable for a component and may be precomputed once per
    garment mesh.  Literal target geometry only supplies occupancy and, for deep wrong-side contacts,
    the outward hemisphere; its high-frequency surface is never copied into the cloth field.
    """
    S=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64)
    if len(S)==0 or len(F)==0:return V,{"adjusted":False,"rounds":0}
    envV=np.asarray(env["V"],dtype=np.float64);envF=np.asarray(env["F"],dtype=np.int64);base_support=np.asarray(cache["target_support_V"],dtype=np.float64)[np.asarray(cache["target_support_F"],dtype=np.int64)]
    literal=_target_fit_collision_triangles(cache)
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0);ea,eb=edges[:,0],edges[:,1];deg=np.bincount(np.concatenate((ea,eb)),minlength=len(V)).astype(np.float64);feature=_structural_component_features(S,F);rigid=(len(S)<=650 and feature["max_extent"]<=.055) or (len(S)<=220 and feature["max_extent"]<=.090 and feature["thin_ratio"]<=.18) or (len(S)<=180 and feature["max_extent"]<=.170 and feature["boundary_fraction"]>=.50 and feature["middle_ratio"]<=.35)
    if precomputed is None:
        vf=_coupled_support_frame(S,cache,envV);ff=_coupled_support_frame(S[F].mean(axis=1),cache,envV)
        if vf is None:return V,{"adjusted":False,"rounds":0,"reason":"support frame unavailable"}
        v_contact,v_normal,v_distance,_=vf
        if ff is None:f_contact=f_normal=f_distance=None
        else:f_contact,f_normal,f_distance,_=ff
    else:
        v_contact=np.asarray(precomputed["v_contact"],dtype=np.float64);v_normal=np.asarray(precomputed["v_normal"],dtype=np.float64);v_distance=np.asarray(precomputed["v_distance"],dtype=np.float64)
        f_contact=np.asarray(precomputed.get("f_contact"),dtype=np.float64) if precomputed.get("f_contact") is not None else None;f_normal=np.asarray(precomputed.get("f_normal"),dtype=np.float64) if precomputed.get("f_normal") is not None else None;f_distance=np.asarray(precomputed.get("f_distance"),dtype=np.float64) if precomputed.get("f_distance") is not None else None
    initial=V.copy();round_reports=[];hemisphere_flips=0;patch_fallbacks=0

    def patchwise_accept(before: np.ndarray, proposed: np.ndarray, source: np.ndarray, active_seed: np.ndarray):
        nonlocal patch_fallbacks
        mask=np.asarray(active_seed,dtype=bool).copy()
        # Four topology rings provide a feather boundary around the collision core.
        for _ in range(4):
            grown=mask.copy();hit=mask[ea]|mask[eb];grown[ea[hit]]=True;grown[eb[hit]]=True;mask=grown
        comps=_masked_vertex_components(F,mask);out=before.copy();accepted_any=False
        for core in comps:
            if len(core)<1:continue
            face_mask=np.any(np.isin(F,core),axis=1);fids=np.flatnonzero(face_mask)
            if not len(fids):continue
            patch=np.unique(F[fids].reshape(-1));lookup=np.full(len(before),-1,dtype=np.int64);lookup[patch]=np.arange(len(patch),dtype=np.int64);LF=lookup[F[fids]]
            local_before=out[patch].copy();local_prop=local_before.copy();core_set=np.zeros(len(before),dtype=bool);core_set[core]=True;moving=patch[core_set[patch]]
            if not len(moving):continue
            li=lookup[moving];local_prop[li]=proposed[moving]
            safe,alpha,_,_=_coupled_topology_safe_alpha(source[patch],local_before,local_prop,LF)
            if alpha>0.0 and np.max(np.linalg.norm(safe-local_before,axis=1),initial=0.0)>1e-8:out[patch]=safe;accepted_any=True
        if accepted_any:patch_fallbacks+=1
        return out,accepted_any

    max_rounds=5 if enforce_support else 7
    for round_index in range(max_rounds):
        contact=v_contact;normal=v_normal.copy();source_distance=v_distance
        if enforce_support:
            source_floor=np.clip(source_distance,float(margin),.030);projection=np.einsum("ij,ij->i",V-contact,normal);seed=np.maximum(source_floor-projection,0.0);seed[source_distance>.035]=0.0
        else:
            seed=np.zeros(len(V),dtype=np.float64)
        face_bad=0
        if enforce_support and f_contact is not None:
            floor=np.clip(f_distance,float(margin),.030);proj=np.einsum("ij,ij->i",V[F].mean(axis=1)-f_contact,f_normal);deficit=np.maximum(floor-proj,0.0);active=deficit>.00003;face_bad=int(np.count_nonzero(active))
            if np.any(active):
                rows=np.flatnonzero(active);contrib=deficit[rows]
                for corner in range(3):np.maximum.at(seed,F[rows,corner],contrib)
        # Literal target geometry is absolute occupancy authority: never replace a genuine nearest-body
        # collision with a farther support-consistent triangle.  The smooth source-corresponding support
        # still controls the cloth displacement field; literal normals are used only to disambiguate the
        # outward hemisphere for deep wrong-side contacts, so local anatomy cannot emboss the garment.
        samples=np.vstack((V,V[F].mean(axis=1)));_,literal_normals,signed,_,_=_nearest_literal_occupancy(samples,literal,k=48,exact_band=float(margin)+.0020);need=np.maximum(float(margin)-signed,0.0);vb=need[:len(V)];seed=np.maximum(seed,vb);fb=need[len(V):];rows=np.flatnonzero(fb>.00003)
        if len(rows):
            for corner in range(3):np.maximum.at(seed,F[rows,corner],fb[rows])
        # Deep literal penetration may reveal that source correspondence selected the opposite side of a close body fold.
        literal_vn=literal_normals[:len(V)];agreement=np.einsum("ij,ij->i",normal,literal_vn);wrong=(signed[:len(V)]<-.0010)&(agreement<-.10)
        if np.any(wrong):normal[wrong]*=-1.0;hemisphere_flips+=int(np.count_nonzero(wrong))
        # On the final literal-only polish, a narrow/rigid authored piece moves as one object.  Literal
        # normals may therefore choose its translation direction without embossing their local shape.
        if rigid and not enforce_support:
            lit_use=(need[:len(V)]>.00003)
            if np.any(lit_use):normal[lit_use]=literal_vn[lit_use]
        max_seed=float(np.max(seed,initial=0.0))
        if max_seed<=.00003:break
        before=V.copy();active_seed=seed>.00003
        if rigid:
            active=active_seed;direction=np.sum(normal[active]*seed[active,None],axis=0) if np.any(active) else np.zeros(3);dn=float(np.linalg.norm(direction))
            if dn<1e-12:break
            direction/=dn;proj=np.einsum("ij,j->i",normal[active],direction);good=proj>.12
            if not np.any(good):break
            step=float(min(.0040,np.max(seed[active][good]/np.maximum(proj[good],.12))+.00005));proposed=V+direction*step
        else:
            if not enforce_support:
                # Final literal-only collision patches diffuse the *vector* displacement rather than a
                # scalar magnitude multiplied by every local body normal.  This bridges folds over a
                # coherent cloth patch and avoids reprinting crotch/under-glute/nipple-scale relief.
                anchor=normal*seed[:,None];field_vec=anchor.copy();active_anchor=seed>.00003
                for _ in range(8):
                    sums=np.zeros_like(field_vec);np.add.at(sums,ea,field_vec[eb]);np.add.at(sums,eb,field_vec[ea]);avg=sums/np.maximum(deg[:,None],1.0)
                    field_vec=np.where(active_anchor[:,None],.78*anchor+.22*avg,.78*avg)
                proj=np.einsum("ij,ij->i",field_vec,normal);low=active_anchor&(proj<seed*.55)
                if np.any(low):field_vec[low]+=normal[low]*(seed[low]*.55-proj[low])[:,None]
                mag=np.linalg.norm(field_vec,axis=1);field_vec*=np.minimum(1.0,.0022/np.maximum(mag,1e-12))[:,None];proposed=V+field_vec
            else:
                field=seed.copy()
                for _ in range(6):
                    nbr=np.zeros(len(V),dtype=np.float64);np.maximum.at(nbr,ea,field[eb]);np.maximum.at(nbr,eb,field[ea]);field=np.maximum(field,.76*nbr)
                for _ in range(3):
                    sums=np.zeros(len(V),dtype=np.float64);np.add.at(sums,ea,field[eb]);np.add.at(sums,eb,field[ea]);avg=sums/np.maximum(deg,1.0);field=np.maximum(seed,.68*field+.32*avg)
                field=np.minimum(field,.0035);proposed=V+normal*field[:,None]
        accepted,alpha,topology,edge_metric=_coupled_topology_safe_alpha(S,V,proposed,F)
        if alpha<=0.0 and not rigid:
            accepted,ok=patchwise_accept(V,proposed,S,active_seed)
            if ok:
                alpha=-1.0;topology=_source_relative_topology_summary(S,accepted,F);edge_metric=_coupled_edge_metric(S,accepted,F)
        V=accepted;moved=np.linalg.norm(V-before,axis=1);round_reports.append({"round":round_index+1,"rigid":bool(rigid),"max_seed_mm":max_seed*1000.0,"face_deficits":face_bad,"accepted_alpha":float(alpha),"moved_vertices":int(np.count_nonzero(moved>1e-7)),"move_p95_mm":float(np.percentile(moved,95)*1000.0),"topology":topology,"edge":edge_metric})
        if alpha==0.0 or float(np.max(moved,initial=0.0))<1e-7:break
    total=np.linalg.norm(V-initial,axis=1);return V,{"adjusted":bool(np.any(total>1e-7)),"rounds":len(round_reports),"move_p95_mm":float(np.percentile(total,95)*1000.0),"move_max_mm":float(np.max(total,initial=0.0)*1000.0),"hemisphere_flips":int(hemisphere_flips),"patch_fallbacks":int(patch_fallbacks),"round_reports":round_reports,"policy":"precomputed source-corresponding support clearance + broad topology-coherent literal/auxiliary collision patches; literal geometry only disambiguates occupancy/outward hemisphere","support_authority":bool(enforce_support)}

def _coupled_target_clearance_guard(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]], cache: dict[str,Any], margin: float=.00065, enforce_support: bool=True):
    """Late coupled clearance with one batched occupancy/support gate per mesh.

    The expensive component solver runs only for components that demonstrably violate either the
    source-corresponding support field (first pass) or the literal target body (all passes).  Later
    convergence passes intentionally use literal occupancy only so they cannot re-imprint local target
    relief or fight the already-established source-relative garment shape.
    """
    out={k:np.asarray(v,dtype=np.float64).copy() for k,v in positions.items()};changed=set();rows=[];env=_build_auxiliary_envelope_support(cache);envV=np.asarray(env["V"],dtype=np.float64)
    literal=_target_fit_collision_triangles(cache)
    for name in sorted(out):
        context=contexts[name];S=np.asarray(context["data"]["V"],dtype=np.float64);F=np.asarray(context["data"]["F"],dtype=np.int64);current=out[name];components=_masked_vertex_components(F,np.ones(len(S),dtype=bool));component_reports=[]
        pre_key=(id(cache),len(S),len(F),int(np.asarray(F,dtype=np.int64).sum(dtype=np.int64)))
        pre=context.get("_ravafit_coupled_clearance_precomputed")
        if not isinstance(pre,dict) or pre.get("key")!=pre_key:
            vf=_coupled_support_frame(S,cache,envV);ff=_coupled_support_frame(S[F].mean(axis=1),cache,envV)
            if vf is None:
                rows.append({"mesh":name,"components":[],"reason":"support frame unavailable"});continue
            vc,vn,vd,_=vf
            if ff is None:fc=fn=fd=None
            else:fc,fn,fd,_=ff
            comp_label=np.full(len(S),-1,dtype=np.int64)
            for ci,ids in enumerate(components):comp_label[np.asarray(ids,dtype=np.int64)]=ci
            face_component=comp_label[F[:,0]]
            pre={"key":pre_key,"vc":vc,"vn":vn,"vd":vd,"fc":fc,"fn":fn,"fd":fd,"face_component":face_component}
            context["_ravafit_coupled_clearance_precomputed"]=pre
        else:
            vc=np.asarray(pre["vc"],dtype=np.float64);vn=np.asarray(pre["vn"],dtype=np.float64);vd=np.asarray(pre["vd"],dtype=np.float64);fc=pre.get("fc");fn=pre.get("fn");fd=pre.get("fd");face_component=np.asarray(pre["face_component"],dtype=np.int64)

        face_centres=current[F].mean(axis=1) if len(F) else np.empty((0,3),dtype=np.float64)
        samples=np.vstack((current,face_centres));_,_,literal_signed,_,_=_nearest_literal_occupancy(samples,literal,k=48,exact_band=float(margin)+.0020)
        literal_v_bad=literal_signed[:len(current)] < float(margin)-.00003
        literal_f_bad=literal_signed[len(current):] < float(margin)-.00003 if len(F) else np.zeros(0,dtype=bool)

        # Build direct component maps once.  Source-relative support is a cloth-shaping authority, not a
        # reason to repeatedly process hundreds of tiny buckles/rings/details.  Only substantial or
        # demonstrably sheet-like components participate in support convergence; literal occupancy still
        # applies to every component regardless of size.
        vertex_component=np.full(len(S),-1,dtype=np.int64);support_eligible=np.zeros(len(components),dtype=bool)
        for ci,ids in enumerate(components):
            ids=np.asarray(ids,dtype=np.int64);vertex_component[ids]=ci;LF=_component_local_faces(F,ids,len(S))
            if len(LF):
                feature=_structural_component_features(S[ids],LF);support_eligible[ci]=(len(ids)>=96 or feature["max_extent"]>=.055) and not (len(ids)<=48 and feature["max_extent"]<.040)
        support_v_bad=np.zeros(len(current),dtype=bool);support_f_bad=np.zeros(len(F),dtype=bool)
        if enforce_support:
            source_floor=np.clip(vd,float(margin),.030);projection=np.einsum("ij,ij->i",current-vc,vn);support_v_bad=(vd<=.035)&((source_floor-projection)>.00003)
            support_v_bad &= support_eligible[np.maximum(vertex_component,0)]
            if fc is not None and len(F):
                floor=np.clip(np.asarray(fd,dtype=np.float64),float(margin),.030);proj=np.einsum("ij,ij->i",face_centres-np.asarray(fc,dtype=np.float64),np.asarray(fn,dtype=np.float64));support_f_bad=(floor-proj)>.00003
                support_f_bad &= support_eligible[np.maximum(np.asarray(pre["face_component"],dtype=np.int64),0)]
        active_v=literal_v_bad|support_v_bad;active_f=literal_f_bad|support_f_bad
        active_components=set(int(x) for x in np.unique(np.concatenate((vertex_component[active_v],np.asarray(pre["face_component"],dtype=np.int64)[active_f]))) if int(x)>=0)

        skipped=0
        for ci,ids in enumerate(components):
            ids=np.asarray(ids,dtype=np.int64);fids=np.flatnonzero(face_component==ci);LF=_component_local_faces(F,ids,len(S))
            if len(LF)==0:continue
            if ci not in active_components:
                skipped+=1;continue
            local_pre={"v_contact":vc[ids],"v_normal":vn[ids],"v_distance":vd[ids]}
            if fc is not None:local_pre.update({"f_contact":fc[fids],"f_normal":fn[fids],"f_distance":fd[fids]})
            candidate,rep=_coupled_component_clearance(S[ids],current[ids],LF,cache,env,margin,precomputed=local_pre,enforce_support=enforce_support);current[ids]=candidate
            if rep.get("adjusted"):changed.add(name)
            component_reports.append({"component":int(ci),"vertices":int(len(ids)),**rep})
        out[name]=current;rows.append({"mesh":name,"components":component_reports,"active_components":int(len(active_components)),"skipped_components":int(skipped),"literal_bad_samples":int(np.count_nonzero(literal_v_bad)+np.count_nonzero(literal_f_bad)),"support_bad_samples":int(np.count_nonzero(support_v_bad)+np.count_nonzero(support_f_bad))})
    return out,changed,{"enabled":True,"adjusted_mesh_count":len(changed),"meshes":rows,"auxiliary_envelope":env["report"],"support_authority":bool(enforce_support),"policy":"single batched literal occupancy gate + component-local broad collision patches; support frame is enforced only on the first convergence pass, while later passes preserve the fitted shape and enforce literal occupancy only"}

def _coupled_validation_summary(positions: dict[str,np.ndarray], contexts: dict[str,dict[str,Any]], cache: dict[str,Any]):
    env=_build_auxiliary_envelope_support(cache);env_tri=np.asarray(env["V"],dtype=np.float64)[np.asarray(env["F"],dtype=np.int64)];rows=[];worst_edge=1.0;worst_edge_max=1.0;worst_edge_min=1.0;worst_flip=0.0;worst_env=999.0
    for name in sorted(positions):
        S=np.asarray(contexts[name]["data"]["V"],dtype=np.float64);F=np.asarray(contexts[name]["data"]["F"],dtype=np.int64);V=np.asarray(positions[name],dtype=np.float64);top=_source_relative_topology_summary(S,V,F);edge=_coupled_edge_metric(S,V,F);samples=np.vstack((V,V[F].mean(axis=1)));_,_,signed,_,_=_nearest_literal_occupancy(samples,env_tri,k=48,exact_band=.0030);p01=float(np.percentile(signed,1)*1000.0);rows.append({"mesh":name,"topology":top,"edge":edge,"envelope_signed_p01_mm":p01,"envelope_signed_min_mm":float(np.min(signed)*1000.0)});worst_edge=max(worst_edge,edge["p999"]);worst_edge_max=max(worst_edge_max,edge["max"]);worst_edge_min=min(worst_edge_min,edge["min"]);worst_flip=max(worst_flip,top["flip_fraction"]);worst_env=min(worst_env,p01)
    return {"meshes":rows,"worst_edge_p999":float(worst_edge),"worst_edge_max":float(worst_edge_max),"worst_edge_min":float(worst_edge_min),"worst_flip_fraction":float(worst_flip),"worst_envelope_p01_mm":float(worst_env),"auxiliary":cache.get("_ravafit_target_auxiliary_report",{})}


def _finalize_modded_coupled_solution(source: Any, cache: dict[str,Any], positions: dict[str,np.ndarray], skinning: dict[str,dict[str,Any]], records: dict[str,Any], contexts: dict[str,dict[str,Any]], local_affine_quality_rms_mm: float):
    """B14 macro transfer followed by bounded local-structure/layer/collision convergence."""
    candidate,skinning,records,struct_stats=_finalize_modded_structural_solution(source,cache,positions,skinning,records,contexts,local_affine_quality_rms_mm)
    source_tri=cache.get("_ravafit_source_support_triangles")
    if source_tri is None:source_tri=_triangles_from_surface(cache["source_support_V"],cache["source_support_F"]);cache["_ravafit_source_support_triangles"]=source_tri
    target_tri=cache.get("_ravafit_target_support_triangles")
    if target_tri is None:target_tri=_triangles_from_surface(cache["target_support_V"],cache["target_support_F"]);cache["_ravafit_target_support_triangles"]=target_tri
    meshes={name:{"V":np.asarray(contexts[name]["data"]["V"],dtype=np.float64),"F":np.asarray(contexts[name]["data"]["F"],dtype=np.int64)} for name in candidate}
    local_relations=_infer_local_authored_layer_relations(meshes,source_tri)

    candidate,layered,layer_stage1=_preserve_authored_garment_layers(meshes,candidate,source_tri,target_tri)
    candidate,local_layered,local_layer_stage1=_preserve_local_authored_garment_layers(meshes,candidate,source_tri,target_tri,relations=local_relations,pass_count=2,quick_verify=False)
    pairs={frozenset((str(r.get("inner")),str(r.get("outer")))) for r in layer_stage1.get("relations",[]) if r.get("inner") and r.get("outer")};pairs|={frozenset((str(a),str(b))) for a,b in local_layer_stage1.get("cross_mesh_pairs",[]) if a and b}
    candidate,cross,cross_stage1=_preserve_authored_cross_mesh_assembly(meshes,candidate,pairs)
    candidate,seams,seam_stage1=_preserve_authored_weld_splits(candidate,contexts)
    candidate,long_meshes,long_stage=_long_panel_authored_shape_guard(candidate,contexts,cache);candidate,narrow,narrow_stage=_narrow_authored_strip_shape_guard(candidate,contexts,cache);candidate,ribbons,ribbon_stage=_preserve_authored_ribbon_components(candidate,contexts)

    candidate,clear1,clear_stage1=_coupled_target_clearance_guard(candidate,contexts,cache,margin=.00065,enforce_support=True)
    candidate,seams2,seam_stage2=_preserve_authored_weld_splits(candidate,contexts)
    candidate,layered2,layer_stage2=_preserve_authored_garment_layers(meshes,candidate,source_tri,target_tri)
    # Only relations actually disturbed by clearance are expensive enough to revisit.
    candidate,local_layered2,local_layer_stage2=_preserve_local_authored_garment_layers(meshes,candidate,source_tri,target_tri,relations=local_relations,pass_count=1,quick_verify=True)
    pairs2={frozenset((str(r.get("inner")),str(r.get("outer")))) for r in layer_stage2.get("relations",[]) if r.get("inner") and r.get("outer")};pairs2|={frozenset((str(a),str(b))) for a,b in local_layer_stage2.get("cross_mesh_pairs",[]) if a and b}
    candidate,cross2,cross_stage2=_preserve_authored_cross_mesh_assembly(meshes,candidate,pairs2)
    candidate,clear2,clear_stage2=_coupled_target_clearance_guard(candidate,contexts,cache,margin=.00065,enforce_support=False)
    candidate,seams3,seam_stage3=_preserve_authored_weld_splits(candidate,contexts)
    # Seam reconciliation is no longer allowed to be the last geometry mutation.  Literal target clearance wins final authority.
    candidate,clear3,clear_stage3=_coupled_target_clearance_guard(candidate,contexts,cache,margin=.00070,enforce_support=False)
    # Last-resort contact polish uses a tiny positive margin.  This lets topology-safe incremental patches
    # and rigid authored strips clear sparse residual contacts without re-sculpting the established fit.
    candidate,clear4,clear_stage4=_coupled_target_clearance_guard(candidate,contexts,cache,margin=.00012,enforce_support=False)
    candidate,unilateral_pair_stage=_preserve_final_unilateral_pair_separation(source,candidate)

    for name in sorted(candidate):
        context=contexts[name];data=context["data"];w=context["w"];retargeted,stage=_retarget_garment_skinning(candidate[name],data["V"],data["W"],data["joint_names"],cache,context["behavior"],context["effective_behavior"],context["labels"],context["classes"],w["raw_to_weld"]);_assert_finite_stage(name,"coupled final skinning retarget",retargeted);skinning[name]={"weights":retargeted,"joint_names":list(data["joint_names"]),"stage":stage};records.setdefault(name,{})["skinning"]=stage;records[name]["coupled_core_finalizer_applied"]=True
        if name in (set(layered)|set(local_layered)|set(layered2)|set(local_layered2)):records[name]["layer_order_preserved"]=True
        if name in (set(cross)|set(cross2)):records[name]["authored_cross_mesh_assembly_preserved"]=True
        if name in (set(seams)|set(seams2)|set(seams3)):records[name]["authored_split_seams_preserved"]=True

    validation=_coupled_validation_summary(candidate,contexts,cache);proxy=_garment_support_proxy(cache)
    struct_stats.update({
        "authored_layer_relations":int(layer_stage2.get("relation_count",0)),"authored_layer_adjusted_meshes":int(layer_stage2.get("adjusted_mesh_count",0)),"authored_layer_report":{"first":layer_stage1,"final":layer_stage2},
        "local_authored_layer_relations":int(local_layer_stage2.get("relation_count",0)),"local_authored_layer_adjusted_meshes":int(local_layer_stage2.get("adjusted_mesh_count",0)),"local_authored_layer_report":{"first":local_layer_stage1,"final":local_layer_stage2},
        "authored_cross_mesh_relations":int(cross_stage2.get("relation_count",0)),"authored_cross_mesh_adjusted_meshes":int(cross_stage2.get("adjusted_mesh_count",0)),"authored_cross_mesh_report":{"first":cross_stage1,"final":cross_stage2},
        "authored_split_seams":{"first":seam_stage1,"middle":seam_stage2,"final":seam_stage3},"authored_long_panel_shape_guard":long_stage,"authored_narrow_strip_shape_guard":narrow_stage,"authored_ribbon_topology":ribbon_stage,
        "coupled_target_clearance":{"first":clear_stage1,"second":clear_stage2,"final_after_seams":clear_stage3,"residual_contact_polish":clear_stage4},"final_unilateral_pair_separation":unilateral_pair_stage,"coupled_validation":validation,"target_body_auxiliary_obstacles":cache.get("_ravafit_target_auxiliary_report",{}),"garment_support_proxy":proxy.get("report",{}),
        "core_policy":"B14 macro fit -> source-relative differential shape -> whole/local authored layers -> component/topology-safe seams -> broad body+aux clearance -> verified layer reconciliation -> final literal clearance -> source-proven unilateral-pair separation"})
    return candidate,skinning,records,struct_stats

def _finalize_garment_solution(source: Any, cache: dict[str, Any], positions: dict[str,np.ndarray], skinning: dict[str,dict[str,Any]], records: dict[str,Any], local_affine_quality_rms_mm: float, _assembly_in_process: bool = False):
    """Run the shared authored-assembly/final-clearance lane after independent mesh solves."""
    if not positions:
        raise ValueError("No garment meshes were eligible for B14 fitting after removing the source body.")
    dense_vanilla_proxy=bool(cache.get("dense_vanilla_source_proxy",False))
    strict_b14_contract=bool(cache.get("_ravafit_strict_b14_contract",False))
    if strict_b14_contract:
        source_tri=cache.get("_ravafit_strict_source_surface_triangles")
        if source_tri is None:
            source_tri=_triangles_from_surface(cache.get('_ravafit_strict_source_surface_V',cache['source_surface_V']),cache.get('_ravafit_strict_source_surface_F',cache['source_surface_F']));cache["_ravafit_strict_source_surface_triangles"]=source_tri
        target_tri=cache.get("_ravafit_strict_target_surface_triangles")
        if target_tri is None:
            target_tri=_triangles_from_surface(cache.get('_ravafit_strict_target_surface_V',cache['target_surface_V']),cache.get('_ravafit_strict_target_surface_F',cache['target_surface_F']));cache["_ravafit_strict_target_surface_triangles"]=target_tri
    else:
        source_tri=cache.get("_ravafit_source_support_triangles")
        if source_tri is None:
            source_tri=_triangles_from_surface(cache["source_support_V"],cache["source_support_F"]);cache["_ravafit_source_support_triangles"]=source_tri
        target_tri=cache.get("_ravafit_target_support_triangles")
        if target_tri is None:
            target_tri=_triangles_from_surface(cache["target_support_V"],cache["target_support_F"]);cache["_ravafit_target_support_triangles"]=target_tri

    layer_meshes: dict[str,dict[str,Any]]={}
    retarget_contexts: dict[str,dict[str,Any]]={}
    for name in sorted(positions):
        data=source.data(name)
        w=weld_mesh(data)
        behavior,features,labels,classes,_details=infer_shell_behavior(w,source_tri)
        record=records.get(name,{})
        effective_behavior=str(record.get("behavior") or behavior)
        layer_meshes[name]={"V":np.asarray(data["V"],dtype=np.float64),"F":np.asarray(data["F"],dtype=np.int64)}
        retarget_contexts[name]={"data":data,"w":w,"behavior":behavior,"effective_behavior":effective_behavior,"features":features,"labels":labels,"classes":classes}

    if not dense_vanilla_proxy:
        return _finalize_modded_coupled_solution(source,cache,positions,skinning,records,retarget_contexts,local_affine_quality_rms_mm)

    if _assembly_in_process:
        def _assembly_local():
            p1,layered,layer_stage=_preserve_authored_garment_layers(layer_meshes,positions,source_tri,target_tri)
            layer_pairs={frozenset((str(r.get("inner")),str(r.get("outer")))) for r in layer_stage.get("relations",[]) if r.get("inner") and r.get("outer")}
            p2,cross,cross_stage=_preserve_authored_cross_mesh_assembly(layer_meshes,p1,layer_pairs)
            return p2,layered,layer_stage,cross,cross_stage
        positions,layered_meshes,layer_stage,cross_meshes,cross_mesh_stage=_finite_stage_call("garment assembly","authored layer/cross-mesh assembly preservation",_assembly_local)
        layer_stage["worker"]={"mode":"in-process-after-isolated-garment-workers"};cross_mesh_stage["worker"]={"mode":"in-process-after-isolated-garment-workers"}
    else:
        positions, layered_meshes, layer_stage, cross_meshes, cross_mesh_stage = _finite_stage_call("garment assembly", "authored layer/cross-mesh assembly preservation", lambda: _preserve_authored_assembly_fresh(layer_meshes, positions, source_tri, target_tri))
    far_structural_meshes=set();far_structural_stage={"enabled":False,"reason":"dense vanilla or no extreme far structure"}
    if not dense_vanilla_proxy:
        positions,far_structural_meshes,far_structural_stage=_finite_stage_call("garment assembly","far authored structural anchor",lambda:_far_structural_assembly_anchor_guard(positions,retarget_contexts,cache))
        for changed_name in far_structural_meshes:
            if changed_name in records:records[changed_name]["far_authored_structural_anchor"]=far_structural_stage
    positions, seam_meshes, seam_stage = _finite_stage_call("garment assembly", "authored split seam preservation", lambda: _preserve_authored_weld_splits(positions, retarget_contexts))
    ribbon_meshes=set();ribbon_stage={"enabled":False,"reason":"runs after clearance so ribbon topology is final"}
    dense_clearance_meshes=set();dense_clearance_stage={"enabled":False,"reason":"not dense vanilla proxy"}
    modded_coverage_meshes=set();modded_coverage_stage={"enabled":False,"reason":"dense vanilla uses its own guard"}
    if dense_vanilla_proxy:
        positions,dense_clearance_meshes,dense_clearance_stage=_finite_stage_call("garment assembly","dense vanilla outward-expansion clearance",lambda:_dense_vanilla_expansion_clearance_guard(positions,retarget_contexts,cache))
        for changed_name in dense_clearance_meshes:
            if changed_name in records:records[changed_name]["dense_vanilla_expansion_clearance"]=dense_clearance_stage
    else:
        positions,modded_coverage_meshes,modded_coverage_stage=_finite_stage_call("garment assembly","source-authored literal-body coverage clearance",lambda:_modded_source_coverage_clearance_guard(positions,retarget_contexts,cache))
        for changed_name in modded_coverage_meshes:
            if changed_name in records:records[changed_name]["source_authored_literal_body_clearance"]=modded_coverage_stage
    long_panel_meshes=set();long_panel_stage={"enabled":False,"reason":"dense vanilla or no qualifying rough long panel"}
    if not dense_vanilla_proxy:
        positions,long_panel_meshes,long_panel_stage=_finite_stage_call("garment assembly","authored long-panel shape guard",lambda:_long_panel_authored_shape_guard(positions,retarget_contexts,cache))
        for changed_name in long_panel_meshes:
            if changed_name in records:records[changed_name]["authored_long_panel_shape_guard"]=long_panel_stage
    narrow_strip_meshes=set();narrow_strip_stage={"enabled":False,"reason":"no distorted qualifying narrow raw strip"}
    if not dense_vanilla_proxy:
        positions,narrow_strip_meshes,narrow_strip_stage=_finite_stage_call("garment assembly","authored narrow-strip shape guard",lambda:_narrow_authored_strip_shape_guard(positions,retarget_contexts,cache))
        for changed_name in narrow_strip_meshes:
            if changed_name in records:records[changed_name]["authored_narrow_strip_shape_guard"]=narrow_strip_stage
    positions,ribbon_meshes,ribbon_stage=_finite_stage_call("garment assembly","authored ribbon topology preservation",lambda:_preserve_authored_ribbon_components(positions,retarget_contexts))

    skinning_refresh_meshes=set(layered_meshes)|set(cross_meshes)|set(seam_meshes)|set(far_structural_meshes)|set(ribbon_meshes)|set(narrow_strip_meshes)|set(dense_clearance_meshes)|set(modded_coverage_meshes)|set(long_panel_meshes)
    for name in sorted(skinning_refresh_meshes):
        context=retarget_contexts[name];data=context["data"];w=context["w"]
        retargeted_weights,skinning_stage=_retarget_garment_skinning(positions[name],data["V"],data["W"],data["joint_names"],cache,context["behavior"],context["effective_behavior"],context["labels"],context["classes"],w["raw_to_weld"])
        _assert_finite_stage(name,"assembly-preserved skinning retarget",retargeted_weights)
        skinning[name]={"weights":retargeted_weights,"joint_names":list(data["joint_names"]),"stage":skinning_stage}
        records.setdefault(name,{})["skinning"]=skinning_stage
        if name in layered_meshes:records[name]["layer_order_preserved"]=True
        if name in cross_meshes:records[name]["authored_cross_mesh_assembly_preserved"]=True
        if name in seam_meshes:records[name]["authored_split_seams_preserved"]=True
        if name in ribbon_meshes:records[name]["authored_ribbon_topology_preserved"]=True
        if name in narrow_strip_meshes:records[name]["authored_narrow_strip_shape_preserved"]=True
        if name in modded_coverage_meshes:records[name]["source_authored_literal_body_clearance_applied"]=True

    return positions,skinning,records,{
        "local_affine_quality_rms_mm":float(local_affine_quality_rms_mm),
        "garment_mesh_count":len(positions),
        "skinning_retargeted_mesh_count":len(skinning),
        "authored_layer_relations":int(layer_stage.get("relation_count",0)),
        "authored_layer_adjusted_meshes":int(layer_stage.get("adjusted_mesh_count",0)),
        "authored_layer_report":layer_stage,
        "authored_cross_mesh_relations":int(cross_mesh_stage.get("relation_count",0)),
        "authored_cross_mesh_adjusted_meshes":int(cross_mesh_stage.get("adjusted_mesh_count",0)),
        "authored_cross_mesh_report":cross_mesh_stage,
        "authored_split_seams":seam_stage,
        "far_authored_structural_anchor":far_structural_stage,
        "authored_long_panel_shape_guard":long_panel_stage,
        "authored_ribbon_topology":ribbon_stage,
        "authored_narrow_strip_shape_guard":narrow_strip_stage,
        "modded_source_coverage_clearance":modded_coverage_stage,
        "dense_vanilla_expansion_clearance":dense_clearance_stage,
        "body_support_proxy":{"enabled":True,"mode":"dense target-topology vanilla proxy + local outward-expansion clearance" if dense_vanilla_proxy else "source-topology macro envelope + filtered local relief + literal collision","slots":cache.get("slot_stats",[])},
        "source_body_suppression":_public_source_body_suppression(cache.get("_ravafit_source_body_suppression")),
    }


@dataclass(frozen=True)
class GarmentLayerMember:
    mesh_name: str
    component_index: int
    vertex_ids: tuple[int, ...]
    face_ids: tuple[int, ...]
    material: str
    structural_classification: str
    source_clearance_median_mm: float
    source_clearance_p95_mm: float
    source_signed_clearance_median_mm: float

    @property
    def stable_component_id(self) -> str:
        return f"{self.mesh_name}::component-{self.component_index}"


@dataclass(frozen=True)
class GarmentLayer:
    """Geometry-inferred authored garment layer used only to orchestrate strict B14.

    The representation is deliberately garment-name agnostic.  Faces are stored in layer-local
    vertex indexing; members retain the exact source mesh/component identities so disconnected,
    bilateral, decorative and peer structures never lose their authored identity.
    """
    stable_id: str
    members: tuple[GarmentLayerMember, ...]
    faces: np.ndarray
    materials: tuple[str, ...]
    source_clearance_median_mm: float
    source_clearance_p95_mm: float
    connected_components: tuple[str, ...]
    structural_classification: str
    source_signed_clearance_median_mm: float = 0.0
    source_ordering_relationships: tuple[str, ...] = ()

    @property
    def source_meshes(self) -> tuple[str, ...]:
        return tuple(sorted({member.mesh_name for member in self.members}))


@dataclass(frozen=True)
class B14LayerResult:
    """Immutable geometry returned by the untouched strict B14 worker for one authored layer."""
    stable_id: str
    positions: np.ndarray
    behavior: str
    source_clearance_median_mm: float
    source_clearance_p95_mm: float
    solve_time_sec: float
    stage: dict[str, Any]


def _readonly_array(value: np.ndarray, dtype=np.float64) -> np.ndarray:
    out=np.asarray(value,dtype=dtype).copy()
    out.setflags(write=False)
    return out


def _component_classification(vertices: np.ndarray, faces: np.ndarray) -> str:
    feature=_structural_component_features(vertices,faces)
    n=int(len(vertices));extent=float(feature.get("max_extent",0.0));middle=float(feature.get("middle_ratio",0.0));thin=float(feature.get("thin_ratio",0.0));boundary=float(feature.get("boundary_fraction",0.0))
    # Physical scale and shape outrank tessellation density. XIV accessories can contain hundreds of
    # seam/normal-split vertices in a centimetre-scale buckle or charm; vertex count must not turn
    # those into cloth shells.
    if extent<=.018 or (n<96 and extent<.045):
        return "rigid_detail"
    # Narrow open strips such as bands/straps keep their own authority. They are deformable by the
    # body correspondence but must not enter a broad shell optimiser merely because they are dense.
    if extent<=.055 and thin<=.08 and middle<=.38 and boundary>=.18:
        return "ribbon_or_strap"
    if extent>.045 and (thin<.10 or middle<.26) and boundary>.08:
        return "ribbon_or_strap"
    if boundary>.10:
        return "open_shell"
    return "shell"


def _component_clearance_stats(vertices: np.ndarray, source_body_triangles: np.ndarray) -> tuple[float,float,float]:
    P=np.asarray(vertices,dtype=np.float64)
    if not len(P):
        return 0.0,0.0,0.0
    _,_,signed,distance,_=_b14_nearest_surface(P,source_body_triangles,k=32)
    distance=np.asarray(distance,dtype=np.float64);signed=np.asarray(signed,dtype=np.float64)
    return float(np.median(distance)*1000.0),float(np.percentile(distance,95)*1000.0),float(np.median(signed)*1000.0)


def _discover_garment_components(source: Any, body_mesh_names: set[str], mesh_filter: set[str] | None, source_body_triangles: np.ndarray) -> list[dict[str,Any]]:
    """Discover authored connected structures on welded/rendered topology.

    XIV MDLs commonly duplicate coincident vertices at UV/material/smoothing seams.  Raw-index
    connectivity therefore fragments one authored panel into many apparent components.  Component
    identity is discovered on ``weld_mesh`` topology, then expanded back to the exact raw vertex and
    face storage so the production solve still preserves authored seams and untouched dead vertices.
    """
    components=[]
    for mesh_name in source.mesh_names():
        if not mesh_name or mesh_name in body_mesh_names or (mesh_filter is not None and mesh_name not in mesh_filter):
            continue
        try:
            data=source.data(mesh_name)
        except Exception:
            continue
        V=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64)
        if not len(V) or not len(F):
            continue
        welded=weld_mesh(data)
        WV=np.asarray(welded.get("V",[]),dtype=np.float64);WF=np.asarray(welded.get("F",[]),dtype=np.int64);raw_to_weld=np.asarray(welded.get("raw_to_weld",[]),dtype=np.int64)
        if not len(WV) or not len(WF) or raw_to_weld.shape!=(len(V),):
            raise ValueError(f"{mesh_name} could not expose stable welded topology for layer discovery.")
        welded_groups=_masked_vertex_components(WF,np.ones(len(WV),dtype=bool))
        sortable=[]
        for welded_ids in welded_groups:
            welded_ids=np.asarray(welded_ids,dtype=np.int64)
            if not len(welded_ids):continue
            weld_mask=np.zeros(len(WV),dtype=bool);weld_mask[welded_ids]=True
            raw_ids=np.flatnonzero(weld_mask[raw_to_weld]).astype(np.int64)
            if not len(raw_ids):continue
            raw_membership=np.zeros(len(V),dtype=bool);raw_membership[raw_ids]=True
            face_ids=np.flatnonzero(np.all(raw_membership[F],axis=1)).astype(np.int64)
            if not len(face_ids):continue
            raw_local_faces=_component_local_faces(F,raw_ids,len(V))
            welded_membership=np.zeros(len(WV),dtype=bool);welded_membership[welded_ids]=True
            welded_face_ids=np.flatnonzero(np.all(welded_membership[WF],axis=1))
            welded_local_faces=_component_local_faces(WF,welded_ids,len(WV)) if len(welded_face_ids) else np.zeros((0,3),dtype=np.int64)
            raw_P=V[raw_ids];structural_P=WV[welded_ids]
            median,p95,signed_median=_component_clearance_stats(structural_P,source_body_triangles)
            classification=_component_classification(structural_P,welded_local_faces)
            sortable.append((int(np.min(raw_ids)),{
                "mesh_name":str(mesh_name),
                "component_index":0,
                "vertex_ids":raw_ids,
                "face_ids":face_ids,
                "V":raw_P,
                "F":raw_local_faces,
                "material":_normalise_material(data.get("material")),
                "classification":classification,
                "clearance_median_mm":median,
                "clearance_p95_mm":p95,
                "signed_clearance_median_mm":signed_median,
                "centroid":np.mean(structural_P,axis=0),
                "extent":np.ptp(structural_P,axis=0),
                "vertex_count":int(len(raw_P)),
                "rendered_vertex_count":int(len(structural_P)),
                "raw_seam_duplicate_count":int(len(raw_P)-len(structural_P)),
            }))
        for component_index,(_,row) in enumerate(sorted(sortable,key=lambda item:item[0])):
            row["component_index"]=int(component_index);components.append(row)
    return components


def _component_distance_sample(component: dict[str,Any]) -> tuple[np.ndarray,cKDTree]:
    cached=component.get("_distance_sample")
    if cached is not None:return cached
    V=np.asarray(component["V"],dtype=np.float64)
    if len(V)>256:V=V[np.unique(np.linspace(0,len(V)-1,256,dtype=np.int64))]
    cached=(V,cKDTree(V) if len(V) else None);component["_distance_sample"]=cached
    return cached

def _sample_min_component_distance(a: dict[str,Any], b: dict[str,Any]) -> float:
    av,at=_component_distance_sample(a);bv,bt=_component_distance_sample(b)
    if not len(av) or not len(bv):return float("inf")
    return float(min(np.min(bt.query(av,k=1)[0]),np.min(at.query(bv,k=1)[0])))


def _components_are_nested_layers(a: dict[str,Any], b: dict[str,Any], source_body_triangles: np.ndarray) -> bool:
    # Small rigid decorations are assembly members, not competing garment shells.  Skipping the
    # expensive directional shell test here also keeps dense ornament packs linear enough to run.
    if a.get("classification")=="rigid_detail" or b.get("classification")=="rigid_detail":return False
    av=np.asarray(a["V"],dtype=np.float64);bv=np.asarray(b["V"],dtype=np.float64)
    if not _expanded_aabb_overlap(av,bv,.010):
        return False
    at=av[np.asarray(a["F"],dtype=np.int64)];bt=bv[np.asarray(b["F"],dtype=np.int64)]
    ae=_local_layer_directional_evidence(av,bt,source_body_triangles,close_distance=.010)
    be=_local_layer_directional_evidence(bv,at,source_body_triangles,close_distance=.010)
    if ae is not None and be is not None:
        opposing=(ae["positive_fraction"]>=.72 and be["negative_fraction"]>=.72) or (be["positive_fraction"]>=.72 and ae["negative_fraction"]>=.72)
        radial=max(abs(float(ae["median"])),abs(float(be["median"])))
        if opposing and radial>=.00018:
            return True
    clearance_delta=abs(float(a["clearance_median_mm"])-float(b["clearance_median_mm"]))
    return clearance_delta>=1.10 and _sample_min_component_distance(a,b)<=.010


def _mirrored_component_peers(a: dict[str,Any], b: dict[str,Any]) -> bool:
    ca=np.asarray(a["centroid"],dtype=np.float64);cb=np.asarray(b["centroid"],dtype=np.float64)
    if ca[0]*cb[0]>=0.0 or min(abs(ca[0]),abs(cb[0]))<.004:
        return False
    ea=np.asarray(a["extent"],dtype=np.float64);eb=np.asarray(b["extent"],dtype=np.float64)
    scale=max(float(np.linalg.norm(ea)),float(np.linalg.norm(eb)),1e-6)
    lateral=abs(abs(float(ca[0]))-abs(float(cb[0])))
    non_lateral=float(np.linalg.norm(ca[1:]-cb[1:]))
    extent_error=float(np.linalg.norm(ea-eb))/scale
    count_ratio=min(a["vertex_count"],b["vertex_count"])/max(a["vertex_count"],b["vertex_count"],1)
    return lateral<=max(.010,.18*scale) and non_lateral<=max(.025,.32*scale) and extent_error<=.38 and count_ratio>=.45


def _component_layer_affinity(a: dict[str,Any], b: dict[str,Any], source_body_triangles: np.ndarray) -> float:
    mirrored=_mirrored_component_peers(a,b)
    if not mirrored and _components_are_nested_layers(a,b,source_body_triangles):
        return -100.0
    same_material=bool(a["material"]) and a["material"]==b["material"]
    clearance_delta=abs(float(a["clearance_median_mm"])-float(b["clearance_median_mm"]))
    clearance_scale=max(1.25,.30*max(float(a["clearance_p95_mm"]),float(b["clearance_p95_mm"]),2.0))
    clearance_similar=clearance_delta<=clearance_scale
    near=_sample_min_component_distance(a,b)
    close=near<=.0065
    overlap=_expanded_aabb_overlap(np.asarray(a["V"]),np.asarray(b["V"]),.004)
    same_kind=a["classification"]==b["classification"]
    same_mesh=a["mesh_name"]==b["mesh_name"]
    score=0.0
    if same_material:score+=3.0
    if clearance_similar:score+=1.75
    if mirrored:score+=3.0
    if close:score+=2.0
    if overlap:score+=.75
    if same_kind:score+=.75
    if same_mesh:score+=.50
    detail_attachment=(a["classification"]=="rigid_detail")^(b["classification"]=="rigid_detail")
    if detail_attachment and same_material and near<=.009:
        score+=2.25
    # Two substantial disconnected structures with different materials should not fuse merely
    # because they occupy the same body region. Mirrored peers remain a deliberate exception.
    substantial=a["classification"]!="rigid_detail" and b["classification"]!="rigid_detail"
    if substantial and not same_material and not mirrored:
        score-=2.0
    # Two substantial overlapping shells require stronger evidence than merely sharing a material.
    if substantial and overlap and not mirrored and not close:
        score-=1.5
    return score


def _layer_stable_id(members: list[dict[str,Any]]) -> str:
    identity="|".join(sorted(f"{m['mesh_name']}:{m['component_index']}:{m['material']}" for m in members))
    return "layer-"+hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def _layer_local_faces_and_members(source: Any, members: list[dict[str,Any]]) -> tuple[np.ndarray,tuple[GarmentLayerMember,...]]:
    faces=[];offset=0;public=[]
    for comp in sorted(members,key=lambda x:(x["mesh_name"],x["component_index"])):
        local=np.asarray(comp["F"],dtype=np.int64)
        faces.append(local+offset)
        public.append(GarmentLayerMember(
            mesh_name=str(comp["mesh_name"]),
            component_index=int(comp["component_index"]),
            vertex_ids=tuple(int(x) for x in np.asarray(comp["vertex_ids"],dtype=np.int64)),
            face_ids=tuple(int(x) for x in np.asarray(comp["face_ids"],dtype=np.int64)),
            material=str(comp["material"]),
            structural_classification=str(comp["classification"]),
            source_clearance_median_mm=float(comp["clearance_median_mm"]),
            source_clearance_p95_mm=float(comp["clearance_p95_mm"]),
            source_signed_clearance_median_mm=float(comp["signed_clearance_median_mm"]),
        ))
        offset+=len(comp["vertex_ids"])
    F=np.vstack(faces) if faces else np.zeros((0,3),dtype=np.int64)
    F.setflags(write=False)
    return F,tuple(public)


def _infer_garment_layers(source: Any, body_mesh_names: set[str], mesh_filter: set[str] | None, source_body_triangles: np.ndarray) -> tuple[list[GarmentLayer],dict[str,dict[str,Any]]]:
    components=_discover_garment_components(source,body_mesh_names,mesh_filter,source_body_triangles)
    if not components:
        return [],{}
    parent=list(range(len(components)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:
            return
        left=[i for i in range(len(components)) if find(i)==ra]
        right=[i for i in range(len(components)) if find(i)==rb]
        # Prevent a decorative/peer bridge from transitively collapsing two authored nested shells.
        if any(_components_are_nested_layers(components[li],components[ri],source_body_triangles) and not _mirrored_component_peers(components[li],components[ri]) for li in left for ri in right):
            return
        parent[rb]=ra
    # Only substantial structures participate in authored-layer union.  Disconnected rigid details
    # deliberately remain independent deformation authorities; assembly membership is relationship
    # metadata, never permission for a neighbouring shell to deform them.  Materials are also a hard
    # layer boundary here: nested shirt/bra/fishnet shells may occupy the same slot but must never
    # collapse into one solve merely because their AABBs overlap.
    for ai,a in enumerate(components):
        if a["classification"]=="rigid_detail":continue
        for bi in range(ai+1,len(components)):
            b=components[bi]
            if b["classification"]=="rigid_detail" or a["material"]!=b["material"]:continue
            if _component_layer_affinity(a,b,source_body_triangles)>=5.0:
                union(ai,bi)
    groups={}
    for i,comp in enumerate(components):
        groups.setdefault(find(i),[]).append(comp)
    layers=[];component_lookup={}
    for members in sorted(groups.values(),key=lambda g:min((m["mesh_name"],m["component_index"]) for m in g)):
        layer_id=_layer_stable_id(members)
        F,public_members=_layer_local_faces_and_members(source,members)
        layer_vertices=np.vstack([np.asarray(m["V"],dtype=np.float64) for m in members])
        layer_clearance_median,layer_clearance_p95,layer_signed_median=_component_clearance_stats(layer_vertices,source_body_triangles)
        classifications=[m["classification"] for m in members]
        if any(c in {"shell","open_shell"} for c in classifications):layer_class="shell"
        elif any(c=="ribbon_or_strap" for c in classifications):layer_class="ribbon_or_strap"
        else:layer_class="rigid_assembly"
        layer=GarmentLayer(
            stable_id=layer_id,
            members=public_members,
            faces=F,
            materials=tuple(sorted({str(m["material"]) for m in members})),
            source_clearance_median_mm=float(layer_clearance_median),
            source_clearance_p95_mm=float(layer_clearance_p95),
            connected_components=tuple(member.stable_component_id for member in public_members),
            structural_classification=layer_class,
            source_signed_clearance_median_mm=float(layer_signed_median),
        )
        layers.append(layer)
        for m in members:
            component_lookup[f"{m['mesh_name']}::{m['component_index']}"]=m
    return layers,component_lookup


def _layer_virtual_data(source: Any, layer: GarmentLayer) -> tuple[dict[str,Any],tuple[tuple[str,int],...]]:
    union_names=[]
    for member in layer.members:
        for name in source.data(member.mesh_name).get("joint_names",[]):
            if name not in union_names:union_names.append(name)
    name_index={name:i for i,name in enumerate(union_names)}
    verts=[];weights=[];uv=[];have_uv=True;mapping=[]
    for member in layer.members:
        data=source.data(member.mesh_name);ids=np.asarray(member.vertex_ids,dtype=np.int64)
        verts.append(np.asarray(data["V"],dtype=np.float64)[ids])
        src_w=np.asarray(data["W"],dtype=np.float64)[ids];src_names=list(data.get("joint_names",[]))
        W=np.zeros((len(ids),len(union_names)),dtype=np.float64)
        for col,name in enumerate(src_names):
            if col<src_w.shape[1]:W[:,name_index[name]]+=src_w[:,col]
        total=W.sum(axis=1,keepdims=True);good=total[:,0]>1e-12
        if not np.all(good):raise ValueError(f"{layer.stable_id} contains vertices with no skin weights.")
        W/=total;weights.append(W)
        source_uv=data.get("UV")
        if source_uv is None or len(source_uv)!=len(data["V"]):
            have_uv=False
        else:
            uv.append(np.asarray(source_uv,dtype=np.float64)[ids])
        mapping.extend((member.mesh_name,int(i)) for i in ids)
    V=np.vstack(verts) if verts else np.zeros((0,3),dtype=np.float64)
    W=np.vstack(weights) if weights else np.zeros((0,len(union_names)),dtype=np.float64)
    UV=np.vstack(uv) if have_uv and uv else None
    material="|".join(layer.materials)
    return {"V":V,"F":np.asarray(layer.faces,dtype=np.int64),"W":W,"UV":UV,"N":None,"joint_names":union_names,"material":material,"name":layer.stable_id},tuple(mapping)


def _surface_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    V=np.asarray(vertices,dtype=np.float64);F=np.asarray(faces,dtype=np.int64);N=np.zeros_like(V)
    if len(V)==0 or len(F)==0:return N
    face_n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);np.add.at(N,F[:,0],face_n);np.add.at(N,F[:,1],face_n);np.add.at(N,F[:,2],face_n)
    N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12);return N


def _infer_thin_volume_surface(w: dict[str,Any], source_body_triangles: np.ndarray) -> dict[str,Any] | None:
    """Recognise a source-authored thin closed volume and expose its garment-facing surface.

    This is source construction discovery, not a fitting rule.  The target body is never consulted.
    A qualifying volume must prove that almost all welded vertices have a mutual, opposite-facing
    partner at a stable millimetre-scale thickness.  B14 then fits only the authored outside surface;
    the full volume is reconstructed from that solved surface using source-local offsets.
    """
    V=np.asarray(w.get("V",[]),dtype=np.float64);F=np.asarray(w.get("F",[]),dtype=np.int64)
    if len(V)<24 or len(F)<24:return None
    feature=_structural_component_features(V,F)
    if float(feature.get("boundary_fraction",1.0))>.035:return None
    N=_surface_vertex_normals(V,F);tree=cKDTree(V);k=min(32,len(V));distance,index=tree.query(V,k=k);distance=distance if distance.ndim>1 else distance[:,None];index=index if index.ndim>1 else index[:,None]
    partner=np.full(len(V),-1,dtype=np.int64);pair_distance=np.full(len(V),np.inf,dtype=np.float64)
    for col in range(1,index.shape[1]):
        candidate=index[:,col];alignment=np.einsum("ij,ij->i",N,N[candidate]);valid=(alignment<=-.55)&(distance[:,col]>=.00035)&(distance[:,col]<=.0060)&(distance[:,col]<pair_distance)
        partner[valid]=candidate[valid];pair_distance[valid]=distance[valid,col]
    safe_partner=np.maximum(partner,0);mutual=(partner>=0)&(partner[safe_partner]==np.arange(len(V),dtype=np.int64));coverage=float(np.mean(mutual))
    if coverage<.90:return None
    thickness=pair_distance[mutual]
    if not len(thickness):return None
    p50=float(np.median(thickness));p95=float(np.percentile(thickness,95))
    if p50<.00045 or p50>.0050 or p95>max(.0060,p50*1.45):return None
    _,body_normals,_,body_distance,_=_b14_nearest_surface(V,np.asarray(source_body_triangles,dtype=np.float64),k=32);body_distance=np.asarray(body_distance,dtype=np.float64);body_normals=np.asarray(body_normals,dtype=np.float64)
    outer=np.zeros(len(V),dtype=bool);seen=set()
    for i in np.flatnonzero(mutual):
        j=int(partner[i]);key=(min(int(i),j),max(int(i),j))
        if key in seen:continue
        seen.add(key)
        if abs(float(body_distance[i]-body_distance[j]))>.00015:chosen=int(i) if body_distance[i]>body_distance[j] else j
        else:
            ai=float(np.dot(N[i],body_normals[i]));aj=float(np.dot(N[j],body_normals[j]));chosen=int(i) if ai>=aj else j
        outer[chosen]=True
    # Rare unpaired wall/seam vertices use authored normal orientation only for surface selection.
    for i in np.flatnonzero(~mutual):outer[i]=float(np.dot(N[i],body_normals[i]))>=0.0
    outer_face_mask=np.all(outer[F],axis=1);outer_faces_global=F[outer_face_mask]
    if len(outer_faces_global)<max(12,int(len(F)*.25)):return None
    outer_ids=np.unique(outer_faces_global.reshape(-1));remap=np.full(len(V),-1,dtype=np.int64);remap[outer_ids]=np.arange(len(outer_ids),dtype=np.int64);outer_faces=remap[outer_faces_global]
    outer_feature=_structural_component_features(V[outer_ids],outer_faces)
    # A useful garment surface must reveal authored openings/boundaries hidden by the closed volume.
    if float(outer_feature.get("boundary_fraction",0.0))<.025:return None
    return {"outer_ids":_readonly_array(outer_ids,dtype=np.int64),"outer_faces":_readonly_array(outer_faces,dtype=np.int64),"paired_fraction":coverage,"thickness_p50_mm":p50*1000.0,"thickness_p95_mm":p95*1000.0,"outer_vertex_count":int(len(outer_ids)),"outer_face_count":int(len(outer_faces)),"source_vertex_count":int(len(V)),"source_face_count":int(len(F)),"outer_boundary_fraction":float(outer_feature.get("boundary_fraction",0.0))}


def _thin_volume_surface_data(w: dict[str,Any], thin: dict[str,Any], name: str) -> dict[str,Any]:
    ids=np.asarray(thin["outer_ids"],dtype=np.int64);UV=w.get("UV");N=w.get("N")
    return {"V":np.asarray(w["V"],dtype=np.float64)[ids].copy(),"F":np.asarray(thin["outer_faces"],dtype=np.int64).copy(),"W":np.asarray(w["W"],dtype=np.float64)[ids].copy(),"UV":None if UV is None else np.asarray(UV,dtype=np.float64)[ids].copy(),"N":None if N is None else np.asarray(N,dtype=np.float64)[ids].copy(),"joint_names":list(w.get("joint_names",[])),"material":str(w.get("material","")),"name":name}


def _orthonormal_triangle_frames(triangles: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    T=np.asarray(triangles,dtype=np.float64);tangent=T[:,1]-T[:,0];tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12);normal=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12);bitangent=np.cross(normal,tangent);bitangent/=np.maximum(np.linalg.norm(bitangent,axis=1,keepdims=True),1e-12);return tangent,bitangent,normal


def _reconstruct_thin_volume_from_surface(w: dict[str,Any], thin: dict[str,Any], solved_outer: np.ndarray) -> np.ndarray:
    """Expand a solved garment-facing surface back to its authored source thickness.

    Only the source shell supplies offsets/thickness.  No target geometry participates here.
    """
    V=np.asarray(w["V"],dtype=np.float64);outer_ids=np.asarray(thin["outer_ids"],dtype=np.int64);outer_faces=np.asarray(thin["outer_faces"],dtype=np.int64);Q=np.asarray(solved_outer,dtype=np.float64)
    if Q.shape!=(len(outer_ids),3):raise ValueError(f"Thin-volume solved surface shape {Q.shape} != {(len(outer_ids),3)}")
    source_outer=V[outer_ids];source_tri=source_outer[outer_faces];target_tri=Q[outer_faces];closest,_,_,_,face_index=_b14_nearest_surface(V,source_tri,k=24);face_index=np.asarray(face_index,dtype=np.int64);bary=trimesh.triangles.points_to_barycentric(source_tri[face_index],np.asarray(closest,dtype=np.float64));target_contact=np.einsum("ni,nij->nj",bary,target_tri[face_index]);offset=V-np.asarray(closest,dtype=np.float64)
    se,sb,sn=_orthonormal_triangle_frames(source_tri[face_index]);te,tb,tn=_orthonormal_triangle_frames(target_tri[face_index]);local=np.column_stack((np.einsum("ij,ij->i",offset,se),np.einsum("ij,ij->i",offset,sb),np.einsum("ij,ij->i",offset,sn)))
    out=target_contact+te*local[:,0,None]+tb*local[:,1,None]+tn*local[:,2,None];out[outer_ids]=Q;return out

def _layer_member_virtual_data(source: Any, layer: GarmentLayer, member: GarmentLayerMember) -> dict[str,Any]:
    data=source.data(member.mesh_name);ids=np.asarray(member.vertex_ids,dtype=np.int64)
    Fraw=np.asarray(data["F"],dtype=np.int64)[np.asarray(member.face_ids,dtype=np.int64)]
    remap=np.full(len(data["V"]),-1,dtype=np.int64);remap[ids]=np.arange(len(ids),dtype=np.int64)
    F=remap[Fraw]
    if np.any(F<0):raise ValueError(f"{member.stable_component_id} face remap escaped its authored component.")
    UV=data.get("UV");N=data.get("N")
    return {
        "V":np.asarray(data["V"],dtype=np.float64)[ids].copy(),
        "F":np.asarray(F,dtype=np.int64).copy(),
        "W":np.asarray(data["W"],dtype=np.float64)[ids].copy(),
        "UV":None if UV is None else np.asarray(UV,dtype=np.float64)[ids].copy(),
        "N":None if N is None else np.asarray(N,dtype=np.float64)[ids].copy(),
        "joint_names":list(data.get("joint_names",[])),
        "material":str(data.get("material","")),
        "name":f"{layer.stable_id}/{member.stable_component_id}",
    }


def _layer_geometry_from_source(source: Any, layer: GarmentLayer) -> np.ndarray:
    rows=[]
    for member in layer.members:
        data=source.data(member.mesh_name);rows.append(np.asarray(data["V"],dtype=np.float64)[np.asarray(member.vertex_ids,dtype=np.int64)])
    return np.vstack(rows) if rows else np.zeros((0,3),dtype=np.float64)


def _infer_layer_order_graph(source: Any, layers: list[GarmentLayer], source_body_triangles: np.ndarray) -> tuple[list[GarmentLayer],list[dict[str,Any]]]:
    geometry={layer.stable_id:_layer_geometry_from_source(source,layer) for layer in layers}
    relations=[]
    for ai,a in enumerate(layers):
        # Rigid ornaments do not define garment inside/outside shell ordering. Their placement is
        # independently body-corresponded and frozen; including them here creates false layer edges
        # and quadratic work in ornament-heavy outfits.
        if a.structural_classification=="rigid_assembly":continue
        AV=geometry[a.stable_id];AF=np.asarray(a.faces,dtype=np.int64)
        if not len(AV) or not len(AF):continue
        for b in layers[ai+1:]:
            if b.structural_classification=="rigid_assembly":continue
            # Ordering describes distinct authored layers, not disconnected peers/trims belonging to
            # the same material construction. Same-material components retain their own deformation
            # authority but must never manufacture an inside/outside constraint between each other.
            if set(a.materials)&set(b.materials):continue
            BV=geometry[b.stable_id];BF=np.asarray(b.faces,dtype=np.int64)
            if not len(BV) or not len(BF) or not _expanded_aabb_overlap(AV,BV,.012):continue
            ae=_layer_directional_evidence(AV,BV[BF],source_body_triangles,close_distance=.012)
            be=_layer_directional_evidence(BV,AV[AF],source_body_triangles,close_distance=.012)
            outer=inner=None;OV=IV=None;OF=IF=None
            evidence=None
            # A real authored stack must be evidenced in both directions over a meaningful patch.
            # One-sided proximity is common for straps, trims, buckles and crossing decorations and
            # must not manufacture a global inside/outside relationship. The graph is deliberately
            # conservative: ambiguous overlap remains component-local rather than becoming assembly
            # authority.
            if ae is not None and be is not None and ae["fraction"]>=.08 and be["fraction"]>=.08:
                if ae["positive_fraction"]>=.68 and be["negative_fraction"]>=.68 and ae["median"]>.00010:
                    outer,inner=a,b;OV,OF,IV,IF=AV,AF,BV,BF;evidence=ae
                elif be["positive_fraction"]>=.68 and ae["negative_fraction"]>=.68 and be["median"]>.00010:
                    outer,inner=b,a;OV,OF,IV,IF=BV,BF,AV,AF;evidence=be
            if outer is None:continue
            closest,_,_,distance,inner_face_index=_b14_nearest_surface(OV,IV[IF],k=32)
            inner_face_index=np.asarray(inner_face_index,dtype=np.int64)
            source_inner_tri=IV[IF][inner_face_index]
            inner_barycentric=trimesh.triangles.points_to_barycentric(source_inner_tri,np.asarray(closest,dtype=np.float64))
            _,body_normals,_,_,_=_b14_nearest_surface(OV,source_body_triangles,k=32)
            radial=np.sum((OV-closest)*body_normals,axis=1)
            mask=(distance<=.012)&np.isfinite(radial)&(radial>.00002)
            if int(np.count_nonzero(mask))<max(6,int(len(OV)*.01)):continue
            source_gap=np.zeros(len(OV),dtype=np.float64);source_gap[mask]=radial[mask]
            relation_id=f"{inner.stable_id}->{outer.stable_id}"
            relations.append({
                "id":relation_id,"inner":inner.stable_id,"outer":outer.stable_id,
                "source_gap":_readonly_array(source_gap),
                # Closest-surface correspondence is frozen in source topology. Reconciliation must
                # compare the same authored surface patch after independent fitting rather than
                # performing a new nearest-surface lookup that can jump across folds/openings.
                "inner_face_index":_readonly_array(inner_face_index),
                "inner_barycentric":_readonly_array(np.asarray(inner_barycentric,dtype=np.float64)),
                "source_gap_p50_mm":float(np.median(radial[mask])*1000.0),
                "source_gap_p95_mm":float(np.percentile(radial[mask],95)*1000.0),
                "overlap_vertices":int(np.count_nonzero(mask)),"influence_m":float(np.clip(np.percentile(distance[mask],95)*1.5,.004,.014)),
                "source_directional_median_mm":float((evidence or {}).get("median",0.0)*1000.0),
            })
    relation_ids={layer.stable_id:[] for layer in layers}
    for relation in relations:
        relation_ids[relation["inner"]].append(relation["id"]);relation_ids[relation["outer"]].append(relation["id"])
    layers=[replace(layer,source_ordering_relationships=tuple(sorted(relation_ids[layer.stable_id]))) for layer in layers]
    return layers,relations


def _source_relative_body_penetration_state(source_vertices: np.ndarray, vertices: np.ndarray, source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, tolerance: float=.00002, body_margin_m: float=.00002):
    """Classify target-body penetration relative to the authored source/body relationship.

    A negative nearest-normal sign is not, by itself, evidence that a solved garment has crossed the
    body: concave and bilateral anatomy can legitimately produce the same sign in the untouched
    source.  Source-negative vertices are therefore allowed to retain their authored signed relation;
    only a target result that becomes materially *more* negative is a new penetration.  Vertices that
    were not source-negative retain the normal literal-body rule.
    """
    S=np.asarray(source_vertices,dtype=np.float64);V=np.asarray(vertices,dtype=np.float64)
    if S.shape!=V.shape:raise ValueError(f"Source-relative body validation requires matching geometry, got {S.shape} and {V.shape}.")
    if not len(V):
        return {"count":0,"source_signed":np.zeros(0,dtype=np.float64),"target_signed":np.zeros(0,dtype=np.float64),"target_normals":np.zeros((0,3),dtype=np.float64),"required":np.zeros(0,dtype=np.float64),"authored_negative":np.zeros(0,dtype=bool)}
    _,_,source_signed,_,_=_b14_nearest_surface(S,source_body_triangles,k=32)
    _,target_normals,target_signed,_,_=_b14_nearest_surface(V,target_body_triangles,k=32)
    source_signed=np.asarray(source_signed,dtype=np.float64);target_signed=np.asarray(target_signed,dtype=np.float64);target_normals=np.asarray(target_normals,dtype=np.float64)
    tol=abs(float(tolerance));authored_negative=source_signed<-tol
    # Source-negative points may keep the relationship the source garment was actually authored with.
    # Everything else is expected not to cross the target body's literal zero surface.
    violation_floor=np.where(authored_negative,source_signed-tol,-tol)
    desired=np.where(authored_negative,source_signed,float(body_margin_m))
    violation=target_signed<violation_floor
    required=np.where(violation,np.maximum(0.0,desired-target_signed),0.0)
    return {"count":int(np.count_nonzero(violation)),"source_signed":source_signed,"target_signed":target_signed,"target_normals":target_normals,"required":required,"authored_negative":authored_negative}


def _layer_body_penetration_count(vertices: np.ndarray, target_body_triangles: np.ndarray, tolerance: float=.00002, source_vertices: np.ndarray | None=None, source_body_triangles: np.ndarray | None=None) -> int:
    if source_vertices is not None and source_body_triangles is not None:
        return int(_source_relative_body_penetration_state(source_vertices,vertices,source_body_triangles,target_body_triangles,tolerance=tolerance)["count"])
    if not len(vertices):return 0
    _,_,signed,_,_=_b14_nearest_surface(np.asarray(vertices,dtype=np.float64),target_body_triangles,k=32)
    return int(np.count_nonzero(np.asarray(signed)<-abs(float(tolerance))))


def _layer_relation_mapped_inner(inner: np.ndarray, inner_faces: np.ndarray, relation: dict[str,Any]) -> np.ndarray:
    face_rows=np.asarray(relation.get("inner_face_index"),dtype=np.int64)
    bary=np.asarray(relation.get("inner_barycentric"),dtype=np.float64)
    faces=np.asarray(inner_faces,dtype=np.int64)
    if face_rows.ndim!=1 or bary.shape!=(len(face_rows),3):
        raise ValueError(f"{relation.get('id','layer relation')} has invalid source closest-surface correspondence metadata.")
    if np.any(face_rows<0) or np.any(face_rows>=len(faces)):
        raise ValueError(f"{relation.get('id','layer relation')} source closest-surface face mapping is out of range.")
    return np.einsum("ni,nij->nj",bary,np.asarray(inner,dtype=np.float64)[faces[face_rows]])


def _layer_order_violation_count(outer: np.ndarray, inner: np.ndarray, inner_faces: np.ndarray, target_body_triangles: np.ndarray, relation: dict[str,Any]) -> int:
    if not len(outer) or not len(inner_faces):return 0
    closest=_layer_relation_mapped_inner(inner,inner_faces,relation)
    distance=np.linalg.norm(np.asarray(outer,dtype=np.float64)-closest,axis=1)
    _,body_normals,_,_,_=_b14_nearest_surface(outer,target_body_triangles,k=32)
    radial=np.sum((outer-closest)*body_normals,axis=1);gap=np.asarray(relation["source_gap"],dtype=np.float64)
    eligible=(gap>0)&(distance<=float(relation["influence_m"])*1.25)
    # Source gap is relationship metadata, not a target-space modelling instruction. A violation is
    # therefore a genuine crossing/collapse, not merely a smaller-but-valid target-space gap.
    minimum_gap=np.minimum(gap*.10,.00025)
    return int(np.count_nonzero(eligible&(radial<minimum_gap-.00003)))


def _reconcile_frozen_b14_layers(layers: list[GarmentLayer], frozen: dict[str,B14LayerResult], relations: list[dict[str,Any]], target_body_triangles: np.ndarray, displacement_cap_m: float=.0020, body_margin_m: float=.00002, source_layer_geometry: dict[str,np.ndarray] | None=None, source_body_triangles: np.ndarray | None=None):
    """Tiny bounded post-B14 assembly pass: ordering, source gap, then literal body penetration only."""
    final={layer.stable_id:np.asarray(frozen[layer.stable_id].positions,dtype=np.float64).copy() for layer in layers}
    faces={layer.stable_id:np.asarray(layer.faces,dtype=np.int64) for layer in layers}
    original={layer.stable_id:np.asarray(frozen[layer.stable_id].positions,dtype=np.float64) for layer in layers}
    total={layer.stable_id:np.zeros(len(final[layer.stable_id]),dtype=np.float64) for layer in layers}
    per_layer_order_before={layer.stable_id:0 for layer in layers}
    source_relative_body_validation=source_layer_geometry is not None and source_body_triangles is not None
    per_layer_body_before={layer.stable_id:_layer_body_penetration_count(final[layer.stable_id],target_body_triangles,source_vertices=(source_layer_geometry or {}).get(layer.stable_id) if source_relative_body_validation else None,source_body_triangles=source_body_triangles if source_relative_body_validation else None) for layer in layers}
    for r in relations:
        count=_layer_order_violation_count(final[r["outer"]],final[r["inner"]],faces[r["inner"]],target_body_triangles,r)
        per_layer_order_before[r["outer"]]+=count
    relation_before=sum(per_layer_order_before.values())
    body_before=sum(per_layer_body_before.values())
    unresolved_required_mm=0.0

    # Inner/outer reconciliation: move only the authored outer layer, never refit or smooth either shell.
    for relation in relations:
        outer_id=relation["outer"];inner_id=relation["inner"];OV=final[outer_id];IV=final[inner_id];IF=faces[inner_id]
        closest=_layer_relation_mapped_inner(IV,IF,relation)
        distance=np.linalg.norm(np.asarray(OV,dtype=np.float64)-closest,axis=1)
        _,body_normals,_,_,_=_b14_nearest_surface(OV,target_body_triangles,k=32)
        radial=np.sum((OV-closest)*body_normals,axis=1);gap=np.asarray(relation["source_gap"],dtype=np.float64)
        eligible=(gap>0)&(distance<=float(relation["influence_m"])*1.25)
        # Reconciliation is assembly, never a second fit. Preserve ordering and repair only a
        # collapsed inter-layer clearance. Do not force the source body's full millimetre gap onto
        # different target anatomy. The source distribution determines the collapse floor; movement
        # remains the smallest displacement that restores a physically separate stack.
        minimum_gap=np.minimum(gap*.10,.00025)
        required=np.where(eligible,np.maximum(0.0,minimum_gap-radial),0.0)
        unresolved_required_mm=max(unresolved_required_mm,float(np.max(required,initial=0.0)*1000.0))
        remaining=np.maximum(0.0,float(displacement_cap_m)-total[outer_id])
        move=np.minimum(required,remaining)
        if np.any(move>0):
            final[outer_id]=OV+body_normals*move[:,None];total[outer_id]+=move

    # Literal target-body correction remains tiny and bounded, but penetration classification is
    # source-relative when the authored source/body relationship is available. This prevents a
    # nearest-normal sign flip in concave/bilateral anatomy from being mistaken for a new crossing.
    for layer in layers:
        lid=layer.stable_id;V=final[lid]
        if source_relative_body_validation:
            state=_source_relative_body_penetration_state((source_layer_geometry or {})[lid],V,source_body_triangles,target_body_triangles,tolerance=body_margin_m,body_margin_m=body_margin_m)
            normals=np.asarray(state["target_normals"],dtype=np.float64);required=np.asarray(state["required"],dtype=np.float64)
        else:
            _,normals,signed,_,_=_b14_nearest_surface(V,target_body_triangles,k=32)
            signed=np.asarray(signed,dtype=np.float64);normals=np.asarray(normals,dtype=np.float64)
            required=np.where(signed<0.0,np.maximum(0.0,float(body_margin_m)-signed),0.0)
        unresolved_required_mm=max(unresolved_required_mm,float(np.max(required,initial=0.0)*1000.0))
        remaining=np.maximum(0.0,float(displacement_cap_m)-total[lid]);move=np.minimum(required,remaining)
        if np.any(move>0):
            final[lid]=V+normals*move[:,None];total[lid]+=move

    per_layer_order_after={layer.stable_id:0 for layer in layers}
    per_layer_body_after={layer.stable_id:_layer_body_penetration_count(final[layer.stable_id],target_body_triangles,source_vertices=(source_layer_geometry or {}).get(layer.stable_id) if source_relative_body_validation else None,source_body_triangles=source_body_triangles if source_relative_body_validation else None) for layer in layers}
    for r in relations:
        count=_layer_order_violation_count(final[r["outer"]],final[r["inner"]],faces[r["inner"]],target_body_triangles,r)
        per_layer_order_after[r["outer"]]+=count
    relation_after=sum(per_layer_order_after.values())
    body_after=sum(per_layer_body_after.values())
    layer_reports={}
    warnings=[]
    for layer in layers:
        lid=layer.stable_id;delta=np.linalg.norm(final[lid]-original[lid],axis=1)
        moved=delta>1e-12
        p50=float(np.percentile(delta,50)*1000.0) if len(delta) else 0.0
        p95=float(np.percentile(delta,95)*1000.0) if len(delta) else 0.0
        maximum=float(np.max(delta,initial=0.0)*1000.0)
        warning=p95>1.0 or maximum>=float(displacement_cap_m)*1000.0*.90
        if warning:warnings.append(f"{lid}: reconciliation p95={p95:.3f} mm max={maximum:.3f} mm")
        layer_reports[lid]={"reconciled_vertices":int(np.count_nonzero(moved)),"p50_mm":p50,"p95_mm":p95,"max_mm":maximum,"warning":warning,
                            "layer_order_violations_before":int(per_layer_order_before[lid]),"layer_order_violations_after":int(per_layer_order_after[lid]),
                            "body_penetrations_before":int(per_layer_body_before[lid]),"body_penetrations_after":int(per_layer_body_after[lid])}
    if unresolved_required_mm>float(displacement_cap_m)*1000.0+1e-9:
        warnings.append(f"B14 layer solve requires {unresolved_required_mm:.3f} mm of assembly correction, beyond the {displacement_cap_m*1000.0:.3f} mm hard cap; treat this solve as failed.")
    if relation_after:
        warnings.append(f"{relation_after} source layer-order violations remain after bounded reconciliation; treat this solve as failed.")
    if body_after:
        warnings.append(f"{body_after} target-body penetrations remain after bounded reconciliation; treat this solve as failed.")
    report={
        "policy":"bounded relationship-only reconciliation; no post-B14 macro/body fitting",
        "source_relative_body_validation":bool(source_relative_body_validation),
        "displacement_cap_mm":float(displacement_cap_m*1000.0),
        "layer_order_violations_before":int(relation_before),"layer_order_violations_after":int(relation_after),
        "body_penetrations_before":int(body_before),"body_penetrations_after":int(body_after),
        "largest_unbounded_requested_correction_mm":float(unresolved_required_mm),
        "passed":bool(relation_after==0 and body_after==0 and unresolved_required_mm<=float(displacement_cap_m)*1000.0+1e-9),
        "layers":layer_reports,"warnings":warnings,
    }
    final={layer_id:_readonly_array(value) for layer_id,value in final.items()}
    return final,report


def _scatter_layer_positions_to_meshes(source: Any, layers: list[GarmentLayer], layer_positions: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    out={}
    for layer in layers:
        candidate=np.asarray(layer_positions[layer.stable_id],dtype=np.float64);offset=0
        for member in layer.members:
            ids=np.asarray(member.vertex_ids,dtype=np.int64);count=len(ids)
            if member.mesh_name not in out:out[member.mesh_name]=np.asarray(source.data(member.mesh_name)["V"],dtype=np.float64).copy()
            out[member.mesh_name][ids]=candidate[offset:offset+count];offset+=count
        if offset!=len(candidate):raise ValueError(f"{layer.stable_id} scatter map consumed {offset} of {len(candidate)} vertices.")
    return out


def _weld_positions_from_raw(w: dict[str,Any], raw_positions: np.ndarray) -> np.ndarray:
    raw=np.asarray(raw_positions,dtype=np.float64);mapping=np.asarray(w["raw_to_weld"],dtype=np.int64);n=len(w["V"])
    if len(raw)!=len(mapping):raise ValueError(f"Raw position count {len(raw)} does not match weld map {len(mapping)}")
    out=np.zeros((n,3),dtype=np.float64);count=np.zeros(n,dtype=np.float64);np.add.at(out,mapping,raw);np.add.at(count,mapping,1.0)
    return out/np.maximum(count[:,None],1.0)


def _apply_b14_local_structural_retarget(source: Any, positions: dict[str,np.ndarray], cache: dict[str,Any], source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, source_asset_has_embedded_body: bool=False) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Apply the universal coherent refit to already-good Frozen B14 geometry.

    Construction/detail remains source/B14 authority.  The target body contributes only a smooth,
    continuous source->target deformation field whose influence is determined from the authored
    source-body clearance.  Spatially attached disconnected pieces share that same field, so a
    strap/ring/trim cannot wander away from the garment assembly.  Literal body geometry is used
    only as distributed collision authority.
    """
    X=np.asarray(cache["X"],dtype=np.float64);Y=np.asarray(cache["Y"],dtype=np.float64);BW=np.asarray(cache["BW"],dtype=np.float64)
    local_affines=cache.get("_ravafit_local_affines")
    if local_affines is None:
        local_affines=precompute_body_local_affines(X,Y,BW);cache["_ravafit_local_affines"]=local_affines
    A,_,tree=local_affines
    body_motion=np.linalg.norm(Y-X,axis=1)
    body_motion_p95=float(np.percentile(body_motion,95)) if len(body_motion) else 0.0
    body_macro_transform=fit_body_macro_transform(X,Y) if len(X)>=4 else None
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};reports={};changed_meshes=[]
    for name in sorted(out):
        data=source.data(name);w=weld_mesh(data);b14_w=_weld_positions_from_raw(w,out[name])
        Wg,_=solver_body_weights(w["V"],X,BW,tree)
        correspondence=local_body_field_map_soft(w["V"],Wg,X,Y,BW,A,tree=tree,tau=.004)[0]
        corrected,report=coherent_universal_refit(
            w["V"],w["F"],b14_w,correspondence,source_body_triangles,target_body_triangles,
            body_motion_p95_m=body_motion_p95,nearest_surface_fn=_b14_nearest_surface,macro_transform=body_macro_transform,source_weights=w.get("W"),allow_authored_standoff_noop=not bool(source_asset_has_embedded_body))
        corrected,ribbon_report=retarget_attached_ribbons_to_assembly(
            w["V"],w["F"],b14_w,correspondence,corrected,source_body_triangles,target_body_triangles,
            nearest_surface_fn=_b14_nearest_surface,collision_polish_fn=_b14_residual_collision_polish)
        combined_move=np.linalg.norm(np.asarray(corrected,dtype=np.float64)-b14_w,axis=1)
        report["attached_ribbon_retarget"]=ribbon_report
        report["changed_vertex_count"]=int(np.count_nonzero(combined_move>1e-12))
        report["displacement_p95_mm"]=float(np.percentile(combined_move,95)*1000.0) if len(combined_move) else 0.0
        report["displacement_max_mm"]=float(np.max(combined_move,initial=0.0)*1000.0)
        reports[name]=report
        if int(report.get("changed_vertex_count",0))<=0:continue
        raw=expand_welded(w,corrected)
        if raw.shape!=out[name].shape:raise ValueError(f"{name}: universal coherent refit expansion produced {raw.shape}, expected {out[name].shape}")
        out[name]=np.asarray(raw,dtype=np.float64);changed_meshes.append(name)
    return out,{"enabled":True,"policy":"same authored garment, one continuous source-clearance-weighted body field, distributed literal collision","body_motion_p95_mm":body_motion_p95*1000.0,"changed_meshes":changed_meshes,"changed_mesh_count":len(changed_meshes),"meshes":reports}


def _apply_final_local_shell_bridge(source: Any, positions: dict[str,np.ndarray], source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, target_support_triangles: np.ndarray | None=None, max_passes: int=3, detail_gate_m: float=.00075, cumulative_cap_m: float=.00320) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Bridge only literal target relief that is absent from the smooth target support surface.

    The target support surface owns macro fit (breast volume/position, hips, abdomen, etc.).  The literal
    target body is collision evidence only for local surface relief.  A post-fit bridge is therefore
    allowed to alter B14 geometry only where the literal target body materially departs from the smooth
    target support surface *and* the source-aware bridge detector independently identifies new local
    curvature.  This prevents repeated cleanup passes from flattening genuine target macro shape or
    source-authored cup/apex relief.
    """
    if not positions:return {},{"enabled":False,"reason":"no garment meshes"}
    literal=np.asarray(target_body_triangles,dtype=np.float64)
    support=np.asarray(target_support_triangles,dtype=np.float64) if target_support_triangles is not None else np.zeros((0,3,3),dtype=np.float64)
    have_support=bool(support.ndim==3 and support.shape[1:]==(3,3) and len(support))
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};mesh_reports={};changed=[]
    for name in sorted(out):
        data=source.data(name);w=weld_mesh(data);source_w=np.asarray(w["V"],dtype=np.float64);faces=np.asarray(w["F"],dtype=np.int64)
        current=_weld_positions_from_raw(w,out[name]);initial=current.copy();passes=[]
        neighbours=[set() for _ in range(len(current))]
        if len(faces):
            for a,b in np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0):
                neighbours[int(a)].add(int(b));neighbours[int(b)].add(int(a))
        for pass_index in range(max(1,int(max_passes))):
            candidate,report=_bridge_new_local_curvature(source_w,faces,current,source_body_triangles,literal,nearest_surface_fn=_b14_nearest_surface,config=None)
            candidate=np.asarray(candidate,dtype=np.float64);step=np.linalg.norm(candidate-current,axis=1);step_mask=step>1e-12
            raw_changed=int(np.count_nonzero(step_mask));detail_seed_count=0;detail_p95=0.0
            if raw_changed and have_support:
                ids=np.flatnonzero(step_mask)
                literal_cp,_,_,_,_=_nearest_surface_reference_chunked(current[ids],literal,k=32)
                support_cp,_,_,_,_=_nearest_surface_reference_chunked(current[ids],support,k=32)
                detail=np.linalg.norm(np.asarray(literal_cp)-np.asarray(support_cp),axis=1)
                detail_p95=float(np.percentile(detail,95)*1000.0) if len(detail) else 0.0
                seeds=ids[detail>=float(detail_gate_m)];detail_seed_count=int(len(seeds))
                if not len(seeds):
                    candidate=current.copy();step_mask[:]=False
                else:
                    allowed=np.zeros(len(current),dtype=bool);allowed[seeds]=True;front=set(int(i) for i in seeds.tolist())
                    # Keep the bridge spatially smooth around target-only detail, but do not allow it to
                    # spread into unrelated macro shell/cup regions.
                    for _ in range(2):
                        nxt=set()
                        for vi in front:nxt.update(int(nb) for nb in neighbours[vi])
                        nxt={vi for vi in nxt if step_mask[vi] and not allowed[vi]}
                        if not nxt:break
                        allowed[list(nxt)]=True;front=nxt
                    candidate[~allowed]=current[~allowed]
                    step_mask=allowed&(np.linalg.norm(candidate-current,axis=1)>1e-12)
            elif raw_changed and not have_support:
                # Compatibility fallback: retain the historical detector, but still enforce the hard
                # cumulative cap below. Production strict-B14 supplies a support surface.
                detail_seed_count=raw_changed

            if np.any(step_mask):
                cumulative_vec=candidate-initial;cumulative_mag=np.linalg.norm(cumulative_vec,axis=1)
                over=cumulative_mag>float(cumulative_cap_m)
                if np.any(over):
                    cumulative_vec[over]*=(float(cumulative_cap_m)/np.maximum(cumulative_mag[over],1e-12))[:,None]
                    candidate=initial+cumulative_vec
                step=np.linalg.norm(candidate-current,axis=1);changed_count=int(np.count_nonzero(step>1e-12))
            else:
                candidate=current.copy();step=np.zeros(len(current),dtype=np.float64);changed_count=0
            passes.append({"pass":pass_index+1,"raw_bridge_changed_vertex_count":raw_changed,"target_only_detail_seed_vertices":detail_seed_count,"literal_vs_support_detail_p95_mm":detail_p95,"changed_vertex_count":changed_count,"step_p95_mm":float(np.percentile(step[step>1e-12],95)*1000.0) if changed_count else 0.0,"step_max_mm":float(np.max(step,initial=0.0)*1000.0),"bridge":report})
            if changed_count<=0:break
            current=candidate
        cumulative=np.linalg.norm(current-initial,axis=1);mesh_changed=int(np.count_nonzero(cumulative>1e-12))
        mesh_reports[name]={"passes":passes,"pass_count":len(passes),"changed_vertex_count":mesh_changed,"cumulative_p95_mm":float(np.percentile(cumulative[cumulative>1e-12],95)*1000.0) if mesh_changed else 0.0,"cumulative_max_mm":float(np.max(cumulative,initial=0.0)*1000.0)}
        if mesh_changed:
            raw=expand_welded(w,current)
            if raw.shape!=out[name].shape:raise ValueError(f"{name}: final local shell bridge produced {raw.shape}, expected {out[name].shape}")
            out[name]=np.asarray(raw,dtype=np.float64);changed.append(name)
    return out,{"enabled":bool(changed),"policy":"source-aware local bridge may act only where literal target relief departs from smooth target macro support; source-authored and target-macro relief are protected","max_passes":int(max_passes),"target_only_detail_gate_mm":float(detail_gate_m*1000.0),"cumulative_vertex_cap_mm":float(cumulative_cap_m*1000.0),"changed_meshes":changed,"changed_mesh_count":len(changed),"meshes":mesh_reports}


def _preserve_final_unilateral_pair_separation(source: Any, positions: dict[str,np.ndarray], *, preserve_proximal_structure: bool = True) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Reassert source-proven left/right leg-component separation after all body/layer fitting.

    Production requires unilateral leg-skinning evidence so mirrored non-leg structures (cups, gloves,
    shoulder ornaments, etc.) cannot enter this lane on geometry symmetry alone.  Connectivity and gap
    measurements are performed on welded render topology, then expanded back to authored raw storage.

    ``preserve_proximal_structure=False`` is used only for the final post-carrier guard: at that point
    raw authored bands/cuffs have already inherited their coherent solved carrier frame, so the welded
    bilateral authority owns the centreline separation floor only and must not reshape those structures.
    """
    if not positions:return {},{"enabled":False,"reason":"no garment meshes"}
    cfg=UnilateralPairSeparationConfig(require_leg_evidence=True,preserve_proximal_structure=bool(preserve_proximal_structure))
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};mesh_reports={};changed=[]
    total_pairs=0;total_moved=0;maximum_correction=0.0
    for name in sorted(out):
        data=source.data(name);w=weld_mesh(data);source_w=np.asarray(w.get("V",[]),dtype=np.float64);faces=np.asarray(w.get("F",[]),dtype=np.int64);weights=np.asarray(w.get("W",[]),dtype=np.float64);joint_names=list(data.get("joint_names") or w.get("joint_names") or [])
        if not len(source_w) or not len(faces) or weights.ndim!=2 or len(weights)!=len(source_w) or weights.shape[1]!=len(joint_names):
            mesh_reports[name]={"enabled":False,"reason":"missing welded topology or compatible source skinning","pair_count":0,"moved_vertices":0};continue
        left_columns,right_columns=_leg_joint_columns(joint_names)
        if not left_columns or not right_columns:
            mesh_reports[name]={"enabled":False,"reason":"no unilateral leg-joint columns","pair_count":0,"moved_vertices":0};continue
        current_w=_weld_positions_from_raw(w,out[name]);left_mass=weights[:,left_columns].sum(axis=1);right_mass=weights[:,right_columns].sum(axis=1)
        corrected,report=preserve_source_unilateral_pair_separation(source_w,current_w,faces,left_leg_mass=left_mass,right_leg_mass=right_mass,config=cfg)
        mesh_reports[name]=report;total_pairs+=int(report.get("pair_count",0));total_moved+=int(report.get("moved_vertices",0));maximum_correction=max(maximum_correction,float(report.get("maximum_correction_mm",0.0)))
        if int(report.get("moved_vertices",0))<=0:continue
        raw=expand_welded(w,corrected)
        if raw.shape!=out[name].shape:raise ValueError(f"{name}: unilateral pair separation expansion produced {raw.shape}, expected {out[name].shape}")
        _assert_finite_stage(name,"final unilateral pair separation",raw);out[name]=np.asarray(raw,dtype=np.float64);changed.append(name)
    policy=("source-proven disconnected unilateral leg pairs preserve target-scaled authored centreline separation plus proximal structure after fitting" if preserve_proximal_structure else "source-proven disconnected unilateral leg pairs preserve the target-scaled authored centreline separation floor only after coherent structural-carrier placement")
    return out,{"enabled":bool(total_pairs),"policy":policy,"preserve_proximal_structure":bool(preserve_proximal_structure),"pair_count":int(total_pairs),"moved_vertices":int(total_moved),"maximum_correction_mm":float(maximum_correction),"changed_meshes":changed,"changed_mesh_count":len(changed),"meshes":mesh_reports}



def _preserve_final_surface_relative_detail_layout(source: Any, positions: dict[str,np.ndarray]) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Keep compact disconnected authored details in the local frame of the solved garment beneath them.

    The carrier is selected entirely from untouched source connectivity/proximity.  Source geometry
    supplies layout/shape only; the already-solved carrier supplies target placement and scale.
    """
    if not positions:return {},{"enabled":False,"reason":"no garment meshes"}
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};names=[];source_parts=[];current_parts=[];face_parts=[];ranges={};offset=0
    for name in sorted(out):
        data=source.data(name);w=weld_mesh(data);source_w=np.asarray(w.get("V",[]),dtype=np.float64);faces=np.asarray(w.get("F",[]),dtype=np.int64)
        if not len(source_w) or not len(faces):continue
        current_w=_weld_positions_from_raw(w,out[name]);start=offset;end=start+len(source_w);ranges[name]=(start,end,w);names.append(name);source_parts.append(source_w);current_parts.append(current_w);face_parts.append(faces+start);offset=end
    if not source_parts:return out,{"enabled":False,"reason":"no welded garment topology"}
    source_all=np.vstack(source_parts);current_all=np.vstack(current_parts);faces_all=np.vstack(face_parts)
    corrected,report=preserve_surface_relative_detail_layout(source_all,current_all,faces_all,config=SurfaceRelativeDetailConfig())
    corrected=np.asarray(corrected,dtype=np.float64);changed=[]
    for name in names:
        start,end,w=ranges[name];piece=corrected[start:end];before=current_all[start:end]
        if np.max(np.linalg.norm(piece-before,axis=1),initial=0.0)<=1.0e-12:continue
        raw=expand_welded(w,piece)
        if raw.shape!=out[name].shape:raise ValueError(f"{name}: surface-relative detail layout expansion produced {raw.shape}, expected {out[name].shape}")
        _assert_finite_stage(name,"surface-relative detail layout",raw);out[name]=np.asarray(raw,dtype=np.float64);changed.append(name)
    report=dict(report);report["changed_meshes"]=changed;report["changed_mesh_count"]=len(changed)
    return out,report


def _preserve_final_source_relative_structural_carriers(source: Any, positions: dict[str,np.ndarray]) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Preserve raw authored carried bands and compact multi-piece assemblies in solved carrier frames.

    Unlike attachment/seam logic, this authority intentionally uses the untouched *raw render topology*.
    A decorative cuff or nested ornament can be authored as a disconnected source component whose seam
    vertices happen to occupy the same coordinates as its host. Welding would erase that authored
    structural boundary and make the very band we need to preserve indistinguishable from cloth.
    Concatenating raw meshes still permits cross-mesh proximity clustering without using item names.
    """
    if not positions:return {},{"enabled":False,"reason":"no garment meshes"}
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};names=[];source_parts=[];current_parts=[];face_parts=[];group_parts=[];ranges={};offset=0
    for group_index,name in enumerate(sorted(out)):
        data=source.data(name);source_v=np.asarray(data.get("V",[]),dtype=np.float64);faces=np.asarray(data.get("F",[]),dtype=np.int64);current=np.asarray(out[name],dtype=np.float64)
        if not len(source_v) or not len(faces):continue
        if current.shape!=source_v.shape:raise ValueError(f"{name}: structural carrier raw geometry mismatch source={source_v.shape}, current={current.shape}")
        start=offset;end=start+len(source_v);ranges[name]=(start,end);names.append(name);source_parts.append(source_v);current_parts.append(current);face_parts.append(faces+start);group_parts.append(np.full(len(source_v),group_index,dtype=np.int64));offset=end
    if not source_parts:return out,{"enabled":False,"reason":"no raw garment topology"}
    source_all=np.vstack(source_parts);current_all=np.vstack(current_parts);faces_all=np.vstack(face_parts);group_all=np.concatenate(group_parts)
    corrected,report=preserve_source_relative_structural_carriers(source_all,current_all,faces_all,config=StructuralCarrierConfig(),vertex_group_ids=group_all)
    corrected=np.asarray(corrected,dtype=np.float64);changed=[]
    for name in names:
        start,end=ranges[name];piece=np.asarray(corrected[start:end],dtype=np.float64);before=current_all[start:end]
        if np.max(np.linalg.norm(piece-before,axis=1),initial=0.0)<=1.0e-12:continue
        if piece.shape!=out[name].shape:raise ValueError(f"{name}: structural carrier raw slice produced {piece.shape}, expected {out[name].shape}")
        _assert_finite_stage(name,"source-relative raw structural carrier preservation",piece);out[name]=piece;changed.append(name)
    report=dict(report);report["changed_meshes"]=changed;report["changed_mesh_count"]=len(changed);report["topology_authority"]="untouched raw render connectivity (no welding)"
    return out,report


def _preserve_support_stable_components(source: Any, positions: dict[str,np.ndarray], cache: dict[str,Any], source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, *, min_component_vertices: int=180, displacement_gate_m: float=.012, support_gate_m: float=.010, displacement_support_ratio: float=2.4, allowed_worsening_m: float=.00020) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Preserve exact source geometry for components whose local body support is effectively unchanged.

    This is intentionally *regional* evidence, not a slot-wide shortcut.  If a connected garment
    component moves far more than the source->target body support directly under it, and keeping the
    untouched source component does not materially worsen the exact source-relative target-body audit,
    then the body solver is overreaching.  Preserve that component exactly instead of inventing a new
    refit.  This catches long sleeve/boot-style false positives while leaving genuinely body-coupled
    regions under normal target-fit authority.
    """
    if not positions:
        return {}, {"enabled":False,"reason":"no garment meshes"}
    X=np.asarray(cache["X"],dtype=np.float64);Y=np.asarray(cache["Y"],dtype=np.float64);BW=np.asarray(cache["BW"],dtype=np.float64)
    local_affines=cache.get("_ravafit_local_affines")
    if local_affines is None:
        local_affines=precompute_body_local_affines(X,Y,BW);cache["_ravafit_local_affines"]=local_affines
    A,_,tree=local_affines
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()}
    mesh_reports={};changed_meshes=[];total_components=0;preserved_components=0;total_vertices=0;max_component_motion=0.0
    tol=2.0e-5
    for name in sorted(out):
        data=source.data(name)
        S=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64);P=np.asarray(out[name],dtype=np.float64)
        if len(S)==0 or len(F)==0 or S.shape!=P.shape:
            mesh_reports[name]={"enabled":False,"reason":"missing geometry or mismatched shape"}
            continue
        try:
            Wg,_=solver_body_weights(S,X,BW,tree)
            support=np.asarray(local_body_field_map_soft(S,Wg,X,Y,BW,A,tree=tree,tau=.004)[0],dtype=np.float64)
        except Exception as exc:
            mesh_reports[name]={"enabled":False,"reason":f"support field failed: {exc}"}
            continue
        support_motion=np.linalg.norm(support-S,axis=1)
        solved_motion=np.linalg.norm(P-S,axis=1)
        source_state=_source_relative_body_penetration_state(S,S,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
        current_state=_source_relative_body_penetration_state(S,P,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
        source_required=np.asarray(source_state["required"],dtype=np.float64)
        current_required=np.asarray(current_state["required"],dtype=np.float64)
        comps=_masked_vertex_components(F,np.ones(len(S),dtype=bool))
        comp_reports=[];mesh_changed=False
        for comp_index,ids in enumerate(comps):
            ids=np.asarray(ids,dtype=np.int64);total_components+=1
            comp_report={"component":int(comp_index),"vertex_count":int(len(ids))}
            if len(ids)<int(min_component_vertices):
                comp_report.update({"status":"skipped_small"});comp_reports.append(comp_report);continue
            comp_support=support_motion[ids];comp_solved=solved_motion[ids]
            support_p95=float(np.percentile(comp_support,95)) if len(comp_support) else 0.0
            solved_p95=float(np.percentile(comp_solved,95)) if len(comp_solved) else 0.0
            support_p50=float(np.percentile(comp_support,50)) if len(comp_support) else 0.0
            solved_p50=float(np.percentile(comp_solved,50)) if len(comp_solved) else 0.0
            source_req_max=float(np.max(source_required[ids],initial=0.0))
            current_req_max=float(np.max(current_required[ids],initial=0.0))
            worsened_vertices=int(np.count_nonzero(source_required[ids]>current_required[ids]+float(allowed_worsening_m)+1e-12))
            exact_source_safe=source_req_max<=current_req_max+float(allowed_worsening_m)+1e-12 and worsened_vertices==0
            overreacting=solved_p95>=max(float(displacement_gate_m), support_p95*float(displacement_support_ratio))
            support_stable=support_p95<=float(support_gate_m)
            comp_report.update({
                "support_motion_p50_mm":support_p50*1000.0,
                "support_motion_p95_mm":support_p95*1000.0,
                "solved_displacement_p50_mm":solved_p50*1000.0,
                "solved_displacement_p95_mm":solved_p95*1000.0,
                "source_required_max_mm":source_req_max*1000.0,
                "current_required_max_mm":current_req_max*1000.0,
                "worsened_vertices_if_source_preserved":int(worsened_vertices),
                "support_stable":bool(support_stable),
                "exact_source_safe":bool(exact_source_safe),
                "overreacting":bool(overreacting),
            })
            if not (support_stable and exact_source_safe and overreacting):
                reason=[]
                if not support_stable:reason.append("support_changed")
                if not exact_source_safe:reason.append("source_worsens_body_relation")
                if not overreacting:reason.append("deformation_not_disproportionate")
                comp_report["status"]="skipped_"+"+".join(reason)
                comp_reports.append(comp_report)
                continue
            moved=np.linalg.norm(P[ids]-S[ids],axis=1)
            out[name][ids]=S[ids]
            mesh_changed=True;preserved_components+=1;total_vertices+=int(np.count_nonzero(moved>1e-12));max_component_motion=max(max_component_motion,float(np.max(moved,initial=0.0)))
            comp_report.update({"status":"preserved_exact_source","moved_vertex_count":int(np.count_nonzero(moved>1e-12)),"maximum_reverted_motion_mm":float(np.max(moved,initial=0.0)*1000.0)})
            comp_reports.append(comp_report)
        mesh_reports[name]={"enabled":bool(mesh_changed),"component_count":len(comps),"preserved_component_count":sum(1 for row in comp_reports if row.get("status")=="preserved_exact_source"),"components":comp_reports}
        if mesh_changed:changed_meshes.append(name)
    return out,{"enabled":bool(changed_meshes),"policy":"connected components whose local source->target support field changes little, whose exact source geometry stays body-safe, and whose current deformation is disproportionate are preserved exactly as source authority","changed_meshes":changed_meshes,"changed_mesh_count":len(changed_meshes),"preserved_component_count":int(preserved_components),"preserved_vertex_count":int(total_vertices),"maximum_reverted_motion_mm":float(max_component_motion*1000.0),"min_component_vertices":int(min_component_vertices),"displacement_gate_mm":float(displacement_gate_m*1000.0),"support_gate_mm":float(support_gate_m*1000.0),"displacement_support_ratio":float(displacement_support_ratio),"allowed_body_worsening_mm":float(allowed_worsening_m*1000.0),"meshes":mesh_reports}


def _veto_worsened_source_relative_body_penetration(source: Any, before_positions: dict[str,np.ndarray], proposed_positions: dict[str,np.ndarray], source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, tolerance: float=.00002, allowed_worsening_m: float=.00002) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Keep post-fit structural authorities from making an already-safer body relationship worse.

    The normal B14/body fit remains authoritative.  Later source-structure authorities may preserve an
    opening, attachment, or carried detail, but they are not allowed to introduce a new source-relative
    target-body crossing or materially deepen one that already existed.  Only the structural-authority
    displacement is backtracked; the underlying target refit is never replaced by source geometry.

    Per-vertex safe fractions are found on the segment from the pre-authority solve to the proposed
    structural result, then conservatively diffused over raw mesh edges so the veto itself cannot make
    isolated dents.  This is deliberately generic and topology/body-evidence based.
    """
    if not proposed_positions:
        return {}, {"enabled":False,"reason":"no garment meshes"}
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in proposed_positions.items()}
    reports={};total_restricted=0;total_new_before=0;total_new_after=0;max_reverted=0.0
    tol=abs(float(tolerance));worsen=max(0.0,float(allowed_worsening_m))
    for name in sorted(out):
        if name not in before_positions:
            continue
        data=source.data(name);S=np.asarray(data["V"],dtype=np.float64);B=np.asarray(before_positions[name],dtype=np.float64);A=np.asarray(out[name],dtype=np.float64)
        if S.shape!=B.shape or B.shape!=A.shape:
            raise ValueError(f"{name}: collision-veto geometry mismatch source={S.shape}, before={B.shape}, proposed={A.shape}")
        before_state=_source_relative_body_penetration_state(S,B,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
        proposed_state=_source_relative_body_penetration_state(S,A,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
        before_required=np.asarray(before_state["required"],dtype=np.float64);proposed_required=np.asarray(proposed_state["required"],dtype=np.float64)
        allowed=before_required+worsen
        coherent_band_rescues=[]
        # A source-proven broad band may already be a perfect carrier-relative similarity when the
        # body veto sees it.  Per-vertex backtracking would make that safe by bending the band back out
        # of shape.  First try to keep the band coherent and restore its source-relative standoff by
        # expanding only its two broad axes around the already-solved carrier centre.  This changes
        # neither its bend/orientation nor its thin-axis profile and is accepted only when the exact
        # source-relative body audit proves the whole component safe.
        faces=np.asarray(data.get("F",np.zeros((0,3),dtype=np.int64)),dtype=np.int64)
        initial_worsened=proposed_required>allowed+1e-12
        if len(faces) and np.any(initial_worsened):
            cfg=StructuralCarrierConfig()
            for ids in _masked_vertex_components(faces,np.ones(len(S),dtype=bool)):
                ids=np.asarray(ids,dtype=np.int64);n=len(ids)
                if n<int(cfg.band_min_vertices) or n>int(cfg.band_max_vertices) or not np.any(initial_worsened[ids]):continue
                extent=np.ptp(S[ids],axis=0);ordered=np.sort(np.asarray(extent,dtype=np.float64))
                if ordered[2]<float(cfg.band_min_major_extent_m) or ordered[1]<float(cfg.band_min_middle_extent_m) or ordered[0]>float(cfg.band_max_minor_extent_m):continue
                if ordered[0]/max(float(ordered[1]),1e-12)>float(cfg.band_max_minor_to_middle_ratio):continue
                coherent,base_scale=_similarity_fit_points(S[ids],A[ids],scale_min=.55,scale_max=1.55)
                coherence_error=np.linalg.norm(coherent-A[ids],axis=1);coherence_p95=float(np.percentile(coherence_error,95))
                if coherence_p95>.00035:continue
                centre=np.mean(coherent,axis=0);centred=coherent-centre
                try:_,_,vt=np.linalg.svd(centred,full_matrices=False)
                except np.linalg.LinAlgError:continue
                if len(vt)<3:continue
                minor=np.asarray(vt[-1],dtype=np.float64);minor/=max(float(np.linalg.norm(minor)),1e-12)
                axial=(centred@minor)[:,None]*minor[None,:];broad=centred-axial
                def expanded(factor: float) -> np.ndarray:
                    return centre+axial+broad*float(factor)
                safe_factor=None;safe_points=None;prev_factor=1.0
                # Search only a bounded structural expansion.  If the required standoff needs more,
                # fall back to the conservative vertex veto rather than inventing garment scale.
                for factor in np.linspace(1.0,1.35,36):
                    Q=expanded(float(factor))
                    qstate=_source_relative_body_penetration_state(S[ids],Q,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
                    qrequired=np.asarray(qstate["required"],dtype=np.float64)
                    if np.all(qrequired<=allowed[ids]+1e-12):
                        lo=prev_factor;hi=float(factor);best=Q
                        for _ in range(8):
                            mid=(lo+hi)*.5;M=expanded(mid)
                            mstate=_source_relative_body_penetration_state(S[ids],M,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
                            if np.all(np.asarray(mstate["required"],dtype=np.float64)<=allowed[ids]+1e-12):hi=mid;best=M
                            else:lo=mid
                        safe_factor=hi;safe_points=best;break
                    prev_factor=float(factor)
                if safe_points is None:continue
                max_delta=float(np.max(np.linalg.norm(safe_points-A[ids],axis=1),initial=0.0))
                if max_delta>.030:continue
                A[ids]=safe_points
                coherent_band_rescues.append({
                    "vertex_count":int(n),"base_similarity_scale":float(base_scale),"broad_plane_expansion":float(safe_factor),
                    "coherence_error_p95_mm":coherence_p95*1000.0,"maximum_body_safe_adjustment_mm":max_delta*1000.0,
                    "status":"coherent_broad_band_body_safe",
                })
            if coherent_band_rescues:
                proposed_state=_source_relative_body_penetration_state(S,A,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
                proposed_required=np.asarray(proposed_state["required"],dtype=np.float64)
        worsened=proposed_required>allowed+1e-12
        alpha=np.ones(len(A),dtype=np.float64)
        restricted=np.flatnonzero(worsened)
        # Find the largest structural-authority fraction that retains the pre-authority body relationship.
        for idx in restricted.tolist():
            lo=0.0;hi=1.0;src=S[idx:idx+1];base=B[idx:idx+1];delta=(A-B)[idx:idx+1];limit=float(allowed[idx])
            for _ in range(10):
                mid=(lo+hi)*.5
                state=_source_relative_body_penetration_state(src,base+delta*mid,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
                req=float(np.asarray(state["required"],dtype=np.float64)[0])
                if req<=limit+1e-12:lo=mid
                else:hi=mid
            alpha[idx]=lo
        # Lower neighbouring authority gradually rather than leaving sharp one-vertex reversions.
        if len(restricted) and len(faces):
            edges=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
            edges=np.sort(edges,axis=1);edges=np.unique(edges,axis=0)
            for _ in range(4):
                prev=alpha.copy()
                for i,j in edges.tolist():
                    alpha[i]=min(alpha[i],prev[j]+.22);alpha[j]=min(alpha[j],prev[i]+.22)
        candidate=B+(A-B)*alpha[:,None]
        # After the safety veto, broad source-proven bands can end up safe but slightly slanted because
        # neighbouring vertices retained different fractions of the structural authority.  Project such a
        # candidate back to its nearest coherent source-relative similarity *only when* the exact body
        # audit proves that the straighter result remains safe.  This restores levelness/planarity without
        # giving the band permission to float off the fitted host leg.
        if len(faces):
            cfg=StructuralCarrierConfig()
            for ids in _masked_vertex_components(faces,np.ones(len(S),dtype=bool)):
                ids=np.asarray(ids,dtype=np.int64);n=len(ids)
                if n<int(cfg.band_min_vertices) or n>int(cfg.band_max_vertices) or not np.any(alpha[ids] < 1.0-1e-12):
                    continue
                extent=np.ptp(S[ids],axis=0);ordered=np.sort(np.asarray(extent,dtype=np.float64))
                if ordered[2]<float(cfg.band_min_major_extent_m) or ordered[1]<float(cfg.band_min_middle_extent_m) or ordered[0]>float(cfg.band_max_minor_extent_m):
                    continue
                if ordered[0]/max(float(ordered[1]),1e-12)>float(cfg.band_max_minor_to_middle_ratio):
                    continue
                coherent,_=_similarity_fit_points(S[ids],candidate[ids],scale_min=.55,scale_max=1.55)
                pre_err=np.linalg.norm(candidate[ids]-coherent,axis=1)
                pre_p95=float(np.percentile(pre_err,95)) if len(pre_err) else 0.0
                if pre_p95<=0.00035:
                    continue
                sim_state=_source_relative_body_penetration_state(S[ids],coherent,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
                sim_required=np.asarray(sim_state["required"],dtype=np.float64)
                if np.all(sim_required<=allowed[ids]+1e-12):
                    delta=np.linalg.norm(coherent-candidate[ids],axis=1)
                    if float(np.max(delta,initial=0.0))<=0.015:
                        candidate[ids]=coherent
        # A final exact audit is authoritative.  Any rare non-monotonic nearest-surface case that still
        # worsens is returned all the way to the already-fitted pre-authority position.
        final_state=_source_relative_body_penetration_state(S,candidate,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
        final_required=np.asarray(final_state["required"],dtype=np.float64)
        still=final_required>allowed+1e-12
        if np.any(still):
            candidate[still]=B[still];alpha[still]=0.0
            final_state=_source_relative_body_penetration_state(S,candidate,source_body_triangles,target_body_triangles,tolerance=tol,body_margin_m=tol)
            final_required=np.asarray(final_state["required"],dtype=np.float64)
        changed=alpha<1.0-1e-12
        displacement=np.linalg.norm(A-candidate,axis=1)
        out[name]=candidate
        introduced_before=int(np.count_nonzero((proposed_required>allowed+1e-12)))
        introduced_after=int(np.count_nonzero(final_required>allowed+1e-12))
        total_restricted+=int(np.count_nonzero(changed));total_new_before+=introduced_before;total_new_after+=introduced_after
        max_reverted=max(max_reverted,float(np.max(displacement,initial=0.0)))
        reports[name]={
            "before_penetrations":int(before_state["count"]),
            "proposed_penetrations":int(proposed_state["count"]),
            "final_penetrations":int(final_state["count"]),
            "materially_worsened_before_veto":introduced_before,
            "materially_worsened_after_veto":introduced_after,
            "restricted_vertex_count":int(np.count_nonzero(changed)),
            "hard_reverted_vertex_count":int(np.count_nonzero(still)),
            "authority_fraction_p50":float(np.median(alpha[changed])) if np.any(changed) else 1.0,
            "authority_fraction_min":float(np.min(alpha,initial=1.0)),
            "veto_displacement_p95_mm":float(np.percentile(displacement[changed],95)*1000.0) if np.any(changed) else 0.0,
            "veto_displacement_max_mm":float(np.max(displacement,initial=0.0)*1000.0),
            "coherent_band_body_safe_rescues":coherent_band_rescues,
            "coherent_band_body_safe_rescue_count":int(len(coherent_band_rescues)),
        }
    return out,{
        "enabled":True,
        "policy":"post-fit structural authority may preserve source structure only while not materially worsening the source-relative target-body relationship",
        "allowed_worsening_mm":worsen*1000.0,
        "restricted_vertex_count":int(total_restricted),
        "materially_worsened_before_veto":int(total_new_before),
        "materially_worsened_after_veto":int(total_new_after),
        "veto_displacement_max_mm":max_reverted*1000.0,
        "meshes":reports,
    }


def _preserve_final_source_attachment_continuity(source: Any, positions: dict[str,np.ndarray]) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Run the source-proven attachment closure only after all layer reconciliation is final.

    Each mesh is welded before concatenation so duplicate raw storage/seam rows cannot manufacture
    fake disconnected components.  Concatenating the welded meshes also lets real attachments cross
    source mesh boundaries.  The corrected welded result is expanded back into the original raw
    storage layout without changing topology or any non-position attribute.
    """
    if not positions:return {},{"enabled":False,"reason":"no garment meshes"}
    names=sorted(positions);parts=[];faces=[];meta=[];offset=0
    for name in names:
        data=source.data(name);w=weld_mesh(data);source_w=np.asarray(w["V"],dtype=np.float64);current_w=_weld_positions_from_raw(w,np.asarray(positions[name],dtype=np.float64));F=np.asarray(w["F"],dtype=np.int64)
        parts.append(source_w);faces.append(F+offset if len(F) else F);meta.append((name,w,offset,len(source_w),current_w));offset+=len(source_w)
    if not parts:return {name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()},{"enabled":False,"reason":"no welded garment vertices"}
    source_all=np.vstack(parts);current_all=np.vstack([row[4] for row in meta]);faces_all=np.vstack(faces) if any(len(x) for x in faces) else np.zeros((0,3),dtype=np.int64)
    corrected,report=preserve_source_proven_attachment_continuity(source_all,faces_all,current_all)
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};changed=[]
    for name,w,start,count,_ in meta:
        solved=np.asarray(corrected[start:start+count],dtype=np.float64);raw=expand_welded(w,solved)
        if raw.shape!=out[name].shape:raise ValueError(f"{name}: final attachment closure expansion produced {raw.shape}, expected {out[name].shape}")
        if not np.array_equal(raw,out[name]):changed.append(name)
        out[name]=raw
    report=dict(report);report["changed_meshes"]=changed;report["changed_mesh_count"]=len(changed);report["cross_mesh_capable"]=True
    return out,report



def _target_fit_collision_triangles(cache: dict[str,Any]) -> np.ndarray:
    """Use complete target anatomy for fitting, independently of render cut-outs.

    Source-authored body removal is an output visibility decision. Its open edges
    are not a signed collision solid: using them can classify cloth outside a
    breast as deeply inside its cut rim. The strict contract retains the original
    indexed anatomical material separately from hair/accessory package meshes.
    Those optional open surfaces must not deform the clothing either.
    """
    cached = cache.get("_ravafit_target_collision_triangles")
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)
    vertices = cache.get("_ravafit_strict_target_surface_V")
    faces = cache.get("_ravafit_strict_target_surface_F")
    if vertices is None or faces is None:
        vertices, faces = cache["target_surface_V"], cache["target_surface_F"]
    triangles = _triangles_from_surface(vertices, faces)
    triangles, _ = _sanitise_strict_b14_surface_triangles(triangles, "complete target anatomy")
    cache["_ravafit_target_collision_triangles"] = triangles
    return triangles


def _strict_source_literal_body_triangles(cache: dict[str,Any], fallback: np.ndarray) -> np.ndarray:
    """Use the body geometry that actually existed in the source outfit as final contact authority.

    A modded outfit may deliberately delete anatomy under a garment.  The catalogue source body is still
    useful for correspondence, but it must never resurrect that deleted anatomy at the end of the solve.
    The suppression plan therefore publishes the embedded source-body triangles it actually observed.
    """
    suppression=cache.get("_ravafit_source_body_suppression") or {}
    if "_source_collision_triangles" in suppression:
        actual=np.asarray(suppression.get("_source_collision_triangles"),dtype=np.float64)
        if actual.ndim==3 and actual.shape[1:]==(3,3):
            if len(actual)==0:return actual
            actual,_=_sanitise_strict_b14_surface_triangles(actual,"embedded source final-clearance")
            return actual
    triangles=[]
    for pair in cache.get("slot_pairs",[]):
        vertices=np.asarray(pair.get("source_literal_V",[]),dtype=np.float64)
        faces=np.asarray(pair.get("source_literal_F",[]),dtype=np.int64)
        if vertices.ndim!=2 or vertices.shape[1:]!=(3,) or faces.ndim!=2 or faces.shape[1:]!=(3,) or len(vertices)<3 or len(faces)==0:
            continue
        if int(np.min(faces,initial=0))<0 or int(np.max(faces,initial=-1))>=len(vertices):continue
        triangles.append(vertices[faces])
    if not triangles:return np.asarray(fallback,dtype=np.float64)
    combined=np.vstack(triangles);combined,_=_sanitise_strict_b14_surface_triangles(combined,"source literal final-clearance")
    return combined


def _final_target_body_clearance(source: Any, positions: dict[str,np.ndarray], target_body_triangles: np.ndarray, target_support_triangles: np.ndarray, margin_m: float=.00010, maximum_vertex_move_m: float=.00800, maximum_step_m: float=.00150, max_passes: int=8) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Clear the target while source-proven seam copies share one displacement."""
    kwargs=dict(margin_m=margin_m,maximum_vertex_move_m=maximum_vertex_move_m,
                maximum_step_m=maximum_step_m,max_passes=max_passes)
    pairs,discovery=_source_proven_cross_mesh_seam_pairs(source,positions,include_same_mesh=True)
    if not pairs:
        return _clear_target_body_surfaces(source,positions,target_body_triangles,target_support_triangles,**kwargs)
    joined,seams=_preserve_final_source_shared_seams(source,positions)
    names=sorted(joined);starts={};source_parts=[];current_parts=[];faces=[];offset=0
    for name in names:
        data=source.data(name);S=np.asarray(data["V"],dtype=np.float64);F=np.asarray(data["F"],dtype=np.int64)
        starts[name]=offset;source_parts.append(S);current_parts.append(joined[name]);faces.append(F+offset);offset+=len(S)
    S=np.vstack(source_parts);P=np.vstack(current_parts);F=np.vstack(faces);parent=np.arange(len(S))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    for an,ai,bn,bi,_ in pairs:
        a=root(starts[an]+ai);b=root(starts[bn]+bi)
        if a!=b:parent[b]=a
    roots=np.asarray([root(i) for i in range(len(S))],dtype=np.int64)
    _,inverse=np.unique(roots,return_inverse=True);count=np.bincount(inverse).astype(np.float64)
    source_centres=np.zeros((len(count),3));displacement=np.zeros_like(source_centres)
    np.add.at(source_centres,inverse,S);source_centres/=count[:,None]
    np.add.at(displacement,inverse,P-S);displacement/=count[:,None]
    centres=source_centres+displacement;authored_offsets=S-source_centres[inverse]
    proxy_faces=inverse[F]
    proxy_faces=proxy_faces[(proxy_faces[:,0]!=proxy_faces[:,1])&(proxy_faces[:,1]!=proxy_faces[:,2])&(proxy_faces[:,2]!=proxy_faces[:,0])]
    class SeamSurface:
        def data(self,name):return {"V":source_centres,"F":proxy_faces}
    # Near-coincident authored seams retain their tiny offsets on expansion; this
    # extra clearance bounds the maximum difference from the shared solve surface.
    kwargs["margin_m"]+=float(np.max(np.linalg.norm(authored_offsets,axis=1),initial=0.))
    solved,report=_clear_target_body_surfaces(SeamSurface(),{"seam_surface":centres},target_body_triangles,target_support_triangles,**kwargs)
    expanded=solved["seam_surface"][inverse]+authored_offsets
    out={name:expanded[starts[name]:starts[name]+len(joined[name])].copy() for name in names}
    report["source_seams"]={"discovery":discovery,"closure":seams,"shared_displacement":True}
    report["changed_meshes"]=[name for name in names if np.any(np.linalg.norm(out[name]-positions[name],axis=1)>1e-12)]
    report["changed_mesh_count"]=len(report["changed_meshes"])
    report["maximum_move_mm"]=max(float(np.max(np.linalg.norm(out[name]-positions[name],axis=1),initial=0.))*1000 for name in names)
    return out,report


def _clear_target_body_surfaces(source: Any, positions: dict[str,np.ndarray], target_body_triangles: np.ndarray, target_support_triangles: np.ndarray, margin_m: float=.00010, maximum_vertex_move_m: float=.00800, maximum_step_m: float=.00150, max_passes: int=8) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Repair sampled residual garment penetration against the selected target body without embossing it.

    The selected target body is final fit/occupancy authority even where the untouched source outfit had
    body geometry removed.  Literal target geometry answers only one question here: *is cloth inside the
    target?*  The accumulated correction is spread over a physical neighbourhood of connected cloth,
    relative to the already-fitted garment normals. This also handles anatomy embedded in the main
    body material, where the nominal target support and literal body are the same surface.

    Nearby clear cloth can follow a contact to bridge small relief. Unaffected components and authored
    garment geometry stay intact: this filters collision displacement, not the garment positions.
    """
    if not positions:return {},{"enabled":False,"reason":"no garment meshes"}
    target_tri=np.asarray(target_body_triangles,dtype=np.float64);support_tri=np.asarray(target_support_triangles,dtype=np.float64)
    if target_tri.ndim!=3 or target_tri.shape[1:]!=(3,3) or len(target_tri)==0:
        return {name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()},{"enabled":False,"reason":"no target collision surface"}
    have_support=bool(support_tri.ndim==3 and support_tri.shape[1:]==(3,3) and len(support_tri))
    # Vertices, edge midpoints and the centroid miss narrow crossings between those probes.
    # Include the quarter-grid interiors as well; final MDL topology can put a body ridge
    # directly underneath one of these previously untested parts of a garment triangle.
    bary=np.asarray([(i/4.,j/4.,(4-i-j)/4.) for i in range(5) for j in range(5-i)]+[(1./3.,1./3.,1./3.)],dtype=np.float64)
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()};reports={};changed=[];total_faces=0;maximum_move=0.0;minimum_after=None;unresolved_total=0;initial_minimum=None
    for name in sorted(out):
        data=source.data(name);F=np.asarray(data.get("F",[]),dtype=np.int64);P=np.asarray(out[name],dtype=np.float64)
        if P.ndim!=2 or P.shape[1:]!=(3,) or F.ndim!=2 or F.shape[1:]!=(3,) or len(F)==0 or int(np.max(F,initial=-1))>=len(P):
            reports[name]={"enabled":False,"reason":"no compatible indexed garment surface"};continue
        original=P.copy();V=P.copy();affected=set();opposite_rejections=0;bisector_pushes=0;iterations=0;initial_penetrating_faces=0
        edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0);neighbours=[[] for _ in range(len(V))]
        for a,b in edges:neighbours[int(a)].append(int(b));neighbours[int(b)].append(int(a))
        base_area=np.cross(original[F[:,1]]-original[F[:,0]],original[F[:,2]]-original[F[:,0]]);base_norm=np.linalg.norm(base_area,axis=1);valid_area=base_norm>1e-12
        for pass_index in range(max(1,int(max_passes))):
            grid=np.einsum("bk,fkj->fbj",bary,V[F]);samples=grid.reshape(-1,3)
            _,literal_normals,signed,_,_,rejections=_nearest_literal_surface_consistent_with_support(samples,target_tri,support_tri if have_support else None,exact_base=(pass_index==max(1,int(max_passes))-1))
            opposite_rejections+=int(rejections);signed=np.asarray(signed,dtype=np.float64).reshape(len(F),len(bary));literal_normals=np.asarray(literal_normals,dtype=np.float64).reshape(len(F),len(bary),3)
            if pass_index==0:
                initial_minimum=float(np.min(signed))*1000.0 if initial_minimum is None else min(initial_minimum,float(np.min(signed))*1000.0)
                initial_penetrating_faces=int(np.count_nonzero(np.any(signed<-.00002,axis=1)))
            deficit=np.maximum(0.0,float(margin_m)-signed);deficit[signed>=-.00002]=0.0
            bad_rows=np.flatnonzero(np.max(deficit,axis=1)>1e-8)
            if len(bad_rows)==0:break
            iterations=pass_index+1;affected.update(int(x) for x in bad_rows.tolist())
            chosen=np.argmax(deficit[bad_rows],axis=1);sample_points=grid[bad_rows,chosen];literal=np.asarray(literal_normals[bad_rows,chosen],dtype=np.float64)
            if have_support:
                _,support_normals,_,_,_=_nearest_surface_reference_chunked(sample_points,support_tri,k=32);support_normals=np.asarray(support_normals,dtype=np.float64)
                push_normals,bisector_report=_clearance_push_normals_with_bisector(support_normals,literal,signed[bad_rows,chosen]);bisector_pushes+=int(bisector_report.get("used",0))
                flip=np.einsum("ij,ij->i",push_normals,literal)<-.05;push_normals[flip]*=-1.0
            else:push_normals=literal.copy()
            push_normals/=np.maximum(np.linalg.norm(push_normals,axis=1)[:,None],1e-12)
            need=np.minimum(deficit[bad_rows,chosen],float(maximum_step_m))
            acc=np.zeros_like(V);weight=np.zeros(len(V),dtype=np.float64);vertex_ids=F[bad_rows];corner_weights=.30+.70*bary[chosen]
            contrib=push_normals[:,None,:]*need[:,None,None]*corner_weights[:,:,None];np.add.at(acc,vertex_ids.reshape(-1),contrib.reshape(-1,3));np.add.at(weight,vertex_ids.reshape(-1),corner_weights.reshape(-1))
            direct=weight>0;field=np.zeros_like(V);field[direct]=acc[direct]/weight[direct,None]
            # Build a smooth cloth-support envelope.  Literal anatomy determines the required amount at
            # the collision core; neighbouring cloth follows a rapidly decaying garment-topology field.
            seen=set(int(i) for i in np.flatnonzero(direct).tolist());front=set(seen)
            for scale in (.42,.18):
                nxt=set()
                for vi in front:nxt.update(int(nb) for nb in neighbours[vi] if int(nb) not in seen)
                if not nxt:break
                for vi in nxt:
                    refs=[nb for nb in neighbours[vi] if float(np.linalg.norm(field[nb]))>0.0]
                    if refs:field[vi]=float(scale)*np.mean(field[refs],axis=0)
                seen.update(nxt);front=nxt
            trial=V+field
            cumulative=trial-original;mag=np.linalg.norm(cumulative,axis=1);over=mag>float(maximum_vertex_move_m)
            if np.any(over):cumulative[over]*=(float(maximum_vertex_move_m)/np.maximum(mag[over],1e-12))[:,None]
            trial=original+cumulative
            # Never accept an inverted/collapsed local triangle.  Halve the local step a few times before
            # giving up on those vertices so deep target penetrations can still converge smoothly.
            local_step=trial-V
            for _ in range(4):
                trial_area=np.cross(trial[F[:,1]]-trial[F[:,0]],trial[F[:,2]]-trial[F[:,0]]);trial_norm=np.linalg.norm(trial_area,axis=1);dot=np.ones(len(F));ratio=np.ones(len(F));dot[valid_area]=np.einsum("ij,ij->i",base_area[valid_area],trial_area[valid_area])/np.maximum(base_norm[valid_area]*trial_norm[valid_area],1e-24);ratio[valid_area]=trial_norm[valid_area]/np.maximum(base_norm[valid_area],1e-12);unsafe=valid_area&((dot<=.05)|(ratio<=.12))
                if not np.any(unsafe):break
                unsafe_vertices=np.unique(F[unsafe].reshape(-1));local_step[unsafe_vertices]*=.5;trial=V+local_step
            if np.any(unsafe):
                unsafe_vertices=np.unique(F[unsafe].reshape(-1));trial[unsafe_vertices]=V[unsafe_vertices]
            if float(np.max(np.linalg.norm(trial-V,axis=1),initial=0.0))<1e-10:break
            V=trial
        envelope, envelope_report = clearance_envelope(original, F, V-original,
                                                        maximum_move_m=float(maximum_vertex_move_m))
        V, envelope_report["topology"] = accept_envelope_step(V, F, original+envelope)
        # Audit a denser grid after the envelope. Subsequent local moves must
        # include adjacent faces in the recheck because they share moved corners.
        audit_bary=np.asarray([(i/8.,j/8.,(8-i-j)/8.) for i in range(9) for j in range(9-i)]+[(1./3.,1./3.,1./3.)],dtype=np.float64)
        micro_iterations=0
        audit_grid=np.einsum("bk,fkj->fbj",audit_bary,V[F]);_,_,audit_signed,_,_,audit_rejections=_nearest_literal_surface_consistent_with_support(audit_grid.reshape(-1,3),target_tri,support_tri if have_support else None,exact_base=True);opposite_rejections+=int(audit_rejections);audit_signed=np.asarray(audit_signed,dtype=np.float64).reshape(len(F),len(audit_bary))
        micro_faces=np.flatnonzero(np.any(audit_signed<float(margin_m),axis=1))
        for micro_index in range(12):
            if len(micro_faces)==0:break
            local_grid=np.einsum("bk,fkj->fbj",audit_bary,V[F[micro_faces]]);_,local_literal_normals,local_signed,_,_,local_rejections=_nearest_literal_surface_consistent_with_support(local_grid.reshape(-1,3),target_tri,support_tri if have_support else None,exact_base=True);opposite_rejections+=int(local_rejections);local_signed=np.asarray(local_signed,dtype=np.float64).reshape(len(micro_faces),len(audit_bary));local_literal_normals=np.asarray(local_literal_normals,dtype=np.float64).reshape(len(micro_faces),len(audit_bary),3)
            deficit=np.maximum(0.0,float(margin_m)-local_signed);bad=np.flatnonzero(np.max(deficit,axis=1)>1e-8)
            if len(bad)==0:break
            micro_iterations=micro_index+1;rows=micro_faces[bad];chosen=np.argmax(deficit[bad],axis=1);points=local_grid[bad,chosen];literal_n=local_literal_normals[bad,chosen]
            if have_support:
                _,support_normals,_,_,_=_nearest_surface_reference_chunked(points,support_tri,k=32);support_normals=np.asarray(support_normals,dtype=np.float64);push_normals,_=_clearance_push_normals_with_bisector(support_normals,literal_n,local_signed[bad,chosen]);flip=np.einsum("ij,ij->i",push_normals,literal_n)<-.05;push_normals[flip]*=-1.0
            else:push_normals=np.asarray(literal_n,dtype=np.float64).copy()
            push_normals/=np.maximum(np.linalg.norm(push_normals,axis=1)[:,None],1e-12)
            # A small deterministic overshoot avoids asymptotically hovering a few microns inside due
            # shared-vertex averaging while remaining far below visible garment-shape scale.
            need=np.minimum(deficit[bad,chosen]*1.20+.000015,.00075)
            acc=np.zeros_like(V);weight=np.zeros(len(V),dtype=np.float64);vertex_ids=F[rows];corner_weights=.45+.55*audit_bary[chosen];contrib=push_normals[:,None,:]*need[:,None,None]*corner_weights[:,:,None];np.add.at(acc,vertex_ids.reshape(-1),contrib.reshape(-1,3));np.add.at(weight,vertex_ids.reshape(-1),corner_weights.reshape(-1));direct=weight>0;field=np.zeros_like(V);field[direct]=acc[direct]/weight[direct,None]
            trial=V+field;cumulative=trial-original;mag=np.linalg.norm(cumulative,axis=1);over=mag>float(maximum_vertex_move_m)
            if np.any(over):cumulative[over]*=(float(maximum_vertex_move_m)/np.maximum(mag[over],1e-12))[:,None]
            trial=original+cumulative
            trial,_=accept_envelope_step(V,F,trial)
            if float(np.max(np.linalg.norm(trial-V,axis=1),initial=0.0))<1e-10:break
            V=trial;affected.update(int(x) for x in rows.tolist())
            micro_faces=np.union1d(micro_faces,np.flatnonzero(np.any(direct[F],axis=1)))
            local_grid=np.einsum("bk,fkj->fbj",audit_bary,V[F[micro_faces]]);_,_,local_signed,_,_,local_rejections=_nearest_literal_surface_consistent_with_support(local_grid.reshape(-1,3),target_tri,support_tri if have_support else None,exact_base=True);opposite_rejections+=int(local_rejections);local_signed=np.asarray(local_signed,dtype=np.float64).reshape(len(micro_faces),len(audit_bary));micro_faces=micro_faces[np.any(local_signed<float(margin_m)-1e-8,axis=1)]
        final_grid=np.einsum("bk,fkj->fbj",audit_bary,V[F]).reshape(-1,3);_,_,final_signed,_,_,final_rejections=_nearest_literal_surface_consistent_with_support(final_grid,target_tri,support_tri if have_support else None,exact_base=True);opposite_rejections+=int(final_rejections);final_signed=np.asarray(final_signed,dtype=np.float64).reshape(len(F),len(audit_bary));remaining=np.any(final_signed<-.00002,axis=1);unresolved=int(np.count_nonzero(remaining));unresolved_total+=unresolved
        after=float(np.min(final_signed)) if final_signed.size else None;moved=np.linalg.norm(V-original,axis=1);move_max=float(np.max(moved,initial=0.0))
        if move_max>1e-12:out[name]=V;changed.append(name)
        total_faces+=len(affected);maximum_move=max(maximum_move,move_max)
        if after is not None and np.isfinite(after):minimum_after=after*1000.0 if minimum_after is None else min(minimum_after,after*1000.0)
        reports[name]={"enabled":True,"initial_penetrating_faces":int(initial_penetrating_faces),"affected_faces":int(len(affected)),"changed_vertex_count":int(np.count_nonzero(moved>1e-12)),"maximum_move_mm":move_max*1000.0,"iterations":int(iterations),"micro_iterations":int(micro_iterations),"unresolved_penetrating_faces":unresolved,"opposite_facing_literal_rejections":opposite_rejections,"bisector_pushes":bisector_pushes,"sample_min_after_mm":after*1000.0 if after is not None else None,"cloth_envelope":envelope_report}
    return out,{"enabled":True,"policy":"selected target body is final occupancy authority; connected cloth spreads collision displacement at physical scale without smoothing authored geometry; source-body holes never suppress target clearance","penetration_clearance_mm":float(margin_m*1000.0),"maximum_vertex_move_mm":float(maximum_vertex_move_m*1000.0),"maximum_step_mm":float(maximum_step_m*1000.0),"max_passes":int(max_passes),"changed_meshes":changed,"changed_mesh_count":len(changed),"affected_face_count":int(total_faces),"unresolved_penetrating_face_count":int(unresolved_total),"maximum_move_mm":float(maximum_move*1000.0),"minimum_sample_before_mm":initial_minimum,"minimum_contact_sample_after_mm":minimum_after,"meshes":reports}


def _final_source_authored_body_clearance(source: Any, positions: dict[str,np.ndarray], source_body_triangles: np.ndarray, target_body_triangles: np.ndarray, target_support_triangles: np.ndarray, margin_m: float=.00010, maximum_vertex_move_m: float=.00800) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Compatibility wrapper for the pre-1.1.9 name; source-body geometry no longer gates target fit."""
    return _final_target_body_clearance(source,positions,target_body_triangles,target_support_triangles,margin_m=margin_m,maximum_vertex_move_m=maximum_vertex_move_m)


def _source_proven_cross_mesh_seam_pairs(source: Any, positions: dict[str,np.ndarray], tolerance_m: float=.000075, minimum_pair_witnesses: int=3, *, include_same_mesh: bool=False) -> tuple[list[tuple[str,int,str,int,np.ndarray]],dict[str,Any]]:
    """Discover repeated source seams, optionally including split boundaries within a mesh."""
    names=sorted(positions)
    if not names or (len(names)<2 and not include_same_mesh):
        return [],{"candidate_pair_count":0,"accepted_pair_count":0,"mesh_pair_witnesses":{}}
    source_parts=[];owners=[];locals_=[];boundaries=[];mesh_edges={}
    for owner,name in enumerate(names):
        vertices=np.asarray(source.data(name).get("V",[]),dtype=np.float64)
        current=np.asarray(positions[name],dtype=np.float64)
        if vertices.shape!=current.shape or vertices.ndim!=2 or vertices.shape[1:]!=(3,):
            continue
        boundary=np.zeros(len(vertices),dtype=bool)
        faces=np.asarray(source.data(name).get("F",[]),dtype=np.int64)
        if faces.ndim==2 and faces.shape[1:]==(3,) and len(faces):
            edges=np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
            edges,counts=np.unique(edges,axis=0,return_counts=True)
            boundary[np.unique(edges[counts==1])]=True
            mesh_edges[owner]=set(map(tuple,edges.tolist()))
        source_parts.append(vertices);owners.extend([owner]*len(vertices));locals_.extend(range(len(vertices)));boundaries.extend(boundary.tolist())
    if not source_parts:
        return [],{"candidate_pair_count":0,"accepted_pair_count":0,"mesh_pair_witnesses":{}}
    source_all=np.vstack(source_parts);owners=np.asarray(owners,dtype=np.int64);locals_=np.asarray(locals_,dtype=np.int64);boundaries=np.asarray(boundaries,dtype=bool)
    tree=cKDTree(source_all)
    raw=list(tree.query_pairs(r=max(float(tolerance_m),1e-9)))
    cross=[];counts={}
    for a,b in raw:
        oa=int(owners[a]);ob=int(owners[b])
        if oa==ob and (not include_same_mesh or not (boundaries[a] and boundaries[b])):continue
        if oa==ob and (int(locals_[a]),int(locals_[b])) in mesh_edges.get(oa,set()):continue
        if oa>ob:a,b=b,a;oa,ob=ob,oa
        key=(oa,ob);counts[key]=counts.get(key,0)+1;cross.append((int(a),int(b),key))
    eligible={key for key,count in counts.items() if int(count)>=max(2,int(minimum_pair_witnesses))}
    result=[]
    for a,b,key in cross:
        if key not in eligible:continue
        oa=int(owners[a]);ob=int(owners[b]);sa=source_all[a];sb=source_all[b]
        result.append((names[oa],int(locals_[a]),names[ob],int(locals_[b]),np.asarray(sa-sb,dtype=np.float64)))
    witness_report={f"{names[a]} <-> {names[b]}":int(count) for (a,b),count in sorted(counts.items())}
    return result,{"candidate_pair_count":int(len(cross)),"accepted_pair_count":int(len(result)),"accepted_mesh_pair_count":int(len(eligible)),"mesh_pair_witnesses":witness_report,"source_tolerance_mm":float(tolerance_m*1000.0),"minimum_pair_witnesses":int(minimum_pair_witnesses)}


def _preserve_final_source_shared_seams(source: Any, positions: dict[str,np.ndarray], tolerance_m: float=.000075, minimum_pair_witnesses: int=3) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Keep source-proven seams joined across meshes and split boundaries within one mesh.

    Independent layer fitting is intentionally allowed, but two source meshes that share many
    effectively-coincident boundary witnesses are one authored construction at that boundary. Their
    tiny source offset is retained while all copies share one displacement. A local two-ring feather keeps
    the correction from producing a hard kink.  This is intentionally much stricter than generic
    proximity: isolated contacts never become seam authority.
    """
    if not positions:
        return {},{"enabled":False,"reason":"no garment meshes"}
    pairs,discovery=_source_proven_cross_mesh_seam_pairs(source,positions,tolerance_m=tolerance_m,minimum_pair_witnesses=minimum_pair_witnesses,include_same_mesh=True)
    if not pairs:
        return {name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()},{"enabled":False,"reason":"no repeated source-proven cross-mesh seam","discovery":discovery}
    out={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()}
    target_sum={name:np.zeros_like(out[name]) for name in out};target_weight={name:np.zeros(len(out[name]),dtype=np.float64) for name in out}
    before_gaps=[];parent={}
    def root(node):
        parent.setdefault(node,node)
        while parent[node]!=node:
            parent[node]=parent[parent[node]];node=parent[node]
        return node
    for an,ai,bn,bi,source_delta in pairs:
        pa=np.asarray(out[an][ai],dtype=np.float64);pb=np.asarray(out[bn][bi],dtype=np.float64)
        before_gaps.append(float(np.linalg.norm((pa-pb)-source_delta)))
        ar=root((an,ai));br=root((bn,bi))
        if ar!=br:parent[br]=ar
    groups={}
    for node in parent:groups.setdefault(root(node),[]).append(node)
    # All copies of an authored seam share one displacement. Pairwise midpoint
    # averaging leaves a gap when a junction has three or more storage copies.
    for group in groups.values():
        displacement=np.mean([out[name][idx]-np.asarray(source.data(name)["V"])[idx] for name,idx in group],axis=0)
        for name,idx in group:
            target_sum[name][idx]=np.asarray(source.data(name)["V"])[idx]+displacement
            target_weight[name][idx]=1.0
    changed=[];rejected=[];per_mesh={}
    for name in sorted(out):
        current=out[name];weights=target_weight[name];anchors=weights>0
        if not np.any(anchors):continue
        target=current.copy();target[anchors]=target_sum[name][anchors]/weights[anchors,None]
        anchor_field=target-current;field=np.zeros_like(current);field[anchors]=anchor_field[anchors]
        data=source.data(name);S=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64)
        if F.ndim==2 and F.shape[1:] == (3,) and len(F):
            edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0);neighbours=[[] for _ in range(len(current))]
            for a,b in edges:neighbours[int(a)].append(int(b));neighbours[int(b)].append(int(a))
            seen=set(int(i) for i in np.flatnonzero(anchors));front=set(seen)
            for ring,scale in ((1,.40),(2,.16)):
                nxt=set()
                for vi in front:nxt.update(int(nb) for nb in neighbours[vi] if int(nb) not in seen)
                if not nxt:break
                for vi in nxt:
                    refs=[nb for nb in neighbours[vi] if nb in seen and float(np.linalg.norm(field[nb]))>0.0]
                    if refs:field[vi]=np.mean(field[refs],axis=0)*scale
                seen.update(nxt);front=nxt
        magnitude=np.linalg.norm(field,axis=1);non_anchor=~anchors
        field[non_anchor]*=np.minimum(1.0,.004/np.maximum(magnitude[non_anchor],1e-12))[:,None]
        candidate=current+field
        # Exact seam vertices retain the source relation; the feather is expendable if it harms topology.
        before_topology=_source_relative_topology_summary(S,current,F) if len(S)==len(current) and F.ndim==2 and F.shape[1:] == (3,) and len(F) else None
        after_topology=_source_relative_topology_summary(S,candidate,F) if before_topology is not None else None
        unsafe=bool(before_topology is not None and after_topology is not None and (
            after_topology["flip_fraction"]>before_topology["flip_fraction"]+.0025
            or after_topology["area_p01"]<min(before_topology["area_p01"]*.55,.20)
            or after_topology["edge_p99"]>max(before_topology["edge_p99"]*1.75,3.0)))
        status="preserved"
        if unsafe:
            # Feathering is optional; the source-proven seam itself is not.  Falling back to exact
            # seam anchors prevents independent mesh/layer solves from leaving a visible split.
            candidate=current.copy();candidate[anchors]=target[anchors]
            after_topology=_source_relative_topology_summary(S,candidate,F)
            anchor_warning=bool(after_topology["flip_fraction"]>before_topology["flip_fraction"]+.0025 or after_topology["area_p01"]<min(before_topology["area_p01"]*.45,.15))
            status="preserved_hard_anchor_topology_warning" if anchor_warning else "preserved_anchor_only_topology_guard"
            if anchor_warning:rejected.append(name)  # diagnostic only: geometry is still seam-correct.
        moved=np.linalg.norm(candidate-current,axis=1);out[name]=candidate;changed.append(name)
        per_mesh[name]={"status":status,"seam_vertices":int(np.count_nonzero(anchors)),"changed_vertices":int(np.count_nonzero(moved>1e-12)),"move_p95_mm":float(np.percentile(moved[moved>1e-12],95)*1000.0) if np.any(moved>1e-12) else 0.0,"move_max_mm":float(np.max(moved,initial=0.0)*1000.0),"topology_before":before_topology,"topology_after":after_topology}
    after_gaps=[]
    for an,ai,bn,bi,source_delta in pairs:
        after_gaps.append(float(np.linalg.norm((out[an][ai]-out[bn][bi])-source_delta)))
    return out,{
        "enabled":True,
        "policy":"garment-only source seam witnesses retain their authored relative position after independent fitting; body geometry is not an input",
        "discovery":discovery,
        "changed_meshes":changed,
        "topology_warning_meshes":rejected,
        "seam_error_p50_before_mm":float(np.median(before_gaps)*1000.0) if before_gaps else 0.0,
        "seam_error_p95_before_mm":float(np.percentile(before_gaps,95)*1000.0) if before_gaps else 0.0,
        "seam_error_p50_after_mm":float(np.median(after_gaps)*1000.0) if after_gaps else 0.0,
        "seam_error_p95_after_mm":float(np.percentile(after_gaps,95)*1000.0) if after_gaps else 0.0,
        "meshes":per_mesh,
    }


def _gather_layer_positions_from_meshes(layers: list[GarmentLayer], positions: dict[str,np.ndarray]) -> dict[str,B14LayerResult]:
    """Rebuild layer-coordinate arrays after universal refit so authored layer ordering can be revalidated."""
    out={}
    for layer in layers:
        parts=[]
        for member in layer.members:
            ids=np.asarray(member.vertex_ids,dtype=np.int64)
            parts.append(np.asarray(positions[member.mesh_name],dtype=np.float64)[ids])
        V=np.vstack(parts) if parts else np.zeros((0,3),dtype=np.float64)
        out[layer.stable_id]=B14LayerResult(layer.stable_id,_readonly_array(V),"universal_coherent_refit",float(layer.source_clearance_median_mm),float(layer.source_clearance_p95_mm),0.0,{"mode":"universal_coherent_refit"})
    return out

def _preserve_source_shared_seam_skinning(source: Any, positions: dict[str,np.ndarray], skinning: dict[str,dict[str,Any]], tolerance_m: float=.000075, minimum_pair_witnesses: int=3) -> tuple[dict[str,dict[str,Any]],dict[str,Any]]:
    """Preserve the source deformation relationship across source-proven cross-mesh seams.

    Geometry can be perfectly joined in bind pose yet open in game if the two meshes are independently
    reweighted.  For every repeated source-proven seam witness, preserve the source pair's weight delta
    around the target-adapted average.  Equal source seam weights therefore remain equal; deliberately
    different authored seam weights retain that difference.  This changes skinning only, never geometry.
    """
    if not skinning:return skinning,{"enabled":False,"reason":"no skinning payload"}
    if all(np.array_equal(np.asarray(row["weights"]), np.asarray(source.data(name)["W"]))
           and list(row["joint_names"]) == list(source.data(name)["joint_names"])
           for name,row in skinning.items()):
        return skinning,{"enabled":False,"reason":"source skinning already exact; preserve authored seam weights without arithmetic","changed_vertices":0}
    pairs,discovery=_source_proven_cross_mesh_seam_pairs(source,positions,tolerance_m=tolerance_m,minimum_pair_witnesses=minimum_pair_witnesses)
    if not pairs:return skinning,{"enabled":False,"reason":"no repeated source-proven cross-mesh seam","discovery":discovery}
    out={name:{**row,"weights":np.asarray(row["weights"],dtype=np.float64).copy(),"joint_names":list(row["joint_names"])} for name,row in skinning.items()}
    accum={name:np.zeros_like(np.asarray(row["weights"],dtype=np.float64)) for name,row in out.items()};count={name:np.zeros(len(np.asarray(row["weights"])),dtype=np.float64) for name,row in out.items()}
    before=[];after=[];used=0;skipped=0
    for an,ai,bn,bi,_ in pairs:
        if an not in out or bn not in out:skipped+=1;continue
        a_names=list(out[an]["joint_names"]);b_names=list(out[bn]["joint_names"])
        if a_names!=b_names:skipped+=1;continue
        A=np.asarray(out[an]["weights"],dtype=np.float64);B=np.asarray(out[bn]["weights"],dtype=np.float64)
        sa=np.asarray(source.data(an).get("W",[]),dtype=np.float64);sb=np.asarray(source.data(bn).get("W",[]),dtype=np.float64)
        if ai>=len(A) or bi>=len(B) or ai>=len(sa) or bi>=len(sb) or A.shape[1]!=B.shape[1] or sa.shape[1]!=A.shape[1] or sb.shape[1]!=B.shape[1]:skipped+=1;continue
        wa=A[ai];wb=B[bi];swa=sa[ai];swb=sb[bi];source_delta=swa-swb;target_avg=(wa+wb)*.5
        da=np.maximum(target_avg+source_delta*.5,0.0);db=np.maximum(target_avg-source_delta*.5,0.0)
        ta=float(da.sum());tb=float(db.sum())
        if ta<=1e-12 or tb<=1e-12:skipped+=1;continue
        da/=ta;db/=tb
        before.append(float(np.abs((wa-wb)-source_delta).sum()));after.append(float(np.abs((da-db)-source_delta).sum()))
        accum[an][ai]+=da;count[an][ai]+=1.0;accum[bn][bi]+=db;count[bn][bi]+=1.0;used+=1
    changed_vertices=0
    for name,row in out.items():
        mask=count[name]>0
        if not np.any(mask):continue
        W=np.asarray(row["weights"],dtype=np.float64);W[mask]=accum[name][mask]/count[name][mask,None];W[mask]/=np.maximum(W[mask].sum(axis=1,keepdims=True),1e-12);row["weights"]=W;changed_vertices+=int(np.count_nonzero(mask))
    return out,{"enabled":bool(used),"policy":"source-proven cross-mesh seams preserve their authored skin-weight relationship after target retargeting","discovery":discovery,"used_pair_witnesses":int(used),"skipped_pair_witnesses":int(skipped),"changed_vertices":int(changed_vertices),"pair_weight_delta_error_l1_p95_before":float(np.percentile(before,95)) if before else 0.0,"pair_weight_delta_error_l1_p95_after":float(np.percentile(after,95)) if after else 0.0}


def _retarget_frozen_layer_skinning(source: Any, positions: dict[str,np.ndarray], cache: dict[str,Any], source_body_triangles: np.ndarray):
    """Retarget weights once, after final geometry is frozen. This function never edits positions."""
    before={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()}
    skinning={};records={}
    for name,final_pos in positions.items():
        data=source.data(name);w=weld_mesh(data);behavior,features,labels,classes,details=infer_shell_behavior(w,source_body_triangles)
        weights,stage=_retarget_garment_skinning(np.asarray(final_pos,dtype=np.float64),np.asarray(data["V"],dtype=np.float64),np.asarray(data["W"],dtype=np.float64),list(data["joint_names"]),cache,behavior,behavior,labels,classes,w["raw_to_weld"])
        skinning[name]={"weights":weights,"joint_names":list(data["joint_names"]),"stage":stage}
        records[name]={"behavior":behavior,"features":features,"component_count":len(details),"skinning":stage}
    skinning,seam_skinning=_preserve_source_shared_seam_skinning(source,positions,skinning,tolerance_m=.000075,minimum_pair_witnesses=3)
    for name in records:records[name]["source_shared_seam_skinning"]=seam_skinning
    for name in positions:
        if not np.array_equal(before[name],np.asarray(positions[name])):
            raise AssertionError(f"Skinning retarget moved geometry for {name}; geometry must be frozen before skinning.")
    return skinning,records


def _solve_strict_b14_layers(source: Any, cache: dict[str,Any], body_mesh_names: set[str], mesh_filter: set[str] | None=None):
    """Discover authored layers, run untouched strict B14 independently, freeze, reconcile, then skin once."""
    X=cache["X"];Y=cache["Y"];BW=cache["BW"];NS=cache["NS"];NT=cache["NT"];names=list(cache["names"])
    local_affines=cache.get("_ravafit_local_affines")
    if local_affines is None:
        local_affines=precompute_body_local_affines(X,Y,BW);cache["_ravafit_local_affines"]=local_affines
    A,quality,tree=local_affines
    axes=skeleton_bone_axes(source,names)
    source_tri=cache.get("_ravafit_strict_source_surface_triangles")
    if source_tri is None:source_tri=_triangles_from_surface(cache.get("_ravafit_strict_source_surface_V",cache["source_support_V"]),cache.get("_ravafit_strict_source_surface_F",cache["source_support_F"]))
    target_tri=cache.get("_ravafit_strict_target_surface_triangles")
    if target_tri is None:target_tri=_triangles_from_surface(cache.get("_ravafit_strict_target_surface_V",cache["target_support_V"]),cache.get("_ravafit_strict_target_surface_F",cache["target_support_F"]))
    source_tri,strict_source_surface_sanitisation=_sanitise_strict_b14_surface_triangles(source_tri,"source")
    target_tri,strict_target_surface_sanitisation=_sanitise_strict_b14_surface_triangles(target_tri,"target")
    target_collision_tri=_target_fit_collision_triangles(cache)
    layers,_=_infer_garment_layers(source,body_mesh_names,mesh_filter,source_tri)
    if not layers:raise ValueError("No garment layers were inferred after removing the source body.")
    layers,relations=_infer_layer_order_graph(source,layers,source_tri)
    frozen={};layer_diagnostics=[]
    # B14 layer solves are independent by invariant, so solve structural/cloth layers before large
    # waves of rigid ornament layers. This keeps expensive historical flexible/shell solves in a
    # clean numerical state without allowing solve order to affect geometry or authority.
    solve_layers=sorted(layers,key=lambda layer:(layer.structural_classification=="rigid_assembly", layer.stable_id))
    for layer in solve_layers:
        member_results=[];member_stages=[];layer_started=time.perf_counter()
        for member in layer.members:
            data=_layer_member_virtual_data(source,layer,member)
            component_id=member.stable_component_id
            _assert_finite_stage(component_id,"component source import",data["V"])
            source_w=weld_mesh(data);_assert_finite_stage(component_id,"component welding",source_w["V"])
            thin=_infer_thin_volume_surface(source_w,source_tri)
            solve_data=data;w=source_w
            if thin is not None:
                solve_data=_thin_volume_surface_data(source_w,thin,component_id+"::garment-surface")
                w=weld_mesh(solve_data);_assert_finite_stage(component_id,"thin-volume garment surface",w["V"])
            Wg,_=_finite_stage_call(component_id,"component body-weight transfer",lambda w=w:solver_body_weights(w["V"],X,BW,tree))
            base_result=_finite_stage_call(component_id,"strict historical local body-field mapping",lambda w=w,Wg=Wg:local_body_field_map_soft(w["V"],Wg,X,Y,BW,A,tree=tree,tau=.004))
            base,contact,_,_,_,_=base_result
            behavior,features,labels,classes,details=infer_shell_behavior(w,source_tri)
            member_shape=_structural_component_features(np.asarray(w["V"],dtype=np.float64),np.asarray(w["F"],dtype=np.int64))
            compact_strip=(float(member_shape.get("max_extent",0.0))<=.055 and float(member_shape.get("middle_ratio",1.0))<=.40 and float(member_shape.get("thin_ratio",1.0))<=.10)
            if member.structural_classification in {"rigid_detail","ribbon_or_strap"} or compact_strip:
                behavior="conservative_component_assembly"
                if member.structural_classification=="rigid_detail":
                    classes={int(c):"rigid" for c in np.unique(labels)}
            started=time.perf_counter()
            U,ids,F,stage=_finite_stage_call(component_id,"strict frozen B14 component solve",lambda solve_data=solve_data,w=w,Wg=Wg,base=base,contact=contact,labels=labels,classes=classes,features=features,behavior=behavior,component_id=component_id:_strict_frozen_b14_solve(source,component_id,solve_data,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,classes,features,behavior))
            elapsed=time.perf_counter()-started
            solved_w=expand_welded(w,U);_assert_finite_stage(component_id,"strict frozen B14 component expansion",solved_w)
            if thin is not None:
                reconstructed=_finite_stage_call(component_id,"source-authored thin-volume reconstruction",lambda source_w=source_w,thin=thin,solved_w=solved_w:_reconstruct_thin_volume_from_surface(source_w,thin,solved_w))
                raw=expand_welded(source_w,reconstructed)
                thin_stage={k:v for k,v in thin.items() if k not in {"outer_ids","outer_faces"}}
                stage={"mode":"source_authored_thin_volume","surface_behavior":behavior,"thin_volume":thin_stage,"b14":stage}
            else:
                raw=solved_w
            _assert_finite_stage(component_id,"strict frozen B14 final component expansion",raw)
            if raw.shape!=data["V"].shape:raise ValueError(f"{component_id} strict B14 produced {raw.shape}, expected {data['V'].shape}")
            member_results.append(np.asarray(raw,dtype=np.float64))
            member_stages.append({
                "component_id":component_id,"mesh_name":member.mesh_name,"component_index":int(member.component_index),
                "structural_classification":member.structural_classification,"behavior":behavior,"solve_time_sec":float(elapsed),
                "thin_volume":bool(thin is not None),
                "source_clearance_median_mm":float(member.source_clearance_median_mm),"source_clearance_p95_mm":float(member.source_clearance_p95_mm),"stage":stage,
            })
        raw=np.vstack(member_results) if member_results else np.zeros((0,3),dtype=np.float64)
        elapsed=time.perf_counter()-layer_started
        behaviors=sorted({str(row["behavior"]) for row in member_stages})
        behavior=behaviors[0] if len(behaviors)==1 else "component_independent:"+",".join(behaviors)
        stage={"mode":"strict_component_independent_layer","components":member_stages}
        frozen[layer.stable_id]=B14LayerResult(layer.stable_id,_readonly_array(raw),behavior,float(layer.source_clearance_median_mm),float(layer.source_clearance_p95_mm),float(elapsed),stage)
        layer_diagnostics.append({"layer_id":layer.stable_id,"components":list(layer.connected_components),"source_meshes":list(layer.source_meshes),"materials":list(layer.materials),"structural_classification":layer.structural_classification,"source_ordering_relationships":list(layer.source_ordering_relationships),"b14_behavior":behavior,"source_clearance_median_mm":float(layer.source_clearance_median_mm),"source_clearance_p95_mm":float(layer.source_clearance_p95_mm),"source_signed_clearance_median_mm":float(layer.source_signed_clearance_median_mm),"b14_solve_time_sec":float(elapsed),"stage":stage})

    source_layer_geometry={layer.stable_id:_layer_geometry_from_source(source,layer) for layer in layers}
    final_layers,reconciliation=_reconcile_frozen_b14_layers(layers,frozen,relations,target_collision_tri,displacement_cap_m=.0020,source_layer_geometry=source_layer_geometry,source_body_triangles=source_tri)
    positions=_scatter_layer_positions_to_meshes(source,layers,final_layers)
    positions,local_retarget=_apply_b14_local_structural_retarget(source,positions,cache,source_tri,target_collision_tri,source_asset_has_embedded_body=bool(body_mesh_names))
    # The universal field is allowed to change fit, but never authored assembly ordering. Re-run only
    # the tiny relationship/body reconciler after the fit so layered garments remain one construction.
    post_universal_frozen=_gather_layer_positions_from_meshes(layers,positions)
    post_layers,post_universal_reconciliation=_reconcile_frozen_b14_layers(layers,post_universal_frozen,relations,target_collision_tri,displacement_cap_m=.0020,source_layer_geometry=source_layer_geometry,source_body_triangles=source_tri)
    positions=_scatter_layer_positions_to_meshes(source,layers,post_layers)
    # Local target-only relief (nipples, sharp body detail, similar small convexities) is collision
    # evidence, not garment-shape authority.  Bridge only source-disproved convexities after the final
    # fit, then run attachment closure last so straps/rings remain attached to that final shell.
    positions,final_local_shell_bridge=_apply_final_local_shell_bridge(source,positions,source_tri,target_collision_tri,target_tri,max_passes=3)
    positions,support_stable_component_preserve=_preserve_support_stable_components(source,positions,cache,source_tri,target_collision_tri)
    pre_structural_authority_positions={name:np.asarray(value,dtype=np.float64).copy() for name,value in positions.items()}
    # Reassert disconnected unilateral leg-pair separation after every body/layer/reconciliation stage
    # that can move medial edges.  Attachment closure stays last so source-proven physical joints keep
    # their final construction authority.
    positions,pre_attachment_unilateral_pair_separation=_preserve_final_unilateral_pair_separation(source,positions)
    # Attachment continuity still closes genuine source-proven physical joints.  It is not allowed to
    # become placement authority for nearby disconnected ornamentation.
    positions,final_attachment_continuity=_preserve_final_source_attachment_continuity(source,positions)
    # Attachment closure can perturb a previously-correct unilateral opening.  Reassert both its
    # target-scaled separation and its short source-proximal similarity structure afterwards.
    positions,final_unilateral_pair_separation=_preserve_final_unilateral_pair_separation(source,positions)
    # Finally, compact disconnected details inherit the local frame of the now-final carrier surface.
    # This keeps authored motifs/layout coherent without freezing the carrier's actual target refit.
    positions,final_surface_relative_detail_layout=_preserve_final_surface_relative_detail_layout(source,positions)
    # Individual compact details are not enough for authored multi-piece structures: nested ornaments,
    # chain/link assemblies and broad carried cuffs now share one solved carrier frame so their internal
    # shape/layout cannot drift while the underlying garment still genuinely refits the target.
    positions,final_source_relative_structural_carriers=_preserve_final_source_relative_structural_carriers(source,positions)
    # Structural carrier placement never owns bilateral separation; reassert the target-scaled floor.
    positions,post_detail_unilateral_pair_separation=_preserve_final_unilateral_pair_separation(source,positions,preserve_proximal_structure=False)
    # Structural preservation must never trade a clean target fit for a prettier source-relative shape.
    # Backtrack only the post-fit structural delta where literal body evidence proves it made the
    # already-resolved source-relative body relationship worse.
    positions,final_structural_body_veto=_veto_worsened_source_relative_body_penetration(source,pre_structural_authority_positions,positions,source_tri,target_collision_tri)
    # The selected target body is final occupancy authority, including anatomy absent from the source
    # outfit body. Repair only actual residual penetration and use smooth target support for the push
    # field so literal local anatomy cannot become embossed garment-shape authority.
    positions,final_target_body_clearance=_final_target_body_clearance(source,positions,target_collision_tri,target_tri,margin_m=.00010,maximum_vertex_move_m=.00800,maximum_step_m=.00150,max_passes=8)
    # Independent mesh solves can pull apart a boundary that was physically shared in the source.
    # Rejoin those source-proven seams once as the final garment-construction authority.  Seam
    # placement is derived only from the untouched garment; body spacing is never used to shape it.
    positions,final_source_shared_seams=_preserve_final_source_shared_seams(source,positions,tolerance_m=.000075,minimum_pair_witnesses=3)
    post_seam_body_clearance={"enabled":False,"reason":"single bounded penetration pass; seam closure does not own body clearance"}
    final_source_shared_seam_reassert={"enabled":False,"reason":"single authored seam closure is final"}
    skinning,mesh_skin_records=_retarget_frozen_layer_skinning(source,positions,cache,source_tri)
    records={}
    for layer_diag in layer_diagnostics:
        rr=reconciliation["layers"][layer_diag["layer_id"]]
        layer_diag.update({"post_b14_reconciliation_vertex_count":int(rr["reconciled_vertices"]),"reconciliation_p50_mm":float(rr["p50_mm"]),"reconciliation_p95_mm":float(rr["p95_mm"]),"reconciliation_max_mm":float(rr["max_mm"]),"reconciliation_warning":bool(rr["warning"]),
                           "layer_order_violations_before":int(rr["layer_order_violations_before"]),"layer_order_violations_after":int(rr["layer_order_violations_after"]),
                           "body_penetrations_before":int(rr["body_penetrations_before"]),"body_penetrations_after":int(rr["body_penetrations_after"])})
        print(f"[RavaFit layer] {layer_diag['layer_id']} components={layer_diag['components']} behavior={layer_diag['b14_behavior']} "
              f"clearance_p50/p95={layer_diag['source_clearance_median_mm']:.3f}/{layer_diag['source_clearance_p95_mm']:.3f} mm "
              f"B14={layer_diag['b14_solve_time_sec']:.3f}s reconcile_vertices={layer_diag['post_b14_reconciliation_vertex_count']} "
              f"reconcile_p50/p95/max={layer_diag['reconciliation_p50_mm']:.3f}/{layer_diag['reconciliation_p95_mm']:.3f}/{layer_diag['reconciliation_max_mm']:.3f} mm "
              f"order={layer_diag['layer_order_violations_before']}->{layer_diag['layer_order_violations_after']} body={layer_diag['body_penetrations_before']}->{layer_diag['body_penetrations_after']}",
              file=sys.stderr,flush=True)
    for warning in reconciliation.get("warnings",[]):
        print(f"[RavaFit layer WARNING] {warning}",file=sys.stderr,flush=True)
    for name in positions:
        records[name]={"mode":"strict_layered_b14","layers":[layer.stable_id for layer in layers if name in layer.source_meshes],**mesh_skin_records[name]}
    stats={
        "strict_frozen_b14":True,"strict_layer_orchestration":True,
        "principle":"B14 fits layers. RavaFit only discovers the layers, supplies the correct body correspondence, and preserves their authored relationships afterward.",
        "strict_contract_report":cache.get("_ravafit_strict_b14_report"),
        "strict_surface_sanitisation":{"source":strict_source_surface_sanitisation,"target":strict_target_surface_sanitisation},
        "garment_layer_count":len(layers),"layer_order_relation_count":len(relations),
        "layers":layer_diagnostics,
        "layer_order_graph":[{k:v for k,v in relation.items() if k!="source_gap"} for relation in relations],
        "reconciliation":reconciliation,
        "local_structural_retarget":local_retarget,
        "post_universal_reconciliation":post_universal_reconciliation,
        "final_local_shell_bridge":final_local_shell_bridge,
        "support_stable_component_preserve":support_stable_component_preserve,
        "pre_attachment_unilateral_pair_separation":pre_attachment_unilateral_pair_separation,
        "final_source_attachment_continuity":final_attachment_continuity,
        "final_unilateral_pair_separation":final_unilateral_pair_separation,
        "final_surface_relative_detail_layout":final_surface_relative_detail_layout,
        "final_source_relative_structural_carriers":final_source_relative_structural_carriers,
        "post_detail_unilateral_pair_separation":post_detail_unilateral_pair_separation,
        "final_structural_body_veto":final_structural_body_veto,
        "final_target_body_clearance":final_target_body_clearance,
        "final_source_body_clearance":final_target_body_clearance,
        "final_source_shared_seams":final_source_shared_seams,
        "post_seam_body_clearance":post_seam_body_clearance,
        "final_source_shared_seam_reassert":final_source_shared_seam_reassert,
        "local_affine_quality_rms_mm":float(np.sqrt(np.mean(np.asarray(quality,float)**2))*1000.0),
        "garment_mesh_count":len(positions),"skinning_retargeted_mesh_count":len(skinning),
        "post_b14_geometry_mutation":"bounded_layer_reconciler_then_evidence_gated_local_retarget_then_iterative_local_shell_bridge_then_support_stable_component_preserve_then_unilateral_proximal_guard_then_source_attachment_closure_then_unilateral_proximal_reassert_then_surface_relative_detail_layout_then_source_relative_structural_carriers_then_final_unilateral_floor_then_source_relative_body_veto_then_target_authoritative_smooth_envelope_penetration_repair_then_source_proven_cross_mesh_seam_closure",
        "post_b14_body_or_macro_solver_called":False,
        "skinning_after_geometry_freeze":True,
    }
    return positions,skinning,records,stats


_STRICT_B14_INPROCESS_LOCK=threading.RLock()

def _strict_frozen_flexible_inprocess(w: dict[str,Any], Wg: np.ndarray, base: np.ndarray, axes: np.ndarray, target_tri: np.ndarray, labels: np.ndarray, features: dict[str,Any]):
    """Run the exact frozen flexible function in-process with Runtime11 patches temporarily removed.

    This avoids repeated heavyweight child-process startup for large flexible layers while preserving
    frozen B14 semantics.  Every monkey-patched import surface touched by ``b14_compat`` is restored
    to its original function for the duration of the call, under a process-wide lock, then restored
    exactly afterward.  Frozen source files remain untouched.
    """
    import b14_compat as compat
    with _STRICT_B14_INPROCESS_LOCK:
        worker=compat._worker
        saved={
            "worker_nearest":worker.nearest_surface,
            "worker_refine_component":worker.refine_component,
            "worker_refine_lobofit":worker.refine_lobofit,
            "worker_collision":worker.smooth_collision_polish,
            "collision_nearest":compat._collision_eval.nearest_surface,
            "construction_nearest":compat._construction_fields.nearest_surface,
            "lobofit_nearest":compat._lobofit_official.nearest_surface,
        }
        try:
            worker.nearest_surface=compat._original_nearest_surface
            worker.refine_component=compat._original_refine_component
            worker.refine_lobofit=compat._original_refine_lobofit
            worker.smooth_collision_polish=compat._original_smooth_collision_polish
            compat._collision_eval.nearest_surface=compat._original_nearest_surface
            compat._construction_fields.nearest_surface=compat._original_nearest_surface
            compat._lobofit_official.nearest_surface=compat._original_nearest_surface
            return worker.solve_flexible(w,Wg,base,axes,target_tri,labels,features)
        finally:
            worker.nearest_surface=saved["worker_nearest"]
            worker.refine_component=saved["worker_refine_component"]
            worker.refine_lobofit=saved["worker_refine_lobofit"]
            worker.smooth_collision_polish=saved["worker_collision"]
            compat._collision_eval.nearest_surface=saved["collision_nearest"]
            compat._construction_fields.nearest_surface=saved["construction_nearest"]
            compat._lobofit_official.nearest_surface=saved["lobofit_nearest"]

def _strict_frozen_b14_solve(source: Any, name: str, data: dict[str,Any], w: dict[str,Any], Wg: np.ndarray, base: np.ndarray, contact: np.ndarray, X: np.ndarray, Y: np.ndarray, BW: np.ndarray, NS: np.ndarray, NT: np.ndarray, names: list[str], axes: np.ndarray, source_tri: np.ndarray, target_tri: np.ndarray, labels: np.ndarray, classes: dict[int,str], features: dict[str,Any], behavior: str):
    """Execute untouched historical B14 behaviour for one frozen layer.

    Conservative component assembly is intentionally executed in-process.  Frozen B14's
    ``solve_conservative`` is geometry-local: it only applies the already-computed body field and
    rigid Kabsch fits; it does not call the Runtime11-patched collision/refinement machinery.
    Running that exact function directly avoids spawning hundreds of short-lived Python workers for
    buckles, trims and other disconnected rigid details while keeping cloth/shell solves isolated.
    Structured/flexible behaviours still execute in the clean worker whose import path excludes
    ``b14_compat``.
    """
    if behavior not in {"constructed_close_shell","stand_off_structured_shell","body_following_flexible_layer"}:
        started=time.perf_counter()
        U,ids,F,stage=solve_conservative(w,Wg,base,labels,classes,features)
        _assert_finite_stage(name,"strict frozen B14 direct conservative solve",U)
        return np.asarray(U,dtype=np.float64),np.asarray(ids,dtype=np.int64),np.asarray(F,dtype=np.int64),{
            "mode":"strict_frozen_b14_direct_conservative",
            "solve":stage,
            "worker":{"wall_sec":float(time.perf_counter()-started),"in_process":True,"exact_frozen_function":True},
        }
    if behavior=="body_following_flexible_layer":
        started=time.perf_counter()
        U,ids,F,stage=_strict_frozen_flexible_inprocess(w,Wg,base,axes,target_tri,labels,features)
        _assert_finite_stage(name,"strict frozen B14 in-process flexible solve",U)
        return np.asarray(U,dtype=np.float64),np.asarray(ids,dtype=np.int64),np.asarray(F,dtype=np.int64),{
            "mode":"strict_frozen_b14_inprocess_flexible",
            "solve":stage,
            "worker":{"wall_sec":float(time.perf_counter()-started),"in_process":True,"runtime11_patches_temporarily_removed":True},
        }

    # Structured B14 remains process-isolated so Runtime11 compatibility patches can never
    # affect its historical refinement/collision behaviour.

    worker=_MODULE_DIR/'strict_b14_shell_worker.py'
    if not worker.exists():raise FileNotFoundError(worker)
    work_dir=Path(tempfile.mkdtemp(prefix='ravafit-strict-b14-'));input_path=work_dir/'input.npz';meta_path=work_dir/'meta.json';output_path=work_dir/'output.npz';stage_path=work_dir/'stage.json';started=time.perf_counter()
    try:
        np.savez_compressed(input_path,w_V=np.asarray(w['V'],dtype=np.float64),w_F=np.asarray(w['F'],dtype=np.int64),w_W=np.asarray(w.get('W',np.zeros((len(w['V']),0))),dtype=np.float64),Wg=np.asarray(Wg,dtype=np.float64),base=np.asarray(base,dtype=np.float64),contact=np.asarray(contact,dtype=np.int64),X=np.asarray(X,dtype=np.float64),Y=np.asarray(Y,dtype=np.float64),BW=np.asarray(BW,dtype=np.float64),NS=np.asarray(NS,dtype=np.float64),NT=np.asarray(NT,dtype=np.float64),axes=np.asarray(axes,dtype=np.float64),source_tri=np.asarray(source_tri,dtype=np.float64),target_tri=np.asarray(target_tri,dtype=np.float64),labels=np.asarray(labels,dtype=np.int64))
        meta={'source_js':source.js,'w_joint_names':list(w.get('joint_names',[])),'names':list(names),'features':_strict_json_safe(features),'classes':{str(k):str(v) for k,v in classes.items()},'behavior':str(behavior),'material':str(data.get('material','')),'mesh_name':str(name)}
        meta_path.write_text(json.dumps(meta,separators=(',',':')),encoding='utf-8')
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform.startswith('win') else 0
        proc=subprocess.run([sys.executable,str(worker),str(input_path),str(meta_path),str(output_path),str(stage_path)],capture_output=True,text=True,timeout=240,creationflags=creationflags,env=env)
        if proc.returncode!=0 or not output_path.exists():raise RuntimeError(f"strict frozen B14 worker exited {proc.returncode}: {(proc.stderr or proc.stdout)[-3000:]}")
        with np.load(output_path,allow_pickle=False) as out:
            U=np.asarray(out['U'],dtype=np.float64).copy();ids=np.asarray(out['ids'],dtype=np.int64).copy();F=np.asarray(out['F'],dtype=np.int64).copy()
        payload=json.loads(stage_path.read_text(encoding='utf-8')) if stage_path.exists() else {};stage=payload.get('stage',{});stage={'mode':'strict_frozen_b14','solve':stage,'worker':{'wall_sec':time.perf_counter()-started,'worker_sec':payload.get('elapsed_sec'),'pid':payload.get('pid')}}
        return U,ids,F,stage
    finally:
        shutil.rmtree(work_dir,ignore_errors=True)


def _solve_garment_meshes(source: GLB, cache: dict[str, Any], body_mesh_names: set[str], mesh_filter: set[str] | None = None, _skip_final_assembly: bool = False):
    _warm_tri=np.asarray([[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]]],dtype=np.float64)
    _b14_nearest_surface(np.asarray([[0.1,0.1,0.1]],dtype=np.float64),_warm_tri,k=1)
    X = cache["X"]
    Y = cache["Y"]
    BW = cache["BW"]
    NS = cache["NS"]
    NT = cache["NT"]
    names = list(cache["names"])
    strict_b14_contract=bool(cache.get("_ravafit_strict_b14_contract",False))
    dense_vanilla_proxy=bool(cache.get("dense_vanilla_source_proxy",False))
    modded_coverage_finalizer=not dense_vanilla_proxy and not strict_b14_contract
    local_affines=cache.get("_ravafit_local_affines")
    if local_affines is None:
        local_affines=precompute_body_local_affines(X,Y,BW);cache["_ravafit_local_affines"]=local_affines
    A,quality,tree=local_affines
    axes = skeleton_bone_axes(source, names)
    longitudinal_axis = _skeleton_longitudinal_axis(source, names)
    source_tri=cache.get("_ravafit_source_support_triangles")
    if source_tri is None:
        source_tri=_triangles_from_surface(cache["source_support_V"],cache["source_support_F"]);cache["_ravafit_source_support_triangles"]=source_tri
    target_tri=cache.get("_ravafit_target_support_triangles")
    if target_tri is None:
        target_tri=_triangles_from_surface(cache["target_support_V"],cache["target_support_F"]);cache["_ravafit_target_support_triangles"]=target_tri
    if strict_b14_contract:
        source_tri,_=_sanitise_strict_b14_surface_triangles(source_tri,"source")
        target_tri,_=_sanitise_strict_b14_surface_triangles(target_tri,"target")
    target_collision_tri=_target_fit_collision_triangles(cache)

    positions: dict[str, np.ndarray] = {}
    skinning: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {}
    layer_meshes: dict[str, dict[str, Any]] = {}
    retarget_contexts: dict[str, dict[str, Any]] = {}

    if bool(cache.get("identity_body_mapping",False)):
        for name in source.mesh_names():
            if not name or name in body_mesh_names:continue
            if mesh_filter is not None and name not in mesh_filter:
                records[name]={"mode":"unchanged","reason":"excluded by diagnostic mesh filter"};continue
            try:data=source.data(name)
            except Exception as ex:
                records[name]={"mode":"unchanged","reason":f"not B14-skinnable: {ex}"};continue
            if not len(data["V"]) or not len(data["F"]):
                records[name]={"mode":"unchanged","reason":"empty geometry"};continue
            positions[name]=np.asarray(data["V"],dtype=np.float64).copy()
            skinning[name]={"weights":np.asarray(data["W"],dtype=np.float64).copy(),"joint_names":list(data["joint_names"]),"stage":{"mode":"authored_identity","reason":"source and target body payloads are identical"}}
            records[name]={"mode":"identity","behavior":"authored_identity","stage":{"mode":"authored_identity"},"skinning":skinning[name]["stage"]}
        if not positions:raise ValueError("No garment meshes were eligible for identity-body fitting after removing the source body.")
        return positions,skinning,records,{"identity_body_fast_path":True,"garment_mesh_count":len(positions),"skinning_retargeted_mesh_count":0,"authored_layer_relations":0,"authored_layer_adjusted_meshes":0,"authored_layer_report":{"relation_count":0,"adjusted_mesh_count":0,"relations":[],"skipped":"identity body mapping"},"authored_cross_mesh_relations":0,"authored_cross_mesh_adjusted_meshes":0,"authored_cross_mesh_report":{"relation_count":0,"adjusted_mesh_count":0,"relations":[],"skipped":"identity body mapping"},"authored_split_seams":{"enabled":False,"reason":"identity body mapping preserves authored raw geometry exactly"},"local_affine_quality_rms_mm":0.0}

    for name in source.mesh_names():
        if not name or name in body_mesh_names:
            continue
        if mesh_filter is not None and name not in mesh_filter:
            records[name] = {"mode": "unchanged", "reason": "excluded by diagnostic mesh filter"}
            continue
        try:
            data = source.data(name)
        except Exception as ex:
            # Helper/non-skinned nodes are not garment geometry.
            records[name] = {"mode": "unchanged", "reason": f"not B14-skinnable: {ex}"}
            continue
        if not len(data["V"]) or not len(data["F"]):
            records[name] = {"mode": "unchanged", "reason": "empty geometry"}
            continue

        _assert_finite_stage(name, "source garment import", data["V"])
        w = weld_mesh(data)
        _assert_finite_stage(name, "mesh welding", w["V"])
        Wg, _ = _finite_stage_call(name, "body-weight transfer", lambda: solver_body_weights(w["V"], X, BW, tree))
        _assert_finite_stage(name, "body-weight transfer", Wg)
        base_result = _finite_stage_call(name, "local body-field mapping", lambda: local_body_field_map_soft(w["V"], Wg, X, Y, BW, A, tree=tree, tau=.004))
        base, contact, _, _, relief_blend, relief_ids = base_result
        if strict_b14_contract:
            _assert_finite_stage(name,"strict historical local body-field mapping",base)
            behavior,features,labels,classes,details=infer_shell_behavior(w,source_tri)
            U,ids,F,stage=_finite_stage_call(name,"strict frozen B14 solve",lambda:_strict_frozen_b14_solve(source,name,data,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,classes,features,behavior))
            _assert_finite_stage(name,"strict frozen B14 solve",U)
            raw=expand_welded(w,U)
            if raw.shape!=data["V"].shape:raise ValueError(f"Strict B14 produced invalid geometry shape for {name}: {raw.shape} vs {data['V'].shape}")
            _assert_finite_stage(name,"strict frozen B14 expansion",raw)
            retargeted_weights,skinning_stage=_retarget_garment_skinning(raw,data["V"],data["W"],data["joint_names"],cache,behavior,behavior,labels,classes,w["raw_to_weld"])
            _assert_finite_stage(name,"strict frozen B14 skinning retarget",retargeted_weights)
            positions[name]=raw;skinning[name]={"weights":retargeted_weights,"joint_names":list(data["joint_names"]),"stage":skinning_stage}
            layer_meshes[name]={"V":np.asarray(data["V"],dtype=np.float64),"F":np.asarray(data["F"],dtype=np.int64)}
            retarget_contexts[name]={"data":data,"w":w,"behavior":behavior,"effective_behavior":behavior,"features":features,"labels":labels,"classes":classes}
            records[name]={"mode":"solved","behavior":behavior,"inferred_behavior":behavior,"features":features,"component_count":len(details),"target_relief":{"enabled":False,"reason":"strict frozen B14 contract"},"surface_frame":{"enabled":False,"reason":"strict frozen B14 contract"},"stage":stage,"skinning":skinning_stage}
            continue
        macro_result, macro_field = _finite_stage_call(name, "extreme macro body-field mapping", lambda: _extreme_macro_body_field(w["V"], Wg, cache))
        if macro_result is not None:
            base, contact, _, _, relief_blend, relief_ids = macro_result
        _assert_finite_stage(name, "local body-field mapping", base)
        _assert_finite_stage(name, "local body contact mapping", contact)
        behavior, features, labels, classes, details = infer_shell_behavior(w, source_tri)
        base, surface_frame_stage = _finite_stage_call(name, "extreme source-to-target surface-frame transfer", lambda: _extreme_surface_frame_transfer(w, base, labels, classes, cache, bool(macro_field.get("enabled",False)), source_tri))
        _assert_finite_stage(name, "extreme source-to-target surface-frame transfer", base)
        close_support_authority = modded_coverage_finalizer and behavior == "body_following_flexible_layer" and float(features.get("source_clearance_median_mm",999.0)) <= 4.50
        if close_support_authority:
            relief_stage={"enabled":False,"reason":"close body-following cloth uses smoothed support authority; literal detail remains collision authority","source_clearance_median_mm":float(features.get("source_clearance_median_mm",0.0))}
        else:
            base, relief_stage = _finite_stage_call(name, "target relief filtering", lambda: _apply_target_relief_correction(w["V"], w["F"], base, relief_blend, relief_ids, cache, behavior, features))
            _assert_finite_stage(name, "target relief filtering", base)
        base, extreme_refit = _finite_stage_call(name, "extreme refit topology stabilisation", lambda: _stabilise_extreme_refit_components(w, base, labels, classes, source_tri, bool(macro_field.get("enabled",False))))
        _assert_finite_stage(name, "extreme refit topology stabilisation", base)
        base, authored_rig_geometry = _finite_stage_call(name, "authored rig geometry", lambda: _preserve_authored_rig_geometry(w, base, cache, labels))
        _assert_finite_stage(name, "authored rig geometry", base)
        base, support_frame_clearance = _finite_stage_call(name, "support-frame clearance", lambda: _support_frame_clearance_guard(w, base, relief_blend, relief_ids, cache, labels, classes, source_tri))
        _assert_finite_stage(name, "support-frame clearance", base)
        base, literal_body_envelope = _finite_stage_call(name, "literal-body broad envelope", lambda: _literal_body_envelope_guard(w, base, relief_blend, relief_ids, cache, labels, classes, source_tri, bool(macro_field.get("enabled",False))))
        _assert_finite_stage(name, "literal-body broad envelope", base)
        effective_behavior = behavior
        peer_ids=_peer_shell_component_ids(details)
        peer_group_precomputed=False
        if _use_anchored_free_panel(features, details):
            U, stage = _finite_stage_call(name, "anchored free-panel solve", lambda: _solve_anchored_free_panel(w, base, cache["source_surface_V"])); effective_behavior = "anchored_free_panel"
        elif _use_rigid_component_preservation(behavior, features, details):
            U, stage = _finite_stage_call(name, "rigid-component solve", lambda: _solve_component_rigid_preservation(w, base, labels)); effective_behavior = "rigid_component_preservation"
        elif _use_fragmented_decorated_shell_frame(behavior,features,details):
            U, ids, F, frame_stage = _finite_stage_call(name, "fragmented decorated structural frame", lambda: solve_conservative(w,Wg,base,labels,classes,features))
            stage={"mode":"fragmented_decorated_structural_frame","original_behavior":behavior,"component_count":int(features.get("component_count",0)),"root_vertices":int(features.get("root_vertices",0)),"solve":frame_stage};effective_behavior="fragmented_decorated_structural_frame"
        elif peer_ids and behavior in {"stand_off_structured_shell","constructed_close_shell"}:
            U, ids, stage = _finite_stage_call(name, "independent peer-shell group solve", lambda: _solve_peer_shell_group_fresh(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features,behavior,peer_ids))
            F=np.empty((0,3),dtype=np.int64)
            peer_group_precomputed=True
            effective_behavior=f"{behavior}_peer_shells"
        elif behavior == "stand_off_structured_shell":
            U, ids, F, stage = _finite_stage_call(name, "stand-off structured-shell solve", lambda: _solve_single_shell_fresh_or_legacy(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features,behavior))
        elif behavior == "constructed_close_shell":
            U, ids, F, stage = _finite_stage_call(name, "constructed close-shell solve", lambda: _solve_single_shell_fresh_or_legacy(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features,behavior))
        elif behavior == "body_following_flexible_layer":
            close_translation_lane=modded_coverage_finalizer and float(features.get("source_clearance_median_mm",999.0))<=4.50
            if close_translation_lane:
                U, translation_stage = _finite_stage_call(name, "close body-following translation field", lambda: _smooth_translation_body_field(w["V"],w["F"],Wg,X,Y,BW,tree,tau=.004,iterations=3))
                ids=np.where(labels==int(features["root_component"]))[0];F=_component_local_faces(w["F"],ids,len(w["V"]))
                stage={"mode":"close_body_following_translation_field","translation":translation_stage,"source_clearance_median_mm":float(features.get("source_clearance_median_mm",0.0))}
                effective_behavior="close_body_following_translation_field"
            else:
                U, ids, F, stage = _finite_stage_call(name, "body-following flexible solve", lambda: _solve_single_shell_fresh_or_legacy(source,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features,behavior))
        else:
            U, ids, F, stage = _finite_stage_call(name, "conservative component solve", lambda: solve_conservative(w, Wg, base, labels, classes, features))

        _assert_finite_stage(name, f"{effective_behavior} initial solve", U)
        if int(authored_rig_geometry.get("adjusted_vertices",0))>0:
            stage={"mode":"authored_rig_geometry","solve":stage,"authored_rig_geometry":authored_rig_geometry}
        if int(support_frame_clearance.get("adjusted_vertices",0))>0:
            stage={"mode":"support_frame_clearance","solve":stage,"support_frame_clearance":support_frame_clearance}
        if int(literal_body_envelope.get("adjusted_vertices",0))>0:
            stage={"mode":"literal_body_broad_envelope","solve":stage,"literal_body_envelope":literal_body_envelope}
        if bool(macro_field.get("enabled",False)):
            stage={"mode":"extreme_macro_body_field","solve":stage,"macro_field":macro_field}

        if peer_ids and not peer_group_precomputed and effective_behavior not in {"anchored_free_panel", "rigid_component_preservation"} and behavior == "body_following_flexible_layer":
            peer_result=np.asarray(U,dtype=np.float64).copy();peer_stages=[]
            for peer_id in peer_ids:
                peer_features=dict(features);peer_features["root_component"]=int(peer_id)
                candidate,peer_vertices,_,peer_stage=_finite_stage_call(name,f"peer shell {int(peer_id)} flexible solve",lambda:solve_flexible(w,Wg,base,axes,target_tri,labels,peer_features))
                peer_result[peer_vertices]=candidate[peer_vertices]
                del candidate
                peer_stages.append({"component":int(peer_id),"vertices":int(len(peer_vertices)),"stage":peer_stage})
            U=peer_result
            _assert_finite_stage(name, "independent peer-shell solves", U)
            stage={"mode":"independent_peer_shells","initial":stage,"peers":peer_stages}
            effective_behavior=f"{behavior}_peer_shells"

        use_smooth_flow = _use_b14_smooth_flow_curve(features, details) and effective_behavior not in {"anchored_free_panel", "rigid_component_preservation"}
        if effective_behavior == "stand_off_structured_shell" and not use_smooth_flow:
            U, boundary_design = _finite_stage_call(name, "structured boundary design", lambda: _preserve_structured_shell_longitudinal_design(w, U, labels, features, longitudinal_axis))
            _assert_finite_stage(name, "structured boundary design", U)
            if bool(boundary_design.get("active", False)):
                stage = {"mode": "structured_boundary_design", "solve": stage, "boundary_design": boundary_design}

        if use_smooth_flow:
            U, flow_stage = _finite_stage_call(name, "smooth panel-flow refinement", lambda: _smooth_b14_panel_flow_curve(w, U, cache["source_surface_V"]))
            _assert_finite_stage(name, "smooth panel-flow refinement", U)
            stage = {"mode": "b14_smooth_flow_panel", "b14": stage, "flow": flow_stage}
            effective_behavior = "b14_smooth_flow_panel"

        U, leg_separation_stage = _finite_stage_call(name, "bilateral leg separation", lambda: _preserve_leg_component_separation(w["V"], U, w["W"], w["joint_names"], labels))
        _assert_finite_stage(name, "bilateral leg separation", U)
        if int(leg_separation_stage.get("moved_vertices", 0)) > 0:
            stage = {"mode": "bilateral_leg_separation", "solve": stage, "leg_separation": leg_separation_stage}

        U, extreme_clearance_stage = _finite_stage_call(name, "extreme literal-body coarse clearance", lambda: _extreme_literal_shell_clearance_guard(w["V"], U, w["F"], labels, classes, target_collision_tri, bool(macro_field.get("enabled",False)), w["W"], w["joint_names"], margin=.00060))
        _assert_finite_stage(name, "extreme literal-body coarse clearance", U)
        if int(extreme_clearance_stage.get("adjusted_vertices",0))>0:
            stage={"mode":"extreme_literal_body_coarse_clearance","solve":stage,"extreme_clearance":extreme_clearance_stage}

        if dense_vanilla_proxy or modded_coverage_finalizer:
            reason="dense vanilla proxy uses assembly-level outward-expansion clearance" if dense_vanilla_proxy else "source-authored literal-body coverage finalizer handles garment/body contact after assembly"
            clearance_stage={"affected":0,"skipped":True,"reason":reason}
        else:
            U, clearance_stage = _finite_stage_call(name, "residual surface-clearance guard", lambda: _residual_surface_clearance_guard(U, w["F"], target_tri))
            _assert_finite_stage(name, "residual surface-clearance guard", U)
            if int(clearance_stage.get("affected", 0)) > 0:
                stage = {"mode": "residual_surface_clearance", "solve": stage, "clearance": clearance_stage}

        if effective_behavior == "rigid_component_preservation":
            U, rigid_clearance_stage = _finite_stage_call(name, "rigid-component clearance guard", lambda: _rigid_component_clearance_guard(U, w["F"], labels, target_tri))
            _assert_finite_stage(name, "rigid-component clearance guard", U)
            if int(rigid_clearance_stage.get("moved_components", 0)) > 0:
                stage = {"mode": "rigid_component_clearance", "solve": stage, "rigid_clearance": rigid_clearance_stage}

        preserve_source_orientation = behavior != "conservative_component_assembly"
        skip_legacy_surface = dense_vanilla_proxy or modded_coverage_finalizer or (behavior == "conservative_component_assembly" and len(details) > 20)
        surface_stage = {"affected_faces": 0, "sample_min_after_mm": None, "skipped_for_dense_obstacle": skip_legacy_surface, "dense_vanilla_expansion_guard": dense_vanilla_proxy}
        if not skip_legacy_surface:
            U, surface_stage = _finite_stage_call(name, "continuous surface-clearance guard", lambda: _componentwise_surface_clearance_guard(w["V"], U, w["F"], labels, source_tri, target_tri, preserve_source_orientation))
            _assert_finite_stage(name, "continuous surface-clearance guard", U)
            if int(surface_stage.get("affected_faces", 0)) > 0:
                stage = {"mode": "continuous_surface_clearance", "solve": stage, "surface_clearance": surface_stage}

        # Translate only B14-rigid details; deformable shells get local collision polish.
        if not skip_legacy_surface and surface_stage.get("sample_min_after_mm") is not None and float(surface_stage["sample_min_after_mm"]) < 0.49:
            rigid_ids={int(c) for c,kind in classes.items() if str(kind).casefold()=="rigid"}
            U, rigid_fallback = _finite_stage_call(name, "typed rigid-component clearance fallback", lambda: _rigid_component_clearance_guard(U, w["F"], labels, target_tri, max_component_extent=.18, eligible_components=rigid_ids))
            _assert_finite_stage(name, "typed rigid-component clearance fallback", U)
            U, shell_fallback = _finite_stage_call(name, "typed shell collision fallback", lambda: _shell_component_collision_guard(U, w["F"], labels, classes, target_tri))
            _assert_finite_stage(name, "typed shell collision fallback", U)
            if int(rigid_fallback.get("moved_components",0))>0 or int(shell_fallback.get("affected_components",0))>0:
                stage={"mode":"typed_component_clearance_fallback","solve":stage,"rigid":rigid_fallback,"shell":shell_fallback}
                U, surface_stage = _finite_stage_call(name, "continuous surface-clearance after typed fallback", lambda: _componentwise_surface_clearance_guard(w["V"], U, w["F"], labels, source_tri, target_tri, preserve_source_orientation))
                _assert_finite_stage(name, "continuous surface-clearance after typed fallback", U)
                if int(surface_stage.get("affected_faces",0))>0:
                    stage={"mode":"continuous_surface_clearance_after_typed_fallback","solve":stage,"surface_clearance":surface_stage}

        if not skip_legacy_surface and surface_stage.get("sample_min_after_mm") is not None and float(surface_stage["sample_min_after_mm"]) < 0.0:
            U, face_shell_stage = _finite_stage_call(name, "shell-face clearance inflation", lambda: _shell_face_clearance_inflation_guard(U, w["F"], labels, classes, target_tri))
            _assert_finite_stage(name, "shell-face clearance inflation", U)
            if int(face_shell_stage.get("affected_components",0))>0:
                stage={"mode":"shell_face_clearance_inflation","solve":stage,"shell_face":face_shell_stage}
                U, surface_stage = _finite_stage_call(name, "continuous surface-clearance after shell-face inflation", lambda: _componentwise_surface_clearance_guard(w["V"], U, w["F"], labels, source_tri, target_tri, preserve_source_orientation))
                _assert_finite_stage(name, "continuous surface-clearance after shell-face inflation", U)
                if int(surface_stage.get("affected_faces",0))>0:
                    stage={"mode":"continuous_surface_clearance_after_shell_face","solve":stage,"surface_clearance":surface_stage}

        if dense_vanilla_proxy or modded_coverage_finalizer:
            reason="dense vanilla proxy uses assembly-level outward-expansion clearance" if dense_vanilla_proxy else "source-authored literal-body coverage finalizer handles intended body contact after assembly"
            obstacle_stage={"affected_components":0,"skipped":True,"reason":reason}
            literal_collision_stage={"affected_components":0,"skipped":True,"reason":reason}
        else:
            U, obstacle_stage = _finite_stage_call(name, "dense support-envelope clearance", lambda: _dense_target_obstacle_clearance_guard(w["V"], U, w["F"], labels, classes, target_tri, longitudinal_axis, margin=.00050, preserve_source_orientation=preserve_source_orientation))
            _assert_finite_stage(name, "dense support-envelope clearance", U)
            if int(obstacle_stage.get("affected_components", 0)) > 0:
                stage = {"mode": "support_envelope_clearance", "solve": stage, "obstacle_clearance": obstacle_stage}

            U, literal_collision_stage = _finite_stage_call(name, "literal target collision", lambda: _dense_target_obstacle_clearance_guard(w["V"], U, w["F"], labels, classes, target_collision_tri, longitudinal_axis, margin=.00035, preserve_source_orientation=preserve_source_orientation))
            _assert_finite_stage(name, "literal target collision", U)
            if int(literal_collision_stage.get("affected_components", 0)) > 0:
                stage = {"mode": "literal_target_collision", "solve": stage, "literal_collision": literal_collision_stage}

        raw = expand_welded(w, U)
        if raw.shape != data["V"].shape:
            raise ValueError(f"B14 produced invalid geometry shape for {name}: {raw.shape} vs {data['V'].shape}")
        _assert_finite_stage(name, "final welded expansion", raw)
        retargeted_weights, skinning_stage = _retarget_garment_skinning(
            raw, data["V"], data["W"], data["joint_names"], cache, behavior, effective_behavior, labels, classes, w["raw_to_weld"]
        )
        _assert_finite_stage(name, "skinning retarget", retargeted_weights)
        positions[name] = raw
        skinning[name] = {"weights": retargeted_weights, "joint_names": list(data["joint_names"]), "stage": skinning_stage}
        layer_meshes[name] = {"V": np.asarray(data["V"], dtype=np.float64), "F": np.asarray(data["F"], dtype=np.int64)}
        retarget_contexts[name] = {"data": data, "w": w, "behavior": behavior, "effective_behavior": effective_behavior, "features": features, "labels": labels, "classes": classes}
        records[name] = {
            "mode": "solved",
            "behavior": effective_behavior,
            "inferred_behavior": behavior,
            "features": features,
            "component_count": len(details),
            "target_relief": relief_stage,
            "surface_frame": surface_frame_stage,
            "stage": stage,
            "skinning": skinning_stage,
        }

    if not positions:
        raise ValueError("No garment meshes were eligible for B14 fitting after removing the source body.")
    local_rms=float(np.sqrt(np.mean(np.asarray(quality,float)**2))*1000.0)
    if strict_b14_contract or _skip_final_assembly:
        return positions,skinning,records,{"preassembly_only":bool(_skip_final_assembly),"strict_frozen_b14":bool(strict_b14_contract),"strict_contract_report":cache.get("_ravafit_strict_b14_report"),"local_affine_quality_rms_mm":local_rms,"garment_mesh_count":len(positions),"skinning_retargeted_mesh_count":len(skinning),"post_b14_geometry_mutation":False}
    return _finalize_garment_solution(source,cache,positions,skinning,records,local_rms)


def _solve_modded_garment_meshes_isolated(source: Any, cache: dict[str,Any], body_mesh_names: set[str], mesh_filter: set[str] | None = None):
    """Solve authored modded garment meshes in fresh clean processes and finalise once."""
    if mesh_filter is not None:return _solve_garment_meshes(source,cache,body_mesh_names,mesh_filter)
    mesh_names=[]
    for name in source.mesh_names():
        if not name or name in body_mesh_names:continue
        try:data=source.data(name)
        except Exception:continue
        if len(data.get('V',[])) and len(data.get('F',[])):mesh_names.append(name)
    if len(mesh_names)<=1:return _solve_garment_meshes(source,cache,body_mesh_names,mesh_filter)
    if len(mesh_names)>12:
        result=_solve_garment_meshes(source,cache,body_mesh_names,mesh_filter)
        result[3]['garment_mesh_workers']={'enabled':False,'reason':f'{len(mesh_names)} garment meshes exceeds safe isolated-worker cap of 12'}
        return result
    worker=_MODULE_DIR/'garment_mesh_worker.py';supervisor=_MODULE_DIR/'garment_group_supervisor.py';finalizer=_MODULE_DIR/'garment_finalize_worker.py'
    if not worker.exists():raise FileNotFoundError(worker)
    if not supervisor.exists():raise FileNotFoundError(supervisor)
    if not finalizer.exists():raise FileNotFoundError(finalizer)
    import pickle
    work_dir=Path(tempfile.mkdtemp(prefix='ravafit-garment-pipeline-'));started=time.perf_counter();processes=[];final_proc=None
    try:
        worker_cache={k:v for k,v in cache.items() if k not in {'_ravafit_local_affines','_ravafit_source_support_triangles','_ravafit_target_support_triangles','_ravafit_target_collision_triangles'}}
        common_path=work_dir/'common.pkl'
        with common_path.open('wb') as f:pickle.dump({'source_js':source.js,'cache':worker_cache},f,protocol=pickle.HIGHEST_PROTOCOL)
        rows=[]
        for index,name in enumerate(mesh_names):
            mesh_path=work_dir/f'mesh-{index}.pkl';output=work_dir/f'mesh-{index}.npz';report=work_dir/f'mesh-{index}.json'
            with mesh_path.open('wb') as f:pickle.dump({'name':name,'data':source.data(name)},f,protocol=pickle.HIGHEST_PROTOCOL)
            rows.append({'name':name,'mesh_payload':str(mesh_path),'output':str(output),'report':str(report)})
        manifest_path=work_dir/'manifest.json';manifest_path.write_text(json.dumps({'meshes':rows},separators=(',',':')),encoding='utf-8')
        worker_cache=None
        source=None
        cache=None
        parent_trim=_best_effort_trim_process_memory()
        ready_path=work_dir/'ready.flag';final_output=work_dir/'final.pkl';final_report=work_dir/'final.json'
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform.startswith('win') else 0;env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        group_report=work_dir/'group.json'
        group_proc=subprocess.Popen([sys.executable,str(supervisor),str(worker),str(common_path),str(manifest_path),str(group_report)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=creationflags,env=env)
        try:group_code=group_proc.wait(timeout=720)
        except subprocess.TimeoutExpired:
            group_proc.kill();group_code=group_proc.wait();raise RuntimeError('isolated garment supervisor timed out')
        group_payload=json.loads(group_report.read_text(encoding='utf-8')) if group_report.exists() else {'ok':False,'error':f'exit {group_code} without report'}
        if group_code!=0 or not group_payload.get('ok',False):raise RuntimeError(f'isolated garment supervisor failed: {group_payload.get("error") or group_code}')
        for row,result_row in zip(rows,group_payload.get('meshes',[])):
            row['wall_sec']=result_row.get('wall_sec')
        ready_path.write_text('ready',encoding='ascii')
        final_proc=subprocess.Popen([sys.executable,str(finalizer),str(common_path),str(manifest_path),str(ready_path),str(final_output),str(final_report)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=creationflags,env=env)
        try:final_code=final_proc.wait(timeout=240)
        except subprocess.TimeoutExpired:
            final_proc.kill();final_code=final_proc.wait();raise RuntimeError('fresh garment finalizer timed out')
        final_payload=json.loads(final_report.read_text(encoding='utf-8')) if final_report.exists() else {'ok':False,'error':f'exit {final_code} without report'}
        if final_code!=0 or not final_payload.get('ok',False) or not final_output.exists():raise RuntimeError(f'fresh garment finalizer failed: {final_payload.get("error") or final_code}')
        with final_output.open('rb') as f:result=pickle.load(f)
        stats=result['stats'];worker_stats=stats.setdefault('garment_mesh_workers',{});worker_stats['parent_wall_sec']=time.perf_counter()-started;worker_stats['prelaunched_finalizer']=False;worker_stats['fresh_finalizer_after_meshes']=True;worker_stats['authored_mesh_wave_size']=group_payload.get('wave_size',len(rows));worker_stats['supervisor_wall_sec']=group_payload.get('elapsed_sec');worker_stats['clean_group_supervisor']=True;worker_stats['clean_batch_worker']=group_payload.get('mode')=='one-clean-multimesh-worker';worker_stats['parent_trim']=parent_trim
        worker_stats['workers']=[dict(w,wall_sec=row.get('wall_sec')) for w,row in zip(worker_stats.get('workers',[]),rows)] if worker_stats.get('workers') else [{'mesh':row['name'],'wall_sec':row.get('wall_sec')} for row in rows]
        return result['positions'],result['skinning'],result['records'],stats
    finally:
        for _row,proc,_started in processes:
            if proc.poll() is None:
                try:proc.kill();proc.wait(timeout=5)
                except Exception:pass
        if final_proc is not None and final_proc.poll() is None:
            try:final_proc.kill();final_proc.wait(timeout=5)
            except Exception:pass
        shutil.rmtree(work_dir,ignore_errors=True)


def _write_retargeted_skinning(ed: GLBEditor, primitive: dict[str, Any], dense_weights: np.ndarray, mesh_name: str):
    attrs=primitive.get("attributes",{});pairs=[]
    for suffix in ("0","1"):
        jk=f"JOINTS_{suffix}";wk=f"WEIGHTS_{suffix}"
        if (jk in attrs)!=(wk in attrs):raise ValueError(f"{mesh_name} has incomplete {jk}/{wk} skinning attributes.")
        if jk in attrs:pairs.append((attrs[jk],attrs[wk]))
    if not pairs:raise ValueError(f"{mesh_name} has no writable JOINTS/WEIGHTS attributes for target-body skin retargeting.")
    capacity=4*len(pairs);weights=np.asarray(dense_weights,dtype=np.float64)
    if weights.ndim!=2:raise ValueError(f"{mesh_name} retargeted weights are not a dense 2D matrix.")
    # Geometry-only fits must not re-sort, re-normalise or re-quantise the
    # authored joint/weight accessors on export. Compare in the same normalised
    # dense representation used by the solver, then leave the original bytes.
    authored=np.zeros_like(weights)
    for joint_accessor,weight_accessor in pairs:
        joints=np.asarray(ed.accessor(joint_accessor),dtype=np.int64)
        values=np.asarray(ed.accessor(weight_accessor),dtype=np.float64)
        if len(joints)!=len(weights) or joints.shape!=values.shape:
            raise ValueError(f"{mesh_name} skin accessor count does not match solved weights.")
        if np.any(joints<0) or np.any(joints>=weights.shape[1]):
            raise ValueError(f"{mesh_name} source joint index exceeds its skin.")
        for corner in range(joints.shape[1]):
            authored[np.arange(len(weights)),joints[:,corner]]+=values[:,corner]
    total=authored.sum(axis=1);valid=total>1e-12
    authored[valid]/=total[valid,None]
    if np.array_equal(authored,weights):
        return {"mode":"source_authored_accessors_preserved","attributes_preserved":True,
                "influence_capacity":capacity,"max_active_influences":int(np.max(np.count_nonzero(weights>1e-8,axis=1),initial=0))}
    order=np.argsort(-weights,axis=1,kind="stable")[:,:capacity];packed=np.take_along_axis(weights,order,axis=1)
    total=packed.sum(axis=1,keepdims=True);good=total[:,0]>1e-12
    if not np.all(good):raise ValueError(f"{mesh_name} has {int(np.count_nonzero(~good))} vertices with no encodable skin weights.")
    packed/=total
    for pair_index,(joint_accessor,weight_accessor) in enumerate(pairs):
        jmeta=ed.js["accessors"][joint_accessor];wmeta=ed.js["accessors"][weight_accessor]
        jarr=ed.accessor(joint_accessor);warr=ed.accessor(weight_accessor)
        if jarr.shape[1]!=4 or warr.shape[1]!=4:raise ValueError(f"{mesh_name} skin accessor is not VEC4.")
        max_joint=np.iinfo(jarr.dtype).max if np.issubdtype(jarr.dtype,np.integer) else None
        joints=order[:,pair_index*4:(pair_index+1)*4]
        values=packed[:,pair_index*4:(pair_index+1)*4]
        if max_joint is None or (joints.size and int(np.max(joints))>int(max_joint)):
            raise ValueError(f"{mesh_name} target-body joint index exceeds {jarr.dtype} storage.")
        ed.write_accessor(joint_accessor,joints.astype(jarr.dtype))
        component=int(wmeta.get("componentType",5126));normalized=bool(wmeta.get("normalized",False))
        if component==5126:
            encoded=values.astype(np.float32)
        elif normalized and component==5121:
            encoded=np.rint(np.clip(values,0.0,1.0)*255.0).astype(np.uint8)
        elif normalized and component==5123:
            encoded=np.rint(np.clip(values,0.0,1.0)*65535.0).astype(np.uint16)
        else:
            raise ValueError(f"{mesh_name} uses unsupported WEIGHTS accessor encoding componentType={component}, normalized={normalized}.")
        ed.write_accessor(weight_accessor,encoded)
    return {"influence_capacity":capacity,"max_active_influences":int(np.max(np.count_nonzero(packed>1e-8,axis=1))) if len(packed) else 0}


def _rebuild_normals_preserving_authored_splits(vertices: np.ndarray, faces: np.ndarray, source_positions: np.ndarray, source_normals: np.ndarray):
    tri=trimesh.Trimesh(vertices=np.asarray(vertices,dtype=np.float64),faces=np.asarray(faces,dtype=np.int64),process=False)
    normals=np.asarray(tri.vertex_normals,dtype=np.float64).copy()
    source_positions=np.asarray(source_positions,dtype=np.float64);source_normals=np.asarray(source_normals,dtype=np.float64)
    key=np.round(source_positions/1e-7).astype(np.int64);_,inverse=np.unique(key,axis=0,return_inverse=True)
    grouped: dict[int,list[int]]={}
    for index,group in enumerate(inverse):grouped.setdefault(int(group),[]).append(index)
    smoothed_groups=0;smoothed_vertices=0
    for ids in grouped.values():
        if len(ids)<2:continue
        pending=set(ids)
        while pending:
            seed=pending.pop();cluster=[seed];seed_normal=source_normals[seed];seed_len=float(np.linalg.norm(seed_normal))
            if seed_len<=1e-12:continue
            seed_normal=seed_normal/seed_len
            matches=[]
            for candidate in pending:
                cn=source_normals[candidate];cl=float(np.linalg.norm(cn))
                if cl>1e-12 and float(np.dot(seed_normal,cn/cl))>=.98:matches.append(candidate)
            for candidate in matches:pending.remove(candidate);cluster.append(candidate)
            if len(cluster)<2:continue
            average=np.sum(normals[cluster],axis=0);length=float(np.linalg.norm(average))
            if length<=1e-12:continue
            normals[cluster]=average/length;smoothed_groups+=1;smoothed_vertices+=len(cluster)
    length=np.linalg.norm(normals,axis=1);good=length>1e-12;normals[good]/=length[good,None]
    return normals.astype(np.float32),{"smoothed_groups":smoothed_groups,"smoothed_vertices":smoothed_vertices}


def _patch_positions_and_detach_body(source_path: Path, output_path: Path, positions: dict[str, np.ndarray], skinning: dict[str, dict[str, Any]], body_mesh_names: set[str]):
    ed = GLBEditor(source_path)
    body_mesh_indices = {mi for mi, mesh in enumerate(ed.js.get("meshes", [])) if mesh.get("name") in body_mesh_names}
    normal_rebuild={}
    primitive_report={}

    # Production must treat every glTF primitive belonging to one XIV mesh as a
    # single authored mesh. Frozen B14's legacy GLB helpers expose primitives[0]
    # only, which silently discarded MeshParts from outfits such as Duskwing.
    read_view = GLB(source_path)
    for name, newpos in positions.items():
        source_data=aggregate_mesh_data(read_view,name)
        source_v=np.asarray(source_data["V"],dtype=np.float64)
        source_n=np.asarray(source_data["N"],dtype=np.float64) if source_data.get("N") is not None else None
        faces=np.asarray(source_data["F"],dtype=np.int64)
        new=np.asarray(newpos,dtype=np.float32)
        if source_v.shape != new.shape:
            raise ValueError(f"{name} aggregate position count mismatch {source_v.shape} vs {new.shape}")
        if source_n is not None:
            normals,normal_stage=_rebuild_normals_preserving_authored_splits(new,faces,source_v,source_n);normal_rebuild[name]=normal_stage
        else:
            tri=trimesh.Trimesh(vertices=new,faces=faces,process=False);normals=np.asarray(tri.vertex_normals,dtype=np.float32);normal_rebuild[name]={"smoothed_groups":0,"smoothed_vertices":0}

        mi,domains,primitives=mesh_domains(ed,name)
        written_accessors=set();domain_rows=[]
        skin = skinning.get(name)
        if skin is None:
            raise ValueError(f"Solved garment mesh {name} has no retargeted skinning payload.")
        solved_weights=np.asarray(skin["weights"],dtype=np.float64)
        if solved_weights.shape[0] != len(new):
            raise ValueError(f"{name} solved skinning rows {solved_weights.shape[0]} != aggregate vertices {len(new)}")
        packing_rows=[]

        for domain in domains:
            sl=slice(domain.start,domain.start+domain.count)
            domain_new=new[sl]
            domain_normals=normals[sl]
            if domain.position_accessor not in written_accessors:
                ed.write_accessor(domain.position_accessor,domain_new);written_accessors.add(domain.position_accessor)
                ed.js["accessors"][domain.position_accessor]["min"]=domain_new.min(axis=0).astype(float).tolist()
                ed.js["accessors"][domain.position_accessor]["max"]=domain_new.max(axis=0).astype(float).tolist()
            if domain.normal_accessor is not None and domain.normal_accessor not in written_accessors:
                ed.write_accessor(domain.normal_accessor,domain_normals);written_accessors.add(domain.normal_accessor)

            # Tangents need every face in this vertex domain, not merely primitive[0].
            domain_faces=[]
            for pi in domain.primitive_indices:
                primitive=primitives[pi]
                idx=np.asarray(ed.accessor(primitive["indices"]),dtype=np.int64).reshape(-1)
                if len(idx)%3:raise ValueError(f"{name} primitive {pi} index count is not triangular.")
                domain_faces.append(idx.reshape(-1,3))
            domain_faces=np.vstack(domain_faces) if domain_faces else np.zeros((0,3),dtype=np.int64)
            if domain.tangent_accessor is not None and domain.uv_accessor is not None and domain.normal_accessor is not None and domain.tangent_accessor not in written_accessors:
                uv=ed.accessor(domain.uv_accessor).astype(np.float32);fallback=ed.accessor(domain.tangent_accessor).astype(np.float32)
                ed.write_accessor(domain.tangent_accessor,compute_tangents(domain_new,domain_faces,uv,domain_normals,fallback));written_accessors.add(domain.tangent_accessor)

            representative=primitives[domain.primitive_indices[0]]
            packing_rows.append(_write_retargeted_skinning(ed,representative,solved_weights[sl],name))
            domain_rows.append({"start":int(domain.start),"vertices":int(domain.count),"primitives":[int(v) for v in domain.primitive_indices]})

        # All domains of a logical XIV mesh carry one solver skinning stage. Preserve
        # the old public shape while recording the actual primitive/domain fan-out.
        skin["packing"] = packing_rows[0] if len(packing_rows)==1 else {"mode":"multi_primitive_domains","domains":packing_rows}
        primitive_report[name]={"primitive_count":len(primitives),"vertex_domain_count":len(domains),"domains":domain_rows}

    detached_nodes = []
    for ni, node in enumerate(ed.js.get("nodes", [])):
        if node.get("mesh") in body_mesh_indices:
            detached_nodes.append(ni)
            node.pop("mesh", None)
            node.pop("skin", None)

    for scene in ed.js.get("scenes", []):
        scene["nodes"] = [ni for ni in scene.get("nodes", []) if ni not in detached_nodes]

    bounds = _refresh_float_accessor_bounds(ed)
    _validate_float_accessor_bounds(ed)
    ed.save(output_path)
    serialized_bounds = _validate_float_accessor_bounds(GLBEditor(output_path))
    return {
        "body_mesh_indices": sorted(body_mesh_indices),
        "detached_nodes": detached_nodes,
        "normal_rebuild": normal_rebuild,
        "multi_primitive_meshes": primitive_report,
        "retargeted_skinning": {name: {"stage": value.get("stage"), "packing": value.get("packing")} for name, value in skinning.items()},
        "accessor_bounds": {
            "checked": bounds["checked"],
            "repaired_count": bounds["repaired_count"],
            "repaired": bounds["repaired"],
            "serialized_checked": serialized_bounds["checked"],
            "serialized_sparse_checked": serialized_bounds.get("sparse_checked", 0),
        },
    }

def _align4(buf: bytearray):
    while len(buf) % 4:
        buf.append(0)


def _accessor_values_with_sparse(ed: GLBEditor, index: int) -> np.ndarray:
    """Return the resolved accessor payload, including sparse overrides."""
    component_types = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    component_counts = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    accessor = ed.js["accessors"][index]
    component_type = int(accessor["componentType"])
    if component_type not in component_types:
        raise ValueError(f"Accessor {index} uses unsupported componentType {component_type}.")
    accessor_type = str(accessor["type"])
    if accessor_type not in component_counts:
        raise ValueError(f"Accessor {index} uses unsupported type {accessor_type}.")

    dtype = np.dtype(component_types[component_type]).newbyteorder("<")
    components = component_counts[accessor_type]
    count = int(accessor.get("count", 0))

    if "bufferView" in accessor:
        values = np.asarray(ed.accessor(index), dtype=dtype).copy()
    else:
        # glTF permits a missing base bufferView for sparse accessors; the base is zero.
        values = np.zeros((count, components), dtype=dtype)

    sparse = accessor.get("sparse")
    if not sparse:
        return values

    sparse_count = int(sparse.get("count", 0))
    if sparse_count <= 0:
        return values

    indices = sparse["indices"]
    index_component_type = int(indices["componentType"])
    if index_component_type not in {5121, 5123, 5125}:
        raise ValueError(f"Accessor {index} sparse indices use invalid componentType {index_component_type}.")
    index_dtype = np.dtype(component_types[index_component_type]).newbyteorder("<")
    index_view = ed.js["bufferViews"][int(indices["bufferView"])]
    index_offset = int(index_view.get("byteOffset", 0)) + int(indices.get("byteOffset", 0))
    sparse_indices = np.frombuffer(ed.bin, dtype=index_dtype, count=sparse_count, offset=index_offset).astype(np.int64)
    if np.any(sparse_indices < 0) or np.any(sparse_indices >= count):
        raise ValueError(f"Accessor {index} sparse indices point outside its {count} elements.")

    sparse_values = sparse["values"]
    value_view = ed.js["bufferViews"][int(sparse_values["bufferView"])]
    value_offset = int(value_view.get("byteOffset", 0)) + int(sparse_values.get("byteOffset", 0))
    replacement = np.frombuffer(ed.bin, dtype=dtype, count=sparse_count * components, offset=value_offset).reshape(sparse_count, components)
    values[sparse_indices] = replacement
    return values


def _refresh_float_accessor_bounds(ed: GLBEditor) -> dict[str, Any]:
    """Refresh float accessor bounds from the resolved binary payload."""
    repaired: list[dict[str, Any]] = []
    checked = 0
    position_accessors = {
        ai
        for mesh in ed.js.get("meshes", [])
        for primitive in mesh.get("primitives", [])
        for ai in ([primitive.get("attributes", {}).get("POSITION")]
                   + [target.get("POSITION") for target in primitive.get("targets", [])])
        if ai is not None
    }
    for index, accessor in enumerate(ed.js.get("accessors", [])):
        if accessor.get("componentType") != 5126:  # FLOAT
            continue
        if accessor.get("type") not in {"SCALAR", "VEC2", "VEC3", "VEC4"}:
            continue
        if "bufferView" not in accessor and "sparse" not in accessor:
            continue

        # POSITION bounds are mandatory; refresh other float bounds only when they already existed.
        is_position = index in position_accessors
        if not is_position and "min" not in accessor and "max" not in accessor:
            continue

        values = np.asarray(_accessor_values_with_sparse(ed, index), dtype=np.float32)
        if values.size == 0:
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"Accessor {index} contains non-finite float data.")

        actual_min = values.min(axis=0).astype(np.float32)
        actual_max = values.max(axis=0).astype(np.float32)
        new_min = [float(v) for v in actual_min.tolist()]
        new_max = [float(v) for v in actual_max.tolist()]
        old_min = copy.deepcopy(accessor.get("min"))
        old_max = copy.deepcopy(accessor.get("max"))
        accessor["min"] = new_min
        accessor["max"] = new_max
        checked += 1
        if old_min != new_min or old_max != new_max:
            repaired.append({
                "accessor": index,
                "type": accessor.get("type"),
                "position": is_position,
                "sparse": "sparse" in accessor,
                "old_min": old_min,
                "old_max": old_max,
                "new_min": new_min,
                "new_max": new_max,
            })

    return {"checked": checked, "repaired": repaired, "repaired_count": len(repaired)}


def _validate_float_accessor_bounds(ed: GLBEditor) -> dict[str, Any]:
    """Fail if serialised float bounds disagree with the resolved payload."""
    checked = 0
    sparse_checked = 0
    for index, accessor in enumerate(ed.js.get("accessors", [])):
        if accessor.get("componentType") != 5126:
            continue
        if "bufferView" not in accessor and "sparse" not in accessor:
            continue
        if "min" not in accessor and "max" not in accessor:
            continue
        if accessor.get("type") not in {"SCALAR", "VEC2", "VEC3", "VEC4"}:
            continue

        values = np.asarray(_accessor_values_with_sparse(ed, index), dtype=np.float32)
        if values.size == 0:
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"Accessor {index} contains non-finite float data.")
        actual_min = values.min(axis=0).astype(np.float32)
        actual_max = values.max(axis=0).astype(np.float32)
        declared_min = np.asarray(accessor.get("min", actual_min), dtype=np.float32)
        declared_max = np.asarray(accessor.get("max", actual_max), dtype=np.float32)
        if declared_min.shape != actual_min.shape or declared_max.shape != actual_max.shape:
            raise ValueError(f"Accessor {index} bounds have the wrong component count.")
        if np.any(actual_min < declared_min) or np.any(actual_max > declared_max):
            raise ValueError(
                f"Accessor {index} bounds do not enclose serialized data: "
                f"declared min={declared_min.tolist()} max={declared_max.tolist()} "
                f"actual min={actual_min.tolist()} max={actual_max.tolist()} sparse={'sparse' in accessor}"
            )
        checked += 1
        if "sparse" in accessor:
            sparse_checked += 1
    return {"checked": checked, "sparse_checked": sparse_checked}


def _clone_accessor(dst: GLBEditor, src: GLB, accessor_index: int, cache: dict[tuple[int, tuple[tuple[int, int], ...] | None], int], joint_map: dict[int, int] | None = None) -> int:
    mapping_key = tuple(sorted(joint_map.items())) if joint_map is not None else None
    key = (accessor_index, mapping_key)
    if key in cache:
        return cache[key]
    accessor = src.js["accessors"][accessor_index]
    # Resolve sparse target-body accessors and store them densely in the fresh output buffer.
    arr = np.asarray(_accessor_values_with_sparse(src, accessor_index)).copy()
    joint_component_type = None
    if joint_map is not None:
        source_component_type = int(accessor.get("componentType", 0))
        if source_component_type not in (5121, 5123):
            raise ValueError(f"Target body JOINTS accessor uses unsupported componentType {source_component_type}; expected UNSIGNED_BYTE or UNSIGNED_SHORT.")
        flat = arr.astype(np.int64, copy=False)
        mapped = np.empty_like(flat)
        for index in np.ndindex(flat.shape):
            old = int(flat[index])
            if old not in joint_map:
                raise ValueError(f"Target body JOINTS accessor references unmapped joint index {old}.")
            mapped[index] = joint_map[old]
        max_joint = int(mapped.max(initial=0)) if mapped.size else 0
        if max_joint > 65535:
            raise ValueError(f"Target-body skin joint index {max_joint} exceeds glTF UNSIGNED_SHORT storage.")
        if max_joint > 255 or source_component_type == 5123:
            arr = mapped.astype(np.dtype('<u2'))
            joint_component_type = 5123
        else:
            arr = mapped.astype(np.uint8)
            joint_component_type = 5121

    _align4(dst.bin)
    byte_offset = len(dst.bin)
    raw = np.ascontiguousarray(arr).tobytes(order="C")
    dst.bin.extend(raw)
    source_bv = src.js["bufferViews"][accessor["bufferView"]] if "bufferView" in accessor else None
    bv = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(raw)}
    if source_bv is not None and "target" in source_bv:
        bv["target"] = source_bv["target"]
    dst.js.setdefault("bufferViews", []).append(bv)
    bvi = len(dst.js["bufferViews"]) - 1
    cloned = {k: copy.deepcopy(v) for k, v in accessor.items() if k not in ("bufferView", "byteOffset", "sparse")}
    cloned["bufferView"] = bvi
    cloned["byteOffset"] = 0
    if joint_map is not None:
        cloned["componentType"] = joint_component_type
        cloned.pop("min", None)
        cloned.pop("max", None)
    dst.js.setdefault("accessors", []).append(cloned)
    ai = len(dst.js["accessors"]) - 1
    cache[key] = ai
    return ai




def _validate_xiv_weighted_bone_budget(ed: GLBEditor) -> dict[str, Any]:
    meshes = []
    for mesh_index, mesh in enumerate(ed.js.get("meshes", [])):
        used: set[int] = set()
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            attrs = primitive.get("attributes", {})
            joint_sets = sorted(name for name in attrs if name.startswith("JOINTS_"))
            for joint_semantic in joint_sets:
                suffix = joint_semantic.split("_", 1)[1]
                weight_semantic = f"WEIGHTS_{suffix}"
                if weight_semantic not in attrs:
                    raise ValueError(f"Mesh {mesh_index} primitive {primitive_index} has {joint_semantic} without {weight_semantic}.")
                joints = np.asarray(_accessor_values_with_sparse(ed, attrs[joint_semantic]), dtype=np.int64)
                weights = np.asarray(_accessor_values_with_sparse(ed, attrs[weight_semantic]))
                if joints.shape != weights.shape:
                    raise ValueError(f"Mesh {mesh_index} primitive {primitive_index} joint/weight accessor shape mismatch: {joints.shape} vs {weights.shape}.")
                positive = weights.astype(np.float64, copy=False) > 0.0
                used.update(int(value) for value in joints[positive].tolist())
        if len(used) > 64:
            name = str(mesh.get("name") or f"mesh {mesh_index}")
            raise ValueError(f"{name} uses {len(used)} weighted joints; FFXIV MDL supports at most 64 weighted bone mappings per mesh.")
        meshes.append({"mesh_index": mesh_index, "name": str(mesh.get("name") or ""), "weighted_joints": len(used)})
    return {"maximum_weighted_joints_per_mesh": 64, "meshes": meshes}

def _ensure_material(dst: GLBEditor, name: str) -> int:
    """Return or add the XIV material-name placeholder used by Penumbra's importer."""
    wanted = _normalise_material(name)
    materials = dst.js.setdefault("materials", [])
    for i, material in enumerate(materials):
        if _normalise_material(material.get("name", "")) == wanted:
            return i
    materials.append({"name": str(name).replace("\\", "/").strip()})
    return len(materials) - 1


def _parse_mesh_name(name: str):
    m = _MESH_RE.match(str(name or ""))
    if not m:
        return None
    return int(m.group("mesh")), int(m.group("sub") or 0)


def _bone_region(name: str) -> str | None:
    """Map XIV skeleton joints to body-library regions."""
    n = str(name or "").casefold()
    if n.startswith((
        "j_asi_a_", "j_asi_b_", "j_asi_c_",
        "iv_daitai", "ya_daitai", "iv_shiri", "ya_shiri",
        "j_sk_", "n_hiza",
    )):
        return "Legs"
    if n.startswith(("j_asi_d_", "j_asi_e_", "iv_asi_", "ya_asi_")):
        return "Feet"
    if n.startswith((
        "j_te_", "j_hito_", "j_ko_", "j_kusu_", "j_naka_", "j_oya_",
        "iv_hito_", "iv_ko_", "iv_kusu_", "iv_naka_", "iv_oya_",
    )) and "asi_" not in n:
        return "Hands"
    if n.startswith((
        "j_sebo_", "j_mune_", "j_sako_", "j_ude_", "j_kata_", "j_kubi",
        "iv_mune", "iv_sako", "iv_ude", "ya_mune", "ya_sako", "ya_ude",
    )):
        return "Chest"
    return None


def _entry_region_evidence(glb: GLB, entry: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    data = glb.data(entry["mesh_name"])
    weights = np.asarray(data["W"], dtype=np.float64)
    names = list(data["joint_names"])
    count = int(len(weights))
    out: dict[str, dict[str, float | int]] = {}
    for slot in _SLOT_SUFFIX.values():
        ids = [i for i, name in enumerate(names) if _bone_region(name) == slot]
        region_weight = np.sum(weights[:, ids], axis=1) if ids else np.zeros(count, dtype=np.float64)
        strong = int(np.count_nonzero(region_weight >= 0.42))
        out[slot] = {
            "vertices": count,
            "strong_vertices": strong,
            "strong_fraction": float(strong / max(count, 1)),
            "mean_weight": float(np.mean(region_weight)) if count else 0.0,
            "max_weight": float(np.max(region_weight)) if count else 0.0,
        }
    return out


def _strong_region(row: dict[str, float | int]) -> bool:
    vertices = int(row.get("vertices", 0))
    strong = int(row.get("strong_vertices", 0))
    fraction = float(row.get("strong_fraction", 0.0))
    mean = float(row.get("mean_weight", 0.0))
    return strong >= max(16, int(np.ceil(vertices * 0.01))) and (fraction >= 0.03 or mean >= 0.08)


def _source_body_info(glb: GLB, source_materials: set[str]):
    entries = []
    for mi, mesh in enumerate(glb.js.get("meshes", [])):
        mats = _primitive_material_names(glb, mi)
        matching = [m for m in mats if _material_matches_body_package(m,source_materials)]
        if not matching:
            continue
        # If a glTF mesh ever mixes body and garment materials, fail rather than deleting the whole node.
        non_body = [m for m in mats if not _material_matches_body_package(m,source_materials)]
        if non_body:
            raise ValueError(
                f"Body/garment material mixing was found inside {mesh.get('name') or f'mesh {mi}'}: "
                f"body={matching}, non-body={non_body}. RavaFit will not detach a mixed mesh."
            )
        ni = _mesh_node_index(glb, mi)
        if ni is None:
            continue
        node = glb.js["nodes"][ni]
        skin_index = node.get("skin")
        if skin_index is None:
            raise ValueError(f"Source body mesh {mesh.get('name')} is not skinned.")
        entries.append({
            "mesh_index": mi,
            "mesh_name": mesh.get("name") or f"mesh {mi}",
            "node_index": ni,
            "skin_index": int(skin_index),
            "material": matching[0],
        })
    if not entries:
        raise ValueError("No source body nodes were found for body transplant.")
    return entries


def _assign_source_body_regions(glb: GLB, entries: list[dict[str, Any]], source_materials_by_slot: dict[str, set[str]], model_slot: str):
    selected_slots = [slot for slot in ("Chest", "Legs", "Hands", "Feet") if slot in source_materials_by_slot]
    if model_slot not in selected_slots:
        raise ValueError(f"The source body context does not contain the model's primary {model_slot} slot.")

    by_slot: dict[str, list[dict[str, Any]]] = {slot: [] for slot in selected_slots}
    evidence_report: list[dict[str, Any]] = []
    for entry in entries:
        evidence = _entry_region_evidence(glb, entry)
        material_slots = [slot for slot in selected_slots if _material_matches_body_package(entry["material"],source_materials_by_slot.get(slot,set()))]
        if len(material_slots) == 1:
            assigned = material_slots[0]
            reason = "unique source material"
        else:
            ranked = sorted(selected_slots, key=lambda slot: (
                int(evidence[slot]["strong_vertices"]),
                float(evidence[slot]["strong_fraction"]),
                float(evidence[slot]["mean_weight"]),
            ), reverse=True)
            best = ranked[0]
            if _strong_region(evidence[best]):
                assigned = best
                reason = "direct skeleton-weight anatomy"
            elif model_slot in material_slots or not material_slots:
                assigned = model_slot
                reason = "primary model-slot fallback"
            else:
                assigned = material_slots[0]
                reason = "shared-material fallback"
        by_slot.setdefault(assigned, []).append(entry)
        evidence_report.append({
            "mesh": entry["mesh_name"],
            "material": entry["material"],
            "assigned_slot": assigned,
            "reason": reason,
            "material_slots": material_slots,
            "evidence": evidence,
        })

    if not by_slot.get(model_slot):
        raise ValueError(f"Could not associate any embedded source-body geometry with the primary {model_slot} slot.")

    present = [slot for slot in selected_slots if by_slot.get(slot)]
    return by_slot, present, evidence_report


def _masked_vertex_components(faces: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    F=np.asarray(faces,dtype=np.int64);mask=np.asarray(mask,dtype=bool);neighbours=[set() for _ in range(len(mask))]
    for a,b,c in F:
        a=int(a);b=int(b);c=int(c)
        if mask[a] and mask[b]:neighbours[a].add(b);neighbours[b].add(a)
        if mask[b] and mask[c]:neighbours[b].add(c);neighbours[c].add(b)
        if mask[c] and mask[a]:neighbours[c].add(a);neighbours[a].add(c)
    seen=set();out=[]
    for first in np.where(mask)[0]:
        first=int(first)
        if first in seen:continue
        stack=[first];seen.add(first);rows=[]
        while stack:
            current=stack.pop();rows.append(current)
            for neighbour in neighbours[current]:
                if neighbour not in seen:seen.add(neighbour);stack.append(neighbour)
        out.append(np.asarray(rows,dtype=np.int64))
    return out


def _masked_face_components(faces: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Connected components of a selected triangle subset, joined only by shared edges."""
    F=np.asarray(faces,dtype=np.int64);mask=np.asarray(mask,dtype=bool)
    selected=np.flatnonzero(mask)
    if not len(selected):return []
    edge_faces={}
    for fi in selected.tolist():
        a,b,c=(int(x) for x in F[fi])
        for u,v in ((a,b),(b,c),(c,a)):
            key=(u,v) if u<v else (v,u);edge_faces.setdefault(key,[]).append(int(fi))
    neighbours={int(fi):set() for fi in selected.tolist()}
    for rows in edge_faces.values():
        if len(rows)<2:continue
        for i in range(len(rows)):
            for j in range(i+1,len(rows)):
                neighbours[rows[i]].add(rows[j]);neighbours[rows[j]].add(rows[i])
    seen=set();out=[]
    for first in selected.tolist():
        if first in seen:continue
        stack=[int(first)];seen.add(int(first));rows=[]
        while stack:
            current=stack.pop();rows.append(current)
            for neighbour in neighbours.get(current,()):
                if neighbour not in seen:seen.add(neighbour);stack.append(neighbour)
        out.append(np.asarray(rows,dtype=np.int64))
    return sorted(out,key=len,reverse=True)


def _source_body_suppression_plan(source: GLB, cache: dict[str, Any], source_by_slot: dict[str, list[dict[str, Any]]], present_slots: list[str], body_mesh_names: set[str], mesh_filter: set[str] | None):
    garment_triangles=[]
    for mesh_name in source.mesh_names():
        if not mesh_name or mesh_name in body_mesh_names or (mesh_filter is not None and mesh_name not in mesh_filter):continue
        try:data=source.data(mesh_name)
        except Exception:continue
        V=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64)
        if len(V) and len(F):garment_triangles.append(V[F])
    garment_tri=np.vstack(garment_triangles) if garment_triangles else np.zeros((0,3,3),dtype=np.float64)
    pair_by_slot={str(pair.get("slot")):pair for pair in cache.get("slot_pairs",[])}
    literal_collision=[];slot_reports=[];native_suppression=[];source_collision=[];source_collision_meshes=set()
    for slot in ("Chest","Legs","Hands","Feet"):
        pair=pair_by_slot.get(slot)
        if pair is None:continue
        target_V=np.asarray(pair["target_literal_V"],dtype=np.float64);target_F=np.asarray(pair["target_literal_F"],dtype=np.int64);keep=np.ones(len(target_F),dtype=bool)
        report={"slot":slot,"source_body_present":slot in present_slots,"status":"no source-body removal evidence","accepted_components":[],"suppressed_target_triangles":0,"native_meshes":[]}
        if slot in present_slots and source_by_slot.get(slot) and len(garment_tri):
            actual=[]
            for entry in source_by_slot[slot]:
                mesh_name=str(entry.get("mesh_name") or "")
                try:body_data=source.data(mesh_name)
                except Exception:continue
                V=np.asarray(body_data.get("V",[]),dtype=np.float64);F=np.asarray(body_data.get("F",[]),dtype=np.int64)
                if len(V) and len(F):
                    triangles=V[F];actual.append(triangles)
                    if mesh_name and mesh_name not in source_collision_meshes:
                        source_collision.append(triangles);source_collision_meshes.add(mesh_name)
            source_V=np.asarray(pair["source_literal_V"],dtype=np.float64);source_F=np.asarray(pair["source_literal_F"],dtype=np.int64)
            if actual and len(source_V):
                actual_tri=np.vstack(actual);_,_,_,body_distance,_=_b14_nearest_surface(source_V,actual_tri,k=32)
                p50=float(np.percentile(body_distance,50));p75=float(np.percentile(body_distance,75));p90=float(np.percentile(body_distance,90))
                report["source_match_mm"]={"p50":p50*1000.0,"p75":p75*1000.0,"p90":p90*1000.0}
                if p50<=.0035 and p75<=.0065:
                    threshold=max(.0055,p75+.0035);lo=np.min(source_V,axis=0);hi=np.max(source_V,axis=0);span=np.maximum(hi-lo,1e-6)
                    x_band=max(.005,float(span[0])*.04);y_band=max(.005,float(span[1])*.04)
                    outer=(source_V[:,0]-lo[0]<=x_band)|(hi[0]-source_V[:,0]<=x_band)|(source_V[:,1]-lo[1]<=y_band)|(hi[1]-source_V[:,1]<=y_band)
                    missing=(body_distance>threshold)&(~outer)
                    _,_,_,garment_distance,_=_b14_nearest_surface(source_V,garment_tri,k=32)
                    covered=garment_distance<=.030
                    accepted=[]
                    for component in _masked_vertex_components(source_F,missing):
                        if len(component)<max(20,int(np.ceil(len(source_V)*.004))):continue
                        cover_fraction=float(np.mean(covered[component]));median_cover=float(np.median(garment_distance[component]));median_missing=float(np.median(body_distance[component]))
                        if cover_fraction<.70 or median_cover>.022 or median_missing<threshold+.001:continue
                        accepted.append(component);report["accepted_components"].append({"vertices":int(len(component)),"coverage_fraction":cover_fraction,"garment_distance_p50_mm":median_cover*1000.0,"missing_depth_p50_mm":median_missing*1000.0})
                    if accepted:
                        ids=np.unique(np.concatenate(accepted));mapped=np.asarray(pair["Y"],dtype=np.float64)[ids];source_weights=np.asarray(pair["BW"],dtype=np.float64)[ids]
                        edges=np.vstack((source_F[:,[0,1]],source_F[:,[1,2]],source_F[:,[2,0]]));edge_len=np.linalg.norm(source_V[edges[:,0]]-source_V[edges[:,1]],axis=1);radius=float(np.clip(np.median(edge_len)*4.0,.006,.018))
                        centres=target_V[target_F].mean(axis=1);tree=cKDTree(mapped);distance,index=tree.query(centres,k=1);target_weights=np.asarray(pair["target_literal_W"],dtype=np.float64)[target_F].mean(axis=1);alignment=np.einsum("ij,ij->i",target_weights,source_weights[np.asarray(index,dtype=np.int64)])
                        suppress=(distance<=radius)&(alignment>=.20);keep&=~suppress
                        report["status"]="source body explicitly removes garment-covered geometry";report["suppression_radius_mm"]=radius*1000.0;report["suppressed_target_triangles"]=int(np.count_nonzero(suppress))
                        records=[]
                        for record in pair.get("target_mesh_records",[]):
                            start=int(record.get("face_offset",0));count=int(record.get("face_count",0));local=np.where(suppress[start:start+count])[0].astype(int).tolist()
                            if not local:continue
                            row={"slot":slot,"native_mesh":int(record.get("mesh_index",0)),"triangles":local};records.append(row);native_suppression.append(row)
                        report["native_meshes"]=[{"native_mesh":row["native_mesh"],"suppressed_triangles":len(row["triangles"])} for row in records]
                    if not accepted:
                        # Fallback: classify the *target* body triangles directly.  This catches explicit
                        # source-body cutaways whose missing region is too sparse or fragmented to form a
                        # large component on the canonical source reference (common around crotch/chest
                        # garment cutouts).  Target centres are mapped back through the authoritative
                        # X<->Y body correspondence, then tested against the body geometry physically
                        # present in the untouched outfit.  The garment must also cover the mapped source
                        # location, so arbitrary source holes/LOD cuts cannot delete unrelated target body.
                        X=np.asarray(pair.get("X",[]),dtype=np.float64);Y=np.asarray(pair.get("Y",[]),dtype=np.float64);BW=np.asarray(pair.get("BW",[]),dtype=np.float64)
                        target_weights_all=np.asarray(pair.get("target_literal_W",[]),dtype=np.float64)
                        if len(X)==len(Y) and len(X)>=8 and len(target_V) and len(target_F) and len(target_weights_all)==len(target_V):
                            centres=target_V[target_F].mean(axis=1);k=min(8,len(Y));distance,index=cKDTree(Y).query(centres,k=k)
                            if np.asarray(index).ndim==1:index=np.asarray(index)[:,None];distance=np.asarray(distance)[:,None]
                            distance=np.asarray(distance,dtype=np.float64);index=np.asarray(index,dtype=np.int64)
                            inv=1.0/np.maximum(distance,.00035)**2;inv/=np.maximum(inv.sum(axis=1,keepdims=True),1e-12)
                            mapped_source=np.einsum("nk,nkj->nj",inv,X[index]);mapped_source_weights=np.einsum("nk,nkj->nj",inv,BW[index]) if BW.ndim==2 and len(BW)==len(X) else None
                            _,_,_,mapped_body_distance,_=_b14_nearest_surface(mapped_source,actual_tri,k=32);_,_,_,mapped_garment_distance,_=_b14_nearest_surface(mapped_source,garment_tri,k=32)
                            mapped_body_distance=np.asarray(mapped_body_distance,dtype=np.float64);mapped_garment_distance=np.asarray(mapped_garment_distance,dtype=np.float64)
                            target_nearest=np.min(distance,axis=1);candidate=(mapped_body_distance>threshold)&(mapped_garment_distance<=.022)&(target_nearest<=.010)
                            if mapped_source_weights is not None and mapped_source_weights.shape[1]==target_weights_all.shape[1]:
                                target_face_weights=target_weights_all[target_F].mean(axis=1);alignment=np.einsum("ij,ij->i",target_face_weights,mapped_source_weights);candidate&=alignment>=.20
                            selected_components=[];suppress=np.zeros(len(target_F),dtype=bool)
                            for component in _masked_face_components(target_F,candidate):
                                if len(component)<max(8,int(np.ceil(len(target_F)*.001))):continue
                                missing_p50=float(np.median(mapped_body_distance[component]));garment_p50=float(np.median(mapped_garment_distance[component]));map_p95=float(np.percentile(target_nearest[component],95))
                                if missing_p50<threshold+.001 or garment_p50>.022 or map_p95>.010:continue
                                suppress[component]=True;selected_components.append(component)
                                report["accepted_components"].append({"target_triangles":int(len(component)),"coverage_fraction":1.0,"garment_distance_p50_mm":garment_p50*1000.0,"missing_depth_p50_mm":missing_p50*1000.0,"mode":"target-centric explicit source cutaway"})
                            if np.any(suppress):
                                keep&=~suppress;report["status"]="source body explicitly removes garment-covered geometry";report["suppressed_target_triangles"]=int(np.count_nonzero(suppress));report["target_centric_fallback"]=True
                                records=[]
                                for record in pair.get("target_mesh_records",[]):
                                    start=int(record.get("face_offset",0));count=int(record.get("face_count",0));local=np.where(suppress[start:start+count])[0].astype(int).tolist()
                                    if not local:continue
                                    row={"slot":slot,"native_mesh":int(record.get("mesh_index",0)),"triangles":local};records.append(row);native_suppression.append(row)
                                report["native_meshes"]=[{"native_mesh":row["native_mesh"],"suppressed_triangles":len(row["triangles"])} for row in records]
                else:report["status"]="source body does not match selected source closely enough; suppression disabled"
        elif slot not in present_slots:
            report["status"]="fit-context only; no embedded source body, so nothing may be removed"
        literal_collision.append(target_V[target_F[keep]]);slot_reports.append(report)
    collision_tri=np.vstack(literal_collision) if literal_collision else np.zeros((0,3,3),dtype=np.float64)
    source_collision_tri=np.vstack(source_collision) if source_collision else np.zeros((0,3,3),dtype=np.float64)
    if len(source_collision_tri):source_collision_tri,_=_sanitise_strict_b14_surface_triangles(source_collision_tri,"embedded source collision")
    return {"enabled":True,"policy":"selected target stays complete unless the embedded source body explicitly removes a coherent garment-covered region","slots":slot_reports,"native_suppression":native_suppression,"_collision_triangles":collision_tri,"_source_collision_triangles":source_collision_tri}


def _refine_source_body_suppression_after_fit(source, positions, cache, plan):
    """Keep authored cutaways only where the final garment still covers the body."""
    proposed=plan.get("native_suppression",[])
    if not proposed:return plan
    garment=[]
    for name,vertices in positions.items():
        faces=np.asarray(source.data(name)["F"],dtype=np.int64)
        if len(faces):garment.append(np.asarray(vertices,dtype=np.float64)[faces])
    cloth=np.vstack(garment) if garment else np.zeros((0,3,3))
    cloth,_=_sanitise_strict_b14_surface_triangles(cloth,"fitted garment visibility")
    pairs={str(pair["slot"]):pair for pair in cache.get("slot_pairs",[])}
    result=dict(plan);accepted=[];counts={};total_before=0;total_after=0
    for row in proposed:
        slot=str(row["slot"]);pair=pairs.get(slot)
        ids=np.asarray(row["triangles"],dtype=np.int64);total_before+=len(ids)
        if pair is None:continue
        record=next((r for r in pair.get("target_mesh_records",[]) if int(r["mesh_index"])==int(row["native_mesh"])),None)
        if record is None:continue
        if np.any(ids<0) or np.any(ids>=int(record["face_count"])):
            raise ValueError(f"Invalid target-body cutaway indices for {slot} mesh {row['native_mesh']}.")
        faces=np.asarray(pair["target_literal_F"],dtype=np.int64)[ids+int(record["face_offset"])]
        triangles=np.asarray(pair["target_literal_V"],dtype=np.float64)[faces]
        retained=ids[covered_body_faces(triangles,cloth)].astype(int).tolist();total_after+=len(retained)
        counts.setdefault(slot,[]).append({"native_mesh":int(row["native_mesh"]),"suppressed_triangles":len(retained)})
        if retained:accepted.append({**row,"triangles":retained})
    result["native_suppression"]=accepted
    result["slots"]=[{**row,"suppressed_target_triangles":sum(r["suppressed_triangles"] for r in counts.get(str(row["slot"]),[])),
                      "native_meshes":counts.get(str(row["slot"]),[])} for row in plan.get("slots",[])]
    result["fitted_coverage"]={"enabled":True,"proposed_triangles":total_before,"suppressed_triangles":total_after,
                               "retained_at_openings":total_before-total_after,
                               "policy":"source cutaway evidence plus final outward cloth coverage at every corner and centre; never changes fit geometry"}
    return result


def _public_source_body_suppression(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:return {"enabled":False}
    return {key:value for key,value in plan.items() if not str(key).startswith("_")}


def _target_body_nodes(glb: GLB, target_mesh_materials: dict[int, dict[str, Any]]):
    """Bind target GLB pieces to RBODY's per-mesh material assignments."""
    entries = []
    found_mesh_indices: set[int] = set()
    for ni, node in enumerate(glb.js.get("nodes", [])):
        mi = node.get("mesh")
        if mi is None:
            continue
        mesh = glb.js["meshes"][mi]
        mesh_name = mesh.get("name") or f"mesh {mi}"
        parsed = _parse_mesh_name(mesh_name)
        if parsed is None:
            continue
        xiv_mesh_index, _ = parsed
        assignment = target_mesh_materials.get(xiv_mesh_index)
        if assignment is None:
            continue
        if node.get("skin") is None:
            raise ValueError(f"Target body mesh {mesh_name} is not skinned.")

        actual = _primitive_material_names(glb, mi)
        expected = assignment["normalised_material"]
        if not actual:
            raise ValueError(f"Target body mesh {mesh_name} has no glTF primitives/material assignment.")
        mismatched = [material for material in actual if material != expected]
        if mismatched:
            raise ValueError(
                f"Target body export material mismatch for XIV mesh {xiv_mesh_index}: RBODY assigns "
                f"{assignment['material']!r}, but {mesh_name} exported {actual}."
            )
        found_mesh_indices.add(xiv_mesh_index)
        entries.append({
            "node_index": ni,
            "mesh_index": int(mi),
            "xiv_mesh_index": xiv_mesh_index,
            "skin_index": int(node["skin"]),
            "primitive_indices": list(range(len(mesh.get("primitives", [])))),
            "mesh_name": mesh_name,
            "material": assignment["material"],
            "material_table_index": assignment["material_table_index"],
            "rbody_record_ordinal": assignment["record_ordinal"],
        })
    missing = sorted(set(target_mesh_materials) - found_mesh_indices)
    if missing:
        available = sorted(str(mesh.get("name") or f"mesh {i}") for i, mesh in enumerate(glb.js.get("meshes", [])))
        raise ValueError(
            f"Target body export is missing RBODY standard mesh index(es) {missing}. "
            f"Exported target meshes were {available}."
        )
    if not entries:
        raise ValueError("Selected target RBODY payload contains no transplantable body-package geometry in the exported GLB.")
    return entries


def _filter_target_entries_for_slot(glb: GLB, entries: list[dict[str, Any]], slot: str, selected_slots: list[str], target_mesh_materials_by_slot: dict[str, dict[int, dict[str, Any]]] | None = None):
    """De-duplicate only body roles genuinely shared by selected slot payloads."""
    target_mesh_materials_by_slot = target_mesh_materials_by_slot or {}
    kept = []
    report = []
    any_strong = False
    for entry in entries:
        role_material = _normalise_material(entry.get("material", ""))
        owner_slots = []
        for candidate in selected_slots:
            assignment = target_mesh_materials_by_slot.get(candidate, {}).get(int(entry.get("xiv_mesh_index", -1)))
            if assignment is not None and _normalise_material(assignment.get("material", "")) == role_material:
                owner_slots.append(candidate)

        if owner_slots == [slot]:
            kept.append(entry)
            report.append({
                "mesh": entry["mesh_name"],
                "material": entry.get("material"),
                "owner_slots": owner_slots,
                "included": True,
                "reason": "unique RBODY slot ownership",
            })
            continue

        evidence = _entry_region_evidence(glb, entry)
        ranked = sorted(selected_slots, key=lambda candidate: (
            int(evidence[candidate]["strong_vertices"]),
            float(evidence[candidate]["strong_fraction"]),
            float(evidence[candidate]["mean_weight"]),
        ), reverse=True)
        best = ranked[0] if ranked else slot
        best_is_strong = bool(ranked) and _strong_region(evidence[best])
        any_strong |= best_is_strong
        include = not best_is_strong or best == slot
        if include:
            kept.append(entry)
        report.append({
            "mesh": entry["mesh_name"],
            "material": entry.get("material"),
            "owner_slots": owner_slots,
            "best_slot": best,
            "best_strong": best_is_strong,
            "included": include,
            "reason": "shared RBODY role; skeleton-weight anatomy" if len(owner_slots) > 1 else "anatomical fallback",
            "evidence": evidence,
        })
    if kept:
        return kept, report
    if not any_strong:
        return entries, report
    raise ValueError(f"The selected {slot} target payload contains shared body-package geometry, but every candidate is strongly owned by another selected body region.")


def _skin_joint_names(glb: GLB, skin_index: int) -> list[str]:
    skin = glb.js["skins"][skin_index]
    return [glb.js["nodes"][j].get("name", "") for j in skin.get("joints", [])]


def _clone_body_mesh(dst: GLBEditor, target: GLB, target_entry: dict[str, Any], output_name: str, destination_skin: int, accessor_cache: dict[tuple[int, tuple[tuple[int, int], ...] | None], int]):
    src_mesh = target.js["meshes"][target_entry["mesh_index"]]
    target_joint_names = _skin_joint_names(target, target_entry["skin_index"])
    dest_skin = dst.js["skins"][destination_skin]
    dest_joint_names = [dst.js["nodes"][j].get("name", "") for j in dest_skin.get("joints", [])]
    dest_by_name = {name: i for i, name in enumerate(dest_joint_names) if name}
    missing = sorted({name for name in target_joint_names if name and name not in dest_by_name})
    if missing:
        raise ValueError(f"Target body requires skeleton joints missing from the retained source armature: {', '.join(missing[:20])}")
    joint_map = {i: dest_by_name[name] for i, name in enumerate(target_joint_names) if name in dest_by_name}
    material_index = _ensure_material(dst, target_entry["material"])

    mesh = copy.deepcopy(src_mesh)
    mesh["name"] = output_name
    mesh["primitives"] = []
    for pi in target_entry["primitive_indices"]:
        p = src_mesh.get("primitives", [])[pi]
        q = copy.deepcopy(p)
        attrs = {}
        for semantic, ai in p.get("attributes", {}).items():
            attrs[semantic] = _clone_accessor(dst, target, ai, accessor_cache, joint_map if semantic.startswith("JOINTS_") else None)
        q["attributes"] = attrs
        if "indices" in p:
            q["indices"] = _clone_accessor(dst, target, p["indices"], accessor_cache)
        if "targets" in p:
            q["targets"] = [
                {semantic: _clone_accessor(dst, target, ai, accessor_cache) for semantic, ai in morph.items()}
                for morph in p["targets"]
            ]
        q["material"] = material_index
        mesh["primitives"].append(q)
    return mesh


def _region_output_names(target_entries: list[dict[str, Any]], reserved_names: set[str]) -> list[str]:
    """Preserve XIV mesh grouping while avoiding retained garment-name collisions."""
    if not target_entries:
        return []
    reserved_bases = {parsed[0] for name in reserved_names if (parsed := _parse_mesh_name(name)) is not None}
    parsed_entries: list[tuple[int, int]] = []
    groups: dict[int, list[int]] = {}
    for i, entry in enumerate(target_entries):
        parsed = _parse_mesh_name(entry["mesh_name"])
        if parsed is None:
            raise ValueError(f"Target body mesh name is not an XIV mesh/submesh name: {entry['mesh_name']!r}")
        parsed_entries.append(parsed)
        groups.setdefault(parsed[0], []).append(i)

    base_map: dict[int, int] = {}
    for target_base in sorted(groups):
        output_base = target_base
        if output_base in reserved_bases:
            output_base = next((candidate for candidate in range(256) if candidate not in reserved_bases), -1)
            if output_base < 0:
                raise ValueError("Could not allocate a free XIV mesh group for transplanted target-body geometry.")
        base_map[target_base] = output_base
        reserved_bases.add(output_base)

    names = [f"mesh {base_map[base]}.{sub}" for base, sub in parsed_entries]
    if len(set(names)) != len(names):
        raise ValueError(f"Target body export contains duplicate logical mesh names after grouping: {names}")
    reserved_names.update(names)
    return names


def _same_accessor_payload(a: GLBEditor, ai: int, b: GLBEditor, bi: int) -> bool:
    left = _accessor_values_with_sparse(a, ai)
    right = _accessor_values_with_sparse(b, bi)
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)


def _audit_garment_integrity(original_source_glb: Path, output_glb: Path, source_body_meshes: set[str], reweighted_meshes: set[str]):
    """Check that garment topology, materials and immutable attributes survived."""
    src = GLBEditor(original_source_glb)
    dst = GLBEditor(output_glb)
    src_by_name: dict[str, list[dict[str, Any]]] = {}
    dst_by_name: dict[str, list[dict[str, Any]]] = {}
    for mesh in src.js.get("meshes", []):
        src_by_name.setdefault(str(mesh.get("name") or ""), []).append(mesh)
    for mesh in dst.js.get("meshes", []):
        dst_by_name.setdefault(str(mesh.get("name") or ""), []).append(mesh)

    audited = []
    frame_mutable = {"POSITION", "NORMAL", "TANGENT"}
    skin_mutable = {"JOINTS_0", "JOINTS_1", "WEIGHTS_0", "WEIGHTS_1"}
    for name, src_meshes in src_by_name.items():
        if not name or name in source_body_meshes:
            continue
        if len(src_meshes) != 1 or len(dst_by_name.get(name, [])) != 1:
            raise ValueError(f"Garment integrity failure: mesh {name!r} is not present exactly once in the final GLB.")
        src_mesh = src_meshes[0]
        dst_mesh = dst_by_name[name][0]
        src_primitives = src_mesh.get("primitives", [])
        dst_primitives = dst_mesh.get("primitives", [])
        if len(src_primitives) != len(dst_primitives):
            raise ValueError(f"Garment integrity failure: {name} primitive count changed.")
        for pi, (sp, dp) in enumerate(zip(src_primitives, dst_primitives)):
            sm = sp.get("material")
            dm = dp.get("material")
            sm_name = "" if sm is None else _normalise_material(src.js.get("materials", [])[sm].get("name", ""))
            dm_name = "" if dm is None else _normalise_material(dst.js.get("materials", [])[dm].get("name", ""))
            if sm_name != dm_name:
                raise ValueError(f"Garment integrity failure: {name} primitive {pi} material changed from {sm_name!r} to {dm_name!r}.")
            if ("indices" in sp) != ("indices" in dp) or ("indices" in sp and not _same_accessor_payload(src, sp["indices"], dst, dp["indices"])):
                raise ValueError(f"Garment integrity failure: {name} primitive {pi} topology/index data changed.")
            sa = sp.get("attributes", {})
            da = dp.get("attributes", {})
            if set(sa) != set(da):
                raise ValueError(f"Garment integrity failure: {name} primitive {pi} vertex attribute layout changed.")
            for semantic in sa:
                mutable = semantic in frame_mutable or (name in reweighted_meshes and semantic in skin_mutable)
                if mutable:
                    if src.js["accessors"][sa[semantic]].get("count") != dst.js["accessors"][da[semantic]].get("count"):
                        raise ValueError(f"Garment integrity failure: {name} primitive {pi} {semantic} vertex count changed.")
                    continue
                if not _same_accessor_payload(src, sa[semantic], dst, da[semantic]):
                    raise ValueError(f"Garment integrity failure: {name} primitive {pi} immutable {semantic} data changed.")
            st = sp.get("targets", [])
            dt = dp.get("targets", [])
            if len(st) != len(dt):
                raise ValueError(f"Garment integrity failure: {name} primitive {pi} morph-target count changed.")
            for ti, (source_target, output_target) in enumerate(zip(st, dt)):
                if set(source_target) != set(output_target):
                    raise ValueError(f"Garment integrity failure: {name} primitive {pi} morph target {ti} layout changed.")
                for semantic in source_target:
                    if not _same_accessor_payload(src, source_target[semantic], dst, output_target[semantic]):
                        raise ValueError(f"Garment integrity failure: {name} primitive {pi} morph target {ti} {semantic} changed.")
        audited.append(name)
    return {"mesh_count": len(audited), "meshes": audited, "immutable_data_preserved": True, "reweighted_meshes": sorted(reweighted_meshes)}


def _transplant_target_bodies(
    fitted_glb: Path,
    original_source_glb: Path,
    target_glbs_by_slot: dict[str, Path],
    output_glb: Path,
    source_materials: set[str],
    source_materials_by_slot: dict[str, set[str]],
    target_materials_by_slot: dict[str, set[str]],
    target_mesh_materials_by_slot: dict[str, dict[int, dict[str, Any]]],
    model_slot: str,
    reweighted_meshes: set[str],
    body_suppression: dict[str, Any] | None = None,
):
    source_before = GLB(original_source_glb)
    source_entries = _source_body_info(source_before, source_materials)
    source_by_slot, present_slots, source_region_report = _assign_source_body_regions(source_before, source_entries, source_materials_by_slot, model_slot)
    missing_targets = [slot for slot in present_slots if slot not in target_glbs_by_slot]
    if missing_targets:
        raise ValueError(f"RavaFit would remove embedded source body region(s) {missing_targets} without a corresponding selected target payload. Conversion aborted.")

    dst = GLBEditor(fitted_glb)
    source_node_indices = [entry["node_index"] for entry in source_entries]
    active_nodes: list[int] = []
    inserted: list[dict[str, Any]] = []
    target_region_report: dict[str, Any] = {}
    source_body_names = {entry["mesh_name"] for entry in source_entries}
    reserved_names = {
        str(mesh.get("name") or "")
        for mesh in source_before.js.get("meshes", [])
        if mesh.get("name") and str(mesh.get("name")) not in source_body_names
    }
    transplanted_slots: list[str] = []

    for slot in ("Chest", "Legs", "Hands", "Feet"):
        region_sources = source_by_slot.get(slot, [])
        if not region_sources:
            continue
        target_materials = target_materials_by_slot.get(slot, set())
        target_mesh_materials = target_mesh_materials_by_slot.get(slot, {})
        if not target_mesh_materials:
            raise ValueError(f"Embedded source {slot} geometry was detected, but the selected {slot} RBODY target exposes no mesh/material assignments.")
        target_path = target_glbs_by_slot[slot]
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        target = GLB(target_path)
        target_entries = _target_body_nodes(target, target_mesh_materials)
        target_entries, region_detail = _filter_target_entries_for_slot(target, target_entries, slot, present_slots, target_mesh_materials_by_slot)
        if not target_entries:
            raise ValueError(f"The selected {slot} target contains no transplantable body geometry.")

        destination_skin = region_sources[0]["skin_index"]
        output_names = _region_output_names(target_entries, reserved_names)
        accessor_cache: dict[tuple[int, tuple[tuple[int, int], ...] | None], int] = {}

        for i, (entry, output_name) in enumerate(zip(target_entries, output_names)):
            material = entry["material"]
            new_mesh = _clone_body_mesh(dst, target, entry, output_name, destination_skin, accessor_cache)
            if i < len(region_sources):
                source_entry = region_sources[i]
                mesh_index = source_entry["mesh_index"]
                dst.js["meshes"][mesh_index] = new_mesh
                node_index = source_entry["node_index"]
                node = dst.js["nodes"][node_index]
                node["mesh"] = mesh_index
                node["skin"] = destination_skin
                node["name"] = output_name
                node["extras"] = copy.deepcopy(target.js["nodes"][entry["node_index"]].get("extras", {}))
            else:
                mesh_index = len(dst.js["meshes"])
                dst.js["meshes"].append(new_mesh)
                node_index = len(dst.js["nodes"])
                dst.js["nodes"].append({
                    "name": output_name,
                    "mesh": mesh_index,
                    "skin": destination_skin,
                    "extras": copy.deepcopy(target.js["nodes"][entry["node_index"]].get("extras", {})),
                })
            active_nodes.append(node_index)
            inserted.append({
                "slot": slot,
                "name": output_name,
                "node": node_index,
                "mesh": mesh_index,
                "material": material,
                "target_mesh": entry["mesh_name"],
                "target_xiv_mesh_index": entry["xiv_mesh_index"],
                "rbody_material_table_index": entry["material_table_index"],
                "rbody_record_ordinal": entry["rbody_record_ordinal"],
            })

        # Leave unused source-body pieces detached so they cannot leak into the result.
        for source_entry in region_sources[len(target_entries):]:
            node = dst.js["nodes"][source_entry["node_index"]]
            node.pop("mesh", None)
            node.pop("skin", None)

        target_region_report[slot] = {
            "target_glb": str(target_path),
            "source_meshes": [entry["mesh_name"] for entry in region_sources],
            "target_meshes": [entry["mesh_name"] for entry in target_entries],
            "selection": region_detail,
            "retained_skin": destination_skin,
            "target_package_materials": sorted(target_materials),
            "target_mesh_material_assignments": [target_mesh_materials[index] for index in sorted(target_mesh_materials)],
            "output_materials": [entry["material"] for entry in target_entries],
        }
        transplanted_slots.append(slot)

    if set(transplanted_slots) != set(present_slots):
        raise ValueError(f"Body transplant parity failure: source regions={present_slots}, transplanted regions={transplanted_slots}.")

    # Rebuild scene membership from the transplanted body nodes only.
    for node_index in source_node_indices:
        if node_index not in active_nodes:
            dst.js["nodes"][node_index].pop("mesh", None)
            dst.js["nodes"][node_index].pop("skin", None)
    for scene in dst.js.get("scenes", []):
        current = [ni for ni in scene.get("nodes", []) if ni not in source_node_indices and ni not in active_nodes]
        scene["nodes"] = current + active_nodes

    dst.js["buffers"][0]["byteLength"] = len(dst.bin)
    bounds = _refresh_float_accessor_bounds(dst)
    _validate_float_accessor_bounds(dst)
    xiv_skinning = _validate_xiv_weighted_bone_budget(dst)
    dst.save(output_glb)
    serialized_bounds = _validate_float_accessor_bounds(GLBEditor(output_glb))
    garment_audit = _audit_garment_integrity(original_source_glb, output_glb, {entry["mesh_name"] for entry in source_entries}, reweighted_meshes)
    return {
        "source_body_meshes": [entry["mesh_name"] for entry in source_entries],
        "source_regions": present_slots,
        "transplanted_regions": transplanted_slots,
        "source_region_assignment": source_region_report,
        "target_regions": target_region_report,
        "inserted": inserted,
        "body_suppression": body_suppression or {"enabled":False},
        "garment_integrity": garment_audit,
        "xiv_skinning": xiv_skinning,
        "accessor_bounds": {
            "checked": bounds["checked"],
            "repaired_count": bounds["repaired_count"],
            "repaired": bounds["repaired"],
            "serialized_checked": serialized_bounds["checked"],
            "serialized_sparse_checked": serialized_bounds.get("sparse_checked", 0),
        },
    }

def _strict_json_safe(value: Any) -> Any:
    """Normalise diagnostic values for strict JSON."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return _strict_json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: _strict_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_safe(item) for item in value]
    return value


def _support_equivalent_source_preserve_decision(cache: dict[str,Any], *, max_rms_mm: float=3.25, max_spatial_p95_mm: float=6.50) -> dict[str,Any]:
    """Decide whether every selected source->target support region is close enough to preserve garment geometry exactly.

    This is deliberately slot-agnostic.  If *all* selected body regions move only a few millimetres and
    their correspondence remains locally coherent, solving can do more harm than good to authored
    structures such as heels, rigid footwear, cuffs and accessories.  In that case the garment itself
    remains untouched while normal body detach/transplant still proceeds.
    """
    rows=list(cache.get("slot_stats") or [])
    if not rows:
        return {"eligible":False,"reason":"no slot support statistics","slots":[]}
    if bool(cache.get("dense_cross_sex_source_proxy",False)):
        return {"eligible":False,"reason":"cross-sex dense support requires a real refit","slots":rows}
    reports=[]
    eligible=True
    for row in rows:
        rms=float(row.get("displacement_rms_mm",float("inf")))
        raw_rms=float(row.get("raw_displacement_rms_mm",rms))
        spatial_p95=float(row.get("chosen_spatial_p95_mm",float("inf")))
        finite=bool(np.isfinite(rms) and np.isfinite(raw_rms) and np.isfinite(spatial_p95))
        slot_ok=finite and rms<=float(max_rms_mm) and raw_rms<=float(max_rms_mm) and spatial_p95<=float(max_spatial_p95_mm)
        reports.append({"slot":str(row.get("slot") or "Body"),"eligible":bool(slot_ok),"displacement_rms_mm":rms,"raw_displacement_rms_mm":raw_rms,"chosen_spatial_p95_mm":spatial_p95})
        eligible=eligible and slot_ok
    return {"eligible":bool(eligible),"reason":"all selected body supports are materially equivalent" if eligible else "at least one selected body support changes materially","max_rms_mm":float(max_rms_mm),"max_spatial_p95_mm":float(max_spatial_p95_mm),"slots":reports}


def _source_preserved_garment_solution(source: Any, body_mesh_names: set[str], mesh_filter: set[str] | None=None):
    positions={};skinning={};records={}
    for name in source.mesh_names():
        if name in body_mesh_names or (mesh_filter is not None and name not in mesh_filter):
            continue
        data=source.data(name)
        V=np.asarray(data.get("V",[]),dtype=np.float64);F=np.asarray(data.get("F",[]),dtype=np.int64);W=np.asarray(data.get("W",[]),dtype=np.float64)
        if len(V)==0 or len(F)==0:
            continue
        if W.ndim!=2 or W.shape[0]!=len(V):
            raise ValueError(f"{name}: source-preserve lane requires valid authored skinning weights.")
        positions[name]=V.copy()
        stage={"mode":"source_preserved_support_equivalent","policy":"source and target support regions are materially equivalent; preserve authored garment geometry and skinning exactly"}
        skinning[name]={"weights":W.copy(),"joint_names":list(data.get("joint_names") or []),"stage":stage}
        records[name]={"mode":"source_preserved_support_equivalent","behavior":"source_preserved","stage":stage,"skinning":stage}
    if not positions:
        raise ValueError("No garment meshes were eligible for source-preserve conversion after removing the source body.")
    stats={"support_equivalent_source_preserve":{"enabled":True,"mesh_count":len(positions),"meshes":sorted(positions),"policy":"skip geometric refit when every selected source->target body support is materially equivalent; body detach/transplant remains authoritative"},"garment_mesh_count":len(positions),"skinning_retargeted_mesh_count":len(skinning),"post_b14_geometry_mutation":False}
    return positions,skinning,records,stats


def _validate_final_fit_clearance(solve_stats):
    report=solve_stats.get("final_target_body_clearance") or {}
    unresolved=int(report.get("unresolved_penetrating_face_count",0))
    if unresolved:
        depth=report.get("minimum_contact_sample_after_mm")
        detail=f" (deepest sampled contact {float(depth):.3f} mm)" if depth is not None else ""
        raise ValueError(f"Garment fit still intersects the selected target body on {unresolved} triangles{detail}. No fitted output was published.")


def convert(spec: dict[str, Any]) -> dict[str, Any]:
    source_glb = Path(spec["source_glb"]).resolve()
    output_glb = Path(spec["output_glb"]).resolve()
    model_slot = spec.get("model_slot") or _model_slot(spec.get("game_path", ""))
    if not source_glb.exists():
        raise FileNotFoundError(source_glb)

    raw_targets = spec.get("target_body_glbs") or {}
    # One-version fallback for old hand-written diagnostic specs.
    if not raw_targets and spec.get("target_body_glb"):
        raw_targets = {model_slot: spec["target_body_glb"]}
    target_glbs_by_slot = {str(slot): Path(path).resolve() for slot, path in raw_targets.items()}
    if not target_glbs_by_slot:
        raise ValueError("Conversion spec contains no target body payloads.")
    for target_path in target_glbs_by_slot.values():
        if not target_path.exists():
            raise FileNotFoundError(target_path)
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    _reset_b14_runtime_caches()
    _set_surface_query_cache_enabled(True)
    source = GLB(source_glb)
    names_for_rig=_rig_names_from_loaded_glb(source,spec.get("rig_mesh"))
    race_retarget_context=_race_retarget_context(spec,names_for_rig)
    race_retarget_report=_retarget_loaded_source_glb(source,race_retarget_context) if race_retarget_context is not None else {"enabled":False}
    names, cache, source_materials, source_materials_by_slot, target_materials_by_slot, target_mesh_materials_by_slot, payload_details = _load_pairs(spec, names_for_rig, source)
    cache=dict(cache)
    _attach_target_auxiliary_obstacles(cache,target_glbs_by_slot,payload_details)
    if model_slot not in target_materials_by_slot:
        raise ValueError(f"The selected model is a {model_slot} model, but no {model_slot} body target was selected.")

    source_body_mode=str(spec.get("source_body_mode") or "rbody").casefold()
    default_source_contains_body=bool(source_materials) if source_body_mode=="vanilla" else True
    source_contains_body=bool(spec.get("source_contains_body",default_source_contains_body))
    fit_only,transplant_target_body,accessory_container_slot=_resolve_output_policy(spec,source_contains_body)
    body_mesh_names = set(_find_body_meshes(source, source_materials)) if source_contains_body else set()
    mesh_filter=set(spec["mesh_filter"]) if spec.get("mesh_filter") else None
    source_entries=_source_body_info(source,source_materials) if source_contains_body else []
    if source_entries:
        source_by_slot,present_source_slots,_=_assign_source_body_regions(source,source_entries,source_materials_by_slot,model_slot)
    else:
        source_by_slot,present_source_slots={},[]
    dense_vanilla=source_body_mode=="vanilla" and bool(cache.get("dense_vanilla_source_proxy",False))
    suppression_plan=_complete_vanilla_target_body_plan(cache,present_source_slots) if dense_vanilla else _source_body_suppression_plan(source,cache,source_by_slot,present_source_slots,body_mesh_names,mesh_filter)
    cache["_ravafit_source_body_suppression"]=suppression_plan
    cache.pop("_ravafit_target_collision_triangles",None)
    strict_b14_contract=bool(cache.get("_ravafit_strict_b14_contract",False))
    if dense_vanilla:
        solve_source=_DenseVanillaGarmentSource(source,body_mesh_names)
    elif strict_b14_contract:
        solve_source=_IndexedRenderGarmentSource(source,body_mesh_names)
    else:
        solve_source=source
    dense_garment_report=solve_source.report if dense_vanilla else {}
    indexed_garment_report=solve_source.report if strict_b14_contract else {}
    support_equivalent_preserve=_support_equivalent_source_preserve_decision(cache)
    reference_retry = None
    try:
        if support_equivalent_preserve.get("eligible",False):
            positions,skinning,mesh_records,solve_stats=_source_preserved_garment_solution(source,body_mesh_names,mesh_filter)
            solve_stats["support_equivalent_source_preserve"]["decision"]=support_equivalent_preserve
        elif strict_b14_contract:
            positions,skinning,mesh_records,solve_stats=_solve_strict_b14_layers(solve_source,cache,body_mesh_names,mesh_filter)
        elif not dense_vanilla and mesh_filter is None:
            try:
                positions,skinning,mesh_records,solve_stats=_solve_modded_garment_meshes_isolated(solve_source,cache,body_mesh_names,mesh_filter)
            except Exception as worker_error:
                print(f"[RavaFit perf] isolated garment workers failed; using exact in-process solve: {worker_error}",file=sys.stderr,flush=True)
                positions,skinning,mesh_records,solve_stats=_solve_garment_meshes(solve_source,cache,body_mesh_names,mesh_filter)
                solve_stats["garment_mesh_workers"]={"enabled":False,"fallback":"exact-in-process","error":str(worker_error)}
        else:
            positions, skinning, mesh_records, solve_stats = _solve_garment_meshes(solve_source, cache, body_mesh_names, mesh_filter)
        if strict_b14_contract:
            positions,skinning=_collapse_indexed_render_garment_solution(solve_source,positions,skinning)
            solve_stats["indexed_render_garment_proxy"]={"enabled":True,"meshes":indexed_garment_report,"policy":"solve only triangle-referenced rendered vertices; restore exact authored raw storage afterward"}
        elif dense_vanilla:
            positions,skinning=_collapse_dense_vanilla_garment_solution(solve_source,positions,skinning)
            original_contexts={name:{"data":source.data(name)} for name in positions if name in source.mesh_names()}
            positions,post_collapse_meshes,post_collapse_clearance=_dense_vanilla_expansion_clearance_guard(positions,original_contexts,cache)
            solve_stats["dense_vanilla_garment_proxy"]={"enabled":True,"meshes":dense_garment_report,"policy":"temporary subdivided solve proxy collapsed to exact authored output topology"}
            solve_stats["dense_vanilla_post_collapse_clearance"]=post_collapse_clearance
    except NonFiniteGeometryError as first_error:
        _reset_b14_runtime_caches()
        _set_surface_query_cache_enabled(False)
        _PAIR_CACHE.clear()
        close_cached_rbodies()
        source = GLB(source_glb)
        names_for_rig=_rig_names_from_loaded_glb(source,spec.get("rig_mesh"))
        race_retarget_context=_race_retarget_context(spec,names_for_rig)
        race_retarget_report=_retarget_loaded_source_glb(source,race_retarget_context) if race_retarget_context is not None else {"enabled":False}
        names, cache, source_materials, source_materials_by_slot, target_materials_by_slot, target_mesh_materials_by_slot, payload_details = _load_pairs(spec, names_for_rig, source)
        cache=dict(cache)
        _attach_target_auxiliary_obstacles(cache,target_glbs_by_slot,payload_details)
        body_mesh_names = set(_find_body_meshes(source, source_materials)) if source_contains_body else set()
        source_entries=_source_body_info(source,source_materials) if source_contains_body else []
        if source_entries:
            source_by_slot,present_source_slots,_=_assign_source_body_regions(source,source_entries,source_materials_by_slot,model_slot)
        else:
            source_by_slot,present_source_slots={},[]
        dense_vanilla=source_body_mode=="vanilla" and bool(cache.get("dense_vanilla_source_proxy",False))
        suppression_plan=_complete_vanilla_target_body_plan(cache,present_source_slots) if dense_vanilla else _source_body_suppression_plan(source,cache,source_by_slot,present_source_slots,body_mesh_names,mesh_filter)
        cache["_ravafit_source_body_suppression"]=suppression_plan
        cache.pop("_ravafit_target_collision_triangles",None)
        strict_b14_contract=bool(cache.get("_ravafit_strict_b14_contract",False))
        if dense_vanilla:
            solve_source=_DenseVanillaGarmentSource(source,body_mesh_names)
        elif strict_b14_contract:
            solve_source=_IndexedRenderGarmentSource(source,body_mesh_names)
        else:
            solve_source=source
        dense_garment_report=solve_source.report if dense_vanilla else {}
        indexed_garment_report=solve_source.report if strict_b14_contract else {}
        support_equivalent_preserve=_support_equivalent_source_preserve_decision(cache)
        try:
            if support_equivalent_preserve.get("eligible",False):
                positions,skinning,mesh_records,solve_stats=_source_preserved_garment_solution(source,body_mesh_names,mesh_filter)
                solve_stats["support_equivalent_source_preserve"]["decision"]=support_equivalent_preserve
            elif strict_b14_contract:
                positions, skinning, mesh_records, solve_stats = _solve_strict_b14_layers(solve_source, cache, body_mesh_names, mesh_filter)
            else:
                positions, skinning, mesh_records, solve_stats = _solve_garment_meshes(solve_source, cache, body_mesh_names, mesh_filter)
            if strict_b14_contract:
                positions,skinning=_collapse_indexed_render_garment_solution(solve_source,positions,skinning)
                solve_stats["indexed_render_garment_proxy"]={"enabled":True,"meshes":indexed_garment_report,"policy":"solve only triangle-referenced rendered vertices; restore exact authored raw storage afterward"}
            elif dense_vanilla:
                positions,skinning=_collapse_dense_vanilla_garment_solution(solve_source,positions,skinning)
                original_contexts={name:{"data":source.data(name)} for name in positions if name in source.mesh_names()}
                positions,post_collapse_meshes,post_collapse_clearance=_dense_vanilla_expansion_clearance_guard(positions,original_contexts,cache)
                solve_stats["dense_vanilla_garment_proxy"]={"enabled":True,"meshes":dense_garment_report,"policy":"temporary subdivided solve proxy collapsed to exact authored output topology"}
                solve_stats["dense_vanilla_post_collapse_clearance"]=post_collapse_clearance
        except NonFiniteGeometryError as reference_error:
            raise NonFiniteGeometryError(
                f"{reference_error} (optimised-path first failure: {first_error})"
            ) from reference_error
        reference_retry = {"used": True, "trigger": str(first_error), "surface_query": "reference"}
    finally:
        _set_surface_query_cache_enabled(True)
    if reference_retry is not None:
        solve_stats["reference_retry"] = reference_retry

    _validate_final_fit_clearance(solve_stats)
    suppression_plan=_refine_source_body_suppression_after_fit(source,positions,cache,suppression_plan)
    cache["_ravafit_source_body_suppression"]=suppression_plan
    fitted_without_body = output_glb.with_name(output_glb.stem + ".garment-only.glb")
    detach = _patch_positions_and_detach_body(source_glb, fitted_without_body, positions, skinning, body_mesh_names)
    if transplant_target_body:
        transplant = _transplant_target_bodies(
            fitted_without_body,
            source_glb,
            target_glbs_by_slot,
            output_glb,
            source_materials,
            source_materials_by_slot,
            target_materials_by_slot,
            target_mesh_materials_by_slot,
            model_slot,
            set(skinning),
            _public_source_body_suppression(suppression_plan),
        )
    else:
        shutil.copyfile(fitted_without_body,output_glb)
        transplant={"enabled":False,"reason":"fit-only source; preserve destination body outside this solver transaction"}

    result = {
        "ok": True,
        "source_glb": str(source_glb),
        "target_body_glbs": {slot: str(path) for slot, path in target_glbs_by_slot.items()},
        "output_glb": str(output_glb),
        "model_slot": model_slot,
        "accessory_container_slot": accessory_container_slot,
        "fit_only": fit_only,
        "joint_count": len(names),
        "race_skeleton_retarget": race_retarget_report,
        "source_body_materials": sorted(source_materials),
        "source_body_materials_by_slot": {slot: sorted(values) for slot, values in source_materials_by_slot.items()},
        "target_body_materials_by_slot": {slot: sorted(values) for slot, values in target_materials_by_slot.items()},
        "target_mesh_materials_by_slot": {
            slot: [values[index] for index in sorted(values)]
            for slot, values in target_mesh_materials_by_slot.items()
        },
        "source_body_meshes": sorted(body_mesh_names),
        "source_contains_body": source_contains_body,
        "transplant_target_body": transplant_target_body,
        "payloads": payload_details,
        "solve": solve_stats,
        "meshes": mesh_records,
        "detach": detach,
        "transplant": transplant,
    }
    result = _strict_json_safe(result)
    report_path = output_glb.with_suffix(".ravafit.json")
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    result["report"] = str(report_path)
    _reset_b14_runtime_caches()
    return result
