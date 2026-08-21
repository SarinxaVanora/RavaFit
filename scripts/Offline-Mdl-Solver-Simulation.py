#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'/'rbody'));sys.path.insert(0,str(ROOT/'runtime'/'solver'))
from rbody_v3_core import parse_rigged_mdl, dense_weights
from rbody_v3_loader import RBodyV3
from rbody_b14_adapter import collect_body_pairs, build_dense_source_proxy
import production_b14 as prod

# Offline timing visibility only; production behavior is untouched.
_orig_finite_stage_call = prod._finite_stage_call
def _timed_finite_stage_call(mesh_name, stage_name, fn):
    t0=time.time(); print(json.dumps({"mesh":mesh_name,"stage_begin":stage_name}),flush=True)
    result=_orig_finite_stage_call(mesh_name,stage_name,fn)
    print(json.dumps({"mesh":mesh_name,"stage_end":stage_name,"elapsed_sec":round(time.time()-t0,3)}),flush=True)
    return result
prod._finite_stage_call=_timed_finite_stage_call

class MdlSolverView:
    def __init__(self,path:Path,cache,rig_joint_names=None,supplemental_skeleton=None):
        self.path=Path(path);self.parsed=parse_rigged_mdl(self.path.read_bytes());self._data={};self._names=[]
        rig=list(rig_joint_names or self.parsed['joint_names']); own_names=list(self.parsed['joint_names'])
        V=np.asarray(self.parsed['positions'],dtype=np.float64);N=np.asarray(self.parsed['normals'],dtype=np.float64);UV=np.asarray(self.parsed['uv0'],dtype=np.float64);I=np.asarray(self.parsed['indices'],dtype=np.int64)
        ownW=dense_weights(self.parsed); W=np.zeros((len(V),len(rig)),dtype=np.float64); rig_index={n:i for i,n in enumerate(rig)}
        for j,n in enumerate(own_names):
            if n in rig_index: W[:,rig_index[n]]=ownW[:,j]
        for ordinal,rec in enumerate(self.parsed['mesh_records']):
            vo=int(rec['vertex_offset']);vc=int(rec['vertex_count']);io=int(rec['index_offset']);ic=int(rec['index_count']);name=f'mesh {ordinal}';mat=self.parsed['materials'][ordinal] if ordinal<len(self.parsed['materials']) else ''
            self._names.append(name);f=(I[io:io+ic].reshape(-1,3)-vo).astype(np.int64)
            self._data[name]={'name':name,'V':V[vo:vo+vc].copy(),'F':f,'UV':UV[vo:vo+vc].copy() if len(UV)==len(V) else None,'N':N[vo:vo+vc].copy() if len(N)==len(V) else None,'W':W[vo:vo+vc].copy(),'joint_names':list(rig),'material':mat}
        X=np.asarray(cache['X'],dtype=np.float64);BW=np.asarray(cache['BW'],dtype=np.float64);nodes=[]
        supplemental_positions={}
        if supplemental_skeleton is not None:
            sr=supplemental_skeleton; sV=np.asarray(sr['positions'],dtype=np.float64); sW=dense_weights(sr)
            for j,n in enumerate(sr['joint_names']):
                ww=sW[:,j]; mass=float(ww.sum())
                if mass>1e-7: supplemental_positions[n]=np.sum(sV*ww[:,None],axis=0)/mass
        for i,name in enumerate(cache['names']):
            w=BW[:,i] if i<BW.shape[1] else np.zeros(len(X));mass=float(w.sum())
            if mass>1e-7: pos=np.sum(X*w[:,None],axis=0)/mass
            elif name in supplemental_positions: pos=supplemental_positions[name]
            else: pos=np.array([0.,0.,0.])
            nodes.append({'name':name,'translation':pos.tolist()})
        self.js={'nodes':nodes,'meshes':[{'name':n} for n in self._names]}
    def mesh_names(self):return list(self._names)
    def data(self,name):return self._data[name]

def body_mesh_names(view,source_materials):
    mats={prod._normalise_material(x) for x in source_materials};out=set()
    for name in view.mesh_names():
        if prod._material_matches_body_package(view.data(name)['material'],mats):out.add(name)
    return out

def source_materials(lib,body,slot,variant):
    pid=lib.resolve_payload_id(body,slot,variant,'0201');return {prod._normalise_material(x['material']) for x in lib.mesh_material_assignments(pid).values()}


def _union_joint_names(*name_lists):
    out=[]
    for seq in name_lists:
        for name in seq:
            if name not in out: out.append(name)
    return out

def build_source_ref_from_body_mdl(source_body_mdl,rig_joint_names,slot,payload_id='xiv-vanilla-e0000'):
    parsed=parse_rigged_mdl(Path(source_body_mdl).read_bytes()); own_names=list(parsed['joint_names'])
    V=np.asarray(parsed['positions'],dtype=np.float64);N=np.asarray(parsed['normals'],dtype=np.float64);UV=np.asarray(parsed['uv0'],dtype=np.float64);I=np.asarray(parsed['indices'],dtype=np.int64)
    ownW=dense_weights(parsed); rig=list(rig_joint_names or parsed['joint_names']); W=np.zeros((len(V),len(rig)),dtype=np.float64); rig_index={n:i for i,n in enumerate(rig)}
    for j,n in enumerate(own_names):
        if n in rig_index: W[:,rig_index[n]]=ownW[:,j]
    rec=parsed['mesh_records'][0]; vo=int(rec['vertex_offset']); vc=int(rec['vertex_count']); io=int(rec['index_offset']); ic=int(rec['index_count'])
    F=(I[io:io+ic].reshape(-1,3)-vo).astype(np.int64)
    return {'V':V[vo:vo+vc].copy(),'N':N[vo:vo+vc].copy(),'UV':UV[vo:vo+vc].copy(),'F':F,'W':W[vo:vo+vc].copy(),'joint_names':list(rig),'slot':slot,'surface_mode':'body','race_code':'0201','payload_id':f'{payload_id}:{slot}'}


def run_case_from_source_refs(label,source_mdl,source_refs,rbody,target_slots,out,mesh_filter=None):
    t=time.time(); print(json.dumps({"case":label,"stage":"load"}),flush=True)
    with RBodyV3(rbody) as lib:
        # target_slots: list of (target_body, slot, target_variant)
        rig=list(source_refs[0]["joint_names"])
        pairs=[]; primary_target=None
        for target_body, slot, target_variant in target_slots:
            tgt_entry=lib.entry(target_body,slot,target_variant); surface_mode=str(tgt_entry.get("support_surface") or "body")
            tgt=lib.reference(target_body,slot,target_variant,race_code='0201',rig_joint_names=rig,surface_mode=surface_mode)
            if primary_target is None: primary_target=tgt
            literal_src=next(ref for ref in source_refs if str(ref.get("slot"))==str(slot))
            src=build_dense_source_proxy(literal_src,tgt)
            print(json.dumps({"case":label,"stage":"body-correspondence","slot":slot,"literal_source_vertices":len(literal_src['V']),"dense_source_vertices":len(src['V']),"target_vertices":len(tgt['V']),"target_support_surface":surface_mode}),flush=True)
            pairs.append((src,tgt))
        cache=collect_body_pairs(pairs)
        view=MdlSolverView(Path(source_mdl),cache,rig_joint_names=rig,supplemental_skeleton=None); body_names={'mesh 0'} if any(str(ref.get('slot'))=='Chest' for ref in source_refs) else set()
        solve_view=prod._DenseVanillaGarmentSource(view,body_names)
        print(json.dumps({"case":label,"stage":"solve","body_meshes":sorted(body_names),'garment_meshes':[n for n in view.mesh_names() if n not in body_names],"dense_garment_proxy":solve_view.report}),flush=True)
        prod._reset_b14_runtime_caches(); prod._set_surface_query_cache_enabled(True)
        positions,skinning,records,stats=prod._solve_garment_meshes(solve_view,cache,body_names,set(mesh_filter) if mesh_filter else None)
        positions,skinning=prod._collapse_dense_vanilla_garment_solution(solve_view,positions,skinning)
        original_contexts={name:{"data":view.data(name)} for name in positions if name in view.mesh_names()}
        positions,_,post_clearance=prod._dense_vanilla_expansion_clearance_guard(positions,original_contexts,cache)
        stats['dense_vanilla_garment_proxy']={"enabled":True,"meshes":solve_view.report,"policy":"temporary subdivided solve proxy collapsed to exact authored output topology"}
        stats['dense_vanilla_post_collapse_clearance']=post_clearance
        stats['source_body_suppression']={"enabled":False,"policy":"vanilla full target-body authority"}
        report={"case":label,"source_body":"xiv-vanilla-e0000+dense-proxy","source_variant":"canonical","target_slots":[{"body":b,"slot":s,"variant":v} for b,s,v in target_slots],"body_meshes":sorted(body_names),"solver_stats":stats,"mesh_records":records,"slot_stats":cache.get('slot_stats',[]),"elapsed_sec":time.time()-t}
        serialise_npz(Path(out),view,positions,body_names,primary_target,report)
        print(json.dumps({"case":label,"stage":"done","elapsed_sec":report['elapsed_sec'],"output":str(out),"seams":stats.get('authored_split_seams',{})},default=str),flush=True)

def run_case_from_source_ref(label,source_mdl,source_ref,rbody,target_body,slot,target_variant,out,mesh_filter=None):
    t=time.time(); print(json.dumps({'case':label,'stage':'load'}),flush=True)
    with RBodyV3(rbody) as lib:
        rig=list(source_ref['joint_names'])
        tgt_entry=lib.entry(target_body,slot,target_variant);surface_mode=str(tgt_entry.get('support_surface') or 'body')
        tgt=lib.reference(target_body,slot,target_variant,race_code='0201',rig_joint_names=rig,surface_mode=surface_mode)
        print(json.dumps({'case':label,'stage':'body-correspondence','source_vertices':len(source_ref['V']),'target_vertices':len(tgt['V']),'target_support_surface':surface_mode}),flush=True)
        cache=collect_body_pairs([(source_ref,tgt)])
        view=MdlSolverView(Path(source_mdl),cache,rig_joint_names=rig,supplemental_skeleton=None);body_names={'mesh 0'} if slot=='Chest' else body_mesh_names(view,source_materials(lib,target_body,slot,target_variant))
        print(json.dumps({'case':label,'stage':'solve','body_meshes':sorted(body_names),'garment_meshes':[n for n in view.mesh_names() if n not in body_names]}),flush=True)
        prod._reset_b14_runtime_caches(); prod._set_surface_query_cache_enabled(True)
        positions,skinning,records,stats=prod._solve_garment_meshes(view,cache,body_names,set(mesh_filter) if mesh_filter else None)
        report={'case':label,'source_body':'xiv-vanilla-e0000','source_variant':'canonical','target_body':target_body,'target_variant':target_variant,'target_support_surface':surface_mode,'body_meshes':sorted(body_names),'solver_stats':stats,'mesh_records':records,'elapsed_sec':time.time()-t}
        serialise_npz(Path(out),view,positions,body_names,tgt,report)
        print(json.dumps({'case':label,'stage':'done','elapsed_sec':report['elapsed_sec'],'output':str(out),'seams':stats.get('authored_split_seams',{})},default=str),flush=True)

def serialise_npz(path,view,positions,body_names,target_ref,report):
    arrays={};meta={'meshes':[],'body_mesh_names':sorted(body_names),'report':report}
    for name in view.mesh_names():
        d=view.data(name);key=name.replace(' ','_');arrays[key+'_source_V']=d['V'];arrays[key+'_F']=d['F'];arrays[key+'_new_V']=positions.get(name,d['V']);meta['meshes'].append({'name':name,'material':d['material'],'garment':name not in body_names,'vertices':len(d['V']),'triangles':len(d['F'])})
    arrays['target_body_V']=np.asarray(target_ref['V'],dtype=np.float64);arrays['target_body_F']=np.asarray(target_ref['F'],dtype=np.int64)
    np.savez_compressed(path,**arrays);path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

def run_case(label,source_mdl,rbody,source_body,slot,source_variant,target_body,target_variant,out,mesh_filter=None):
    t=time.time();print(json.dumps({'case':label,'stage':'load'}),flush=True)
    with RBodyV3(rbody) as lib:
        parsed=parse_rigged_mdl(Path(source_mdl).read_bytes());rig=list(parsed['joint_names']); supplemental=None
        if slot=='Legs':
            top_path=Path(source_mdl).with_name('l top nb m.mdl')
            if top_path.exists():
                supplemental=parse_rigged_mdl(top_path.read_bytes())
                for joint in supplemental['joint_names']:
                    if joint not in rig: rig.append(joint)
        src=lib.reference(source_body,slot,source_variant,race_code='0201',rig_joint_names=rig,surface_mode='body')
        tgt_entry=lib.entry(target_body,slot,target_variant);surface_mode=str(tgt_entry.get('support_surface') or 'body')
        tgt=lib.reference(target_body,slot,target_variant,race_code='0201',rig_joint_names=rig,surface_mode=surface_mode)
        print(json.dumps({'case':label,'stage':'body-correspondence','source_vertices':len(src['V']),'target_vertices':len(tgt['V']),'target_support_surface':surface_mode}),flush=True)
        cache=collect_body_pairs([(src,tgt)])
        view=MdlSolverView(Path(source_mdl),cache,rig_joint_names=rig,supplemental_skeleton=supplemental);body_names=body_mesh_names(view,source_materials(lib,source_body,slot,source_variant))
        print(json.dumps({'case':label,'stage':'solve','body_meshes':sorted(body_names),'garment_meshes':[n for n in view.mesh_names() if n not in body_names]}),flush=True)
        prod._reset_b14_runtime_caches(); prod._set_surface_query_cache_enabled(True)
        positions,skinning,records,stats=prod._solve_garment_meshes(view,cache,body_names,set(mesh_filter) if mesh_filter else None)
        report={'case':label,'source_body':source_body,'source_variant':source_variant,'target_body':target_body,'target_variant':target_variant,'target_support_surface':surface_mode,'body_meshes':sorted(body_names),'solver_stats':stats,'mesh_records':records,'elapsed_sec':time.time()-t}
        serialise_npz(Path(out),view,positions,body_names,tgt,report)
        print(json.dumps({'case':label,'stage':'done','elapsed_sec':report['elapsed_sec'],'output':str(out),'seams':stats.get('authored_split_seams',{})},default=str),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['lemon-top','lemon-bottom','dress-top'],required=True);ap.add_argument('--rbody',required=True);ap.add_argument('--source-root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--mesh',action='append',default=[]);ap.add_argument('--fast-preview',action='store_true');ap.add_argument('--source-body-mdl',default='');ap.add_argument('--source-body-mdl-legs',default='');a=ap.parse_args();root=Path(a.source_root)
    if a.fast_preview:
        def _preview_componentwise(source_vertices,vertices,faces,labels,source_triangles,target_triangles,preserve_source_orientation=True):
            return np.asarray(vertices,dtype=np.float64).copy(), {'affected_faces':0,'affected_components':0,'sample_min_before_mm':None,'sample_min_after_mm':None,'max_vertex_move_mm':0.0,'rms_vertex_move_mm':0.0,'components':[],'offline_preview_skipped':True}
        def _preview_dense(source_vertices,vertices,faces,labels,classes,target_triangles,longitudinal_axis=None,margin=.00050,preserve_source_orientation=True):
            return np.asarray(vertices,dtype=np.float64).copy(), {'affected_components':0,'penetrating_after':0,'dense_min_after_mm':None,'max_vertex_move_mm':0.0,'batched':True,'offline_preview_skipped':True}
        prod._componentwise_surface_clearance_guard=_preview_componentwise
        prod._dense_target_obstacle_clearance_guard=_preview_dense
    if a.case=='lemon-top':
        run_case(a.case,root/'chara/equipment/e6072/model/l top nb m.mdl',a.rbody,'neolithe','Chest','neolithe.chest.neobelly-m','neolithe','neolithe.chest.m',a.out,a.mesh)
    elif a.case=='lemon-bottom':
        run_case(a.case,root/'chara/equipment/e6072/model/l bottom nb m.mdl',a.rbody,'neolithe','Legs','neolithe.legs.neobelly-gen-a-medium','yab','yab.legs.small-watermelon-crushers-a',a.out,a.mesh)
    else:
        dress_path=root/'Vanilla/Chest/c0201e0648_top.mdl'
        if not a.source_body_mdl:
            raise ValueError('--source-body-mdl is required for dress-top')
        chest_mdl=Path(a.source_body_mdl)
        legs_mdl=Path(a.source_body_mdl_legs) if a.source_body_mdl_legs else chest_mdl.with_name(chest_mdl.name.replace('_top.mdl','_dwn.mdl'))
        dress_parsed=parse_rigged_mdl(dress_path.read_bytes())
        rig_names=_union_joint_names(dress_parsed['joint_names'], parse_rigged_mdl(chest_mdl.read_bytes())['joint_names'], parse_rigged_mdl(legs_mdl.read_bytes())['joint_names'] if legs_mdl.exists() else [])
        source_refs=[build_source_ref_from_body_mdl(chest_mdl,rig_names,'Chest')]
        target_slots=[('neolithe','Chest','neolithe.chest.m')]
        if legs_mdl.exists():
            source_refs.append(build_source_ref_from_body_mdl(legs_mdl,rig_names,'Legs'))
            target_slots.append(('yab','Legs','yab.legs.small-watermelon-crushers-a'))
        run_case_from_source_refs(a.case,dress_path,source_refs,a.rbody,target_slots,a.out,a.mesh)
