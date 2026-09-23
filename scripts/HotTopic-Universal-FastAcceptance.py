#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OFFLINE=ROOT/'scripts'/'Offline-Mdl-Solver-Simulation.py'
spec=importlib.util.spec_from_file_location('ravafit_offline_sim',OFFLINE)
sim=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(sim)

CASES={
    'legs': ('legs_ruexb.mdl','ruexb-plus','Legs','Gen A','neolithe','Gen A Medium'),
    'gloves': ('gloves.mdl','bibo-plus','Hands','Default','neolithe','Short Nails'),
    'boots': ('shoes.mdl','yab','Feet','Default','neolithe','Default'),
    'vest': ('vest_ruexb.mdl','ruexb-plus','Chest','Puffy','neolithe','M'),
    'necklace': ('necklace.mdl','ruexb-plus','Chest','Puffy','neolithe','M'),
}

def run_case(key,assets,rbody,out_dir):
    filename,source_body,slot,source_variant,target_body,target_variant=CASES[key]; source_mdl=assets/filename; started=time.time()
    with sim.RBodyV3(rbody) as lib:
        parsed=sim.parse_rigged_mdl(source_mdl.read_bytes())
        native_src=lib.reference(source_body,slot,source_variant,race_code='0201',rig_joint_names=None,surface_mode='body')
        rig=sim._union_joint_names(parsed['joint_names'],native_src['joint_names'])
        target_slot_variants={case_slot:case_target_variant for _,(_,_,case_slot,_,case_target_body,case_target_variant) in CASES.items() if case_target_body==target_body}
        source_slot_variants={case_slot:case_source_variant for _,(_,case_source_body,case_slot,case_source_variant,_,_) in CASES.items() if case_source_body==source_body}
        inferred=sim.infer_spanning_support_pairs(source_mdl,lib,source_body,slot,source_variant,target_body,target_variant,rig_joint_names=rig,source_slot_variants=source_slot_variants,target_slot_variants=target_slot_variants)
        pairs=inferred['pairs']; cache=sim.prod._apply_strict_historical_uv_contract(pairs,sim.collect_body_pairs(pairs))
        primary_target=next(t for s,t in pairs if str(s.get('slot'))==str(slot))
        view=sim.MdlSolverView(source_mdl,cache,rig_joint_names=inferred['rig_joint_names'],supplemental_skeleton=None)
        body_names=sim.body_mesh_names(view,sim.source_materials(lib,source_body,slot,source_variant))
        sim.prod._reset_b14_runtime_caches(); sim.prod._set_surface_query_cache_enabled(True)
        positions,skinning,records,stats=sim.prod._solve_garment_meshes(view,cache,body_names,None)
        source_tri=sim.prod._triangles_from_surface(cache.get('_ravafit_strict_source_surface_V',cache['source_support_V']),cache.get('_ravafit_strict_source_surface_F',cache['source_support_F']))
        target_tri=sim.prod._triangles_from_surface(cache.get('_ravafit_strict_target_surface_V',cache['target_support_V']),cache.get('_ravafit_strict_target_surface_F',cache['target_support_F']))
        positions,universal=sim.prod._apply_b14_local_structural_retarget(view,positions,cache,source_tri,target_tri,source_asset_has_embedded_body=bool(body_names))
        stats['universal_coherent_refit']=universal
        report={'case':key,'source_body':source_body,'source_variant':source_variant,'target_body':target_body,'target_variant':target_variant,'support_slots':inferred['selected_slots'],'support_diagnostics':inferred['diagnostics'],'body_meshes':sorted(body_names),'target_body_output_present':bool(body_names),'solver_stats':stats,'mesh_records':records,'elapsed_sec':time.time()-started}
        sim.serialise_npz(out_dir/f'{key}.npz',view,positions,body_names,primary_target,report)
        print(json.dumps({'case':key,'elapsed_sec':report['elapsed_sec'],'support_slots':inferred['selected_slots'],'changed_meshes':universal.get('changed_meshes',[]),'output':str(out_dir/f'{key}.npz')}),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',required=True); ap.add_argument('--rbody',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--case',action='append',choices=sorted(CASES),default=[]); a=ap.parse_args()
    assets=Path(a.assets); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    for key in (a.case or list(CASES)):run_case(key,assets,Path(a.rbody),out)
if __name__=='__main__':main()
