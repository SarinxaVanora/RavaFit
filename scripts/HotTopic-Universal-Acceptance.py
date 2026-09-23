#!/usr/bin/env python3
from __future__ import annotations

"""Whole-outfit offline acceptance for the universal coherent refit.

Only source garment/body geometry and selected target RBODY data enter solving. Authored YAB/default
comparison garments are intentionally absent from this script and are loaded only by the separate
post-freeze board renderer.
"""

import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OFFLINE=ROOT/'scripts'/'Offline-Mdl-Solver-Simulation.py'
spec=importlib.util.spec_from_file_location('ravafit_offline_sim',OFFLINE)
sim=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(sim)

CASES={
    'torso': ('top_ruexb_perky.mdl','ruexb-plus','Chest','Puffy','neolithe','M'),
    'legs': ('legs_ruexb.mdl','ruexb-plus','Legs','Gen A','neolithe','Gen A Medium'),
    'gloves': ('gloves.mdl','bibo-plus','Hands','Default','neolithe','Short Nails'),
    'boots': ('shoes.mdl','yab','Feet','Default','neolithe','Default'),
    'vest': ('vest_ruexb.mdl','ruexb-plus','Chest','Puffy','neolithe','M'),
    'necklace': ('necklace.mdl','ruexb-plus','Chest','Puffy','neolithe','M'),
}


def run_case(key: str, assets: Path, rbody: Path, out_dir: Path):
    filename,source_body,slot,source_variant,target_body,target_variant=CASES[key]
    source_mdl=assets/filename; started=time.time()
    print(json.dumps({'case':key,'stage':'load'}),flush=True)
    with sim.RBodyV3(rbody) as lib:
        parsed=sim.parse_rigged_mdl(source_mdl.read_bytes()); rig=list(parsed['joint_names'])
        target_slot_variants={case_slot:case_target_variant for _,(_,_,case_slot,_,case_target_body,case_target_variant) in CASES.items() if case_target_body==target_body}
        source_slot_variants={case_slot:case_source_variant for _,(_,case_source_body,case_slot,case_source_variant,_,_) in CASES.items() if case_source_body==source_body}
        inferred=sim.infer_spanning_support_pairs(source_mdl,lib,source_body,slot,source_variant,target_body,target_variant,rig_joint_names=rig,source_slot_variants=source_slot_variants,target_slot_variants=target_slot_variants)
        pairs=inferred['pairs']; cache=sim.collect_body_pairs(pairs)
        cache=sim.prod._apply_strict_historical_uv_contract(pairs,cache)
        target_surface=str(lib.entry(target_body,slot,target_variant).get('support_surface') or 'body')
        primary_target=next(t for s,t in pairs if str(s.get('slot'))==str(slot))
        view=sim.MdlSolverView(source_mdl,cache,rig_joint_names=inferred['rig_joint_names'],supplemental_skeleton=None)
        body_names=sim.body_mesh_names(view,sim.source_materials(lib,source_body,slot,source_variant))
        print(json.dumps({'case':key,'stage':'solve','strict':bool(cache.get('_ravafit_strict_b14_contract')),'support_slots':inferred['selected_slots'],'support_diagnostics':inferred['diagnostics'],'body_meshes':sorted(body_names),'garment_meshes':[n for n in view.mesh_names() if n not in body_names]}),flush=True)
        solve_view=sim.prod._IndexedRenderGarmentSource(view,body_names)
        sim.prod._reset_b14_runtime_caches(); sim.prod._set_surface_query_cache_enabled(True)
        positions,skinning,records,stats=sim.prod._solve_strict_b14_layers(solve_view,cache,body_names,None)
        positions,skinning=sim.prod._collapse_indexed_render_garment_solution(solve_view,positions,skinning)
        report={
            'case':key,'source_body':source_body,'source_variant':source_variant,
            'target_body':target_body,'target_variant':target_variant,'target_support_surface':target_surface,
            'support_slots':inferred['selected_slots'],'support_diagnostics':inferred['diagnostics'],
            'body_meshes':sorted(body_names),'target_body_output_present':bool(body_names),'solver_stats':stats,'mesh_records':records,
            'elapsed_sec':time.time()-started,
        }
        sim.serialise_npz(out_dir/f'{key}.npz',view,positions,body_names,primary_target,report)
        print(json.dumps({'case':key,'stage':'done','elapsed_sec':report['elapsed_sec'],'output':str(out_dir/f'{key}.npz'),'support_slots':inferred['selected_slots'],'changed_meshes':stats.get('local_structural_retarget',{}).get('changed_meshes',[])}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--rbody',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--case',action='append',choices=sorted(CASES),default=[]);args=ap.parse_args()
    assets=Path(args.assets);rbody=Path(args.rbody);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    for key in (args.case or list(CASES)):run_case(key,assets,rbody,out)

if __name__=='__main__':main()
