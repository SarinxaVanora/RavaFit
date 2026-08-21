#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'runtime'/'solver'));sys.path.insert(0,str(ROOT/'runtime'/'rbody'))
from rbody_v3_loader import RBodyV3
from rbody_v3_core import parse_rigged_mdl
from rbody_b14_adapter import collect_body_pairs
import production_b14 as prod
from importlib.machinery import SourceFileLoader
sim=SourceFileLoader('offsim',str(ROOT/'scripts'/'Offline-Mdl-Solver-Simulation.py')).load_module()
source=Path('/mnt/data/ravafit_work/common/chara/equipment/e6072/model/l bottom nb m.mdl')
rbody=Path('/mnt/data/ravafit_work/Bodies_NeoBelly_Complete.rbody')
parts={4:Path('/mnt/data/ravafit_work/lemon_bottom_mesh4_preview.npz'),5:Path('/mnt/data/ravafit_work/lemon_bottom_mesh5_preview.npz'),7:Path('/mnt/data/ravafit_work/lemon_bottom_mesh7_preview.npz')}
parsed=parse_rigged_mdl(source.read_bytes());rig=list(parsed['joint_names']);top=parse_rigged_mdl(source.with_name('l top nb m.mdl').read_bytes())
for j in top['joint_names']:
    if j not in rig:rig.append(j)
with RBodyV3(rbody) as lib:
    src=lib.reference('neolithe','Legs','neolithe.legs.neobelly-sfw-medium',race_code='0201',rig_joint_names=rig,surface_mode='body')
    tgt=lib.reference('yab','Legs','yab.legs.small-watermelon-crushers-a',race_code='0201',rig_joint_names=rig,surface_mode='body')
    cache=collect_body_pairs([(src,tgt)])
    view=sim.MdlSolverView(source,cache,rig_joint_names=rig,supplemental_skeleton=top)
    body_names=sim.body_mesh_names(view,sim.source_materials(lib,'neolithe','Legs','neolithe.legs.neobelly-sfw-medium'))
    positions={}
    for name in view.mesh_names():
        if name in body_names:continue
        idx=int(name.split()[-1]);d=view.data(name)
        if idx in parts:
            z=np.load(parts[idx]);positions[name]=np.asarray(z[name.replace(' ','_')+'_new_V'],dtype=np.float64)
        else:
            positions[name]=np.asarray(d['V'],dtype=np.float64).copy()
    source_tri=prod._triangles_from_surface(cache['source_support_V'],cache['source_support_F'])
    target_tri=prod._triangles_from_surface(cache['target_support_V'],cache['target_support_F'])
    layer_meshes={n:{'V':np.asarray(view.data(n)['V'],dtype=np.float64),'F':np.asarray(view.data(n)['F'],dtype=np.int64)} for n in positions}
    positions,changed,layer_report=prod._preserve_authored_garment_layers(layer_meshes,positions,source_tri,target_tri)
    contexts={n:{'data':view.data(n),'w':prod.weld_mesh(view.data(n))} for n in positions}
    positions,seam_changed,seam_report=prod._preserve_authored_weld_splits(positions,contexts)
    sim.serialise_npz(Path('/mnt/data/ravafit_work/lemon_bottom_new_preview_merged.npz'),view,positions,body_names,tgt,{
        'case':'lemon-bottom','mode':'offline_structural_preview','full_production_collision_guards':False,
        'mesh_6_ties':'untouched_source_geometry_due_frozen_B14_standoff_offline_runtime',
        'solved_meshes':['mesh 4','mesh 5','mesh 7'],'layer_report':layer_report,'seam_report':seam_report,
    })
    print(json.dumps({'body_meshes':sorted(body_names),'solved_meshes':['mesh 4','mesh 5','mesh 7'],'source_preserved_meshes':['mesh 6'],'layer_changed':sorted(changed),'seam_changed':sorted(seam_changed),'output':'/mnt/data/ravafit_work/lemon_bottom_new_preview_merged.npz'},indent=2))
