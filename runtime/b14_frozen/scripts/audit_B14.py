from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from ffxiv_lobofit import GLB,sha
from collision_eval import body_triangles,evaluate
from structural_refine import unique_edges
from transplant_selected_body import audit_transplant
SRC=ROOT/'inputs/Original_outfit.glb';TGT=ROOT/'inputs/Selected_Body.glb';FIT=ROOT/'candidates/B14_final_garment.glb';MERGED=ROOT/'candidates/B14_with_Selected_Body.glb'

def main():
    s=GLB(SRC);t=GLB(TGT);g=GLB(FIT);m=GLB(MERGED);tt=body_triangles(t)
    rep={'candidate':'B14_reproduced','source_hash':sha(SRC),'target_hash':sha(TGT),'fit_hash':sha(FIT),'merged_hash':sha(MERGED),'meshes':{}}
    for name in s.mesh_names():
        if not name or name.startswith('mesh 0.') or name not in g.mesh_names():continue
        a=s.data(name);b=g.data(name);V0,V,F=a['V'],b['V'],a['F']
        topology=bool(np.array_equal(F,b['F']))
        E=unique_edges(F);l0=np.linalg.norm(V0[E[:,1]]-V0[E[:,0]],axis=1);l=np.linalg.norm(V[E[:,1]]-V[E[:,0]],axis=1);er=l/np.maximum(l0,1e-12)
        n0=np.cross(V0[F[:,1]]-V0[F[:,0]],V0[F[:,2]]-V0[F[:,0]]);n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
        ar0=np.linalg.norm(n0,axis=1)*.5;ar=np.linalg.norm(n,axis=1)*.5;dot=np.sum(n0*n,axis=1)/np.maximum(np.linalg.norm(n0,axis=1)*np.linalg.norm(n,axis=1),1e-20)
        coll0=evaluate(V,tt,k=20,epsilon=.0003);coll={k:v for k,v in coll0.items() if k not in ('signed','distance')}
        _,ps=s.primitive(name);_,pg=g.primitive(name);attrs={}
        for k,ai in ps['attributes'].items():
            if k in ('POSITION','NORMAL','TANGENT'):continue
            x=s.accessor(ai);y=g.accessor(pg['attributes'][k]);attrs[k]=bool(np.array_equal(x,y))
        rep['meshes'][name]={'vertices':len(V),'faces':len(F),'topology_exact':topology,'finite':bool(np.isfinite(V).all()),'edge_ratio_p':np.percentile(er,[1,5,50,95,99]).tolist(),'area_ratio_p':np.percentile(ar/np.maximum(ar0,1e-20),[1,5,50,95,99]).tolist(),'degenerate_faces':int(np.sum(ar<1e-10)),'orientation_dot_p':np.percentile(dot,[0,1,5,50]).tolist(),'opposed_faces':int(np.sum(dot<0)),'collision':coll,'preserved_attributes':attrs,'all_non_geometry_attributes_exact':all(attrs.values())}
    def clean(n):return {k:v for k,v in n.items() if k not in ('mesh','skin','extras')}
    joint_count=min(172,len(s.js.get('nodes',[])),len(g.js.get('nodes',[])),len(m.js.get('nodes',[])))
    rep['skeleton_exact']=all(clean(s.js['nodes'][i])==clean(g.js['nodes'][i]) for i in range(joint_count))
    rep['materials_exact']=s.js.get('materials')==g.js.get('materials')
    tr=audit_transplant(str(MERGED),str(TGT));rep['body_transplant']=tr
    rep['merged_skeleton_exact']=all(clean(s.js['nodes'][i])==clean(m.js['nodes'][i]) for i in range(joint_count))
    rep['merged_materials_exact']=s.js.get('materials')==m.js.get('materials')
    rep['old_source_body_reachable']=False
    for name in ['mesh 0.1','mesh 0.2','mesh 0.3']:
        if name in s.mesh_names() and name in m.mesh_names() and name in t.mesh_names():
            sd=s.data(name);md=m.data(name)
            if len(md['V'])==len(sd['V']) and np.array_equal(md['V'],sd['V']):rep['old_source_body_reachable']=True
    mesh_ok=all(x['finite'] and x['topology_exact'] and x['degenerate_faces']==0 and x['opposed_faces']==0 and x['all_non_geometry_attributes_exact'] for x in rep['meshes'].values())
    rep['pass']=bool(mesh_ok and rep['skeleton_exact'] and rep['materials_exact'] and tr['all_exact'] and rep['merged_skeleton_exact'] and rep['merged_materials_exact'] and not rep['old_source_body_reachable'])
    (ROOT/'metrics/B14_audit.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps({'pass':rep['pass'],'body_exact':tr['all_exact'],'source_body_reachable':rep['old_source_body_reachable'],'meshes':{k:{'opposed':v['opposed_faces'],'degenerate':v['degenerate_faces']} for k,v in rep['meshes'].items()}},indent=2))
    if not rep['pass']:raise SystemExit(2)
if __name__=='__main__':main()
