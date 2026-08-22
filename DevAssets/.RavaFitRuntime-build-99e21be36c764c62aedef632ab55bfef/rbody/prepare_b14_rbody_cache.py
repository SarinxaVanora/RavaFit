from __future__ import annotations
import argparse,json,struct,sys
from pathlib import Path
import numpy as np
from rbody_v3_loader import RBodyV3
from rbody_b14_adapter import collect_body_pairs

def glb_json(path):
    b=Path(path).read_bytes();magic,version,length=struct.unpack_from('<4sII',b,0)
    if magic!=b'glTF':raise ValueError('Not a GLB')
    off=12
    while off<length:
        n,t=struct.unpack_from('<II',b,off);off+=8;chunk=b[off:off+n];off+=n
        if t==0x4E4F534A:return json.loads(chunk.decode('utf8').rstrip('\x00 '))
    raise ValueError('GLB has no JSON chunk')

def rig_names(path,mesh_name=None):
    js=glb_json(path);mi=None
    if mesh_name is not None:
        for i,m in enumerate(js.get('meshes',[])):
            if m.get('name')==mesh_name:mi=i;break
        if mi is None:raise KeyError(f'No mesh {mesh_name!r}')
    node=None
    for n in js.get('nodes',[]):
        if 'skin' not in n or 'mesh' not in n:continue
        if mi is None or n['mesh']==mi:node=n;break
    if node is None:raise ValueError('No skinned mesh found in rig GLB')
    sk=js['skins'][node['skin']];return [js['nodes'][j].get('name','') for j in sk['joints']]

def main():
    ap=argparse.ArgumentParser(description='Build B14 body-reference/collision cache directly from RBODY V3 selections.')
    ap.add_argument('--rig-glb',required=True);ap.add_argument('--spec',required=True);ap.add_argument('--output',required=True);ap.add_argument('--rig-mesh')
    a=ap.parse_args();spec=json.loads(Path(a.spec).read_text(encoding='utf8'));names=rig_names(a.rig_glb,a.rig_mesh)
    loaders={};pairs=[]
    try:
        for p in spec['pairs']:
            sp=str(Path(p['source_rbody']).resolve());tp=str(Path(p['target_rbody']).resolve())
            loaders.setdefault(sp,RBodyV3(sp));loaders.setdefault(tp,RBodyV3(tp))
            source_race=p.get('source_race_code',p.get('race_code'));target_race=p.get('target_race_code',p.get('race_code'));s=loaders[sp].reference(p['source_body'],p['slot'],p['source_variant'],race_code=source_race,rig_joint_names=names);t=loaders[tp].reference(p['target_body'],p['slot'],p['target_variant'],race_code=target_race,rig_joint_names=names,surface_mode=str(p.get('target_support_surface') or 'body'));pairs.append((s,t))
        c=collect_body_pairs(pairs)
        np.savez_compressed(a.output,X=c['X'],Y=c['Y'],BW=c['BW'],NS=c['NS'],NT=c['NT'],names=np.asarray(c['names'],dtype=object),parts=c['parts'],source_surface_V=c['source_surface_V'],source_surface_F=c['source_surface_F'],target_surface_V=c['target_surface_V'],target_surface_F=c['target_surface_F'])
        meta={'rig_glb':str(Path(a.rig_glb).resolve()),'rig_joint_count':len(names),'pairs':spec['pairs'],'slot_stats':c['slot_stats'],'vertices':len(c['X']),'source_surface_triangles':len(c['source_surface_F']),'target_surface_triangles':len(c['target_surface_F'])}
        Path(str(a.output)+'.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(meta,indent=2))
    finally:
        for l in loaders.values():l.close()
if __name__=='__main__':main()
