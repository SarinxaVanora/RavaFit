import sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from ffxiv_lobofit import *
s=GLB(ROOT/'inputs/Original_outfit.glb');t=GLB(ROOT/'inputs/Selected_Body.glb')
X,Y,BW,NS,NT,names,parts=collect_body_pairs_full(s,t);A,q,tree=precompute_body_local_affines(X,Y,BW);axes=skeleton_bone_axes(s,names)
out=ROOT/'workers/body_cache.npz';np.savez_compressed(out,X=X,Y=Y,BW=BW,NS=NS,NT=NT,A=A,q=q,axes=axes,names=np.asarray(names,dtype=object),parts=parts)
meta={'source_hash':sha(ROOT/'inputs/Original_outfit.glb'),'target_hash':sha(ROOT/'inputs/Selected_Body.glb'),'vertices':len(X),'joint_count':len(names)};(ROOT/'workers/body_cache.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
