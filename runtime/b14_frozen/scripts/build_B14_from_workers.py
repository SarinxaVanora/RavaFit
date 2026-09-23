from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from ffxiv_lobofit import GLB,patch_positions,sha
from transplant_selected_body import transplant_selected_body

SRC=ROOT/'inputs/Original_outfit.glb'; TGT=ROOT/'inputs/Selected_Body.glb'
FIT=ROOT/'candidates/B14_final_garment.glb'; MERGED=ROOT/'candidates/B14_with_Selected_Body.glb'
MESHES=['mesh 1.0','mesh 2.0','mesh 2.1','mesh 2.2','mesh 2.3','mesh 2.4','mesh 3.0','mesh 4.0']


def worker_stem(name:str)->str:return name.replace(' ','_').replace('.','_')+'_B14'

def main():
    s=GLB(SRC); pos={}; workers={}
    for name in MESHES:
        npz=ROOT/'workers'/f'{worker_stem(name)}.npz'; js=ROOT/'workers'/f'{worker_stem(name)}.json'
        if not npz.exists(): raise FileNotFoundError(f'Missing worker result: {npz}')
        pos[name]=np.load(npz)['raw']
        workers[name]=json.loads(js.read_text()) if js.exists() else {'mesh':name}
    FIT.parent.mkdir(exist_ok=True); patch_positions(SRC,FIT,pos,True)
    body_info=transplant_selected_body(str(FIT),str(TGT),str(MERGED))
    rec={
        'candidate':'B14_reproduced',
        'fresh_from_original':True,
        'generation_uses_validation':False,
        'source_hash':sha(SRC),'target_hash':sha(TGT),
        'fit_hash':sha(FIT),'merged_hash':sha(MERGED),
        'metrics':{n:workers[n].get('metrics') for n in MESHES},
        'behaviors':{n:workers[n].get('behavior') for n in MESHES},
        'workers':workers,
        'body_transplant_info':body_info,
    }
    (ROOT/'metrics').mkdir(exist_ok=True);(ROOT/'metrics/B14.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps({'fit':str(FIT),'with_selected_body':str(MERGED),'fit_hash':rec['fit_hash'],'merged_hash':rec['merged_hash']},indent=2))
if __name__=='__main__':main()
