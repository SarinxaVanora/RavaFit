from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMEXPR_NUM_THREADS']='1'
import numpy as np
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent;RUNTIME_ROOT=HERE.parent
for path in (HERE,RUNTIME_ROOT/'rbody',RUNTIME_ROOT/'b14_frozen'/'scripts'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from b14_compat import solve_constructed_close, solve_standoff, solve_flexible

def safe(v):
    if v is None or isinstance(v,(str,bool,int,float)):return v
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    return str(v)

def main():
    if len(sys.argv)!=5:raise SystemExit('shell_solve_worker.py <input.npz> <meta.json> <output.npz> <stage.json>')
    input_path,meta_path,output_path,stage_path=map(Path,sys.argv[1:])
    meta=json.loads(meta_path.read_text(encoding='utf-8'));z=np.load(input_path,allow_pickle=False)
    rig=SimpleNamespace(js=meta['source_js']);w={'V':z['w_V'],'F':z['w_F'],'W':z['w_W'],'joint_names':list(meta['w_joint_names'])}
    Wg=z['Wg'];base=z['base'];contact=z['contact'];X=z['X'];Y=z['Y'];BW=z['BW'];NS=z['NS'];NT=z['NT'];names=list(meta['names']);axes=z['axes'];source_tri=z['source_tri'];target_tri=z['target_tri'];labels=z['labels'];features=dict(meta['features']);behavior=str(meta['behavior']);t0=time.perf_counter()
    if behavior=='constructed_close_shell':
        U,ids,F,stage=solve_constructed_close(rig,rig,w,Wg,base,X,Y,BW,NS,NT,names,axes,cKDTree(X),source_tri,target_tri,labels,features)
    elif behavior=='stand_off_structured_shell':
        U,ids,F,stage=solve_standoff(rig,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features)
    elif behavior=='body_following_flexible_layer':
        U,ids,F,stage=solve_flexible(w,Wg,base,axes,target_tri,labels,features)
    else:raise ValueError(f'Fresh shell worker does not support {behavior}')
    np.savez(output_path,U=np.asarray(U,dtype=np.float64),ids=np.asarray(ids,dtype=np.int64),F=np.asarray(F,dtype=np.int64))
    stage_path.write_text(json.dumps({'stage':safe(stage),'elapsed_sec':time.perf_counter()-t0,'pid':os.getpid()},separators=(',',':')),encoding='utf-8')
    return 0
if __name__=='__main__':
    code=int(main() or 0);sys.stdout.flush();sys.stderr.flush();os._exit(code)
