from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMEXPR_NUM_THREADS']='1'
import numpy as np
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent;RUNTIME_ROOT=HERE.parent;B14=RUNTIME_ROOT/'b14_frozen'/'scripts'
# Deliberately exclude HERE from sys.path before importing the frozen worker: this process must never load b14_compat.
sys.path=[str(B14)]+[p for p in sys.path if Path(p or '.').resolve()!=HERE.resolve()]
import b14_mesh_worker as _worker
solve_constructed_close=_worker.solve_constructed_close
solve_standoff=_worker.solve_standoff
solve_flexible=_worker.solve_flexible
solve_conservative=_worker.solve_conservative

def safe(v):
    if v is None or isinstance(v,(str,bool,int,float)):return v
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    return str(v)

def main():
    if len(sys.argv)!=5:raise SystemExit('strict_b14_shell_worker.py <input.npz> <meta.json> <output.npz> <stage.json>')
    input_path,meta_path,output_path,stage_path=map(Path,sys.argv[1:])
    meta=json.loads(meta_path.read_text(encoding='utf-8'));z=np.load(input_path,allow_pickle=False)
    rig=SimpleNamespace(js=meta['source_js'])
    w={'V':z['w_V'],'F':z['w_F'],'W':z['w_W'],'joint_names':list(meta['w_joint_names']),'material':meta.get('material',''),'name':meta.get('mesh_name','mesh')}
    Wg=z['Wg'];base=z['base'];contact=z['contact'];X=z['X'];Y=z['Y'];BW=z['BW'];NS=z['NS'];NT=z['NT'];names=list(meta['names']);axes=z['axes'];source_tri=z['source_tri'];target_tri=z['target_tri'];labels=z['labels'];features=dict(meta['features']);classes={int(k):str(v) for k,v in meta.get('classes',{}).items()};behavior=str(meta['behavior']);t0=time.perf_counter()
    unique_components=np.unique(labels)
    old_refine=None
    if len(unique_components)<=1 and behavior in {'constructed_close_shell','stand_off_structured_shell'}:
        old_refine=_worker.refine_assembly
        def _identity_refine(Vsrc,Vinit,F,lab,root_component=0,iterations=800,lr=.025,max_gap=.0030,log_every=100):
            return np.asarray(Vinit,float).copy(),{},[],[{'iter':0,'loss':0.0,'attach':0.0,'frame':0.0,'shape':0.0,'reg':0.0,'skipped':True,'reason':'single-component strict B14 assembly has no movable peer parameters','gaps':[]}]
        _worker.refine_assembly=_identity_refine
    def _run():
        requires_longitudinal_frame=behavior in {'constructed_close_shell','stand_off_structured_shell'}
        if requires_longitudinal_frame and ('j_kubi' not in names or 'j_kosi' not in names):
            U,ids,F,stage=solve_conservative(w,Wg,base,labels,classes,features)
            return U,ids,F,{'mode':'strict_b14_missing_longitudinal_rig_guard','requested_behavior':behavior,'solve':stage,'reason':'required longitudinal skeleton anchors are absent from this accessory rig'}
        if behavior=='constructed_close_shell':
            return solve_constructed_close(rig,rig,w,Wg,base,X,Y,BW,NS,NT,names,axes,cKDTree(X),source_tri,target_tri,labels,features)
        if behavior=='stand_off_structured_shell':
            return solve_standoff(rig,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features)
        if behavior=='body_following_flexible_layer':
            return solve_flexible(w,Wg,base,axes,target_tri,labels,features)
        return solve_conservative(w,Wg,base,labels,classes,features)
    try:
        try:
            U,ids,F,stage=_run()
        except ValueError as ex:
            if 'empty parameter list' not in str(ex):
                raise
            if old_refine is None:
                old_refine=_worker.refine_assembly
            def _identity_refine_retry(Vsrc,Vinit,F,lab,root_component=0,iterations=800,lr=.025,max_gap=.0030,log_every=100):
                return np.asarray(Vinit,float).copy(),{},[],[{'iter':0,'loss':0.0,'attach':0.0,'frame':0.0,'shape':0.0,'reg':0.0,'skipped':True,'reason':'strict B14 assembly retry: no movable rigid/ribbon parameters','gaps':[]}]
            _worker.refine_assembly=_identity_refine_retry
            U,ids,F,stage=_run()
            stage={'mode':'strict_b14_empty_assembly_guard','solve':stage}
    finally:
        if old_refine is not None:_worker.refine_assembly=old_refine
    np.savez_compressed(output_path,U=np.asarray(U,dtype=np.float64),ids=np.asarray(ids,dtype=np.int64),F=np.asarray(F,dtype=np.int64))
    stage_path.write_text(json.dumps({'stage':safe(stage),'elapsed_sec':time.perf_counter()-t0,'pid':os.getpid(),'strict_frozen_b14':True},separators=(',',':')),encoding='utf-8')
    return 0
if __name__=='__main__':
    code=int(main() or 0);sys.stdout.flush();sys.stderr.flush();os._exit(code)
