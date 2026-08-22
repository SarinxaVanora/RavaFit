from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMEXPR_NUM_THREADS']='1'
import numpy as np
HERE=Path(__file__).resolve().parent;RUNTIME_ROOT=HERE.parent
for path in (HERE,RUNTIME_ROOT/'rbody',RUNTIME_ROOT/'b14_frozen'/'scripts'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import production_b14 as prod

def safe(v):
    if v is None or isinstance(v,(str,bool,int,float)):return v
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    return str(v)

def main():
    if len(sys.argv)!=5:raise SystemExit('assembly_worker.py <input.npz> <meta.json> <output.npz> <report.json>')
    input_path,meta_path,output_path,report_path=map(Path,sys.argv[1:]);meta=json.loads(meta_path.read_text(encoding='utf-8'));z=np.load(input_path,allow_pickle=False);started=time.perf_counter();source_meshes={};positions={}
    for row in meta['meshes']:
        key=str(row['key']);name=str(row['name']);source_meshes[name]={'V':np.asarray(z[f'{key}_source_v'],dtype=np.float64),'F':np.asarray(z[f'{key}_faces'],dtype=np.int64)};positions[name]=np.asarray(z[f'{key}_position'],dtype=np.float64)
    source_tri=np.asarray(z['source_tri'],dtype=np.float64);target_tri=np.asarray(z['target_tri'],dtype=np.float64)
    positions,layered,layer_stage=prod._preserve_authored_garment_layers(source_meshes,positions,source_tri,target_tri)
    layer_pairs={frozenset((str(r.get('inner')),str(r.get('outer')))) for r in layer_stage.get('relations',[]) if r.get('inner') and r.get('outer')}
    positions,cross,cross_stage=prod._preserve_authored_cross_mesh_assembly(source_meshes,positions,layer_pairs)
    arrays={}
    for row in meta['meshes']:
        key=str(row['key']);name=str(row['name']);arrays[f'{key}_position']=np.asarray(positions[name],dtype=np.float64)
    np.savez(output_path,**arrays);report_path.write_text(json.dumps({'elapsed_sec':time.perf_counter()-started,'layered':sorted(layered),'cross':sorted(cross),'layer_stage':safe(layer_stage),'cross_stage':safe(cross_stage)},separators=(',',':')),encoding='utf-8');return 0
if __name__=='__main__':
    code=int(main() or 0);sys.stdout.flush();sys.stderr.flush();os._exit(code)
