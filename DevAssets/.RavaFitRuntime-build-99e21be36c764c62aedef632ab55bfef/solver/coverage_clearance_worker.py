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
    if len(sys.argv)!=5:raise SystemExit('coverage_clearance_worker.py <input.npz> <meta.json> <output.npz> <report.json>')
    input_path,meta_path,output_path,report_path=map(Path,sys.argv[1:]);meta=json.loads(meta_path.read_text(encoding='utf-8'));z=np.load(input_path,allow_pickle=False);started=time.perf_counter();out_arrays={};reports=[]
    for job in meta['jobs']:
        jid=str(job['id']);cur=np.asarray(z[f'{jid}_current'],dtype=np.float64);src=np.asarray(z[f'{jid}_source'],dtype=np.float64);faces=np.asarray(z[f'{jid}_faces'],dtype=np.int64);slot_reports=[];source_limit=float(job.get('source_limit',.0040))
        for si,slot in enumerate(job['slots']):
            x=np.asarray(z[f'{jid}_slot{si}_x'],dtype=np.float64);y=np.asarray(z[f'{jid}_slot{si}_y'],dtype=np.float64);ns=np.asarray(z[f'{jid}_slot{si}_ns'],dtype=np.float64);nt=np.asarray(z[f'{jid}_slot{si}_nt'],dtype=np.float64)
            sv=np.asarray(z[f'{jid}_slot{si}_source_v'],dtype=np.float64);sf=np.asarray(z[f'{jid}_slot{si}_source_f'],dtype=np.int64);tv=np.asarray(z[f'{jid}_slot{si}_target_v'],dtype=np.float64);tf=np.asarray(z[f'{jid}_slot{si}_target_f'],dtype=np.int64);tsv=np.asarray(z[f'{jid}_slot{si}_target_support_v'],dtype=np.float64);tsf=np.asarray(z[f'{jid}_slot{si}_target_support_f'],dtype=np.int64)
            if bool(job.get('coverage_enabled',True)):
                candidate,coverage=prod._preserve_source_authored_body_coverage(src,cur,faces,x,y,ns,nt,source_limit=source_limit,margin=float(meta.get('margin',.00065)))
                if np.any(np.linalg.norm(candidate-cur,axis=1)>1e-10):cur=np.asarray(candidate,dtype=np.float64)
            else:
                coverage={'enabled':False,'reason':'clearance-only source-contact lane preserves non-close garment geometry'}
            candidate,clearance=prod._source_covered_face_clearance(src,cur,faces,sv[sf],tv[tf],margin=float(meta.get('margin',.00065)),target_support_triangles=tsv[tsf])
            if int(clearance.get('affected_faces',0))>0:cur=np.asarray(candidate,dtype=np.float64)
            slot_reports.append({'slot':str(slot),'body_coverage':coverage,'face_clearance':clearance})
        out_arrays[f'{jid}_current']=cur;reports.append({'id':jid,'mesh':job['mesh'],'source_limit_mm':source_limit*1000.0,'slots':slot_reports})
    np.savez(output_path,**out_arrays);report_path.write_text(json.dumps({'elapsed_sec':time.perf_counter()-started,'reports':safe(reports)},separators=(',',':')),encoding='utf-8');return 0
if __name__=='__main__':
    code=int(main() or 0);sys.stdout.flush();sys.stderr.flush();os._exit(code)
