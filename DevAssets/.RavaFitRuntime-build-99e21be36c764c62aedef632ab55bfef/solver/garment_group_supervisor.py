#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        print('usage: garment_group_supervisor.py <worker.py> <common.pkl> <manifest.json> <report.json>', file=sys.stderr)
        return 2
    worker=Path(sys.argv[1]); common=Path(sys.argv[2]); manifest=Path(sys.argv[3]); report=Path(sys.argv[4])
    rows=json.loads(manifest.read_text(encoding='utf-8')).get('meshes',[])
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform.startswith('win') else 0
    started=time.perf_counter()
    try:
        # One isolated worker handles the full garment and shares immutable body-field caches.
        proc=subprocess.Popen([sys.executable,str(worker),'--batch',str(common),str(manifest),str(report)+'.worker'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=creationflags,env=env)
        try: code=proc.wait(timeout=720)
        except subprocess.TimeoutExpired:
            proc.kill(); code=proc.wait(); raise RuntimeError('batched garment worker timed out')
        worker_report=Path(str(report)+'.worker')
        payload=json.loads(worker_report.read_text(encoding='utf-8')) if worker_report.exists() else {'ok':False,'error':f'exit {code} without report'}
        if code != 0 or not payload.get('ok',False):
            raise RuntimeError(f"batched garment worker failed: {payload.get('error') or code}")
        results=[]
        by_name={str(row.get('mesh')):row for row in payload.get('meshes',[])}
        for row in rows:
            name=str(row.get('name'))
            mesh_payload=json.loads(Path(row['report']).read_text(encoding='utf-8')) if Path(row['report']).exists() else {'ok':False,'error':'missing per-mesh report'}
            if not mesh_payload.get('ok',False):
                raise RuntimeError(f"worker failed for {name}: {mesh_payload.get('error')}")
            item=by_name.get(name,{})
            results.append({'mesh':name,'wall_sec':item.get('wall_sec',mesh_payload.get('elapsed_sec')),'worker':mesh_payload})
        report.write_text(json.dumps({'ok':True,'elapsed_sec':time.perf_counter()-started,'wave_size':len(rows),'mode':'one-clean-multimesh-worker','worker_pid':payload.get('pid'),'meshes':results},separators=(',',':')),encoding='utf-8')
        try: worker_report.unlink(missing_ok=True)
        except Exception: pass
        return 0
    except Exception as ex:
        report.write_text(json.dumps({'ok':False,'elapsed_sec':time.perf_counter()-started,'wave_size':len(rows),'mode':'one-clean-multimesh-worker','error':str(ex)},separators=(',',':')),encoding='utf-8')
        return 1

if __name__=='__main__':
    raise SystemExit(main())
