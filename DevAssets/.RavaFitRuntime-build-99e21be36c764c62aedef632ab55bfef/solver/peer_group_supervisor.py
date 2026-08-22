from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

# Keep this supervisor free of numerical imports so worker startup remains isolated.
def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit('peer_group_supervisor.py <peer_worker.py> <input.npz> <manifest.json> <report.json>')
    worker=Path(sys.argv[1]); input_path=Path(sys.argv[2]); manifest_path=Path(sys.argv[3]); report_path=Path(sys.argv[4])
    rows=json.loads(manifest_path.read_text(encoding='utf-8')).get('peers',[]); started=time.perf_counter();reports=[]
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform.startswith('win') else 0
    running=[]
    for row in rows:
        peer_id=int(row['peer_id']);meta=Path(row['meta']);output=Path(row['output']);stage=Path(row['stage']);output.unlink(missing_ok=True);stage.unlink(missing_ok=True)
        proc=subprocess.Popen([sys.executable,str(worker),str(input_path),str(meta),str(output),str(stage)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=creationflags,env=env)
        running.append((peer_id,proc,output,stage,time.perf_counter()))
    failed=[]
    for peer_id,proc,output,stage,peer_started in running:
        try:
            stdout,stderr=proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate();failed.append({'peer_id':peer_id,'error':f'timeout: {(stderr or stdout)[-2000:]}' });continue
        if proc.returncode!=0 or not output.exists() or not stage.exists():
            failed.append({'peer_id':peer_id,'error':f'exit {proc.returncode}: {(stderr or stdout)[-2000:]}' });continue
        payload=json.loads(stage.read_text(encoding='utf-8'));reports.append({'peer_id':peer_id,'attempts':1,'wall_sec':time.perf_counter()-peer_started,'worker_sec':payload.get('elapsed_sec'),'pid':payload.get('pid')})
    if failed:
        report_path.write_text(json.dumps({'ok':False,'error':'peer wave failure','failed':failed,'reports':reports},separators=(',',':')),encoding='utf-8');return 2
    report_path.write_text(json.dumps({'ok':True,'elapsed_sec':time.perf_counter()-started,'reports':reports},separators=(',',':')),encoding='utf-8');return 0

if __name__=='__main__':
    code=int(main() or 0);sys.stdout.flush();sys.stderr.flush();os._exit(code)
