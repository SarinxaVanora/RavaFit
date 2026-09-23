from __future__ import annotations
import subprocess,sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PYTHON=sys.executable
MESHES=['mesh 1.0','mesh 2.0','mesh 2.1','mesh 2.2','mesh 2.3','mesh 2.4','mesh 3.0','mesh 4.0']
def run(*args):
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'0'})
    p=subprocess.run([PYTHON,*map(str,args)],cwd=ROOT,text=True,capture_output=True,env=env)
    if p.stdout.strip():print(p.stdout.strip())
    if p.returncode:
        if p.stderr.strip():print(p.stderr,file=sys.stderr)
        raise subprocess.CalledProcessError(p.returncode,p.args,p.stdout,p.stderr)
    return p.stdout
def main():
    t0=time.time();(ROOT/'workers').mkdir(exist_ok=True);(ROOT/'candidates').mkdir(exist_ok=True);(ROOT/'metrics').mkdir(exist_ok=True)
    print('[B14] Preparing immutable source/target body correspondence cache...');run(ROOT/'scripts/prepare_body_cache.py')
    for i,name in enumerate(MESHES,1):
        print(f'[B14] Solving {i}/{len(MESHES)} {name} in isolated process...');run(ROOT/'scripts/b14_mesh_worker.py',name)
    print('[B14] Assembling garment and exact-transplanting Selected Body...');run(ROOT/'scripts/build_B14_from_workers.py')
    print('[B14] Running integrity audit...');run(ROOT/'scripts/audit_B14.py')
    audit=json.loads((ROOT/'metrics/B14_audit.json').read_text())
    if not audit.get('pass'):raise RuntimeError('B14 audit did not pass')
    print(f'[B14] COMPLETE / AUDIT PASS in {time.time()-t0:.1f}s')
if __name__=='__main__':main()
