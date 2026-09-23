from __future__ import annotations
import json,pickle,sys,time,traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import production_b14 as prod
import target_fit_authority as target_fit
import adaptive_skinning

class SerializedMultiGarmentSource:
    def __init__(self,js:dict,meshes:dict[str,dict]):self.js=js;self._meshes=meshes
    def mesh_names(self):return list(self._meshes)
    def data(self,name):return self._meshes[name]


def main()->int:
    if len(sys.argv)!=6:raise SystemExit('garment_finalize_worker.py <common.pkl> <manifest.json> <ready.flag> <output.pkl> <report.json>')
    common_path=Path(sys.argv[1]);manifest_path=Path(sys.argv[2]);ready_path=Path(sys.argv[3]);output_path=Path(sys.argv[4]);report_path=Path(sys.argv[5]);started=time.perf_counter()
    try:
        deadline=time.monotonic()+420.0
        while not ready_path.exists():
            if time.monotonic()>deadline:raise TimeoutError('timed out waiting for garment workers')
            time.sleep(.05)
        with common_path.open('rb') as f:common=pickle.load(f)
        rows=json.loads(manifest_path.read_text(encoding='utf-8')).get('meshes',[]);meshes={};positions={};skinning={};records={};local_rms=None;worker_reports=[]
        for row in rows:
            name=str(row['name'])
            with Path(row['mesh_payload']).open('rb') as f:mesh_payload=pickle.load(f)
            meshes[name]=mesh_payload['data']
            payload=json.loads(Path(row['report']).read_text(encoding='utf-8'))
            if not payload.get('ok',False):raise RuntimeError(f'garment worker failed for {name}: {payload.get("error")}')
            with np.load(Path(row['output']),allow_pickle=False) as z:
                positions[name]=np.asarray(z['position'],dtype=np.float64).copy();weights=np.asarray(z['weights'],dtype=np.float64).copy()
            skinning[name]={'weights':weights,'joint_names':list(payload.get('joint_names',[])),'stage':payload.get('skinning_stage',{})};records[name]=payload.get('record',{})
            value=(payload.get('stats') or {}).get('local_affine_quality_rms_mm')
            if local_rms is None and value is not None:local_rms=float(value)
            worker_reports.append({'mesh':name,'worker_sec':payload.get('elapsed_sec'),'pid':payload.get('pid')})
        source=SerializedMultiGarmentSource(common['source_js'],meshes);cache=common['cache']
        capacity_report=adaptive_skinning.register_source_influence_capacities(cache,source)
        target_fit.install_close_shell_macro_authority(prod);adaptive_skinning.install_adaptive_body_skinning(prod)
        if local_rms is None:
            _A,quality,_tree=prod.precompute_body_local_affines(cache['X'],cache['Y'],cache['BW']);local_rms=float(np.sqrt(np.mean(np.asarray(quality,float)**2))*1000.0)
        positions,skinning,records,stats=prod._finalize_garment_solution(source,cache,positions,skinning,records,float(local_rms),_assembly_in_process=True)
        stats['source_influence_capacity_registry']=capacity_report
        stats['garment_mesh_workers']={'enabled':True,'mode':'one-clean-multimesh-worker+fresh-finalizer','mesh_count':len(rows),'workers':worker_reports,'finalizer_pid':__import__('os').getpid(),'pipeline_wall_sec':time.perf_counter()-started}
        output_path.parent.mkdir(parents=True,exist_ok=True)
        with output_path.open('wb') as f:pickle.dump({'positions':positions,'skinning':skinning,'records':records,'stats':stats},f,protocol=pickle.HIGHEST_PROTOCOL)
        report_path.write_text(json.dumps({'ok':True,'elapsed_sec':time.perf_counter()-started,'mesh_count':len(rows),'pid':__import__('os').getpid()},separators=(',',':')),encoding='utf-8');return 0
    except Exception as ex:
        report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps({'ok':False,'error':str(ex),'traceback':traceback.format_exc()[-10000:]},separators=(',',':')),encoding='utf-8');return 2

if __name__=='__main__':raise SystemExit(main())
