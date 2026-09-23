from __future__ import annotations
import json,pickle,sys,time,traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import production_b14 as prod
import target_fit_authority as target_fit


class SerializedMultiGarmentSource:
    def __init__(self,js:dict,meshes:dict[str,dict]):self.js=js;self._meshes=meshes
    def mesh_names(self):return list(self._meshes)
    def data(self,name):return self._meshes[name]

class SerializedGarmentSource:
    def __init__(self,js:dict,mesh_name:str,mesh_data:dict):
        self.js=js;self._name=mesh_name;self._data=mesh_data
    def mesh_names(self):return [self._name]
    def data(self,name):
        if name!=self._name:raise KeyError(name)
        return self._data


def _solve_one(common:dict, mesh:dict, output_path:Path, report_path:Path, shared_cache:dict|None=None, reset_runtime:bool=True)->dict:
    started=time.perf_counter();mesh_name=str(mesh['name'])
    source=SerializedGarmentSource(common['source_js'],mesh_name,mesh['data'])
    cache=shared_cache if shared_cache is not None else dict(common['cache'])
    target_fit.install_close_shell_macro_authority(prod)
    if reset_runtime:
        for key in ('_ravafit_local_affines','_ravafit_source_support_triangles','_ravafit_target_support_triangles','_ravafit_target_collision_triangles'):
            cache.pop(key,None)
        prod._reset_b14_runtime_caches();prod._set_surface_query_cache_enabled(True)
    positions,skinning,records,stats=prod._solve_garment_meshes(source,cache,set(),{mesh_name},_skip_final_assembly=True)
    if mesh_name not in positions or mesh_name not in skinning:raise ValueError(f'mesh {mesh_name!r} did not produce a solved garment result')
    output_path.parent.mkdir(parents=True,exist_ok=True);report_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output_path,position=np.asarray(positions[mesh_name],dtype=np.float64),weights=np.asarray(skinning[mesh_name]['weights'],dtype=np.float64))
    report={'ok':True,'mesh':mesh_name,'elapsed_sec':time.perf_counter()-started,'pid':__import__('os').getpid(),'joint_names':list(skinning[mesh_name]['joint_names']),'skinning_stage':prod._strict_json_safe(skinning[mesh_name].get('stage',{})),'record':prod._strict_json_safe(records.get(mesh_name,{})),'stats':prod._strict_json_safe(stats)}
    report_path.write_text(json.dumps(report,separators=(',',':')),encoding='utf-8')
    return report


def _batch_main()->int:
    if len(sys.argv)!=5:raise SystemExit('garment_mesh_worker.py --batch <common.pkl> <manifest.json> <report.json>')
    common_path=Path(sys.argv[2]);manifest_path=Path(sys.argv[3]);batch_report=Path(sys.argv[4]);started=time.perf_counter();results=[]
    try:
        with common_path.open('rb') as f:common=pickle.load(f)
        rows=json.loads(manifest_path.read_text(encoding='utf-8')).get('meshes',[])
        if not rows:raise ValueError('garment batch manifest contains no authored meshes')

        # Solve the complete authored garment in one isolated worker to preserve peer context.
        meshes={}
        ordered=[]
        for row in rows:
            with Path(row['mesh_payload']).open('rb') as f:mesh=pickle.load(f)
            name=str(mesh['name']);ordered.append(name);meshes[name]=mesh['data']
        source=SerializedMultiGarmentSource(common['source_js'],meshes)
        cache=dict(common['cache'])
        target_fit.install_close_shell_macro_authority(prod)
        for key in ('_ravafit_local_affines','_ravafit_source_support_triangles','_ravafit_target_support_triangles','_ravafit_target_collision_triangles'):
            cache.pop(key,None)
        prod._reset_b14_runtime_caches();prod._set_surface_query_cache_enabled(True)
        positions,skinning,records,stats=prod._solve_garment_meshes(source,cache,set(),set(ordered),_skip_final_assembly=True)
        solve_elapsed=time.perf_counter()-started

        for row,name in zip(rows,ordered):
            if name not in positions or name not in skinning:raise ValueError(f'mesh {name!r} did not produce a solved garment result')
            output_path=Path(row['output']);report_path=Path(row['report']);output_path.parent.mkdir(parents=True,exist_ok=True);report_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(output_path,position=np.asarray(positions[name],dtype=np.float64),weights=np.asarray(skinning[name]['weights'],dtype=np.float64))
            payload={'ok':True,'mesh':name,'elapsed_sec':solve_elapsed,'pid':__import__('os').getpid(),'joint_names':list(skinning[name]['joint_names']),'skinning_stage':prod._strict_json_safe(skinning[name].get('stage',{})),'record':prod._strict_json_safe(records.get(name,{})),'stats':prod._strict_json_safe(stats)}
            report_path.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
            results.append({'mesh':name,'wall_sec':solve_elapsed,'worker_sec':solve_elapsed})
        batch_report.parent.mkdir(parents=True,exist_ok=True)
        batch_report.write_text(json.dumps({'ok':True,'elapsed_sec':time.perf_counter()-started,'solve_elapsed_sec':solve_elapsed,'pid':__import__('os').getpid(),'mesh_count':len(rows),'mode':'one-clean-multimesh-worker','meshes':results},separators=(',',':')),encoding='utf-8')
        return 0
    except Exception as ex:
        batch_report.parent.mkdir(parents=True,exist_ok=True)
        batch_report.write_text(json.dumps({'ok':False,'error':str(ex),'traceback':traceback.format_exc()[-10000:],'meshes':results},separators=(',',':')),encoding='utf-8')
        return 2


def main()->int:
    if len(sys.argv)>1 and sys.argv[1]=='--batch':return _batch_main()
    if len(sys.argv)!=5:raise SystemExit('garment_mesh_worker.py <common.pkl> <mesh.pkl> <output.npz> <report.json>')
    common_path=Path(sys.argv[1]);mesh_path=Path(sys.argv[2]);output_path=Path(sys.argv[3]);report_path=Path(sys.argv[4])
    try:
        with common_path.open('rb') as f:common=pickle.load(f)
        with mesh_path.open('rb') as f:mesh=pickle.load(f)
        _solve_one(common,mesh,output_path,report_path);return 0
    except Exception as ex:
        report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps({'ok':False,'error':str(ex),'traceback':traceback.format_exc()[-8000:]},separators=(',',':')),encoding='utf-8');return 2

if __name__=='__main__':raise SystemExit(main())
