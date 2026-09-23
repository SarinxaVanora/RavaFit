#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, sys, zipfile
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'/'rbody'))
from rbody_v3_core import parse_rigged_mdl, body_surface_view, hash_solver_identity

PACKAGE='Neolithe [MAIN - NEOBELLY]'
GROUPS={
    'Chest': [('CHEST: SmallClothes','group_005_chest_ smallclothes.json'),("CHEST: The Emperor's New Robe","group_007_chest_ the emperor's new robe.json")],
    'Legs': [('LEGS: SmallClothes','group_009_legs_ smallclothes.json'),("LEGS: The Emperor's New Breeches","group_011_legs_ the emperor's new breeches.json")],
}

def slug(value:str)->str:
    return re.sub(r'-+','-',re.sub(r'[^a-z0-9]+','-',value.casefold())).strip('-')

def analysis_geometry_id(surface:dict[str,Any])->str:
    h=hashlib.sha256();h.update(surface['V'].astype('<f4',copy=False).tobytes());h.update(surface['F'].astype('<i4',copy=False).tobytes());return h.hexdigest()

def payload_meta(raw:bytes, source:dict[str,Any], selected_by:dict[str,Any])->tuple[str,dict[str,Any],str]:
    pid=hashlib.sha256(raw).hexdigest();ref=parse_rigged_mdl(raw);surface=body_surface_view(ref);aid=analysis_geometry_id(surface)
    meta={
        'id':pid,'raw_mdl_sha256':pid,'solver_identity_sha256':hash_solver_identity(ref),'mdl_version':int(ref['mdl_version']),
        'vertex_count':int(len(ref['positions'])),'triangle_count':int(len(ref['indices'])//3),'body_surface_vertex_count':int(len(surface['V'])),'body_surface_triangle_count':int(len(surface['F'])),
        'joint_count':int(len(ref['joint_names'])),'joint_names':list(ref['joint_names']),'joint_name_sha256':hashlib.sha256('\0'.join(ref['joint_names']).encode()).hexdigest(),
        'solver_compatible':bool(ref['solver_compatible']),'has_normals':bool(any(r.get('has_normal') for r in ref['mesh_records'])),'has_uv0':bool(any(r.get('has_uv0') for r in ref['mesh_records'])),'has_skin':bool(any(r.get('has_skin') for r in ref['mesh_records'])),
        'bounds_min':ref['bounds_min'],'bounds_max':ref['bounds_max'],'materials':list(ref['materials']),'body_surface_materials':[r['material'] for r in surface.get('mesh_records',[])],
        'mesh_records':[dict(r,material=(ref['materials'][i] if i<len(ref['materials']) else '')) for i,r in enumerate(ref['mesh_records'])],
        'warning_count':int(ref.get('warning_count',0)),'warnings':list(ref.get('warnings') or []),'canonical_source':source,'selected_by':[selected_by],
    }
    return pid,meta,aid

def read_pmp_rows(pmp:Path):
    rows=[]
    with zipfile.ZipFile(pmp) as z:
        lookup={n.replace('\\','/').casefold():n for n in z.namelist()}
        for slot, groups in GROUPS.items():
            for group_name, group_file in groups:
                group=json.loads(z.read(lookup[group_file.casefold()]))
                context='Emperor' if 'emperor' in group_name.casefold() else 'SmallClothes'
                for option in group.get('Options') or []:
                    option_name=str(option.get('Name') or '').strip()
                    for game_path, rel in (option.get('Files') or {}).items():
                        if not str(rel).casefold().endswith('.mdl'): continue
                        key=str(rel).replace('\\','/').casefold();raw=z.read(lookup[key])
                        ref=parse_rigged_mdl(raw)
                        rows.append({'slot':slot,'group_name':group_name,'context':context,'option':option_name,'game_path':str(game_path).replace('\\','/'),'source_path':str(rel).replace('\\','/'),'raw':raw,'raw_sha':hashlib.sha256(raw).hexdigest(),'solver_sha':hash_solver_identity(ref)})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input_rbody',type=Path);ap.add_argument('neobelly_pmp',type=Path);ap.add_argument('output_rbody',type=Path);args=ap.parse_args()
    rows=read_pmp_rows(args.neobelly_pmp)
    by_shape={}
    for row in rows:
        by_shape.setdefault((row['slot'],row['option'].casefold(),row['solver_sha']),[]).append(row)
    assert len(rows)==46, len(rows);assert len(by_shape)==23,len(by_shape)

    with zipfile.ZipFile(args.input_rbody) as zin:
        manifest=json.loads(zin.read('manifest.json'));catalogue=json.loads(zin.read('catalogue.json'));payload_index=json.loads(zin.read('payload_index.json'));target_options=json.loads(zin.read('target_options.json'))
        body=next(b for b in catalogue['bodies'] if b.get('id')=='neolithe');body.setdefault('source_packages',[])
        if PACKAGE not in body['source_packages']: body['source_packages'].append(PACKAGE)
        body['source_packages']=sorted(set(body['source_packages']),key=str.casefold)
        existing_by_solver={}
        for slot,variants in body['slots'].items():
            for v in variants:
                for rp in v.get('race_payloads') or []:
                    pm=payload_index.get(rp.get('solver_payload_id')) or {}
                    ss=pm.get('solver_identity_sha256')
                    if ss: existing_by_solver[(slot,ss)]=v

        new_payload_bytes={};added_variants=[];linked_contexts=[]
        for (slot,option_key,solver_sha), group_rows in sorted(by_shape.items()):
            small=next(r for r in group_rows if r['context']=='SmallClothes');emp=next(r for r in group_rows if r['context']=='Emperor')
            variant=existing_by_solver.get((slot,solver_sha))
            option=small['option']
            if variant is None:
                display=f"NeoBelly {option}"
                vid=f"neolithe.{slot.casefold()}.neobelly-{slug(option)}"
                existing_ids={v['id'] for v in body['slots'][slot]};base=vid;n=2
                while vid in existing_ids: vid=f'{base}-{n}';n+=1
                selected={'entry_id':vid,'body':'Neolithe','slot':slot,'variant':display,'race_code':'0201'}
                source={'package':PACKAGE,'label':f"{small['group_name']} / {option}",'source':small['source_path'],'target':small['game_path'],'race_code':'0201','slot':slot}
                pid,meta,aid=payload_meta(small['raw'],source,selected);payload_index.setdefault(pid,meta);new_payload_bytes.setdefault(pid,small['raw'])
                variant={'id':vid,'display_name':display,'slot':slot,'source_packages':[PACKAGE],'source_labels':[source['label']],'sexes':['female'],'race_payloads':[{'race_code':'0201','solver_payload_id':pid,'replacement_payload_id':pid,'analysis_geometry_id':aid,'alternate_model_sha256s':[],'source_label':source['label'],'source_package':PACKAGE,'target_path':small['game_path']}],'canonical_solver_payload_id':pid,'canonical_replacement_payload_id':pid,'analysis_geometry_id':aid}
                if option.casefold().startswith('sfw '): variant['support_surface']='smallclothes'
                body['slots'][slot].append(variant);existing_by_solver[(slot,solver_sha)]=variant;added_variants.append(vid)
            else:
                # Make legacy NeoBelly labels explicit without changing stable IDs.
                if variant['id'] in {f'neolithe.chest.neobelly-{x}' for x in ('xs','s','m','l')}:
                    size=variant['display_name'].split()[-1];variant['display_name']=f'NeoBelly NSFW {size}'
                elif 'neobelly' in variant['display_name'].casefold():
                    variant['display_name']=variant['display_name'].replace('Neobelly','NeoBelly')
                variant.setdefault('source_packages',[])
                if PACKAGE not in variant['source_packages']: variant['source_packages'].append(PACKAGE)
                variant['source_packages']=sorted(set(variant['source_packages']),key=str.casefold)

            rp=variant['race_payloads'][0];alts=rp.setdefault('alternate_model_sha256s',[])
            if emp['raw_sha'] not in alts: alts.append(emp['raw_sha'])
            alts.sort()
            for row in group_rows:
                selected={'entry_id':variant['id'],'body':'Neolithe','slot':slot,'variant':variant['display_name'],'race_code':'0201'}
                source={'package':PACKAGE,'label':f"{row['group_name']} / {row['option']}",'source':row['source_path'],'target':row['game_path'],'race_code':'0201','slot':slot}
                pid=row['raw_sha']
                if pid not in payload_index:
                    _,meta,_=payload_meta(row['raw'],source,selected);payload_index[pid]=meta;new_payload_bytes[pid]=row['raw']
                else:
                    pm=payload_index[pid];pm.setdefault('selected_by',[])
                    if not any(x.get('entry_id')==variant['id'] and x.get('race_code')=='0201' for x in pm['selected_by']): pm['selected_by'].append(selected)
                linked_contexts.append({'variant':variant['id'],'context':row['context'],'raw_sha256':pid,'solver_identity_sha256':solver_sha})

        for slot in body['slots']:
            body['slots'][slot].sort(key=lambda v:(str(v.get('display_name','')).casefold(),str(v.get('id',''))))
        body['slot_counts']={slot:len(body['slots'].get(slot) or []) for slot in ('Chest','Legs','Hands','Feet')}
        catalogue['body_count']=len(catalogue['bodies']);catalogue['variant_count']=sum(len(vs or []) for b in catalogue['bodies'] for vs in (b.get('slots') or {}).values())
        manifest['body_count']=catalogue['body_count'];manifest['variant_count']=catalogue['variant_count'];manifest['payload_count']=len(payload_index)
        manifest['neobelly_complete']='46 authored NeoBelly MDLs retained as 23 solver-unique UI shapes; SmallClothes canonical + Emperor alternate contexts.'

        stage=args.output_rbody.with_suffix(args.output_rbody.suffix + '.stage')
        if stage.exists():
            import shutil; shutil.rmtree(stage)
        (stage/'payloads').mkdir(parents=True,exist_ok=True)
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        (stage/'catalogue.json').write_text(json.dumps(catalogue,indent=2,ensure_ascii=False),encoding='utf-8')
        (stage/'payload_index.json').write_text(json.dumps(payload_index,indent=2,ensure_ascii=False),encoding='utf-8')
        (stage/'target_options.json').write_text(json.dumps(target_options,indent=2,ensure_ascii=False),encoding='utf-8')
        for pid,raw in new_payload_bytes.items(): (stage/'payloads'/f'{pid}.mdl').write_bytes(raw)
    report={'authored_model_contexts':len(rows),'solver_unique_shapes':len(by_shape),'new_ui_variants':added_variants,'new_payloads':len(new_payload_bytes),'linked_contexts':linked_contexts,'output':str(args.output_rbody),'stage':str(stage)}
    args.output_rbody.with_suffix('.neobelly-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='linked_contexts'},indent=2))
if __name__=='__main__':main()
