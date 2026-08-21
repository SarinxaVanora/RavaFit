#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, re, sys, zipfile
from pathlib import Path
from typing import Any

RUNTIME_RBODY=Path(__file__).resolve().parents[1]/'runtime'/'rbody'
sys.path.insert(0,str(RUNTIME_RBODY))
from rbody_v3_core import parse_rigged_mdl, apply_model_view, body_surface_view, hash_solver_identity

SOURCE_PACKAGE='[HS] Otopop (Default)'
BODY_ID='otopop'
COLLECTION='Otopop'

def slug(s:str)->str:
    return re.sub(r'[^a-z0-9]+','-',s.casefold()).strip('-') or 'variant'

def friendly(s:str)->str:
    return str(s).replace('Sqaure','Square')

def analysis_id(surface:dict[str,Any])->str:
    h=hashlib.sha256();h.update(surface['V'].astype('<f4',copy=False).tobytes());h.update(surface['F'].astype('<i4',copy=False).tobytes());return h.hexdigest()

def payload_meta(raw:bytes, source:dict[str,Any], selected_by:dict[str,Any])->tuple[str,dict[str,Any]]:
    pid=hashlib.sha256(raw).hexdigest();ref=parse_rigged_mdl(raw);surface=body_surface_view(ref)
    return pid,{
        'id':pid,'raw_mdl_sha256':pid,'solver_identity_sha256':hash_solver_identity(ref),'mdl_version':int(ref['mdl_version']),
        'vertex_count':int(len(ref['positions'])),'triangle_count':int(len(ref['indices'])//3),
        'body_surface_vertex_count':int(len(surface['V'])),'body_surface_triangle_count':int(len(surface['F'])),
        'joint_count':int(len(ref['joint_names'])),'joint_names':list(ref['joint_names']),
        'joint_name_sha256':hashlib.sha256('\0'.join(ref['joint_names']).encode()).hexdigest(),
        'solver_compatible':bool(ref['solver_compatible']),'has_normals':bool(any(r.get('has_normal') for r in ref['mesh_records'])),
        'has_uv0':bool(any(r.get('has_uv0') for r in ref['mesh_records'])),'has_skin':bool(any(r.get('has_skin') for r in ref['mesh_records'])),
        'bounds_min':ref['bounds_min'],'bounds_max':ref['bounds_max'],'materials':list(ref['materials']),
        'body_surface_materials':[r['material'] for r in surface.get('mesh_records',[])],
        'mesh_records':[dict(record,material=(ref['materials'][index] if index<len(ref['materials']) else '')) for index,record in enumerate(ref['mesh_records'])],'warning_count':int(ref.get('warning_count',0)),'warnings':list(ref.get('warnings') or []),
        'canonical_source':source,'selected_by':[selected_by],
    }

def group(z:zipfile.ZipFile,index:int)->tuple[str,dict[str,Any]]:
    prefix=f'group_{index:03d}_'
    name=next(n for n in z.namelist() if Path(n).name.casefold().startswith(prefix))
    return Path(name).name,json.loads(z.read(name))

def option_view(g:dict[str,Any],index:int)->dict[str,Any]:
    options=[o for o in g.get('Options',[]) if isinstance(o,dict)]
    selected=options[index]
    # An omitted attribute in another option is false when that Single-group option is not selected.
    controlled_attrs=set()
    for o in options:
        for m in o.get('Manipulations',[]):
            if m.get('Type')=='Atr':
                a=(m.get('Manipulation') or {}).get('Attribute')
                if a: controlled_attrs.add(str(a))
    attrs={a:False for a in controlled_attrs};shapes=[]
    for m in selected.get('Manipulations',[]):
        typ=m.get('Type');man=m.get('Manipulation') or {}
        if typ=='Atr' and man.get('Attribute'):
            attrs[str(man['Attribute'])]=bool(man.get('Entry'))
        elif typ=='Shp' and bool(man.get('Entry')) and man.get('Shape'):
            shapes.append(str(man['Shape']))
    view={'attributes':dict(sorted(attrs.items())),'shapes':shapes,'strict':True}
    if not attrs:view.pop('attributes')
    if not shapes:view.pop('shapes')
    return view

def merge_views(*views:dict[str,Any])->dict[str,Any]:
    attrs={};shapes=[];aliases={}
    for view in views:
        attrs.update(view.get('attributes') or {})
        for shape in view.get('shapes') or []:
            if shape not in shapes:shapes.append(shape)
        aliases.update(view.get('shape_aliases') or {})
    out={'strict':True}
    if attrs:out['attributes']=dict(sorted(attrs.items()))
    if shapes:out['shapes']=shapes
    if aliases:out['shape_aliases']=aliases
    return out

def target_option_defaults(*pairs:tuple[str,int])->dict[str,int]:return {name:index for name,index in pairs}

def build_variant(raw:bytes, source_path:str, target_path:str, race_code:str, slot:str, display:str, sex:str, view:dict[str,Any], defaults:dict[str,int], surface_mode:str='body'):
    entry_id=f'{BODY_ID}.{slot.casefold()}.{slug(display)}';selected={'entry_id':entry_id,'body':'Otopop','slot':slot,'variant':display,'race_code':race_code}
    source={'package':SOURCE_PACKAGE,'label':display,'source':source_path,'target':target_path,'race_code':race_code,'slot':slot}
    pid,meta=payload_meta(raw,source,selected)
    parsed=parse_rigged_mdl(raw);viewed=apply_model_view(parsed,view);surface=body_surface_view(viewed,surface_mode=surface_mode);aid=analysis_id(surface)
    variant={
        'id':entry_id,'display_name':display,'slot':slot,'source_packages':[SOURCE_PACKAGE],'source_labels':[display],'sexes':[sex],
        'race_payloads':[{'race_code':race_code,'solver_payload_id':pid,'replacement_payload_id':pid,'analysis_geometry_id':aid,'alternate_model_sha256s':[],
                          'source_label':display,'source_package':SOURCE_PACKAGE,'target_path':target_path}],
        'canonical_solver_payload_id':pid,'canonical_replacement_payload_id':pid,'analysis_geometry_id':aid,
        'model_view':view,'target_option_defaults':defaults,
    }
    if surface_mode!='body':variant['support_surface']=surface_mode
    return variant,pid,meta

def add_selected_by(meta:dict[str,Any], row:dict[str,Any]):
    key=(row.get('entry_id'),row.get('race_code'));seen={(x.get('entry_id'),x.get('race_code')) for x in meta.get('selected_by',[])}
    if key not in seen:meta.setdefault('selected_by',[]).append(row)

def main()->int:
    ap=argparse.ArgumentParser(description='Add Otopop 2.7.4 Lalafell body states to a unified RavaFit RBODY v4 library.')
    ap.add_argument('--rbody',required=True,type=Path);ap.add_argument('--otopop',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--report',type=Path)
    a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.rbody) as base, zipfile.ZipFile(a.otopop) as oto:
        manifest=json.loads(base.read('manifest.json'));catalogue=json.loads(base.read('catalogue.json'));payload_index=json.loads(base.read('payload_index.json'))
        # Idempotent replacement.
        catalogue['bodies']=[b for b in catalogue.get('bodies',[]) if str(b.get('id')).casefold()!=BODY_ID]
        if COLLECTION not in catalogue.get('collections',[]):catalogue.setdefault('collections',[]).append(COLLECTION)
        gnames={};groups={}
        for idx in (3,4,7,8,16,17,18,19,22):gnames[idx],groups[idx]=group(oto,idx)
        lookup={n.casefold():n for n in oto.namelist()}
        def read(rel:str)->bytes:return oto.read(lookup[rel.replace('\\','/').casefold()])
        paths={
            'mtop':'files/default/chara/equipment/e0000/model/c1101e0000_top.mdl',
            'ftop_off':'files/chest slider/off/chara/equipment/e0000/model/c1201e0000_top.mdl',
            'ftop_on':'files/chest slider/on/chara/equipment/e0000/model/c1201e0000_top.mdl',
            'mlegs':'files/default/chara/equipment/e0000/model/c1101e0000_dwn.mdl',
            'flegs':'files/default/chara/equipment/e0000/model/c1201e0000_dwn.mdl',
            'hands':'files/common/3/c1101e0000_glv.mdl','feet':'files/common/4/c1101e0000_sho.mdl',
        }
        raw={k:read(v) for k,v in paths.items()}
        slots={'Chest':[],'Legs':[],'Hands':[],'Feet':[]};new_payload_bytes={};new_payload_count_before=len(payload_index)
        def accept(result):
            variant,pid,meta=result;slots[variant['slot']].append(variant);new_payload_bytes.setdefault(pid,raw_by_pid[pid])
            if pid in payload_index:add_selected_by(payload_index[pid],meta['selected_by'][0])
            else:payload_index[pid]=meta
        raw_by_pid={hashlib.sha256(b).hexdigest():b for b in raw.values()}
        # Male chest: authored body shape + default Zip Up smallclothes top.
        for bi,bo in enumerate(groups[4]['Options']):
            display=f"Male {friendly(bo['Name'])}"
            body_view=option_view(groups[4],bi)
            if bo['Name'].casefold()=='kelly':
                body_view.setdefault('attributes',{})['atrx_oto_a']=False
                body_view['attributes']['atrx_oto_c']=True
            view=merge_views(body_view,option_view(groups[7],int(groups[7].get('DefaultSettings',0))))
            accept(build_variant(raw['mtop'],paths['mtop'],'chara/equipment/e0000/model/c1101e0000_top.mdl','1101','Chest',display,'male',view,
                                 target_option_defaults((gnames[4],bi),(gnames[7],int(groups[7].get('DefaultSettings',0))))))
        for slider_index,slider in enumerate(groups[16]['Options']):
            key='ftop_on' if slider['Name'].casefold()=='on' else 'ftop_off'
            for bi,bo in enumerate(groups[19]['Options']):
                display=f"Female {friendly(bo['Name'])} - Chest Slider {friendly(slider['Name'])}"
                body_view=option_view(groups[19],bi)
                if bo['Name'].casefold()=='kelly':body_view.setdefault('shape_aliases',{})['shpx_oto_kelly_allan_z']='shpx_oto_kelly'
                view=merge_views(body_view,option_view(groups[22],int(groups[22].get('DefaultSettings',0))))
                accept(build_variant(raw[key],paths[key],'chara/equipment/e0000/model/c1201e0000_top.mdl','1201','Chest',display,'female',view,
                                     target_option_defaults((gnames[16],slider_index),(gnames[19],bi),(gnames[22],int(groups[22].get('DefaultSettings',0))))))
        # Legs are explicit SFW support surfaces and retain Otopop's underwear / packed variants.
        for sex,race,gidx,key in [('Female','1201',3,'flegs'),('Male','1101',8,'mlegs')]:
            for oi,opt in enumerate(groups[gidx]['Options']):
                display=f"{sex} {friendly(opt['Name'])}";view=option_view(groups[gidx],oi)
                accept(build_variant(raw[key],paths[key],f'chara/equipment/e0000/model/c{race}e0000_dwn.mdl',race,'Legs',display,sex.casefold(),view,
                                     target_option_defaults((gnames[gidx],oi)),surface_mode='smallclothes'))
        # Shared hand MDL is explicitly mapped by Otopop to both c1101 and c1201.
        for sex,race,gidx in [('Female','1201',17),('Male','1101',18)]:
            for oi,opt in enumerate(groups[gidx]['Options']):
                display=f"{sex} {friendly(opt['Name'])}";view=option_view(groups[gidx],oi)
                accept(build_variant(raw['hands'],paths['hands'],f'chara/equipment/e0000/model/c{race}e0000_glv.mdl',race,'Hands',display,sex.casefold(),view,target_option_defaults((gnames[gidx],oi))))
        # Otopop 2.7.4 only installs a c1101 smallclothes feet model; do not invent a c1201 payload.
        accept(build_variant(raw['feet'],paths['feet'],'chara/equipment/e0000/model/c1101e0000_sho.mdl','1101','Feet','Male Otopop Feet','male',{'strict':True},{},surface_mode='body'))
        for values in slots.values():values.sort(key=lambda x:x['display_name'].casefold())
        body={'id':BODY_ID,'display_name':'Otopop','collection':COLLECTION,'source_packages':[SOURCE_PACKAGE],'slots':slots}
        catalogue['bodies'].append(body)
        catalogue['body_count']=len(catalogue['bodies']);catalogue['variant_count']=sum(len(v) for b in catalogue['bodies'] for v in (b.get('slots') or {}).values())
        manifest['collections']=catalogue['collections'];manifest['body_count']=catalogue['body_count'];manifest['variant_count']=catalogue['variant_count'];manifest['payload_count']=len(payload_index)
        manifest['otopop']='Otopop 2.7.4 catalogue states include MDL shape-key and submesh-attribute views; target extraction bakes the selected view without changing topology.'
        temp=a.output.with_suffix(a.output.suffix+'.tmp')
        if temp.exists():temp.unlink()
        skip={'manifest.json','catalogue.json','payload_index.json','README.txt'}
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as out:
            for info in base.infolist():
                if info.filename in skip:continue
                out.writestr(info,base.read(info.filename))
            readme=(base.read('README.txt').decode('utf8','replace') if 'README.txt' in base.namelist() else 'RavaFit unified RBODY v4.\n')
            if 'Otopop' not in readme:readme=readme.rstrip()+"\nIncludes Otopop 2.7.4 Lalafell body-state catalogue support.\n"
            out.writestr('README.txt',readme);out.writestr('manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False));out.writestr('catalogue.json',json.dumps(catalogue,indent=2,ensure_ascii=False));out.writestr('payload_index.json',json.dumps(payload_index,indent=2,ensure_ascii=False))
            existing=set(base.namelist())
            for pid,b in sorted(new_payload_bytes.items()):
                name=f'payloads/{pid}.mdl'
                if name not in existing:out.writestr(name,b)
        temp.replace(a.output)
    report={'output':str(a.output),'body_id':BODY_ID,'collection':COLLECTION,'variants':{slot:len(values) for slot,values in slots.items()},'total_otopop_variants':sum(map(len,slots.values())),'new_payloads':len(payload_index)-new_payload_count_before,'payload_count':len(payload_index),'body_count':catalogue['body_count'],'variant_count':catalogue['variant_count']}
    if a.report:a.report.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
