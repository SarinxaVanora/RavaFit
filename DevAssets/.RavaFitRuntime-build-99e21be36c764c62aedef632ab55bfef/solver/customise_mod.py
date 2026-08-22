from __future__ import annotations

import copy
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np


def _rbody_core():
    from rbody_v3_core import parse_rigged_mdl
    return parse_rigged_mdl


def _normalise_material(value: str) -> str:
    return str(value or '').replace('\\', '/').strip().casefold()


def _is_piercing_material(value: str) -> bool:
    n = _normalise_material(value)
    return any(token in n for token in ('pierc', 'jewel', 'dermal', 'barbell', 'bellyring'))


def _cluster_preview_part(positions: np.ndarray, normals: np.ndarray, uv0: np.ndarray, tri: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(tri) <= target_faces:
        used=np.unique(tri.reshape(-1))
        remap=np.full(len(positions),-1,dtype=np.int64)
        remap[used]=np.arange(len(used),dtype=np.int64)
        out_n=normals[used] if len(normals)==len(positions) else np.zeros((len(used),3),dtype=np.float32)
        out_u=uv0[used] if len(uv0)==len(positions) else np.zeros((len(used),2),dtype=np.float32)
        return positions[used], out_n, out_u, remap[tri]

    used=np.unique(tri.reshape(-1))
    pts=positions[used]
    lo=pts.min(axis=0);hi=pts.max(axis=0);span=np.maximum(hi-lo,1e-6)
    best=None
    low,high=3,96
    for _ in range(9):
        grid=(low+high)//2
        q=np.floor(((pts-lo)/span)*grid).astype(np.int32)
        keys,inv=np.unique(q,axis=0,return_inverse=True)
        local=np.full(len(positions),-1,dtype=np.int64);local[used]=inv
        faces=local[tri]
        keep=(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,2]!=faces[:,0])
        faces=faces[keep]
        if len(faces):
            canon=np.sort(faces,axis=1)
            _,unique_idx=np.unique(canon,axis=0,return_index=True)
            faces=faces[np.sort(unique_idx)]
        current=(abs(len(faces)-target_faces),grid,keys,inv,faces)
        if best is None or current[0]<best[0]:best=current
        if len(faces)>target_faces:high=max(low,grid-1)
        else:low=min(high,grid+1)
    _,_,keys,inv,faces=best
    cluster_count=len(keys)
    counts=np.bincount(inv,minlength=cluster_count).astype(np.float32)
    out_p=np.zeros((cluster_count,3),dtype=np.float32)
    out_n=np.zeros((cluster_count,3),dtype=np.float32)
    out_u=np.zeros((cluster_count,2),dtype=np.float32)
    for axis in range(3):out_p[:,axis]=np.bincount(inv,weights=pts[:,axis],minlength=cluster_count)/np.maximum(counts,1)
    src_n=normals[used] if len(normals)==len(positions) else np.zeros_like(pts)
    for axis in range(3):out_n[:,axis]=np.bincount(inv,weights=src_n[:,axis],minlength=cluster_count)/np.maximum(counts,1)
    nlen=np.linalg.norm(out_n,axis=1,keepdims=True);out_n=np.divide(out_n,np.maximum(nlen,1e-8),out=np.zeros_like(out_n),where=nlen>1e-8)
    src_u=uv0[used] if len(uv0)==len(positions) else np.zeros((len(used),2),dtype=np.float32)
    for axis in range(2):out_u[:,axis]=np.bincount(inv,weights=src_u[:,axis],minlength=cluster_count)/np.maximum(counts,1)
    return out_p,out_n,out_u,faces


def _preview_geometry(parsed: dict[str, Any], parts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],list[dict[str, Any]]]:
    positions=np.asarray(parsed.get('positions'),dtype=np.float32)
    normals=np.asarray(parsed.get('normals'),dtype=np.float32)
    uv0=np.asarray(parsed.get('uv0'),dtype=np.float32)
    indices=np.asarray(parsed.get('indices'),dtype=np.uint32)
    mesh_by_index={int(m['mesh_index']):m for m in parsed.get('mesh_records',[])}
    if not len(positions) or not len(indices) or not parts:return [],[]
    raw=[];total_faces=0
    for part in parts:
        mesh=mesh_by_index.get(int(part['mesh_index']))
        if mesh is None:continue
        lo=int(mesh['index_offset'])+int(part['local_index_offset']);count=int(part['index_count']);hi=min(len(indices),lo+count)
        tri=indices[lo:hi-(hi-lo)%3].reshape(-1,3) if hi>lo else np.zeros((0,3),dtype=np.uint32)
        tri=tri[np.max(tri,axis=1)<len(positions)] if len(tri) else tri
        raw.append((part,tri));total_faces+=len(tri)
    target_total=10000
    triangles=[];edges=[]
    for part,tri in raw:
        if not len(tri):continue
        share=len(tri)/max(1,total_faces)
        target=max(120,min(len(tri),int(round(target_total*share))))
        p,n,u,faces=_cluster_preview_part(positions,normals,uv0,tri,target)
        if not len(faces):continue
        # Boundary edges only: selected-region outlines are clean instead of drawing every face edge.
        edge_counts={}
        for f in faces:
            for a,b in ((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))):
                key=(a,b) if a<b else (b,a);edge_counts[key]=edge_counts.get(key,0)+1
        for a,b in (key for key,count in edge_counts.items() if count==1):
            edges.append({'part_index':int(part['part_index']),'p':[float(x) for x in np.concatenate((p[a],p[b]))]})
        for f in faces:
            a,b,c=(int(f[0]),int(f[1]),int(f[2]))
            face_normal=n[[a,b,c]].mean(axis=0)
            ln=float(np.linalg.norm(face_normal))
            if ln<1e-8:
                face_normal=np.cross(p[b]-p[a],p[c]-p[a]);ln=float(np.linalg.norm(face_normal))
            if ln>1e-8:face_normal=face_normal/ln
            triangles.append({
                'part_index':int(part['part_index']),
                'p':[float(x) for x in np.concatenate((p[a],p[b],p[c]))],
                'n':[float(x) for x in face_normal],
                'uv':[float(x) for x in np.concatenate((u[a],u[b],u[c]))],
            })
    return triangles,edges


def inspect_mdl_parts(path: str | Path) -> dict[str, Any]:
    source=Path(path).resolve()
    data=source.read_bytes()
    parsed=_rbody_core()(data)
    materials=list(parsed.get('materials') or [])
    parts=[]
    for mesh_ordinal,mesh in enumerate(parsed.get('mesh_records',[])):
        mesh_index=int(mesh['mesh_index'])
        material_index=int(mesh.get('material_index',-1))
        if 0<=mesh_ordinal<len(materials):
            material=materials[mesh_ordinal]
        elif 0<=material_index<len(materials):
            material=materials[material_index]
        else:
            material=''
        submeshes=list(mesh.get('submeshes') or [])
        authored_submeshes=bool(submeshes)
        if not submeshes:
            submeshes=[{'part_index':mesh_index,'local_index_offset':0,'index_count':int(mesh['index_count']),'attributes':[],'attribute_mask':0}]
        for local_ordinal,sm in enumerate(submeshes):
            local_start=max(0,int(sm.get('local_index_offset',0)))
            count=max(0,int(sm.get('index_count',0)))
            global_start=int(mesh['index_offset'])+local_start
            used=np.asarray(parsed['indices'][global_start:global_start+count],dtype=np.uint32)
            tri=used[:len(used)-(len(used)%3)].reshape(-1,3) if len(used)>=3 else np.zeros((0,3),dtype=np.uint32)
            visible_triangles=int(np.count_nonzero((tri[:,0]!=tri[:,1])&(tri[:,1]!=tri[:,2])&(tri[:,2]!=tri[:,0]))) if len(tri) else 0
            # Hide zero-geometry parts from Customise.
            if visible_triangles<=0:continue
            attrs=sorted({str(a) for a in sm.get('attributes',[]) if a})
            part_index=int(sm.get('part_index',mesh_index))
            parts.append({
                'part_index':part_index,
                'mesh_index':mesh_index,
                'submesh_index':int(local_ordinal),
                'material':str(material or ''),
                'vertex_count':int(len(np.unique(used))) if len(used) else 0,
                'index_count':count,
                'triangle_count':int(len(tri)),
                'visible_triangle_count':visible_triangles,
                'local_index_offset':local_start,
                'attributes':attrs,
                'piercing_like':_is_piercing_material(material) or any(any(token in a.casefold() for token in ('pierc', 'jewel', 'dermal', 'barbell', 'bellyring')) for a in attrs),
                'attribute_capable':authored_submeshes,
            })
    bounds_min=[float(x) for x in parsed.get('bounds_min',[0,0,0])]
    bounds_max=[float(x) for x in parsed.get('bounds_max',[0,0,0])]
    triangles,edges=_preview_geometry(parsed,parts)
    return {
        'ok':True,'path':str(source),'parts':parts,'part_count':len(parts),
        'preview':{'bounds_min':bounds_min,'bounds_max':bounds_max,'triangles':triangles,'edges':edges},
    }


def _part_signature(record: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (_normalise_material(record.get('material','')),tuple(sorted(str(x) for x in record.get('attributes',[]) if x)))


def _resolve_lod_record(target: dict[str, Any], lod_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not lod_records:return None
    target_material,target_attrs=_part_signature(target)
    ordinal=int(target.get('standard_ordinal',-1))
    if 0<=ordinal<len(lod_records):
        candidate=lod_records[ordinal]
        if _part_signature(candidate)==(target_material,target_attrs):return candidate
    exact=[r for r in lod_records if _part_signature(r)==(target_material,target_attrs)]
    if len(exact)==1:return exact[0]
    same=[r for r in lod_records if _normalise_material(r.get('material',''))==target_material]
    if len(same)==1:return same[0]
    if not same:return None
    raise ValueError(f"Could not safely identify '{target.get('material') or 'mesh'}' in a lower MDL LOD; customisation refused to touch the wrong mesh.")


def _resolve_lod_submesh(target_mesh: dict[str,Any], target_submesh: dict[str,Any], lod_records: list[dict[str,Any]]) -> dict[str,Any] | None:
    mesh=_resolve_lod_record(target_mesh,lod_records)
    if mesh is None:return None
    subs=list(mesh.get('submeshes') or [])
    if not subs:return None
    ordinal=int(target_submesh.get('submesh_ordinal',target_submesh.get('_local_ordinal',-1)))
    target_attrs=tuple(sorted(str(x) for x in target_submesh.get('attributes',[]) if x))
    if 0<=ordinal<len(subs):
        candidate=subs[ordinal]
        cand_attrs=tuple(sorted(str(x) for x in candidate.get('attributes',[]) if x))
        if cand_attrs==target_attrs:return candidate
    exact=[sm for sm in subs if tuple(sorted(str(x) for x in sm.get('attributes',[]) if x))==target_attrs]
    if len(exact)==1:return exact[0]
    if len(subs)==1:return subs[0]
    raise ValueError(f"Could not safely identify sub-part {ordinal+1} of '{target_mesh.get('material') or 'mesh'}' in a lower MDL LOD.")


def _append_model_attribute(raw: bytes, name: str) -> tuple[bytes,int]:
    from rbody_v3_core import _parse_model_data
    parsed=_rbody_core()(raw)
    names=list(parsed.get('attribute_names') or [])
    if name in names:return raw,names.index(name)
    if len(names)>=32:raise ValueError('This MDL already uses all 32 model attributes; RavaFit cannot add another independent visibility toggle safely.')
    encoded=name.encode('utf-8')+b'\0'
    mesh_count=struct.unpack_from('<H',raw,12)[0]
    path_header=68+136*mesh_count
    path_count,path_size=struct.unpack_from('<ii',raw,path_header)
    path_block=path_header+8
    model_start=path_block+path_size
    md,after=_parse_model_data(raw,model_start)
    old_count=int(md['AttributeCount'])
    lod_start=after+int(md['ElementIdCount'])*32
    q=lod_start;lod_meta=[]
    for _ in range(3):
        _,std_count=struct.unpack_from('<HH',raw,q);q+=4;q+=8
        water=struct.unpack_from('<HH',raw,q);q+=4
        shadow=struct.unpack_from('<HH',raw,q);q+=4
        _terrain=struct.unpack_from('<HH',raw,q);q+=4
        fog=struct.unpack_from('<HH',raw,q);q+=4;q+=32
        lod_meta.append((std_count,water,shadow,fog))
    info_count=sum(int(std)+int(w[1])+int(sh[1])+int(f[1]) for std,w,sh,f in lod_meta)
    attr_table=lod_start+3*60+info_count*36
    string_delta=len(encoded);new_string_offset=path_size
    step1=bytearray(raw[:path_block+path_size]+encoded+raw[path_block+path_size:])
    attr_insert=attr_table+string_delta+old_count*4
    out=bytearray(step1[:attr_insert]+struct.pack('<I',new_string_offset)+step1[attr_insert:])
    total=string_delta+4
    struct.pack_into('<i',out,8,struct.unpack_from('<i',raw,8)[0]+total)
    for base in (16,28):
        for i in range(3):
            old=struct.unpack_from('<I',raw,base+4*i)[0]
            struct.pack_into('<I',out,base+4*i,old+total if old else 0)
    struct.pack_into('<ii',out,path_header,path_count+1,path_size+string_delta)
    new_model_start=model_start+string_delta
    struct.pack_into('<h',out,new_model_start+6,old_count+1)
    md2,after2=_parse_model_data(out,new_model_start)
    lod2=after2+int(md2['ElementIdCount'])*32
    for lod_index in range(3):
        base=lod2+lod_index*60
        for rel in (32,52,56):
            value=struct.unpack_from('<i',out,base+rel)[0]
            if value:struct.pack_into('<i',out,base+rel,value+total)
    return bytes(out),old_count


def tag_mdl_parts(source_path: str|Path, output_path: str|Path, part_indices: list[int], attribute_name: str) -> dict[str,Any]:
    source=Path(source_path).resolve();output=Path(output_path).resolve();raw=source.read_bytes()
    before=_rbody_core()(raw)
    selected=sorted({int(x) for x in part_indices})
    if not selected:raise ValueError('At least one model part must be selected.')
    lods=list(before.get('lod_standard_mesh_records') or [])
    if not lods:raise ValueError('This MDL does not expose authored LOD mesh metadata.')
    # Build a lookup for the exact LOD0 submesh IDs shown by the UI.
    lod0=list(lods[0].get('records') or [])
    lookup={}
    for mesh in lod0:
        for local,sm in enumerate(mesh.get('submeshes') or []):
            copy_sm=dict(sm);copy_sm['_local_ordinal']=local
            lookup[int(sm['part_index'])]=(mesh,copy_sm)
    missing=[x for x in selected if x not in lookup]
    if missing:raise ValueError(f'Model part selection is no longer valid: {missing}. Refresh the model preview and retry.')

    clean=re.sub(r'[^a-z0-9_]+','_',str(attribute_name).casefold()).strip('_')
    if not clean.startswith('atrx_'):clean='atrx_'+clean
    if len(clean)>63:clean=clean[:63]
    tagged,new_attr_index=_append_model_attribute(raw,clean)
    reparsed=_rbody_core()(tagged)
    out=bytearray(tagged);bit=1<<new_attr_index
    touched=[]
    # Re-resolve after the insertion because every structure offset after the path block moved.
    lods_new=list(reparsed.get('lod_standard_mesh_records') or [])
    lod0_new=list(lods_new[0].get('records') or [])
    lookup_new={}
    for mesh in lod0_new:
        for local,sm in enumerate(mesh.get('submeshes') or []):
            copy_sm=dict(sm);copy_sm['_local_ordinal']=local
            lookup_new[int(sm['part_index'])]=(mesh,copy_sm)
    for part_id in selected:
        target_mesh,target_sm=lookup_new[part_id]
        for lod in lods_new:
            candidate=_resolve_lod_submesh(target_mesh,target_sm,list(lod.get('records') or []))
            if candidate is None:continue
            offset=int(candidate['struct_offset'])+8
            old_mask=struct.unpack_from('<I',out,offset)[0]
            struct.pack_into('<I',out,offset,old_mask|bit)
            touched.append({'lod':int(lod.get('lod',0)),'part_index':int(candidate['part_index']),'struct_offset':int(candidate['struct_offset'])})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(out)
    after=_rbody_core()(bytes(out))
    checks=(
        np.array_equal(before['positions'],after['positions']),np.array_equal(before['indices'],after['indices']),
        np.array_equal(before['normals'],after['normals']),np.array_equal(before['uv0'],after['uv0']),
        np.array_equal(before['joints'],after['joints']),np.array_equal(before['weights'],after['weights']),
        before['materials']==after['materials'],before['joint_names']==after['joint_names'])
    if not all(checks):
        try:output.unlink()
        except OSError:pass
        raise ValueError('Attribute-tagged MDL changed solver geometry or skinning; transaction refused.')
    return {'ok':True,'path':str(output),'attribute':clean,'attribute_index':int(new_attr_index),'tagged_parts':selected,'touched_lods':touched,'part_count':len(lookup_new)}


def hide_mdl_parts(source_path: str | Path, output_path: str | Path, part_indices: list[int]) -> dict[str, Any]:
    source=Path(source_path).resolve();output=Path(output_path).resolve();raw=source.read_bytes();parsed=_rbody_core()(raw)
    selected=sorted({int(x) for x in part_indices})
    if not selected:raise ValueError('At least one model part must be selected.')
    lods=list(parsed.get('lod_standard_mesh_records') or [])
    if not lods:raise ValueError('This MDL does not expose authored LOD mesh metadata.')
    lod0=list(lods[0].get('records') or [])
    lookup={int(sm['part_index']):(mesh,dict(sm,_local_ordinal=i)) for mesh in lod0 for i,sm in enumerate(mesh.get('submeshes') or [])}
    missing=[x for x in selected if x not in lookup]
    if missing:raise ValueError(f'Model part selection is outside the available submeshes: {missing}.')
    out=bytearray(raw);hidden_lods={};touched=[]
    for part in selected:
        target_mesh,target_sm=lookup[part]
        for lod in lods:
            candidate=_resolve_lod_submesh(target_mesh,target_sm,list(lod.get('records') or []))
            if candidate is None:continue
            mesh=_resolve_lod_record(target_mesh,list(lod.get('records') or []))
            count=int(candidate.get('index_count',0))
            if count<=0 or mesh is None:continue
            absolute=int(lod['index_buffer_offset'])+(int(mesh['source_index_offset'])+int(candidate['local_index_offset']))*2
            byte_count=count*2
            indices=np.frombuffer(raw,dtype='<u2',count=count,offset=absolute)
            first=int(indices[0]) if len(indices) else 0
            out[absolute:absolute+byte_count]=np.full(count,first,dtype='<u2').tobytes()
            touched.append((absolute,absolute+byte_count));hidden_lods.setdefault(int(lod.get('lod',0)),[]).append(int(candidate['part_index']))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(out)
    return {'ok':True,'path':str(output),'hidden_parts':selected,'hidden_lods':{str(k):v for k,v in hidden_lods.items()},'touched_ranges':[[a,b] for a,b in touched]}


def split_mdl_parts(source_path: str | Path, remaining_output_path: str | Path, accessory_output_path: str | Path, part_indices: list[int]) -> dict[str, Any]:
    """Split authored MDL parts into two fixed-size MDLs without rebuilding XIV mesh structures."""
    source=Path(source_path).resolve();remaining=Path(remaining_output_path).resolve();accessory=Path(accessory_output_path).resolve()
    inspection=inspect_mdl_parts(source)
    available=sorted({int(part['part_index']) for part in inspection.get('parts',[])})
    selected=sorted({int(x) for x in part_indices})
    if not selected:raise ValueError('Select at least one model part to split into the accessory.')
    missing=[x for x in selected if x not in available]
    if missing:raise ValueError(f'Model part selection is no longer valid: {missing}. Refresh the model preview and retry.')
    remaining_parts=[x for x in available if x not in selected]
    if not remaining_parts:raise ValueError('Splitting every part would leave the source equipment model empty. Keep at least one part on the source model.')

    remaining.parent.mkdir(parents=True,exist_ok=True);accessory.parent.mkdir(parents=True,exist_ok=True)
    remaining_report=hide_mdl_parts(source,remaining,selected)
    accessory_report=hide_mdl_parts(source,accessory,remaining_parts)
    # Fixed-size surgery is an important safety property for import/template compatibility.
    source_size=source.stat().st_size
    if remaining.stat().st_size!=source_size or accessory.stat().st_size!=source_size:
        for path in (remaining,accessory):
            try:path.unlink()
            except OSError:pass
        raise ValueError('Accessory split changed native MDL size; transaction refused.')
    return {
        'ok':True,
        'source':str(source),
        'remaining_path':str(remaining),
        'accessory_path':str(accessory),
        'selected_parts':selected,
        'remaining_parts':remaining_parts,
        'available_parts':available,
        'remaining_hidden_lods':remaining_report.get('hidden_lods',{}),
        'accessory_hidden_lods':accessory_report.get('hidden_lods',{}),
        'source_size':source_size,
        'policy':'selected authored parts remain together in accessory; source hides selected; accessory hides all unselected; no body inference/transplant',
    }


def _load_glb_helpers():
    # These modules are part of RavaFit's frozen B14 runtime and are already used by production_b14.
    from ffxiv_lobofit import GLB
    from glb_patch_legacy import GLBEditor
    from production_b14 import _clone_accessor, _ensure_material, _skin_joint_names
    return GLB, GLBEditor, _clone_accessor, _ensure_material, _skin_joint_names


def _mesh_skin_map(glb) -> dict[int, int]:
    result: dict[int, int] = {}
    for node in glb.js.get('nodes', []):
        if 'mesh' in node and 'skin' in node:
            result.setdefault(int(node['mesh']), int(node['skin']))
    return result


def _piercing_primitive_refs(glb) -> list[tuple[int, int, str, int | None]]:
    skin_by_mesh = _mesh_skin_map(glb)
    materials = glb.js.get('materials', [])
    out = []
    for mesh_index, mesh in enumerate(glb.js.get('meshes', [])):
        for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
            material_index = primitive.get('material')
            material_name = ''
            if isinstance(material_index, int) and 0 <= material_index < len(materials):
                material_name = str(materials[material_index].get('name', '') or '')
            if _is_piercing_material(material_name):
                out.append((mesh_index, primitive_index, material_name, skin_by_mesh.get(mesh_index)))
    return out



def extract_piercings_glb(source_glb: str | Path, output_glb: str | Path) -> dict[str, Any]:
    GLB, GLBEditor, _, _, _ = _load_glb_helpers()
    source_path = Path(source_glb).resolve()
    source = GLB(str(source_path))
    refs = _piercing_primitive_refs(source)
    if not refs:
        raise ValueError('The selected body export contains no piercing material geometry.')
    keep = {(mi, pi) for mi, pi, _, _ in refs}
    dst = GLBEditor(str(source_path))
    for mesh_index, mesh in enumerate(dst.js.get('meshes', [])):
        primitives = mesh.get('primitives', [])
        mesh['primitives'] = [p for pi, p in enumerate(primitives) if (mesh_index, pi) in keep]
        if not mesh['primitives']:
            mesh['name'] = ''
    empty_meshes = {i for i, mesh in enumerate(dst.js.get('meshes', [])) if not mesh.get('primitives')}
    empty_nodes = {i for i, node in enumerate(dst.js.get('nodes', [])) if node.get('mesh') in empty_meshes}
    for scene in dst.js.get('scenes', []):
        scene['nodes'] = [ni for ni in scene.get('nodes', []) if ni not in empty_nodes]
    output = Path(output_glb).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dst.save(str(output))
    return {'ok': True, 'path': str(output), 'piercing_primitives': len(refs), 'materials': sorted({r[2] for r in refs})}

def replace_piercings_glb(source_glb: str | Path, donor_glb: str | Path, output_glb: str | Path) -> dict[str, Any]:
    GLB, GLBEditor, clone_accessor, ensure_material, skin_joint_names = _load_glb_helpers()
    source = GLB(str(Path(source_glb).resolve()))
    donor = GLB(str(Path(donor_glb).resolve()))
    dst = GLBEditor(str(Path(source_glb).resolve()))

    source_refs = _piercing_primitive_refs(source)
    donor_refs = _piercing_primitive_refs(donor)
    if not donor_refs:
        raise ValueError('The selected target body has piercing controls but its exported model contains no piercing material geometry.')

    # Remove source piercing primitives, but keep every non-piercing mesh/primitive untouched.
    source_remove = {(mi, pi) for mi, pi, _, _ in source_refs}
    for mesh_index, mesh in enumerate(dst.js.get('meshes', [])):
        primitives = mesh.get('primitives', [])
        mesh['primitives'] = [p for pi, p in enumerate(primitives) if (mesh_index, pi) not in source_remove]
        if not mesh['primitives']:
            mesh['name'] = ''

    empty_meshes = {i for i, mesh in enumerate(dst.js.get('meshes', [])) if not mesh.get('primitives')}
    empty_nodes = {i for i, node in enumerate(dst.js.get('nodes', [])) if node.get('mesh') in empty_meshes}
    for node_index in empty_nodes:
        node = dst.js['nodes'][node_index]
        node.pop('mesh', None)
        node.pop('skin', None)
    for scene in dst.js.get('scenes', []):
        scene['nodes'] = [ni for ni in scene.get('nodes', []) if ni not in empty_nodes]

    destination_skin = None
    for node in dst.js.get('nodes', []):
        mesh_index = node.get('mesh')
        if isinstance(mesh_index, int) and dst.js.get('meshes', [])[mesh_index].get('primitives') and isinstance(node.get('skin'), int):
            destination_skin = int(node['skin'])
            break
    if destination_skin is None:
        raise ValueError('The source model has no retained skinned mesh to receive piercing geometry.')

    dest_skin_names = skin_joint_names(source, destination_skin)
    dest_by_name = {name: i for i, name in enumerate(dest_skin_names) if name}
    accessor_cache = {}
    added = 0
    added_materials: list[str] = []

    for mesh_index, primitive_index, material_name, donor_skin_index in donor_refs:
        if donor_skin_index is None:
            raise ValueError(f'Donor piercing mesh {mesh_index} is not skinned.')
        donor_joint_names = skin_joint_names(donor, donor_skin_index)
        missing = sorted({name for name in donor_joint_names if name and name not in dest_by_name})
        if missing:
            raise ValueError('Target piercing geometry requires skeleton joints missing from the selected model: ' + ', '.join(missing[:20]))
        joint_map = {i: dest_by_name[name] for i, name in enumerate(donor_joint_names) if name in dest_by_name}
        src_primitive = donor.js['meshes'][mesh_index]['primitives'][primitive_index]
        q = copy.deepcopy(src_primitive)
        q['attributes'] = {
            semantic: clone_accessor(dst, donor, ai, accessor_cache, joint_map if semantic.startswith('JOINTS_') else None)
            for semantic, ai in src_primitive.get('attributes', {}).items()
        }
        if 'indices' in src_primitive:
            q['indices'] = clone_accessor(dst, donor, src_primitive['indices'], accessor_cache)
        if 'targets' in src_primitive:
            q['targets'] = [
                {semantic: clone_accessor(dst, donor, ai, accessor_cache) for semantic, ai in morph.items()}
                for morph in src_primitive['targets']
            ]
        q['material'] = ensure_material(dst, material_name)
        new_mesh = {'name': f'ravafit piercing {added}', 'primitives': [q]}
        dst.js.setdefault('meshes', []).append(new_mesh)
        new_mesh_index = len(dst.js['meshes']) - 1
        # Attach to the same scene graph as the retained source model, using its existing skin.
        node_name = f'ravafit piercing {added}'
        dst.js.setdefault('nodes', []).append({'name': node_name, 'mesh': new_mesh_index, 'skin': destination_skin})
        new_node_index = len(dst.js['nodes']) - 1
        scene_index = int(dst.js.get('scene', 0) or 0)
        scenes = dst.js.setdefault('scenes', [{'nodes': []}])
        while len(scenes) <= scene_index:
            scenes.append({'nodes': []})
        scenes[scene_index].setdefault('nodes', []).append(new_node_index)
        added += 1
        added_materials.append(material_name)

    if added == 0:
        raise ValueError('No piercing geometry was copied from the selected body.')
    output = Path(output_glb).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dst.save(str(output))
    return {
        'ok': True,
        'path': str(output),
        'removed_source_primitives': len(source_refs),
        'added_target_primitives': added,
        'materials': sorted(set(added_materials)),
    }
