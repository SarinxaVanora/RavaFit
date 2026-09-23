from __future__ import annotations
import copy, json, struct
from pathlib import Path
import numpy as np

from glb_patch_legacy import GLBEditor
from ffxiv_lobofit import GLB, COMP, NCOMP


def _align4(buf: bytearray):
    while len(buf) % 4:
        buf.append(0)


def _clone_accessor(dst: GLBEditor, src: GLB, accessor_index: int, cache: dict[int,int]) -> int:
    if accessor_index in cache:
        return cache[accessor_index]
    a = src.js['accessors'][accessor_index]
    if 'sparse' in a:
        raise NotImplementedError('Sparse accessors are not supported in body transplant')
    arr = src.accessor(accessor_index).copy()
    _align4(dst.bin)
    byte_offset = len(dst.bin)
    raw = np.ascontiguousarray(arr).tobytes(order='C')
    dst.bin.extend(raw)
    src_bv = src.js['bufferViews'][a['bufferView']]
    bv = {'buffer': 0, 'byteOffset': byte_offset, 'byteLength': len(raw)}
    if 'target' in src_bv:
        bv['target'] = src_bv['target']
    dst.js.setdefault('bufferViews', []).append(bv)
    bvi = len(dst.js['bufferViews']) - 1
    na = {k: copy.deepcopy(v) for k,v in a.items() if k not in ('bufferView','byteOffset','sparse')}
    na['bufferView'] = bvi
    na['byteOffset'] = 0
    dst.js.setdefault('accessors', []).append(na)
    ai = len(dst.js['accessors']) - 1
    cache[accessor_index] = ai
    return ai


def _material_map(dst: GLBEditor, src: GLB, src_mat_index: int | None) -> int | None:
    if src_mat_index is None:
        return None
    sm = src.js.get('materials', [])[src_mat_index]
    name = sm.get('name','')
    for i,m in enumerate(dst.js.get('materials', [])):
        if m.get('name','') == name:
            return i
    # Body material should already exist in the outfit; fail rather than silently duplicate textures/material dependencies.
    raise KeyError(f'Target body material {name!r} does not exist in outfit GLB')


def _clone_primitive(dst: GLBEditor, src: GLB, p: dict, acc_cache: dict[int,int]) -> dict:
    q = copy.deepcopy(p)
    q['attributes'] = {k:_clone_accessor(dst, src, ai, acc_cache) for k,ai in p.get('attributes',{}).items()}
    if 'indices' in p:
        q['indices'] = _clone_accessor(dst, src, p['indices'], acc_cache)
    if 'targets' in p:
        q['targets'] = [{k:_clone_accessor(dst, src, ai, acc_cache) for k,ai in t.items()} for t in p['targets']]
    if 'material' in p:
        q['material'] = _material_map(dst, src, p.get('material'))
    return q


def _clone_mesh(dst: GLBEditor, src: GLB, mesh_index: int, acc_cache: dict[int,int]) -> dict:
    m = copy.deepcopy(src.js['meshes'][mesh_index])
    m['primitives'] = [_clone_primitive(dst, src, p, acc_cache) for p in src.js['meshes'][mesh_index].get('primitives',[])]
    return m


def _mesh_index_by_name(js: dict, name: str) -> int:
    for i,m in enumerate(js.get('meshes',[])):
        if m.get('name') == name:
            return i
    raise KeyError(name)


def _target_body_mesh_names(target: GLB) -> list[str]:
    names=[]
    for mi,m in enumerate(target.js.get('meshes',[])):
        if not m.get('primitives'):
            continue
        p=m['primitives'][0]
        mat=''
        if 'material' in p:
            mat=target.js.get('materials',[{}])[p['material']].get('name','')
        if 'bibo' in mat.lower() or ('skin' in mat.lower() and 'outfit' not in mat.lower()):
            names.append(m.get('name'))
    return names


def transplant_selected_body(fitted_outfit_glb: str, selected_body_glb: str, output_glb: str):
    """
    Replace the original outfit's active Bibo body with the Selected Body geometry, byte-for-byte at accessor-value level,
    while retaining the outfit GLB's existing skeleton/armature nodes and garment meshes.

    Existing source body mesh slots mesh 0.1/0.2/0.3 are replaced in-place where possible so garment mesh indices do not shift.
    Any target-only body pieces (currently mesh 0.0/0.4) are appended as meshes/nodes, bound to the same existing skeleton nodes.
    """
    dst = GLBEditor(fitted_outfit_glb)
    tgt = GLB(selected_body_glb)
    acc_cache: dict[int,int] = {}

    # Hard compatibility gate: target skeleton node prefix must be exactly identical to outfit's skeleton node prefix.
    target_body_names = _target_body_mesh_names(tgt)
    if not target_body_names:
        raise RuntimeError('No target body meshes detected')

    # FFXIV files here share 172 rig nodes. Infer from target body skin joints rather than hardcoding.
    body_target_indices = [_mesh_index_by_name(tgt.js, n) for n in target_body_names]
    first_target_node = next(i for i,n in enumerate(tgt.js.get('nodes',[])) if n.get('mesh') in body_target_indices)
    target_skin = tgt.js['skins'][tgt.js['nodes'][first_target_node]['skin']]
    rig_joint_count = len(target_skin['joints'])
    if len(dst.js.get('nodes',[])) < rig_joint_count:
        raise RuntimeError('Outfit has fewer nodes than target rig joint count')
    for i in range(rig_joint_count):
        a=dst.js['nodes'][i]; b=tgt.js['nodes'][i]
        # Mesh/skin are not expected on skeleton nodes. Compare all transform/hierarchy/name data.
        def clean(n): return {k:v for k,v in n.items() if k not in ('mesh','skin','extras')}
        if clean(a) != clean(b):
            raise RuntimeError(f'Skeleton node mismatch at {i}: {a.get("name")} vs {b.get("name")}')

    # Existing body mesh/node records in destination, including currently-hidden B06 body nodes.
    mats=dst.js.get('materials',[])
    dst_body_mesh_indices=[]
    for mi,m in enumerate(dst.js.get('meshes',[])):
        is_body=False
        for p in m.get('primitives',[]):
            mat=p.get('material'); mn=mats[mat].get('name','') if mat is not None else ''
            if 'bibo' in mn.lower() or ('skin' in mn.lower() and 'outfit' not in mn.lower()):
                is_body=True; break
        if is_body: dst_body_mesh_indices.append(mi)
    dst_mesh_to_node={n.get('mesh'):i for i,n in enumerate(dst.js.get('nodes',[])) if 'mesh' in n}

    # Map each target body mesh by its semantic mesh name. Replace in-place where the outfit already has the same body piece.
    active_body_nodes=[]
    for tname in target_body_names:
        tmi=_mesh_index_by_name(tgt.js,tname)
        tnode_i=next(i for i,n in enumerate(tgt.js['nodes']) if n.get('mesh')==tmi)
        tnode=tgt.js['nodes'][tnode_i]
        cloned_mesh=_clone_mesh(dst,tgt,tmi,acc_cache)
        try:
            dmi=_mesh_index_by_name(dst.js,tname)
            if dmi not in dst_body_mesh_indices:
                raise KeyError(tname)
            dst.js['meshes'][dmi]=cloned_mesh
            dni=dst_mesh_to_node[dmi]
            # Keep same node index but adopt target's body-piece extras. Bind to an existing compatible outfit skin.
            # All target body skins use the exact same joints/IBMs as source in this asset; validate below.
            dst.js['nodes'][dni]['extras']=copy.deepcopy(tnode.get('extras',{}))
            active_body_nodes.append(dni)
        except KeyError:
            dmi=len(dst.js['meshes']); dst.js['meshes'].append(cloned_mesh)
            # Reuse destination skin 0 after validating it is exactly compatible with the target body skin.
            new_node={'mesh':dmi,'skin':0,'extras':copy.deepcopy(tnode.get('extras',{}))}
            dni=len(dst.js['nodes']); dst.js['nodes'].append(new_node)
            active_body_nodes.append(dni)

    # Validate destination skin 0 against every target body skin by joint node names + inverse bind matrices.
    base_skin=dst.js['skins'][0]
    base_names=[dst.js['nodes'][j].get('name','') for j in base_skin['joints']]
    base_ibm=GLB(fitted_outfit_glb).accessor(base_skin['inverseBindMatrices'])
    for tname in target_body_names:
        tmi=_mesh_index_by_name(tgt.js,tname); tni=next(i for i,n in enumerate(tgt.js['nodes']) if n.get('mesh')==tmi)
        tsi=tgt.js['nodes'][tni]['skin']; ts=tgt.js['skins'][tsi]
        tnames=[tgt.js['nodes'][j].get('name','') for j in ts['joints']]
        if tnames != base_names:
            raise RuntimeError(f'Skin joint mismatch for target {tname}')
        tibm=tgt.accessor(ts['inverseBindMatrices'])
        if tibm.shape != base_ibm.shape or np.max(np.abs(tibm.astype(float)-base_ibm.astype(float))) > 1e-7:
            raise RuntimeError(f'Inverse bind matrix mismatch for target {tname}')

    # Ensure all body nodes use the known-compatible existing outfit skin 0.
    for ni in active_body_nodes:
        dst.js['nodes'][ni]['skin']=0

    # Remove any old source-body nodes not re-used above from every active scene, then add all Selected Body pieces.
    old_body_nodes=[dst_mesh_to_node[mi] for mi in dst_body_mesh_indices if mi in dst_mesh_to_node]
    for scene in dst.js.get('scenes',[]):
        current=[ni for ni in scene.get('nodes',[]) if ni not in old_body_nodes and ni not in active_body_nodes]
        # Preserve root first, then body pieces in target semantic order, then original garment nodes.
        roots=[ni for ni in current if ni < rig_joint_count]
        others=[ni for ni in current if ni not in roots]
        scene['nodes']=roots + active_body_nodes + others

    # Update declared buffer byte length.
    dst.js['buffers'][0]['byteLength']=len(dst.bin)
    dst.save(output_glb)
    return {'target_body_meshes':target_body_names,'active_body_nodes':active_body_nodes,'old_body_nodes':old_body_nodes}


def audit_transplant(out_glb: str, selected_body_glb: str):
    out=GLB(out_glb); tgt=GLB(selected_body_glb)
    report={'meshes':{},'all_exact':True}
    for name in _target_body_mesh_names(tgt):
        od=out.data(name); td=tgt.data(name)
        item={'position_max_abs':float(np.max(np.abs(od['V']-td['V']))),
              'normal_max_abs':float(np.max(np.abs(od['N']-td['N'])) if od['N'] is not None else np.nan),
              'faces_exact':bool(np.array_equal(od['F'],td['F'])),
              'vertex_count':int(len(od['V'])),'face_count':int(len(od['F']))}
        _,op=out.primitive(name); _,tp=tgt.primitive(name)
        for attr in tp.get('attributes',{}):
            oa=out.accessor(op['attributes'][attr]); ta=tgt.accessor(tp['attributes'][attr])
            if np.issubdtype(oa.dtype,np.floating): exact=bool(np.array_equal(oa,ta)); md=float(np.max(np.abs(oa.astype(float)-ta.astype(float))))
            else: exact=bool(np.array_equal(oa,ta)); md=0.0 if exact else float('nan')
            item[f'{attr}_exact']=exact; item[f'{attr}_max_abs']=md
            report['all_exact'] &= exact
        report['meshes'][name]=item
        report['all_exact'] &= item['faces_exact']
    # Active body nodes
    scene=out.js['scenes'][out.js.get('scene',0)]['nodes']; active=[]
    for ni in scene:
        n=out.js['nodes'][ni]
        if 'mesh' in n:
            mn=out.js['meshes'][n['mesh']].get('name','')
            try:
                _,p=out.primitive(mn); mat=out.material_name(p)
            except Exception: mat=''
            if 'bibo' in mat.lower(): active.append((ni,mn,n.get('skin'),n.get('extras',{})))
    report['active_body_nodes']=active
    report['active_body_names']=[x[1] for x in active]
    report['all_target_body_active']=set(report['active_body_names'])==set(_target_body_mesh_names(tgt))
    report['all_exact'] &= report['all_target_body_active']
    return report

if __name__=='__main__':
    base=Path('/mnt/data/garment_refit_lobofit')
    inp=base/'candidates/B06_recovered_lobofit_full.glb'
    tgt=base/'inputs/Selected_Body.glb'
    out=base/'diagnostics/B06_EXPORT_TEST_with_Selected_Body.glb'
    info=transplant_selected_body(str(inp),str(tgt),str(out))
    rep=audit_transplant(str(out),str(tgt))
    (base/'metrics/B06_export_transplant_audit.json').write_text(json.dumps({'transplant':info,'audit':rep},indent=2))
    print(json.dumps({'output':str(out),'transplant':info,'audit':rep},indent=2))
