from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys
import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_ROOT = _MODULE_DIR.parent if _MODULE_DIR.name.casefold() == 'solver' else _MODULE_DIR
_B14_SCRIPTS = _RUNTIME_ROOT / 'b14_frozen' / 'scripts'
if str(_B14_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_B14_SCRIPTS))

from ffxiv_lobofit import GLB as FrozenGLB


@dataclass(frozen=True)
class VertexDomain:
    key: tuple[int, ...]
    primitive_indices: tuple[int, ...]
    start: int
    count: int
    position_accessor: int
    normal_accessor: int | None
    uv_accessor: int | None
    tangent_accessor: int | None
    joints_accessors: tuple[int | None, int | None]
    weights_accessors: tuple[int | None, int | None]


def _attrs_key(primitive: dict[str, Any]) -> tuple[int, ...]:
    attrs = primitive.get('attributes', {})
    semantics = ('POSITION','NORMAL','TEXCOORD_0','TANGENT','JOINTS_0','WEIGHTS_0','JOINTS_1','WEIGHTS_1')
    return tuple(int(attrs.get(name, -1)) for name in semantics)


def mesh_domains(glb: Any, name: str) -> tuple[int, list[VertexDomain], list[dict[str, Any]]]:
    for mi, mesh in enumerate(glb.js.get('meshes', [])):
        if mesh.get('name') != name:
            continue
        primitives = list(mesh.get('primitives') or [])
        if not primitives:
            raise ValueError(f'{name} has no glTF primitives.')
        groups: dict[tuple[int, ...], list[int]] = {}
        for pi, primitive in enumerate(primitives):
            attrs = primitive.get('attributes', {})
            if 'POSITION' not in attrs:
                raise ValueError(f'{name} primitive {pi} has no POSITION accessor.')
            groups.setdefault(_attrs_key(primitive), []).append(pi)
        domains: list[VertexDomain] = []
        start = 0
        for key, primitive_indices in groups.items():
            p = primitives[primitive_indices[0]]
            attrs = p.get('attributes', {})
            count = int(glb.js['accessors'][attrs['POSITION']]['count'])
            for pi in primitive_indices[1:]:
                other = primitives[pi].get('attributes', {})
                if _attrs_key(primitives[pi]) != key:
                    raise AssertionError('domain grouping key changed unexpectedly')
                if int(glb.js['accessors'][other['POSITION']]['count']) != count:
                    raise ValueError(f'{name} primitives in one vertex domain disagree on vertex count.')
            domains.append(VertexDomain(
                key=key,
                primitive_indices=tuple(primitive_indices),
                start=start,
                count=count,
                position_accessor=int(attrs['POSITION']),
                normal_accessor=int(attrs['NORMAL']) if 'NORMAL' in attrs else None,
                uv_accessor=int(attrs['TEXCOORD_0']) if 'TEXCOORD_0' in attrs else None,
                tangent_accessor=int(attrs['TANGENT']) if 'TANGENT' in attrs else None,
                joints_accessors=(int(attrs['JOINTS_0']) if 'JOINTS_0' in attrs else None, int(attrs['JOINTS_1']) if 'JOINTS_1' in attrs else None),
                weights_accessors=(int(attrs['WEIGHTS_0']) if 'WEIGHTS_0' in attrs else None, int(attrs['WEIGHTS_1']) if 'WEIGHTS_1' in attrs else None),
            ))
            start += count
        return mi, domains, primitives
    raise KeyError(name)


def _domain_material(glb: Any, primitives: list[dict[str, Any]], name: str) -> str:
    values=[]
    for p in primitives:
        material_index=p.get('material')
        value='' if material_index is None else str(glb.js.get('materials',[{}])[material_index].get('name',''))
        values.append(value)
    normalized={v.replace('\\','/').strip().casefold() for v in values}
    if len(normalized)>1:
        raise ValueError(f'{name} spans multiple materials across glTF primitives: {values}. XIV mesh material identity must remain singular.')
    return values[0] if values else ''


def aggregate_mesh_data(glb: Any, name: str) -> dict[str, Any]:
    mi, domains, primitives = mesh_domains(glb, name)
    node_index = glb._mesh_nodes.get(mi)
    if node_index is None:
        raise ValueError(f'{name} has no scene node.')
    node=glb.js['nodes'][node_index]
    skin_index=node.get('skin')
    if skin_index is None:
        raise ValueError(f'{name} has no skin.')
    skin=glb.js['skins'][skin_index]
    joint_names=[glb.js['nodes'][ni].get('name','') for ni in skin.get('joints',[])]

    V_parts=[];N_parts=[];UV_parts=[];W_parts=[];faces=[]
    have_normals=True;have_uv=True
    for domain in domains:
        V=np.asarray(glb.accessor(domain.position_accessor),dtype=np.float64)
        V_parts.append(V)
        if domain.normal_accessor is None:
            have_normals=False
        else:
            N_parts.append(np.asarray(glb.accessor(domain.normal_accessor),dtype=np.float64))
        if domain.uv_accessor is None:
            have_uv=False
        else:
            UV_parts.append(np.asarray(glb.accessor(domain.uv_accessor),dtype=np.float64))
        W=np.zeros((domain.count,len(joint_names)),dtype=np.float64)
        for ja,wa in zip(domain.joints_accessors,domain.weights_accessors):
            if ja is None and wa is None:
                continue
            if ja is None or wa is None:
                raise ValueError(f'{name} has incomplete JOINTS/WEIGHTS attributes.')
            J=np.asarray(glb.accessor(ja),dtype=np.int64)
            WW=np.asarray(glb.accessor(wa),dtype=np.float64)
            if J.shape != WW.shape or len(J) != domain.count:
                raise ValueError(f'{name} skin accessor shape mismatch in a vertex domain.')
            valid=(J>=0)&(J<len(joint_names))
            for corner in range(J.shape[1]):
                rows=np.flatnonzero(valid[:,corner])
                if len(rows):
                    np.add.at(W,(rows,J[rows,corner]),WW[rows,corner])
        total=W.sum(axis=1);good=total>1e-12;W[good]/=total[good,None]
        W_parts.append(W)
        for pi in domain.primitive_indices:
            primitive=primitives[pi]
            if 'indices' not in primitive:
                raise ValueError(f'{name} primitive {pi} has no indices.')
            idx=np.asarray(glb.accessor(primitive['indices']),dtype=np.int64).reshape(-1)
            if len(idx)%3:
                raise ValueError(f'{name} primitive {pi} index count is not triangular.')
            if len(idx) and (int(idx.min())<0 or int(idx.max())>=domain.count):
                raise ValueError(f'{name} primitive {pi} index references outside its vertex domain.')
            faces.append(idx.reshape(-1,3)+domain.start)

    V=np.vstack(V_parts) if V_parts else np.zeros((0,3),dtype=np.float64)
    F=np.vstack(faces) if faces else np.zeros((0,3),dtype=np.int64)
    N=np.vstack(N_parts) if have_normals and len(N_parts)==len(domains) else None
    UV=np.vstack(UV_parts) if have_uv and len(UV_parts)==len(domains) else None
    W=np.vstack(W_parts) if W_parts else np.zeros((len(V),len(joint_names)),dtype=np.float64)
    return {
        'name':name,'mi':mi,'V':V,'F':F,'UV':UV,'N':N,'W':W,'joint_names':joint_names,
        'material':_domain_material(glb,primitives,name),
        '_primitive_count':len(primitives),'_vertex_domains':domains,
    }


def scatter_domain_attribute(glb: Any, name: str, values: np.ndarray, semantic: str) -> dict[str, Any]:
    """Scatter one aggregate solver attribute back to every authored glTF vertex domain."""
    _, domains, primitives = mesh_domains(glb, name)
    aggregate=np.asarray(values)
    total=sum(domain.count for domain in domains)
    if len(aggregate) != total:
        raise ValueError(f'{name} aggregate {semantic} rows {len(aggregate)} != domain rows {total}.')
    attr_name={
        'POSITION':'position_accessor','NORMAL':'normal_accessor','TANGENT':'tangent_accessor','TEXCOORD_0':'uv_accessor',
    }.get(semantic)
    if attr_name is None:
        raise ValueError(f'Unsupported scattered semantic {semantic!r}.')
    written=set();rows=[]
    for domain in domains:
        accessor=getattr(domain,attr_name)
        if accessor is None:
            raise ValueError(f'{name} domain has no {semantic} accessor.')
        section=aggregate[domain.start:domain.start+domain.count]
        if accessor not in written:
            glb.write_accessor(accessor,section)
            written.add(accessor)
            if semantic == 'POSITION':
                glb.js['accessors'][accessor]['min']=np.asarray(section).min(axis=0).astype(float).tolist()
                glb.js['accessors'][accessor]['max']=np.asarray(section).max(axis=0).astype(float).tolist()
        rows.append({'start':int(domain.start),'vertices':int(domain.count),'primitives':[int(v) for v in domain.primitive_indices],'accessor':int(accessor)})
    return {'primitive_count':len(primitives),'vertex_domain_count':len(domains),'domains':rows}


class ProductionGLB(FrozenGLB):
    """Frozen-B14-compatible GLB reader that exposes every primitive of an XIV mesh."""
    def primitive(self,name):
        mi,_,primitives=mesh_domains(self,name)
        if len(primitives)!=1:
            raise ValueError(f'{name} has {len(primitives)} primitives; use data() so no authored MeshPart is discarded.')
        return mi,primitives[0]

    def data(self,name):
        return aggregate_mesh_data(self,name)
