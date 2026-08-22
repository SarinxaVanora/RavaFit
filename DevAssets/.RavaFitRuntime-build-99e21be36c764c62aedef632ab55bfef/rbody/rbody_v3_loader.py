from __future__ import annotations
import json, zipfile, tempfile, hashlib, os
from pathlib import Path
from collections import OrderedDict
from typing import Iterable
import sys

try:
    from rbody_v3_core import parse_rigged_mdl, apply_model_view, body_surface_view
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from rbody_v3_core import parse_rigged_mdl, apply_model_view, body_surface_view

# Cache open catalogues by file stamp; replacing an RBODY invalidates the reader automatically.
_RBODY_READER_LIMIT=4
_RBODY_READERS: OrderedDict[str, tuple[tuple[int,int], "RBodyV3"]] = OrderedDict()

def _file_stamp(path: Path) -> tuple[int,int]:
    stat=path.stat();return int(stat.st_size),int(stat.st_mtime_ns)

def get_cached_rbody(path) -> "RBodyV3":
    resolved=Path(path).resolve();key=str(resolved);stamp=_file_stamp(resolved)
    cached=_RBODY_READERS.get(key)
    if cached is not None:
        old_stamp,reader=cached
        if old_stamp==stamp:
            _RBODY_READERS.move_to_end(key)
            return reader
        try:reader.close()
        finally:_RBODY_READERS.pop(key,None)
    reader=RBodyV3(resolved);_RBODY_READERS[key]=(stamp,reader);_RBODY_READERS.move_to_end(key)
    while len(_RBODY_READERS)>_RBODY_READER_LIMIT:
        _,(_,old)=_RBODY_READERS.popitem(last=False)
        old.close()
    return reader

def close_cached_rbodies():
    while _RBODY_READERS:
        _,(_,reader)=_RBODY_READERS.popitem(last=False)
        reader.close()

class RBodyV3:
    _RAW_CACHE_LIMIT=12
    _PARSED_CACHE_LIMIT=8
    def __init__(self,path):
        self.path=Path(path)
        self.z=zipfile.ZipFile(self.path,'r')
        self.manifest=json.loads(self.z.read('manifest.json'))
        if self.manifest.get('format')!='RBODY' or int(self.manifest.get('version',0))<3:
            raise ValueError(f'{self.path} is not RBODY v3')
        self.catalogue=json.loads(self.z.read('catalogue.json'))
        self.payload_index=json.loads(self.z.read('payload_index.json'))
        self._body_by_id={b['id'].casefold():b for b in self.catalogue['bodies']}
        self._bodies_by_name={}
        for b in self.catalogue['bodies']:
            self._bodies_by_name.setdefault(b['display_name'].casefold(),[]).append(b)
        self._body_by_name={k:v[0] for k,v in self._bodies_by_name.items() if len(v)==1}
        self._ambiguous_body_names={k:tuple(v) for k,v in self._bodies_by_name.items() if len(v)>1}
        self._raw=OrderedDict();self._parsed=OrderedDict()
    def close(self):
        self._raw.clear();self._parsed.clear();self.z.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def bodies(self):return [b['display_name'] for b in self.catalogue['bodies']]
    def body_ids(self):return [b['id'] for b in self.catalogue['bodies']]
    def body(self,name):
        k=str(name).casefold()
        # IDs are authoritative; names are only a fallback when they are unambiguous.
        b=self._body_by_id.get(k)
        if b is not None:return b
        b=self._body_by_name.get(k)
        if b is not None:return b
        ambiguous=self._ambiguous_body_names.get(k)
        if ambiguous:
            choices=', '.join(f"{row.get('collection','?')}:{row.get('id','?')}" for row in ambiguous)
            raise KeyError(f'Ambiguous body display name {name!r}; use a body id instead ({choices})')
        raise KeyError(f'Unknown body {name!r}')
    def slots(self,body):return list(self.body(body).get('slots',{}))
    def variants(self,body,slot):
        b=self.body(body); entries=b.get('slots',{}).get(slot)
        if entries is None:raise KeyError(f'{b["display_name"]} has no {slot} slot')
        return [e['display_name'] for e in entries]
    def entry(self,body,slot,variant):
        b=self.body(body); entries=b.get('slots',{}).get(slot)
        if entries is None:raise KeyError(f'{b["display_name"]} has no {slot} slot')
        k=str(variant).casefold()
        for e in entries:
            if e['display_name'].casefold()==k or e['id'].casefold()==k:return e
        raise KeyError(f'Unknown {b["display_name"]} {slot} variant {variant!r}')
    def resolve_payload_id(self,body,slot,variant,race_code=None):
        e=self.entry(body,slot,variant)
        if race_code is not None:
            rc=str(race_code)
            for r in e.get('race_payloads',[]):
                if r['race_code']==rc:return r['solver_payload_id']
            raise KeyError(f'{body} / {slot} / {variant} has no race {race_code}')
        pid=e.get('canonical_solver_payload_id')
        if not pid:raise KeyError(f'{body} / {slot} / {variant} has no canonical payload')
        return pid
    def payload_meta(self,payload_id):return self.payload_index[payload_id]
    def mesh_material_assignments(self,payload_id):
        """Return standard-mesh material assignments for an RBODY payload."""
        meta=self.payload_meta(payload_id);records=list(meta.get('mesh_records') or []);materials=list(meta.get('materials') or [])
        if len(records)!=len(materials):raise ValueError(f'RBODY payload {payload_id} has {len(records)} mesh records but {len(materials)} per-mesh materials')
        out={}
        for ordinal,(record,material) in enumerate(zip(records,materials)):
            vertex_count=int(record.get('vertex_count',0) or 0);index_count=int(record.get('index_count',0) or 0)
            if vertex_count<=0 or index_count<=0:continue
            mesh_index=int(record.get('mesh_index',ordinal));material=str(material or '').replace('\\','/').strip()
            if not material:raise ValueError(f'RBODY payload {payload_id} mesh {mesh_index} has geometry but no material')
            assignment={'mesh_index':mesh_index,'material':material,'material_table_index':int(record.get('material_index',-1)),'record_ordinal':ordinal,'vertex_count':vertex_count,'index_count':index_count}
            previous=out.get(mesh_index)
            if previous is not None and previous['material'].casefold()!=material.casefold():raise ValueError(f'RBODY payload {payload_id} mesh {mesh_index} has conflicting materials')
            out[mesh_index]=assignment
        if not out:raise ValueError(f'RBODY payload {payload_id} exposes no standard mesh/material assignments with geometry')
        return out
    def raw_mdl(self,payload_id):
        cached=self._raw.get(payload_id)
        if cached is not None:
            self._raw.move_to_end(payload_id);return cached
        b=self.z.read(f'payloads/{payload_id}.mdl')
        got=hashlib.sha256(b).hexdigest()
        if got!=payload_id:raise IOError(f'RBODY payload hash mismatch: expected {payload_id}, got {got}')
        self._raw[payload_id]=b;self._raw.move_to_end(payload_id)
        while len(self._raw)>self._RAW_CACHE_LIMIT:self._raw.popitem(last=False)
        return b
    def parsed_payload(self,payload_id):
        cached=self._parsed.get(payload_id)
        if cached is not None:
            self._parsed.move_to_end(payload_id);return cached
        parsed=parse_rigged_mdl(self.raw_mdl(payload_id));self._parsed[payload_id]=parsed;self._parsed.move_to_end(payload_id)
        while len(self._parsed)>self._PARSED_CACHE_LIMIT:self._parsed.popitem(last=False)
        return parsed
    def reference(self,body,slot,variant,race_code=None,rig_joint_names=None,surface_mode="body"):
        pid=self.resolve_payload_id(body,slot,variant,race_code)
        entry=self.entry(body,slot,variant)
        d=apply_model_view(self.parsed_payload(pid),entry.get('model_view'))
        ref=body_surface_view(d,rig_joint_names=rig_joint_names,surface_mode=surface_mode)
        ref['payload_id']=pid;ref['body']=self.body(body)['display_name'];ref['slot']=slot;ref['variant']=entry['display_name'];ref['race_code']=race_code;ref['model_view']=entry.get('model_view');ref['surface_mode']=str(surface_mode or 'body')
        return ref
    def write_pristine_mdl(self,destination,body,slot,variant,race_code=None):
        pid=self.resolve_payload_id(body,slot,variant,race_code); p=Path(destination);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(self.raw_mdl(pid));return p
    def extract_pristine_mdl_temp(self,body,slot,variant,race_code=None,directory=None):
        pid=self.resolve_payload_id(body,slot,variant,race_code); fd,p=tempfile.mkstemp(prefix=f'rbody_{pid[:12]}_',suffix='.mdl',dir=directory);Path(p).write_bytes(self.raw_mdl(pid));os.close(fd);return Path(p)
